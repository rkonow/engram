//! Building a vector field's sections.
//!
//! Build options (keys of [`BuildOptions`], all optional):
//!
//! | key | default | meaning |
//! |---|---|---|
//! | `vector.index` | schema's kind | `flat`, `ivf` or `hnsw` |
//! | `vector.codes` | schema's codes (`sq8`) | `sq8` or `binary` |
//! | `vector.ivf.nlist` | schema's `nlist`, else ≈ 4·√n | number of clusters |
//! | `vector.ivf.iters` | 10 | k-means iterations |
//! | `vector.ivf.sample` | 64 | k-means training points per cluster |
//! | `vector.hnsw.m` | schema's `m` (16) | links per node (2·M on layer 0) |
//! | `vector.hnsw.ef_construction` | schema's (200) | construction beam width |
//! | `vector.hnsw.build` | `sq8` with SQ8 codes | `sq8`: construction scores with the codes, as search does; `f32`: with float32 rows |
//! | `vector.threads` | 1 | threads for k-means and graph construction |
//! | `vector.seed` | fixed | seed for sampling and HNSW levels |
//!
//! A field that is not `indexed` gets the forward vectors only (flat).

use crate::postings::bytes::ByteWriter;

use crate::error::{Error, Result};
use crate::schema::{Field, FieldType, VectorCodes, VectorIndexKind};
use crate::segment::build::BuildOptions;
use crate::segment::{DocId, FieldIndexBuilder, SegmentFile, SegmentWriter};
use crate::value::Value;
use crate::vector::hnsw::{self, HnswConfig, HnswView};
use crate::vector::ivf::{self, IvfConfig};
use crate::vector::bits::Bits;
use crate::vector::codes::Quantiser;
use crate::vector::sq8::Sq8;
use crate::vector::{section, VectorInfo};

/// Float32 rows of `stride` floats (the first `dim` hold the vector).
pub(crate) struct Rows<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub stride: usize,
    pub n: usize,
}

impl Rows<'_> {
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.stride..(i + 1) * self.stride]
    }
}

/// Row strides: float32 rows padded to 16 floats and SQ8 code rows to 64
/// bytes, so every row starts on a cache line and kernels need no tail
/// handling (binary code rows: [`VectorCodes::stride`]).
pub(crate) fn strides(dim: usize) -> (usize, usize) {
    (dim.next_multiple_of(16), dim.next_multiple_of(64))
}

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Flat,
    Ivf { nlist: Option<usize>, iters: usize, sample: usize },
    Hnsw { m: usize, ef_construction: usize, f32: bool },
}

fn parse<T: std::str::FromStr>(opts: &BuildOptions, key: &str) -> Result<Option<T>> {
    opts.get(key).map(|v| v.parse::<T>().map_err(|_| Error::Schema(format!("bad value {v:?} for build option {key}")))).transpose()
}

pub(crate) struct VectorBuilder {
    field: String,
    dim: usize,
    stride: usize,
    kind: Kind,
    codes: VectorCodes,
    /// Segments with more vectors than this build IVF instead of HNSW
    /// (`vector.hnsw.max_docs`): an HNSW graph's build time grows faster
    /// than the segment, and is paid again at every merge.
    hnsw_max_docs: Option<usize>,
    threads: usize,
    seed: u64,
    /// Segments whose graphs this one's starts from, and the docID each
    /// one's first document has here ([`hnsw::StartGraph`]); the first that
    /// fits keeps its links, the others are joined to it.
    start: Vec<(SegmentFile, u32)>,
    /// Float32 rows as little-endian bytes, so the buffer becomes the
    /// section without a copy.
    rows: Vec<u8>,
    present: Vec<u64>,
    n: usize,
    missing: usize,
    /// A multi-vector field's row offsets (`n` counts vectors), and whether
    /// it keeps its float32 vectors.
    multi: Option<(Vec<u32>, bool)>,
}

impl VectorBuilder {
    pub fn new(field: &Field, opts: &BuildOptions) -> Result<Self> {
        let FieldType::Vector { dim, index } = &field.ty else {
            return Err(Error::Schema(format!("field {} is not a vector", field.name)));
        };
        let schema_kind = if field.indexed { index.clone() } else { VectorIndexKind::Flat };
        let name = match opts.get("vector.index") {
            Some(k) => k.to_string(),
            None => match schema_kind {
                VectorIndexKind::Flat => "flat".into(),
                VectorIndexKind::Ivf { .. } => "ivf".into(),
                VectorIndexKind::Hnsw { .. } => "hnsw".into(),
            },
        };
        let (schema_nlist, schema_m, schema_ef) = match schema_kind {
            VectorIndexKind::Ivf { nlist } => (Some(nlist).filter(|&n| n > 0), 16, 200),
            VectorIndexKind::Hnsw { m, ef_construction } => (None, m, ef_construction),
            VectorIndexKind::Flat => (None, 16, 200),
        };
        let kind = match name.as_str() {
            "flat" => Kind::Flat,
            "ivf" => Kind::Ivf {
                nlist: parse(opts, "vector.ivf.nlist")?.or(schema_nlist),
                iters: parse(opts, "vector.ivf.iters")?.unwrap_or(10),
                sample: parse(opts, "vector.ivf.sample")?.unwrap_or(64),
            },
            "hnsw" => Kind::Hnsw {
                m: parse(opts, "vector.hnsw.m")?.unwrap_or(schema_m),
                ef_construction: parse(opts, "vector.hnsw.ef_construction")?.unwrap_or(schema_ef),
                f32: match opts.get("vector.hnsw.build") {
                    None | Some("sq8") => false,
                    Some("f32") => true,
                    Some(b) => return Err(Error::Schema(format!("bad value {b:?} for build option vector.hnsw.build (sq8 or f32)"))),
                },
            },
            other => return Err(Error::Schema(format!("unknown vector index {other:?}"))),
        };
        Ok(VectorBuilder {
            field: field.key(),
            dim: *dim,
            stride: strides(*dim).0,
            kind,
            codes: match opts.get("vector.codes") {
                None => field.codes,
                Some(c) => VectorCodes::parse(c).ok_or_else(|| Error::Schema(format!("unknown vector codes {c:?}")))?,
            },
            hnsw_max_docs: parse(opts, "vector.hnsw.max_docs")?,
            threads: parse(opts, "vector.threads")?.unwrap_or(1),
            seed: parse(opts, "vector.seed")?.unwrap_or(0x5EE_D0F5_747A),
            start: Vec::new(),
            rows: Vec::new(),
            present: Vec::new(),
            n: 0,
            missing: 0,
            multi: field.is_multi_vector().then(|| (vec![0], field.keep_f32)),
        })
    }
}

impl VectorBuilder {
    /// Starts the field's HNSW graph from the graphs of `segs`, whose
    /// documents are this build's from each one's docID `base` on, all of
    /// them with the same vectors: the first that fits keeps its links and
    /// the others are joined to it.
    pub fn start_from(mut self, segs: Vec<(SegmentFile, u32)>) -> Self {
        self.start = segs;
        self
    }
}

/// `seg`'s HNSW graph of `field` and its node count, if it has one of
/// `dim` dimensions.
fn start_graph(seg: &SegmentFile, field: &str, dim: usize) -> Option<(HnswView, usize)> {
    let info: VectorInfo = serde_json::from_slice(&seg.section(&section(field, "meta")).ok()??).ok()?;
    if info.index != "hnsw" || info.dim != dim || info.n != seg.n_docs() {
        return None;
    }
    let g = HnswView::open(info.m?, info.entry?, seg.section(&section(field, "hnsw")).ok()??, seg.section(&section(field, "hnsw_upper")).ok()??, info.n).ok()?;
    Some((g, info.n))
}

impl FieldIndexBuilder for VectorBuilder {
    fn add(&mut self, doc: DocId, value: &Value) -> Result<()> {
        if let Some((offsets, _)) = &self.multi {
            if doc as usize + 1 != offsets.len() {
                return Err(Error::Schema(format!("vector field {}: document {doc} out of order", self.field)));
            }
            for v in value.items() {
                match v {
                    Value::Vector(v) if v.len() == self.dim => {
                        if self.n.is_multiple_of(64) {
                            self.present.push(0);
                        }
                        self.rows.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                        self.rows.resize(self.rows.len() + (self.stride - self.dim) * 4, 0);
                        self.present[self.n / 64] |= 1 << (self.n % 64);
                        self.n += 1;
                    }
                    _ => return Err(Error::Schema(format!("vector field {}: bad value for document {doc}", self.field))),
                }
            }
            let n = u32::try_from(self.n).map_err(|_| Error::Schema(format!("vector field {}: more than 2^32 vectors in a segment", self.field)))?;
            self.multi.as_mut().expect("multi").0.push(n);
            return Ok(());
        }
        if doc as usize != self.n {
            return Err(Error::Schema(format!("vector field {}: document {doc} out of order", self.field)));
        }
        if self.n.is_multiple_of(64) {
            self.present.push(0);
        }
        match value {
            Value::Vector(v) if v.len() == self.dim => {
                self.rows.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                self.present[self.n / 64] |= 1 << (self.n % 64);
            }
            Value::Null => {
                self.rows.resize(self.rows.len() + self.dim * 4, 0);
                self.missing += 1;
            }
            _ => return Err(Error::Schema(format!("vector field {}: bad value for document {doc}", self.field))),
        }
        self.rows.resize(self.rows.len() + (self.stride - self.dim) * 4, 0);
        self.n += 1;
        Ok(())
    }

    fn finish(self: Box<Self>, out: &mut SegmentWriter) -> Result<()> {
        let VectorBuilder { field, dim, stride, kind, codes: code_kind, hnsw_max_docs, threads, seed, start, rows, present, n, missing, multi } = *self;
        let code_stride = code_kind.stride(dim);
        let is_present = |d: u32| present[d as usize / 64] >> (d % 64) & 1 == 1;
        let ids: Vec<u32> = (0..n as u32).filter(|&d| is_present(d)).collect();
        let kind = match kind {
            Kind::Hnsw { .. } if hnsw_max_docs.is_some_and(|m| ids.len() > m) => Kind::Ivf { nlist: None, iters: 10, sample: 64 },
            // A multi-vector segment of NULL rows has no vectors to index.
            _ if multi.is_some() && n == 0 => Kind::Flat,
            k => k,
        };
        // A merge's graph is a graph of rows, not of a multi-vector field's vectors.
        let start = if multi.is_some() { Vec::new() } else { start };
        // SAFETY: f32 has no invalid bit patterns; `align_to` only yields the
        // aligned middle part, and we require it to be everything.
        let (head, data, tail) = unsafe { rows.align_to::<f32>() };
        assert!(head.is_empty() && tail.is_empty(), "allocator returned a misaligned vector buffer");
        let view = Rows { data, dim, stride, n };

        let present_rows = ids.iter().map(|&d| &view.row(d as usize)[..dim]);
        let quant = match code_kind {
            VectorCodes::Sq8 => Quantiser::Sq8(Sq8::fit(dim, present_rows)),
            VectorCodes::Binary => Quantiser::Bits(Bits::fit(dim, present_rows)),
        };
        let mut codes = vec![0u8; n * code_stride];
        for &d in &ids {
            let d = d as usize;
            quant.encode(&view.row(d)[..dim], &mut codes[d * code_stride..(d + 1) * code_stride]);
        }

        let mut info = VectorInfo {
            dim,
            n,
            missing,
            index: "flat".into(),
            stride_f32: stride,
            stride_u8: code_stride,
            codes: code_kind,
            nlist: None,
            train_points: None,
            iters: None,
            m: None,
            ef_construction: None,
            entry: None,
            max_level: None,
            rows: multi.as_ref().map(|(o, _)| o.len() - 1),
            f32: multi.as_ref().is_none_or(|&(_, keep)| keep),
        };
        let mut index_sections: Vec<(String, Vec<u8>)> = Vec::new();
        match kind {
            Kind::Flat => {}
            Kind::Ivf { nlist, iters, sample } => {
                let cfg = IvfConfig { nlist: nlist.unwrap_or_else(|| ivf::default_nlist(ids.len())), iters, sample_per_list: sample, threads, seed };
                let built = ivf::build(&view, &ids, &codes, code_stride, &cfg);
                info.index = "ivf".into();
                info.nlist = Some(built.nlist);
                info.train_points = Some(built.train_points);
                info.iters = Some(iters);
                index_sections.push((section(&field, "ivf_centroids"), built.centroid_bytes()));
                index_sections.push((section(&field, "ivf_dir"), built.dir_bytes()));
                index_sections.push((section(&field, "ivf"), built.lists));
            }
            Kind::Hnsw { m, ef_construction, f32 } => {
                let graphs: Vec<((HnswView, usize), u32)> = start.iter().filter_map(|(seg, base)| Some((start_graph(seg, &field, dim)?, *base))).collect();
                let starts: Vec<hnsw::StartGraph> = graphs.iter().map(|((g, n), base)| hnsw::StartGraph { view: g, base: *base, n: *n }).collect();
                let on = match &quant {
                    Quantiser::Sq8(sq8) if !f32 => Some(hnsw::BuildCodes { sq8, codes: &codes, stride: code_stride }),
                    _ => None,
                };
                let g = hnsw::build(&view, &is_present, &HnswConfig { m, ef_construction, threads, seed }, &starts, on.as_ref());
                info.index = "hnsw".into();
                info.m = Some(g.m);
                info.ef_construction = Some(ef_construction);
                info.entry = Some(g.entry);
                info.max_level = Some(g.max_level);
                index_sections.push((section(&field, "hnsw_upper"), g.upper_bytes()));
                index_sections.push((section(&field, "hnsw"), g.layer0_bytes()));
            }
        }

        let mut p = ByteWriter::new();
        let (codes_part, params_part) = match &quant {
            Quantiser::Sq8(sq) => {
                p.array(&sq.min);
                p.array(&sq.scale);
                ("sq8", "sq8p")
            }
            Quantiser::Bits(b) => {
                p.array(&b.mean);
                p.array(&b.spread);
                ("bin", "binp")
            }
        };
        out.put(section(&field, "meta"), serde_json::to_vec(&info).expect("serialisable"));
        out.put(section(&field, params_part), p.finish());
        if missing > 0 {
            out.put(section(&field, "present"), present.iter().flat_map(|w| w.to_le_bytes()).collect());
        }
        for (name, bytes) in index_sections {
            out.put(name, bytes);
        }
        out.put(section(&field, codes_part), codes);
        if let Some((offsets, _)) = &multi {
            let mut owner = Vec::with_capacity(n * 4);
            for (r, w) in offsets.windows(2).enumerate() {
                for _ in w[0]..w[1] {
                    owner.extend((r as u32).to_le_bytes());
                }
            }
            out.put(section(&field, "rows"), offsets.iter().flat_map(|o| o.to_le_bytes()).collect());
            out.put(section(&field, "owner"), owner);
            // Rows are scored exactly with SQ8 codes when binary codes are
            // searched and the float32 vectors are not kept.
            if !info.f32 && code_kind == VectorCodes::Binary {
                let sq = Sq8::fit(dim, (0..n).map(|d| &view.row(d)[..dim]));
                let s = strides(dim).1;
                let mut c = vec![0u8; n * s];
                for d in 0..n {
                    sq.encode(&view.row(d)[..dim], &mut c[d * s..(d + 1) * s]);
                }
                let mut p = ByteWriter::new();
                p.array(&sq.min);
                p.array(&sq.scale);
                out.put(section(&field, "rescore"), c);
                out.put(section(&field, "rescorep"), p.finish());
            }
        }
        if info.f32 {
            out.put(section(&field, "f32"), rows);
        }
        Ok(())
    }
}
