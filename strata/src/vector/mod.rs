//! Vector storage and ANN search (DESIGN.md §5.3).
//!
//! Every vector field stores, per segment:
//!
//! | section | contents |
//! |---|---|
//! | `vec.<f>.meta` | JSON header ([`VectorInfo`]) |
//! | `vec.<f>.f32` | float32 vectors, `n × stride` (rows padded to 16 floats) |
//! | `vec.<f>.sq8` | SQ8 codes, `n × code_stride` (rows padded to 64 bytes) |
//! | `vec.<f>.sq8p` | per-dimension SQ8 minimum and scale |
//! | `vec.<f>.bin` | binary codes instead of SQ8 ones, when the field asks for them (rows padded to 8 bytes) |
//! | `vec.<f>.binp` | per-dimension threshold and spread of the binary codes |
//! | `vec.<f>.present` | bitset of documents with a vector (only if some lack one) |
//!
//! plus the sections of its index: IVF (`ivf_centroids`, `ivf_dir`, `ivf`) or
//! HNSW (`hnsw`, `hnsw_upper`); see [`ivf`] and [`hnsw`] for the layouts.
//! [`VectorReader`] views all of them in place. A multi-vector field
//! (`VECTOR(d)[]`) has the same sections over its vectors rather than its
//! rows, and maps between the two ([`multi`]).
//!
//! **Search** scores SQ8 codes with an int8-quantised query (AVX-512 VNNI or
//! NEON `usdot`), keeps the best `rerank` candidates and re-ranks them with
//! float32. A field with binary codes ([`crate::schema::VectorCodes`]) is
//! searched the same way over a bit per dimension ([`bits`]) and keeps
//! [`codes::BINARY_RERANK`]` × rerank` candidates. Only the codes (and the
//! index) need to stay in memory: the float32 rows are read for the
//! candidates alone, so they can stay on disk. The [`Strategy`] decides which
//! codes are scored:
//!
//! * `Flat`: every code (a pre-filter scan when there is a filter).
//! * `Ivf`: the clusters closest to the query; with a filter, each probed
//!   cluster's docIDs are intersected with it before scoring.
//! * `Hnsw`: a graph walk; with a filter, non-members are traversed but not
//!   collected.
//! * `PreFilter`: only the filter's members (exact up to SQ8 ties).
//! * `PostFilter`: an unfiltered index search for `k·margin/selectivity`
//!   candidates, then the filter; retried with 4× more if short.
//! * `Auto`: the cheapest by the [`CostModel`].

pub mod bits;
pub mod codes;
pub mod cost;
pub mod exact;
pub mod kernels;
pub mod multi;
pub mod sq8;

mod build;
mod hnsw;
mod io;
mod ivf;
mod topk;

use crate::postings::bytes::{ByteReader, Slice};
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use crate::error::{corrupt, Result};
use crate::schema::{Field, VectorCodes};
use crate::value::Value;
use crate::segment::build::BuildOptions;
use crate::segment::{DocFilter, DocId, FieldIndexBuilder, SegmentFile};

pub use cost::{CostModel, IndexShape, Work};
pub use ivf::{default_nlist, filtered_probe_weight, filtered_probes};

use hnsw::HnswView;
use ivf::IvfView;
use bits::Bits;
use codes::{Quantiser, QueryCode, BINARY_RERANK};
use kernels::{dot_f32, prefetch};
use sq8::Sq8;
use topk::{Members, TopK};

/// Builder for a VECTOR field (see `build.rs` for the build options).
pub fn field_builder(field: &Field, opts: &BuildOptions) -> Result<Box<dyn FieldIndexBuilder>> {
    Ok(Box::new(build::VectorBuilder::new(field, opts)?))
}

/// [`field_builder`] whose HNSW graph, if it builds one, starts from
/// `seg`'s: `seg`'s documents are this build's from docID `base` on, all of
/// them, with the same vectors (a merge's input that lost none).
pub fn field_builder_from(field: &Field, opts: &BuildOptions, seg: SegmentFile, base: u32) -> Result<Box<dyn FieldIndexBuilder>> {
    Ok(Box::new(build::VectorBuilder::new(field, opts)?.start_from(seg, base)))
}

/// Name of section `part` of vector field `field`.
pub fn section(field: &str, part: &str) -> String {
    format!("vec.{field}.{part}")
}

/// The JSON header of a vector field's sections.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorInfo {
    pub dim: usize,
    /// Documents in the segment (with or without a vector).
    pub n: usize,
    /// Documents without a vector.
    pub missing: usize,
    /// `"flat"`, `"ivf"` or `"hnsw"`.
    pub index: String,
    /// Floats per stored vector row.
    pub stride_f32: usize,
    /// Bytes per code row.
    pub stride_u8: usize,
    /// The codes kept (SQ8 unless the field asks for binary ones).
    #[serde(default, skip_serializing_if = "is_sq8")]
    pub codes: VectorCodes,
    pub nlist: Option<usize>,
    pub train_points: Option<usize>,
    pub iters: Option<usize>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub entry: Option<u32>,
    pub max_level: Option<usize>,
    /// A multi-vector field's rows; `n` then counts its vectors
    /// ([`multi`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<usize>,
    /// Whether section `f32` holds the float32 vectors: always for a
    /// one-vector field, with `keep_f32` for a multi-vector one.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub f32: bool,
}

fn is_sq8(c: &VectorCodes) -> bool {
    *c == VectorCodes::Sq8
}

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

/// A query prepared for exact scores: float32, or the SQ8 codes that stand
/// for vectors a multi-vector field does not keep in float32.
pub enum ExactQuery {
    F32(Vec<f32>),
    Sq8(sq8::Sq8Query),
}

/// One search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc: DocId,
    /// Exact float32 dot product with the query.
    pub score: f32,
}

/// How a search finds its candidates (see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Strategy {
    #[default]
    Auto,
    Flat,
    Ivf,
    Hnsw,
    PreFilter,
    PostFilter,
}

impl Strategy {
    pub const ALL: [Strategy; 6] = [Strategy::Auto, Strategy::Flat, Strategy::Ivf, Strategy::Hnsw, Strategy::PreFilter, Strategy::PostFilter];

    pub fn as_str(&self) -> &'static str {
        match self {
            Strategy::Auto => "auto",
            Strategy::Flat => "flat",
            Strategy::Ivf => "ivf",
            Strategy::Hnsw => "hnsw",
            Strategy::PreFilter => "prefilter",
            Strategy::PostFilter => "postfilter",
        }
    }

    pub fn parse(s: &str) -> Option<Strategy> {
        Strategy::ALL.into_iter().find(|x| x.as_str().eq_ignore_ascii_case(s))
    }
}

/// Search knobs. `rerank` is the number of candidates re-ranked with
/// float32 (at least `k`; times [`BINARY_RERANK`] over binary codes); HNSW
/// searches with `max(ef, rerank)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub strategy: Strategy,
    pub nprobe: usize,
    pub ef: usize,
    pub rerank: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams { strategy: Strategy::Auto, nprobe: 32, ef: 64, rerank: 40 }
    }
}

/// What a search did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchStats {
    /// The strategy that ran (never `Auto`).
    pub strategy: Strategy,
    /// Codes scored (SQ8 or binary).
    pub sq8_distances: usize,
    /// Of those, codes scored in contiguous runs (flat scan, unfiltered IVF
    /// clusters) rather than at scattered positions.
    pub sq8_contiguous: usize,
    /// Float32 re-rank dot products.
    pub f32_distances: usize,
    /// IVF centroids scored.
    pub centroids: usize,
    /// Candidates handed to the float32 re-ranking.
    pub candidates: usize,
    pub clusters_probed: usize,
    /// HNSW nodes expanded on layer 0.
    pub nodes_visited: usize,
    /// Filter membership tests.
    pub filter_checks: usize,
    /// Post-filter searches repeated with more candidates.
    pub retries: usize,
    /// The cost model's prediction for the chosen strategy (µs).
    pub estimated_us: f64,
}

/// Hits (best first) and statistics.
#[derive(Clone, Debug, Default)]
pub struct SearchOutput {
    pub hits: Vec<Hit>,
    pub stats: SearchStats,
}

/// How an in-filter HNSW search moves through non-members (see
/// [`VectorReader::set_hnsw_filter`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HnswFilterMode {
    /// Score and expand every reached node; collect members only. Recall
    /// holds at any selectivity; work grows like `1/selectivity`.
    #[default]
    Traverse,
    /// Score and expand members only, looking through non-members to their
    /// neighbours (ACORN-1). Cheaper, but on an ordinary `M = 16` graph
    /// recall collapses below ~10% selectivity or when the members lie away
    /// from the query (see `notes/vector.md`).
    TwoHop,
}

/// Code rows by docID.
pub(crate) struct Codes<'a> {
    data: &'a [u8],
    stride: usize,
}

impl<'a> Codes<'a> {
    #[inline]
    pub fn row(&self, d: DocId) -> &'a [u8] {
        &self.data[d as usize * self.stride..(d as usize + 1) * self.stride]
    }
    pub fn n(&self) -> usize {
        self.data.len() / self.stride
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// Scores `count` code rows of `block` (row `pos[i]`, or row `i` without
/// `pos`) and calls `f(i, score)`. Four rows go through the kernel at once;
/// scattered rows are prefetched one group ahead.
#[inline]
pub(crate) fn score_rows(block: &[u8], stride: usize, count: usize, pos: Option<&[u32]>, q: &QueryCode, mut f: impl FnMut(usize, i32)) {
    let row = |i: usize| {
        let r = pos.map_or(i, |p| p[i] as usize);
        &block[r * stride..(r + 1) * stride]
    };
    let mut i = 0;
    while i + 4 <= count {
        if pos.is_some() {
            for j in i + 4..(i + 8).min(count) {
                prefetch(row(j).as_ptr(), stride);
            }
        }
        let s = q.dot4([row(i), row(i + 1), row(i + 2), row(i + 3)]);
        for (j, s) in s.into_iter().enumerate() {
            f(i + j, s);
        }
        i += 4;
    }
    while i < count {
        f(i, q.dot(row(i)));
        i += 1;
    }
}

/// Zero-copy reader of one vector field in one segment.
pub struct VectorReader {
    info: VectorInfo,
    vectors: Slice<f32>,
    codes: Slice<u8>,
    quant: Quantiser,
    /// SQ8 codes and ranges of exact scores without float32 vectors (the
    /// search codes when they are SQ8).
    rescore: Option<(Slice<u8>, Sq8)>,
    multi: Option<multi::Rows>,
    present: Option<Slice<u64>>,
    ivf: Option<IvfView>,
    hnsw: Option<HnswView>,
    shape: IndexShape,
    cost: CostModel,
    io_hints: bool,
    hnsw_filter: HnswFilterMode,
}

impl VectorReader {
    pub fn open(seg: &SegmentFile, field: &str) -> Result<Self> {
        let info: VectorInfo = serde_json::from_slice(&seg.require(&section(field, "meta"))?).map_err(corrupt)?;
        let (n, dim) = (info.n, info.dim);
        let rows = info.rows.unwrap_or(n);
        if (info.stride_f32, info.stride_u8) != (build::strides(dim).0, info.codes.stride(dim)) || rows != seg.n_docs() || (info.rows.is_none() && !info.f32) {
            return Err(corrupt(format!("vector field {field}: header does not match the segment")));
        }
        let (codes_part, params_part) = match info.codes {
            VectorCodes::Sq8 => ("sq8", "sq8p"),
            VectorCodes::Binary => ("bin", "binp"),
        };
        let vectors: Slice<f32> = if info.f32 { Slice::view(seg.require(&section(field, "f32"))?).map_err(corrupt)? } else { Slice::Owned(Vec::new()) };
        let codes: Slice<u8> = Slice::view(seg.require(&section(field, codes_part))?).map_err(corrupt)?;
        if (info.f32 && vectors.len() != n * info.stride_f32) || codes.len() != n * info.stride_u8 {
            return Err(corrupt(format!("vector field {field}: sections have the wrong size")));
        }
        let mut r = ByteReader::new(seg.require(&section(field, params_part))?);
        let (a, b): (Slice<f32>, Slice<f32>) = (r.array().map_err(corrupt)?, r.array().map_err(corrupt)?);
        if a.len() != dim || b.len() != dim {
            return Err(corrupt(format!("vector field {field}: bad code parameters")));
        }
        let quant = match info.codes {
            VectorCodes::Sq8 => Quantiser::Sq8(Sq8 { min: a, scale: b }),
            VectorCodes::Binary => Quantiser::Bits(Bits { mean: a, spread: b }),
        };
        let multi = info.rows.map(|r| multi::Rows::open(seg, field, r, n)).transpose()?;
        let rescore = match (&quant, info.f32) {
            (_, true) => None,
            (Quantiser::Sq8(sq), false) => Some((codes.clone(), sq.clone())),
            (Quantiser::Bits(_), false) => {
                let c: Slice<u8> = Slice::view(seg.require(&section(field, "rescore"))?).map_err(corrupt)?;
                let mut r = ByteReader::new(seg.require(&section(field, "rescorep"))?);
                let (min, scale): (Slice<f32>, Slice<f32>) = (r.array().map_err(corrupt)?, r.array().map_err(corrupt)?);
                if c.len() != n * build::strides(dim).1 || min.len() != dim || scale.len() != dim {
                    return Err(corrupt(format!("vector field {field}: bad rescoring codes")));
                }
                Some((c, Sq8 { min, scale }))
            }
        };
        let present = match seg.section(&section(field, "present"))? {
            Some(b) => {
                let p: Slice<u64> = Slice::view(b).map_err(corrupt)?;
                if p.len() != n.div_ceil(64) {
                    return Err(corrupt(format!("vector field {field}: bad presence bitset")));
                }
                Some(p)
            }
            None => None,
        };
        let (mut ivf, mut hnsw) = (None, None);
        match info.index.as_str() {
            "flat" => {}
            "ivf" => {
                let nlist = info.nlist.ok_or_else(|| corrupt("ivf without nlist"))?;
                ivf = Some(IvfView::open(
                    nlist,
                    dim,
                    info.stride_u8,
                    seg.require(&section(field, "ivf_centroids"))?,
                    seg.require(&section(field, "ivf_dir"))?,
                    seg.require(&section(field, "ivf"))?,
                )?);
            }
            "hnsw" => {
                let (m, entry) = (info.m.ok_or_else(|| corrupt("hnsw without m"))?, info.entry.ok_or_else(|| corrupt("hnsw without entry"))?);
                hnsw = Some(HnswView::open(m, entry, seg.require(&section(field, "hnsw"))?, seg.require(&section(field, "hnsw_upper"))?, n)?);
            }
            other => return Err(corrupt(format!("unknown vector index {other}"))),
        }
        let shape = IndexShape { n: n - info.missing, kind: info.index.clone(), nlist: info.nlist.unwrap_or(1), hnsw_m: info.m.unwrap_or(16) };
        Ok(VectorReader { info, vectors, codes, quant, rescore, multi, present, ivf, hnsw, shape, cost: CostModel::default(), io_hints: false, hnsw_filter: HnswFilterMode::Traverse })
    }

    pub fn dim(&self) -> usize {
        self.info.dim
    }

    pub fn n_docs(&self) -> u32 {
        self.info.n as u32
    }

    /// `"flat"`, `"ivf"` or `"hnsw"`.
    pub fn index_kind(&self) -> &str {
        &self.info.index
    }

    pub fn info(&self) -> &VectorInfo {
        &self.info
    }

    /// The index shape the cost model works from.
    pub fn shape(&self) -> &IndexShape {
        &self.shape
    }

    pub fn cost_model(&self) -> &CostModel {
        &self.cost
    }

    /// Replaces the cost model (e.g. with one calibrated for this machine).
    pub fn set_cost_model(&mut self, cost: CostModel) {
        self.cost = cost;
    }

    /// Read-ahead hints for segments mapped from a file that may not be in
    /// the page cache: before touching them, a search announces the probed
    /// IVF clusters, the code rows of HNSW neighbours and pre-filter members,
    /// and the float32 rows to re-rank (`madvise(WILLNEED)`), so a cold query
    /// issues its reads together instead of one page fault at a time. Off by
    /// default: on a warm segment the system calls cost more than they save.
    pub fn set_io_hints(&mut self, on: bool) {
        self.io_hints = on;
    }

    /// How in-filter HNSW searches move through non-members (default
    /// [`HnswFilterMode::Traverse`]).
    pub fn set_hnsw_filter(&mut self, mode: HnswFilterMode) {
        self.hnsw_filter = mode;
    }

    /// Whether document `doc` has a vector (documents without one are never
    /// returned by a search).
    pub fn has_vector(&self, doc: DocId) -> bool {
        self.present.as_ref().is_none_or(|p| p[doc as usize / 64] >> (doc % 64) & 1 == 1)
    }

    /// A multi-vector field's rows, by vector and vector by row.
    pub fn rows(&self) -> Option<&multi::Rows> {
        self.multi.as_ref()
    }

    /// Whether the float32 vectors are kept ([`VectorInfo::f32`]).
    pub fn has_f32(&self) -> bool {
        self.info.f32
    }

    /// `query` prepared for [`VectorReader::exact_score`].
    pub fn prepare_exact(&self, query: &[f32]) -> ExactQuery {
        match &self.rescore {
            None => ExactQuery::F32(query.to_vec()),
            Some((_, sq)) => ExactQuery::Sq8(sq.prepare(query, build::strides(self.info.dim).1)),
        }
    }

    /// The dot product of vector `v` with a prepared query: exact in
    /// float32, or of the SQ8 codes that stand for it.
    #[inline]
    pub fn exact_score(&self, q: &ExactQuery, v: DocId) -> f32 {
        match (q, &self.rescore) {
            (ExactQuery::F32(q), _) => dot_f32(self.vector(v), q),
            (ExactQuery::Sq8(q), Some((codes, _))) => {
                let s = build::strides(self.info.dim).1;
                q.score(q.dot(&codes[v as usize * s..(v as usize + 1) * s]))
            }
            (ExactQuery::Sq8(_), None) => unreachable!("prepared by this reader"),
        }
    }

    /// Vector `v` as stored: float32, or its SQ8 codes' values.
    pub fn vector_value(&self, v: DocId) -> Arc<[f32]> {
        match &self.rescore {
            None => self.vector(v).into(),
            Some((codes, sq)) => {
                let s = build::strides(self.info.dim).1;
                let mut out = vec![0f32; self.info.dim];
                sq.decode(&codes[v as usize * s..(v as usize + 1) * s], &mut out);
                out.into()
            }
        }
    }

    /// Row `d`'s value: its vector, a multi-vector field's list of vectors,
    /// or NULL.
    pub fn row_value(&self, d: DocId) -> Value {
        match &self.multi {
            Some(rows) => {
                let vs = rows.vectors(d);
                if vs.is_empty() {
                    Value::Null
                } else {
                    Value::List(vs.map(|v| Value::Vector(self.vector_value(v))).collect())
                }
            }
            None if self.has_vector(d) => Value::Vector(self.vector(d).into()),
            None => Value::Null,
        }
    }

    /// The stored vector of `doc` (zeros if it has none).
    pub fn vector(&self, doc: DocId) -> &[f32] {
        let s = self.info.stride_f32;
        &self.vectors[doc as usize * s..doc as usize * s + self.info.dim]
    }

    /// Exact dot product of `doc`'s vector with `query`.
    pub fn similarity(&self, doc: DocId, query: &[f32]) -> f32 {
        dot_f32(self.vector(doc), query)
    }

    /// The codes of `doc` (padded to the code stride).
    pub fn codes(&self, doc: DocId) -> &[u8] {
        Codes { data: &self.codes, stride: self.info.stride_u8 }.row(doc)
    }

    /// The SQ8 quantiser (per-dimension minimum and scale), if the field
    /// keeps SQ8 codes.
    pub fn sq8(&self) -> Option<&Sq8> {
        match &self.quant {
            Quantiser::Sq8(s) => Some(s),
            Quantiser::Bits(_) => None,
        }
    }

    /// The field's quantiser.
    pub fn quantiser(&self) -> &Quantiser {
        &self.quant
    }

    /// Bytes of codes, the part of the field a search reads in full.
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// True when every array is a view of the segment's bytes (no copies).
    pub fn is_zero_copy(&self) -> bool {
        (matches!(self.vectors, Slice::Shared { .. }) || !self.info.f32)
            && matches!(self.codes, Slice::Shared { .. })
            && self.multi.as_ref().is_none_or(|m| m.is_shared())
            && match &self.quant {
                Quantiser::Sq8(s) => matches!(s.min, Slice::Shared { .. }),
                Quantiser::Bits(b) => matches!(b.mean, Slice::Shared { .. }),
            }
            && self.present.as_ref().is_none_or(|p| matches!(p, Slice::Shared { .. }))
            && self.ivf.as_ref().is_none_or(|v| v.is_shared())
            && self.hnsw.as_ref().is_none_or(|v| v.is_shared())
    }

    /// Predicted latency in microseconds of `strategy` for a top-`k` search
    /// with a filter of `filter_cardinality` members (`None`: unfiltered).
    /// `Auto` gives the cheapest applicable strategy's estimate.
    pub fn estimate_cost(&self, strategy: Strategy, k: usize, filter_cardinality: Option<usize>, params: &SearchParams) -> f64 {
        match strategy {
            Strategy::Auto => self.cost.choose(&self.shape, k, filter_cardinality, params).1,
            s => self.cost.estimate(&self.shape, s, k, filter_cardinality, params),
        }
    }

    /// The strategy a search with these inputs runs.
    pub fn plan(&self, k: usize, filter_cardinality: Option<usize>, params: &SearchParams) -> Strategy {
        match cost::resolve(&self.shape, params.strategy, filter_cardinality.is_some()) {
            Strategy::Auto => self.cost.choose(&self.shape, k, filter_cardinality, params).0,
            s => s,
        }
    }

    /// The top-`k` documents by similarity to `query`, best first.
    pub fn search(&self, query: &[f32], k: usize, params: &SearchParams, filter: Option<&dyn DocFilter>) -> SearchOutput {
        assert_eq!(query.len(), self.info.dim, "query dimension");
        // No search returns more rows than the segment holds, and every
        // size derived from k (heaps, rerank pools) stays within it.
        let k = k.min(self.shape.n);
        let clamped;
        let params = if params.rerank > self.shape.n || params.ef > self.shape.n {
            clamped = SearchParams { rerank: params.rerank.min(self.shape.n), ef: params.ef.min(self.shape.n.max(1)), ..params.clone() };
            &clamped
        } else {
            params
        };
        let card = filter.map(|f| f.cardinality());
        let strategy = self.plan(k, card, params);
        let mut stats = SearchStats { strategy, estimated_us: self.cost.estimate(&self.shape, strategy, k, card, params), ..Default::default() };
        if k == 0 || card == Some(0) || self.shape.n == 0 {
            return SearchOutput { hits: Vec::new(), stats };
        }
        let mut qf = query.to_vec();
        qf.resize(self.info.stride_f32, 0.0);
        let qc = self.quant.prepare(query, self.info.stride_u8);
        let r = params.rerank.max(k) * self.oversample();
        let members = filter.map(Members::of);
        // IVF and HNSW test far more documents than a sparse filter has
        // members: probe a bitset rather than search a sorted list.
        let bits: Vec<u64>;
        let members = match (strategy, members) {
            (Strategy::Ivf | Strategy::Hnsw, Some(Members::Sorted(docs))) => {
                let mut b = vec![0u64; self.info.n.div_ceil(64)];
                for &d in docs {
                    b[d as usize / 64] |= 1 << (d % 64);
                }
                bits = b;
                Some(Members::Bits(&bits))
            }
            (_, m) => m,
        };
        let s = card.map_or(1.0, |c| c as f64 / self.shape.n as f64);
        let cands = match strategy {
            Strategy::Flat => self.scan(&qc, r, None, &mut stats),
            Strategy::PreFilter => self.scan(&qc, r, members, &mut stats),
            Strategy::Ivf => self.ivf.as_ref().expect("planned").search(&qc, &qf, r, params.nprobe, members.map(|m| (m, s)), self.io_hints, &mut stats),
            Strategy::Hnsw => self.hnsw_search(&qc, params.ef.max(r), r, members, &mut stats),
            Strategy::PostFilter => self.post_filter(&qc, &qf, k, r, params, members.expect("filtered"), s, &mut stats),
            Strategy::Auto => unreachable!("planned"),
        };
        stats.candidates = cands.len();
        let hits = self.rerank(query, &cands, k, &mut stats);
        SearchOutput { hits, stats }
    }

    /// Exact top-`k` by float32 over every document with a vector (or the
    /// filter's members): the reference for recall.
    pub fn exact(&self, query: &[f32], k: usize, filter: Option<&dyn DocFilter>) -> Vec<Hit> {
        let q = self.prepare_exact(query);
        let mut all: Vec<Hit> = (0..self.n_docs())
            .filter(|&d| self.has_vector(d) && filter.is_none_or(|f| f.contains(d)))
            .map(|d| Hit { doc: d, score: self.exact_score(&q, d) })
            .collect();
        sort_hits(&mut all);
        all.truncate(k);
        all
    }

    /// How many times more candidates the codes need rescored: 1 for SQ8,
    /// [`BINARY_RERANK`] for binary codes.
    fn oversample(&self) -> usize {
        match self.info.codes {
            VectorCodes::Sq8 => 1,
            VectorCodes::Binary => BINARY_RERANK,
        }
    }

    fn code_rows(&self) -> Codes<'_> {
        Codes { data: &self.codes, stride: self.info.stride_u8 }
    }

    /// Code scan of every document, or of a filter's members.
    fn scan(&self, q: &QueryCode, r: usize, members: Option<Members>, stats: &mut SearchStats) -> Vec<(i32, DocId)> {
        let mut top = TopK::new(r);
        let n = self.info.n as u32;
        let stride = self.info.stride_u8;
        let present = self.present.as_deref();
        let keep = |d: DocId| present.is_none_or(|p| p[d as usize / 64] >> (d % 64) & 1 == 1);
        match members {
            None => {
                // Blocks of 1 MiB; with hints, the next blocks are announced
                // while one is scored (a single hint for the whole section
                // is far slower: the kernel reads it page by page).
                const BLOCK: usize = 1 << 20;
                let rows = (BLOCK / stride).max(1);
                let ahead = |b: usize| {
                    let lo = (b * rows * stride).min(self.codes.len());
                    io::will_need_range(self.codes[lo..].as_ptr(), (rows * stride).min(self.codes.len() - lo));
                };
                for b in 0..(n as usize).div_ceil(rows) {
                    if self.io_hints {
                        if b == 0 {
                            ahead(0);
                        }
                        ahead(b + 1);
                        ahead(b + 2);
                    }
                    let lo = b * rows;
                    let cnt = rows.min(n as usize - lo);
                    score_rows(&self.codes[lo * stride..], stride, cnt, None, q, |i, s| {
                        let d = (lo + i) as DocId;
                        if keep(d) {
                            top.push(s, d)
                        }
                    });
                }
                stats.sq8_distances += n as usize;
                stats.sq8_contiguous += n as usize;
            }
            Some(m) => {
                if matches!(m, Members::Probe(_)) {
                    stats.filter_checks += n as usize;
                }
                m.for_each_chunk(n, |chunk| {
                    if self.io_hints {
                        let base = self.codes.as_ptr() as usize;
                        io::will_need(&mut chunk.iter().map(|&d| (base + d as usize * stride, stride)).collect::<Vec<_>>());
                    }
                    score_rows(&self.codes, stride, chunk.len(), Some(chunk), q, |i, s| {
                        if keep(chunk[i]) {
                            top.push(s, chunk[i])
                        }
                    });
                    stats.sq8_distances += chunk.len();
                });
            }
        }
        top.into_sorted()
    }

    /// HNSW search over the members of `members`.
    fn hnsw_search(&self, q: &QueryCode, ef: usize, r: usize, members: Option<Members>, stats: &mut SearchStats) -> Vec<(i32, DocId)> {
        let two_hop = self.hnsw_filter == HnswFilterMode::TwoHop;
        let mut c = self.hnsw.as_ref().expect("planned").search(&self.code_rows(), q, ef, members, two_hop, self.io_hints, stats);
        c.truncate(r);
        c
    }

    /// Unfiltered search for `k·margin/s` candidates, then the filter; the
    /// candidate count grows 4× until `k` members survive or every document
    /// was a candidate. An HNSW walk goes on from where it stopped instead of
    /// starting over (pgvector's iterative scan).
    #[allow(clippy::too_many_arguments)]
    fn post_filter(&self, q: &QueryCode, qf: &[f32], k: usize, r: usize, p: &SearchParams, m: Members, s: f64, stats: &mut SearchStats) -> Vec<(i32, DocId)> {
        let n = self.shape.n;
        // Binary codes order candidates more loosely: keep as many members
        // as they rescore.
        let k = k * self.oversample();
        let mut kp = ((k as f64 * self.cost.post_margin / s).ceil() as usize).max(r).min(n);
        let mut walk = hnsw::Walk::default();
        loop {
            let cands = match self.info.index.as_str() {
                "hnsw" => {
                    let mut c = self.hnsw.as_ref().expect("hnsw").walk(&mut walk, &self.code_rows(), q, p.ef.max(kp), None, false, self.io_hints, stats);
                    c.truncate(kp);
                    c
                }
                "ivf" => self.ivf.as_ref().expect("ivf").search(q, qf, kp, cost::post_filter_nprobe(p.nprobe, kp, r), None, self.io_hints, stats),
                _ => self.scan(q, kp, None, stats),
            };
            stats.filter_checks += cands.len();
            let mut kept: Vec<(i32, DocId)> = cands.into_iter().filter(|&(_, d)| m.contains(d)).collect();
            if kept.len() >= k || kp >= n {
                kept.truncate(r);
                return kept;
            }
            kp = (kp * 4).min(n);
            stats.retries += 1;
        }
    }

    /// Float32 scores of the candidates (best code score first), best `k`
    /// kept.
    fn rerank(&self, query: &[f32], cands: &[(i32, DocId)], k: usize, stats: &mut SearchStats) -> Vec<Hit> {
        if !self.info.f32 {
            // A multi-vector field's SQ8 codes stand for its vectors.
            let q = self.prepare_exact(query);
            let mut hits: Vec<Hit> = cands.iter().map(|&(_, d)| Hit { doc: d, score: self.exact_score(&q, d) }).collect();
            stats.f32_distances += hits.len();
            sort_hits(&mut hits);
            hits.truncate(k);
            return hits;
        }
        let row_bytes = self.info.stride_f32 * 4;
        if self.io_hints {
            io::will_need(&mut cands.iter().map(|&(_, d)| (self.vector(d).as_ptr() as usize, row_bytes)).collect::<Vec<_>>());
        }
        let mut hits: Vec<Hit> = Vec::with_capacity(cands.len());
        for (i, &(_, d)) in cands.iter().enumerate() {
            if let Some(&(_, next)) = cands.get(i + 1) {
                prefetch(self.vector(next).as_ptr() as *const u8, row_bytes);
            }
            hits.push(Hit { doc: d, score: self.similarity(d, query) });
        }
        stats.f32_distances += hits.len();
        sort_hits(&mut hits);
        hits.truncate(k);
        hits
    }
}

/// Best first; ties to the smaller docID.
pub fn sort_hits(hits: &mut [Hit]) {
    hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
}
