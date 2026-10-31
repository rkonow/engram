//! Vector storage and ANN search experiments (DESIGN.md §8, items 4 and 5).
//!
//! For one BEIR collection:
//!
//! 1. exact float32 top-100 per query (cached next to the data), and nDCG@10
//!    of exact dense retrieval;
//! 2. kernel speed (float32, SQ8 with VNNI or NEON dot products, SQ8
//!    asymmetric) and the ranking quality of the two SQ8 scorings;
//! 3. per index kind (flat, IVF, HNSW): build time, section sizes, recall@10
//!    against latency (re-rank, `nprobe` and `ef` sweeps), and cold-cache
//!    latency with pages read (segment file mapped, evicted before each query);
//! 4. the cost model fitted to the measured runs;
//! 5. the filtered-search study: every strategy and `Auto` at selectivities
//!    from 0.1% to 100%, with random filters (`rating` ranges, independent of
//!    the query) and `topic` filters (the query's nearest topics, correlated).
//!
//! Writes `results/vector/<dataset>[-tag].json` and prints a summary.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use memmap2::Mmap;
use strata::postings::bytes::OwnedBytes;
use serde::Deserialize;
use serde_json::{json, Value as Json};
use strata::metrics::{mean, ndcg_at_k, percentile, recall};
use strata::segment::build::BuildOptions;
use strata::segment::{DocFilter, DocSet, SegmentFile, SegmentWriter};
use strata::vector::exact::exact_top_k;
use strata::vector::{kernels, CostModel, Hit, HnswFilterMode, SearchParams, SearchStats, Strategy, VectorReader, Work};
use strata::{data, iostat, Field, FieldType, Value, VectorIndexKind};

#[derive(Parser)]
#[command(about = "Vector storage and ANN search experiments")]
struct Args {
    /// BEIR collection under the data directory.
    #[arg(long, default_value = "fiqa")]
    dataset: String,
    /// Keep only the first N documents.
    #[arg(long)]
    limit: Option<usize>,
    /// Queries for the latency curves (the first N).
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    /// Threads for index building and ground truth (searches run on one).
    #[arg(long, default_value_t = 2)]
    threads: usize,
    /// Index kinds to build, comma-separated.
    #[arg(long, default_value = "flat,ivf,hnsw")]
    kinds: String,
    /// IVF lists (default ≈ 4·√n).
    #[arg(long)]
    nlist: Option<usize>,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    /// What HNSW construction scores with: `sq8` codes or `f32` rows.
    #[arg(long, default_value = "sq8")]
    hnsw_build: String,
    #[arg(long, default_value = "1,2,4,8,16,32,64,128")]
    nprobe: String,
    #[arg(long, default_value = "10,16,24,32,48,64,96,128,256")]
    ef: String,
    #[arg(long, default_value = "10,20,40,80,160")]
    rerank: String,
    /// Operating points (cold cache, filtered study).
    #[arg(long, default_value_t = 32)]
    default_nprobe: usize,
    #[arg(long, default_value_t = 64)]
    default_ef: usize,
    #[arg(long, default_value_t = 40)]
    default_rerank: usize,
    /// Queries timed with a cold page cache per operating point (0: skip).
    #[arg(long, default_value_t = 30)]
    cold_queries: usize,
    /// Queries in the filtered study (0: skip).
    #[arg(long, default_value_t = 200)]
    filter_queries: usize,
    #[arg(long, default_value = "0.001,0.003,0.01,0.03,0.1,0.3,1")]
    selectivities: String,
    /// Time budget per (strategy, filter) cell of the filtered study; the
    /// cell reports the queries that ran within it.
    #[arg(long, default_value_t = 20.0)]
    budget_secs: f64,
    /// Queries for the SQ8 ranking-quality comparison.
    #[arg(long, default_value_t = 100)]
    sq8_queries: usize,
    /// Where segment files are written (deleted afterwards unless --keep).
    #[arg(long)]
    workdir: Option<PathBuf>,
    #[arg(long)]
    keep: bool,
    /// Reuse segment files kept by an earlier run (with --keep) instead of
    /// building; build figures are then absent.
    #[arg(long)]
    reuse: bool,
    #[arg(long, default_value = "results/vector")]
    out: PathBuf,
    /// Suffix for the output file name.
    #[arg(long)]
    tag: Option<String>,
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse().ok().expect("bad list element")).collect()
}

/// A collection mapped from its prepared files (vectors are not copied, so
/// Quora fits next to its index).
struct Dataset {
    name: String,
    n: usize,
    dim: usize,
    docs: Mmap,
    doc_ids: Vec<String>,
    rating: Vec<f32>,
    topic: Vec<u16>,
    queries: Vec<Vec<f32>>,
    query_ids: Vec<String>,
    qrels: data::Qrels,
}

#[derive(Deserialize)]
struct RawDoc {
    id: String,
    rating: f32,
    topic: String,
}

#[derive(Deserialize)]
struct RawQuery {
    id: String,
}

#[derive(Deserialize)]
struct Meta {
    docs: usize,
    queries: usize,
    dim: usize,
}

impl Dataset {
    fn load(name: &str, limit: Option<usize>) -> Dataset {
        let dir = data::data_dir().join(name);
        let meta: Meta = serde_json::from_reader(File::open(dir.join("meta.json")).unwrap()).unwrap();
        let n = limit.map_or(meta.docs, |l| l.min(meta.docs));
        // SAFETY: the prepared files are not modified while mapped.
        let docs = unsafe { Mmap::map(&File::open(dir.join("doc_vecs.f32")).unwrap()).unwrap() };
        assert!(docs.len() >= n * meta.dim * 4);
        let (mut doc_ids, mut rating, mut topic) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for line in BufReader::new(File::open(dir.join("docs.jsonl")).unwrap()).lines().take(n) {
            let r: RawDoc = serde_json::from_str(&line.unwrap()).unwrap();
            doc_ids.push(r.id);
            rating.push(r.rating);
            topic.push(r.topic.trim_start_matches('t').parse::<u16>().unwrap());
        }
        let mut qbytes = Vec::new();
        File::open(dir.join("query_vecs.f32")).unwrap().read_to_end(&mut qbytes).unwrap();
        let queries: Vec<Vec<f32>> =
            qbytes.chunks_exact(meta.dim * 4).take(meta.queries).map(|c| c.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()).collect();
        let query_ids: Vec<String> = BufReader::new(File::open(dir.join("queries.jsonl")).unwrap())
            .lines()
            .map(|l| serde_json::from_str::<RawQuery>(&l.unwrap()).unwrap().id)
            .collect();
        let mut qrels: data::Qrels = HashMap::new();
        for line in BufReader::new(File::open(dir.join("qrels.tsv")).unwrap()).lines() {
            let line = line.unwrap();
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 3 {
                if let Ok(g) = f[2].trim().parse::<u32>() {
                    qrels.entry(f[0].to_string()).or_default().insert(f[1].to_string(), g);
                }
            }
        }
        Dataset { name: name.into(), n, dim: meta.dim, docs, doc_ids, rating, topic, queries, query_ids, qrels }
    }

    fn doc(&self, d: usize) -> &[f32] {
        // SAFETY: the mapping is page aligned and holds n·dim floats.
        let all: &[f32] = unsafe { std::slice::from_raw_parts(self.docs.as_ptr() as *const f32, self.n * self.dim) };
        &all[d * self.dim..(d + 1) * self.dim]
    }

    fn rows(&self) -> Vec<&[f32]> {
        (0..self.n).map(|d| self.doc(d)).collect()
    }

    /// Drops this process's mapping of the document vectors from memory
    /// (they stay in the page cache), keeping the RSS down while indexes are
    /// built.
    fn release(&self) {
        // SAFETY: read-only file mapping; contents are re-read on next access.
        unsafe { iostat::drop_mapping(self.docs.as_ptr(), self.docs.len()).unwrap() };
    }
}

fn peak_rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0
}

/// Exact top-100 of every query, cached in the collection's directory.
fn ground_truth(ds: &Dataset, threads: usize) -> (Vec<Vec<Hit>>, f64, bool) {
    const K: usize = 100;
    let suffix = if ds.n == ds.doc_ids.len() && ds.n == count_docs(&ds.name) { String::new() } else { format!("_n{}", ds.n) };
    let path = data::data_dir().join(&ds.name).join(format!("vec_gt_top{K}{suffix}.bin"));
    if let Ok(bytes) = std::fs::read(&path) {
        let h = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as usize;
        if bytes.len() >= 16 && &bytes[..4] == b"VGT1" && h(1) == ds.queries.len() && h(2) == K && h(3) == ds.n {
            let mut out = Vec::with_capacity(h(1));
            let mut o = 16;
            for _ in 0..h(1) {
                let mut v = Vec::with_capacity(K);
                for _ in 0..K {
                    let doc = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                    let score = f32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap());
                    if doc != u32::MAX {
                        v.push(Hit { doc, score });
                    }
                    o += 8;
                }
                out.push(v);
            }
            return (out, 0.0, true);
        }
    }
    let t = Instant::now();
    let rows = ds.rows();
    let qs: Vec<&[f32]> = ds.queries.iter().map(|q| q.as_slice()).collect();
    let gt = exact_top_k(&rows, &qs, K, None, threads);
    let secs = t.elapsed().as_secs_f64();
    let mut w = BufWriter::new(File::create(&path).unwrap());
    w.write_all(b"VGT1").unwrap();
    for x in [qs.len(), K, ds.n] {
        w.write_all(&(x as u32).to_le_bytes()).unwrap();
    }
    for hits in &gt {
        for i in 0..K {
            let h = hits.get(i).copied().unwrap_or(Hit { doc: u32::MAX, score: 0.0 });
            w.write_all(&h.doc.to_le_bytes()).unwrap();
            w.write_all(&h.score.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
    ds.release();
    (gt, secs, false)
}

fn count_docs(name: &str) -> usize {
    let meta: Meta = serde_json::from_reader(File::open(data::data_dir().join(name).join("meta.json")).unwrap()).unwrap();
    meta.docs
}

fn ndcg(ds: &Dataset, qi: usize, hits: &[Hit]) -> Option<f64> {
    let judged = ds.qrels.get(&ds.query_ids[qi])?;
    let ranked: Vec<String> = hits.iter().take(10).map(|h| ds.doc_ids[h.doc as usize].clone()).collect();
    Some(ndcg_at_k(&ranked, judged, 10))
}

/// ns per call of `f`, best of three rounds.
fn time_ns(iters: usize, mut f: impl FnMut(usize)) -> f64 {
    (0..3)
        .map(|_| {
            let t = Instant::now();
            for i in 0..iters {
                f(i);
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn kernels_study(ds: &Dataset, r: &VectorReader, gt: &[Vec<Hit>], nq: usize) -> Json {
    let dim = ds.dim;
    let stride = r.info().stride_u8;
    // In-cache speed: 64 vectors cycled (256 KiB of float32, 64 KiB of codes).
    let vs: Vec<&[f32]> = (0..64).map(|d| ds.doc(d)).collect();
    let cs: Vec<&[u8]> = (0..64).map(|d| r.codes(d)).collect();
    let q = &ds.queries[0];
    let qc = r.sq8().expect("SQ8 codes").prepare(q, stride);
    let f32_ns = time_ns(2_000_000, |i| {
        std::hint::black_box(kernels::dot_f32(vs[i % 64], q));
    });
    let vnni_ns = time_ns(4_000_000, |i| {
        std::hint::black_box(kernels::dot_u8_i8(cs[i % 64], &qc.q8));
    });
    let x4_ns = time_ns(1_000_000, |i| {
        let j = (i * 4) % 64;
        std::hint::black_box(kernels::dot_u8_i8_x4([cs[j], cs[j + 1], cs[j + 2], cs[j + 3]], &qc.q8));
    }) / 4.0;
    let asym_ns = time_ns(2_000_000, |i| {
        std::hint::black_box(kernels::dot_u8_f32(cs[i % 64], &qc.w));
    });
    let scalar_ns = time_ns(200_000, |i| {
        std::hint::black_box(kernels::dot_u8_i8_scalar(cs[i % 64], &qc.q8));
    });
    let isa = kernels::sq8_kernel();
    // Ranking quality of SQ8 alone (no re-ranking): top-10 by each scoring.
    let n = ds.n as u32;
    let nq = nq.min(ds.queries.len());
    let (mut rv, mut ra) = (Vec::new(), Vec::new());
    for (qi, q) in ds.queries.iter().take(nq).enumerate() {
        let qc = r.sq8().expect("SQ8 codes").prepare(q, stride);
        let mut a: Vec<(i32, u32)> = (0..n).map(|d| (qc.dot(r.codes(d)), d)).collect();
        a.sort_unstable_by(|x, y| y.cmp(x));
        let mut b: Vec<(f32, u32)> = (0..n).map(|d| (kernels::dot_u8_f32(r.codes(d), &qc.w), d)).collect();
        b.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
        let truth: Vec<u32> = gt[qi].iter().take(10).map(|h| h.doc).collect();
        rv.push(recall(&a.iter().take(10).map(|x| x.1).collect::<Vec<_>>(), &truth));
        ra.push(recall(&b.iter().take(10).map(|x| x.1).collect::<Vec<_>>(), &truth));
    }
    println!(
        "kernels ({dim}d): f32 {f32_ns:.1} ns, sq8 {isa} {vnni_ns:.1} ns (x4 {x4_ns:.1} ns/row), sq8 asym {asym_ns:.1} ns, sq8 scalar {scalar_ns:.1} ns; SQ8-only recall@10 {isa} {:.4} asym {:.4}",
        mean(&rv),
        mean(&ra)
    );
    // The `vnni` keys hold whichever int8 kernel ran (`sq8_kernel`).
    json!({
        "dim": dim, "sq8_kernel": isa,
        "ns_dot_f32": f32_ns, "ns_dot_u8_i8_vnni": vnni_ns, "ns_dot_u8_i8_vnni_x4_per_row": x4_ns,
        "ns_dot_u8_f32_asym": asym_ns, "ns_dot_u8_i8_scalar": scalar_ns,
        "sq8_only_recall10_vnni": mean(&rv), "sq8_only_recall10_asym": mean(&ra), "sq8_queries": nq,
    })
}

/// ns per float32 re-rank of a random document, prefetching the next row as
/// the search's re-ranking does.
fn rerank_cost(r: &VectorReader, q: &[f32]) -> f64 {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let docs: Vec<u32> = (0..20_000).map(|_| rng.gen_range(0..r.n_docs())).collect();
    time_ns(docs.len(), |i| {
        if let Some(&next) = docs.get(i + 1) {
            kernels::prefetch(r.vector(next).as_ptr() as *const u8, r.dim() * 4);
        }
        std::hint::black_box(r.similarity(docs[i], q));
    })
}

/// Builds `kind` over the collection and writes the segment to `path`.
fn build(ds: &Dataset, kind: &str, a: &Args, path: &Path) -> Json {
    let field = Field::new("emb", FieldType::Vector { dim: ds.dim, index: VectorIndexKind::Flat }).not_stored();
    let mut opts = BuildOptions::default()
        .set("vector.index", kind)
        .set("vector.threads", a.threads)
        .set("vector.hnsw.m", a.m)
        .set("vector.hnsw.ef_construction", a.ef_construction)
        .set("vector.hnsw.build", &a.hnsw_build);
    if let Some(nl) = a.nlist {
        opts = opts.set("vector.ivf.nlist", nl);
    }
    let t = Instant::now();
    let mut b = strata::vector::field_builder(&field, &opts).unwrap();
    for d in 0..ds.n {
        b.add(d as u32, &Value::Vector(Arc::from(ds.doc(d)))).unwrap();
        if d % 65536 == 65535 {
            ds.release();
        }
    }
    ds.release();
    let mut w = SegmentWriter::new(ds.n);
    b.finish(&mut w).unwrap();
    let build_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut f = BufWriter::with_capacity(1 << 22, File::create(path).unwrap());
    let bytes = w.write_to(&mut f).unwrap();
    f.into_inner().unwrap().sync_all().unwrap();
    let write_secs = t.elapsed().as_secs_f64();
    println!("{kind}: built in {build_secs:.1}s (written in {write_secs:.1}s, {:.1} MiB), peak RSS {:.0} MiB", bytes as f64 / 1048576.0, peak_rss_mb());
    json!({"build_secs": build_secs, "write_secs": write_secs, "file_bytes": bytes, "threads": a.threads, "hnsw_build": a.hnsw_build, "peak_rss_mb": peak_rss_mb()})
}

fn open(path: &Path) -> (SegmentFile, VectorReader) {
    // SAFETY: the segment file is not modified while mapped.
    let mmap = unsafe { Mmap::map(&File::open(path).unwrap()).unwrap() };
    let seg = SegmentFile::open(OwnedBytes::new(mmap)).unwrap();
    let r = VectorReader::open(&seg, "emb").unwrap();
    assert!(r.is_zero_copy());
    (seg, r)
}

/// HNSW work against ef: (ef, SQ8 distances) and (ef, expansions).
type HnswCurve = (Vec<(f64, f64)>, Vec<(f64, f64)>);

struct Run {
    hits: Vec<Vec<Hit>>,
    stats: Vec<SearchStats>,
    us: Vec<f64>,
}

/// Warms up on the first 20 queries, then runs all of them timed (one
/// thread, no overlap), stopping early once `budget` seconds have passed.
fn run(r: &VectorReader, qs: &[&[f32]], k: usize, p: &SearchParams, filters: &dyn Fn(usize) -> Option<Arc<DocSet>>, budget: f64) -> Run {
    let start = Instant::now();
    for (qi, q) in qs.iter().enumerate().take(20) {
        let f = filters(qi);
        std::hint::black_box(r.search(q, k, p, f.as_deref().map(|x| x as &dyn DocFilter)));
        if start.elapsed().as_secs_f64() > budget / 2.0 {
            break;
        }
    }
    let mut out = Run { hits: Vec::new(), stats: Vec::new(), us: Vec::new() };
    let start = Instant::now();
    for (qi, q) in qs.iter().enumerate() {
        let f = filters(qi);
        let t = Instant::now();
        let o = r.search(q, k, p, f.as_deref().map(|x| x as &dyn DocFilter));
        out.us.push(t.elapsed().as_nanos() as f64 / 1000.0);
        out.hits.push(o.hits);
        out.stats.push(o.stats);
        if start.elapsed().as_secs_f64() > budget {
            break;
        }
    }
    out
}

fn mean_of(stats: &[SearchStats], f: impl Fn(&SearchStats) -> usize) -> f64 {
    mean(&stats.iter().map(|s| f(s) as f64).collect::<Vec<_>>())
}

fn summary(run: &Run, truth: &dyn Fn(usize) -> Vec<u32>) -> Json {
    let rec: Vec<f64> = run.hits.iter().enumerate().map(|(qi, h)| recall(&h.iter().map(|h| h.doc).collect::<Vec<_>>(), &truth(qi))).collect();
    let s = &run.stats;
    json!({
        "queries": run.us.len(),
        "recall10": mean(&rec),
        "mean_us": mean(&run.us), "p50_us": percentile(&run.us, 50.0), "p99_us": percentile(&run.us, 99.0),
        "sq8": mean_of(s, |x| x.sq8_distances), "f32": mean_of(s, |x| x.f32_distances),
        "centroids": mean_of(s, |x| x.centroids), "clusters": mean_of(s, |x| x.clusters_probed),
        "nodes": mean_of(s, |x| x.nodes_visited), "checks": mean_of(s, |x| x.filter_checks),
        "candidates": mean_of(s, |x| x.candidates), "retries": mean_of(s, |x| x.retries),
        "est_us": mean(&s.iter().map(|x| x.estimated_us).collect::<Vec<_>>()),
    })
}

/// Calibration sample: mean work and mean latency of a run.
fn sample(run: &Run) -> (Work, f64) {
    let ws: Vec<Work> = run.stats.iter().map(Work::from_stats).collect();
    let n = ws.len().max(1) as f64;
    let mut w = Work::default();
    for x in &ws {
        w.scan += x.scan / n;
        w.gather += x.gather / n;
        w.rerank += x.rerank / n;
        w.centroids += x.centroids / n;
        w.clusters += x.clusters / n;
        w.checks += x.checks / n;
        w.nodes += x.nodes / n;
    }
    (w, mean(&run.us))
}

/// Non-negative least squares on relative error (projected coordinate descent
/// on the normal equations), with the re-rank cost fixed to `rerank_ns`: the
/// re-rank depth hardly varies across runs, so a fit cannot tell it from the
/// fixed cost.
fn fit(samples: &[(Work, f64)], rerank_ns: f64) -> [f64; 8] {
    let rows: Vec<[f64; 8]> = samples.iter().map(|(w, us)| w.features().map(|x| x / (us * 1000.0))).collect();
    let mut g = [[0f64; 8]; 8];
    let mut h = [0f64; 8];
    for r in &rows {
        for i in 0..8 {
            h[i] += r[i];
            for j in 0..8 {
                g[i][j] += r[i] * r[j];
            }
        }
    }
    const RERANK: usize = 3;
    let mut x = [0f64; 8];
    x[RERANK] = rerank_ns;
    for _ in 0..200_000 {
        for j in 0..8 {
            if g[j][j] <= 0.0 || j == RERANK {
                continue;
            }
            let grad: f64 = (0..8).map(|i| g[j][i] * x[i]).sum::<f64>() - h[j];
            x[j] = (x[j] - grad / g[j][j]).max(0.0);
        }
    }
    x
}

/// Least-squares line through (x, y).
fn line(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
    let sxx = pts.iter().map(|p| p.0 * p.0).sum::<f64>();
    let sxy = pts.iter().map(|p| p.0 * p.1).sum::<f64>();
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (a, (sy - a * sx) / n)
}

/// Cold-cache runs: before each query the segment file is evicted from the
/// page cache and the mapping's page-table entries dropped; pages read are
/// the file's resident pages after the query.
/// `advice` is the mapping's read-ahead policy (normal: the kernel reads
/// around every fault; random: only the faulting page); `hints` turns on the
/// reader's own read-ahead hints.
fn cold(seg: &SegmentFile, r: &mut VectorReader, path: &Path, qs: &[&[f32]], p: &SearchParams, advice: libc::c_int, hints: bool) -> Json {
    r.set_io_hints(hints);
    let img = seg.image();
    // SAFETY: madvise on our own read-only mapping only changes readahead.
    unsafe { libc::madvise(img.as_ptr() as *mut libc::c_void, img.len(), advice) };
    let (mut us, mut pages) = (Vec::new(), Vec::new());
    for q in qs {
        // Unmap first: the kernel keeps pages that are still mapped.
        // SAFETY: read-only file mapping that nobody writes through.
        unsafe { iostat::drop_mapping(img.as_ptr(), img.len()).unwrap() };
        iostat::evict(path).unwrap();
        let before = iostat::resident_pages(path).unwrap();
        let t = Instant::now();
        std::hint::black_box(r.search(q, 10, p, None));
        us.push(t.elapsed().as_nanos() as f64 / 1000.0);
        pages.push(iostat::resident_pages(path).unwrap().saturating_sub(before) as f64);
    }
    // SAFETY: as above.
    unsafe { libc::madvise(img.as_ptr() as *mut libc::c_void, img.len(), libc::MADV_NORMAL) };
    r.set_io_hints(false);
    json!({
        "advice": if advice == libc::MADV_RANDOM { "random" } else { "normal" },
        "hints": hints,
        "queries": qs.len(), "mean_us": mean(&us), "p50_us": percentile(&us, 50.0), "p99_us": percentile(&us, 99.0),
        "pages_mean": mean(&pages), "pages_p50": percentile(&pages, 50.0),
    })
}

/// Per-query filters correlated with the query: its nearest topics (by the
/// topic centroid), the last one cut by `rating` to hit `s` exactly.
fn topic_filter(ds: &Dataset, order: &[u16], frac: &[f64], s: f64) -> DocSet {
    let n = ds.n as u32;
    if s >= 1.0 {
        return DocSet::all(n);
    }
    let mut whole = Vec::new();
    let mut cum = 0.0;
    let mut part = None;
    for &t in order {
        if cum + frac[t as usize] <= s {
            whole.push(t);
            cum += frac[t as usize];
        } else {
            part = Some((t, 5.0 * (s - cum) / frac[t as usize]));
            break;
        }
    }
    let mut is_whole = [false; 1024];
    for &t in &whole {
        is_whole[t as usize] = true;
    }
    let docs: Vec<u32> = (0..n)
        .filter(|&d| {
            let t = ds.topic[d as usize];
            is_whole[t as usize] || part.is_some_and(|(pt, cut)| pt == t && (ds.rating[d as usize] as f64) < cut)
        })
        .collect();
    DocSet::from_sorted(docs, n)
}

fn main() {
    let a = Args::parse();
    let t0 = Instant::now();
    let ds = Dataset::load(&a.dataset, a.limit);
    println!("{}: {} documents, {} queries, {} dimensions", ds.name, ds.n, ds.queries.len(), ds.dim);
    let (gt, gt_secs, cached) = ground_truth(&ds, a.threads);
    println!("ground truth: {} (secs {gt_secs:.1})", if cached { "cached" } else { "computed" });
    let ndcg_exact: Vec<f64> = (0..ds.queries.len()).filter_map(|qi| ndcg(&ds, qi, &gt[qi])).collect();
    println!("exact dense retrieval nDCG@10 {:.4} over {} judged queries", mean(&ndcg_exact), ndcg_exact.len());

    let nq = a.queries.min(ds.queries.len());
    let qs: Vec<&[f32]> = ds.queries.iter().take(nq).map(|q| q.as_slice()).collect();
    let truth10 = |qi: usize| gt[qi].iter().take(10).map(|h| h.doc).collect::<Vec<u32>>();
    let workdir = a.workdir.clone().unwrap_or_else(|| std::env::temp_dir().join("strata-vector-bench"));
    std::fs::create_dir_all(&workdir).unwrap();
    let defaults = SearchParams { strategy: Strategy::Auto, nprobe: a.default_nprobe, ef: a.default_ef, rerank: a.default_rerank };

    let mut out = json!({
        "dataset": ds.name, "n": ds.n, "dim": ds.dim, "queries": ds.queries.len(), "timed_queries": nq,
        "threads": a.threads, "cpu": cpu_model(),
        "ground_truth": {"k": 100, "secs": gt_secs, "cached": cached},
        "ndcg10_exact": mean(&ndcg_exact), "ndcg10_queries": ndcg_exact.len(),
        "defaults": {"nprobe": a.default_nprobe, "ef": a.default_ef, "rerank": a.default_rerank},
    });
    let mut indexes = Vec::new();
    let mut samples: Vec<(Work, f64)> = Vec::new();
    let mut hnsw_pts: HnswCurve = (Vec::new(), Vec::new());
    let mut filtered = Vec::new();
    let kinds: Vec<String> = list(&a.kinds);
    let mut rerank_ns = f64::NAN;
    let fq = a.filter_queries.min(ds.queries.len());
    let fqs: Vec<&[f32]> = ds.queries.iter().take(fq).map(|q| q.as_slice()).collect();
    let sels: Vec<f64> = list(&a.selectivities);
    let study = FilterStudy { ds: &ds, qs: &fqs, sels: &sels, threads: a.threads, budget: a.budget_secs, defaults };
    let (random, topic) = if fq > 0 {
        let t = Instant::now();
        let c = (study.cases_random(), study.cases_topic());
        println!("filter cases and exact filtered top-10: {:.1}s", t.elapsed().as_secs_f64());
        c
    } else {
        (Vec::new(), Vec::new())
    };
    for kind in &kinds {
        let path = workdir.join(format!("{}-{kind}.seg", ds.name));
        let built = if a.reuse && path.exists() { Json::Null } else { build(&ds, kind, &a, &path) };
        let mut entry = json!({"kind": kind, "build": built});
        let (seg, mut r) = open(&path);
        let mut sections: Vec<(String, usize)> = seg.section_sizes();
        sections.sort();
        entry["sections"] = json!(sections.iter().map(|(n, b)| json!({"name": n, "bytes": b})).collect::<Vec<_>>());
        entry["info"] = serde_json::to_value(r.info()).unwrap();
        if rerank_ns.is_nan() {
            rerank_ns = rerank_cost(&r, &ds.queries[0]);
            println!("float32 re-rank of a random document: {rerank_ns:.0} ns");
            out["rerank_ns"] = json!(rerank_ns);
        }
        if kind == "flat" {
            out["kernels"] = kernels_study(&ds, &r, &gt, a.sq8_queries);
        }

        // Recall/latency curve.
        let sweep: Vec<(String, usize, SearchParams)> = match kind.as_str() {
            "flat" => list::<usize>(&a.rerank).into_iter().map(|v| ("rerank".into(), v, SearchParams { strategy: Strategy::Flat, rerank: v, ..defaults })).collect(),
            "ivf" => list::<usize>(&a.nprobe).into_iter().map(|v| ("nprobe".into(), v, SearchParams { strategy: Strategy::Ivf, nprobe: v, ..defaults })).collect(),
            // HNSW searches with max(ef, rerank): let ef alone set the depth.
            _ => list::<usize>(&a.ef)
                .into_iter()
                .map(|v| ("ef".into(), v, SearchParams { strategy: Strategy::Hnsw, ef: v, rerank: v.min(a.default_rerank), ..defaults }))
                .collect(),
        };
        let mut curve = Vec::new();
        for (name, v, p) in &sweep {
            // An untimed pass first. The segment was just written, so its
            // pages are cached, but the first touch of each page through the
            // new mapping still faults, and each setting re-ranks different
            // candidates; macOS maps one 16 KiB page per fault, without
            // Linux's fault-around, which made the shallow settings of the
            // sweep look slower than the deep ones.
            run(&r, &qs, 10, p, &|_| None, f64::INFINITY);
            let res = run(&r, &qs, 10, p, &|_| None, f64::INFINITY);
            let mut s = summary(&res, &truth10);
            let nd: Vec<f64> = res.hits.iter().enumerate().filter_map(|(qi, h)| ndcg(&ds, qi, h)).collect();
            s["ndcg10"] = json!(mean(&nd));
            s[name.as_str()] = json!(v);
            println!("  {kind} {name}={v}: recall@10 {:.4}, mean {:.0} us, p99 {:.0} us, sq8 {:.0}", s["recall10"].as_f64().unwrap(), s["mean_us"].as_f64().unwrap(), s["p99_us"].as_f64().unwrap(), s["sq8"].as_f64().unwrap());
            samples.push(sample(&res));
            if kind == "hnsw" {
                let ef = p.ef.max(p.rerank) as f64;
                hnsw_pts.0.push((ef, s["sq8"].as_f64().unwrap()));
                hnsw_pts.1.push((ef, s["nodes"].as_f64().unwrap()));
            }
            curve.push(s);
        }
        entry["curve"] = json!(curve);

        // The operating point, and what the read-ahead hints cost when
        // everything is resident (before the cold runs evict the file).
        let op = SearchParams {
            strategy: match kind.as_str() {
                "ivf" => Strategy::Ivf,
                "hnsw" => Strategy::Hnsw,
                _ => Strategy::Flat,
            },
            ..defaults
        };
        let mut warm = Vec::new();
        for hints in [false, true] {
            r.set_io_hints(hints);
            let res = run(&r, &qs, 10, &op, &|_| None, f64::INFINITY);
            warm.push(json!({"hints": hints, "mean_us": mean(&res.us), "p50_us": percentile(&res.us, 50.0)}));
        }
        r.set_io_hints(false);
        println!("  {kind} warm: {:.0} us without hints, {:.0} us with", warm[0]["mean_us"].as_f64().unwrap(), warm[1]["mean_us"].as_f64().unwrap());
        entry["warm_hints"] = json!(warm);

        // Filtered study on the index kinds that have an in-index strategy.
        if fq > 0 && kind != "flat" {
            let auto = HnswFilterMode::Traverse;
            let mut fixed: Vec<Variant> = vec![("prefilter", Strategy::PreFilter, auto), ("postfilter", Strategy::PostFilter, auto)];
            if kind == "ivf" {
                fixed.push(("ivf", Strategy::Ivf, auto));
            } else {
                fixed.push(("hnsw-traverse", Strategy::Hnsw, HnswFilterMode::Traverse));
                fixed.push(("hnsw-twohop", Strategy::Hnsw, HnswFilterMode::TwoHop));
            }
            filtered.extend(study.measure(&mut r, kind, "random", &random, &fixed, Some(&mut samples)));
            filtered.extend(study.measure(&mut r, kind, "random", &random, &[("auto", Strategy::Auto, auto)], None));
            fixed.push(("auto", Strategy::Auto, auto));
            filtered.extend(study.measure(&mut r, kind, "topic", &topic, &fixed, None));
        }
        // Cold cache at the operating point (last: it evicts the file).
        if a.cold_queries > 0 {
            let cq = &qs[..a.cold_queries.min(nq)];
            let mut c = Vec::new();
            for (advice, hints) in [(libc::MADV_NORMAL, false), (libc::MADV_RANDOM, false), (libc::MADV_RANDOM, true)] {
                let j = cold(&seg, &mut r, &path, cq, &op, advice, hints);
                println!(
                    "  {kind} cold ({}{}): mean {:.0} us, p50 {:.0} us, pages {:.0}",
                    j["advice"].as_str().unwrap(),
                    if hints { " + hints" } else { "" },
                    j["mean_us"].as_f64().unwrap(),
                    j["p50_us"].as_f64().unwrap(),
                    j["pages_mean"].as_f64().unwrap()
                );
                c.push(j);
            }
            entry["cold"] = json!(c);
        }

        indexes.push(entry);
        drop(r);
        drop(seg);
        if !a.keep {
            std::fs::remove_file(&path).unwrap();
        }
    }
    out["indexes"] = json!(indexes);
    out["calibration"] = calibrate(&samples, &hnsw_pts, rerank_ns);
    out["filtered"] = json!(filtered);
    out["peak_rss_mb"] = json!(peak_rss_mb());
    out["total_secs"] = json!(t0.elapsed().as_secs_f64());
    std::fs::create_dir_all(&a.out).unwrap();
    let name = match &a.tag {
        Some(t) => format!("{}-{t}.json", ds.name),
        None => format!("{}.json", ds.name),
    };
    std::fs::write(a.out.join(&name), serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {} ({:.0}s, peak RSS {:.0} MiB)", a.out.join(&name).display(), t0.elapsed().as_secs_f64(), peak_rss_mb());
}

/// Fits the cost model to the measured runs (searches here use the
/// compiled-in model; the fit is what `CostModel::default` is set from).
fn calibrate(samples: &[(Work, f64)], hnsw_pts: &HnswCurve, rerank_ns: f64) -> Json {
    if samples.is_empty() {
        return Json::Null;
    }
    let mut model = CostModel::default();
    model.set_coefficients(fit(samples, rerank_ns));
    if hnsw_pts.0.len() >= 2 {
        (model.hnsw_dist_a, model.hnsw_dist_b) = line(&hnsw_pts.0);
        (model.hnsw_nodes_a, model.hnsw_nodes_b) = line(&hnsw_pts.1);
    }
    let err = |m: &CostModel| -> Vec<f64> { samples.iter().map(|(w, us)| (m.time_us(w) / us - 1.0).abs()).collect() };
    let (fitted, default) = (err(&model), err(&CostModel::default()));
    println!(
        "cost model fit on {} runs: {:?}\n  median |error| fitted {:.1}%, compiled-in {:.1}%",
        samples.len(),
        model,
        100.0 * percentile(&fitted, 50.0),
        100.0 * percentile(&default, 50.0)
    );
    json!({
        "model": model, "samples": samples.len(),
        "median_rel_err": percentile(&fitted, 50.0), "p90_rel_err": percentile(&fitted, 90.0),
        "default_median_rel_err": percentile(&default, 50.0), "default_p90_rel_err": percentile(&default, 90.0),
    })
}

/// A strategy as measured: label, strategy, HNSW in-filter traversal.
type Variant = (&'static str, Strategy, HnswFilterMode);

struct FilterStudy<'a> {
    ds: &'a Dataset,
    qs: &'a [&'a [f32]],
    sels: &'a [f64],
    threads: usize,
    budget: f64,
    defaults: SearchParams,
}

/// One filter setting: per-query filters (or one shared) and the exact
/// filtered top-10 of every query.
struct Case {
    sel: f64,
    filters: Vec<Arc<DocSet>>,
    truth: Vec<Vec<u32>>,
}

impl FilterStudy<'_> {
    fn cases_random(&self) -> Vec<Case> {
        let rows = self.ds.rows();
        let cases = self
            .sels
            .iter()
            .map(|&s| {
                let n = self.ds.n as u32;
                let f = Arc::new(if s >= 1.0 { DocSet::all(n) } else { DocSet::from_sorted((0..n).filter(|&d| (self.ds.rating[d as usize] as f64) < 5.0 * s).collect(), n) });
                let truth = exact_top_k(&rows, self.qs, 10, Some(&*f), self.threads).into_iter().map(|h| h.iter().map(|x| x.doc).collect()).collect();
                Case { sel: f.len() as f64 / n as f64, filters: vec![f], truth }
            })
            .collect();
        self.ds.release();
        cases
    }

    fn cases_topic(&self) -> Vec<Case> {
        let ds = self.ds;
        let nt = *ds.topic.iter().max().unwrap() as usize + 1;
        let mut cent = vec![vec![0f32; ds.dim]; nt];
        let mut count = vec![0usize; nt];
        for d in 0..ds.n {
            let t = ds.topic[d] as usize;
            count[t] += 1;
            for (c, x) in cent[t].iter_mut().zip(ds.doc(d)) {
                *c += x;
            }
        }
        let frac: Vec<f64> = count.iter().map(|&c| c as f64 / ds.n as f64).collect();
        let orders: Vec<Vec<u16>> = self
            .qs
            .iter()
            .map(|q| {
                let mut o: Vec<(f32, u16)> = cent.iter().enumerate().map(|(t, c)| (kernels::dot_f32(c, q), t as u16)).collect();
                o.sort_by(|a, b| b.0.total_cmp(&a.0));
                o.into_iter().map(|x| x.1).collect()
            })
            .collect();
        let rows = ds.rows();
        let cases = self
            .sels
            .iter()
            .map(|&s| {
                let filters: Vec<Arc<DocSet>> = orders.iter().map(|o| Arc::new(topic_filter(ds, o, &frac, s))).collect();
                let mut truth: Vec<Vec<u32>> = vec![Vec::new(); self.qs.len()];
                let per = self.qs.len().div_ceil(self.threads.max(1));
                std::thread::scope(|sc| {
                    for (ci, chunk) in truth.chunks_mut(per).enumerate() {
                        let (rows, filters) = (&rows, &filters);
                        sc.spawn(move || {
                            for (j, t) in chunk.iter_mut().enumerate() {
                                let qi = ci * per + j;
                                *t = exact_top_k(rows, &self.qs[qi..qi + 1], 10, Some(&*filters[qi]), 1)[0].iter().map(|h| h.doc).collect();
                            }
                        });
                    }
                });
                let sel = mean(&filters.iter().map(|f| f.len() as f64 / ds.n as f64).collect::<Vec<_>>());
                Case { sel, filters, truth }
            })
            .collect();
        ds.release();
        cases
    }

    /// Runs every strategy on every case; adds calibration samples if asked.
    fn measure(&self, r: &mut VectorReader, kind: &str, filter: &str, cases: &[Case], variants: &[Variant], mut samples: Option<&mut Vec<(Work, f64)>>) -> Vec<Json> {
        let mut cells = Vec::new();
        for case in cases {
            for &(label, st, mode) in variants {
                r.set_hnsw_filter(mode);
                let p = SearchParams { strategy: st, ..self.defaults };
                let fl = |qi: usize| Some(case.filters[qi.min(case.filters.len() - 1)].clone());
                let res = run(r, self.qs, 10, &p, &fl, self.budget);
                r.set_hnsw_filter(HnswFilterMode::Traverse);
                let mut s = summary(&res, &|qi| case.truth[qi].clone());
                let mut chosen: HashMap<&str, usize> = HashMap::new();
                for x in &res.stats {
                    *chosen.entry(x.strategy.as_str()).or_default() += 1;
                }
                s["index"] = json!(kind);
                s["filter"] = json!(filter);
                s["selectivity"] = json!(case.sel);
                s["target_selectivity"] = json!(self.sels[cases.iter().position(|c| std::ptr::eq(c, case)).unwrap()]);
                s["strategy"] = json!(label);
                s["ran"] = json!(chosen);
                println!(
                    "  {kind} {filter} s={:.4} {:<13} recall@10 {:.4} mean {:>8.0} us (est {:>8.0}) ran {:?}",
                    case.sel,
                    label,
                    s["recall10"].as_f64().unwrap(),
                    s["mean_us"].as_f64().unwrap(),
                    s["est_us"].as_f64().unwrap(),
                    chosen
                );
                if let Some(v) = samples.as_deref_mut() {
                    v.push(sample(&res));
                }
                cells.push(s);
            }
        }
        cells
    }
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}
