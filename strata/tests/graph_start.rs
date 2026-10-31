//! Merged HNSW graphs start from an input's graph: its nodes keep their
//! levels and links and only the other documents are inserted (Lucene's
//! merge, from 9.9); the other inputs' graphs are joined to it (Lucene
//! 10.2). A graph started from must cover documents with the same vectors;
//! one that does not fit is ignored. Construction scores with SQ8 codes
//! unless `vector.hnsw.build` is `f32`.

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use strata::db::Db;
use strata::query::Params;
use strata::segment::build::{build_segment, build_segment_from, BuildOptions, GraphStart};
use strata::segment::SegmentFile;
use strata::storage::{DbOptions, MergePolicy};
use strata::vector::{SearchParams, Strategy, VectorReader};
use strata::{Field, FieldType, Schema, Value, VectorIndexKind};

const DIM: usize = 32;

fn gaussian(rng: &mut impl Rng) -> f32 {
    let (u, v): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen());
    ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
}

/// `n` unit vectors around 50 random centres.
fn clustered(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..50).map(|_| (0..DIM).map(|_| gaussian(&mut rng)).collect()).collect();
    (0..n)
        .map(|_| {
            let c = &centres[rng.gen_range(0..centres.len())];
            let mut v: Vec<f32> = c.iter().map(|x| x + 0.6 * gaussian(&mut rng)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

fn schema(m: usize) -> Schema {
    Schema::new("t", vec![Field::new("id", FieldType::Int), Field::new("emb", FieldType::Vector { dim: DIM, index: VectorIndexKind::Hnsw { m, ef_construction: 100 } })], Some("id")).unwrap()
}

/// Documents `ids` with their vectors; every 13th has none.
fn docs(vectors: &[Vec<f32>], ids: std::ops::Range<usize>) -> Vec<Vec<Value>> {
    ids.map(|i| vec![Value::Int(i as i64), if i % 13 == 5 { Value::Null } else { Value::Vector(Arc::from(vectors[i].as_slice())) }]).collect()
}

/// The name of field `emb`'s sections (by its id).
fn emb() -> String {
    schema(16).key_of("emb").unwrap()
}

fn opts(seed: u64) -> BuildOptions {
    BuildOptions::default().set("vector.seed", seed).set("vector.threads", 1)
}

fn graph(seg: &SegmentFile) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let s = |part: &str| seg.require(&format!("vec.{}.{part}", emb())).unwrap().to_vec();
    (s("meta"), s("hnsw"), s("hnsw_upper"))
}

fn segment(image: Vec<u8>) -> SegmentFile {
    SegmentFile::from_vec(image).unwrap()
}

/// Recall@10 of the graph over `queries` against an exact scan.
fn recall(seg: &SegmentFile, queries: &[Vec<f32>]) -> f64 {
    let r = VectorReader::open(seg, &emb()).unwrap();
    let params = SearchParams { strategy: Strategy::Hnsw, ef: 100, ..SearchParams::default() };
    let mut found = 0;
    for q in queries {
        let exact: Vec<u32> = r.exact(q, 10, None).iter().map(|h| h.doc).collect();
        found += r.search(q, 10, &params, None).hits.iter().filter(|h| exact.contains(&h.doc)).count();
    }
    found as f64 / (10 * queries.len()) as f64
}

#[test]
fn a_graph_started_from_one_over_the_same_documents_is_that_graph() {
    let vectors = clustered(3000, 1);
    let all = docs(&vectors, 0..3000);
    let first = segment(build_segment(&schema(16), &all, &opts(1)).unwrap());
    // Another seed gives other levels and so another graph ...
    let other = segment(build_segment(&schema(16), &all, &opts(2)).unwrap());
    assert_ne!(graph(&other), graph(&first));
    // ... unless it starts from the first one's, which leaves nothing to insert.
    let start = [GraphStart { field: 1, segment: first.clone(), base: 0 }];
    let started = segment(build_segment_from(&schema(16), &all, &opts(2), &start).unwrap());
    assert_eq!(graph(&started), graph(&first));
}

#[test]
fn a_graph_started_from_part_of_the_documents_finds_neighbours_as_well() {
    let vectors = clustered(8000, 3);
    let queries = clustered(200, 4);
    let all = docs(&vectors, 0..8000);
    // The input merged with others: documents 3,000 to 6,000.
    let part = segment(build_segment(&schema(16), &docs(&vectors, 3000..6000), &opts(1)).unwrap());
    let start = [GraphStart { field: 1, segment: part.clone(), base: 3000 }];
    let started = segment(build_segment_from(&schema(16), &all, &opts(1), &start).unwrap());
    let whole = segment(build_segment(&schema(16), &all, &opts(1)).unwrap());
    let (with, without) = (recall(&started, &queries), recall(&whole, &queries));
    assert!(with >= 0.9 && with >= without - 0.02, "recall@10 started {with:.3}, built whole {without:.3}");
}

#[test]
fn a_graph_that_does_not_fit_is_ignored() {
    let vectors = clustered(2000, 5);
    let all = docs(&vectors, 0..2000);
    let plain = build_segment(&schema(16), &all, &opts(1)).unwrap();
    // Another M; other documents (missing vectors elsewhere); past the end.
    let other_m = segment(build_segment(&schema(8), &all, &opts(1)).unwrap());
    let shifted = segment(build_segment(&schema(16), &docs(&vectors, 1..2000), &opts(1)).unwrap());
    let small = segment(build_segment(&schema(16), &docs(&vectors, 0..1000), &opts(1)).unwrap());
    for (seg, base) in [(other_m, 0), (shifted, 0), (small, 1500)] {
        let start = [GraphStart { field: 1, segment: seg, base }];
        assert!(build_segment_from(&schema(16), &all, &opts(1), &start).unwrap() == plain);
    }
}

#[test]
fn a_merge_joins_its_other_inputs_graphs() {
    let vectors = clustered(8000, 8);
    let queries = clustered(200, 9);
    let all = docs(&vectors, 0..8000);
    let inputs: Vec<SegmentFile> = (0..4).map(|s| segment(build_segment(&schema(16), &docs(&vectors, s * 2000..(s + 1) * 2000), &opts(1)).unwrap())).collect();
    let starts: Vec<GraphStart> = inputs.iter().enumerate().map(|(s, seg)| GraphStart { field: 1, segment: seg.clone(), base: (s * 2000) as u32 }).collect();
    let joined = build_segment_from(&schema(16), &all, &opts(1), &starts).unwrap();
    let largest = build_segment_from(&schema(16), &all, &opts(1), &starts[..1]).unwrap();
    let whole = segment(build_segment(&schema(16), &all, &opts(1)).unwrap());
    // The joined inputs change the graph, the same way every time.
    assert!(joined != largest);
    assert!(build_segment_from(&schema(16), &all, &opts(1), &starts).unwrap() == joined);
    let (with, without) = (recall(&segment(joined), &queries), recall(&whole, &queries));
    assert!(with >= 0.9 && with >= without - 0.02, "recall@10 joined {with:.3}, built whole {without:.3}");
    // A graph that does not fit among them is left out: the same graph as
    // without it.
    let mut odd = starts.clone();
    odd.insert(2, GraphStart { field: 1, segment: segment(build_segment(&schema(8), &docs(&vectors, 4000..6000), &opts(1)).unwrap()), base: 4000 });
    odd.remove(3);
    let mut fit = starts.clone();
    fit.remove(2);
    assert!(build_segment_from(&schema(16), &all, &opts(1), &odd).unwrap() == build_segment_from(&schema(16), &all, &opts(1), &fit).unwrap());
}

#[test]
fn graphs_built_on_codes_find_neighbours_as_well() {
    let vectors = clustered(6000, 10);
    let queries = clustered(200, 11);
    let all = docs(&vectors, 0..6000);
    let codes = build_segment(&schema(16), &all, &opts(1)).unwrap();
    let f32 = build_segment(&schema(16), &all, &opts(1).set("vector.hnsw.build", "f32")).unwrap();
    assert!(codes != f32);
    let (c, f) = (recall(&segment(codes), &queries), recall(&segment(f32), &queries));
    assert!(c >= 0.9 && c >= f - 0.02, "recall@10 built on codes {c:.3}, on float32 {f:.3}");
    let err = build_segment(&schema(16), &all, &opts(1).set("vector.hnsw.build", "bf16")).unwrap_err().to_string();
    assert!(err.contains("vector.hnsw.build"), "{err}");
}

#[test]
fn merged_segments_start_from_their_largest_whole_input() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = clustered(9000, 6);
    let queries = clustered(100, 7);
    let o = DbOptions { build: opts(1), memtable_max_docs: 1_000_000, segment_max_docs: 1_000_000, ..DbOptions::default() };
    let db = Db::create(&dir.path().join("g.strata"), schema(16), o).unwrap();
    // Four segments, the largest (4,000 documents) whole, a smaller one
    // with deletions.
    for r in [0..1000, 1000..5000, 5000..7000, 7000..9000] {
        db.storage().bulk_load(docs(&vectors, r)).unwrap();
    }
    db.sql("DELETE FROM t WHERE id >= 5000 AND id < 5100", &Params::new()).unwrap();
    let policy = MergePolicy { merge_factor: 4, floor_docs: 1_000_000, ..MergePolicy::default() };
    let stats = db.storage().merge(&policy).unwrap();
    assert_eq!(stats.merges, 1);
    let snap = db.storage().snapshot();
    let segs = snap.segments();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].file.n_docs(), 8900);
    let got = recall(&segs[0].file, &queries);
    assert!(got >= 0.9, "recall@10 {got:.3}");
    // Every live document is still found by its own vector.
    let r = VectorReader::open(&segs[0].file, &emb()).unwrap();
    let params = SearchParams { strategy: Strategy::Hnsw, ef: 40, ..SearchParams::default() };
    for d in (0..8900).step_by(97).filter(|&d| r.has_vector(d)) {
        assert_eq!(r.search(r.vector(d), 1, &params, None).hits[0].doc, d);
    }
}
