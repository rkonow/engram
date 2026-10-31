//! Vector search on synthetic data: exactness of the flat and pre-filter
//! paths against a naive reference, filter semantics of every strategy,
//! missing vectors, zero-copy views, build options.

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_distr_free::gaussian;
use strata::segment::build::{build_segment, BuildOptions};
use strata::segment::{DocFilter, DocSet, SegmentFile};
use strata::vector::{Hit, HnswFilterMode, SearchParams, Strategy, VectorReader};
use strata::{Field, FieldType, Schema, Value, VectorIndexKind};

/// The sections of field `emb`, the table's only field (id 0).
const EMB: &str = "f0";

/// Box-Muller normal samples (keeps the test free of extra dependencies).
mod rand_distr_free {
    use rand::Rng;
    pub fn gaussian(rng: &mut impl Rng) -> f32 {
        let (u, v): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

/// `n` unit vectors around `clusters` random centres.
fn clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| (0..dim).map(|_| gaussian(&mut rng)).collect()).collect();
    (0..n)
        .map(|_| {
            let c = &centres[rng.gen_range(0..clusters)];
            let mut v: Vec<f32> = c.iter().map(|x| x + 0.6 * gaussian(&mut rng)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

fn segment(vectors: &[Option<Vec<f32>>], dim: usize, opts: &BuildOptions) -> SegmentFile {
    let schema = Schema::new("t", vec![Field::new("emb", FieldType::Vector { dim, index: VectorIndexKind::Flat }).not_stored()], None).unwrap();
    let docs: Vec<Vec<Value>> = vectors.iter().map(|v| vec![v.as_ref().map_or(Value::Null, |v| Value::Vector(Arc::from(v.as_slice())))]).collect();
    SegmentFile::from_vec(build_segment(&schema, &docs, opts).unwrap()).unwrap()
}

fn naive(vectors: &[Option<Vec<f32>>], q: &[f32], k: usize, filter: Option<&dyn DocFilter>) -> Vec<u32> {
    let mut all: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .filter_map(|(d, v)| v.as_ref().map(|v| (d as u32, v)))
        .filter(|(d, _)| filter.is_none_or(|f| f.contains(*d)))
        .map(|(d, v)| (v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>(), d))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|x| x.1).collect()
}

fn ids(h: &[Hit]) -> Vec<u32> {
    h.iter().map(|h| h.doc).collect()
}

fn opts(kind: &str) -> BuildOptions {
    BuildOptions::default().set("vector.index", kind).set("vector.ivf.nlist", 24).set("vector.hnsw.ef_construction", 100)
}

#[test]
fn flat_with_full_rerank_is_exact_search() {
    // 50 dimensions: float32 rows are padded to 64, code rows to 64.
    let dim = 50;
    let vecs: Vec<Option<Vec<f32>>> = clustered(1500, dim, 12, 1).into_iter().map(Some).collect();
    let seg = segment(&vecs, dim, &opts("flat"));
    let r = VectorReader::open(&seg, EMB).unwrap();
    assert_eq!((r.dim(), r.n_docs(), r.index_kind()), (dim, 1500, "flat"));
    assert!(r.is_zero_copy());
    let queries = clustered(20, dim, 12, 2);
    let p = SearchParams { strategy: Strategy::Flat, rerank: 1500, ..Default::default() };
    for q in &queries {
        let out = r.search(q, 10, &p, None);
        assert_eq!(ids(&out.hits), naive(&vecs, q, 10, None));
        assert_eq!(out.stats.strategy, Strategy::Flat);
        assert_eq!(out.stats.sq8_distances, 1500);
        for h in &out.hits {
            assert_eq!(h.score, r.similarity(h.doc, q));
        }
    }
    // The stored vectors are the inputs.
    assert_eq!(r.vector(7), vecs[7].as_deref().unwrap());
}

#[test]
fn prefilter_returns_exact_filtered_top_k() {
    let dim = 64;
    let n = 3000;
    let vecs: Vec<Option<Vec<f32>>> = clustered(n, dim, 16, 3).into_iter().map(Some).collect();
    let seg = segment(&vecs, dim, &opts("flat"));
    let r = VectorReader::open(&seg, EMB).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let queries = clustered(25, dim, 16, 5);
    let mut found = 0;
    let mut total = 0;
    for density in [0.003, 0.05, 0.5] {
        let f = DocSet::from_sorted((0..n as u32).filter(|_| rng.gen_bool(density)).collect(), n as u32);
        for q in &queries {
            let truth = naive(&vecs, q, 10, Some(&f));
            // Re-ranking every member is exact.
            let all = SearchParams { strategy: Strategy::PreFilter, rerank: n, ..Default::default() };
            let out = r.search(q, 10, &all, Some(&f));
            assert_eq!(ids(&out.hits), truth, "density {density}");
            assert_eq!(out.stats.sq8_distances, f.len());
            // The default re-rank depth finds the same documents up to SQ8 ties.
            let out = r.search(q, 10, &SearchParams { strategy: Strategy::PreFilter, ..Default::default() }, Some(&f));
            found += out.hits.iter().filter(|h| truth.contains(&h.doc)).count();
            total += truth.len();
        }
    }
    assert!(found as f64 >= 0.99 * total as f64, "pre-filter recall {found}/{total}");
}

#[test]
fn every_strategy_returns_only_members() {
    let dim = 32;
    let n = 2500;
    let vecs: Vec<Option<Vec<f32>>> = clustered(n, dim, 10, 6).into_iter().map(Some).collect();
    let queries = clustered(10, dim, 10, 7);
    let mut rng = rand::rngs::StdRng::seed_from_u64(8);
    let filters: Vec<DocSet> =
        [0.002, 0.02, 0.2, 0.9].iter().map(|&p| DocSet::from_sorted((0..n as u32).filter(|_| rng.gen_bool(p)).collect(), n as u32)).collect();
    for kind in ["flat", "ivf", "hnsw"] {
        let seg = segment(&vecs, dim, &opts(kind));
        let r = VectorReader::open(&seg, EMB).unwrap();
        assert_eq!(r.index_kind(), kind);
        assert!(r.is_zero_copy(), "{kind}");
        for f in &filters {
            for s in Strategy::ALL {
                let p = SearchParams { strategy: s, ..Default::default() };
                for q in &queries {
                    let out = r.search(q, 10, &p, Some(f));
                    assert!(out.hits.iter().all(|h| f.contains(h.doc)), "{kind} {s:?} returned a non-member");
                    assert!(out.hits.windows(2).all(|w| w[0].score >= w[1].score), "{kind} {s:?} unsorted");
                    assert_ne!(out.stats.strategy, Strategy::Auto);
                    // Every strategy finds k members when the filter has them
                    // (in-filter HNSW may stop short only on a tiny filter).
                    if f.len() >= 50 || out.stats.strategy != Strategy::Hnsw {
                        assert_eq!(out.hits.len(), 10.min(f.len()), "{kind} {:?} short", out.stats.strategy);
                    }
                }
            }
        }
        // The two-hop traversal returns members only, though it may stop short.
        let mut r = r;
        r.set_hnsw_filter(HnswFilterMode::TwoHop);
        for f in &filters {
            for q in &queries {
                let out = r.search(q, 10, &SearchParams { strategy: Strategy::Hnsw, ..Default::default() }, Some(f));
                assert!(out.hits.iter().all(|h| f.contains(h.doc)), "{kind} two-hop returned a non-member");
                if f.len() > 1000 {
                    assert_eq!(out.hits.len(), 10);
                }
            }
        }
        r.set_hnsw_filter(HnswFilterMode::Traverse);
        // An empty filter returns nothing; no filter returns k hits.
        let empty = DocSet::empty(n as u32);
        assert!(r.search(&queries[0], 10, &SearchParams::default(), Some(&empty)).hits.is_empty());
        assert_eq!(r.search(&queries[0], 10, &SearchParams::default(), None).hits.len(), 10);
    }
}

#[test]
fn documents_without_vectors_are_never_returned() {
    let dim = 24;
    let n = 800;
    let base = clustered(n, dim, 6, 9);
    let vecs: Vec<Option<Vec<f32>>> = base.into_iter().enumerate().map(|(d, v)| if d % 5 == 0 { None } else { Some(v) }).collect();
    let queries = clustered(8, dim, 6, 10);
    for kind in ["flat", "ivf", "hnsw"] {
        let seg = segment(&vecs, dim, &opts(kind));
        let r = VectorReader::open(&seg, EMB).unwrap();
        assert!(!r.has_vector(0) && r.has_vector(1));
        assert!(r.vector(0).iter().all(|&x| x == 0.0));
        for s in [Strategy::Flat, Strategy::Ivf, Strategy::Hnsw] {
            let p = SearchParams { strategy: s, rerank: n, ef: 200, nprobe: 24 };
            for q in &queries {
                let out = r.search(q, 20, &p, None);
                assert!(out.hits.iter().all(|h| h.doc % 5 != 0), "{kind} {s:?}");
                assert_eq!(ids(&out.hits), naive(&vecs, q, 20, None), "{kind} {s:?}");
            }
        }
        assert_eq!(ids(&r.exact(&queries[0], 20, None)), naive(&vecs, &queries[0], 20, None));
    }
}

#[test]
fn build_options_select_the_index() {
    let dim = 16;
    let vecs: Vec<Option<Vec<f32>>> = clustered(300, dim, 4, 11).into_iter().map(Some).collect();
    let seg = segment(&vecs, dim, &BuildOptions::default().set("vector.index", "ivf").set("vector.ivf.nlist", 7));
    let r = VectorReader::open(&seg, EMB).unwrap();
    assert_eq!((r.index_kind(), r.info().nlist), ("ivf", Some(7)));
    let seg = segment(&vecs, dim, &BuildOptions::default().set("vector.index", "hnsw").set("vector.hnsw.m", 8));
    let r = VectorReader::open(&seg, EMB).unwrap();
    assert_eq!((r.index_kind(), r.info().m), ("hnsw", Some(8)));
    let schema = Schema::new("t", vec![Field::new("emb", FieldType::Vector { dim, index: VectorIndexKind::Flat })], None).unwrap();
    let docs = vec![vec![Value::Vector(Arc::from(vec![0f32; dim]))]];
    assert!(build_segment(&schema, &docs, &BuildOptions::default().set("vector.index", "lsh")).is_err());
    // An IVF over fewer documents than lists still works.
    let seg = segment(&vecs[..3], dim, &BuildOptions::default().set("vector.index", "ivf").set("vector.ivf.nlist", 50));
    let r = VectorReader::open(&seg, EMB).unwrap();
    let out = r.search(vecs[1].as_deref().unwrap(), 2, &SearchParams { strategy: Strategy::Ivf, ..Default::default() }, None);
    assert_eq!(out.hits[0].doc, 1);
    // An empty segment.
    let seg = segment(&[], dim, &BuildOptions::default().set("vector.index", "hnsw"));
    let r = VectorReader::open(&seg, EMB).unwrap();
    assert!(r.search(&[0.5; 16], 5, &SearchParams::default(), None).hits.is_empty());
}

#[test]
fn auto_follows_the_cost_model() {
    let dim = 32;
    let n = 4000;
    let vecs: Vec<Option<Vec<f32>>> = clustered(n, dim, 10, 12).into_iter().map(Some).collect();
    let seg = segment(&vecs, dim, &opts("hnsw"));
    let r = VectorReader::open(&seg, EMB).unwrap();
    let p = SearchParams::default();
    // A tiny filter is scanned; no filter uses the graph.
    assert_eq!(r.plan(10, Some(4), &p), Strategy::PreFilter);
    assert_eq!(r.plan(10, None, &p), Strategy::Hnsw);
    for card in [None, Some(4), Some(400), Some(4000)] {
        let auto = r.estimate_cost(Strategy::Auto, 10, card, &p);
        for s in [Strategy::Flat, Strategy::Hnsw, Strategy::PreFilter, Strategy::PostFilter] {
            assert!(auto <= r.estimate_cost(s, 10, card, &p) + 1e-9);
        }
        assert!(auto > 0.0);
    }
}

#[test]
fn io_hints_do_not_change_results() {
    let dim = 40;
    let n = 3000;
    let vecs: Vec<Option<Vec<f32>>> = clustered(n, dim, 10, 13).into_iter().map(Some).collect();
    let queries = clustered(10, dim, 10, 14);
    let f = DocSet::from_sorted((0..n as u32).filter(|d| d % 7 == 3).collect(), n as u32);
    for kind in ["flat", "ivf", "hnsw"] {
        let seg = segment(&vecs, dim, &opts(kind));
        let mut r = VectorReader::open(&seg, EMB).unwrap();
        for s in Strategy::ALL {
            let p = SearchParams { strategy: s, ..Default::default() };
            for q in &queries {
                for filter in [None, Some(&f as &dyn DocFilter)] {
                    r.set_io_hints(false);
                    let plain = r.search(q, 10, &p, filter);
                    r.set_io_hints(true);
                    let hinted = r.search(q, 10, &p, filter);
                    assert_eq!(plain.hits, hinted.hits, "{kind} {s:?}");
                    assert_eq!(plain.stats, hinted.stats, "{kind} {s:?}");
                }
            }
        }
    }
}

/// A post-filter whose members lie away from the query (the half of the
/// vectors least like it) needs many more candidates than its selectivity
/// says: the HNSW walk widens from where it stopped until enough pass, and
/// finds the filtered top 10.
#[test]
fn post_filter_widens_its_walk() {
    let dim = 32;
    let n = 4000;
    let vecs: Vec<Option<Vec<f32>>> = clustered(n, dim, 8, 21).into_iter().map(Some).collect();
    let seg = segment(&vecs, dim, &opts("hnsw"));
    let r = VectorReader::open(&seg, EMB).unwrap();
    let queries = clustered(10, dim, 8, 22);
    let (mut found, mut total, mut retried) = (0, 0, 0);
    for q in &queries {
        // The half least like the query.
        let mut by: Vec<(f32, u32)> = vecs.iter().enumerate().map(|(d, v)| (v.as_ref().unwrap().iter().zip(q).map(|(a, b)| a * b).sum::<f32>(), d as u32)).collect();
        by.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut members: Vec<u32> = by[..n / 2].iter().map(|x| x.1).collect();
        members.sort_unstable();
        let f = DocSet::from_sorted(members, n as u32);
        let truth = naive(&vecs, q, 10, Some(&f));
        let out = r.search(q, 10, &SearchParams { strategy: Strategy::PostFilter, ..Default::default() }, Some(&f));
        assert_eq!(out.hits.len(), 10);
        assert!(out.hits.iter().all(|h| f.contains(h.doc)));
        retried += (out.stats.retries > 0) as usize;
        found += out.hits.iter().filter(|h| truth.contains(&h.doc)).count();
        total += truth.len();
    }
    assert_eq!(retried, queries.len(), "every query needs a wider walk");
    assert!(found as f64 >= 0.9 * total as f64, "post-filter recall {found}/{total}");
}
