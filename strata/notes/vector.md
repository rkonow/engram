# Vector storage and ANN search (DESIGN.md §5.3)

`strata/src/vector/`: float32 + SQ8 forward vectors, flat / IVF / HNSW
indexes, filtered search strategies and the cost model the planner uses to
choose between them.

| file | contents |
|---|---|
| `mod.rs` | `VectorReader`, `Strategy`, `SearchParams`, `SearchStats`, search orchestration, re-ranking |
| `build.rs` | the `FieldIndexBuilder` (buffering, SQ8 encoding, build options, sections) |
| `kernels.rs` | float32 / SQ8 dot products (AVX-512 FMA, AVX-512 VNNI), 4×4 block kernel, prefetch |
| `sq8.rs` | per-dimension 8-bit scalar quantisation, query preparation |
| `ivf.rs` | spherical k-means, the cluster layout, probing (filtered and not) |
| `hnsw.rs` | parallel graph construction, fixed-degree layout, search (traverse / two-hop) |
| `cost.rs` | `Work` (operation counts), `CostModel` (ns per operation), `Auto` |
| `exact.rs` | blocked brute force (ground truth, k-means assignment) |
| `io.rs` | `madvise(WILLNEED)` read-ahead hints for cold segments |
| `topk.rs` | bounded top-k, the filter adapter (`Members`) |

Experiment binary: `src/bin/strata-vector-bench.rs`; results in
`results/vector/<dataset>.json`. Tests: unit tests in each file,
`tests/vector_search.rs` (synthetic: exactness, filter semantics, missing
vectors, zero-copy, options, hints), `tests/vector_beir.rs` (SciFact recall).

No dependencies were added.

## Changes outside the scope

Both additive; nothing else outside `src/vector/`, the bench and the tests
was touched.

1. `segment/filter.rs`: `DocFilter::as_doc_set(&self) -> Option<&DocSet>`,
   a provided method returning `None`, overridden by `DocSet` to return
   itself. Without it a search can only probe `contains` for every one of
   the `n` documents, so a pre-filter scan of a 0.1% filter on Quora would
   cost ~500k virtual calls instead of 523 code reads. Other `DocFilter`s
   (e.g. storage's `LiveDocs`) keep working through probing.
2. `segment/mod.rs`: `SegmentWriter::write_to(self, w: &mut impl Write) ->
   io::Result<usize>` streams the segment image section by section; `finish`
   now calls it (same bytes, checked by a new test) through a shared
   `layout()`. Quora's IVF segment is 3.2 GB of sections; `finish` would
   need a second 3.2 GB image.

## Public API

```rust
// strata::vector
pub fn field_builder(field: &Field, opts: &BuildOptions) -> Result<Box<dyn FieldIndexBuilder>>;
pub fn section(field: &str, part: &str) -> String;           // "vec.<field>.<part>"

pub struct VectorReader { .. }
impl VectorReader {
    pub fn open(seg: &SegmentFile, field: &str) -> Result<Self>;   // parses the JSON header and
                                                                   // the IVF directory; all arrays are views
    pub fn dim(&self) -> usize;
    pub fn n_docs(&self) -> u32;
    pub fn index_kind(&self) -> &str;                              // "flat" | "ivf" | "hnsw"
    pub fn info(&self) -> &VectorInfo;                             // the header (nlist, m, entry, ...)
    pub fn shape(&self) -> &IndexShape;                            // what the cost model needs
    pub fn vector(&self, doc: DocId) -> &[f32];
    pub fn similarity(&self, doc: DocId, query: &[f32]) -> f32;    // exact dot product
    pub fn has_vector(&self, doc: DocId) -> bool;                  // NULL vectors are never returned
    pub fn codes(&self, doc: DocId) -> &[u8];                      // SQ8 row (padded to 64 bytes)
    pub fn sq8(&self) -> &Sq8;
    pub fn search(&self, query: &[f32], k: usize, params: &SearchParams,
                  filter: Option<&dyn DocFilter>) -> SearchOutput;
    pub fn exact(&self, query: &[f32], k: usize, filter: Option<&dyn DocFilter>) -> Vec<Hit>;
    pub fn estimate_cost(&self, strategy: Strategy, k: usize, filter_cardinality: Option<usize>,
                         params: &SearchParams) -> f64;            // µs; Auto = cheapest
    pub fn plan(&self, k: usize, filter_cardinality: Option<usize>, params: &SearchParams) -> Strategy;
    pub fn cost_model(&self) -> &CostModel;
    pub fn set_cost_model(&mut self, cost: CostModel);
    pub fn set_io_hints(&mut self, on: bool);                      // default off
    pub fn set_hnsw_filter(&mut self, mode: HnswFilterMode);       // default Traverse
    pub fn is_zero_copy(&self) -> bool;
}
pub enum Strategy { Auto, Flat, Ivf, Hnsw, PreFilter, PostFilter }   // as_str, parse, ALL
pub struct SearchParams { pub strategy: Strategy, pub nprobe: usize, pub ef: usize, pub rerank: usize }
    // Default: Auto, nprobe 32, ef 64, rerank 40; HNSW searches with max(ef, rerank)
pub struct SearchOutput { pub hits: Vec<Hit>, pub stats: SearchStats }
pub struct Hit { pub doc: DocId, pub score: f32 }                // best first, ties to smaller doc
pub struct SearchStats {
    pub strategy: Strategy,          // what ran (never Auto)
    pub sq8_distances: usize, pub sq8_contiguous: usize, pub f32_distances: usize,
    pub centroids: usize, pub candidates: usize, pub clusters_probed: usize,
    pub nodes_visited: usize, pub filter_checks: usize, pub retries: usize,
    pub estimated_us: f64,           // the cost model's prediction for `strategy`
}
pub enum HnswFilterMode { Traverse, TwoHop }
pub struct VectorInfo { dim, n, missing, index, stride_f32, stride_u8, nlist, train_points,
                        iters, m, ef_construction, entry, max_level }
pub fn sort_hits(hits: &mut [Hit]);
pub fn default_nlist(n: usize) -> usize;                                    // ≈ 4·√n
pub fn filtered_probes(nprobe: usize, selectivity: f64, nlist: usize) -> usize; // nprobe/√s
pub fn filtered_probe_weight(members: usize, size: usize, selectivity: f64) -> f64;

// strata::vector::cost (re-exported: CostModel, IndexShape, Work)
pub struct Work { pub scan, pub gather, pub rerank, pub centroids, pub clusters, pub checks, pub nodes: f64 }
impl Work { pub fn from_stats(s: &SearchStats) -> Work; pub fn features(&self) -> [f64; 8]; }
pub struct IndexShape { pub n: usize, pub kind: String, pub nlist: usize, pub hnsw_m: usize }
pub struct CostModel { pub fixed_ns, scan_ns, gather_ns, rerank_ns, centroid_ns, cluster_ns,
                       check_ns, node_ns, hnsw_dist_a, hnsw_dist_b, hnsw_nodes_a, hnsw_nodes_b,
                       post_margin: f64 }                           // Default = calibrated
impl CostModel {
    pub fn work(&self, shape: &IndexShape, s: Strategy, k: usize, card: Option<usize>, p: &SearchParams) -> Work;
    pub fn estimate(&self, shape: &IndexShape, s: Strategy, k: usize, card: Option<usize>, p: &SearchParams) -> f64;
    pub fn choose(&self, shape: &IndexShape, k: usize, card: Option<usize>, p: &SearchParams) -> (Strategy, f64);
    pub fn candidates(shape: &IndexShape, filtered: bool) -> Vec<Strategy>;
    pub fn time_us(&self, w: &Work) -> f64;
    pub fn coefficients(&self) -> [f64; 8]; pub fn set_coefficients(&mut self, c: [f64; 8]);
}
pub fn resolve(shape: &IndexShape, s: Strategy, filtered: bool) -> Strategy;

// strata::vector::{kernels, sq8, exact}
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32;                  // + *_scalar references
pub fn dot_u8_i8(a: &[u8], b: &[i8]) -> i32;
pub fn dot_u8_i8_x4(rows: [&[u8]; 4], q: &[i8]) -> [i32; 4];
pub fn dot_u8_f32(a: &[u8], w: &[f32]) -> f32;
pub fn dot_f32_4x4(a: [&[f32]; 4], b: [&[f32]; 4]) -> [[f32; 4]; 4];
pub struct Sq8 { pub min: Slice<f32>, pub scale: Slice<f32> }   // fit, encode, decode, prepare
pub struct QueryCode { pub q8: Vec<i8>, pub w: Vec<f32>, pub alpha: f32, pub bias: f32 } // dot, score
pub fn exact_top_k(rows: &[&[f32]], queries: &[&[f32]], k: usize,
                   filter: Option<&dyn DocFilter>, threads: usize) -> Vec<Vec<Hit>>;
pub fn block_dots(a: &[&[f32]], b: &[&[f32]], tile: usize, f: impl FnMut(usize, usize, f32));
```

Strategy resolution: `Flat` with a filter is `PreFilter` and vice versa; an
index strategy on a segment without that index falls back to the scan (so
the planner can ask for `Hnsw` on an in-memory flat segment); `PostFilter`
without a filter is the plain index search.

### Build options (`BuildOptions` keys)

| key | default | |
|---|---|---|
| `vector.index` | schema's `VectorIndexKind` | `flat`, `ivf`, `hnsw` |
| `vector.codes` | schema's `Field::codes` (`sq8`) | `sq8` or `binary` |
| `vector.ivf.nlist` | schema's `nlist` if > 0, else ≈ 4·√n | |
| `vector.ivf.iters` | 10 | k-means iterations |
| `vector.ivf.sample` | 64 | training points per list |
| `vector.hnsw.m` | schema's `m` (16) | layer 0 has 2·M |
| `vector.hnsw.ef_construction` | schema's (200) | |
| `vector.hnsw.build` | `sq8` with SQ8 codes | `sq8`: construction scores with the codes, as search does; `f32`: with float32 rows |
| `vector.threads` | 1 | k-means and graph construction |
| `vector.seed` | fixed | sample, HNSW levels |

A field that is not `indexed` stores the forward vectors only (flat).

## On-disk layout

| section | contents |
|---|---|
| `vec.<f>.meta` | JSON `VectorInfo` |
| `vec.<f>.sq8p` | per-dimension min and scale (f32) |
| `vec.<f>.present` | bitset of documents with a vector, only if some are NULL |
| `vec.<f>.f32` | `n × stride` f32, rows padded to 16 floats (one 4 KiB page per vector at 1024 d) |
| `vec.<f>.sq8` | `n × code_stride` u8, rows padded to 64 bytes (1 KiB at 1024 d) |
| `vec.<f>.bin`, `vec.<f>.binp` | instead of `sq8`/`sq8p` for a field with binary codes: `n × ⌈d/64⌉` u64 sign bits (128 B at 1024 d), and the per-dimension threshold and spread |
| `vec.<f>.ivf_centroids` | SQ8 ranges of the centroids, then one code row per centroid |
| `vec.<f>.ivf_dir` | per cluster: byte offset (u64) and size (u32) |
| `vec.<f>.ivf` | clusters back to back, 64-byte aligned: docIDs (u32, ascending), pad, code rows |
| `vec.<f>.hnsw` | layer 0: `n × 2M` u32, padded with `u32::MAX` (128 bytes per node at M = 16) |
| `vec.<f>.hnsw_upper` | per upper level: sorted node ids, `n_l × M` links |

Opening parses the JSON header, the `sq8p` and `ivf_dir` arrays as views,
and checks the IVF offsets (`nlist` entries); no per-vector array is copied
or decoded (`is_zero_copy()`; tested after `SegmentFile::from_vec`). NULL
vectors are stored as zeros, excluded from the IVF lists and the graph, and
skipped by scans.

Sizes (1024 d): float32 4096 B + SQ8 1024 B per document; IVF adds a second
copy of the codes plus 4 B of docID (+25%), HNSW 128 B of layer-0 links
(+2.5%). TREC-COVID (171k): flat 837 MiB, IVF 1006 MiB (centroids 1.6 MiB),
HNSW 858 MiB (graph 21.6 MiB). The IVF docIDs are plain u32: they are 0.4%
of a cluster's bytes, so compressing them (Loom/Pyramid) would not change
I/O, and the intersection with a filter is a bitset probe per docID either
way.

## How measurements were taken

All numbers come from `strata-vector-bench` on the 4-core development
machine while four other agents were building and benchmarking (load
average 5–13): relative comparisons hold, absolute latencies are noisy (a
preempted query adds ~4 ms, visible in the p99 columns of the JSON).
Searches run on one thread, builds on 2 threads. Recall@10 is against the
exact float32 top-10 (top-100 cached in `<data>/<name>/vec_gt_top100.bin`).
Latencies are means over all timed queries after a 20-query warm-up, with
the segment written to a file and mapped (warm page cache). To re-run in
isolation (`--threads 4` for the final build numbers):

```
cargo run --release --bin strata-vector-bench -- --dataset fiqa --queries 648
cargo run --release --bin strata-vector-bench -- --dataset trec-covid --queries 50 --filter-queries 50
cargo run --release --bin strata-vector-bench -- --dataset quora --queries 1000 --cold-queries 10 \
    --rerank 10,20,40 --nprobe 1,2,4,8,16,32,64,128,256 --budget-secs 10
```

`--kinds`, `--nlist`, `--m`, `--ef-construction`, the sweep lists, the
operating point (`--default-nprobe/-ef/-rerank`), `--selectivities`,
`--budget-secs` (per filtered cell), `--keep`/`--reuse` (segment files) and
`--tag` are configurable.

## Design decisions and the measurements behind them

### SQ8 scoring: quantised query and VNNI

Documents are ranked by `Σ v_j·c_j` with u8 codes `c` and the query's
weights `q_j·scale_j` quantised to i8 (`v`); the per-query constant
`Σ q_j·min_j` does not affect the order, so no per-vector correction term is
stored (unlike L2). One `vpdpbusd` covers 64 dimensions.

| kernel (1024 d, in cache) | FiQA run | TREC-COVID run | Quora run |
|---|---|---|---|
| float32 dot (AVX-512 FMA) | 49.8 ns | 70.8 ns | 59.0 ns |
| SQ8 × i8 query (VNNI) | 12.7 ns | 10.0 ns | 15.0 ns |
| … four rows per query load (`x4`) | 8.1 ns/row | 7.8 ns/row | 10.6 ns/row |
| SQ8 × f32 query (asymmetric, widen + FMA) | 47.7 ns | 60.3 ns | 63.9 ns |
| SQ8 × i8, scalar | 78.2 ns | 88.2 ns | 80.5 ns |

Ranking quality of SQ8 alone (top-10 of the SQ8 order against the exact
top-10, no re-ranking, 100 queries): VNNI 0.990 / 0.986 / 0.998, asymmetric
0.989 / 0.990 / 0.998. Quantising the query costs nothing measurable, and the
asymmetric kernel is 4–6× slower (6–8× against the four-row form).
Re-ranking depth (flat scan): 10 candidates → recall 0.992 / 0.986 / 0.996,
20 → 1.000 on all three. The default `rerank` is 40: index searches hand
over candidates that are already approximate, and 40 float32 rows cost
7–40 µs (a random row measured 180 ns on FiQA's 225 MiB of float32, 1.0 µs
on TREC-COVID's 669 MiB and Quora's 2 GiB).

Per-dimension min/max ranges (no clipping) are used: the round-trip error is
bounded by half a step per dimension (tested), and the SQ8-only recall above
leaves little to gain from percentile clipping.

### Binary codes

A field declared with `codes: "binary"` (or built with `vector.codes =
binary`) keeps one bit per dimension instead of SQ8's byte: whether the
value is above the segment's per-dimension mean. Centring matters: learned
embeddings share a common direction, and signs taken around zero would put
most of each code's bits in agreement. A code stands for `t + a∘s`
(threshold `t`, per-dimension mean absolute deviation `a`, signs `s`), so a
document ranks by `Σ_{bits set} q_j·a_j`, the query's weights quantised to
int8 (the query keeps 8 bits per dimension; only documents are reduced).

Kernel: on AVX-512BW the code word is the load mask for the 64 weight bytes
(`vmovdqu8` with a zeroing mask) and `vpsadbw` sums them, 64 dimensions per
step; a `popcnt` of the word removes the +128 offset. Elsewhere the weights
are split into 8 bit planes and each plane is ANDed with the code and
counted (9 popcounts per 64 dimensions); both give the same integer, tested
against a one-bit-at-a-time reference.

Binary codes order candidates more loosely, so a search rescores
`BINARY_RERANK = 8` times `rerank` candidates with float32 (320 by
default), and a post-filter search keeps 8 times as many members.
Everything else (flat scan, IVF lists, HNSW walk, every filter strategy)
runs unchanged over the binary rows, because all of them score through
`QueryCode`.

500,000 synthetic 768-d unit vectors (1,000 centres sharing a common
direction, noise 1.5 per dimension), 200 queries, cloud container (Intel
Xeon 2.80 GHz, 4 cores, AVX-512 VNNI and BW, no VPOPCNTDQ), one search
thread, builds on four (`strata-codes-bench`,
results/vector/codes-500000x768.json and, for the breadth-320 HNSW rows,
codes-500000x768-hnsw-ef320.json):

| codes | index | rescored | codes in memory | recall@10 | p50 ms |
|---|---|---|---|---|---|
| SQ8 | flat | 40 | 384 MB | 1.000 | 60.0 |
| binary | flat | 160 | 48 MB | 0.917 | 14.2 |
| binary | flat | 320 | 48 MB | 0.994 | 17.2 |
| SQ8 | IVF (nprobe 32) | 40 | 384 MB | 1.000 | 1.80 |
| binary | IVF (nprobe 32) | 160 | 48 MB | 0.917 | 0.66 |
| binary | IVF (nprobe 32) | 320 | 48 MB | 0.994 | 0.89 |
| SQ8 | HNSW (ef 64) | 40 | 384 MB | 0.868 | 0.54 |
| SQ8 | HNSW (ef 320) | 40 | 384 MB | 0.993 | 0.86 |
| binary | HNSW (breadth 320) | 320 | 48 MB | 0.991, 0.985 | 0.95, 0.87 |

The float32 rows (1,536 MB) are read only for the rescored candidates.
HNSW searches with breadth `max(ef, rescored)`, so binary codes at the
default rerank of 40 walk the graph 320 wide: compare them with SQ8 at
ef 320, not ef 64. The two binary HNSW figures are two builds (threaded
builds differ). So, with 320 candidates rescored, binary codes make the
flat scan 3.5× faster and IVF 2.0× faster than SQ8 at recall 0.994
against 1.000, and HNSW level with SQ8 at equal breadth (0.985–0.991
against 0.993), each holding an eighth of the code memory.
Synthetic clusters are easy for IVF; real embeddings lose more to binary
codes, and how much depends on the model, so measure recall on the
collection before choosing them.

### Flat scan

Memory-bandwidth bound: FiQA's 57.6k codes in 2.0 ms (35 ns/code, mostly
L3-resident), TREC-COVID's 171k in ~9.4 ms (55 ns/code, 18 GB/s), Quora's
523k in ~39 ms (75 ns/code, 13 GB/s with the other agents' traffic). It is the
exact answer (up to SQ8 ties: recall 1.000 at rerank ≥ 20) and the fallback
for every index strategy on segments without that index.

### IVF

* **Training**: spherical k-means (normalised centroids, assignment by dot
  product), random initial centroids, 10 Lloyd iterations on 64 points per
  list, empty clusters re-seeded from the largest. On SciFact (nprobe 16 of
  288): no iterations 0.792, 4 → 0.881, 10 → 0.877, 20 → 0.879, so 10 is
  past the knee. Ranking probes by the unnormalised cluster mean instead of
  the normalised centroid: 0.872 vs 0.877. Build (2 threads): FiQA 17 s,
  TREC-COVID 38 s, dominated by the 4×4-blocked assignment.
* **nlist ≈ 4·√n** (FiQA 960, TREC-COVID 1656, Quora 2893). At the same
  fraction of the data scanned, more lists were slightly better on SciFact
  (nlist 72/144/288 at 5.5%: 0.770/0.838/0.877).
* **SQ8 centroids.** Every query scores all centroids. In float32 that is
  `nlist × 4 KiB` (3.8 MiB for FiQA), measured at 160 ns per centroid, which
  dominated small probes (FiQA nprobe 16: 204 µs). As SQ8 codes scored with
  VNNI: 90 µs, recall unchanged (0.909). Centroids are stored only as codes.
* **Default nprobe 32**: recall 0.947 (FiQA), 0.950 (TREC-COVID), 0.936
  (Quora); nprobe 16 gives 0.903 / 0.938 / 0.904. Filtered at nprobe 32,
  Quora: random filters 0.961 / 0.945 / 0.938 / 0.927 at 0.1 / 1 / 3 / 10%,
  topic filters 0.915 / 0.913 / 0.918 / 0.936.
* **Filtered probing** (the stopping rule). A filtered search must probe
  further than `nprobe`: the filtered top-k of a filter with selectivity `s`
  sits near rank k/s of the unfiltered order. Rules tried (TREC-COVID,
  nprobe 16, recall@10 against the exact filtered top-10; `topic` filters
  select the query's nearest topics, i.e. whole k-means regions):

  | rule | random 1% | random 10% | topic 1% | topic 3% | topic 10% |
  |---|---|---|---|---|---|
  | probe `nprobe/√s` clusters | 0.958 | 0.934 | 0.690 | 0.642 | 0.720 |
  | score `nprobe·avg·√s` members | 0.958 | 0.934 | 0.746 | 0.750 | 0.774 |
  | stop at `√(members·docIDs) ≥ nprobe·avg` | 0.952 | 0.928 | 0.692 | 0.666 | 0.756 |
  | **each cluster counts `min(1, f_c/√s)`** | 0.964 | 0.938 | 0.832 | 0.820 | 0.818 |
  | same, nprobe 32 (default) | 0.990 | 0.964 | 0.916 | 0.904 | 0.884 |

  With nprobe = nlist the filtered search equals the pre-filter (recall
  1.000), so the loss is probing depth. A topic filter selects whole IVF
  clusters, often not the ones nearest the query; budgets counted in
  clusters or members stop after a handful of member clusters. Counting each
  probed cluster by its own member fraction `f_c` (relative to √s, capped at
  1) means `nprobe/√s` clusters for a filter independent of the query, and
  about `nprobe` clusters *of members* when members come in whole clusters,
  wherever they start in the probe order. The remaining gap on topic filters
  is IVF's known weakness for queries far from the probed region (centroid
  similarity discriminates poorly there).
* **Sparse filters as bitsets.** A sorted filter (`DocSet` keeps up to n/32
  members sorted) was intersected with each cluster's docIDs by galloping.
  Converting it to a bitset once per query (n/8 bytes) made filtered IVF
  4–6× faster on TREC-COVID (0.1%: 1919 → 311 µs; 1%: 1191 → 185 µs; 3%:
  1136 → 285 µs). HNSW searches use the same conversion.

### HNSW

* **Construction**: hnswlib's algorithm (heuristic neighbour selection,
  2·M links on layer 0, per-node locks, levels from a seeded hash so the
  layer structure does not depend on the thread count). Inserts are
  memory-latency bound (4 KiB per float32 vector, random): prefetching
  the next neighbour's vector cut FiQA's build from 50.5 s to 33.5 s
  (prefetching in the selection heuristic and link shrinking was neutral).
  Build with 2 threads under load, float32 distances: FiQA 54 s,
  TREC-COVID 113 s, Quora 593 s.
* **Construction on SQ8 codes** (`vector.hnsw.build`, default `sq8`): the
  inserted vector is prepared as a query (int8 weights, as searches are)
  and scored against the others' codes, a byte per dimension; the
  selection heuristic prepares each neighbour it keeps the same way.
  TREC-COVID, 4 threads, idle container: 36.6 s against 82.0 s on float32
  rows, recall@10 within 0.014 at every ef from 10 to 256 (one run each).
  At 64 dimensions it saves nothing (a float32 row is 256 bytes, and
  preparing a query per kept neighbour costs about what it saves).
* **Merges** start from the inputs' graphs (`GraphStart`). The largest
  input that lost no documents keeps its levels and links (Lucene 9.9);
  every other whole input is joined to it (Lucene 10.2's
  `MergingHnswGraphBuilder`): a join set covering its graph (greedy, each
  node needing `max(2, ⌈degree/4⌉)` neighbours in the set, Lucene's
  `UpdateGraphsUtils.computeJoinSet`) is inserted as usual, then every
  other node searches layer 0 from its old neighbours already inserted
  (the join set, and the nodes before it) and their new neighbours. Four
  TREC-COVID segments of 25,000 (1,024 dimensions, 4 threads, 50 queries
  and 500 held-out documents, `examples/graph_start.rs`):

  | layer-0 beam of the other nodes | seeds | seconds | recall@10 ef 64 | ef 100 |
  |---|---|---|---|---|
  | `min(ef/2, 3M)` = 48 (Lucene) | join set | 10.3 | 0.984 | 0.991 |
  | `min(ef/2, 3M)` = 48 | join set and nodes before | 10.1 | 0.983 | 0.991 |
  | `ef/2` = 100 | join set | 12.1 | 0.988 | 0.992 |
  | `ef/2` = 100 (kept) | join set and nodes before | 12.0 | 0.990 | 0.995 |
  | largest input's graph only | | 15.9 | 0.991 | 0.995 |
  | from scratch | | 20.4 | 0.989 | 0.993 |

  One run each on codes; float32 rows: 20.9 to 21.1 s joined, 29.6 to
  30.0 s from the largest, 37.9 s from scratch.
* **Search on SQ8 codes** (1 KiB per hop instead of 4 KiB), four neighbours
  per VNNI call, then float32 re-ranking; `ef = max(ef, rerank)`. Default
  ef 64: recall 0.984 (FiQA), 0.988 (TREC-COVID), 0.991 (Quora) at 90–175
  µs.
* **Filtered HNSW: two traversals** (`HnswFilterMode`), TREC-COVID, recall@10
  / mean µs, random filters | topic filters:

  | s | traverse | two-hop (ACORN-1) |
  |---|---|---|
  | 0.1% | 0.996 / 36000 · 0.996 / 17774 | 0.016 / 21 · 0.158 / 44 |
  | 1% | 1.000 / 5150 · 0.992 / 7229 | 0.620 / 202 · 0.570 / 177 |
  | 3% | 0.996 / 1594 · 0.976 / 4136 | 0.928 / 244 · 0.600 / 147 |
  | 10% | 1.000 / 553 · 0.996 / 1021 | 0.984 / 274 · 0.864 / 176 |
  | 30% | 0.998 / 233 · 0.998 / 161 | 0.984 / 299 · 0.998 / 208 |

  *Traverse* (score and expand every reached node, collect members only)
  keeps recall at every selectivity because the whole graph stays
  connected; its work grows like `ef/s`, and at 0.1% it walks most of the
  graph. It breaks down on cost: slower than the pre-filter scan below ~10%
  and than the post-filter above. *Two-hop* (expand members only, looking
  through non-members to their neighbours, up to 2M members per expansion)
  costs about the same at every selectivity, but on a graph built with
  M = 16 the members' two-hop graph is not navigable below ~10% (about
  s·(2M)² members per two-hop neighbourhood) or when the members lie away
  from the query (topic filters), and recall collapses (0.62 at 1%, 0.07 at
  0.3%). ACORN avoids this by building a denser graph (γ·M links); not done
  here. Traverse is the default, and the planner rarely needs either
  (below).

### Cold cache and read-ahead hints

Latency and pages read per query at the operating point (segment file
evicted and unmapped before each query). Three ways to read: the kernel's
read-around on every fault (`normal`), one page per fault (`MADV_RANDOM`),
and `MADV_RANDOM` plus the reader's hints (`set_io_hints(true)`:
`madvise(WILLNEED)` on the probed clusters, the neighbours about to be
scored, pre-filter members, the re-rank rows, and a sliding 2 MiB window
for full scans).

| index | normal | random | random + hints | warm: hints off → on |
|---|---|---|---|---|
| FiQA flat | 28.9 ms, 15731 p | 524 ms, 14453 p | 31.0 ms, 14453 p | 1.79 → 2.38 ms |
| FiQA IVF (nprobe 32) | 14.6 ms, 3754 p | 52.7 ms, 862 p | **3.7 ms**, 862 p | 148 → 178 µs |
| FiQA HNSW (ef 64) | 53.3 ms, 14466 p | 35.4 ms, 1101 p | **14.0 ms**, 1101 p | 103 → 415 µs |
| TREC-COVID flat | 70.1 ms, 44144 p | 1605 ms, 42876 p | 95.5 ms, 42876 p | 9.0 → 10.8 ms |
| TREC-COVID IVF | 12.2 ms, 4449 p | 78.2 ms, 1435 p | **4.8 ms**, 1435 p | 241 → 281 µs |
| TREC-COVID HNSW | 79.6 ms, 19849 p | 57.4 ms, 946 p | **15.7 ms**, 946 p | 202 → 563 µs |
| Quora flat | 227 ms, 131853 p | 5644 ms, 130776 p | 308 ms, 130776 p | 41.3 → 46.5 ms |
| Quora IVF | 17.9 ms, 5278 p | 83.9 ms, 2409 p | **7.5 ms**, 2409 p | 575 → 621 µs |
| Quora HNSW | 125.8 ms, 32172 p | 44.3 ms, 1203 p | **30.0 ms** (p50 15.5), 1203 p | 148 → 548 µs |

IVF is the disk-friendly index, as designed: with hints a cold IVF query
reads only what it needs (TREC-COVID: 1435 pages for the centroids, 32
clusters and 40 re-rank rows) in two waves of parallel reads, 2.4–3.9×
faster than kernel read-around while reading 2.2–4.4× fewer pages. HNSW
stays latency-bound (one wave per hop) and takes 3–4× the IVF time even
though it reads fewer pages; hints cut it 3.8–5.1× against read-around.
Hints cost a system call per range, so they are off by default: +8–20% warm
for IVF, 2.8–4× for HNSW. The storage layer should map files `MADV_RANDOM` and turn hints on
for databases larger than memory (or after a cold open); full scans do as
well with plain read-around.

### Filtered strategies and `Auto`

* **PreFilter** enumerates the filter's members (sorted list or bitset via
  `DocFilter::as_doc_set`, else probing `contains` for all n), scores their
  codes four at a time with the next group prefetched, and re-ranks: exact up
  to SQ8 ties (recall 1.000 in every cell). Cost ∝ members.
* **Ivf** in-filter: the per-cluster rule above; cost ∝ `nprobe/√s` clusters
  of docID checks plus the members' codes.
* **Hnsw** in-filter: traverse (default) or two-hop, above.
* **PostFilter**: an unfiltered index search for `k·2/s` candidates (HNSW
  `ef`, IVF `nprobe` scaled by the same factor, flat: the whole scan), then
  the filter, retried with 4× the candidates if fewer than k members
  survive. Filters correlated *away* from the query need retries (TREC-COVID
  topic 1%: 9.6 ms vs 1.8 ms for a random 1% filter).
* **Auto** asks the cost model for the cheapest applicable strategy: Flat or
  the native index without a filter; PreFilter, the native in-filter
  strategy or PostFilter with one.

Measured with random filters (mean µs, recall@10 in parentheses where below
0.99; every cell, both filter types and the HNSW traversals are in the JSON,
`filtered`):

| s | TREC-COVID HNSW: pre / post / Auto | Quora HNSW: pre / post / Auto | TREC-COVID IVF: pre / ivf / Auto | Quora IVF: pre / ivf / Auto |
|---|---|---|---|---|
| 0.1% | 22 / 15423 / 15 → pre | 33 / 26073 / 23 → pre | 15 / 332 / 12 → pre | 24 / 446 (0.961) / 22 → pre |
| 1% | 82 / 1810 / 68 → pre | 294 / 3007 / 306 → pre | 78 / 193 / 79 → pre | 279 / 339 (0.945) / 273 → pre |
| 3% | 296 / 666 (0.988) / 289 → pre | 853 / 1047 / 907 → pre | 288 / 203 (0.970) / 204 → ivf | 788 / 343 (0.938) / 337 → ivf |
| 10% | 938 / 261 (0.982) / 244 → post | 2971 / 359 (0.988) / 365 → post | 828 / 211 (0.964) / 221 → ivf | 2754 / 420 (0.927) / 356 → ivf |
| 30% | 2413 / 108 (0.980) / 92 → post | 9067 / 151 (0.981) / 140 → post | 2457 / 237 (0.960) / 304 → ivf | 9013 / 486 (0.928) / 446 → ivf |
| 100% | 8796 / 100 (0.988) / 92 → post | 40911 / 147 / 144 → post | 7456 / 237 (0.950) / 269 → ivf | 38829 / 528 (0.933) / 544 → ivf |

On every collection, index and filter type, Auto picks the fastest strategy
among those within ~0.03 of the index's own unfiltered recall (IVF at
nprobe 32: 0.93–0.95; HNSW: 0.98–0.99), or one within measurement noise of
it (Auto's timing differs from the same strategy's own cell by up to ±30%
at these loads). The crossovers: below ~3% the pre-filter scan wins (it
touches only s·n codes and is exact); above ~3–10% the index wins
(post-filter on HNSW, in-filter on IVF), and the crossover moves to lower
selectivity as n grows (the scan costs s·n, the index searches do not grow
with n). In-filter HNSW wins only in one corner: Quora topic filters at 10%,
where traverse (486 µs, recall 0.992) and two-hop (281 µs, 0.980) beat the
post-filter Auto picked (744 µs); elsewhere traverse loses to the
pre-filter below ~10% and to the post-filter above, and two-hop loses
recall where it would be cheaper. On topic filters (members correlated with
the embedding) Auto makes the same choices; the IVF choice at 3–10% costs
recall there (TREC-COVID 0.904 / 0.884, Quora 0.918 / 0.936, against 1.000
for the pre-filter at 1.0–7× the time), because the model cannot see
correlation.

### Cost model

`estimate_cost` = Σ predicted operation counts × ns per operation:

| strategy | predicted work (n documents, s selectivity, R = max(k, rerank)) |
|---|---|
| Flat | n codes scanned, R re-ranks |
| PreFilter | s·n codes gathered, min(R, s·n) re-ranks |
| Ivf | nlist centroids; p = max(nprobe, R/avg) clusters, p·avg codes scanned (filtered: p = max(nprobe/√s, R/(s·avg)), p·avg docID checks, p·avg·s codes gathered) |
| Hnsw | (9.0·ef' + 252)·M/16 codes gathered, 0.99·ef' + 2.5 expansions, ef' = max(ef, R); filtered (traverse): both ÷ s, capped at n |
| PostFilter | the native search for k' = max(R, 2k/s) candidates, plus k' checks |

The per-operation costs are a non-negative least-squares fit (relative
error) of measured latency on measured counts (`Work::from_stats`) over the
71 runs of a bench (sweeps + every filtered cell except Auto), with the
re-rank cost fixed to a direct measurement (the re-rank depth hardly varies
across runs, so a fit cannot separate it from the fixed cost). The compiled
defaults come from TREC-COVID: scan 56 ns/code, gather 34 ns/code, re-rank
620 ns/row, centroid 39 ns, filter check 2.8 ns, HNSW expansion 590 ns
(neighbour list, heap, visited set), fixed 0. Median |error| of the
compiled-in model on the final runs: 23% (FiQA), 21% (TREC-COVID), 33%
(Quora); refitted per collection: 19% / 19% / 14%. Quora's fit (scan
74 ns, gather 40 ns, re-rank 1005 ns, expansion 1025 ns) shows what grows
with the working set: every memory-bound cost rises by 30–75% from 171k to
523k documents, so a single constant set under-predicts large segments
(it does not change the choices above, where the alternatives differ by
2–100×). The HNSW work curve (distances and expansions against ef') is
fitted separately by least squares: 10.9·ef' + 281 (FiQA), 9.0·ef' + 251
(TREC-COVID), 11.8·ef' + 272 (Quora); expansions ≈ 0.99·ef' + 2.4
everywhere.

Two artifacts of the benchmark bias the fit: TREC-COVID has only 50 queries,
repeated across sweep points, so their re-rank rows and gathered codes stay
in the 260 MB L3 (gather is fitted cheaper than scan, and an unconstrained
fit puts re-ranking at 0 with a 7 µs fixed cost and a 10.6% median error);
and the machine was loaded. A per-machine calibration (the bench prints
the fit; `set_cost_model` installs one) is the intended use.

## Results by collection

Operating points (k = 10, rerank 40; nprobe 32; ef 64), warm, one thread:

| | FiQA (57.6k) | TREC-COVID (171k) | Quora (523k) |
|---|---|---|---|
| exact dense nDCG@10 (reference ≈ 0.42 / 0.79 / 0.89) | 0.4214 | 0.8178 | 0.8872 |
| flat: recall@10, mean latency | 1.000, 2.1 ms | 1.000, 9.9 ms | 1.000, 40.7 ms |
| IVF nprobe 32: recall@10, latency | 0.947, 251 µs | 0.950, 334 µs | 0.936, 583 µs |
| HNSW ef 64: recall@10, latency | 0.984, 103 µs | 0.988, 90 µs | 0.991, 174 µs |
| nlist / HNSW levels | 960 / 3 | 1656 / 3 | 2893 / 5 |
| build, 2 threads: flat / IVF / HNSW | 0.3 / 17 / 54 s | 0.9 / 38 / 113 s | 6.7 / 145 / 593 s |
| segment size: flat / IVF / HNSW | 281 / 339 / 289 MiB | 837 / 1006 / 858 MiB | 2553 / 3069 / 2620 MiB |
| ground truth (exact top-100, 2 threads) | 0.5 s | 0.3 s | 104 s |

Recall/latency curves (sweeps of rerank, nprobe 1–256 and ef 10–256 with
nDCG@10, p50/p99 and operation counts) are in the JSON (`indexes[].curve`).
Quora, for example: IVF nprobe 8/16/64/128 → recall 0.862/0.904/0.960/0.977
at 260/368/1009/2048 µs; HNSW ef 16/32/128 → 0.967/0.984/0.993 at
68/92/276 µs. nDCG@10 of ANN results is within 0.01 of exact search from
recall@10 ≈ 0.95 on. Peak RSS: FiQA 0.48 GB, TREC-COVID 1.05 GB, Quora
3.28 GB (the IVF build holds the float32 rows, the codes and their clustered
copy; the segment is streamed to the file with `write_to`).

## Open issues

* **HNSW construction** is still the slowest build step. On SQ8 codes it
  reads a quarter of the bytes (2.2× faster on TREC-COVID); preparing a
  query per kept neighbour in the selection heuristic is now a large part
  of an insert, and caching those, or comparing neighbours code to code,
  would cut it further. Joining a merge's inputs' graphs is 1.3× faster
  than starting from the largest alone (1.4× on float32 rows; with the
  codes, 2.4× faster than float32 from the largest), but the nodes of
  inputs that lost documents are still inserted from scratch.
* **IVF centroid scan.** Every IVF query scores all `nlist` centroids:
  Quora's 2893 SQ8 centroids (2.9 MB, past L2) cost ~170 µs, most of a
  query at nprobe ≤ 8 (nprobe 1: 205 µs). A small HNSW over the centroids,
  or fewer lists for large segments, would remove it.
* **In-filter HNSW** is rarely the cheapest strategy here (one corner:
  Quora topic filters at 10%). A generally useful one needs ACORN-γ (a
  denser graph built for filtering, γ·M links) or per-value subgraphs; the
  planner meanwhile uses pre-filter / post-filter, and the cost model only
  models the traverse mode.
* **Correlated filters.** The cost model assumes filters independent of the
  query. Filters whose members lie away from the query make post-filtering
  retry (TREC-COVID topic 1%: 9.6 ms) and cost IVF recall (0.88–0.94 at
  3–10%); Auto avoids the former only because the pre-filter is cheaper
  there anyway. A planner that knows a filter's correlation (e.g. from a
  sample of the filter's members' scores) could prefer the exact scan.
* **Recall is not in the cost model**: Auto chooses by time, assuming each
  strategy at its parameters is "good enough". The IVF in-filter recall at
  default nprobe is 0.88–0.99, lower than pre-filter (exact) and post-filter
  on HNSW (≥ 0.96).
* **Benchmark noise**: everything was measured with four other agents
  running; the numbers above are relative. The cost model's defaults should
  be refitted from the isolated run (`calibration.model` in the JSON).
* **`vector.threads` defaults to 1** (an embedded library should not grab all
  cores during a background merge); bulk loads should pass more.
* **Visited set**: HNSW allocates an n-bit visited bitset per query (65 KB on
  Quora, a few µs); a per-thread epoch array would avoid it.
* **Filters the reader cannot enumerate** (e.g. storage's `LiveDocs`) are
  probed for all n documents by the pre-filter; the query engine should
  intersect live documents into the predicate's `DocSet` first.
* **Single-threaded search**: intra-query parallelism (splitting the flat
  scan or the IVF probes) is left to the engine (per segment).
