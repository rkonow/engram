---
title: Results
parent: Strata
nav_order: 3
---

# Strata: results
{: .no_toc }

The design holds up. One file, with ACID commits and a SQL front end, serves full text, structured filters, vectors and their combinations. Warm, it is faster than both specialised baselines on most of the workload, and its ranking quality matches theirs. It pays for that in build time and in cold structured filters.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

## Summary

| goal | met | missed |
|---|---|---|
| **One embedded file for three index families** | Every query of the workload, and ten compound statements, run as SQL against one file. Durable commits take 4.1 ms, nearly all of it macOS's `F_FULLFSYNC`; every crash test recovered a committed prefix; snapshots cost at most 0.13 µs; the file is near SQLite's size. | A query reads one table (it joins a table only with itself). Building the HNSW graph makes loads 1.7–5.9× slower than SQLite's (0.2–0.5× without it). |
| **Faster than the specialised engines** | Warm BM25 is 1.7–3.2× faster than Tantivy and conjunctions 2.7–4.1× faster; Strata wins 24 of 28 warm text and filter comparisons. Vector queries take 0.05–0.10 ms against SQLite's 7.7 ms to 0.7 s scans, and are 28–63× faster cold. | Tantivy counts facets 1.1–2.1× faster on three of the four collections. Cold structured filters are 1.4–3.4× slower than Tantivy's. |
| **Hybrid ranking as one plan** | RRF and weighted fusion are single SQL expressions under the same filter as their inputs. Weighted fusion adds 0.02–0.04 nDCG@10 over vectors alone on three collections, at 0.2–0.8 ms. | The top-100 shortcut recovers 89–95% of the exact fused top 10. The best weight differs by collection (0.1–0.3); the relevance lab picks it, the normaliser and RRF's `k` from judged queries ([score fusion](#score-fusion)). |
| **A planner that adapts per query** | For filtered vector search, 1.8–6.1× faster than the best fixed strategy on the three larger collections, with recall@10 0.991–0.998. For filtered BM25 and filters alone, geometric-mean regret is 1.06–1.18. | It spends 1.0–1.6× an oracle's time. On SciFact, always pre-filtering takes 17% less time. |

## Setup

**Data.** Four BEIR collections with Cohere embed-english-v3 vectors (1,024 dimensions) for documents and queries, their relevance judgments, and synthetic structured fields: `year`, `category` and `rating` drawn at random, and `topic`, the k-means cluster of the same embeddings, which correlates with vector queries by construction.

| collection | documents | Strata file | build |
|---|---|---|---|
| SciFact | 5,183 | 33 MiB | 0.5 s |
| FiQA | 57,638 | 334 MiB | 14.2 s |
| TREC-COVID | 171,332 | 1,031 MiB | 40.6 s |
| Quora | 522,931 | 2,689 MiB | 153.1 s |

**Baselines.** SQLite with FTS5, B-tree indexes on the structured fields and vectors as blobs scanned by a scalar function (it has no vector index here), and Tantivy 0.26 (no vectors). All three systems answer the same workload of BM25, boolean, filter, facet, vector and hybrid queries, are measured after a full merge into one segment, and run queries on one thread. Warm numbers are medians over the queries of a kind, each run three times; cold numbers evict the database from the page cache before each query.

**Machine.** A MacBook Air with an Apple M2 (8 cores: 4 performance and 4 efficiency, 24 GB), macOS 26.0.1, Rust 1.92.0, on mains power and otherwise idle. Vector scoring uses NEON: float32 dot products with FMA, and 8-bit codes with `usdot` (i8mm, 16 byte products per instruction). HNSW and IVF builds use four threads. These runs replace earlier ones on a 4-core cloud VM with an Emerald Rapids Xeon (AVX-512); [Reproducing the results](../reproduce.md#what-changed-on-the-laptop) compares the two.

## Build time and size

Strata builds every index of the table in one pass: text, structured, doc store, keys, float32 and 8-bit vectors, and an HNSW graph. That takes 0.5 s for SciFact and 153.1 s for Quora, against 0.3–30.7 s for SQLite and 0.4–9.3 s for Tantivy, neither of which builds a vector index. The graph is most of the difference: 12.9 s of FiQA's 14.2 s, 37 s of TREC-COVID's 40.6 s and 146 s of Quora's 153.1 s, on four threads. The rest of the load is 0.2–0.5× SQLite's time; with IVF instead of HNSW, TREC-COVID builds in 19.2 s and Quora in 60.1 s.

The Strata and SQLite files are close in size because both are dominated by float32 vectors, 4 KiB per document (83% of Strata's TREC-COVID file, 97% of Quora's). Without vectors, Strata's text index is the smallest of the three: with positions it takes 44.5 MiB on TREC-COVID against 60.0 MiB for Tantivy and 70.0 MiB for FTS5. A posting, with its docID and frequency, costs 12.2–12.7 bits.

## Query latency

![Median warm latency by query kind on TREC-COVID: Strata against SQLite with FTS5 and Tantivy, log scale](../assets/img/chart-strata-latency.png)

*TREC-COVID, warm cache. The [full report](../reports/strata.html) has the chart for every collection, warm and cold.*

Warm, Strata is faster than Tantivy in 24 of the 28 text and filter comparisons:

- **BM25 top-10:** 1.7–3.2× faster than Tantivy, and 43–200× faster than FTS5, which scores every match.
- **Two-term conjunctions:** 2.7–4.1× faster than Tantivy.
- **Filtered BM25:** 3.1–4.0× faster than Tantivy.
- **Vector top-10:** 0.05–0.10 ms with HNSW. SQLite's scan takes 7.7 ms on SciFact and 0.7 s on Quora.
- **Hybrid queries:** 0.17–0.77 ms, running BM25 and the vector index and fusing their top 100s.

| median warm ms | SciFact | FiQA | TREC-COVID | Quora |
|---|---|---|---|---|
| BM25 top-10: Strata | **0.048** | **0.158** | **0.391** | **0.098** |
| BM25 top-10: Tantivy | 0.153 | 0.368 | 0.681 | 0.193 |
| BM25 top-10: SQLite FTS5 | 2.06 | 12.0 | 78.3 | 9.72 |
| vector top-10: Strata (HNSW) | **0.046** | **0.072** | **0.066** | **0.097** |
| vector top-10: SQLite (scan) | 7.66 | 82.3 | 265 | 695 |
| hybrid (RRF): Strata | **0.170** | **0.418** | **0.772** | **0.343** |
| hybrid (RRF): SQLite | 10.1 | 95.0 | 344 | 714 |

*Source: `strata/results/strata/strata_<collection>.json` and `strata/results/baselines/{tantivy,sqlite}_<collection>.json`.*

The exceptions are facet counts on SciFact, FiQA and Quora, where Tantivy is 1.1–2.1× faster (on TREC-COVID Strata is 1.5× faster), and SciFact's three-term conjunctions. These queries take 9–27 µs in Tantivy. At that size the fixed cost of a SQL statement shows: even a query that matches nothing is parsed, bound, planned and run. Strata also materialises the match before reading category codes, while Tantivy's collector reads its column inside the intersection.

**Cold**, every system waits on page faults and the gaps narrow. Text queries are within 0.5–1.5× of Tantivy (BM25 is faster on SciFact and TREC-COVID, 6.6 ms against 7.2 ms on the latter). Structured filters are 1.4–3.4× slower than Tantivy's columns, because a range touches histogram buckets, boundary blocks, zone maps and the key index, each on different pages. A cold vector query is 28–63× faster than SQLite's on the two large collections: HNSW touches about 3,700–4,400 pages where the scan reads all 239,000 or 592,000. Pages are counted in 4 KiB units, but macOS pages are 16 KiB, so a random fault reads four times the data of a 4 KiB page; random-access queries such as HNSW therefore read more and run slower cold here than on the earlier VM.

### Cold reads and page-cache hints

Cold latency depends on what the kernel reads around each page fault, and the right policy differs by section. With the kernel's default read-ahead everywhere, a cold HNSW query on TREC-COVID reads 7,364 pages and takes 117 ms, twice the pages it reads when the graph is advised random access. With no read-ahead anywhere, vector queries drop to 3,676 pages on TREC-COVID and 4,416 on Quora and take 102–115 ms, but BM25, facets and filters, which read postings and columns in order, become up to 1.5× slower. Strata's default advises random access only for the sections read at random (doc store, key index, dictionaries, float32 vectors, the graph and its codes), which keeps vector and hybrid queries at the random-access cost and BM25 at 0.8–1.0× its read-ahead time. Structured filters are the exception, 1.2–1.5× slower than with read-ahead everywhere: their key tie-breaks fault one page per key.

## Ranking quality

Strata's BM25 is within 0.018 nDCG@10 of both baselines on every collection, and no system is consistently ahead; the analysers differ (Snowball with stopwords dropped in Strata, Porter in FTS5, Snowball keeping stopwords in Tantivy). HNSW is within 0.006 of the exact scan on three collections; on the 300 Quora queries SQLite also ran, it scores 0.867 against 0.880 exact, with recall@10 0.989, and four queries that lost a relevant duplicate account for the whole difference.

| nDCG@10, BM25 | SciFact | FiQA | TREC-COVID | Quora |
|---|---|---|---|---|
| Strata | 0.689 | 0.252 | 0.658 | 0.756 |
| Tantivy | 0.687 | 0.251 | 0.667 | 0.758 |
| SQLite FTS5 | 0.684 | 0.254 | 0.641 | 0.760 |

All three systems drop NLTK's 153 English function words from BM25 queries, so the rankings are comparable. Keeping them (`stopwords => 0`) costs TREC-COVID 0.055 nDCG@10, whose queries are full questions, and gains Quora the same 0.055 (0.811 against 0.756): its questions are short, and "how", "why" or "can" carry much of their meaning.

### Ranking fixes

The engineering review's ranking fixes (query letters such as "d" and "t" kept, repeated query words weighted, BM25's document count and average length taken over the documents that have the field, and a word's typos and one-word synonyms sharing its document frequency) were measured on seven BEIR collections with `strata-bm25-eval`, BM25 through SQL with stopwords dropped, plain and with typo tolerance (`fuzzy => 1`). Before is the code just before the fixes; both ran at the same time in the cloud container (Intel Xeon @ 2.80 GHz, 4 cores, Linux), so their query times are not compared. Results: `strata/results/text/bm25_before.json` and `bm25_after.json`.

| nDCG@10 | SciFact | NFCorpus | ArguAna | FiQA | SCIDOCS | TREC-COVID | Quora |
|---|---|---|---|---|---|---|---|
| BM25, before | 0.689 | 0.327 | 0.310 | 0.252 | 0.157 | 0.658 | 0.756 |
| BM25, after | 0.691 | 0.329 | 0.362 | 0.253 | 0.156 | 0.664 | 0.756 |
| typo-tolerant, before | 0.626 | 0.270 | 0.212 | 0.148 | 0.119 | 0.151 | 0.689 |
| typo-tolerant, after | 0.673 | 0.294 | 0.331 | 0.207 | 0.139 | 0.565 | 0.724 |

Plain BM25 moves little except on ArguAna (+0.052), whose queries are whole arguments; every one of its 8,674 documents has text, so the gain comes from the query-side fixes (repeated words, kept letters), which this run does not separate. Typo tolerance gains on every collection, most on TREC-COVID (+0.414): before, a query word's rare typo neighbours scored with their own, higher IDF and could outrank the word itself.

### Cross-encoder reranking

BM25's top 100 ranked again by the cross-encoder `ms-marco-MiniLM-L-6-v2` in a second phase (`cross_encoder()`, ranking 7b), over every test query, with `strata-bm25-eval --variants plain,ce_rerank100`; p is the paired randomization test of nDCG@10 against BM25 over the queries (10,000 rounds). Results: `strata/results/text/bm25_ce_quality.json`; that run shared the cloud container with builds, so its query times are not latencies.

| | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| nDCG@10, BM25 | 0.691 | 0.329 | 0.362 |
| nDCG@10, reranked | 0.687 (p = 0.79) | 0.352 (p < 0.001) | 0.312 (p < 0.001) |
| recall@10, BM25 | 0.827 | 0.156 | 0.758 |
| recall@10, reranked | 0.803 | 0.162 | 0.666 |

The model, trained on MS MARCO's web questions, helps on NFCorpus's medical questions, not on SciFact's scientific claims, and hurts on ArguAna, where each query is a whole argument and the document to find is its counterargument, a task unlike answering a question. The BEIR paper measured the same reranking at 0.311 on ArguAna. The relevance gate keeps a cheaper variant, BM25's top 20 reranked on SciFact and NFCorpus (`strata/tests/fixtures/relevance_gate_cross_encoder.json`).

Latency, alone on the cloud container (Intel Xeon @ 2.30 GHz, 4 cores, Linux), the first 50 test queries of each collection (`bm25_ce_latency.json`), ms:

| p50 / p99 | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| BM25 | 0.07 / 1.19 | 0.04 / 0.65 | 1.10 / 8.32 |
| top 20 reranked | 997 / 1,320 | 872 / 1,151 | 1,000 / 1,645 |
| top 100 reranked | 4,789 / 5,866 | 4,512 / 5,380 | 5,225 / 7,269 |

About 50 ms a row on four cores. A server's `--max-rerank` defaults to 20 rows, whose p99 is a third of its default 5 s deadline; the top 100's median is about the whole deadline.

## Hybrid ranking

![Weighted hybrid ranking: nDCG@10 by BM25 weight on four collections, with RRF of the same two rankings as dashed lines](../assets/img/chart-strata-fusion.png)

Weighted fusion beats both of its inputs on three of the four collections. The best BM25 weight is small, 0.1–0.3: the score scales differ, and on these collections the embeddings are the stronger ranking. On TREC-COVID it reaches nDCG@10 0.851, against 0.815 for vectors alone, 0.835 for RRF and 0.658 for BM25; on Quora 0.920 against 0.899 and 0.860. On FiQA, where BM25 is weak, fusion only matches the vector ranking (0.437) and RRF falls below it (0.381), because RRF gives both rankings equal weight. Strata's RRF is within 0.005 of SQLite's RRF over the same two rankings.

The top-100 shortcut is cheap and close to exact. Against weighted ranking computed over every document, it recovers 89–95% of the exact top 10 with nDCG@10 within 0.01–0.03, at 0.2–0.8 ms per query, where the exact computation costs 0.9 ms on SciFact, 12 ms on FiQA and 62 ms on TREC-COVID.

## Score fusion

Which normaliser to put in a weighted sum, and whether a weighted sum beats RRF once each is tuned. The judged queries of each collection alternate between a dev half, on which each family's one parameter is chosen (the BM25 weight from 0.02 to 0.9, or RRF's `k` from 1 to 100), and a test half, which gives the numbers below. Inputs are `bm25(body, ...)` and `similarity(emb, :q)`, 100 candidates each.

| nDCG@10, test half | SciFact (150) | FiQA (324) | TREC-COVID (25) |
|---|---|---|---|
| BM25 | 0.697 | 0.259 | 0.718 |
| vectors | 0.702 | 0.407 | 0.839 |
| RRF, `k => 60` | 0.722 | 0.362 | **0.883** |
| RRF, `k` tuned | 0.731 (k 1) | 0.390 (k 1) | 0.883 (k 60) |
| `minmax`, weight tuned | **0.748** (0.3) | 0.417 (0.02) | 0.877 (0.2) |
| `tmm`, weight tuned | 0.743 (0.1) | **0.419** (0.02) | 0.877 (0.1) |
| `zscore`, weight tuned | 0.730 (0.2) | 0.417 (0.02) | 0.867 (0.15) |

Tuned convex fusion beats RRF with its usual `k => 60` on SciFact (+0.021 to +0.026, paired randomisation p 0.01 to 0.04) and FiQA (+0.055 to +0.057, p < 0.001), where BM25 is weak and RRF's equal votes pull the vector ranking down. On TREC-COVID RRF is 0.006 ahead, on 25 test queries (p 0.57 to 0.69). `tmm` and `minmax` are within 0.005 of each other everywhere; `zscore` is no better than either.

What sets `tmm` apart is that its ranking does not depend on how many candidates were read. The share of the top 10 at depth 400 that a shallower depth already returns:

| top-10 overlap with depth 400 | depth | SciFact | FiQA | TREC-COVID |
|---|---|---|---|---|
| `tmm` | 100 | 0.999 | 0.996 | 1.000 |
| `minmax` | 100 | 0.967 | 0.993 | 0.960 |
| RRF | 100 | 0.991 | 0.993 | 0.944 |
| `tmm` | 20 | 0.986 | 0.990 | 0.976 |
| `minmax` | 20 | 0.924 | 0.983 | 0.952 |
| RRF | 20 | 0.963 | 0.959 | 0.632 |

With `tmm` a row's value moves only when the best candidate changes; with `minmax` every row moves when the weakest one does. The normalisers read the same candidates and do the same work; this run shared its machine with a build, so it gives no latencies. It ran on a 4-core cloud container (Intel Xeon @ 2.10 GHz, 15 GB), Rust 1.97.0: `strata-eval --collections scifact,fiqa,trec-covid --studies fusion --study-queries 1000`, which writes `results/strata/fusion_<collection>.json`.

## Posting formats and top-k

The text index stores long lists (at least 512 documents) as [Loom](../posting-lists.md#loom) and short ones as BP128 blocks. Against BP128 for every list, on the same index:

- **Conjunctions** are where Loom pays off: AND is 7.6–11× faster (7.5 against 57 µs on FiQA, 21 against 230 µs on TREC-COVID), because Loom's cursors skip with sampled pointers instead of decoding whole blocks. OR gains nothing: the two formats are within 2% of each other on TREC-COVID and Quora, and Loom is 1.2× slower on FiQA.
- **Size** is 2–4% smaller, at 11.7–12.7 bits per posting including frequencies.
- **Ranked top-10** is close across formats (MaxScore within 0.95–1.0×). MaxScore beats Block-Max WAND by 1.9–2.6× on these queries.

## Vector indexes

![Recall@10 against median latency on Quora: HNSW ef sweep, IVF nprobe sweep and the exact SQ8 scan with re-ranking](../assets/img/chart-strata-recall-latency.png)

On Quora (522,931 vectors, 1,024 dimensions), HNSW reaches recall@10 0.960 at 0.036 ms and 0.990 at 0.086 ms (ef 64, the default). IVF needs 0.22 ms for 0.936 (nprobe 32) and 0.67 ms for 0.977. The exact scan of 8-bit codes with float32 re-ranking reaches 1.000 in 9.7 ms. HNSW took 146 s to build with four threads, IVF 53 s.

The quantised dot product costs 14 ns per vector with NEON `usdot` (9.8 ns when four rows are scored at once), against 70 ns in float32; on x86-64 the same kernel uses AVX-512 VNNI. Scoring on the codes alone already finds 99.8% of the exact top 10 over 100 sampled queries; float32 re-ranking settles the rest.

Cold, access patterns decide. With the kernel's default read-ahead, one HNSW query reads 9,672 pages (136 ms). Advising the graph and its codes `MADV_RANDOM` cuts that to 4,836 pages (113 ms), and prefetching each hop's neighbours to 28 ms. IVF reads its clusters sequentially, so it is fastest with read-ahead: 26 ms, or 10 ms with prefetching.

## Compressed vectors

A `VECTOR` field can keep 1-bit codes instead of 8-bit ones (`"codes": "binary"`), and searches then rescore 8 times as many candidates in float32. This was measured separately from the rest of this page: 500,000 synthetic 768-dimension unit vectors (1,000 centres sharing a common direction, as learned embeddings do), 200 queries, one search thread, on the 4-core cloud container (Intel Xeon at 2.80 GHz with AVX-512 VNNI and BW), not the M2.

| codes | index | rescored | codes in memory | recall@10 | p50 |
|---|---|---|---|---|---|
| 8-bit | flat | 40 | 384 MB | 1.000 | 60.0 ms |
| 1-bit | flat | 320 | 48 MB | 0.994 | 17.2 ms |
| 8-bit | IVF, nprobe 32 | 40 | 384 MB | 1.000 | 1.80 ms |
| 1-bit | IVF, nprobe 32 | 320 | 48 MB | 0.994 | 0.89 ms |
| 8-bit | HNSW, ef 320 | 40 | 384 MB | 0.993 | 0.86 ms |
| 1-bit | HNSW, breadth 320 | 320 | 48 MB | 0.985–0.991 | 0.87–0.95 ms |

Binary codes hold an eighth of the memory of 8-bit codes (a 32nd of the float32 vectors, 1,536 MB here, which are read only for the rescored candidates). The flat scan gets 3.5× faster and IVF 2.0× faster for 0.006 of recall; HNSW is level. HNSW searches as wide as the candidates it rescores, so the 1-bit rows walk the graph 320 wide and are compared with 8-bit codes at ef 320; at the default ef of 64, 8-bit HNSW reaches 0.868 on this data. The two 1-bit HNSW figures come from two builds.

The data is synthetic because the container could not download embeddings. How much recall binary codes cost depends on the embedding model, so measure it on the collection before choosing them. [`strata/notes/vector.md`](https://github.com/rkonow/strata/blob/main/strata/notes/vector.md#binary-codes) has every depth and the kernels; the numbers are in `strata/results/vector/`.

## MaxSim

`maxsim(emb, vectors)` runs one vector search per query vector and scores the union of their top 10 exactly against every query vector. Measured on 200,000 synthetic 384-dimension unit vectors (1,000 centres sharing a common direction, as for [compressed vectors](#compressed-vectors)), 100 lists of query vectors, `LIMIT 10`, one query thread, on the 4-core cloud container (Intel Xeon at 2.10 GHz, AVX-512 VNNI), not the M2:

| query vectors | HNSW recall@10 | HNSW mean | HNSW p99 | flat recall@10 | flat mean |
|---|---|---|---|---|---|
| `similarity` | 0.994 | 0.19 ms | 0.32 ms | 1.000 | 4.0 ms |
| 1 | 0.994 | 0.17 ms | 0.29 ms | 1.000 | 4.3 ms |
| 2 | 1.000 | 0.37 ms | 0.47 ms | 1.000 | 7.1 ms |
| 4 | 0.999 | 0.87 ms | 1.16 ms | 1.000 | 14.9 ms |
| 8 | 0.998 | 1.63 ms | 2.05 ms | 1.000 | 29.6 ms |
| 16 | 0.999 | 3.29 ms | 4.21 ms | 1.000 | 63.5 ms |
| 32 | 0.998 | 6.52 ms | 8.24 ms | 1.000 | 118.8 ms |
| 64 | 0.999 | 14.18 ms | 17.46 ms | 1.000 | 247.2 ms |

Recall is against the exact answer, each row's best dot product with the query vectors over every row. Each added query vector costs about one more search: 0.22 ms with HNSW and 3.85 ms with the flat scan of 8-bit codes (the slope of mean latency over the number of vectors), against 0.19 ms and 4.0 ms for one `similarity` search; scoring the union exactly adds little. With one vector `maxsim` is `similarity`. `strata-maxsim-bench` measures it; the numbers are in `strata/results/vector/maxsim-200000x384.json`.

## Multi-vector fields

A `VECTOR(d)[]` field holds several vectors per row, and `maxsim(field, vectors)` ranks rows by the sum over query vectors of each one's best match among the row's vectors (ColBERT's and ColPali's late interaction). Measured on ViDoRe's DocVQA subset (`vidore/docvqa_test_subsampled_tesseract`: 500 pages, 500 questions, one page answering each), with pages and questions encoded by `vidore/colSmol-256M` (128 dimensions; 1,060 patch vectors per page and 20 per question on average; encoded on the CPU in bfloat16), `LIMIT 5`, one query thread, on the 4-core cloud container (Intel Xeon at 2.10 GHz, AVX-512 VNNI), not the M2:

| layout | vectors a page | bytes a page | NDCG@5, cap 25 | p50 | NDCG@5, cap 200 (default) | p50 | p99 | NDCG@5, every page scored |
|---|---|---|---|---|---|---|---|---|
| binary codes searched, 8-bit codes scored (default) | 1,060 | 297 KB | 0.505 | 7.9 ms | 0.505 | 26.3 ms | 55.5 ms | 0.506 |
| the same, float32 kept to score with (`keep_f32`) | 1,060 | 704 KB | 0.507 | 11.7 ms | 0.506 | 46.2 ms | 94.3 ms | 0.507 |
| pooled at write, cosine ≥ 0.95 (`pool = 0.95`) | 666 | 187 KB | 0.503 | 7.5 ms | 0.503 | 18.9 ms | 38.5 ms | 0.503 |
| pooled at write, cosine ≥ 0.9 (`pool = 0.9`) | 410 | 115 KB | 0.497 | 7.3 ms | 0.497 | 14.3 ms | 28.6 ms | 0.497 |

The reference is the model's own float32 scores over every page: NDCG@5 0.507. BM25 over the pages' OCR text (the dataset's Tesseract output) scores 0.361 in 0.01 ms, and `rrf(bm25, maxsim)` 0.437: fusing with the weaker OCR ranking costs more than it adds here. Each query vector searches the field's binary codes (HNSW, 64 nearest vectors), a page's score from those searches picks the candidates, and the candidates are scored exactly with 8-bit codes; the cap is the number of candidates (`candidates =>`, at least `LIMIT`). Scoring every page exactly instead takes 1.4 s a question with 8-bit codes and 2.1 s with float32.

Most of the default's 297 KB a page is the 8-bit codes (128 bytes a vector) and the HNSW graph over the vectors (128 bytes a vector at `m = 16`); the binary codes are 16 bytes a vector, against 512 for float32. On 500 pages a cap of 25 already finds what 200 does, at a third of the time; the default stays at 200 for larger collections, where the right page is more often outside the first candidates. Pooling at write merges a page's vectors whose cosine similarity is at least the threshold into their mean: 0.9 keeps 39% of the vectors for 0.008 NDCG@5 and halves the time at the default cap. `strata-multivector-bench` measures it from what `strata/scripts/vidore_encode.py` writes; the numbers are in `strata/results/vector/multivector-docvqa_test_subsampled_tesseract.json`.

## Filtered vector search and the planner

Filtered vector search is where one fixed strategy cannot win:

- **Pre-filtered scan:** its cost grows with the number of matching documents.
- **HNSW with the filter inside:** cheap for permissive filters, very expensive for selective ones and for filters whose matches sit far from the query.
- **Post-filtering:** sensitive to both.

![Filtered vector search on Quora: mean time per strategy by filter selectivity, with the planner and the best plan per query](../assets/img/chart-strata-filtered-vectors.png)

*Quora; the filters include random ones and topic filters drawn near to and far from the query.*

Across 300 filtered vector queries per collection, the planner is **2.7× faster** than the best fixed strategy on TREC-COVID (121 ms in total against 325 ms for always pre-filtering) and **6.1× faster** on Quora (179 ms against 1,089 ms), with recall@10 0.991–0.998. Always using the HNSW index would take 2.9–6.1 s. Against an oracle that knows the fastest plan per query with recall ≥ 0.9, the planner's total time is 1.0–1.6× the oracle's.

Two refinements made that gap small (the timings in this list come from the development runs on the earlier Xeon VM):

1. **Selectivity near the query.** Planning on global selectivity alone, 100 TREC-COVID queries took 328 ms against an oracle's 48 ms; a 32-neighbour sample brought that to 61 ms. On Quora it still sent far-topic 10% filters into 60–100 ms graph walks where a 5 ms scan sufficed (1,076 ms in total). A second, 160-neighbour sample, taken when the first finds at most one match, brought Quora to 451 ms.
2. **Post-filtering costed with the smaller** of the global and local selectivity, since its candidate pool is sized from the global one.

On SciFact, with 5,183 documents, scanning a filter's members is never expensive, and planning, probing and the occasional index plan add 1–29 µs per query: always pre-filtering takes 17% less time there.

For BM25 with filters and for structured filters alone, geometric-mean regret is 1.06–1.18. Range filters choose between a zone-map scan and histogram bucket postings, and filtered BM25 chooses between driving scoring from the filter and probing it.

## Compound queries

Ten statements combine the workload's shapes: phrases with negation, prefixes, `OR` and `NOT` filters, both fusion methods, facets over a match, column ordering, and exact distance with a forced strategy. Each runs on every collection, and the same statement gets different plans as the data changes. The `OR` filter in the fourth query matches about 15% of documents everywhere: on SciFact the planner scans those 800 documents; on the larger collections it measures 11% near the query and post-filters the HNSW result. The RRF query's topic filter matches 11% of Quora globally but only 3.4% near the query, so the planner switches to an exact scan of its 58,767 members, at 3.1 ms.

## Transactions and storage

![Write throughput through the WAL by transaction size, with and without fsync at commit](../assets/img/chart-strata-wal.png)

- **Commits.** A durable single-document commit is one WAL append and one `fsync`: 4.1 ms at the median, 227 commits a second. On macOS the sync is `F_FULLFSYNC`, which makes the SSD flush its write cache; the earlier VM's `fdatasync` on a virtual disk did not have to, and took 0.27 ms. With 1,000 documents per transaction the WAL takes 98,000 documents a second; without `fsync`, 155,000–438,000 (a process crash still loses nothing, a power failure may lose recent commits).
- **Read your writes.** Committed documents are searchable at once. The first query after a write rebuilds the memtable's last 1,024-document chunk with every index of the evaluation table, which takes 17–20 ms; an unchunked memtable of 10,000 documents took 169 ms.
- **Open.** Cold, opening takes 6.8 ms for 10 segments, 40 ms for 100 and 279 ms for 994 (6 ms warm). Replaying 10,000 WAL documents takes 34 ms at open, plus 207 ms to index them at the first query.
- **Merges.** A tiered merge of eight 5,000-document segments, 10% deleted, rebuilds every index in 0.75 s, about 48,000 documents a second. After a delete-heavy workload the file is 1.22× its compacted size until `vacuum`.
- **Recovery.** A crash test kills a writer with `SIGKILL` at a random moment during inserts, deletes, commits and flushes, then reopens. Every restart recovered a committed prefix: 20 runs by default, and 60 and 80 in longer runs. A process killed while it creates databases, one after another, leaves each path a whole database or nothing. A damaged meta slot falls back to the other one, and so does a damaged newest version, after every commit of a workload of flushes, merges and a vacuum (`tests/storage_recovery.rs`). A simulated power loss at each of a workload's several hundred writes, truncations and syncs, with unsynced data kept or lost sector by sector, reopens to every acknowledged transaction with `sync`, and to the state after some transaction since the last flush without it; a full disk or a failing sync at each call leaves what was committed (`tests/storage_faults.rs`).

## Limitations

- **One machine.** A MacBook Air with an Apple M2, timed with the NEON kernels. Cold numbers come from evicting the page cache with `msync(MS_INVALIDATE)` on the Mac's local NVMe drive (APFS, 16 KiB pages); a spinning disk, a network volume or 4 KiB pages would shift them, and durable commits depend on what the drive's cache flush costs.
- **Synthetic structured fields.** `year`, `category` and `rating` are random and so independent of the text, which the planner assumes; `topic` correlates with vector queries by construction.
- **Baselines.** SQLite has no vector index here, so its vector numbers measure a scan, not a competing ANN index, and Tantivy has no vectors. A fuller comparison would add sqlite-vec and a dedicated vector store.
- **One segment.** All three systems are measured after a full merge. Queries over many segments are supported and tested, but not timed here.
- **Single-threaded queries**, by design for the comparison.

## Next steps

- **I/O planning:** let the planner choose page-cache hints per query, and keep read-ahead on the docID-to-key half of the key index, which would close the cold filter gap.
- **Correlation-aware estimates:** sample multi-predicate conjunctions the way the vector planner samples near the query.
- **Fusion without tuning:** fit the weighted-fusion weight per collection from score distributions, or learn it from clicks.
- **Incremental HNSW merges**, so that merges stop rebuilding the graph.

The long-form report, with every table and an interactive chart for each collection, is [Strata: the database and its evaluation](../reports/strata.html). [Reproducing the results](../reproduce.md#strata) has the commands.
