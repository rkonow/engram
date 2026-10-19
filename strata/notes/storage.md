# Storage engine (DESIGN.md §3)

`strata/src/storage/`: one `.strata` file plus a `-wal` sidecar, a single
writer, any number of snapshot readers. Everything else in Strata sees the
database through `Snapshot::segments()`: segment files viewed in place from
the mapping, each with a live-documents filter.

| file | contents |
|---|---|
| `pager.rs` | page/extent types, meta slots A/B, the read-only `mmap` |
| `freelist.rs` | free page runs (best-fit), deferred frees, encode/decode |
| `manifest.rs` | one on-disk version as JSON: schema, segments, delete sets, ids |
| `wal.rs` | the `-wal` sidecar: checksummed records, replay, torn tails |
| `live.rs` | delete sets (`LiveDocs`, a `DocFilter`) |
| `pk.rs` | the `pk` section: builder and zero-copy reader |
| `merge.rs` | the tiered merge policy |
| `state.rs` | in-memory state: open segments, memtable, pending deletes |
| `snapshot.rs` | `Snapshot`, `SegmentView`, snapshot pins, memtable segments |
| `db.rs` | `Database`, `WriteTxn`, the shadow-paging commit, flush, bulk load, merge, vacuum |

No dependencies were added (memmap2, crc32fast, libc, serde_json, rand
were already in `Cargo.toml`). Nothing outside the scope was changed.

## Public API

```rust
// strata::storage
pub struct DbOptions {                       // Default
    pub build: BuildOptions,                 // default
    pub memtable_max_docs: usize,            // 10_000 (documents + pending deletes)
    pub memtable_chunk_docs: usize,          // 1024 (rounded up to a multiple of 64)
    pub segment_max_docs: usize,             // 100_000 (flush and bulk-load segments)
    pub sync: bool,                          // true
    pub doc_source: Option<Arc<dyn DocumentSource>>, // None (needed by merge)
}
pub trait DocumentSource: Send + Sync {
    fn documents(&self, schema: &Schema, seg: &SegmentFile, docs: &[DocId]) -> Result<Vec<Document>>;
}

impl Database {                              // Send + Sync
    pub fn create(path: &Path, schema: Schema, opts: DbOptions) -> Result<Database>; // fails if the file exists
    pub fn open(path: &Path, opts: DbOptions) -> Result<Database>;  // recovery + WAL replay
    pub fn schema(&self) -> &Schema;
    pub fn path(&self) -> &Path;
    pub fn options(&self) -> &DbOptions;
    pub fn begin(&self) -> Result<WriteTxn<'_>>;   // Error::Busy while another write runs
    pub fn snapshot(&self) -> Snapshot;            // never waits for the writer
    pub fn flush(&self) -> Result<()>;
    pub fn bulk_load(&self, docs: impl IntoIterator<Item = Document>) -> Result<()>;
    pub fn merge(&self, policy: &MergePolicy) -> Result<MergeStats>;
    pub fn vacuum(&self) -> Result<u64>;            // bytes returned to the file system
    pub fn file_stats(&self) -> FileStats;
    pub fn check(&self) -> Result<()>;              // fsck: checksums, page accounting
}
impl WriteTxn<'_> {
    pub fn insert(&mut self, doc: Document) -> Result<()>;   // Error::Schema on a live key
    pub fn upsert(&mut self, doc: Document) -> Result<()>;
    pub fn delete(&mut self, key: &Value) -> Result<bool>;   // false if no live document
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn commit(self) -> Result<()>;
    pub fn rollback(self);                                    // dropping also rolls back
}
impl Snapshot {                              // Clone + Send + Sync, Arc inside
    pub fn txn(&self) -> u64;
    pub fn schema(&self) -> &Schema;
    pub fn segments(&self) -> &[SegmentView];     // on-disk (oldest first), then in-memory
    pub fn n_live_docs(&self) -> u64;
    pub fn pk(&self, seg: usize) -> Option<&PkIndex>;
    pub fn lookup(&self, key: &Value) -> Option<(usize, DocId)>;
    pub fn key(&self, seg: usize, doc: DocId) -> Option<Value>;
    pub fn evict_from_cache(&self) -> Result<()>;
}
pub struct SegmentView { pub id: u64, pub file: SegmentFile, pub live: Option<Arc<dyn DocFilter>>,
                         pub n_live: u32, pub in_memory: bool }

pub fn build_pk_index(schema: &Schema, docs: &[Document], out: &mut SegmentWriter) -> Result<()>;
impl PkIndex {
    pub fn open(seg: &SegmentFile) -> Result<Option<PkIndex>>;
    pub fn from_section(bytes: OwnedBytes) -> Result<PkIndex>;
    pub fn len(&self) -> usize;
    pub fn lookup(&self, key: &Value) -> Option<DocId>;
    pub fn lookup_all<'a>(&'a self, key: &'a Value) -> impl Iterator<Item = DocId> + 'a;
    pub fn key(&self, doc: DocId) -> Value;
    pub fn key_ref(&self, doc: DocId) -> KeyRef<'_>;
}
pub fn key_hash(k: KeyRef) -> u64;           // part of the file format

impl LiveDocs {                              // DocFilter: contains = not deleted
    pub fn from_deleted(words: Vec<u64>, n_docs: u32) -> LiveDocs;
    pub fn with_deleted(base: Option<&LiveDocs>, n_docs: u32, more: impl IntoIterator<Item = DocId>) -> LiveDocs;
    pub fn n_docs/n_live/n_deleted(&self) -> u32;
    pub fn is_deleted(&self, d: DocId) -> bool;
    pub fn deleted_words(&self) -> &[u64];
    pub fn to_docset(&self) -> DocSet;
}

pub struct MergePolicy { pub merge_factor: usize /* 8 */, pub floor_docs: usize /* 1000 */,
                         pub max_merged_docs: usize /* 2M */, pub max_deleted_ratio: f64 /* 0.3 */ }
impl MergePolicy { pub fn select(&self, segs: &[SegmentInfo]) -> Option<Vec<u64>>; }
pub struct MergeStats { merges, segments_in, docs_merged, docs_dropped, bytes_written, seconds }
pub struct FileStats { file_bytes, page_count, live_pages, free_pages, pending_pages, wal_bytes,
                       memtable_docs, pending_deletes, segments: Vec<SegmentStats> }
```

`pager`, `freelist`, `manifest`, `wal`, `live`, `pk` and `merge` are public
modules (formats and building blocks for tests and tools).

Contracts the other components rely on:

* A segment id names immutable segment data. A flush that changes an on-disk
  segment's delete set keeps its id (only `live` changes). In-memory segments
  have the top bit set and get a fresh id every time they are rebuilt, so an
  id can key a per-segment cache.
* The memtable appears as one in-memory segment per chunk of
  `memtable_chunk_docs` documents, after the on-disk segments. They are built
  with `build_segment`, so every index family works on them unchanged.
* `DocumentSource::documents` must return the documents in the order asked
  for. The integration layer implements it over the doc store and vector
  sections; the tests and the benchmark use a map from primary key to
  document.
* Tables without a primary key can insert and bulk load but not delete or
  upsert (`Error::Schema`).

## File format

```text
page 0, 1   meta slots A/B (84 bytes + crc32, rest zero): magic "STRATADB", format 1,
            page size, db_id, txn, manifest extent+crc, freelist extent+crc, page count
page 2..    extents: segment images, delete sets, manifests, freelists
```

* Pages are 4096 bytes; an extent is `(first page, length in bytes)`.
  Segments start on a page, so `SegmentFile::open` over the mapping gives
  page-aligned sections.
* Manifest: JSON (`schema`, `txn`, `wal_txn`, `next_segment_id`, and per
  segment `id`, extent, crc32 of the image, `n_docs`, `n_deleted`, delete-set
  extent and crc, the txn that wrote it).
* Freelist: `n_runs, n_pending, (start, len)*, (freed_at, page, pages)*` as
  u64, zero-padded to the extent size reserved before allocating it.
* Delete set: `n_docs.div_ceil(64)` little-endian words, viewed in place.
* `pk` section: `kind | n | dir_bits | dir: [u32; 2^bits + 1] |
  entries: [(hash, docID) as u64; n], sorted | KEYWORD: offsets [u64; n+1],
  bytes | INT: keys [i64; n]` (ByteWriter arrays, 8-byte aligned).
* WAL: `"STRWAL01" | db_id` then records `len u32 | crc32 u32 | kind u8 |
  payload`: insert (document), delete (key), commit (txn u64, n_ops u32).
  Values: a tag byte and an LE payload (strings and vectors with a u32
  length; vectors as f32). A FiQA document with its 1024-d vector takes
  5.7 KB.

## How it works

**Reads.** One `MAP_SHARED`, read-only mapping of the whole file. When a
commit grows the file past the mapping, a new mapping is made for later
snapshots; older ones stay alive while any `OwnedBytes` slice of them does.
Writes are `pwrite`s and never touch the mapping. Every segment is opened at
`open` (its directory and `pk` header are parsed), so `snapshot()` cannot
fail.

**Commit (shadow paging).** Flush, bulk load, merge and vacuum each produce
a new version via `VersionWriter`, which works on copies of the manifest and
freelist:

1. Write the new segments and delete sets into free pages, or at the end of
   the file.
2. Write the new manifest and freelist the same way, then `fdatasync`.
3. Write the meta slot that does not hold the current version, then
   `fdatasync`.

On open, the valid slot with the larger txn wins, and a file longer than its
page count is cut back (that tail is an unfinished commit's). If anything
fails after the meta write starts, the database is *poisoned*: later writes
fail until it is reopened.

**Page reuse.** A commit's frees are pending, tagged with its txn. They
become free when two conditions hold:

* no live snapshot has a smaller txn (snapshots register their txn with a
  reference count, `Pins`), and
* a newer commit has been written, so the other meta slot no longer names
  the version that used them.

The second rule is stricter than LMDB's. It keeps *both* meta slots
pointing at intact versions, so a damaged newest slot falls back to a whole
previous version. A test damages the newest slot after a merge that freed
everything the previous version used, then checks that the previous
version is intact. Free runs are coalesced and allocated best-fit; free
runs at the end of the file are truncated at commit.

**WAL and memtable.** A transaction buffers its operations. Each
`delete`/`insert` resolves its key against the current state plus the
transaction's own changes, so `delete` can return whether a live document
existed and `insert` can reject a live key. `commit` writes all records
with one `pwrite`, runs one `fdatasync`, then applies them to the memtable
under the state lock. A delete of an on-disk document becomes a *pending
delete*: an overlay on that segment's delete set, visible to the next
snapshot and written at flush. Replay applies the same operations in the
same order. Transactions with `txn <= manifest.wal_txn` are skipped,
because a flush may commit and then crash before it truncates the WAL. A
torn tail is cut off on open, so later appends stay reachable. The memtable
flushes automatically when documents plus pending deletes exceed
`memtable_max_docs`. `bulk_load` flushes the memtable first: segments
written behind the WAL's back must not contain keys the WAL might later
re-resolve.

**Snapshots.** A snapshot is an `Arc` holding the version's segment views,
live filters, `pk` readers, mappings and its pin. It is cached until the
next change. The state lock is held only to copy pointers. The memtable is
kept in chunks of `memtable_chunk_docs` documents, and each chunk with live
documents becomes one in-memory segment, built outside the state lock. A
full chunk's segment is built once (until the flush). After a write, only
the last chunk's segment is rebuilt, from the chunk in place (no document
copies), into a page-aligned buffer. Live filters merge the stored delete
set with pending deletes and are cached per segment.

**Published snapshots.** By default the first `snapshot()` after a write
builds the last chunk's segment, and concurrent readers queue on
`build_lock` behind it. With `DbOptions::publish_snapshots` the writer
builds the new version's snapshot before a commit, flush, bulk load, merge
or vacuum returns, and stores it in `published`. `snapshot()` then only
clones that `Arc`: a reader never builds a segment or waits for one, and a
write returns once readers can see it. The write pays for the build instead
of the next query. Engram turns this on.

**Merges.** `merge(policy)` loops: take an on-disk snapshot, let the policy
pick a group, and fetch the group's live documents through `doc_source`.
It builds the merged segment with neither the writer lock nor the state
lock held. Only then does it take the writer lock to commit. Documents of
the inputs deleted in the meantime (pending, or already flushed into new
delete sets) become the merged segment's delete set. The inputs' pending
deletes are dropped, since their documents are either gone or now in that
delete set. A test does deletes, a flush and upserts from inside
`DocumentSource::documents`, which runs between the merge's snapshot and
its commit. The WAL is untouched. Replaying a delete whose document was
merged away or is already deleted finds no live document and does nothing,
so replay stays exact.

**Tiered policy.** Tier `k` holds segments of `floor·f^k ..floor·f^(k+1)`
live documents. When a tier has `f` segments, its `f` smallest are merged;
a segment with more than `max_deleted_ratio` deleted is rewritten alone;
segments over `max_merged_docs / 2` are left alone.

**Vacuum.** `vacuum()` copies segments and delete sets, highest page first,
into free runs that lie wholly below them (`FreeList::alloc_below`), then
commits. It repeats until nothing moves and the file stops shrinking (frees
become reusable two commits later), and the trailing free run is truncated
at each commit. Pages that a held snapshot can reach are pending, so they
are neither reused nor moved.

**Locks.** `writer: Mutex<Wal>` is the single-writer lock, held by a
`WriteTxn`, flush, bulk load, vacuum and a merge's commit. `begin`, `flush`,
`bulk_load`, `vacuum` and `check` use `try_lock` and return `Error::Busy`.
A merge's commit waits for the lock, so do not call `merge` from a thread
that holds a `WriteTxn`. `state` is held only for pointer copies and
in-memory updates. `build_lock` makes concurrent readers build a memtable
segment once. `published` is an `RwLock` written only under the writer
lock (and at open), so versions are published in commit order; readers
hold its read side for one `Arc` clone. An exclusive `flock` on the file admits one open `Database`
per file across processes; the kernel drops it if the process dies.

## Tests

`cargo test --release`: 13 storage unit tests, plus 19 integration tests in
`storage_basic.rs` and 3 in `storage_crash.rs`, all passing.

* Unit tests:
  * pk index against random string and int keys; forced hash collisions and
    duplicate keys; directory spread over 65k keys.
  * Freelist against a page bitmap (20k random operations); best-fit, split,
    coalesce, trim; `alloc_below`; release rules; round trip.
  * WAL cut at every byte, with and without garbage appended; foreign or
    missing log; append, reopen and append again.
  * Meta slot torn at several bytes; delete-set view; tiered selection.
* `tests/storage_basic.rs`:
  * commit, reopen (WAL replay), flush, reopen;
  * WAL torn tail;
  * deletes and upserts visible before and after flush and reopen, including
    re-inserting a deleted key and duplicate versions in a memtable segment;
  * memtable chunks built once;
  * snapshot isolation: a held snapshot's segment bytes are unchanged
    through 6 rounds of deletes, flushes, merges and bulk loads, and its
    pages are released after it is dropped;
  * freed pages reused: 25 full-table rewrites keep the file bounded;
  * merge with a mock `DocumentSource`, and merge folding in writes made
    during it;
  * vacuum: shrinks the file and spares a held snapshot;
  * damaged newest meta slot falls back to the previous version, and both
    slots damaged gives `Corrupt`;
  * `Busy` for a second writer and a second `open`;
  * bulk-load segmenting and duplicate keys; a table without a primary key;
  * two reader threads checking every snapshot (consistent key sets,
    monotonic) while one thread commits with automatic flushes and another
    merges, with and without published snapshots;
  * published snapshots: each commit, flush, bulk load, merge, vacuum and
    reopen publishes its version; readers get the same one until the next
    write, see the last commit while a transaction is open, and a full
    chunk's segment is not rebuilt;
  * random operations (transactions, rollbacks, flushes, merges, vacuums,
    bulk loads, reopens) against a model, 6 seeds × 400 steps, and 3 more
    seeds with published snapshots.

  Most tests end with `check()`. That verifies every segment and delete-set
  checksum, that every key is found by its segment's index, and that every
  page is used exactly once.
* `tests/storage_crash.rs`: the test binary re-executes itself as a child
  that runs random transactions, flushes, merges, vacuums and bulk loads
  (random options, fsync on for even seeds, off for odd ones). The child is
  SIGKILLed 40–400 ms in. It logs `B n ops` before each commit and `C n`
  after it returns, each line in one `write`. The parent reopens the
  database and requires:
  * the live keys equal the state after some transaction between the last
    acknowledged one and the last attempted one;
  * no key is live twice;
  * `check()` passes.

  Each iteration continues from the previous iteration's recovered state.
  The test runs 20 iterations (`STRATA_CRASH_ITERS`); 60 and 80 iterations
  were also run, all passing.

Limitation: with the component builders still stubs, segments contain only
the `pk` section. So tests compare live key sets and counts, not document
contents.

## Measurements

`cargo run --release --bin strata-storage-bench -- --dataset fiqa --reps 5`
writes `results/storage/{wal,bulk,open,flush,merge,space,snapshot,pk}.json`.
Options: `--exp`, `--limit`, `--reps`, `--dir` (scratch database files,
default `target/storage-bench`, must be on the disk under test),
`--wal-secs`, `--wal-docs`.

Medians of 5 repetitions, FiQA (57,638 documents, 1024-d vectors), ext4 on
the VM's virtual disk. Other index builders were stubs, so segments hold
only the `pk` index (30 bytes per document). **The machine was shared: the
1-minute load average was 8.3–10 on 4 cores throughout** (recorded as
`loadavg_at_start` in each file). fsync-bound numbers vary by 10× between
quiet and busy moments: an isolated SciFact run gave 192 µs per 1-document
commit, against 271 µs here and ~4 ms in an earlier run. Treat these as
relative numbers; they are to be re-measured in isolation.

**WAL commits** (`wal.json`; the memtable never flushes during the run):

| txn docs | fsync: docs/s | commits/s | p50 / p99 µs | no fsync: docs/s | commits/s | p50 µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,844 | 2,844 | 271 / 2,101 | 154,276 | 154,276 | 5 |
| 10 | 27,312 | 2,731 | 319 / 1,091 | 239,961 | 23,996 | 37 |
| 100 | 31,835 | 318 | 3,881 / 7,314 | 274,270 | 2,743 | 349 |
| 1000 | 81,871 | 82 | 12,678 / 15,690 | 251,759 | 252 | 3,804 |

Each commit is one `pwrite` and one `fdatasync`. With fsync off, the cost
is encoding about 5.7 KB per document (mostly the vector) plus the memtable
update, about 4 µs per document.

**Bulk load** (`bulk.json`): 865k docs/s with 10k-document segments (6
segments, 0.067 s) and 769k docs/s as one segment. `build_segment` takes
0.012 s of that; the rest is validation, the duplicate-key set, the pk
build, writes and the commit's two fsyncs.

**Open** (`open.json`; cold = `posix_fadvise(DONTNEED)` on both files
before opening):

| segments | cold open | warm open | first snapshot |
|---:|---:|---:|---:|
| 10 | 2.2 ms | 0.06 ms | 0.006 ms |
| 100 | 1.7 ms | 0.13 ms | 0.02 ms |
| 994 | 6.1 ms | 1.06 ms | 0.17 ms |

Cold WAL replay: 4.7 ms for 1,000 documents (5.7 MB of WAL) and 83 ms for
10,000 (57 MB), about 120k docs/s.

**Flush** (`flush.json`): 27 ms for 1,000 memtable documents and 61 ms for
10,000. Three fsyncs: data, meta slot, WAL truncation.

**Merge** (`merge.json`): 8 segments × 5,000 documents with 10% deleted, in
80 ms (564k docs/s). This includes fetching the documents from the source,
the build and the commit.

**Space** (`space.json`): load FiQA as 6 segments, then 10 rounds that each
delete 5% of the documents (transactions of 100, then a flush). After
rounds 5 and 10: `merge(default)`, then `vacuum()`. Space amplification is
the file size divided by live data, where live data = segment bytes × the
live fraction.

| step | live docs | file KiB | free / pending pages | amp |
|---|---:|---:|---:|---:|
| load | 57,638 | 1,784 | 0 / 2 | 1.02 |
| round 5, deletes | 43,233 | 1,864 | 0 / 16 | 1.42 |
| round 5, merge (nothing over 30% deleted) | 43,233 | 1,864 | 0 / 16 | 1.42 |
| round 5, vacuum | 43,233 | 1,816 | 0 / 4 | 1.39 |
| round 10, deletes | 28,828 | 1,864 | 0 / 16 | 2.13 |
| round 10, merge (6 rewrites) | 28,828 | 2,136 | 149 / 157 | 2.41 |
| round 10, vacuum | 28,828 | 936 | 2 / 4 | 1.06 |

The same 28,828 documents bulk-loaded into a fresh file take 904 KiB, so
the file after vacuum is 1.04× a freshly loaded one. Before merging, the
amplification is the deleted documents still in segments, as in any LSM;
the merge policy bounds it through `max_deleted_ratio`. Right after a merge
the file is at its largest: the new segments were written while the old
ones were still live, and the old ones' pages become reusable only two
commits later.

## Design decisions and the measurements behind them

* **Radix directory for the pk index**, not binary search (`pk.json`, 58k
  keys, 100k lookups):

  | | hit ns | miss ns | pages per hit | pages per miss |
  |---|---:|---:|---:|---:|
  | radix directory (as built) | 87 | 49 | 4.02 | 2.01 |
  | binary search, same entries | 128 | 123 | 9.77 | 7.76 |

  "Pages" counts the distinct 4 KiB pages a lookup touches: the cold-cache
  I/O. Each bucket holds about 4 entries. (hash, docID) pairs are
  interleaved so that a probe reads one cache line. The directory costs
  0.5–1 byte per document.
* **Chunked memtable** instead of one memtable segment (`snapshot.json`,
  first snapshot after a 1-document commit, stub builders):

  | memtable docs | chunk 1024 | chunk 4096 | one segment |
  |---:|---:|---:|---:|
  | 1,000 | 0.08 ms | | 0.08 ms |
  | 10,000 | 0.07 ms (10 segments) | 0.20 ms (3) | 1.40 ms |

  The rebuild cost is proportional to the documents re-indexed. With real
  text and vector builders the ratio holds and the absolute gap grows to
  tens of milliseconds. A first version that cloned all memtable documents
  into one segment took 34 ms at 10k documents (an earlier SciFact run, on
  a busy machine). The cost is up to
  `memtable_max_docs / chunk` extra small segments per query (10 with the
  defaults). Cached snapshots cost 0.1 µs.
* **Plain appending WAL**, not a pre-zeroed or `fallocate`d one. A probe of
  4 KiB writes, each followed by `fdatasync`, over 3 runs on this ext4,
  median latency per write:

  | file | run 1 | run 2 | run 3 |
  |---|---:|---:|---:|
  | appending | 157 µs | 189 µs | 129 µs |
  | pre-zeroed | 163 µs | 190 µs | 136 µs |
  | fallocated | 172 µs | 115 µs | 118 µs |

  No consistent gain, so no preallocation and no zero-tail handling.
* **Best-fit allocation** kept: first-fit by address gave identical file
  sizes in the space experiment. The lever for space is moving data, hence
  `vacuum` (2,136 → 936 KiB).
* **Two-commit reuse delay.** It keeps one more commit's frees pending: the
  4–16 pending pages in the space table. It buys a whole previous version
  behind a damaged newest meta slot (tested).
* **Eager segment open.** Opening every segment at `open` costs 1 ms warm
  and 6 ms cold for ~1,000 segments, and makes `snapshot()` infallible.
  Tiered merging keeps segment counts far lower.

## Open issues

* **Writers wait on maintenance.** A merge builds without locks, but writes
  and fsyncs the merged segment under the writer lock. Flush and bulk load
  build their segments under it. Meanwhile `begin()` returns `Busy`, as the
  API requires; a caller wanting to wait must retry. Fix: write merged
  extents into reserved pages before taking the lock. That needs the
  persisted freelist to treat reservations as free.
* **Merge memory.** All input documents of a merge are in memory at once
  (bounded by `max_merged_docs`). With 1024-d vectors that is about 4 KB
  per document plus text. A streaming `build_segment` would remove it.
* **Space** is returned only by `vacuum` (and tail truncation). Running
  vacuum after a merge would be easy (auto-vacuum).
* **Long-held snapshots** pin every page freed after them, as in LMDB, so
  the file grows while a reader holds an old snapshot.
* **Integrity checks.** Segment and delete-set checksums are verified only
  by `check()`, never on the read path. Meta slots, manifest and freelist
  are verified at every open.
* **`SegmentView::live` is a `dyn DocFilter`.** The query layer cannot get
  at the bitset for word-level set operations. `LiveDocs::to_docset` exists;
  exposing the concrete `LiveDocs` (or adding a bitset accessor to
  `DocFilter`) would help.
* **Measurement caveats.** All numbers were measured with stub index
  builders and on a shared machine. Bulk load, flush, merge and snapshot
  costs will be dominated by the real builders after integration. The bench
  runs unchanged and records `build_segment` time separately for bulk loads.
* **Single process.** `flock` admits one `Database` per file. There is no
  shared-memory reader table, so other processes cannot read concurrently.
