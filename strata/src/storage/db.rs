//! The database: one file plus a `-wal` sidecar (or a directory of files,
//! [`Layout::Directory`]), one writer, any number of snapshot readers.
//!
//! A database holds one or more tables ([`Database::create_table`]), each
//! with its own schema, segments and memtable. They share the WAL, the
//! writer lock and the commit: one transaction can write several tables,
//! and a snapshot sees every table at the same version. The table a
//! database was created with is its first; the methods that take no table
//! name ([`WriteTxn::insert`], [`Database::bulk_load`], ...) act on it.
//!
//! * Write transactions append to the WAL and `fsync` at commit; their
//!   documents and deletes land in the memtable, visible to the next
//!   snapshot. Deletes of on-disk documents stay *pending* (an overlay on
//!   the segment's delete set) until a flush.
//! * A flush, a bulk load and a merge each write a new on-disk version with
//!   the shadow-paging commit ([`VersionWriter`]): new extents into free
//!   pages, `fsync`, then the other meta slot, `fsync`.
//! * Merges run on segments no other merge has claimed, several at once.
//!   A merge writes its image before it takes the writer lock, into pages
//!   (or a file) reserved for it, and holds the lock only to install it.
//! * Readers never take the writer lock; the state lock is held only to
//!   copy pointers, never across I/O or index builds.
//! * With [`DbOptions::publish_snapshots`] the writer also builds each new
//!   version's snapshot before its commit returns, and readers only clone
//!   the published one: they never build a memtable segment or wait for
//!   one to be built.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock, TryLockError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::error::{corrupt, Error, Result};
use crate::schema::{FieldType, Schema};
use crate::sid::{Codebook, Ties};
use crate::segment::build::{build_segment, build_threads, BuildOptions, GraphStart, SegmentBuild};
use crate::segment::{DocId, SegmentFile, SegmentWriter};
use crate::storage::freelist::FreeList;
use crate::storage::io;
use crate::storage::dir::{self, Current};
use crate::storage::live::LiveDocs;
use crate::storage::lock;
use crate::storage::manifest::{sources_schema, Manifest, Numbering, RankProfile, SegmentMeta, SeqNo, TableMeta, SOURCES, SOURCES_ID};
use crate::storage::merge::{memory_limit, merge_memory, MergeEntry, MERGE_BATCH_BYTES, MERGE_BATCH_DOCS, MergePolicy, MergeRecord, MergeStats, MergeStatus, MergerOptions, Pacer, SegmentInfo};
use crate::storage::pager::{self, Extent, Meta, META_PAGES, PAGE};
use crate::postings::bytes::OwnedBytes;
use crate::storage::snapshot::{build_memseg, Pins, SegmentView, Snapshot};
use crate::storage::rowseq::{self, Seqs};
use crate::storage::state::{mem_seq, Check, Key, Loc, SegHandle, State, TableState};
use crate::storage::updates::{Updates, SEQ};
use crate::storage::ship::{Applied, LogRead, ShipLog, Shipped};
use crate::storage::wal::{self, Op, Wal, WalTxn};
use crate::value::{conform, validate, Document, Value};

/// Ids of in-memory segments have this bit set.
const MEM_ID: u64 = 1 << 63;

/// Rebuilds full documents from a segment, for merges. Provided by the
/// layer that owns the doc store and vector sections.
pub trait DocumentSource: Send + Sync {
    /// The documents `docs` (increasing docIDs) of `seg`, in that order.
    fn documents(&self, schema: &Schema, seg: &SegmentFile, docs: &[DocId]) -> Result<Vec<Document>>;
}

/// Options of an open database (not stored in the file).
#[derive(Clone)]
pub struct DbOptions {
    /// Options of every segment build (memtable, flush, bulk load, merge).
    pub build: BuildOptions,
    /// Options of memtable segment builds instead of `build`. A memtable
    /// chunk is rebuilt by the first snapshot after each write, so an index
    /// that is slow to build but fast to search (HNSW) costs more there
    /// than a scan of a few hundred vectors (`vector.index = flat`).
    pub mem_build: Option<BuildOptions>,
    /// Flush when the memtable's documents plus pending deletes exceed this.
    pub memtable_max_docs: usize,
    /// Documents per memtable chunk (rounded up to a multiple of 64). Each
    /// chunk is one in-memory segment of a snapshot; the first snapshot
    /// after a write rebuilds only the last chunk's.
    pub memtable_chunk_docs: usize,
    /// Largest segment written by flushes and bulk loads.
    pub segment_max_docs: usize,
    /// `fsync` at every commit. Without it a process crash still loses
    /// nothing, but a power failure may lose the transactions committed
    /// since the last flush (a bulk load flushes first), newest first.
    /// Flushes, merges, bulk loads and vacuums sync either way: the WAL's
    /// unsynced appends, then the pages of the version they write, then the
    /// meta slot that names it.
    pub sync: bool,
    /// Needed by [`Database::merge`].
    pub doc_source: Option<Arc<dyn DocumentSource>>,
    /// Kernel read-ahead on the file mapping. With it off (`MADV_RANDOM`) a
    /// page fault reads one page instead of up to 32, which suits random
    /// access (dictionary probes, graph hops, doc store fetches) when the
    /// data is not cached.
    pub read_ahead: bool,
    /// With read-ahead on, advise the sections that are read at random
    /// positions (graph, float32 vectors, norms, dictionaries, doc store,
    /// keys) `MADV_RANDOM`, keeping read-ahead for scanned sections.
    pub random_hints: bool,
    /// Build each version's snapshot when it is made, in the writer (a
    /// commit, flush, bulk load, merge or vacuum returns once readers can
    /// see it), and give readers that one. A reader then never builds a
    /// memtable segment or waits for another reader's build; each write
    /// pays for the build instead of the first query after it.
    pub publish_snapshots: bool,
    /// How a database created with these options is laid out on disk (an
    /// existing one opens in the layout it has).
    pub layout: Layout,
    /// Open a database whose WAL was damaged after it was made durable
    /// (more than one commit follows the damage) by dropping the
    /// transactions from the damage on; the log is copied to
    /// `<wal>.damaged` first. Without it, opening such a database with
    /// `sync` fails. ([`Database::damaged`] says what was dropped.)
    pub wal_drop_damaged: bool,
    /// Check every segment against its checksum when the database opens,
    /// which reads every byte of it; a version whose segment fails is
    /// damaged and opening falls back to the one before it. Without it a
    /// segment is trusted as far as its structure checks go (a text list,
    /// for one, is checked in full when first used) until
    /// [`Database::check`] runs. Delete sets and updates are checked
    /// either way, and merges check the segments they read.
    pub verify_checksums: bool,
    /// Check each section of an on-disk segment against its crc32 the
    /// first time a query reads it, which reads that whole section once
    /// (and a segment's keys when it opens); a query that needs a damaged
    /// section fails. Segments written before sections had a crc32 are
    /// left to the checks above. Off, sections are trusted as far as
    /// their structure checks go.
    pub check_sections: bool,
    /// Bytes of recent numbered transactions to keep in memory, as they
    /// were logged, for replicas to read ([`Database::log_from`]); 0 keeps
    /// none. A replica further behind than this needs a copy of the
    /// segments first.
    pub ship_log_bytes: usize,
    /// Open a database written before field ids
    /// ([`crate::storage::manifest::FIELD_IDS`]), which opening refuses
    /// otherwise: to copy it into a new database ([`crate::upgrade`]). It
    /// stays in its format: what is written to it meanwhile is written as
    /// the versions of Strata before field ids wrote it.
    pub open_old: bool,
}

/// How a database is laid out on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum Layout {
    /// One file of pages plus a `-wal` sidecar: extents, two meta slots and
    /// the shadow-paging commit.
    #[default]
    SingleFile,
    /// A directory with one file per segment and delete set, and versioned
    /// manifests named by `CURRENT` ([`crate::storage::dir`]).
    Directory,
}

impl DbOptions {
    /// How the segments of the version a database opens at are checked.
    fn check_at_open(&self) -> Check {
        match (self.verify_checksums, self.check_sections) {
            (true, _) => Check::Open,
            (false, true) => Check::Read,
            (false, false) => Check::Never,
        }
    }
}

impl Default for DbOptions {
    fn default() -> Self {
        DbOptions {
            build: BuildOptions::default(),
            mem_build: None,
            memtable_max_docs: 10_000,
            memtable_chunk_docs: 1024,
            segment_max_docs: 100_000,
            sync: true,
            doc_source: None,
            read_ahead: true,
            random_hints: true,
            publish_snapshots: false,
            layout: Layout::SingleFile,
            wal_drop_damaged: false,
            verify_checksums: false,
            check_sections: true,
            ship_log_bytes: 0,
            open_old: false,
        }
    }
}

/// How [`Database::checkpoint`] writes its copy.
#[derive(Clone, Debug, Default)]
pub struct CheckpointOptions {
    /// The copy's layout, whatever the database's (default: one file).
    pub layout: Layout,
    /// Directory layout: an earlier checkpoint of the same database, whose
    /// segment files the copy hard-links instead of writing again when the
    /// segment is unchanged (same id and checksum). Segments are immutable
    /// and keep their ids in a checkpoint, so successive checkpoints share
    /// every segment no merge has replaced since.
    pub link_from: Option<PathBuf>,
}

/// What [`Database::checkpoint`] copied.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CheckpointStats {
    /// The commit of the database that the copy holds.
    pub txn: u64,
    pub tables: usize,
    pub segments: usize,
    /// Of `segments`, those hard-linked from `link_from`.
    pub segments_linked: usize,
    pub live_docs: u64,
    /// Bytes written (linked segments not counted).
    pub bytes_written: u64,
    pub seconds: f64,
}

/// Space use of the file and the WAL.
#[derive(Clone, Debug, Serialize)]
pub struct FileStats {
    pub file_bytes: u64,
    pub page_count: u64,
    /// Pages reachable from the current version (meta slots included).
    pub live_pages: u64,
    pub free_pages: u64,
    /// Freed pages not yet reusable (held for a snapshot or the older slot).
    pub pending_pages: u64,
    pub wal_bytes: u64,
    pub memtable_docs: usize,
    pub pending_deletes: usize,
    pub segments: Vec<SegmentStats>,
}

/// Size of one on-disk segment.
#[derive(Clone, Debug, Serialize)]
pub struct SegmentStats {
    /// The table the segment belongs to.
    pub table: String,
    pub id: u64,
    pub n_docs: u32,
    pub n_deleted: u32,
    pub bytes: u64,
    pub delete_set_bytes: u64,
}

/// An open database. `Send + Sync`: share it (e.g. in an `Arc`) between a
/// writer thread, readers and a background merger.
pub struct Database {
    path: PathBuf,
    wal_path: PathBuf,
    opts: DbOptions,
    /// The first table's schema (that table cannot be dropped).
    schema: Arc<Schema>,
    /// The database file, or the directory layout's `LOCK` file.
    file: File,
    /// The directory, in the directory layout.
    dir: Option<PathBuf>,
    /// The single-writer lock; it owns the WAL.
    writer: Mutex<Wal>,
    state: Mutex<State>,
    /// Serialises memtable segment builds, so concurrent readers build once.
    build_lock: Mutex<()>,
    /// The snapshot readers get, with `DbOptions::publish_snapshots`.
    published: RwLock<Option<Snapshot>>,
    /// The merges running, with what a crash interrupted ([`Merging`]).
    merging: Mutex<Merging>,
    /// Expired rows per segment ([`Database::expired_in`]): the segment's
    /// live count, the minute they were counted at and whether all were
    /// expired, and the count.
    expired: Mutex<HashMap<u64, ((u32, i64, bool), u32)>>,
    /// Where `merging.record` is kept.
    merge_record_path: PathBuf,
    /// Set by [`Database::spawn_merger`].
    merger: OnceLock<Arc<Merger>>,
    merger_threads: Mutex<Vec<JoinHandle<()>>>,
    pins: Arc<Pins>,
    mem_ids: AtomicU64,
    /// Set when a write failed in a way that leaves memory and disk out of
    /// step; every later write fails until the database is reopened.
    poisoned: AtomicBool,
    /// Why the last flush failed, until one succeeds.
    flush_error: Mutex<Option<String>>,
    /// What opening found damaged and skipped: the newest version on disk
    /// (the one before it was opened), commits at the end of the WAL.
    damaged: Option<String>,
    /// The txns of the sealed WAL files a background flush has not
    /// installed yet ([`wal::sealed_path`]), oldest first.
    sealed_wals: Mutex<Vec<u64>>,
    /// One background flush at a time.
    background: Mutex<()>,
    /// Set by [`Database::spawn_flusher`]: commits that fill the memtable
    /// wake the flusher instead of flushing.
    flush_signal: OnceLock<Arc<FlushSignal>>,
    flusher: Mutex<Option<JoinHandle<()>>>,
    /// How many background flushes and merges wait for the writer lock,
    /// and hold it (at most one): writes that find the lock taken wait for
    /// it ([`Database::try_writer`]).
    background_waiting: AtomicUsize,
    background_holding: AtomicUsize,
    /// Told the term and number of each numbered transaction once it is
    /// committed ([`Database::on_commit`]).
    commit_hook: RwLock<Option<CommitHook>>,
    /// The codebooks of semantic-ID fields ([`Database::set_codebook`]).
    codebooks: RwLock<Vec<Coded>>,
}

/// The codebook of semantic-ID field `field` of the table with id `table`,
/// the one whose file has hash `hash`.
struct Coded {
    table: u32,
    field: usize,
    hash: String,
    codebook: Arc<Codebook>,
}

type CommitHook = Arc<dyn Fn(SeqNo) + Send + Sync>;

/// The writer lock as a background flush or a merge holds it, briefly: to
/// seal the memtables, to reserve space for an image, or to install
/// segments. Until it is released, writes that find the lock taken wait
/// instead of failing with `Busy`.
struct BackgroundWriter<'a> {
    wal: Option<MutexGuard<'a, Wal>>,
    holding: &'a AtomicUsize,
}

impl std::ops::Deref for BackgroundWriter<'_> {
    type Target = Wal;
    fn deref(&self) -> &Wal {
        self.wal.as_ref().expect("held until dropped")
    }
}

impl std::ops::DerefMut for BackgroundWriter<'_> {
    fn deref_mut(&mut self) -> &mut Wal {
        self.wal.as_mut().expect("held until dropped")
    }
}

impl Drop for BackgroundWriter<'_> {
    fn drop(&mut self) {
        // Released first, so that a waiting write finds it free.
        self.wal = None;
        self.holding.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The merges running: each one's inputs, which no other merge may take,
/// and estimated memory, in `record.running`, with the merges a crash
/// interrupted before ([`MergeRecord`]).
#[derive(Default)]
struct Merging {
    record: MergeRecord,
    /// Counters and the last error, for [`MergeStatus`].
    merges: u64,
    limited: u64,
    error: Option<String>,
}

impl Merging {
    fn claimed(&self, id: u64) -> bool {
        self.record.running.iter().any(|m| m.segments.contains(&id))
    }

    fn memory(&self) -> u64 {
        self.record.running.iter().map(|m| m.memory).sum()
    }
}

/// A merge's claim on its input segments, which it releases when dropped.
struct MergeClaim<'a> {
    db: &'a Database,
    table: u32,
    snap: Snapshot,
    /// Oldest first, so the merged segment keeps documents in age order.
    ids: Vec<u64>,
    /// Selected larger and cut to fit its memory budget.
    cut: bool,
    /// The table's retention field and the time before which its rows
    /// expire, when the merge was claimed ([`Schema::retention`]).
    retention: Option<(usize, i64)>,
}

impl Drop for MergeClaim<'_> {
    fn drop(&mut self) {
        let live: HashSet<u64> = lock(&self.db.state).handles().map(|h| h.meta.id).collect();
        let mut m = lock(&self.db.merging);
        m.record.running.retain(|e| e.segments != self.ids);
        m.record.retain_live(|s| live.contains(&s));
        let _ = m.record.store(&self.db.merge_record_path);
    }
}

/// Space reserved for a merged image while it is written without the
/// writer lock: a segment id, and pages of the file (or the segment's file,
/// in the directory layout) that no commit hands out meanwhile. Until the
/// image is installed, the freelist on disk counts the pages as free (and
/// opening removes a file no version names), so a crash leaves nothing
/// behind.
#[derive(Clone, Debug)]
pub(crate) struct Reserved {
    pub id: u64,
    pub extent: Extent,
    pub file: Option<String>,
}

/// Images are written in pieces of this size, each paced.
const IMAGE_CHUNK: usize = 1 << 20;

/// Writes an image streamed to it at `at` in `file`, in [`IMAGE_CHUNK`]
/// pieces, and sums its checksum.
struct ImageOut<'a> {
    file: &'a File,
    at: u64,
    buf: Vec<u8>,
    pacer: Option<&'a Pacer>,
    crc: crc32fast::Hasher,
}

impl ImageOut<'_> {
    /// Writes what is buffered.
    fn drain(&mut self) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        if let Some(p) = self.pacer {
            p.pace(self.buf.len());
        }
        io::write_at(self.file, &self.buf, self.at)?;
        self.crc.update(&self.buf);
        self.at += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }
}

impl std::io::Write for ImageOut<'_> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let n = data.len().min(IMAGE_CHUNK - self.buf.len());
        self.buf.extend_from_slice(&data[..n]);
        if self.buf.len() == IMAGE_CHUNK {
            self.drain()?;
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.drain()
    }
}

/// The background merger's settings and wake-ups ([`Database::spawn_merger`]).
struct Merger {
    policy: MergePolicy,
    budget: Option<u64>,
    pacer: Option<Pacer>,
    wake: Mutex<MergerWake>,
    cv: Condvar,
    /// The table a merger thread looks at first, in turn.
    next_table: AtomicUsize,
}

#[derive(Default)]
struct MergerWake {
    /// Bumped by each version a commit publishes: segments may be due a
    /// merge. Every merger thread looks.
    generation: u64,
    closed: bool,
}

impl Merger {
    fn wake(&self) {
        lock(&self.wake).generation += 1;
        self.cv.notify_all();
    }

    fn close(&self) {
        lock(&self.wake).closed = true;
        self.cv.notify_all();
    }
}

/// A table's rows as a flush or a bulk load writes them to segments:
/// without those past `retention` ([`Schema::retention`]), and sorted by
/// its cluster_by field, rows of one value in the order written
/// ([`Schema::with_cluster_by`]). `with` holds a value for each row (its
/// sequence number), which goes along.
fn arrange<T>(schema: &Schema, mut live: Vec<Document>, mut with: Vec<T>, retention: Option<(usize, i64)>) -> (Vec<Document>, Vec<T>) {
    if live.iter().any(|d| expired(retention, d)) {
        (live, with) = live.into_iter().zip(with).filter(|(d, _)| !expired(retention, d)).unzip();
    }
    if let Some(f) = schema.cluster_by_field() {
        let keys: Vec<ClusterKey> = live.iter().map(|d| ClusterKey::of(&d[f])).collect();
        let mut by: Vec<u32> = (0..live.len() as u32).collect();
        by.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));
        let mut rows: Vec<Option<(Document, T)>> = live.into_iter().zip(with).map(Some).collect();
        (live, with) = by.into_iter().map(|i| rows[i as usize].take().expect("each row once")).unzip();
    }
    (live, with)
}

/// Whether row `d` is past `retention`, its table's ([`Schema::retention`]).
fn expired(retention: Option<(usize, i64)>, d: &Document) -> bool {
    retention.is_some_and(|(f, cutoff)| matches!(d[f], Value::Timestamp(t) | Value::Int(t) if t < cutoff))
}

/// A row's place in a segment ordered by its cluster_by field: rows
/// without a value first.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ClusterKey<'a> {
    Null,
    Int(i64),
    Str(&'a str),
}

impl<'a> ClusterKey<'a> {
    fn of(v: &'a Value) -> ClusterKey<'a> {
        match v {
            Value::Int(x) | Value::Timestamp(x) => ClusterKey::Int(*x),
            Value::Str(s) => ClusterKey::Str(s),
            _ => ClusterKey::Null,
        }
    }
}

/// The sort keys of a merge's documents (`order`, as (input, docID)) by
/// cluster_by field `f`, read from the inputs' columns: a keyword's is its
/// rank among every input's values, so that keys compare as the values do.
fn cluster_keys(schema: &Schema, f: usize, views: &[&SegmentView], order: &[(u32, DocId)]) -> Result<Vec<Option<i64>>> {
    let name = &schema.fields[f].key();
    match schema.fields[f].ty {
        FieldType::Keyword => {
            let cols = views.iter().map(|v| crate::forward::KeywordColumn::open(&v.file, name)).collect::<Result<Vec<_>>>()?;
            let mut all: Vec<&str> = cols.iter().flat_map(|c| (0..c.n_values() as u32).map(|code| c.value(code))).collect();
            all.sort_unstable();
            all.dedup();
            let ranks: Vec<Vec<i64>> = cols.iter().map(|c| (0..c.n_values() as u32).map(|code| all.partition_point(|&x| x < c.value(code)) as i64).collect()).collect();
            Ok(order.iter().map(|&(k, d)| cols[k as usize].code_of_doc(d).map(|code| ranks[k as usize][code as usize])).collect())
        }
        _ => {
            let cols = views.iter().map(|v| crate::forward::NumericColumn::open(&v.file, name)).collect::<Result<Vec<_>>>()?;
            Ok(order.iter().map(|&(k, d)| cols[k as usize].get_i64(d)).collect())
        }
    }
}

/// A merger thread: merges what the policy selects whenever a version is
/// published (new segments, deletes written) and every few seconds, while
/// other threads take other segments. A failed merge is retried after a
/// pause that doubles up to a minute.
fn merger(db: Weak<Database>, m: Arc<Merger>) {
    let mut seen = u64::MAX;
    let mut backoff = Duration::ZERO;
    loop {
        {
            let mut w = lock(&m.wake);
            let t0 = Instant::now();
            let wait = if backoff.is_zero() { Duration::from_secs(5) } else { backoff };
            while !w.closed && (!backoff.is_zero() || w.generation == seen) {
                let left = wait.saturating_sub(t0.elapsed());
                if left.is_zero() {
                    break;
                }
                w = m.cv.wait_timeout(w, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            if w.closed {
                return;
            }
            seen = w.generation;
        }
        let Some(db) = db.upgrade() else { return };
        let mut failed = false;
        while let Some(claim) = db.next_merge(&m) {
            let mut stats = MergeStats::default();
            let t0 = Instant::now();
            let r = db.merge_claimed(&claim, m.pacer.as_ref(), &mut stats);
            let segments = claim.ids.len();
            drop(claim);
            let mut st = lock(&db.merging);
            if let Err(e) = r {
                tracing::warn!(segments, error = %e, "a merge failed; merging pauses, then tries again");
                st.error = Some(e.to_string());
                failed = true;
                break;
            }
            tracing::debug!(segments, docs = stats.docs_merged, dropped = stats.docs_dropped, bytes = stats.bytes_written, secs = t0.elapsed().as_secs_f64(), "merged");
            st.error = None;
        }
        backoff = if failed { (backoff * 2).clamp(Duration::from_secs(1), Duration::from_secs(60)) } else { Duration::ZERO };
        drop(db);
    }
}

/// Wakes the flusher thread.
#[derive(Default)]
struct FlushSignal {
    state: Mutex<FlushWake>,
    cv: Condvar,
}

#[derive(Default)]
struct FlushWake {
    /// A commit filled the memtable since the flusher last looked.
    due: bool,
    /// The flusher is flushing.
    running: bool,
    /// The database is closing.
    closed: bool,
}

impl FlushSignal {
    fn wake(&self) {
        lock(&self.state).due = true;
        self.cv.notify_all();
    }

    fn set_running(&self, running: bool) {
        lock(&self.state).running = running;
        self.cv.notify_all();
    }

    fn close(&self) {
        lock(&self.state).closed = true;
        self.cv.notify_all();
    }
}

/// The flusher thread: a background flush each time a commit fills the
/// memtable, until the database closes. A flush that failed is retried
/// after a pause rather than at every commit.
fn flusher(db: Weak<Database>, sig: Arc<FlushSignal>) {
    loop {
        {
            let mut w = lock(&sig.state);
            while !w.due && !w.closed {
                w = sig.cv.wait(w).unwrap_or_else(|e| e.into_inner());
            }
            if w.closed {
                return;
            }
            w.due = false;
        }
        let Some(db) = db.upgrade() else { return };
        let mut failed = false;
        if db.flush_due() {
            sig.set_running(true);
            failed = db.background_flush().is_err();
            sig.set_running(false);
        }
        drop(db);
        if failed {
            let w = lock(&sig.state);
            let _ = sig.cv.wait_timeout_while(w, Duration::from_secs(1), |w| !w.closed);
        }
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        if let Some(sig) = self.flush_signal.get() {
            sig.close();
        }
        if let Some(m) = self.merger.get() {
            m.close();
        }
        let mut threads: Vec<JoinHandle<()>> = self.flusher.get_mut().unwrap_or_else(|e| e.into_inner()).take().into_iter().collect();
        threads.append(self.merger_threads.get_mut().unwrap_or_else(|e| e.into_inner()));
        for h in threads {
            // The flusher and the mergers hold a reference only while they
            // work; when one held the last one, the database closes on its
            // thread.
            if h.thread().id() != std::thread::current().id() {
                let _ = h.join();
            }
        }
    }
}

/// What a background flush took from a table's memtable when it sealed
/// it: its `len` documents, with the deletes and values written in place
/// up to then.
struct SealedMem {
    table: u32,
    schema: Arc<Schema>,
    gen: u64,
    len: usize,
    chunks: Vec<Arc<Vec<Document>>>,
    deleted: Vec<u64>,
    updates: Updates,
    seqs: Vec<u64>,
    /// Whether the database numbers its writes.
    numbered: bool,
}


/// A sealed position that was deleted when the memtable was sealed.
const GONE: u32 = u32::MAX;

impl SealedMem {
    /// Seals the whole memtable: the version the flush installs holds every
    /// transaction up to the seal (the WAL from there on is all the
    /// memtable keeps).
    fn of(t: &TableState, numbered: bool) -> SealedMem {
        let m = &t.mem;
        SealedMem {
            table: t.id,
            schema: t.schema.clone(),
            gen: m.gen,
            len: m.len,
            chunks: m.chunks.clone(),
            deleted: m.deleted.clone(),
            updates: m.updates.clone(),
            seqs: m.seqs.clone(),
            numbered,
        }
    }

    fn was_deleted(&self, p: usize) -> bool {
        self.deleted[p / 64] >> (p % 64) & 1 == 1
    }

    /// The live documents, with their values written in place, their
    /// sequence numbers, and each position's docID among them (`GONE` for
    /// deleted ones).
    fn live(&self) -> (Vec<Document>, Vec<u64>, Vec<u32>) {
        let mut docids = vec![GONE; self.len];
        let (mut live, mut seqs) = (Vec::new(), Vec::new());
        for (p, d) in self.chunks.iter().flat_map(|c| c.iter()).take(self.len).enumerate() {
            if !self.was_deleted(p) {
                docids[p] = live.len() as u32;
                let mut d = d.clone();
                self.updates.apply(p as DocId, &mut d);
                live.push(d);
                seqs.push(mem_seq(&self.updates, &self.seqs, p as u32));
            }
        }
        (live, seqs, docids)
    }

    /// The deletes and values written in place since the seal on sealed
    /// documents, as docIDs of the segments built from them (`max`
    /// documents each, `n_segs` of them), or `None` when the table is gone.
    fn since(&self, st: &State, docids: &[u32], n_segs: usize, max: usize) -> Option<Vec<(Vec<DocId>, Updates)>> {
        let mem = &st.tables[st.table_index(self.table)?].mem;
        let mut out = vec![(Vec::new(), Updates::default()); n_segs];
        for (p, &d) in docids.iter().enumerate() {
            if d != GONE && mem.deleted[p / 64] >> (p % 64) & 1 == 1 {
                out[d as usize / max].0.push(d % max as u32);
            }
        }
        for (f, p, v) in mem.updates.iter() {
            let Some(&d) = docids.get(p as usize).filter(|&&d| d != GONE) else { continue };
            if self.updates.get(f, p) != Some(v) {
                out[d as usize / max].1.set(f, d % max as u32, v.clone());
            }
        }
        Some(out)
    }
}

impl Database {
    /// Creates a new database (the path must not exist) in the layout
    /// `opts.layout` names, and its WAL.
    ///
    /// The database is written under a temporary name beside `path` and
    /// then given its name whole, so a crash leaves a database at `path` or
    /// nothing (and a hidden `.<name>.creating-…` beside it, which can be
    /// deleted): `path` can be created again.
    pub fn create(path: &Path, schema: Schema, opts: DbOptions) -> Result<Database> {
        // A table read from a database written before field ids gets them.
        let schema = if schema.is_legacy() { schema.with_new_ids()? } else { schema };
        schema.check_ids()?;
        if let Some(f) = schema.fields.iter().find(|f| f.references.as_ref().is_some_and(|r| !r.eq_ignore_ascii_case(&schema.table))) {
            let r = f.references.as_deref().unwrap_or_default();
            return Err(Error::Schema(format!("field {}: references {r}, and a new database has no table but {}", f.name, schema.table)));
        }
        if path.exists() {
            return Err(Error::Io(std::io::Error::new(std::io::ErrorKind::AlreadyExists, format!("{} exists", path.display()))));
        }
        let tmp = beside(path, "creating")?;
        if opts.layout == Layout::Directory {
            return Database::create_dir(path, &tmp, schema, opts);
        }
        let db_id: u64 = rand::random();
        let built = (|| -> Result<File> {
            let file = OpenOptions::new().read(true).write(true).create_new(true).open(&tmp)?;
            lock_exclusive(&file)?;
            let manifest = Manifest::new(schema);
            let mut fl = FreeList::new(META_PAGES);
            let mbytes = manifest.encode();
            let mext = Extent { page: fl.alloc(pages_for(mbytes.len())), len: mbytes.len() as u64 };
            let (fext, fbytes) = place_freelist(&mut fl, &[]);
            io::set_len(&file, fl.page_count() * PAGE)?;
            io::write_at(&file, &mbytes, mext.page * PAGE)?;
            io::write_at(&file, &fbytes, fext.page * PAGE)?;
            let meta = Meta {
                format: manifest.format(),
                db_id,
                txn: 1,
                manifest: mext,
                manifest_crc: crc32fast::hash(&mbytes),
                freelist: fext,
                freelist_crc: crc32fast::hash(&fbytes),
                page_count: fl.page_count(),
            };
            pager::write_meta(&file, 0, &meta)?;
            io::sync_all(&file)?;
            // A link, unlike a rename, fails if `path` appeared meanwhile.
            std::fs::hard_link(&tmp, path)?;
            Ok(file)
        })();
        let _ = io::remove_file(&tmp);
        let file = built?;
        // After the link, so that a create that fails leaves another
        // database's log alone. A log that is missing, or is another
        // database's (a crash before this), opens empty.
        Wal::create(&wal_path(path), db_id, false, true)?;
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        io::sync_dir(dir)?;
        Database::load(path, file, opts)
    }

    /// Opens an existing database, recovering from a crash: the newest valid
    /// meta slot (or `CURRENT`) names the on-disk version, and committed WAL
    /// transactions not yet in it are replayed into the memtable.
    pub fn open(path: &Path, opts: DbOptions) -> Result<Database> {
        if path.is_dir() {
            let lockf = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.join(dir::LOCK))?;
            lock_exclusive(&lockf)?;
            return Database::load_dir(path, lockf, opts);
        }
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        lock_exclusive(&file)?;
        Database::load(path, file, opts)
    }

    /// [`Database::create`] in the directory layout, built in `tmp`.
    fn create_dir(path: &Path, tmp: &Path, schema: Schema, opts: DbOptions) -> Result<Database> {
        std::fs::create_dir(tmp)?;
        let built = (|| -> Result<File> {
            let lockf = OpenOptions::new().read(true).write(true).create_new(true).open(tmp.join(dir::LOCK))?;
            lock_exclusive(&lockf)?;
            let db_id: u64 = rand::random();
            let manifest = Manifest::new(schema);
            let mbytes = manifest.encode();
            let name = dir::manifest_name(1);
            dir::write_file(tmp, &name, &mbytes, true)?;
            dir::write_current(tmp, &Current { db_id, txn: 1, manifest: name, crc: crc32fast::hash(&mbytes), format: manifest.format() }, true)?;
            Wal::create(&tmp.join(dir::WAL), db_id, false, true)?;
            dir::sync_dir(tmp)?;
            // Fails if `path` appeared meanwhile and holds files.
            std::fs::rename(tmp, path)?;
            Ok(lockf)
        })();
        let lockf = match built {
            Ok(f) => f,
            Err(e) => {
                let _ = std::fs::remove_dir_all(tmp);
                return Err(e);
            }
        };
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        io::sync_dir(parent)?;
        Database::load_dir(path, lockf, opts)
    }

    fn load_dir(path: &Path, lockf: File, opts: DbOptions) -> Result<Database> {
        let (cur, manifest) = dir::read_current(path)?;
        // Files a crash left behind: written for a commit that did not
        // finish, or no longer named after one that did.
        for (f, _) in dir::orphans(path, &dir::referenced(&cur, &manifest))? {
            io::remove_file(&f)?;
        }
        let handles = manifest
            .tables
            .iter()
            .map(|t| t.segments.iter().map(|m| SegHandle::open_in(path, m, &t.schema, opts.read_ahead, opts.read_ahead && opts.random_hints, opts.check_at_open()).map(Arc::new)).collect())
            .collect::<Result<_>>()?;
        let meta = dir_meta(&cur);
        Database::assemble(path, path.join(dir::WAL), Some(path.to_path_buf()), lockf, opts, manifest, meta, 0, FreeList::new(META_PAGES), OwnedBytes::empty(), handles)
    }

    /// Opens the newest version named by a valid meta slot, or, when that
    /// version is damaged (its manifest or free list fails its checksum,
    /// or a segment it names does not open), the version in the other
    /// slot: the one before it, which the newer commit left intact
    /// ([`crate::storage::freelist`]). With `sync` off, a power loss can
    /// leave the newest slot naming pages that never reached the disk;
    /// that loses the newest commits, not the database.
    fn load(path: &Path, file: File, opts: DbOptions) -> Result<Database> {
        let metas = pager::read_metas(&file)?;
        let mut damaged = None;
        for (i, &(meta, slot)) in metas.iter().enumerate() {
            match Database::load_version(&file, &opts, meta, i > 0) {
                Ok((manifest, freelist, map, handles)) => {
                    let mut db = Database::assemble(path, wal_path(path), None, file, opts, manifest, meta, slot, freelist, map, handles)?;
                    db.damaged = match (damaged, db.damaged.take()) {
                        (Some(a), Some(b)) => Some(format!("{a}; {b}")),
                        (a, b) => a.or(b),
                    };
                    return Ok(db);
                }
                Err(Error::Corrupt(m)) if i + 1 < metas.len() => damaged = Some(format!("version {} skipped: {m}", meta.txn)),
                Err(e) => return Err(e),
            }
        }
        Err(corrupt("no valid meta slot"))
    }

    /// Reads and checks the version `meta` names. A commit cuts free pages
    /// off the end of the file, which can leave it shorter than the
    /// version before it says (`older`); those pages are none that version
    /// references (every page it does is either still in use or a pending
    /// free of the newer commit), so the file is extended back.
    #[allow(clippy::type_complexity)]
    fn load_version(file: &File, opts: &DbOptions, meta: Meta, older: bool) -> Result<(Manifest, FreeList, OwnedBytes, Vec<Vec<Arc<SegHandle>>>)> {
        let manifest = Manifest::decode(&pager::read_extent(file, meta.manifest, meta.manifest_crc)?)?;
        let freelist = FreeList::decode(&pager::read_extent(file, meta.freelist, meta.freelist_crc)?, meta.page_count)?;
        let len = file.metadata()?.len();
        if len < meta.page_count * PAGE {
            if !older {
                return Err(corrupt("file is shorter than its page count"));
            }
            io::set_len(file, meta.page_count * PAGE)?;
        }
        if len > meta.page_count * PAGE {
            // Left by a commit that did not finish, or by the damaged
            // version skipped; nothing this version uses lives there.
            io::set_len(file, meta.page_count * PAGE)?;
        }
        let map = pager::map_file(file, opts.read_ahead)?;
        let handles = manifest
            .tables
            .iter()
            .map(|t| t.segments.iter().map(|m| SegHandle::open(&map, m, &t.schema, opts.read_ahead, opts.read_ahead && opts.random_hints, opts.check_at_open()).map(Arc::new)).collect())
            .collect::<Result<_>>()?;
        Ok((manifest, freelist, map, handles))
    }

    /// The open database around an on-disk version (`handles`: each table's
    /// opened segments): replays the WAL into the memtables.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        path: &Path,
        wal_path: PathBuf,
        dir: Option<PathBuf>,
        file: File,
        opts: DbOptions,
        manifest: Manifest,
        meta: Meta,
        slot: usize,
        freelist: FreeList,
        map: OwnedBytes,
        handles: Vec<Vec<Arc<SegHandle>>>,
    ) -> Result<Database> {
        if manifest.is_legacy() && !opts.open_old {
            return Err(Error::Schema(format!(
                "{} was written by a version of Strata from before field ids, which this one opens only to copy it into a new database: run strata::upgrade on it (strata-server upgrade, or strata.upgrade in Python; engram upgrades its stores itself)",
                path.display()
            )));
        }
        let fits = |table: u32, op: &Op| manifest.table(table).is_none_or(|t| op_fits(&t.schema, op));
        let numbered = manifest.seq_no.is_some();
        let (sealed, mut txns, cut) = read_sealed_wals(&wal_path, &manifest, meta.db_id, opts.wal_drop_damaged, &fits)?;
        let (mut wal, wal_dropped) = match cut {
            None => {
                let (wal, active) = Wal::open(&wal_path, meta.db_id, numbered, opts.sync, opts.wal_drop_damaged, &fits)?;
                txns.extend(active);
                let dropped = wal.dropped().map(str::to_string);
                (wal, dropped)
            }
            Some(what) => {
                // Every transaction after the damage goes, the open log's too.
                if std::fs::metadata(&wal_path).is_ok_and(|m| m.len() > 0) {
                    std::fs::copy(&wal_path, wal::damaged_path(&wal_path))?;
                }
                (Wal::create(&wal_path, meta.db_id, numbered, opts.sync)?, Some(what))
            }
        };
        let tables: Vec<TableState> = manifest
            .tables
            .iter()
            .zip(handles)
            .map(|(t, h)| {
                let mut ts = TableState::new(t.id, Arc::new(t.schema.clone()), h, opts.memtable_chunk_docs, 0);
                ts.profiles = Arc::new(t.profiles.clone());
                ts
            })
            .collect();
        let schema = tables[0].schema.clone();
        let seq = manifest.seq_no.as_ref().map_or(0, |n| n.seq);
        let mut st = State {
            manifest,
            meta,
            slot,
            freelist,
            file_pages: meta.page_count,
            map,
            tables,
            txn: meta.txn,
            seq,
            version: 0,
            pending: Default::default(),
            pending_updates: Default::default(),
            n_pending: 0,
            live_cache: HashMap::new(),
            updates_cache: HashMap::new(),
            cached: None,
            reserved: Vec::new(),
            next_segment_id: 0,
            ship: ShipLog::new(opts.ship_log_bytes),
        };
        let mut flushed = false;
        for t in txns {
            if t.seq.is_some() != numbered {
                return Err(corrupt(format!("a sealed write-ahead log holds transactions {} sequence numbers, and the database {}", if numbered { "without" } else { "with" }, if numbered { "numbers its writes" } else { "does not number its writes" })));
            }
            st.seq = st.seq.max(t.seq.map_or(0, |n| n.seq));
            if let Some(seq) = t.seq.filter(|_| st.ship.keeps()) {
                st.ship.push(Shipped { seq, bytes: wal::encode_txn(t.txn, t.seq, &t.ops).into() });
            }
            if t.txn <= st.manifest.wal_txn {
                // Flushed before a crash cut the WAL truncation short.
                flushed = true;
                continue;
            }
            for (table, op) in t.ops {
                // A dropped table's writes were flushed before the drop,
                // so none can follow it here; skip them all the same.
                if let Some(i) = st.table_index(table) {
                    st.apply(i, op, t.seq.map_or(0, |n| n.seq));
                }
            }
            st.txn = st.txn.max(t.txn);
        }
        let mut sealed = sealed;
        if flushed && st.mem_len() == 0 && st.pending.is_empty() && st.pending_updates.is_empty() {
            wal.reset(opts.sync)?;
            for t in sealed.drain(..) {
                let _ = io::remove_file(&wal::sealed_path(&wal_path, t));
            }
        }
        // Merges a crash interrupted; those of segments that are gone are
        // forgotten.
        let merge_record_path = match &dir {
            Some(d) => d.join(dir::MERGE_RECORD),
            None => {
                let mut s = path.as_os_str().to_owned();
                s.push("-merging");
                PathBuf::from(s)
            }
        };
        let mut record = MergeRecord::load(&merge_record_path);
        record.retain_live(|id| st.handle(id).is_some());
        record.store(&merge_record_path)?;
        let db = Database {
            path: path.to_path_buf(),
            wal_path,
            opts,
            schema,
            file,
            dir,
            writer: Mutex::new(wal),
            state: Mutex::new(st),
            build_lock: Mutex::new(()),
            published: RwLock::new(None),
            merging: Mutex::new(Merging { record, ..Merging::default() }),
            expired: Mutex::new(HashMap::new()),
            merge_record_path,
            merger: OnceLock::new(),
            merger_threads: Mutex::new(Vec::new()),
            pins: Arc::new(Pins::default()),
            mem_ids: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            flush_error: Mutex::new(None),
            damaged: wal_dropped,
            sealed_wals: Mutex::new(sealed),
            background: Mutex::new(()),
            flush_signal: OnceLock::new(),
            flusher: Mutex::new(None),
            background_waiting: AtomicUsize::new(0),
            background_holding: AtomicUsize::new(0),
            commit_hook: RwLock::new(None),
            codebooks: RwLock::new(Vec::new()),
        };
        db.republish();
        Ok(db)
    }

    /// The schema of the database's first table.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The schemas of the database's tables, in creation order (the first
    /// is the table the database was created with). The system table
    /// [`SOURCES`] is not among them: [`Self::table`] and SQL read it by
    /// name.
    pub fn tables(&self) -> Vec<Arc<Schema>> {
        lock(&self.state).tables.iter().filter(|t| t.id != SOURCES_ID).map(|t| t.schema.clone()).collect()
    }

    /// The schema of the table named `name` (any case).
    pub fn table(&self, name: &str) -> Option<Arc<Schema>> {
        lock(&self.state).tables.iter().find(|t| t.schema.table.eq_ignore_ascii_case(name)).map(|t| t.schema.clone())
    }

    /// Adds an empty table with this schema, in one commit. Its name must
    /// be a new one (in any case) and a plain identifier: letters, digits
    /// and underscores, not starting with a digit.
    ///
    /// A database with more than one table is written in format 2, which
    /// versions of Strata before tables cannot open; dropping back to the
    /// first table alone writes format 1 again.
    ///
    /// A field that references another table ([`crate::schema::Field::references`])
    /// must name a table of this database whose primary key has the field's
    /// type; the reference is spelled as that table's name.
    pub fn create_table(&self, mut schema: Schema) -> Result<()> {
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        check_table_name(&schema.table)?;
        if schema.table.eq_ignore_ascii_case(SOURCES) {
            return Err(Error::Schema(format!("{SOURCES} is a system table, which streams make and write (Database::ensure_sources)")));
        }
        if self.table(&schema.table).is_some() {
            return Err(Error::Schema(format!("table {} already exists", schema.table)));
        }
        let tables = self.tables();
        for f in schema.fields.iter_mut() {
            let Some(r) = &f.references else { continue };
            if r.eq_ignore_ascii_case(&schema.table) {
                continue;
            }
            let t = tables.iter().find(|t| t.table.eq_ignore_ascii_case(r)).ok_or_else(|| Error::Schema(format!("field {}: references {r}, which is not a table", f.name)))?;
            let pk = t.primary_key.map(|k| &t.fields[k]).ok_or_else(|| Error::Schema(format!("field {}: references {}, which has no primary key", f.name, t.table)))?;
            if pk.ty != f.ty {
                return Err(Error::Schema(format!(
                    "field {}: references {}, whose primary key {} is {}, not {}",
                    f.name,
                    t.table,
                    pk.name,
                    crate::schema::type_name(&pk.ty),
                    crate::schema::type_name(&f.ty)
                )));
            }
            f.references = Some(t.table.clone());
        }
        let mut vw = VersionWriter::begin(self);
        if vw.manifest.is_legacy() {
            // Opened to be copied: it stays as it was written.
            schema = schema.without_ids();
        } else if schema.is_legacy() {
            // A table read from a database written before field ids.
            schema = schema.with_new_ids()?;
        }
        schema.check_ids()?;
        let id = vw.manifest.next_table_id;
        vw.manifest.next_table_id += 1;
        vw.manifest.tables.push(TableMeta { id, schema, segments: Vec::new(), profiles: Vec::new() });
        let c = vw.commit(&mut wal)?;
        self.publish(c, |_| {})
    }

    /// Gives table `table` (any case) the rank profile `profile`, in place
    /// of one of the same name (any case) only when `replace`. It is
    /// logged as a write is, in a transaction of its own, so it is
    /// numbered, replayed and shipped to replicas as writes are, and it
    /// changes no row. The database then requires feature `rank_profiles`
    /// (format 3) while a table has a profile, and its log names the
    /// feature from the first profile record until it is next emptied
    /// ([`Wal::allow_profiles`]: that first record flushes the memtables).
    ///
    /// Strata keeps a profile as written; [`crate::query::profile`] reads
    /// it when a query ranks by it.
    pub fn set_rank_profile(&self, table: &str, profile: RankProfile, replace: bool) -> Result<()> {
        check_name("rank profile", &profile.name)?;
        let mut txn = self.begin()?;
        let t = self.table_ref(Some(table))?;
        if t.id == SOURCES_ID {
            return Err(Error::Schema(format!("{SOURCES} is a system table, which takes no rank profile")));
        }
        crate::query::profile::check_columns(&profile, &t.schema)?;
        let exists = lock(&self.state).tables[t.pos].profiles.iter().any(|p| p.name.eq_ignore_ascii_case(&profile.name));
        if exists && !replace {
            return Err(Error::Schema(format!("{} has a rank profile named {} already; CREATE OR REPLACE RANK PROFILE replaces it", t.schema.table, profile.name)));
        }
        txn.ops.push((t.id, Op::Profile { name: profile.name.clone(), def: Some(Arc::new(profile)) }));
        txn.commit()
    }

    /// Drops table `table`'s rank profile named `name` (any case), logged
    /// as [`Database::set_rank_profile`] logs one. Returns whether there
    /// was one.
    pub fn drop_rank_profile(&self, table: &str, name: &str) -> Result<bool> {
        let mut txn = self.begin()?;
        let t = self.table_ref(Some(table))?;
        let Some(name) = lock(&self.state).tables[t.pos].profiles.iter().find(|p| p.name.eq_ignore_ascii_case(name)).map(|p| p.name.clone()) else { return Ok(false) };
        txn.ops.push((t.id, Op::Profile { name, def: None }));
        txn.commit()?;
        Ok(true)
    }

    /// The rank profiles of table `table` (any case), by name.
    pub fn rank_profiles(&self, table: &str) -> Result<Vec<RankProfile>> {
        let st = lock(&self.state);
        let t = st.tables.iter().find(|t| t.schema.table.eq_ignore_ascii_case(table)).ok_or_else(|| Error::Schema(format!("no table {table}")))?;
        Ok(t.profiles.to_vec())
    }

    /// Starts the log again naming `rank_profiles`, before the first
    /// profile record goes into it: what it holds is in a version first
    /// (the memtables flushed, then a commit that records the last
    /// transaction), so emptying it drops nothing.
    fn allow_profiles(&self, wal: &mut Wal) -> Result<()> {
        self.flush_locked(wal)?;
        if !wal.is_empty() {
            // Transactions that left nothing to flush.
            let txn = lock(&self.state).txn;
            let mut vw = VersionWriter::begin(self);
            vw.manifest.wal_txn = txn;
            let c = vw.commit(wal)?;
            self.publish(c, |_| {})?;
        }
        wal.allow_profiles(self.opts.sync).map_err(|e| self.poison(e))
    }

    /// Adds the system table [`SOURCES`] if the database lacks it, in one
    /// commit: streams keep there, per source, the offset of the last
    /// event they wrote, in the transaction that wrote it. Every copy of a
    /// database gives it the same id ([`SOURCES_ID`]), so a replica makes
    /// its own when its primary's first offset arrives
    /// ([`Database::apply_shipped`]). The database then needs feature
    /// `sources`, so versions of Strata before it refuse to open it.
    pub fn ensure_sources(&self) -> Result<()> {
        if self.has_sources() {
            return Ok(());
        }
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        self.add_sources(&mut wal)
    }

    fn has_sources(&self) -> bool {
        lock(&self.state).tables.iter().any(|t| t.id == SOURCES_ID)
    }

    /// [`Database::ensure_sources`], under the writer lock.
    fn add_sources(&self, wal: &mut Wal) -> Result<()> {
        if self.has_sources() {
            return Ok(());
        }
        let mut vw = VersionWriter::begin(self);
        let schema = if vw.manifest.is_legacy() { sources_schema().without_ids() } else { sources_schema() };
        vw.manifest.tables.push(TableMeta { id: SOURCES_ID, schema, segments: Vec::new(), profiles: Vec::new() });
        let c = vw.commit(wal)?;
        self.publish(c, |_| {})
    }

    /// Numbers this database's writes from now on: every write transaction
    /// gets the next sequence number (from 1), under the database's term
    /// (1 at first, raised by [`Database::begin_term`]), logged with it and
    /// kept across flushes, crashes and reopening. Replication orders
    /// writes by them. The memtables are flushed first, and the WAL starts
    /// again numbered (`STRWAL02`).
    ///
    /// It cannot be undone: the database then requires feature `seq_no`
    /// (format 3), so versions of Strata from before it refuse to open it
    /// rather than lose its numbers. Bulk loads and attached segments,
    /// which skip the WAL, take no sequence number. Starting again is a
    /// no-op.
    pub fn start_seq_no(&self) -> Result<()> {
        // No background flush may seal the log in between.
        let _one = lock(&self.background);
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        if lock(&self.state).manifest.seq_no.is_some() {
            return Ok(());
        }
        self.flush_locked(&mut wal)?;
        // What a flush left in the log is in the segments already; a crash
        // after the commit below finds the log empty, and starts it again
        // numbered.
        if !wal.is_empty() {
            wal.reset(self.opts.sync).map_err(|e| self.poison(e))?;
        }
        let mut vw = VersionWriter::begin(self);
        vw.manifest.seq_no = Some(Numbering::start());
        let c = vw.commit(&mut wal)?;
        self.publish(c, |_| {})?;
        wal.renumber(self.opts.sync).map_err(|e| self.poison(e))
    }

    /// Raises the term this database numbers its writes under to `term`,
    /// which must be greater than its term now; durable on return. A
    /// replica made primary does this, so that its writes are told apart
    /// from those of the primary before it.
    pub fn begin_term(&self, term: u64) -> Result<()> {
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        self.raise_term(&mut wal, term)
    }

    /// [`Database::begin_term`], under the writer lock.
    fn raise_term(&self, wal: &mut Wal, term: u64) -> Result<()> {
        let (now, seq) = {
            let st = lock(&self.state);
            (st.manifest.seq_no.clone().ok_or_else(|| Error::Schema("the database does not number its writes (Database::start_seq_no)".into()))?, st.seq)
        };
        if term <= now.term {
            return Err(Error::Schema(format!("term {term} is not after the database's term, {}", now.term)));
        }
        let mut vw = VersionWriter::begin(self);
        let n = vw.manifest.seq_no.as_mut().expect("numbered");
        n.term = term;
        n.terms.push((term, seq + 1));
        let c = vw.commit(wal)?;
        self.publish(c, |_| {})
    }

    /// The numbered transactions from sequence number `from` on, as they
    /// were logged, up to `max_bytes` of them (at least one), for a replica
    /// to apply ([`Database::apply_shipped`]): none when `from` is past the
    /// last number given, and [`LogRead::Gone`] when the ship log
    /// ([`DbOptions::ship_log_bytes`]) no longer holds it.
    pub fn log_from(&self, from: u64, max_bytes: usize) -> LogRead {
        let st = lock(&self.state);
        st.ship.read(from, st.seq, max_bytes)
    }

    /// Whether it keeps a ship log ([`DbOptions::ship_log_bytes`]).
    pub fn keeps_ship_log(&self) -> bool {
        self.opts.ship_log_bytes > 0
    }

    /// Holds the transactions from number `from` on in the ship log until
    /// [`Database::release_log`], as a primary does for a replica it sends
    /// a copy, which then follows from the copy's last transaction: the
    /// log keeps them past [`DbOptions::ship_log_bytes`], up to
    /// [`crate::storage::ship::HELD`] times that. Returns the hold's id.
    pub fn hold_log(&self, from: u64) -> u64 {
        lock(&self.state).ship.hold(from)
    }

    /// Ends hold `id` ([`Database::hold_log`]).
    pub fn release_log(&self, id: u64) {
        lock(&self.state).ship.release(id);
    }

    /// Has `f` told the term and number of each numbered transaction once
    /// it is committed, in the order they commit (replacing any earlier
    /// `f`). It runs in the writer, so it must not block or write.
    pub fn on_commit(&self, f: impl Fn(SeqNo) + Send + Sync + 'static) {
        *self.commit_hook.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
    }

    /// Applies transactions a primary shipped ([`Database::log_from`]), in
    /// order, each as one transaction under the primary's term and number,
    /// so that this database holds the same rows with the same numbers;
    /// durable on return with `DbOptions::sync`. `prev` is the transaction
    /// before the first, as the primary has it (its number, and that
    /// number's term): when it is not this database's last
    /// ([`Numbering::last`]), nothing is applied ([`Applied::Mismatch`]).
    /// A transaction with a newer term than this database's raises its
    /// term first.
    ///
    /// Each transaction is checked before any is applied: an error for one
    /// that is damaged, does not follow the one before it, or comes from an
    /// older term than the one before it (an old primary's, which a newer
    /// one has replaced). An error also for one whose operations do not
    /// apply to the rows here, as they did on the primary (a delete or a
    /// set of a row that is not live, an insert of one that is), which
    /// means that the two databases were not copies of each other: it is
    /// not applied, and those before it are.
    pub fn apply_shipped(&self, prev: SeqNo, txns: &[&[u8]]) -> Result<Applied> {
        let now = || self.numbering().ok_or_else(|| Error::Schema("the database does not number its writes (Database::start_seq_no)".into()));
        // None: a primary asks the last transaction, which needs no writer
        // (a flush or a merge may hold it).
        let held = if txns.is_empty() { None } else { Some(self.begin()?) };
        let now = now()?;
        if now.last() != prev {
            let last = now.last();
            return Ok(Applied::Mismatch { seq: last.seq, term: last.term });
        }
        let Some(WriteTxn { mut wal, .. }) = held else { return Ok(Applied::At(prev)) };
        let mut schemas: Vec<(u32, Arc<Schema>)> = lock(&self.state).tables.iter().map(|t| (t.id, t.schema.clone())).collect();
        // The primary's sources' offsets, before this copy has a table for
        // them: it makes its own, as the primary did.
        if !schemas.iter().any(|(id, _)| *id == SOURCES_ID) {
            schemas.push((SOURCES_ID, Arc::new(sources_schema())));
        }
        let fits = |table: u32, op: &Op| schemas.iter().find(|(id, _)| *id == table).is_some_and(|(_, s)| op_fits(s, op));
        let mut shipped = Vec::with_capacity(txns.len());
        let mut at = now.last();
        for bytes in txns {
            let t = wal::decode_shipped(bytes, &fits)?;
            let s = t.seq.expect("a shipped transaction is numbered");
            if s.seq != at.seq + 1 {
                return Err(corrupt(format!("shipped transaction {} does not follow {}", s.seq, at.seq)));
            }
            if s.term < at.term.max(now.term) {
                return Err(Error::Schema(format!(
                    "shipped transaction {} has term {}, older than term {}: it comes from a primary that another has replaced",
                    s.seq,
                    s.term,
                    at.term.max(now.term)
                )));
            }
            at = s;
            shipped.push((s, t.ops));
        }
        if shipped.iter().any(|(_, ops)| ops.iter().any(|(id, _)| *id == SOURCES_ID)) {
            self.add_sources(&mut wal)?;
        }
        let mut at = now.last();
        let mut failed = None;
        for (s, ops) in shipped {
            if s.term > lock(&self.state).manifest.seq_no.as_ref().map_or(0, |n| n.term) {
                if let Err(e) = self.raise_term(&mut wal, s.term) {
                    failed = Some(e);
                    break;
                }
            }
            let mut txn = WriteTxn { db: self, wal, ops: Vec::new(), keys: Vec::new(), ties: Vec::new() };
            let replayed = txn.replay(ops);
            let WriteTxn { wal: w, ops, .. } = txn;
            wal = w;
            // The term is the transaction's and it follows the last, so the
            // commit gives it the primary's number, unless it wrote nothing.
            match replayed.and_then(|_| self.commit_logged(&mut wal, ops, false)) {
                Ok(Some(n)) if n == s.seq => {}
                Ok(got) => {
                    failed = Some(corrupt(format!("shipped transaction {} committed here as {got:?}", s.seq)));
                    break;
                }
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
            at = s;
        }
        if self.opts.sync {
            wal.sync_appends().map_err(|e| self.poison(e.into()))?;
        }
        match failed {
            Some(e) => Err(e),
            None => Ok(Applied::At(at)),
        }
    }

    /// The term writes are numbered under and the last sequence number
    /// given, once the database numbers its writes
    /// ([`Database::start_seq_no`]).
    pub fn seq_no(&self) -> Option<SeqNo> {
        self.numbering().map(|n| n.at())
    }

    /// The sequence number of the write that last changed the live row
    /// with primary key `key` in table `table` (the first when `None`),
    /// which a conditional write compares with: `None` when no row has the
    /// key, 0 for a row written before the database numbered its writes or
    /// bulk loaded. An error when the database does not number its writes.
    pub fn seq_of(&self, table: Option<&str>, key: &Value) -> Result<Option<u64>> {
        // The table is found under the same lock as the row: without the
        // writer lock, a drop could move it in between.
        let st = lock(&self.state);
        if st.manifest.seq_no.is_none() {
            return Err(Error::Schema("the database does not number its writes (Database::start_seq_no)".into()));
        }
        let pos = match table {
            None => 0,
            Some(n) => st.tables.iter().position(|t| t.schema.table.eq_ignore_ascii_case(n)).ok_or_else(|| Error::Schema(format!("no table {n}")))?,
        };
        let t = &st.tables[pos];
        TableRef { id: t.id, pos, schema: t.schema.clone() }.key_of(key)?;
        st.find_live(pos, key).map(|loc| st.seq_at(pos, loc)).transpose()
    }

    /// Replaces this database's rows with those of `copy`, a checkpoint of
    /// its primary ([`Database::checkpoint`]), in one commit, and numbers
    /// on from the copy's last transaction under the copy's terms: what a
    /// replica that its primary's ship log no longer reaches does, before
    /// it applies the transactions that follow
    /// ([`Database::apply_shipped`]). Every row, row number and term here
    /// is replaced, the WAL and the ship log are emptied, and snapshots
    /// taken before go on reading what they read; a merge running here
    /// installs nothing. Both must number their writes and hold the same
    /// tables (names and schemas, in order), and the copy's term must not
    /// be older than this database's: a copy of a replaced primary is
    /// refused. Durable on return. Returns the copy's last transaction.
    pub fn install_copy(&self, copy: &Database) -> Result<SeqNo> {
        if std::ptr::eq(self, copy) {
            return Err(Error::Schema("a database is not a copy of itself".into()));
        }
        // What the copy's memtables hold goes into its segments first.
        copy.flush()?;
        let (theirs, tables) = {
            let st = lock(&copy.state);
            let n = st.manifest.seq_no.clone().ok_or_else(|| Error::Schema("the copy does not number its writes".into()))?;
            let mut tables = Vec::new();
            let mut sources = None;
            for t in &st.manifest.tables {
                let segs = t.segments.iter().map(|m| Ok((m.n_docs, st.handle(m.id).cloned().ok_or_else(|| corrupt(format!("segment {} is in the manifest but not open", m.id)))?))).collect::<Result<Vec<_>>>()?;
                match t.id == SOURCES_ID {
                    true => sources = Some(segs),
                    false => {
                        // Logged since its last version, maybe.
                        let profiles = st.tables.iter().find(|ts| ts.id == t.id).map_or_else(Vec::new, |ts| ts.profiles.to_vec());
                        tables.push((t.schema.clone(), segs, profiles));
                    }
                }
            }
            (Numbering { seq: st.seq, ..n }, (tables, sources))
        };
        let (tables, sources) = tables;
        // No background flush may seal the log in between.
        let _one = lock(&self.background);
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        let ours = self.numbering().ok_or_else(|| Error::Schema("the database does not number its writes (Database::start_seq_no)".into()))?;
        {
            // Sources aside, which either may have made first.
            let st = lock(&self.state);
            let mine: Vec<&Schema> = st.manifest.tables.iter().filter(|t| t.id != SOURCES_ID).map(|t| &t.schema).collect();
            let names = |ts: &mut dyn Iterator<Item = &Schema>| ts.map(|s| s.table.clone()).collect::<Vec<_>>().join(", ");
            if mine.len() != tables.len() || mine.iter().zip(&tables).any(|(t, (s, _, _))| **t != *s) {
                return Err(Error::Schema(format!(
                    "the copy's tables ({}) are not this database's ({}), or their schemas differ",
                    names(&mut tables.iter().map(|t| &t.0)),
                    names(&mut mine.into_iter())
                )));
            }
        }
        if sources.is_some() {
            self.add_sources(&mut wal)?;
        }
        if theirs.term < ours.term {
            return Err(Error::Schema(format!("the copy is of term {}, older than this database's ({}): it is a copy of a replaced primary", theirs.term, ours.term)));
        }
        // What the memtables and pending deletes hold goes into segments,
        // which the commit below drops with the others, and the log they
        // were in is emptied: a crash before that commit finds this
        // database as it was, and one after it finds the copy.
        self.flush_locked(&mut wal)?;
        if !wal.is_empty() {
            wal.reset(self.opts.sync).map_err(|e| self.poison(e))?;
        }
        let mut vw = VersionWriter::begin(self);
        let ids: Vec<u32> = vw.manifest.tables.iter().map(|t| t.id).filter(|&id| id != SOURCES_ID).collect();
        for id in vw.manifest.tables.iter().flat_map(|t| &t.segments).map(|s| s.id).collect::<Vec<_>>() {
            vw.remove_segment(id);
        }
        let mut into: Vec<_> = ids.into_iter().zip(tables.iter().map(|(_, segs, profiles)| (segs, profiles.clone()))).collect();
        if let Some(segs) = &sources {
            into.push((SOURCES_ID, (segs, Vec::new())));
        }
        for (to, (segs, profiles)) in into {
            for (n_docs, h) in segs {
                vw.add_segment(to, None, h.file.image(), *n_docs, h.base.as_deref(), h.updates.as_deref())?;
            }
            vw.table_mut(to).profiles = profiles;
        }
        vw.manifest.seq_no = Some(theirs.clone());
        // The commit keeps the last number given here when it is higher: the
        // copy's is the one to keep.
        let before = std::mem::replace(&mut lock(&self.state).seq, theirs.seq);
        let c = match vw.commit(&mut wal) {
            Ok(c) => c,
            Err(e) => {
                lock(&self.state).seq = before;
                return Err(e);
            }
        };
        let max = self.opts.ship_log_bytes;
        self.publish(c, |st| {
            st.live_cache.clear();
            st.updates_cache.clear();
            st.ship = ShipLog::new(max);
            for t in &mut st.tables {
                t.ties = None;
            }
        })?;
        Ok(theirs.last())
    }

    /// How the database numbers its writes ([`Numbering`]): its term, the
    /// last sequence number given and every term with the first number it
    /// gave.
    pub fn numbering(&self) -> Option<Numbering> {
        let st = lock(&self.state);
        st.manifest.seq_no.clone().map(|n| Numbering { seq: st.seq, ..n })
    }

    /// Drops the table named `name` (any case) and its rows, in one commit.
    /// The first table (the one the database was created with) cannot be
    /// dropped, nor a table that another one references
    /// ([`Database::drop_tables`] can). The memtables are flushed first, so
    /// that no logged write outlives its table. Its pages are freed once no
    /// snapshot reads them.
    pub fn drop_table(&self, name: &str) -> Result<()> {
        self.drop_tables(&[name], DropOptions::default())
    }

    /// Drops the tables named in `names` (any case) and their rows, in one
    /// commit, as [`Database::drop_table`] drops one. A table that a table
    /// left standing references ([`crate::schema::Field::references`]) is
    /// dropped only with `opts.cascade`, which removes those references in
    /// the same commit; a name that is not a table is an error unless
    /// `opts.if_exists`.
    pub fn drop_tables(&self, names: &[&str], opts: DropOptions) -> Result<()> {
        let mut wal = self.try_writer()?;
        let (ids, dropped) = {
            let st = lock(&self.state);
            let mut ids = Vec::new();
            let mut dropped: Vec<String> = Vec::new();
            for name in names {
                match st.tables.iter().find(|t| t.schema.table.eq_ignore_ascii_case(name)) {
                    None if opts.if_exists => {}
                    None => return Err(Error::Schema(format!("no table {name}"))),
                    Some(t) if t.id == 0 => return Err(Error::Schema(format!("{} is the database's first table, which cannot be dropped", t.schema.table))),
                    // A replica's would come back with its primary's next offset.
                    Some(t) if t.id == SOURCES_ID => return Err(Error::Schema(format!("{SOURCES} is a system table, which cannot be dropped: delete its rows to forget sources"))),
                    Some(t) if ids.contains(&t.id) => {}
                    Some(t) => {
                        ids.push(t.id);
                        dropped.push(t.schema.table.clone());
                    }
                }
            }
            if !opts.cascade {
                for t in st.tables.iter().filter(|t| !ids.contains(&t.id)) {
                    if let Some((f, r)) = dropped.iter().find_map(|d| t.schema.referencing(d).next().map(|f| (f, d))) {
                        let (t, f) = (&t.schema.table, &t.schema.fields[f].name);
                        return Err(Error::Schema(format!("{t}.{f} references {r}: drop {t} as well, or drop with CASCADE, which removes the reference")));
                    }
                }
            }
            (ids, dropped)
        };
        if ids.is_empty() {
            return Ok(());
        }
        self.flush_locked(&mut wal)?;
        if !wal.is_empty() {
            // Only transactions the segments already hold can be left.
            wal.reset(self.opts.sync).map_err(|e| self.poison(e))?;
        }
        let mut vw = VersionWriter::begin(self);
        for &id in &ids {
            vw.drop_table(id);
        }
        for t in vw.manifest.tables.iter_mut() {
            for f in t.schema.fields.iter_mut() {
                if f.references.as_ref().is_some_and(|r| dropped.iter().any(|d| d.eq_ignore_ascii_case(r))) {
                    f.references = None;
                }
            }
        }
        let before = lock(&self.state).manifest.format();
        let c = vw.commit(&mut wal)?;
        self.publish(c, |st| {
            st.live_cache.clear();
            st.updates_cache.clear();
        })?;
        if self.dir.is_none() && lock(&self.state).manifest.format() < before {
            // The other meta slot still names a version of the newer
            // format (several tables, or a format 3 feature), and versions
            // of Strata that do not read it refuse a file with such a slot
            // even when it is the older one. Commit again so that both
            // slots hold the older format.
            let c = VersionWriter::begin(self).commit(&mut wal)?;
            self.publish(c, |_| {})?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn options(&self) -> &DbOptions {
        &self.opts
    }

    /// Starts the write transaction; `Error::Busy` while another one (or a
    /// flush, bulk load or merge commit) is in progress.
    pub fn begin(&self) -> Result<WriteTxn<'_>> {
        self.wait_for_flusher();
        let wal = self.try_writer()?;
        self.check_poisoned()?;
        Ok(WriteTxn { db: self, wal, ops: Vec::new(), keys: Vec::new(), ties: Vec::new() })
    }

    /// The current version. Never waits for a writer; the first snapshot
    /// after a write rebuilds the last memtable chunk's segment, unless
    /// `DbOptions::publish_snapshots` has the writer build it.
    pub fn snapshot(&self) -> Snapshot {
        if let Some(s) = &*self.published.read().unwrap_or_else(|e| e.into_inner()) {
            return s.clone();
        }
        self.snapshot_inner(true)
    }

    /// With `publish_snapshots`, builds the current version's snapshot and
    /// hands it to readers from now on. Callers hold the writer lock (or
    /// are opening the database), so versions are published in order.
    fn republish(&self) {
        if self.opts.publish_snapshots {
            let s = self.snapshot_inner(true);
            *self.published.write().unwrap_or_else(|e| e.into_inner()) = Some(s);
        }
    }

    /// A snapshot without the memtable (merges read only on-disk segments).
    fn snapshot_inner(&self, with_mem: bool) -> Snapshot {
        let mut cap = {
            let mut st = lock(&self.state);
            if let (true, Some(s)) = (with_mem, &st.cached) {
                return Snapshot::of(s.clone());
            }
            st.capture(&self.pins, with_mem)
        };
        if cap.tables.iter().any(|t| t.mem.iter().any(|p| p.seg.is_none())) {
            let _b = lock(&self.build_lock);
            let opts = self.opts.mem_build.as_ref().unwrap_or(&self.opts.build);
            for t in &mut cap.tables {
                for part in t.mem.iter_mut().filter(|p| p.seg.is_none()) {
                    // Another reader may have built it while we waited.
                    let built = lock(&self.state).memseg(part.table, part.gen, part.chunk, part.len);
                    part.seg = Some(built.unwrap_or_else(|| {
                        let id = MEM_ID | self.mem_ids.fetch_add(1, Ordering::Relaxed);
                        let seg = build_memseg(&t.schema, opts, part, id);
                        lock(&self.state).keep_memseg(part.table, &seg);
                        seg
                    }));
                }
            }
        }
        let version = cap.version;
        let inner = Arc::new(cap.assemble(self.path.clone()));
        if with_mem {
            let mut st = lock(&self.state);
            if st.version == version {
                st.cached = Some(inner.clone());
            }
        }
        Snapshot::of(inner)
    }

    /// Writes the memtables as segments and pending deletes into delete
    /// sets, all tables in one commit, then empties the WAL.
    pub fn flush(&self) -> Result<()> {
        let mut wal = self.try_writer()?;
        self.flush_locked(&mut wal)
    }

    /// Why the last flush failed, if no flush has succeeded since. Writes
    /// still commit meanwhile (the WAL holds them) until the memtable
    /// reaches twice its limit.
    pub fn flush_error(&self) -> Option<String> {
        lock(&self.flush_error).clone()
    }

    /// Set when opening skipped something damaged: the newest version on
    /// disk, when it failed a check and the one before it was opened (the
    /// next commit overwrites its meta slot), or committed transactions
    /// after damage in the WAL, which were dropped after the log was copied
    /// aside. Says what, why and where the copy is.
    pub fn damaged(&self) -> Option<&str> {
        self.damaged.as_deref()
    }

    /// Whether a write failed partway (a WAL append, a meta page), after
    /// which every write and merge fails until the database is reopened.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    fn flush_locked(&self, wal: &mut Wal) -> Result<()> {
        let r = self.flush_inner(wal);
        *lock(&self.flush_error) = r.as_ref().err().map(|e| e.to_string());
        r
    }

    fn flush_inner(&self, wal: &mut Wal) -> Result<()> {
        self.check_poisoned()?;
        let (mems, pending, pending_updates, txn, numbered) = {
            let st = lock(&self.state);
            // Rank profiles changed since the last version go into one too.
            let profiles_kept = st.tables.iter().all(|t| st.manifest.table(t.id).is_some_and(|m| m.profiles == *t.profiles));
            if st.mem_len() == 0 && st.pending.is_empty() && st.pending_updates.is_empty() && profiles_kept {
                return Ok(());
            }
            let mems: Vec<_> = st
                .tables
                .iter()
                .filter(|t| t.mem.len > 0)
                .map(|t| (t.id, t.schema.clone(), t.mem.chunks.clone(), t.mem.deleted.clone(), t.mem.updates.clone(), t.mem.seqs.clone()))
                .collect();
            (mems, st.pending.clone(), st.pending_updates.clone(), st.txn, st.manifest.seq_no.is_some())
        };
        let mut vw = VersionWriter::begin(self);
        vw.manifest.wal_txn = txn;
        for (table, schema, chunks, deleted, mem_updates, mem_seqs) in mems {
            // The codes of the deleted and expired documents, which the
            // segments drop.
            let now = crate::timestamp::now();
            let retention = schema.retention(now);
            let mut gone = Vec::new();
            if !crate::sid::fields(&schema).is_empty() {
                let (mut dropped, mut kept) = (Ties::default(), Ties::default());
                for (i, d) in chunks.iter().flat_map(|c| c.iter()).enumerate() {
                    let t = if deleted[i / 64] >> (i % 64) & 1 == 1 || expired(retention, d) { &mut dropped } else { &mut kept };
                    t.saw_doc(&schema, d);
                }
                gone = crate::sid::gone_sections(&schema, &dropped, &kept);
            }
            let (live, seqs): (Vec<Document>, Vec<u64>) = chunks
                .iter()
                .flat_map(|c| c.iter())
                .enumerate()
                .filter(|(i, _)| deleted[i / 64] >> (i % 64) & 1 == 0)
                .map(|(i, d)| {
                    let mut d = d.clone();
                    mem_updates.apply(i as DocId, &mut d);
                    (d, mem_seq(&mem_updates, &mem_seqs, i as u32))
                })
                .unzip();
            drop(chunks);
            let (live, seqs) = arrange(&schema, live, seqs, retention);
            let max = self.opts.segment_max_docs.max(1);
            if live.is_empty() && !gone.is_empty() {
                // A segment without documents keeps what they were given.
                let image = rowseq::build_with(&schema, &[], numbered.then_some(&[]), &self.opts.build, std::mem::take(&mut gone))?;
                vw.add_segment(table, None, &image, 0, None, None)?;
            }
            for (chunk, seqs) in live.chunks(max).zip(seqs.chunks(max)) {
                let image = rowseq::build_with(&schema, chunk, numbered.then_some(seqs), &self.opts.build, std::mem::take(&mut gone))?;
                vw.add_segment(table, None, &image, chunk.len() as u32, None, None)?;
            }
        }
        for (&id, docs) in &pending {
            vw.add_deletes(id, docs)?;
        }
        for (&id, (_, ups)) in &pending_updates {
            vw.add_updates(id, ups)?;
        }
        let c = vw.commit(wal)?;
        self.publish(c, |st| {
            for t in &mut st.tables {
                t.clear_mem();
            }
            st.pending.clear();
            st.pending_updates.clear();
            st.updates_cache.clear();
            st.n_pending = 0;
        })?;
        wal.reset(self.opts.sync).map_err(|e| self.poison(e))?;
        self.remove_sealed(txn);
        Ok(())
    }

    /// Deletes the sealed WAL files the on-disk version holds (sealed at or
    /// before `txn`). One that stays is deleted when the database opens.
    fn remove_sealed(&self, txn: u64) {
        lock(&self.sealed_wals).retain(|&t| {
            let flushed = t <= txn;
            if flushed {
                let _ = io::remove_file(&wal::sealed_path(&self.wal_path, t));
            }
            !flushed
        });
    }

    /// Whether the memtables and pending deletes are past
    /// `memtable_max_docs`.
    fn flush_due(&self) -> bool {
        let st = lock(&self.state);
        st.mem_len() + st.n_pending > self.opts.memtable_max_docs
    }

    /// Whether writes fell behind the flusher: past twice
    /// `memtable_max_docs` while the flusher flushes (or is about to), a
    /// commit waits for its flush ([`Self::spawn_flusher`]). A server stops
    /// reading a stream meanwhile and refuses plain writes (429). Not while
    /// the flusher is idle, as after a reopen replays a long log: then the
    /// next commit flushes, or wakes it; nor while flushes fail, which
    /// commits report.
    pub fn backlogged(&self) -> bool {
        let Some(sig) = self.flush_signal.get() else { return false };
        if !self.over_twice() || self.flush_error().is_some() {
            return false;
        }
        let w = lock(&sig.state);
        (w.running || w.due) && !w.closed
    }

    /// Whether the memtables and pending deletes are past twice
    /// `memtable_max_docs`, where commits stop leaving flushes to the
    /// flusher.
    fn over_twice(&self) -> bool {
        let st = lock(&self.state);
        st.mem_len() + st.n_pending > 2 * self.opts.memtable_max_docs.max(1)
    }

    /// Past twice `memtable_max_docs` while the flusher flushes (or is about
    /// to), waits for it, without the writer lock: writes then go at the
    /// pace of flushes, rather than a commit flushing the whole memtable
    /// itself. A flusher whose flushes fail is not waited for (commits
    /// flush, or are refused).
    fn wait_for_flusher(&self) {
        let Some(sig) = self.flush_signal.get() else { return };
        let mut w = lock(&sig.state);
        while (w.running || w.due) && !w.closed && self.over_twice() {
            w = sig.cv.wait_timeout(w, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    /// Starts a thread that flushes in the background
    /// ([`Database::background_flush`]) when a commit fills the memtable,
    /// in place of that commit flushing before it returns. Past twice
    /// `memtable_max_docs` (the flusher fell behind), new writes wait for
    /// its flush; if its flushes fail, commits flush the memtable whole or
    /// are refused, as without a flusher. The thread ends with the
    /// database; calling this again does nothing.
    pub fn spawn_flusher(self: &Arc<Self>) {
        let sig = Arc::new(FlushSignal::default());
        if self.flush_signal.set(sig.clone()).is_err() {
            return;
        }
        let db = Arc::downgrade(self);
        let h = std::thread::Builder::new().name("strata-flusher".into()).spawn(move || flusher(db, sig)).expect("spawn the flusher thread");
        *lock(&self.flusher) = Some(h);
        if self.flush_due() {
            if let Some(sig) = self.flush_signal.get() {
                sig.wake();
            }
        }
    }

    /// Flushes as [`Database::flush`] does, but holds the writer lock only
    /// to seal the memtables and to install the segments built from them,
    /// so that commits go on while they build: into a new WAL file (the
    /// sealed one, `<wal>.<txn>`, is deleted once the flush is installed)
    /// and into the memtable, which keeps them. A sealed document deleted
    /// or written in place meanwhile is so in its new segment. A flush of
    /// everything meanwhile ([`Database::flush`], a bulk load, dropping a
    /// table) makes this one a no-op. Writes that start while it holds the
    /// lock wait for it rather than fail with `Busy`. It waits for the
    /// writer lock, so it must not be called from a thread that holds a
    /// [`WriteTxn`].
    pub fn background_flush(&self) -> Result<()> {
        self.background_flush_with(|| {})
    }

    /// [`Database::background_flush`], running `between` after the seal and
    /// before the install (tests write there).
    #[doc(hidden)]
    pub fn background_flush_with(&self, between: impl FnOnce()) -> Result<()> {
        let _one = lock(&self.background);
        let t0 = Instant::now();
        let r = self.background_flush_inner(between);
        let now = r.as_ref().err().map(|e| e.to_string());
        let mut before = lock(&self.flush_error);
        // A failing flush is tried again every second: logged when it
        // starts failing (or fails otherwise) and when it works again.
        match (&now, &*before) {
            (None, None) => tracing::debug!(secs = t0.elapsed().as_secs_f64(), "flushed the memtable"),
            (None, Some(_)) => tracing::info!("background flushes work again"),
            (Some(e), b) if b.as_ref() != Some(e) => tracing::warn!(error = %e, "a background flush failed; writes go on into the write-ahead log"),
            _ => {}
        }
        *before = now;
        r
    }

    /// The writer lock, for a background flush or a merge
    /// ([`BackgroundWriter`]).
    fn background_writer(&self) -> BackgroundWriter<'_> {
        self.background_waiting.fetch_add(1, Ordering::AcqRel);
        let wal = lock(&self.writer);
        self.background_holding.fetch_add(1, Ordering::AcqRel);
        self.background_waiting.fetch_sub(1, Ordering::AcqRel);
        BackgroundWriter { wal: Some(wal), holding: &self.background_holding }
    }

    fn background_flush_inner(&self, between: impl FnOnce()) -> Result<()> {
        let (sealed, txn) = {
            let mut wal = self.background_writer();
            self.check_poisoned()?;
            let st = lock(&self.state);
            if st.mem_len() == 0 && st.pending.is_empty() && st.pending_updates.is_empty() {
                return Ok(());
            }
            let numbered = st.manifest.seq_no.is_some();
            let sealed: Vec<SealedMem> = st.tables.iter().filter(|t| t.mem.len > 0).map(|t| SealedMem::of(t, numbered)).collect();
            let txn = st.txn;
            drop(st);
            // A log with no commits since the last seal stays as it is.
            if !wal.is_empty() {
                wal.sync_appends()?;
                let to = wal::sealed_path(&self.wal_path, txn);
                io::rename(&self.wal_path, &to)?;
                lock(&self.sealed_wals).push(txn);
                // Commits must not go on in the sealed file: the install
                // deletes it.
                wal.restart(&self.wal_path, self.opts.sync).and_then(|_| Ok(io::sync_dir(self.wal_dir())?)).map_err(|e| self.poison(e))?;
            }
            (sealed, txn)
        };
        between();
        let max = self.opts.segment_max_docs.max(1);
        let mut built = Vec::with_capacity(sealed.len());
        for s in &sealed {
            let (live, seqs, docids) = s.live();
            let images = live
                .chunks(max)
                .zip(seqs.chunks(max))
                .map(|(c, seqs)| Ok((rowseq::build(&s.schema, c, s.numbered.then_some(seqs), &self.opts.build)?, c.len() as u32)))
                .collect::<Result<Vec<_>>>()?;
            built.push((images, docids));
        }

        let mut wal = self.background_writer();
        self.check_poisoned()?;
        let (carried, pending, pending_updates) = {
            let st = lock(&self.state);
            // A flush of everything since the seal holds what this one built.
            if st.manifest.wal_txn >= txn || sealed.iter().any(|s| st.table_index(s.table).is_some_and(|i| st.tables[i].mem.gen != s.gen)) {
                return Ok(());
            }
            let carried: Vec<_> = sealed.iter().zip(&built).map(|(s, (images, docids))| s.since(&st, docids, images.len(), max)).collect();
            (carried, st.pending.clone(), st.pending_updates.clone())
        };
        let mut vw = VersionWriter::begin(self);
        vw.manifest.wal_txn = txn;
        for ((s, (images, _)), carried) in sealed.iter().zip(&built).zip(carried) {
            let Some(carried) = carried else { continue };
            for ((image, n), (deleted, updated)) in images.iter().zip(carried) {
                let dels = (!deleted.is_empty()).then(|| LiveDocs::with_deleted(None, *n, deleted.iter().copied()));
                vw.add_segment(s.table, None, image, *n, dels.as_ref(), Some(&updated).filter(|u| !u.is_empty()))?;
            }
        }
        for (&id, docs) in &pending {
            vw.add_deletes(id, docs)?;
        }
        for (&id, (_, ups)) in &pending_updates {
            vw.add_updates(id, ups)?;
        }
        let c = vw.commit(&mut wal)?;
        self.publish(c, |st| {
            for s in &sealed {
                if let Some(i) = st.table_index(s.table) {
                    st.tables[i].split_mem(s.len);
                }
            }
            st.pending.clear();
            st.pending_updates.clear();
            st.updates_cache.clear();
            st.n_pending = 0;
        })?;
        drop(wal);
        self.remove_sealed(txn);
        Ok(())
    }

    /// The directory that holds the WAL.
    fn wal_dir(&self) -> &Path {
        self.wal_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
    }

    /// Loads documents into the first table straight into segments of
    /// `segment_max_docs`, in one commit and without the WAL. Keys must be
    /// new (as with `insert`).
    pub fn bulk_load(&self, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        self.try_bulk_load(docs.into_iter().map(Ok))
    }

    /// [`Database::bulk_load`] of documents that may fail to arrive: the
    /// first error ends the load, and nothing is committed.
    pub fn try_bulk_load(&self, docs: impl IntoIterator<Item = Result<Document>>) -> Result<()> {
        self.bulk_load_table(None, docs)
    }

    /// [`Database::bulk_load`] into the table named `table`.
    pub fn bulk_load_into(&self, table: &str, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        self.bulk_load_table(Some(table), docs.into_iter().map(Ok))
    }

    /// [`Database::try_bulk_load`] into the table named `table`.
    pub fn try_bulk_load_into(&self, table: &str, docs: impl IntoIterator<Item = Result<Document>>) -> Result<()> {
        self.bulk_load_table(Some(table), docs)
    }

    fn bulk_load_table(&self, table: Option<&str>, docs: impl IntoIterator<Item = Result<Document>>) -> Result<()> {
        let mut wal = self.try_writer()?;
        let t = self.table_ref(table)?;
        // WAL deletes are replayed against the on-disk version; segments
        // written behind the WAL's back must not hold keys it touches.
        self.flush_locked(&mut wal)?;
        let mut vw = VersionWriter::begin(self);
        let schema = &t.schema;
        let pk = schema.primary_key;
        let check_existing = !lock(&self.state).tables[t.pos].handles.is_empty();
        let mut seen = HashSet::new();
        let mut chunk = Vec::new();
        let max = self.opts.segment_max_docs.max(1);
        let mut ties = Ties::default();
        for doc in docs {
            let mut doc = doc?;
            conform(schema, &mut doc)?;
            self.fill_sids(&t, &mut doc, &mut ties, &[])?;
            if let Some(pk) = pk {
                let dup = !seen.insert(Key::of(&doc[pk]).expect("validated key"))
                    || (check_existing && lock(&self.state).find_live(t.pos, &doc[pk]).is_some());
                if dup {
                    return Err(Error::Schema(format!("duplicate primary key {:?}", doc[pk])));
                }
            }
            chunk.push(doc);
            if chunk.len() == max {
                self.add_loaded(&mut vw, &t, std::mem::take(&mut chunk))?;
            }
        }
        if !chunk.is_empty() {
            self.add_loaded(&mut vw, &t, chunk)?;
        }
        if vw.segments_added == 0 {
            return Ok(());
        }
        let c = vw.commit(&mut wal)?;
        self.publish(c, |st| {
            if let Some(known) = st.table_index(t.id).and_then(|i| st.tables[i].ties.as_mut()) {
                known.absorb(&ties);
            }
        })
    }

    /// Adds a segment of bulk-loaded rows, ordered by the table's
    /// cluster_by field. Rows past its retention stay until a merge.
    fn add_loaded(&self, vw: &mut VersionWriter, t: &TableRef, chunk: Vec<Document>) -> Result<()> {
        let n = chunk.len();
        let (chunk, _) = arrange(&t.schema, chunk, vec![(); n], None);
        let image = build_segment(&t.schema, &chunk, &self.opts.build)?;
        vw.add_segment(t.id, None, &image, n as u32, None, None)
    }

    /// Writes segment `id`'s image to `dest` (a new file), for another
    /// database to attach ([`Database::attach_segments`]): the same bytes in
    /// either layout. An image carries no delete set, so a segment with
    /// deleted documents must be merged first.
    pub fn export_segment(&self, id: u64, dest: &Path) -> Result<()> {
        let (h, pending) = {
            let st = lock(&self.state);
            let h = st.handle(id).cloned().ok_or_else(|| Error::Schema(format!("no segment {id}")))?;
            (h, st.pending.get(&id).map_or(0, |p| p.len()) + st.pending_updates.get(&id).map_or(0, |p| p.1.len()))
        };
        if h.meta.n_deleted > 0 || h.updates.is_some() || pending > 0 {
            return Err(Error::Schema(format!("segment {id} has deleted or updated documents; merge it first")));
        }
        let image = if self.dir.is_some() { h.map.clone() } else { h.map.slice(h.meta.extent.range()) };
        let f = OpenOptions::new().write(true).create_new(true).open(dest)?;
        io::write_at(&f, &image, 0)?;
        io::sync(&f)?;
        Ok(())
    }

    /// Adds segment images built elsewhere (by [`Database::export_segment`]
    /// or [`crate::segment::build::build_segment`] with the first table's
    /// schema) to the first table in one commit, after flushing the
    /// memtables. Their primary keys must be new. Their rows take no
    /// sequence number: they have number 0, whatever an exported image
    /// held ([`rowseq`]). Returns the new segments' ids.
    pub fn attach_segments(&self, images: &[PathBuf]) -> Result<Vec<u64>> {
        let mut wal = self.try_writer()?;
        // As for bulk loads: WAL deletes are replayed against the on-disk
        // version, so segments added behind the WAL's back must follow it.
        self.flush_locked(&mut wal)?;
        let pk = self.schema.primary_key;
        let mut seen = HashSet::new();
        let mut loaded = Vec::with_capacity(images.len());
        for path in images {
            let bytes = std::fs::read(path)?;
            let file = SegmentFile::open(OwnedBytes::new(bytes.clone()))
                .and_then(|f| f.verify_sections().map(|()| f))
                .map_err(|e| Error::Schema(format!("{}: {e}", path.display())))?;
            // Written for the table as it is: its fields, by the same ids.
            let ids = (!self.schema.is_legacy()).then(|| self.schema.fields.iter().map(|f| f.id).collect::<Vec<_>>());
            if file.field_ids()? != ids {
                return Err(Error::Schema(format!("{}: was written for other fields than table {}'s (the field ids differ)", path.display(), self.schema.table)));
            }
            let keys = crate::storage::pk::PkIndex::open(&file)?;
            if pk.is_some() != keys.is_some() {
                return Err(Error::Schema(format!("{}: primary key index does not match the table", path.display())));
            }
            if let Some(keys) = keys {
                let st = lock(&self.state);
                for d in 0..file.n_docs() as DocId {
                    let k = keys.key(d);
                    if !seen.insert(Key::of(&k).ok_or_else(|| corrupt("bad key"))?) || st.find_live(0, &k).is_some() {
                        return Err(Error::Schema(format!("{}: duplicate primary key {k:?}", path.display())));
                    }
                }
            }
            // Its rows have number 0 here, like loaded ones.
            let bytes = rowseq::without(&file)?.unwrap_or(bytes);
            loaded.push((bytes, file.n_docs() as u32));
        }
        let mut vw = VersionWriter::begin(self);
        let first = vw.manifest.next_segment_id;
        for (bytes, n) in &loaded {
            vw.add_segment(0, None, bytes, *n, None, None)?;
        }
        let ids: Vec<u64> = (first..vw.manifest.next_segment_id).collect();
        if ids.is_empty() {
            return Ok(ids);
        }
        let c = vw.commit(&mut wal)?;
        // Their semantic IDs are counted again when a write needs them.
        self.publish(c, |st| st.tables[0].ties = None)?;
        Ok(ids)
    }

    /// Merges each table's on-disk segments as `policy` selects until it
    /// selects nothing. Inputs are rebuilt from `DbOptions::doc_source`
    /// and the merged image is written without blocking writers; the
    /// install, under the writer lock, marks documents deleted in the
    /// meantime as deleted in the merged segment. Merges running at once
    /// (other calls, the threads of [`Database::spawn_merger`]) take
    /// different segments: this one leaves those another merge holds
    /// alone, and returns when the policy selects nothing among the rest.
    /// Segments of a merge a crash interrupted are merged in smaller
    /// groups ([`MergeRecord`]). Must not be called by a thread that holds
    /// a [`WriteTxn`] (the install waits for the writer lock).
    pub fn merge(&self, policy: &MergePolicy) -> Result<MergeStats> {
        self.merge_with(policy, || true)
    }

    /// [`Database::merge`], calling `between` once each merged image is
    /// written and before it is installed; the merge is given up when it
    /// returns false (tests write, check and fail there).
    #[doc(hidden)]
    pub fn merge_with(&self, policy: &MergePolicy, mut between: impl FnMut() -> bool) -> Result<MergeStats> {
        let source = self.opts.doc_source.clone().ok_or_else(|| Error::Schema("merging needs DbOptions::doc_source".into()))?;
        // A database that cannot commit would build the merged segment
        // only to throw it away.
        self.check_poisoned()?;
        let t0 = Instant::now();
        let mut stats = MergeStats::default();
        let pacer = self.merger.get().and_then(|m| m.pacer.as_ref());
        let tables: Vec<(u32, usize)> = lock(&self.state).tables.iter().map(|t| (t.id, t.handles.len())).collect();
        for (table, n) in tables {
            // Each merge removes a segment, except rewrites of a single
            // segment to drop deletes; the bound stops those from repeating
            // forever under a concurrent stream of deletes.
            let mut rounds = 2 * n + 1;
            while rounds > 0 {
                rounds -= 1;
                let Some(claim) = self.claim_merge(table, policy, None) else { break };
                self.merge_group(&claim, source.as_ref(), pacer, &mut between, &mut stats)?;
            }
        }
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }

    /// Starts `opts.threads` threads that merge segments as `opts.policy`
    /// selects, in place of calls to [`Database::merge`]: whenever a commit
    /// publishes a version (a flush, a load, a merge, deletes written) and
    /// every 5 s. Up to that many merges run at once, on different
    /// segments, within `opts.memory_budget`, writing no faster than
    /// `opts.write_bytes_per_sec` (calls to [`Database::merge`] keep to that
    /// rate too). [`Database::merge_status`] says what they do. The threads
    /// end with the database; calling this again does nothing.
    pub fn spawn_merger(self: &Arc<Self>, opts: MergerOptions) {
        let m = Arc::new(Merger {
            policy: opts.policy,
            budget: opts.memory_budget.or_else(|| memory_limit().map(|l| l / 4)),
            pacer: opts.write_bytes_per_sec.map(Pacer::new),
            wake: Mutex::new(MergerWake::default()),
            cv: Condvar::new(),
            next_table: AtomicUsize::new(0),
        });
        if self.merger.set(m.clone()).is_err() {
            return;
        }
        if self.opts.doc_source.is_none() {
            lock(&self.merging).error = Some("merging needs DbOptions::doc_source".into());
            return;
        }
        let mut threads = lock(&self.merger_threads);
        for i in 0..opts.threads.max(1) {
            let (db, m) = (Arc::downgrade(self), m.clone());
            threads.push(std::thread::Builder::new().name(format!("strata-merger-{i}")).spawn(move || merger(db, m)).expect("spawn a merger thread"));
        }
    }

    /// What the merges are doing.
    pub fn merge_status(&self) -> MergeStatus {
        let m = lock(&self.merging);
        MergeStatus {
            running: m.record.running.len(),
            segments_merging: m.record.running.iter().map(|e| e.segments.len()).sum(),
            memory: m.memory(),
            memory_budget: self.merger.get().and_then(|x| x.budget),
            merges: m.merges,
            merges_limited: m.limited,
            interrupted: m.record.interrupted.len(),
            error: m.error.clone(),
        }
    }

    /// The next merge for a merger thread: of the tables in turn, the first
    /// one with segments to merge within the budget.
    fn next_merge(&self, m: &Merger) -> Option<MergeClaim<'_>> {
        if self.is_poisoned() {
            return None;
        }
        let tables: Vec<u32> = lock(&self.state).tables.iter().map(|t| t.id).collect();
        let first = m.next_table.fetch_add(1, Ordering::Relaxed);
        (0..tables.len()).find_map(|i| self.claim_merge(tables[(first + i) % tables.len()], &m.policy, m.budget))
    }

    /// Claims the next group of `table`'s segments that `policy` selects
    /// among those no running merge holds, cut to fit what is left of
    /// `budget` and the record of interrupted merges ([`MergeRecord`]).
    /// The merge is recorded as running until the claim is dropped.
    fn claim_merge(&self, table: u32, policy: &MergePolicy, budget: Option<u64>) -> Option<MergeClaim<'_>> {
        // The snapshot is taken under the lock that claims are given up
        // under: a merge installed (and its claim given up) after a snapshot
        // taken before the lock would leave its inputs, gone by then, to
        // this one.
        let mut m = lock(&self.merging);
        let snap = self.snapshot_inner(false).table_by_id(table)?;
        let bytes: HashMap<u64, u64> = lock(&self.state).handles().map(|h| (h.meta.id, h.meta.extent.len)).collect();
        // Rows past the table's retention count as deleted: a segment of
        // them only is dropped, and one with many is rewritten without them.
        let retention = snap.schema().retention(crate::timestamp::now());
        let infos: Vec<SegmentInfo> = snap
            .segments()
            .iter()
            .map(|v| {
                let expired = retention.map_or(0, |(f, cutoff)| self.expired_in(v, &snap.schema().fields[f].key(), cutoff));
                SegmentInfo { id: v.id, n_docs: v.file.n_docs() as u32, n_live: v.n_live.saturating_sub(expired), bytes: bytes.get(&v.id).copied().unwrap_or(0) }
            })
            .collect();
        let cap = |g: &[u64]| match (m.record.cap(g), budget) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let threads = build_threads(&self.opts.build).unwrap_or(1);
        let (group, cut) = policy.select_within(&infos, |id| m.claimed(id), cap, threads);
        let mut group = group?;
        let memory = merge_memory(&infos.iter().filter(|s| group.contains(&s.id)).copied().collect::<Vec<_>>(), threads);
        // It fits the budget alone: it waits for running merges to finish.
        if budget.is_some_and(|b| m.memory() + memory > b) {
            return None;
        }
        group.sort_by_key(|id| infos.iter().position(|s| s.id == *id));
        m.record.running.push(MergeEntry { segments: group.clone(), memory, times: 0 });
        m.limited += cut as u64;
        let _ = m.record.store(&self.merge_record_path);
        Some(MergeClaim { db: self, table, snap, ids: group, cut, retention })
    }

    /// The live rows of segment `v` whose TIMESTAMP `field` is before
    /// `cutoff` ([`Schema::with_retain`]). The merger asks every few
    /// seconds, so a segment whose rows are partly past it is counted once
    /// per delete set and minute; one whose rows all are, or none, is told
    /// by its column's least and greatest times at once.
    fn expired_in(&self, v: &SegmentView, field: &str, cutoff: i64) -> u32 {
        // A column that cannot be read counts nothing: the merge reads it
        // again, and fails there.
        let Ok(c) = crate::forward::NumericColumn::open(&v.file, field) else { return 0 };
        let (Some(lo), Some(hi)) = (c.min(), c.max()) else { return 0 };
        if lo as i64 >= cutoff {
            return 0;
        }
        let key = (v.n_live, cutoff.div_euclid(60_000_000), (hi as i64) < cutoff);
        let mut cache = lock(&self.expired);
        if let Some(&(k, n)) = cache.get(&v.id) {
            if k == key {
                return n;
            }
        }
        let docs = c.scan_keys(c.int_key_range(i64::MIN, cutoff - 1));
        let n = docs.iter().filter(|&d| v.live.as_ref().is_none_or(|l| l.contains(d))).count() as u32;
        if cache.len() >= 4096 {
            // Mostly segments merged away since.
            cache.clear();
        }
        cache.insert(v.id, (key, n));
        n
    }

    /// Runs a merge a merger thread claimed.
    fn merge_claimed(&self, claim: &MergeClaim, pacer: Option<&Pacer>, stats: &mut MergeStats) -> Result<()> {
        let source = self.opts.doc_source.clone().ok_or_else(|| Error::Schema("merging needs DbOptions::doc_source".into()))?;
        self.check_poisoned()?;
        self.merge_group(claim, source.as_ref(), pacer, &mut || true, stats)
    }

    fn merge_group(&self, claim: &MergeClaim, source: &dyn DocumentSource, pacer: Option<&Pacer>, between: &mut dyn FnMut() -> bool, stats: &mut MergeStats) -> Result<()> {
        let (snap, ids) = (&claim.snap, &claim.ids[..]);
        let schema = snap.schema();
        let mut views = Vec::with_capacity(ids.len());
        let mut inputs: Vec<(u64, Vec<DocId>)> = Vec::new();
        for &id in ids {
            let v = snap.segments().iter().find(|v| v.id == id).expect("selected from this snapshot");
            // The merged segment gets a checksum of its own, so a damaged
            // input must fail here rather than become a valid segment.
            let h = lock(&self.state).handle(id).cloned().ok_or_else(|| corrupt("merge input vanished"))?;
            h.verify(self.dir.as_deref())?;
            let n = v.file.n_docs() as DocId;
            let mut live: Vec<DocId> = (0..n).filter(|&d| v.live.as_ref().is_none_or(|l| l.contains(d))).collect();
            stats.docs_dropped += (n as usize - live.len()) as u64;
            // Rows past the table's retention are dropped as deleted ones
            // are (their time is not changed in place).
            if let Some((f, cutoff)) = claim.retention {
                let times = crate::forward::NumericColumn::open(&v.file, &schema.fields[f].key())?;
                let before = live.len();
                live.retain(|&d| times.get_i64(d).is_none_or(|t| t >= cutoff));
                stats.docs_expired += (before - live.len()) as u64;
            }
            views.push(v);
            inputs.push((id, live));
        }
        // The merged segment's documents in order, as (input, docID): the
        // inputs concatenated, oldest first, or sorted by the table's
        // cluster_by field, rows of one value in that order.
        let mut order: Vec<(u32, DocId)> = inputs.iter().enumerate().flat_map(|(k, (_, live))| live.iter().map(move |&d| (k as u32, d))).collect();
        let clustered = match schema.cluster_by_field() {
            None => false,
            Some(f) => {
                let keys = cluster_keys(schema, f, &views, &order)?;
                let mut by: Vec<u32> = (0..order.len() as u32).collect();
                by.sort_by_key(|&i| keys[i as usize]);
                order = by.into_iter().map(|i| order[i as usize]).collect();
                true
            }
        };
        let n_new = order.len();
        // Batches of about MERGE_BATCH_BYTES of the inputs' images, each
        // decoded while the segment builds from the one before, and the
        // image streamed to its reserved space: the merge never holds every
        // document decoded, nor the whole image.
        let per_doc: Vec<usize> = views.iter().map(|v| (v.file.image().len() / v.file.n_docs().max(1)).max(1)).collect();
        let mut batches: Vec<&[(u32, DocId)]> = Vec::new();
        let (mut from, mut docs, mut bytes) = (0usize, 0usize, 0usize);
        for (i, &(k, _)) in order.iter().enumerate() {
            (docs, bytes) = (docs + 1, bytes + per_doc[k as usize]);
            if docs >= MERGE_BATCH_DOCS.end || (docs >= MERGE_BATCH_DOCS.start && bytes >= MERGE_BATCH_BYTES) {
                batches.push(&order[from..=i]);
                (from, docs, bytes) = (i + 1, 0, 0);
            }
        }
        if from < order.len() {
            batches.push(&order[from..]);
        }
        // Semantic-ID codes the merge drops (its inputs' deleted and expired
        // documents' and those they had dropped) stay in a section of the merged
        // segment, unless a document it keeps has a higher one.
        let sids = !crate::sid::fields(schema).is_empty();
        let mut dropped = Ties::default();
        if sids {
            for v in &views {
                dropped.scan(schema, &v.file)?;
            }
        }
        let kept = Mutex::new(Ties::default());
        let decode = |batch: &[(u32, DocId)]| -> Result<Vec<Document>> {
            // Each input's documents of the batch, read in docID order.
            let mut wanted: Vec<Vec<(DocId, usize)>> = vec![Vec::new(); views.len()];
            for (i, &(k, d)) in batch.iter().enumerate() {
                wanted[k as usize].push((d, i));
            }
            let mut out: Vec<Document> = vec![Vec::new(); batch.len()];
            for (k, mut want) in wanted.into_iter().enumerate() {
                if want.is_empty() {
                    continue;
                }
                want.sort_unstable();
                let v = views[k];
                let live: Vec<DocId> = want.iter().map(|w| w.0).collect();
                let got = source.documents(schema, &v.file, &live)?;
                if got.len() != live.len() {
                    return Err(corrupt(format!("document source returned {} of {} documents", got.len(), live.len())));
                }
                for (mut doc, (d, i)) in got.into_iter().zip(want) {
                    // The merged segment holds the values written in place.
                    if let Some(u) = &v.updates {
                        u.apply(d, &mut doc);
                    }
                    out[i] = doc;
                }
            }
            if sids {
                let mut k = lock(&kept);
                for d in &out {
                    k.saw_doc(schema, d);
                }
            }
            Ok(out)
        };
        // The merged vector graphs start from the largest input's and join
        // the other inputs' graphs to it, of the inputs that lost no
        // documents and keep their order.
        let mut starts = Vec::new();
        let mut base = 0u32;
        let mut whole: Vec<(usize, &SegmentFile, u32)> = Vec::new();
        for (k, (_, live)) in inputs.iter().enumerate() {
            let v = views[k];
            if !clustered && live.len() == v.file.n_docs() {
                whole.push((live.len(), &v.file, base));
            }
            base += live.len() as u32;
        }
        // Largest first (the first of equals): its graph keeps its links.
        whole.sort_by_key(|&(n, _, base)| (std::cmp::Reverse(n), base));
        for (i, f) in schema.fields.iter().enumerate() {
            // A multi-vector field numbers its vectors apart from the rows,
            // so its graph is built anew.
            if matches!(f.ty, FieldType::Vector { .. }) && !f.multi {
                starts.extend(whole.iter().map(|&(_, seg, base)| GraphStart { field: i, segment: seg.clone(), base }));
            }
        }
        // The merged documents' sequence numbers, as the snapshot has them;
        // values written in place since come with their numbers.
        let seqs = match lock(&self.state).manifest.seq_no.is_some() {
            false => None,
            true => {
                let built = views.iter().map(|v| Seqs::of(&v.file)).collect::<Result<Vec<_>>>()?;
                let seq = |&(k, d): &(u32, DocId)| match views[k as usize].updates.as_ref().and_then(|u| u.get(SEQ, d)) {
                    Some(x) => x.as_i64().unwrap_or(0) as u64,
                    None => built[k as usize].as_ref().map_or(0, |b| b.get(d)),
                };
                Some(order.iter().map(seq).collect::<Vec<u64>>())
            }
        };
        // Where each input's live documents went, when not concatenated.
        let slots: Option<Vec<Vec<u32>>> = clustered.then(|| {
            let mut slots: Vec<Vec<u32>> = inputs.iter().map(|(_, live)| vec![0; live.len()]).collect();
            for (at, &(k, d)) in order.iter().enumerate() {
                let live = &inputs[k as usize].1;
                slots[k as usize][live.binary_search(&d).expect("an input's live document")] = at as u32;
            }
            slots
        });
        // Written before the writer lock is taken, into space reserved for
        // it: commits go on meanwhile.
        let gone = |sections: &mut SegmentWriter| {
            for (name, bytes) in crate::sid::gone_sections(schema, &dropped, &lock(&kept)) {
                sections.put(name, bytes);
            }
        };
        let written = match batches.len() {
            // Every document was deleted; a segment without documents keeps
            // the codes they were given.
            0 if sids && !crate::sid::gone_sections(schema, &dropped, &Ties::default()).is_empty() => {
                let mut sections = SegmentBuild::new(schema, &self.opts.build, 0)?.finish(&[])?;
                if seqs.is_some() {
                    sections.put(rowseq::SECTION, rowseq::encode(&[]));
                }
                gone(&mut sections);
                Some(self.write_reserved(sections, pacer)?)
            }
            0 => None,
            n => {
                let mut build = SegmentBuild::with_starts(schema, &self.opts.build, n_new, &starts)?;
                let mut sections = std::thread::scope(|s| {
                    let mut next = decode(batches[0]);
                    for i in 1..n {
                        let (decode, batch) = (&decode, batches[i]);
                        let ahead = s.spawn(move || decode(batch));
                        let added = next.and_then(|docs| build.add(&docs));
                        next = ahead.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
                        added?;
                    }
                    build.finish(&next?)
                })?;
                if let Some(seqs) = &seqs {
                    sections.put(rowseq::SECTION, rowseq::encode(seqs));
                }
                gone(&mut sections);
                Some(self.write_reserved(sections, pacer)?)
            }
        };
        let n_new = n_new as u32;
        let go = between();
        let mut wal = self.background_writer();
        let r = if go { self.install_merge(&mut wal, claim, written.as_ref(), &inputs, slots.as_deref(), n_new, stats) } else { Ok(false) };
        // A commit that failed before it could reach the disk leaves
        // nothing naming the image; one that may have reached it poisons
        // the database, and then the image stays (opening removes it if no
        // version names it).
        if !matches!(r, Ok(true)) && !self.is_poisoned() {
            if let Some((res, _)) = &written {
                self.unreserve(&wal, res);
            }
        }
        r.map(|_| ())
    }

    /// Replaces a merge's inputs with the merged segment (`written`, of
    /// `n_new` documents; none when every input document was deleted) in
    /// one commit. Returns whether it did: not when the table was dropped
    /// meanwhile.
    #[allow(clippy::too_many_arguments)]
    fn install_merge(&self, wal: &mut Wal, claim: &MergeClaim, written: Option<&(Reserved, u32)>, inputs: &[(u64, Vec<DocId>)], slots: Option<&[Vec<u32>]>, n_new: u32, stats: &mut MergeStats) -> Result<bool> {
        let (table, snap, ids) = (claim.table, &claim.snap, &claim.ids[..]);
        self.check_poisoned()?;
        {
            let st = lock(&self.state);
            if st.table_index(table).is_none() || inputs.iter().any(|(id, _)| st.handle(*id).is_none()) {
                // Dropped while the merge was building, or replaced by a
                // copy of the primary ([`Database::install_copy`]).
                return Ok(false);
            }
        }
        // Documents of the inputs deleted after the snapshot, as docIDs of
        // the merged segment (the inputs concatenated in order, or where
        // `slots` puts them), and values written in place after it.
        let mut deleted = Vec::new();
        let mut updated = Updates::default();
        {
            let mut st = lock(&self.state);
            let mut base = 0u32;
            for (k, (id, live)) in inputs.iter().enumerate() {
                let slot = |j: usize| slots.map_or(base + j as u32, |s| s[k][j]);
                let h = st.handle(*id).cloned().ok_or_else(|| corrupt("merge input vanished"))?;
                deleted.extend(live.iter().enumerate().filter(|&(_, &d)| st.is_deleted(&h, d)).map(|(j, _)| slot(j)));
                let then = snap.segments().iter().find(|v| v.id == *id).and_then(|v| v.updates.clone());
                if let Some(now) = st.updates_of(&h) {
                    for (f, d, v) in now.iter() {
                        if then.as_ref().and_then(|t| t.get(f, d)) != Some(v) {
                            if let Ok(j) = live.binary_search(&d) {
                                updated.set(f, slot(j), v.clone());
                            }
                        }
                    }
                }
                base += live.len() as u32;
            }
        }
        let mut vw = VersionWriter::begin(self);
        // The merged segment takes the place of the oldest input.
        let pos = vw.table_mut(table).segments.iter().position(|s| ids.contains(&s.id)).expect("inputs present");
        for &id in ids {
            vw.remove_segment(id);
        }
        if let Some((r, crc)) = written {
            let dels = (!deleted.is_empty()).then(|| LiveDocs::with_deleted(None, n_new, deleted.iter().copied()));
            vw.add_reserved_segment(table, Some(pos), r, *crc, n_new, dels.as_ref(), Some(&updated).filter(|u| !u.is_empty()))?;
        }
        let c = vw.commit(wal)?;
        stats.bytes_written += c.bytes_written + written.map_or(0, |(r, _)| r.extent.len);
        // Pending deletes of the inputs are now either dropped documents or
        // stored in the merged segment's delete set.
        self.publish(c, |st| {
            for id in ids {
                if let Some(p) = st.pending.remove(id) {
                    st.n_pending -= p.len();
                }
                if let Some((_, u)) = st.pending_updates.remove(id) {
                    st.n_pending -= u.len();
                }
                st.updates_cache.remove(id);
            }
            if let Some((r, _)) = written {
                st.reserved.retain(|x| x.id != r.id);
            }
        })?;
        stats.merges += 1;
        stats.merges_limited += claim.cut as usize;
        stats.segments_in += ids.len();
        stats.docs_merged += (n_new as usize - deleted.len()) as u64;
        lock(&self.merging).merges += 1;
        Ok(true)
    }

    /// Reserves space for the merged image of `sections`, streams it there
    /// (each section released once written) and syncs it, at the pace
    /// `pacer` allows; returns the reservation and the image's checksum.
    /// Takes the writer lock only to reserve.
    fn write_reserved(&self, sections: SegmentWriter, pacer: Option<&Pacer>) -> Result<(Reserved, u32)> {
        let r = self.reserve(sections.len() as u64)?;
        let written = (|| -> Result<u32> {
            let (f, base) = match (&self.dir, &r.file) {
                (Some(d), Some(name)) => (Some(OpenOptions::new().write(true).create_new(true).open(d.join(name))?), 0),
                _ => (None, r.extent.page * PAGE),
            };
            let f = f.as_ref().unwrap_or(&self.file);
            let mut out = ImageOut { file: f, at: base, buf: Vec::with_capacity(IMAGE_CHUNK), pacer, crc: crc32fast::Hasher::new() };
            sections.write_to(&mut out)?;
            out.drain()?;
            io::sync(f)?;
            Ok(out.crc.finalize())
        })();
        match written {
            Ok(crc) => Ok((r, crc)),
            Err(e) => {
                let wal = self.background_writer();
                self.unreserve(&wal, &r);
                Err(e)
            }
        }
    }

    /// A segment id and room for an image of `len` bytes ([`Reserved`]).
    fn reserve(&self, len: u64) -> Result<Reserved> {
        let _wal = self.background_writer();
        self.check_poisoned()?;
        let mut st = lock(&self.state);
        let id = st.next_segment_id.max(st.manifest.next_segment_id);
        let r = match &self.dir {
            Some(_) => Reserved { id, extent: Extent { page: 0, len }, file: Some(dir::segment_name(id)) },
            None => {
                let pages = pages_for(len as usize);
                let page = st.freelist.alloc(pages);
                if st.freelist.page_count() > st.file_pages {
                    if let Err(e) = io::set_len(&self.file, st.freelist.page_count() * PAGE) {
                        st.freelist.free(page, pages);
                        let floor = st.meta.page_count;
                        st.freelist.trim_tail_to(floor);
                        return Err(e.into());
                    }
                    st.file_pages = st.freelist.page_count();
                }
                Reserved { id, extent: Extent { page, len }, file: None }
            }
        };
        st.next_segment_id = id + 1;
        st.reserved.push(r.clone());
        Ok(r)
    }

    /// Gives back a reservation that will not be installed. The caller
    /// holds the writer lock (`_wal`), so no version is being built from
    /// the freelist meanwhile.
    fn unreserve(&self, _wal: &Wal, r: &Reserved) {
        {
            let mut st = lock(&self.state);
            st.reserved.retain(|x| x.id != r.id);
            if self.dir.is_none() {
                st.freelist.free(r.extent.page, r.extent.pages());
                // Pages the reservation added to the end of the file are
                // not in the version on disk.
                let floor = st.meta.page_count;
                st.freelist.trim_tail_to(floor);
            }
        }
        if let (Some(d), Some(f)) = (&self.dir, &r.file) {
            let _ = io::remove_file(&d.join(f));
        }
    }

    /// Writes a consistent copy of the database to `dest`, which must not
    /// exist: a backup that is itself a database, opened like any other.
    ///
    /// Under the writer lock the memtables are flushed, so that the version
    /// on disk holds every committed transaction, and that version is
    /// pinned; then the lock is released and the pinned segments, delete
    /// sets and in-place updates are copied while writes and merges go on.
    /// Copying a database's files instead is not safe while it is open: a
    /// commit reuses pages an older version freed, and the directory layout
    /// unlinks files a version stops naming.
    ///
    /// The copy holds only live data (no free pages, no WAL), keeps every
    /// table and segment id and the commit number, and is written under a
    /// temporary name, checked against its checksums ([`Database::check`])
    /// and only then renamed to `dest`, so a failed checkpoint leaves no
    /// `dest` behind.
    pub fn checkpoint(&self, dest: &Path, opts: &CheckpointOptions) -> Result<CheckpointStats> {
        let t0 = Instant::now();
        if dest.exists() {
            return Err(Error::Io(std::io::Error::new(std::io::ErrorKind::AlreadyExists, format!("{} exists", dest.display()))));
        }
        let link = match &opts.link_from {
            Some(prev) if opts.layout == Layout::Directory => {
                let (_, m) = dir::read_current(prev)?;
                let segs: HashMap<String, (u32, u64)> = m.tables.iter().flat_map(|t| &t.segments).map(|s| (dir::segment_name(s.id), (s.crc, s.extent.len))).collect();
                Some((prev.clone(), segs))
            }
            Some(_) => return Err(Error::Schema("link_from needs a checkpoint in the directory layout".into())),
            None => None,
        };
        let (_snap, manifest, handles) = {
            let mut wal = lock(&self.writer);
            self.check_poisoned()?;
            self.flush_locked(&mut wal)?;
            // Pins the version: no later commit reuses its pages.
            let snap = self.snapshot_inner(false);
            let st = lock(&self.state);
            if !st.pending.is_empty() || !st.pending_updates.is_empty() {
                return Err(Error::Busy("deletes or updates are pending after a flush".into()));
            }
            let mut manifest = st.manifest.clone();
            if let Some(n) = &mut manifest.seq_no {
                n.seq = st.seq;
            }
            // Profiles logged since the last version are in no segment.
            for t in manifest.tables.iter_mut() {
                if let Some(ts) = st.tables.iter().find(|ts| ts.id == t.id) {
                    t.profiles = ts.profiles.to_vec();
                }
            }
            (snap, manifest, st.handles().cloned().collect::<Vec<_>>())
        };
        let tmp = beside(dest, "checkpoint")?;
        let r = self.write_checkpoint(&tmp, opts.layout, &manifest, &handles, link);
        let mut stats = match r {
            Ok(stats) => stats,
            Err(e) => {
                let _ = if tmp.is_dir() { std::fs::remove_dir_all(&tmp) } else { std::fs::remove_file(&tmp) };
                let _ = std::fs::remove_file(wal_path(&tmp));
                return Err(e);
            }
        };
        if opts.layout == Layout::SingleFile {
            // Empty; opening creates it.
            io::remove_file(&wal_path(&tmp))?;
        }
        std::fs::rename(&tmp, dest)?;
        let parent = dest.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        io::sync_dir(parent)?;
        stats.txn = manifest.txn;
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }

    fn write_checkpoint(&self, tmp: &Path, layout: Layout, manifest: &Manifest, handles: &[Arc<SegHandle>], link: Option<(PathBuf, HashMap<String, (u32, u64)>)>) -> Result<CheckpointStats> {
        let o = DbOptions { layout, sync: true, publish_snapshots: false, doc_source: None, ..DbOptions::default() };
        let out = Database::create(tmp, manifest.tables[0].schema.clone(), o)?;
        for t in &manifest.tables[1..] {
            match t.id == SOURCES_ID {
                true => out.ensure_sources()?,
                false => out.create_table(t.schema.clone())?,
            }
        }
        let mut stats = CheckpointStats { tables: manifest.tables.len(), ..CheckpointStats::default() };
        {
            let mut wal = lock(&out.writer);
            let mut vw = VersionWriter::begin(&out);
            vw.link = link;
            // The copy's version takes the source's commit number, which
            // names its manifest and delete sets in the directory layout:
            // two checkpoints' files then share a name only if they hold
            // the same bytes, and can be synced into one place.
            vw.txn = vw.txn.max(manifest.txn);
            // The copy's tables were created in the same order; their ids
            // may differ (dropped tables leave gaps in the source's).
            let ids: Vec<u32> = vw.manifest.tables.iter().map(|t| t.id).collect();
            for (t, &to) in manifest.tables.iter().zip(&ids) {
                vw.table_mut(to).profiles = t.profiles.clone();
                for m in &t.segments {
                    let h = handles.iter().find(|h| h.meta.id == m.id).ok_or_else(|| corrupt(format!("segment {} is in the manifest but not open", m.id)))?;
                    vw.manifest.next_segment_id = m.id;
                    vw.add_segment(to, None, h.file.image(), m.n_docs, h.base.as_deref(), h.updates.as_deref())?;
                    stats.segments += 1;
                    stats.live_docs += (m.n_docs - m.n_deleted) as u64;
                }
            }
            vw.manifest.next_segment_id = manifest.next_segment_id;
            // The copy numbers its writes on from the source's.
            vw.manifest.seq_no = manifest.seq_no.clone();
            stats.segments_linked = vw.segments_linked;
            let c = vw.commit(&mut wal)?;
            stats.bytes_written = c.bytes_written;
            out.publish(c, |_| {})?;
        }
        out.check()?;
        Ok(stats)
    }

    /// Returns space to the file system: moves segments and delete sets from
    /// the end of the file into free runs below them and truncates the free
    /// tail, committing until nothing moves (pages freed by a commit become
    /// reusable two commits later). Space a live snapshot can still reach
    /// stays. Returns the number of bytes the file shrank by.
    pub fn vacuum(&self) -> Result<u64> {
        let mut wal = self.try_writer()?;
        self.check_poisoned()?;
        if let Some(d) = &self.dir {
            // Each segment is its own file, so there is nothing to compact:
            // only files a crash or a failed unlink left behind.
            let keep = {
                let st = lock(&self.state);
                let cur = Current {
                    db_id: st.meta.db_id,
                    txn: st.meta.txn,
                    manifest: dir::manifest_name(st.manifest.txn),
                    crc: st.meta.manifest_crc,
                    format: st.meta.format,
                };
                let mut keep = dir::referenced(&cur, &st.manifest);
                // Merged images being written.
                keep.extend(st.reserved.iter().filter_map(|r| r.file.clone()));
                keep
            };
            let mut freed = 0;
            for (f, len) in dir::orphans(d, &keep)? {
                io::remove_file(&f)?;
                freed += len;
            }
            return Ok(freed);
        }
        let before = lock(&self.state).meta.page_count;
        let mut idle = 0;
        while idle < 2 {
            let mut vw = VersionWriter::begin(self);
            let moved = vw.relocate_down()?;
            let c = vw.commit(&mut wal)?;
            let shrank = c.meta.page_count < lock(&self.state).meta.page_count;
            self.publish(c, |_| {})?;
            idle = if moved > 0 || shrank { 0 } else { idle + 1 };
        }
        Ok(before.saturating_sub(lock(&self.state).meta.page_count) * PAGE)
    }

    /// The size of the WAL, sealed files included.
    fn wal_bytes(&self) -> u64 {
        let size = |p: &Path| std::fs::metadata(p).map_or(0, |m| m.len());
        size(&self.wal_path) + lock(&self.sealed_wals).iter().map(|&t| size(&wal::sealed_path(&self.wal_path, t))).sum::<u64>()
    }

    pub fn file_stats(&self) -> FileStats {
        let st = lock(&self.state);
        if let Some(d) = &self.dir {
            return FileStats {
                file_bytes: dir::size(d),
                page_count: 0,
                live_pages: 0,
                free_pages: 0,
                pending_pages: 0,
                wal_bytes: self.wal_bytes(),
                memtable_docs: st.mem_live_docs(),
                pending_deletes: st.n_pending,
                segments: segment_stats(&st.manifest),
            };
        }
        let segments = segment_stats(&st.manifest);
        let seg_pages: u64 =
            st.manifest.segments().map(|s| s.extent.pages() + s.deletes.map_or(0, |e| e.pages()) + s.updates.map_or(0, |e| e.pages())).sum();
        FileStats {
            file_bytes: self.file.metadata().map_or(st.file_pages * PAGE, |m| m.len()),
            page_count: st.meta.page_count,
            live_pages: META_PAGES + st.meta.manifest.pages() + st.meta.freelist.pages() + seg_pages,
            free_pages: st.freelist.free_pages(),
            pending_pages: st.freelist.pending_pages(),
            wal_bytes: self.wal_bytes(),
            memtable_docs: st.mem_live_docs(),
            pending_deletes: st.n_pending,
            segments,
        }
    }

    /// Verifies the file: the meta slot and manifest on disk match the open
    /// version, every segment and delete set matches its checksum, every
    /// key is found by its segment's index, and every page is used exactly
    /// once (meta slot, extent, free run or pending free). `Error::Busy`
    /// while a write is in progress.
    pub fn check(&self) -> Result<()> {
        let _w = self.try_writer()?;
        if let Some(d) = &self.dir {
            return self.check_dir(d);
        }
        // The writer lock keeps the version still; copy it so readers are
        // not held up while every byte is checksummed.
        let (st_meta, st_slot, st_manifest, freelist, handles, map, reserved) = {
            let st = lock(&self.state);
            let reserved: Vec<Extent> = st.reserved.iter().map(|r| r.extent).collect();
            (st.meta, st.slot, st.manifest.clone(), st.freelist.clone(), st.handles().cloned().collect::<Vec<_>>(), st.map.clone(), reserved)
        };
        // The open version's slot: after opening fell back from a damaged
        // newer version, the other slot names that one until a commit.
        let meta = pager::read_metas(&self.file)?.into_iter().find(|&(_, s)| s == st_slot).ok_or_else(|| corrupt("the open version's meta slot is not valid"))?.0;
        let manifest = Manifest::decode(&pager::read_extent(&self.file, meta.manifest, meta.manifest_crc)?)?;
        // Compared as written: a format-1 manifest does not hold the next
        // table id, which the open one keeps counting.
        if meta != st_meta || manifest.encode() != st_manifest.encode() {
            return Err(corrupt("on-disk version differs from the open one"));
        }
        check_field_lists(&manifest, &handles)?;
        let mut used = vec![false; meta.page_count as usize];
        let mut claim = |what: &str, page: u64, pages: u64| -> Result<()> {
            for p in page..page + pages {
                match used.get_mut(p as usize) {
                    Some(u) if !*u => *u = true,
                    _ => return Err(corrupt(format!("page {p} ({what}) is out of range or used twice"))),
                }
            }
            Ok(())
        };
        claim("meta", 0, META_PAGES)?;
        claim("manifest", meta.manifest.page, meta.manifest.pages())?;
        claim("freelist", meta.freelist.page, meta.freelist.pages())?;
        for (s, h) in manifest.segments().zip(&handles) {
            claim("segment", s.extent.page, s.extent.pages())?;
            if crc32fast::hash(&map[s.extent.range()]) != s.crc {
                return Err(corrupt(format!("segment {} fails its checksum", s.id)));
            }
            if let Some(e) = s.deletes {
                claim("delete set", e.page, e.pages())?;
                if crc32fast::hash(&map[e.range()]) != s.deletes_crc {
                    return Err(corrupt(format!("delete set of segment {} fails its checksum", s.id)));
                }
                let n: u32 = h.base.as_ref().map_or(0, |b| b.deleted_words().iter().map(|w| w.count_ones()).sum());
                if n != s.n_deleted {
                    return Err(corrupt(format!("segment {} has {n} deletes, manifest says {}", s.id, s.n_deleted)));
                }
            }
            if let Some(e) = s.updates {
                claim("updates", e.page, e.pages())?;
                if crc32fast::hash(&map[e.range()]) != s.updates_crc {
                    return Err(corrupt(format!("updates of segment {} fail their checksum", s.id)));
                }
            }
            if let Some(pk) = &h.pk {
                for d in 0..s.n_docs {
                    if !pk.lookup_all(&pk.key(d)).any(|x| x == d) {
                        return Err(corrupt(format!("segment {} key of document {d} not found", s.id)));
                    }
                }
            }
        }
        for (start, len) in freelist.runs() {
            claim("free run", start, len)?;
        }
        for p in freelist.pending() {
            claim("pending free", p.page, p.pages)?;
        }
        // Reserved for merged images being written: the version on disk
        // counts them free, and may end before them.
        for e in reserved {
            let end = (e.page + e.pages()).min(meta.page_count);
            if e.page < end {
                claim("reserved for a merge", e.page, end - e.page)?;
            }
        }
        match used.iter().position(|u| !u) {
            Some(p) => Err(corrupt(format!("page {p} is neither used nor free"))),
            None => Ok(()),
        }
    }

    /// [`Database::check`] in the directory layout: `CURRENT` names the open
    /// version, every file matches its checksum, every key is found, and no
    /// file is left over.
    fn check_dir(&self, d: &Path) -> Result<()> {
        let (manifest, handles, meta, reserved) = {
            let st = lock(&self.state);
            (st.manifest.clone(), st.handles().cloned().collect::<Vec<_>>(), st.meta, st.reserved.iter().filter_map(|r| r.file.clone()).collect::<Vec<_>>())
        };
        let (cur, on_disk) = dir::read_current(d)?;
        if on_disk.encode() != manifest.encode() || cur.txn != meta.txn {
            return Err(corrupt("on-disk version differs from the open one"));
        }
        check_field_lists(&manifest, &handles)?;
        for (s, h) in manifest.segments().zip(&handles) {
            let name = s.file.as_deref().ok_or_else(|| corrupt(format!("segment {} has no file", s.id)))?;
            if crc32fast::hash(&std::fs::read(d.join(name))?) != s.crc {
                return Err(corrupt(format!("{name} fails its checksum")));
            }
            if let Some(df) = &s.deletes_file {
                if crc32fast::hash(&std::fs::read(d.join(df))?) != s.deletes_crc {
                    return Err(corrupt(format!("{df} fails its checksum")));
                }
                let n: u32 = h.base.as_ref().map_or(0, |b| b.deleted_words().iter().map(|w| w.count_ones()).sum());
                if n != s.n_deleted {
                    return Err(corrupt(format!("segment {} has {n} deletes, manifest says {}", s.id, s.n_deleted)));
                }
            }
            if let Some(uf) = &s.updates_file {
                if crc32fast::hash(&std::fs::read(d.join(uf))?) != s.updates_crc {
                    return Err(corrupt(format!("{uf} fails its checksum")));
                }
            }
            if let Some(pk) = &h.pk {
                for doc in 0..s.n_docs {
                    if !pk.lookup_all(&pk.key(doc)).any(|x| x == doc) {
                        return Err(corrupt(format!("segment {} key of document {doc} not found", s.id)));
                    }
                }
            }
        }
        // Merged images being written are named by no version yet.
        let mut keep = dir::referenced(&cur, &manifest);
        keep.extend(reserved);
        match dir::orphans(d, &keep)?.first() {
            Some((f, _)) => Err(corrupt(format!("{} is named by no version", f.display()))),
            None => Ok(()),
        }
    }

    /// The writer lock, or `Busy` when another write holds it. A background
    /// flush holds it only to seal or install, briefly: that is waited for.
    /// While one waits for the lock, so does this, for a few milliseconds,
    /// in case the lock is being handed to the flush.
    fn try_writer(&self) -> Result<MutexGuard<'_, Wal>> {
        let t0 = Instant::now();
        loop {
            match self.writer.try_lock() {
                Ok(g) => return Ok(g),
                Err(TryLockError::Poisoned(p)) => return Ok(p.into_inner()),
                Err(TryLockError::WouldBlock) => {
                    let holding = self.background_holding.load(Ordering::Acquire) > 0;
                    let waiting = self.background_waiting.load(Ordering::Acquire) > 0;
                    if !holding && !(waiting && t0.elapsed() < Duration::from_millis(5)) {
                        return Err(Error::Busy("another write is in progress".into()));
                    }
                }
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    fn check_poisoned(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::Io(std::io::Error::other("an earlier write failed; reopen the database")));
        }
        Ok(())
    }

    fn poison(&self, e: Error) -> Error {
        if !self.poisoned.swap(true, Ordering::AcqRel) {
            tracing::error!(error = %e, "a write failed partway: the database refuses writes until it is reopened");
        }
        e
    }

    /// Makes a committed version the current one.
    fn publish(&self, c: Committed, update: impl FnOnce(&mut State)) -> Result<()> {
        let (old, map) = {
            let st = lock(&self.state);
            (st.handles().cloned().collect::<Vec<_>>(), st.map.clone())
        };
        let grown = self.dir.is_none() && c.meta.page_count * PAGE > map.len() as u64;
        let map = if grown { pager::map_file(&self.file, self.opts.read_ahead).map_err(|e| self.poison(e))? } else { map };
        let hints = self.opts.read_ahead && self.opts.random_hints;
        let check = if self.opts.check_sections { Check::Read } else { Check::Never };
        let handles = c
            .manifest
            .tables
            .iter()
            .map(|t| {
                t.segments
                    .iter()
                    .map(|m| match (old.iter().find(|h| h.meta == *m), &self.dir) {
                        (Some(h), _) => Ok(h.clone()),
                        (None, Some(d)) => SegHandle::open_in(d, m, &t.schema, self.opts.read_ahead, hints, check).map(Arc::new),
                        (None, None) => SegHandle::open(&map, m, &t.schema, self.opts.read_ahead, hints, check).map(Arc::new),
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| self.poison(e))?;
        let obsolete = c.obsolete;
        {
            let mut st = lock(&self.state);
            // Tables keep their memtables; created ones start empty and
            // dropped ones go.
            let mut old_tables = std::mem::take(&mut st.tables);
            st.tables = c
                .manifest
                .tables
                .iter()
                .zip(handles)
                .map(|(t, h)| match old_tables.iter().position(|o| o.id == t.id) {
                    Some(i) => {
                        let mut o = old_tables.swap_remove(i);
                        o.handles = h;
                        // A drop with CASCADE removes references; nothing
                        // else changes a table's schema, and never the
                        // first table's (whose references are its own).
                        if *o.schema != t.schema {
                            debug_assert!(t.id != 0, "the first table's schema changed");
                            o.schema = Arc::new(t.schema.clone());
                        }
                        // The version took them from here, unless it
                        // installs others (a copy's).
                        if *o.profiles != t.profiles {
                            o.profiles = Arc::new(t.profiles.clone());
                        }
                        o
                    }
                    None => {
                        let mut ts = TableState::new(t.id, Arc::new(t.schema.clone()), h, self.opts.memtable_chunk_docs, 0);
                        ts.profiles = Arc::new(t.profiles.clone());
                        ts
                    }
                })
                .collect();
            st.txn = c.meta.txn;
            st.manifest = c.manifest;
            st.meta = c.meta;
            st.slot = c.slot;
            st.freelist = c.fl;
            st.file_pages = c.file_pages;
            st.map = map;
            st.live_cache.clear();
            update(&mut st);
            st.changed();
        }
        self.republish();
        if let Some(m) = self.merger.get() {
            m.wake();
        }
        // Directory layout: files the new version no longer names. Open
        // snapshots keep reading them through their mappings.
        if let Some(d) = &self.dir {
            for f in obsolete {
                let _ = io::remove_file(&d.join(f));
            }
        }
        Ok(())
    }

    /// The table named `name` (the first table for `None`). Callers hold
    /// the writer lock, so it stays where it is until they release it.
    fn table_ref(&self, name: Option<&str>) -> Result<TableRef> {
        let Some(n) = name else { return Ok(self.first_table()) };
        let st = lock(&self.state);
        let pos = st.tables.iter().position(|t| t.schema.table.eq_ignore_ascii_case(n)).ok_or_else(|| Error::Schema(format!("no table {n}")))?;
        let t = &st.tables[pos];
        Ok(TableRef { id: t.id, pos, schema: t.schema.clone() })
    }

    /// The first table: id 0, always first since it cannot be dropped.
    fn first_table(&self) -> TableRef {
        TableRef { id: 0, pos: 0, schema: self.schema.clone() }
    }

    /// Gives semantic-ID field `field` of the table named `table` its
    /// codebook, whose file has hash `hash` ([`crate::sid::Codebook::open`]):
    /// writes then fill the field from the row's vector. It must be the
    /// codebook the field records.
    pub fn set_codebook(&self, table: &str, field: &str, codebook: Arc<Codebook>, hash: &str) -> Result<()> {
        let t = self.table_ref(Some(table))?;
        let f = t.schema.field_index(field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        let Some(sid) = &t.schema.fields[f].semantic_id else { return Err(Error::Schema(format!("field {field} is not a semantic-ID field"))) };
        if sid.hash != hash {
            return Err(Error::Schema(format!("field {field} records codebook {} with hash {}, not {hash}", sid.codebook, sid.hash)));
        }
        let dim = t.schema.field(&sid.from).and_then(|v| if let FieldType::Vector { dim, .. } = v.ty { Some(dim) } else { None });
        if dim != Some(codebook.dim) {
            return Err(Error::Schema(format!("codebook {} codes {}-dimension vectors; {} holds {}", sid.codebook, codebook.dim, sid.from, dim.unwrap_or(0))));
        }
        let mut all = self.codebooks.write().unwrap_or_else(|e| e.into_inner());
        all.retain(|c| !(c.table == t.id && c.field == f));
        all.push(Coded { table: t.id, field: f, hash: hash.to_string(), codebook });
        Ok(())
    }

    /// The codebook of semantic-ID field `field` of the table named `table`,
    /// if it has one ([`Database::set_codebook`]).
    pub fn codebook(&self, table: &str, field: &str) -> Option<Arc<Codebook>> {
        let t = self.table_ref(Some(table)).ok()?;
        self.codebook_of(&t, t.schema.field_index(field)?)
    }

    fn codebook_of(&self, t: &TableRef, f: usize) -> Option<Arc<Codebook>> {
        let hash = &t.schema.fields.get(f)?.semantic_id.as_ref()?.hash;
        let all = self.codebooks.read().unwrap_or_else(|e| e.into_inner());
        all.iter().find(|c| c.table == t.id && c.field == f && c.hash == *hash).map(|c| c.codebook.clone())
    }

    /// Fills the semantic-ID fields of `doc`, a row of table `t` that a
    /// write under the writer lock gives ([`crate::sid`]): a code given in
    /// full is kept, one of the levels alone gets its tie, and a row with a
    /// vector and no code gets one from the codebook. `live` holds the
    /// fields' values in the row an upsert replaces, whose code the row
    /// keeps when its levels are the same; `local` the ties the write gave
    /// so far.
    fn fill_sids(&self, t: &TableRef, doc: &mut Document, local: &mut Ties, live: &[(usize, Value)]) -> Result<()> {
        for (f, vf) in crate::sid::fields(&t.schema) {
            let field = &t.schema.fields[f];
            let cb = self.codebook_of(t, f);
            let levels = match &doc[f] {
                Value::Null => match (&doc[vf], &cb) {
                    (Value::Vector(v), Some(cb)) => cb.code(v),
                    (Value::Vector(_), None) => {
                        let sid = field.semantic_id.as_ref().expect("a semantic-ID field");
                        return Err(Error::Schema(format!("field {}: its codebook {} is not open here, so rows cannot get codes (Db::open_codebooks)", field.name, sid.codebook)));
                    }
                    _ => continue,
                },
                v => {
                    let code = crate::sid::code_of(v).filter(|c| c.len() >= 2).ok_or_else(|| Error::Schema(format!("bad value for {}: {v:?} (a semantic ID is every prefix of a code such as 12.7.201.0)", field.name)))?;
                    match &cb {
                        Some(cb) if code.len() == cb.n_levels() && code.iter().all(|&x| (x as usize) < cb.k) => code,
                        Some(cb) if code.len() == cb.n_levels() + 1 && code[..cb.n_levels()].iter().all(|&x| (x as usize) < cb.k) => {
                            local.saw(f, &code);
                            continue;
                        }
                        Some(cb) => return Err(Error::Schema(format!("bad value for {}: {v:?} (codes have {} levels below {}, and a tie)", field.name, cb.n_levels(), cb.k))),
                        // Without the codebook, a code is taken as given.
                        None if code[..code.len() - 1].iter().all(|&x| x < 256) && code.len() <= crate::sid::MAX_LEVELS + 1 => {
                            local.saw(f, &code);
                            continue;
                        }
                        None => return Err(Error::Schema(format!("bad value for {}: {v:?}", field.name))),
                    }
                }
            };
            let kept = live.iter().find(|(g, _)| *g == f).and_then(|(_, v)| crate::sid::code_of(v).filter(|c| c.len() == levels.len() + 1 && c[..levels.len()] == levels[..]));
            let code = match kept {
                Some(c) => c,
                None => {
                    let p = crate::sid::pack(&levels);
                    let tie = self.committed_tie(t, f, p)?.max(local.get(f, p));
                    levels.iter().copied().chain([tie]).collect()
                }
            };
            local.saw(f, &code);
            doc[f] = crate::sid::value(&code[..code.len() - 1], code[code.len() - 1]);
        }
        Ok(())
    }

    /// The next tie of semantic-ID field `f` of table `t` under prefix `p`
    /// that committed writes leave, counting the table's ties first if no
    /// write has needed them since it opened. Under the writer lock, which
    /// keeps its segments and memtable as they are.
    fn committed_tie(&self, t: &TableRef, f: usize, p: u32) -> Result<u32> {
        let (files, chunks) = {
            let st = lock(&self.state);
            let table = &st.tables[t.pos];
            if let Some(ties) = &table.ties {
                return Ok(ties.get(f, p));
            }
            (table.handles.iter().map(|h| h.file.clone()).collect::<Vec<_>>(), table.mem.chunks.clone())
        };
        let mut ties = Ties::default();
        for file in &files {
            ties.scan(&t.schema, file)?;
        }
        // Deleted memtable rows too: their numbers are not given again.
        for doc in chunks.iter().flat_map(|c| c.iter()) {
            ties.saw_doc(&t.schema, doc);
        }
        let next = ties.get(f, p);
        lock(&self.state).tables[t.pos].ties = Some(ties);
        Ok(next)
    }

    /// INT field `f` of the live document of table `t` with primary key
    /// `key`: the value written in place if any, else the document's.
    fn live_int(&self, t: &TableRef, key: &Value, f: usize) -> Result<Option<i64>> {
        Ok(self.live_value(t, key, f)?.as_i64())
    }

    /// Field `f` of the live document of table `t` with primary key `key`
    /// (NULL without one): a number's value written in place if any, else
    /// the document's (a field not numeric must be stored).
    fn live_value(&self, t: &TableRef, key: &Value, f: usize) -> Result<Value> {
        let st = lock(&self.state);
        let (h, d) = match st.find_live(t.pos, key) {
            None => return Ok(Value::Null),
            Some(Loc::Mem(i)) => {
                let mem = &st.tables[t.pos].mem;
                return Ok(match mem.updates.get(f, i) {
                    Some(v) => v.clone(),
                    None => mem.chunks[i as usize / mem.chunk][i as usize % mem.chunk][f].clone(),
                });
            }
            Some(Loc::Disk(id, d)) => (st.handle(id).cloned().expect("segment of a live key"), d),
        };
        let written = st.pending_updates.get(&h.meta.id).and_then(|(_, u)| u.get(f, d)).or_else(|| h.updates.as_ref().and_then(|u| u.get(f, d)));
        if let Some(v) = written {
            return Ok(v.clone());
        }
        drop(st);
        let field = &t.schema.fields[f];
        Ok(if field.indexed && matches!(field.ty, FieldType::Int | FieldType::Float) {
            crate::forward::NumericColumn::open(&h.file, &field.key())?.get(d)
        } else {
            crate::forward::DocStore::open(&h.file, &t.schema)?.get_fields(d, &[f]).pop().unwrap_or(Value::Null)
        })
    }

    /// Commits `ops` as one transaction; returns its sequence number when
    /// the database numbers its writes.
    fn commit_ops(&self, wal: &mut Wal, ops: Vec<(u32, Op)>) -> Result<Option<u64>> {
        self.commit_logged(wal, ops, self.opts.sync)
    }

    /// [`Database::commit_ops`], logged durably only with `sync`.
    fn commit_logged(&self, wal: &mut Wal, ops: Vec<(u32, Op)>, sync: bool) -> Result<Option<u64>> {
        if ops.is_empty() {
            return Ok(None);
        }
        self.check_poisoned()?;
        if !wal.allows_profiles() && ops.iter().any(|(_, op)| matches!(op, Op::Profile { .. })) {
            self.allow_profiles(wal)?;
        }
        // While flushes fail or the flusher falls behind, the memtable
        // grows; past twice its limit, writes wait for a flush to succeed,
        // refused before they are logged.
        let flusher = self.flush_signal.get();
        if self.over_twice() && (flusher.is_some() || self.flush_error().is_some()) {
            self.flush_locked(wal).map_err(|e| Error::Io(std::io::Error::other(format!("writes wait for a flush to succeed: {e}"))))?;
        }
        let (txn, seq) = {
            let st = lock(&self.state);
            (st.txn + 1, st.manifest.seq_no.as_ref().map(|n| SeqNo { term: n.term, seq: st.seq + 1 }))
        };
        // A failed append is cut off the log, so the transaction is not
        // committed; but the page cache may be in doubt after a failed
        // fsync, so stop writing.
        let logged = wal.append(txn, seq, &ops, sync).map_err(|e| self.poison(e))?;
        let full = {
            let mut st = lock(&self.state);
            for (table, op) in ops {
                // The writer lock kept every table of the transaction.
                let t = st.table_index(table).expect("table of a logged write");
                st.apply(t, op, seq.map_or(0, |n| n.seq));
            }
            st.txn = txn;
            if let Some(n) = seq {
                st.seq = n.seq;
                if st.ship.keeps() {
                    st.ship.push(Shipped { seq: n, bytes: logged.into() });
                }
            }
            st.mem_len() + st.n_pending > self.opts.memtable_max_docs
        };
        let numbered = seq;
        let seq = seq.map(|n| n.seq);
        // From here the transaction is durable, and its result is Ok: an
        // error would make callers retry a write that already happened (an
        // `add` applied twice). A failed flush is kept in `flush_error`
        // and tried again by the next commit.
        if full {
            match flusher {
                Some(sig) => sig.wake(),
                // The flush published its version.
                None if self.flush_locked(wal).is_ok() => {
                    self.committed(numbered);
                    return Ok(seq);
                }
                None => {}
            }
        }
        self.republish();
        self.committed(numbered);
        Ok(seq)
    }

    /// Tells the commit hook ([`Database::on_commit`]) of a numbered
    /// transaction.
    fn committed(&self, seq: Option<SeqNo>) {
        if let (Some(n), Some(f)) = (seq, &*self.commit_hook.read().unwrap_or_else(|e| e.into_inner())) {
            f(n);
        }
    }

    /// The table with id `id`, under the writer lock as [`Database::table_ref`].
    fn table_by_id(&self, id: u32) -> Result<TableRef> {
        let st = lock(&self.state);
        let pos = st.table_index(id).ok_or_else(|| Error::Schema(format!("no table has id {id}")))?;
        Ok(TableRef { id, pos, schema: st.tables[pos].schema.clone() })
    }
}

/// The single write transaction. Operations are buffered and logged at
/// commit; dropping the transaction rolls it back. Its methods write the
/// first table; [`WriteTxn::table`] writes another, in the same
/// transaction.
pub struct WriteTxn<'a> {
    db: &'a Database,
    wal: MutexGuard<'a, Wal>,
    ops: Vec<(u32, Op)>,
    /// What the transaction did to each table it wrote, by table id.
    keys: Vec<(u32, TxnKeys)>,
    /// The semantic-ID ties the transaction gave, by table id.
    ties: Vec<(u32, Ties)>,
}

/// What a transaction did to one table's keys.
#[derive(Default)]
struct TxnKeys {
    /// Keys this transaction made live (true) or deleted (false).
    touched: HashMap<Key, bool>,
    /// Documents this transaction wrote, by key, as they now stand.
    written: HashMap<Key, Arc<Document>>,
    /// Fields this transaction set in place on documents it did not write.
    sets: HashMap<(Key, usize), Value>,
}

/// A table resolved under the writer lock, which keeps it from being
/// dropped or moved until the lock is released.
#[derive(Clone)]
struct TableRef {
    id: u32,
    /// Its position in `State::tables`.
    pos: usize,
    schema: Arc<Schema>,
}

impl TableRef {
    fn primary_key(&self) -> Result<usize> {
        self.schema.primary_key.ok_or_else(|| Error::Schema(format!("table {} has no primary key", self.schema.table)))
    }

    /// Refuses to change field `i` in place when it places the table's
    /// rows on shards ([`crate::schema::Schema::with_shard_by`]): the row
    /// would stay on the shard of its old value. Nor when it orders the
    /// segments' rows or keeps them ([`crate::schema::Schema::with_cluster_by`],
    /// [`crate::schema::Schema::with_retain`]), which merges read from the
    /// segments' columns.
    fn check_not_placing(&self, i: usize) -> Result<()> {
        let s = &self.schema;
        let why = match () {
            _ if s.shard_by_field() == Some(i) => "places the rows on shards (shard_by)",
            _ if s.cluster_by_field() == Some(i) => "orders the rows (cluster_by)",
            _ if s.retention(0).is_some_and(|(f, _)| f == i) => "keeps the rows (retain_by)",
            _ if s.edge_fields().is_some_and(|(a, b, _)| a == i || b == i) => "links the rows (edges)",
            _ => return Ok(()),
        };
        Err(Error::Schema(format!("{} {why} of {}, so it is not changed in place: delete the row and write it again", s.fields[i].name, s.table)))
    }

    /// Checks that `key` has the primary key's type.
    fn key_of(&self, key: &Value) -> Result<Key> {
        let pk = self.primary_key()?;
        match (&self.schema.fields[pk].ty, key) {
            (FieldType::Keyword, Value::Str(_)) | (FieldType::Int, Value::Int(_)) => Ok(Key::of(key).unwrap()),
            _ => Err(Error::Schema(format!("{key:?} is not a value of primary key {}", self.schema.fields[pk].name))),
        }
    }
}

/// Writes to one table of a [`WriteTxn`], from [`WriteTxn::table`].
pub struct TableTxn<'t, 'a> {
    txn: &'t mut WriteTxn<'a>,
    t: TableRef,
}

impl TableTxn<'_, '_> {
    /// [`WriteTxn::insert`] into this table.
    pub fn insert(&mut self, doc: Document) -> Result<()> {
        self.txn.insert_in(&self.t, doc)
    }

    /// [`WriteTxn::upsert`] into this table.
    pub fn upsert(&mut self, doc: Document) -> Result<()> {
        self.txn.upsert_in(&self.t, doc)
    }

    /// [`WriteTxn::upsert_if_newer`] into this table.
    pub fn upsert_if_newer(&mut self, doc: Document, version: &str) -> Result<bool> {
        self.txn.upsert_if_newer_in(&self.t, doc, version)
    }

    /// [`WriteTxn::delete`] from this table.
    pub fn delete(&mut self, key: &Value) -> Result<bool> {
        self.txn.delete_in(&self.t, key)
    }

    /// [`WriteTxn::seq_of`] in this table.
    pub fn seq_of(&self, key: &Value) -> Result<Option<u64>> {
        self.txn.seq_of_in(&self.t, key)
    }

    /// [`WriteTxn::set`] in this table.
    pub fn set(&mut self, key: &Value, field: &str, value: Value) -> Result<bool> {
        self.txn.set_in(&self.t, key, field, value)
    }

    /// [`WriteTxn::add`] in this table.
    pub fn add(&mut self, key: &Value, field: &str, delta: &Value) -> Result<bool> {
        self.txn.add_in(&self.t, key, field, delta)
    }

    /// [`WriteTxn::add_or_insert`] in this table.
    pub fn add_or_insert(&mut self, doc: Document, field: &str, delta: &Value) -> Result<()> {
        self.txn.add_or_insert_in(&self.t, doc, field, delta)
    }

    /// Field `field` of the live row with this key, as this transaction
    /// finds it, its own writes included; `None` when no row with the key
    /// is live.
    pub fn value(&mut self, key: &Value, field: &str) -> Result<Option<Value>> {
        self.txn.value_in(&self.t, key, field)
    }

    pub fn schema(&self) -> &Schema {
        &self.t.schema
    }
}

impl<'a> WriteTxn<'a> {
    /// Writes to the table named `name` (any case) in this transaction.
    pub fn table(&mut self, name: &str) -> Result<TableTxn<'_, 'a>> {
        let t = self.db.table_ref(Some(name))?;
        Ok(TableTxn { txn: self, t })
    }

    /// The offset of the last event written from stream source `name`, as
    /// this transaction finds it ([`SOURCES`]); `None` for a source not
    /// written yet.
    pub fn source_offset(&mut self, name: &str) -> Result<Option<i64>> {
        if !self.db.has_sources() {
            return Ok(None);
        }
        let v = self.table(SOURCES)?.value(&Value::Str(name.to_string()), "last_offset")?;
        Ok(v.and_then(|v| v.as_i64()))
    }

    /// Records `offset` as the last event written from source `name`, at
    /// `now` (microseconds since the epoch), in this transaction: committed
    /// with the rows the events wrote, or not at all. The database needs
    /// the table first ([`Database::ensure_sources`]).
    pub fn set_source(&mut self, name: &str, offset: i64, now: i64) -> Result<()> {
        if !self.db.has_sources() {
            return Err(Error::Schema(format!("the database has no {SOURCES} table yet: Database::ensure_sources makes it")));
        }
        self.table(SOURCES)?.upsert(vec![Value::Str(name.to_string()), Value::Int(offset), Value::Timestamp(now)])
    }

    /// Inserts a document; its primary key must not be live.
    pub fn insert(&mut self, doc: Document) -> Result<()> {
        let t = self.db.first_table();
        self.insert_in(&t, doc)
    }

    /// Inserts a document, replacing the live one with the same key.
    pub fn upsert(&mut self, doc: Document) -> Result<()> {
        let t = self.db.first_table();
        self.upsert_in(&t, doc)
    }

    /// Upserts `doc` only if its INT field `version` is greater than the
    /// live document's, or no document has its key: external versions, so
    /// that writers whose updates arrive out of order keep the newest.
    /// Returns whether it was written.
    pub fn upsert_if_newer(&mut self, doc: Document, version: &str) -> Result<bool> {
        let t = self.db.first_table();
        self.upsert_if_newer_in(&t, doc, version)
    }

    /// Deletes the live document with this primary key; false if none.
    pub fn delete(&mut self, key: &Value) -> Result<bool> {
        let t = self.db.first_table();
        self.delete_in(&t, key)
    }

    /// The sequence number of the write that last changed the live row
    /// with this primary key, as this transaction finds it
    /// ([`Database::seq_of`]): `None` also when this transaction wrote or
    /// deleted the row, since its own number comes at commit. A write
    /// conditional on a row's number checks it here, under the writer lock,
    /// so that no other write comes in between.
    pub fn seq_of(&self, key: &Value) -> Result<Option<u64>> {
        let t = self.db.first_table();
        self.seq_of_in(&t, key)
    }

    fn seq_of_in(&self, t: &TableRef, key: &Value) -> Result<Option<u64>> {
        let k = t.key_of(key)?;
        let st = lock(&self.db.state);
        if st.manifest.seq_no.is_none() {
            return Err(Error::Schema("the database does not number its writes (Database::start_seq_no)".into()));
        }
        let mine = self.keys.iter().find(|(id, _)| *id == t.id).is_some_and(|(_, keys)| keys.touched.contains_key(&k) || keys.sets.keys().any(|(s, _)| *s == k));
        if mine {
            return Ok(None);
        }
        st.find_live(t.pos, key).map(|loc| st.seq_at(t.pos, loc)).transpose()
    }

    /// Sets a numeric field of the live document with this primary key in
    /// place, without rewriting the document: its text and vectors stay as
    /// they are. The field must be a single-valued INT or FLOAT. False if no
    /// document has the key.
    pub fn set(&mut self, key: &Value, field: &str, value: Value) -> Result<bool> {
        let t = self.db.first_table();
        self.set_in(&t, key, field, value)
    }

    /// Adds `delta` to a numeric field of the live document with this
    /// primary key, in place as [`WriteTxn::set`] does. The base is the
    /// value as this transaction sees it, its own writes included (NULL
    /// counts as 0). An INT field takes an integer and refuses a sum that
    /// overflows; a FLOAT sum must be finite. False if no document has the
    /// key.
    pub fn add(&mut self, key: &Value, field: &str, delta: &Value) -> Result<bool> {
        let t = self.db.first_table();
        self.add_in(&t, key, field, delta)
    }

    /// [`WriteTxn::add`] to the live row with `doc`'s primary key, or, when
    /// no row has it, inserts `doc` with `delta` added to its `field` (NULL
    /// counting as 0): a counter, or an edge's weight, that starts with its
    /// first event.
    pub fn add_or_insert(&mut self, doc: Document, field: &str, delta: &Value) -> Result<()> {
        let t = self.db.first_table();
        self.add_or_insert_in(&t, doc, field, delta)
    }

    fn insert_in(&mut self, t: &TableRef, mut doc: Document) -> Result<()> {
        conform(&t.schema, &mut doc)?;
        self.fill_sids(t, &mut doc, false)?;
        if let Some(pk) = t.schema.primary_key {
            let key = Key::of(&doc[pk]).expect("validated key");
            if self.is_live(t, &key, &doc[pk]) {
                return Err(Error::Schema(format!("duplicate primary key {:?}", doc[pk])));
            }
            let doc = Arc::new(doc);
            let k = self.keys(t.id);
            k.touched.insert(key.clone(), true);
            k.written.insert(key, doc.clone());
            self.ops.push((t.id, Op::Insert(doc)));
            return Ok(());
        }
        self.ops.push((t.id, Op::Insert(Arc::new(doc))));
        Ok(())
    }

    fn upsert_in(&mut self, t: &TableRef, mut doc: Document) -> Result<()> {
        conform(&t.schema, &mut doc)?;
        let pk = t.primary_key()?;
        self.fill_sids(t, &mut doc, true)?;
        self.delete_in(t, &doc[pk].clone())?;
        let key = Key::of(&doc[pk]).expect("validated key");
        let doc = Arc::new(doc);
        let k = self.keys(t.id);
        k.touched.insert(key.clone(), true);
        k.written.insert(key, doc.clone());
        self.ops.push((t.id, Op::Insert(doc)));
        Ok(())
    }

    fn upsert_if_newer_in(&mut self, t: &TableRef, mut doc: Document, version: &str) -> Result<bool> {
        conform(&t.schema, &mut doc)?;
        let pk = t.primary_key()?;
        let vf = t.schema.fields.iter().position(|f| f.name == version).ok_or_else(|| Error::Schema(format!("no field {version}")))?;
        if t.schema.fields[vf].ty != FieldType::Int || t.schema.fields[vf].multi || t.schema.fields[vf].logical == Some(crate::schema::Logical::Bool) {
            return Err(Error::Schema(format!("version field {version} must be a single-valued INT or TIMESTAMP")));
        }
        let new = doc[vf].as_i64().ok_or_else(|| Error::Schema(format!("document has no {version}")))?;
        let key = Key::of(&doc[pk]).expect("validated key");
        let k = self.keys(t.id);
        let old = match (k.touched.get(&key), k.written.get(&key)) {
            (Some(true), Some(d)) => d[vf].as_i64(),
            (Some(false), _) => None,
            _ => match k.sets.get(&(key.clone(), vf)) {
                Some(v) => v.as_i64(),
                None => self.db.live_int(t, &doc[pk], vf)?,
            },
        };
        if old.is_some_and(|o| o >= new) {
            return Ok(false);
        }
        self.upsert_in(t, doc)?;
        Ok(true)
    }

    fn delete_in(&mut self, t: &TableRef, key: &Value) -> Result<bool> {
        let k = t.key_of(key)?;
        if !self.is_live(t, &k, key) {
            return Ok(false);
        }
        let keys = self.keys(t.id);
        keys.touched.insert(k.clone(), false);
        keys.written.remove(&k);
        self.ops.push((t.id, Op::Delete(key.clone())));
        Ok(true)
    }

    fn set_in(&mut self, t: &TableRef, key: &Value, field: &str, value: Value) -> Result<bool> {
        let k = t.key_of(key)?;
        let i = t.schema.fields.iter().position(|f| f.name == field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        t.check_not_placing(i)?;
        let f = &t.schema.fields[i];
        let value = match (&f.ty, f.store(value)?) {
            _ if f.multi || Some(i) == t.schema.primary_key => {
                return Err(Error::Schema(format!("{field} cannot be set in place: it is multi-valued or the key")))
            }
            (_, Value::Null) if f.not_null => return Err(Error::Schema(format!("{field} is NOT NULL: give it a value"))),
            (FieldType::Int, v @ (Value::Int(_) | Value::Null)) => v,
            (FieldType::Float, Value::Int(x)) => Value::Float(x as f64),
            (FieldType::Float, Value::Float(x)) if !x.is_finite() => return Err(Error::Schema(format!("bad value for {field}: {x} (FLOAT values are finite)"))),
            (FieldType::Float, v @ (Value::Float(_) | Value::Null)) => v,
            (FieldType::Int | FieldType::Float, v) => return Err(Error::Schema(format!("bad value for {field}: {v:?}"))),
            _ => return Err(Error::Schema(format!("{field} is not a numeric field; only INT and FLOAT are set in place"))),
        };
        if !self.is_live(t, &k, key) {
            return Ok(false);
        }
        let keys = self.keys(t.id);
        match keys.written.get_mut(&k) {
            Some(d) => Arc::make_mut(d)[i] = value.clone(),
            None => {
                keys.sets.insert((k, i), value.clone());
            }
        }
        self.ops.push((t.id, Op::Set { key: key.clone(), field: i as u16, value }));
        Ok(true)
    }

    fn add_in(&mut self, t: &TableRef, key: &Value, field: &str, delta: &Value) -> Result<bool> {
        let k = t.key_of(key)?;
        let i = t.schema.fields.iter().position(|f| f.name == field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        t.check_not_placing(i)?;
        if let Some(l) = t.schema.fields[i].logical {
            return Err(Error::Schema(format!("{field} is a {}: set it rather than add to it", l.type_name())));
        }
        let Some(base) = self.value_of(t, &k, key, i)? else { return Ok(false) };
        let sum = added(&t.schema, i, &base, delta)?;
        self.set_in(t, key, field, sum)
    }

    /// [`WriteTxn::add`] to the live row with `doc`'s key, or else inserts
    /// `doc` with `delta` added to its `field` (NULL as 0).
    fn add_or_insert_in(&mut self, t: &TableRef, mut doc: Document, field: &str, delta: &Value) -> Result<()> {
        let pk = t.primary_key()?;
        conform(&t.schema, &mut doc)?;
        let key = doc[pk].clone();
        if self.add_in(t, &key, field, delta)? {
            return Ok(());
        }
        let i = t.schema.fields.iter().position(|f| f.name == field).expect("add_in found the field");
        doc[i] = added(&t.schema, i, &doc[i], delta)?;
        self.insert_in(t, doc)
    }

    /// Field `field` of the live row with this key, as this transaction
    /// finds it, its own writes included; `None` when no row with the key is
    /// live.
    fn value_in(&mut self, t: &TableRef, key: &Value, field: &str) -> Result<Option<Value>> {
        let k = t.key_of(key)?;
        let i = t.schema.fields.iter().position(|f| f.name == field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        self.value_of(t, &k, key, i)
    }

    fn value_of(&mut self, t: &TableRef, k: &Key, key: &Value, i: usize) -> Result<Option<Value>> {
        if !self.is_live(t, k, key) {
            return Ok(None);
        }
        let keys = self.keys(t.id);
        let v = match (keys.touched.get(k), keys.written.get(k)) {
            (Some(true), Some(d)) => d[i].clone(),
            _ => match keys.sets.get(&(k.clone(), i)) {
                Some(v) => v.clone(),
                None => self.db.live_value(t, key, i)?,
            },
        };
        Ok(Some(v))
    }

    /// Operations buffered so far.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Logs the transaction (durable on return when `DbOptions::sync`) and
    /// applies it. `Ok` means it committed and `Err` that it did not, so a
    /// failed commit can be retried. If the memtables are then full they
    /// are flushed; a failed flush does not fail the commit, and shows in
    /// [`Database::flush_error`].
    pub fn commit(self) -> Result<()> {
        self.commit_seq().map(|_| ())
    }

    /// Commits, and returns the transaction's sequence number when the
    /// database numbers its writes ([`Database::start_seq_no`]); `None`
    /// then only for a transaction that wrote nothing.
    pub fn commit_seq(self) -> Result<Option<u64>> {
        let WriteTxn { db, mut wal, ops, .. } = self;
        db.commit_ops(&mut wal, ops)
    }

    pub fn rollback(self) {}

    /// Writes a primary's logged operations, which must apply here as they
    /// did there ([`Database::apply_shipped`]).
    fn replay(&mut self, ops: Vec<(u32, Op)>) -> Result<()> {
        let differs = |what: String| Error::Schema(format!("a shipped transaction {what}: this database is not a copy of its primary's"));
        for (table, op) in ops {
            let t = self.db.table_by_id(table)?;
            match op {
                Op::Insert(doc) => self.insert_in(&t, Arc::unwrap_or_clone(doc)).map_err(|e| differs(format!("inserts a row it cannot ({e})")))?,
                Op::Delete(key) => {
                    if !self.delete_in(&t, &key)? {
                        return Err(differs(format!("deletes {key:?}, which no row here has")));
                    }
                }
                Op::Set { key, field, value } => {
                    let name = t.schema.fields.get(field as usize).map(|f| f.name.clone()).ok_or_else(|| differs(format!("sets field #{field}")))?;
                    if !self.set_in(&t, &key, &name, value)? {
                        return Err(differs(format!("sets {name} of {key:?}, which no row here has")));
                    }
                }
                op @ Op::Profile { .. } => self.ops.push((t.id, op)),
            }
        }
        Ok(())
    }

    /// Fills the semantic-ID fields `doc` leaves out ([`crate::sid`]); an
    /// upsert keeps the code of the row it replaces when the new vector has
    /// the same levels.
    fn fill_sids(&mut self, t: &TableRef, doc: &mut Document, upsert: bool) -> Result<()> {
        let fields = crate::sid::fields(&t.schema);
        if fields.is_empty() {
            return Ok(());
        }
        let mut live = Vec::new();
        if let (true, Some(pk)) = (upsert, t.schema.primary_key) {
            let key = Key::of(&doc[pk]).expect("validated key");
            for &(f, _) in &fields {
                let k = self.keys(t.id);
                let v = match (k.touched.get(&key), k.written.get(&key)) {
                    (Some(true), Some(d)) => d[f].clone(),
                    (Some(false), _) => Value::Null,
                    _ => self.db.live_value(t, &doc[pk], f)?,
                };
                live.push((f, v));
            }
        }
        let i = match self.ties.iter().position(|(id, _)| *id == t.id) {
            Some(i) => i,
            None => {
                self.ties.push((t.id, Ties::default()));
                self.ties.len() - 1
            }
        };
        self.db.fill_sids(t, doc, &mut self.ties[i].1, &live)
    }

    fn keys(&mut self, table: u32) -> &mut TxnKeys {
        let i = match self.keys.iter().position(|(id, _)| *id == table) {
            Some(i) => i,
            None => {
                self.keys.push((table, TxnKeys::default()));
                self.keys.len() - 1
            }
        };
        &mut self.keys[i].1
    }

    fn is_live(&self, t: &TableRef, k: &Key, v: &Value) -> bool {
        match self.keys.iter().find(|(id, _)| *id == t.id).and_then(|(_, keys)| keys.touched.get(k)) {
            Some(&live) => live,
            None => lock(&self.db.state).find_live(t.pos, v).is_some(),
        }
    }
}

/// The outcome of a [`VersionWriter`] commit, to be published.
struct Committed {
    meta: Meta,
    slot: usize,
    manifest: Manifest,
    fl: FreeList,
    file_pages: u64,
    bytes_written: u64,
    /// Directory layout: files to unlink once the version is published.
    obsolete: Vec<String>,
}

/// Builds the next on-disk version (shadow paging). Holds working copies
/// of the manifest and freelist; nothing changes in memory until the commit
/// is published, and nothing on disk is reachable until the meta slot is
/// written. Callers hold the writer lock.
struct VersionWriter<'a> {
    db: &'a Database,
    txn: u64,
    manifest: Manifest,
    fl: FreeList,
    meta: Meta,
    slot: usize,
    file_pages: u64,
    bytes_written: u64,
    segments_added: usize,
    /// Directory layout: files the new version stops naming.
    obsolete: Vec<String>,
    /// A checkpoint's earlier checkpoint: its directory and its segment
    /// files' checksums and lengths, by name.
    link: Option<(PathBuf, HashMap<String, (u32, u64)>)>,
    segments_linked: usize,
    /// Ids of the reservations this version installs.
    installs: Vec<u64>,
}

impl<'a> VersionWriter<'a> {
    fn begin(db: &'a Database) -> VersionWriter<'a> {
        let st = lock(&db.state);
        let mut fl = st.freelist.clone();
        // Reusable: freed before the newest on-disk commit (so the older
        // meta slot cannot reach them) and by a commit no live snapshot
        // predates.
        fl.release((st.meta.txn - 1).min(db.pins.min().unwrap_or(u64::MAX)));
        let mut manifest = st.manifest.clone();
        // Past the ids reserved for merges.
        manifest.next_segment_id = manifest.next_segment_id.max(st.next_segment_id);
        // Profiles logged since the last version. The writer lock, which
        // every version is written under, keeps them until the commit.
        for t in manifest.tables.iter_mut() {
            if let Some(ts) = st.tables.iter().find(|ts| ts.id == t.id) {
                t.profiles = ts.profiles.to_vec();
            }
        }
        VersionWriter {
            db,
            txn: st.txn + 1,
            manifest,
            fl,
            meta: st.meta,
            slot: st.slot,
            file_pages: st.file_pages,
            bytes_written: 0,
            segments_added: 0,
            obsolete: Vec::new(),
            link: None,
            segments_linked: 0,
            installs: Vec::new(),
        }
    }

    /// Writes `bytes` as file `name` in the directory layout, or into a new
    /// extent of the single file.
    fn write_as(&mut self, bytes: &[u8], name: String) -> Result<(Extent, Option<String>)> {
        match &self.db.dir {
            Some(d) => {
                dir::write_version_file(d, &name, bytes, true)?;
                self.bytes_written += bytes.len() as u64;
                Ok((Extent { page: 0, len: bytes.len() as u64 }, Some(name)))
            }
            None => Ok((self.write(bytes)?, None)),
        }
    }

    /// Allocates an extent for `bytes` and writes them.
    fn write(&mut self, bytes: &[u8]) -> Result<Extent> {
        debug_assert!(!bytes.is_empty());
        let page = self.fl.alloc(pages_for(bytes.len()));
        self.grow()?;
        io::write_at(&self.db.file, bytes, page * PAGE)?;
        self.bytes_written += bytes.len() as u64;
        Ok(Extent { page, len: bytes.len() as u64 })
    }

    fn grow(&mut self) -> Result<()> {
        if self.fl.page_count() > self.file_pages {
            io::set_len(&self.db.file, self.fl.page_count() * PAGE)?;
            self.file_pages = self.fl.page_count();
        }
        Ok(())
    }

    /// Adds a segment to table `table` at position `pos` (default: last).
    #[allow(clippy::too_many_arguments)]
    fn add_segment(
        &mut self,
        table: u32,
        pos: Option<usize>,
        image: &[u8],
        n_docs: u32,
        deleted: Option<&LiveDocs>,
        updates: Option<&Updates>,
    ) -> Result<()> {
        let id = self.manifest.next_segment_id;
        self.manifest.next_segment_id += 1;
        let (extent, file) = match (&self.db.dir, self.linkable(id, image)) {
            (Some(d), Some(from)) => {
                std::fs::hard_link(&from, d.join(dir::segment_name(id)))?;
                self.segments_linked += 1;
                (Extent { page: 0, len: image.len() as u64 }, Some(dir::segment_name(id)))
            }
            _ => self.write_as(image, dir::segment_name(id))?,
        };
        self.insert_segment(table, pos, id, extent, file, crc32fast::hash(image), n_docs, deleted, updates)
    }

    /// Adds a segment whose image was written into space reserved for it
    /// ([`Reserved`]), with checksum `crc`.
    #[allow(clippy::too_many_arguments)]
    fn add_reserved_segment(
        &mut self,
        table: u32,
        pos: Option<usize>,
        r: &Reserved,
        crc: u32,
        n_docs: u32,
        deleted: Option<&LiveDocs>,
        updates: Option<&Updates>,
    ) -> Result<()> {
        self.installs.push(r.id);
        self.insert_segment(table, pos, r.id, r.extent, r.file.clone(), crc, n_docs, deleted, updates)
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_segment(
        &mut self,
        table: u32,
        pos: Option<usize>,
        id: u64,
        extent: Extent,
        file: Option<String>,
        crc: u32,
        n_docs: u32,
        deleted: Option<&LiveDocs>,
        updates: Option<&Updates>,
    ) -> Result<()> {
        let (mut deletes, mut deletes_crc, mut n_deleted, mut deletes_file) = (None, 0, 0, None);
        if let Some(l) = deleted.filter(|l| l.n_deleted() > 0) {
            let b = l.encode();
            let (e, f) = self.write_as(&b, dir::deletes_name(id, self.txn))?;
            (deletes, deletes_file) = (Some(e), f);
            deletes_crc = crc32fast::hash(&b);
            n_deleted = l.n_deleted();
        }
        let (mut upd, mut updates_crc, mut updates_file) = (None, 0, None);
        if let Some(u) = updates {
            let b = u.encode(&self.manifest.table(table).expect("table of a new segment").schema);
            let (e, f) = self.write_as(&b, dir::updates_name(id, self.txn))?;
            (upd, updates_file, updates_crc) = (Some(e), f, crc32fast::hash(&b));
        }
        let meta = SegmentMeta {
            id,
            extent,
            crc,
            n_docs,
            n_deleted,
            deletes,
            deletes_crc,
            txn: self.txn,
            file,
            deletes_file,
            updates: upd,
            updates_file,
            updates_crc,
        };
        let segs = &mut self.table_mut(table).segments;
        segs.insert(pos.unwrap_or(segs.len()), meta);
        self.segments_added += 1;
        Ok(())
    }

    /// The earlier checkpoint's file of segment `id`, if it holds `image`.
    fn linkable(&self, id: u64, image: &[u8]) -> Option<PathBuf> {
        let (prev, segs) = self.link.as_ref()?;
        let name = dir::segment_name(id);
        let &(crc, len) = segs.get(&name)?;
        (len == image.len() as u64 && crc == crc32fast::hash(image)).then(|| prev.join(name))
    }

    fn table_mut(&mut self, id: u32) -> &mut TableMeta {
        self.manifest.table_mut(id).expect("table in manifest")
    }

    /// Replaces a segment's delete set with its current one plus `docs`.
    fn add_deletes(&mut self, id: u64, docs: &BTreeSet<DocId>) -> Result<()> {
        let h = lock(&self.db.state).handle(id).cloned().expect("segment of a pending delete");
        let live = LiveDocs::with_deleted(h.base.as_deref(), h.meta.n_docs, docs.iter().copied());
        let b = live.encode();
        let (e, file) = self.write_as(&b, dir::deletes_name(id, self.txn))?;
        let s = self.manifest.segment_mut(id).expect("segment in manifest");
        match (s.deletes, s.deletes_file.take()) {
            (_, Some(old)) => self.obsolete.push(old),
            (Some(old), None) => self.fl.defer(self.txn, old),
            _ => {}
        }
        s.deletes_file = file;
        s.deletes = Some(e);
        s.deletes_crc = crc32fast::hash(&b);
        s.n_deleted = live.n_deleted();
        Ok(())
    }

    /// Replaces a segment's in-place updates with its current ones plus
    /// `newer`.
    fn add_updates(&mut self, id: u64, newer: &Updates) -> Result<()> {
        let h = lock(&self.db.state).handle(id).cloned().expect("segment of a pending update");
        let schema = &self.manifest.tables.iter().find(|t| t.segments.iter().any(|s| s.id == id)).expect("table of a segment").schema;
        let b = Updates::merged(h.updates.as_deref(), newer).encode(schema);
        let (e, file) = self.write_as(&b, dir::updates_name(id, self.txn))?;
        let s = self.manifest.segment_mut(id).expect("segment in manifest");
        match (s.updates, s.updates_file.take()) {
            (_, Some(old)) => self.obsolete.push(old),
            (Some(old), None) => self.fl.defer(self.txn, old),
            _ => {}
        }
        s.updates_file = file;
        s.updates = Some(e);
        s.updates_crc = crc32fast::hash(&b);
        Ok(())
    }

    /// Copies extents into free runs below them, highest first. Returns how
    /// many moved.
    fn relocate_down(&mut self) -> Result<usize> {
        if self.db.dir.is_some() {
            return Ok(0);
        }
        let map = lock(&self.db.state).map.clone();
        // (extent, (table, segment), what): 0 the image, 1 the delete set,
        // 2 updates.
        let mut extents: Vec<(Extent, (usize, usize), u8)> = Vec::new();
        for (t, table) in self.manifest.tables.iter().enumerate() {
            for (i, s) in table.segments.iter().enumerate() {
                extents.push((s.extent, (t, i), 0));
                extents.extend(s.deletes.map(|e| (e, (t, i), 1)));
                extents.extend(s.updates.map(|e| (e, (t, i), 2)));
            }
        }
        extents.sort_by_key(|&(e, _, _)| std::cmp::Reverse(e.page));
        let mut moved = 0;
        for (e, (t, i), what) in extents {
            let Some(page) = self.fl.alloc_below(e.pages(), e.page) else { continue };
            // Copy out of the mapping first: on macOS, a pwrite whose source
            // is a mapping of the same file can deadlock in the kernel.
            let bytes = map[e.range()].to_vec();
            io::write_at(&self.db.file, &bytes, page * PAGE)?;
            self.bytes_written += e.len;
            self.fl.defer(self.txn, e);
            let to = Extent { page, len: e.len };
            let s = &mut self.manifest.tables[t].segments[i];
            if what == 2 {
                s.updates = Some(to);
            } else if what == 1 {
                s.deletes = Some(to);
            } else {
                s.extent = to;
            }
            moved += 1;
        }
        Ok(moved)
    }

    fn remove_segment(&mut self, id: u64) {
        let (t, i) = self
            .manifest
            .tables
            .iter()
            .enumerate()
            .find_map(|(t, table)| Some((t, table.segments.iter().position(|s| s.id == id)?)))
            .expect("segment in manifest");
        let s = self.manifest.tables[t].segments.remove(i);
        self.free_segment(s);
    }

    /// Drops table `id` and frees its segments.
    fn drop_table(&mut self, id: u32) {
        let i = self.manifest.tables.iter().position(|t| t.id == id).expect("table in manifest");
        for s in self.manifest.tables.remove(i).segments {
            self.free_segment(s);
        }
    }

    /// Frees the pages (or names the files) of a segment the new version
    /// no longer holds.
    fn free_segment(&mut self, s: SegmentMeta) {
        if self.db.dir.is_some() {
            self.obsolete.extend(s.file);
            self.obsolete.extend(s.deletes_file);
            self.obsolete.extend(s.updates_file);
            return;
        }
        self.fl.defer(self.txn, s.extent);
        if let Some(u) = s.updates {
            self.fl.defer(self.txn, u);
        }
        if let Some(d) = s.deletes {
            self.fl.defer(self.txn, d);
        }
    }

    /// Writes the manifest and freelist, syncs, then writes the other meta
    /// slot and syncs again.
    fn commit(mut self, wal: &mut Wal) -> Result<Committed> {
        // The transactions in the WAL come before this version, which may
        // hold their effects (a merge drops the documents they deleted):
        // without `sync` they are made durable now, before it can be.
        wal.sync_appends()?;
        // The last sequence number given so far, below which none is given
        // again after a crash, even once the WAL that held it is gone.
        if let Some(n) = &mut self.manifest.seq_no {
            n.seq = n.seq.max(lock(&self.db.state).seq);
        }

        if let Some(d) = self.db.dir.clone() {
            return self.commit_dir(&d);
        }
        let db = self.db;
        let c = self.txn;
        self.fl.defer(c, self.meta.manifest);
        self.fl.defer(c, self.meta.freelist);
        self.fl.trim_tail();
        self.manifest.txn = c;
        let mbytes = self.manifest.encode();
        let manifest = self.write(&mbytes)?;
        let reserved: Vec<Extent> = lock(&db.state).reserved.iter().filter(|r| !self.installs.contains(&r.id)).map(|r| r.extent).collect();
        let (freelist, fbytes) = place_freelist(&mut self.fl, &reserved);
        self.grow()?;
        io::write_at(&db.file, &fbytes, freelist.page * PAGE)?;
        self.bytes_written += fbytes.len() as u64;
        io::sync(&db.file)?;
        let meta = Meta {
            format: self.manifest.format(),
            db_id: self.meta.db_id,
            txn: c,
            manifest,
            manifest_crc: crc32fast::hash(&mbytes),
            freelist,
            freelist_crc: crc32fast::hash(&fbytes),
            page_count: self.fl.page_count(),
        };
        let slot = 1 - self.slot;
        // Past this point the on-disk version may already be the new one.
        pager::write_meta(&db.file, slot, &meta).map_err(|e| db.poison(e))?;
        // Synced whatever `DbOptions::sync` is: the next commit overwrites
        // the other slot and reuses the pages this version freed, so this
        // one must be the intact version a power loss falls back to. Its
        // pages were synced above, before the slot named them.
        io::sync(&db.file).map_err(|e| db.poison(e.into()))?;
        // Free pages at the end are reachable from neither slot. A failed
        // truncation only leaves the file longer than needed.
        if self.fl.page_count() < self.file_pages && io::set_len(&db.file, self.fl.page_count() * PAGE).is_ok() {
            self.file_pages = self.fl.page_count();
        }
        Ok(Committed { meta, slot, manifest: self.manifest, fl: self.fl, file_pages: self.file_pages, bytes_written: self.bytes_written, obsolete: Vec::new() })
    }

    /// Directory layout: writes the manifest file, then points `CURRENT` at
    /// it with an atomic rename.
    fn commit_dir(mut self, d: &Path) -> Result<Committed> {
        let db = self.db;
        let c = self.txn;
        let previous = dir::manifest_name(lock(&db.state).manifest.txn);
        self.manifest.txn = c;
        let mbytes = self.manifest.encode();
        let name = dir::manifest_name(c);
        dir::write_version_file(d, &name, &mbytes, true)?;
        self.bytes_written += mbytes.len() as u64;
        let cur = Current { db_id: self.meta.db_id, txn: c, manifest: name, crc: crc32fast::hash(&mbytes), format: self.manifest.format() };
        // Past the rename the on-disk version may already be the new one.
        dir::write_current(d, &cur, true).map_err(|e| db.poison(e))?;
        self.obsolete.push(previous);
        Ok(Committed {
            meta: dir_meta(&cur),
            slot: self.slot,
            manifest: self.manifest,
            fl: self.fl,
            file_pages: 0,
            bytes_written: self.bytes_written,
            obsolete: self.obsolete,
        })
    }
}

/// The meta of a directory-layout version (it has no pages).
fn dir_meta(cur: &Current) -> Meta {
    let none = Extent { page: 0, len: 0 };
    Meta { format: cur.format, db_id: cur.db_id, txn: cur.txn, manifest: none, manifest_crc: cur.crc, freelist: none, freelist_crc: 0, page_count: 0 }
}

/// Every segment's size, by table.
fn segment_stats(m: &Manifest) -> Vec<SegmentStats> {
    m.tables
        .iter()
        .flat_map(|t| {
            t.segments.iter().map(|s| SegmentStats {
                table: t.schema.table.clone(),
                id: s.id,
                n_docs: s.n_docs,
                n_deleted: s.n_deleted,
                bytes: s.extent.len,
                delete_set_bytes: s.deletes.map_or(0, |e| e.len),
            })
        })
        .collect()
}

/// What [`Database::drop_tables`] does about references and missing tables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropOptions {
    /// Remove the references that tables left standing hold to the dropped
    /// ones (SQL `CASCADE`), instead of refusing to drop.
    pub cascade: bool,
    /// Skip names that are not tables (SQL `IF EXISTS`), instead of failing.
    pub if_exists: bool,
}

/// Checks that each segment lists the fields it was written with
/// ([`crate::segment::FIELDS`]: ids, none twice, each one its table has
/// given), or, in a database written before field ids, lists none.
/// `handles` are the version's segments in the manifest's order.
fn check_field_lists(manifest: &Manifest, handles: &[Arc<SegHandle>]) -> Result<()> {
    let mut hs = handles.iter();
    for t in &manifest.tables {
        for s in &t.segments {
            let h = hs.next().ok_or_else(|| corrupt("the open version has fewer segments than its manifest"))?;
            match (t.schema.is_legacy(), h.file.field_ids()?) {
                (true, None) => {}
                (false, Some(mut ids)) => {
                    ids.sort_unstable();
                    if ids.windows(2).any(|w| w[0] == w[1]) || ids.last().is_some_and(|&id| id >= t.schema.next_field_id) {
                        return Err(corrupt(format!("segment {} lists fields its table {} never had, or one twice", s.id, t.schema.table)));
                    }
                }
                (true, Some(_)) => return Err(corrupt(format!("segment {} lists field ids, and its table {} has none", s.id, t.schema.table))),
                (false, None) => return Err(corrupt(format!("segment {} does not list the fields it was written with", s.id))),
            }
        }
    }
    Ok(())
}

/// Table names are plain identifiers, so SQL can name them unquoted.
fn check_table_name(name: &str) -> Result<()> {
    check_name("table", name)
}

/// A table's or a rank profile's name: letters, digits and underscores,
/// not starting with a digit.
fn check_name(what: &str, name: &str) -> Result<()> {
    let mut chars = name.chars();
    let ok = chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        Ok(())
    } else {
        Err(Error::Schema(format!("bad {what} name {name:?}: use letters, digits and underscores, not starting with a digit")))
    }
}

/// Allocates the freelist's own extent and encodes the freelist after that
/// allocation, padded to the size measured before it (allocating never adds
/// a run, so the encoding only shrinks). Space `reserved` for merged images
/// not installed yet is free on disk, as pages a commit freed before any
/// other (reusable at once), so that a crash before their install leaves
/// no page that is neither used nor free.
/// `base + delta` for numeric field `i` ([`WriteTxn::add`]): NULL counts as
/// 0, an INT takes an integer and refuses a sum that overflows, and a FLOAT
/// sum must be finite.
fn added(schema: &Schema, i: usize, base: &Value, delta: &Value) -> Result<Value> {
    let field = &schema.fields[i].name;
    let overflow = || Error::Schema(format!("{field} + {delta:?} overflows"));
    Ok(match (&schema.fields[i].ty, delta) {
        (FieldType::Int, Value::Int(x)) => Value::Int(base.as_i64().unwrap_or(0).checked_add(*x).ok_or_else(overflow)?),
        (FieldType::Int, _) => return Err(Error::Schema(format!("{field} is INT: add an integer"))),
        (FieldType::Float, d @ (Value::Int(_) | Value::Float(_))) => {
            let y = base.as_f64().unwrap_or(0.0) + d.as_f64().unwrap_or(0.0);
            if !y.is_finite() {
                return Err(overflow());
            }
            Value::Float(y)
        }
        (FieldType::Float, d) => return Err(Error::Schema(format!("cannot add {d:?} to {field}"))),
        _ => return Err(Error::Schema(format!("{field} is not a numeric field"))),
    })
}

fn place_freelist(fl: &mut FreeList, reserved: &[Extent]) -> (Extent, Vec<u8>) {
    let on_disk = |fl: &FreeList| {
        let mut d = fl.clone();
        for &e in reserved {
            d.defer(0, e);
        }
        d
    };
    let bound = if reserved.is_empty() { fl.encoded_len() } else { on_disk(fl).encoded_len() };
    let page = fl.alloc(pages_for(bound));
    let mut bytes = if reserved.is_empty() { fl.encode() } else { on_disk(fl).encode() };
    bytes.resize(bound, 0);
    (Extent { page, len: bound as u64 }, bytes)
}

fn pages_for(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(PAGE).max(1)
}

/// Whether a logged operation is one a transaction on a table of `schema`
/// could have committed (see [`WriteTxn`]'s checks). Replay takes one that
/// is not, behind an intact checksum, as damage: applied, it would break
/// what readers rely on.
fn op_fits(schema: &Schema, op: &Op) -> bool {
    let is_key = |k: &Value| {
        schema.primary_key.is_some_and(|pk| matches!((&schema.fields[pk].ty, k), (FieldType::Keyword, Value::Str(_)) | (FieldType::Int, Value::Int(_))))
    };
    match op {
        Op::Insert(doc) => validate(schema, doc).is_ok(),
        Op::Delete(k) => is_key(k),
        Op::Profile { name, def } => check_name("rank profile", name).is_ok() && def.as_ref().is_none_or(|p| p.name == *name),
        Op::Set { key, field, value } => {
            let i = *field as usize;
            is_key(key)
                && Some(i) != schema.primary_key
                && schema.fields.get(i).is_some_and(|f| {
                    !f.multi
                        && match (&f.ty, value) {
                            (FieldType::Int, Value::Int(_)) | (FieldType::Int | FieldType::Float, Value::Null) => true,
                            (FieldType::Float, Value::Float(x)) => x.is_finite(),
                            _ => false,
                        }
                })
        }
    }
}

/// The sealed WAL files ([`wal::sealed_path`]) a background flush did not
/// install, oldest first: their txns, their committed transactions, and,
/// when one is damaged and `drop_damaged` says to drop what follows, what
/// was dropped (opening then drops the open log's transactions too). The
/// ones a flush installed (sealed at or before `manifest.wal_txn`) and
/// other databases' are deleted.
#[allow(clippy::type_complexity)]
fn read_sealed_wals(
    wal_path: &Path,
    manifest: &Manifest,
    db_id: u64,
    drop_damaged: bool,
    fits: &dyn Fn(u32, &Op) -> bool,
) -> Result<(Vec<u64>, Vec<WalTxn>, Option<String>)> {
    let (mut kept, mut txns, mut cut) = (Vec::new(), Vec::new(), None::<String>);
    for (t, p) in wal::sealed_files(wal_path)? {
        if t <= manifest.wal_txn {
            let _ = io::remove_file(&p);
            continue;
        }
        if cut.is_some() {
            std::fs::copy(&p, wal::damaged_path(&p))?;
            io::remove_file(&p)?;
            continue;
        }
        let Some((got, damage)) = wal::read_sealed(&p, db_id, fits)? else {
            let _ = io::remove_file(&p);
            continue;
        };
        txns.extend(got);
        kept.push(t);
        if let Some(at) = damage {
            // It was durable before it was sealed: this is no torn append.
            let what = format!("{} is damaged at byte {at}", p.display());
            if !drop_damaged {
                return Err(corrupt(format!(
                    "{what}; open with DbOptions::wal_drop_damaged to drop the transactions from there on, in it and in the logs after it (each is copied to <log>.damaged first)"
                )));
            }
            std::fs::copy(&p, wal::damaged_path(&p))?;
            let f = OpenOptions::new().write(true).open(&p)?;
            io::set_len(&f, at)?;
            io::sync(&f)?;
            cut = Some(format!("{what}: dropped the transactions from there on, in it and in the logs after it, each copied to <log>.damaged"));
        }
    }
    Ok((kept, txns, cut))
}

/// A hidden temporary name beside `path` to write it under:
/// `.<name>.<what>-<random>`.
fn beside(path: &Path, what: &str) -> Result<PathBuf> {
    let name = path.file_name().ok_or_else(|| Error::Schema(format!("{} names no file", path.display())))?.to_string_lossy();
    Ok(path.with_file_name(format!(".{name}.{what}-{:016x}", rand::random::<u64>())))
}

fn wal_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push("-wal");
    PathBuf::from(s)
}

/// How long an open waits for a lock that is held.
pub const LOCK_WAIT: Duration = Duration::from_millis(500);

/// One open `Database` per file: an exclusive `flock`, released when the
/// file is closed (also by a crash). A child process holds a copy of every
/// descriptor of its parent from its fork until its exec closes them, so a
/// database that a process closes and opens again while it starts a child
/// can find its own lock still held for a moment: a held lock is retried
/// for [`LOCK_WAIT`] before the open fails.
fn lock_exclusive(file: &File) -> Result<()> {
    let deadline = Instant::now() + LOCK_WAIT;
    loop {
        // SAFETY: flock on a valid descriptor.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(e.into());
        }
        if Instant::now() >= deadline {
            return Err(Error::Busy("the database is open elsewhere".into()));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
