//! The memory table in a Strata file: appends, hybrid search, SQL.
//!
//! Reads run on the caller's thread against a snapshot, any number at once.
//! Writes go to one writer thread ([`Writer`]), which commits whatever is
//! queued as one transaction and numbers new records in commit order.
//! Strata publishes each version before the writer answers
//! (`DbOptions::publish_snapshots`), so a caller reads its own writes, and a
//! reader never waits for a write or for the index build that follows it. A
//! full memtable becomes a segment, and segments merge, on Strata's
//! background threads; a write waits for a flush only once the memtable
//! holds twice `flush_docs` before the flush under way ends.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{mpsc, Arc};

use serde::Serialize;
use strata::db::Db;
use strata::query::exec::ResultSet;
use strata::query::Params;
use strata::storage::{DbOptions, MergerOptions, Snapshot};
use strata::{Document, Value, VectorIndexKind};

use crate::embed::Embedder;
use crate::record::{entity_id, file_entity_id, links_of, record_id, schema, Draft, Kind, Link, Record, COLUMNS};
use crate::{err, Error, Result};

/// One search result with its component scores.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub record: Record,
    /// Raw BM25 score (None if not in the BM25 list).
    pub bm25: Option<f64>,
    /// Cosine similarity (None if not in the vector list).
    pub cos: Option<f64>,
    /// Final ranking score.
    pub score: f64,
}

/// What a search may return.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// Exclude records of this session with `seq >= window_start` (the live window).
    pub exclude_window: Option<(String, i64)>,
    /// Only these kinds (empty: any).
    pub kinds: Vec<Kind>,
    /// Only records at or after this time.
    pub since_ts: Option<i64>,
    /// Only records before this time.
    pub until_ts: Option<i64>,
    /// Only this session.
    pub session: Option<String>,
}

impl Filter {
    fn sql(&self) -> String {
        let parts = self.conds();
        if parts.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", parts.join(" AND "))
        }
    }

    /// The filter's conditions, which name the values of
    /// [`Filter::params`] (`:window_session`, `:session`) rather than
    /// quote them. Entity records are never results unless asked for by
    /// kind.
    pub fn conds(&self) -> Vec<String> {
        let mut parts = Vec::new();
        if let Some((_, w)) = &self.exclude_window {
            parts.push(format!("NOT (session = :window_session AND seq >= {w})"));
        }
        if !self.kinds.is_empty() {
            let ks: Vec<String> = self.kinds.iter().map(|k| format!("'{}'", k.as_str())).collect();
            parts.push(format!("kind IN ({})", ks.join(", ")));
        }
        if let Some(t) = self.since_ts {
            parts.push(format!("ts >= {t}"));
        }
        if let Some(t) = self.until_ts {
            parts.push(format!("ts < {t}"));
        }
        if self.session.is_some() {
            parts.push("session = :session".into());
        }
        if self.kinds.is_empty() {
            parts.push("kind != 'entity'".into());
        }
        parts
    }

    /// The values [`Filter::conds`] names, for a query that runs them.
    pub fn params(&self) -> Params {
        let mut p = Params::new();
        if let Some((s, _)) = &self.exclude_window {
            p = p.set("window_session", Value::Str(s.clone()));
        }
        if let Some(s) = &self.session {
            p = p.set("session", Value::Str(s.clone()));
        }
        p
    }
}

/// How recall walks the links (see [`Store::reach`]).
#[derive(Clone, Debug)]
pub struct Reach {
    pub hops: usize,
    /// Kinds a walk may pass through (seeds and end points are exempt).
    pub through: Vec<Kind>,
    /// The most records a walk may meet; past it, the walk takes fewer
    /// hops ([`Store::reach`]).
    pub max_nodes: usize,
}

#[derive(Clone, Debug)]
pub struct StoreOptions {
    pub vector_index: VectorIndexKind,
    /// Flush the memtable to a segment past this many records (in the
    /// background; writes wait for a flush only once it holds twice this).
    pub flush_docs: usize,
    /// fsync every commit.
    pub sync: bool,
    /// Records per memtable chunk. The first query after a write rebuilds
    /// the last chunk's in-memory segment, so this bounds that work.
    pub mem_chunk_docs: usize,
    /// Memtable segments scan their vectors instead of building a graph.
    pub mem_flat: bool,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions { vector_index: VectorIndexKind::Hnsw { m: 16, ef_construction: 100 }, flush_docs: 4096, sync: true, mem_chunk_docs: 256, mem_flat: true }
    }
}

pub struct Store {
    pub db: Arc<Db>,
    pub embedder: Arc<dyn Embedder>,
    writer: Arc<Writer>,
    /// The snapshot this handle reads, if pinned (see [`Store::pinned`]).
    snap: Option<Snapshot>,
}

impl Store {
    /// Opens the store at `path`, creating it if needed.
    pub fn open(path: &Path, embedder: Arc<dyn Embedder>, opts: StoreOptions) -> Result<Store> {
        let mut dbo = DbOptions::default();
        dbo.sync = opts.sync;
        dbo.memtable_max_docs = opts.flush_docs;
        dbo.memtable_chunk_docs = opts.mem_chunk_docs.max(64);
        dbo.publish_snapshots = true;
        if opts.mem_flat {
            dbo.mem_build = Some(dbo.build.clone().set("vector.index", "flat"));
        }
        let db = if path.exists() || strata::upgrade::old_path(path).exists() {
            // A store written before Strata had field ids is copied into
            // this version's format once, the old one kept beside it.
            if let Some(u) = strata::upgrade::upgrade(path)? {
                eprintln!("engram: copied {} records of {} into the current store format (the old file is {})", u.rows, path.display(), u.old.display());
            }
            let db = Db::open(path, dbo.clone())?;
            if db.schema().field("links").is_some() {
                db
            } else {
                drop(db);
                migrate(path, &embedder, &opts)?;
                Db::open(path, dbo)?
            }
        } else {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            Db::create(path, schema(embedder.dim(), opts.vector_index.clone())?, dbo)?
        };
        let want = embedder.dim();
        if let Some(strata::FieldType::Vector { dim, .. }) = db.schema().field("emb").map(|f| f.ty.clone()) {
            if dim != want {
                return Err(err(format!("{} holds {dim}-dimension vectors; the embedder {} makes {want}", path.display(), embedder.name())));
            }
        }
        count_entity_links(&db)?;
        let last = db.query("SELECT seq FROM memory ORDER BY seq DESC LIMIT 1", &Params::new())?;
        let next = last.rows.first().and_then(|r| r[0].as_i64()).map_or(1, |s| s + 1);
        let db = Arc::new(db);
        db.spawn_flusher();
        // One merge at a time: a daemon serves one project, and there may be
        // several daemons.
        db.spawn_merger(MergerOptions { threads: 1, ..MergerOptions::default() });
        let writer = Arc::new(Writer::start(db.clone(), next)?);
        Ok(Store { db, embedder, writer, snap: None })
    }

    /// A handle whose reads all see the current version, however much is
    /// written meanwhile: the searches of one request agree with each other.
    /// (Its writes go to the writer as usual; its reads do not see them.)
    pub fn pinned(&self) -> Store {
        Store { db: self.db.clone(), embedder: self.embedder.clone(), writer: self.writer.clone(), snap: Some(self.snapshot()) }
    }

    /// The version reads see: the pinned one, else the latest.
    pub fn snapshot(&self) -> Snapshot {
        self.snap.clone().unwrap_or_else(|| self.db.snapshot())
    }

    pub fn last_seq(&self) -> i64 {
        self.writer.next_seq.load(Ordering::SeqCst) - 1
    }

    pub fn len(&self) -> u64 {
        self.snapshot().n_live_docs()
    }

    /// Embeds and stores records with the ids and sequence numbers they
    /// have, replacing the stored ones with the same ids.
    pub fn insert(&self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = records.iter().map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs = self.embedder.embed(&refs)?;
        self.insert_embedded(records, embs)
    }

    /// Stores records with precomputed embeddings.
    pub fn insert_embedded(&self, records: &[Record], embs: Vec<Vec<f32>>) -> Result<()> {
        let recs = records.iter().cloned().zip(embs).map(|(rec, emb)| Pending { rec, emb, how: How::Replace }).collect();
        self.writer.run(Op::Put(recs)).map(|_| ())
    }

    /// Bulk-loads records (for benchmarks and transcript back-fill).
    pub fn bulk_insert(&self, records: &[Record], embs: Vec<Vec<f32>>) -> Result<()> {
        let schema = self.db.schema().clone();
        let docs: Vec<Document> = records
            .iter()
            .zip(embs)
            .map(|(r, e)| {
                let mut d = r.to_doc(e);
                schema.fill_generated(&mut d);
                d
            })
            .collect();
        self.writer.run(Op::Load(docs))?;
        if let Some(max) = records.iter().map(|r| r.seq).max() {
            self.writer.next_seq.fetch_max(max + 1, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Assigns sequence numbers and stores drafts for one session and turn.
    /// A draft's entities and files become links, and entities the store has
    /// not seen become records of their own.
    pub fn append(&self, session: &str, turn: i64, ts: i64, drafts: Vec<Draft>) -> Result<Vec<Record>> {
        let mut drafts = drafts;
        let mut names: Vec<(String, String)> = Vec::new();
        for d in &mut drafts {
            for n in std::mem::take(&mut d.entities) {
                if let Some(id) = entity_id(&n) {
                    if !d.links.iter().any(|l| l.to == id) {
                        d.links.push(Link::new("about", id.clone()));
                    }
                    names.push((id, n.trim().to_string()));
                }
            }
            if links_files(d.kind) {
                for f in &d.files {
                    let id = file_entity_id(f);
                    if !d.links.iter().any(|l| l.to == id) {
                        d.links.push(Link::new("touches", id.clone()));
                    }
                    names.push((id, f.clone()));
                }
            }
        }
        // The writer numbers them (and so gives their ids) in commit order.
        let records: Vec<Record> = drafts.into_iter().map(|d| Record::from_draft(d, session, turn, 0, ts)).collect();
        let entities = self.new_entities(names, ts)?;
        if records.is_empty() && entities.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = records.iter().chain(&entities).map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut embs = self.embedder.embed(&refs)?.into_iter();
        let mut recs: Vec<Pending> = records.into_iter().map(|rec| Pending { rec, emb: embs.next().unwrap_or_default(), how: How::New }).collect();
        recs.extend(entities.into_iter().map(|rec| Pending { rec, emb: embs.next().unwrap_or_default(), how: How::Entity }));
        Ok(self.writer.run(Op::Put(recs))?.stored)
    }

    /// Entity records for the names not stored yet (the writer checks again,
    /// against what was stored since).
    fn new_entities(&self, names: Vec<(String, String)>, ts: i64) -> Result<Vec<Record>> {
        let mut seen = std::collections::HashSet::new();
        let names: Vec<(String, String)> = names.into_iter().filter(|(id, _)| seen.insert(id.clone())).collect();
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<String> = names.iter().map(|(id, _)| id.clone()).collect();
        let have: std::collections::HashSet<String> = self.get(&ids)?.into_iter().map(|r| r.id).collect();
        Ok(names
            .into_iter()
            .filter(|(id, _)| !have.contains(id))
            .map(|(id, name)| {
                let mut r = Record::from_draft(Draft::new(Kind::Entity, name), "-", 0, 0, ts);
                r.id = id;
                r
            })
            .collect())
    }

    /// Opens the current snapshot's segment readers now, so the next query
    /// does not pay for it.
    pub fn warm(&self) {
        let _ = self.db.readers(&self.snapshot());
    }

    /// Writes everything to segments (on shutdown).
    pub fn flush(&self) -> Result<()> {
        self.writer.run(Op::Flush).map(|_| ())
    }

    /// Changes a record's tag (e.g. planned work that is now done).
    pub fn retag(&self, id: &str, tag: &str) -> Result<bool> {
        let Some(mut r) = self.get(&[id.to_string()])?.pop() else { return Ok(false) };
        r.tag = tag.to_string();
        self.insert(&[r])?;
        Ok(true)
    }

    /// Deletes a record.
    pub fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.writer.run(Op::Delete(id.to_string()))?.found)
    }

    /// Runs SQL against the memory table.
    pub fn query(&self, sql: &str) -> Result<ResultSet> {
        self.query_with(sql, &Params::new())
    }

    pub fn query_with(&self, sql: &str, params: &Params) -> Result<ResultSet> {
        Ok(self.db.query_at(&self.snapshot(), sql, params, &self.db.exec)?)
    }

    /// Records by id (any order).
    pub fn get(&self, ids: &[String]) -> Result<Vec<Record>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let rs = self.query_with(&format!("SELECT {COLUMNS} FROM memory WHERE id IN (:ids) LIMIT {}", ids.len()), &Params::new().set("ids", strs(ids)))?;
        Ok(rs.rows.iter().filter_map(|r| row_record(r)).collect())
    }

    /// The records of `session` with `seq` in `[lo, hi]`, in order.
    pub fn range(&self, session: &str, lo: i64, hi: i64) -> Result<Vec<Record>> {
        let rs = self.query_with(
            &format!("SELECT {COLUMNS} FROM memory WHERE session = :session AND seq BETWEEN {lo} AND {hi} ORDER BY seq ASC LIMIT {}", (hi - lo + 1).max(1)),
            &Params::new().set("session", Value::Str(session.to_string())),
        )?;
        Ok(rs.rows.iter().filter_map(|r| row_record(r)).collect())
    }

    /// The newest records of the given kinds.
    pub fn latest(&self, kinds: &[Kind], limit: usize, filter: &Filter) -> Result<Vec<Record>> {
        let mut f = filter.clone();
        f.kinds = kinds.to_vec();
        let rs = self.query_with(&format!("SELECT {COLUMNS} FROM memory {} ORDER BY seq DESC LIMIT {limit}", f.sql()), &f.params())?;
        Ok(rs.rows.iter().filter_map(|r| row_record(r)).collect())
    }

    /// The top `n` by BM25 over `body`, with raw scores.
    pub fn bm25_top(&self, query: &str, n: usize, filter: &Filter) -> Result<Vec<(Record, f64)>> {
        let sql = format!("SELECT {COLUMNS}, score() FROM memory {} ORDER BY bm25(body, :q, syntax => 'plain') DESC LIMIT {n}", filter.sql());
        let rs = self.query_with(&sql, &filter.params().set("q", Value::Str(query.to_string())))?;
        Ok(rs.rows.iter().filter_map(|r| Some((row_record(r)?, r.last()?.as_f64()?))).collect())
    }

    /// The records reachable from `seeds` over links within `r.hops`, both
    /// ways, passing only through the kinds `r.through` allows, best
    /// `n` by cosine similarity to `v`. Seeds are left out. A walk that
    /// would meet more than `r.max_nodes` records takes a hop fewer (Strata
    /// refuses to cut it), and none at all if one hop meets too many.
    pub fn reach(&self, seeds: &[String], v: &[f32], n: usize, filter: &Filter, r: &Reach) -> Result<Vec<(Record, f64)>> {
        if seeds.is_empty() || r.hops == 0 {
            return Ok(Vec::new());
        }
        let through = if r.through.is_empty() {
            String::new()
        } else {
            let ks: Vec<String> = r.through.iter().map(|k| format!("'{}'", k.as_str())).collect();
            format!(", through => kind IN ({})", ks.join(", "))
        };
        let params = filter.params().set("v", Value::Vector(v.to_vec().into())).set("seeds", strs(seeds));
        for hops in (1..=r.hops).rev() {
            let mut conds = filter.conds();
            conds.push(format!("REACHABLE(links, :seeds, hops => {hops}, dir => 'both', max_nodes => {}{through})", r.max_nodes));
            let sql = format!("SELECT {COLUMNS}, score() FROM memory WHERE {} ORDER BY similarity(emb, :v) DESC LIMIT {n}", conds.join(" AND "));
            let rs = match self.query_with(&sql, &params) {
                Err(e) if e.to_string().contains("max_nodes") => continue,
                rs => rs?,
            };
            let seeds: std::collections::HashSet<&str> = seeds.iter().map(|s| s.as_str()).collect();
            return Ok(rs.rows.iter().filter_map(|r| Some((row_record(r)?, r.last()?.as_f64()?))).filter(|(r, _)| !seeds.contains(r.id.as_str())).collect());
        }
        Ok(Vec::new())
    }

    /// The top `n` by cosine similarity to `v`.
    pub fn vector_top(&self, v: &[f32], n: usize, filter: &Filter) -> Result<Vec<(Record, f64)>> {
        let sql = format!("SELECT {COLUMNS}, score() FROM memory {} ORDER BY similarity(emb, :v) DESC LIMIT {n}", filter.sql());
        let rs = self.query_with(&sql, &filter.params().set("v", Value::Vector(v.to_vec().into())))?;
        Ok(rs.rows.iter().filter_map(|r| Some((row_record(r)?, r.last()?.as_f64()?))).collect())
    }

    /// The newest `n` records of `session`, newest first, with their vectors.
    pub fn tail(&self, session: &str, n: usize) -> Result<Vec<(Record, Option<Vec<f32>>)>> {
        let sql = format!("SELECT {COLUMNS}, emb FROM memory WHERE session = :session ORDER BY seq DESC LIMIT {n}");
        let rs = self.query_with(&sql, &Params::new().set("session", Value::Str(session.to_string())))?;
        Ok(rs
            .rows
            .iter()
            .filter_map(|r| {
                let v = match r.last()? {
                    Value::Vector(v) => Some(v.to_vec()),
                    _ => None,
                };
                Some((row_record(r)?, v))
            })
            .collect())
    }

    /// The top `n` by `maxsim` with `vs`: each record's best cosine
    /// similarity to any of them.
    pub fn maxsim_top(&self, vs: &[Vec<f32>], n: usize, filter: &Filter) -> Result<Vec<(Record, f64)>> {
        if vs.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!("SELECT {COLUMNS}, score() FROM memory {} ORDER BY maxsim(emb, :vs) DESC LIMIT {n}", filter.sql());
        let list = Value::List(vs.iter().map(|v| Value::Vector(v.clone().into())).collect());
        let rs = self.query_with(&sql, &filter.params().set("vs", list))?;
        Ok(rs.rows.iter().filter_map(|r| Some((row_record(r)?, r.last()?.as_f64()?))).collect())
    }

    /// Inverse document frequency of each analysed query term over `body`,
    /// from the segments' term dictionaries (deleted records still count).
    pub fn idf(&self, terms: &[String]) -> Result<Vec<f64>> {
        let readers = self.db.readers(&self.snapshot())?;
        let field = self.db.schema().field_index("body").ok_or_else(|| err("no body field"))?;
        let mut n = 0u64;
        let mut df = vec![0u64; terms.len()];
        for r in &readers {
            n += r.n_docs as u64;
            if let Some(t) = r.text(field)? {
                for (d, term) in df.iter_mut().zip(terms) {
                    *d += t.doc_freq(term) as u64;
                }
            }
        }
        let n = n.max(1) as f64;
        Ok(df.into_iter().map(|d| (1.0 + (n - d as f64 + 0.5) / (d as f64 + 0.5)).ln()).collect())
    }
}

/// A record for the writer to store.
struct Pending {
    rec: Record,
    emb: Vec<f32>,
    how: How,
}

#[derive(Clone, Copy, PartialEq)]
enum How {
    /// Takes the next sequence number, and the id that goes with it.
    New,
    /// An entity: takes the next sequence number, unless its id is stored.
    Entity,
    /// Replaces the stored record with its id.
    Replace,
}

enum Op {
    Put(Vec<Pending>),
    Delete(String),
    Load(Vec<Document>),
    Flush,
}

/// What an operation gives back.
#[derive(Default)]
struct Done {
    /// The new records it stored, numbered.
    stored: Vec<Record>,
    /// A delete found its record.
    found: bool,
}

type Job = (Op, mpsc::Sender<Result<Done>>);

/// Jobs that may share one transaction.
const MAX_GROUP: usize = 256;

/// The single writer: a thread that makes every write to the store.
struct Writer {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The next sequence number, advanced by the writer at each commit.
    next_seq: Arc<AtomicI64>,
}

impl Writer {
    fn start(db: Arc<Db>, next: i64) -> Result<Writer> {
        let next_seq = Arc::new(AtomicI64::new(next));
        let (tx, rx) = mpsc::channel();
        let seq = next_seq.clone();
        let thread = std::thread::Builder::new().name("engram-writer".into()).spawn(move || write_loop(&db, &rx, &seq))?;
        Ok(Writer { jobs: Some(tx), thread: Some(thread), next_seq })
    }

    /// Queues an operation and waits until it is committed and readers see it.
    fn run(&self, op: Op) -> Result<Done> {
        let stopped = || err("the store's writer has stopped");
        let (tx, rx) = mpsc::channel();
        self.jobs.as_ref().ok_or_else(stopped)?.send((op, tx)).map_err(|_| stopped())?;
        rx.recv().map_err(|_| stopped())?
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // The thread finishes what is queued, then ends.
        drop(self.jobs.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The writer thread. Whatever is queued when it is free is committed
/// together (one WAL sync for all of it).
fn write_loop(db: &Db, jobs: &mpsc::Receiver<Job>, next_seq: &AtomicI64) {
    while let Ok(first) = jobs.recv() {
        let mut group = Vec::new();
        for (op, reply) in std::iter::once(first).chain(std::iter::from_fn(|| jobs.try_recv().ok()).take(MAX_GROUP - 1)) {
            let r = match op {
                Op::Put(_) | Op::Delete(_) => {
                    group.push((op, reply));
                    continue;
                }
                Op::Load(docs) => {
                    commit_group(db, next_seq, std::mem::take(&mut group));
                    db.storage().bulk_load(docs)
                }
                Op::Flush => {
                    commit_group(db, next_seq, std::mem::take(&mut group));
                    db.storage().flush()
                }
            };
            let _ = reply.send(r.map(|_| Done::default()).map_err(Error::from));
        }
        commit_group(db, next_seq, group);
    }
}

/// Commits puts and deletes as one transaction and answers each. If an
/// operation is refused, nothing is committed and each job is tried on its
/// own, so that it fails only its own caller.
fn commit_group(db: &Db, next_seq: &AtomicI64, mut group: Vec<Job>) {
    if group.is_empty() {
        return;
    }
    match apply(db, next_seq, &mut group) {
        Ok(done) => {
            for ((_, reply), d) in group.into_iter().zip(done) {
                let _ = reply.send(Ok(d));
            }
        }
        Err((e, committed)) if committed || group.len() == 1 => {
            for (_, reply) in group {
                let _ = reply.send(Err(err(e.0.clone())));
            }
        }
        Err(_) => {
            for job in group {
                commit_group(db, next_seq, vec![job]);
            }
        }
    }
}

/// Applies a group in one transaction. An error says whether the commit
/// was reached (then it may have happened, and the group is not retried).
fn apply(db: &Db, next_seq: &AtomicI64, group: &mut [Job]) -> std::result::Result<Vec<Done>, (Error, bool)> {
    let refused = |e: strata::Error| (Error::from(e), false);
    let schema = db.schema().clone();
    // Every earlier commit is published, so this is the latest version.
    let snap = db.snapshot();
    let mut txn = db.storage().begin().map_err(refused)?;
    let mut next = next_seq.load(Ordering::SeqCst);
    let mut entities = HashSet::new();
    // Links each entity gains (or loses, with a deleted record).
    let mut counts: HashMap<String, f64> = HashMap::new();
    let mut out = Vec::with_capacity(group.len());
    for (op, _) in group.iter_mut() {
        let mut done = Done::default();
        match op {
            Op::Put(recs) => {
                for p in recs.iter_mut() {
                    if p.how == How::Entity && (snap.lookup(&Value::Str(p.rec.id.clone())).is_some() || !entities.insert(p.rec.id.clone())) {
                        continue; // stored already, or earlier in this group
                    }
                    if p.how != How::Replace {
                        p.rec.seq = next;
                        next += 1;
                    }
                    if p.how == How::New {
                        p.rec.id = record_id(&p.rec.session, p.rec.seq);
                    }
                    let mut doc = p.rec.to_doc(p.emb.clone());
                    schema.fill_generated(&mut doc);
                    txn.upsert(doc).map_err(refused)?;
                    if p.how == How::New {
                        count_links(&mut counts, &p.rec.links, 1.0);
                        done.stored.push(p.rec.clone());
                    }
                }
            }
            Op::Delete(id) => {
                let key = Value::Str(id.clone());
                if let Some(v) = txn.table("memory").and_then(|mut t| t.value(&key, "links")).map_err(refused)? {
                    count_links(&mut counts, &links_of(&v), -1.0);
                }
                done.found = txn.delete(&key).map_err(refused)?;
            }
            Op::Load(_) | Op::Flush => unreachable!("only puts and deletes are grouped"),
        }
        out.push(done);
    }
    for (id, n) in counts {
        if n != 0.0 {
            txn.add(&Value::Str(id), "importance", &Value::Float(n)).map_err(refused)?;
        }
    }
    let r = txn.commit();
    // A failed commit may have been logged: its numbers are not given again.
    next_seq.store(next, Ordering::SeqCst);
    r.map_err(|e| (e.into(), true))?;
    Ok(out)
}

/// Adds `by` to the count of each entity `links` points to.
fn count_links(counts: &mut HashMap<String, f64>, links: &[Link], by: f64) {
    for l in links.iter().filter(|l| is_entity(&l.to)) {
        *counts.entry(l.to.clone()).or_default() += by;
    }
}

/// Entity ids: `e:` names and `f:` files.
fn is_entity(id: &str) -> bool {
    id.starts_with("e:") || id.starts_with("f:")
}

/// Counts the links to each entity in a store written before entities kept
/// that count (their `importance` was 0), in one transaction.
fn count_entity_links(db: &Db) -> Result<()> {
    let p = Params::new();
    let any = |sql: &str| db.query(sql, &p).map(|r| !r.rows.is_empty());
    if any("SELECT id FROM memory WHERE kind = 'entity' AND importance > 0 LIMIT 1")? || !any("SELECT id FROM memory WHERE kind = 'entity' LIMIT 1")? {
        return Ok(());
    }
    let mut counts = HashMap::new();
    for r in db.query("SELECT links FROM memory WHERE kind != 'entity' LIMIT 1000000000", &p)?.rows {
        count_links(&mut counts, &r.first().map(links_of).unwrap_or_default(), 1.0);
    }
    let mut txn = db.storage().begin()?;
    for (id, n) in counts {
        txn.add(&Value::Str(id), "importance", &Value::Float(n))?;
    }
    txn.commit()?;
    Ok(())
}

/// A row projected with [`COLUMNS`] as a record.
/// Strings as a list parameter (`id IN (:ids)`, `REACHABLE(links, :seeds)`).
fn strs(xs: &[String]) -> Value {
    Value::List(xs.iter().map(|x| Value::Str(x.clone())).collect())
}

pub fn row_record(r: &[Value]) -> Option<Record> {
    let s = |i: usize| r.get(i).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let n = |i: usize| r.get(i).and_then(|v| v.as_i64()).unwrap_or(0);
    Some(Record {
        id: s(0),
        session: s(1),
        kind: Kind::parse(r.get(2)?.as_str()?)?,
        tag: s(3),
        turn: n(4),
        seq: n(5),
        ts: n(6),
        importance: r.get(7).and_then(|v| v.as_f64()).unwrap_or(0.0),
        files: s(8),
        text: s(9),
        ctx: s(10),
        links: r.get(11).map(links_of).unwrap_or_default(),
    })
}

/// Kinds whose files become links: what was decided or went wrong about a
/// file, not every read of it.
fn links_files(k: Kind) -> bool {
    k.is_knowledge() || matches!(k, Kind::Error | Kind::Episode)
}

/// Rewrites a store made before records had links into the current schema
/// (vectors are not stored, so the records are embedded again). The old
/// file is kept next to it as `<name>.v1`.
fn migrate(path: &Path, embedder: &Arc<dyn Embedder>, opts: &StoreOptions) -> Result<()> {
    let old = Db::open(path, DbOptions::default())?;
    let cols = "id, session, kind, tag, turn, seq, ts, importance, files, text, ctx";
    let rs = old.query(&format!("SELECT {cols} FROM memory ORDER BY seq ASC LIMIT 1000000000"), &Params::new())?;
    let records: Vec<Record> = rs.rows.iter().filter_map(|r| row_record(r)).collect();
    drop(old);
    let tmp = path.with_extension("migrating");
    let side = |p: &Path, suffix: &str| {
        let mut s = p.as_os_str().to_owned();
        s.push(suffix);
        std::path::PathBuf::from(s)
    };
    for p in [tmp.clone(), side(&tmp, "-wal")] {
        let _ = std::fs::remove_file(p);
    }
    {
        let new = Store::open(&tmp, embedder.clone(), StoreOptions { sync: false, ..opts.clone() })?;
        for chunk in records.chunks(4096) {
            let texts: Vec<String> = chunk.iter().map(|r| r.embed_text()).collect();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            new.bulk_insert(chunk, embedder.embed(&refs)?)?;
        }
        new.flush()?;
    }
    let bak = path.with_extension("v1");
    std::fs::rename(path, &bak)?;
    let _ = std::fs::rename(side(path, "-wal"), side(&bak, "-wal"));
    std::fs::rename(&tmp, path)?;
    let _ = std::fs::rename(side(&tmp, "-wal"), side(path, "-wal"));
    eprintln!("engram: rewrote {} records of {} with links (the old file is {})", records.len(), path.display(), bak.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;

    fn store(dir: &Path) -> Store {
        Store::open(&dir.join("m.strata"), Arc::new(HashEmbedder { dim: 64 }), StoreOptions { sync: false, ..Default::default() }).unwrap()
    }

    #[test]
    fn append_search_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.append("sess-a", 1, 100, vec![Draft::new(Kind::Prompt, "Use BM25 with k1 1.2 for the ranking")]).unwrap();
        s.append("sess-a", 1, 101, vec![Draft::new(Kind::Answer, "The planner probes local selectivity").ctx("about the planner")]).unwrap();
        s.append("sess-b", 1, 200, vec![Draft::new(Kind::Decision, "We decided the vector index is HNSW")]).unwrap();
        assert_eq!(s.len(), 3);
        let hits = s.bm25_top("ranking bm25", 10, &Filter::default()).unwrap();
        assert_eq!(hits[0].0.kind, Kind::Prompt);
        // Context is searchable.
        assert_eq!(s.bm25_top("planner", 10, &Filter::default()).unwrap().len(), 1);
        // The live window of sess-a (seq >= 2) is excluded.
        let f = Filter { exclude_window: Some(("sess-a".into(), 2)), ..Default::default() };
        assert!(s.bm25_top("planner", 10, &f).unwrap().is_empty());
        let v = s.embedder.embed_query("vector index HNSW").unwrap();
        assert_eq!(s.vector_top(&v, 1, &Filter::default()).unwrap()[0].0.kind, Kind::Decision);
        let idf = s.idf(&["planner".into(), "zzz".into()]).unwrap();
        assert!(idf[1] > idf[0]);
        drop(s);
        let s = store(dir.path());
        assert_eq!(s.last_seq(), 3);
        assert_eq!(s.range("sess-a", 1, 2).unwrap().len(), 2);
        assert_eq!(s.latest(&[Kind::Decision], 5, &Filter::default()).unwrap().len(), 1);
        assert!(s.delete("sessb:3").unwrap());
        assert_eq!(s.len(), 2);
    }

    /// Sessions, ids and entities with quotes reach Strata as query
    /// parameters, not as text quoted by hand.
    #[test]
    fn values_with_quotes_are_parameters() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let sid = "it's \"a\" session; DROP TABLE memory --";
        let p = s.append(sid, 1, 100, vec![Draft::new(Kind::Prompt, "Where is Roberto's laptop charger")]).unwrap()[0].id.clone();
        let f = s.append(sid, 1, 101, vec![Draft::new(Kind::Fact, "Roberto's laptop charger is in the drawer").entities(vec!["Roberto's laptop".into()]).link("from", p.clone())]).unwrap()[0].id.clone();
        s.append("other", 1, 200, vec![Draft::new(Kind::Fact, "The office's printer is on the second floor")]).unwrap();
        assert_eq!(s.range(sid, 1, 2).unwrap().len(), 2);
        let e = entity_id("Roberto's laptop").unwrap();
        assert!(e.contains('\''), "{e}");
        let entity = s.get(&[p.clone(), e.clone()]).unwrap();
        assert_eq!(entity.len(), 2, "{entity:?}");
        let only = Filter { session: Some(sid.into()), ..Default::default() };
        assert_eq!(s.latest(&[Kind::Fact], 5, &only).unwrap().len(), 1);
        assert_eq!(s.bm25_top("roberto's charger", 5, &only).unwrap().len(), 2);
        let window = Filter { exclude_window: Some((sid.into(), 1)), ..Default::default() };
        assert!(s.bm25_top("charger", 5, &window).unwrap().is_empty());
        let v = s.embedder.embed_query("charger").unwrap();
        assert_eq!(s.vector_top(&v, 5, &window).unwrap().len(), 1);
        let r = Reach { hops: 2, through: crate::recall::THROUGH.to_vec(), max_nodes: 100 };
        let got: Vec<String> = s.reach(&[e], &v, 10, &Filter::default(), &r).unwrap().into_iter().map(|(r, _)| r.id).collect();
        assert!(got.contains(&f), "{got:?}");
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn entities_link_records_across_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let id = |v: Vec<Record>| v[0].id.clone();
        let prompt = id(s.append("sess-a", 1, 100, vec![Draft::new(Kind::Prompt, "Build the count sheet in walking order")]).unwrap());
        let todo = id(s.append("sess-a", 1, 101, vec![Draft::new(Kind::Todo, "The count sheet is sorted by bin location").entities(vec!["Count sheet".into(), "bins".into()]).link("from", prompt.clone())]).unwrap());
        let fact_b = id(s.append("sess-b", 1, 200, vec![Draft::new(Kind::Fact, "The auditor wants a blind count sheet").entities(vec!["count sheets".into(), "auditor".into()])]).unwrap());
        let fact_c = id(s.append("sess-c", 1, 300, vec![Draft::new(Kind::Fact, "Prices are stored in integer cents").entities(vec!["prices".into()])]).unwrap());
        // One prompt, three notes and four entities (count sheet, bin,
        // auditor, price): the plural meets the singular.
        assert_eq!(s.len(), 8);
        assert_eq!(s.get(&["e:count sheet".into()]).unwrap()[0].text, "Count sheet");
        // Entities are not search results.
        assert!(s.bm25_top("count sheet", 10, &Filter::default()).unwrap().iter().all(|(r, _)| r.kind != Kind::Entity));
        // From the first note: its source turn (1 hop) and, through the
        // shared entity, the other session's note (2 hops); not the prices.
        let v = s.embedder.embed_query("count sheet").unwrap();
        let r = Reach { hops: 2, through: crate::recall::THROUGH.to_vec(), max_nodes: 100 };
        let got: Vec<String> = s.reach(&[todo.clone()], &v, 10, &Filter::default(), &r).unwrap().into_iter().map(|(r, _)| r.id).collect();
        assert!(got.contains(&prompt) && got.contains(&fact_b), "{got:?}");
        assert!(!got.iter().any(|i| i.starts_with("e:") || *i == fact_c || *i == todo), "{got:?}");
        // Two hops meet more than 3 records, one hop 3 (the prompt and two
        // entities): the walk takes one; with 2, none.
        let one = Reach { max_nodes: 3, ..r.clone() };
        let got: Vec<String> = s.reach(&[todo.clone()], &v, 10, &Filter::default(), &one).unwrap().into_iter().map(|(r, _)| r.id).collect();
        assert_eq!(got, vec![prompt.clone()]);
        assert!(s.reach(&[todo.clone()], &v, 10, &Filter::default(), &Reach { max_nodes: 2, ..r }).unwrap().is_empty());
        // A retag keeps the links.
        s.retag(&todo, "done").unwrap();
        assert_eq!(s.get(&[todo]).unwrap()[0].links.len(), 3);
    }

    /// Threads append at once (one writer commits them, in groups) while
    /// others read: numbers follow commit order, so a reader never sees a
    /// record without every earlier one, and an entity named by several
    /// appends at once is stored once.
    #[test]
    fn concurrent_appends_are_numbered_in_commit_order() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let done = std::sync::atomic::AtomicBool::new(false);
        let ids: Vec<Vec<String>> = std::thread::scope(|sc| {
            for _ in 0..2 {
                sc.spawn(|| {
                    while !done.load(Ordering::Acquire) {
                        let p = s.pinned();
                        let max = p.query("SELECT seq FROM memory ORDER BY seq DESC LIMIT 1").unwrap().rows.first().map_or(0, |r| r[0].as_i64().unwrap());
                        assert_eq!(p.len() as i64, max, "every number up to the newest is stored");
                        let n = p.query("SELECT id FROM memory WHERE kind = 'entity' AND id = 'e:count sheet' LIMIT 5").unwrap().rows.len();
                        assert!(n <= 1, "{n} copies of one entity");
                    }
                });
            }
            let ws: Vec<_> = (0..8)
                .map(|t| {
                    let s = &s;
                    sc.spawn(move || {
                        let mut ids = Vec::new();
                        for i in 0..40 {
                            let d = Draft::new(Kind::Fact, format!("Writer {t} note {i} about the count sheet")).entities(vec!["count sheet".into(), format!("bin {t}")]);
                            let got = s.append(&format!("sess-{t}"), 1, 100, vec![d, Draft::new(Kind::Prompt, format!("prompt {t} {i}"))]).unwrap();
                            assert_eq!(got.len(), 2);
                            assert_eq!(got[1].seq, got[0].seq + 1);
                            ids.extend(got.into_iter().map(|r| r.id));
                        }
                        ids
                    })
                })
                .collect();
            let ids = ws.into_iter().map(|h| h.join().unwrap()).collect();
            done.store(true, Ordering::Release);
            ids
        });
        let all: std::collections::HashSet<&String> = ids.iter().flatten().collect();
        assert_eq!(all.len(), 8 * 40 * 2);
        // 640 records, the shared entity and one bin per writer.
        assert_eq!(s.len(), 640 + 1 + 8);
        assert_eq!(s.last_seq(), 649);
        assert_eq!(s.get(&["e:count sheet".into()]).unwrap().len(), 1);
        // Every note counted once on the entities it links to.
        let links = |s: &Store, id: &str| s.get(&[id.to_string()]).unwrap()[0].importance;
        assert_eq!((links(&s, "e:count sheet"), links(&s, "e:bin 3")), (320.0, 40.0));
        assert!(s.delete(&ids[0][0]).unwrap());
        assert!(!s.delete(&ids[0][0]).unwrap());
        assert_eq!((links(&s, "e:count sheet"), links(&s, "e:bin 0")), (319.0, 39.0));
        s.flush().unwrap();
        drop(s);
        let s = store(dir.path());
        assert_eq!((s.len(), s.last_seq()), (648, 649));
        assert_eq!(links(&s, "e:count sheet"), 319.0);
    }

    /// Entities count the records that link to them, and a store whose
    /// entities have no counts yet gets them when it is opened.
    #[test]
    fn entities_count_their_links() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let d = |text: &str, ents: &[&str], files: &[&str]| Draft::new(Kind::Fact, text).entities(ents.iter().map(|e| e.to_string()).collect()).files(files.iter().map(|f| f.to_string()).collect());
        s.append("a", 1, 100, vec![d("The count sheet lists bins", &["count sheet", "bins"], &["src/sheet.py"]), d("Bins are numbered", &["bin"], &[])]).unwrap();
        s.append("b", 1, 200, vec![d("The sheet module prints the count sheet", &["Count sheets"], &["src/sheet.py"])]).unwrap();
        // A tool call names a file without linking to it.
        s.append("b", 1, 201, vec![Draft::new(Kind::ToolCall, "Read src/sheet.py").files(vec!["src/sheet.py".into()])]).unwrap();
        let counts = |s: &Store| -> Vec<(String, f64)> {
            let rs = s.query("SELECT id, importance FROM memory WHERE kind = 'entity' ORDER BY importance DESC, id LIMIT 10").unwrap();
            rs.rows.iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_f64().unwrap())).collect()
        };
        let want = vec![("e:bin".to_string(), 2.0), ("e:count sheet".into(), 2.0), ("f:src/sheet.py".into(), 2.0)];
        assert_eq!(counts(&s), want);
        // Completion: entity names and files by a prefix of their id, most
        // linked first.
        let rs = s.query("SELECT id FROM memory WHERE kind = 'entity' AND (PREFIX(id, 'e:co') OR PREFIX(id, 'f:src/')) ORDER BY importance DESC, id LIMIT 10").unwrap();
        assert_eq!(rs.rows.len(), 2);
        // With a typo, as the inspector fills its list.
        let rs = s.query("SELECT id FROM memory WHERE kind = 'entity' AND (PREFIX(id, 'e:cuont', fuzzy => 1) OR PREFIX(id, 'f:cuont', fuzzy => 1)) ORDER BY importance DESC, id LIMIT 10").unwrap();
        assert_eq!(rs.rows, vec![vec![Value::Str("e:count sheet".into())]]);
        // A store from before the counts: they are made when it opens.
        let mut txn = s.db.storage().begin().unwrap();
        for (id, _) in &want {
            txn.set(&Value::Str(id.clone()), "importance", Value::Float(0.0)).unwrap();
        }
        txn.commit().unwrap();
        assert!(counts(&s).iter().all(|(_, n)| *n == 0.0));
        drop(s);
        let s = store(dir.path());
        assert_eq!(counts(&s), want);
    }

    /// Strata's background threads flush the memtable and merge segments
    /// while appends and reads go on: nothing is lost or seen out of order,
    /// and the segments get merged without the writer doing it.
    #[test]
    fn flushes_and_merges_run_beside_the_writer() {
        let dir = tempfile::tempdir().unwrap();
        let opts = StoreOptions { sync: false, flush_docs: 64, mem_chunk_docs: 64, ..Default::default() };
        let open = || Store::open(&dir.path().join("m.strata"), Arc::new(HashEmbedder { dim: 64 }), opts.clone()).unwrap();
        let s = open();
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|sc| {
            sc.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    let p = s.pinned();
                    let max = p.query("SELECT seq FROM memory ORDER BY seq DESC LIMIT 1").unwrap().rows.first().map_or(0, |r| r[0].as_i64().unwrap());
                    assert_eq!(p.len() as i64, max, "every number up to the newest is stored");
                }
            });
            let ws: Vec<_> = (0..4)
                .map(|t| {
                    let s = &s;
                    sc.spawn(move || {
                        for i in 0..300 {
                            let drafts = vec![Draft::new(Kind::ToolCall, format!("cargo test {t} {i}")), Draft::new(Kind::ToolResult, format!("test result {t} {i}: ok"))];
                            s.append(&format!("sess-{t}"), i, 100, drafts).unwrap();
                        }
                    })
                })
                .collect();
            ws.into_iter().for_each(|h| h.join().unwrap());
            done.store(true, Ordering::Release);
        });
        assert_eq!((s.len(), s.last_seq()), (2400, 2400));
        // About 37 flushes of 64 records; the merger brings them down.
        let t = std::time::Instant::now();
        while s.snapshot().segments().len() > 15 && t.elapsed() < std::time::Duration::from_secs(60) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let segs = s.snapshot().segments().len();
        assert!((1..=15).contains(&segs), "{segs} segments");
        assert_eq!(s.bm25_top("result 3 299", 1, &Filter::default()).unwrap()[0].0.text, "test result 3 299: ok");
        drop(s);
        let s = open();
        assert_eq!((s.len(), s.last_seq()), (2400, 2400));
    }

    #[test]
    fn stores_without_links_are_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.strata");
        let mut fields = schema(64, VectorIndexKind::Flat).unwrap().fields;
        fields.retain(|f| f.name != "links");
        let old = strata::Schema::new("memory", fields, Some("id")).unwrap();
        {
            let db = Db::create(&path, old.clone(), DbOptions::default()).unwrap();
            let mut txn = db.storage().begin().unwrap();
            let r = Record::from_draft(Draft::new(Kind::Decision, "We use HNSW for the vector index"), "sess-a", 1, 1, 100);
            let mut doc = r.to_doc(vec![0.1; 64]);
            doc.pop();
            old.fill_generated(&mut doc);
            txn.upsert(doc).unwrap();
            txn.commit().unwrap();
            db.storage().flush().unwrap();
        }
        let s = store(dir.path());
        assert!(s.db.schema().field("links").is_some());
        assert_eq!(s.len(), 1);
        assert_eq!(s.latest(&[Kind::Decision], 5, &Filter::default()).unwrap()[0].text, "We use HNSW for the vector index");
        assert!(dir.path().join("m.v1").exists());
    }
}
