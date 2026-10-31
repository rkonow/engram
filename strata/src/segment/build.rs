//! Turning documents into a segment.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use crate::error::{Error, Result};
use crate::schema::{FieldType, Schema};
use crate::segment::{DocId, FieldIndexBuilder, SegmentFile, SegmentWriter};
use crate::forward::StoreBuilder;
use crate::graph::{EdgeTableBuilder, GraphBuilder};
use crate::storage::KeyList;
use crate::text::TextFieldBuilder;
use crate::value::Document;

/// Build-time knobs, mostly for experiments that compare encodings
/// (e.g. `text.postings = loom | bp128`). Components document their keys.
#[derive(Clone, Debug, Default)]
pub struct BuildOptions {
    pub params: BTreeMap<String, String>,
}

impl BuildOptions {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }
    pub fn get_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.get(key).unwrap_or(default)
    }
    pub fn set(mut self, key: &str, value: impl ToString) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
}

/// Seconds spent in each part of segment builds since the last
/// [`take_build_times`], summed over all builds and threads.
static BUILD_TIMES: Mutex<BTreeMap<String, f64>> = Mutex::new(BTreeMap::new());

fn add_build_time(part: &str, since: Instant) {
    let mut t = BUILD_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    *t.entry(part.to_string()).or_insert(0.0) += since.elapsed().as_secs_f64();
}

/// Seconds spent in each part of segment builds since the last call, by
/// part: `<field> (<type>)` for an index (`emb (vector)` includes the SQ8
/// codes and the ANN graph), `doc store`, `keys`. Resets the counters.
pub fn take_build_times() -> BTreeMap<String, f64> {
    std::mem::take(&mut *BUILD_TIMES.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Builds of fewer documents run on the calling thread alone.
const PARALLEL_MIN_DOCS: usize = 1024;
/// A TEXT field is analysed in runs of at least this many documents, one
/// run per thread.
const TEXT_RUN_DOCS: usize = 4096;

/// Threads one segment build uses: `build.threads`, by default the
/// machine's cores, at most 8. Fields build at once, and TEXT fields analyse
/// runs of documents at once; the image is the same bytes whatever the
/// count. A vector field's graph uses `vector.threads` of its own.
pub fn build_threads(opts: &BuildOptions) -> Result<usize> {
    match opts.get("build.threads") {
        Some(s) => s.parse().ok().filter(|&n: &usize| n > 0).ok_or_else(|| Error::Schema(format!("build.threads: bad value {s:?}"))),
        None => {
            // Asked once: on Linux it reads the cgroup's CPU quota.
            static CORES: OnceLock<usize> = OnceLock::new();
            Ok(*CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)))
        }
    }
}

/// A field's index under construction.
enum Slot {
    /// A builder given every document's value.
    Index(Box<dyn FieldIndexBuilder>),
    /// An EDGES field, which needs every document's key when it is written.
    Graph(GraphBuilder),
    /// An edge table's graph, of each row's source, target and weight.
    Edges(EdgeTableBuilder),
    /// A TEXT field, analysed in runs of documents (as many per batch as
    /// threads), joined when it is written.
    Text(Vec<Option<TextFieldBuilder>>),
}

/// One piece of a batch, run on any thread.
enum Task {
    /// The batch's values for slot `.0`.
    Slot(usize),
    /// Documents `range` of the batch for TEXT slot `slot`, whose run
    /// `run` it is.
    TextRun { slot: usize, run: usize, range: Range<usize>, builder: TextFieldBuilder },
    /// Writes TEXT slot `.0` from runs of earlier batches (a last batch
    /// with no documents).
    TextJoin(usize),
    DocStore,
    Keys,
}

fn kind(ty: &FieldType) -> &'static str {
    match ty {
        FieldType::Text { .. } => "text",
        FieldType::Keyword => "keyword",
        FieldType::Int => "int",
        FieldType::Float => "float",
        FieldType::Vector { .. } => "vector",
        FieldType::Sparse => "sparse",
        FieldType::Edges => "edges",
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Builds the segment image for `docs` (local docIDs follow slice order) on
/// [`build_threads`] threads.
pub fn build_segment(schema: &Schema, docs: &[Document], opts: &BuildOptions) -> Result<Vec<u8>> {
    build_segment_from(schema, docs, opts, &[])
}

/// A vector field's HNSW graph to start a build from: `segment`'s graph of
/// the field, whose documents are the build's from docID `base` on, all of
/// them, with the same vectors. A merge names every input that lost no
/// documents, the largest first: the first graph that fits keeps its links
/// and the others are joined to it.
#[derive(Clone)]
pub struct GraphStart {
    pub field: usize,
    pub segment: SegmentFile,
    pub base: u32,
}

/// [`build_segment`], with the vector graphs `starts` names started from
/// those segments' graphs (ignored where the field's graph does not fit:
/// another `M`, no graph, other documents).
pub fn build_segment_from(schema: &Schema, docs: &[Document], opts: &BuildOptions, starts: &[GraphStart]) -> Result<Vec<u8>> {
    Ok(SegmentBuild::with_starts(schema, opts, docs.len(), starts)?.finish(docs)?.finish())
}

/// A segment built from batches of documents in docID order, so that a
/// merge holds one batch of decoded documents at a time instead of all of
/// them: the same image as [`build_segment`] over all of them. Each batch
/// builds on [`build_threads`] threads, its fields at once and TEXT fields
/// in runs of documents; the last batch goes to [`SegmentBuild::finish`],
/// which also writes the indexes.
pub struct SegmentBuild<'a> {
    schema: &'a Schema,
    opts: &'a BuildOptions,
    threads: usize,
    n_docs: usize,
    added: usize,
    /// The field of each slot, and the slot (taken when written).
    fields: Vec<usize>,
    slots: Vec<Mutex<Option<Slot>>>,
    store: Mutex<Option<StoreBuilder>>,
    /// Every document's primary key so far, for the key index and EDGES
    /// fields.
    keys: KeyList,
}

impl<'a> SegmentBuild<'a> {
    /// A build of `n_docs` documents in all.
    pub fn new(schema: &'a Schema, opts: &'a BuildOptions, n_docs: usize) -> Result<Self> {
        Self::with_starts(schema, opts, n_docs, &[])
    }

    /// A build of `n_docs` documents in all, with the vector graphs
    /// `starts` names started from those segments' graphs
    /// ([`build_segment_from`]).
    pub fn with_starts(schema: &'a Schema, opts: &'a BuildOptions, n_docs: usize, starts: &[GraphStart]) -> Result<Self> {
        let threads = build_threads(opts)?;
        let threads = if n_docs < PARALLEL_MIN_DOCS { 1 } else { threads };
        let (mut fields, mut slots) = (Vec::new(), Vec::new());
        for (i, f) in schema.fields.iter().enumerate() {
            let slot = match &f.ty {
                FieldType::Text { .. } if f.indexed => {
                    // Checks the field's options now rather than in a batch.
                    crate::text::text_builder(f, opts)?;
                    Some(Slot::Text(Vec::new()))
                }
                FieldType::Sparse if f.indexed => Some(Slot::Index(crate::text::sparse::field_builder(f, opts)?)),
                FieldType::Keyword | FieldType::Int | FieldType::Float if f.indexed => Some(Slot::Index(crate::forward::field_builder(f, opts)?)),
                // Vector fields always store their vectors; `indexed` adds the ANN index.
                FieldType::Vector { .. } => {
                    let from: Vec<_> = starts.iter().filter(|g| g.field == i).map(|g| (g.segment.clone(), g.base)).collect();
                    Some(Slot::Index(if from.is_empty() { crate::vector::field_builder(f, opts)? } else { crate::vector::field_builder_from(f, opts, from)? }))
                }
                FieldType::Edges if f.indexed => Some(Slot::Graph(crate::graph::keyless_builder(f, schema)?)),
                _ => None,
            };
            if let Some(slot) = slot {
                fields.push(i);
                slots.push(Mutex::new(Some(slot)));
            }
        }
        if let (Some(b), Some((src, ..))) = (crate::graph::edge_table::builder(schema), schema.edge_fields()) {
            fields.push(src);
            slots.push(Mutex::new(Some(Slot::Edges(b))));
        }
        let store = Mutex::new(Some(StoreBuilder::new(schema, opts)?));
        let keys = match schema.primary_key {
            Some(pk) => KeyList::new(&schema.fields[pk].ty)?,
            None => KeyList::of(false),
        };
        Ok(SegmentBuild { schema, opts, threads, n_docs, added: 0, fields, slots, store, keys })
    }

    /// Adds the next documents.
    pub fn add(&mut self, docs: &[Document]) -> Result<()> {
        if !docs.is_empty() {
            self.batch(docs, false)?;
        }
        Ok(())
    }

    /// Adds the last documents (any number, none included) and writes the
    /// segment's sections.
    pub fn finish(mut self, docs: &[Document]) -> Result<SegmentWriter> {
        let built = self.batch(docs, true)?;
        // A TEXT field's runs are adjacent tasks, so its sections, written
        // by one of them, land where the field's go.
        let mut out = SegmentWriter::new(self.n_docs);
        for s in built.into_iter().flatten() {
            out.append(s);
        }
        if !self.schema.is_legacy() {
            out.put(super::FIELDS, super::encode_field_ids(self.schema));
        }
        Ok(out)
    }

    /// Runs a batch's tasks; on the last batch, each task's sections, in
    /// the order they are written (none for the TEXT runs whose thread did
    /// not write their field).
    fn batch(&mut self, docs: &[Document], last: bool) -> Result<Vec<Option<SegmentWriter>>> {
        for d in docs {
            crate::value::validate(self.schema, d)?;
        }
        let total = self.added + docs.len();
        if total > self.n_docs || (last && total != self.n_docs) {
            return Err(Error::Schema(format!("a segment build of {} documents was given {total}", self.n_docs)));
        }
        if let Some(pk) = self.schema.primary_key {
            for d in docs {
                self.keys.push(&d[pk])?;
            }
        }
        let (schema, threads, n_docs, base) = (self.schema, self.threads, self.n_docs, self.added);
        // In the order their sections are written.
        let mut tasks: Vec<Task> = Vec::new();
        for (j, slot) in self.slots.iter_mut().enumerate() {
            match slot.get_mut().unwrap_or_else(|e| e.into_inner()) {
                Some(Slot::Text(runs)) if docs.is_empty() && !runs.is_empty() => tasks.push(Task::TextJoin(j)),
                Some(Slot::Text(runs)) => {
                    let n = (docs.len() / TEXT_RUN_DOCS).clamp(1, threads);
                    let per = docs.len().div_ceil(n).max(1);
                    for at in (0..docs.len().max(1)).step_by(per) {
                        let builder = crate::text::text_builder(&schema.fields[self.fields[j]], self.opts)?;
                        tasks.push(Task::TextRun { slot: j, run: runs.len(), range: at..(at + per).min(docs.len()), builder });
                        runs.push(None);
                    }
                }
                _ => tasks.push(Task::Slot(j)),
            }
        }
        tasks.push(Task::DocStore);
        if last {
            tasks.push(Task::Keys);
        }
        let (fields, slots, store, keys) = (&self.fields, &self.slots, &self.store, &self.keys);
        let run = |task: Task| -> Result<Option<SegmentWriter>> {
            let t = Instant::now();
            let mut out = SegmentWriter::new(n_docs);
            let name = match task {
                Task::Slot(j) => {
                    let i = fields[j];
                    let mut slot = lock(&slots[j]);
                    match slot.as_mut().expect("a slot is taken by the last batch only") {
                        Slot::Index(b) => {
                            for (d, doc) in docs.iter().enumerate() {
                                b.add((base + d) as DocId, &doc[i])?;
                            }
                        }
                        Slot::Graph(b) => {
                            for (d, doc) in docs.iter().enumerate() {
                                b.add((base + d) as DocId, &doc[i])?;
                            }
                        }
                        Slot::Edges(b) => {
                            for (d, doc) in docs.iter().enumerate() {
                                b.add((base + d) as DocId, doc)?;
                            }
                        }
                        Slot::Text(_) => unreachable!("TEXT slots run as runs"),
                    }
                    let edges = matches!(slot.as_ref(), Some(Slot::Edges(_)));
                    if last {
                        match slot.take() {
                            Some(Slot::Index(b)) => b.finish(&mut out)?,
                            Some(Slot::Graph(b)) => b.finish_with_keys(keys, &mut out)?,
                            Some(Slot::Edges(b)) => b.finish(&mut out)?,
                            _ => unreachable!("taken once"),
                        }
                    }
                    match edges {
                        true => "edge table".to_string(),
                        false => format!("{} ({})", schema.fields[i].name, kind(&schema.fields[i].ty)),
                    }
                }
                Task::TextRun { slot: j, run, range, mut builder } => {
                    let i = fields[j];
                    for (d, doc) in docs[range].iter().enumerate() {
                        builder.add(d as DocId, &doc[i])?;
                    }
                    builder.seal();
                    if !last {
                        // Held until the last batch's runs are done.
                        builder.pack();
                    }
                    let name = format!("{} (text)", schema.fields[i].name);
                    add_build_time(&name, t);
                    let mut slot = lock(&slots[j]);
                    let Some(Slot::Text(runs)) = slot.as_mut() else { unreachable!("a TEXT slot") };
                    runs[run] = Some(builder);
                    if !last || runs.iter().any(Option::is_none) {
                        return Ok(None);
                    }
                    let runs: Vec<TextFieldBuilder> = runs.drain(..).flatten().collect();
                    drop(slot);
                    let t = Instant::now();
                    TextFieldBuilder::write_joined(runs, threads, &mut out)?;
                    add_build_time(&name, t);
                    return Ok(Some(out));
                }
                Task::TextJoin(j) => {
                    let Some(Slot::Text(runs)) = lock(&slots[j]).take() else { unreachable!("a TEXT slot") };
                    TextFieldBuilder::write_joined(runs.into_iter().flatten().collect(), threads, &mut out)?;
                    format!("{} (text)", schema.fields[fields[j]].name)
                }
                Task::DocStore => {
                    let mut b = lock(store);
                    for doc in docs {
                        b.as_mut().expect("taken by the last batch only").add(doc);
                    }
                    if last {
                        b.take().expect("taken once").finish(&mut out);
                    }
                    "doc store".to_string()
                }
                Task::Keys => {
                    crate::storage::build_pk_index_of(schema, keys, &mut out)?;
                    "keys".to_string()
                }
            };
            add_build_time(&name, t);
            Ok(last.then_some(out))
        };
        // Longest first, so that no thread starts a vector graph last.
        let cost = |t: &Task| match t {
            Task::Slot(j) if matches!(schema.fields[fields[*j]].ty, FieldType::Vector { .. }) => 0,
            Task::TextRun { .. } | Task::TextJoin(_) => 1,
            Task::DocStore => 2,
            _ => 3,
        };
        let mut order: Vec<usize> = (0..tasks.len()).collect();
        order.sort_by_key(|&k| cost(&tasks[k]));
        let n_tasks = tasks.len();
        let tasks: Vec<Mutex<Option<Task>>> = tasks.into_iter().map(|t| Mutex::new(Some(t))).collect();
        let built: Vec<Mutex<Option<Result<Option<SegmentWriter>>>>> = (0..n_tasks).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        let work = || loop {
            let Some(&k) = order.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
            let task = lock(&tasks[k]).take().expect("each task runs once");
            let r = run(task);
            *lock(&built[k]) = Some(r);
        };
        if threads > 1 {
            std::thread::scope(|s| {
                for _ in 0..threads.min(n_tasks) {
                    s.spawn(work);
                }
            });
        } else {
            work();
        }
        self.added = total;
        built.into_iter().map(|b| b.into_inner().unwrap_or_else(|e| e.into_inner()).expect("every task ran")).collect()
    }
}
