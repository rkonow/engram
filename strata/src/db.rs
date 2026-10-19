//! The database: storage plus the query engine, behind SQL.
//!
//! ```no_run
//! # use strata::{db::Db, query::Params, storage::DbOptions, data};
//! let db = Db::open(std::path::Path::new("papers.strata"), DbOptions::default()).unwrap();
//! let rows = db.query("SELECT id, score() FROM docs ORDER BY bm25(body, 'mrna vaccine') DESC LIMIT 10", &Params::new()).unwrap();
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::error::Result;
use crate::forward::{DocStore, KeywordColumn, NumericColumn};
use crate::query::exec::{ExecOptions, Executor, ResultSet, SegmentReaders};
use crate::query::{bind, Params, Select};
use crate::schema::{FieldType, Schema};
use crate::segment::{DocId, SegmentFile};
use crate::storage::{Database, DbOptions, DocumentSource, Snapshot};
use crate::value::{Document, Value};
use crate::vector::VectorReader;

pub struct Db {
    storage: Database,
    /// Readers by segment id. On-disk segments keep their id for life
    /// (deletes travel with the snapshot, not the readers); in-memory ones
    /// get a fresh id per rebuild.
    readers: RwLock<HashMap<u64, Arc<SegmentReaders>>>,
    pub exec: ExecOptions,
}

impl Db {
    pub fn create(path: &Path, schema: Schema, opts: DbOptions) -> Result<Db> {
        Ok(Db::wrap(Database::create(path, schema, with_source(opts))?))
    }

    pub fn open(path: &Path, opts: DbOptions) -> Result<Db> {
        Ok(Db::wrap(Database::open(path, with_source(opts))?))
    }

    fn wrap(storage: Database) -> Db {
        Db { storage, readers: RwLock::new(HashMap::new()), exec: ExecOptions::default() }
    }

    pub fn storage(&self) -> &Database {
        &self.storage
    }

    pub fn schema(&self) -> &Schema {
        self.storage.schema()
    }

    pub fn snapshot(&self) -> Snapshot {
        self.storage.snapshot()
    }

    /// Bulk-loads documents (generated columns are filled in).
    pub fn load(&self, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        let schema = self.schema().clone();
        self.storage.bulk_load(docs.into_iter().map(move |mut d| {
            schema.fill_generated(&mut d);
            d
        }))
    }

    pub fn query(&self, sql: &str, params: &Params) -> Result<ResultSet> {
        self.query_with(sql, params, &self.exec)
    }

    pub fn query_with(&self, sql: &str, params: &Params, opts: &ExecOptions) -> Result<ResultSet> {
        self.query_at(&self.snapshot(), sql, params, opts)
    }

    /// Runs a query against a given snapshot (several queries that must
    /// see the same version share one).
    pub fn query_at(&self, snap: &Snapshot, sql: &str, params: &Params, opts: &ExecOptions) -> Result<ResultSet> {
        let select = bind(sql, snap.schema(), params)?;
        self.execute(snap, &select, opts)
    }

    pub fn execute(&self, snap: &Snapshot, select: &Select, opts: &ExecOptions) -> Result<ResultSet> {
        let readers = self.readers(snap)?;
        Executor::new(snap, &readers, opts).execute(select)
    }

    /// Readers for every segment of `snap`, opened on first use.
    pub fn readers(&self, snap: &Snapshot) -> Result<Vec<Arc<SegmentReaders>>> {
        {
            let cache = self.readers.read().unwrap_or_else(|e| e.into_inner());
            if let Some(all) = snap.segments().iter().map(|v| cache.get(&v.id).cloned()).collect() {
                return Ok(all);
            }
        }
        let mut cache = self.readers.write().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(snap.segments().len());
        for v in snap.segments() {
            let r = match cache.get(&v.id) {
                Some(r) => r.clone(),
                None => {
                    let r = Arc::new(SegmentReaders::open(&v.file, snap.schema())?);
                    cache.insert(v.id, r.clone());
                    r
                }
            };
            out.push(r);
        }
        if cache.len() > 2 * snap.segments().len() + 16 {
            let live: std::collections::HashSet<u64> = snap.segments().iter().map(|v| v.id).collect();
            cache.retain(|id, _| live.contains(id));
        }
        Ok(out)
    }

    /// Drops cached readers (e.g. before cold-cache measurements).
    pub fn clear_readers(&self) {
        self.readers.write().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

fn with_source(mut opts: DbOptions) -> DbOptions {
    if opts.doc_source.is_none() {
        opts.doc_source = Some(Arc::new(StoredSource));
    }
    opts
}

/// Rebuilds documents for merges: stored fields from the doc store, vectors
/// and unstored structured fields from their indexes, generated text from
/// its sources.
pub struct StoredSource;

impl DocumentSource for StoredSource {
    fn documents(&self, schema: &Schema, seg: &SegmentFile, docs: &[DocId]) -> Result<Vec<Document>> {
        let store = if schema.fields.iter().any(|f| f.stored) { Some(DocStore::open(seg, schema)?) } else { None };
        let mut vectors = Vec::new();
        let mut numeric = Vec::new();
        let mut keyword = Vec::new();
        let mut edges = Vec::new();
        for (i, f) in schema.fields.iter().enumerate() {
            match f.ty {
                FieldType::Edges if !f.stored && f.indexed => {
                    if let Some(g) = crate::graph::GraphReader::open(seg, &f.name)? {
                        edges.push((i, g));
                    }
                }
                FieldType::Vector { .. } => vectors.push((i, VectorReader::open(seg, &f.name)?)),
                FieldType::Int | FieldType::Float if !f.stored && f.indexed => numeric.push((i, NumericColumn::open(seg, &f.name)?)),
                FieldType::Keyword if !f.stored && f.indexed => keyword.push((i, KeywordColumn::open(seg, &f.name)?)),
                _ => {}
            }
        }
        let mut out = Vec::with_capacity(docs.len());
        for &d in docs {
            let mut doc = match &store {
                Some(s) => s.get(d),
                None => vec![Value::Null; schema.fields.len()],
            };
            for (i, v) in &vectors {
                if v.has_vector(d) {
                    doc[*i] = Value::Vector(v.vector(d).into());
                }
            }
            for (i, c) in &numeric {
                doc[*i] = c.get(d);
            }
            for (i, k) in &keyword {
                doc[*i] = k.value_of_doc(d).map_or(Value::Null, |v| Value::Str(v.to_string()));
            }
            for (i, g) in &edges {
                doc[*i] = g.edges_value(d);
            }
            schema.fill_generated(&mut doc);
            out.push(doc);
        }
        Ok(out)
    }
}
