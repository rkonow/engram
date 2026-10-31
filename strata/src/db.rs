//! The database: storage plus the query engine, behind SQL.
//!
//! ```no_run
//! # use strata::{db::Db, query::Params, storage::DbOptions, data};
//! let db = Db::open(std::path::Path::new("papers.strata"), DbOptions::default()).unwrap();
//! let rows = db.query("SELECT id, score() FROM docs ORDER BY bm25(body, 'mrna vaccine') DESC LIMIT 10", &Params::new()).unwrap();
//! ```
//!
//! SQL reads and writes every table of the database by name, and creates
//! and drops tables (`CREATE TABLE`, `DROP TABLE`); a query reads one table.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::derive::Deriver;
use crate::embed::{Embedder, Input, Modality};
use crate::error::{Error, Result};
use crate::forward::{DocStore, KeywordColumn, NumericColumn};
use crate::media::{Media, Resolver};
use crate::query::agg::Part;
use crate::query::exec::{ExecOptions, ExecStats, Executor, ResultSet, SegmentReaders};
use crate::query::ir::eval::{self, Ref};
use crate::query::ir::Plan;
use crate::query::model::Models;
use crate::query::stats::TermStats;
use crate::query::{bind_command, compound, ops, profile, sql as qsql, Assign, Command, Conflict, Params, Select, Write};
use crate::rank_log::{self, Export, Log, Logged};
use crate::schema::{Derive, Embed, FieldType, Schema};
use crate::segment::{DocId, SegmentFile};
use crate::storage::{Database, DbOptions, DocumentSource, DropOptions, Mask, Snapshot, TableTxn, SOURCES};
use crate::value::{Document, Value};
use crate::vector::VectorReader;

/// What [`Db::sql`] returns.
#[derive(Debug)]
pub enum SqlResult {
    Rows(ResultSet),
    /// Documents inserted, updated or deleted (0 for `CREATE TABLE` and
    /// `DROP TABLE`).
    Written(u64),
}

/// An INSERT, UPDATE or DELETE bound to its table, with the vectors it
/// writes computed ([`Db::prepare_write`]), to run with
/// [`Db::write_prepared`].
#[derive(Clone, Debug)]
pub struct PreparedWrite {
    schema: Arc<Schema>,
    write: Write,
    known: Known,
    derived: Derived,
    /// Rows an UPDATE or DELETE passes over ([`PreparedWrite::masked`]).
    mask: Option<Arc<Mask>>,
    /// Asked as the write runs ([`PreparedWrite::guarded`]).
    guard: Option<Guard>,
}

/// A check a write passes as it runs, under the writer lock, answering the
/// rows an UPDATE or DELETE passes over then ([`PreparedWrite::guarded`]).
#[derive(Clone)]
struct Guard(Arc<dyn Fn() -> Result<Option<Arc<Mask>>> + Send + Sync>);

impl std::fmt::Debug for Guard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Guard")
    }
}

impl PreparedWrite {
    /// The primary keys of the documents an INSERT writes; none for an
    /// UPDATE or a DELETE, which change rows already in the table (and
    /// cannot change a key).
    pub fn inserted_keys(&self) -> Vec<&Value> {
        match (&self.write, self.schema.primary_key) {
            (Write::Insert { docs, .. }, Some(pk)) => docs.iter().map(|d| &d[pk]).collect(),
            _ => Vec::new(),
        }
    }

    /// The values that place the documents an INSERT writes on shards:
    /// their keys ([`Self::inserted_keys`]), or for a table placed by a
    /// column ([`Schema::with_shard_by`]) that column's values.
    pub fn inserted_places(&self) -> Vec<&Value> {
        match (&self.write, self.schema.shard_by_field()) {
            (Write::Insert { docs, .. }, Some(f)) => docs.iter().map(|d| &d[f]).collect(),
            _ => self.inserted_keys(),
        }
    }

    /// The table it writes.
    pub fn table(&self) -> &Schema {
        &self.schema
    }

    /// The same UPDATE or DELETE passing over the rows `mask` hides
    /// ([`Snapshot::masked`]): a shard runs a statement over some of its
    /// slots so. Refused for an INSERT, whose rows are its own.
    pub fn masked(mut self, mask: Arc<Mask>) -> Result<PreparedWrite> {
        if matches!(self.write, Write::Insert { .. }) {
            return Err(Error::Query("an INSERT writes its own rows: only an UPDATE or DELETE passes over some".into()));
        }
        self.mask = Some(mask);
        Ok(self)
    }

    /// The same write running only if `guard` passes, asked under the
    /// writer lock as the write starts; an UPDATE or DELETE passes over
    /// the rows of the mask it answers too (an INSERT's rows are its own).
    /// A shard checks so that the rows it holds without serving, and the
    /// slots it serves, are still those the statement was routed by.
    pub fn guarded(mut self, guard: impl Fn() -> Result<Option<Arc<Mask>>> + Send + Sync + 'static) -> PreparedWrite {
        self.guard = Some(Guard(Arc::new(guard)));
        self
    }
}

/// Vectors computed ahead of a write, by embedded field and source text.
type Known = HashMap<usize, HashMap<String, Arc<[f32]>>>;

/// Text made ahead of a write, by derived field and the MEDIA value it is
/// made from.
type Derived = HashMap<usize, HashMap<String, String>>;

/// A stream's source, as the system table [`SOURCES`] keeps it: the
/// offset of the last event written from it, and when that was written
/// (microseconds since the epoch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub name: String,
    pub offset: i64,
    pub updated: Option<i64>,
}

pub struct Db {
    storage: Arc<Database>,
    /// Readers by segment id, with the address of the image they read.
    /// On-disk segments keep their id for life (deletes travel with the
    /// snapshot, not the readers), but a vacuum can move their bytes;
    /// in-memory ones get a fresh id per rebuild.
    readers: RwLock<HashMap<u64, (usize, Arc<SegmentReaders>)>>,
    pub exec: ExecOptions,
    /// Models of embedded VECTOR fields ([`Db::set_embedder_in`]).
    embedders: RwLock<Vec<Model>>,
    /// Models of derived TEXT fields ([`Db::set_deriver_in`]).
    derivers: RwLock<Vec<Deriving>>,
    /// Where [`Db::open_embedders`] found models, for the tables created
    /// after it, and where ranking models are ([`Db::open_ranking_models`]).
    models: RwLock<Option<PathBuf>>,
    /// The ranking models queries have used.
    ranking: Models,
    /// How writes resolve MEDIA values ([`Db::set_media_resolver`]).
    media: RwLock<Arc<Resolver>>,
}

/// The model of an embedded VECTOR field, and the field it was set for: it
/// serves a field of that name, in a table of that name, with the same
/// embedding and dimensions (so a table dropped and created again with
/// another embedding does not get it).
struct Model {
    table: String,
    field: String,
    embed: Embed,
    dim: usize,
    model: Arc<dyn Embedder>,
}

/// The model of a derived TEXT field, and the field it was set for (as
/// [`Model`] is).
struct Deriving {
    table: String,
    field: String,
    derive: Derive,
    model: Arc<dyn Deriver>,
}

impl Db {
    /// A new database of one table. Its semantic-ID fields' codebooks
    /// come from the default models directory ([`Db::create_in`]).
    pub fn create(path: &Path, schema: Schema, opts: DbOptions) -> Result<Db> {
        Db::create_in(path, schema, opts, &crate::embed::models_dir())
    }

    /// [`Db::create`], with the codebooks of semantic-ID fields from
    /// `models`: the table records each one's hash ([`crate::sid::resolve`]),
    /// and writes get codes from it.
    pub fn create_in(path: &Path, mut schema: Schema, opts: DbOptions, models: &Path) -> Result<Db> {
        crate::sid::resolve(&mut schema, models)?;
        let db = Db::wrap(Database::create(path, schema, with_source(opts))?);
        db.open_codebooks(models);
        Ok(db)
    }

    /// Opens a database; the codebooks of its semantic-ID fields come from
    /// the default models directory, or from [`Db::open_embedders`]'s.
    pub fn open(path: &Path, opts: DbOptions) -> Result<Db> {
        let db = Db::wrap(Database::open(path, with_source(opts))?);
        db.open_codebooks(&crate::embed::models_dir());
        Ok(db)
    }

    fn wrap(storage: Database) -> Db {
        Db {
            storage: Arc::new(storage),
            readers: RwLock::new(HashMap::new()),
            exec: ExecOptions::default(),
            embedders: RwLock::new(Vec::new()),
            derivers: RwLock::new(Vec::new()),
            models: RwLock::new(None),
            media: RwLock::new(Arc::new(Resolver::default())),
            ranking: Models::default(),
        }
    }

    pub fn storage(&self) -> &Database {
        &self.storage
    }

    /// Flushes in the background from now on ([`Database::spawn_flusher`]).
    pub fn spawn_flusher(&self) {
        self.storage.spawn_flusher();
    }

    /// Merges in the background from now on ([`Database::spawn_merger`]).
    pub fn spawn_merger(&self, opts: crate::storage::MergerOptions) {
        self.storage.spawn_merger(opts);
    }

    /// The schema of the database's first table, which the methods here
    /// that name no table write.
    pub fn schema(&self) -> &Schema {
        self.storage.schema()
    }

    /// The schemas of the database's tables, the first one first
    /// ([`Database::tables`]).
    pub fn catalog(&self) -> Vec<Arc<Schema>> {
        self.storage.tables()
    }

    /// The stream sources written from ([`Database::ensure_sources`]), by
    /// name, with the offset of the last event each wrote; none before
    /// the first.
    pub fn sources(&self) -> Result<Vec<Source>> {
        let snap = self.snapshot();
        if snap.table(SOURCES).is_none() {
            return Ok(Vec::new());
        }
        let sql = format!("SELECT name, last_offset, updated FROM {SOURCES} LIMIT {}", u32::MAX);
        let rs = self.query_at(&snap, &sql, &Params::new(), &ExecOptions { max_rows: None, ..self.exec.clone() })?;
        let mut sources: Vec<Source> = rs
            .rows
            .into_iter()
            .filter_map(|r| {
                let mut r = r.into_iter();
                let name = match r.next()? {
                    Value::Str(s) => s,
                    _ => return None,
                };
                let offset = r.next()?.as_i64()?;
                let updated = r.next().and_then(|v| match v {
                    Value::Timestamp(t) | Value::Int(t) => Some(t),
                    _ => None,
                });
                Some(Source { name, offset, updated })
            })
            .collect();
        sources.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(sources)
    }

    /// The schema of the table named `table` (any case).
    fn table(&self, table: &str) -> Result<Arc<Schema>> {
        self.storage.table(table).ok_or_else(|| Error::Schema(format!("no table {table}")))
    }

    /// Adds an empty table ([`Database::create_table`], SQL `CREATE
    /// TABLE`). SQL reads and writes it by name, and so do
    /// [`crate::storage::WriteTxn::table`] and [`Snapshot::table`]. After
    /// [`Db::open_embedders`], its embedded fields get their models from the
    /// same place (a model it cannot open leaves the vectors to the writes).
    pub fn create_table(&self, mut schema: Schema) -> Result<()> {
        let name = schema.table.clone();
        let dir = self.models.read().unwrap_or_else(|e| e.into_inner()).clone();
        let models = dir.clone().unwrap_or_else(crate::embed::models_dir);
        crate::sid::resolve(&mut schema, &models)?;
        self.storage.create_table(schema)?;
        if let Some(dir) = dir {
            self.open_embedders_of(&name, &dir);
        }
        self.open_codebooks_of(&name, &models);
        Ok(())
    }

    /// Drops a table other than the first ([`Database::drop_table`]).
    pub fn drop_table(&self, name: &str) -> Result<()> {
        self.drop_tables(&[name], DropOptions::default())
    }

    /// Drops tables ([`Database::drop_tables`], SQL `DROP TABLE`).
    pub fn drop_tables(&self, names: &[&str], opts: DropOptions) -> Result<()> {
        let r = self.storage.drop_tables(names, opts);
        let tables = self.catalog();
        self.embedders.write().unwrap_or_else(|e| e.into_inner()).retain(|m| tables.iter().any(|t| t.table.eq_ignore_ascii_case(&m.table)));
        r
    }

    pub fn snapshot(&self) -> Snapshot {
        self.storage.snapshot()
    }

    /// Bulk-loads documents into the first table (generated columns are
    /// filled in, and embedded vectors the documents leave out are
    /// computed).
    pub fn load(&self, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        self.try_load(docs.into_iter().map(Ok))
    }

    /// [`Db::load`] of documents that may fail to arrive (rows read from a
    /// file): the first error ends the load, and nothing is committed.
    pub fn try_load(&self, docs: impl IntoIterator<Item = Result<Document>>) -> Result<()> {
        let first = self.schema().table.clone();
        self.try_load_into(&first, docs)
    }

    /// [`Db::load`] into the table named `table`.
    pub fn load_into(&self, table: &str, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        self.try_load_into(table, docs.into_iter().map(Ok))
    }

    /// [`Db::try_load`] into the table named `table`.
    pub fn try_load_into(&self, table: &str, docs: impl IntoIterator<Item = Result<Document>>) -> Result<()> {
        let schema = self.table(table)?;
        let models = self.models_of(&schema);
        let derivers = self.derivers_of(&schema);
        let media = self.media_resolver();
        let mut docs = docs.into_iter();
        if models.iter().all(Option::is_none) && !schema.fields.iter().any(|f| f.media.is_some()) {
            return self.storage.try_bulk_load_into(table, docs.map(move |d| {
                d.map(|mut d| {
                    schema.fill_generated(&mut d);
                    d
                })
            }));
        }
        // Resolved and embedded in batches, so that fetches run together
        // and a model sees many texts at once; fewer rows of images, whose
        // bytes are held until a model has read them.
        let batch_size = if (0..schema.fields.len()).any(|f| read_by_model(&schema, f)) { MEDIA_BATCH } else { EMBED_BATCH };
        let mut ready: std::vec::IntoIter<Result<Document>> = Vec::new().into_iter();
        self.storage.try_bulk_load_into(
            table,
            std::iter::from_fn(|| loop {
                if let Some(d) = ready.next() {
                    return Some(d);
                }
                let mut batch = Vec::new();
                for d in docs.by_ref().take(batch_size) {
                    match d {
                        Ok(d) => batch.push(d),
                        Err(e) => {
                            // The rows before the error are not loaded either.
                            return Some(Err(e));
                        }
                    }
                }
                if batch.is_empty() {
                    return None;
                }
                batch.iter_mut().for_each(|d| schema.fill_generated(d));
                let embedded = resolve_media(&media, &schema, &mut batch).and_then(|held| {
                    let files = Files { resolver: &media, held };
                    derive_known(&derivers, &schema, &mut batch, &Derived::new(), &files)?;
                    embed_with(&models, &schema, &mut batch, &files)
                });
                if let Err(e) = embedded {
                    return Some(Err(e));
                }
                ready = batch.into_iter().map(Ok).collect::<Vec<_>>().into_iter();
            }),
        )
    }

    /// Every live document of `snap` as merges rebuild them (stored fields,
    /// vectors, values written in place, generated text), segment by
    /// segment. Fields neither stored nor recoverable from an index are null.
    pub fn documents<'a>(&self, snap: &'a Snapshot) -> impl Iterator<Item = Result<Document>> + 'a {
        let (mut seg, mut next) = (0usize, 0 as DocId);
        let mut ready = Vec::new().into_iter();
        std::iter::from_fn(move || loop {
            if let Some(d) = ready.next() {
                return Some(Ok(d));
            }
            let v = snap.segments().get(seg)?;
            let n = v.file.n_docs() as DocId;
            if next >= n {
                (seg, next) = (seg + 1, 0);
                continue;
            }
            let end = n.min(next.saturating_add(4096));
            let live: Vec<DocId> = (next..end).filter(|&d| v.live.as_ref().is_none_or(|l| l.contains(d))).collect();
            next = end;
            match StoredSource.documents(snap.schema(), &v.file, &live) {
                Ok(mut docs) => {
                    if let Some(u) = &v.updates {
                        for (doc, &d) in docs.iter_mut().zip(&live) {
                            u.apply(d, doc);
                        }
                    }
                    ready = docs.into_iter();
                }
                Err(e) => {
                    seg = usize::MAX;
                    return Some(Err(e));
                }
            }
        })
    }

    /// Rows `rows` of the table `snap` looks at, each a segment ordinal
    /// and a docID, increasing within each segment ([`Snapshot::covered`]
    /// gives them so), rebuilt as [`Db::documents`] rebuilds every row.
    pub fn documents_of(&self, snap: &Snapshot, rows: &[(usize, DocId)]) -> Result<Vec<Document>> {
        let mut out = Vec::with_capacity(rows.len());
        for group in rows.chunk_by(|a, b| a.0 == b.0) {
            let v = snap.segments().get(group[0].0).ok_or_else(|| Error::Query(format!("segment {} of {}", group[0].0, snap.segments().len())))?;
            let docs: Vec<DocId> = group.iter().map(|r| r.1).collect();
            let mut got = StoredSource.documents(snap.schema(), &v.file, &docs)?;
            if let Some(u) = &v.updates {
                for (doc, &d) in got.iter_mut().zip(&docs) {
                    u.apply(d, doc);
                }
            }
            out.extend(got);
        }
        Ok(out)
    }

    /// Bulk-loads the rows of `other`, a database with another schema,
    /// matching fields by name: to add, drop or re-embed fields, create a
    /// database with the new schema and load the old one into it. A VECTOR
    /// field is copied when the old field has the same dimensions and
    /// embedding ([`crate::schema::Embed`]); otherwise this database embeds
    /// it anew (given the model, [`Db::set_embedder`]). A semantic-ID field
    /// is copied when the old field has the same codebook, and derived anew
    /// otherwise: a new codebook means a new field, filled this way. So is
    /// a derived TEXT field, by its model ([`crate::schema::Derive`]).
    /// Returns the rows loaded.
    pub fn load_from(&self, other: &Db) -> Result<u64> {
        self.load_table_from(&self.schema().table, other, &other.schema().table)
    }

    /// [`Db::load_from`] of the rows of table `from` of `other` into this
    /// database's table `table`.
    pub fn load_table_from(&self, table: &str, other: &Db, from: &str) -> Result<u64> {
        let (old, new) = (&*other.table(from)?, &*self.table(table)?);
        if let Some(f) = old.unrebuildable() {
            return Err(crate::Error::Schema(format!("{} is not stored, so its values cannot be copied", f.name)));
        }
        let map: Vec<Option<usize>> = new
            .fields
            .iter()
            .map(|f| {
                let j = old.field_index(&f.name)?;
                let g = &old.fields[j];
                let keep = match (&f.ty, &g.ty) {
                    (FieldType::Vector { dim: a, .. }, FieldType::Vector { dim: b, .. }) => a == b && f.embed == g.embed,
                    // Codes of another codebook are derived anew.
                    _ if f.semantic_id.is_some() || g.semantic_id.is_some() => f.semantic_id == g.semantic_id,
                    // Text of another model is made anew.
                    _ if f.derive.is_some() || g.derive.is_some() => f.derive == g.derive,
                    (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b) && f.multi == g.multi && f.media == g.media,
                };
                keep.then_some(j)
            })
            .collect();
        let snap = other.snapshot().table(from).ok_or_else(|| Error::Query(format!("no such table: {from}")))?;
        let mut failed = None;
        let mut n = 0u64;
        let docs = other.documents(&snap).map_while(|d| match d {
            Ok(d) => {
                n += 1;
                Some(map.iter().map(|j| j.map_or(Value::Null, |j| d[j].clone())).collect())
            }
            Err(e) => {
                failed = Some(e);
                None
            }
        });
        self.load_into(table, docs)?;
        match failed {
            Some(e) => Err(e),
            None => Ok(n),
        }
    }

    /// Gives embedded VECTOR field `field` ([`crate::schema::Embed`]) of the
    /// first table its model. Writes through this `Db` that leave the vector
    /// out get it computed ([`Db::load`], SQL `INSERT` and `UPDATE`,
    /// [`Db::embed_missing`] before a [`crate::storage::WriteTxn`]).
    pub fn set_embedder(&self, field: &str, e: Arc<dyn Embedder>) -> Result<()> {
        let first = self.schema().table.clone();
        self.set_embedder_in(&first, field, e)
    }

    /// [`Db::set_embedder`] for a field of the table named `table`.
    pub fn set_embedder_in(&self, table: &str, field: &str, e: Arc<dyn Embedder>) -> Result<()> {
        let schema = self.table(table)?;
        let i = schema.field_index(field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        let f = &schema.fields[i];
        let (FieldType::Vector { dim, .. }, Some(spec)) = (&f.ty, &f.embed) else {
            return Err(Error::Schema(format!("field {field} is not an embedded VECTOR field")));
        };
        if e.dim() != *dim {
            return Err(Error::Schema(format!("model {} makes {}-dimension vectors; field {field} holds {dim}", spec.model, e.dim())));
        }
        if let Some(m) = schema.embeds_media(i) {
            let (wants, what) = match schema.fields[m].media {
                Some(crate::media::MediaKind::Audio) => (Modality::Audio, "sound"),
                _ => (Modality::Image, "images"),
            };
            if !e.modalities().contains(&wants) {
                return Err(Error::Schema(format!("model {} does not embed {what}, and field {field} is the embedding of the {what} of {}", spec.model, schema.fields[m].name)));
            }
        }
        let mut all = self.embedders.write().unwrap_or_else(|e| e.into_inner());
        all.retain(|m| !(m.table.eq_ignore_ascii_case(&schema.table) && m.field.eq_ignore_ascii_case(&f.name)));
        all.push(Model { table: schema.table.clone(), field: f.name.clone(), embed: spec.clone(), dim: *dim, model: e });
        Ok(())
    }

    /// The model of embedded field `field` of the first table, if it has one.
    pub fn embedder(&self, field: &str) -> Option<Arc<dyn Embedder>> {
        self.embedder_in(&self.schema().table, field)
    }

    /// The model of embedded field `field` of the table named `table`.
    pub fn embedder_in(&self, table: &str, field: &str) -> Option<Arc<dyn Embedder>> {
        let schema = self.storage.table(table)?;
        let i = schema.field_index(field)?;
        self.models_of(&schema).swap_remove(i)
    }

    /// The models of the embedded fields of the table `schema` describes,
    /// by field.
    fn models_of(&self, schema: &Schema) -> Vec<Option<Arc<dyn Embedder>>> {
        let all = self.embedders.read().unwrap_or_else(|e| e.into_inner());
        schema
            .fields
            .iter()
            .map(|f| {
                let serves = |m: &&Model| {
                    m.table.eq_ignore_ascii_case(&schema.table) && m.field.eq_ignore_ascii_case(&f.name) && f.embed.as_ref() == Some(&m.embed) && matches!(f.ty, FieldType::Vector { dim, .. } if dim == m.dim)
                };
                all.iter().find(serves).map(|m| m.model.clone())
            })
            .collect()
    }

    /// Gives derived TEXT field `field` ([`crate::schema::Derive`]) of the
    /// table named `table` its model. Writes through this `Db` that leave
    /// the text out get it made, as they get embedded vectors.
    pub fn set_deriver_in(&self, table: &str, field: &str, d: Arc<dyn Deriver>) -> Result<()> {
        let schema = self.table(table)?;
        let f = schema.field(field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        let Some(spec) = &f.derive else { return Err(Error::Schema(format!("field {field} is not a derived TEXT field"))) };
        let kind = schema.field(&spec.from).and_then(|m| m.media).expect("checked by the schema");
        let wants = match d.modality() {
            Modality::Image => crate::media::MediaKind::Image,
            Modality::Audio => crate::media::MediaKind::Audio,
            Modality::Text => return Err(Error::Schema(format!("model {} makes text of text, not of media", spec.model))),
        };
        if kind != wants {
            return Err(Error::Schema(format!(
                "model {} makes text of {}, and {} is a MEDIA field of kind {}: declare kind = '{}'",
                spec.model,
                if wants == crate::media::MediaKind::Image { "images" } else { "audio" },
                spec.from,
                kind.as_str(),
                wants.as_str()
            )));
        }
        let mut all = self.derivers.write().unwrap_or_else(|e| e.into_inner());
        all.retain(|m| !(m.table.eq_ignore_ascii_case(&schema.table) && m.field.eq_ignore_ascii_case(&f.name)));
        all.push(Deriving { table: schema.table.clone(), field: f.name.clone(), derive: spec.clone(), model: d });
        Ok(())
    }

    /// The model of derived field `field` of the table named `table`.
    pub fn deriver_in(&self, table: &str, field: &str) -> Option<Arc<dyn Deriver>> {
        let schema = self.storage.table(table)?;
        let i = schema.field_index(field)?;
        self.derivers_of(&schema).swap_remove(i)
    }

    /// The models of the derived fields of the table `schema` describes, by
    /// field.
    fn derivers_of(&self, schema: &Schema) -> Vec<Option<Arc<dyn Deriver>>> {
        let all = self.derivers.read().unwrap_or_else(|e| e.into_inner());
        schema
            .fields
            .iter()
            .map(|f| {
                let serves = |m: &&Deriving| m.table.eq_ignore_ascii_case(&schema.table) && m.field.eq_ignore_ascii_case(&f.name) && f.derive.as_ref() == Some(&m.derive);
                all.iter().find(serves).map(|m| m.model.clone())
            })
            .collect()
    }

    /// Opens the model of every embedded field of every table from `models`
    /// ([`crate::embed::open`]), and of every derived field
    /// ([`crate::derive::open`]), and those of the tables created later
    /// ([`Db::create_table`]); returns the fields whose model did not open,
    /// with the reason (named `table.field` outside the first table). Their
    /// vectors or text must come with the writes.
    pub fn open_embedders(&self, models: &Path) -> Vec<(String, Error)> {
        *self.models.write().unwrap_or_else(|e| e.into_inner()) = Some(models.to_path_buf());
        let first = self.schema().table.clone();
        let mut failed = Vec::new();
        for t in self.catalog() {
            for (f, e) in self.open_embedders_of(&t.table, models) {
                failed.push((if t.table == first { f } else { format!("{}.{f}", t.table) }, e));
            }
        }
        failed.extend(self.open_codebooks(models));
        failed
    }

    /// Opens the codebook of every semantic-ID field from `models`
    /// ([`crate::sid::Codebook::open`]), checked against the hash the field
    /// records; returns the fields whose codebook did not open, as
    /// [`Db::open_embedders`] does. Writes that need codes from those fail.
    pub fn open_codebooks(&self, models: &Path) -> Vec<(String, Error)> {
        let first = self.schema().table.clone();
        let mut failed = Vec::new();
        for t in self.catalog() {
            for (f, e) in self.open_codebooks_of(&t.table, models) {
                failed.push((if t.table == first { f } else { format!("{}.{f}", t.table) }, e));
            }
        }
        failed
    }

    fn open_codebooks_of(&self, table: &str, models: &Path) -> Vec<(String, Error)> {
        let Some(schema) = self.storage.table(table) else { return Vec::new() };
        let mut failed = Vec::new();
        for f in &schema.fields {
            let Some(sid) = &f.semantic_id else { continue };
            let opened = crate::sid::Codebook::open(&sid.codebook, models).and_then(|(cb, hash)| self.storage.set_codebook(table, &f.name, cb, &hash));
            if let Err(e) = opened {
                failed.push((f.name.clone(), e));
            }
        }
        failed
    }

    fn open_embedders_of(&self, table: &str, models: &Path) -> Vec<(String, Error)> {
        let Some(schema) = self.storage.table(table) else { return Vec::new() };
        let mut failed = Vec::new();
        for f in &schema.fields {
            let opened = match (&f.embed, &f.derive) {
                (Some(spec), _) => crate::embed::open_for(&spec.model, match f.ty { FieldType::Vector { dim, .. } => Some(dim), _ => None }, models).and_then(|m| self.set_embedder_in(table, &f.name, m)),
                (_, Some(spec)) => crate::derive::open(&spec.model, models).and_then(|m| self.set_deriver_in(table, &f.name, m)),
                _ => continue,
            };
            if let Err(e) = opened {
                failed.push((f.name.clone(), e));
            }
        }
        failed
    }

    /// Opens the ranking models a plan's `gbdt()` and `cross_encoder()`
    /// calls name ([`crate::query::model`]) from the models directory: the
    /// one [`Db::open_embedders`] was given, else
    /// [`crate::embed::models_dir`]. A `shard`
    /// ([`crate::query::exec::ExecOptions::shard`]) leaves cross-encoders
    /// to its coordinator. A plan that names the hash of another file of a
    /// model is refused; one without gets the hashes of the files opened.
    /// [`Db::run`] and [`Db::query`] open a plan's models themselves.
    pub fn open_ranking_models(&self, plan: &mut Plan, shard: bool) -> Result<()> {
        if !plan.root.has_models() {
            return Ok(());
        }
        let dir = self.models.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(crate::embed::models_dir);
        self.ranking.open(&dir, plan, !shard)
    }

    /// Trains a semantic-ID codebook ([`crate::sid::Codebook::train`]) on the
    /// vectors of VECTOR field `field` of the table named `table`, and saves
    /// it as `name` under `models`, for semantic-ID fields to name. Returns
    /// the SHA-256 of its file.
    pub fn train_codebook(&self, table: &str, field: &str, name: &str, models: &Path, opts: &crate::sid::TrainOptions) -> Result<String> {
        let snap = self.snapshot().table(table).ok_or_else(|| Error::Schema(format!("no table {table}")))?;
        let schema = snap.schema();
        let f = schema.field(field).ok_or_else(|| Error::Schema(format!("no field {field}")))?;
        let FieldType::Vector { dim, .. } = f.ty else { return Err(Error::Schema(format!("field {field} is not a VECTOR field"))) };
        let mut vectors = Vec::new();
        for v in snap.segments() {
            let r = VectorReader::open(&v.file, &f.key())?;
            for d in 0..v.file.n_docs() as DocId {
                if v.live.as_ref().is_none_or(|l| l.contains(d)) && r.has_vector(d) {
                    vectors.extend_from_slice(r.vector(d));
                }
            }
        }
        if vectors.is_empty() {
            return Err(Error::Schema(format!("{table}.{field} holds no vectors to train a codebook on")));
        }
        crate::sid::Codebook::train(&vectors, dim, opts)?.save(name, models)
    }

    /// Computes the embedded vectors that documents of the first table
    /// leave out (null) from their sources, one batch per field split
    /// across the cores. A document whose sources are all empty gets no
    /// vector. Generated fields must be filled in first.
    pub fn embed_missing(&self, docs: &mut [Document]) -> Result<()> {
        let first = self.schema().table.clone();
        self.embed_missing_in(&first, docs)
    }

    /// [`Db::embed_missing`] for documents of the table named `table`.
    /// The images of a MEDIA field are read from where their values say
    /// ([`crate::media::Resolver::bytes_of`]).
    pub fn embed_missing_in(&self, table: &str, docs: &mut [Document]) -> Result<()> {
        let schema = self.table(table)?;
        embed_with(&self.models_of(&schema), &schema, docs, &Files { resolver: &self.media_resolver(), held: HashMap::new() })
    }

    /// Does for documents of the table named `table` the work a write does
    /// before the writer, so that the rows it logs hold every value: their
    /// MEDIA values are resolved ([`Db::resolve_media_in`]), the derived
    /// text they leave out made (and the generated fields that read it
    /// filled in again), then the embedded vectors they leave out computed
    /// ([`Db::embed_missing_in`]). Generated fields must be filled in
    /// first. [`Db::load`] and SQL writes do this themselves.
    pub fn fill_in(&self, table: &str, docs: &mut [Document]) -> Result<()> {
        let schema = self.table(table)?;
        let media = self.media_resolver();
        let files = Files { resolver: &media, held: resolve_media(&media, &schema, docs)? };
        derive_known(&self.derivers_of(&schema), &schema, docs, &Derived::new(), &files)?;
        embed_with(&self.models_of(&schema), &schema, docs, &files)
    }

    /// Sets how writes through this `Db` resolve the MEDIA values they give
    /// ([`crate::media::Resolver`]: the sources it may fetch, its size cap
    /// and its media store). The default fetches nothing and keeps
    /// nothing: it takes complete values, and bytes given with a URI.
    pub fn set_media_resolver(&self, r: Resolver) {
        *self.media.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(r);
    }

    /// How writes resolve MEDIA values ([`Db::set_media_resolver`]).
    pub fn media_resolver(&self) -> Arc<Resolver> {
        self.media.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Turns the MEDIA values of documents of the table named `table`, as
    /// writes give them ([`crate::media::input_text`]), into the values
    /// they store ([`crate::media::Resolver::resolve`]): files fetched,
    /// hashed and described here rather than in the writer, so that the
    /// rows a primary logs hold them and replicas never fetch. The first
    /// value that fails fails them all.
    pub fn resolve_media_in(&self, table: &str, docs: &mut [Document]) -> Result<()> {
        let schema = self.table(table)?;
        resolve_media(&self.media_resolver(), &schema, docs).map(drop)
    }

    pub fn query(&self, sql: &str, params: &Params) -> Result<ResultSet> {
        self.query_with(sql, params, &self.exec)
    }

    pub fn query_with(&self, sql: &str, params: &Params, opts: &ExecOptions) -> Result<ResultSet> {
        self.query_at(&self.snapshot(), sql, params, opts)
    }

    /// Runs a query against a given snapshot (several queries that must
    /// see the same version share one), on the table it names (`FROM`).
    ///
    /// A compound query (`WITH`, a subquery, a join, `SELECT DISTINCT`) runs
    /// as a plan of operators over single-table queries
    /// ([`crate::query::compound`]), all on the same snapshot; its parts
    /// may read different tables (a join of two tables, a subquery over
    /// another). Its parts may hold up to
    /// [`crate::query::compound::MAX_ROWS`] rows each; its answer is held
    /// to [`ExecOptions::max_rows`].
    pub fn query_at(&self, snap: &Snapshot, sql: &str, params: &Params, opts: &ExecOptions) -> Result<ResultSet> {
        let q = qsql::parse_query(sql)?;
        let plan = plan_of(snap, &q, params)?;
        self.run_on(snap, &plan, opts)
    }

    /// Runs a query ([`Db::query`]) and logs what it served to a log table
    /// ([`crate::rank_log`]), made by the first query logged to it: one row
    /// per row answered, or per candidate ([`Log::candidates`]), with the
    /// values its ranking read. The answer has the columns the query asked
    /// for. The query reads one snapshot and the log is written after it in
    /// one transaction, as [`Db::sql`] writes.
    pub fn query_logged(&self, sql: &str, params: &Params, log: &Log) -> Result<Logged> {
        if self.storage.table(&log.table).is_none() {
            self.sql(&rank_log::create_sql(&log.table)?, &Params::new())?;
        }
        let schema = self.table(&log.table)?;
        rank_log::check(&schema)?;
        let p = rank_log::prepare(sql, params, log)?;
        let served = rank_log::serve(self.query(&p.sql, params)?, &p, log)?;
        let now = crate::timestamp::now();
        let mut t = self.storage.begin()?;
        {
            let mut tt = t.table(&schema.table)?;
            for e in &served.entries {
                tt.upsert(crate::ingest::document_at(&schema, &e.to_json(), now)?)?;
            }
        }
        let seq_no = t.commit_seq()?;
        Ok(Logged { rs: served.rs, request: served.request, seq_no })
    }

    /// Exports a log table ([`rank_log::export`]) from one snapshot: one
    /// record per logged row, with its features as served and, from an
    /// events table, its history and label as of its time. Returns the
    /// records given to `out`.
    pub fn export_rank_log(&self, x: &Export, mut out: impl FnMut(serde_json::Value) -> Result<()>) -> Result<u64> {
        let snap = self.snapshot();
        let tables: Vec<&Schema> = snap.schemas().collect();
        let opts = ExecOptions { max_rows: None, ..self.exec.clone() };
        rank_log::export(&mut |sql, params| self.query_at(&snap, sql, params, &opts), &tables, x, &mut out)
    }

    /// Runs a read plan on `snap`'s version: a leaf on the executor, any
    /// other with operators ([`crate::query::ops`]), each of whose leaves
    /// reads its own table and may hold up to
    /// [`crate::query::compound::MAX_ROWS`] rows.
    fn run_on(&self, snap: &Snapshot, plan: &Plan, opts: &ExecOptions) -> Result<ResultSet> {
        profile::check_plan(plan, &|t, name| snap.table(t).and_then(|v| v.rank_profile(name).map(|p| p.hash())))?;
        let opened;
        let plan = match plan.root.has_models() {
            true => {
                let mut p = plan.clone();
                self.open_ranking_models(&mut p, false)?;
                opened = p;
                &opened
            }
            false => plan,
        };
        let view_of = |table: &str| snap.table(table).ok_or_else(|| Error::Query(format!("no such table: {table}")));
        if !plan.root.uses_operators() {
            let view = view_of(plan.table())?;
            return self.execute(&view, &plan.to_select(view.schema())?, opts);
        }
        let leaves = ExecOptions { max_rows: None, ..opts.clone() };
        let catalog: Vec<&Schema> = snap.schemas().collect();
        let rs = ops::run_in(
            plan,
            &catalog,
            &mut |leaf| {
                let view = view_of(leaf.table())?;
                self.execute(&view, &leaf.to_select(view.schema())?, &leaves)
            },
            opts,
        )?;
        if let Some(max) = opts.max_rows.filter(|&m| rs.rows.len() > m) {
            return Err(Error::Query(format!("the query matches more than {max} rows, the most a query returns; add a LIMIT")));
        }
        Ok(rs)
    }

    /// Runs a plan ([`crate::query::ir`]) on the table it names: a read
    /// returns its rows, a write the number of rows it wrote, in one
    /// transaction. SQL runs the same way: [`Db::query`] and [`Db::sql`]
    /// bind a statement into a plan, and the executor answers its Search.
    /// A plan made elsewhere is checked against the table's schema as the
    /// binder checks SQL.
    pub fn run(&self, plan: &Plan) -> Result<SqlResult> {
        match plan.is_write() {
            true => self.write_prepared(&self.prepare_plan(plan)?).map(SqlResult::Written),
            false => self.run_at(&self.snapshot(), plan, &self.exec).map(SqlResult::Rows),
        }
    }

    /// [`Db::run`] of a plan that reads, against a given snapshot.
    pub fn run_at(&self, snap: &Snapshot, plan: &Plan, opts: &ExecOptions) -> Result<ResultSet> {
        self.run_on(snap, plan, opts)
    }

    /// [`Db::prepare_write`] of a plan that writes.
    pub fn prepare_plan(&self, plan: &Plan) -> Result<PreparedWrite> {
        if !plan.is_write() {
            return Err(Error::Query("the plan reads; run it with Db::run".into()));
        }
        let schema = self.table(plan.table())?;
        let write = plan.to_write(&schema)?;
        self.prepare(schema, write)
    }

    /// Runs any statement, on any table: a SELECT returns its rows; an
    /// INSERT, UPDATE or DELETE the number of documents it wrote, in one
    /// transaction; `CREATE TABLE` ([`Db::create_table`]) and `DROP TABLE`
    /// ([`Db::drop_tables`]) 0.
    ///
    /// UPDATE sets numeric fields in place ([`crate::storage::WriteTxn::set`])
    /// and rewrites the whole document for any other field, which needs every
    /// text field stored or generated. The rows to change are read after the
    /// transaction takes the writer lock, so no other write lands between.
    pub fn sql(&self, sql: &str, params: &Params) -> Result<SqlResult> {
        let catalog = self.catalog();
        let tables: Vec<&Schema> = catalog.iter().map(|t| &**t).collect();
        match bind_command(sql, &tables, params)? {
            Command::Query => self.query(sql, params).map(SqlResult::Rows),
            Command::Explain { json } => {
                let sqlparser::ast::Statement::Explain { statement, .. } = qsql::parse_one(sql)? else { unreachable!("bound as EXPLAIN") };
                let plan = self.explain_statement(&statement, params)?;
                let rows = match json {
                    true => vec![vec![Value::Str(plan.to_json_text()?)]],
                    false => plan.explain(&tables).into_iter().map(|l| vec![Value::Str(l)]).collect(),
                };
                let stats = ExecStats::default();
                Ok(SqlResult::Rows(ResultSet { columns: vec!["plan".into()], rows, keys: Vec::new(), scores: Vec::new(), parts: None, stats }))
            }
            Command::CreateTable { schema, if_not_exists } => {
                if !(if_not_exists && self.storage.table(&schema.table).is_some()) {
                    self.create_table(schema)?;
                }
                Ok(SqlResult::Written(0))
            }
            Command::DropTable { tables, opts } => {
                let names: Vec<&str> = tables.iter().map(|t| t.as_str()).collect();
                self.drop_tables(&names, opts)?;
                Ok(SqlResult::Written(0))
            }
            Command::CreateRankProfile { table, profile, replace } => {
                self.storage.set_rank_profile(&table, profile, replace)?;
                Ok(SqlResult::Written(0))
            }
            Command::DropRankProfile { table, name, if_exists } => {
                let table = match table {
                    Some(t) => Some(t),
                    None => {
                        let mut with: Vec<String> = Vec::new();
                        for t in &catalog {
                            if self.storage.rank_profiles(&t.table)?.iter().any(|p| p.name.eq_ignore_ascii_case(&name)) {
                                with.push(t.table.clone());
                            }
                        }
                        if with.len() > 1 {
                            return Err(Error::Query(format!("{} have a rank profile named {name}: DROP RANK PROFILE {name} ON table", with.join(" and "))));
                        }
                        with.pop()
                    }
                };
                let dropped = match &table {
                    Some(t) => self.storage.drop_rank_profile(t, &name)?,
                    None => false,
                };
                if !dropped && !if_exists {
                    return Err(Error::Query(format!("no rank profile named {name}{}", table.map_or(String::new(), |t| format!(" on {t}")))));
                }
                Ok(SqlResult::Written(0))
            }
            Command::Write { table, write } => {
                let schema = catalog.iter().find(|t| t.table == table).expect("bound to a table of the catalog");
                let write = Plan::of_write(schema, &write).to_write(schema)?;
                self.write_prepared(&self.prepare(schema.clone(), write)?).map(SqlResult::Written)
            }
        }
    }

    /// The plan of one statement ([`crate::query::ir`]): a SELECT, INSERT,
    /// UPDATE or DELETE, bound as it would run but not run. `EXPLAIN
    /// <statement>` in [`Db::sql`] gives it as text or JSON. A compound
    /// SELECT (`WITH`, a subquery, a join, `SELECT DISTINCT`) plans into
    /// operators over single-table queries ([`crate::query::compound`]).
    pub fn explain(&self, sql: &str, params: &Params) -> Result<Plan> {
        self.explain_statement(&qsql::parse_one(sql)?, params)
    }

    fn explain_statement(&self, stmt: &sqlparser::ast::Statement, params: &Params) -> Result<Plan> {
        if let sqlparser::ast::Statement::Query(q) = stmt {
            // The plan names the files of its ranking models.
            let mut plan = plan_of(&self.snapshot(), q, params)?;
            self.open_ranking_models(&mut plan, false)?;
            return Ok(plan);
        }
        let catalog = self.catalog();
        let tables: Vec<&Schema> = catalog.iter().map(|t| &**t).collect();
        match qsql::bind_statement(stmt, &tables, params)? {
            Command::Write { table, write } => {
                let schema = catalog.iter().find(|t| t.table == table).expect("bound to a table of the catalog");
                Ok(Plan::of_write(schema, &write))
            }
            _ => Err(Error::Query("EXPLAIN shows the plan of a SELECT, INSERT, UPDATE or DELETE".into())),
        }
    }

    /// Binds an INSERT, UPDATE or DELETE, and computes the vectors it
    /// writes (of the embedded fields it leaves out, and of those whose
    /// sources an UPDATE changes) before [`Db::write_prepared`] takes the
    /// writer lock: a model is slow, and every other write waits while the
    /// lock is held. `None` for any other statement. [`Db::sql`] does both.
    pub fn prepare_write(&self, sql: &str, params: &Params) -> Result<Option<PreparedWrite>> {
        let catalog = self.catalog();
        let tables: Vec<&Schema> = catalog.iter().map(|t| &**t).collect();
        let Command::Write { table, write } = bind_command(sql, &tables, params)? else { return Ok(None) };
        let schema = catalog.iter().find(|t| t.table == table).expect("bound to a table of the catalog");
        let write = Plan::of_write(schema, &write).to_write(schema)?;
        Ok(Some(self.prepare(schema.clone(), write)?))
    }

    fn prepare(&self, schema: Arc<Schema>, mut write: Write) -> Result<PreparedWrite> {
        let media = self.media_resolver();
        let mut held = Held::new();
        match &mut write {
            Write::Insert { docs, conflict } => {
                docs.iter_mut().for_each(|d| schema.fill_generated(d));
                held = resolve_media(&media, &schema, docs)?;
                if let Conflict::Update(sets) = conflict {
                    held.extend(resolve_sets(&media, &schema, sets)?);
                }
            }
            Write::Update { sets, .. } => held = resolve_sets(&media, &schema, sets)?,
            Write::Delete { .. } => {}
        }
        let (known, derived) = self.embed_ahead(&schema, &write, &Files { resolver: &media, held })?;
        Ok(PreparedWrite { schema, write, known, derived, mask: None, guard: None })
    }

    /// The text `write` will derive and the vectors it will compute, from
    /// the latest version. The rows an UPDATE or `ON CONFLICT DO UPDATE`
    /// rewrites are read here and again under the writer lock, where only
    /// the files and texts that changed in between go to a model. An error
    /// reading the rows (the table dropped, a sum that overflows) is left
    /// for the write to report; an image that cannot be read fails the
    /// statement.
    fn embed_ahead(&self, schema: &Schema, write: &Write, files: &Files) -> Result<(Known, Derived)> {
        let models = self.models_of(schema);
        let derivers = self.derivers_of(schema);
        if models.iter().all(Option::is_none) && derivers.iter().all(Option::is_none) {
            return Ok((Known::new(), Derived::new()));
        }
        let docs = || -> Result<Vec<Document>> {
            let Some(snap) = self.snapshot().table(&schema.table) else { return Ok(Vec::new()) };
            let pk = || schema.primary_key.ok_or_else(|| Error::Query(format!("{} has no primary key", schema.table)));
            Ok(match write {
                Write::Insert { docs, conflict: Conflict::Fail | Conflict::Replace } => docs.clone(),
                Write::Insert { docs, conflict: Conflict::Skip } => not_live(&snap, pk()?, docs.clone()),
                Write::Insert { docs, conflict: Conflict::Update(sets) } => {
                    let pk = pk()?;
                    let (live, mut new): (Vec<Document>, Vec<Document>) = docs.iter().cloned().partition(|d| snap.lookup(&d[pk]).is_some());
                    if !in_place(schema, sets) {
                        new.extend(plan_sets(schema, &models, &derivers, self.conflict_rows(&snap, schema, pk, live)?, sets)?.rewrites);
                    }
                    new
                }
                // Numbers set in place embed nothing.
                Write::Update { sets, .. } if in_place(schema, sets) => Vec::new(),
                Write::Update { select, sets } => {
                    let rows = self.execute(&snap, select, &self.exec)?.rows;
                    plan_sets(schema, &models, &derivers, rows.into_iter().map(|r| (r, Vec::new())).collect(), sets)?.rewrites
                }
                Write::Delete { .. } => Vec::new(),
            })
        };
        let Ok(mut docs) = docs() else { return Ok((Known::new(), Derived::new())) };
        let derived = derived_of(&derivers, schema, &docs, files)?;
        derive_known(&derivers, schema, &mut docs, &derived, files)?;
        Ok((vectors_of(&models, schema, &docs, files)?, derived))
    }

    /// Runs a write [`Db::prepare_write`] bound, in one transaction, and
    /// returns the number of documents written. The rows an UPDATE or
    /// DELETE changes are read under the writer lock, so no other write
    /// lands between; vectors computed ahead are used where their source
    /// texts did not change. It can run again after an error (a busy
    /// writer).
    pub fn write_prepared(&self, p: &PreparedWrite) -> Result<u64> {
        Ok(self.write_prepared_seq(p)?.0)
    }

    /// [`Db::write_prepared`], also returning the transaction's sequence
    /// number in a database that numbers its writes (none if it wrote
    /// nothing).
    pub fn write_prepared_seq(&self, p: &PreparedWrite) -> Result<(u64, Option<u64>)> {
        let schema = &*p.schema;
        let mut t = self.storage.begin()?;
        let mut tt = t.table(&schema.table)?;
        if tt.schema() != schema {
            return Err(Error::Query(format!("table {} was dropped and created again while the statement ran; run it again", schema.table)));
        }
        let models = self.models_of(schema);
        let derivers = self.derivers_of(schema);
        // Text and vectors computed ahead cover the rows unless they changed
        // since: files of those are read here.
        let media = self.media_resolver();
        let files = Files { resolver: &media, held: Held::new() };
        // Read under the writer lock: no other write lands between.
        let guarded = match &p.guard {
            Some(g) => (g.0)()?,
            None => None,
        };
        let mut snap = self.snapshot();
        if !matches!(p.write, Write::Insert { .. }) {
            for m in guarded.iter().chain(&p.mask) {
                snap = snap.masked(m)?;
            }
        }
        let snap = snap.table(&schema.table).expect("the writer lock keeps the table");
        let pk = || schema.primary_key.ok_or_else(|| Error::Query(format!("{} has no primary key", schema.table)));
        let mut n = 0u64;
        match p.write.clone() {
            Write::Insert { mut docs, conflict } => {
                match conflict {
                    Conflict::Fail | Conflict::Replace => {
                        derive_known(&derivers, schema, &mut docs, &p.derived, &files)?;
                        embed_known(&models, schema, &mut docs, &p.known, &files)?;
                        for d in docs {
                            if conflict == Conflict::Replace {
                                tt.upsert(d)?;
                            } else {
                                tt.insert(d)?;
                            }
                            n += 1;
                        }
                    }
                    Conflict::Skip => {
                        let mut docs = not_live(&snap, pk()?, docs);
                        derive_known(&derivers, schema, &mut docs, &p.derived, &files)?;
                        embed_known(&models, schema, &mut docs, &p.known, &files)?;
                        for d in docs {
                            tt.insert(d)?;
                            n += 1;
                        }
                    }
                    Conflict::Update(sets) => {
                        let pk = pk()?;
                        let mut seen = HashSet::new();
                        if let Some(d) = docs.iter().find(|d| !seen.insert(Part::of(&d[pk]))) {
                            return Err(Error::Query(format!("ON CONFLICT DO UPDATE: key {:?} comes twice; a statement updates a row once", d[pk])));
                        }
                        let (live, mut new): (Vec<Document>, Vec<Document>) = docs.into_iter().partition(|d| snap.lookup(&d[pk]).is_some());
                        derive_known(&derivers, schema, &mut new, &p.derived, &files)?;
                        embed_known(&models, schema, &mut new, &p.known, &files)?;
                        for d in new {
                            tt.insert(d)?;
                            n += 1;
                        }
                        let pairs = self.conflict_rows(&snap, schema, pk, live)?;
                        n += set_rows(&mut tt, schema, (&models, &derivers), pairs, &sets, (&p.known, &p.derived), &files)?;
                    }
                }
            }
            Write::Delete { select } => {
                let pk = pk()?;
                for row in self.execute(&snap, &select, &self.exec)?.rows {
                    n += tt.delete(&row[pk])? as u64;
                }
            }
            Write::Update { select, sets } => {
                let rows = self.execute(&snap, &select, &self.exec)?.rows;
                n += set_rows(&mut tt, schema, (&models, &derivers), rows.into_iter().map(|r| (r, Vec::new())).collect(), &sets, (&p.known, &p.derived), &files)?;
            }
        }
        drop(tt);
        let seq = t.commit_seq()?;
        Ok((n, seq))
    }

    /// The live rows of the documents an INSERT proposed (`live`), every
    /// field, each paired with its document, for `ON CONFLICT DO UPDATE`.
    fn conflict_rows(&self, snap: &Snapshot, schema: &Schema, pk: usize, live: Vec<Document>) -> Result<Vec<(Vec<Value>, Document)>> {
        if live.is_empty() {
            return Ok(Vec::new());
        }
        let keys = live.iter().map(|d| d[pk].clone()).collect();
        let rows = self.execute(snap, &qsql::keyed_rows(schema, keys)?, &self.exec)?.rows;
        let mut by_key: HashMap<Part, Vec<Value>> = rows.into_iter().map(|r| (Part::of(&r[pk]), r)).collect();
        Ok(live.into_iter().filter_map(|d| by_key.remove(&Part::of(&d[pk])).map(|r| (r, d))).collect())
    }

    pub fn execute(&self, snap: &Snapshot, select: &Select, opts: &ExecOptions) -> Result<ResultSet> {
        // Fixed before the first query's first step begins.
        crate::query::steps::epoch();
        let t0 = std::time::Instant::now();
        let readers = self.readers(snap)?;
        let opened = t0.elapsed();
        let mut rs = Executor::new(snap, &readers, opts).execute(select)?;
        // Opening segments' readers takes microseconds once they are
        // cached; a step when it took longer (a cold cache, new segments).
        if opened.as_micros() >= 20 {
            let step = crate::query::steps::Step { op: "open readers".into(), detail: format!("{} segments", readers.len()), start_us: crate::query::steps::us_at(t0), us: crate::query::steps::tenth(opened.as_nanos() as f64 / 1e3), ..Default::default() };
            rs.stats.steps.insert(0, step);
        }
        Ok(rs)
    }

    /// This database's BM25 statistics for the terms `select` scores with,
    /// for a coordinator to sum over shards and pass back through
    /// [`ExecOptions::stats`] ([`crate::cluster`]).
    pub fn term_stats(&self, snap: &Snapshot, select: &Select) -> Result<TermStats> {
        self.term_stats_with(snap, select, &self.exec)
    }

    /// [`Db::term_stats`] with options (their synonyms widen the terms).
    pub fn term_stats_with(&self, snap: &Snapshot, select: &Select, opts: &ExecOptions) -> Result<TermStats> {
        let readers = self.readers(snap)?;
        Executor::new(snap, &readers, opts).term_stats(select)
    }

    /// The loosest level of `select`'s guard (`min_rows`, `max_rows`) and
    /// each level's rows on snapshot `snap`, from it to the strictest
    /// ([`Executor::guard_counts`]): what a coordinator sums over its
    /// shards to choose the level ([`crate::query::logical::Guard::level`]).
    pub fn guard_counts(&self, snap: &Snapshot, select: &Select, opts: &ExecOptions) -> Result<(usize, Vec<u64>)> {
        if select.guard.is_none() {
            return Err(Error::Query("the query has no guard (min_rows, max_rows) to count the levels of".into()));
        }
        let readers = self.readers(snap)?;
        Executor::new(snap, &readers, opts).guard_counts(select)
    }

    /// The codes that can follow each of `prefixes` in semantic-ID field
    /// `field` of table `table`, among the rows that SQL condition `filter`
    /// (with `params`) matches, or all rows: per prefix, each code one
    /// number longer that such rows have, with how many rows have it, in
    /// the order of their last numbers. The prefix `""` asks for the first
    /// numbers. At most [`crate::sid::MAX_PREFIXES`] prefixes.
    pub fn sid_next(&self, table: &str, field: &str, prefixes: &[String], filter: Option<&str>, params: &Params) -> Result<Vec<Vec<(String, u64)>>> {
        let snap = self.snapshot().table(table).ok_or_else(|| Error::Query(format!("no such table: {table}")))?;
        let sql = format!("SELECT COUNT(*) FROM {}{}", qsql::ident(&snap.schema().table), filter.map(|c| format!(" WHERE {c}")).unwrap_or_default());
        let select = qsql::bind(&sql, snap.schema(), params)?;
        self.sid_next_with(&snap, &select, field, prefixes, &self.exec)
    }

    /// [`Db::sid_next`] on snapshot `snap`, among the rows `select`'s
    /// filter matches (the rest of it is not used), with options.
    pub fn sid_next_with(&self, snap: &Snapshot, select: &Select, field: &str, prefixes: &[String], opts: &ExecOptions) -> Result<Vec<Vec<(String, u64)>>> {
        let schema = snap.schema();
        let f = schema.field_index(field).ok_or_else(|| Error::Query(format!("no field {field} in {}", schema.table)))?;
        if schema.fields[f].semantic_id.is_none() {
            return Err(Error::Query(format!("{field} is not a semantic-ID field")));
        }
        if prefixes.is_empty() || prefixes.len() > crate::sid::MAX_PREFIXES {
            return Err(Error::Query(format!("ask for the next codes of 1 to {} prefixes, not {}", crate::sid::MAX_PREFIXES, prefixes.len())));
        }
        if let Some(p) = prefixes.iter().find(|p| !p.is_empty() && crate::sid::parse(p).is_none_or(|c| c.len() > crate::sid::MAX_LEVELS)) {
            return Err(Error::Query(format!("{p:?} is not a prefix of a semantic ID: numbers joined by dots, such as 12.7, or \"\" for the first")));
        }
        let readers = self.readers(snap)?;
        Executor::new(snap, &readers, opts).sid_next(select, f, prefixes)
    }

    /// Builds a segment image of `docs` for this schema into `dest` (a new
    /// file), to attach to a database elsewhere ([`Db::attach_segments`]).
    pub fn write_segment(schema: &Schema, docs: Vec<Document>, opts: &crate::segment::build::BuildOptions, dest: &Path) -> Result<()> {
        let docs: Vec<Document> = docs
            .into_iter()
            .map(|mut d| {
                crate::value::conform(schema, &mut d)?;
                schema.fill_generated(&mut d);
                Ok(d)
            })
            .collect::<Result<_>>()?;
        let image = crate::segment::build::build_segment(schema, &docs, opts)?;
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(dest)?;
        std::io::Write::write_all(&mut f, &image)?;
        f.sync_data()?;
        Ok(())
    }

    /// Attaches segment images built elsewhere, in one commit, after
    /// checking that every field of this table opens in each of them.
    pub fn attach_segments(&self, images: &[std::path::PathBuf]) -> Result<Vec<u64>> {
        let schema = self.schema().clone();
        for p in images {
            let file = SegmentFile::open(crate::postings::bytes::OwnedBytes::new(std::fs::read(p)?))?;
            file.verify_sections().map_err(|e| crate::Error::Schema(format!("{}: {e}", p.display())))?;
            let r = SegmentReaders::open(&file, &schema)?;
            for (i, f) in schema.fields.iter().enumerate() {
                let ok = match f.ty {
                    FieldType::Text { .. } | FieldType::Sparse => r.text(i).map(|_| ()),
                    FieldType::Keyword => r.keyword(i).map(|_| ()),
                    FieldType::Vector { .. } => r.vector(i).map(|_| ()),
                    FieldType::Edges => r.graph(i).map(|_| ()),
                    FieldType::Int | FieldType::Float => Ok(()),
                };
                ok.map_err(|e| crate::Error::Schema(format!("{}: field {}: {e}", p.display(), f.name)))?;
            }
            r.store()?;
        }
        self.storage.attach_segments(images)
    }

    /// Readers for every segment of `snap`, opened on first use.
    pub fn readers(&self, snap: &Snapshot) -> Result<Vec<Arc<SegmentReaders>>> {
        let at = |v: &crate::storage::SegmentView| v.file.image().as_ptr() as usize;
        {
            let cache = self.readers.read().unwrap_or_else(|e| e.into_inner());
            let hit = |v: &crate::storage::SegmentView| cache.get(&v.id).filter(|(p, _)| *p == at(v)).map(|(_, r)| r.clone());
            if let Some(all) = snap.segments().iter().map(hit).collect() {
                return Ok(all);
            }
        }
        let mut cache = self.readers.write().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(snap.segments().len());
        for v in snap.segments() {
            let r = match cache.get(&v.id).filter(|(p, _)| *p == at(v)) {
                Some((_, r)) => r.clone(),
                None => {
                    let r = Arc::new(SegmentReaders::open(&v.file, snap.schema())?);
                    cache.insert(v.id, (at(v), r.clone()));
                    r
                }
            };
            out.push(r);
        }
        // Segment ids are unique across tables; keep the readers of every
        // table, so that queries on different tables do not evict each other's.
        if cache.len() > 2 * snap.all_segments().count() + 16 {
            let live: HashSet<u64> = snap.all_segments().map(|v| v.id).collect();
            cache.retain(|id, _| live.contains(id));
        }
        Ok(out)
    }

    /// Drops cached readers (e.g. before cold-cache measurements).
    pub fn clear_readers(&self) {
        self.readers.write().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Documents [`Db::load`] embeds at a time.
const EMBED_BATCH: usize = 4096;

/// Documents [`Db::load`] embeds at a time into a table with a field of
/// images' vectors: their files are held until they are embedded.
const MEDIA_BATCH: usize = 64;

/// The bytes of the files a write's MEDIA values named, by hash: read
/// once when the values were resolved, for the vectors embedded from them.
type Held = HashMap<String, Arc<[u8]>>;

/// Where the images an embedded field embeds come from: the bytes a write
/// held, else where the value says ([`Resolver::bytes_of`]).
struct Files<'a> {
    resolver: &'a Resolver,
    held: Held,
}

impl Files<'_> {
    fn bytes(&self, m: &Media) -> Result<Arc<[u8]>> {
        match self.held.get(&m.sha256) {
            Some(b) => Ok(b.clone()),
            None => Ok(self.resolver.bytes_of(m)?.into()),
        }
    }
}

/// Whether field `g` is a source of a generated field among `from`.
fn generated_from_set(schema: &Schema, from: &[String], g: usize) -> bool {
    from.iter().filter_map(|s| schema.field(s)).any(|f| f.generated_from.iter().any(|x| schema.field_index(x) == Some(g)))
}

/// [`Db::resolve_media_in`] with resolver `r`, for documents of `schema`.
/// Returns the bytes read of the files a field embeds.
fn resolve_media(r: &Resolver, schema: &Schema, docs: &mut [Document]) -> Result<Held> {
    let mut held = Held::new();
    for (f, field) in schema.fields.iter().enumerate() {
        let Some(kind) = field.media else { continue };
        let at: Vec<usize> = (0..docs.len()).filter(|&j| matches!(docs[j][f], Value::Str(_))).collect();
        let resolved = {
            let inputs: Vec<&str> = at.iter().filter_map(|&j| docs[j][f].as_str()).collect();
            r.resolve_all(kind, &inputs)
        };
        let read = read_by_model(schema, f);
        for (j, m) in at.into_iter().zip(resolved) {
            let m = m.map_err(|e| media_error(&field.name, e))?;
            if let Some(data) = m.data.filter(|_| read) {
                held.insert(m.media.sha256.clone(), data);
            }
            docs[j][f] = Value::Str(m.media.to_text());
        }
    }
    Ok(held)
}

/// Resolves the MEDIA values an UPDATE's `SET` gives as constants.
fn resolve_sets(r: &Resolver, schema: &Schema, sets: &mut [(usize, Assign)]) -> Result<Held> {
    let mut held = Held::new();
    for (f, a) in sets {
        let (Some(kind), Assign::Value(Value::Str(t))) = (schema.fields[*f].media, &*a) else { continue };
        let m = r.resolve(kind, t).map_err(|e| media_error(&schema.fields[*f].name, e))?;
        if let Some(data) = m.data.filter(|_| read_by_model(schema, *f)) {
            held.insert(m.media.sha256.clone(), data);
        }
        *a = Assign::Value(Value::Str(m.media.to_text()));
    }
    Ok(held)
}

/// Whether a model reads the files of MEDIA field `f`: a field embeds it,
/// or is derived from it.
fn read_by_model(schema: &Schema, f: usize) -> bool {
    (0..schema.fields.len()).any(|g| schema.embeds_media(g) == Some(f) || schema.derives_from(g) == Some(f))
}

fn media_error(field: &str, e: Error) -> Error {
    match e {
        Error::Schema(m) => Error::Schema(format!("field {field}: {m}")),
        Error::Timeout(m) => Error::Timeout(format!("field {field}: {m}")),
        e => e,
    }
}

/// Computes the embedded vectors `docs` leave out (null) with `models`
/// (by field of `schema`), one batch per field split across the cores. A
/// document whose sources are all empty gets no vector.
fn embed_with(models: &[Option<Arc<dyn Embedder>>], schema: &Schema, docs: &mut [Document], files: &Files) -> Result<()> {
    embed_known(models, schema, docs, &Known::new(), files)
}

/// [`embed_with`], taking the vectors of the texts `known` has from it.
fn embed_known(models: &[Option<Arc<dyn Embedder>>], schema: &Schema, docs: &mut [Document], known: &Known, files: &Files) -> Result<()> {
    for (f, e) in models.iter().enumerate() {
        let Some(e) = e else { continue };
        let ahead = known.get(&f);
        let mut todo = Vec::new();
        let mut texts = Vec::new();
        for (j, d) in docs.iter_mut().enumerate() {
            if matches!(d[f], Value::Null) {
                let t = schema.embed_text(f, d);
                if let Some(v) = ahead.and_then(|a| a.get(&t)) {
                    d[f] = Value::Vector(v.clone());
                } else if !t.is_empty() {
                    todo.push(j);
                    texts.push(t);
                }
            }
        }
        for (j, v) in todo.into_iter().zip(embed_sources(e.as_ref(), schema, f, &texts, files)?) {
            docs[j][f] = Value::Vector(v.into());
        }
    }
    Ok(())
}

/// The vectors `docs` leave out (null) of their embedded fields, each text
/// embedded once.
fn vectors_of(models: &[Option<Arc<dyn Embedder>>], schema: &Schema, docs: &[Document], files: &Files) -> Result<Known> {
    let mut known = Known::new();
    for (f, e) in models.iter().enumerate() {
        let Some(e) = e else { continue };
        let mut seen = HashSet::new();
        let texts: Vec<String> = docs.iter().filter(|d| matches!(d[f], Value::Null)).map(|d| schema.embed_text(f, d)).filter(|t| !t.is_empty() && seen.insert(t.clone())).collect();
        if !texts.is_empty() {
            let vectors = embed_sources(e.as_ref(), schema, f, &texts, files)?;
            known.insert(f, texts.into_iter().zip(vectors).map(|(t, v)| (t, v.into())).collect());
        }
    }
    Ok(known)
}

/// The vectors of field `f` of `schema` for `sources` (as
/// [`Schema::embed_text`] gives them): of the texts, or of the images or
/// sounds that the MEDIA values name, decoded a batch at a time.
fn embed_sources(e: &dyn Embedder, schema: &Schema, f: usize, sources: &[String], files: &Files) -> Result<Vec<Vec<f32>>> {
    let Some(m) = schema.embeds_media(f) else { return embed_all(e, sources) };
    if schema.fields[m].media == Some(crate::media::MediaKind::Audio) {
        return in_sound_batches(&schema.fields[m].name, sources, files, |inputs| e.embed_inputs(inputs));
    }
    let mut out = Vec::with_capacity(sources.len());
    for batch in sources.chunks(16) {
        let images = images_of(&schema.fields[m].name, batch, files)?;
        let inputs: Vec<Input> = images.iter().map(Input::Image).collect();
        out.extend(e.embed_inputs(&inputs)?);
    }
    Ok(out)
}

/// The images MEDIA values of field `field` name, decoded.
fn images_of(field: &str, values: &[String], files: &Files) -> Result<Vec<crate::embed::Image>> {
    values
        .iter()
        .map(|t| {
            let m = Media::parse(t).map_err(Error::Schema)?;
            let data = files.bytes(&m)?;
            crate::media::decode_image(&data).map_err(|e| Error::Schema(format!("{}: {e}", m.uri)))
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|e| media_error(field, e))
}

/// Makes the derived text `docs` leave out (null) with `derivers` (by
/// field of `schema`), taking it from `known` where it has the file's, and
/// fills in again the generated fields of the documents that got text. A
/// document without a file gets none.
fn derive_known(derivers: &[Option<Arc<dyn Deriver>>], schema: &Schema, docs: &mut [Document], known: &Derived, files: &Files) -> Result<()> {
    let mut changed = vec![false; docs.len()];
    for (f, d) in derivers.iter().enumerate() {
        let (Some(d), Some(m)) = (d, schema.derives_from(f)) else { continue };
        let ahead = known.get(&f);
        let mut todo = Vec::new();
        let mut values = Vec::new();
        for (j, doc) in docs.iter_mut().enumerate() {
            let Value::Str(v) = &doc[m] else { continue };
            if !matches!(doc[f], Value::Null) {
                continue;
            }
            changed[j] = true;
            match ahead.and_then(|a| a.get(v)) {
                Some(t) => doc[f] = Value::Str(t.clone()),
                None => {
                    todo.push(j);
                    values.push(v.clone());
                }
            }
        }
        for (j, t) in todo.into_iter().zip(derive_sources(d.as_ref(), schema, m, &values, files)?) {
            docs[j][f] = Value::Str(t);
        }
    }
    if schema.fields.iter().any(|f| !f.generated_from.is_empty()) {
        docs.iter_mut().zip(changed).filter(|(_, c)| *c).for_each(|(d, _)| schema.fill_generated(d));
    }
    Ok(())
}

/// The text `docs` leave out (null) of their derived fields, each file
/// read once.
fn derived_of(derivers: &[Option<Arc<dyn Deriver>>], schema: &Schema, docs: &[Document], files: &Files) -> Result<Derived> {
    let mut known = Derived::new();
    for (f, d) in derivers.iter().enumerate() {
        let (Some(d), Some(m)) = (d, schema.derives_from(f)) else { continue };
        let mut seen = HashSet::new();
        let values: Vec<String> = docs.iter().filter(|doc| matches!(doc[f], Value::Null)).filter_map(|doc| doc[m].as_str()).filter(|v| seen.insert(v.to_string())).map(String::from).collect();
        if !values.is_empty() {
            let texts = derive_sources(d.as_ref(), schema, m, &values, files)?;
            known.insert(f, values.into_iter().zip(texts).collect());
        }
    }
    Ok(known)
}

/// The text `d` makes of the files that `values` of MEDIA field `m` name,
/// decoded a batch at a time: 16 images, or up to 4 sounds of 2 minutes
/// together (a sound can be an hour long).
fn derive_sources(d: &dyn Deriver, schema: &Schema, m: usize, values: &[String], files: &Files) -> Result<Vec<String>> {
    let field = &schema.fields[m].name;
    if d.modality() == Modality::Audio {
        return in_sound_batches(field, values, files, |inputs| d.derive(inputs));
    }
    let mut out = Vec::with_capacity(values.len());
    for batch in values.chunks(16) {
        let images = images_of(field, batch, files)?;
        let inputs: Vec<Input> = images.iter().map(Input::Image).collect();
        out.extend(d.derive(&inputs)?);
    }
    Ok(out)
}

/// What `f` makes of the sounds that MEDIA values of field `field` name,
/// decoded and given to it up to 4 at a time, or 2 minutes of sound
/// together (a sound can be an hour long).
fn in_sound_batches<T>(field: &str, values: &[String], files: &Files, mut f: impl FnMut(&[Input<'_>]) -> Result<Vec<T>>) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(values.len());
    let mut batch: Vec<crate::embed::Audio> = Vec::new();
    for (i, v) in values.iter().enumerate() {
        batch.extend(sounds_of(field, std::slice::from_ref(v), files)?);
        let seconds: f64 = batch.iter().map(|a| a.samples.len() as f64 / a.rate.max(1) as f64).sum();
        if batch.len() == 4 || seconds >= 120.0 || i + 1 == values.len() {
            out.extend(f(&batch.iter().map(Input::Audio).collect::<Vec<_>>())?);
            batch.clear();
        }
    }
    Ok(out)
}

/// The sounds MEDIA values of field `field` name, decoded.
fn sounds_of(field: &str, values: &[String], files: &Files) -> Result<Vec<crate::embed::Audio>> {
    values
        .iter()
        .map(|t| {
            let m = Media::parse(t).map_err(Error::Schema)?;
            let data = files.bytes(&m)?;
            crate::media::decode_audio(&data).map_err(|e| Error::Schema(format!("{}: {e}", m.uri)))
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|e| media_error(field, e))
}

/// What setting fields of live rows writes ([`plan_sets`]).
struct Planned {
    /// Key, field and value of each number set in place.
    in_place: Vec<(Value, usize, Value)>,
    /// The rows rewritten, the vectors their new sources make stale left
    /// out (null).
    rewrites: Vec<Document>,
    rows: u64,
}

/// Plans setting fields of live rows: each pair is a row (every field, as
/// a SELECT of them all returns it) and, for `ON CONFLICT DO UPDATE`, the
/// document the INSERT proposed ([`Assign::Excluded`]). Only single-valued
/// numbers change in place; any other field rewrites the row, which needs
/// every text field stored or generated.
fn plan_sets(schema: &Schema, models: &[Option<Arc<dyn Embedder>>], derivers: &[Option<Arc<dyn Deriver>>], rows: Vec<(Vec<Value>, Document)>, sets: &[(usize, Assign)]) -> Result<Planned> {
    let pk = schema.primary_key.ok_or_else(|| Error::Query(format!("UPDATE needs a primary key, and {} has none", schema.table)))?;
    if let Some(f) = sets.iter().map(|(f, _)| &schema.fields[*f]).find(|f| f.semantic_id.is_some()) {
        let from = &f.semantic_id.as_ref().expect("a semantic-ID field").from;
        return Err(Error::Query(format!("{} is a semantic ID: the writer derives it from {from}; set {from} instead", f.name)));
    }
    if let Some(f) = schema.shard_by_field().filter(|f| sets.iter().any(|(g, _)| g == f)) {
        return Err(Error::Query(format!("{} places {}'s rows on shards (shard_by), so UPDATE does not change it: delete the row and insert it again", schema.fields[f].name, schema.table)));
    }
    let rewrite = !in_place(schema, sets);
    if rewrite {
        if let Some(f) = schema.fields.iter().find(|f| matches!(f.ty, FieldType::Text { .. }) && !f.stored && f.generated_from.is_empty()) {
            return Err(Error::Query(format!("UPDATE of non-numeric fields rewrites documents, and {} is not stored", f.name)));
        }
    }
    // A rewrite that changes the file of a derived field makes its text
    // again, unless it sets the text itself.
    let remade: Vec<usize> = (0..schema.fields.len())
        .filter(|&f| derivers[f].is_some() && !sets.iter().any(|(g, _)| *g == f))
        .filter(|&f| sets.iter().any(|(g, _)| schema.derives_from(f) == Some(*g)))
        .collect();
    // A rewrite that changes a source of an embedded field (or the text
    // made again) recomputes its vector, unless it sets the vector itself.
    let stale: Vec<usize> = (0..schema.fields.len())
        .filter(|&f| models[f].is_some() && !sets.iter().any(|(g, _)| *g == f))
        .filter(|&f| {
            let from = &schema.fields[f].embed.as_ref().expect("embedded").from;
            sets.iter().map(|(g, _)| *g).chain(remade.iter().copied()).any(|g| from.iter().any(|s| schema.field_index(s) == Some(g)) || generated_from_set(schema, from, g))
        })
        .collect();
    // Expressions read generated fields, which need not be stored.
    let generated = schema.fields.iter().any(|f| !f.generated_from.is_empty()) && sets.iter().any(|(_, a)| matches!(a, Assign::Expr(_)));
    let (mut in_place, mut rewrites, mut n) = (Vec::new(), Vec::new(), 0u64);
    for (mut row, proposed) in rows {
        if generated {
            schema.fill_generated(&mut row);
        }
        // The row as an expression reads it: every right-hand side sees
        // the row as it was. A multi-valued field without values is an
        // empty list, as queries compare it.
        let read = |r: Ref| -> Result<Value> {
            let (doc, g) = match r {
                Ref::Field(g) => (&row, g as usize),
                Ref::Excluded(g) => (&proposed, g as usize),
                r => return Err(Error::Query(format!("SET reads fields of the row, not {r:?}"))),
            };
            let field = schema.fields.get(g).ok_or_else(|| Error::Query(format!("{} has no field #{g}", schema.table)))?;
            Ok(match doc.get(g).cloned().unwrap_or(Value::Null) {
                Value::Null if field.multi => Value::List(Vec::new()),
                v => field.output(v),
            })
        };
        let new = |f: usize, a: &Assign| -> Result<Value> {
            Ok(match a {
                Assign::Value(v) => v.clone(),
                Assign::Excluded(g) => proposed[*g].clone(),
                Assign::Expr(e) => qsql::assigned(&schema.fields[f], eval::eval(e, &read)?)?,
                // NULL counts as zero; a sum that overflows is an
                // error, as in Postgres.
                Assign::Add(x) => {
                    let overflow = || Error::Query(format!("{} + {x:?} overflows", schema.fields[f].name));
                    match (&schema.fields[f].ty, x) {
                        (FieldType::Int, Value::Int(x)) => Value::Int(row[f].as_i64().unwrap_or(0).checked_add(*x).ok_or_else(overflow)?),
                        (FieldType::Int, Value::Float(y)) if y.fract() == 0.0 && y.abs() < 9.2e18 => {
                            Value::Int(row[f].as_i64().unwrap_or(0).checked_add(*y as i64).ok_or_else(overflow)?)
                        }
                        (FieldType::Int, _) => return Err(Error::Query(format!("{} is INT: add an integer", schema.fields[f].name))),
                        _ => {
                            let y = row[f].as_f64().unwrap_or(0.0) + x.as_f64().unwrap_or(0.0);
                            if !y.is_finite() {
                                return Err(overflow());
                            }
                            Value::Float(y)
                        }
                    }
                }
            })
        };
        if !rewrite {
            for (f, a) in sets {
                in_place.push((row[pk].clone(), *f, new(*f, a)?));
            }
        } else {
            let mut doc = row.clone();
            for (f, a) in sets {
                doc[*f] = new(*f, a)?;
            }
            for &f in stale.iter().chain(&remade) {
                doc[f] = Value::Null;
            }
            // A semantic ID whose vector changes is derived again (the
            // writer keeps it if the levels stay the same).
            for (f, vf) in crate::sid::fields(schema) {
                if stale.contains(&vf) || sets.iter().any(|(g, _)| *g == vf) {
                    doc[f] = Value::Null;
                }
            }
            schema.fill_generated(&mut doc);
            rewrites.push(doc);
        }
        n += 1;
    }
    Ok(Planned { in_place, rewrites, rows: n })
}

/// Whether setting `sets` changes rows in place: every field set is a
/// single-valued number.
fn in_place(schema: &Schema, sets: &[(usize, Assign)]) -> bool {
    sets.iter().all(|(f, _)| matches!(schema.fields[*f].ty, FieldType::Int | FieldType::Float) && !schema.fields[*f].multi)
}

/// Sets fields of live rows ([`plan_sets`]), making the text and the
/// vectors the rewrites leave out (from `known` where it has them).
fn set_rows(
    tt: &mut TableTxn,
    schema: &Schema,
    (models, derivers): (&[Option<Arc<dyn Embedder>>], &[Option<Arc<dyn Deriver>>]),
    rows: Vec<(Vec<Value>, Document)>,
    sets: &[(usize, Assign)],
    (known, derived): (&Known, &Derived),
    files: &Files,
) -> Result<u64> {
    let mut p = plan_sets(schema, models, derivers, rows, sets)?;
    for (key, f, v) in p.in_place {
        tt.set(&key, &schema.fields[f].name, v)?;
    }
    derive_known(derivers, schema, &mut p.rewrites, derived, files)?;
    embed_known(models, schema, &mut p.rewrites, known, files)?;
    for doc in p.rewrites {
        tt.upsert(doc)?;
    }
    Ok(p.rows)
}

/// The documents of an INSERT whose key is neither live nor proposed
/// earlier by the same statement (`INSERT OR IGNORE`, `ON CONFLICT DO
/// NOTHING`).
fn not_live(snap: &Snapshot, pk: usize, mut docs: Vec<Document>) -> Vec<Document> {
    let mut seen = HashSet::new();
    docs.retain(|d| snap.lookup(&d[pk]).is_none() && seen.insert(Part::of(&d[pk])));
    docs
}

/// `e`'s vectors of `texts`, split across the cores for a model computed
/// here ([`Embedder::parallel`]).
fn embed_all(e: &dyn Embedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let threads = if e.parallel() { std::thread::available_parallelism().map_or(1, |n| n.get()).min(refs.len() / 64).max(1) } else { 1 };
    let vectors = if threads == 1 {
        e.embed(&refs)?
    } else {
        std::thread::scope(|s| {
            let parts: Vec<_> = refs.chunks(refs.len().div_ceil(threads)).map(|c| s.spawn(move || e.embed(c))).collect();
            parts.into_iter().map(|p| p.join().expect("embedding thread")).collect::<Result<Vec<_>>>()
        })?
        .into_iter()
        .flatten()
        .collect()
    };
    if vectors.len() != texts.len() {
        return Err(Error::Schema(format!("model {} made {} vectors of {} texts", e.name(), vectors.len(), texts.len())));
    }
    Ok(vectors)
}

fn with_source(mut opts: DbOptions) -> DbOptions {
    if opts.doc_source.is_none() {
        opts.doc_source = Some(Arc::new(StoredSource));
    }
    opts
}

/// Rebuilds documents for merges: stored fields from the doc store, vectors
/// and unstored structured fields from their indexes, generated text from
/// its sources. Unstored TEXT or SPARSE has none, so a merge of a schema
/// with one (from before [`Schema::new`] refused them) fails instead of
/// emptying its index.
pub struct StoredSource;

impl DocumentSource for StoredSource {
    fn documents(&self, schema: &Schema, seg: &SegmentFile, docs: &[DocId]) -> Result<Vec<Document>> {
        if let Some(f) = schema.unrebuildable() {
            return Err(crate::Error::Schema(format!("cannot merge: {} is neither stored nor generated, so a merge would erase its index", f.name)));
        }
        let store = if schema.fields.iter().any(|f| f.stored) { Some(DocStore::open(seg, schema)?) } else { None };
        let mut vectors = Vec::new();
        let mut numeric = Vec::new();
        let mut keyword = Vec::new();
        let mut edges = Vec::new();
        for (i, f) in schema.fields.iter().enumerate() {
            match f.ty {
                FieldType::Edges if !f.stored && f.indexed => {
                    if let Some(g) = crate::graph::GraphReader::open(seg, &f.key())? {
                        edges.push((i, g));
                    }
                }
                FieldType::Vector { .. } => vectors.push((i, VectorReader::open(seg, &f.key())?)),
                FieldType::Int | FieldType::Float if !f.stored && f.indexed => numeric.push((i, NumericColumn::open(seg, &f.key())?)),
                FieldType::Keyword if !f.stored && f.indexed => keyword.push((i, KeywordColumn::open(seg, &f.key())?)),
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
                match v.row_value(d) {
                    Value::Null => {}
                    x => doc[*i] = x,
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

/// The plan of a SELECT over `snap`'s tables: the executor's query of one
/// table, or a compound one's operators ([`compound::needs`]) over any
/// ([`compound::plan_read`]).
fn plan_of(snap: &Snapshot, q: &sqlparser::ast::Query, params: &Params) -> Result<Plan> {
    let catalog: Vec<&Schema> = snap.schemas().collect();
    compound::plan_read_query(q, &catalog, params, &|t, name| snap.table(t).and_then(|v| v.rank_profile(name).cloned()))
}
