//! Python bindings for Strata, the database in-process: the `strata._strata`
//! extension module under the `strata` package (`python/strata`), which
//! wraps it in the documented API.
//!
//! Rows cross the boundary as JSON-like Python values (dicts, lists,
//! strings, numbers, None), read by the same code as files and the HTTP
//! API ([`strata::ingest`]). Every call that reads or writes the database
//! releases the GIL.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value as Json};
use strata::db::{Db, SqlResult};
use strata::ingest::{self, Format};
use strata::query::ir::Plan;
use strata::query::sql::{ident, quoted};
use strata::query::Params;
use strata::query::exec::ResultSet;
use strata::rank_log::{Export, Log};
use strata::schema::Schema;
use strata::sid::{Codebook, TrainOptions};
use strata::sqlite;
use strata::storage::{DbOptions, Layout, MergePolicy};
use strata::{FieldType, Value};

create_exception!(_strata, StrataError, PyException, "An error from the database: a bad schema, query, value or file.");

fn err(e: strata::Error) -> PyErr {
    StrataError::new_err(e.to_string())
}

/// Rows a write transaction carries (as `strata-server import` does).
const BATCH: usize = ingest::IMPORT_BATCH;
/// Inserts of at least this many rows are bulk-loaded into new segments in
/// one commit; fewer go through the write-ahead log.
const BULK_INSERT: usize = 1000;

/// A Python value as JSON. Numpy arrays and scalars (anything with
/// `tolist()`) become lists and numbers.
fn to_json(o: &Bound<'_, PyAny>) -> PyResult<Json> {
    if o.is_none() {
        return Ok(Json::Null);
    }
    if let Ok(b) = o.cast::<PyBool>() {
        return Ok(Json::Bool(b.is_true()));
    }
    if let Ok(i) = o.cast::<PyInt>() {
        return Ok(Json::Number(i.extract::<i64>()?.into()));
    }
    if let Ok(f) = o.cast::<PyFloat>() {
        return Ok(Number::from_f64(f.value()).map_or(Json::Null, Json::Number));
    }
    if let Ok(s) = o.cast::<PyString>() {
        return Ok(Json::String(s.to_str()?.to_string()));
    }
    if let Ok(d) = o.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in d.iter() {
            let k = k.cast::<PyString>().map_err(|_| PyValueError::new_err(format!("keys are strings, not {}", k.repr().map(|r| r.to_string()).unwrap_or_default())))?;
            m.insert(k.to_str()?.to_string(), to_json(&v)?);
        }
        return Ok(Json::Object(m));
    }
    if let Ok(l) = o.cast::<PyList>() {
        return l.iter().map(|x| to_json(&x)).collect::<PyResult<_>>().map(Json::Array);
    }
    if let Ok(t) = o.cast::<PyTuple>() {
        return t.iter().map(|x| to_json(&x)).collect::<PyResult<_>>().map(Json::Array);
    }
    if o.hasattr("tolist")? {
        return to_json(&o.call_method0("tolist")?);
    }
    // A datetime or date, as ISO 8601 text (a TIMESTAMP's form; a naive
    // datetime is taken as UTC).
    if o.hasattr("isoformat")? && o.hasattr("year")? {
        return Ok(Json::String(o.call_method0("isoformat")?.extract()?));
    }
    Err(PyValueError::new_err(format!("cannot store a {}: values are None, bool, int, float, str, datetime, lists and dicts", o.get_type().name()?)))
}

/// A stored value as Python.
fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Vector(xs) => PyList::new(py, xs.iter().map(|&x| x as f64))?.into_any(),
        Value::List(xs) => PyList::new(py, xs.iter().map(|x| to_py(py, x)).collect::<PyResult<Vec<_>>>()?)?.into_any(),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        // As the HTTP API gives it, so that Database and Client agree.
        Value::Timestamp(t) => strata::timestamp::format(*t).into_pyobject(py)?.into_any(),
    })
}

/// A query's answer as a dict: columns, rows (lists), keys, scores, stats.
fn result_py<'py>(py: Python<'py>, rs: &ResultSet) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("columns", PyList::new(py, &rs.columns)?)?;
    let rows = rs.rows.iter().map(|r| PyList::new(py, r.iter().map(|v| to_py(py, v)).collect::<PyResult<Vec<_>>>()?)).collect::<PyResult<Vec<_>>>()?;
    d.set_item("rows", PyList::new(py, rows)?)?;
    d.set_item("keys", PyList::new(py, rs.keys.iter().map(|v| to_py(py, v)).collect::<PyResult<Vec<_>>>()?)?)?;
    d.set_item("scores", PyList::new(py, &rs.scores)?)?;
    d.set_item("stats", serde_json::to_string(&rs.stats).unwrap_or_default())?;
    Ok(d)
}

/// A query parameter, as the HTTP API reads one: a number, string, None,
/// a list of numbers (a vector), a list, a dict of term weights (SPARSE),
/// `{"embed": text, "field": f}` (the text's vector by `f`'s model, or an
/// image or sound file's for `{"embed": {"image": "<base64>"}}` or
/// `{"embed": {"audio": "<base64>"}}`) or
/// `{"vector_of": key, "field": f}` (a stored row's vector), each with
/// `"table"` for a field of another table than the first.
fn param(db: &Db, j: &Json) -> strata::Result<Value> {
    let bad = |m: String| strata::Error::Query(m);
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => match n.as_i64() {
            Some(x) => Value::Int(x),
            None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Json::String(s) => Value::Str(s.clone()),
        // Whole numbers stay exact integers (`IN (:ids)` with 64-bit ids);
        // where a vector is expected, a list of numbers is one.
        Json::Array(xs) if !xs.is_empty() && xs.iter().all(|x| x.is_i64()) => Value::List(xs.iter().map(|x| Value::Int(x.as_i64().unwrap_or(0))).collect()),
        Json::Array(xs) if !xs.is_empty() && xs.iter().all(|x| x.is_number()) => Value::Vector(xs.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>().into()),
        Json::Array(xs) => Value::List(xs.iter().map(|x| param(db, x)).collect::<strata::Result<_>>()?),
        Json::Object(o) if o.get("embed").is_some_and(|e| e.is_string() || e.get("image").is_some() || e.get("audio").is_some()) && o.keys().all(|k| k == "embed" || k == "field" || k == "table") => {
            let s = table_of(db, o.get("table").and_then(|t| t.as_str()))?;
            let field = match o.get("field").and_then(|f| f.as_str()) {
                Some(f) => f.to_string(),
                None => s.fields.iter().find(|f| db.embedder_in(&s.table, &f.name).is_some()).map(|f| f.name.clone()).ok_or_else(|| bad("no embedded field has its model open".into()))?,
            };
            let e = db.embedder_in(&s.table, &field).ok_or_else(|| bad(format!("{field} has no embedding model open (see Database.open's models)")))?;
            match &o["embed"] {
                Json::String(t) => Value::Vector(e.embed_query(t)?.into()),
                Json::Object(i) if i.len() == 1 && i.values().all(|v| v.is_string()) => {
                    let (kind, b64) = i.iter().next().expect("one key");
                    let (_, data) = strata::embed::query_file(kind, b64.as_str().unwrap_or_default())?;
                    Value::Vector(strata::embed::embed_file(e.as_ref(), &data)?.into())
                }
                _ => return Err(bad("\"embed\" is a text, {\"image\": \"<base64>\"} or {\"audio\": \"<base64>\"}".into())),
            }
        }
        Json::Object(o) if o.contains_key("vector_of") && o.keys().all(|k| k == "vector_of" || k == "field" || k == "table") => {
            let s = &*table_of(db, o.get("table").and_then(|t| t.as_str()))?;
            let pk = s.primary_key.ok_or_else(|| bad("vector_of needs a primary key".into()))?;
            let field = match o.get("field").and_then(|f| f.as_str()) {
                Some(f) => f.to_string(),
                None => s.fields.iter().find(|f| matches!(f.ty, FieldType::Vector { .. })).map(|f| f.name.clone()).ok_or_else(|| bad("the table has no VECTOR field".into()))?,
            };
            // Quoted: a field may be named like a keyword (`order`).
            let sql = format!("SELECT {} FROM {} WHERE {} = :k LIMIT 1", quoted(&field), ident(&s.table), quoted(&s.fields[pk].name));
            let rs = db.query(&sql, &Params::new().set("k", ingest::value(s, pk, &o["vector_of"])?))?;
            match rs.rows.into_iter().next().and_then(|r| r.into_iter().next()) {
                Some(v @ Value::Vector(_)) => v,
                Some(_) => return Err(bad(format!("row {} has no {field}", o["vector_of"]))),
                None => return Err(bad(format!("no row {}", o["vector_of"]))),
            }
        }
        Json::Object(m) if m.values().all(|w| w.is_number()) => Value::Str(j.to_string()),
        Json::Object(_) => return Err(bad("parameters are numbers, strings, None, lists, dicts of term weights, {\"embed\": text} or {\"vector_of\": key}".into())),
    })
}

fn params(db: &Db, j: &[(String, Json)]) -> strata::Result<Params> {
    let mut p = Params::new();
    for (k, v) in j {
        p = p.set(k, param(db, v)?);
    }
    Ok(p)
}

fn params_json(p: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Json)>> {
    let Some(p) = p else { return Ok(Vec::new()) };
    p.iter().map(|(k, v)| Ok((k.extract::<String>()?, to_json(&v)?))).collect()
}

fn rows_json(rows: &Bound<'_, PyAny>) -> PyResult<Vec<Json>> {
    if rows.cast::<PyDict>().is_ok() {
        return Ok(vec![to_json(rows)?]);
    }
    rows.try_iter()?.map(|r| to_json(&r?)).collect()
}

/// A database file (or directory) opened in this process. Only one process
/// may have a database open at a time.
#[pyclass(frozen, module = "strata._strata")]
struct Database {
    db: RwLock<Option<Arc<Db>>>,
    /// Embedded fields whose model did not open, with the reason.
    #[pyo3(get)]
    embed_errors: Vec<(String, String)>,
}

impl Database {
    fn new(db: Db, models: Option<PathBuf>) -> Self {
        let models = models.unwrap_or_else(strata::embed::models_dir);
        let embed_errors = db.open_embedders(&models).into_iter().map(|(f, e)| (f, e.to_string())).collect();
        Database { db: RwLock::new(Some(Arc::new(db))), embed_errors }
    }

    fn get(&self) -> PyResult<Arc<Db>> {
        self.db.read().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| StrataError::new_err("the database is closed"))
    }

    fn sqlite_options(tables: Option<Vec<String>>, schemas: Option<&str>, blobs_base64: bool) -> PyResult<sqlite::ImportOptions> {
        let schemas = schemas.map(sqlite::schemas_from_json).transpose().map_err(err)?.unwrap_or_default();
        Ok(sqlite::ImportOptions { tables: tables.unwrap_or_default(), schemas, blobs_base64 })
    }

    fn documents(db: &Db, schema: &Schema, rows: &[Json]) -> strata::Result<Vec<strata::Document>> {
        let now = strata::timestamp::now();
        let mut docs = rows.iter().enumerate().map(|(i, j)| ingest::document_at(schema, j, now).map_err(|e| strata::Error::Schema(format!("row {}: {e}", i + 1)))).collect::<strata::Result<Vec<_>>>()?;
        db.fill_in(&schema.table, &mut docs)?;
        Ok(docs)
    }
}

/// The schema of the table named `table` (any case), or of the first.
fn table_of(db: &Db, table: Option<&str>) -> strata::Result<Arc<Schema>> {
    match table {
        None => Ok(db.catalog().swap_remove(0)),
        Some(t) => db.storage().table(t).ok_or_else(|| strata::Error::Query(format!("no such table: {t}"))),
    }
}

#[pymethods]
impl Database {
    /// Creates a database at `path` with a schema (JSON text, either form).
    #[staticmethod]
    #[pyo3(signature = (path, schema, directory = false, models = None))]
    fn create(py: Python<'_>, path: PathBuf, schema: &str, directory: bool, models: Option<PathBuf>) -> PyResult<Self> {
        let schema = Schema::from_json(schema).map_err(err)?;
        let opts = DbOptions { layout: if directory { Layout::Directory } else { Layout::SingleFile }, ..DbOptions::default() };
        let models = models.unwrap_or_else(strata::embed::models_dir);
        let db = py.detach(|| Db::create_in(&path, schema, opts, &models)).map_err(err)?;
        Ok(Database::new(db, Some(models)))
    }

    /// Opens the database at `path`.
    #[staticmethod]
    #[pyo3(signature = (path, models = None))]
    fn open(py: Python<'_>, path: PathBuf, models: Option<PathBuf>) -> PyResult<Self> {
        let db = py.detach(|| Db::open(&path, DbOptions::default())).map_err(err)?;
        Ok(Database::new(db, models))
    }

    /// The schema of the first table, or of `table`, in its short JSON form.
    #[pyo3(signature = (table = None))]
    fn schema_json(&self, table: Option<&str>) -> PyResult<String> {
        Ok(table_of(&*self.get()?, table).map_err(err)?.to_short_json().to_string())
    }

    /// Every table's schema in its short JSON form, the first first.
    fn tables_json(&self) -> PyResult<Vec<String>> {
        Ok(self.get()?.catalog().iter().map(|s| s.to_short_json().to_string()).collect())
    }

    /// Live rows.
    fn count(&self) -> PyResult<u64> {
        Ok(self.get()?.snapshot().n_live_docs())
    }

    /// Runs a SELECT; returns {columns, rows, keys, scores, stats}.
    #[pyo3(signature = (sql, params = None))]
    fn query<'py>(&self, py: Python<'py>, sql: &str, params: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        let db = self.get()?;
        let p = params_json(params)?;
        let rs = py.detach(|| self::params(&db, &p).and_then(|p| db.query(sql, &p))).map_err(err)?;
        result_py(py, &rs)
    }

    /// Runs any statement: a SELECT's answer as `query` gives it, or the
    /// number of rows an INSERT, UPDATE or DELETE wrote.
    #[pyo3(signature = (sql, params = None))]
    fn sql<'py>(&self, py: Python<'py>, sql: &str, params: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
        let db = self.get()?;
        let p = params_json(params)?;
        match py.detach(|| self::params(&db, &p).and_then(|p| db.sql(sql, &p))).map_err(err)? {
            SqlResult::Rows(rs) => Ok(result_py(py, &rs)?.into_any()),
            SqlResult::Written(n) => Ok(n.into_pyobject(py)?.into_any()),
        }
    }

    /// Runs a SELECT and logs what it served (`log`: the options as JSON
    /// text, as strata-server's `"log"` takes them); returns its answer as
    /// `query` does, with the request's id.
    fn query_logged<'py>(&self, py: Python<'py>, sql: &str, params: Option<&Bound<'py, PyDict>>, log: &str) -> PyResult<Bound<'py, PyDict>> {
        let db = self.get()?;
        let p = params_json(params)?;
        let log: Json = serde_json::from_str(log).map_err(|e| StrataError::new_err(format!("log options are JSON: {e}")))?;
        let l = py.detach(|| Log::from_json(&log).and_then(|log| self::params(&db, &p).and_then(|p| db.query_logged(sql, &p, &log)))).map_err(err)?;
        let d = result_py(py, &l.rs)?;
        d.set_item("request", l.request)?;
        Ok(d)
    }

    /// The records of a log table (`export`: the options as JSON text, as
    /// strata-server's `/v1/rank_log/export` takes them), as JSON text of a
    /// list.
    fn export_rank_log(&self, py: Python<'_>, export: &str) -> PyResult<String> {
        let db = self.get()?;
        let x: Json = serde_json::from_str(export).map_err(|e| StrataError::new_err(format!("export options are JSON: {e}")))?;
        py.detach(|| {
            let x = Export::from_json(&x)?;
            let mut out = Vec::new();
            db.export_rank_log(&x, |r| Ok(out.push(r)))?;
            Ok(Json::Array(out).to_string())
        })
        .map_err(err)
    }

    /// The plan of a SELECT, INSERT, UPDATE or DELETE as JSON text, without
    /// running it.
    #[pyo3(signature = (sql, params = None))]
    fn plan(&self, py: Python<'_>, sql: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        let db = self.get()?;
        let p = params_json(params)?;
        py.detach(|| self::params(&db, &p).and_then(|p| db.explain(sql, &p)).and_then(|plan| plan.to_json_text())).map_err(err)
    }

    /// Runs a plan given as JSON text, as `sql` runs a statement.
    fn run<'py>(&self, py: Python<'py>, plan: &str) -> PyResult<Bound<'py, PyAny>> {
        let db = self.get()?;
        let plan: Json = serde_json::from_str(plan).map_err(|e| StrataError::new_err(format!("a plan is JSON: {e}")))?;
        match py.detach(|| Plan::from_json(&plan).and_then(|plan| db.run(&plan))).map_err(err)? {
            SqlResult::Rows(rs) => Ok(result_py(py, &rs)?.into_any()),
            SqlResult::Written(n) => Ok(n.into_pyobject(py)?.into_any()),
        }
    }

    /// Inserts rows (keys must be new) into the first table or `table`;
    /// all or none are written.
    #[pyo3(signature = (rows, table = None))]
    fn insert(&self, py: Python<'_>, rows: &Bound<'_, PyAny>, table: Option<&str>) -> PyResult<u64> {
        let db = self.get()?;
        let rows = rows_json(rows)?;
        py.detach(|| -> strata::Result<u64> {
            let schema = table_of(&db, table)?;
            let docs = Database::documents(&db, &schema, &rows)?;
            let n = docs.len() as u64;
            if docs.len() >= BULK_INSERT {
                db.load_into(&schema.table, docs)?;
            } else {
                let mut t = db.storage().begin()?;
                let mut tt = t.table(&schema.table)?;
                for d in docs {
                    tt.insert(d)?;
                }
                t.commit()?;
            }
            Ok(n)
        })
        .map_err(err)
    }

    /// Inserts rows or replaces the rows with their keys, in the first
    /// table or `table`, in transactions of 10,000 rows.
    #[pyo3(signature = (rows, table = None))]
    fn upsert(&self, py: Python<'_>, rows: &Bound<'_, PyAny>, table: Option<&str>) -> PyResult<u64> {
        let db = self.get()?;
        let rows = rows_json(rows)?;
        py.detach(|| -> strata::Result<u64> {
            let schema = table_of(&db, table)?;
            let mut n = 0;
            for chunk in rows.chunks(BATCH) {
                let docs = Database::documents(&db, &schema, chunk)?;
                let mut t = db.storage().begin()?;
                let mut tt = t.table(&schema.table)?;
                for d in docs {
                    tt.upsert(d)?;
                }
                t.commit()?;
                n += chunk.len() as u64;
            }
            Ok(n)
        })
        .map_err(err)
    }

    /// Deletes the rows with these keys from the first table or `table`;
    /// returns how many there were.
    #[pyo3(signature = (keys, table = None))]
    fn delete(&self, py: Python<'_>, keys: &Bound<'_, PyAny>, table: Option<&str>) -> PyResult<u64> {
        let db = self.get()?;
        let keys = if keys.cast::<PyString>().is_ok() || keys.cast::<PyInt>().is_ok() { vec![to_json(keys)?] } else { rows_json(keys)? };
        py.detach(|| -> strata::Result<u64> {
            let s = table_of(&db, table)?;
            let pk = s.primary_key.ok_or_else(|| strata::Error::Query("deleting by key needs a primary key".into()))?;
            let mut t = db.storage().begin()?;
            let mut tt = t.table(&s.table)?;
            let mut n = 0;
            for k in &keys {
                n += tt.delete(&ingest::value(&s, pk, k)?)? as u64;
            }
            t.commit()?;
            Ok(n)
        })
        .map_err(err)
    }

    /// Applies write operations as strata-server's /v1/write takes them, in
    /// one transaction: insert, upsert and delete, and set and add of a
    /// numeric field in place (an add with a "doc" inserts it when no row
    /// has its key). Returns the rows written.
    fn write(&self, py: Python<'_>, ops: &Bound<'_, PyAny>) -> PyResult<u64> {
        let db = self.get()?;
        let ops = rows_json(ops)?;
        py.detach(|| -> strata::Result<u64> {
            let bad = |i: usize, m: &str| strata::Error::Query(format!("operation {}: {m}", i + 1));
            let mut t = db.storage().begin()?;
            let mut n = 0;
            for (i, op) in ops.iter().enumerate() {
                let get = |k: &str| op.get(k).filter(|v| !v.is_null());
                let kind = get("op").and_then(Json::as_str).ok_or_else(|| bad(i, "it needs \"op\""))?;
                let s = table_of(&db, get("table").and_then(Json::as_str))?;
                let doc = || match get("doc") {
                    Some(d) => Ok(Database::documents(&db, &s, std::slice::from_ref(d))?.remove(0)),
                    None => Err(bad(i, &format!("{kind} needs \"doc\""))),
                };
                let key = || {
                    let pk = s.primary_key.ok_or_else(|| bad(i, "writing by key needs a primary key"))?;
                    ingest::value(&s, pk, get("key").ok_or_else(|| bad(i, &format!("{kind} needs \"key\"")))?)
                };
                let field = || get("field").and_then(Json::as_str).ok_or_else(|| bad(i, &format!("{kind} needs \"field\"")));
                let mut tt = t.table(&s.table)?;
                let wrote = match kind {
                    "insert" => tt.insert(doc()?).map(|_| true)?,
                    "upsert" => tt.upsert(doc()?).map(|_| true)?,
                    "delete" => tt.delete(&key()?)?,
                    "set" => {
                        let f = field()?;
                        let fi = s.field_index(f).ok_or_else(|| bad(i, &format!("no field {f}")))?;
                        tt.set(&key()?, f, ingest::value(&s, fi, get("value").unwrap_or(&Json::Null))?)?
                    }
                    "add" => {
                        let delta = match get("value") {
                            Some(Json::Number(x)) => x.as_i64().map(Value::Int).or_else(|| x.as_f64().map(Value::Float)),
                            _ => None,
                        };
                        let delta = delta.ok_or_else(|| bad(i, "add needs a numeric value"))?;
                        if get("doc").is_some() {
                            let d = doc()?;
                            if get("key").is_some() && Some(&key()?) != s.primary_key.map(|pk| &d[pk]) {
                                return Err(bad(i, "add's key and its doc's key differ: give one"));
                            }
                            tt.add_or_insert(d, field()?, &delta).map(|_| true)?
                        } else {
                            tt.add(&key()?, field()?, &delta)?
                        }
                    }
                    other => return Err(bad(i, &format!("unknown op {other:?}"))),
                };
                n += wrote as u64;
            }
            t.commit()?;
            Ok(n)
        })
        .map_err(err)
    }

    /// Loads a JSON Lines, JSON, CSV or TSV file; returns (rows, bulk,
    /// skipped fields).
    #[pyo3(signature = (path, format = None, table = None))]
    fn import_file(&self, py: Python<'_>, path: PathBuf, format: Option<&str>, table: Option<&str>) -> PyResult<(u64, bool, Vec<String>)> {
        let db = self.get()?;
        let format = format.map(|f| Format::parse(f).ok_or_else(|| PyValueError::new_err(format!("unknown format {f:?}: jsonl, json, csv or tsv")))).transpose()?;
        let r = py.detach(|| table_of(&db, table).and_then(|s| ingest::import_file_into(&db, &s.table, &path, format))).map_err(err)?;
        Ok((r.rows, r.bulk, r.unknown))
    }

    /// Imports the tables of the SQLite file `path` ([`sqlite::import`]);
    /// returns what it did as JSON.
    #[pyo3(signature = (path, tables = None, schemas = None, blobs_base64 = false))]
    fn import_sqlite(&self, py: Python<'_>, path: PathBuf, tables: Option<Vec<String>>, schemas: Option<&str>, blobs_base64: bool) -> PyResult<String> {
        let db = self.get()?;
        let opts = Database::sqlite_options(tables, schemas, blobs_base64)?;
        let r = py.detach(|| sqlite::import(&db, &path, &opts)).map_err(err)?;
        Ok(r.to_json().to_string())
    }

    /// Writes the tables to a new SQLite file ([`sqlite::export`]); returns
    /// what it wrote as JSON.
    #[pyo3(signature = (path, tables = None, fts = true, indexes = true, schemas = true))]
    fn export_sqlite(&self, py: Python<'_>, path: PathBuf, tables: Option<Vec<String>>, fts: bool, indexes: bool, schemas: bool) -> PyResult<String> {
        let db = self.get()?;
        let opts = sqlite::ExportOptions { tables: tables.unwrap_or_default(), fts, indexes, meta: schemas };
        let r = py.detach(|| sqlite::export(&db, &path, &opts)).map_err(err)?;
        Ok(r.to_json().to_string())
    }

    /// Creates a database at `path` from the tables of the SQLite file
    /// `sqlite_path`; returns it and what the import did as JSON.
    #[staticmethod]
    #[pyo3(signature = (sqlite_path, path, tables = None, schemas = None, blobs_base64 = false, directory = false, models = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_sqlite(
        py: Python<'_>,
        sqlite_path: PathBuf,
        path: PathBuf,
        tables: Option<Vec<String>>,
        schemas: Option<&str>,
        blobs_base64: bool,
        directory: bool,
        models: Option<PathBuf>,
    ) -> PyResult<(Self, String)> {
        let opts = Database::sqlite_options(tables, schemas, blobs_base64)?;
        let db_opts = DbOptions { layout: if directory { Layout::Directory } else { Layout::SingleFile }, ..DbOptions::default() };
        let models = models.unwrap_or_else(strata::embed::models_dir);
        let (db, embed_errors, report) = py
            .detach(|| -> strata::Result<_> {
                let src = sqlite::Source::open(&sqlite_path)?;
                let plan = src.plan(None, &opts)?;
                let first = plan.tables.first().ok_or_else(|| strata::Error::Schema(format!("{}: no table to import", sqlite_path.display())))?;
                let db = Db::create_in(&path, first.schema.clone(), db_opts, &models)?;
                // Before the rows, so that they get their embeddings.
                let embed_errors: Vec<(String, String)> = db.open_embedders(&models).into_iter().map(|(f, e)| (f, e.to_string())).collect();
                let r = src.load(&db, &plan)?;
                Ok((db, embed_errors, r.to_json().to_string()))
            })
            .map_err(err)?;
        Ok((Database { db: RwLock::new(Some(Arc::new(db))), embed_errors }, report))
    }

    /// What an import of the SQLite file `sqlite_path` into a new database
    /// would do ([`sqlite::Plan::to_json`]).
    #[staticmethod]
    #[pyo3(signature = (sqlite_path, tables = None, schemas = None, blobs_base64 = false))]
    fn plan_sqlite(py: Python<'_>, sqlite_path: PathBuf, tables: Option<Vec<String>>, schemas: Option<&str>, blobs_base64: bool) -> PyResult<String> {
        let opts = Database::sqlite_options(tables, schemas, blobs_base64)?;
        let plan = py.detach(|| sqlite::plan(&sqlite_path, None, &opts)).map_err(err)?;
        Ok(plan.to_json().to_string())
    }

    /// The vector `field`'s model gives `text` as a query, or `image` or
    /// `audio`, the bytes of an image or sound file, for a model of that.
    #[pyo3(signature = (text = None, field = None, table = None, image = None, audio = None))]
    fn embed(&self, py: Python<'_>, text: Option<&str>, field: Option<&str>, table: Option<&str>, image: Option<Vec<u8>>, audio: Option<Vec<u8>>) -> PyResult<Vec<f32>> {
        use strata::embed::Modality;
        let db = self.get()?;
        let file = match (image, audio) {
            (Some(_), Some(_)) => return Err(PyValueError::new_err("embed takes a text, an image or a sound, one of them")),
            (Some(d), None) => Some((Modality::Image, d)),
            (None, Some(d)) => Some((Modality::Audio, d)),
            (None, None) => None,
        };
        if let Some((m, data)) = file {
            let s = table_of(&db, table).map_err(err)?;
            let what = strata::embed::modality_name(m);
            let field = match field {
                Some(f) => f.to_string(),
                None => s.fields.iter().find(|f| db.embedder_in(&s.table, &f.name).is_some_and(|e| e.modalities().contains(&m))).map(|f| f.name.clone()).ok_or_else(|| PyValueError::new_err(format!("no embedded field has a model of {what} open")))?,
            };
            let e = db.embedder_in(&s.table, &field).ok_or_else(|| PyValueError::new_err(format!("{field} has no embedding model open (see Database.open's models)")))?;
            strata::embed::file_is(m, &data).map_err(err)?;
            return py.detach(|| strata::embed::embed_file(e.as_ref(), &data)).map_err(err);
        }
        let text = text.ok_or_else(|| PyValueError::new_err("embed takes a text, an image or a sound"))?;
        let mut o = Map::new();
        o.insert("embed".into(), Json::String(text.into()));
        if let Some(f) = field {
            o.insert("field".into(), Json::String(f.into()));
        }
        if let Some(t) = table {
            o.insert("table".into(), Json::String(t.into()));
        }
        match py.detach(|| param(&db, &Json::Object(o))).map_err(err)? {
            Value::Vector(v) => Ok(v.to_vec()),
            _ => unreachable!("an embedding is a vector"),
        }
    }

    /// The codes that can follow each of `prefixes` in semantic-ID field
    /// `field` of `table` (the first by default) among the rows SQL
    /// condition `filter` matches, with their rows ([`Db::sid_next`]).
    #[pyo3(signature = (field, prefixes, filter = None, params = None, table = None))]
    fn sid_next(&self, py: Python<'_>, field: &str, prefixes: Vec<String>, filter: Option<String>, params: Option<&Bound<'_, PyDict>>, table: Option<&str>) -> PyResult<Vec<Vec<(String, u64)>>> {
        let db = self.get()?;
        let p = params_json(params)?;
        let table = table_of(&db, table).map_err(err)?.table.clone();
        py.detach(|| self::params(&db, &p).and_then(|p| db.sid_next(&table, field, &prefixes, filter.as_deref(), &p))).map_err(err)
    }

    /// Trains a semantic-ID codebook on the vectors of `field` of `table`
    /// (the first table by default), saves it as `name` under `models`, and
    /// returns its SHA-256.
    #[pyo3(signature = (field, name, table = None, models = None, levels = 3, k = 256, iterations = 25, sample = None, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn train_codebook(&self, py: Python<'_>, field: &str, name: &str, table: Option<String>, models: Option<PathBuf>, levels: usize, k: usize, iterations: usize, sample: Option<usize>, seed: u64) -> PyResult<String> {
        let db = self.get()?;
        let table = table.unwrap_or_else(|| db.schema().table.clone());
        let models = models.unwrap_or_else(strata::embed::models_dir);
        let opts = train_options(levels, k, iterations, sample, seed);
        py.detach(|| db.train_codebook(&table, field, name, &models, &opts)).map_err(err)
    }

    /// Merges small segments, as a server does in the background.
    fn optimize(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.get()?;
        py.detach(|| db.storage().merge(&MergePolicy::default()).map(|_| ())).map_err(err)
    }

    /// Merges small segments and closes the database (also when the last
    /// reference to it goes).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.db.write().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(db) = db {
            py.detach(|| db.storage().merge(&MergePolicy::default()).map(|_| ())).map_err(err)?;
        }
        Ok(())
    }

    fn closed(&self) -> bool {
        self.db.read().unwrap_or_else(|e| e.into_inner()).is_none()
    }
}

fn train_options(levels: usize, k: usize, iters: usize, sample: Option<usize>, seed: u64) -> TrainOptions {
    TrainOptions { levels, k, iters, sample: sample.unwrap_or(TrainOptions::default().sample), seed }
}

/// Trains a semantic-ID codebook on `vectors` (a list of vectors, or
/// anything with `tolist()`), saves it as `name` under `models`, and
/// returns its SHA-256.
#[pyfunction]
#[pyo3(signature = (vectors, name, models = None, levels = 3, k = 256, iterations = 25, sample = None, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn train_codebook(py: Python<'_>, vectors: &Bound<'_, PyAny>, name: &str, models: Option<PathBuf>, levels: usize, k: usize, iterations: usize, sample: Option<usize>, seed: u64) -> PyResult<String> {
    let Json::Array(rows) = to_json(vectors)? else { return Err(PyValueError::new_err("vectors are a list of lists of numbers")) };
    let dim = match rows.first() {
        Some(Json::Array(v)) => v.len(),
        _ => return Err(PyValueError::new_err("vectors are a list of lists of numbers, at least one")),
    };
    let mut flat = Vec::with_capacity(rows.len() * dim);
    for (i, r) in rows.iter().enumerate() {
        match r {
            Json::Array(v) if v.len() == dim => {
                for x in v {
                    flat.push(x.as_f64().ok_or_else(|| PyValueError::new_err(format!("vector {}: {x} is not a number", i + 1)))? as f32);
                }
            }
            _ => return Err(PyValueError::new_err(format!("vector {} does not have {dim} numbers, as the first does", i + 1))),
        }
    }
    let models = models.unwrap_or_else(strata::embed::models_dir);
    let opts = train_options(levels, k, iterations, sample, seed);
    py.detach(|| Codebook::train(&flat, dim, &opts)?.save(name, &models)).map_err(err)
}

/// Copies the database at `path`, written by a version of Strata from
/// before field ids (which no longer opens), into this version's format in
/// its place, keeping the old one as `<path>.before-field-ids`. Returns
/// `None` when it needs no upgrade, else the old database's path, the
/// tables copied and their rows.
#[pyfunction]
fn upgrade(py: Python<'_>, path: PathBuf) -> PyResult<Option<(PathBuf, usize, u64)>> {
    Ok(py.detach(|| strata::upgrade::upgrade(&path)).map_err(err)?.map(|u| (u.old, u.tables, u.rows)))
}

#[pymodule]
fn _strata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_function(wrap_pyfunction!(train_codebook, m)?)?;
    m.add_function(wrap_pyfunction!(upgrade, m)?)?;
    m.add("StrataError", m.py().get_type::<StrataError>())?;
    Ok(())
}
