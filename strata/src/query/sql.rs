//! SQL front end: parses Strata's dialect with `sqlparser` and binds it to a
//! table schema, producing a [`Select`].
//!
//! Supported: `SELECT <cols | * | score() | COUNT(*)> FROM <table>
//! [WHERE <pred>] [GROUP BY <keyword>] [ORDER BY <score> [ASC|DESC]]
//! [LIMIT n [OFFSET m]]`, where predicates combine `MATCH(field, q)`,
//! comparisons, `BETWEEN`, `IN` with AND/OR/NOT, and scores are `bm25`,
//! `similarity` (on a VECTOR or a SPARSE field), `distance`, `decay`, `rrf`,
//! weighted sums and products of those, or a column.
//! Parameters (`:name`) are bound from [`Params`].
//!
//! Writes ([`bind_command`], [`bind_write`]): `INSERT INTO t (cols) VALUES
//! (...), ...` with `OR REPLACE`, `OR IGNORE` or `ON CONFLICT (key) DO
//! NOTHING | DO UPDATE SET ...`, `UPDATE t SET f = v, g = g + 1 [WHERE ...]
//! [LIMIT n]` and `DELETE FROM t [WHERE ...] [ORDER BY ...] [LIMIT n]`, of
//! any table of the database. Lists are `ARRAY[...]`; vectors are
//! parameters, `ARRAY[...]` or a JSON array string. `CREATE TABLE` and
//! `DROP TABLE` are bound in [`crate::query::ddl`]. A clause the binder does
//! not handle is an error, never ignored.
//!
//! Graph queries over EDGES fields (DESIGN.md §6.1): the predicates
//! `REACHABLE(field, seeds, ...)` and `PATH(field, from, to, ...)`, the score
//! `ppr(field, seeds, ...)` and the column `hops()`. Seeds are a key, a
//! parameter, a tuple of keys or a subquery over the same table.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use sqlparser::ast::{self as sql, BinaryOperator, Expr, FunctionArg, FunctionArgExpr, FunctionArguments};

use crate::error::{Error, Result};
use crate::query::compound::outer_field;
use crate::query::ir::{ArithOp, CmpOp, Func, NormKind};
use crate::query::{ddl, ir};
use crate::query::rank::Rank;
use crate::query::logical::{AggFn, Decay, DecayShape, Direction, Expand, GraphKind, GraphQuery, Guard, GuardSite, Having, OrderBy, Opts, Pred, Projection, Score, Seeds, Select, Tiebreak, Vectors};
use crate::schema::{Field, FieldType, Logical, Schema};
use crate::storage::DropOptions;
use crate::value::Value;

/// Values for `:name` placeholders.
#[derive(Clone, Debug, Default)]
pub struct Params(pub HashMap<String, Value>);

impl Params {
    pub fn new() -> Self {
        Params::default()
    }
    pub fn set(mut self, name: &str, v: Value) -> Self {
        self.0.insert(name.trim_start_matches([':', '$', '@']).to_string(), v);
        self
    }
    fn get(&self, placeholder: &str) -> Result<&Value> {
        let name = placeholder.trim_start_matches([':', '$', '@', '?']);
        self.0.get(name).ok_or_else(|| err(format!("unbound parameter {placeholder}")))
    }
}

fn err(m: impl Into<String>) -> Error {
    Error::Query(m.into())
}

/// A bound INSERT, UPDATE or DELETE.
#[derive(Clone, Debug, PartialEq)]
pub enum Write {
    /// Documents in schema order (unnamed columns NULL), and what to do
    /// with one whose primary key is live.
    Insert { docs: Vec<Vec<Value>>, conflict: Conflict },
    /// The documents `select` returns (their primary keys) get `sets`.
    Update { select: Select, sets: Vec<(usize, Assign)> },
    /// The documents `select` returns are deleted.
    Delete { select: Select },
}

/// What an INSERT does with a document whose primary key is already live
/// (or written earlier by the same statement).
#[derive(Clone, Debug, PartialEq)]
pub enum Conflict {
    /// The statement fails: `INSERT`.
    Fail,
    /// The new document replaces the live one: `INSERT OR REPLACE`,
    /// `REPLACE INTO`.
    Replace,
    /// The live document stays and the new one is skipped: `INSERT OR
    /// IGNORE`, `ON CONFLICT DO NOTHING`.
    Skip,
    /// The live document gets these fields set, as by an UPDATE: `ON
    /// CONFLICT (key) DO UPDATE SET ...`.
    Update(Vec<(usize, Assign)>),
}

/// The right-hand side of `SET field = ...`.
#[derive(Clone, Debug, PartialEq)]
pub enum Assign {
    Value(Value),
    /// `field = field + x` (or `- x`): an INT or a FLOAT, integers exact.
    Add(Value),
    /// `field = excluded.g`, in `ON CONFLICT DO UPDATE`: field `g` of the
    /// document the INSERT proposed.
    Excluded(usize),
    /// Any other expression of the row's fields (and, in `ON CONFLICT DO
    /// UPDATE`, of `excluded.g`), computed for each row by the plan IR's
    /// evaluator ([`crate::query::ir::eval`]) and stored as the field's
    /// type: a whole FLOAT fits an INT, and one value a multi-valued field.
    /// NULL in gives NULL out, as in SQL; only `field + number` counts a
    /// NULL as zero.
    Expr(crate::query::ir::Expr),
}

/// A bound statement ([`bind_command`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// A SELECT, bound when it runs ([`crate::db::Db::query`]).
    Query,
    /// An INSERT, UPDATE or DELETE of the table named `table`.
    Write { table: String, write: Write },
    /// `CREATE TABLE [IF NOT EXISTS] ...`.
    CreateTable { schema: Schema, if_not_exists: bool },
    /// `DROP TABLE [IF EXISTS] a, b [CASCADE | RESTRICT]`.
    DropTable { tables: Vec<String>, opts: DropOptions },
    /// `CREATE [OR REPLACE] RANK PROFILE` ([`crate::query::profile`]).
    CreateRankProfile { table: String, profile: crate::storage::RankProfile, replace: bool },
    /// `DROP RANK PROFILE [IF EXISTS] name [ON table]`.
    DropRankProfile { table: Option<String>, name: String, if_exists: bool },
    /// `EXPLAIN [FORMAT JSON] <statement>`: the statement's plan
    /// ([`crate::query::ir`]), as lines of text or as JSON, made when it
    /// runs ([`crate::db::Db::explain`]).
    Explain { json: bool },
}

/// Parses one statement.
pub(crate) fn parse_one(sql_text: &str) -> Result<sql::Statement> {
    let mut stmts = super::parse::parse(sql_text)?;
    if stmts.len() != 1 {
        return Err(err("expected exactly one statement"));
    }
    Ok(stmts.pop().unwrap())
}

/// Parses one SELECT statement.
pub(crate) fn parse_query(sql_text: &str) -> Result<Box<sql::Query>> {
    match parse_one(sql_text)? {
        sql::Statement::Query(q) => Ok(q),
        _ => Err(err("only SELECT is supported")),
    }
}

/// Parses and binds one INSERT, UPDATE or DELETE of `schema`'s table;
/// `None` for a SELECT.
pub fn bind_write(sql_text: &str, schema: &Schema, params: &Params) -> Result<Option<Write>> {
    match bind_command(sql_text, &[schema], params)? {
        Command::Query => Ok(None),
        Command::Write { write, .. } => Ok(Some(write)),
        _ => Err(err("only SELECT, INSERT, UPDATE and DELETE are supported")),
    }
}

/// Parses and binds one statement of a database whose tables are `tables`:
/// a SELECT (bound when it runs), an INSERT, UPDATE or DELETE of one of
/// them, `CREATE TABLE` or `DROP TABLE`, or `CREATE RANK PROFILE` or
/// `DROP RANK PROFILE` ([`crate::query::profile`]).
pub fn bind_command(sql_text: &str, tables: &[&Schema], params: &Params) -> Result<Command> {
    if let Some(p) = super::profile::parse_statement(sql_text)? {
        let table = |name: &str| tables.iter().find(|t| t.table.eq_ignore_ascii_case(name)).map(|t| t.table.clone()).ok_or_else(|| err(format!("no such table: {name}")));
        return Ok(match p {
            super::profile::Statement::Create { table: t, profile, replace } => Command::CreateRankProfile { table: table(&t)?, profile, replace },
            super::profile::Statement::Drop { table: t, name, if_exists } => Command::DropRankProfile { table: t.as_deref().map(table).transpose()?, name, if_exists },
        });
    }
    bind_statement(&parse_one(sql_text)?, tables, params)
}

/// [`bind_command`] of a parsed statement.
pub(crate) fn bind_statement(stmt: &sql::Statement, tables: &[&Schema], params: &Params) -> Result<Command> {
    let table = |name: &sql::ObjectName| -> Result<&Schema> {
        let n = table_name(name);
        tables.iter().copied().find(|t| t.table.eq_ignore_ascii_case(&n)).ok_or_else(|| err(format!("no such table: {n}")))
    };
    let write = |schema: &Schema, write: Write| Command::Write { table: schema.table.clone(), write };
    Ok(match stmt {
        sql::Statement::Query(_) => Command::Query,
        sql::Statement::Explain { describe_alias: _, analyze, verbose, query_plan: _, estimate, statement, format, options } => {
            if *analyze || *verbose || *estimate {
                return Err(err("EXPLAIN shows a statement's plan without running it: ANALYZE, VERBOSE and ESTIMATE are not supported (POST /v1/explain runs a query and reports what it did)"));
            }
            if matches!(**statement, sql::Statement::Explain { .. }) {
                return Err(err("EXPLAIN of EXPLAIN"));
            }
            let mut json = false;
            let mut format_of = |f: &str| -> Result<()> {
                match f.to_ascii_uppercase().as_str() {
                    "JSON" => json = true,
                    "TEXT" | "TREE" => json = false,
                    f => return Err(err(format!("EXPLAIN FORMAT {f} is not supported: TEXT or JSON"))),
                }
                Ok(())
            };
            if let Some(sql::AnalyzeFormatKind::Keyword(f) | sql::AnalyzeFormatKind::Assignment(f)) = format {
                format_of(&f.to_string())?;
            }
            for o in options.iter().flatten() {
                match (o.name.value.to_ascii_uppercase().as_str(), &o.arg) {
                    ("FORMAT", Some(Expr::Identifier(f))) => format_of(&f.value)?,
                    _ => return Err(err(format!("EXPLAIN option {o} is not supported: FORMAT TEXT or FORMAT JSON"))),
                }
            }
            Command::Explain { json }
        }
        sql::Statement::Insert(ins) => {
            let sql::TableObject::TableName(n) = &ins.table else { return Err(err("INSERT needs a table name")) };
            let schema = table(n)?;
            write(schema, insert(ins, schema, params)?)
        }
        sql::Statement::Update { table: t, assignments, from, selection, returning, or, limit } => {
            if !t.joins.is_empty() || from.is_some() {
                return Err(err("UPDATE ... FROM and joins are not supported: UPDATE one table, with a WHERE over its own columns"));
            }
            if returning.is_some() {
                return Err(err("UPDATE ... RETURNING is not supported"));
            }
            if let Some(o) = or {
                return Err(err(format!("UPDATE {o} is not supported")));
            }
            let schema = table(plain_table(&t.relation, "UPDATE")?)?;
            let alias = match &t.relation {
                sql::TableFactor::Table { alias: Some(a), .. } => Some(a.name.value.as_str()),
                _ => None,
            };
            let sets = Binder::new(schema, params).assignments(assignments, alias, false)?;
            write(schema, Write::Update { select: selecting(schema, params, &t.relation, selection, &[], limit)?, sets })
        }
        sql::Statement::Delete(d) => {
            let sql::Delete { tables: named, from, using, selection, returning, order_by, limit } = d;
            if !named.is_empty() || using.is_some() {
                return Err(err("DELETE ... USING and multi-table DELETE are not supported: DELETE FROM one table, with a WHERE over its own columns"));
            }
            if returning.is_some() {
                return Err(err("DELETE ... RETURNING is not supported"));
            }
            let (sql::FromTable::WithFromKeyword(from) | sql::FromTable::WithoutKeyword(from)) = from;
            let [t] = from.as_slice() else { return Err(err("DELETE from exactly one table")) };
            if !t.joins.is_empty() {
                return Err(err("DELETE from exactly one table"));
            }
            let schema = table(plain_table(&t.relation, "DELETE")?)?;
            write(schema, Write::Delete { select: selecting(schema, params, &t.relation, selection, order_by, limit)? })
        }
        sql::Statement::CreateTable(ct) => {
            let (schema, if_not_exists) = ddl::create_table(ct, tables)?;
            Command::CreateTable { schema, if_not_exists }
        }
        stmt @ sql::Statement::Drop { .. } => {
            let (tables, opts) = ddl::drop_table(stmt)?;
            Command::DropTable { tables, opts }
        }
        _ => return Err(err("only SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE, CREATE RANK PROFILE, DROP RANK PROFILE and EXPLAIN are supported")),
    })
}

/// The name of a plain table in FROM (an alias allowed), for `what`.
/// The table a name names: a quoted name is the name inside the quotes
/// (`"books"` is books, `"my table"` is my table).
pub(crate) fn table_name(n: &sql::ObjectName) -> String {
    match n.0.as_slice() {
        [sql::ObjectNamePart::Identifier(id)] => id.value.clone(),
        _ => n.to_string(),
    }
}

fn plain_table<'t>(t: &'t sql::TableFactor, what: &str) -> Result<&'t sql::ObjectName> {
    match t {
        sql::TableFactor::Table { name, alias: _, args: None, with_hints, version: None, with_ordinality: false, partitions, json_path: None, sample: None, index_hints }
            if with_hints.is_empty() && partitions.is_empty() && index_hints.is_empty() =>
        {
            Ok(name)
        }
        _ => Err(err(format!("{what} needs a table name, not {t}"))),
    }
}

/// Binds an INSERT into `schema`'s table.
fn insert(ins: &sql::Insert, schema: &Schema, params: &Params) -> Result<Write> {
    let sql::Insert {
        or,
        ignore,
        into: _,
        table: _,
        table_alias,
        columns,
        overwrite,
        source,
        assignments,
        partitioned,
        after_columns,
        has_table_keyword,
        on,
        returning,
        replace_into,
        priority,
        insert_alias,
        settings,
        format_clause,
    } = ins;
    if returning.is_some() {
        return Err(err("INSERT ... RETURNING is not supported"));
    }
    if !assignments.is_empty() {
        return Err(err("INSERT ... SET is not supported: INSERT INTO t (columns) VALUES (...)"));
    }
    if table_alias.is_some() || *overwrite || partitioned.is_some() || !after_columns.is_empty() || *has_table_keyword || priority.is_some() || insert_alias.is_some() || settings.is_some() || format_clause.is_some() {
        return Err(err(format!("this form of INSERT is not supported: {ins}")));
    }
    let b = Binder::new(schema, params);
    let mut conflict = match (or, *replace_into, *ignore) {
        (Some(sql::SqliteOnConflict::Replace), ..) | (None, true, false) => Conflict::Replace,
        (Some(sql::SqliteOnConflict::Ignore), ..) | (None, false, true) => Conflict::Skip,
        (None, false, false) => Conflict::Fail,
        (Some(o), ..) => return Err(err(format!("INSERT {o} is not supported: INSERT fails on a key that is taken, OR REPLACE replaces that row and OR IGNORE keeps it"))),
        (None, true, true) => return Err(err("REPLACE IGNORE is not supported")),
    };
    match on {
        None => {}
        Some(_) if conflict != Conflict::Fail => return Err(err("INSERT OR ... and ON CONFLICT cannot be combined")),
        Some(sql::OnInsert::DuplicateKeyUpdate(_)) => return Err(err("ON DUPLICATE KEY UPDATE is not supported: use ON CONFLICT (key) DO UPDATE SET ...")),
        Some(sql::OnInsert::OnConflict(oc)) => {
            let pk = schema.primary_key.ok_or_else(|| err(format!("ON CONFLICT needs a primary key, and {} has none", schema.table)))?;
            match &oc.conflict_target {
                None => {}
                Some(sql::ConflictTarget::Columns(cols)) if cols.len() == 1 && schema.field_index(&cols[0].value) == Some(pk) => {}
                Some(_) => return Err(err(format!("ON CONFLICT names the primary key: ON CONFLICT ({})", schema.fields[pk].name))),
            }
            conflict = match &oc.action {
                sql::OnConflictAction::DoNothing => Conflict::Skip,
                sql::OnConflictAction::DoUpdate(u) => {
                    if u.selection.is_some() {
                        return Err(err("ON CONFLICT ... DO UPDATE ... WHERE is not supported"));
                    }
                    Conflict::Update(b.assignments(&u.assignments, None, true)?)
                }
            };
        }
        Some(o) => return Err(err(format!("INSERT ...{o} is not supported"))),
    }
    let cols: Vec<usize> = if columns.is_empty() {
        (0..schema.fields.len()).collect()
    } else {
        columns.iter().map(|c| schema.field_index(&c.value).ok_or_else(|| err(format!("no such column: {}", c.value)))).collect::<Result<_>>()?
    };
    let values = match source.as_deref() {
        Some(q) if no_clauses(q).is_none() => match q.body.as_ref() {
            sql::SetExpr::Values(v) => v,
            _ => return Err(err("INSERT needs VALUES")),
        },
        Some(q) => return Err(err(format!("INSERT ... VALUES takes no {}", no_clauses(q).unwrap_or_default()))),
        None => return Err(err("INSERT needs VALUES")),
    };
    // Columns the statement leaves out take their defaults.
    let start = schema.defaults(b.now)?;
    let mut docs = Vec::with_capacity(values.rows.len());
    for row in &values.rows {
        if row.len() != cols.len() {
            return Err(err(format!("{} values for {} columns", row.len(), cols.len())));
        }
        let mut doc = start.clone();
        for (&f, e) in cols.iter().zip(row) {
            doc[f] = b.write_value(f, e)?;
        }
        docs.push(doc);
    }
    Ok(Write::Insert { docs, conflict })
}

/// The first clause of `q` other than its body: `WITH`, `ORDER BY`,
/// `LIMIT` and the clauses no statement here takes.
fn no_clauses(q: &sql::Query) -> Option<String> {
    let sql::Query { with, body: _, order_by, limit_clause, fetch, locks, for_clause, settings, format_clause, pipe_operators } = q;
    if with.is_some() {
        Some("WITH".into())
    } else if order_by.is_some() {
        Some("ORDER BY".into())
    } else if limit_clause.is_some() {
        Some("LIMIT".into())
    } else {
        unsupported_query_clause(fetch, locks, for_clause, settings, format_clause, pipe_operators)
    }
}

/// The clauses of a query that no statement here takes, by name.
fn unsupported_query_clause(
    fetch: &Option<sql::Fetch>,
    locks: &[sql::LockClause],
    for_clause: &Option<sql::ForClause>,
    settings: &Option<Vec<sql::Setting>>,
    format_clause: &Option<sql::FormatClause>,
    pipe_operators: &[sql::PipeOperator],
) -> Option<String> {
    if fetch.is_some() {
        Some("FETCH (use LIMIT)".into())
    } else if !locks.is_empty() {
        Some("FOR UPDATE or FOR SHARE (reads never lock: they see a snapshot)".into())
    } else if let Some(f) = for_clause {
        Some(f.to_string())
    } else if settings.is_some() {
        Some("SETTINGS".into())
    } else if let Some(f) = format_clause {
        Some(f.to_string())
    } else if !pipe_operators.is_empty() {
        Some("pipe operators".into())
    } else {
        None
    }
}

/// The rows a WHERE clause selects, as a query of every field (an UPDATE
/// may rewrite them; a DELETE needs their keys), cut by `ORDER BY` and
/// `LIMIT` as a SELECT's rows are. The statement's own table and clauses
/// are bound, never printed and parsed again: printed, `- -1` is `--1`, a
/// comment, and a column named `order` is a keyword.
fn selecting(schema: &Schema, params: &Params, table: &sql::TableFactor, selection: &Option<Expr>, order_by: &[sql::OrderByExpr], limit: &Option<Expr>) -> Result<Select> {
    let mut q = parse_query("SELECT * FROM t")?;
    let sql::SetExpr::Select(s) = q.body.as_mut() else { return Err(err("a SELECT of every field")) };
    s.from[0].relation = table.clone();
    s.selection = selection.clone();
    if !order_by.is_empty() {
        q.order_by = Some(sql::OrderBy { kind: sql::OrderByKind::Expressions(order_by.to_vec()), interpolate: None });
    }
    q.limit_clause = limit.clone().map(|l| sql::LimitClause::LimitOffset { limit: Some(l), offset: None, limit_by: Vec::new() });
    let mut select = bind_query(&q, schema, params)?;
    select.projection = every_field(schema);
    select.types = projection_types(schema, &select.projection);
    Ok(select)
}

/// Every field of the rows whose primary key is one of `keys`, as a query
/// (`ON CONFLICT DO UPDATE` reads the live rows of its keys).
pub(crate) fn keyed_rows(schema: &Schema, keys: Vec<Value>) -> Result<Select> {
    let pk = schema.primary_key.ok_or_else(|| err(format!("{} has no primary key", schema.table)))?;
    let projection = every_field(schema);
    let types = projection_types(schema, &projection);
    let filter = if keys.is_empty() { Pred::False } else { Pred::In { field: pk, values: keys } };
    Ok(Select { projection, filter, order: None, group_by: Vec::new(), having: None, tiebreak: None, phases: Vec::new(), limit: None, offset: 0, graphs: Vec::new(), types, guard: None, per: None })
}

fn every_field(schema: &Schema) -> Vec<(String, Projection)> {
    schema.fields.iter().enumerate().map(|(i, f)| (f.name.clone(), Projection::Field(i))).collect()
}

/// A column's name quoted, which SQL reads as that column whatever it is:
/// a keyword such as `order`, or any characters.
pub fn quoted(n: &str) -> String {
    format!("\"{}\"", n.replace('"', "\"\""))
}

/// Why `similarity` refuses a multi-vector field.
pub(crate) fn multi_similarity(field: &str) -> String {
    format!("{field} is a multi-vector field: rank it with maxsim({field}, vectors), the sum of each query vector's best match")
}

/// A value coerced to `field`'s type, in the form its column holds
/// ([`crate::schema::Field::store`]).
fn coerce(field: &Field, v: Value) -> Result<Value> {
    if let Some(l) = field.logical {
        // In SQL a time is text or a timestamp parameter; a bare
        // number would be microseconds nobody meant.
        if l == Logical::Timestamp && matches!(v, Value::Int(_) | Value::Float(_)) {
            return Err(err(format!("{} is a TIMESTAMP: give it text such as '2026-10-01T12:30:00Z', not the number {v:?}", field.name)));
        }
        return field.store(v).map_err(|e| match e {
            Error::Schema(m) => err(m),
            e => e,
        });
    }
    match (&field.ty, v) {
        (FieldType::Keyword, Value::Str(s)) => Ok(Value::Str(s)),
        (FieldType::Keyword, Value::Int(x)) => Ok(Value::Str(x.to_string())),
        (FieldType::Int, Value::Int(x)) => Ok(Value::Int(x)),
        (FieldType::Int, Value::Bool(b)) => Ok(Value::Int(b as i64)),
        (FieldType::Float, v @ (Value::Int(_) | Value::Float(_) | Value::Bool(_))) => Ok(Value::Float(v.as_f64().unwrap())),
        (t, v) => Err(err(format!("value {v:?} does not fit a {t:?} field"))),
    }
}

/// A value compared with `field` in its type, as a condition reads it
/// ([`coerce`]): a subquery's values, which [`crate::query::ops`] runs.
pub(crate) fn coerced(field: &Field, v: Value) -> Result<Value> {
    coerce(field, v)
}

/// What a write gives MEDIA field `field` as text: a URI, or a JSON
/// object's text ([`crate::media::input_text`]), resolved before the writer.
fn media_input(field: &Field, text: String) -> Result<Value> {
    crate::media::input_text(&serde_json::Value::String(text)).map(Value::Str).map_err(|e| err(format!("{}: {e}", field.name)))
}

/// The value an expression gives `field` in `SET` ([`Assign::Expr`]), in
/// the form its column holds: [`coerce`]d, a whole FLOAT as an INT, and
/// one value as a multi-valued field's list of it.
pub(crate) fn assigned(field: &Field, v: Value) -> Result<Value> {
    let one = |v: Value| match (&field.ty, v) {
        (FieldType::Int, Value::Float(x)) if field.logical.is_none() => {
            if x.fract() == 0.0 && x.abs() < 9.2e18 {
                Ok(Value::Int(x as i64))
            } else {
                Err(err(format!("{} is INT, and {x} is not an integer it holds", field.name)))
            }
        }
        (FieldType::Float, Value::Float(x)) if !x.is_finite() => Err(err(format!("{} is FLOAT, and {x} is not a finite number", field.name))),
        (_, v) => coerce(field, v),
    };
    Ok(match (&field.ty, v) {
        (_, Value::Null) => Value::Null,
        (FieldType::Vector { .. }, v) if field.multi => crate::vector::multi::normalize(field, v)?,
        (FieldType::Vector { .. }, v @ Value::Vector(_)) => v,
        (_, Value::Str(t)) if field.media.is_some() => media_input(field, t)?,
        (_, v) if field.media.is_some() => return Err(err(format!("{} is a MEDIA field: SET gives it a URI or a JSON object's text, not {v:?}", field.name))),
        (FieldType::Text { .. } | FieldType::Edges, v @ Value::Str(_)) => v,
        (FieldType::Sparse, v @ Value::Str(_)) => {
            crate::text::sparse::parse(&v)?;
            v
        }
        (FieldType::Vector { .. } | FieldType::Text { .. } | FieldType::Edges | FieldType::Sparse, v) => return Err(err(format!("{} is a {} field: SET cannot give it {v:?}", field.name, field.type_name()))),
        (_, Value::List(xs)) if field.multi => Value::List(xs.into_iter().map(one).collect::<Result<_>>()?),
        (_, Value::List(xs)) => return Err(err(format!("{} holds one value, not the list {xs:?}", field.name))),
        (_, v) if field.multi => Value::List(vec![one(v)?]),
        (_, v) => one(v)?,
    })
}

/// The name of a column of the answer without an alias: a column's own
/// name, quoted or not (`"order"` is `order`), else the expression's text.
pub(crate) fn name_of(e: &Expr) -> String {
    match e {
        Expr::Identifier(id) => id.value.clone(),
        Expr::CompoundIdentifier(ids) => ids.iter().map(|i| i.value.as_str()).collect::<Vec<_>>().join("."),
        e => e.to_string(),
    }
}

/// A name as SQL spells it: quoted unless it is a plain identifier.
pub fn ident(n: &str) -> String {
    if n.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        n.to_string()
    } else {
        quoted(n)
    }
}

/// Parses and binds one SELECT statement.
pub fn bind(sql_text: &str, schema: &Schema, params: &Params) -> Result<Select> {
    bind_query(&*parse_query(sql_text)?, schema, params)
}

/// [`bind`] of a parsed query.
pub(crate) fn bind_query(q: &sql::Query, schema: &Schema, params: &Params) -> Result<Select> {
    Binder::new(schema, params).query(q)
}

/// [`bind_query`] in a database of `tables`, whose edge tables its walks
/// may follow.
pub(crate) fn bind_query_in(q: &sql::Query, schema: &Schema, tables: &[&Schema], params: &Params) -> Result<Select> {
    Binder { tables, ..Binder::new(schema, params) }.query(q)
}

/// Where a subquery sits in a compound query's SELECT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sub {
    /// `x IN (SELECT ...)`: its one column's values.
    In,
    /// `EXISTS (SELECT ...)`.
    Exists,
    /// `x < (SELECT ...)`, or a bound of `BETWEEN`: its one value.
    Scalar,
    /// A traversal's seeds: its rows' keys, or its one column's values when
    /// it reads another table.
    Seeds,
    /// `maxsim`'s query vectors: its one column's values, in order.
    Vectors,
    /// A number in a ranking expression: its one value, for each row
    /// ranked when it names the row (a correlated subquery).
    Score,
    /// A history for `encode(..)`: its rows, one to three columns
    /// ([`ir::Expr::Encode`]).
    Rows,
}

/// Plans a subquery for the binder ([`bind_query_with`]): the expression
/// it stands for, or for seeds `None` when the executor runs it itself.
pub(crate) type Subs<'s> = dyn Fn(&sql::Query, Sub) -> Result<Option<ir::Expr>> + 's;

/// [`bind_query`] of a compound query's SELECT, whose subqueries in
/// conditions and seeds `subs` plans: they bind as [`Pred::Sub`] and
/// [`Seeds::Plan`], which its plan runs first.
pub(crate) fn bind_query_with(q: &sql::Query, schema: &Schema, tables: &[&Schema], params: &Params, subs: &Subs) -> Result<Select> {
    Binder { subs: Some(subs), tables, ..Binder::new(schema, params) }.query(q)
}

/// [`bind_query_with`] for the last step of rerank phases: `earlier` are the
/// ORDER BY expressions of the phases before it, first to last, which
/// `features()` lists with its own.
pub(crate) fn bind_phase_with(q: &sql::Query, schema: &Schema, tables: &[&Schema], params: &Params, subs: &Subs, earlier: &[Expr]) -> Result<Select> {
    let s = Binder { subs: Some(subs), earlier, tables, ..Binder::new(schema, params) }.query(q)?;
    if s.guard.is_some() {
        return Err(err("min_rows and max_rows guard a search's first ranking, not a rerank phase"));
    }
    Ok(s)
}

/// `sql_text`, one SELECT that ranks rows (a table's, or rerank phases'),
/// with `score()` and `features()` added to its columns: the values each
/// row was ranked by, which `/v1/explain` shows for the top rows.
pub fn with_features(sql_text: &str) -> Result<String> {
    let mut q = ranked_query(sql_text).ok_or_else(|| err("hits explain the rows of one SELECT ranked by an ORDER BY"))?;
    add_features(&mut q)?;
    Ok(q.to_string())
}

/// `sql_text` when it is one SELECT ranked by an ORDER BY.
pub(crate) fn ranked_query(sql_text: &str) -> Option<sql::Query> {
    let mut stmts = super::parse::parse(sql_text).ok()?;
    let [sql::Statement::Query(q)] = stmts.as_mut_slice() else { return None };
    (matches!(q.body.as_ref(), sql::SetExpr::Select(_)) && q.order_by.is_some()).then(|| (**q).clone())
}

/// Adds `score()` and `features()` to the columns of a query from
/// [`ranked_query`].
pub(crate) fn add_features(q: &mut sql::Query) -> Result<()> {
    let sql::SetExpr::Select(s) = q.body.as_mut() else { return Err(err("not a SELECT")) };
    let mut extra = super::parse::parse("SELECT score(), features()")?;
    let Some(sql::Statement::Query(x)) = extra.pop() else { unreachable!("a SELECT") };
    let sql::SetExpr::Select(x) = *x.body else { unreachable!("a SELECT") };
    s.projection.extend(x.projection);
    Ok(())
}

/// The query of `(SELECT ...)`, in parentheses or not.
fn subquery_of(e: &Expr) -> Option<&sql::Query> {
    match e {
        Expr::Subquery(q) => Some(q),
        Expr::Nested(x) => subquery_of(x),
        _ => None,
    }
}

struct Binder<'a> {
    schema: &'a Schema,
    params: &'a Params,
    /// A compound query's subqueries ([`bind_query_with`]).
    subs: Option<&'a Subs<'a>>,
    /// Graph computations bound so far (`Select::graphs`).
    graphs: RefCell<Vec<GraphQuery>>,
    /// The statement's time, in microseconds since the epoch: `now()` in
    /// `decay` and the `"now"` default of a TIMESTAMP.
    now: i64,
    /// The ORDER BY expressions of the rerank phases before this query's
    /// ([`bind_phase_with`]).
    earlier: &'a [Expr],
    /// The database's tables, whose edge tables a walk may follow
    /// (`ppr(cooc, ...)`); empty when the query is bound alone.
    tables: &'a [&'a Schema],
    /// Guards bound so far (`max_rows`, `min_rows`), each with the `WHERE`
    /// condition it guards (none for a `bm25`'s) and whether it was bound
    /// in `ORDER BY`.
    guards: RefCell<Vec<(Guard, Option<Pred>, bool)>>,
    /// Whether `ORDER BY` is being bound.
    ordering: Cell<bool>,
}

impl<'a> Binder<'a> {
    fn new(schema: &'a Schema, params: &'a Params) -> Self {
        Binder {
            schema,
            params,
            subs: None,
            graphs: RefCell::new(Vec::new()),
            now: crate::timestamp::now(),
            earlier: &[],
            tables: &[],
            guards: RefCell::new(Vec::new()),
            ordering: Cell::new(false),
        }
    }

    /// Records a guard of `site` if `opts` give `min_rows` or `max_rows`,
    /// taking them out of `opts`; `term` is the `WHERE` condition it guards.
    fn guard(&self, opts: &mut Opts, site: impl FnOnce() -> GuardSite, need: Option<crate::text::query::Need>, term: Option<&Pred>, what: &str) -> Result<()> {
        let mut rows = |k: &str| {
            opts.remove(k)
                .map(|v| v.as_f64().filter(|x| x.fract() == 0.0 && (1.0..1e15).contains(x)).map(|x| x as u64).ok_or_else(|| err(format!("{k} => a count of rows, 1 or more (500), not {v:?}"))))
                .transpose()
        };
        let (min_rows, max_rows) = (rows("min_rows")?, rows("max_rows")?);
        if min_rows.is_none() && max_rows.is_none() {
            return Ok(());
        }
        if let (Some(lo), Some(hi)) = (min_rows, max_rows) {
            if lo > hi {
                return Err(err(format!("min_rows => {lo} is above max_rows => {hi}: no level has at least {lo} rows and at most {hi}")));
            }
        }
        let need = need.ok_or_else(|| err(format!("min_rows and max_rows on {what} relax its min => n, which it needs")))?;
        self.guards.borrow_mut().push((Guard { site: site(), need, min_rows, max_rows }, term.cloned(), self.ordering.get()));
        Ok(())
    }

    /// `filter` without the condition a guard of `WHERE` guards, and the
    /// query's guard: one at most, on a condition joined to the rest by
    /// `AND`, or on a `bm25` of `ORDER BY`.
    fn take_guard(&self, filter: Pred) -> Result<(Pred, Option<Guard>)> {
        let mut guards = self.guards.take();
        if guards.len() > 1 {
            return Err(err("a query has one guard (min_rows, max_rows) at most"));
        }
        let Some((guard, term, in_order)) = guards.pop() else { return Ok((filter, None)) };
        let multi = |f: usize| self.schema.fields[f].multi;
        let guard = match guard.site {
            GuardSite::AtLeast(items) => Guard { site: GuardSite::AtLeast(items.into_iter().map(|p| p.three_valued(&multi).simplify()).collect()), ..guard },
            _ => guard,
        };
        let Some(term) = term else {
            return match in_order {
                true => Ok((filter, Some(guard))),
                false => Err(err("min_rows and max_rows on bm25 guard the ranking of ORDER BY")),
            };
        };
        let mut terms = match filter {
            Pred::And(ps) => ps,
            p => vec![p],
        };
        match terms.iter().position(|p| *p == term) {
            Some(i) => {
                terms.remove(i);
                Ok((Pred::And(terms), Some(guard)))
            }
            None => Err(err("min_rows and max_rows guard a condition of WHERE joined to the others by AND, not one inside OR, NOT or another condition")),
        }
    }

    /// The expression a compound query's subquery stands for.
    fn sub(&self, q: &sql::Query, kind: Sub) -> Result<ir::Expr> {
        let subs = self.subs.ok_or_else(|| err("a subquery runs as a compound query (Db::query), not in one SELECT"))?;
        subs(q, kind)?.ok_or_else(|| err("the subquery has no plan"))
    }

    fn query(&self, q: &sql::Query) -> Result<Select> {
        // `SELECT DISTINCT a, b` is `GROUP BY a, b`, as a compound query
        // plans it, so that a shard answers it with groups to merge.
        if let sql::SetExpr::Select(s) = q.body.as_ref() {
            if matches!(s.distinct, Some(sql::Distinct::Distinct)) {
                let mut s = s.clone();
                super::compound::distinct_to_group(&mut s)?;
                return self.query(&sql::Query { body: Box::new(sql::SetExpr::Select(s)), ..q.clone() });
            }
        }
        let sql::Query { with, body, order_by: _, limit_clause, fetch, locks, for_clause, settings, format_clause, pipe_operators } = q;
        if with.is_some() {
            return Err(err("WITH runs as a compound query (Db::query), not as one SELECT"));
        }
        if let Some(c) = unsupported_query_clause(fetch, locks, for_clause, settings, format_clause, pipe_operators) {
            return Err(err(format!("{c} is not supported")));
        }
        let sql::SetExpr::Select(s) = body.as_ref() else { return Err(err("only simple SELECT is supported")) };
        check_select(s)?;
        if s.from.len() != 1 || !s.from[0].joins.is_empty() {
            return Err(err("exactly one table, no joins"));
        }
        let t = table_name(plain_table(&s.from[0].relation, "SELECT")?);
        if !t.eq_ignore_ascii_case(&self.schema.table) {
            return Err(err(format!("no such table: {t}")));
        }
        if let Some(sql::LimitClause::LimitOffset { limit_by, .. }) = limit_clause {
            if !limit_by.is_empty() {
                return Err(err("LIMIT ... BY is not supported"));
            }
        }
        if let Some(ob) = &q.order_by {
            if ob.interpolate.is_some() || matches!(&ob.kind, sql::OrderByKind::Expressions(es) if es.iter().any(|e| e.with_fill.is_some())) {
                return Err(err("ORDER BY ... WITH FILL and INTERPOLATE are not supported"));
            }
        }
        // Simplified once ORDER BY's guard is known too.
        let filter = match &s.selection {
            Some(e) => self.pred(e)?,
            None => Pred::True,
        };
        let group_by: Vec<usize> = match &s.group_by {
            sql::GroupByExpr::Expressions(es, _) => es.iter().map(|e| self.group_field(e, &s.projection)).collect::<Result<_>>()?,
            _ => return Err(err("GROUP BY ALL is not supported")),
        };
        let mut projection = Vec::new();
        let mut features = false;
        for item in &s.projection {
            let (e, alias) = match item {
                sql::SelectItem::UnnamedExpr(Expr::Function(f)) if fname(f) == "features" => {
                    if !self.args_raw(f).is_ok_and(|(a, n)| a.is_empty() && n.is_empty()) {
                        return Err(err("features() takes no arguments"));
                    }
                    let own = match &q.order_by {
                        Some(sql::OrderBy { kind: sql::OrderByKind::Expressions(es), .. }) if !es.is_empty() => Some(self.ordered_expr(&es[0].expr, &s.projection)?),
                        _ => None,
                    };
                    let orders: Vec<&Expr> = self.earlier.iter().chain(own).collect();
                    if orders.is_empty() {
                        return Err(err("features() lists the inputs of the query's ranking, and it has no ORDER BY"));
                    }
                    for e in self.feature_items(&orders)? {
                        projection.push((name_of(&e), self.projection(&e)?));
                    }
                    features = true;
                    continue;
                }
                sql::SelectItem::UnnamedExpr(e) => (e, None),
                sql::SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.clone())),
                sql::SelectItem::Wildcard(_) => {
                    for (i, f) in self.schema.fields.iter().enumerate() {
                        if f.stored && !matches!(f.ty, FieldType::Vector { .. }) {
                            projection.push((f.name.clone(), Projection::Field(i)));
                        }
                    }
                    continue;
                }
                _ => return Err(err("unsupported select item")),
            };
            let p = self.projection(e)?;
            projection.push((alias.unwrap_or_else(|| name_of(e)), p));
        }
        let aggregated = !group_by.is_empty() || projection.iter().any(|(_, p)| matches!(p, Projection::Count | Projection::Agg { .. }));
        if aggregated {
            if features {
                return Err(err("features() lists the inputs of a ranking of rows, not of groups"));
            }
            for (name, p) in &projection {
                match p {
                    Projection::Feature(_) => return Err(err(format!("{name}: a ranking expression scores rows, not groups"))),
                    Projection::Field(f) if !group_by.contains(f) => {
                        return Err(err(format!("{name} must be in GROUP BY or inside an aggregate (COUNT, SUM, AVG, MIN, MAX)")))
                    }
                    Projection::Score | Projection::Hops(_) | Projection::SeqNo => return Err(err("score(), hops() and seq_no() cannot be aggregated")),
                    _ => {}
                }
            }
        }
        let having = match &s.having {
            None => None,
            Some(_) if !aggregated => return Err(err("HAVING needs GROUP BY or aggregates")),
            Some(h) => Some(self.having(h, &projection, &group_by)?),
        };
        let (order, tiebreak) = match &q.order_by {
            None => (None, None),
            Some(ob) => match &ob.kind {
                sql::OrderByKind::Expressions(es) if (1..=2).contains(&es.len()) => {
                    let desc = es[0].options.asc == Some(false);
                    let mut order = if aggregated {
                        self.group_order(&es[0].expr, desc, &projection, &group_by)?
                    } else {
                        self.ordering.set(true);
                        let order = self.order(&es[0].expr, desc, &projection);
                        self.ordering.set(false);
                        order?
                    };
                    // Rows without the column's value come last, either way
                    // (as in Elasticsearch); `NULLS FIRST` puts them first.
                    if es[0].options.nulls_first == Some(true) {
                        match &mut order {
                            OrderBy::Score { score: Score::Column { .. }, nulls_first, .. } => *nulls_first = true,
                            _ => return Err(err("NULLS FIRST applies to ORDER BY a column")),
                        }
                    }
                    let tiebreak = match es.get(1) {
                        None => None,
                        Some(e) => Some(self.tiebreak(&e.expr, e.options.asc == Some(false), &projection, &group_by)?),
                    };
                    (Some(order), tiebreak)
                }
                _ => return Err(err("ORDER BY supports a key and an optional tie-break")),
            },
        };
        let (limit, offset) = match &q.limit_clause {
            None => (None, 0),
            Some(sql::LimitClause::LimitOffset { limit, offset, .. }) => (
                limit.as_ref().map(|e| self.usize_lit(e)).transpose()?,
                offset.as_ref().map(|o| self.usize_lit(&o.value)).transpose()?.unwrap_or(0),
            ),
            Some(sql::LimitClause::OffsetCommaLimit { offset, limit }) => {
                (Some(self.usize_lit(limit)?), self.usize_lit(offset)?)
            }
        };
        let (filter, guard) = self.take_guard(filter)?;
        let filter = filter.three_valued(&|f| self.schema.fields[f].multi).simplify();
        let graphs = self.graphs.take();
        let types = projection_types(self.schema, &projection);
        Ok(Select { projection, filter, order, group_by, having, tiebreak, phases: Vec::new(), limit, offset, graphs, types, guard, per: None })
    }

    /// The graph predicate `hops()` refers to: the first REACHABLE or PATH.
    fn hops_graph(&self) -> Result<usize> {
        self.graphs
            .borrow()
            .iter()
            .position(|g| !matches!(g.kind, GraphKind::Ppr { .. }))
            .ok_or_else(|| err("hops() needs a REACHABLE or PATH predicate in WHERE"))
    }

    /// The edge table a walk's first argument names, rather than a field
    /// of this table ([`crate::schema::Schema::with_edges`]).
    fn edge_table(&self, e: &Expr) -> Result<Option<String>> {
        let Expr::Identifier(id) = e else { return Ok(None) };
        if self.schema.field_index(&id.value).is_some() {
            return Ok(None);
        }
        let Some(t) = self.tables.iter().find(|t| t.table.eq_ignore_ascii_case(&id.value)) else { return Ok(None) };
        let Some((src, ..)) = t.edge_fields() else {
            return Err(err(format!("table {} is not an edge table: a walk follows an EDGES field or a table with OPTIONS(edges = 'src, dst, w')", t.table)));
        };
        let pk = self.schema.primary_key.ok_or_else(|| err(format!("a walk over {} needs {} to have a primary key", t.table, self.schema.table)))?;
        if let Some(r) = t.fields[src].references.as_ref().filter(|r| !r.eq_ignore_ascii_case(&self.schema.table)) {
            return Err(err(format!("{} links rows of {r}, not of {}", t.table, self.schema.table)));
        }
        if t.fields[src].ty != self.schema.fields[pk].ty {
            return Err(err(format!("{} links {} keys, and {}'s are {}", t.table, crate::schema::type_name(&t.fields[src].ty), self.schema.table, crate::schema::type_name(&self.schema.fields[pk].ty))));
        }
        Ok(Some(t.table.clone()))
    }

    /// Binds `REACHABLE(field, seeds, ...)`, `PATH(field, from, to, ...)` or
    /// `ppr(field, seeds, ...)` and returns its index in `Select::graphs`.
    /// The field may be an edge table instead ([`Binder::edge_table`]).
    fn graph(&self, f: &sql::Function, name: &str) -> Result<usize> {
        let (args, named) = self.args_raw(f)?;
        let want = if name == "path" { 3 } else { 2 };
        if args.len() != want {
            return Err(err(match name {
                "path" => "PATH(field, from, to, ...)",
                "ppr" => "ppr(field, seeds, ...)",
                _ => "REACHABLE(field, seeds, ...)",
            }));
        }
        let edges = self.edge_table(args[0])?;
        let field = match &edges {
            Some(_) => self.schema.primary_key.expect("checked by edge_table"),
            None => self.typed_field(args[0], |t| matches!(t, FieldType::Edges | FieldType::Vector { .. }), "EDGES or VECTOR")?,
        };
        let vector = edges.is_none() && matches!(self.schema.fields[field].ty, FieldType::Vector { .. });
        if edges.is_none() && !vector && !self.schema.fields[field].indexed {
            return Err(err(format!("{} is not indexed", self.schema.fields[field].name)));
        }
        if vector && name == "ppr" {
            return Err(err("ppr() needs an EDGES field"));
        }
        if vector && self.schema.fields[field].multi {
            return Err(err(format!("{name}() follows one vector per row: {} is a multi-vector field", self.schema.fields[field].name)));
        }
        let seeds = self.seeds(args[1])?;
        let mut opts = Opts::new();
        let mut through = None;
        for (k, e) in named {
            if k == "through" {
                let p = self.pred(e)?.three_valued(&|f| self.schema.fields[f].multi).simplify();
                if contains_graph(&p) {
                    return Err(err("a through predicate cannot hold a graph predicate"));
                }
                through = Some(p);
            } else {
                opts.insert(k, self.literal(e)?);
            }
        }
        let allowed: &[&str] = match name {
            "path" => &["max_hops", "dir", "label", "labels", "max_nodes", "k", "min_sim"],
            "ppr" => &["alpha", "eps", "max_pushes", "weighted", "dir", "label", "labels"],
            _ => &["hops", "min_hops", "dir", "label", "labels", "max_nodes", "k", "min_sim"],
        };
        if let Some(k) = opts.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(err(format!("unknown option {k} for {name}")));
        }
        if through.is_some() && name == "ppr" {
            return Err(err("ppr() has no through option"));
        }
        let num = |k: &str, d: f64| -> Result<f64> {
            match opts.get(k) {
                None => Ok(d),
                Some(v) => v.as_f64().ok_or_else(|| err(format!("{k} must be a number"))),
            }
        };
        let count = |k: &str, d: f64| -> Result<u32> {
            let x = num(k, d)?;
            if x < 0.0 || x.fract() != 0.0 || x > 1e6 {
                return Err(err(format!("{k} must be a non-negative integer")));
            }
            Ok(x as u32)
        };
        // A limit: 0 would be read as nothing, so it is refused.
        let at_least_one = |k: &str, d: f64, most: f64| -> Result<f64> {
            let x = num(k, d)?;
            if x < 1.0 || x.fract() != 0.0 || x > most {
                return Err(err(format!("{k} must be a whole number from 1 to {most}")));
            }
            Ok(x)
        };
        let dir = match opts.get("dir").map(|v| v.as_str().unwrap_or("")) {
            None => if name == "ppr" { Direction::Both } else { Direction::Out },
            Some("out") => Direction::Out,
            Some("in") => Direction::In,
            Some("both") => Direction::Both,
            Some(d) => return Err(err(format!("dir must be 'out', 'in' or 'both', not {d:?}"))),
        };
        let labels: Vec<String> = ["label", "labels"]
            .iter()
            .filter_map(|k| opts.get(*k))
            .flat_map(|v| v.as_str().unwrap_or("").split(',').map(|l| l.trim().to_string()).collect::<Vec<_>>())
            .filter(|l| !l.is_empty())
            .collect();
        if let Some(t) = edges.as_ref().filter(|_| !labels.is_empty()) {
            return Err(err(format!("the edges of edge table {t} have no labels")));
        }
        let kind = match name {
            // `max_hops => 0`: only a seed that is also a target.
            "path" => GraphKind::Path { to: self.seeds(args[2])?, max_hops: count("max_hops", 6.0)? },
            "ppr" => {
                let alpha = num("alpha", 0.15)?;
                let eps = num("eps", 1e-4)?;
                if !(alpha > 0.0 && alpha < 1.0) || eps <= 0.0 {
                    return Err(err("ppr needs 0 < alpha < 1 and eps > 0"));
                }
                GraphKind::Ppr { alpha, eps, max_pushes: num("max_pushes", 1e6)? as u64, weighted: num("weighted", 0.0)? != 0.0 }
            }
            _ => {
                let max_hops = count("hops", 1.0)?;
                let min_hops = count("min_hops", 1.0_f64.min(max_hops as f64))?;
                if min_hops > max_hops {
                    return Err(err("min_hops is larger than hops"));
                }
                GraphKind::Reach { min_hops, max_hops }
            }
        };
        let g = GraphQuery {
            field,
            edges,
            seeds,
            kind,
            dir,
            labels,
            through,
            max_nodes: at_least_one("max_nodes", 100_000.0, 1e9)? as usize,
            k: at_least_one("k", 10.0, 1e6)? as usize,
            min_sim: num("min_sim", f64::NEG_INFINITY)?,
        };
        // The same computation named twice (in SELECT and ORDER BY) runs
        // once.
        let mut gs = self.graphs.borrow_mut();
        if let Some(i) = gs.iter().position(|x| *x == g) {
            return Ok(i);
        }
        gs.push(g);
        Ok(gs.len() - 1)
    }

    /// Seeds: a key, a parameter, a tuple or array of keys, or a subquery.
    fn seeds(&self, e: &Expr) -> Result<Seeds> {
        match e {
            Expr::Subquery(q) => {
                if let Some(e) = self.subs.map(|subs| subs(q, Sub::Seeds)).transpose()?.flatten() {
                    return Ok(Seeds::Plan(Box::new(e)));
                }
                Ok(Seeds::Query(Box::new(Binder { subs: self.subs, tables: self.tables, ..Binder::new(self.schema, self.params) }.query(q)?)))
            }
            Expr::Nested(inner) => self.seeds(inner),
            Expr::Tuple(items) => Ok(Seeds::Keys(items.iter().map(|i| self.key_value(i)).collect::<Result<_>>()?)),
            Expr::Array(a) => Ok(Seeds::Keys(a.elem.iter().map(|i| self.key_value(i)).collect::<Result<_>>()?)),
            // A list-valued parameter is a list of keys.
            _ => match self.literal(e).map(as_list) {
                Ok(Ok(xs)) => {
                    let pk = self.schema.primary_key.ok_or_else(|| err("graph queries need a primary key"))?;
                    Ok(Seeds::Keys(xs.into_iter().map(|x| self.coerce(pk, x)).collect::<Result<_>>()?))
                }
                _ => Ok(Seeds::Keys(vec![self.key_value(e)?])),
            },
        }
    }

    /// The keys of `ranked(keys)`: a tuple or array of keys, or a
    /// parameter holding a list of them (or one key).
    fn ranked_keys(&self, e: &Expr) -> Result<Vec<Value>> {
        let pk = self.schema.primary_key.ok_or_else(|| err("ranked(keys) ranks rows by their primary keys, and the table has none"))?;
        match e {
            Expr::Nested(inner) => self.ranked_keys(inner),
            Expr::Tuple(items) => items.iter().map(|i| self.field_value(pk, i)).collect(),
            Expr::Array(a) => a.elem.iter().map(|i| self.field_value(pk, i)).collect(),
            _ => match as_list(self.literal(e)?) {
                Ok(xs) => xs.into_iter().map(|x| self.coerce(pk, x)).collect(),
                Err(v) => Ok(vec![self.coerce(pk, v)?]),
            },
        }
    }

    /// A literal as a primary-key value.
    fn key_value(&self, e: &Expr) -> Result<Value> {
        let pk = self.schema.primary_key.ok_or_else(|| err("graph queries need a primary key"))?;
        self.field_value(pk, e)
    }

    /// The second ORDER BY key: the primary key, or the (first) group value.
    fn tiebreak(&self, e: &Expr, desc: bool, proj: &[(String, Projection)], group: &[usize]) -> Result<Tiebreak> {
        let field = match literal_usize(e) {
            Some(n) => match proj.get(n.wrapping_sub(1)) {
                Some((_, Projection::Field(f))) => *f,
                _ => return Err(err(format!("ORDER BY ..., {n}: not a column"))),
            },
            None => self.field_of(e)?,
        };
        if group.first() == Some(&field) {
            Ok(Tiebreak::GroupValue { desc })
        } else if Some(field) == self.schema.primary_key && group.is_empty() {
            Ok(Tiebreak::Key { desc })
        } else {
            Err(err("the second ORDER BY key must be the primary key or the first group field"))
        }
    }

    /// A GROUP BY item: a field, or the position of a selected field.
    fn group_field(&self, e: &Expr, items: &[sql::SelectItem]) -> Result<usize> {
        let f = match literal_usize(e) {
            Some(n) => match items.get(n.wrapping_sub(1)) {
                Some(sql::SelectItem::UnnamedExpr(e)) | Some(sql::SelectItem::ExprWithAlias { expr: e, .. }) => self.field_of(e)?,
                _ => return Err(err(format!("GROUP BY {n}: not a selected column"))),
            },
            None => match self.field_of(e) {
                Ok(f) => f,
                // An alias of a selected field.
                Err(e2) => {
                    let alias = match e {
                        Expr::Identifier(id) => &id.value,
                        _ => return Err(e2),
                    };
                    let found = items.iter().find_map(|i| match i {
                        sql::SelectItem::ExprWithAlias { expr, alias: a } if a.value.eq_ignore_ascii_case(alias) => Some(expr),
                        _ => None,
                    });
                    self.field_of(found.ok_or(e2)?)?
                }
            },
        };
        let fld = &self.schema.fields[f];
        let ok = match fld.ty {
            FieldType::Keyword | FieldType::Int => true,
            FieldType::Float => !fld.multi,
            _ => false,
        };
        if !ok {
            return Err(err(format!("GROUP BY {}: group by KEYWORD, INT or FLOAT fields", fld.name)));
        }
        if !fld.indexed {
            return Err(err(format!("GROUP BY {}: the field is not indexed", fld.name)));
        }
        Ok(f)
    }

    fn projection(&self, e: &Expr) -> Result<Projection> {
        match e {
            Expr::Identifier(_) | Expr::CompoundIdentifier(_) => Ok(Projection::Field(self.field_of(e)?)),
            Expr::Function(f) => match fname(f).as_str() {
                "score" => Ok(Projection::Score),
                "hops" => Ok(Projection::Hops(self.hops_graph()?)),
                "seq_no" => Ok(Projection::SeqNo),
                "count" | "sum" | "avg" | "min" | "max" => self.aggregate(f),
                "features" => Err(err("features() stands for several columns: it takes no alias and is not part of an expression")),
                _ => self.feature(e),
            },
            Expr::Nested(_) | Expr::UnaryOp { .. } | Expr::BinaryOp { .. } | Expr::Case { .. } => self.feature(e),
            // A number of each row, from the rows it names.
            Expr::Subquery(_) if self.subs.is_some() => self.feature(e),
            _ => Err(err(format!("unsupported select expression: {e}"))),
        }
    }

    /// A ranking expression in SELECT: one that scores each row by itself.
    fn feature(&self, e: &Expr) -> Result<Projection> {
        let s = self.score(e)?;
        if Rank::of(&s, 0).is_some_and(|r| !r.is_row_local()) {
            return Err(err(format!("{e}: a set function (minmax, rank), rrf or a weighted fusion ranks rows against each other, so it is not a column; select score()")));
        }
        Ok(Projection::Feature(s))
    }

    /// The expression an ORDER BY key stands for: the selected column it
    /// names by position, or itself.
    fn ordered_expr<'e>(&self, e: &'e Expr, items: &'e [sql::SelectItem]) -> Result<&'e Expr> {
        let Some(n) = literal_usize(e) else { return Ok(e) };
        match items.get(n.wrapping_sub(1)) {
            Some(sql::SelectItem::UnnamedExpr(Expr::Function(f))) if fname(f) == "features" => Err(err(format!("ORDER BY {n}: features() is not one column"))),
            Some(sql::SelectItem::UnnamedExpr(x) | sql::SelectItem::ExprWithAlias { expr: x, .. }) => Ok(x),
            _ => Err(err(format!("ORDER BY {n}: no such column"))),
        }
    }

    /// What `features()` stands for in a query ranked by `orders` (the
    /// ORDER BY of each rerank phase, first to last): each leaf of the
    /// ranking (a column, or a score such as `bm25(body, :q)`), then each
    /// phase's expression that is not a leaf and scores a row by itself,
    /// each once, named by its SQL text. A ranking by `distance(..)`
    /// reads its `similarity(..)`. A cross-encoder, which scores only in a
    /// global phase, is not one.
    fn feature_items(&self, orders: &[&Expr]) -> Result<Vec<Expr>> {
        // Functions of other scores, and the rows they read.
        let inner = |f: &sql::Function| matches!(fname(f).as_str(), "rrf" | "minmax" | "rank" | "gbdt") || Func::numeric(&fname(f)).is_some();
        let leaf = |e: &Expr| !matches!(e, Expr::Nested(_) | Expr::UnaryOp { .. } | Expr::BinaryOp { .. } | Expr::Case { .. } | Expr::Value(_)) && !matches!(e, Expr::Function(f) if inner(f));
        fn leaves(b: &Binder, e: &Expr, leaf: &dyn Fn(&Expr) -> bool, out: &mut Vec<Expr>) -> Result<()> {
            match e {
                Expr::Nested(x) | Expr::UnaryOp { expr: x, .. } => leaves(b, x, leaf, out)?,
                Expr::BinaryOp { left, right, .. } => {
                    leaves(b, left, leaf, out)?;
                    leaves(b, right, leaf, out)?;
                }
                Expr::Case { conditions, else_result, .. } => {
                    for x in conditions.iter().map(|c| &c.result).chain(else_result.as_deref()) {
                        leaves(b, x, leaf, out)?;
                    }
                }
                Expr::Value(_) => {}
                Expr::Function(f) if !leaf(e) => {
                    for a in b.args_raw(f)?.0 {
                        leaves(b, a, leaf, out)?;
                    }
                }
                // A cross-encoder scores in a global phase alone.
                Expr::Function(f) if fname(f) == "cross_encoder" => {}
                Expr::Function(f) if fname(f) == "distance" => {
                    let mut g = f.clone();
                    g.name = sql::ObjectName::from(vec![sql::Ident::new("similarity")]);
                    out.push(Expr::Function(g));
                }
                e => out.push(e.clone()),
            }
            Ok(())
        }
        let mut items = Vec::new();
        for e in orders {
            leaves(self, e, &leaf, &mut items)?;
        }
        for e in orders {
            let e = strip_nested(e);
            if leaf(e) {
                continue;
            }
            let score = self.score(e)?;
            if !score.has_cross_encoder() && Rank::of(&score, 0).is_none_or(|r| r.is_row_local()) {
                items.push(e.clone());
            }
        }
        let mut seen = std::collections::HashSet::new();
        items.retain(|e| seen.insert(name_of(e)));
        Ok(items)
    }

    /// `COUNT(*)`, `COUNT([DISTINCT] f)`, `SUM(f)`, `AVG(f)`, `MIN(f)`, `MAX(f)`.
    fn aggregate(&self, f: &sql::Function) -> Result<Projection> {
        let name = fname(f);
        let distinct = matches!(&f.args, FunctionArguments::List(l) if l.duplicate_treatment == Some(sql::DuplicateTreatment::Distinct));
        let (args, _) = self.args_raw(f)?;
        let star = args.is_empty() || (name == "count" && args.len() == 1 && literal_usize(args[0]).is_some());
        if star {
            return if name == "count" && !distinct { Ok(Projection::Count) } else { Err(err(format!("{}(field)", name.to_uppercase()))) };
        }
        let [arg] = args.as_slice() else { return Err(err(format!("{} takes one field", name.to_uppercase()))) };
        let field = self.field_of(arg)?;
        let fld = &self.schema.fields[field];
        let numeric = matches!(fld.ty, FieldType::Int | FieldType::Float);
        let func = match name.as_str() {
            "count" if distinct => AggFn::CountDistinct,
            "count" => AggFn::Count,
            "sum" => AggFn::Sum,
            "avg" => AggFn::Avg,
            "min" => AggFn::Min,
            _ => AggFn::Max,
        };
        match func {
            AggFn::Sum | AggFn::Avg if fld.logical == Some(Logical::Timestamp) => {
                return Err(err(format!("{}({}): times do not add up; a TIMESTAMP takes MIN, MAX and COUNT", name.to_uppercase(), fld.name)))
            }
            AggFn::Sum | AggFn::Avg if !numeric => return Err(err(format!("{}({}): not a numeric field", name.to_uppercase(), fld.name))),
            AggFn::Min | AggFn::Max | AggFn::CountDistinct if !(numeric || fld.ty == FieldType::Keyword) => {
                return Err(err(format!("{}({}): needs a KEYWORD, INT or FLOAT field", name.to_uppercase(), fld.name)))
            }
            AggFn::Count if !(numeric || fld.ty == FieldType::Keyword) => return Err(err(format!("COUNT({}): count a KEYWORD, INT or FLOAT field, or COUNT(*)", fld.name))),
            _ => {}
        }
        if distinct && func != AggFn::CountDistinct {
            return Err(err("DISTINCT works in COUNT only"));
        }
        if !fld.indexed {
            return Err(err(format!("{}: aggregates read the field's column, and it is not indexed", fld.name)));
        }
        Ok(Projection::Agg { func, field })
    }

    /// An aggregate or group value named in HAVING or ORDER BY: a function
    /// call, a selected column's alias or position, or a group field.
    fn group_value(&self, e: &Expr, proj: &[(String, Projection)], group: &[usize]) -> Result<Projection> {
        if let Some(n) = literal_usize(e) {
            return proj.get(n.wrapping_sub(1)).map(|(_, p)| p.clone()).ok_or_else(|| err(format!("{n}: no such column")));
        }
        match e {
            Expr::Nested(inner) => self.group_value(inner, proj, group),
            Expr::Function(f) if matches!(fname(f).as_str(), "count" | "sum" | "avg" | "min" | "max") => self.aggregate(f),
            Expr::Identifier(id) if proj.iter().any(|(n, p)| n.eq_ignore_ascii_case(&id.value) && !matches!(p, Projection::Field(_))) => {
                Ok(proj.iter().find(|(n, _)| n.eq_ignore_ascii_case(&id.value)).unwrap().1.clone())
            }
            _ => {
                let f = self.field_of(e)?;
                if group.contains(&f) {
                    Ok(Projection::Field(f))
                } else {
                    Err(err(format!("{e}: use a group field or an aggregate")))
                }
            }
        }
    }

    fn group_order(&self, e: &Expr, desc: bool, proj: &[(String, Projection)], group: &[usize]) -> Result<OrderBy> {
        Ok(match self.group_value(e, proj, group)? {
            Projection::Count => OrderBy::Count { desc },
            Projection::Field(f) if group.first() == Some(&f) => OrderBy::GroupValue { desc },
            what => OrderBy::Agg { what, desc },
        })
    }

    fn having(&self, e: &Expr, proj: &[(String, Projection)], group: &[usize]) -> Result<Having> {
        use BinaryOperator::*;
        Ok(match e {
            Expr::Nested(inner) => self.having(inner, proj, group)?,
            Expr::BinaryOp { left, op: And, right } => Having::And(vec![self.having(left, proj, group)?, self.having(right, proj, group)?]),
            Expr::BinaryOp { left, op: Or, right } => Having::Or(vec![self.having(left, proj, group)?, self.having(right, proj, group)?]),
            Expr::UnaryOp { op: sql::UnaryOperator::Not, expr } => Having::Not(Box::new(self.having(expr, proj, group)?)),
            Expr::Between { expr, negated, low, high } => {
                let what = self.group_value(expr, proj, group)?;
                let h = Having::Range { lo: self.having_bound(&what, low)?, hi: self.having_bound(&what, high)?, what };
                if *negated { Having::Not(Box::new(h)) } else { h }
            }
            Expr::BinaryOp { left, op, right } => {
                let (what, lit, op) = match self.group_value(left, proj, group) {
                    Ok(w) => (w, right, op.clone()),
                    Err(_) => {
                        let flipped = match op {
                            Lt => Gt,
                            LtEq => GtEq,
                            Gt => Lt,
                            GtEq => LtEq,
                            o => o.clone(),
                        };
                        (self.group_value(right, proj, group)?, left, flipped)
                    }
                };
                let v = match self.typed_source(&what) {
                    Some(f) => self.coerce(f, self.literal(lit)?)?,
                    None => self.literal(lit)?,
                };
                let Some(x) = v.as_f64() else {
                    return match op {
                        Eq => Ok(Having::Eq { what, value: v }),
                        NotEq => Ok(Having::Not(Box::new(Having::Eq { what, value: v }))),
                        _ => Err(err("HAVING compares strings with = and != only")),
                    };
                };
                let (inf, ninf) = (f64::INFINITY, f64::NEG_INFINITY);
                let range = |lo, hi| Having::Range { what: what.clone(), lo, hi };
                match op {
                    Eq => range(x, x),
                    NotEq => Having::Not(Box::new(range(x, x))),
                    Gt => range(next_up(x), inf),
                    GtEq => range(x, inf),
                    Lt => range(ninf, next_down(x)),
                    LtEq => range(ninf, x),
                    o => return Err(err(format!("unsupported operator {o} in HAVING"))),
                }
            }
            _ => return Err(err(format!("unsupported HAVING condition: {e}"))),
        })
    }

    fn typed_source(&self, p: &Projection) -> Option<usize> {
        typed_source(self.schema, p)
    }

    /// A bound of `BETWEEN` in HAVING: a number, or a value of the type of
    /// a BOOL or TIMESTAMP field (as its stored integer).
    fn having_bound(&self, what: &Projection, e: &Expr) -> Result<f64> {
        match self.typed_source(what) {
            Some(f) => Ok(self.field_value(f, e)?.as_f64().unwrap_or(f64::NAN)),
            None => self.num_lit(e),
        }
    }

    fn order(&self, e: &Expr, desc: bool, proj: &[(String, Projection)]) -> Result<OrderBy> {
        // Positional `ORDER BY 2` addresses a selected column.
        if let Some(n) = literal_usize(e) {
            let (_, p) = proj.get(n.wrapping_sub(1)).ok_or_else(|| err(format!("ORDER BY {n}: no such column")))?;
            return match p {
                Projection::Field(f) => Ok(OrderBy::Score { score: Score::Column { field: *f }, desc, nulls_first: false }),
                Projection::Score => Err(err("ORDER BY score(): order by the score expression itself")),
                Projection::SeqNo => Err(err("ORDER BY seq_no(): a query cannot order by it")),
                Projection::Hops(g) => Ok(OrderBy::Score { score: Score::Hops { graph: *g, far_first: desc }, desc: true, nulls_first: false }),
                Projection::Feature(s) => Ok(OrderBy::Score { score: s.clone(), desc, nulls_first: false }),
                Projection::Count | Projection::Agg { .. } => Err(err("ORDER BY an aggregate needs GROUP BY")),
            };
        }
        // `ORDER BY hops()` is nearest first; ranking is always best first.
        if let Expr::Function(f) = e {
            if fname(f) == "hops" {
                return Ok(OrderBy::Score { score: Score::Hops { graph: self.hops_graph()?, far_first: desc }, desc: true, nulls_first: false });
            }
        }
        // `ORDER BY distance(..) ASC` ranks like `similarity DESC`.
        if let Expr::Function(f) = e {
            if fname(f) == "distance" {
                let (args, opts) = self.args(f)?;
                return Ok(OrderBy::Score { score: self.similarity(&args, opts, true)?, desc: !desc, nulls_first: false });
            }
        }
        Ok(OrderBy::Score { score: self.score(e)?, desc, nulls_first: false })
    }

    /// A ranking: a score, a column, or an expression of them. `+` adds raw
    /// values; a sum of `w * minmax(score)` is the min-max fusion that `+`
    /// meant before ranking expressions, and a product of scores keeps its
    /// rule (a factor below 0, or missing, counts as 0). Both are written
    /// to plans in their old form, which older shards read.
    fn score(&self, e: &Expr) -> Result<Score> {
        Ok(self.rank(e)?.fused_shapes())
    }

    /// A part of a ranking expression.
    fn rank(&self, e: &Expr) -> Result<Score> {
        let bx = |e: &Expr| self.rank(e).map(Box::new);
        match e {
            Expr::Nested(inner) => self.rank(inner),
            // A number of each row ranked, from the rows it names.
            Expr::Subquery(q) => Ok(Score::Sub(Box::new(self.sub(q, Sub::Score)?))),
            Expr::Identifier(_) | Expr::CompoundIdentifier(_) => {
                let f = self.field_of(e)?;
                if self.schema.fields[f].multi {
                    return Err(err(format!("cannot order by multi-valued field {}", self.schema.fields[f].name)));
                }
                match self.schema.fields[f].ty {
                    FieldType::Int | FieldType::Float => Ok(Score::Column { field: f }),
                    _ => Err(err("ORDER BY a column needs a numeric field")),
                }
            }
            Expr::Value(_) | Expr::UnaryOp { op: sql::UnaryOperator::Minus, expr: _ } if self.number(e).is_some() => match self.number(e) {
                Some(x) if x.is_finite() => Ok(Score::Num(x)),
                _ => Err(err(format!("a number in a ranking expression is finite, not {e}"))),
            },
            Expr::Value(_) => Err(err(format!("a ranking expression takes numbers, not {e}"))),
            Expr::UnaryOp { op: sql::UnaryOperator::Minus, expr } => Ok(Score::Neg(bx(expr)?)),
            Expr::UnaryOp { op: sql::UnaryOperator::Plus, expr } => self.rank(expr),
            Expr::BinaryOp { left, op, right } => {
                let op = match op {
                    BinaryOperator::Plus => ArithOp::Add,
                    BinaryOperator::Minus => ArithOp::Sub,
                    BinaryOperator::Multiply => ArithOp::Mul,
                    BinaryOperator::Divide => ArithOp::Div,
                    BinaryOperator::Modulo => ArithOp::Mod,
                    o => return Err(err(format!("a ranking expression takes + - * / %, not {o}"))),
                };
                Ok(Score::Arith { op, left: bx(left)?, right: bx(right)? })
            }
            Expr::Case { operand, conditions, else_result, .. } => {
                let whens = conditions
                    .iter()
                    .map(|c| {
                        // `CASE x WHEN v` tests `x = v`.
                        let cond = match operand {
                            Some(x) => Expr::BinaryOp { left: x.clone(), op: BinaryOperator::Eq, right: Box::new(c.condition.clone()) },
                            None => c.condition.clone(),
                        };
                        let p = self.pred(&cond)?.three_valued(&|f| self.schema.fields[f].multi).simplify();
                        Ok((Score::Cond(Box::new(p)), self.rank(&c.result)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Score::Case { whens, otherwise: else_result.as_deref().map(bx).transpose()? })
            }
            Expr::Function(f) => {
                let name = fname(f);
                let (args, opts) = self.args(f)?;
                let one = |what: &str| match args.as_slice() {
                    [x] => Ok(x),
                    _ => Err(err(format!("{name}({what}) takes one argument"))),
                };
                match name.as_str() {
                    "bm25" => {
                        let [field, text] = args.as_slice() else { return Err(err("bm25(field, text)")) };
                        let field = self.typed_field(field, |t| matches!(t, FieldType::Text { .. }), "TEXT")?;
                        let text = self.literal(text)?.as_str().ok_or_else(|| err("bm25 text must be a string"))?.to_string();
                        let mut opts = opts;
                        Expand::from_opts(&opts).map_err(err)?;
                        crate::query::exec::plain_syntax(&opts)?;
                        let need = crate::query::exec::min_need(&opts)?;
                        self.guard(&mut opts, || GuardSite::Bm25 { field, text: text.clone() }, need, None, "bm25")?;
                        Ok(Score::Bm25 { field, text, opts: without_norm(opts) })
                    }
                    "similarity" => self.similarity(&args, opts, false),
                    // A field's dot product with a vector is its similarity
                    // (NULL for a row without one), in either order.
                    "dot" => match args.as_slice() {
                        [a, b] if self.field_of(a).is_err() && self.field_of(b).is_ok() => self.similarity(&[*b, *a], opts, false),
                        [_, _] => self.similarity(&args, opts, false),
                        _ => Err(err("dot(field, vector) takes two arguments")),
                    },
                    "distance" => Err(err("distance(...) ranks on its own, ORDER BY distance(...) ASC; in an expression, use similarity(...), which ranks larger first")),
                    "maxsim" => self.maxsim(&args, opts),
                    "rrf" => {
                        if args.len() < 2 {
                            return Err(err("rrf needs at least two scores"));
                        }
                        let inputs = args
                            .iter()
                            .map(|a| match a {
                                // A ranking reads hops nearest first.
                                Expr::Function(g) if fname(g) == "hops" => Ok(Score::Hops { graph: self.hops_graph()?, far_first: false }),
                                a => match self.rank(a)? {
                                    s if s.is_leaf() && !matches!(s, Score::SeqNo | Score::Cond(_)) => Ok(s),
                                    _ => Err(err(format!("rrf fuses the rankings of scores and columns, not of {a}"))),
                                },
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let num = |k: &str, d: f64| opts.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
                        Ok(Score::Rrf { inputs, k: num("k", 60.0), depth: num("depth", 100.0) as usize })
                    }
                    "ppr" => Ok(Score::Ppr { graph: self.graph(f, "ppr")? }),
                    "decay" => self.decay(f),
                    // The number of hops, in an expression.
                    "hops" if args.is_empty() => Ok(Score::Hops { graph: self.hops_graph()?, far_first: true }),
                    "seq_no" if args.is_empty() => Ok(Score::SeqNo),
                    "ranked" => match args.as_slice() {
                        [keys] => Ok(Score::Ranked { keys: self.ranked_keys(keys)? }),
                        _ => Err(err("ranked(keys) takes one argument: a list of primary keys, best first")),
                    },
                    "features" => Err(err("features() stands for several columns: it takes no alias and is not part of an expression")),
                    "minmax" => Ok(Score::Norm { kind: NormKind::MinMax, input: bx(one("x")?)?, lo: None }),
                    "rank" => Ok(Score::Norm { kind: NormKind::Rank, input: bx(one("x")?)?, lo: None }),
                    "zscore" => Ok(Score::Norm { kind: NormKind::ZScore, input: bx(one("x")?)?, lo: None }),
                    "tmm" => self.tmm(*one("x")?, &opts),
                    "gbdt" => {
                        let usage = "gbdt('model', feature, ...) takes a model's name and the features it reads, in order";
                        let [model, features @ ..] = args.as_slice() else { return Err(err(usage)) };
                        let model = self.literal(model).ok().and_then(|v| v.as_str().map(str::to_string)).ok_or_else(|| err(usage))?;
                        if features.is_empty() || !opts.is_empty() {
                            return Err(err(usage));
                        }
                        Ok(Score::Gbdt { model, hash: String::new(), args: features.iter().map(|a| self.rank(a)).collect::<Result<_>>()?, open: Default::default() })
                    }
                    "cross_encoder" => {
                        let usage = "cross_encoder('model', field, ..., query) takes a model's name, the fields whose text it reads and the query";
                        let [model, fields @ .., query] = args.as_slice() else { return Err(err(usage)) };
                        let model = self.literal(model).ok().and_then(|v| v.as_str().map(str::to_string)).ok_or_else(|| err(usage))?;
                        let query = self.literal(query).ok().and_then(|v| v.as_str().map(str::to_string)).ok_or_else(|| err(usage))?;
                        if fields.is_empty() || !opts.is_empty() {
                            return Err(err(usage));
                        }
                        let fields = fields
                            .iter()
                            .map(|f| {
                                let f = self.typed_field(f, |t| matches!(t, FieldType::Text { .. } | FieldType::Keyword), "TEXT or KEYWORD")?;
                                match self.schema.fields[f].stored {
                                    true => Ok(f),
                                    false => Err(err(format!("cross_encoder reads a stored field's text, and {} is not stored", self.schema.fields[f].name))),
                                }
                            })
                            .collect::<Result<_>>()?;
                        Ok(Score::CrossEncoder { model, hash: String::new(), fields, query, open: Default::default() })
                    }
                    // Each condition 1 where WHERE would keep the row, else 0.
                    "count_true" => {
                        if args.is_empty() || !opts.is_empty() {
                            return Err(err("count_true(condition, ...) takes one or more conditions"));
                        }
                        let cond = |c: &Expr| Ok(Score::Cond(Box::new(self.pred(c)?.three_valued(&|f| self.schema.fields[f].multi).simplify())));
                        Ok(Score::Call { func: Func::CountTrue, args: args.iter().map(|a| cond(a)).collect::<Result<_>>()? })
                    }
                    n => match Func::numeric(n) {
                        Some(func) => {
                            func.arity(args.len()).map_err(|a| err(format!("{n}() takes {a}")))?;
                            Ok(Score::Call { func, args: args.iter().map(|a| self.rank(a)).collect::<Result<_>>()? })
                        }
                        None => Err(err(format!("unknown score function {n}"))),
                    },
                }
            }
            _ => Err(err(format!("unsupported ORDER BY expression: {e}"))),
        }
    }

    /// `similarity(field, vector)`, or `distance(field, vector)` (which
    /// ranks the same, ascending, on its own in ORDER BY).
    /// `tmm(x [, lo => a] [, hi => b])`: theoretical min-max. `lo` is the
    /// least value `x` can take, `x`'s own by default ([`Score::least`]);
    /// with `hi` too the bounds are fixed, and it is the row's own value
    /// `(x - a) / (b - a)`, 0 where `x` is NULL.
    fn tmm(&self, x: &Expr, opts: &Opts) -> Result<Score> {
        let input = self.rank(x)?;
        let bound = |k: &str| -> Result<Option<f64>> {
            match opts.get(k) {
                None => Ok(None),
                Some(v) => match v.as_f64() {
                    Some(b) if b.is_finite() => Ok(Some(b)),
                    _ => Err(err(format!("tmm's {k} is a number, not {v:?}"))),
                },
            }
        };
        if let Some(k) = opts.keys().find(|k| *k != "lo" && *k != "hi") {
            return Err(err(format!("tmm(x, lo => a, hi => b) takes the options lo and hi, not {k}")));
        }
        let lo = match bound("lo")?.or_else(|| input.least(self.schema)) {
            Some(lo) => lo,
            None => return Err(err(format!("tmm({x}) needs the least value {x} can take: tmm({x}, lo => n)"))),
        };
        match bound("hi")? {
            None => Ok(Score::Norm { kind: NormKind::Tmm, input: Box::new(input), lo: Some(lo) }),
            Some(hi) if hi > lo => {
                let scaled = Score::Arith {
                    op: ArithOp::Div,
                    left: Box::new(Score::Arith { op: ArithOp::Sub, left: Box::new(input), right: Box::new(Score::Num(lo)) }),
                    right: Box::new(Score::Num(hi - lo)),
                };
                Ok(Score::Call { func: Func::Coalesce, args: vec![scaled, Score::Num(0.0)] })
            }
            Some(hi) => Err(err(format!("tmm's hi ({hi}) must be above its lo ({lo})"))),
        }
    }

    fn similarity(&self, args: &[&Expr], opts: Opts, distance: bool) -> Result<Score> {
        let [field, v] = args else { return Err(err("similarity(field, vector)")) };
        let field = self.typed_field(field, |t| matches!(t, FieldType::Vector { .. } | FieldType::Sparse), "VECTOR or SPARSE")?;
        if self.schema.fields[field].ty == FieldType::Sparse {
            if distance {
                return Err(err("a SPARSE field takes similarity(...), not distance(...)"));
            }
            let terms = crate::text::sparse::parse(&self.literal(v)?).map_err(|e| err(format!("similarity({}, ..): {e}", self.schema.fields[field].name)))?;
            return Ok(Score::Sparse { field, terms, opts: without_norm(opts) });
        }
        let FieldType::Vector { dim, .. } = self.schema.fields[field].ty else { unreachable!() };
        if self.schema.fields[field].multi {
            return Err(err(multi_similarity(&self.schema.fields[field].name)));
        }
        if let Some(e) = self.encode(v)? {
            if distance {
                return Err(err("encode(..) ranks by similarity(field, encode(..)), larger first, not by distance"));
            }
            return Ok(Score::MaxSim { field, vectors: Vectors::Plan(Box::new(e)), opts: without_norm(opts) });
        }
        let vector = self.vector(v)?;
        if vector.len() != dim {
            return Err(err(format!("vector has {} dimensions, field has {dim}", vector.len())));
        }
        Ok(Score::Similarity { field, vector, opts: without_norm(opts) })
    }

    /// `maxsim(field, vectors)`: the vectors are a list parameter, a tuple
    /// or array of vectors, a JSON array of them, or a subquery of one
    /// column of vectors (in a compound query, which runs it first).
    fn maxsim(&self, args: &[&Expr], opts: Opts) -> Result<Score> {
        let [field, vs] = args else { return Err(err("maxsim(field, vectors)")) };
        let field = self.typed_field(field, |t| matches!(t, FieldType::Vector { .. }), "VECTOR")?;
        let FieldType::Vector { dim, .. } = self.schema.fields[field].ty else { unreachable!("a VECTOR field") };
        for k in ["max_vectors", "depth", "candidates"] {
            if let Some(v) = opts.get(k) {
                v.as_f64().filter(|x| *x >= 1.0 && x.fract() == 0.0).ok_or_else(|| err(format!("{k} => a whole number, at least 1")))?;
            }
        }
        if let Some(q) = subquery_of(vs) {
            return Ok(Score::MaxSim { field, vectors: Vectors::Plan(Box::new(self.sub(q, Sub::Vectors)?)), opts: without_norm(opts) });
        }
        if let Some(e) = self.encode(vs)? {
            return Ok(Score::MaxSim { field, vectors: Vectors::Plan(Box::new(e)), opts: without_norm(opts) });
        }
        let vectors: Vec<Arc<[f32]>> = match vs {
            Expr::Tuple(items) => items.iter().map(|i| self.vector(i)).collect::<Result<_>>()?,
            Expr::Array(a) => a.elem.iter().map(|i| self.vector(i)).collect::<Result<_>>()?,
            Expr::Nested(inner) => vec![self.vector(inner)?],
            _ => match self.literal(vs)? {
                // A JSON array of vectors.
                Value::Str(s) => {
                    let vs: Vec<Vec<f32>> = serde_json::from_str(&s).map_err(|_| err("maxsim's vectors as text are a JSON array of arrays of numbers"))?;
                    vs.into_iter().map(Arc::from).collect()
                }
                Value::List(xs) if xs.iter().all(|x| matches!(x, Value::Int(_) | Value::Float(_))) => vec![self.vector_value(Value::List(xs))?],
                Value::List(xs) => xs.into_iter().map(|x| self.vector_value(x)).collect::<Result<_>>()?,
                v => vec![self.vector_value(v)?],
            },
        };
        if let Some(v) = vectors.iter().find(|v| v.len() != dim) {
            return Err(err(format!("a query vector has {} dimensions, field has {dim}", v.len())));
        }
        if vectors.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
            return Err(err("a vector's components must be finite numbers"));
        }
        Ok(Score::MaxSim { field, vectors: Vectors::List(vectors), opts: without_norm(opts) })
    }

    /// `encode('model', (SELECT item, action, time FROM ... ORDER BY time
    /// DESC LIMIT 200), now => t)`, if `e` is a call of it: the history's
    /// vector by a sequence model ([`crate::query::sequence`]), once the
    /// subquery ran.
    fn encode(&self, e: &Expr) -> Result<Option<ir::Expr>> {
        let e = match e {
            Expr::Nested(x) => x,
            e => e,
        };
        let Expr::Function(f) = e else { return Ok(None) };
        if fname(f) != "encode" {
            return Ok(None);
        }
        encode_call(f, &|e| self.literal(e), &|q| self.sub(q, Sub::Rows)).map(Some)
    }

    /// `decay(field, origin, scale, offset => 0, decay => 0.5, shape =>
    /// 'exp' | 'gauss' | 'linear')`; the origin may be `now()` (seconds
    /// since the Unix epoch). On a TIMESTAMP field the origin is `now()` or
    /// a time, and the scale and offset are seconds.
    fn decay(&self, f: &sql::Function) -> Result<Score> {
        let (args, opts) = self.args(f)?;
        let usage = "decay(field, origin, scale, offset => .., decay => .., shape => 'exp' | 'gauss' | 'linear')";
        let [field, origin, scale] = args.as_slice() else { return Err(err(usage)) };
        let field = self.numeric_field(field)?;
        if self.schema.fields[field].multi {
            return Err(err(format!("cannot decay multi-valued field {}", self.schema.fields[field].name)));
        }
        // A TIMESTAMP's origin is a time, and its scale and offset are
        // seconds, as now() is for a number.
        let time = self.schema.fields[field].logical == Some(Logical::Timestamp);
        let unit = if time { 1e6 } else { 1.0 };
        let now = || self.now as f64 / 1e6 * unit;
        let origin = match origin {
            Expr::Function(g) if fname(g) == "now" => now(),
            e => match self.literal(e)? {
                Value::Str(s) if s.eq_ignore_ascii_case("now") => now(),
                v if time => self.coerce(field, v)?.as_f64().unwrap_or(0.0),
                v => v.as_f64().ok_or_else(|| err("decay origin must be a number or now()"))?,
            },
        };
        let scale = self.num_lit(scale)? * unit;
        let num = |k: &str, d: f64| opts.get(k).map_or(Ok(d), |v| v.as_f64().ok_or_else(|| err(format!("decay {k} must be a number"))));
        let (offset, decay) = (num("offset", 0.0)? * unit, num("decay", 0.5)?);
        if scale <= 0.0 || offset < 0.0 || !(decay > 0.0 && decay < 1.0) {
            return Err(err("decay needs scale > 0, offset >= 0 and 0 < decay < 1"));
        }
        let shape = match opts.get("shape").map(|v| v.as_str().unwrap_or("").to_ascii_lowercase()).as_deref() {
            None | Some("exp") => DecayShape::Exp,
            Some("gauss") => DecayShape::Gauss,
            Some("linear") => DecayShape::Linear,
            Some(s) => return Err(err(format!("unknown decay shape {s:?}: exp, gauss or linear"))),
        };
        Ok(Score::Decay { field, decay: Decay { origin, scale, offset, decay, shape }, opts })
    }

    fn pred(&self, e: &Expr) -> Result<Pred> {
        Ok(match e {
            Expr::Nested(inner) => self.pred(inner)?,
            Expr::BinaryOp { left, op: BinaryOperator::And, right } => Pred::And(vec![self.pred(left)?, self.pred(right)?]),
            Expr::BinaryOp { left, op: BinaryOperator::Or, right } => Pred::Or(vec![self.pred(left)?, self.pred(right)?]),
            Expr::UnaryOp { op: sql::UnaryOperator::Not, expr } => Pred::Not(Box::new(self.pred(expr)?)),
            Expr::BinaryOp { left, op, right } if self.subs.is_some() && (subquery_of(left).is_some() || subquery_of(right).is_some()) => self.sub_comparison(left, op, right)?,
            Expr::BinaryOp { left, op, right } if outer_field(left).is_some() || outer_field(right).is_some() => self.outer_comparison(left, op, right)?,
            Expr::BinaryOp { left, op, right } => self.comparison(left, op, right)?,
            Expr::Between { expr, negated, low, high } if self.subs.is_some() && (subquery_of(low).is_some() || subquery_of(high).is_some()) => {
                let f = self.numeric_field(expr)?;
                let bound = |b: &Expr| match subquery_of(b) {
                    Some(q) => self.sub(q, Sub::Scalar),
                    None => Ok(ir::Expr::Lit(self.literal(b)?)),
                };
                let p = Pred::Sub(Box::new(ir::Expr::Between { expr: Box::new(ir::Expr::Field(f as u32)), lo: Some(Box::new(bound(low)?)), hi: Some(Box::new(bound(high)?)) }));
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::InSubquery { expr, subquery, negated } if self.subs.is_some() => {
                let f = self.field_of(expr)?;
                if matches!(self.schema.fields[f].ty, FieldType::Edges | FieldType::Text { .. } | FieldType::Vector { .. } | FieldType::Sparse) {
                    return Err(err(format!("IN compares KEYWORD, INT and FLOAT fields, not {}", self.schema.fields[f].name)));
                }
                let p = Pred::Sub(Box::new(ir::Expr::In { expr: Box::new(ir::Expr::Field(f as u32)), list: vec![self.sub(subquery, Sub::In)?] }));
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::Exists { subquery, negated } if self.subs.is_some() => {
                let p = Pred::Sub(Box::new(self.sub(subquery, Sub::Exists)?));
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::Between { expr, negated, low, high } => {
                let f = self.numeric_field(expr)?;
                let p = if self.schema.fields[f].ty == FieldType::Int {
                    int_range(f, ceil(&self.int_bound(f, low)?), floor(&self.int_bound(f, high)?))
                } else {
                    Pred::Range { field: f, lo: self.num_lit(low)?, hi: self.num_lit(high)? }
                };
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::IsNull(x) | Expr::IsNotNull(x) => {
                let f = self.field_of(x)?;
                let field = &self.schema.fields[f];
                if !matches!(field.ty, FieldType::Int | FieldType::Float | FieldType::Keyword) || !field.indexed {
                    return Err(err(format!("IS NULL works on indexed INT, FLOAT and KEYWORD fields, not {}", field.name)));
                }
                let p = Pred::Present { field: f };
                if matches!(e, Expr::IsNull(_)) { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::InList { expr, list, negated } => {
                let f = self.field_of(expr)?;
                if self.schema.fields[f].ty == FieldType::Edges {
                    return Err(err(format!("{} is an EDGES field: use REACHABLE({}, keys) for rows linked to keys", self.schema.fields[f].name, self.schema.fields[f].name)));
                }
                // A list naming the outer row's columns: a correlated
                // subquery's, compared once the outer row is known.
                if list.iter().any(|v| outer_field(v).is_some()) {
                    let items = list
                        .iter()
                        .map(|v| match outer_field(v) {
                            Some(o) => Ok(ir::Expr::Outer(o)),
                            None => Ok(ir::Expr::Lit(match self.literal(v)? {
                                Value::Null => Value::Null,
                                x => self.compared(f, x)?,
                            })),
                        })
                        .collect::<Result<_>>()?;
                    let p = Pred::Sub(Box::new(ir::Expr::In { expr: Box::new(ir::Expr::Field(f as u32)), list: items }));
                    return Ok(if *negated { Pred::Not(Box::new(p)) } else { p });
                }
                // A list-valued parameter stands for its values (`IN (:ids)`).
                // A NULL is kept: `x IN (1, NULL)` is UNKNOWN where x is not
                // 1, so `NOT IN` holds for no row ([`Pred::three_valued`]).
                let mut values = Vec::with_capacity(list.len());
                for v in list {
                    match as_list(self.literal(v)?) {
                        Ok(xs) => {
                            for x in xs {
                                values.push(self.compared(f, x)?);
                            }
                        }
                        Err(Value::Null) => values.push(Value::Null),
                        Err(x) => values.push(self.compared(f, x)?),
                    }
                }
                let p = if values.is_empty() { Pred::False } else { Pred::In { field: f, values } };
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::Function(func) if fname(func) == "match" => {
                let (args, opts) = self.args(func)?;
                let [field, q] = args.as_slice() else { return Err(err("MATCH(field, query, synonyms => 0 | 1, fuzzy => 0 | 1 | 2 | 'auto', min => n | 'p%')")) };
                let field = self.typed_field(field, |t| matches!(t, FieldType::Text { .. }), "TEXT")?;
                if !self.schema.fields[field].indexed {
                    return Err(err(format!("{} is not indexed", self.schema.fields[field].name)));
                }
                if let Some(k) = opts.keys().find(|k| !matches!(k.as_str(), "synonyms" | "fuzzy" | "min" | "min_rows" | "max_rows")) {
                    return Err(err(format!("unknown option {k} for MATCH: synonyms, fuzzy, min, min_rows, max_rows")));
                }
                let text = self.literal(q)?.as_str().ok_or_else(|| err("MATCH query must be a string"))?.to_string();
                crate::text::query::parse_match_with(&text, self.schema.fields[field].text_analyzer())?;
                let need = crate::query::exec::min_need(&opts)?;
                let mut guarded = opts.clone();
                let expand = Expand::from_opts(&opts).map_err(err)?;
                // `min => n`: the query as one group a row needs n of.
                let query = match need {
                    Some(need) => format!("({text})@{need}"),
                    None => text.clone(),
                };
                let p = Pred::Match { field, query, expand };
                self.guard(&mut guarded, || GuardSite::Match { field, text, expand }, need, Some(&p), "MATCH")?;
                p
            }
            // At least n of the conditions; n may be a share of them.
            Expr::Function(func) if fname(func) == "at_least" => {
                let usage = "AT_LEAST(n, condition, ...): n a count (2) or a percentage ('75%') of the conditions";
                let (args, mut opts) = self.args(func)?;
                let [n, conds @ ..] = args.as_slice() else { return Err(err(usage)) };
                if conds.is_empty() {
                    return Err(err(usage));
                }
                if let Some(k) = opts.keys().find(|k| !matches!(k.as_str(), "min_rows" | "max_rows")) {
                    return Err(err(format!("unknown option {k} for AT_LEAST: min_rows, max_rows")));
                }
                let need = crate::query::exec::need_of(&self.literal(n)?, "AT_LEAST's n is")?.of(conds.len());
                let items: Vec<Pred> = conds.iter().map(|c| self.pred(c)).collect::<Result<_>>()?;
                let p = Pred::AtLeast { need, items: items.clone() };
                self.guard(&mut opts, || GuardSite::AtLeast(items), Some(crate::text::query::Need::Count(need)), Some(&p), "AT_LEAST")?;
                p
            }
            Expr::Like { negated, any: false, expr, pattern, escape_char } => {
                let f = self.prefix_field(expr)?;
                let escape = match escape_char {
                    None => None,
                    Some(sql::Value::SingleQuotedString(c)) if c.chars().count() == 1 => c.chars().next(),
                    Some(c) => return Err(err(format!("LIKE's ESCAPE takes one character, not {c}"))),
                };
                let pattern = self.literal(pattern)?;
                let pattern = pattern.as_str().ok_or_else(|| err("LIKE's pattern must be text"))?;
                let p = match like_prefix(pattern, escape)? {
                    LikePrefix::Exact(v) => Pred::In { field: f, values: vec![Value::Str(v)] },
                    // The empty prefix is any value, and UNKNOWN without
                    // one: `NOT LIKE '%'` holds for no row
                    // ([`Pred::simplify`] makes it `IS NOT NULL`).
                    LikePrefix::Prefix(p) => Pred::Prefix { field: f, prefix: p, edits: 0 },
                };
                if *negated { Pred::Not(Box::new(p)) } else { p }
            }
            Expr::ILike { .. } => return Err(err("ILIKE cannot use a KEYWORD field's index: store the value lowercased and use LIKE")),
            Expr::Function(func) if fname(func) == "prefix" => {
                let (args, opts) = self.args(func)?;
                let [field, p] = args.as_slice() else { return Err(err("PREFIX(field, prefix [, fuzzy => 0 | 1 | 2 | 'auto'])")) };
                if let Some(k) = opts.keys().find(|k| k.as_str() != "fuzzy") {
                    return Err(err(format!("unknown option {k} for PREFIX")));
                }
                let fuzzy = Expand::from_opts(&opts).map_err(err)?.fuzzy;
                let f = self.prefix_field(field)?;
                match self.literal(p)? {
                    Value::Str(p) if p.is_empty() => Pred::Prefix { field: f, prefix: p, edits: 0 },
                    // 'auto' by the typed prefix's length, as MATCH does
                    // by a word's.
                    Value::Str(p) => Pred::Prefix { edits: fuzzy.map_or(0, |z| z.edits(p.chars().count())), field: f, prefix: p },
                    v => return Err(err(format!("PREFIX takes text, not {v:?}"))),
                }
            }
            Expr::Function(func) if matches!(fname(func).as_str(), "reachable" | "path") => Pred::Graph { id: self.graph(func, &fname(func))? },
            Expr::Value(v) if matches!(v.value, sql::Value::Boolean(_)) => {
                if matches!(v.value, sql::Value::Boolean(true)) { Pred::True } else { Pred::False }
            }
            e if self.bool_field(e).is_some() => int_range(self.bool_field(e).unwrap(), 1, 1),
            Expr::IsTrue(x) | Expr::IsFalse(x) | Expr::IsNotTrue(x) | Expr::IsNotFalse(x) => {
                let f = self.bool_field(x).ok_or_else(|| err(format!("{e}: IS TRUE and IS FALSE test a BOOL field")))?;
                let want = matches!(e, Expr::IsTrue(_) | Expr::IsNotFalse(_)) as i128;
                match e {
                    Expr::IsTrue(_) | Expr::IsFalse(_) => int_range(f, want, want),
                    // Unlike `NOT done`, `done IS NOT TRUE` holds for NULL.
                    _ => Pred::Or(vec![Pred::Not(Box::new(Pred::Present { field: f })), int_range(f, want, want)]),
                }
            }
            _ => return Err(err(format!("unsupported predicate: {e}"))),
        })
    }

    /// `field op (SELECT ...)`, or the other way round.
    /// A comparison of a field with a column of the row a correlated
    /// subquery is computed for (`cat = first.cat`), which takes its value
    /// for each distinct one ([`ir::Expr::Outer`]).
    fn outer_comparison(&self, left: &Expr, op: &BinaryOperator, right: &Expr) -> Result<Pred> {
        let (field, outer, flipped) = match (outer_field(left), outer_field(right)) {
            (None, Some(o)) => (left, o, false),
            (Some(o), None) => (right, o, true),
            _ => return Err(err(format!("{left} {op} {right}: a correlated subquery compares its own columns with the outer row's"))),
        };
        let f = self.field_of(field).map_err(|_| err(format!("{left} {op} {right}: a correlated subquery of a table compares a column of the table with one of the outer row (cat = first.cat)")))?;
        if matches!(self.schema.fields[f].ty, FieldType::Edges | FieldType::Text { .. } | FieldType::Vector { .. } | FieldType::Sparse) {
            return Err(err(format!("{left} {op} {right}: a correlated subquery compares KEYWORD, INT, FLOAT, BOOL or TIMESTAMP fields")));
        }
        let op = match (op, flipped) {
            (BinaryOperator::Eq, _) => CmpOp::Eq,
            (BinaryOperator::NotEq, _) => CmpOp::Ne,
            (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => CmpOp::Lt,
            (BinaryOperator::LtEq, false) | (BinaryOperator::GtEq, true) => CmpOp::Le,
            (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => CmpOp::Gt,
            (BinaryOperator::GtEq, false) | (BinaryOperator::LtEq, true) => CmpOp::Ge,
            (o, _) => return Err(err(format!("a correlated subquery compares with =, !=, <, <=, > or >=, not {o}"))),
        };
        Ok(Pred::Sub(Box::new(ir::Expr::Cmp { op, left: Box::new(ir::Expr::Field(f as u32)), right: Box::new(ir::Expr::Outer(outer)) })))
    }

    fn sub_comparison(&self, left: &Expr, op: &BinaryOperator, right: &Expr) -> Result<Pred> {
        use BinaryOperator::*;
        let (field_e, q, op) = match (subquery_of(left), subquery_of(right)) {
            (_, Some(q)) => (left, q, op.clone()),
            (Some(q), None) => {
                let flipped = match op {
                    Lt => Gt,
                    LtEq => GtEq,
                    Gt => Lt,
                    GtEq => LtEq,
                    o => o.clone(),
                };
                (right, q, flipped)
            }
            (None, None) => unreachable!("a side is a subquery"),
        };
        let f = self.field_of(field_e).map_err(|_| err(format!("a scalar subquery compares with a field, not {field_e}")))?;
        let op = match op {
            Eq => ir::CmpOp::Eq,
            NotEq => ir::CmpOp::Ne,
            Lt => ir::CmpOp::Lt,
            LtEq => ir::CmpOp::Le,
            Gt => ir::CmpOp::Gt,
            GtEq => ir::CmpOp::Ge,
            o => return Err(err(format!("unsupported operator {o}"))),
        };
        match self.schema.fields[f].ty {
            FieldType::Keyword if !matches!(op, ir::CmpOp::Eq | ir::CmpOp::Ne) => return Err(err("keyword fields support =, != and IN")),
            FieldType::Keyword | FieldType::Int | FieldType::Float => {}
            _ => return Err(err(format!("cannot compare field {}", self.schema.fields[f].name))),
        }
        Ok(Pred::Sub(Box::new(ir::Expr::Cmp { op, left: Box::new(ir::Expr::Field(f as u32)), right: Box::new(self.sub(q, Sub::Scalar)?) })))
    }

    fn comparison(&self, left: &Expr, op: &BinaryOperator, right: &Expr) -> Result<Pred> {
        use BinaryOperator::*;
        // Normalise to `field op literal`.
        let (field_e, lit_e, op) = if self.field_of(left).is_ok() {
            (left, right, op.clone())
        } else {
            let flipped = match op {
                Lt => Gt,
                LtEq => GtEq,
                Gt => Lt,
                GtEq => LtEq,
                o => o.clone(),
            };
            (right, left, flipped)
        };
        let f = self.field_of(field_e)?;
        let ty = &self.schema.fields[f].ty;
        if matches!(self.literal(lit_e), Ok(Value::Null)) {
            return Err(err(format!("{e} compares with NULL, which is never true: use IS NULL or IS NOT NULL", e = Expr::BinaryOp { left: Box::new(left.clone()), op: op.clone(), right: Box::new(right.clone()) })));
        }
        if matches!(ty, FieldType::Keyword) {
            let v = self.compared(f, self.literal(lit_e)?)?;
            return match op {
                Eq => Ok(Pred::In { field: f, values: vec![v] }),
                NotEq => Ok(Pred::Not(Box::new(Pred::In { field: f, values: vec![v] }))),
                _ => Err(err("keyword fields support =, != and IN")),
            };
        }
        if matches!(ty, FieldType::Int) {
            // Integers compare exactly, in 128 bits so that the bounds of
            // `> i64::MAX` or `< 1e30` need no special cases.
            let v = self.int_bound(f, lit_e)?;
            let (min, max) = (i64::MIN as i128, i64::MAX as i128);
            return Ok(match op {
                Eq => int_range(f, ceil(&v), floor(&v)),
                NotEq => Pred::Not(Box::new(int_range(f, ceil(&v), floor(&v)))),
                Gt => int_range(f, floor(&v) + 1, max),
                GtEq => int_range(f, ceil(&v), max),
                Lt => int_range(f, min, ceil(&v) - 1),
                LtEq => int_range(f, min, floor(&v)),
                o => return Err(err(format!("unsupported operator {o}"))),
            });
        }
        if !matches!(ty, FieldType::Float) {
            return Err(err(format!("cannot compare field {}", self.schema.fields[f].name)));
        }
        let x = self.num_lit(lit_e)?;
        let (inf, ninf) = (f64::INFINITY, f64::NEG_INFINITY);
        let range = |lo, hi| Pred::Range { field: f, lo, hi };
        Ok(match op {
            Eq => range(x, x),
            NotEq => Pred::Not(Box::new(range(x, x))),
            Gt => range(next_up(x), inf),
            GtEq => range(x, inf),
            Lt => range(ninf, next_down(x)),
            LtEq => range(ninf, x),
            o => return Err(err(format!("unsupported operator {o}"))),
        })
    }

    /// Positional and named (`name => value`) arguments of a function call.
    fn args<'e>(&self, f: &'e sql::Function) -> Result<(Vec<&'e Expr>, Opts)> {
        let (pos, named) = self.args_raw(f)?;
        let mut opts = Opts::new();
        for (k, e) in named {
            opts.insert(k, self.literal(e)?);
        }
        Ok((pos, opts))
    }

    /// Positional arguments and unevaluated named ones.
    fn args_raw<'e>(&self, f: &'e sql::Function) -> Result<(Vec<&'e Expr>, Vec<(String, &'e Expr)>)> {
        let FunctionArguments::List(list) = &f.args else {
            return Ok((Vec::new(), Vec::new()));
        };
        let (mut pos, mut named) = (Vec::new(), Vec::new());
        for a in &list.args {
            match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => pos.push(e),
                FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {}
                FunctionArg::Named { name, arg: FunctionArgExpr::Expr(e), .. } => named.push((name.value.to_ascii_lowercase(), e)),
                _ => return Err(err(format!("unsupported argument {a}"))),
            }
        }
        Ok((pos, named))
    }

    fn field_of(&self, e: &Expr) -> Result<usize> {
        let name = match e {
            Expr::Identifier(id) => &id.value,
            Expr::CompoundIdentifier(ids) => &ids.last().unwrap().value,
            _ => return Err(err(format!("expected a column, got {e}"))),
        };
        self.schema.field_index(name).ok_or_else(|| err(format!("no such column: {name}")))
    }

    fn typed_field(&self, e: &Expr, ok: impl Fn(&FieldType) -> bool, what: &str) -> Result<usize> {
        let f = self.field_of(e)?;
        if ok(&self.schema.fields[f].ty) { Ok(f) } else { Err(err(format!("{} is not a {what} field", self.schema.fields[f].name))) }
    }

    /// The field of `LIKE` and `PREFIX()`: an indexed KEYWORD field.
    fn prefix_field(&self, e: &Expr) -> Result<usize> {
        let f = self.typed_field(e, |t| *t == FieldType::Keyword, "KEYWORD")?;
        if !self.schema.fields[f].indexed {
            return Err(err(format!("{} is not indexed", self.schema.fields[f].name)));
        }
        Ok(f)
    }

    fn numeric_field(&self, e: &Expr) -> Result<usize> {
        self.typed_field(e, |t| matches!(t, FieldType::Int | FieldType::Float), "numeric")
    }

    fn literal(&self, e: &Expr) -> Result<Value> {
        match e {
            Expr::Nested(x) => self.literal(x),
            // The statement's time, as text a TIMESTAMP takes.
            e if is_now(e) => Ok(Value::Str(crate::timestamp::format(self.now))),
            Expr::Value(v) => match &v.value {
                sql::Value::Number(s, _) => {
                    Ok(s.parse::<i64>().map(Value::Int).or_else(|_| s.parse::<f64>().map(Value::Float)).map_err(|_| err("bad number"))?)
                }
                sql::Value::SingleQuotedString(s) | sql::Value::DoubleQuotedString(s) => Ok(Value::Str(s.clone())),
                sql::Value::Boolean(b) => Ok(Value::Bool(*b)),
                sql::Value::Null => Ok(Value::Null),
                sql::Value::Placeholder(p) => Ok(self.params.get(p)?.clone()),
                other => Err(err(format!("unsupported literal {other}"))),
            },
            Expr::UnaryOp { op: sql::UnaryOperator::Minus, expr } if negative_integer(expr).is_some() => Ok(Value::Int(negative_integer(expr).expect("an integer"))),
            Expr::UnaryOp { op: sql::UnaryOperator::Minus, expr } => match self.literal(expr)? {
                // -i64::MIN is the one integer without a negation.
                Value::Int(x) => Ok(x.checked_neg().map_or(Value::Float(-(x as f64)), Value::Int)),
                Value::Float(x) => Ok(Value::Float(-x)),
                _ => Err(err("cannot negate a non-number")),
            },
            // "x" reads as a string, as in SQLite and MySQL; an unquoted
            // name is a column, which is no value.
            Expr::Identifier(id) if id.quote_style == Some('"') => Ok(Value::Str(id.value.clone())),
            Expr::Identifier(id) if self.schema.field_index(&id.value).is_some() => {
                Err(err(format!("{} is a column, and a value is expected here (a string is quoted: '{}')", id.value, id.value)))
            }
            Expr::Identifier(id) => Err(err(format!("no such column: {} (a string is quoted: '{}')", id.value, id.value))),
            Expr::Subquery(_) => Err(err(format!("{e}: a subquery works in a condition (IN, EXISTS, a comparison with a field) or as graph seeds, not as a value here"))),
            _ => Err(err(format!("expected a literal or parameter, got {e}"))),
        }
    }

    fn number(&self, e: &Expr) -> Option<f64> {
        if self.field_of(e).is_ok() {
            return None;
        }
        self.literal(e).ok()?.as_f64()
    }

    fn num_lit(&self, e: &Expr) -> Result<f64> {
        self.number(e).ok_or_else(|| err(format!("expected a number, got {e}")))
    }

    /// A finite numeric literal or parameter, an integer kept exact.
    fn num_literal(&self, e: &Expr) -> Result<Value> {
        if self.field_of(e).is_err() {
            match self.literal(e) {
                Ok(v @ Value::Int(_)) => return Ok(v),
                Ok(Value::Float(x)) if x.is_finite() => return Ok(Value::Float(x)),
                _ => {}
            }
        }
        Err(err(format!("expected a number, got {e}")))
    }

    fn usize_lit(&self, e: &Expr) -> Result<usize> {
        let x = self.num_lit(e)?;
        if x < 0.0 || x.fract() != 0.0 {
            return Err(err("LIMIT/OFFSET must be non-negative integers"));
        }
        Ok(x as usize)
    }

    /// A literal coerced to field `f`'s type.
    fn field_value(&self, f: usize, e: &Expr) -> Result<Value> {
        self.coerce(f, self.literal(e)?)
    }

    /// A value coerced to field `f`'s type, in the form its column holds
    /// ([`crate::schema::Field::store`]).
    fn coerce(&self, f: usize, v: Value) -> Result<Value> {
        coerce(&self.schema.fields[f], v)
    }

    /// A value a condition compares field `f` with, coerced as a write's
    /// is ([`Binder::coerce`]), except that a KEYWORD is compared with
    /// text only: a write may store the number 1 as '1', but `k = 1` would
    /// match the text '1' alone, as no row of numbers does.
    fn compared(&self, f: usize, v: Value) -> Result<Value> {
        let field = &self.schema.fields[f];
        if field.ty == FieldType::Keyword && field.logical.is_none() {
            if let Value::Int(_) | Value::Float(_) = v {
                let n = v.as_f64().map_or_else(String::new, |x| x.to_string());
                return Err(err(format!("{} is a KEYWORD: compare it with text ('{n}'), not the number {n}", field.name)));
            }
        }
        self.coerce(f, v)
    }

    /// A bound of an INT field's range: a number, or for a BOOL or
    /// TIMESTAMP field a value of its type, as the integer it is stored as.
    fn int_bound(&self, f: usize, e: &Expr) -> Result<Value> {
        if self.schema.fields[f].logical.is_some() {
            self.field_value(f, e)
        } else {
            self.num_literal(e)
        }
    }

    /// A BOOL field named alone (`WHERE done`, `NOT done`).
    fn bool_field(&self, e: &Expr) -> Option<usize> {
        self.field_of(e).ok().filter(|&f| self.schema.fields[f].logical == Some(Logical::Bool))
    }

    /// A value written to field `f`: a literal or parameter of its type,
    /// `ARRAY[...]` for lists and vectors, or `DEFAULT`.
    fn write_value(&self, f: usize, e: &Expr) -> Result<Value> {
        let field = &self.schema.fields[f];
        // `DEFAULT`: the column's default, or NULL without one.
        if matches!(e, Expr::Identifier(id) if id.quote_style.is_none() && id.value.eq_ignore_ascii_case("default")) {
            return field.default_value(self.now);
        }
        if let Expr::Array(a) = e {
            return match &field.ty {
                // ARRAY[ARRAY[...], ...], or ARRAY[:v1, :v2] of vector parameters.
                FieldType::Vector { .. } if field.multi => {
                    let rows = a
                        .elem
                        .iter()
                        .map(|x| match x {
                            Expr::Array(r) => Ok(Value::Vector(r.elem.iter().map(|x| self.num_lit(x).map(|v| v as f32)).collect::<Result<Vec<f32>>>()?.into())),
                            x => Ok(Value::Vector(self.vector(x)?)),
                        })
                        .collect::<Result<Vec<Value>>>()?;
                    Ok(crate::vector::multi::normalize(field, Value::List(rows))?)
                }
                FieldType::Vector { .. } => {
                    let xs: Vec<f32> = a.elem.iter().map(|x| self.num_lit(x).map(|v| v as f32)).collect::<Result<_>>()?;
                    Ok(Value::Vector(xs.into()))
                }
                _ if field.multi => Ok(Value::List(a.elem.iter().map(|x| self.field_value(f, x)).collect::<Result<_>>()?)),
                _ => Err(err(format!("{} does not take a list", field.name))),
            };
        }
        let v = self.literal(e)?;
        Ok(match (&field.ty, v) {
            (_, Value::Null) => Value::Null,
            (FieldType::Vector { .. }, v) if field.multi => crate::vector::multi::normalize(field, v)?,
            (FieldType::Vector { .. }, _) => Value::Vector(self.vector(e)?),
            (_, Value::Str(t)) if field.media.is_some() => media_input(field, t)?,
            (_, v) if field.media.is_some() => return Err(err(format!("{} is a MEDIA field: give it a URI or a JSON object's text, not {v:?}", field.name))),
            (_, v @ Value::List(_)) if field.multi => v,
            (FieldType::Text { .. } | FieldType::Edges, v @ Value::Str(_)) => v,
            (FieldType::Sparse, v @ Value::Str(_)) => {
                crate::text::sparse::parse(&v)?;
                v
            }
            (FieldType::Keyword | FieldType::Int | FieldType::Float, _) if field.multi => Value::List(vec![self.field_value(f, e)?]),
            _ => self.field_value(f, e)?,
        })
    }

    /// `SET f = value, g = g + 1, h = expression, ...`, of the table or its
    /// `alias`; with `excluded`, as in `ON CONFLICT DO UPDATE`, also of the
    /// row the INSERT proposed (`excluded.h`).
    fn assignments(&self, assignments: &[sql::Assignment], alias: Option<&str>, excluded: bool) -> Result<Vec<(usize, Assign)>> {
        let mut sets: Vec<(usize, Assign)> = Vec::new();
        for a in assignments {
            let sql::AssignmentTarget::ColumnName(name) = &a.target else { return Err(err("SET one column at a time")) };
            // The name, not its SQL (`"order"`, quoted).
            let column = name.0.last().and_then(|p| p.as_ident()).map_or_else(|| name.to_string(), |id| id.value.clone());
            let f = self.schema.field_index(&column).ok_or_else(|| err(format!("no such column: {name}")))?;
            let fld = &self.schema.fields[f];
            if Some(f) == self.schema.primary_key {
                return Err(err("the primary key cannot be updated; delete and insert instead"));
            }
            if sets.iter().any(|(g, _)| *g == f) {
                return Err(err(format!("{} is set twice", fld.name)));
            }
            let assign = match &a.value {
                Expr::BinaryOp { left, op: op @ (BinaryOperator::Plus | BinaryOperator::Minus), right } if matches!(self.set_ref(left, alias, excluded), Some(Ok(ir::Expr::Field(g))) if g as usize == f) && self.num_literal(right).is_ok() => {
                    if !matches!(fld.ty, FieldType::Int | FieldType::Float) || fld.logical.is_some() {
                        return Err(err(format!("{} is not numeric", fld.name)));
                    }
                    if fld.multi {
                        return Err(err(format!("{} holds a list: SET it to a list", fld.name)));
                    }
                    Assign::Add(match (self.num_literal(right)?, op) {
                        (x, BinaryOperator::Plus) => x,
                        (Value::Int(x), _) => x.checked_neg().map_or(Value::Float(-(x as f64)), Value::Int),
                        (x, _) => Value::Float(-x.as_f64().unwrap_or(0.0)),
                    })
                }
                Expr::CompoundIdentifier(ids) if excluded && ids.len() == 2 && ids[0].value.eq_ignore_ascii_case("excluded") => {
                    let g = self.schema.field_index(&ids[1].value).ok_or_else(|| err(format!("no such column: {}", ids[1].value)))?;
                    let src = &self.schema.fields[g];
                    if src.ty != fld.ty || src.multi != fld.multi || src.logical != fld.logical {
                        return Err(err(format!("{} = excluded.{}: the columns have different types", fld.name, src.name)));
                    }
                    Assign::Excluded(g)
                }
                e => match self.write_value(f, e) {
                    Ok(v) => Assign::Value(v),
                    Err(not_value) => self.set_expr(e, alias, excluded, not_value)?,
                },
            };
            sets.push((f, assign));
        }
        Ok(sets)
    }

    /// What a column names in `SET`: a field of the row (`f`, or `t.f` of
    /// the table or its alias), or with `excluded` of the row an INSERT
    /// proposed (`excluded.f`); `None` when `x` is no column.
    fn set_ref(&self, x: &Expr, alias: Option<&str>, excluded: bool) -> Option<Result<ir::Expr>> {
        let ids = match x {
            Expr::Identifier(_) => return Some(self.field_of(x).map(|f| ir::Expr::Field(f as u32))),
            Expr::CompoundIdentifier(ids) if ids.len() == 2 => ids,
            Expr::CompoundIdentifier(_) => return Some(Err(err(format!("no such column: {x}")))),
            _ => return None,
        };
        let (t, name) = (&ids[0].value, &ids[1].value);
        let f = || self.schema.field_index(name).ok_or_else(|| err(format!("no such column: {x}")));
        Some(if excluded && t.eq_ignore_ascii_case("excluded") {
            f().map(|g| ir::Expr::Excluded(g as u32))
        } else if t.eq_ignore_ascii_case(&self.schema.table) || alias.is_some_and(|a| t.eq_ignore_ascii_case(a)) {
            f().map(|g| ir::Expr::Field(g as u32))
        } else {
            Err(err(format!("no such column: {x}")))
        })
    }

    /// `SET field = e` for an `e` that is not a value: an expression of the
    /// row, which the plan's checks of a SET take or refuse
    /// ([`crate::query::ir::Plan::to_write`]). `not_value` is the error of
    /// `e` as a value.
    fn set_expr(&self, e: &Expr, alias: Option<&str>, excluded: bool, not_value: Error) -> Result<Assign> {
        match super::compound::compile(e, self.params, &|x| self.set_ref(x, alias, excluded), &[]) {
            Ok(ir::Expr::Lit(_)) => Err(not_value),
            Ok(x) => Ok(Assign::Expr(x)),
            // A list, or a name that is no column: the value's error says more.
            Err(_) if matches!(e, Expr::Array(_) | Expr::Identifier(_)) => Err(not_value),
            Err(e) => Err(e),
        }
    }

    fn vector(&self, e: &Expr) -> Result<Arc<[f32]>> {
        self.vector_value(self.literal(e)?)
    }

    /// A value as a vector: a vector, a JSON array as text, or a list of
    /// numbers.
    fn vector_value(&self, v: Value) -> Result<Arc<[f32]>> {
        let v: Arc<[f32]> = match v {
            Value::Vector(v) => v,
            Value::Str(s) => {
                let v: Vec<f32> = serde_json::from_str(&s).map_err(|_| err("vector literal must be a JSON array"))?;
                v.into()
            }
            // Clients send a list of whole numbers as a list of integers.
            Value::List(xs) if !xs.is_empty() && xs.iter().all(|x| matches!(x, Value::Int(_) | Value::Float(_))) => {
                xs.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>().into()
            }
            v => return Err(err(format!("expected a vector, got {v:?}"))),
        };
        if v.iter().any(|x| !x.is_finite()) {
            return Err(err("a vector's components must be finite numbers"));
        }
        Ok(v)
    }
}

/// The BOOL or TIMESTAMP field whose values a group value or aggregate
/// holds: the field itself, or its MIN or MAX.
fn typed_source(schema: &Schema, p: &Projection) -> Option<usize> {
    match p {
        Projection::Field(f) | Projection::Agg { func: AggFn::Min | AggFn::Max, field: f } => Some(*f).filter(|&f| schema.fields[f].logical.is_some()),
        _ => None,
    }
}

/// [`Select::types`] of a projection.
pub(crate) fn projection_types(schema: &Schema, projection: &[(String, Projection)]) -> Vec<Option<Logical>> {
    projection.iter().map(|(_, p)| typed_source(schema, p).and_then(|f| schema.fields[f].logical)).collect()
}

/// Refuses the clauses of a SELECT that the binder does not handle, so that
/// none is silently ignored.
fn check_select(s: &sql::Select) -> Result<()> {
    let sql::Select {
        select_token: _,
        distinct,
        top,
        top_before_distinct: _,
        projection: _,
        exclude,
        into,
        from: _,
        lateral_views,
        prewhere,
        selection: _,
        group_by,
        cluster_by,
        distribute_by,
        sort_by,
        having: _,
        named_window,
        qualify,
        window_before_qualify: _,
        value_table_mode,
        connect_by,
        flavor,
    } = s;
    let modifiers = matches!(group_by, sql::GroupByExpr::Expressions(_, m) if !m.is_empty());
    let clause = if distinct.is_some() {
        "SELECT DISTINCT ON (it runs as a compound query, through Db::query)"
    } else if top.is_some() {
        "TOP (use LIMIT)"
    } else if into.is_some() {
        "SELECT INTO"
    } else if named_window.is_empty() && qualify.is_none() && !modifiers && exclude.is_none() && lateral_views.is_empty() && prewhere.is_none() && cluster_by.is_empty() && distribute_by.is_empty() && sort_by.is_empty() && value_table_mode.is_none() && connect_by.is_none() && *flavor == sql::SelectFlavor::Standard {
        return Ok(());
    } else if !named_window.is_empty() || qualify.is_some() {
        "WINDOW or QUALIFY"
    } else if modifiers {
        "GROUP BY ... WITH ROLLUP, CUBE or TOTALS"
    } else {
        return Err(err(format!("this form of SELECT is not supported: {s}")));
    };
    Err(err(format!("{clause} is not supported")))
}

fn contains_graph(p: &Pred) -> bool {
    match p {
        Pred::Graph { .. } => true,
        Pred::And(ps) | Pred::Or(ps) | Pred::AtLeast { items: ps, .. } => ps.iter().any(contains_graph),
        Pred::Not(q) => contains_graph(q),
        _ => false,
    }
}

/// A score's options without `norm`, which chose raw values over min-max
/// fusion in a sum of scores before ranking expressions: `+` adds raw
/// values now, and `minmax()` normalises.
fn without_norm(mut opts: Opts) -> Opts {
    opts.remove("norm");
    opts
}

/// `e` without the parentheses around it.
fn strip_nested(e: &Expr) -> &Expr {
    match e {
        Expr::Nested(x) => strip_nested(x),
        e => e,
    }
}

/// An `encode(..)` call ([`Binder::encode`]), with `literal` for its
/// model's name and `now`, and `sub` to plan its history.
pub(crate) fn encode_call(f: &sql::Function, literal: &dyn Fn(&Expr) -> Result<Value>, sub: &dyn Fn(&sql::Query) -> Result<ir::Expr>) -> Result<ir::Expr> {
    let usage = "encode('model', (SELECT item, action, time FROM ... ORDER BY time DESC LIMIT 200), now => time)";
    let FunctionArguments::List(list) = &f.args else { return Err(err(usage)) };
    let (mut pos, mut now) = (Vec::new(), None);
    for a in &list.args {
        match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => pos.push(e),
            FunctionArg::Named { name, arg: FunctionArgExpr::Expr(e), .. } if name.value.eq_ignore_ascii_case("now") => now = Some(e),
            _ => return Err(err(format!("encode takes a model and a history, and now => a time: {usage}"))),
        }
    }
    let [model, rows] = pos[..] else { return Err(err(usage)) };
    let Value::Str(model) = literal(model)? else { return Err(err(format!("encode's model is its name: {usage}"))) };
    let q = subquery_of(rows).ok_or_else(|| err(format!("encode's history is a subquery of its events, newest first: {usage}")))?;
    let ir::Expr::Values(rows) = sub(q)? else { unreachable!("a subquery's rows") };
    let now = match now.map(literal).transpose()? {
        None | Some(Value::Null) => None,
        Some(Value::Timestamp(t) | Value::Int(t)) => Some(t as f64 / 1e6),
        Some(Value::Float(t)) if t.is_finite() => Some(t / 1e6),
        Some(Value::Str(t)) => Some(crate::timestamp::parse(&t).ok_or_else(|| err(format!("encode's now is a time, not {t:?}")))? as f64 / 1e6),
        Some(v) => return Err(err(format!("encode's now is a time (a TIMESTAMP, or microseconds since the Unix epoch), not {v:?}"))),
    };
    Ok(ir::Expr::Encode { model, hash: String::new(), rows, now, open: Default::default() })
}

fn fname(f: &sql::Function) -> String {
    f.name.to_string().to_ascii_lowercase()
}

/// Whether `e` is the time of the statement: `CURRENT_TIMESTAMP`,
/// `LOCALTIMESTAMP` or `now()`.
pub(crate) fn is_now(e: &Expr) -> bool {
    let now = |n: &str| matches!(n.to_ascii_lowercase().as_str(), "current_timestamp" | "now" | "localtimestamp");
    match e {
        Expr::Function(f) => {
            now(&f.name.to_string())
                && match &f.args {
                    sql::FunctionArguments::None => true,
                    sql::FunctionArguments::List(l) => l.args.is_empty(),
                    _ => false,
                }
        }
        Expr::Identifier(id) => id.quote_style.is_none() && now(&id.value),
        _ => false,
    }
}

/// `-n` written as a minus and a number, as one integer: the only way to
/// write `-9223372036854775808` (i64::MIN), whose number alone is past an
/// INT.
pub(crate) fn negative_integer(e: &Expr) -> Option<i64> {
    match e {
        Expr::Value(v) => match &v.value {
            sql::Value::Number(s, _) => format!("-{s}").parse().ok(),
            _ => None,
        },
        _ => None,
    }
}

fn literal_usize(e: &Expr) -> Option<usize> {
    match e {
        Expr::Value(v) => match &v.value {
            sql::Value::Number(s, _) => s.parse().ok(),
            _ => None,
        },
        _ => None,
    }
}

/// The smallest integer at least `v` (a number), in 128 bits so that
/// doubles beyond the 64-bit range stay beyond it.
pub(crate) fn ceil(v: &Value) -> i128 {
    match v {
        Value::Int(x) => *x as i128,
        v => v.as_f64().unwrap_or(0.0).ceil().clamp(-1e20, 1e20) as i128,
    }
}

/// The largest integer at most `v` (a number).
pub(crate) fn floor(v: &Value) -> i128 {
    match v {
        Value::Int(x) => *x as i128,
        v => v.as_f64().unwrap_or(0.0).floor().clamp(-1e20, 1e20) as i128,
    }
}

/// INT field `f` within `[lo, hi]`, cut to the 64-bit range.
pub(crate) fn int_range(f: usize, lo: i128, hi: i128) -> Pred {
    let (lo, hi) = (lo.max(i64::MIN as i128), hi.min(i64::MAX as i128));
    if lo > hi {
        return Pred::False;
    }
    Pred::IntRange { field: f, lo: lo as i64, hi: hi as i64 }
}

pub(crate) fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    f64::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

pub(crate) fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// The values of a list-valued parameter. A JSON list of numbers arrives as
/// a vector of f32, so a vector of whole numbers that f32 holds exactly
/// counts as a list of integers.
fn as_list(v: Value) -> std::result::Result<Vec<Value>, Value> {
    match v {
        Value::List(xs) => Ok(xs),
        Value::Vector(xs) if xs.iter().all(|x| x.fract() == 0.0 && x.abs() < 16_777_216.0) => Ok(xs.iter().map(|&x| Value::Int(x as i64)).collect()),
        v => Err(v),
    }
}

/// A `LIKE` pattern a KEYWORD field's index answers.
#[derive(Debug, PartialEq)]
enum LikePrefix {
    /// No wildcard: equality with the value.
    Exact(String),
    /// Text, then `%` (one or more) at the end: a prefix.
    Prefix(String),
}

/// `pattern` as an exact value or a prefix, with `escape` (if any) making
/// the character after it literal; `_`, or `%` before the end, is refused.
fn like_prefix(pattern: &str, escape: Option<char>) -> Result<LikePrefix> {
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if Some(c) == escape => out.push(chars.next().ok_or_else(|| err("LIKE pattern ends with its escape character"))?),
            '%' if chars.all(|c| c == '%') => return Ok(LikePrefix::Prefix(out)),
            '%' | '_' => {
                return Err(err(format!(
                    "LIKE on a table's KEYWORD field takes a value or a prefix ('abc%'), not {pattern:?}: '_', or '%' before the end, is not indexed"
                )))
            }
            c => out.push(c),
        }
    }
    Ok(LikePrefix::Exact(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::beir_schema;
    use crate::schema::VectorIndexKind;

    fn schema() -> Schema {
        beir_schema(4, VectorIndexKind::Flat)
    }

    fn q() -> Params {
        Params::new().set("q", Value::Vector(vec![0.5f32; 4].into())).set("t", Value::Str("mrna vaccine".into()))
    }

    #[test]
    fn binds_filtered_bm25() {
        let s = bind(
            "SELECT id, title, score() FROM docs WHERE MATCH(body, 'vaccine -animal') AND year BETWEEN 2019 AND 2021 \
             AND category IN ('c01', 'c02') AND rating > 4 ORDER BY bm25(body, :t) DESC LIMIT 10",
            &schema(),
            &q(),
        )
        .unwrap();
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.projection.len(), 3);
        let Pred::And(ps) = &s.filter else { panic!("{:?}", s.filter) };
        assert_eq!(ps.len(), 4);
        assert!(matches!(&ps[0], Pred::Match { field: 3, query, .. } if query == "vaccine -animal"));
        assert_eq!(ps[1], Pred::IntRange { field: 4, lo: 2019, hi: 2021 });
        assert!(matches!(&ps[2], Pred::In { field: 5, values } if values.len() == 2));
        assert!(matches!(ps[3], Pred::Range { field: 6, lo, hi } if lo > 4.0 && lo < 4.0001 && hi.is_infinite()));
        assert!(matches!(&s.order, Some(OrderBy::Score { score: Score::Bm25 { field: 3, text, .. }, desc: true, .. }) if text == "mrna vaccine"));
    }

    #[test]
    fn binds_vector_and_hybrid_scores() {
        let s = bind("SELECT id FROM docs WHERE year > 2019 ORDER BY similarity(emb, :q, nprobe => 32) DESC LIMIT 5", &schema(), &q())
            .unwrap();
        assert_eq!(s.filter, Pred::IntRange { field: 4, lo: 2020, hi: i64::MAX });
        let Some(OrderBy::Score { score: Score::Similarity { opts, .. }, desc: true, .. }) = &s.order else { panic!() };
        assert_eq!(opts.get("nprobe"), Some(&Value::Int(32)));

        let s = bind("SELECT id FROM docs ORDER BY distance(emb, :q) LIMIT 5", &schema(), &q()).unwrap();
        assert!(matches!(s.order, Some(OrderBy::Score { score: Score::Similarity { .. }, desc: true, .. })));

        let s = bind("SELECT id FROM docs ORDER BY rrf(bm25(body, :t), similarity(emb, :q), k => 20) DESC LIMIT 5", &schema(), &q())
            .unwrap();
        let Some(OrderBy::Score { score: Score::Rrf { inputs, k, depth }, .. }) = &s.order else { panic!() };
        assert_eq!((inputs.len(), *k, *depth), (2, 20.0, 100));

        // A weighted sum of min-max normalised scores is the old fusion;
        // `+` alone adds raw values.
        let s = bind("SELECT id FROM docs ORDER BY 0.3 * minmax(bm25(body, :t)) + 0.7 * minmax(similarity(emb, :q)) DESC LIMIT 5", &schema(), &q())
            .unwrap();
        let Some(OrderBy::Score { score: Score::Weighted(ts), .. }) = &s.order else { panic!() };
        assert_eq!(ts.iter().map(|(w, _)| *w).collect::<Vec<_>>(), vec![0.3, 0.7]);
        let s = bind("SELECT id FROM docs ORDER BY 0.3 * bm25(body, :t) + 0.7 * similarity(emb, :q) DESC LIMIT 5", &schema(), &q()).unwrap();
        assert!(matches!(&s.order, Some(OrderBy::Score { score: Score::Arith { op: ArithOp::Add, .. }, .. })), "{:?}", s.order);
        // distance() ranks on its own, not in an expression.
        let e = bind("SELECT id FROM docs ORDER BY 1 - distance(emb, :q) DESC", &schema(), &q()).unwrap_err().to_string();
        assert!(e.contains("use similarity(...)"), "{e}");
    }

    #[test]
    fn binds_facets_and_rejects_errors() {
        let s = bind("SELECT category, COUNT(*) FROM docs WHERE MATCH(body, 'x') GROUP BY category ORDER BY 2 DESC LIMIT 5", &schema(), &q())
            .unwrap();
        assert_eq!(s.group_by, vec![5]);
        assert_eq!(s.order, Some(OrderBy::Count { desc: true }));
        assert_eq!(s.projection[1].1, Projection::Count);
        let s = bind("SELECT category, COUNT(*) FROM docs GROUP BY category ORDER BY 2 DESC, 1 LIMIT 5", &schema(), &q()).unwrap();
        assert_eq!(s.tiebreak, Some(Tiebreak::GroupValue { desc: false }));
        let s = bind("SELECT id FROM docs WHERE rating BETWEEN 2.0 AND 2.0 ORDER BY rating DESC, id LIMIT 10", &schema(), &q()).unwrap();
        assert_eq!(s.tiebreak, Some(Tiebreak::Key { desc: false }));
        assert!(matches!(s.order, Some(OrderBy::Score { score: Score::Column { field: 6 }, desc: true, .. })));
        // A product of scores (a boost), not a weighted sum.
        let s = bind("SELECT id FROM docs ORDER BY bm25(body, :t) * similarity(emb, :q) DESC", &schema(), &q()).unwrap();
        assert!(matches!(&s.order, Some(OrderBy::Score { score: Score::Product(f), desc: true, .. }) if f.len() == 2), "{:?}", s.order);
        // With a constant among its factors, it is an expression.
        let s = bind("SELECT id FROM docs ORDER BY bm25(body, :t) * 2 * similarity(emb, :q) DESC", &schema(), &q()).unwrap();
        assert!(matches!(&s.order, Some(OrderBy::Score { score: Score::Arith { op: ArithOp::Mul, .. }, .. })), "{:?}", s.order);

        for bad in [
            "SELECT id FROM other",
            "SELECT id FROM docs WHERE MATCH(year, 'x')",
            "SELECT id FROM docs WHERE category > 'c01'",
            "SELECT id FROM docs ORDER BY similarity(emb, :missing)",
            "SELECT id FROM docs ORDER BY bm25(body, :t) >= 2",
            "SELECT id FROM docs ORDER BY ln(similarity(emb, :q), 2, 3)",
            "SELECT id FROM docs ORDER BY year, rating",
            "DELETE FROM docs",
        ] {
            assert!(bind(bad, &schema(), &q()).is_err(), "{bad}");
        }
    }

    #[test]
    fn star_and_strict_float_bounds() {
        let s = bind("SELECT * FROM docs WHERE rating < 2.5 OR NOT category = 'c00' LIMIT 3 OFFSET 1", &schema(), &q()).unwrap();
        let names: Vec<&str> = s.projection.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["id", "title", "text", "year", "category", "rating", "topic"]);
        assert_eq!((s.limit, s.offset), (Some(3), 1));
        let Pred::Or(ps) = &s.filter else { panic!() };
        assert!(matches!(ps[0], Pred::Range { lo, hi, .. } if lo.is_infinite() && hi < 2.5 && hi > 2.4999));
        // A negated comparison also needs the value (NULL != 'c00' is not true).
        assert!(matches!(&ps[1], Pred::And(qs) if matches!(qs[..], [Pred::Present { field: 5 }, Pred::Not(_)])), "{:?}", ps[1]);
    }
}
