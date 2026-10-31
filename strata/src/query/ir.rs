//! The query plan: one typed, versioned and serialisable form of a
//! statement (DESIGN.md §6.2). The SQL binder makes it, and other
//! front-ends (a Python builder, a coordinator handing a query to its
//! shards) are to make it too, so that nothing writes SQL text to pass a
//! query on.
//!
//! A plan is a tree of [`Node`]s over [`Expr`]essions. Its leaf is a
//! [`Search`]: the rows of one table that a filter holds for, ranked, which
//! is what the executor answers ([`crate::query::exec`]). Above it,
//! [`Node::Aggregate`] groups rows, [`Node::Filter`] keeps groups,
//! [`Node::Sort`] orders them, [`Node::Limit`] cuts the rows and
//! [`Node::Project`] names the columns of the answer. Writes are nodes too:
//! [`Node::Insert`], and [`Node::Update`] and [`Node::Delete`] over the rows
//! a Search finds.
//!
//! * **Fields** are named by id, the field's position in its table's
//!   schema: tables have no ALTER, so a field keeps its position.
//! * **Values** are typed literals (`{"int": 5}`, `{"float": 5.0}`,
//!   `{"text": "5"}`, `{"timestamp": ...}`), so the INT 5, the FLOAT 5 and
//!   the text '5' stay apart, and a BOOL or TIMESTAMP field's values are
//!   booleans and times rather than the integers stored.
//! * **Versions.** A plan carries [`VERSION`] and the features it uses. A
//!   reader refuses a newer version, or a feature it does not know, by
//!   name ([`Plan::from_json`]), as a database refuses a format 3 feature
//!   ([`crate::storage::manifest`]); a later node or expression is a new
//!   feature, not a new version.
//!
//! [`Plan::of_select`] and [`Plan::of_write`] make the plan of a bound
//! statement; [`Plan::to_select`] and [`Plan::to_write`] give back the
//! statement the executor runs, for the shapes it runs:
//! `Project [Limit] Search` and `Project [Limit] [Sort] [Filter] Aggregate
//! Search`. Plans travel as JSON and read as a tree in `EXPLAIN`
//! ([`Plan::explain`]). Every SQL statement runs as its plan, and
//! [`crate::db::Db::run`] runs a plan made elsewhere.
//!
//! A compound query (joins, `WITH`, subqueries, `SELECT DISTINCT`,
//! `DISTINCT ON`) is a plan of operators over those shapes, its leaves
//! ([`crate::query::compound`] makes it): [`Node::Join`], [`Node::With`]
//! and [`Node::Ref`], [`Node::Distinct`], [`Node::DistinctOn`]
//! ([`DISTINCT_ON`]), [`Node::KeyOrder`], the nodes above over rows in
//! memory, and subqueries as expressions ([`Expr::Values`], [`Expr::Keys`],
//! [`Expr::Exists`], [`Expr::Scalar`]). [`crate::query::ops`] runs it: each
//! leaf on the executor, the rest in memory. Such a plan needs the feature
//! [`OPERATORS`].

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::query::logical::{AggFn, Decay, Direction, Expand, Opts};
use crate::query::cross_encoder::CrossEncoder;
use crate::query::model::Opened;
use crate::query::sequence::SequenceModel;
use crate::value::Value;

pub mod eval;
mod explain;
pub(crate) mod lower;
mod raise;

/// The plan version this version of Strata writes and reads.
pub const VERSION: u32 = 1;

/// The plan features this version of Strata reads.
pub const FEATURES: [&str; 22] = [
    OPERATORS, SEQ_NO, RANK_EXPRESSIONS, SEMANTIC_ID, RANK_PHASES, RANK_FEATURES, MAXSIM, RANK_PROFILES, RANK_MODELS, DISTINCT_ON, EDGE_TABLE, RANKED, TEXT_OPERATORS, PREFIX, CORRELATED, PREFIX_EDITS, MIN_MATCH, AT_LEAST, GUARD, PER_CAP, ENCODE,
    SCORE_NORMS,
];

/// The feature of a plan with operators over rows in memory: a join,
/// `WITH`, a subquery, `DISTINCT`, or a node over another's rows than a
/// Search's ([module docs](self)).
pub const OPERATORS: &str = "operators";

/// The feature of a plan with a column of rows' sequence numbers
/// ([`Expr::SeqNo`]).
pub const SEQ_NO: &str = "seq_no";

/// The feature of a plan whose Search ranks by an expression of scores,
/// row values and numbers ([`Expr::Arith`], [`Expr::Call`], [`Expr::Case`],
/// [`Expr::Norm`] and literals around score leaves), rather than by a score
/// or one of the fused shapes ([`Expr::Rrf`], [`Expr::Weighted`],
/// [`Expr::Product`]).
pub const RANK_EXPRESSIONS: &str = "rank_expressions";

/// The feature of a plan with a set function that came after `minmax` and
/// `rank`: `tmm(x)` or `zscore(x)` ([`NormKind::Tmm`], [`NormKind::ZScore`]),
/// which a Strata from before them would read as `minmax(x)`.
pub const SCORE_NORMS: &str = "score_norms";

/// The feature of a plan that scores with several query vectors
/// ([`Expr::MaxSim`]), takes a dot product of vectors ([`Func::Dot`]), or
/// seeds a walk with weights ([`Graph::weights`]).
pub const MAXSIM: &str = "maxsim";
/// The feature of a plan that cuts semantic IDs ([`Func::SidAt`]).
pub const SEMANTIC_ID: &str = "semantic_id";

/// The feature of a plan with a walk over an edge table ([`Graph::edges`]).
pub const EDGE_TABLE: &str = "edge_table";

/// The feature of a plan whose Search reranks its top rows in phases
/// ([`Search::phases`]).
pub const RANK_PHASES: &str = "rank_phases";

/// The feature of a plan with a column of a Search that is a ranking
/// expression scoring each row by itself (`bm25(title, :q)`, what SQL's
/// `features()` lists), rather than a field, `score()`, `hops()` or
/// `seq_no()`.
pub const RANK_FEATURES: &str = "rank_features";

/// The feature of a plan whose Search ranks as a rank profile does
/// ([`Search::profile`]).
pub const RANK_PROFILES: &str = "rank_profiles";

/// The feature of a plan that ranks with a model ([`Expr::Gbdt`],
/// [`Expr::CrossEncoder`]).
pub const RANK_MODELS: &str = "rank_models";

/// The feature of a plan that keeps the first row of each group
/// ([`Node::DistinctOn`]).
pub const DISTINCT_ON: &str = "distinct_on";
/// The feature of a plan that ranks by a ranking given as primary keys
/// ([`Expr::Ranked`], `ranked(:keys)`).
pub const RANKED: &str = "ranked";

/// The feature of a plan whose text a Strata from before text operators
/// reads as other words: a `bm25` with a sign or weight (`+word`, `-word`,
/// `word^2`; [`crate::text::query::has_operators`]), or a `MATCH` with a
/// weight.
pub const TEXT_OPERATORS: &str = "text_operators";

/// The feature of a plan that tests whether text starts with a prefix
/// ([`Expr::Prefix`]).
pub const PREFIX: &str = "prefix";

/// The feature of a plan with a correlated subquery ([`Expr::Outer`]), or
/// the values one took for a Search's rows ([`Expr::ByKey`]).
pub const CORRELATED: &str = "correlated";

/// The feature of a plan with a prefix that allows typos ([`Expr::Prefix`]
/// with `edits`), which a Strata from before them would read as exact.
pub const PREFIX_EDITS: &str = "prefix_edits";

/// The feature of a plan whose text a row must match some of: a `MATCH`
/// group `(a b c)@2`, or `bm25`'s `min` option. A Strata from before them
/// reads `)@2` as a word and ignores `min`, and so returns other rows.
pub const MIN_MATCH: &str = "min_match";

/// The feature of a plan that counts conditions: `AT_LEAST(n, ..)`
/// ([`Expr::AtLeast`]) or `count_true(..)` ([`Func::CountTrue`]).
pub const AT_LEAST: &str = "at_least";

/// The feature of a plan with a guarded Search ([`Search::guard`]:
/// `min_rows`, `max_rows`).
pub const GUARD: &str = "guard";

/// The feature of a plan that keeps more than the first row of each
/// group ([`Node::DistinctOn`] with `per` above 1: `MAX k PER field`). A
/// Strata from before it would keep one.
pub const PER_CAP: &str = "per_cap";

/// The feature of a plan that turns a history into a vector with a
/// sequence model ([`Expr::Encode`]).
pub const ENCODE: &str = "encode";

/// A statement as a tree of operators ([module docs](self)).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub version: u32,
    /// Features a reader must have to run the plan.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
    pub root: Node,
}

/// An operator: rows in (from `input`, or a table), rows out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum Node {
    /// The rows of one table (all its fields, and the score they are
    /// ranked by) that its filter holds for, best first.
    Search(Search),
    /// One row per group of the input's rows with equal `keys` (a row per
    /// value of a multi-valued key), or one row in all without keys: the
    /// keys, then the aggregates. Expressions above refer to those columns
    /// with [`Expr::Output`].
    Aggregate { input: Box<Node>, keys: Vec<Expr>, aggs: Vec<Agg> },
    /// The input's rows for which `cond` is true.
    Filter { input: Box<Node>, cond: Expr },
    /// The input's rows by `keys`, the first key first.
    Sort { input: Box<Node>, keys: Vec<SortKey> },
    /// The input's rows from `offset`, at most `limit` of them.
    Limit {
        input: Box<Node>,
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u64>,
    },
    /// The answer's columns, by name.
    Project { input: Box<Node>, columns: Vec<Column> },
    /// Rows written to `table`, each a value per field in field order.
    Insert {
        table: String,
        rows: Vec<Vec<Expr>>,
        #[serde(default, skip_serializing_if = "OnConflict::is_fail")]
        on_conflict: OnConflict,
    },
    /// The rows of `input` (a Search of `table`, with its fields) get
    /// `sets`.
    Update { table: String, input: Box<Node>, sets: Vec<Set> },
    /// The rows of `input` (a Search of `table`) are deleted.
    Delete { table: String, input: Box<Node> },
    /// Each row of `left` beside each row of `right` whose `keys` match it
    /// and for which `on` holds: the left's columns, then the right's. A
    /// left join keeps a left row that matches none, with NULLs on the
    /// right. Rows keep the left's order.
    Join {
        left: Box<Node>,
        right: Box<Node>,
        #[serde(default, skip_serializing_if = "JoinKind::is_inner")]
        kind: JoinKind,
        keys: Vec<JoinKey>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        on: Option<Expr>,
    },
    /// `input`, which reads the rows of named plans with [`Node::Ref`]:
    /// each runs once, in order, and may read those before it.
    With { ctes: Vec<Cte>, input: Box<Node> },
    /// The rows of a plan a [`Node::With`] names.
    Ref { name: String },
    /// The input's rows, each once.
    Distinct { input: Box<Node> },
    /// The first `per` of the input's rows with equal `keys`, in the
    /// input's order (`DISTINCT ON`, and `MAX k PER field` for `per` k):
    /// the other rows of a group are dropped, the rows kept stay in their
    /// order. Keys are equal when their values are, NULL equal to NULL and
    /// a list to the same list. Needs [`DISTINCT_ON`], and [`PER_CAP`]
    /// with `per` above 1.
    DistinctOn {
        input: Box<Node>,
        keys: Vec<Expr>,
        #[serde(default = "one", skip_serializing_if = "is_one")]
        per: u64,
    },
    /// The input's rows (rows of one table) in the order of the rows of
    /// `by` with the same primary key.
    KeyOrder { input: Box<Node>, by: Box<Node> },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinKind {
    #[default]
    Inner,
    Left,
}

impl JoinKind {
    fn is_inner(&self) -> bool {
        *self == JoinKind::Inner
    }
}

/// A key of a [`Node::Join`]: a value of the left row equals a value of
/// the right row. Numbers equal by value whatever their type; a list
/// matches by any of its values, and an EDGES field (`links`) by the
/// targets of its links.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JoinKey {
    /// Over the left rows.
    pub left: Expr,
    /// A column of the right rows ([`Expr::Output`]).
    pub right: Expr,
    #[serde(default, skip_serializing_if = "is_false")]
    pub left_links: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub right_links: bool,
}

/// A named plan of a [`Node::With`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cte {
    pub name: String,
    pub input: Node,
}

/// The rows of one table that `filter` holds for, ordered by `order`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Search {
    pub table: String,
    #[serde(default = "Expr::always", skip_serializing_if = "Expr::is_always")]
    pub filter: Expr,
    /// A score to rank by (or a column), then possibly the primary key for
    /// ties.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<SortKey>,
    /// Rerank phases after `order`, in turn (feature [`RANK_PHASES`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phases: Vec<Phase>,
    /// The rank profile `order` and `phases` were written out from
    /// (feature [`RANK_PROFILES`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileRef>,
    /// Guarded relaxation (feature [`GUARD`]): the rows choose how many of
    /// an `AT_LEAST`'s conditions, or of a `MATCH`'s or a `bm25`'s words,
    /// a row needs ([`crate::query::logical::Guard`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<Box<Guard>>,
    /// That a [`Node::DistinctOn`] above keeps at most `k` rows of each
    /// value of a field, so the Search may return only those and stop
    /// reading once it has them ([`PerCap`]). A hint, not a condition: a
    /// reader that does not cap (one from before it, or a path that has no
    /// capped scan) returns the rows uncapped, and the node above caps them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per: Option<PerCap>,
}

/// The cap of a [`Search::per`]: at most `k` rows of each value of the
/// single-valued `field`, the first in the Search's order, NULL being one
/// value; rows tied with a value's `k`-th, or with the LIMIT's last row,
/// may be kept as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerCap {
    pub field: u32,
    pub k: u64,
}

/// A [`Search`]'s guard: what it relaxes, the loosest level and the row
/// limits that choose the level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Guard {
    pub site: GuardSite,
    /// The loosest level: a count (`2`) or a share of the items (`"75%"`).
    pub need: crate::text::query::Need,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_rows: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows: Option<u64>,
}

/// What a [`Guard`] relaxes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardSite {
    /// `AT_LEAST(level, conds...)`, joined to the Search's filter.
    AtLeast { conds: Vec<Expr> },
    /// `MATCH(field, '(text)@level')`, joined to the Search's filter.
    Match {
        field: u32,
        text: String,
        #[serde(default, skip_serializing_if = "Expand::is_none")]
        expand: Expand,
    },
    /// The `min` of the Search's `bm25(field, text, min => ...)`.
    Bm25 { field: u32, text: String },
}

/// A rank profile a Search ranks as ([`crate::query::profile`]): its name
/// and the hash of its definition
/// ([`crate::storage::manifest::RankProfile::hash`]). A database whose
/// table has no profile of that name, or one with another hash, refuses
/// the plan rather than rank otherwise than the plan was made to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileRef {
    pub name: String,
    pub hash: String,
}

/// A rerank phase of a [`Search`]: the top `count` rows of the phase
/// before it (the first phase is the Search's `order`) ranked again by
/// `expr`, best first, rows tied on it in the order they had; only those
/// rows go on. A phase that is not `global` is row-local, and each shard
/// scores its own rows; a global one runs once over every row it reranks
/// (on the coordinator over shards), so it may hold set functions
/// (`minmax`, `rank`, `rrf`). A phase after a global one is global.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub count: u64,
    pub expr: Expr,
    #[serde(default, skip_serializing_if = "is_false")]
    pub global: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SortKey {
    pub expr: Expr,
    #[serde(default, skip_serializing_if = "is_false")]
    pub desc: bool,
    /// Rows without a value first (`NULLS FIRST`); they come last
    /// otherwise, either way.
    #[serde(default, skip_serializing_if = "is_false")]
    pub nulls_first: bool,
}

/// A column of the answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub expr: Expr,
}

/// An aggregate of a group's rows: `COUNT(*)` without an argument.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Agg {
    pub func: AggFn,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg: Option<Expr>,
}

/// `SET field = value`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Set {
    pub field: u32,
    pub value: Expr,
}

/// What an insert does with a row whose primary key is live.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnConflict {
    /// The statement fails.
    #[default]
    Fail,
    /// The new row replaces the live one.
    Replace,
    /// The live row stays and the new one is skipped.
    Skip,
    /// The live row gets these fields set; [`Expr::Excluded`] names the
    /// fields of the row the insert proposed.
    Update(Vec<Set>),
}

impl OnConflict {
    fn is_fail(&self) -> bool {
        *self == OnConflict::Fail
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    /// Always a FLOAT; NULL for a zero divisor.
    Div,
    /// The remainder; NULL for a zero divisor.
    Mod,
}

/// A scalar function of values. The functions from [`Func::Ln`] on are
/// numeric, came with [`RANK_EXPRESSIONS`], and are NULL where the result
/// is not a finite number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Func {
    Lower,
    Upper,
    /// Characters of a text.
    Length,
    Abs,
    /// To the nearest integer, or to `args[1]` decimals.
    Round,
    /// The first argument that is not NULL (nor a list without values).
    Coalesce,
    /// The natural logarithm.
    Ln,
    /// The base-10 logarithm of `args[0]`, or with two arguments the
    /// logarithm of `args[1]` to base `args[0]` (as PostgreSQL's `log`).
    Log,
    Exp,
    Sqrt,
    /// `args[0]` to the power `args[1]`.
    Pow,
    /// The smallest argument that is not NULL; NULL if all are.
    Least,
    /// The largest argument that is not NULL; NULL if all are.
    Greatest,
    /// `1 / (1 + e^-x)`.
    Sigmoid,
    /// `args[0]` within `args[1]` and `args[2]`.
    Clamp,
    /// The dot product of two vectors of one length (feature [`MAXSIM`]);
    /// not a ranking expression's function: `similarity` is.
    Dot,
    /// The prefix of `args[1]` levels of a semantic ID `args[0]`
    /// ([`crate::sid::sid_at`]); needs [`SEMANTIC_ID`].
    SidAt,
    /// How many of its arguments, conditions, are TRUE (UNKNOWN counts
    /// as not); needs [`AT_LEAST`].
    CountTrue,
}

impl Func {
    /// The function's name in SQL.
    pub fn name(self) -> &'static str {
        match self {
            Func::Lower => "lower",
            Func::Upper => "upper",
            Func::Length => "length",
            Func::Abs => "abs",
            Func::Round => "round",
            Func::Coalesce => "coalesce",
            Func::Ln => "ln",
            Func::Log => "log",
            Func::Exp => "exp",
            Func::Sqrt => "sqrt",
            Func::Pow => "pow",
            Func::Least => "least",
            Func::Greatest => "greatest",
            Func::Sigmoid => "sigmoid",
            Func::Clamp => "clamp",
            Func::Dot => "dot",
            Func::SidAt => "sid_at",
            Func::CountTrue => "count_true",
        }
    }

    /// The numeric function named `name` (as SQL writes it), which a
    /// ranking expression can use.
    pub fn numeric(name: &str) -> Option<Func> {
        Some(match name {
            "abs" => Func::Abs,
            "round" => Func::Round,
            "coalesce" => Func::Coalesce,
            "ln" => Func::Ln,
            "log" => Func::Log,
            "exp" => Func::Exp,
            "sqrt" => Func::Sqrt,
            "pow" | "power" => Func::Pow,
            "least" => Func::Least,
            "greatest" => Func::Greatest,
            "sigmoid" => Func::Sigmoid,
            "clamp" => Func::Clamp,
            _ => return None,
        })
    }

    /// Whether the function takes `n` arguments, or else how many it takes.
    pub fn arity(self, n: usize) -> std::result::Result<(), &'static str> {
        let ok = match self {
            Func::Lower | Func::Upper | Func::Length | Func::Abs | Func::Ln | Func::Exp | Func::Sqrt | Func::Sigmoid => n == 1,
            Func::Round | Func::Log => n == 1 || n == 2,
            Func::Pow | Func::Dot | Func::SidAt => n == 2,
            Func::Clamp => n == 3,
            Func::Coalesce | Func::Least | Func::Greatest | Func::CountTrue => n >= 1,
        };
        match (ok, self) {
            (true, _) => Ok(()),
            (_, Func::Round | Func::Log) => Err("one or two arguments"),
            (_, Func::Pow | Func::Dot) => Err("two arguments"),
            (_, Func::SidAt) => Err("two arguments: a semantic ID and a number of levels"),
            (_, Func::Clamp) => Err("three arguments: a value, its least and its greatest"),
            (_, Func::Coalesce | Func::Least | Func::Greatest | Func::CountTrue) => Err("at least one argument"),
            _ => Err("one argument"),
        }
    }

    /// The numeric function of `args`, NaN standing for NULL: NaN where the
    /// result has no finite value (`ln(0)`, `sqrt(-1)`, an overflow).
    /// [`Func::Lower`], [`Func::Upper`], [`Func::Length`] and
    /// [`Func::SidAt`] take text, and are NaN here.
    pub fn eval_f64(self, args: &[f64]) -> f64 {
        // NULL in, NULL out, except where a function skips it: `pow(1,
        // NULL)` and `clamp(NULL, 0, 1)` are not 1 and 0, as f64's pow and
        // max would make them.
        if !matches!(self, Func::Coalesce | Func::Least | Func::Greatest | Func::CountTrue) && args.iter().any(|a| a.is_nan()) {
            return f64::NAN;
        }
        let x = args.first().copied().unwrap_or(f64::NAN);
        let r = match self {
            Func::Lower | Func::Upper | Func::Length | Func::Dot | Func::SidAt => f64::NAN,
            Func::Abs => x.abs(),
            Func::Round => {
                let d = args.get(1).map_or(0.0, |d| d.clamp(-30.0, 30.0).trunc());
                let m = 10f64.powi(d as i32);
                (x * m).round() / m
            }
            Func::Coalesce => args.iter().copied().find(|v| !v.is_nan()).unwrap_or(f64::NAN),
            Func::Ln if x > 0.0 => x.ln(),
            Func::Log if args.len() == 1 && x > 0.0 => x.log10(),
            Func::Log => match args.get(1) {
                Some(&y) if x > 0.0 && x != 1.0 && y > 0.0 => y.ln() / x.ln(),
                _ => f64::NAN,
            },
            Func::Ln => f64::NAN,
            Func::Exp => x.exp(),
            Func::Sqrt if x >= 0.0 => x.sqrt(),
            Func::Sqrt => f64::NAN,
            Func::Pow => x.powf(args.get(1).copied().unwrap_or(f64::NAN)),
            Func::Least => args.iter().copied().filter(|v| !v.is_nan()).reduce(f64::min).unwrap_or(f64::NAN),
            Func::Greatest => args.iter().copied().filter(|v| !v.is_nan()).reduce(f64::max).unwrap_or(f64::NAN),
            Func::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            // Conditions as 1 and 0.
            Func::CountTrue => args.iter().filter(|&&v| v != 0.0 && !v.is_nan()).count() as f64,
            Func::Clamp => match (args.get(1), args.get(2)) {
                (Some(&lo), Some(&hi)) => x.max(lo).min(hi),
                _ => f64::NAN,
            },
        };
        if r.is_finite() {
            r
        } else {
            f64::NAN
        }
    }
}

/// A set function of [`Expr::Norm`]: a value seen against every candidate
/// a ranking scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    /// `minmax(x)`: `(x - min) / (max - min)` over the candidates, 1 where
    /// they all have one value, and 0 where `x` is NULL.
    MinMax,
    /// `rank(x)`: 1 for the largest `x`, plus the number of candidates with
    /// a larger value; rows where `x` is NULL rank after every other.
    Rank,
    /// `tmm(x)`, theoretical min-max (Bruch, Gai and Ingber, TOIS 2023):
    /// `(x - lo) / (max - lo)`, where `lo` is the least value `x` can take
    /// ([`Expr::Norm`]'s `lo`) and `max` the largest over the candidates;
    /// 0 where `x` is NULL, and 0 for every row where no candidate is above
    /// `lo`. Unlike `minmax`, a row's value does not depend on how deep the
    /// candidates go: the largest is the best row, at any depth.
    Tmm,
    /// `zscore(x)`: `(x - mean) / sd` over the candidates (the population
    /// deviation), 0 where they all have one value; a row where `x` is NULL
    /// takes the least value of the candidates.
    ZScore,
}

impl NormKind {
    /// The set function's name in SQL.
    pub fn name(self) -> &'static str {
        match self {
            NormKind::MinMax => "minmax",
            NormKind::Rank => "rank",
            NormKind::Tmm => "tmm",
            NormKind::ZScore => "zscore",
        }
    }
}

/// A branch of [`Expr::Case`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct When {
    pub when: Expr,
    pub then: Expr,
}

/// An expression over a row. Conditions follow SQL's three-valued logic:
/// a comparison with NULL is neither true nor false, and so is its
/// negation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    /// A field of the row, by id.
    Field(u32),
    /// The score the Search ranked the row by (`score()`).
    Score,
    /// The row's hops from the seeds of the Search's first `REACHABLE` or
    /// `PATH` (`hops()`).
    Hops,
    /// The sequence number of the write that last changed the row
    /// (`seq_no()`), a column of a Search in a database that numbers its
    /// writes. Feature [`SEQ_NO`].
    SeqNo,
    /// Column `n` of the input's rows: an Aggregate's keys, then its
    /// aggregates; a Project's columns; a Join's left columns, then its
    /// right ones; a [`Node::Ref`]'s.
    Output(u32),
    /// A field of the row an insert proposed, in [`OnConflict::Update`]
    /// (`excluded.f`).
    Excluded(u32),
    Lit(#[serde(with = "lit")] Value),
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    /// Whether at least `need` of `conds` are TRUE (`AT_LEAST(n, c1,
    /// ..)`): FALSE where fewer are TRUE and UNKNOWN, UNKNOWN where the
    /// UNKNOWN ones decide it (feature [`AT_LEAST`]).
    AtLeast { need: u32, conds: Vec<Expr> },
    /// The expression has no value (a multi-valued field: holds none).
    IsNull(Box<Expr>),
    Cmp { op: CmpOp, left: Box<Expr>, right: Box<Expr> },
    /// `lo <= expr <= hi`; a missing bound is open.
    Between {
        expr: Box<Expr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lo: Option<Box<Expr>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hi: Option<Box<Expr>>,
    },
    /// `expr` equals one of `list` (a multi-valued field: one of its
    /// values does).
    In { expr: Box<Expr>, list: Vec<Expr> },
    /// `left + right`, `-`, `*`, `/` or `%`: integers exact, and a FLOAT
    /// where an integer result would overflow.
    Arith { op: ArithOp, left: Box<Expr>, right: Box<Expr> },
    /// `-expr`.
    Neg(Box<Expr>),
    /// The values as text, one after another (`||`); NULL adds nothing.
    Concat(Vec<Expr>),
    /// `expr LIKE pattern` (`%` any run of characters, `_` one), or
    /// `ILIKE` ignoring case; NULL unless both are text.
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        #[serde(default, skip_serializing_if = "is_false")]
        case_insensitive: bool,
    },
    /// Whether text `expr` starts with text `prefix` (`PREFIX(x, p)`, and a
    /// table's `kw LIKE 'p%'`): NULL unless both are text. A Search's
    /// filter reads it as a run of a KEYWORD field's dictionary (feature
    /// [`PREFIX`]). With `edits`, whether it starts with a prefix within
    /// that many edits of `prefix` whose first letter is its own
    /// (`PREFIX(x, p, fuzzy => 1)`, feature [`PREFIX_EDITS`]).
    Prefix {
        expr: Box<Expr>,
        prefix: Box<Expr>,
        #[serde(default, skip_serializing_if = "is_zero_u8")]
        edits: u8,
    },
    /// `CASE [operand] WHEN .. THEN .. [ELSE ..] END`: the first branch
    /// whose `when` is TRUE (or equals the operand), else `otherwise`, else
    /// NULL.
    Case {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operand: Option<Box<Expr>>,
        whens: Vec<When>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        otherwise: Option<Box<Expr>>,
    },
    /// A scalar function.
    Call { func: Func, args: Vec<Expr> },
    /// `MATCH(field, query)`: a TEXT field's full-text query.
    Match {
        field: u32,
        query: String,
        #[serde(default, skip_serializing_if = "Expand::is_none")]
        expand: Expand,
    },
    /// The rows a traversal reaches: `REACHABLE` or `PATH`.
    Graph(Box<Graph>),
    /// BM25 of a TEXT field for `text`.
    Bm25 {
        field: u32,
        text: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "lits")]
        opts: Opts,
    },
    /// Similarity of a VECTOR field with `vector`.
    Similarity {
        field: u32,
        vector: Arc<[f32]>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "lits")]
        opts: Opts,
    },
    /// The best similarity of a VECTOR field with any of `vectors`: a list
    /// literal of vectors, or [`Expr::Values`] of a subquery, which runs
    /// first (feature [`MAXSIM`]).
    #[serde(rename = "maxsim")]
    MaxSim {
        field: u32,
        vectors: Box<Expr>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "lits")]
        opts: Opts,
    },
    /// Similarity of a SPARSE field with `terms`.
    Sparse {
        field: u32,
        terms: Vec<(String, f32)>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "lits")]
        opts: Opts,
    },
    /// A numeric field's closeness to an origin.
    Decay {
        field: u32,
        decay: Decay,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "lits")]
        opts: Opts,
    },
    /// Reciprocal rank fusion of the rankings of `inputs`.
    Rrf { inputs: Vec<Expr>, k: f64, depth: u64 },
    /// `Σ weight · expr` of scores.
    Weighted(Vec<(f64, Expr)>),
    /// The product of scores.
    Product(Vec<Expr>),
    /// A set function of a ranking expression, over the rows it ranks
    /// (feature [`RANK_EXPRESSIONS`]; [`SCORE_NORMS`] for `tmm` and
    /// `zscore`). `lo` is the least value `input` can take, which `tmm`
    /// reads: absent in a plan written by hand, where the reader takes the
    /// input's own (0 for `bm25`, -1 for `similarity`, ...).
    Norm {
        kind: NormKind,
        input: Box<Expr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lo: Option<f64>,
    },
    /// A ranking model's score of a row from its features `args`, ranking
    /// expressions of the row ([`crate::query::model`], feature
    /// [`RANK_MODELS`]). `model` names it in the models directory, and
    /// `hash` is the hash of its file that the plan was made with: empty
    /// until a database or coordinator opens it, which then runs the plan
    /// only with a model of that hash.
    Gbdt {
        model: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        hash: String,
        args: Vec<Expr>,
        #[serde(skip)]
        open: Opened,
    },
    /// A cross-encoder's score of the row's text for `query`: `fields`'
    /// values joined by a space ([`crate::query::cross_encoder`], feature
    /// [`RANK_MODELS`]); `model` and `hash` as a [`Expr::Gbdt`]'s. Only a
    /// global rerank phase ranks by one, on the database or coordinator:
    /// a shard leaves it to its coordinator, which fetches the text.
    CrossEncoder {
        model: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        hash: String,
        fields: Vec<u32>,
        query: String,
        #[serde(skip)]
        open: Opened<CrossEncoder>,
    },
    /// Personalized PageRank from a traversal's seeds.
    Ppr(Box<Graph>),
    /// A ranking given as primary keys, best first (`ranked(:keys)`):
    /// `-(position)` from 1 for a listed row, NULL for others. Feature
    /// [`RANKED`].
    Ranked {
        #[serde(with = "lit_list")]
        keys: Vec<Value>,
    },
    /// The rows of a plan (graph seeds from a query of the table, which
    /// the executor runs).
    Subquery(Box<Node>),
    /// The values of the one column of a plan's rows, each value of a list
    /// and NULL none, in an [`Expr::In`] list or as graph seeds.
    Values(Box<Node>),
    /// The primary keys of a plan's rows (rows of one table), in an
    /// [`Expr::In`] list or as graph seeds.
    Keys(Box<Node>),
    /// Whether a plan has a row.
    Exists(Box<Node>),
    /// The value of a plan's one row and one column; NULL without a row.
    Scalar(Box<Node>),
    /// In the plan of a correlated subquery, a column of the row the
    /// query around it is on: a field of a Search's row by id, or a column
    /// of rows in memory. An [`Expr::Scalar`] whose plan names the row so
    /// runs once for each distinct value of what it names among the rows
    /// it is computed for, those values in their place (feature
    /// [`CORRELATED`]).
    Outer(u32),
    /// A number for each row of a Search by its primary key, NULL for a
    /// row it does not list: the values a correlated subquery took for the
    /// Search's rows, once it ran (feature [`CORRELATED`]).
    ByKey(Vec<KeyValue>),
    /// A history's vector by a sequence model
    /// ([`crate::query::sequence`], feature [`ENCODE`]): `rows` are the
    /// history's events, newest first, each the item's key, then its
    /// action and time when the plan has those columns (a TIMESTAMP, or
    /// microseconds); `now` is the request's time in seconds since the
    /// Unix epoch (the clock's when absent). It runs before the rows it ranks, as a subquery does, and
    /// stands for a vector of the item field the model names. `model` and
    /// `hash` are as [`Expr::Gbdt`]'s.
    Encode {
        model: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        hash: String,
        rows: Box<Node>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        now: Option<f64>,
        #[serde(skip)]
        open: Opened<SequenceModel>,
    },
}

/// A row's primary key and a number for it ([`Expr::ByKey`]); `None` is
/// NULL.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyValue {
    #[serde(with = "lit")]
    pub key: Value,
    pub value: Option<f64>,
}

/// A traversal over an EDGES field, over an edge table, or over a VECTOR
/// field whose `k` most similar rows count as a row's neighbours.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// The EDGES or VECTOR field; with `edges`, the primary key.
    pub field: u32,
    /// The edge table walked instead of a field, whose rows link this
    /// table's (feature [`EDGE_TABLE`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<String>,
    /// Primary keys (a list literal), or a [`Expr::Subquery`].
    pub seeds: Expr,
    /// The weight of each of the keys `seeds` lists, which PageRank
    /// restarts at in proportion (`weighted => 1`); none, all 1 (feature
    /// [`MAXSIM`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<f64>,
    pub kind: GraphKind,
    pub dir: Direction,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// Traversals pass only through rows this holds for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub through: Option<Expr>,
    pub max_nodes: u64,
    pub k: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_sim: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphKind {
    Reachable { min_hops: u32, max_hops: u32 },
    Path { to: Expr, max_hops: u32 },
    Ppr { alpha: f64, eps: f64, max_pushes: u64, weighted: bool },
}

impl Expr {
    fn always() -> Expr {
        Expr::Lit(Value::Bool(true))
    }

    fn is_always(&self) -> bool {
        *self == Expr::always()
    }
}

impl Node {
    fn is_write(&self) -> bool {
        matches!(self, Node::Insert { .. } | Node::Update { .. } | Node::Delete { .. })
    }

    /// The nodes whose rows the node reads.
    pub fn inputs(&self) -> Vec<&Node> {
        match self {
            Node::Search(_) | Node::Insert { .. } | Node::Ref { .. } => Vec::new(),
            Node::Aggregate { input, .. }
            | Node::Filter { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => vec![input],
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => vec![a, b],
            Node::With { ctes, input } => ctes.iter().map(|c| &c.input).chain([&**input]).collect(),
        }
    }

    /// [`Node::inputs`], to change.
    pub(crate) fn inputs_mut(&mut self) -> Vec<&mut Node> {
        match self {
            Node::Search(_) | Node::Insert { .. } | Node::Ref { .. } => Vec::new(),
            Node::Aggregate { input, .. }
            | Node::Filter { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => vec![input],
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => vec![a, b],
            Node::With { ctes, input } => ctes.iter_mut().map(|c| &mut c.input).chain([&mut **input]).collect(),
        }
    }

    /// Whether the node or one below it (outside subqueries) is a
    /// [`Node::DistinctOn`].
    fn keeps_firsts(&self) -> bool {
        matches!(self, Node::DistinctOn { .. }) || self.inputs().into_iter().any(Node::keeps_firsts)
    }

    /// Whether the node or one below it (outside subqueries) keeps more
    /// than the first row of a group ([`PER_CAP`]).
    fn caps_per(&self) -> bool {
        matches!(self, Node::DistinctOn { per, .. } if *per > 1) || self.inputs().into_iter().any(Node::caps_per)
    }

    /// Whether the node has the shape the executor runs: `Project [Limit]
    /// [Sort] [Filter] [Aggregate] Search` ([`Plan::to_select`] says
    /// whether it is one it takes). Its subqueries ([`Expr::Values`] and
    /// the like) run first.
    pub fn is_leaf(&self) -> bool {
        let Node::Project { input, .. } = self else { return false };
        let mut n: &Node = input;
        loop {
            match n {
                Node::Limit { input, .. } | Node::Sort { input, .. } | Node::Filter { input, .. } | Node::Aggregate { input, .. } => n = input,
                Node::Search(_) => return true,
                _ => return false,
            }
        }
    }

    /// Whether running the node takes operators: it is not a leaf
    /// ([`Node::is_leaf`]), or holds a subquery that runs first.
    pub fn uses_operators(&self) -> bool {
        if !self.is_leaf() {
            return true;
        }
        let mut found = false;
        self.leaf_exprs(&mut |e| found |= e.has_operators());
        found
    }

    /// Whether the plan needs [`MAXSIM`]: an [`Expr::MaxSim`], a
    /// [`Func::Dot`] or weighted seeds anywhere in it, subqueries too.
    pub fn uses_maxsim(&self) -> bool {
        let mut found = false;
        self.visit_exprs(&mut |e| {
            found |= match e {
                Expr::MaxSim { .. } | Expr::Call { func: Func::Dot, .. } => true,
                Expr::Graph(g) | Expr::Ppr(g) => !g.weights.is_empty(),
                _ => false,
            }
        });
        found
    }

    /// Whether a Search in the plan reranks in phases ([`RANK_PHASES`]).
    pub fn has_phases(&self) -> bool {
        fn within(e: &Expr) -> bool {
            let mut found = false;
            e.visit(&mut |x| {
                if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = x {
                    found |= n.has_phases();
                }
            });
            found
        }
        match self {
            Node::Search(s) => !s.phases.is_empty() || within(&s.filter),
            Node::Filter { input, cond } => input.has_phases() || within(cond),
            Node::Aggregate { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => input.has_phases(),
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => a.has_phases() || b.has_phases(),
            Node::With { ctes, input } => input.has_phases() || ctes.iter().any(|c| c.input.has_phases()),
            Node::Insert { .. } | Node::Ref { .. } => false,
        }
    }

    /// Calls `f` on each Search of the node, those of subqueries in its
    /// expressions too.
    pub fn each_search(&self, f: &mut dyn FnMut(&Search)) {
        fn within(e: &Expr, f: &mut dyn FnMut(&Search)) {
            e.visit(&mut |x| {
                if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = x {
                    n.each_search(f);
                }
            });
        }
        match self {
            Node::Search(s) => {
                f(s);
                within(&s.filter, f);
            }
            Node::Filter { input, cond } => {
                input.each_search(f);
                within(cond, f);
            }
            Node::Aggregate { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => input.each_search(f),
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => {
                a.each_search(f);
                b.each_search(f);
            }
            Node::With { ctes, input } => {
                ctes.iter().for_each(|c| c.input.each_search(f));
                input.each_search(f);
            }
            Node::Insert { .. } | Node::Ref { .. } => {}
        }
    }

    /// Whether a Search in the plan ranks as a rank profile
    /// ([`RANK_PROFILES`]).
    pub fn has_profiles(&self) -> bool {
        let mut found = false;
        self.each_search(&mut |s| found |= s.profile.is_some());
        found
    }

    /// Whether a Search's columns in the plan hold a ranking expression
    /// ([`RANK_FEATURES`]).
    pub fn has_features(&self) -> bool {
        fn within(e: &Expr) -> bool {
            let mut found = false;
            e.visit(&mut |x| {
                if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = x {
                    found |= n.has_features();
                }
            });
            found
        }
        // The columns of `Project [Limit] Search`.
        fn over_search(n: &Node) -> bool {
            match n {
                Node::Limit { input, .. } => over_search(input),
                n => matches!(n, Node::Search(_)),
            }
        }
        match self {
            Node::Search(s) => within(&s.filter),
            Node::Project { input, columns } if over_search(input) => {
                columns.iter().any(|c| !matches!(c.expr, Expr::Field(_) | Expr::Score | Expr::Hops | Expr::SeqNo)) || input.has_features()
            }
            Node::Filter { input, cond } => input.has_features() || within(cond),
            Node::Aggregate { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => input.has_features(),
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => a.has_features() || b.has_features(),
            Node::With { ctes, input } => input.has_features() || ctes.iter().any(|c| c.input.has_features()),
            Node::Insert { .. } | Node::Ref { .. } => false,
        }
    }

    /// Whether a graph in the plan walks an edge table ([`EDGE_TABLE`]).
    pub fn walks_edge_tables(&self) -> bool {
        let mut found = false;
        self.visit_exprs(&mut |e| found |= matches!(e, Expr::Graph(g) | Expr::Ppr(g) if g.edges.is_some()));
        found
    }

    /// Whether a text of the plan has operators ([`TEXT_OPERATORS`]).
    pub fn has_text_operators(&self) -> bool {
        let mut found = false;
        self.visit_exprs(&mut |e| {
            found |= match e {
                Expr::Bm25 { text, opts, .. } => !crate::query::exec::plain_syntax(opts).unwrap_or(false) && crate::text::query::has_operators(text),
                Expr::Match { query, .. } => crate::text::query::match_features(query).weights,
                _ => false,
            }
        });
        found
    }

    /// Whether a text of the plan needs some of its items ([`MIN_MATCH`]).
    pub fn has_min_match(&self) -> bool {
        let mut found = false;
        self.visit_exprs(&mut |e| {
            found |= match e {
                Expr::Bm25 { opts, .. } => opts.contains_key("min"),
                Expr::Match { query, .. } => crate::text::query::match_features(query).at_least,
                _ => false,
            }
        });
        found
    }

    /// Whether the plan names a model to open: a ranking model
    /// ([`RANK_MODELS`]) or a sequence model ([`ENCODE`]).
    pub fn has_models(&self) -> bool {
        let mut found = false;
        self.visit_exprs(&mut |e| found |= matches!(e, Expr::Gbdt { .. } | Expr::CrossEncoder { .. } | Expr::Encode { .. }));
        found
    }

    /// Calls `f` on every expression of the node and of the nodes under
    /// it, and of the plans of subqueries in them, each expression before
    /// those within it.
    pub fn visit_exprs(&self, f: &mut dyn FnMut(&Expr)) {
        let mut each = |e: &Expr| {
            e.visit(&mut |x| {
                f(x);
                if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = x {
                    n.visit_exprs(f);
                }
            })
        };
        match self {
            Node::Search(s) => {
                each(&s.filter);
                s.order.iter().for_each(|k| each(&k.expr));
                s.phases.iter().for_each(|p| each(&p.expr));
                if let Some(GuardSite::AtLeast { conds }) = s.guard.as_deref().map(|g| &g.site) {
                    conds.iter().for_each(&mut each);
                }
            }
            Node::Aggregate { input, keys, aggs } => {
                keys.iter().chain(aggs.iter().filter_map(|a| a.arg.as_ref())).for_each(&mut each);
                input.visit_exprs(f);
            }
            Node::Filter { input, cond } => {
                each(cond);
                input.visit_exprs(f);
            }
            Node::Sort { input, keys } => {
                keys.iter().for_each(|k| each(&k.expr));
                input.visit_exprs(f);
            }
            Node::Project { input, columns } => {
                columns.iter().for_each(|c| each(&c.expr));
                input.visit_exprs(f);
            }
            Node::Insert { rows, on_conflict, .. } => {
                rows.iter().flatten().for_each(&mut each);
                if let OnConflict::Update(sets) = on_conflict {
                    sets.iter().for_each(|s| each(&s.value));
                }
            }
            Node::Update { input, sets, .. } => {
                sets.iter().for_each(|s| each(&s.value));
                input.visit_exprs(f);
            }
            Node::Join { left, right, keys, on, .. } => {
                keys.iter().for_each(|k| {
                    each(&k.left);
                    each(&k.right);
                });
                on.iter().for_each(&mut each);
                left.visit_exprs(f);
                right.visit_exprs(f);
            }
            Node::With { ctes, input } => {
                ctes.iter().for_each(|c| c.input.visit_exprs(f));
                input.visit_exprs(f);
            }
            Node::KeyOrder { input, by } => {
                input.visit_exprs(f);
                by.visit_exprs(f);
            }
            Node::DistinctOn { input, keys, .. } => {
                keys.iter().for_each(&mut each);
                input.visit_exprs(f);
            }
            Node::Limit { input, .. } | Node::Delete { input, .. } | Node::Distinct { input } => input.visit_exprs(f),
            Node::Ref { .. } => {}
        }
    }

    /// [`Node::visit_exprs`], changing the expressions.
    pub fn visit_exprs_mut(&mut self, f: &mut dyn FnMut(&mut Expr) -> Result<()>) -> Result<()> {
        let mut each = |e: &mut Expr| {
            e.visit_mut(&mut |x| {
                f(x)?;
                match x {
                    Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } => n.visit_exprs_mut(f),
                    _ => Ok(()),
                }
            })
        };
        match self {
            Node::Search(s) => {
                each(&mut s.filter)?;
                for e in s.order.iter_mut().map(|k| &mut k.expr).chain(s.phases.iter_mut().map(|p| &mut p.expr)) {
                    each(e)?;
                }
                if let Some(GuardSite::AtLeast { conds }) = s.guard.as_deref_mut().map(|g| &mut g.site) {
                    for e in conds {
                        each(e)?;
                    }
                }
                Ok(())
            }
            Node::Aggregate { input, keys, aggs } => {
                for e in keys.iter_mut().chain(aggs.iter_mut().filter_map(|a| a.arg.as_mut())) {
                    each(e)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::Filter { input, cond } => {
                each(cond)?;
                input.visit_exprs_mut(f)
            }
            Node::Sort { input, keys } => {
                for k in keys {
                    each(&mut k.expr)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::Project { input, columns } => {
                for c in columns {
                    each(&mut c.expr)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::Insert { rows, on_conflict, .. } => {
                for e in rows.iter_mut().flatten() {
                    each(e)?;
                }
                if let OnConflict::Update(sets) = on_conflict {
                    for s in sets {
                        each(&mut s.value)?;
                    }
                }
                Ok(())
            }
            Node::Update { input, sets, .. } => {
                for s in sets {
                    each(&mut s.value)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::Join { left, right, keys, on, .. } => {
                for k in keys {
                    each(&mut k.left)?;
                    each(&mut k.right)?;
                }
                if let Some(on) = on {
                    each(on)?;
                }
                left.visit_exprs_mut(f)?;
                right.visit_exprs_mut(f)
            }
            Node::With { ctes, input } => {
                for c in ctes {
                    c.input.visit_exprs_mut(f)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::KeyOrder { input, by } => {
                input.visit_exprs_mut(f)?;
                by.visit_exprs_mut(f)
            }
            Node::DistinctOn { input, keys, .. } => {
                for k in keys {
                    each(k)?;
                }
                input.visit_exprs_mut(f)
            }
            Node::Limit { input, .. } | Node::Delete { input, .. } | Node::Distinct { input } => input.visit_exprs_mut(f),
            Node::Ref { .. } => Ok(()),
        }
    }

    /// Whether a Search in the plan ranks by an expression that needs
    /// [`RANK_EXPRESSIONS`].
    pub fn ranks_by_expression(&self) -> bool {
        fn within(e: &Expr) -> bool {
            let mut found = false;
            e.visit(&mut |x| {
                if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = x {
                    found |= n.ranks_by_expression();
                }
            });
            found
        }
        match self {
            Node::Search(s) => s.order.first().is_some_and(|k| k.expr.is_rank_expression()) || s.phases.iter().any(|p| p.expr.is_rank_expression()) || within(&s.filter),
            Node::Filter { input, cond } => input.ranks_by_expression() || within(cond),
            Node::Aggregate { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::Project { input, .. }
            | Node::Update { input, .. }
            | Node::Delete { input, .. }
            | Node::Distinct { input }
            | Node::DistinctOn { input, .. } => input.ranks_by_expression(),
            Node::Join { left: a, right: b, .. } | Node::KeyOrder { input: a, by: b } => a.ranks_by_expression() || b.ranks_by_expression(),
            Node::With { ctes, input } => input.ranks_by_expression() || ctes.iter().any(|c| c.input.ranks_by_expression()),
            Node::Insert { .. } | Node::Ref { .. } => false,
        }
    }

    /// Calls `f` on each expression of a leaf's nodes.
    pub(crate) fn leaf_exprs(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            Node::Project { input, columns } => {
                columns.iter().for_each(|c| f(&c.expr));
                input.leaf_exprs(f);
            }
            Node::Limit { input, .. } => input.leaf_exprs(f),
            Node::Sort { input, keys } => {
                keys.iter().for_each(|k| f(&k.expr));
                input.leaf_exprs(f);
            }
            Node::Filter { input, cond } => {
                f(cond);
                input.leaf_exprs(f);
            }
            Node::Aggregate { input, keys, aggs } => {
                keys.iter().chain(aggs.iter().filter_map(|a| a.arg.as_ref())).for_each(&mut *f);
                input.leaf_exprs(f);
            }
            Node::Search(s) => {
                f(&s.filter);
                s.order.iter().for_each(|k| f(&k.expr));
                s.phases.iter().for_each(|p| f(&p.expr));
            }
            _ => {}
        }
    }

    /// [`Node::leaf_exprs`], changing them.
    pub(crate) fn leaf_exprs_mut(&mut self, f: &mut dyn FnMut(&mut Expr) -> Result<()>) -> Result<()> {
        match self {
            Node::Project { input, columns } => {
                for c in columns {
                    f(&mut c.expr)?;
                }
                input.leaf_exprs_mut(f)
            }
            Node::Limit { input, .. } => input.leaf_exprs_mut(f),
            Node::Sort { input, keys } => {
                for k in keys {
                    f(&mut k.expr)?;
                }
                input.leaf_exprs_mut(f)
            }
            Node::Filter { input, cond } => {
                f(cond)?;
                input.leaf_exprs_mut(f)
            }
            Node::Aggregate { input, keys, aggs } => {
                for k in keys.iter_mut().chain(aggs.iter_mut().filter_map(|a| a.arg.as_mut())) {
                    f(k)?;
                }
                input.leaf_exprs_mut(f)
            }
            Node::Search(s) => {
                f(&mut s.filter)?;
                for k in &mut s.order {
                    f(&mut k.expr)?;
                }
                for p in &mut s.phases {
                    f(&mut p.expr)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The Search of a leaf.
    /// A leaf's Search.
    pub(crate) fn search(&self) -> Option<&Search> {
        match self {
            Node::Search(s) => Some(s),
            Node::Project { input, .. } | Node::Limit { input, .. } | Node::Sort { input, .. } | Node::Filter { input, .. } | Node::Aggregate { input, .. } | Node::DistinctOn { input, .. } => input.search(),
            _ => None,
        }
    }

    /// A leaf's Search, to change.
    pub fn search_mut(&mut self) -> Option<&mut Search> {
        match self {
            Node::Search(s) => Some(s),
            Node::Project { input, .. } | Node::Limit { input, .. } | Node::Sort { input, .. } | Node::Filter { input, .. } | Node::Aggregate { input, .. } | Node::DistinctOn { input, .. } => input.search_mut(),
            _ => None,
        }
    }
}

impl Expr {
    /// Whether the expression holds a subquery that operators run
    /// ([`Expr::Values`], [`Expr::Keys`], [`Expr::Exists`],
    /// [`Expr::Scalar`]).
    /// Whether the expression, a Search's ranking, needs
    /// [`RANK_EXPRESSIONS`]: anything but a score, a field, `hops()` and
    /// the fused shapes of them that came before.
    pub fn is_rank_expression(&self) -> bool {
        match self {
            Expr::Field(_) | Expr::Hops | Expr::Bm25 { .. } | Expr::Similarity { .. } | Expr::Sparse { .. } | Expr::MaxSim { .. } | Expr::Decay { .. } | Expr::Ppr(_) | Expr::Ranked { .. } => false,
            Expr::Rrf { inputs, .. } | Expr::Product(inputs) => inputs.iter().any(Expr::is_rank_expression),
            Expr::Weighted(terms) => terms.iter().any(|(_, x)| x.is_rank_expression()),
            _ => true,
        }
    }

    /// Whether the expression adds raw relevance scores (`bm25(..) +
    /// similarity(..)`), which meant min-max fusion before ranking
    /// expressions: EXPLAIN says so, for one release.
    pub fn adds_raw_scores(&self) -> bool {
        // The scores and fields an expression reads raw: not through a set
        // function or a fused shape.
        fn raw<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
            match e {
                Expr::Arith { left, right, .. } => {
                    raw(left, out);
                    raw(right, out);
                }
                Expr::Neg(x) => raw(x, out),
                Expr::Call { args, .. } => args.iter().for_each(|a| raw(a, out)),
                Expr::Case { whens, otherwise, .. } => whens.iter().map(|w| &w.then).chain(otherwise.as_deref()).for_each(|x| raw(x, out)),
                Expr::Field(_) | Expr::Hops | Expr::SeqNo | Expr::Bm25 { .. } | Expr::Similarity { .. } | Expr::Sparse { .. } | Expr::Decay { .. } | Expr::Ppr(_) => out.push(e),
                _ => {}
            }
        }
        let relevance = |e: &&Expr| matches!(e, Expr::Bm25 { .. } | Expr::Similarity { .. } | Expr::Sparse { .. } | Expr::Ppr(_) | Expr::Hops);
        let mut found = false;
        self.visit(&mut |e| {
            if let Expr::Arith { op: ArithOp::Add | ArithOp::Sub, left, right } = e {
                let (mut l, mut r) = (Vec::new(), Vec::new());
                raw(left, &mut l);
                raw(right, &mut r);
                found |= !l.is_empty() && !r.is_empty() && l != r && (l.iter().any(relevance) || r.iter().any(relevance));
            }
        });
        found
    }

    pub fn has_operators(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| found |= matches!(e, Expr::Values(_) | Expr::Keys(_) | Expr::Exists(_) | Expr::Scalar(_) | Expr::Encode { .. }));
        found
    }

    /// Whether the expression is a correlated subquery: an
    /// [`Expr::Scalar`] whose plan names the row around it
    /// ([`Expr::Outer`]).
    pub fn is_correlated(&self) -> bool {
        let Expr::Scalar(n) = self else { return false };
        let mut found = false;
        n.visit_exprs(&mut |e| found |= matches!(e, Expr::Outer(_)));
        found
    }

    /// Whether the expression holds a correlated subquery
    /// ([`Expr::is_correlated`]).
    pub fn has_correlated(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| found |= e.is_correlated());
        found
    }

    /// Calls `f` on this expression and each within it, not inside the
    /// plans of subqueries.
    pub fn visit(&self, f: &mut dyn FnMut(&Expr)) {
        f(self);
        let graph = |g: &Graph, f: &mut dyn FnMut(&Expr)| {
            g.seeds.visit(f);
            if let GraphKind::Path { to, .. } = &g.kind {
                to.visit(f);
            }
            if let Some(t) = &g.through {
                t.visit(f);
            }
        };
        match self {
            Expr::Not(x) | Expr::IsNull(x) | Expr::Neg(x) | Expr::Norm { input: x, .. } => x.visit(f),
            Expr::And(xs) | Expr::Or(xs) | Expr::AtLeast { conds: xs, .. } | Expr::Concat(xs) | Expr::Product(xs) => xs.iter().for_each(|x| x.visit(f)),
            Expr::Cmp { left, right, .. } | Expr::Arith { left, right, .. } => {
                left.visit(f);
                right.visit(f);
            }
            Expr::Between { expr, lo, hi } => {
                expr.visit(f);
                lo.iter().chain(hi.iter()).for_each(|x| x.visit(f));
            }
            Expr::In { expr, list } => {
                expr.visit(f);
                list.iter().for_each(|x| x.visit(f));
            }
            Expr::Like { expr, pattern, .. } | Expr::Prefix { expr, prefix: pattern, .. } => {
                expr.visit(f);
                pattern.visit(f);
            }
            Expr::Case { operand, whens, otherwise } => {
                operand.iter().for_each(|x| x.visit(f));
                for w in whens {
                    w.when.visit(f);
                    w.then.visit(f);
                }
                otherwise.iter().for_each(|x| x.visit(f));
            }
            Expr::Call { args, .. } | Expr::Gbdt { args, .. } => args.iter().for_each(|x| x.visit(f)),
            Expr::MaxSim { vectors, .. } => vectors.visit(f),
            Expr::Rrf { inputs, .. } => inputs.iter().for_each(|x| x.visit(f)),
            Expr::Weighted(ts) => ts.iter().for_each(|(_, x)| x.visit(f)),
            Expr::Graph(g) | Expr::Ppr(g) => graph(g, f),
            _ => {}
        }
    }

    /// [`Expr::visit`], changing the expressions: `f` sees each before
    /// those within it.
    pub fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Expr) -> Result<()>) -> Result<()> {
        f(self)?;
        let mut within: Vec<&mut Expr> = Vec::new();
        match self {
            Expr::Not(x) | Expr::IsNull(x) | Expr::Neg(x) | Expr::Norm { input: x, .. } | Expr::MaxSim { vectors: x, .. } => within.push(x),
            Expr::And(xs)
            | Expr::Or(xs)
            | Expr::AtLeast { conds: xs, .. }
            | Expr::Concat(xs)
            | Expr::Product(xs)
            | Expr::Call { args: xs, .. }
            | Expr::Gbdt { args: xs, .. }
            | Expr::Rrf { inputs: xs, .. } => within.extend(xs.iter_mut()),
            Expr::Cmp { left, right, .. } | Expr::Arith { left, right, .. } | Expr::Like { expr: left, pattern: right, .. }
            | Expr::Prefix { expr: left, prefix: right, .. } => within.extend([&mut **left, &mut **right]),
            Expr::Between { expr, lo, hi } => {
                within.push(expr);
                within.extend(lo.iter_mut().chain(hi.iter_mut()).map(|x| &mut **x));
            }
            Expr::In { expr, list } => {
                within.push(expr);
                within.extend(list.iter_mut());
            }
            Expr::Case { operand, whens, otherwise } => {
                within.extend(operand.iter_mut().map(|x| &mut **x));
                for w in whens {
                    within.extend([&mut w.when, &mut w.then]);
                }
                within.extend(otherwise.iter_mut().map(|x| &mut **x));
            }
            Expr::Weighted(ts) => within.extend(ts.iter_mut().map(|(_, x)| x)),
            Expr::Graph(g) | Expr::Ppr(g) => {
                let Graph { seeds, kind, through, .. } = &mut **g;
                within.push(seeds);
                if let GraphKind::Path { to, .. } = kind {
                    within.push(to);
                }
                within.extend(through.iter_mut());
            }
            _ => {}
        }
        for x in within {
            x.visit_mut(f)?;
        }
        Ok(())
    }
}

fn is_zero(x: &u64) -> bool {
    *x == 0
}

fn one() -> u64 {
    1
}

fn is_one(x: &u64) -> bool {
    *x == 1
}

fn is_zero_u8(x: &u8) -> bool {
    *x == 0
}

fn is_false(x: &bool) -> bool {
    !*x
}

fn err(m: impl Into<String>) -> Error {
    Error::Query(m.into())
}

impl Plan {
    /// A plan of the current version, with the features it uses.
    pub fn new(root: Node) -> Plan {
        let mut features = Vec::new();
        if !root.is_write() && root.uses_operators() {
            features.push(OPERATORS.to_string());
        }
        let mut seq_no = false;
        if root.is_leaf() {
            root.leaf_exprs(&mut |e| e.visit(&mut |x| seq_no |= *x == Expr::SeqNo));
        }
        if seq_no {
            features.push(SEQ_NO.to_string());
        }
        if root.ranks_by_expression() {
            features.push(RANK_EXPRESSIONS.to_string());
        }
        let mut sid_at = false;
        root.visit_exprs(&mut |e| sid_at |= matches!(e, Expr::Call { func: Func::SidAt, .. }));
        if sid_at {
            features.push(SEMANTIC_ID.to_string());
        }
        if root.has_phases() {
            features.push(RANK_PHASES.to_string());
        }
        if root.has_features() {
            features.push(RANK_FEATURES.to_string());
        }
        if root.uses_maxsim() {
            features.push(MAXSIM.to_string());
        }
        if root.has_profiles() {
            features.push(RANK_PROFILES.to_string());
        }
        let (mut gbdt, mut encode) = (false, false);
        root.visit_exprs(&mut |e| {
            gbdt |= matches!(e, Expr::Gbdt { .. } | Expr::CrossEncoder { .. });
            encode |= matches!(e, Expr::Encode { .. });
        });
        if gbdt {
            features.push(RANK_MODELS.to_string());
        }
        let (mut prefix, mut edits) = (false, false);
        root.visit_exprs(&mut |e| {
            if let Expr::Prefix { edits: k, .. } = e {
                (prefix, edits) = (true, edits || *k > 0);
            }
        });
        if prefix {
            features.push(PREFIX.to_string());
        }
        if edits {
            features.push(PREFIX_EDITS.to_string());
        }
        let (mut distinct_on, mut per_cap) = (root.keeps_firsts(), root.caps_per());
        root.visit_exprs(&mut |e| {
            if let Expr::Subquery(n) | Expr::Values(n) | Expr::Keys(n) | Expr::Exists(n) | Expr::Scalar(n) | Expr::Encode { rows: n, .. } = e {
                distinct_on |= n.keeps_firsts();
                per_cap |= n.caps_per();
            }
        });
        if distinct_on {
            features.push(DISTINCT_ON.to_string());
        }
        if per_cap {
            features.push(PER_CAP.to_string());
        }
        if root.walks_edge_tables() {
            features.push(EDGE_TABLE.to_string());
        }
        let mut ranked = false;
        root.visit_exprs(&mut |e| ranked |= matches!(e, Expr::Ranked { .. }));
        if ranked {
            features.push(RANKED.to_string());
        }
        let mut norms = false;
        root.visit_exprs(&mut |e| norms |= matches!(e, Expr::Norm { kind: NormKind::Tmm | NormKind::ZScore, .. }));
        if norms {
            features.push(SCORE_NORMS.to_string());
        }
        if root.has_text_operators() {
            features.push(TEXT_OPERATORS.to_string());
        }
        let mut correlated = false;
        root.visit_exprs(&mut |e| correlated |= matches!(e, Expr::Outer(_) | Expr::ByKey(_)));
        if correlated {
            features.push(CORRELATED.to_string());
        }
        if root.has_min_match() {
            features.push(MIN_MATCH.to_string());
        }
        let mut counts = false;
        root.visit_exprs(&mut |e| counts |= matches!(e, Expr::AtLeast { .. } | Expr::Call { func: Func::CountTrue, .. }));
        if counts {
            features.push(AT_LEAST.to_string());
        }
        let mut guarded = false;
        root.each_search(&mut |s| guarded |= s.guard.is_some());
        if guarded {
            features.push(GUARD.to_string());
        }
        if encode {
            features.push(ENCODE.to_string());
        }
        Plan { version: VERSION, features, root }
    }

    /// The table the plan reads or writes: its (first) Search's, or its
    /// write's.
    pub fn table(&self) -> &str {
        fn of(n: &Node) -> Option<&str> {
            match n {
                Node::Search(s) => Some(&s.table),
                Node::Insert { table, .. } | Node::Update { table, .. } | Node::Delete { table, .. } => Some(table),
                Node::Aggregate { input, .. }
                | Node::Filter { input, .. }
                | Node::Sort { input, .. }
                | Node::Limit { input, .. }
                | Node::Project { input, .. }
                | Node::Distinct { input }
                | Node::DistinctOn { input, .. }
                | Node::KeyOrder { input, .. } => of(input),
                Node::Join { left, right, .. } => of(left).or_else(|| of(right)),
                Node::With { ctes, input } => of(input).or_else(|| ctes.iter().find_map(|c| of(&c.input))),
                Node::Ref { .. } => None,
            }
        }
        of(&self.root).unwrap_or("")
    }

    /// Whether the plan writes (an Insert, Update or Delete) rather than
    /// reads.
    pub fn is_write(&self) -> bool {
        self.root.is_write()
    }

    /// Whether any node of the plan writes, at its root or below it: what
    /// a server's authorization asks.
    pub fn writes(&self) -> bool {
        fn any(n: &Node) -> bool {
            n.is_write() || n.inputs().into_iter().any(any)
        }
        any(&self.root)
    }

    pub fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| err(format!("the plan cannot be written as JSON: {e}")))
    }

    /// [`Plan::to_json`] as text, each object's keys in the order of the
    /// types here (a node's kind first).
    pub fn to_json_text(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| err(format!("the plan cannot be written as JSON: {e}")))
    }

    /// Reads a plan, refusing first a version newer than [`VERSION`] and a
    /// feature this version lacks, by name, and then anything else it does
    /// not know.
    pub fn from_json(j: &serde_json::Value) -> Result<Plan> {
        let version = j.get("version").and_then(|v| v.as_u64()).ok_or_else(|| err("a plan needs a version"))?;
        let features = match j.get("features") {
            None => Vec::new(),
            Some(serde_json::Value::Array(fs)) => fs.iter().map(|f| f.as_str().ok_or_else(|| err("a plan's features are names"))).collect::<Result<_>>()?,
            Some(_) => return Err(err("a plan's features are a list of names")),
        };
        Plan::readable(version, &features)?;
        serde_json::from_value(j.clone()).map_err(|e| err(format!("not a plan of version {VERSION}: {e}")))
    }

    /// Whether this version of Strata reads a plan of `version` that needs
    /// `features`: an error naming what it lacks if not. A plan in another
    /// form than JSON (a shard's protobuf) is checked with it too.
    pub fn readable(version: u64, features: &[&str]) -> Result<()> {
        Plan::readable_by(version, features, &FEATURES)
    }

    /// [`Plan::readable`] by a reader of the plan features `known`: an
    /// older version of Strata knows fewer.
    pub fn readable_by(version: u64, features: &[&str], known: &[&str]) -> Result<()> {
        if version == 0 || version > VERSION as u64 {
            return Err(err(format!("plan version {version} is not one this version of Strata reads (it reads version {VERSION}): it may be newer")));
        }
        match features.iter().find(|f| !known.contains(f)) {
            Some(name) => Err(err(format!("the plan needs feature {name:?}, which this version of Strata does not have: it may be newer"))),
            None => Ok(()),
        }
    }
}

/// Typed literals in JSON: `null`, `{"bool": true}`, `{"int": 5}`,
/// `{"float": 0.5}`, `{"text": "a"}`, `{"timestamp": micros}`,
/// `{"vector": [...]}` and `{"list": [...]}`. A FLOAT must be finite.
mod lit {
    use std::sync::Arc;

    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};

    use crate::value::Value;

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum Lit {
        Null,
        Bool(bool),
        Int(i64),
        Float(f64),
        Text(String),
        Timestamp(i64),
        Vector(Arc<[f32]>),
        List(Vec<Lit>),
    }

    impl Lit {
        pub(super) fn of(v: &Value) -> std::result::Result<Lit, String> {
            Ok(match v {
                Value::Null => Lit::Null,
                Value::Bool(b) => Lit::Bool(*b),
                Value::Int(x) => Lit::Int(*x),
                Value::Float(x) if x.is_finite() => Lit::Float(*x),
                Value::Float(x) => return Err(format!("a FLOAT literal must be finite, not {x}")),
                Value::Str(s) => Lit::Text(s.clone()),
                Value::Timestamp(t) => Lit::Timestamp(*t),
                Value::Vector(v) if v.iter().all(|x| x.is_finite()) => Lit::Vector(v.clone()),
                Value::Vector(_) => return Err("a vector's components must be finite".into()),
                Value::List(xs) => Lit::List(xs.iter().map(Lit::of).collect::<std::result::Result<_, _>>()?),
            })
        }

        pub(super) fn value(self) -> Value {
            match self {
                Lit::Null => Value::Null,
                Lit::Bool(b) => Value::Bool(b),
                Lit::Int(x) => Value::Int(x),
                Lit::Float(x) => Value::Float(x),
                Lit::Text(s) => Value::Str(s),
                Lit::Timestamp(t) => Value::Timestamp(t),
                Lit::Vector(v) => Value::Vector(v),
                Lit::List(xs) => Value::List(xs.into_iter().map(Lit::value).collect()),
            }
        }
    }

    pub fn serialize<S: Serializer>(v: &Value, s: S) -> std::result::Result<S::Ok, S::Error> {
        Lit::of(v).map_err(ser::Error::custom)?.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Value, D::Error> {
        let v = Lit::deserialize(d)?.value();
        Lit::of(&v).map_err(de::Error::custom)?;
        Ok(v)
    }
}

/// A list of literals, each as [`lit`] writes it.
mod lit_list {
    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};

    use super::lit::Lit;
    use crate::value::Value;

    pub fn serialize<S: Serializer>(vs: &[Value], s: S) -> std::result::Result<S::Ok, S::Error> {
        vs.iter().map(Lit::of).collect::<std::result::Result<Vec<_>, String>>().map_err(ser::Error::custom)?.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<Value>, D::Error> {
        let vs: Vec<Value> = Vec::<Lit>::deserialize(d)?.into_iter().map(Lit::value).collect();
        for v in &vs {
            Lit::of(v).map_err(de::Error::custom)?;
        }
        Ok(vs)
    }
}

/// Named options of typed literals (`{"k": {"int": 60}}`).
mod lits {
    use std::collections::BTreeMap;

    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};

    use super::lit::Lit;
    use crate::query::logical::Opts;

    pub fn serialize<S: Serializer>(opts: &Opts, s: S) -> std::result::Result<S::Ok, S::Error> {
        let m: BTreeMap<&str, Lit> = opts.iter().map(|(k, v)| Ok((k.as_str(), Lit::of(v)?))).collect::<std::result::Result<_, String>>().map_err(ser::Error::custom)?;
        m.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Opts, D::Error> {
        let m = BTreeMap::<String, Lit>::deserialize(d)?;
        let opts: Opts = m.into_iter().map(|(k, v)| (k, v.value())).collect();
        for v in opts.values() {
            Lit::of(v).map_err(de::Error::custom)?;
        }
        Ok(opts)
    }
}
