//! Plans ([`strata::query::ir`]) as protobuf (`proto/strata.proto`'s
//! `Plan`), the form a coordinator sends its shards from protocol 2. The
//! two forms hold the same tree: a plan read back from protobuf equals the
//! plan written, as one read back from JSON does. Reading checks what
//! JSON's reading checks: the plan's version and features, and that every
//! number is finite.

use std::collections::{BTreeMap, HashMap};

use strata::query::ir::{Agg, ArithOp, CmpOp, Column, Cte, Expr, Func, Graph, GraphKind, Guard, GuardSite, JoinKey, JoinKind, KeyValue, Node, NormKind, OnConflict, Phase, Plan, ProfileRef, Search, Set, SortKey, When};
use strata::query::{AggFn, Decay, DecayShape, Direction, Expand, Fuzzy};
use strata::text::query::Need;
use strata::value::Value;

use crate::grpc::{from_pb, to_pb};
use crate::pb::{self, expr::Expr as E, node::Node as N};
use crate::ApiError;

/// `plan` as protobuf.
pub fn to_pb_plan(plan: &Plan) -> pb::Plan {
    pb::Plan { version: plan.version, features: plan.features.clone(), root: Some(node(&plan.root)) }
}

/// The plan a request carries, checked as [`Plan::from_json`] checks one.
pub fn from_pb_plan(p: pb::Plan) -> Result<Plan, ApiError> {
    let features: Vec<&str> = p.features.iter().map(String::as_str).collect();
    Plan::readable(p.version as u64, &features)?;
    let root = from_node(p.root)?;
    Ok(Plan { version: p.version, features: p.features, root })
}

fn bad(m: impl Into<String>) -> ApiError {
    ApiError::bad(format!("bad plan: {}", m.into()))
}

fn boxed(n: &Node) -> Option<Box<pb::Node>> {
    Some(Box::new(node(n)))
}

fn ex(e: &Expr) -> pb::Expr {
    pb::Expr { expr: Some(expr(e)) }
}

fn bex(e: &Expr) -> Option<Box<pb::Expr>> {
    Some(Box::new(ex(e)))
}

fn exprs(es: &[Expr]) -> pb::Exprs {
    pb::Exprs { v: es.iter().map(ex).collect() }
}

fn sort_key(k: &SortKey) -> pb::SortKey {
    pb::SortKey { expr: Some(ex(&k.expr)), desc: k.desc, nulls_first: k.nulls_first }
}

fn set(s: &Set) -> pb::Set {
    pb::Set { field: s.field, value: Some(ex(&s.value)) }
}

fn match_pb(field: u32, query: &str, expand: &Expand) -> pb::Match {
    pb::Match {
        field,
        query: query.to_string(),
        expand: Some(pb::Expand {
            synonyms: expand.synonyms,
            fuzzy: expand.fuzzy.map(|f| match f {
                Fuzzy::Auto => pb::expand::Fuzzy::Auto(pb::Empty {}),
                Fuzzy::Edits(n) => pb::expand::Fuzzy::Edits(n as u32),
            }),
        }),
    }
}

fn guard(g: &Guard) -> pb::Guard {
    use pb::guard::{Need as PbNeed, Site};
    pb::Guard {
        site: Some(match &g.site {
            GuardSite::AtLeast { conds } => Site::AtLeast(pb::GuardAtLeast { conds: conds.iter().map(ex).collect() }),
            GuardSite::Match { field, text, expand } => Site::Match(match_pb(*field, text, expand)),
            GuardSite::Bm25 { field, text } => Site::Bm25(pb::GuardBm25 { field: *field, text: text.clone() }),
        }),
        need: Some(match g.need {
            Need::Count(n) => PbNeed::Count(n as u64),
            Need::Percent(p) => PbNeed::Percent(p),
        }),
        min_rows: g.min_rows,
        max_rows: g.max_rows,
    }
}

fn opts(o: &BTreeMap<String, Value>) -> HashMap<String, pb::Value> {
    o.iter().map(|(k, v)| (k.clone(), to_pb(v))).collect()
}

fn node(n: &Node) -> pb::Node {
    let n = match n {
        Node::Search(s) => N::Search(Box::new(pb::Search {
            table: s.table.clone(),
            filter: bex(&s.filter),
            order: s.order.iter().map(sort_key).collect(),
            phases: s.phases.iter().map(|p| pb::Phase { count: p.count, expr: Some(ex(&p.expr)), global: p.global }).collect(),
            profile: s.profile.as_ref().map(|r| pb::RankProfileRef { name: r.name.clone(), hash: r.hash.clone() }),
            guard: s.guard.as_deref().map(guard),
        })),
        Node::Aggregate { input, keys, aggs } => N::Aggregate(Box::new(pb::Aggregate {
            input: boxed(input),
            keys: keys.iter().map(ex).collect(),
            aggs: aggs.iter().map(|a| pb::Agg { func: agg_fn(a.func) as i32, arg: a.arg.as_ref().map(ex) }).collect(),
        })),
        Node::Filter { input, cond } => N::Filter(Box::new(pb::Filter { input: boxed(input), cond: bex(cond) })),
        Node::Sort { input, keys } => N::Sort(Box::new(pb::Sort { input: boxed(input), keys: keys.iter().map(sort_key).collect() })),
        Node::Limit { input, offset, limit } => N::Limit(Box::new(pb::Limit { input: boxed(input), offset: *offset, limit: *limit })),
        Node::Project { input, columns } => N::Project(Box::new(pb::Project { input: boxed(input), columns: columns.iter().map(|c| pb::Column { name: c.name.clone(), expr: Some(ex(&c.expr)) }).collect() })),
        Node::Insert { table, rows, on_conflict } => {
            use pb::on_conflict::Kind;
            let kind = match on_conflict {
                OnConflict::Fail => None,
                OnConflict::Replace => Some(Kind::Replace(pb::Empty {})),
                OnConflict::Skip => Some(Kind::Skip(pb::Empty {})),
                OnConflict::Update(sets) => Some(Kind::Update(pb::Sets { v: sets.iter().map(set).collect() })),
            };
            N::Insert(pb::Insert { table: table.clone(), rows: rows.iter().map(|r| exprs(r)).collect(), on_conflict: kind.map(|k| pb::OnConflict { kind: Some(k) }) })
        }
        Node::Update { table, input, sets } => N::Update(Box::new(pb::Update { table: table.clone(), input: boxed(input), sets: sets.iter().map(set).collect() })),
        Node::Delete { table, input } => N::Delete(Box::new(pb::Delete { table: table.clone(), input: boxed(input) })),
        Node::Join { left, right, kind, keys, on } => N::Join(Box::new(pb::Join {
            left: boxed(left),
            right: boxed(right),
            kind: match kind {
                JoinKind::Inner => pb::JoinKind::Inner,
                JoinKind::Left => pb::JoinKind::Left,
            } as i32,
            keys: keys.iter().map(|k| pb::JoinKey { left: Some(ex(&k.left)), right: Some(ex(&k.right)), left_links: k.left_links, right_links: k.right_links }).collect(),
            on: on.as_ref().and_then(bex),
        })),
        Node::With { ctes, input } => N::With(Box::new(pb::With { ctes: ctes.iter().map(|c| pb::Cte { name: c.name.clone(), input: Some(node(&c.input)) }).collect(), input: boxed(input) })),
        Node::Ref { name } => N::Ref(name.clone()),
        Node::Distinct { input } => N::Distinct(Box::new(node(input))),
        Node::KeyOrder { input, by } => N::KeyOrder(Box::new(pb::KeyOrder { input: boxed(input), by: boxed(by) })),
        Node::DistinctOn { input, keys, per } => N::DistinctOn(Box::new(pb::DistinctOn { input: boxed(input), keys: keys.iter().map(ex).collect(), per: *per })),
    };
    pb::Node { node: Some(n) }
}

fn agg_fn(f: AggFn) -> pb::AggFn {
    match f {
        AggFn::Count => pb::AggFn::Count,
        AggFn::Sum => pb::AggFn::Sum,
        AggFn::Avg => pb::AggFn::Avg,
        AggFn::Min => pb::AggFn::Min,
        AggFn::Max => pb::AggFn::Max,
        AggFn::CountDistinct => pb::AggFn::CountDistinct,
    }
}

fn graph(g: &Graph) -> pb::Graph {
    use pb::graph::Kind;
    let kind = match &g.kind {
        GraphKind::Reachable { min_hops, max_hops } => Kind::Reachable(pb::Reachable { min_hops: *min_hops, max_hops: *max_hops }),
        GraphKind::Path { to, max_hops } => Kind::Path(Box::new(pb::Path { to: bex(to), max_hops: *max_hops })),
        GraphKind::Ppr { alpha, eps, max_pushes, weighted } => Kind::Ppr(pb::Ppr { alpha: *alpha, eps: *eps, max_pushes: *max_pushes, weighted: *weighted }),
    };
    pb::Graph {
        field: g.field,
        edges: g.edges.clone(),
        seeds: bex(&g.seeds),
        kind: Some(kind),
        dir: match g.dir {
            Direction::Out => pb::Direction::Out,
            Direction::In => pb::Direction::In,
            Direction::Both => pb::Direction::Both,
        } as i32,
        labels: g.labels.clone(),
        through: g.through.as_ref().and_then(bex),
        max_nodes: g.max_nodes,
        k: g.k,
        min_sim: g.min_sim,
        weights: g.weights.clone(),
    }
}

fn expr(e: &Expr) -> E {
    match e {
        Expr::Field(f) => E::Field(*f),
        Expr::Score => E::Score(pb::Empty {}),
        Expr::Hops => E::Hops(pb::Empty {}),
        Expr::SeqNo => E::SeqNo(pb::Empty {}),
        Expr::Output(n) => E::Output(*n),
        Expr::Excluded(f) => E::Excluded(*f),
        Expr::Lit(v) => E::Lit(to_pb(v)),
        Expr::Not(x) => E::Not(Box::new(ex(x))),
        Expr::And(xs) => E::And(exprs(xs)),
        Expr::Or(xs) => E::Or(exprs(xs)),
        Expr::AtLeast { need, conds } => E::AtLeast(pb::AtLeast { need: *need, conds: conds.iter().map(ex).collect() }),
        Expr::IsNull(x) => E::IsNull(Box::new(ex(x))),
        Expr::Cmp { op, left, right } => E::Cmp(Box::new(pb::Cmp {
            op: match op {
                CmpOp::Eq => pb::CmpOp::Eq,
                CmpOp::Ne => pb::CmpOp::Ne,
                CmpOp::Lt => pb::CmpOp::Lt,
                CmpOp::Le => pb::CmpOp::Le,
                CmpOp::Gt => pb::CmpOp::Gt,
                CmpOp::Ge => pb::CmpOp::Ge,
            } as i32,
            left: bex(left),
            right: bex(right),
        })),
        Expr::Between { expr, lo, hi } => E::Between(Box::new(pb::Between { expr: bex(expr), lo: lo.as_deref().and_then(bex), hi: hi.as_deref().and_then(bex) })),
        Expr::In { expr, list } => E::In(Box::new(pb::In { expr: bex(expr), list: list.iter().map(ex).collect() })),
        Expr::Arith { op, left, right } => E::Arith(Box::new(pb::Arith {
            op: match op {
                ArithOp::Add => pb::ArithOp::Add,
                ArithOp::Sub => pb::ArithOp::Sub,
                ArithOp::Mul => pb::ArithOp::Mul,
                ArithOp::Div => pb::ArithOp::Div,
                ArithOp::Mod => pb::ArithOp::Mod,
            } as i32,
            left: bex(left),
            right: bex(right),
        })),
        Expr::Neg(x) => E::Neg(Box::new(ex(x))),
        Expr::Concat(xs) => E::Concat(exprs(xs)),
        Expr::Like { expr, pattern, case_insensitive } => E::Like(Box::new(pb::Like { expr: bex(expr), pattern: bex(pattern), case_insensitive: *case_insensitive })),
        Expr::Prefix { expr, prefix, edits } => E::Prefix(Box::new(pb::Prefix { expr: bex(expr), prefix: bex(prefix), edits: *edits as u32 })),
        Expr::Case { operand, whens, otherwise } => E::Case(Box::new(pb::Case {
            operand: operand.as_deref().and_then(bex),
            whens: whens.iter().map(|w| pb::When { when: Some(ex(&w.when)), then: Some(ex(&w.then)) }).collect(),
            otherwise: otherwise.as_deref().and_then(bex),
        })),
        Expr::Call { func, args } => E::Call(pb::Call {
            func: match func {
                Func::Lower => pb::Func::Lower,
                Func::Upper => pb::Func::Upper,
                Func::Length => pb::Func::Length,
                Func::Abs => pb::Func::Abs,
                Func::Round => pb::Func::Round,
                Func::Coalesce => pb::Func::Coalesce,
                Func::Ln => pb::Func::Ln,
                Func::Log => pb::Func::Log,
                Func::Exp => pb::Func::Exp,
                Func::Sqrt => pb::Func::Sqrt,
                Func::Pow => pb::Func::Pow,
                Func::Least => pb::Func::Least,
                Func::Greatest => pb::Func::Greatest,
                Func::Sigmoid => pb::Func::Sigmoid,
                Func::Clamp => pb::Func::Clamp,
                Func::Dot => pb::Func::Dot,
                Func::SidAt => pb::Func::SidAt,
                Func::CountTrue => pb::Func::CountTrue,
            } as i32,
            args: args.iter().map(ex).collect(),
        }),
        Expr::Match { field, query, expand } => E::Match(match_pb(*field, query, expand)),
        Expr::Graph(g) => E::Graph(Box::new(graph(g))),
        Expr::Bm25 { field, text, opts: o } => E::Bm25(pb::Bm25 { field: *field, text: text.clone(), opts: opts(o) }),
        Expr::Similarity { field, vector, opts: o } => E::Similarity(pb::Similarity { field: *field, vector: Some(pb::Floats { v: vector.to_vec() }), opts: opts(o) }),
        Expr::MaxSim { field, vectors, opts: o } => E::Maxsim(Box::new(pb::MaxSim { field: *field, vectors: bex(vectors), opts: opts(o) })),
        Expr::Sparse { field, terms, opts: o } => E::Sparse(pb::SparseScore { field: *field, terms: terms.iter().map(|(t, w)| pb::SparseTerm { term: t.clone(), weight: *w }).collect(), opts: opts(o) }),
        Expr::Decay { field, decay, opts: o } => E::Decay(pb::DecayScore {
            field: *field,
            decay: Some(pb::Decay {
                origin: decay.origin,
                scale: decay.scale,
                offset: decay.offset,
                decay: decay.decay,
                shape: match decay.shape {
                    DecayShape::Exp => pb::DecayShape::Exp,
                    DecayShape::Gauss => pb::DecayShape::Gauss,
                    DecayShape::Linear => pb::DecayShape::Linear,
                } as i32,
            }),
            opts: opts(o),
        }),
        Expr::Rrf { inputs, k, depth } => E::Rrf(pb::Rrf { inputs: inputs.iter().map(ex).collect(), k: *k, depth: *depth }),
        Expr::Weighted(terms) => E::Weighted(pb::Weighted { terms: terms.iter().map(|(w, e)| pb::WeightedTerm { weight: *w, expr: Some(ex(e)) }).collect() }),
        Expr::Product(xs) => E::Product(exprs(xs)),
        Expr::Norm { kind, input, lo } => E::Norm(Box::new(pb::Norm {
            kind: match kind {
                NormKind::MinMax => pb::NormKind::MinMax,
                NormKind::Rank => pb::NormKind::Rank,
                NormKind::Tmm => pb::NormKind::Tmm,
                NormKind::ZScore => pb::NormKind::ZScore,
            } as i32,
            input: bex(input),
            lo: *lo,
        })),
        Expr::Gbdt { model, hash, args, .. } => E::Gbdt(pb::Gbdt { model: model.clone(), hash: hash.clone(), args: args.iter().map(ex).collect() }),
        Expr::CrossEncoder { model, hash, fields, query, .. } => E::CrossEncoder(pb::CrossEncoder { model: model.clone(), hash: hash.clone(), fields: fields.clone(), query: query.clone() }),
        Expr::Ppr(g) => E::Ppr(Box::new(graph(g))),
        Expr::Ranked { keys } => E::Ranked(pb::Ranked { keys: keys.iter().map(to_pb).collect() }),
        Expr::ByKey(vs) => E::ByKey(pb::ByKey { v: vs.iter().map(|kv| pb::KeyValue { key: Some(to_pb(&kv.key)), value: kv.value }).collect() }),
        Expr::Outer(f) => E::Outer(*f),
        Expr::Encode { model, hash, rows, now, .. } => E::Encode(Box::new(pb::Encode { model: model.clone(), hash: hash.clone(), rows: Some(Box::new(node(rows))), now: *now })),
        Expr::Subquery(n) => E::Subquery(Box::new(node(n))),
        Expr::Values(n) => E::Values(Box::new(node(n))),
        Expr::Keys(n) => E::Keys(Box::new(node(n))),
        Expr::Exists(n) => E::Exists(Box::new(node(n))),
        Expr::Scalar(n) => E::Scalar(Box::new(node(n))),
    }
}

// Reading.

/// `x`, which must be finite.
fn finite(x: f64, what: &str) -> Result<f64, ApiError> {
    match x.is_finite() {
        true => Ok(x),
        false => Err(bad(format!("{what} must be finite, not {x}"))),
    }
}

/// A literal: a FLOAT and a vector's components finite, as in JSON.
fn lit(v: pb::Value) -> Result<Value, ApiError> {
    fn check(v: &Value) -> Result<(), ApiError> {
        match v {
            Value::Float(x) => finite(*x, "a FLOAT literal").map(|_| ()),
            Value::Vector(xs) if xs.iter().any(|x| !x.is_finite()) => Err(bad("a vector's components must be finite")),
            Value::List(xs) => xs.iter().try_for_each(check),
            _ => Ok(()),
        }
    }
    let v = from_pb(v);
    check(&v)?;
    Ok(v)
}

fn from_opts(o: HashMap<String, pb::Value>) -> Result<BTreeMap<String, Value>, ApiError> {
    o.into_iter().map(|(k, v)| Ok((k, lit(v)?))).collect()
}

fn need<T>(x: Option<T>, what: &str) -> Result<T, ApiError> {
    x.ok_or_else(|| bad(format!("{what} is missing")))
}

fn from_node(n: Option<pb::Node>) -> Result<Node, ApiError> {
    from_node_in(need(n, "a node")?)
}

fn from_bnode(n: Option<Box<pb::Node>>) -> Result<Box<Node>, ApiError> {
    Ok(Box::new(from_node_in(*need(n, "a node's input")?)?))
}

fn from_ex(e: Option<pb::Expr>) -> Result<Expr, ApiError> {
    from_expr(need(need(e, "an expression")?.expr, "an expression")?)
}

fn from_bex(e: Option<Box<pb::Expr>>) -> Result<Box<Expr>, ApiError> {
    Ok(Box::new(from_ex(e.map(|e| *e))?))
}

fn from_opt_bex(e: Option<Box<pb::Expr>>) -> Result<Option<Box<Expr>>, ApiError> {
    e.map(|e| from_bex(Some(e))).transpose()
}

fn from_match(m: pb::Match) -> Result<(u32, String, Expand), ApiError> {
    let x = m.expand.unwrap_or_default();
    let fuzzy = match x.fuzzy {
        None => None,
        Some(pb::expand::Fuzzy::Auto(_)) => Some(Fuzzy::Auto),
        Some(pb::expand::Fuzzy::Edits(n)) => Some(Fuzzy::Edits(u8::try_from(n).map_err(|_| bad(format!("{n} edits is too many")))?)),
    };
    Ok((m.field, m.query, Expand { synonyms: x.synonyms, fuzzy }))
}

fn from_guard(g: pb::Guard) -> Result<Guard, ApiError> {
    use pb::guard::{Need as PbNeed, Site};
    let site = match need(g.site, "a guard's site")? {
        Site::AtLeast(a) => GuardSite::AtLeast { conds: from_exprs(a.conds)? },
        Site::Match(m) => {
            let (field, text, expand) = from_match(m)?;
            GuardSite::Match { field, text, expand }
        }
        Site::Bm25(b) => GuardSite::Bm25 { field: b.field, text: b.text },
    };
    let need = match need(g.need, "a guard's level")? {
        PbNeed::Count(n) => Need::Count(usize::try_from(n).map_err(|_| bad(format!("a guard's level of {n}")))?),
        PbNeed::Percent(p) => Need::Percent(p),
    };
    Ok(Guard { site, need, min_rows: g.min_rows, max_rows: g.max_rows })
}

fn from_exprs(es: Vec<pb::Expr>) -> Result<Vec<Expr>, ApiError> {
    es.into_iter().map(|e| from_ex(Some(e))).collect()
}

fn from_sort_key(k: pb::SortKey) -> Result<SortKey, ApiError> {
    Ok(SortKey { expr: from_ex(k.expr)?, desc: k.desc, nulls_first: k.nulls_first })
}

fn from_set(s: pb::Set) -> Result<Set, ApiError> {
    Ok(Set { field: s.field, value: from_ex(s.value)? })
}

/// An enum's value, by its number.
fn en<T: TryFrom<i32>>(n: i32, what: &str) -> Result<T, ApiError> {
    T::try_from(n).map_err(|_| bad(format!("{n} is not a {what}")))
}

fn from_node_in(n: pb::Node) -> Result<Node, ApiError> {
    Ok(match need(n.node, "a node")? {
        N::Search(s) => {
            let s = *s;
            let filter = match s.filter {
                Some(f) => from_ex(Some(*f))?,
                None => Expr::Lit(Value::Bool(true)),
            };
            let phases = s.phases.into_iter().map(|p| Ok(Phase { count: p.count, expr: from_ex(p.expr)?, global: p.global })).collect::<Result<_, ApiError>>()?;
            let profile = s.profile.map(|r| ProfileRef { name: r.name, hash: r.hash });
            let guard = s.guard.map(from_guard).transpose()?.map(Box::new);
            // The typed plan does not carry Search::per, a hint the operator
            // above a Search does without.
            Node::Search(Search { table: s.table, filter, order: s.order.into_iter().map(from_sort_key).collect::<Result<_, _>>()?, phases, profile, guard, per: None })
        }
        N::Aggregate(a) => {
            let a = *a;
            let aggs = a
                .aggs
                .into_iter()
                .map(|g| {
                    let func = match en::<pb::AggFn>(g.func, "aggregate")? {
                        pb::AggFn::Count => AggFn::Count,
                        pb::AggFn::Sum => AggFn::Sum,
                        pb::AggFn::Avg => AggFn::Avg,
                        pb::AggFn::Min => AggFn::Min,
                        pb::AggFn::Max => AggFn::Max,
                        pb::AggFn::CountDistinct => AggFn::CountDistinct,
                    };
                    Ok(Agg { func, arg: g.arg.map(|e| from_ex(Some(e))).transpose()? })
                })
                .collect::<Result<_, ApiError>>()?;
            Node::Aggregate { input: from_bnode(a.input)?, keys: from_exprs(a.keys)?, aggs }
        }
        N::Filter(f) => Node::Filter { input: from_bnode(f.input)?, cond: from_ex(f.cond.map(|c| *c))? },
        N::Sort(s) => Node::Sort { input: from_bnode(s.input)?, keys: s.keys.into_iter().map(from_sort_key).collect::<Result<_, _>>()? },
        N::Limit(l) => Node::Limit { input: from_bnode(l.input)?, offset: l.offset, limit: l.limit },
        N::Project(p) => {
            let p = *p;
            let columns = p.columns.into_iter().map(|c| Ok(Column { name: c.name, expr: from_ex(c.expr)? })).collect::<Result<_, ApiError>>()?;
            Node::Project { input: from_bnode(p.input)?, columns }
        }
        N::Insert(i) => {
            use pb::on_conflict::Kind;
            let on_conflict = match i.on_conflict.and_then(|c| c.kind) {
                None | Some(Kind::Fail(_)) => OnConflict::Fail,
                Some(Kind::Replace(_)) => OnConflict::Replace,
                Some(Kind::Skip(_)) => OnConflict::Skip,
                Some(Kind::Update(sets)) => OnConflict::Update(sets.v.into_iter().map(from_set).collect::<Result<_, _>>()?),
            };
            Node::Insert { table: i.table, rows: i.rows.into_iter().map(|r| from_exprs(r.v)).collect::<Result<_, _>>()?, on_conflict }
        }
        N::Update(u) => {
            let u = *u;
            Node::Update { table: u.table, input: from_bnode(u.input)?, sets: u.sets.into_iter().map(from_set).collect::<Result<_, _>>()? }
        }
        N::Delete(d) => {
            let d = *d;
            Node::Delete { table: d.table, input: from_bnode(d.input)? }
        }
        N::Join(j) => {
            let j = *j;
            let kind = match en::<pb::JoinKind>(j.kind, "join kind")? {
                pb::JoinKind::Inner => JoinKind::Inner,
                pb::JoinKind::Left => JoinKind::Left,
            };
            let keys = j
                .keys
                .into_iter()
                .map(|k| Ok(JoinKey { left: from_ex(k.left)?, right: from_ex(k.right)?, left_links: k.left_links, right_links: k.right_links }))
                .collect::<Result<_, ApiError>>()?;
            let on = j.on.map(|e| from_ex(Some(*e))).transpose()?;
            Node::Join { left: from_bnode(j.left)?, right: from_bnode(j.right)?, kind, keys, on }
        }
        N::With(w) => {
            let w = *w;
            let ctes = w.ctes.into_iter().map(|c| Ok(Cte { name: c.name, input: from_node(c.input)? })).collect::<Result<_, ApiError>>()?;
            Node::With { ctes, input: from_bnode(w.input)? }
        }
        N::Ref(name) => Node::Ref { name },
        N::Distinct(n) => Node::Distinct { input: Box::new(from_node_in(*n)?) },
        N::KeyOrder(k) => {
            let k = *k;
            Node::KeyOrder { input: from_bnode(k.input)?, by: from_bnode(k.by)? }
        }
        N::DistinctOn(d) => {
            let d = *d;
            Node::DistinctOn { input: from_bnode(d.input)?, keys: from_exprs(d.keys)?, per: d.per.max(1) }
        }
    })
}

fn from_graph(g: pb::Graph) -> Result<Graph, ApiError> {
    use pb::graph::Kind;
    let kind = match need(g.kind, "a traversal's kind")? {
        Kind::Reachable(r) => GraphKind::Reachable { min_hops: r.min_hops, max_hops: r.max_hops },
        Kind::Path(p) => {
            let p = *p;
            GraphKind::Path { to: from_ex(p.to.map(|e| *e))?, max_hops: p.max_hops }
        }
        Kind::Ppr(p) => GraphKind::Ppr { alpha: finite(p.alpha, "alpha")?, eps: finite(p.eps, "eps")?, max_pushes: p.max_pushes, weighted: p.weighted },
    };
    let dir = match en::<pb::Direction>(g.dir, "direction")? {
        pb::Direction::Out => Direction::Out,
        pb::Direction::In => Direction::In,
        pb::Direction::Both => Direction::Both,
    };
    Ok(Graph {
        field: g.field,
        edges: g.edges,
        seeds: from_ex(g.seeds.map(|e| *e))?,
        kind,
        dir,
        labels: g.labels,
        through: g.through.map(|e| from_ex(Some(*e))).transpose()?,
        max_nodes: g.max_nodes,
        k: g.k,
        min_sim: g.min_sim.map(|x| finite(x, "min_sim")).transpose()?,
        weights: g.weights.into_iter().map(|w| finite(w, "a seed's weight")).collect::<Result<_, ApiError>>()?,
    })
}

fn from_expr(e: E) -> Result<Expr, ApiError> {
    Ok(match e {
        E::Field(f) => Expr::Field(f),
        E::Score(_) => Expr::Score,
        E::Hops(_) => Expr::Hops,
        E::SeqNo(_) => Expr::SeqNo,
        E::Output(n) => Expr::Output(n),
        E::Excluded(f) => Expr::Excluded(f),
        E::Lit(v) => Expr::Lit(lit(v)?),
        E::Not(x) => Expr::Not(Box::new(from_ex(Some(*x))?)),
        E::And(xs) => Expr::And(from_exprs(xs.v)?),
        E::Or(xs) => Expr::Or(from_exprs(xs.v)?),
        E::AtLeast(a) => Expr::AtLeast { need: a.need, conds: from_exprs(a.conds)? },
        E::IsNull(x) => Expr::IsNull(Box::new(from_ex(Some(*x))?)),
        E::Cmp(c) => {
            let c = *c;
            let op = match en::<pb::CmpOp>(c.op, "comparison")? {
                pb::CmpOp::Eq => CmpOp::Eq,
                pb::CmpOp::Ne => CmpOp::Ne,
                pb::CmpOp::Lt => CmpOp::Lt,
                pb::CmpOp::Le => CmpOp::Le,
                pb::CmpOp::Gt => CmpOp::Gt,
                pb::CmpOp::Ge => CmpOp::Ge,
            };
            Expr::Cmp { op, left: from_bex(c.left)?, right: from_bex(c.right)? }
        }
        E::Between(b) => {
            let b = *b;
            Expr::Between { expr: from_bex(b.expr)?, lo: from_opt_bex(b.lo)?, hi: from_opt_bex(b.hi)? }
        }
        E::In(i) => {
            let i = *i;
            Expr::In { expr: from_bex(i.expr)?, list: from_exprs(i.list)? }
        }
        E::Arith(a) => {
            let a = *a;
            let op = match en::<pb::ArithOp>(a.op, "arithmetic operator")? {
                pb::ArithOp::Add => ArithOp::Add,
                pb::ArithOp::Sub => ArithOp::Sub,
                pb::ArithOp::Mul => ArithOp::Mul,
                pb::ArithOp::Div => ArithOp::Div,
                pb::ArithOp::Mod => ArithOp::Mod,
            };
            Expr::Arith { op, left: from_bex(a.left)?, right: from_bex(a.right)? }
        }
        E::Neg(x) => Expr::Neg(Box::new(from_ex(Some(*x))?)),
        E::Concat(xs) => Expr::Concat(from_exprs(xs.v)?),
        E::Like(l) => {
            let l = *l;
            Expr::Like { expr: from_bex(l.expr)?, pattern: from_bex(l.pattern)?, case_insensitive: l.case_insensitive }
        }
        E::Prefix(p) => {
            let p = *p;
            let edits = u8::try_from(p.edits).ok().filter(|&e| e <= 2).ok_or_else(|| bad(format!("a prefix allows 0 to 2 edits, not {}", p.edits)))?;
            Expr::Prefix { expr: from_bex(p.expr)?, prefix: from_bex(p.prefix)?, edits }
        }
        E::Case(c) => {
            let c = *c;
            let whens = c.whens.into_iter().map(|w| Ok(When { when: from_ex(w.when)?, then: from_ex(w.then)? })).collect::<Result<_, ApiError>>()?;
            Expr::Case { operand: from_opt_bex(c.operand)?, whens, otherwise: from_opt_bex(c.otherwise)? }
        }
        E::Call(c) => {
            let func = match en::<pb::Func>(c.func, "function")? {
                pb::Func::Lower => Func::Lower,
                pb::Func::Upper => Func::Upper,
                pb::Func::Length => Func::Length,
                pb::Func::Abs => Func::Abs,
                pb::Func::Round => Func::Round,
                pb::Func::Coalesce => Func::Coalesce,
                pb::Func::Ln => Func::Ln,
                pb::Func::Log => Func::Log,
                pb::Func::Exp => Func::Exp,
                pb::Func::Sqrt => Func::Sqrt,
                pb::Func::Pow => Func::Pow,
                pb::Func::Least => Func::Least,
                pb::Func::Greatest => Func::Greatest,
                pb::Func::Sigmoid => Func::Sigmoid,
                pb::Func::Clamp => Func::Clamp,
                pb::Func::Dot => Func::Dot,
                pb::Func::SidAt => Func::SidAt,
                pb::Func::CountTrue => Func::CountTrue,
            };
            Expr::Call { func, args: from_exprs(c.args)? }
        }
        E::Match(m) => {
            let (field, query, expand) = from_match(m)?;
            Expr::Match { field, query, expand }
        }
        E::Graph(g) => Expr::Graph(Box::new(from_graph(*g)?)),
        E::Bm25(b) => Expr::Bm25 { field: b.field, text: b.text, opts: from_opts(b.opts)? },
        E::Similarity(s) => {
            let v = need(s.vector, "a similarity's vector")?.v;
            if v.iter().any(|x| !x.is_finite()) {
                return Err(bad("a vector's components must be finite"));
            }
            Expr::Similarity { field: s.field, vector: v.into(), opts: from_opts(s.opts)? }
        }
        E::Maxsim(m) => {
            let m = *m;
            Expr::MaxSim { field: m.field, vectors: Box::new(from_ex(m.vectors.map(|e| *e))?), opts: from_opts(m.opts)? }
        }
        E::Sparse(s) => {
            let terms = s.terms.into_iter().map(|t| Ok((t.term, finite(t.weight as f64, "a term's weight")? as f32))).collect::<Result<_, ApiError>>()?;
            Expr::Sparse { field: s.field, terms, opts: from_opts(s.opts)? }
        }
        E::Decay(d) => {
            let x = need(d.decay, "a decay")?;
            let shape = match en::<pb::DecayShape>(x.shape, "decay shape")? {
                pb::DecayShape::Exp => DecayShape::Exp,
                pb::DecayShape::Gauss => DecayShape::Gauss,
                pb::DecayShape::Linear => DecayShape::Linear,
            };
            let decay = Decay { origin: finite(x.origin, "origin")?, scale: finite(x.scale, "scale")?, offset: finite(x.offset, "offset")?, decay: finite(x.decay, "decay")?, shape };
            Expr::Decay { field: d.field, decay, opts: from_opts(d.opts)? }
        }
        E::Rrf(r) => Expr::Rrf { inputs: from_exprs(r.inputs)?, k: finite(r.k, "k")?, depth: r.depth },
        E::Weighted(w) => Expr::Weighted(w.terms.into_iter().map(|t| Ok((finite(t.weight, "a weight")?, from_ex(t.expr)?))).collect::<Result<_, ApiError>>()?),
        E::Product(xs) => Expr::Product(from_exprs(xs.v)?),
        E::Norm(n) => {
            let kind = match en::<pb::NormKind>(n.kind, "set function")? {
                pb::NormKind::MinMax => NormKind::MinMax,
                pb::NormKind::Rank => NormKind::Rank,
                pb::NormKind::Tmm => NormKind::Tmm,
                pb::NormKind::ZScore => NormKind::ZScore,
            };
            let lo = n.lo.map(|lo| finite(lo, "lo")).transpose()?;
            Expr::Norm { kind, input: Box::new(from_ex(n.input.map(|x| *x))?), lo }
        }
        E::Gbdt(g) => Expr::Gbdt { model: g.model, hash: g.hash, args: from_exprs(g.args)?, open: Default::default() },
        E::CrossEncoder(c) => Expr::CrossEncoder { model: c.model, hash: c.hash, fields: c.fields, query: c.query, open: Default::default() },
        E::Ppr(g) => Expr::Ppr(Box::new(from_graph(*g)?)),
        E::Ranked(r) => Expr::Ranked { keys: r.keys.into_iter().map(lit).collect::<Result<_, _>>()? },
        E::ByKey(b) => Expr::ByKey(
            b.v.into_iter()
                .map(|kv| Ok(KeyValue { key: lit(need(kv.key, "a row's key")?)?, value: kv.value.map(|x| finite(x, "a row's value")).transpose()? }))
                .collect::<Result<_, ApiError>>()?,
        ),
        E::Outer(f) => Expr::Outer(f),
        E::Encode(e) => Expr::Encode {
            model: e.model,
            hash: e.hash,
            rows: Box::new(from_node_in(*need(e.rows, "encode's history")?)?),
            now: e.now.map(|t| finite(t, "encode's now")).transpose()?,
            open: Default::default(),
        },
        E::Subquery(n) => Expr::Subquery(Box::new(from_node_in(*n)?)),
        E::Values(n) => Expr::Values(Box::new(from_node_in(*n)?)),
        E::Keys(n) => Expr::Keys(Box::new(from_node_in(*n)?)),
        E::Exists(n) => Expr::Exists(Box::new(from_node_in(*n)?)),
        E::Scalar(n) => Expr::Scalar(Box::new(from_node_in(*n)?)),
    })
}

