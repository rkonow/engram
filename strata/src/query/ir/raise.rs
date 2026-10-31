//! The statements the executor runs, from plans: [`Plan::to_select`] and
//! [`Plan::to_write`]. A plan [`Plan::of_select`] or [`Plan::of_write`]
//! made gives back the statement it was made of; a plan from elsewhere is
//! checked against the schema as the binder checks SQL.

use super::{err, Agg, ArithOp, CmpOp, Expr, Func, Graph, GraphKind, Node, NormKind, OnConflict, Plan};
use crate::error::Result;
use crate::query::logical::{self, AggFn, GraphQuery, Having, OrderBy, Pred, Projection, Score, Seeds, Select, Tiebreak, Vectors, CROSS_ENCODER_FIRST};
use crate::query::rank::Rank;
use crate::query::sql::{self, Assign, Conflict, Write};
use crate::schema::{FieldType, Schema};
use crate::value::Value;

impl Plan {
    /// The SELECT of `schema`'s table this plan runs as. The executor runs
    /// `Project [Limit] Search` and `Project [Limit] [Sort] [Filter]
    /// Aggregate Search`; other shapes need operators of their own.
    pub fn to_select(&self, schema: &Schema) -> Result<Select> {
        Raise::new(schema).select(&self.root)
    }

    /// The INSERT, UPDATE or DELETE of `schema`'s table this plan runs as.
    pub fn to_write(&self, schema: &Schema) -> Result<Write> {
        let r = Raise::new(schema);
        Ok(match &self.root {
            Node::Insert { table, rows, on_conflict } => {
                r.table(table)?;
                let n = schema.fields.len();
                let docs = rows
                    .iter()
                    .map(|row| {
                        if row.len() != n {
                            return Err(err(format!("an inserted row has {} values, and {} has {n} fields", row.len(), schema.table)));
                        }
                        row.iter().enumerate().map(|(f, e)| r.stored(f, r.lit(e)?)).collect()
                    })
                    .collect::<Result<_>>()?;
                let conflict = match on_conflict {
                    OnConflict::Fail => Conflict::Fail,
                    OnConflict::Replace => Conflict::Replace,
                    OnConflict::Skip => Conflict::Skip,
                    OnConflict::Update(sets) => Conflict::Update(sets.iter().map(|s| r.assign(s, true)).collect::<Result<_>>()?),
                };
                Write::Insert { docs, conflict }
            }
            Node::Update { table, input, sets } => {
                r.table(table)?;
                let sets = sets.iter().map(|s| r.assign(s, false)).collect::<Result<_>>()?;
                Write::Update { select: Raise::new(schema).select(input)?, sets }
            }
            Node::Delete { table, input } => {
                r.table(table)?;
                Write::Delete { select: Raise::new(schema).select(input)? }
            }
            n => return Err(err(format!("a {} plan is not a write", name(n)))),
        })
    }
}

/// A node's name, for messages.
fn name(n: &Node) -> &'static str {
    match n {
        Node::Search(_) => "Search",
        Node::Aggregate { .. } => "Aggregate",
        Node::Filter { .. } => "Filter",
        Node::Sort { .. } => "Sort",
        Node::Limit { .. } => "Limit",
        Node::Project { .. } => "Project",
        Node::Insert { .. } => "Insert",
        Node::Update { .. } => "Update",
        Node::Delete { .. } => "Delete",
        Node::Join { .. } => "Join",
        Node::With { .. } => "With",
        Node::Ref { .. } => "Ref",
        Node::Distinct { .. } => "Distinct",
        Node::DistinctOn { .. } => "DistinctOn",
        Node::KeyOrder { .. } => "KeyOrder",
    }
}

const SHAPES: &str = "the executor runs Project [Limit] Search and Project [Limit] [Sort] [Filter] Aggregate Search (other plans run with operators)";

struct Raise<'a> {
    schema: &'a Schema,
    /// The graph computations raised so far (`Select::graphs`).
    graphs: Vec<GraphQuery>,
}

impl<'a> Raise<'a> {
    fn new(schema: &'a Schema) -> Self {
        Raise { schema, graphs: Vec::new() }
    }

    fn table(&self, t: &str) -> Result<()> {
        if t.eq_ignore_ascii_case(&self.schema.table) {
            Ok(())
        } else {
            Err(err(format!("the plan reads table {t}, not {}", self.schema.table)))
        }
    }

    fn field(&self, f: u32) -> Result<usize> {
        let f = f as usize;
        if f < self.schema.fields.len() {
            Ok(f)
        } else {
            Err(err(format!("{} has no field {f}", self.schema.table)))
        }
    }

    fn field_of(&self, e: &Expr) -> Result<usize> {
        match e {
            Expr::Field(f) => self.field(*f),
            e => Err(err(format!("expected a field, not {e:?}"))),
        }
    }

    fn lit(&self, e: &Expr) -> Result<Value> {
        match e {
            Expr::Lit(v) => Ok(v.clone()),
            e => Err(err(format!("expected a value, not {e:?}"))),
        }
    }

    /// A value of field `f` as stored: a BOOL's 1 or 0, a TIMESTAMP's
    /// microseconds.
    fn stored(&self, f: usize, v: Value) -> Result<Value> {
        let field = &self.schema.fields[f];
        match field.logical {
            Some(_) => field.store(v).map_err(|e| err(e.to_string())),
            None => Ok(v),
        }
    }

    fn select(mut self, node: &Node) -> Result<Select> {
        let shape = |n: &Node, want: &str| err(format!("{SHAPES}: this plan has a {} where a {want} goes", name(n)));
        let Node::Project { input, columns } = node else { return Err(shape(node, "Project")) };
        let mut n: &Node = input;
        let (mut offset, mut limit) = (0, None);
        if let Node::Limit { input, offset: o, limit: l } = n {
            (offset, limit) = (*o as usize, l.map(|l| l as usize));
            n = input;
        }
        let mut sort = None;
        if let Node::Sort { input, keys } = n {
            sort = Some(keys);
            n = input;
        }
        let mut having = None;
        if let Node::Filter { input, cond } = n {
            having = Some(cond);
            n = input;
        }
        let mut aggregate = None;
        if let Node::Aggregate { input, keys, aggs } = n {
            aggregate = Some((keys, aggs));
            n = input;
        }
        let Node::Search(search) = n else { return Err(shape(n, "Search")) };
        self.table(&search.table)?;
        let filter = self.filter(&search.filter)?;
        let guard = search.guard.as_deref().map(|g| self.guard(g)).transpose()?;
        // A cap the executor can keep: a single-valued field of the table,
        // under a ranked, ungrouped Search; any other is left to the node
        // above.
        let per = search.per.and_then(|c| {
            let f = self.schema.fields.get(c.field as usize)?;
            (!f.multi && c.k > 0).then(|| logical::Per { field: c.field as usize, k: usize::try_from(c.k).unwrap_or(usize::MAX) })
        });
        let pk = self.schema.primary_key;
        let mut phases = Vec::new();
        let (projection, group_by, having, order, tiebreak) = match aggregate {
            None => {
                if let Some(n) = sort.map(|_| "Sort").or(having.map(|_| "Filter")) {
                    return Err(err(format!("{SHAPES}: this plan has a {n} over a Search; order a Search by its own keys")));
                }
                let projection = columns
                    .iter()
                    .map(|c| {
                        Ok((
                            c.name.clone(),
                            match &c.expr {
                                Expr::Field(f) => Projection::Field(self.field(*f)?),
                                Expr::Score => Projection::Score,
                                Expr::Hops => Projection::Hops(self.hops_graph()?),
                                Expr::SeqNo => Projection::SeqNo,
                                // A ranking expression (feature
                                // RANK_FEATURES), which scores each row by
                                // itself.
                                e => match self.rank(e, true).map(Score::fused_shapes) {
                                    Ok(s) if s.has_cross_encoder() => return Err(err("a cross-encoder ranks in a rerank phase, not in a column: rank by it and select score()")),
                                    Ok(s) if Rank::is_global(&s) => {
                                        return Err(err(format!("a column of a Search scores each row by itself: minmax(), rank(), rrf() and weighted sums rank rows against each other, not {e:?}")))
                                    }
                                    Ok(s) => Projection::Feature(s),
                                    Err(crate::error::Error::Query(x)) => return Err(err(format!("a column of a Search is a field, score(), hops(), seq_no() or a ranking expression: {x}"))),
                                    Err(x) => return Err(x),
                                },
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let order = match search.order.first() {
                    None => None,
                    Some(k) if k.expr == Expr::Hops => Some(OrderBy::Score { score: Score::Hops { graph: self.hops_graph()?, far_first: k.desc }, desc: true, nulls_first: false }),
                    Some(k) => Some(OrderBy::Score { score: self.score(&k.expr)?, desc: k.desc, nulls_first: k.nulls_first }),
                };
                if matches!(&order, Some(OrderBy::Score { score, .. }) if score.has_cross_encoder()) {
                    return Err(err(CROSS_ENCODER_FIRST));
                }
                let tiebreak = match &search.order[..] {
                    [] | [_] => None,
                    [_, k] if pk.is_some() && k.expr == Expr::Field(pk.unwrap() as u32) => Some(Tiebreak::Key { desc: k.desc }),
                    _ => return Err(err("a Search orders by a score, then possibly by the primary key")),
                };
                phases = self.phases(&search.phases, order.is_some())?;
                (projection, Vec::new(), None, order, tiebreak)
            }
            Some((keys, aggs)) => {
                if !search.order.is_empty() || !search.phases.is_empty() {
                    return Err(err("a Search under an Aggregate is not ordered"));
                }
                let group_by: Vec<usize> = keys.iter().map(|k| self.field_of(k)).collect::<Result<_>>()?;
                for a in aggs.iter() {
                    match (&a.func, &a.arg) {
                        (AggFn::Count, None) => {}
                        (_, Some(e)) => {
                            self.field_of(e)?;
                        }
                        (f, None) => return Err(err(format!("{f:?} takes a field"))),
                    }
                }
                let out = |e: &Expr| output(e, &group_by, aggs);
                let projection = columns.iter().map(|c| Ok((c.name.clone(), out(&c.expr)?))).collect::<Result<Vec<_>>>()?;
                let having = having.map(|h| self.having(h, &out)).transpose()?;
                let keys = sort.map_or(&[][..], |k| &k[..]);
                if keys.iter().any(|k| k.nulls_first) {
                    return Err(err("NULLS FIRST applies to ordering a Search by a column"));
                }
                let first_group = |e: &Expr| !group_by.is_empty() && *e == Expr::Output(0);
                let order = match keys.first() {
                    None => None,
                    Some(k) if first_group(&k.expr) => Some(OrderBy::GroupValue { desc: k.desc }),
                    Some(k) => match out(&k.expr)? {
                        Projection::Count => Some(OrderBy::Count { desc: k.desc }),
                        what => Some(OrderBy::Agg { what, desc: k.desc }),
                    },
                };
                let tiebreak = match keys {
                    [] | [_] => None,
                    [_, k] if first_group(&k.expr) => Some(Tiebreak::GroupValue { desc: k.desc }),
                    [_, k] if group_by.is_empty() && pk.is_some() && k.expr == Expr::Field(pk.unwrap() as u32) => Some(Tiebreak::Key { desc: k.desc }),
                    _ => return Err(err("groups are ordered by one key, then possibly by the first group key")),
                };
                (projection, group_by, having, order, tiebreak)
            }
        };
        let types = sql::projection_types(self.schema, &projection);
        let per = per.filter(|_| group_by.is_empty() && phases.is_empty() && order.is_some());
        Ok(Select { projection, filter, order, group_by, having, tiebreak, phases, limit, offset, graphs: self.graphs, types, guard, per })
    }

    /// The graph `hops()` refers to: the first REACHABLE or PATH.
    fn hops_graph(&self) -> Result<usize> {
        self.graphs.iter().position(|g| !matches!(g.kind, logical::GraphKind::Ppr { .. })).ok_or_else(|| err("hops() needs a REACHABLE or PATH in the filter"))
    }

    /// A Search's filter, as the executor reads it: NOTs over plain sets,
    /// with NULL already accounted for ([`Pred::three_valued`]).
    fn filter(&mut self, e: &Expr) -> Result<Pred> {
        Ok(self.pred(e)?.simplify())
    }

    fn guard(&mut self, g: &super::Guard) -> Result<logical::Guard> {
        let text_field = |me: &Self, f: u32| -> Result<usize> {
            let i = me.field(f)?;
            match me.schema.fields[i].ty {
                FieldType::Text { .. } if me.schema.fields[i].indexed => Ok(i),
                _ => Err(err(format!("a guard's MATCH or bm25 reads an indexed TEXT field, not {}", me.schema.fields[i].name))),
            }
        };
        let site = match &g.site {
            super::GuardSite::AtLeast { conds } if conds.is_empty() => return Err(err("a guard's AT_LEAST has conditions")),
            super::GuardSite::AtLeast { conds } => logical::GuardSite::AtLeast(conds.iter().map(|c| self.filter(c)).collect::<Result<_>>()?),
            super::GuardSite::Match { field, text, expand } => {
                let field = text_field(self, *field)?;
                crate::text::query::parse_match_with(text, self.schema.fields[field].text_analyzer())?;
                logical::GuardSite::Match { field, text: text.clone(), expand: *expand }
            }
            super::GuardSite::Bm25 { field, text } => logical::GuardSite::Bm25 { field: text_field(self, *field)?, text: text.clone() },
        };
        if g.min_rows.is_none() && g.max_rows.is_none() {
            return Err(err("a guard has min_rows, max_rows or both"));
        }
        Ok(logical::Guard { site, need: g.need, min_rows: g.min_rows, max_rows: g.max_rows })
    }

    fn pred(&mut self, e: &Expr) -> Result<Pred> {
        Ok(match e {
            Expr::Lit(Value::Bool(b)) => {
                if *b {
                    Pred::True
                } else {
                    Pred::False
                }
            }
            Expr::And(xs) => {
                // `x IS NOT NULL AND NOT (x ...)` is a negated comparison
                // read with SQL's three values already: its NOT is a plain
                // complement.
                let guarded: Vec<&Expr> = xs
                    .iter()
                    .filter_map(|x| match x {
                        Expr::Not(y) => match &**y {
                            Expr::IsNull(f) => Some(&**f),
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect();
                let mut ps = Vec::with_capacity(xs.len());
                for x in xs {
                    ps.push(match x {
                        Expr::Not(y) if comparison_field(y).is_some_and(|f| guarded.contains(&f)) => Pred::Not(Box::new(self.pred(y)?)),
                        x => self.pred(x)?,
                    });
                }
                Pred::And(ps)
            }
            Expr::Or(xs) => Pred::Or(xs.iter().map(|x| self.pred(x)).collect::<Result<_>>()?),
            Expr::AtLeast { need, conds } => Pred::AtLeast { need: *need as usize, items: conds.iter().map(|x| self.pred(x)).collect::<Result<_>>()? },
            Expr::IsNull(x) => Pred::Not(Box::new(Pred::Present { field: self.present_field(x)? })),
            Expr::Not(x) => match &**x {
                Expr::IsNull(y) => Pred::Present { field: self.present_field(y)? },
                // A comparison with NULL is neither true nor false, and so
                // is its negation.
                x => {
                    let p = Pred::Not(Box::new(self.pred(x)?));
                    let multi = |f: usize| self.schema.fields[f].multi;
                    p.three_valued(&multi)
                }
            },
            Expr::Between { expr, lo, hi } => {
                let f = self.field_of(expr)?;
                match self.schema.fields[f].ty {
                    FieldType::Float => {
                        let bound = |b: &Option<Box<Expr>>, open: f64| -> Result<f64> {
                            match b {
                                None => Ok(open),
                                Some(b) => self.lit(b)?.as_f64().ok_or_else(|| err(format!("{} is a FLOAT: compare it with numbers", self.schema.fields[f].name))),
                            }
                        };
                        Pred::Range { field: f, lo: bound(lo, f64::NEG_INFINITY)?, hi: bound(hi, f64::INFINITY)? }
                    }
                    FieldType::Int => {
                        let lo = match lo {
                            None => i64::MIN as i128,
                            Some(b) => sql::ceil(&self.int_value(f, b)?),
                        };
                        let hi = match hi {
                            None => i64::MAX as i128,
                            Some(b) => sql::floor(&self.int_value(f, b)?),
                        };
                        sql::int_range(f, lo, hi)
                    }
                    _ => return Err(err(format!("BETWEEN needs an INT or FLOAT field, not {}", self.schema.fields[f].name))),
                }
            }
            Expr::In { expr, list } => {
                let f = self.field_of(expr)?;
                // A NULL is kept for three_valued and simplify to read.
                let values = list
                    .iter()
                    .map(|v| match self.lit(v)? {
                        Value::Null => Ok(Value::Null),
                        v => self.stored(f, v),
                    })
                    .collect::<Result<Vec<_>>>()?;
                if values.is_empty() {
                    Pred::False
                } else {
                    Pred::In { field: f, values }
                }
            }
            Expr::Cmp { op, left, right } => self.comparison(*op, left, right)?,
            Expr::Match { field, query, expand } => {
                let f = self.field(*field)?;
                let fld = &self.schema.fields[f];
                if !matches!(fld.ty, FieldType::Text { .. }) || !fld.indexed {
                    return Err(err(format!("MATCH needs an indexed TEXT field, not {}", fld.name)));
                }
                crate::text::query::parse_match_with(query, fld.text_analyzer())?;
                Pred::Match { field: f, query: query.clone(), expand: *expand }
            }
            Expr::Prefix { expr, prefix, edits } => {
                let f = self.field_of(expr)?;
                let fld = &self.schema.fields[f];
                if fld.ty != FieldType::Keyword || !fld.indexed {
                    return Err(err(format!("a Search's PREFIX needs an indexed KEYWORD field, not {}", fld.name)));
                }
                if *edits > 2 {
                    return Err(err(format!("PREFIX allows 0 to 2 edits, not {edits}")));
                }
                match self.lit(prefix)? {
                    // The empty prefix (`IS NOT NULL` once simplified), so
                    // that its negation leaves out rows without a value.
                    Value::Str(p) if p.is_empty() && *edits == 0 => Pred::Prefix { field: f, prefix: p, edits: 0 },
                    Value::Str(p) if p.is_empty() => Pred::Present { field: f },
                    Value::Str(p) => Pred::Prefix { field: f, prefix: p, edits: *edits },
                    v => return Err(err(format!("PREFIX of {} takes text, not {v:?}", fld.name))),
                }
            }
            Expr::Graph(g) if !matches!(g.kind, GraphKind::Ppr { .. }) => Pred::Graph { id: self.graph(g)? },
            e => return Err(err(format!("a Search cannot filter by {e:?}"))),
        })
    }

    /// The INT field's bound `b`: a number, or a value of a BOOL or
    /// TIMESTAMP field's type, as the integer stored.
    fn int_value(&self, f: usize, b: &Expr) -> Result<Value> {
        let v = self.lit(b)?;
        if self.schema.fields[f].logical.is_some() {
            return self.stored(f, v);
        }
        match v {
            v @ (Value::Int(_) | Value::Float(_)) => Ok(v),
            v => Err(err(format!("{} is an INT: compare it with numbers, not {v:?}", self.schema.fields[f].name))),
        }
    }

    /// The field of `IS NULL`: an indexed INT, FLOAT or KEYWORD field.
    fn present_field(&self, e: &Expr) -> Result<usize> {
        let f = self.field_of(e)?;
        let fld = &self.schema.fields[f];
        if !matches!(fld.ty, FieldType::Int | FieldType::Float | FieldType::Keyword) || !fld.indexed {
            return Err(err(format!("IS NULL works on indexed INT, FLOAT and KEYWORD fields, not {}", fld.name)));
        }
        Ok(f)
    }

    /// `field op value` (or `value op field`), as the binder reads a
    /// comparison.
    fn comparison(&self, op: CmpOp, left: &Expr, right: &Expr) -> Result<Pred> {
        let (f, v, op) = match (left, right) {
            (Expr::Field(f), v) => (self.field(*f)?, self.lit(v)?, op),
            (v, Expr::Field(f)) => {
                let flipped = match op {
                    CmpOp::Lt => CmpOp::Gt,
                    CmpOp::Le => CmpOp::Ge,
                    CmpOp::Gt => CmpOp::Lt,
                    CmpOp::Ge => CmpOp::Le,
                    o => o,
                };
                (self.field(*f)?, self.lit(v)?, flipped)
            }
            _ => return Err(err("a Search compares a field with a value")),
        };
        if v == Value::Null {
            return Err(err("a comparison with NULL is never true: use IS NULL"));
        }
        let fld = &self.schema.fields[f];
        let not = |p: Pred| Pred::Not(Box::new(p)).three_valued(&|f| self.schema.fields[f].multi);
        Ok(match fld.ty {
            FieldType::Keyword => {
                let p = Pred::In { field: f, values: vec![v] };
                match op {
                    CmpOp::Eq => p,
                    CmpOp::Ne => not(p),
                    _ => return Err(err("keyword fields support =, != and IN")),
                }
            }
            FieldType::Int => {
                let v = self.int_value(f, &Expr::Lit(v))?;
                let (lo, hi) = (sql::ceil(&v), sql::floor(&v));
                let (min, max) = (i64::MIN as i128, i64::MAX as i128);
                match op {
                    CmpOp::Eq => sql::int_range(f, lo, hi),
                    CmpOp::Ne => not(sql::int_range(f, lo, hi)),
                    CmpOp::Gt => sql::int_range(f, hi + 1, max),
                    CmpOp::Ge => sql::int_range(f, lo, max),
                    CmpOp::Lt => sql::int_range(f, min, lo - 1),
                    CmpOp::Le => sql::int_range(f, min, hi),
                }
            }
            FieldType::Float => {
                let x = v.as_f64().ok_or_else(|| err(format!("{} is a FLOAT: compare it with numbers", fld.name)))?;
                let range = |lo, hi| Pred::Range { field: f, lo, hi };
                let (inf, ninf) = (f64::INFINITY, f64::NEG_INFINITY);
                match op {
                    CmpOp::Eq => range(x, x),
                    CmpOp::Ne => not(range(x, x)),
                    CmpOp::Gt => range(sql::next_up(x), inf),
                    CmpOp::Ge => range(x, inf),
                    CmpOp::Lt => range(ninf, sql::next_down(x)),
                    CmpOp::Le => range(ninf, x),
                }
            }
            _ => return Err(err(format!("cannot compare field {}", fld.name))),
        })
    }

    fn graph(&mut self, g: &Graph) -> Result<usize> {
        let field = self.field(g.field)?;
        match &g.edges {
            // The edge table is checked against this one when the walk runs.
            Some(_) if self.schema.primary_key != Some(field) => return Err(err("a walk over an edge table names the primary key as its field")),
            Some(_) if !g.labels.is_empty() => return Err(err("the edges of an edge table have no labels")),
            Some(_) => {}
            None if !matches!(self.schema.fields[field].ty, FieldType::Edges | FieldType::Vector { .. }) => {
                return Err(err(format!("a traversal needs an EDGES or VECTOR field, not {}", self.schema.fields[field].name)));
            }
            None => {}
        }
        if self.schema.fields[field].is_multi_vector() {
            return Err(err(format!("a traversal follows one vector per row: {} is a multi-vector field", self.schema.fields[field].name)));
        }
        let seeds = match (self.seeds(&g.seeds)?, g.weights.as_slice()) {
            (s, []) => s,
            (Seeds::Keys(keys), ws) if keys.len() == ws.len() && ws.iter().all(|w| w.is_finite()) => Seeds::Weighted(keys.into_iter().zip(ws.iter().copied()).collect()),
            _ => return Err(err("a traversal's weights are finite numbers, one per key of its seeds")),
        };
        let kind = match &g.kind {
            GraphKind::Reachable { min_hops, max_hops } => logical::GraphKind::Reach { min_hops: *min_hops, max_hops: *max_hops },
            GraphKind::Path { to, max_hops } => logical::GraphKind::Path { to: self.seeds(to)?, max_hops: *max_hops },
            GraphKind::Ppr { alpha, eps, max_pushes, weighted } => logical::GraphKind::Ppr { alpha: *alpha, eps: *eps, max_pushes: *max_pushes, weighted: *weighted },
        };
        let through = match &g.through {
            None => None,
            Some(t) => {
                let mut inner = Raise::new(self.schema);
                let p = inner.filter(t)?;
                if !inner.graphs.is_empty() {
                    return Err(err("a through condition cannot hold a traversal"));
                }
                Some(p)
            }
        };
        if g.max_nodes == 0 || g.k == 0 {
            return Err(err("a traversal's max_nodes and k are 1 or more"));
        }
        let g = GraphQuery {
            field,
            edges: g.edges.clone(),
            seeds,
            kind,
            dir: g.dir,
            labels: g.labels.clone(),
            through,
            max_nodes: g.max_nodes as usize,
            k: g.k as usize,
            min_sim: g.min_sim.unwrap_or(f64::NEG_INFINITY),
        };
        // The same computation named twice (in a column and the order)
        // runs once, as SQL binds it.
        if let Some(i) = self.graphs.iter().position(|x| *x == g) {
            return Ok(i);
        }
        self.graphs.push(g);
        Ok(self.graphs.len() - 1)
    }

    fn seeds(&self, e: &Expr) -> Result<Seeds> {
        Ok(match e {
            Expr::Lit(Value::List(keys)) => Seeds::Keys(keys.clone()),
            Expr::Lit(key) => Seeds::Keys(vec![key.clone()]),
            Expr::Subquery(n) => Seeds::Query(Box::new(Raise::new(self.schema).select(n)?)),
            e => return Err(err(format!("seeds are keys or a subquery, not {e:?}"))),
        })
    }

    /// A Search's rerank phases, after its order: each reranks at least one
    /// row, and one is global where it holds a set function (`minmax`,
    /// `rank`, `rrf`, a weighted sum) or follows a global one.
    fn phases(&mut self, phases: &[super::Phase], ordered: bool) -> Result<Vec<logical::Phase>> {
        if !phases.is_empty() && !ordered {
            return Err(err("rerank phases follow a Search's order: give it one"));
        }
        let mut after_global = false;
        let mut out = Vec::with_capacity(phases.len());
        for p in phases {
            if p.count == 0 {
                return Err(err("a rerank phase reranks at least one row"));
            }
            let score = self.score(&p.expr)?;
            if !p.global && after_global {
                return Err(err("a rerank phase after a global one is global"));
            }
            if !p.global && Rank::is_global(&score) {
                return Err(err("a rerank phase with minmax(), rank(), rrf(), a weighted sum or a cross-encoder reads every row it reranks: it is global"));
            }
            after_global |= p.global;
            out.push(logical::Phase { score, count: usize::try_from(p.count).unwrap_or(usize::MAX), global: p.global });
        }
        Ok(out)
    }

    /// A Search's ranking, its sums of `w * minmax(score)` and products of
    /// scores in their fused shapes, as SQL binds them.
    fn score(&mut self, e: &Expr) -> Result<Score> {
        Ok(self.rank(e, false)?.fused_shapes())
    }

    /// A Search's ranking, or part of one: `in_expr` within arithmetic, a
    /// function, a set function or CASE (feature
    /// [`super::RANK_EXPRESSIONS`]), where `hops()` is the number of hops,
    /// rather than at the top or within RRF, a weighted sum or a product,
    /// where it ranks nearest first.
    fn rank(&mut self, e: &Expr, in_expr: bool) -> Result<Score> {
        let sub = |this: &mut Self, x: &Expr| this.rank(x, true).map(Box::new);
        Ok(match e {
            // NULL, as a scalar subquery of no row is.
            Expr::Lit(Value::Null) => Score::Num(f64::NAN),
            Expr::Lit(v) => match v.as_f64() {
                Some(x) if x.is_finite() && !matches!(v, Value::Str(_)) => Score::Num(x),
                _ => return Err(err(format!("a number in a ranking expression is finite, not {v:?}"))),
            },
            Expr::SeqNo => Score::SeqNo,
            Expr::Arith { op, left, right } => Score::Arith { op: *op, left: sub(self, left)?, right: sub(self, right)? },
            Expr::Neg(x) => Score::Neg(sub(self, x)?),
            // Each condition 1 where it holds, as WHERE would keep the row.
            Expr::Call { func: Func::CountTrue, args } if !args.is_empty() => {
                Score::Call { func: Func::CountTrue, args: args.iter().map(|a| Ok(Score::Cond(Box::new(self.filter(a)?)))).collect::<Result<_>>()? }
            }
            Expr::Call { func, args } => {
                if Func::numeric(func.name()).is_none() {
                    return Err(err(format!("{}() takes text: a ranking expression is numeric", func.name())));
                }
                func.arity(args.len()).map_err(|n| err(format!("{}() takes {n}", func.name())))?;
                Score::Call { func: *func, args: args.iter().map(|a| self.rank(a, true)).collect::<Result<_>>()? }
            }
            Expr::Norm { kind, input, lo } => {
                let input = sub(self, input)?;
                // tmm reads the least value of its input, the input's own
                // unless the plan gives one.
                let lo = match (kind, lo) {
                    (NormKind::Tmm, Some(lo)) if lo.is_finite() => Some(*lo),
                    (NormKind::Tmm, None) => match input.least(self.schema) {
                        Some(lo) => Some(lo),
                        None => return Err(err("tmm of a ranking with no least value of its own needs lo")),
                    },
                    (NormKind::Tmm, Some(lo)) => return Err(err(format!("tmm's lo is a finite number, not {lo}"))),
                    _ => None,
                };
                Score::Norm { kind: *kind, input, lo }
            }
            Expr::Gbdt { model, hash, args, open } => {
                let Some(m) = &open.0 else {
                    return Err(err(format!("ranking model {model} is not open: a database or coordinator opens a plan's models before it runs it (Db::run)")));
                };
                if args.len() != m.features {
                    return Err(err(format!("ranking model {model} reads {} features, not {}", m.features, args.len())));
                }
                Score::Gbdt { model: model.clone(), hash: hash.clone(), args: args.iter().map(|a| self.rank(a, true)).collect::<Result<_>>()?, open: open.clone() }
            }
            // Opened where it runs: a shard leaves it to its coordinator.
            Expr::CrossEncoder { model, hash, fields, query, open } => {
                if fields.is_empty() {
                    return Err(err(format!("cross-encoder {model} reads the text of at least one field")));
                }
                let fields = fields
                    .iter()
                    .map(|&f| {
                        let f = self.typed(f, |t| matches!(t, FieldType::Text { .. } | FieldType::Keyword), "cross_encoder", "TEXT or KEYWORD")?;
                        match self.schema.fields[f].stored {
                            true => Ok(f),
                            false => Err(err(format!("cross_encoder reads a stored field's text, and {} is not stored", self.schema.fields[f].name))),
                        }
                    })
                    .collect::<Result<_>>()?;
                Score::CrossEncoder { model: model.clone(), hash: hash.clone(), fields, query: query.clone(), open: open.clone() }
            }
            Expr::Case { operand: Some(_), .. } => return Err(err("CASE in a ranking expression takes WHEN conditions, not an operand")),
            Expr::Case { operand: None, whens, otherwise } => Score::Case {
                whens: whens.iter().map(|w| Ok((Score::Cond(Box::new(self.filter(&w.when)?)), self.rank(&w.then, true)?))).collect::<Result<_>>()?,
                otherwise: otherwise.as_ref().map(|x| sub(self, x)).transpose()?,
            },
            Expr::Hops if in_expr => Score::Hops { graph: self.hops_graph()?, far_first: true },
            Expr::Field(f) => {
                let f = self.field(*f)?;
                if !matches!(self.schema.fields[f].ty, FieldType::Int | FieldType::Float) || self.schema.fields[f].multi {
                    return Err(err(format!("order by a single-valued numeric field, not {}", self.schema.fields[f].name)));
                }
                Score::Column { field: f }
            }
            Expr::Bm25 { field, text, opts } => Score::Bm25 { field: self.typed(*field, |t| matches!(t, FieldType::Text { .. }), "bm25", "TEXT")?, text: text.clone(), opts: opts.clone() },
            Expr::Similarity { field, vector, opts } => {
                let f = self.typed(*field, |t| matches!(t, FieldType::Vector { .. }), "similarity", "VECTOR")?;
                if self.schema.fields[f].multi {
                    return Err(err(crate::query::sql::multi_similarity(&self.schema.fields[f].name)));
                }
                if let FieldType::Vector { dim, .. } = self.schema.fields[f].ty {
                    if vector.len() != dim {
                        return Err(err(format!("vector has {} dimensions, field has {dim}", vector.len())));
                    }
                }
                Score::Similarity { field: f, vector: vector.clone(), opts: opts.clone() }
            }
            Expr::MaxSim { field, vectors, opts } => {
                let f = self.typed(*field, |t| matches!(t, FieldType::Vector { .. }), "maxsim", "VECTOR")?;
                let FieldType::Vector { dim, .. } = self.schema.fields[f].ty else { unreachable!("a VECTOR field") };
                let Expr::Lit(Value::List(vs)) = vectors.as_ref() else { return Err(err("maxsim's query vectors are a list of vectors, once its subquery ran")) };
                let vectors = vs
                    .iter()
                    .map(|v| match v {
                        Value::Vector(v) if v.len() == dim => Ok(v.clone()),
                        Value::Vector(v) => Err(err(format!("a query vector has {} dimensions, field has {dim}", v.len()))),
                        v => Err(err(format!("maxsim's query vectors are vectors, not {v:?}"))),
                    })
                    .collect::<Result<_>>()?;
                Score::MaxSim { field: f, vectors: Vectors::List(vectors), opts: opts.clone() }
            }
            Expr::Sparse { field, terms, opts } => Score::Sparse { field: self.typed(*field, |t| *t == FieldType::Sparse, "similarity", "SPARSE")?, terms: terms.clone(), opts: opts.clone() },
            Expr::Decay { field, decay, opts } => Score::Decay { field: self.typed(*field, |t| matches!(t, FieldType::Int | FieldType::Float), "decay", "INT or FLOAT")?, decay: *decay, opts: opts.clone() },
            Expr::Rrf { inputs, k, depth } => Score::Rrf { inputs: inputs.iter().map(|s| self.score(s)).collect::<Result<_>>()?, k: *k, depth: *depth as usize },
            Expr::Weighted(terms) => Score::Weighted(terms.iter().map(|(w, s)| Ok((*w, self.score(s)?))).collect::<Result<_>>()?),
            Expr::Product(factors) => Score::Product(factors.iter().map(|s| self.score(s)).collect::<Result<_>>()?),
            Expr::Ppr(g) if matches!(g.kind, GraphKind::Ppr { .. }) => Score::Ppr { graph: self.graph(g)? },
            Expr::Hops => Score::Hops { graph: self.hops_graph()?, far_first: false },
            Expr::ByKey(values) => {
                let pk = self.schema.primary_key.ok_or_else(|| err("values by key rank the rows of a table with a primary key"))?;
                let values = values.iter().map(|kv| Ok(super::KeyValue { key: self.stored(pk, kv.key.clone())?, value: kv.value })).collect::<Result<_>>()?;
                Score::ByKey(std::sync::Arc::new(crate::query::logical::ByKey::new(values)))
            }
            Expr::Outer(_) => return Err(err("a correlated subquery runs before the Search it ranks, with each row's values in place")),
            Expr::Ranked { keys } => {
                let pk = self.schema.primary_key.ok_or_else(|| err(format!("ranked(keys) ranks rows by their primary keys, and {} has none", self.schema.table)))?;
                Score::Ranked { keys: keys.iter().map(|k| crate::query::sql::coerced(&self.schema.fields[pk], k.clone())).collect::<Result<_>>()? }
            }
            e => return Err(err(format!("not a score: {e:?}"))),
        })
    }

    fn typed(&self, f: u32, ok: impl Fn(&FieldType) -> bool, func: &str, want: &str) -> Result<usize> {
        let f = self.field(f)?;
        if ok(&self.schema.fields[f].ty) {
            Ok(f)
        } else {
            Err(err(format!("{func} needs a {want} field, not {}", self.schema.fields[f].name)))
        }
    }

    fn having(&self, e: &Expr, out: &dyn Fn(&Expr) -> Result<Projection>) -> Result<Having> {
        let bound = |b: &Option<Box<Expr>>, open: f64| -> Result<f64> {
            match b {
                None => Ok(open),
                Some(b) => self.lit(b)?.as_f64().ok_or_else(|| err("a range of groups has numeric bounds")),
            }
        };
        Ok(match e {
            Expr::And(xs) => Having::And(xs.iter().map(|x| self.having(x, out)).collect::<Result<_>>()?),
            Expr::Or(xs) => Having::Or(xs.iter().map(|x| self.having(x, out)).collect::<Result<_>>()?),
            Expr::Not(x) => Having::Not(Box::new(self.having(x, out)?)),
            Expr::Between { expr, lo, hi } => Having::Range { what: out(expr)?, lo: bound(lo, f64::NEG_INFINITY)?, hi: bound(hi, f64::INFINITY)? },
            Expr::Cmp { op: CmpOp::Eq, left, right } => Having::Eq { what: out(left)?, value: self.lit(right)? },
            Expr::Cmp { op: CmpOp::Ne, left, right } => Having::Not(Box::new(Having::Eq { what: out(left)?, value: self.lit(right)? })),
            // `n > 2` is the range above 2, as SQL's binder reads it.
            Expr::Cmp { op, left, right } => {
                let x = self.lit(right)?.as_f64().ok_or_else(|| err("a range of groups has numeric bounds"))?;
                let (lo, hi) = match op {
                    CmpOp::Gt => (sql::next_up(x), f64::INFINITY),
                    CmpOp::Ge => (x, f64::INFINITY),
                    CmpOp::Lt => (f64::NEG_INFINITY, sql::next_down(x)),
                    _ => (f64::NEG_INFINITY, x),
                };
                Having::Range { what: out(left)?, lo, hi }
            }
            e => return Err(err(format!("groups are kept by ranges and comparisons of their outputs, not {e:?}"))),
        })
    }

    fn assign(&self, s: &super::Set, excluded: bool) -> Result<(usize, Assign)> {
        let f = self.field(s.field)?;
        if Some(f) == self.schema.primary_key {
            return Err(err("the primary key cannot be updated; delete and insert instead"));
        }
        let fld = &self.schema.fields[f];
        let a = match &s.value {
            Expr::Lit(v) => Assign::Value(self.stored(f, v.clone())?),
            // A counter: `f = f + number` counts a NULL as zero.
            Expr::Arith { op: op @ (ArithOp::Add | ArithOp::Sub), left, right } if **left == Expr::Field(f as u32) && matches!(**right, Expr::Lit(_)) => {
                if !matches!(fld.ty, FieldType::Int | FieldType::Float) || fld.logical.is_some() || fld.multi {
                    return Err(err(format!("{} is not a single numeric field", fld.name)));
                }
                match (op, self.lit(right)?) {
                    (ArithOp::Add, v @ (Value::Int(_) | Value::Float(_))) => Assign::Add(v),
                    (ArithOp::Sub, Value::Int(x)) => Assign::Add(x.checked_neg().map_or(Value::Float(-(x as f64)), Value::Int)),
                    (ArithOp::Sub, Value::Float(x)) => Assign::Add(Value::Float(-x)),
                    (_, v) => return Err(err(format!("{} adds a number, not {v:?}", fld.name))),
                }
            }
            Expr::Excluded(g) if excluded => {
                let g = self.field(*g)?;
                let src = &self.schema.fields[g];
                if src.ty != fld.ty || src.multi != fld.multi || src.logical != fld.logical {
                    return Err(err(format!("{} = excluded.{}: the columns have different types", fld.name, src.name)));
                }
                Assign::Excluded(g)
            }
            e => {
                self.row_expr(e, excluded)?;
                Assign::Expr(e.clone())
            }
        };
        Ok((f, a))
    }

    /// Checks an expression `SET` computes from each row: of its fields
    /// and, on conflict, the proposed row's (`excluded.f`), with no search,
    /// score or traversal.
    fn row_expr(&self, e: &Expr, excluded: bool) -> Result<()> {
        let go = |x: &Expr| self.row_expr(x, excluded);
        let all = |xs: &[Expr]| xs.iter().try_for_each(go);
        match e {
            Expr::Field(f) => self.field(*f).map(|_| ()),
            Expr::Excluded(f) if excluded => self.field(*f).map(|_| ()),
            Expr::Excluded(f) => Err(err(format!("excluded.{} is the row an INSERT proposed, in ON CONFLICT DO UPDATE", self.schema.fields[self.field(*f)?].name))),
            Expr::Lit(_) => Ok(()),
            Expr::Not(x) | Expr::IsNull(x) | Expr::Neg(x) => go(x),
            Expr::And(xs) | Expr::Or(xs) | Expr::AtLeast { conds: xs, .. } | Expr::Concat(xs) => all(xs),
            Expr::Cmp { left, right, .. } | Expr::Arith { left, right, .. } => go(left).and_then(|_| go(right)),
            Expr::Like { expr, pattern, .. } | Expr::Prefix { expr, prefix: pattern, .. } => go(expr).and_then(|_| go(pattern)),
            Expr::Between { expr, lo, hi } => [Some(expr), lo.as_ref(), hi.as_ref()].into_iter().flatten().try_for_each(|x| go(x)),
            Expr::In { expr, list } => go(expr).and_then(|_| all(list)),
            Expr::Case { operand, whens, otherwise } => {
                operand.iter().chain(otherwise).try_for_each(|x| go(x))?;
                whens.iter().try_for_each(|w| go(&w.when).and_then(|_| go(&w.then)))
            }
            Expr::Call { args, .. } => all(args),
            e => Err(err(format!("SET takes an expression of the row's fields, not {e:?}"))),
        }
    }
}

/// The field a comparison (BETWEEN, IN or `field op value`) tests.
fn comparison_field(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::Between { expr, .. } | Expr::In { expr, .. } => Some(expr),
        Expr::Cmp { left, right, .. } => Some(if matches!(**left, Expr::Field(_)) { left } else { right }),
        _ => None,
    }
}

/// What an Aggregate's output column `e` holds.
fn output(e: &Expr, keys: &[usize], aggs: &[Agg]) -> Result<Projection> {
    let Expr::Output(i) = e else { return Err(err(format!("above an Aggregate, columns are its outputs, not {e:?}"))) };
    let i = *i as usize;
    if let Some(&f) = keys.get(i) {
        return Ok(Projection::Field(f));
    }
    match aggs.get(i - keys.len()) {
        Some(Agg { func: AggFn::Count, arg: None }) => Ok(Projection::Count),
        Some(Agg { func, arg: Some(Expr::Field(f)) }) => Ok(Projection::Agg { func: *func, field: *f as usize }),
        _ => Err(err(format!("the Aggregate has no output {i}"))),
    }
}
