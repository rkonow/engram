//! Plans as text, for `EXPLAIN`: a line per node, its input indented under
//! it, with fields by name and expressions in SQL's words.

use std::cell::RefCell;
use std::collections::HashMap;

use super::{Agg, ArithOp, CmpOp, Expr, Graph, GraphKind, GuardSite, JoinKey, JoinKind, Node, NormKind, OnConflict, Plan, SortKey};
use crate::query::logical::{AggFn, DecayShape, Direction, Expand, Fuzzy, Opts};
use crate::query::sql::ident;
use crate::schema::{FieldType, Logical, Schema};
use crate::value::Value;

/// Rows of an insert shown in full; the rest are counted.
const ROWS_SHOWN: usize = 5;

impl Plan {
    /// The plan as lines of text, the root first and each node's input
    /// indented under it, then the file of each ranking model it names.
    /// `schemas` name the fields of the tables the plan reads (a field of
    /// another table shows as `#id`).
    pub fn explain(&self, schemas: &[&Schema]) -> Vec<String> {
        let mut lines = Text { schemas, refs: RefCell::default() }.node(&self.root).0;
        self.root.visit_exprs(&mut |e| {
            if let Expr::Gbdt { model, hash, .. } | Expr::CrossEncoder { model, hash, .. } = e {
                let line = match hash.as_str() {
                    "" => format!("ranking model: {model}"),
                    h => format!("ranking model: {model} (hash {h})"),
                };
                if !lines.contains(&line) {
                    lines.push(line);
                }
            }
        });
        lines
    }
}

struct Text<'a> {
    schemas: &'a [&'a Schema],
    /// The columns of the plans a With names.
    refs: RefCell<HashMap<String, Names<'a>>>,
}

/// What the expressions over a node name: the fields of the table a
/// Search reads, or the columns a node gives (an Aggregate's, a
/// Project's, a Join's).
#[derive(Clone, Default)]
struct Names<'a> {
    schema: Option<&'a Schema>,
    outputs: Vec<String>,
}

fn indent(lines: Vec<String>) -> impl Iterator<Item = String> {
    lines.into_iter().map(|l| format!("  {l}"))
}

impl<'a> Text<'a> {
    fn schema(&self, table: &str) -> Option<&'a Schema> {
        self.schemas.iter().copied().find(|s| s.table.eq_ignore_ascii_case(table))
    }

    fn node(&self, n: &Node) -> (Vec<String>, Names<'a>) {
        let over = |input: &Node, head: &dyn Fn(&Names<'a>) -> String| {
            let (lines, names) = self.node(input);
            (std::iter::once(head(&names)).chain(indent(lines)).collect::<Vec<_>>(), names)
        };
        match n {
            Node::Search(s) => {
                let names = Names { schema: self.schema(&s.table), outputs: Vec::new() };
                let mut lines = vec![format!("Search {}", ident(&s.table))];
                if !s.filter.is_always() {
                    lines.push(format!("  filter: {}", self.expr(&s.filter, &names)));
                }
                if !s.order.is_empty() {
                    lines.push(format!("  order: {}", self.keys(&s.order, &names)));
                }
                // Where each phase runs: a second phase on each shard's
                // rows, a global one once over every row.
                for p in &s.phases {
                    let at = if p.global { "a global phase: scored once over every row" } else { "a second phase: each shard scores its rows" };
                    lines.push(format!("  rerank: {} DESC over the top {} ({at})", self.expr(&p.expr, &names), p.count));
                }
                if let Some(r) = &s.profile {
                    lines.push(format!("  rank profile: {} (hash {})", ident(&r.name), r.hash));
                }
                if let Some(g) = &s.guard {
                    let site = match &g.site {
                        GuardSite::AtLeast { conds } => format!("AT_LEAST(level, {})", conds.iter().map(|x| self.expr(x, &names)).collect::<Vec<_>>().join(", ")),
                        GuardSite::Match { field: f, text, expand } => format!("MATCH({}, {}{}, min => level)", field(&names, *f), quote(text), expansion(expand)),
                        GuardSite::Bm25 { field: f, text } => format!("the min of bm25({}, {})", field(&names, *f), quote(text)),
                    };
                    let rows: Vec<String> = [("min_rows", g.min_rows), ("max_rows", g.max_rows)].into_iter().filter_map(|(k, v)| v.map(|v| format!("{k} {v}"))).collect();
                    lines.push(format!("  guard: {site}, the level from all of them down to {} as {} choose", g.need, rows.join(" and ")));
                }
                // For one release: `+` between scores meant min-max fusion
                // before.
                if s.order.iter().map(|k| &k.expr).chain(s.phases.iter().map(|p| &p.expr)).any(|e| e.adds_raw_scores()) {
                    lines.push("  note: + adds raw scores; minmax() normalises them, as a sum of scores did before ranking expressions".into());
                }
                (lines, names)
            }
            Node::Aggregate { input, keys, aggs } => {
                let (lines, names) = self.node(input);
                let keys: Vec<String> = keys.iter().map(|k| self.expr(k, &names)).collect();
                let aggs: Vec<String> = aggs.iter().map(|a| self.agg(a, &names)).collect();
                let mut head = "Aggregate".to_string();
                if !keys.is_empty() {
                    head += &format!(" by {}", keys.join(", "));
                }
                if !aggs.is_empty() {
                    head += &format!(": {}", aggs.join(", "));
                }
                let names = Names { schema: names.schema, outputs: keys.into_iter().chain(aggs).collect() };
                (std::iter::once(head).chain(indent(lines)).collect(), names)
            }
            Node::Filter { input, cond } => over(input, &|names| format!("Filter: {}", self.expr(cond, names))),
            Node::Sort { input, keys } => over(input, &|names| format!("Sort: {}", self.keys(keys, names))),
            Node::Limit { input, offset, limit } => over(input, &|_| match (limit, offset) {
                (Some(l), 0) => format!("Limit: {l}"),
                (Some(l), o) => format!("Limit: {l} offset {o}"),
                (None, o) => format!("Offset: {o}"),
            }),
            Node::Project { input, columns } => {
                let (lines, names) = over(input, &|names| {
                    let cols: Vec<String> = columns
                        .iter()
                        .map(|c| {
                            let e = self.expr(&c.expr, names);
                            if e == c.name || e == ident(&c.name) { e } else { format!("{e} AS {}", ident(&c.name)) }
                        })
                        .collect();
                    format!("Project: {}", cols.join(", "))
                });
                (lines, Names { schema: names.schema, outputs: columns.iter().map(|c| ident(&c.name)).collect() })
            }
            Node::Join { left, right, kind, keys, on } => {
                let (l, ln) = self.node(left);
                let (r, rn) = self.node(right);
                let names = Names { schema: None, outputs: ln.outputs.iter().chain(&rn.outputs).cloned().collect() };
                let mut head = match kind {
                    JoinKind::Inner => "Join".to_string(),
                    JoinKind::Left => "Left join".to_string(),
                };
                let mut conds: Vec<String> = keys.iter().map(|k| self.join_key(k, &ln, &rn)).collect();
                conds.extend(on.iter().map(|c| self.operand(c, &names)));
                if !conds.is_empty() {
                    head += &format!(" on {}", conds.join(" AND "));
                }
                (std::iter::once(head).chain(indent(l)).chain(indent(r)).collect(), names)
            }
            Node::With { ctes, input } => {
                let mut lines = vec!["With".to_string()];
                for c in ctes {
                    let (cl, cn) = self.node(&c.input);
                    lines.push(format!("  {} AS", ident(&c.name)));
                    lines.extend(indent(indent(cl).collect()));
                    self.refs.borrow_mut().insert(c.name.to_ascii_lowercase(), cn);
                }
                let (il, names) = self.node(input);
                lines.extend(indent(il));
                (lines, names)
            }
            Node::Ref { name } => {
                let names = self.refs.borrow().get(&name.to_ascii_lowercase()).cloned().unwrap_or_default();
                (vec![format!("Ref {}", ident(name))], Names { schema: None, outputs: names.outputs })
            }
            Node::Distinct { input } => over(input, &|_| "Distinct".into()),
            Node::DistinctOn { input, keys, per } => over(input, &|names| {
                let keys = keys.iter().map(|k| self.expr(k, names)).collect::<Vec<_>>().join(", ");
                match *per {
                    1 => format!("Distinct on: {keys}"),
                    k => format!("At most {k} per: {keys}"),
                }
            }),
            Node::KeyOrder { input, by } => {
                let (il, names) = self.node(input);
                let lines = match &**by {
                    Node::Ref { name } => std::iter::once(format!("Key order of {}", ident(name))).chain(indent(il)).collect(),
                    by => {
                        let (bl, _) = self.node(by);
                        std::iter::once("Key order".to_string()).chain(indent(il)).chain(std::iter::once("  of:".to_string())).chain(indent(indent(bl).collect())).collect()
                    }
                };
                (lines, names)
            }
            Node::Insert { table, rows, on_conflict } => {
                let names = Names { schema: self.schema(table), outputs: Vec::new() };
                let mut head = format!("Insert into {}: {} row{}", ident(table), rows.len(), if rows.len() == 1 { "" } else { "s" });
                match on_conflict {
                    OnConflict::Fail => {}
                    OnConflict::Replace => head += ", replacing a live key",
                    OnConflict::Skip => head += ", skipping a live key",
                    OnConflict::Update(sets) => head += &format!(", on a live key SET {}", self.sets(sets, &names)),
                }
                let mut lines = vec![head];
                if let Some(s) = names.schema {
                    lines.push(format!("  columns: {}", s.fields.iter().map(|f| ident(&f.name)).collect::<Vec<_>>().join(", ")));
                }
                for row in rows.iter().take(ROWS_SHOWN) {
                    lines.push(format!("  ({})", row.iter().map(|e| self.expr(e, &names)).collect::<Vec<_>>().join(", ")));
                }
                if rows.len() > ROWS_SHOWN {
                    lines.push(format!("  … {} more", rows.len() - ROWS_SHOWN));
                }
                (lines, names)
            }
            Node::Update { table, input, sets } => {
                let names = Names { schema: self.schema(table), outputs: Vec::new() };
                let (lines, _) = self.node(input);
                (std::iter::once(format!("Update {} SET {}", ident(table), self.sets(sets, &names))).chain(indent(lines)).collect(), names)
            }
            Node::Delete { table, input } => {
                let (lines, names) = self.node(input);
                (std::iter::once(format!("Delete from {}", ident(table))).chain(indent(lines)).collect(), names)
            }
        }
    }

    fn join_key(&self, k: &JoinKey, left: &Names, right: &Names) -> String {
        let side = |e: &Expr, names: &Names, links: bool| {
            let x = self.operand(e, names);
            if links { format!("targets({x})") } else { x }
        };
        format!("{} = {}", side(&k.left, left, k.left_links), side(&k.right, right, k.right_links))
    }

    /// A subquery's plan on one line.
    fn subquery(&self, n: &Node) -> String {
        let (lines, _) = self.node(n);
        format!("({})", lines.iter().map(|l| l.trim()).collect::<Vec<_>>().join("; "))
    }

    fn keys(&self, keys: &[SortKey], names: &Names) -> String {
        keys.iter()
            .map(|k| format!("{}{}{}", self.expr(&k.expr, names), if k.desc { " DESC" } else { "" }, if k.nulls_first { " NULLS FIRST" } else { "" }))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn sets(&self, sets: &[super::Set], names: &Names) -> String {
        sets.iter().map(|s| format!("{} = {}", field(names, s.field), self.expr(&s.value, names))).collect::<Vec<_>>().join(", ")
    }

    fn agg(&self, a: &Agg, names: &Names) -> String {
        let arg = a.arg.as_ref().map_or("*".to_string(), |e| self.expr(e, names));
        match a.func {
            AggFn::Count => format!("COUNT({arg})"),
            AggFn::CountDistinct => format!("COUNT(DISTINCT {arg})"),
            AggFn::Sum => format!("SUM({arg})"),
            AggFn::Avg => format!("AVG({arg})"),
            AggFn::Min => format!("MIN({arg})"),
            AggFn::Max => format!("MAX({arg})"),
        }
    }

    /// `e`, in parentheses if it is AND, OR or an operator's.
    fn operand(&self, e: &Expr, names: &Names) -> String {
        match e {
            Expr::And(_) | Expr::Or(_) | Expr::Weighted(_) | Expr::Arith { .. } | Expr::Concat(_) | Expr::Like { .. } => format!("({})", self.expr(e, names)),
            e => self.expr(e, names),
        }
    }

    fn expr(&self, e: &Expr, names: &Names) -> String {
        match e {
            Expr::Field(f) => field(names, *f),
            Expr::Score => "score()".into(),
            Expr::Hops => "hops()".into(),
            Expr::SeqNo => "seq_no()".into(),
            Expr::Output(i) => names.outputs.get(*i as usize).cloned().unwrap_or_else(|| format!("#{i}")),
            Expr::Excluded(f) => format!("excluded.{}", field(names, *f)),
            Expr::Lit(v) => lit(v),
            Expr::Not(x) => match &**x {
                Expr::IsNull(y) => format!("{} IS NOT NULL", self.operand(y, names)),
                x @ (Expr::Field(_) | Expr::Lit(_) | Expr::Match { .. } | Expr::Graph(_) | Expr::AtLeast { .. }) => format!("NOT {}", self.expr(x, names)),
                x => format!("NOT ({})", self.expr(x, names)),
            },
            Expr::And(xs) => xs.iter().map(|x| self.operand(x, names)).collect::<Vec<_>>().join(" AND "),
            Expr::Or(xs) => xs.iter().map(|x| self.operand(x, names)).collect::<Vec<_>>().join(" OR "),
            Expr::AtLeast { need, conds } => format!("AT_LEAST({need}, {})", conds.iter().map(|x| self.expr(x, names)).collect::<Vec<_>>().join(", ")),
            Expr::IsNull(x) => format!("{} IS NULL", self.operand(x, names)),
            Expr::Cmp { op, left, right } => {
                let op = match op {
                    CmpOp::Eq => "=",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                };
                format!("{} {op} {}", self.operand(left, names), self.operand(right, names))
            }
            Expr::Between { expr, lo, hi } => {
                let x = self.operand(expr, names);
                match (lo, hi) {
                    (Some(lo), Some(hi)) if lo == hi => format!("{x} = {}", self.expr(lo, names)),
                    (Some(lo), Some(hi)) => format!("{x} BETWEEN {} AND {}", self.expr(lo, names), self.expr(hi, names)),
                    (Some(lo), None) => format!("{x} >= {}", self.expr(lo, names)),
                    (None, Some(hi)) => format!("{x} <= {}", self.expr(hi, names)),
                    (None, None) => format!("{x} BETWEEN -inf AND inf"),
                }
            }
            Expr::In { expr, list } => match &list[..] {
                [v @ (Expr::Values(_) | Expr::Keys(_))] => format!("{} IN {}", self.operand(expr, names), self.expr(v, names)),
                [v] => format!("{} = {}", self.operand(expr, names), self.expr(v, names)),
                vs => format!("{} IN ({})", self.operand(expr, names), vs.iter().map(|v| self.expr(v, names)).collect::<Vec<_>>().join(", ")),
            },
            Expr::Arith { op, left, right } => {
                let op = match op {
                    ArithOp::Add => "+",
                    ArithOp::Sub => "-",
                    ArithOp::Mul => "*",
                    ArithOp::Div => "/",
                    ArithOp::Mod => "%",
                };
                format!("{} {op} {}", self.operand(left, names), self.operand(right, names))
            }
            Expr::Neg(x) => format!("-{}", self.operand(x, names)),
            Expr::Concat(xs) => xs.iter().map(|x| self.operand(x, names)).collect::<Vec<_>>().join(" || "),
            Expr::Like { expr, pattern, case_insensitive } => {
                format!("{} {} {}", self.operand(expr, names), if *case_insensitive { "ILIKE" } else { "LIKE" }, self.operand(pattern, names))
            }
            Expr::Prefix { expr, prefix, edits: 0 } => format!("PREFIX({}, {})", self.expr(expr, names), self.expr(prefix, names)),
            Expr::Prefix { expr, prefix, edits } => format!("PREFIX({}, {}, fuzzy => {edits})", self.expr(expr, names), self.expr(prefix, names)),
            Expr::Case { operand, whens, otherwise } => {
                let mut s = "CASE".to_string();
                if let Some(o) = operand {
                    s += &format!(" {}", self.operand(o, names));
                }
                for w in whens {
                    s += &format!(" WHEN {} THEN {}", self.expr(&w.when, names), self.expr(&w.then, names));
                }
                if let Some(x) = otherwise {
                    s += &format!(" ELSE {}", self.expr(x, names));
                }
                s + " END"
            }
            Expr::Call { func, args } => format!("{}({})", func.name(), args.iter().map(|x| self.expr(x, names)).collect::<Vec<_>>().join(", ")),
            Expr::Gbdt { model, args, .. } => format!("gbdt({})", std::iter::once(quote(model)).chain(args.iter().map(|x| self.expr(x, names))).collect::<Vec<_>>().join(", ")),
            Expr::CrossEncoder { model, fields, query, .. } => {
                format!("cross_encoder({})", std::iter::once(quote(model)).chain(fields.iter().map(|f| field(names, *f))).chain(std::iter::once(quote(query))).collect::<Vec<_>>().join(", "))
            }
            Expr::Norm { kind: NormKind::Tmm, input, lo: Some(lo) } => format!("tmm({}, lo => {lo})", self.expr(input, names)),
            Expr::Norm { kind, input, .. } => format!("{}({})", kind.name(), self.expr(input, names)),
            Expr::Match { field: f, query, expand } => format!("MATCH({}, {}{})", field(names, *f), quote(query), expansion(expand)),
            Expr::Graph(g) | Expr::Ppr(g) => self.graph(g, names),
            Expr::Bm25 { field: f, text, opts } => format!("bm25({}, {}{})", field(names, *f), quote(text), options(opts)),
            Expr::Similarity { field: f, vector, opts } => format!("similarity({}, {}{})", field(names, *f), lit(&Value::Vector(vector.clone())), options(opts)),
            Expr::MaxSim { field: f, vectors, opts } => {
                let vectors = match vectors.as_ref() {
                    // A few vectors, and how many there are.
                    Expr::Lit(Value::List(vs)) if vs.len() > 2 => format!("ARRAY[{}, … ({} vectors)]", vs.iter().take(2).map(lit).collect::<Vec<_>>().join(", "), vs.len()),
                    e => self.expr(e, names),
                };
                format!("maxsim({}, {vectors}{})", field(names, *f), options(opts))
            }
            Expr::Sparse { field: f, terms, opts } => {
                let terms: serde_json::Map<String, serde_json::Value> = terms.iter().map(|(t, w)| (t.clone(), serde_json::json!(w))).collect();
                format!("similarity({}, {}{})", field(names, *f), quote(&serde_json::Value::Object(terms).to_string()), options(opts))
            }
            Expr::Decay { field: f, decay, .. } => {
                let shape = match decay.shape {
                    DecayShape::Exp => "exp",
                    DecayShape::Gauss => "gauss",
                    DecayShape::Linear => "linear",
                };
                // A TIMESTAMP's origin is a time, and its scale and offset
                // seconds, as SQL gives them.
                let time = names.schema.and_then(|s| s.fields.get(*f as usize)).is_some_and(|fl| fl.logical == Some(Logical::Timestamp));
                let (origin, unit) = match time {
                    true => (lit(&Value::Timestamp(decay.origin as i64)), 1e6),
                    false => (decay.origin.to_string(), 1.0),
                };
                format!("decay({}, {origin}, {}, offset => {}, decay => {}, shape => '{shape}')", field(names, *f), decay.scale / unit, decay.offset / unit, decay.decay)
            }
            Expr::Ranked { keys } => format!("ranked({})", lit(&Value::List(keys.clone()))),
            Expr::Rrf { inputs, k, depth } => format!("rrf({}, k => {k}, depth => {depth})", inputs.iter().map(|x| self.expr(x, names)).collect::<Vec<_>>().join(", ")),
            // Min-max fusion, as SQL writes it: each term normalised, unless
            // a term's score keeps raw values (`norm => 'none'`).
            Expr::Weighted(terms) => {
                let raw = terms.iter().any(|(_, x)| match x {
                    Expr::Bm25 { opts, .. } | Expr::Similarity { opts, .. } | Expr::Sparse { opts, .. } | Expr::Decay { opts, .. } => opts.get("norm").and_then(|v| v.as_str()) == Some("none"),
                    _ => false,
                });
                let mut s = String::new();
                for (i, (w, x)) in terms.iter().enumerate() {
                    // A negative weight after the first term is a minus.
                    let w = match i {
                        0 => *w,
                        _ if *w < 0.0 => {
                            s += " - ";
                            -w
                        }
                        _ => {
                            s += " + ";
                            *w
                        }
                    };
                    let x = match raw {
                        true => self.operand(x, names),
                        false => format!("minmax({})", self.expr(x, names)),
                    };
                    s += &match w {
                        w if w == 1.0 => x,
                        w => format!("{w} * {x}"),
                    };
                }
                s
            }
            Expr::Product(xs) => xs.iter().map(|x| self.operand(x, names)).collect::<Vec<_>>().join(" * "),
            Expr::Subquery(n) => self.subquery(n),
            Expr::Values(n) => format!("values of {}", self.subquery(n)),
            Expr::Keys(n) => format!("keys of {}", self.subquery(n)),
            Expr::Exists(n) => format!("EXISTS {}", self.subquery(n)),
            Expr::Scalar(n) => self.subquery(n),
            Expr::Outer(n) => format!("outer ${n}"),
            Expr::ByKey(vs) => format!("by key ({} rows)", vs.len()),
            Expr::Encode { model, rows, now, .. } => match now {
                Some(t) => format!("encode('{model}', {}, now => {t})", self.subquery(rows)),
                None => format!("encode('{model}', {})", self.subquery(rows)),
            },
        }
    }

    fn graph(&self, g: &Graph, names: &Names) -> String {
        let f = g.edges.as_deref().map_or_else(|| field(names, g.field), ident);
        let seeds = self.seeds(&g.seeds, names);
        let dir = match g.dir {
            Direction::Out => "out",
            Direction::In => "in",
            Direction::Both => "both",
        };
        let mut s = match &g.kind {
            GraphKind::Reachable { min_hops, max_hops } => format!("REACHABLE({f}, {seeds}, hops => {max_hops}, min_hops => {min_hops}, dir => '{dir}', max_nodes => {}", g.max_nodes),
            GraphKind::Path { to, max_hops } => format!("PATH({f}, {seeds}, {}, max_hops => {max_hops}, dir => '{dir}', max_nodes => {}", self.seeds(to, names), g.max_nodes),
            GraphKind::Ppr { alpha, eps, max_pushes, weighted } => {
                format!("ppr({f}, {seeds}, alpha => {alpha}, eps => {eps}, max_pushes => {max_pushes}, weighted => {}, dir => '{dir}'", *weighted as u8)
            }
        };
        if !g.weights.is_empty() {
            // A few weights, and how many there are.
            let shown: Vec<String> = g.weights.iter().take(8).map(|w| format!("{w:?}")).collect();
            let more = if g.weights.len() > 8 { format!(", … ({} weights)", g.weights.len()) } else { String::new() };
            s += &format!(", weights => ({}{more})", shown.join(", "));
        }
        if !g.labels.is_empty() {
            s += &format!(", labels => {}", quote(&g.labels.join(",")));
        }
        if let Some(t) = &g.through {
            s += &format!(", through => {}", self.expr(t, names));
        }
        let vector = g.edges.is_none() && names.schema.and_then(|sc| sc.fields.get(g.field as usize)).is_some_and(|fl| matches!(fl.ty, FieldType::Vector { .. }));
        if vector {
            s += &format!(", k => {}", g.k);
        }
        if let Some(m) = g.min_sim {
            s += &format!(", min_sim => {m}");
        }
        s + ")"
    }

    fn seeds(&self, e: &Expr, names: &Names) -> String {
        match e {
            Expr::Lit(Value::List(keys)) if keys.len() == 1 => lit(&keys[0]),
            Expr::Lit(Value::List(keys)) => format!("({})", keys.iter().map(lit).collect::<Vec<_>>().join(", ")),
            e => self.expr(e, names),
        }
    }
}

fn field(names: &Names, f: u32) -> String {
    names.schema.and_then(|s| s.fields.get(f as usize)).map_or_else(|| format!("#{f}"), |fl| ident(&fl.name))
}

fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// A value as SQL writes it; a long vector by its first components.
fn lit(v: &Value) -> String {
    match v {
        Value::Null => "NULL".into(),
        Value::Bool(b) => if *b { "TRUE" } else { "FALSE" }.into(),
        Value::Int(x) => x.to_string(),
        Value::Float(x) => format!("{x:?}"),
        Value::Str(s) => quote(s),
        Value::Timestamp(t) => format!("TIMESTAMP {}", quote(&crate::timestamp::format(*t))),
        Value::Vector(xs) => {
            let shown: Vec<String> = xs.iter().take(4).map(|x| format!("{x:?}")).collect();
            if xs.len() > 4 {
                format!("[{}, … ({} values)]", shown.join(", "), xs.len())
            } else {
                format!("[{}]", shown.join(", "))
            }
        }
        Value::List(xs) => format!("ARRAY[{}]", xs.iter().map(lit).collect::<Vec<_>>().join(", ")),
    }
}

fn expansion(e: &Expand) -> String {
    let mut s = String::new();
    if e.synonyms {
        s += ", synonyms => 1";
    }
    match e.fuzzy {
        None => {}
        Some(Fuzzy::Auto) => s += ", fuzzy => 'auto'",
        Some(Fuzzy::Edits(n)) => s += &format!(", fuzzy => {n}"),
    }
    s
}

fn options(opts: &Opts) -> String {
    opts.iter().map(|(k, v)| format!(", {k} => {}", lit(v))).collect()
}
