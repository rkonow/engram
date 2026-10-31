//! Plans as protobuf (`strata_server::plan_pb`), the form a coordinator
//! sends its shards from protocol 2: every kind of statement's plan, and
//! every kind of node and expression, comes back from protobuf (and from
//! its bytes) as the plan it was; and a plan that is newer, incomplete or
//! holds a number that is not finite is refused, as in JSON.

use std::collections::BTreeSet;

use prost::Message;
use strata::db::Db;
use strata::query::ir::{self, Expr, Node, Plan};
use strata::query::{bind, bind_command, compound, Command, Params};
use strata::schema::{Field, FieldType, Schema};
use strata::storage::{DbOptions, RankProfile};
use strata::value::Value;
use strata_server::pb;
use strata_server::plan_pb::{from_pb_plan, to_pb_plan};

const PAPERS: &str = "CREATE TABLE papers (
    id KEYWORD PRIMARY KEY,
    title TEXT OPTIONS(analyzer = 'simple'),
    body TEXT,
    category KEYWORD,
    tags KEYWORD[],
    year INT,
    refs INT[],
    rating FLOAT,
    done BOOL,
    published TIMESTAMP,
    emb VECTOR(4) OPTIONS(index = 'flat'),
    terms SPARSE,
    cites EDGES
)";

fn papers(dir: &std::path::Path) -> Schema {
    let seed = Schema::new("seed", vec![Field::new("id", FieldType::Keyword)], Some("id")).unwrap();
    let db = Db::create(&dir.join("d.strata"), seed, DbOptions { sync: false, ..DbOptions::default() }).unwrap();
    db.sql(PAPERS, &Params::new()).unwrap();
    let schema = db.storage().table("papers").unwrap();
    schema.as_ref().clone()
}

fn params() -> Params {
    let vs = Value::List(vec![Value::Vector(vec![0.1, 0.2, 0.3, 0.4].into()), Value::Vector(vec![0.4, 0.3, 0.2, 0.1].into())]);
    Params::new().set("v", Value::Vector(vec![0.1, 0.2, 0.3, 0.4].into())).set("ids", Value::List(vec![Value::Str("a".into()), Value::Str("b".into())])).set("vs", vs)
}

/// SELECTs of one table.
const SELECTS: &[&str] = &[
    "SELECT * FROM papers",
    "SELECT id, title FROM papers WHERE MATCH(body, 'graph neural') AND year >= 2019 ORDER BY bm25(body, 'graph neural') DESC LIMIT 10",
    "SELECT id, score() AS s FROM papers WHERE MATCH(title, 'Graph', fuzzy => 'auto', synonyms => 1) ORDER BY bm25(body, 'graph', fuzzy => 1) DESC LIMIT 5 OFFSET 2",
    "SELECT id FROM papers WHERE year BETWEEN 2000 AND 2010 AND rating > 0.5 AND NOT (category = 'ml') AND category IN ('a', 'b', 'c')",
    "SELECT id FROM papers WHERE NOT (year > 5 OR rating <= 1.5) OR tags = 'x' OR NOT (refs = 3)",
    "SELECT id FROM papers WHERE year IS NULL OR rating IS NOT NULL OR category IS NULL",
    "SELECT id FROM papers WHERE rating = 2.5 OR rating != 3 OR year != 7 OR year < 1e30 OR year > 9223372036854775807",
    "SELECT id FROM papers WHERE done AND published >= '2026-10-01' AND published < '2026-11-01T00:00:00Z'",
    "SELECT id FROM papers WHERE category IN (:ids) AND TRUE",
    "SELECT id FROM papers WHERE FALSE",
    "SELECT id, rating FROM papers ORDER BY rating DESC NULLS FIRST, id LIMIT 3",
    "SELECT id FROM papers ORDER BY similarity(emb, :v) DESC LIMIT 5",
    "SELECT id, seq_no() FROM papers WHERE year > 2000 LIMIT 5",
    "SELECT id FROM papers ORDER BY rrf(bm25(body, 'graph'), similarity(emb, :v), k => 10, depth => 50) DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY 0.3 * minmax(bm25(body, 'graph')) + 0.7 * minmax(similarity(emb, :v)) - minmax(rating) DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY 0.3 * tmm(bm25(body, 'graph')) + 0.7 * tmm(similarity(emb, :v)) - 0.1 * tmm(rating, lo => 0) DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY zscore(bm25(body, 'graph')) + zscore(similarity(emb, :v)) DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY ln(1 + bm25(body, 'graph')) * (1 + 0.2 * coalesce(rating, 0)) - 0.1 * rank(similarity(emb, :v)) + CASE WHEN category = 'ml' THEN 0.5 ELSE -seq_no() / 1e9 END DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY bm25(body, 'graph') * decay(year, 2020, 5) DESC LIMIT 5",
    // A ranking given as keys.
    "SELECT id FROM papers ORDER BY rrf(bm25(body, 'graph'), ranked(:ids)) DESC LIMIT 5",
    "SELECT id FROM papers WHERE year > 2000 ORDER BY ranked(('p3', 'p1')) + 0.1 * coalesce(rating, 0) DESC LIMIT 5",
    // Ranking expressions as columns, and features().
    "SELECT id, bm25(body, 'net') AS b, ln(1 + rating) * 2, features() FROM papers ORDER BY bm25(body, 'graph') * decay(year, 2020, 5) + similarity(emb, :v) DESC LIMIT 5",
    "SELECT id FROM papers ORDER BY decay(year, 2020, 5, offset => 1, decay => 0.3, shape => 'gauss') DESC",
    "SELECT id FROM papers ORDER BY decay(rating, 3, 1, shape => 'linear') DESC",
    "SELECT id FROM papers ORDER BY similarity(terms, '{\"graph\": 1.5, \"net\": 0.5}') DESC LIMIT 3",
    "SELECT id, hops() FROM papers WHERE REACHABLE(cites, 'p1', hops => 2, dir => 'both', labels => 'x,y', through => year > 2000) ORDER BY hops() LIMIT 20",
    "SELECT id FROM papers WHERE REACHABLE(cites, 'p1', dir => 'in')",
    "SELECT id FROM papers WHERE PATH(cites, 'p1', ('p5', 'p6'), max_hops => 4) ORDER BY hops() DESC",
    "SELECT id FROM papers WHERE NOT REACHABLE(cites, ('p1', 'p2'), hops => 0, min_hops => 0) AND REACHABLE(emb, 'p1', k => 5, min_sim => 0.5)",
    "SELECT id FROM papers ORDER BY ppr(cites, (SELECT id FROM papers WHERE MATCH(body, 'graph') ORDER BY bm25(body, 'graph') DESC LIMIT 3), alpha => 0.2, weighted => 1) DESC LIMIT 10",
    "SELECT id FROM papers WHERE year > 2000 ORDER BY maxsim(emb, :vs, rerank => 50, max_vectors => 8) * (1 + coalesce(rating, 0)) DESC LIMIT 5",
    "SELECT category, COUNT(*) AS n, AVG(rating), MAX(published), MIN(year), SUM(rating) FROM papers WHERE year > 2000 GROUP BY category HAVING COUNT(*) >= 2 AND AVG(rating) < 3 ORDER BY n DESC LIMIT 5",
    "SELECT COUNT(*), COUNT(DISTINCT category), COUNT(rating) FROM papers WHERE MATCH(body, 'graph')",
    // Some of the conditions, and how many hold.
    "SELECT id, count_true(year > 2000, tags = 'x', MATCH(body, 'net')) AS k FROM papers WHERE AT_LEAST(2, MATCH(body, 'graph'), year >= 2019, NOT AT_LEAST('50%', rating > 1, category IN ('a', 'b'))) ORDER BY bm25(body, 'graph') * (1 + count_true(done, rating IS NULL)) DESC LIMIT 5",
    "SELECT id FROM papers WHERE category LIKE 'm%' OR (PREFIX(id, 'p1') AND id NOT LIKE 'p1!_%' ESCAPE '!') ORDER BY year DESC LIMIT 5",
    "SELECT id FROM papers WHERE PREFIX(category, 'mtah', fuzzy => 'auto') OR PREFIX(id, 'p2', fuzzy => 1) ORDER BY year DESC LIMIT 5",
];

/// Compound queries: operators over single-table queries.
const COMPOUND: &[&str] = &[
    "SELECT DISTINCT category FROM papers",
    // Rerank phases: a second phase, then a global one.
    "WITH a AS (SELECT * FROM papers WHERE year > 2000 ORDER BY bm25(body, 'graph') DESC LIMIT 100), b AS (SELECT * FROM a ORDER BY similarity(emb, :v) DESC LIMIT 30) SELECT id, score() FROM b ORDER BY minmax(bm25(body, 'net')) + rating DESC LIMIT 5 OFFSET 1",
    "WITH a AS (SELECT * FROM papers ORDER BY bm25(body, 'graph') DESC LIMIT 100) SELECT id, score(), features() FROM a ORDER BY similarity(emb, :v) * (1 + rating) DESC LIMIT 5",
    "SELECT a.id, b.title FROM papers a JOIN papers b ON a.cites = b.id WHERE a.year >= 2019 ORDER BY a.year DESC LIMIT 5",
    "SELECT a.id, b.id FROM papers a LEFT JOIN papers b ON a.category = b.category AND b.year > a.year WHERE a.title LIKE 'gr%' OR b.title ILIKE '%NET' ORDER BY a.id LIMIT 7 OFFSET 1",
    "SELECT a.id || '/' || b.id, CASE WHEN a.year > b.year THEN 'newer' WHEN a.year IS NULL THEN NULL ELSE 'older' END, upper(a.category), length(b.title), abs(a.year - b.year), round(a.rating / 2), coalesce(a.rating, - -1) % 3 FROM papers a JOIN papers b ON a.category = b.category",
    "WITH top AS (SELECT id FROM papers ORDER BY bm25(body, 'graph') DESC LIMIT 20) SELECT category, COUNT(*) AS n FROM papers WHERE id IN (SELECT id FROM top) GROUP BY category",
    "WITH r AS (SELECT id, category FROM papers WHERE year >= 2000) SELECT DISTINCT category FROM r LIMIT 2",
    "WITH r AS (SELECT id, year FROM papers WHERE MATCH(body, 'graph') ORDER BY bm25(body, 'graph') DESC LIMIT 10) SELECT id, title FROM r",
    "SELECT id FROM papers WHERE year > (SELECT AVG(year) FROM papers) AND EXISTS (SELECT id FROM papers WHERE done)",
    "SELECT id FROM papers WHERE year IN (SELECT MAX(year) FROM papers)",
    "SELECT d.id FROM (SELECT id, year FROM papers WHERE year > 2000) AS d ORDER BY d.year LIMIT 3",
    "SELECT a.category, COUNT(*) FROM papers a JOIN papers b ON a.cites = b.id GROUP BY a.category HAVING COUNT(*) > 1",
    "SELECT DISTINCT a.category, b.year FROM papers a JOIN papers b ON a.cites = b.id ORDER BY b.year LIMIT 4",
    "SELECT id, dot(emb, :v) FROM papers WHERE year > 2000",
    "SELECT id FROM papers ORDER BY rrf(bm25(body, 'graph'), maxsim(emb, (SELECT emb FROM papers WHERE category = 'ml' LIMIT 4))) DESC LIMIT 5",
    "SELECT DISTINCT ON (category, year % 3) id, category FROM papers WHERE year > 1990 ORDER BY bm25(body, 'graph') DESC LIMIT 3",
    "SELECT DISTINCT ON (category) category, year, COUNT(*) AS n FROM papers GROUP BY category, year ORDER BY category, n DESC",
    "SELECT id, category FROM papers WHERE year > 1990 ORDER BY bm25(body, 'graph') DESC MAX 2 PER category LIMIT 5",
    "SELECT d.id FROM (SELECT id, title FROM papers) AS d WHERE PREFIX(d.title, upper('gr'))",
    // A correlated subquery, which names the row it ranks.
    "WITH a AS (SELECT * FROM papers ORDER BY bm25(body, 'graph') DESC LIMIT 50) SELECT id FROM a ORDER BY bm25(body, 'graph') + (SELECT COUNT(*) FROM papers p WHERE p.category = a.category AND p.year < a.year) DESC LIMIT 5",
    // A history's vector by a sequence model, as a query vector and in a
    // dot product.
    "SELECT id FROM papers ORDER BY similarity(emb, encode('seq', (SELECT id, category, published FROM papers WHERE year > 2000 ORDER BY year DESC LIMIT 20), now => 1800000000.5)) DESC LIMIT 5",
    "SELECT id, dot(emb, encode('seq', (SELECT id FROM papers ORDER BY year DESC LIMIT 3))) FROM papers WHERE year > 2000",
];

/// Writes of each kind.
const WRITES: &[&str] = &[
    "INSERT INTO papers (id, title, year, done, published, tags, emb, terms, cites) VALUES ('n1', 'a', 1, TRUE, '2026-10-01', ARRAY['x', 'y'], ARRAY[0.1, 0.2, 0.3, 0.4], '{\"a\": 1}', '[\"p1\"]'), ('n2', 'b', NULL, FALSE, NULL, ARRAY['z'], NULL, NULL, NULL)",
    "INSERT OR REPLACE INTO papers (id, year) VALUES ('p1', 2)",
    "INSERT INTO papers (id, year) VALUES ('p1', 2) ON CONFLICT (id) DO NOTHING",
    "INSERT INTO papers (id, year, rating) VALUES ('p2', 3, 1.5) ON CONFLICT (id) DO UPDATE SET year = CASE WHEN excluded.year > year THEN excluded.year ELSE year END, rating = coalesce(rating, 0) + excluded.rating",
    "UPDATE papers SET rating = rating * 2 + year, title = lower(coalesce(title, 'untitled')) || '!', done = NOT done WHERE category = 'db' LIMIT 5",
    "DELETE FROM papers WHERE done ORDER BY year DESC LIMIT 3",
];

/// `plan` through protobuf, and through its bytes, and back.
fn through_pb(plan: &Plan) -> Plan {
    let p = to_pb_plan(plan);
    let back = from_pb_plan(p.clone()).unwrap_or_else(|e| panic!("{}: {plan:?}", e.message));
    assert_eq!(&back, plan);
    let bytes = p.encode_to_vec();
    assert_eq!(from_pb_plan(pb::Plan::decode(bytes.as_slice()).unwrap()).unwrap(), back);
    back
}

/// The kinds of nodes and expressions in `plan`, by the names its JSON
/// gives them.
fn kinds(plan: &Plan) -> BTreeSet<String> {
    fn walk(j: &serde_json::Value, out: &mut BTreeSet<String>) {
        match j {
            serde_json::Value::Object(m) => {
                if let Some(serde_json::Value::String(n)) = m.get("node") {
                    out.insert(format!("node {n}"));
                }
                // An expression is an object of one key, its kind.
                if m.len() == 1 && !m.contains_key("node") {
                    out.insert(format!("expr {}", m.keys().next().unwrap()));
                }
                m.values().for_each(|v| walk(v, out));
            }
            serde_json::Value::Array(xs) => xs.iter().for_each(|v| walk(v, out)),
            serde_json::Value::String(s) if s == "score" || s == "hops" || s == "seq_no" => {
                out.insert(format!("expr {s}"));
            }
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    walk(&plan.to_json().unwrap(), &mut out);
    out
}

#[test]
fn every_plan_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let schema = papers(dir.path());
    let mut seen = BTreeSet::new();
    let mut check = |plan: Plan| {
        seen.extend(kinds(&plan));
        through_pb(&plan);
    };
    for sql in SELECTS {
        let select = bind(sql, &schema, &params()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        check(Plan::of_select(&schema, &select));
    }
    for sql in COMPOUND {
        assert!(compound::is_compound(sql, &schema), "{sql}");
        check(compound::plan(sql, &schema, &params()).unwrap_or_else(|e| panic!("{sql}: {e}")));
    }
    // Seeds with weights, which operators fill in from a subquery's scores
    // before a plan reaches a shard.
    let sql = "SELECT id FROM papers ORDER BY ppr(cites, ('p1', 'p2'), weighted => 1) DESC LIMIT 5";
    let mut j = Plan::of_select(&schema, &bind(sql, &schema, &params()).unwrap()).to_json().unwrap();
    fn weigh(j: &mut serde_json::Value) -> bool {
        match j {
            serde_json::Value::Object(m) => match m.get_mut("ppr") {
                Some(serde_json::Value::Object(g)) => g.insert("weights".into(), serde_json::json!([0.25, 3.5])).is_none(),
                _ => m.values_mut().any(weigh),
            },
            serde_json::Value::Array(xs) => xs.iter_mut().any(weigh),
            _ => false,
        }
    }
    assert!(weigh(&mut j));
    let weighted = Plan::new(Plan::from_json(&j).unwrap().root);
    assert!(weighted.features.iter().any(|f| f == ir::MAXSIM), "{:?}", weighted.features);
    check(weighted);
    // A walk over an edge table, which names the table; its field is the
    // key.
    let sql = "SELECT id FROM papers ORDER BY ppr(cites, 'p1') DESC LIMIT 5";
    let mut j = Plan::of_select(&schema, &bind(sql, &schema, &params()).unwrap()).to_json().unwrap();
    fn over_table(j: &mut serde_json::Value) -> bool {
        match j {
            serde_json::Value::Object(m) => match m.get_mut("ppr") {
                Some(serde_json::Value::Object(g)) => {
                    g.insert("field".into(), serde_json::json!(0));
                    g.insert("edges".into(), serde_json::json!("cooc")).is_none()
                }
                _ => m.values_mut().any(over_table),
            },
            serde_json::Value::Array(xs) => xs.iter_mut().any(over_table),
            _ => false,
        }
    }
    assert!(over_table(&mut j));
    let walked = Plan::new(Plan::from_json(&j).unwrap().root);
    assert!(walked.features.iter().any(|f| f == ir::EDGE_TABLE), "{:?}", walked.features);
    check(walked);
    for sql in WRITES {
        let Command::Write { write, .. } = bind_command(sql, &[&schema], &params()).unwrap_or_else(|e| panic!("{sql}: {e}")) else { panic!("{sql}") };
        check(Plan::of_write(&schema, &write));
    }
    // The values a correlated subquery took, by key, as operators send
    // them to a shard.
    let sql = "SELECT id FROM papers ORDER BY bm25(body, 'graph') DESC LIMIT 5";
    let mut plan = Plan::of_select(&schema, &bind(sql, &schema, &params()).unwrap());
    let Some(s) = plan.root.search_mut() else { panic!("a leaf") };
    let values = vec![ir::KeyValue { key: Value::Str("p1".into()), value: Some(2.5) }, ir::KeyValue { key: Value::Str("p2".into()), value: None }];
    s.order[0].expr = Expr::Arith { op: ir::ArithOp::Add, left: Box::new(s.order[0].expr.clone()), right: Box::new(Expr::ByKey(values)) };
    let plan = Plan::new(plan.root);
    assert!(plan.features.iter().any(|f| f == ir::CORRELATED), "{:?}", plan.features);
    check(plan);
    // Ranked by a rank profile: the plan names it.
    let profile = RankProfile {
        name: "recent".into(),
        ranking: "WITH a AS (SELECT * FROM papers ORDER BY bm25(body, :t) DESC LIMIT 50) SELECT * FROM a ORDER BY rating * :w DESC".into(),
        defaults: [("w".to_string(), serde_json::json!(0.5))].into(),
    };
    let sql = "SELECT id, score() FROM papers WHERE year > 2000 ORDER BY rank_profile('recent', t => 'graph') LIMIT 5";
    let plan = compound::plan_read(sql, &[&schema], &params(), &|t, n| (t == "papers" && n == "recent").then(|| profile.clone())).unwrap();
    assert!(plan.features.iter().any(|f| f == ir::RANK_PROFILES), "{:?}", plan.features);
    let back = through_pb(&plan);
    let mut named = Vec::new();
    back.root.each_search(&mut |s| named.extend(s.profile.clone()));
    assert_eq!(named, [ir::ProfileRef { name: "recent".into(), hash: profile.hash() }]);
    seen.extend(kinds(&plan));
    // Ranked by a model: the plan names it and the hash of its file.
    let sql = "SELECT id, score(), features() FROM papers ORDER BY gbdt('m', bm25(body, 'graph'), rating, year % 6) DESC LIMIT 5";
    let mut plan = compound::plan(sql, &schema, &params()).unwrap();
    assert!(plan.features.iter().any(|f| f == ir::RANK_MODELS), "{:?}", plan.features);
    plan.root
        .visit_exprs_mut(&mut |e| {
            if let Expr::Gbdt { hash, .. } = e {
                *hash = "0123456789abcdef".into();
            }
            Ok(())
        })
        .unwrap();
    let back = through_pb(&plan);
    let mut models = Vec::new();
    back.root.visit_exprs(&mut |e| {
        if let Expr::Gbdt { model, hash, args, .. } = e {
            models.push((model.clone(), hash.clone(), args.len()));
        }
    });
    // In the ranking, and as a column of features().
    assert_eq!(models, vec![("m".to_string(), "0123456789abcdef".to_string(), 3); 2]);
    seen.extend(kinds(&plan));
    // Reranked by a cross-encoder, in a global phase.
    let sql = "WITH top AS (SELECT * FROM papers ORDER BY bm25(body, 'graph') DESC LIMIT 20) SELECT id, score() FROM top ORDER BY cross_encoder('ce', title, body, 'graph search') DESC LIMIT 5";
    let plan = compound::plan(sql, &schema, &params()).unwrap();
    assert!(plan.features.iter().any(|f| f == ir::RANK_MODELS), "{:?}", plan.features);
    let back = through_pb(&plan);
    let mut found = Vec::new();
    back.root.each_search(&mut |s| found.extend(s.phases.iter().map(|p| (p.count, p.global, p.expr.clone()))));
    let [(20, true, Expr::CrossEncoder { model, fields, query, .. })] = &found[..] else { panic!("{found:?}") };
    assert_eq!((model.as_str(), fields.len(), query.as_str()), ("ce", 2, "graph search"));
    seen.extend(kinds(&plan));
    // Every kind of node and expression was among them.
    let nodes = ["search", "aggregate", "filter", "sort", "limit", "project", "insert", "update", "delete", "join", "with", "ref", "distinct", "key_order"];
    let exprs = [
        "field", "score", "hops", "output", "excluded", "lit", "not", "and", "or", "is_null", "cmp", "between", "in", "arith", "neg", "concat", "like", "case", "call", "match", "graph", "bm25", "similarity", "sparse", "decay", "rrf", "weighted", "product", "ppr", "subquery", "values", "keys", "exists", "scalar", "seq_no", "norm", "gbdt", "cross_encoder", "maxsim", "prefix", "outer", "by_key", "at_least", "encode",
    ];
    let want: BTreeSet<String> = nodes.iter().map(|n| format!("node {n}")).chain(exprs.iter().map(|e| format!("expr {e}"))).collect();
    let missing: Vec<&String> = want.difference(&seen).collect();
    assert!(missing.is_empty(), "never made: {missing:?}");
}

/// A plan made by hand of the given root.
fn plan(root: Node) -> pb::Plan {
    to_pb_plan(&Plan::new(root))
}

fn search(filter: Expr) -> Node {
    Node::Search(ir::Search { table: "papers".into(), filter, order: Vec::new(), phases: Vec::new(), profile: None, guard: None, per: None })
}

fn refused(p: pb::Plan, says: &str) {
    match from_pb_plan(p) {
        Ok(p) => panic!("read {p:?}; wanted it refused with {says:?}"),
        Err(e) => {
            assert_eq!(e.status, 400, "{}", e.message);
            assert!(e.message.contains(says), "{:?} lacks {says:?}", e.message);
        }
    }
}

#[test]
fn plans_that_cannot_be_read_are_refused() {
    let ok = plan(search(Expr::Lit(Value::Bool(true))));
    from_pb_plan(ok.clone()).unwrap();
    // Newer, or needing what this version lacks, by name.
    refused(pb::Plan { version: ir::VERSION + 1, ..ok.clone() }, "plan version 2");
    refused(pb::Plan { version: 0, ..ok.clone() }, "plan version 0");
    refused(pb::Plan { features: vec!["teleport".into()], ..ok.clone() }, "\"teleport\"");
    // Incomplete.
    refused(pb::Plan { root: None, ..ok.clone() }, "a node is missing");
    refused(pb::Plan { root: Some(pb::Node { node: None }), ..ok.clone() }, "a node is missing");
    let mut p = plan(Node::Limit { input: Box::new(search(Expr::Lit(Value::Bool(true)))), offset: 0, limit: Some(3) });
    let Some(pb::node::Node::Limit(l)) = &mut p.root.as_mut().unwrap().node else { panic!() };
    l.input = None;
    refused(p, "a node's input is missing");
    let mut p = plan(Node::Filter { input: Box::new(search(Expr::Lit(Value::Bool(true)))), cond: Expr::Hops });
    let Some(pb::node::Node::Filter(f)) = &mut p.root.as_mut().unwrap().node else { panic!() };
    f.cond = Some(Box::new(pb::Expr { expr: None }));
    refused(p, "an expression is missing");
    // Numbers that are not finite, as JSON refuses them.
    refused(plan(search(Expr::Lit(Value::Float(f64::NAN)))), "finite");
    refused(plan(search(Expr::Lit(Value::List(vec![Value::Float(f64::INFINITY)])))), "finite");
    refused(plan(search(Expr::Lit(Value::Vector(vec![0.5, f32::NAN].into())))), "finite");
    let sim = |v: Vec<f32>| Expr::Similarity { field: 10, vector: v.into(), opts: Default::default() };
    refused(plan(search(sim(vec![f32::INFINITY]))), "finite");
    refused(plan(search(Expr::Rrf { inputs: vec![sim(vec![1.0])], k: f64::INFINITY, depth: 10 })), "k must be finite");
    refused(plan(search(Expr::Weighted(vec![(f64::NAN, sim(vec![1.0]))]))), "a weight must be finite");
    let mut opts = std::collections::BTreeMap::new();
    opts.insert("k1".to_string(), Value::Float(f64::NEG_INFINITY));
    refused(plan(search(Expr::Bm25 { field: 2, text: "graph".into(), opts })), "finite");
    // An enum's number it does not know, and too many edits for a byte.
    let mut p = plan(search(Expr::Cmp { op: ir::CmpOp::Eq, left: Box::new(Expr::Field(5)), right: Box::new(Expr::Lit(Value::Int(1))) }));
    let Some(pb::node::Node::Search(s)) = &mut p.root.as_mut().unwrap().node else { panic!() };
    let Some(pb::expr::Expr::Cmp(c)) = &mut s.filter.as_mut().unwrap().expr else { panic!() };
    c.op = 99;
    refused(p, "99 is not a comparison");
    let mut p = plan(search(Expr::Match { field: 2, query: "graph".into(), expand: strata::query::Expand { synonyms: false, fuzzy: Some(strata::query::Fuzzy::Edits(1)) } }));
    let Some(pb::node::Node::Search(s)) = &mut p.root.as_mut().unwrap().node else { panic!() };
    let Some(pb::expr::Expr::Match(m)) = &mut s.filter.as_mut().unwrap().expr else { panic!() };
    m.expand.as_mut().unwrap().fuzzy = Some(pb::expand::Fuzzy::Edits(300));
    refused(p, "300 edits");
}

/// A search's filter that is absent holds for every row, as in JSON.
#[test]
fn a_search_without_a_filter_reads_every_row() {
    let mut p = plan(search(Expr::Lit(Value::Bool(true))));
    let Some(pb::node::Node::Search(s)) = &mut p.root.as_mut().unwrap().node else { panic!() };
    s.filter = None;
    assert_eq!(from_pb_plan(p).unwrap(), Plan::new(search(Expr::Lit(Value::Bool(true)))));
}
