//! Ranking expressions (`ORDER BY bm25(body, :t) * (1 + 0.2 * rating)`):
//! one database answers like the same rows split over 1, 2 and 4 shards for
//! every shape, a NULL makes a score NULL and ranks its row last, and a
//! reader without the plan feature refuses an expression it cannot run.

use std::sync::Arc;

use strata::cluster;
use strata::data::{self, col};
use strata::db::Db;
use strata::query::exec::{ExecOptions, ResultSet};
use strata::query::ir::{Plan, OPERATORS, RANK_EXPRESSIONS, SCORE_NORMS, SEQ_NO};
use strata::query::stats::TermStats;
use strata::query::{bind, Params};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

const N: usize = 2400;
const DIM: usize = 16;
const SEG: usize = 300;

fn open(dir: &std::path::Path, name: &str, docs: &[strata::Document]) -> Db {
    let opts = DbOptions { segment_max_docs: SEG, ..DbOptions::default() };
    let db = Db::create(&dir.join(name), data::beir_schema(DIM, VectorIndexKind::Flat), opts).unwrap();
    db.load(docs.iter().cloned()).unwrap();
    db
}

/// The corpus, every seventh row without a rating.
fn docs() -> Vec<strata::Document> {
    let mut docs = data::synthetic(N, DIM, 11);
    for d in docs.iter_mut().step_by(7) {
        d[col::RATING] = Value::Null;
    }
    docs
}

/// The coordinator's rounds over `shards`.
fn sharded(shards: &[Db], sql: &str, params: &Params) -> ResultSet {
    let snaps: Vec<_> = shards.iter().map(|d| d.snapshot()).collect();
    let select = bind(sql, snaps[0].schema(), params).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let parts: Vec<TermStats> = shards.iter().zip(&snaps).map(|(d, s)| d.term_stats(s, &select).unwrap()).collect();
    let opts = ExecOptions { stats: Some(Arc::new(TermStats::sum(&parts))), shard: true, ..ExecOptions::default() };
    let answers = shards.iter().zip(&snaps).map(|(d, s)| d.execute(s, &select, &opts).unwrap_or_else(|e| panic!("{sql}: {e}"))).collect();
    cluster::merge(&select, answers).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The same scores, in order, to 1e-9 relative (NULL as −∞); keys in the
/// same order, except that rows tied on score may swap within their tie,
/// and a tie cut by the page may hold different rows.
fn same(sql: &str, whole: &ResultSet, split: &ResultSet) {
    assert_eq!(whole.keys.len(), split.keys.len(), "{sql}: row counts");
    let close = |a: f64, b: f64| a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
    for (i, (a, b)) in whole.scores.iter().zip(&split.scores).enumerate() {
        assert!(close(*a, *b), "{sql}: row {i}: {a} vs {b}");
    }
    let mut i = 0;
    while i < whole.keys.len() {
        let mut j = i + 1;
        while j < whole.keys.len() && close(whole.scores[i], whole.scores[j]) {
            j += 1;
        }
        if j < whole.keys.len() {
            let mut a: Vec<String> = whole.keys[i..j].iter().map(|k| format!("{k:?}")).collect();
            let mut b: Vec<String> = split.keys[i..j].iter().map(|k| format!("{k:?}")).collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "{sql}: rows {i}..{j}");
        }
        i = j;
    }
}

#[test]
fn shards_rank_every_shape_like_one_database() {
    let docs = docs();
    let dir = tempfile::tempdir().unwrap();
    let whole = open(dir.path(), "whole.strata", &docs);
    let splits: Vec<Vec<Db>> = [1, 2, 4]
        .iter()
        .map(|&k| (0..k).map(|i| open(dir.path(), &format!("s{k}-{i}.strata"), &docs[i * N / k..(i + 1) * N / k])).collect())
        .collect();

    let q = docs[42][col::EMB].clone();
    let title = docs[42][col::TITLE].as_str().unwrap().to_string();
    let text = title.split(' ').take(3).collect::<Vec<_>>().join(" ");
    let qs = Value::List([42, 7, 1234].iter().map(|&i| docs[i][col::EMB].clone()).collect());
    let params = Params::new().set("q", q).set("t", Value::Str(text)).set("qs", qs);
    let shapes = [
        "bm25(body, :t) + similarity(emb, :q)",
        "0.3 * minmax(bm25(body, :t)) + 0.7 * minmax(similarity(emb, :q))",
        "0.5 * minmax(bm25(body, :t)) - 0.5 * minmax(rating)",
        "minmax(bm25(body, :t)) + minmax(similarity(emb, :q)) + 0.1 * minmax(rating)",
        "bm25(body, :t) * (1 + 0.2 * coalesce(rating, 0))",
        "ln(1 + bm25(body, :t)) - 0.1 * rating",
        "-rank(bm25(body, :t)) - rank(similarity(emb, :q))",
        "rrf(bm25(body, :t), similarity(emb, :q)) * (1 + 0.1 * coalesce(rating, 0))",
        "rrf(bm25(body, :t), similarity(emb, :q), rating, k => 10, depth => 40)",
        "CASE WHEN category = 'c01' THEN 2 * bm25(body, :t) ELSE bm25(body, :t) END",
        "CASE WHEN year > 2010 THEN similarity(emb, :q) END",
        "CASE category WHEN 'c02' THEN 1 WHEN 'c03' THEN 0.5 ELSE 0 END + minmax(similarity(emb, :q))",
        "bm25(body, :t) * decay(year, 2015, 3)",
        "sigmoid(bm25(body, :t) - 5) + clamp(rating, 1, 4) / 4",
        "greatest(minmax(bm25(body, :t)), minmax(similarity(emb, :q)))",
        "least(bm25(body, :t), 10) + sqrt(abs(similarity(emb, :q)))",
        "pow(similarity(emb, :q), 2) * 3 - exp(-bm25(body, :t))",
        "rating * 2 + year / 1000",
        "bm25(body, :t, exact => 1) + rating",
        "bm25(body, :t, depth => 30) + similarity(emb, :q)",
        "similarity(emb, :q) / (1 + rank(rating))",
        "round(bm25(body, :t), 0) + round(similarity(emb, :q), 1)",
        "log(2, 1 + bm25(body, :t)) + log(1 + rating)",
        "bm25(body, :t) % 2 + similarity(emb, :q)",
        "minmax(bm25(body, :t) + similarity(emb, :q))",
        "bm25(body, :t) * similarity(emb, :q) * 2",
        "bm25(body, :t) / rating",
        "maxsim(emb, :qs) * (1 + rating)",
        "rrf(bm25(body, :t), maxsim(emb, :qs)) + 0.1 * minmax(rating)",
        "0.3 * tmm(bm25(body, :t)) + 0.7 * tmm(similarity(emb, :q))",
        "0.5 * tmm(bm25(body, :t)) + 0.5 * tmm(maxsim(emb, :qs)) - 0.1 * tmm(rating, lo => 0)",
        "0.2 * tmm(bm25(body, :t), lo => 0, hi => 20) + 0.8 * tmm(similarity(emb, :q), lo => -1, hi => 1)",
        "0.4 * zscore(bm25(body, :t)) + 0.6 * zscore(similarity(emb, :q))",
        "zscore(similarity(emb, :q)) + 0.1 * zscore(rating)",
    ];
    let mut queries = Vec::new();
    for s in shapes {
        queries.push(format!("SELECT id, score() FROM docs ORDER BY {s} DESC LIMIT 10"));
    }
    queries.extend([
        "SELECT id, score() FROM docs WHERE year >= 2005 ORDER BY bm25(body, :t) - 0.5 * minmax(rating) DESC LIMIT 15 OFFSET 5".to_string(),
        "SELECT id, score() FROM docs WHERE category IN ('c00', 'c01') ORDER BY similarity(emb, :q) * coalesce(rating, 1) DESC, id DESC LIMIT 12".into(),
        "SELECT id, score() FROM docs ORDER BY ln(1 + bm25(body, :t)) - 0.1 * rating DESC LIMIT 2000".into(),
    ]);
    let mut nulls = 0;
    for sql in &queries {
        let want = whole.query(sql, &params).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert!(!want.keys.is_empty(), "{sql}");
        nulls += want.scores.iter().filter(|s| **s == f64::NEG_INFINITY).count();
        for shards in &splits {
            same(&format!("{sql} on {} shards", shards.len()), &want, &sharded(shards, sql, &params));
        }
    }
    // Rows with a NULL score were ranked, last, on every split.
    assert!(nulls > 10, "{nulls}");
}

/// A row whose score is NULL (a missing rating, `ln(0)`, a zero divisor)
/// ranks after every other, and score() shows NULL.
#[test]
fn a_null_score_ranks_last() {
    let mut docs = data::synthetic(60, DIM, 3);
    for (i, d) in docs.iter_mut().enumerate() {
        d[col::RATING] = match i % 3 {
            0 => Value::Null,
            1 => Value::Float(0.0),
            _ => Value::Float(i as f64),
        };
    }
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "n.strata", &docs);
    for sql in [
        "SELECT id, score() FROM docs ORDER BY rating * 2 DESC LIMIT 60",
        "SELECT id, score() FROM docs ORDER BY ln(rating) DESC LIMIT 60",
        "SELECT id, score() FROM docs ORDER BY 1 / rating DESC LIMIT 60",
    ] {
        let r = db.query(sql, &Params::new()).unwrap();
        assert_eq!(r.keys.len(), 60, "{sql}");
        let nulls = r.rows.iter().filter(|row| row[1] == Value::Null).count();
        let first_null = r.rows.iter().position(|row| row[1] == Value::Null).unwrap();
        assert_eq!(first_null + nulls, 60, "{sql}: NULL scores are last");
        assert!(r.scores[first_null..].iter().all(|s| *s == f64::NEG_INFINITY), "{sql}");
        assert!(r.scores[..first_null].iter().all(|s| s.is_finite()), "{sql}");
        assert!(r.scores[..first_null].windows(2).all(|w| w[0] >= w[1]), "{sql}");
    }
    // `coalesce` gives the missing a value.
    let r = db.query("SELECT id, score() FROM docs ORDER BY coalesce(rating, 100) DESC LIMIT 20", &Params::new()).unwrap();
    assert!(r.scores.iter().all(|s| *s == 100.0));
}

/// A plan that ranks by an expression needs the `rank_expressions` feature,
/// so a reader from before it refuses the plan by name rather than ranking
/// differently; the shapes from before (a weighted sum of minmax(), a
/// product of scores, RRF) do not need it, and older shards still read
/// them.
#[test]
fn an_old_reader_refuses_an_expression_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "p.strata", &data::synthetic(50, DIM, 5));
    let params = Params::new().set("q", Value::Vector(vec![0.25; DIM].into()));
    let old = [OPERATORS, SEQ_NO];
    let needs = |sql: &str| {
        let plan = db.explain(sql, &params).unwrap();
        let json = plan.to_json().unwrap();
        let plan = Plan::from_json(&json).unwrap();
        let features: Vec<&str> = plan.features.iter().map(String::as_str).collect();
        (features.contains(&RANK_EXPRESSIONS), Plan::readable_by(plan.version as u64, &features, &old).map_err(|e| e.to_string()))
    };
    for sql in [
        "SELECT id FROM docs ORDER BY bm25(body, 'x') + similarity(emb, :q) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY bm25(body, 'x') * (1 + rating) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY minmax(bm25(body, 'x')) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY CASE WHEN year > 2000 THEN 1 ELSE 0 END DESC LIMIT 5",
        "SELECT id FROM docs WHERE id IN (SELECT id FROM docs ORDER BY ln(1 + rating) DESC LIMIT 3)",
    ] {
        let (needed, read) = needs(sql);
        assert!(needed, "{sql}");
        let e = read.unwrap_err();
        assert!(e.contains("rank_expressions") && e.contains("may be newer"), "{sql}: {e}");
    }
    for sql in [
        "SELECT id FROM docs ORDER BY 0.3 * minmax(bm25(body, 'x')) + 0.7 * minmax(similarity(emb, :q)) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY bm25(body, 'x') * decay(year, 2010, 5) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY rrf(bm25(body, 'x'), similarity(emb, :q)) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY rating DESC LIMIT 5",
    ] {
        assert_eq!(needs(sql), (false, Ok(())), "{sql}");
    }
}

/// For one release, EXPLAIN and the plan lines of a query that adds raw
/// scores say that `+` no longer normalises them (it meant min-max fusion
/// before ranking expressions); a sum of minmax() terms, a product and a
/// constant offset say nothing.
#[test]
fn a_sum_of_raw_scores_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "e.strata", &data::synthetic(50, DIM, 5));
    let params = Params::new().set("q", Value::Vector(vec![0.25; DIM].into()));
    let note = "adds raw scores; minmax() normalises them";
    let says = |sql: &str| {
        let explained = db.explain(sql, &params).unwrap().explain(&[db.snapshot().schema()]).join("\n");
        let ran = db.query(sql, &params).unwrap().stats.plan.join("\n");
        assert_eq!(explained.contains(note), ran.contains(note), "{sql}:\n{explained}\n{ran}");
        ran.contains(note)
    };
    for sql in [
        "SELECT id FROM docs ORDER BY bm25(body, 'x') + similarity(emb, :q) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY bm25(body, 'x') - 0.1 * rating DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY ln(1 + bm25(body, 'x')) + CASE WHEN year > 2000 THEN rating END DESC LIMIT 5",
    ] {
        assert!(says(sql), "{sql}");
    }
    for sql in [
        "SELECT id FROM docs ORDER BY minmax(bm25(body, 'x')) + minmax(similarity(emb, :q)) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY 0.3 * minmax(bm25(body, 'x')) + 0.7 * minmax(rating) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY bm25(body, 'x') * similarity(emb, :q) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY bm25(body, 'x') + 1 DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY rating + year DESC LIMIT 5",
    ] {
        assert!(!says(sql), "{sql}");
    }
}

/// `tmm(x)` is `(x - lo) / (max - lo)`, `lo` the least value `x` can take
/// (0 for BM25, -1 for a similarity, given for a column) and `max` the
/// largest over the candidates, so a row keeps its value whatever the
/// depth of the candidates, where `minmax()` moves with the smallest
/// candidate; with `hi` too the bounds are fixed. `zscore(x)` centres and
/// scales by the candidates' mean and deviation.
#[test]
fn tmm_and_zscore_normalise_as_documented() {
    let docs = docs();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "t.strata", &docs);
    let q = docs[42][col::EMB].clone();
    let title = docs[42][col::TITLE].as_str().unwrap().to_string();
    let text = title.split(' ').take(3).collect::<Vec<_>>().join(" ");
    let params = Params::new().set("q", q).set("t", Value::Str(text));
    let run = |sql: &str| db.query(sql, &params).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let by_key = |r: &ResultSet| -> std::collections::HashMap<String, f64> { r.keys.iter().map(|k| format!("{k:?}")).zip(r.scores.iter().copied()).collect() };

    // One input: the best row is 1, and every row its raw value over the
    // best's, from the least possible value.
    let raw = run("SELECT id, score() FROM docs ORDER BY bm25(body, :t) DESC LIMIT 50");
    let t = run("SELECT id, score() FROM docs ORDER BY tmm(bm25(body, :t)) DESC LIMIT 50");
    assert_eq!(raw.keys, t.keys);
    for (r, n) in raw.scores.iter().zip(&t.scores) {
        assert!((n - r / raw.scores[0]).abs() < 1e-9, "{r} {n}");
    }
    let raw = run("SELECT id, score() FROM docs ORDER BY similarity(emb, :q) DESC LIMIT 50");
    let t = run("SELECT id, score() FROM docs ORDER BY tmm(similarity(emb, :q)) DESC LIMIT 50");
    assert_eq!(raw.keys, t.keys);
    for (r, n) in raw.scores.iter().zip(&t.scores) {
        assert!((n - (r + 1.0) / (raw.scores[0] + 1.0)).abs() < 1e-9, "{r} {n}");
    }

    // A fused row's value does not depend on the depth of the candidates
    // under tmm; under minmax it does.
    let fused = |norm: &str, depth: usize| {
        run(&format!("SELECT id, score() FROM docs ORDER BY 0.3 * {norm}(bm25(body, :t, depth => {depth})) + 0.7 * {norm}(similarity(emb, :q)) DESC LIMIT 10"))
    };
    let (shallow, deep) = (by_key(&fused("tmm", 20)), by_key(&fused("tmm", 400)));
    let shared: Vec<&String> = shallow.keys().filter(|k| deep.contains_key(*k)).collect();
    assert!(shared.len() >= 5, "{shared:?}");
    for k in &shared {
        assert!((shallow[*k] - deep[*k]).abs() < 1e-9, "{k}: {} vs {}", shallow[*k], deep[*k]);
    }
    let (shallow, deep) = (by_key(&fused("minmax", 20)), by_key(&fused("minmax", 400)));
    assert!(shallow.keys().filter(|k| deep.contains_key(*k)).any(|k| (shallow[k] - deep[k]).abs() > 1e-6), "minmax moves with depth");

    // Fixed bounds: the row's own value, 0 where it has none.
    let r = run("SELECT id, score(), rating FROM docs ORDER BY tmm(rating, lo => 0, hi => 5) DESC LIMIT 2400");
    for (row, s) in r.rows.iter().zip(&r.scores) {
        match &row[2] {
            Value::Float(x) => assert!((s - x / 5.0).abs() < 1e-12, "{x} {s}"),
            Value::Null => assert_eq!(*s, 0.0),
            v => panic!("{v:?}"),
        }
    }

    // zscore: mean 0 and deviation 1 over the candidates of one input.
    let z = run("SELECT id, score() FROM docs ORDER BY zscore(rating) DESC LIMIT 2400");
    let have: Vec<f64> = z.rows.iter().zip(&z.scores).filter(|(row, _)| row[0] != Value::Null).map(|(_, s)| *s).collect();
    let rated: Vec<f64> = docs.iter().filter_map(|d| d[col::RATING].as_f64()).collect();
    let mean = rated.iter().sum::<f64>() / rated.len() as f64;
    let sd = (rated.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rated.len() as f64).sqrt();
    let want: Vec<f64> = {
        let mut w: Vec<f64> = rated.iter().map(|x| (x - mean) / sd).collect();
        let least = w.iter().copied().fold(f64::INFINITY, f64::min);
        w.extend(std::iter::repeat_n(least, docs.len() - rated.len()));
        w.sort_by(|a, b| b.total_cmp(a));
        w
    };
    assert_eq!(have.len(), want.len());
    for (a, b) in have.iter().zip(&want) {
        assert!((a - b).abs() < 1e-9, "{a} {b}");
    }

    // EXPLAIN writes the bound tmm uses.
    let plan = db.explain("SELECT id FROM docs ORDER BY 0.5 * tmm(bm25(body, :t)) + 0.5 * tmm(similarity(emb, :q)) DESC LIMIT 5", &params).unwrap();
    let text = plan.explain(&[db.snapshot().schema()]).join("\n");
    assert!(text.contains("tmm(similarity(emb") && text.contains("lo => -1") && text.contains("lo => 0"), "{text}");

    // Errors say what to write.
    for (sql, says) in [
        ("SELECT id FROM docs ORDER BY tmm(rating) DESC LIMIT 5", "lo => n"),
        ("SELECT id FROM docs ORDER BY tmm(ln(1 + bm25(body, :t))) DESC LIMIT 5", "lo => n"),
        ("SELECT id FROM docs ORDER BY tmm(bm25(body, :t), hi => 0) DESC LIMIT 5", "must be above"),
        ("SELECT id FROM docs ORDER BY tmm(bm25(body, :t), depth => 5) DESC LIMIT 5", "takes the options lo and hi"),
        ("SELECT id FROM docs ORDER BY zscore(bm25(body, :t), similarity(emb, :q)) DESC LIMIT 5", "one argument"),
    ] {
        let e = db.query(sql, &params).unwrap_err().to_string();
        assert!(e.contains(says), "{sql}: {e}");
    }
}

/// `tmm` and `zscore` need the plan feature `score_norms`: a reader from
/// before them would take the set function for `minmax`. `tmm` with both
/// bounds is arithmetic and does not need it.
#[test]
fn an_old_reader_refuses_tmm_and_zscore_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "f.strata", &data::synthetic(50, DIM, 5));
    let params = Params::new().set("q", Value::Vector(vec![0.25; DIM].into()));
    let old = [OPERATORS, SEQ_NO, RANK_EXPRESSIONS];
    let features = |sql: &str| {
        let plan = Plan::from_json(&db.explain(sql, &params).unwrap().to_json().unwrap()).unwrap();
        let fs: Vec<String> = plan.features.clone();
        let refs: Vec<&str> = fs.iter().map(String::as_str).collect();
        (fs.contains(&SCORE_NORMS.to_string()), Plan::readable_by(plan.version as u64, &refs, &old).map_err(|e| e.to_string()))
    };
    for sql in [
        "SELECT id FROM docs ORDER BY 0.3 * tmm(bm25(body, 'x')) + 0.7 * tmm(similarity(emb, :q)) DESC LIMIT 5",
        "SELECT id FROM docs ORDER BY zscore(similarity(emb, :q)) DESC LIMIT 5",
    ] {
        let (needed, read) = features(sql);
        assert!(needed, "{sql}");
        assert!(read.unwrap_err().contains(SCORE_NORMS), "{sql}");
    }
    let (needed, read) = features("SELECT id FROM docs ORDER BY tmm(bm25(body, 'x'), lo => 0, hi => 10) + tmm(rating, lo => 0, hi => 5) DESC LIMIT 5");
    assert!(!needed);
    assert_eq!(read, Ok(()));
    // A plan written without lo takes the input's own when read.
    let json = db.explain("SELECT id FROM docs ORDER BY tmm(similarity(emb, :q)) DESC LIMIT 5", &params).unwrap().to_json().unwrap().to_string();
    assert!(json.contains("\"lo\":-1"), "{json}");
    let bare = json.replace(",\"lo\":-1.0", "").replace(",\"lo\":-1", "");
    assert!(!bare.contains("\"lo\""), "{bare}");
    let plan = |j: &str| Plan::from_json(&serde_json::from_str(j).unwrap()).unwrap();
    let a = db.run_at(&db.snapshot(), &plan(&json), &ExecOptions::default()).unwrap();
    let b = db.run_at(&db.snapshot(), &plan(&bare), &ExecOptions::default()).unwrap();
    assert_eq!(a.keys, b.keys);
    assert_eq!(a.scores, b.scores);
}
