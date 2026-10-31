//! Evaluates Strata on the shared workloads (DESIGN.md §8).
//!
//! Studies (`--studies`, default all):
//!
//! * `workload`: every workload kind through SQL, warm and cold, with the
//!   same harness as the SQLite and Tantivy baselines
//!   (`results/strata/strata_<tag>.json`).
//! * `hybrid`: retrieval quality (nDCG@10) and latency of BM25, approximate
//!   and exact vector search, RRF and weighted fusion over a weight sweep;
//!   recall of candidate-based weighted fusion against exact weighted
//!   ranking (`results/strata/hybrid_<tag>.json`). The fusion weight is
//!   chosen on half of the judged queries (dev) and every method is
//!   reported on the other half (test), with a paired randomization
//!   p-value against BM25.
//! * `fusion`: the normalisers of a weighted hybrid side by side: RRF over
//!   a sweep of its `k`, and `w * norm(bm25) + (1 - w) * norm(similarity)`
//!   over a sweep of `w` for `minmax`, `tmm` and `zscore`, each family's
//!   parameter chosen on the dev half and reported on the test half, and
//!   how much the chosen ranking moves with the depth of the candidates
//!   (`results/strata/fusion_<tag>.json`).
//! * `planner`: every filtered query re-run under each forced plan; the
//!   planner's choice against the best plan in hindsight
//!   (`results/strata/planner_<tag>.json`).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use clap::Parser;
use serde_json::{json, Value as Json};

use strata::data::{self, Collection};
use strata::db::Db;
use strata::error::{Error, Result};
use strata::forward::RangeStrategy;
use strata::iostat;
use strata::metrics::{mean, ndcg_at_k, paired_randomization, percentile, recall};
use strata::query::exec::ExecOptions;
use strata::query::plan::{FilterMode, VectorStrategy};
use strata::query::Params;
use strata::schema::VectorIndexKind;
use strata::segment::build::BuildOptions;
use strata::storage::DbOptions;
use strata::value::Value;
use strata::workload::run::{run_kind, BuildReport, QueryOutput, RunOptions, System};
use strata::workload::{generate, Kind, Workload, WorkloadConfig, WorkloadQuery};

#[derive(Parser, Debug)]
#[command(about = "Evaluate Strata on the shared workloads")]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "scifact")]
    collections: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "workload,hybrid,planner,showcase")]
    studies: Vec<String>,
    /// Vector index: hnsw, ivf or flat.
    #[arg(long, default_value = "hnsw")]
    index: String,
    /// Threads for building vector indexes.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Documents per segment (default: one segment per collection, like a
    /// fully merged index).
    #[arg(long)]
    segment_docs: Option<usize>,
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    #[arg(long, default_value_t = 60.0)]
    warm_budget: f64,
    #[arg(long, default_value_t = 100)]
    cold_sample: usize,
    #[arg(long, default_value_t = 60.0)]
    cold_budget: f64,
    #[arg(long)]
    max_per_kind: Option<usize>,
    #[arg(long)]
    max_vector: Option<usize>,
    /// Queries per study for `hybrid` and per kind for `planner`.
    #[arg(long, default_value_t = 300)]
    study_queries: usize,
    #[arg(long, default_value = "/home/user/data/strata")]
    db_dir: PathBuf,
    #[arg(long)]
    results_dir: Option<PathBuf>,
    /// Reuse a database left by an earlier `--keep` run.
    #[arg(long)]
    reuse: bool,
    #[arg(long)]
    keep: bool,
    /// Open databases with kernel read-ahead off (`MADV_RANDOM`).
    #[arg(long)]
    no_read_ahead: bool,
    /// Keep read-ahead on every section (no per-section MADV_RANDOM).
    #[arg(long)]
    no_random_hints: bool,
}

fn results_root(args: &Args) -> PathBuf {
    args.results_dir.clone().unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/results")))
}

fn write_json(path: &Path, value: &Json) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, serde_json::to_string_pretty(value).map_err(strata::error::corrupt)? + "\n")?;
    Ok(())
}

fn peak_rss_mb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    u.ru_maxrss as f64 / 1024.0
}

fn wal_path(p: &Path) -> PathBuf {
    PathBuf::from(format!("{}-wal", p.display()))
}

/// Strata behind the workload harness.
struct StrataSys {
    db: Option<Db>,
    path: PathBuf,
    exec: ExecOptions,
    read_ahead: bool,
    random_hints: bool,
}

impl StrataSys {
    fn db(&self) -> &Db {
        self.db.as_ref().unwrap()
    }
}

fn query_params(q: &WorkloadQuery, coll: &Collection) -> Params {
    match q.query {
        Some(i) if q.sql.contains(":q") => Params::new().set("q", Value::Vector(coll.queries[i].vector.clone())),
        _ => Params::new(),
    }
}

impl System for StrataSys {
    fn name(&self) -> &str {
        "strata"
    }
    fn supports(&self, _kind: Kind) -> bool {
        true
    }
    fn run(&mut self, q: &WorkloadQuery, coll: &Collection) -> Result<QueryOutput> {
        let params = query_params(q, coll);
        let rs = self.db().query_with(&q.sql, &params, &self.exec)?;
        let mut out = QueryOutput::default();
        if q.kind == Kind::Facet {
            out.facets = Some(rs.rows.iter().map(|r| (r[0].as_str().unwrap_or("").to_string(), r[1].as_i64().unwrap_or(0) as u64)).collect());
        } else {
            out.ids = rs.rows.iter().map(|r| r[0].as_str().unwrap_or("").to_string()).collect();
        }
        if let Some(c) = &q.count_sql {
            out.count = self.db().query_with(c, &params, &self.exec)?.rows.first().and_then(|r| r[0].as_i64()).map(|n| n as u64);
        }
        Ok(out)
    }
    fn evict_caches(&mut self) -> Result<()> {
        self.db = None;
        for p in self.files() {
            iostat::evict(&p)?;
        }
        self.db = Some(Db::open(&self.path, DbOptions { read_ahead: self.read_ahead, random_hints: self.random_hints, ..DbOptions::default() })?);
        Ok(())
    }
    fn files(&self) -> Vec<PathBuf> {
        [self.path.clone(), wal_path(&self.path)].into_iter().filter(|p| p.exists()).collect()
    }
}

fn index_kind(name: &str) -> Result<VectorIndexKind> {
    Ok(match name {
        "hnsw" => VectorIndexKind::Hnsw { m: 16, ef_construction: 200 },
        "ivf" => VectorIndexKind::Ivf { nlist: 0 },
        "flat" => VectorIndexKind::Flat,
        o => return Err(Error::Query(format!("unknown index {o}"))),
    })
}

/// Builds (or reuses) the collection's database; returns its build report.
fn build(args: &Args, coll: &mut Collection, path: &Path) -> Result<BuildReport> {
    let report_path = PathBuf::from(format!("{}.build.json", path.display()));
    if args.reuse && path.exists() {
        if let Some(r) = std::fs::read(&report_path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            coll.docs = Vec::new();
            return Ok(r);
        }
    }
    for p in [path.to_path_buf(), wal_path(path)] {
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    let n = coll.docs.len();
    let opts = DbOptions {
        build: BuildOptions::default().set("vector.threads", args.threads),
        segment_max_docs: args.segment_docs.unwrap_or(n.max(1)),
        ..DbOptions::default()
    };
    let t = Instant::now();
    let db = Db::create(path, data::beir_schema(coll.dim, index_kind(&args.index)?), opts)?;
    db.load(std::mem::take(&mut coll.docs))?;
    let seconds = t.elapsed().as_secs_f64();
    let stats = db.storage().file_stats();
    // Bytes by component, from the section directories of the segments.
    let mut comp: BTreeMap<String, u64> = BTreeMap::new();
    for v in db.snapshot().segments() {
        for (name, len) in v.file.section_sizes() {
            let key = match name.split('.').next().unwrap_or("") {
                "text" => format!("text ({})", name.split('.').nth(2).unwrap_or("")),
                "vec" => format!("vectors ({})", name.split('.').nth(2).unwrap_or("")),
                "col" | "range" | "kw" => "structured".to_string(),
                "store" => "doc store".to_string(),
                other => other.to_string(),
            };
            *comp.entry(key).or_insert(0) += len as u64;
        }
    }
    let r = BuildReport { seconds, phases: vec![("load".into(), seconds)], bytes: stats.file_bytes, components: comp.into_iter().collect() };
    write_json(&report_path, &json!(r))?;
    Ok(r)
}

fn workload(args: &Args, coll: &Collection, tag: &str) -> Result<Workload> {
    let path = results_root(args).join("workloads").join(format!("{tag}.json"));
    if path.exists() {
        return Workload::load(&path);
    }
    let w = generate(coll, &WorkloadConfig::default());
    w.save(&path)?;
    Ok(w)
}

fn main() {
    let args = Args::parse();
    for name in &args.collections {
        if let Err(e) = run_collection(&args, name) {
            eprintln!("{name}: {e}");
            std::process::exit(1);
        }
    }
}

fn run_collection(args: &Args, name: &str) -> Result<()> {
    eprintln!("== {name}");
    let mut coll = data::load(name, None)?;
    let w = workload(args, &coll, name)?;
    let path = args.db_dir.join(format!("{name}.{}.strata", args.index));
    let t = Instant::now();
    let build_report = build(args, &mut coll, &path)?;
    eprintln!("   built in {:.1}s ({:.1}s here), {:.1} MiB", build_report.seconds, t.elapsed().as_secs_f64(), build_report.bytes as f64 / (1 << 20) as f64);
    let (read_ahead, random_hints) = (!args.no_read_ahead, !args.no_random_hints);
    let mut sys = StrataSys {
        db: Some(Db::open(&path, DbOptions { read_ahead, random_hints, ..DbOptions::default() })?),
        path: path.clone(),
        exec: ExecOptions::default(),
        read_ahead,
        random_hints,
    };
    let out_dir = results_root(args).join("strata");
    let mut suffix = if args.index == "hnsw" { String::new() } else { format!("_{}", args.index) };
    if !read_ahead {
        suffix.push_str("_noreadahead");
    } else if !random_hints {
        suffix.push_str("_readahead");
    }

    if args.studies.iter().any(|s| s == "workload") {
        let kinds: Vec<Kind> = if args.kinds.is_empty() {
            Kind::ALL.to_vec()
        } else {
            args.kinds.iter().map(|k| Kind::parse(k).ok_or_else(|| Error::Query(format!("unknown kind {k}")))).collect::<Result<_>>()?
        };
        let opts = RunOptions {
            reps: args.reps,
            warm_budget_secs: args.warm_budget,
            cold_sample: args.cold_sample,
            cold_budget_secs: args.cold_budget,
            max_per_kind: args.max_per_kind,
            max_vector: args.max_vector,
        };
        let mut reports = Vec::new();
        let mut outputs = BTreeMap::new();
        for k in kinds {
            let run = run_kind(&mut sys, &w, &coll, k, &opts)?;
            eprintln!("   {:<15} warm p50 {:.3} ms, cold p50 {}", k.name(), run.report.warm.p50_ms, run.report.cold.as_ref().map_or("-".into(), |c| format!("{:.2} ms", c.p50_ms)));
            reports.push(run.report);
            outputs.insert(k.name(), run.outputs);
        }
        write_json(
            &out_dir.join(format!("strata{suffix}_{name}.json")),
            &json!({
                "system": "strata",
                "collection": name,
                "n_docs": w.n_docs,
                "workload": format!("results/workloads/{name}.json"),
                "config": {"index": args.index, "threads": args.threads, "read_ahead": read_ahead, "random_hints": random_hints, "segment_docs": args.segment_docs, "segments": sys.db().snapshot().segments().len()},
                "run": {"options": opts, "unix_time": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs()), "peak_rss_mib": peak_rss_mb()},
                "build": build_report,
                "kinds": reports,
                "outputs": outputs,
            }),
        )?;
    }
    if args.studies.iter().any(|s| s == "hybrid") {
        let doc = hybrid_study(args, &sys, &coll)?;
        write_json(&out_dir.join(format!("hybrid{suffix}_{name}.json")), &doc)?;
    }
    if args.studies.iter().any(|s| s == "fusion") {
        let doc = fusion_study(args, &sys, &coll)?;
        write_json(&out_dir.join(format!("fusion{suffix}_{name}.json")), &doc)?;
    }
    if args.studies.iter().any(|s| s == "showcase") {
        let doc = showcase(&sys, &coll)?;
        write_json(&out_dir.join(format!("showcase{suffix}_{name}.json")), &doc)?;
    }
    if args.studies.iter().any(|s| s == "planner") {
        let doc = planner_study(args, &sys, &coll, &w)?;
        write_json(&out_dir.join(format!("planner{suffix}_{name}.json")), &doc)?;
    }
    drop(sys);
    if !args.keep {
        for p in [path.clone(), wal_path(&path), PathBuf::from(format!("{}.build.json", path.display()))] {
            if p.exists() {
                std::fs::remove_file(&p)?;
            }
        }
    }
    Ok(())
}

fn ids(rows: &[Vec<Value>]) -> Vec<String> {
    rows.iter().map(|r| r[0].as_str().unwrap_or("").to_string()).collect()
}

fn sql_text(s: &str) -> String {
    s.replace('\'', "''")
}

/// Median of `reps` timed runs, in ms, and the last output.
fn timed(reps: usize, mut f: impl FnMut() -> Result<Vec<String>>) -> Result<(f64, Vec<String>)> {
    let mut ts = Vec::with_capacity(reps);
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        out = f()?;
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((ts[ts.len() / 2], out))
}

/// nDCG@10 and latency of each ranking method; recall of candidate-based
/// weighted fusion against exact weighted ranking. The judged queries
/// alternate between a dev half, on which the fusion weight is chosen, and
/// a test half, on which every method is reported.
fn hybrid_study(args: &Args, sys: &StrataSys, coll: &Collection) -> Result<Json> {
    let db = sys.db();
    let judged: Vec<usize> = (0..coll.queries.len()).filter(|&i| coll.qrels.contains_key(&coll.queries[i].id)).collect();
    let picked: Vec<usize> = strata::workload::run::spread(judged.len(), args.study_queries).into_iter().map(|j| judged[j]).collect();
    let alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let exact_ok = db.snapshot().n_live_docs() <= 200_000;
    let mut methods: Vec<(String, String)> = vec![
        ("bm25".into(), "ORDER BY bm25(body, '{t}', syntax => 'plain') DESC".into()),
        ("vector".into(), "ORDER BY similarity(emb, :q) DESC".into()),
        ("vector_exact".into(), "ORDER BY similarity(emb, :q, strategy => 'flat', rerank => 100) DESC".into()),
        ("rrf".into(), "ORDER BY rrf(bm25(body, '{t}', syntax => 'plain'), similarity(emb, :q), depth => 100) DESC".into()),
    ];
    for a in alphas.iter().filter(|&&a| a > 0.0 && a < 1.0) {
        methods.push((format!("weighted_{a:.1}"), format!("ORDER BY {a} * minmax(bm25(body, '{{t}}', syntax => 'plain')) + {:.1} * minmax(similarity(emb, :q)) DESC", 1.0 - a)));
    }
    let mut ndcg: HashMap<String, Vec<f64>> = HashMap::new();
    let mut lat: HashMap<String, Vec<f64>> = HashMap::new();
    let mut outs: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for &qi in &picked {
        let q = &coll.queries[qi];
        let params = Params::new().set("q", Value::Vector(q.vector.clone()));
        let judged = &coll.qrels[&q.id];
        for (m, order) in &methods {
            let sql = format!("SELECT id FROM docs {} LIMIT 10", order.replace("{t}", &sql_text(&q.text)));
            let (ms, got) = timed(3, || Ok(ids(&db.query(&sql, &params)?.rows)))?;
            ndcg.entry(m.clone()).or_default().push(ndcg_at_k(&got, judged, 10));
            lat.entry(m.clone()).or_default().push(ms);
            outs.entry(m.clone()).or_default().push(got);
        }
    }
    let half = |v: &[f64], r: usize| v.iter().skip(r).step_by(2).copied().collect::<Vec<f64>>();
    let (dev, test) = (|m: &str| half(&ndcg[m], 0), |m: &str| half(&ndcg[m], 1));
    // The weight that does best on dev; exact weighted ranking at it is the
    // recall reference.
    let best = methods
        .iter()
        .filter(|(m, _)| m.starts_with("weighted_"))
        .max_by(|a, b| mean(&dev(&a.0)).total_cmp(&mean(&dev(&b.0))))
        .map(|(m, o)| (m.clone(), o.clone()));
    let chosen = best.as_ref().map(|(m, _)| m.clone());
    let mut exact = Json::Null;
    if let (Some((m, order)), true) = (best, exact_ok) {
        let exact_order = order.replace("similarity(emb, :q)", "similarity(emb, :q, exact => 1)");
        let mut rec = Vec::new();
        let mut ms_all = Vec::new();
        let mut nd = Vec::new();
        for (j, &qi) in picked.iter().enumerate().skip(1).step_by(2).take(100) {
            let q = &coll.queries[qi];
            let params = Params::new().set("q", Value::Vector(q.vector.clone()));
            let sql = format!("SELECT id FROM docs {} LIMIT 10", exact_order.replace("{t}", &sql_text(&q.text)));
            let (ms, got) = timed(1, || Ok(ids(&db.query(&sql, &params)?.rows)))?;
            rec.push(recall(&outs[&m][j], &got));
            nd.push(ndcg_at_k(&got, &coll.qrels[&q.id], 10));
            ms_all.push(ms);
        }
        exact = json!({"method": m, "queries": rec.len(), "candidate_recall": mean(&rec), "exact_ndcg10": mean(&nd), "exact_ms_mean": mean(&ms_all)});
    }
    let summary: BTreeMap<String, Json> = methods
        .iter()
        .map(|(m, _)| {
            let p = paired_randomization(&test(m), &test("bm25"), 10_000, 1);
            let row = json!({
                "ndcg10": mean(&test(m)),
                "ndcg10_dev": mean(&dev(m)),
                "p_vs_bm25": p,
                "ms_mean": mean(&lat[m]),
                "ms_p50": percentile(&lat[m], 50.0),
                "ms_p99": percentile(&lat[m], 99.0)
            });
            (m.clone(), row)
        })
        .collect();
    eprintln!("   hybrid (test half): {}", methods.iter().map(|(m, _)| format!("{m} {:.3}", mean(&test(m)))).collect::<Vec<_>>().join(", "));
    Ok(json!({
        "collection": coll.name,
        "queries": {"dev": picked.len().div_ceil(2), "test": picked.len() / 2},
        "weight_chosen_on_dev": chosen,
        "methods": summary,
        "weighted_exact": exact
    }))
}

/// RRF, min-max, theoretical min-max and z-score fusion of BM25 and the
/// vector ranking, on the same queries: each family's parameter (RRF's
/// `k`, the others' BM25 weight) chosen on the dev half, every method
/// reported on the test half, with a paired randomization p-value of each
/// family's chosen method against RRF at its default `k` (60); and, at the
/// chosen weight, nDCG@10 and the top 10's overlap with depth 400's when
/// the candidates are each input's top 20, 100 or 400.
fn fusion_study(args: &Args, sys: &StrataSys, coll: &Collection) -> Result<Json> {
    let db = sys.db();
    let judged: Vec<usize> = (0..coll.queries.len()).filter(|&i| coll.qrels.contains_key(&coll.queries[i].id)).collect();
    let picked: Vec<usize> = strata::workload::run::spread(judged.len(), args.study_queries).into_iter().map(|j| judged[j]).collect();
    let bm25 = |depth: usize| format!("bm25(body, '{{t}}', syntax => 'plain', depth => {depth})");
    let sim = "similarity(emb, :q)";
    // (family, parameter, ORDER BY at a depth)
    let mut methods: Vec<(String, f64, Box<dyn Fn(usize) -> String>)> = vec![
        ("bm25".into(), 0.0, Box::new(move |_| format!("{} DESC", bm25(100)))),
        ("vector".into(), 0.0, Box::new(move |_| format!("{sim} DESC"))),
    ];
    for k in [1.0, 10.0, 20.0, 60.0, 100.0] {
        methods.push(("rrf".into(), k, Box::new(move |d| format!("rrf(bm25(body, '{{t}}', syntax => 'plain'), {sim}, k => {k}, depth => {d}) DESC"))));
    }
    for norm in ["minmax", "tmm", "zscore"] {
        for w in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
            methods.push((norm.into(), w, Box::new(move |d| format!("{w} * {norm}({}) + {:.2} * {norm}({sim}) DESC", bm25(d), 1.0 - w))));
        }
    }
    let name = |(f, p): (&str, f64)| if f == "bm25" || f == "vector" { f.to_string() } else { format!("{f}_{p}") };
    let run = |order: &str, qi: usize| -> Result<(f64, Vec<String>)> {
        let q = &coll.queries[qi];
        let params = Params::new().set("q", Value::Vector(q.vector.clone()));
        let sql = format!("SELECT id FROM docs ORDER BY {} LIMIT 10", order.replace("{t}", &sql_text(&q.text)));
        timed(1, || Ok(ids(&db.query(&sql, &params)?.rows)))
    };
    let mut ndcg: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut lat: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for &qi in &picked {
        let judged = &coll.qrels[&coll.queries[qi].id];
        for (f, p, order) in &methods {
            let (ms, got) = run(&order(100), qi)?;
            ndcg.entry(name((f, *p))).or_default().push(ndcg_at_k(&got, judged, 10));
            lat.entry(name((f, *p))).or_default().push(ms);
        }
    }
    let half = |v: &[f64], r: usize| v.iter().skip(r).step_by(2).copied().collect::<Vec<f64>>();
    let (dev, test) = (|m: &str| half(&ndcg[m], 0), |m: &str| half(&ndcg[m], 1));
    let mut families = BTreeMap::new();
    let mut depth = BTreeMap::new();
    for fam in ["rrf", "minmax", "tmm", "zscore"] {
        let ms: Vec<&(String, f64, Box<dyn Fn(usize) -> String>)> = methods.iter().filter(|m| m.0 == fam).collect();
        let chosen = ms.iter().max_by(|a, b| mean(&dev(&name((&a.0, a.1)))).total_cmp(&mean(&dev(&name((&b.0, b.1)))))).unwrap();
        let best_test = ms.iter().map(|m| mean(&test(&name((&m.0, m.1))))).fold(f64::NEG_INFINITY, f64::max);
        let n = name((&chosen.0, chosen.1));
        families.insert(
            fam,
            json!({
                "chosen_on_dev": chosen.1,
                "ndcg10": mean(&test(&n)),
                "best_on_test": best_test,
                "p_vs_rrf60": paired_randomization(&test(&n), &test("rrf_60"), 10_000, 1),
                "by_parameter": ms.iter().map(|m| (format!("{}", m.1), json!(mean(&test(&name((&m.0, m.1))))))).collect::<BTreeMap<_, _>>(),
            }),
        );
        if fam == "minmax" || fam == "tmm" || fam == "rrf" {
            // The chosen ranking at three depths, on the test half.
            let mut by_depth = BTreeMap::new();
            let tops: Vec<Vec<Vec<String>>> = [20, 100, 400]
                .iter()
                .map(|&d| picked.iter().skip(1).step_by(2).map(|&qi| run(&(chosen.2)(d), qi).map(|r| r.1)).collect::<Result<Vec<_>>>())
                .collect::<Result<_>>()?;
            for (i, d) in [20, 100, 400].iter().enumerate() {
                let nd: Vec<f64> = picked.iter().skip(1).step_by(2).zip(&tops[i]).map(|(&qi, got)| ndcg_at_k(got, &coll.qrels[&coll.queries[qi].id], 10)).collect();
                let overlap: Vec<f64> = tops[i].iter().zip(&tops[2]).map(|(a, b)| recall(b, a)).collect();
                by_depth.insert(d.to_string(), json!({"ndcg10": mean(&nd), "top10_overlap_with_depth_400": mean(&overlap)}));
            }
            depth.insert(fam, json!(by_depth));
        }
    }
    let summary: BTreeMap<String, Json> = ndcg.keys().map(|m| (m.clone(), json!({"ndcg10": mean(&test(m)), "ndcg10_dev": mean(&dev(m)), "ms_mean": mean(&lat[m])}))).collect();
    eprintln!(
        "   fusion (test half): {}",
        ["bm25", "vector", "rrf_60"].iter().map(|m| format!("{m} {:.3}", mean(&test(m)))).chain(families.iter().map(|(f, v)| format!("{f} {:.3} (at {})", v["ndcg10"].as_f64().unwrap_or(0.0), v["chosen_on_dev"]))).collect::<Vec<_>>().join(", ")
    );
    Ok(json!({
        "collection": coll.name,
        "queries": {"dev": picked.len().div_ceil(2), "test": picked.len() / 2},
        "families": families,
        "depth": depth,
        "methods": summary,
    }))
}

struct Plan {
    name: &'static str,
    opts: ExecOptions,
}

fn plans_for(kind: Kind) -> Vec<Plan> {
    let base = ExecOptions::default();
    let with = |f: &dyn Fn(&mut ExecOptions)| {
        let mut o = base.clone();
        f(&mut o);
        o
    };
    match kind {
        Kind::FilteredVector | Kind::FilteredHybrid => vec![
            Plan { name: "auto", opts: base.clone() },
            Plan { name: "prefilter", opts: with(&|o| o.vector = Some(VectorStrategy::PreFilter)) },
            Plan {
                name: "index+probe",
                opts: with(&|o| {
                    o.vector = Some(VectorStrategy::InIndex);
                    o.filter_mode = Some(FilterMode::Probe)
                }),
            },
            Plan {
                name: "index+set",
                opts: with(&|o| {
                    o.vector = Some(VectorStrategy::InIndex);
                    o.filter_mode = Some(FilterMode::Materialize)
                }),
            },
            Plan {
                name: "postfilter+probe",
                opts: with(&|o| {
                    o.vector = Some(VectorStrategy::PostFilter);
                    o.filter_mode = Some(FilterMode::Probe)
                }),
            },
        ],
        Kind::FilteredBm25 => vec![
            Plan { name: "auto", opts: base.clone() },
            Plan { name: "set", opts: with(&|o| o.filter_mode = Some(FilterMode::Materialize)) },
            Plan { name: "probe", opts: with(&|o| o.filter_mode = Some(FilterMode::Probe)) },
        ],
        _ => vec![
            Plan { name: "auto", opts: base.clone() },
            Plan { name: "scan", opts: with(&|o| o.range = RangeStrategy::Scan) },
            Plan { name: "postings", opts: with(&|o| o.range = RangeStrategy::Postings) },
        ],
    }
}

/// Each filtered query under every forced plan. A plan qualifies when its
/// recall@10 against exact search is at least 0.9; regret is the planner's
/// time over the fastest qualifying plan's.
fn planner_study(args: &Args, sys: &StrataSys, coll: &Collection, w: &Workload) -> Result<Json> {
    let db = sys.db();
    let mut per_kind = BTreeMap::new();
    for kind in [Kind::FilteredVector, Kind::FilteredBm25, Kind::Filter] {
        let qs: Vec<usize> = {
            let all = w.of_kind(kind);
            strata::workload::run::spread(all.len(), args.study_queries).into_iter().map(|j| all[j]).collect()
        };
        let plans = plans_for(kind);
        let mut rows = Vec::new();
        for &i in &qs {
            let q = &w.queries[i];
            let params = query_params(q, coll);
            let sql = if kind == Kind::Filter { q.count_sql.clone().unwrap_or_else(|| q.sql.clone()) } else { q.sql.clone() };
            let truth = if kind == Kind::FilteredVector {
                let exact = q.sql.replace("similarity(emb, :q)", "similarity(emb, :q, strategy => 'prefilter', rerank => 100)");
                ids(&db.query(&exact, &params)?.rows)
            } else {
                Vec::new()
            };
            let mut res = Vec::new();
            for p in &plans {
                let (ms, got) = timed(3, || Ok(ids(&db.query_with(&sql, &params, &p.opts)?.rows)))?;
                let rec = if truth.is_empty() { 1.0 } else { recall(&got, &truth) };
                res.push((p.name, ms, rec));
            }
            let qualifying: Vec<&(&str, f64, f64)> = res.iter().filter(|r| r.0 != "auto" && r.2 >= 0.9).collect();
            let best = qualifying.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|r| r.1);
            let auto = res[0];
            rows.push(json!({
                "query": i,
                "target": q.target,
                "matches": q.matches,
                "plans": res.iter().map(|r| json!({"plan": r.0, "ms": r.1, "recall": r.2})).collect::<Vec<_>>(),
                "best_ms": best,
                "regret": best.map(|b| auto.1 / b),
                "auto_recall": auto.2,
            }));
        }
        let regrets: Vec<f64> = rows.iter().filter_map(|r| r["regret"].as_f64()).collect();
        let totals: BTreeMap<&str, f64> = plans
            .iter()
            .enumerate()
            .map(|(j, p)| (p.name, rows.iter().map(|r| r["plans"][j]["ms"].as_f64().unwrap_or(0.0)).sum::<f64>()))
            .collect();
        let oracle: f64 = rows.iter().filter_map(|r| r["best_ms"].as_f64()).sum();
        let geo = (regrets.iter().map(|r| r.ln()).sum::<f64>() / regrets.len().max(1) as f64).exp();
        eprintln!(
            "   planner {:<15} {} queries: regret geo-mean {geo:.2}, within 1.25x {:.0}%, total auto {:.0} ms vs oracle {oracle:.0} ms",
            kind.name(),
            rows.len(),
            100.0 * regrets.iter().filter(|&&r| r <= 1.25).count() as f64 / regrets.len().max(1) as f64,
            totals["auto"]
        );
        per_kind.insert(
            kind.name(),
            json!({
                "queries": rows.len(),
                "regret_geomean": geo,
                "regret_p90": percentile(&regrets, 90.0),
                "within_1_25": regrets.iter().filter(|&&r| r <= 1.25).count() as f64 / regrets.len().max(1) as f64,
                "auto_recall_mean": mean(&rows.iter().filter_map(|r| r["auto_recall"].as_f64()).collect::<Vec<_>>()),
                "total_ms": totals,
                "oracle_ms": oracle,
                "rows": rows,
            }),
        );
    }
    Ok(json!({"collection": coll.name, "kinds": per_kind}))
}

/// Compound queries that exercise the dialect end to end: their latency,
/// result counts and the plan each segment ran.
const SHOWCASE: &[(&str, &str)] = &[
    ("phrase, boolean and negation", "SELECT id, title FROM docs WHERE MATCH(body, '\"how do i\" AND (learn OR study) -money') LIMIT 10"),
    ("prefix and conjunction", "SELECT COUNT(*) FROM docs WHERE MATCH(body, 'progr* AND language')"),
    ("ranked text, OR / NOT filter", "SELECT id, score() FROM docs WHERE (category IN ('c01', 'c02') OR rating > 4.5) AND NOT topic = 't03' ORDER BY bm25(body, 'best programming language to learn first') DESC LIMIT 10"),
    ("filtered vector, OR filter", "SELECT id, score() FROM docs WHERE year < 1995 OR category = 'c40' ORDER BY similarity(emb, :q) DESC LIMIT 10"),
    ("filtered hybrid, weighted", "SELECT id, score() FROM docs WHERE rating >= 3 AND year BETWEEN 2005 AND 2020 ORDER BY 0.2 * minmax(bm25(body, :t)) + 0.8 * minmax(similarity(emb, :q)) DESC LIMIT 10"),
    ("hybrid, RRF on a correlated filter", "SELECT id, score() FROM docs WHERE topic IN ('t05', 't06', 't07') ORDER BY rrf(bm25(body, :t), similarity(emb, :q)) DESC LIMIT 10"),
    ("facets over a text match", "SELECT topic, COUNT(*) FROM docs WHERE MATCH(body, 'india') AND year BETWEEN 2000 AND 2010 GROUP BY topic ORDER BY 2 DESC, 1 LIMIT 5"),
    ("text filter, ordered by a column", "SELECT id, rating FROM docs WHERE MATCH(body, 'travel AND visa') ORDER BY rating DESC, id LIMIT 5"),
    ("count over a boolean and a range", "SELECT COUNT(*) FROM docs WHERE MATCH(body, 'iphone OR android') AND rating BETWEEN 2 AND 3"),
    ("nearest by distance, exact", "SELECT id FROM docs WHERE category IN ('c00', 'c01', 'c02', 'c03') ORDER BY distance(emb, :q, strategy => 'prefilter') LIMIT 10"),
];

fn showcase(sys: &StrataSys, coll: &Collection) -> Result<Json> {
    let db = sys.db();
    let q = &coll.queries[0];
    let params = Params::new().set("q", Value::Vector(q.vector.clone())).set("t", Value::Str(q.text.clone()));
    let mut rows = Vec::new();
    for (label, sql) in SHOWCASE {
        let rs = db.query(sql, &params)?;
        let (ms, _) = timed(5, || Ok(ids(&db.query(sql, &params)?.rows)))?;
        let first = rs.rows.first().map(|r| r.iter().map(|v| match v {
            Value::Str(s) => s.chars().take(60).collect::<String>(),
            Value::Int(x) => x.to_string(),
            Value::Float(x) => format!("{x:.4}"),
            _ => String::new(),
        }).collect::<Vec<_>>().join(" · "));
        eprintln!("   showcase {label}: {ms:.3} ms, {} rows", rs.rows.len());
        rows.push(json!({"label": label, "sql": sql, "ms": ms, "rows": rs.rows.len(), "first": first, "plan": rs.stats.plan}));
    }
    Ok(json!({"collection": coll.name, "query_text": q.text, "queries": rows}))
}
