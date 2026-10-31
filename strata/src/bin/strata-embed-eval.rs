//! Which fields to embed: retrieval quality of vectors made from a BEIR
//! collection's titles, texts, or both, by one model, against its relevance
//! judgments, with BM25, hybrid and the collection's own Cohere vectors as
//! references.
//!
//! ```text
//! STRATA_DATA=~/data/beir cargo run --release --bin strata-embed-eval -- --dataset scifact --model potion-retrieval-32M
//! cargo run --release --features bert --bin strata-embed-eval -- --dataset trec-covid --model bge-small-en-v1.5 --out results/embed-fields/trec-covid-bge.json
//! ```
//!
//! The database computes the vectors as the collection loads (the fields are
//! declared with `embed`), and vector search is exact (flat), so that the
//! comparison is of the vectors, not of an index.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde_json::json;
use strata::data;
use strata::db::Db;
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

#[derive(Parser)]
struct Args {
    /// A collection under $STRATA_DATA (scifact, trec-covid, fiqa, quora).
    #[arg(long)]
    dataset: String,
    /// The model, by the name a schema gives it (see `strata::embed::open`).
    #[arg(long, default_value = "potion-retrieval-32M")]
    model: String,
    /// Where models are (default $STRATA_MODELS or ~/.local/share/strata/models).
    #[arg(long)]
    models: Option<PathBuf>,
    /// Keep the first n documents.
    #[arg(long)]
    limit: Option<usize>,
    /// Write the results as JSON here.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// The embedded fields compared, and what each embeds.
const FIELDS: &[(&str, &[&str])] = &[("emb_title", &["title"]), ("emb_text", &["text"]), ("emb_both", &["title", "text"])];

fn main() {
    let a = Args::parse();
    let models = a.models.clone().unwrap_or_else(strata::embed::models_dir);
    let model = strata::embed::open(&a.model, &models).unwrap_or_else(|e| panic!("{e}"));
    let coll = data::load(&a.dataset, a.limit).unwrap_or_else(|e| panic!("{}: {e}", a.dataset));
    eprintln!("{}: {} documents, {} queries; model {} ({} dimensions)", coll.name, coll.docs.len(), coll.queries.len(), model.name(), model.dim());

    let dim = model.dim();
    let mut fields = vec![
        Field::new("id", FieldType::Keyword),
        Field::new("title", FieldType::Text { positions: false }).stored_only(),
        Field::new("text", FieldType::Text { positions: false }).stored_only(),
        Field::new("body", FieldType::Text { positions: true }).not_stored().generated(&["title", "text"]),
        Field::new("cohere", FieldType::Vector { dim: coll.dim, index: VectorIndexKind::Flat }).not_stored().embedded("cohere/embed-english-v3.0", &["title", "text"]),
    ];
    for (name, from) in FIELDS {
        fields.push(Field::new(name, FieldType::Vector { dim, index: VectorIndexKind::Flat }).not_stored().embedded(&a.model, from));
    }
    let schema = Schema::new("docs", fields, Some("id")).expect("valid schema");
    let dir = tempfile_dir();
    let db = Db::create(&dir.join("eval.strata"), schema, DbOptions { sync: false, ..DbOptions::default() }).expect("create");
    for (name, _) in FIELDS {
        db.set_embedder(name, model.clone()).expect("model fits the field");
    }
    let empty_titles = coll.docs.iter().filter(|d| d[data::col::TITLE].as_str().is_none_or(|t| t.is_empty())).count();
    let t0 = Instant::now();
    db.load(coll.docs.iter().map(|d| {
        vec![d[data::col::ID].clone(), d[data::col::TITLE].clone(), d[data::col::TEXT].clone(), Value::Null, d[data::col::EMB].clone(), Value::Null, Value::Null, Value::Null]
    }))
    .expect("load");
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("loaded and embedded 3 fields in {load_s:.1} s ({empty_titles} documents without a title get no title vector)");

    // Rankings: (name, ORDER BY, which query vector :q is).
    let mut runs: Vec<(String, String, Option<&str>)> = vec![("bm25 (title + text)".into(), "bm25(body, :t, syntax => 'plain')".into(), None)];
    for (name, from) in FIELDS {
        runs.push((format!("{} vector of {}", a.model, from.join(" + ")), format!("similarity({name}, :q)"), Some("model")));
    }
    runs.push((format!("rrf(bm25, {} vector of title + text)", a.model), "rrf(bm25(body, :t, syntax => 'plain'), similarity(emb_both, :q))".into(), Some("model")));
    runs.push(("reference: Cohere embed-english-v3 vector of title + text".into(), "similarity(cohere, :q)".into(), Some("cohere")));

    let judged: Vec<_> = coll.queries.iter().filter(|q| coll.qrels.get(&q.id).is_some_and(|j| j.values().any(|&g| g > 0))).collect();
    let qvecs: Vec<Value> = judged.iter().map(|q| Value::Vector(model.embed_query(&q.text).unwrap_or_else(|e| panic!("{e}")).into())).collect();
    let mut rows = Vec::new();
    println!("{:<62} {:>9} {:>11}", format!("{} ({} queries)", coll.name, judged.len()), "nDCG@10", "recall@100");
    for (label, order, vector) in &runs {
        let sql = format!("SELECT id FROM docs ORDER BY {order} DESC LIMIT 100");
        let (mut ndcg, mut rec) = (Vec::new(), Vec::new());
        for (q, mv) in judged.iter().zip(&qvecs) {
            let mut p = Params::new().set("t", Value::Str(q.text.clone()));
            match *vector {
                Some("model") => p = p.set("q", mv.clone()),
                Some(_) => p = p.set("q", Value::Vector(q.vector.clone())),
                None => {}
            }
            let r = db.query(&sql, &p).unwrap_or_else(|e| panic!("{sql}: {e}"));
            let ranked: Vec<String> = r.rows.iter().filter_map(|row| row[0].as_str().map(String::from)).collect();
            let j: &HashMap<String, u32> = &coll.qrels[&q.id];
            let relevant: Vec<String> = j.iter().filter(|(_, &g)| g > 0).map(|(d, _)| d.clone()).collect();
            ndcg.push(strata::metrics::ndcg_at_k(&ranked, j, 10));
            rec.push(strata::metrics::recall(&ranked, &relevant));
        }
        let (n, r) = (strata::metrics::mean(&ndcg), strata::metrics::mean(&rec));
        println!("{label:<62} {n:>9.4} {r:>11.4}");
        rows.push(json!({"ranking": label, "order_by": order, "ndcg_at_10": n, "recall_at_100": r}));
    }
    if let Some(out) = &a.out {
        let report = json!({
            "dataset": coll.name, "documents": coll.docs.len(), "queries": judged.len(), "documents_without_title": empty_titles,
            "model": a.model, "dimensions": dim, "load_and_embed_seconds": load_s, "rankings": rows,
        });
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p).expect("create the output directory");
        }
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap()).expect("write the results");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A fresh directory under the system's temporary directory.
fn tempfile_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("strata-embed-eval-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("create a temporary directory");
    d
}
