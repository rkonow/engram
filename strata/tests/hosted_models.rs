//! Embedding models behind an HTTP API (`strata::hosted`), against a mock
//! server on loopback that answers in each API's shape: writes that leave a
//! vector out get it from the API in batches, queries are embedded as
//! queries and cached, rate limits and server errors are retried, a
//! refused key or a wrong answer fails the write and writes nothing, and a
//! text too long for the model is cut and sent again.

#![cfg(feature = "hosted")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value as J};
use strata::db::Db;
use strata::embed::{Embedder, HashEmbedder};
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

/// A request the mock took.
#[derive(Clone, Debug)]
struct Seen {
    path: String,
    headers: Vec<(String, String)>,
    body: J,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

type Answer = (u16, Vec<(String, String)>, String);
type Handler = dyn Fn(&Seen, usize) -> Answer + Send + Sync;

/// An HTTP server on loopback answering each request with `handler` (given
/// the request and how many came before it).
struct Mock {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Mock {
    fn start(handler: impl Fn(&Seen, usize) -> Answer + Send + Sync + 'static) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let handler: Arc<Handler> = Arc::new(handler);
        let s = seen.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(conn) = conn else { continue };
                let (s, h) = (s.clone(), handler.clone());
                std::thread::spawn(move || serve(conn, &s, &*h));
            }
        });
        Mock { url, seen }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn serve(conn: TcpStream, seen: &Mutex<Vec<Seen>>, handler: &Handler) {
    let mut r = BufReader::new(conn.try_clone().unwrap());
    let mut line = String::new();
    if r.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        r.read_line(&mut h).unwrap();
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').unwrap();
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    let len: usize = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length")).map_or(0, |(_, v)| v.parse().unwrap());
    let mut body = vec![0; len];
    r.read_exact(&mut body).unwrap();
    let req = Seen { path, headers, body: serde_json::from_slice(&body).unwrap_or(J::Null) };
    let n = {
        let mut s = seen.lock().unwrap();
        s.push(req.clone());
        s.len() - 1
    };
    let (status, extra, text) = handler(&req, n);
    let mut out = format!("HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n", text.len());
    for (k, v) in extra {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(&text);
    let mut w = conn;
    let _ = w.write_all(out.as_bytes());
}

/// The mock model's vector of a text: the hash model's, so that search by
/// meaning works as with `hash:N`.
fn vector_of(text: &str, dim: usize) -> Vec<f32> {
    HashEmbedder { dim }.embed(&[text]).unwrap().pop().unwrap()
}

/// Whether two vectors are equal but for rounding (JSON, and normalizing
/// again).
fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// The texts a request asks for, in the shape of `api`.
fn texts_of(api: &str, body: &J) -> Vec<String> {
    let strs = |j: &J| j.as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    match api {
        "cohere" => strs(&body["texts"]),
        "gemini" => body["requests"].as_array().unwrap().iter().map(|r| r["content"]["parts"][0]["text"].as_str().unwrap().to_string()).collect(),
        _ => strs(&body["input"]),
    }
}

/// An answer in the shape of `api` with the vectors of `texts`, those of
/// OpenAI's shape listed last first (they carry their index).
fn answer(api: &str, texts: &[String], dim: usize) -> Answer {
    let vs: Vec<Vec<f32>> = texts.iter().map(|t| vector_of(t, dim)).collect();
    let body = match api {
        "cohere" => json!({"embeddings": {"float": vs}, "meta": {"billed_units": {"input_tokens": texts.len()}}}),
        "gemini" => json!({"embeddings": vs.iter().map(|v| json!({"values": v})).collect::<Vec<_>>()}),
        "ollama" => json!({"embeddings": vs, "prompt_eval_count": texts.len()}),
        _ => json!({"data": vs.iter().enumerate().rev().map(|(i, v)| json!({"embedding": v, "index": i})).collect::<Vec<_>>(), "usage": {"total_tokens": texts.len()}}),
    };
    (200, Vec::new(), body.to_string())
}

/// A models directory holding `name/endpoint.json`.
fn models(dir: &Path, name: &str, config: J) -> std::path::PathBuf {
    let models = dir.join("models");
    std::fs::create_dir_all(models.join(name)).unwrap();
    std::fs::write(models.join(name).join("endpoint.json"), config.to_string()).unwrap();
    models
}

/// id, title, body, emb (`model`'s embedding of title and body).
fn schema(model: &str, dim: usize) -> Schema {
    Schema::new(
        "docs",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("title", FieldType::Text { positions: false }),
            Field::new("body", FieldType::Text { positions: false }),
            Field::new("emb", FieldType::Vector { dim, index: VectorIndexKind::Flat }).not_stored().embedded(model, &["title", "body"]),
        ],
        Some("id"),
    )
    .unwrap()
}

fn open(dir: &Path, model: &str, dim: usize, models: &Path) -> Db {
    let db = Db::create(&dir.join("h.strata"), schema(model, dim), DbOptions::default()).unwrap();
    let failed = db.open_embedders(models);
    assert!(failed.is_empty(), "{failed:?}");
    db
}

fn emb_of(db: &Db, key: &str) -> Option<Vec<f32>> {
    let r = db.query("SELECT emb FROM docs WHERE id = :k", &Params::new().set("k", Value::Str(key.into()))).unwrap();
    r.rows.first().and_then(|row| row[0].as_vector().map(|v| v.to_vec()))
}

fn count(db: &Db) -> usize {
    db.query("SELECT id FROM docs LIMIT 10000", &Params::new()).unwrap().rows.len()
}

fn insert(db: &Db, rows: &[(String, String, String)]) -> strata::Result<u64> {
    let values: Vec<String> = rows.iter().map(|(id, t, b)| format!("('{id}', '{t}', '{b}')")).collect();
    match db.sql(&format!("INSERT INTO docs (id, title, body) VALUES {}", values.join(", ")), &Params::new())? {
        strata::db::SqlResult::Written(n) => Ok(n),
        strata::db::SqlResult::Rows(_) => panic!("an INSERT writes"),
    }
}

fn rows(n: usize, from: usize) -> Vec<(String, String, String)> {
    (from..from + n).map(|i| (format!("d{i:04}"), format!("{} mug {i}", ["red", "blue", "green", "black"][i % 4]), "ceramic".into())).collect()
}

#[test]
fn writes_and_queries_by_an_openai_compatible_server() {
    let mock = Mock::start(|r, _| answer("openai", &texts_of("openai", &r.body), 16));
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("STRATA_TEST_HOSTED_KEY_1", "sk-test-1");
    let config = json!({"api": "openai", "url": format!("{}/v1/embeddings", mock.url), "model": "bge-m3", "dim": 16, "key_env": "STRATA_TEST_HOSTED_KEY_1", "batch": 64, "document_prefix": "passage: ", "query_prefix": "query: ", "headers": {"x-team": "search"}});
    let models = models(dir.path(), "gpu-box", config);
    let db = open(dir.path(), "gpu-box", 16, &models);

    assert_eq!(insert(&db, &rows(300, 0)).unwrap(), 300);
    // 300 texts in batches of 64.
    let seen = mock.seen();
    assert_eq!(seen.len(), 5);
    assert!(seen.iter().all(|s| s.path == "/v1/embeddings" && s.header("authorization") == Some("Bearer sk-test-1") && s.header("x-team") == Some("search")));
    assert!(seen.iter().all(|s| s.body["model"] == "bge-m3" && s.body.get("dimensions").is_none()));
    let mut sent: Vec<String> = seen.iter().flat_map(|s| texts_of("openai", &s.body)).collect();
    sent.sort();
    assert_eq!(sent.len(), 300);
    assert!(sent.iter().all(|t| t.starts_with("passage: ")));
    // Each row has the vector of its own text, whatever order the API answered in.
    assert!(close(&emb_of(&db, "d0007").unwrap(), &vector_of("passage: black mug 7\nceramic", 16)));
    assert!(close(&emb_of(&db, "d0298").unwrap(), &vector_of("passage: green mug 298\nceramic", 16)));

    // A query is embedded as a query, once.
    let e = db.embedder("emb").unwrap();
    let q = e.embed_query("blue mug 5\nceramic").unwrap();
    assert!(close(&q, &vector_of("query: blue mug 5\nceramic", 16)));
    assert_eq!(e.embed_query("blue mug 5\nceramic").unwrap(), q);
    assert_eq!(mock.seen().len(), 6);
    let r = db.query("SELECT id FROM docs ORDER BY similarity(emb, :q) DESC LIMIT 1", &Params::new().set("q", Value::Vector(q.into()))).unwrap();
    assert_eq!(r.rows[0][0], Value::Str("d0005".into()));
    let u = e.usage().unwrap();
    assert_eq!((u.requests, u.retries, u.failures, u.texts, u.tokens), (6, 0, 0, 301, 301));

    // An UPDATE of a source asks again for that row only.
    db.sql("UPDATE docs SET title = 'teapot' WHERE id = 'd0001'", &Params::new()).unwrap();
    assert_eq!(texts_of("openai", &mock.seen()[6].body), ["passage: teapot\nceramic"]);
    assert!(close(&emb_of(&db, "d0001").unwrap(), &vector_of("passage: teapot\nceramic", 16)));
}

#[test]
fn rate_limits_and_server_errors_are_retried() {
    // Two rate limits, a server error, then answers.
    let mock = Mock::start(|r, n| match n {
        0 | 1 => (429, vec![("retry-after".into(), "0".into())], json!({"error": {"message": "Rate limit reached"}}).to_string()),
        2 => (503, Vec::new(), "upstream down".into()),
        _ => answer("openai", &texts_of("openai", &r.body), 16),
    });
    let dir = tempfile::tempdir().unwrap();
    let models = models(dir.path(), "m", json!({"api": "openai", "url": format!("{}/v1/embeddings", mock.url), "dim": 16}));
    let db = open(dir.path(), "m", 16, &models);
    assert_eq!(insert(&db, &rows(10, 0)).unwrap(), 10);
    assert!(close(&emb_of(&db, "d0003").unwrap(), &vector_of("black mug 3\nceramic", 16)));
    let u = db.embedder("emb").unwrap().usage().unwrap();
    assert_eq!((u.requests, u.retries, u.failures), (4, 3, 0));
}

#[test]
fn a_model_that_keeps_failing_fails_the_write() {
    let mock = Mock::start(|_, _| (500, Vec::new(), json!({"message": "internal error"}).to_string()));
    let dir = tempfile::tempdir().unwrap();
    let models = models(dir.path(), "m", json!({"api": "cohere", "url": format!("{}/v2/embed", mock.url), "dim": 16, "retries": 2}));
    let db = open(dir.path(), "m", 16, &models);
    let e = insert(&db, &rows(5, 0)).unwrap_err();
    assert!(matches!(e, strata::Error::Unavailable(_)), "{e}");
    let m = e.to_string();
    assert!(m.contains("answered 500: internal error") && m.contains("after 3 tries"), "{m}");
    assert_eq!(count(&db), 0);
    assert_eq!(mock.seen().len(), 3);
    // A query fails alike.
    assert!(db.embedder("emb").unwrap().embed_query("mug").is_err());
}

#[test]
fn a_refused_key_fails_at_once_without_showing_it() {
    let mock = Mock::start(|_, _| (401, Vec::new(), json!({"error": {"message": "Incorrect API key provided"}}).to_string()));
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("STRATA_TEST_HOSTED_KEY_2", "sk-very-secret");
    let models = models(dir.path(), "m", json!({"api": "openai", "url": format!("{}/v1/embeddings", mock.url), "dim": 16, "key_env": "STRATA_TEST_HOSTED_KEY_2"}));
    let db = open(dir.path(), "m", 16, &models);
    let m = insert(&db, &rows(3, 0)).unwrap_err().to_string();
    assert!(m.contains("401") && m.contains("Incorrect API key") && m.contains("$STRATA_TEST_HOSTED_KEY_2"), "{m}");
    assert!(!m.contains("sk-very-secret"), "{m}");
    assert_eq!(mock.seen().len(), 1);
    assert_eq!(count(&db), 0);
}

#[test]
fn answers_of_the_wrong_size_fail_the_write() {
    // Eight numbers a vector for a field of 16, and too few vectors.
    let mock = Mock::start(|r, n| {
        let texts = texts_of("openai", &r.body);
        match n {
            0 => answer("openai", &texts, 8),
            _ => answer("openai", &texts[1..], 16),
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let models = models(dir.path(), "m", json!({"api": "openai", "url": format!("{}/v1/embeddings", mock.url), "dim": 16}));
    let db = open(dir.path(), "m", 16, &models);
    let m = insert(&db, &rows(3, 0)).unwrap_err().to_string();
    assert!(m.contains("answered 8-dimension vectors; the field holds 16"), "{m}");
    let m = insert(&db, &rows(3, 0)).unwrap_err().to_string();
    assert!(m.contains("answered 2 vectors for 3 texts"), "{m}");
    assert_eq!(count(&db), 0);
}

#[test]
fn texts_too_long_for_the_model_are_cut_and_sent_again() {
    let mock = Mock::start(|r, _| {
        let texts = texts_of("openai", &r.body);
        match texts.iter().any(|t| t.chars().count() > 1000) {
            true => (400, Vec::new(), json!({"error": {"message": "This model's maximum context length is 8192 tokens, however you requested 9000 tokens"}}).to_string()),
            false => answer("openai", &texts, 16),
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let models = models(dir.path(), "m", json!({"api": "openai", "url": format!("{}/v1/embeddings", mock.url), "dim": 16}));
    let db = open(dir.path(), "m", 16, &models);
    let long = "word ".repeat(1000);
    insert(&db, &[("big".into(), "title".into(), long.clone())]).unwrap();
    let seen = mock.seen();
    let lens: Vec<usize> = seen.iter().map(|s| texts_of("openai", &s.body)[0].chars().count()).collect();
    assert_eq!(lens, [5006, 2503, 1251, 625]);
    let sent = texts_of("openai", &seen[3].body).remove(0);
    assert!(close(&emb_of(&db, "big").unwrap(), &vector_of(&sent, 16)));
}

#[test]
fn each_api_in_its_own_shape() {
    for (api, path, key_var, key_header) in [("cohere", "/v2/embed", "STRATA_TEST_COHERE", "authorization"), ("voyage", "/v1/embeddings", "STRATA_TEST_VOYAGE", "authorization"), ("gemini", "/v1beta/models/gemini-embedding-001:batchEmbedContents", "STRATA_TEST_GEMINI", "x-goog-api-key"), ("jina", "/v1/embeddings", "STRATA_TEST_JINA", "authorization"), ("ollama", "/api/embed", "STRATA_TEST_OLLAMA", "authorization")] {
        let (model, dim) = match api {
            "gemini" => ("gemini-embedding-001", 768),
            "jina" => ("jina-embeddings-v3", 1024),
            _ => ("m", 16),
        };
        // The dimensions asked for, else the field's.
        let mock = Mock::start(move |r, _| {
            let asked = r.body.get("output_dimension").or(r.body.get("dimensions")).or(r.body.pointer("/requests/0/outputDimensionality")).and_then(J::as_u64);
            answer(api, &texts_of(api, &r.body), asked.map_or(dim, |d| d as usize))
        });
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var(key_var, "k-123");
        let config = json!({"api": api, "url": format!("{}{path}", mock.url), "model": model, "key_env": key_var});
        let models = models(dir.path(), "m", config);
        let db = open(dir.path(), "m", dim, &models);
        insert(&db, &rows(4, 0)).unwrap();
        assert!(close(&emb_of(&db, "d0002").unwrap(), &vector_of("green mug 2\nceramic", dim)), "{api}");
        let q = db.embedder("emb").unwrap().embed_query("red mug").unwrap();
        assert!(close(&q, &vector_of("red mug", dim)), "{api}");
        let seen = mock.seen();
        assert_eq!(seen.len(), 2, "{api}");
        assert!(seen.iter().all(|s| s.path == path), "{api}");
        let key = if key_header == "authorization" { "Bearer k-123" } else { "k-123" };
        assert_eq!(seen[0].header(key_header), Some(key), "{api}");
        let (doc, query) = (&seen[0].body, &seen[1].body);
        match api {
            "cohere" => assert_eq!((doc["input_type"].as_str(), query["input_type"].as_str()), (Some("search_document"), Some("search_query"))),
            "voyage" => assert_eq!((doc["input_type"].as_str(), query["input_type"].as_str()), (Some("document"), Some("query"))),
            "gemini" => {
                assert_eq!((doc["requests"][0]["taskType"].as_str(), query["requests"][0]["taskType"].as_str()), (Some("RETRIEVAL_DOCUMENT"), Some("RETRIEVAL_QUERY")));
                // gemini-embedding-001 makes 3,072 numbers and is asked for the field's 768.
                assert_eq!(doc["requests"][0]["outputDimensionality"], 768);
            }
            "jina" => {
                assert_eq!((doc["task"].as_str(), query["task"].as_str()), (Some("retrieval.passage"), Some("retrieval.query")));
                assert!(doc.get("dimensions").is_none());
            }
            _ => assert_eq!(doc["model"], "m"),
        }
    }
}

#[test]
fn a_hosted_model_without_its_key_does_not_open() {
    let dir = tempfile::tempdir().unwrap();
    // Not a key Strata reads, so no test sets it.
    let models = models(dir.path(), "m", json!({"api": "openai", "url": "http://127.0.0.1:9/v1/embeddings", "dim": 16, "key_env": "STRATA_TEST_UNSET_KEY"}));
    let db = Db::create(&dir.path().join("h.strata"), schema("m", 16), DbOptions::default()).unwrap();
    let failed = db.open_embedders(&models);
    assert_eq!(failed.len(), 1);
    assert!(failed[0].1.to_string().contains("$STRATA_TEST_UNSET_KEY is not set"), "{}", failed[0].1);
    // A model Strata knows opens for a field of its size, and for fewer
    // dimensions when it can make them.
    let models = dir.path().join("none");
    std::env::set_var("OPENAI_API_KEY", "sk-unused");
    for dim in [1536, 512] {
        let e = strata::embed::open_for("openai/text-embedding-3-small", Some(dim), &models).unwrap();
        assert_eq!(e.dim(), dim);
    }
    assert!(strata::embed::open_for("openai/text-embedding-ada-002", Some(512), &models).is_err());
    assert_eq!(strata::embed::known_dim("voyage/voyage-3.5"), Some(1024));
}
