//! A server whose embedded field names a model behind an API (an
//! `endpoint.json` pointing at a mock of OpenAI's API on loopback): rows
//! written as plain text get their vectors from it, `{"embed": text}`
//! parameters too, `/v1/models` says which models would open here,
//! `/metrics` and `/v1/schema` count the requests, and a model that keeps
//! failing fails the write with 503 and writes nothing.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value as Json};
use strata::db::Db;
use strata::embed::{Embedder, HashEmbedder};
use strata::schema::Schema;
use strata::storage::DbOptions;
use strata_server::auth::Access;
use strata_server::service::{AgentConfig, Backend, Service};
use strata_server::server;
use tokio::net::TcpListener;

const SCHEMA: &str = r#"{"table": "docs", "key": "id", "fields": [
    {"name": "id", "type": "keyword"},
    {"name": "title", "type": "text"},
    {"name": "emb", "type": "vector", "dim": 16, "index": "flat", "embed": {"from": ["title"], "model": "gpu"}}
]}"#;

/// A mock of OpenAI's embeddings API: the hash model's vectors, or 503s
/// while `down`. Counts the requests.
fn mock(down: Arc<AtomicBool>, seen: Arc<AtomicUsize>) -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/embeddings", l.local_addr().unwrap());
    std::thread::spawn(move || {
        for conn in l.incoming().flatten() {
            let (down, seen) = (down.clone(), seen.clone());
            std::thread::spawn(move || answer(conn, &down, &seen));
        }
    });
    url
}

fn answer(conn: TcpStream, down: &AtomicBool, seen: &AtomicUsize) {
    let mut r = BufReader::new(conn.try_clone().unwrap());
    let mut len = 0;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).unwrap_or(0) == 0 {
            return;
        }
        if h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
            let _ = k;
            len = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body).unwrap();
    seen.fetch_add(1, Ordering::SeqCst);
    let (status, text) = match down.load(Ordering::SeqCst) {
        true => (503, json!({"error": {"message": "overloaded"}}).to_string()),
        false => {
            let req: Json = serde_json::from_slice(&body).unwrap();
            let texts: Vec<&str> = req["input"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
            let vs = HashEmbedder { dim: 16 }.embed(&texts).unwrap();
            let data: Vec<Json> = vs.iter().enumerate().map(|(i, v)| json!({"embedding": v, "index": i})).collect();
            (200, json!({"data": data, "usage": {"total_tokens": 2 * texts.len()}}).to_string())
        }
    };
    let mut w = conn;
    let _ = write!(w, "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}", text.len());
}

async fn post(c: &reqwest::Client, url: &str, body: Json) -> (u16, Json) {
    let r = c.post(url).json(&body).send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Json::Null))
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_and_queries_embedded_by_a_hosted_model() {
    let (down, seen) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(0)));
    let url = mock(down.clone(), seen.clone());
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    for (name, config) in [("gpu", json!({"api": "openai", "url": url, "dim": 16, "retries": 0})), ("needs-key", json!({"api": "openai", "url": url, "dim": 16, "key_env": "STRATA_SERVER_TEST_UNSET_KEY"}))] {
        std::fs::create_dir_all(models.join(name)).unwrap();
        std::fs::write(models.join(name).join("endpoint.json"), config.to_string()).unwrap();
    }
    let schema = Schema::from_json(SCHEMA).unwrap();
    let db = Db::create(&dir.path().join("h.strata"), schema.clone(), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let agent = AgentConfig { api_key: None, model: strata_server::agent::DEFAULT_MODEL.into(), base_url: "http://unused".into() };
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema, agent, 16, Duration::from_secs(10));
    svc.open_models(&models);
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(server::serve_http(Arc::new(svc), l, Access::default()));
    let c = reqwest::Client::new();

    let r: Json = c.get(format!("{base}/v1/schema")).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["embeddings"][0]["loaded"], true, "{r}");
    assert_eq!(r["embeddings"][0]["usage"]["requests"], 0, "{r}");

    // Plain text in.
    let colours = ["red", "blue", "green", "black"];
    let ops: Vec<Json> = (0..40).map(|i| json!({"op": "insert", "doc": {"id": format!("d{i:02}"), "title": format!("{} mug {i}", colours[i % 4])}})).collect();
    let (status, w) = post(&c, &format!("{base}/v1/write"), json!({"ops": ops})).await;
    assert_eq!((status, w["written"].as_u64()), (200, Some(40)), "{w}");
    assert_eq!(seen.load(Ordering::SeqCst), 1);

    // Meaning out.
    let find = |text: &str| json!({"sql": "SELECT id FROM docs ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": {"embed": text}}});
    let (status, r) = post(&c, &format!("{base}/v1/query"), find("green mug 6")).await;
    assert_eq!((status, &r["rows"][0][0]), (200, &json!("d06")), "{r}");
    // The same text again is the cached vector.
    post(&c, &format!("{base}/v1/query"), find("green mug 6")).await;
    assert_eq!(seen.load(Ordering::SeqCst), 2);

    // Which models would open here.
    let m: Json = c.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    let named = |n: &str| m["models"].as_array().unwrap().iter().find(|x| x["name"] == n).cloned().unwrap_or(Json::Null);
    assert_eq!((named("gpu")["ready"].clone(), named("gpu")["hosted"].clone(), named("gpu")["dim"].clone()), (json!(true), json!(true), json!(16)), "{m}");
    assert_eq!(named("needs-key")["ready"], false, "{m}");
    assert!(named("needs-key")["error"].as_str().unwrap().contains("$STRATA_SERVER_TEST_UNSET_KEY is not set"), "{m}");
    assert_eq!(named("openai/text-embedding-3-small")["dim"], 1536, "{m}");
    assert_eq!(named("hash:64"), Json::Null);

    // What it cost.
    let metrics = c.get(format!("{base}/metrics")).send().await.unwrap().text().await.unwrap();
    assert!(metrics.contains("strata_embed_requests_total{table=\"docs\",field=\"emb\",model=\"gpu\"} 2"), "{metrics}");
    assert!(metrics.contains("strata_embed_tokens_total{table=\"docs\",field=\"emb\",model=\"gpu\"} 82"), "{metrics}");
    let r: Json = c.get(format!("{base}/v1/schema")).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["embeddings"][0]["usage"], json!({"requests": 2, "retries": 0, "failures": 0, "texts": 41, "tokens": 82}), "{r}");

    // A model that fails fails the write, which writes nothing.
    down.store(true, Ordering::SeqCst);
    let (status, w) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "insert", "doc": {"id": "x1", "title": "teapot"}}]})).await;
    assert_eq!(status, 503, "{w}");
    assert!(w["error"].as_str().unwrap_or_default().contains("overloaded"), "{w}");
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT COUNT(*) FROM docs"})).await;
    assert_eq!(r["rows"][0][0], 40, "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/sql"), json!({"sql": "INSERT INTO docs (id, title) VALUES ('x2', 'kettle')"})).await;
    assert_eq!(status, 503, "{r}");
}
