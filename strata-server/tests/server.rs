//! The server answers like the embedded database: over HTTP, and as a
//! coordinator over shard servers reached by gRPC. Writes over HTTP and
//! through a coordinator, the token, and the agent's tool loop against a
//! stand-in for the Messages API.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value as Json};
use strata::db::Db;
use strata::query::Params;
use strata::storage::{DbOptions, Layout};
use strata::value::Value;
use strata_server::auth::{self, Access, Secret, Tls};
use strata_server::keys::Principal;
use strata_server::replication::{Acks, Follow, Primary, PrimaryOptions};
use strata_server::service::{shard_of, After, AgentConfig, Backend, QueryReq, Replication, Service, MAX_LEASES};
use strata_server::coord::{CoordOptions, Coordinator};
use strata_server::slots::{self, SlotMap};
use strata_server::{demo, json as js, pb, server};
use tokio::net::TcpListener;

const N: usize = 3000;

fn agent_config(base_url: &str) -> AgentConfig {
    AgentConfig { api_key: Some("test-key".into()), model: strata_server::agent::DEFAULT_MODEL.into(), base_url: base_url.into() }
}

fn create(dir: &std::path::Path, name: &str, keep: impl Fn(usize) -> bool) -> Db {
    let opts = DbOptions { segment_max_docs: 700, publish_snapshots: true, ..DbOptions::default() };
    let db = Db::create(&dir.join(name), demo::schema(N), opts).unwrap();
    db.set_embedder("emb", Arc::new(strata::embed::HashEmbedder { dim: demo::DIM })).unwrap();
    db.load((0..N).filter(|&i| keep(i)).map(|i| demo::product(i, 7))).unwrap();
    db
}

/// The demo's vector of product 42 (what `{"vector_of": "p0000042"}` binds).
fn vector_42(db: &Db) -> Value {
    db.query("SELECT emb FROM products WHERE id = 'p0000042'", &Params::new()).unwrap().rows[0][0].clone()
}

fn service(db: Db, base_url: &str) -> Arc<Service> {
    let schema = db.schema().clone();
    Arc::new(Service::new(Backend::Local(Arc::new(db)), schema, agent_config(base_url), 16, Duration::from_secs(10)))
}

/// A server of `db`, set up by `f` (its replication).
fn service_with(db: Arc<Db>, f: impl FnOnce(&mut Service)) -> Arc<Service> {
    let schema = db.schema().clone();
    let mut svc = Service::new(Backend::Local(db), schema, agent_config("http://unused"), 16, Duration::from_secs(10));
    f(&mut svc);
    Arc::new(svc)
}

async fn http(svc: Arc<Service>, token: Option<&str>) -> String {
    http_with(svc, Access { token: token.map(Secret::new), ..Default::default() }).await
}

/// The server's URL, `https://localhost:port` with TLS.
async fn http_with(svc: Arc<Service>, access: Access) -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let url = if access.tls.is_some() { format!("https://localhost:{port}") } else { format!("http://127.0.0.1:{port}") };
    tokio::spawn(server::serve_http(svc, l, access));
    url
}

async fn grpc(svc: Arc<Service>) -> String {
    grpc_with(svc, Access::default()).await
}

async fn grpc_with(svc: Arc<Service>, access: Access) -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let url = if access.tls.is_some() { format!("https://localhost:{port}") } else { format!("http://127.0.0.1:{port}") };
    tokio::spawn(server::serve_grpc(svc, l, access));
    url
}

/// The test certificate for localhost and the CA that signed it.
fn test_tls() -> (Tls, Vec<u8>) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tls");
    let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
    (Tls { cert: read("cert.pem"), key: read("key.pem") }, read("ca.pem"))
}

async fn post(c: &reqwest::Client, url: &str, body: Json) -> (u16, Json) {
    let r = c.post(url).json(&body).send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Json::Null))
}

/// Rows equal, in order, except that rows tied on score may swap and
/// floats may differ in the last digits.
fn same(sql: &str, want: &Json, got: &Json) {
    let (wr, gr) = (want["rows"].as_array().unwrap(), got["rows"].as_array().unwrap());
    assert_eq!(wr.len(), gr.len(), "{sql}: row counts");
    assert_eq!(want["columns"], got["columns"], "{sql}");
    let score = |j: &Json, i: usize| j["scores"].get(i).and_then(|s| s.as_f64());
    let mut i = 0;
    while i < wr.len() {
        let mut j = i + 1;
        while j < wr.len() && score(want, j).zip(score(want, i)).is_some_and(|(a, b)| (a - b).abs() < 1e-9) {
            j += 1;
        }
        let key = |r: &Json| r[0].to_string();
        let mut a: Vec<String> = wr[i..j].iter().map(key).collect();
        let mut b: Vec<String> = gr[i..j].iter().map(key).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{sql}: rows {i}..{j}");
        if j == i + 1 {
            for (u, v) in wr[i].as_array().unwrap().iter().zip(gr[i].as_array().unwrap()) {
                match (u.as_f64(), v.as_f64()) {
                    (Some(p), Some(q)) => assert!((p - q).abs() < 1e-6 * p.abs().max(1.0), "{sql}: {p} vs {q}"),
                    _ => assert_eq!(u, v, "{sql}: row {i}"),
                }
            }
        }
        i = j;
    }
}

fn queries() -> Vec<(&'static str, Json)> {
    let like = json!({"q": {"vector_of": "p0000042"}});
    let none = json!({});
    vec![
        ("SELECT id, title, score() FROM products ORDER BY bm25(body, 'waterproof hiking boots') DESC LIMIT 10", none.clone()),
        ("SELECT id, price, score() FROM products WHERE price < 150 AND category IN ('outdoor', 'sports') ORDER BY bm25(body, 'lightweight') DESC LIMIT 20", none.clone()),
        ("SELECT id, score() FROM products ORDER BY bm25(body, 'quiet kettle') DESC LIMIT 5 OFFSET 4", none.clone()),
        ("SELECT id, score() FROM products ORDER BY similarity(emb, :q) DESC LIMIT 10", like.clone()),
        ("SELECT id, score() FROM products WHERE rating >= 4 ORDER BY similarity(emb, :q) DESC LIMIT 10", like.clone()),
        ("SELECT id, score() FROM products ORDER BY rrf(bm25(body, 'espresso'), similarity(emb, :q)) DESC LIMIT 10", like.clone()),
        ("SELECT id, score() FROM products WHERE year >= 2020 ORDER BY 0.4 * minmax(bm25(body, 'recycled')) + 0.6 * minmax(similarity(emb, :q)) DESC LIMIT 15", like.clone()),
        ("SELECT id, price FROM products WHERE tags = 'bestseller' ORDER BY price ASC, id LIMIT 12", none.clone()),
        ("SELECT id, tags FROM products WHERE tags IN ('sale', 'new') AND MATCH(body, '\"cast iron\"') ORDER BY sold DESC, id LIMIT 10", none.clone()),
        ("SELECT brand, COUNT(*) FROM products WHERE MATCH(body, 'waterproof') GROUP BY brand ORDER BY 2 DESC, 1 LIMIT 8", none.clone()),
        ("SELECT tags, COUNT(*) FROM products GROUP BY tags ORDER BY 2 DESC, 1 LIMIT 10", none.clone()),
        ("SELECT COUNT(*) FROM products WHERE MATCH(body, 'leather OR suede') AND price BETWEEN 20 AND 300", none.clone()),
        ("SELECT id, year, score() FROM products ORDER BY bm25(body, 'waterproof jacket') * decay(year, 2026, 2) DESC LIMIT 10", none.clone()),
        ("SELECT id, price, score() FROM products WHERE category = 'kitchen' ORDER BY decay(price, 50, 20, shape => 'gauss') DESC LIMIT 10", none.clone()),
        ("SELECT id, score() FROM products ORDER BY similarity(emb, :q) * decay(rating, 5, 1) DESC LIMIT 10", like.clone()),
        // Ranking expressions: BM25 scored with the whole collection's
        // statistics, and a rank-safe boost.
        ("SELECT id, score() FROM products ORDER BY ln(1 + bm25(body, 'waterproof hiking')) + 0.1 * coalesce(rating, 0) DESC LIMIT 10", none.clone()),
        ("SELECT id, score() FROM products WHERE price < 200 ORDER BY bm25(body, 'leather boots') * (1 + 0.2 * coalesce(rating, 0)) DESC LIMIT 8", none.clone()),
        // Some of the conditions, and a boost by how many hold.
        ("SELECT id, price FROM products WHERE AT_LEAST(2, MATCH(body, 'waterproof'), price < 200, year >= 2020) ORDER BY price DESC, id LIMIT 10", none.clone()),
        ("SELECT id, score() FROM products WHERE AT_LEAST('50%', rating >= 4, tags = 'sale', category = 'outdoor') ORDER BY bm25(body, 'leather boots') * (1 + count_true(rating >= 4, year >= 2020)) DESC LIMIT 8", none.clone()),
        // Autocomplete: a prefix's most rated, and a prefix counted.
        ("SELECT id, rating FROM products WHERE PREFIX(id, 'p00001') ORDER BY rating DESC, id LIMIT 10", none.clone()),
        ("SELECT COUNT(*) FROM products WHERE category LIKE 'k%' AND id NOT LIKE 'p0000%'", none.clone()),
        ("SELECT id, rating FROM products WHERE PREFIX(category, 'kicthen', fuzzy => 'auto') ORDER BY rating DESC, id LIMIT 10", none.clone()),
    ]
}

/// HTTP answers equal the embedded database's.
#[tokio::test(flavor = "multi_thread")]
async fn http_answers_like_the_embedded_database() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "p.strata", |_| true);
    let mut want = Vec::new();
    for (sql, params) in queries() {
        let p = if params["q"].is_object() { Params::new().set("q", vector_42(&db)) } else { Params::new() };
        want.push(js::result(&db.query(sql, &p).unwrap(), 0.0));
    }
    let base = http(service(db, "http://unused"), None).await;
    let c = reqwest::Client::new();
    for ((sql, params), want) in queries().into_iter().zip(&want) {
        let (status, got) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "params": params})).await;
        assert_eq!(status, 200, "{sql}: {got}");
        same(sql, want, &got);
    }
    let (status, e) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT nope FROM products"})).await;
    assert_eq!(status, 400, "{e}");
    assert!(e["error"].as_str().unwrap().contains("nope"));
    let health: Json = c.get(format!("{base}/v1/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(health["docs"], N);
    let metrics = c.get(format!("{base}/metrics")).send().await.unwrap().text().await.unwrap();
    assert!(metrics.contains("strata_queries_total{kind=\"bm25\",outcome=\"ok\"} 3"), "{metrics}");
    assert!(metrics.contains("strata_queries_total{kind=\"invalid\",outcome=\"error\"} 1"), "{metrics}");
    let cores = std::thread::available_parallelism().unwrap().get();
    assert!(metrics.contains(&format!("strata_query_pool_threads {cores}\n")) && metrics.contains("strata_query_pool_queued 0\n"), "{metrics}");
}

/// Writes over HTTP: operations, SQL, counters, versions; then reads.
#[tokio::test(flavor = "multi_thread")]
async fn writes_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let base = http(service(create(dir.path(), "w.strata", |i| i < 500), "http://unused"), None).await;
    let c = reqwest::Client::new();
    let get = |id: &'static str| {
        let c = c.clone();
        let base = base.clone();
        async move {
            let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id, price, sold, tags, reviews FROM products WHERE id = :k", "params": {"k": id}})).await;
            r["rows"].get(0).cloned()
        }
    };
    let sold0 = get("p0000003").await.unwrap()[2].as_i64().unwrap();
    let doc = json!({"id": "n1", "title": "Test kettle", "description": "A quiet kettle", "brand": "Hearthly", "category": "kitchen", "tags": ["quiet", "new"], "price": 20.5, "rating": 4.0, "reviews": 3, "sold": 0, "year": 2026});
    let ops = json!({"ops": [
        {"op": "insert", "doc": doc},
        {"op": "add", "key": "p0000003", "field": "sold", "value": 5},
        {"op": "add", "key": "p0000003", "field": "sold", "value": 2},
        {"op": "set", "key": "p0000003", "field": "price", "value": 9.99},
        {"op": "upsert_if_newer", "doc": {"id": "n1", "reviews": 2, "price": 1.0}, "version": "reviews"},
        {"op": "delete", "key": "p0000004"},
    ]});
    let (status, r) = post(&c, &format!("{base}/v1/write"), ops).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["written"], 5, "the older version is skipped");
    let p3 = get("p0000003").await.unwrap();
    assert_eq!(p3[1], 9.99);
    assert_eq!(p3[2].as_i64().unwrap(), sold0 + 7);
    assert_eq!(get("n1").await.unwrap()[3], json!(["quiet", "new"]));
    assert!(get("p0000004").await.is_none());

    // SQL writes.
    let sql_url = format!("{base}/v1/sql");
    let sql = |s: &str| post(&c, &sql_url, json!({"sql": s}));
    let (_, r) = sql("UPDATE products SET sold = sold + 1 WHERE id = 'p0000003'").await;
    assert_eq!(r["written"], 1);
    let (_, r) = sql("INSERT OR REPLACE INTO products (id, title, price, tags, reviews) VALUES ('n1', 'Kettle v2', 25.0, ARRAY['steel'], 9)").await;
    assert_eq!(r["written"], 1, "{r}");
    let n1 = get("n1").await.unwrap();
    assert_eq!((n1[1].clone(), n1[3].clone(), n1[4].clone()), (json!(25.0), json!(["steel"]), json!(9)));
    let (_, r) = sql("DELETE FROM products WHERE category = 'kitchen' AND price > 300").await;
    assert!(r["written"].as_u64().unwrap() > 0, "{r}");
    let (_, r) = sql("SELECT COUNT(*) FROM products WHERE category = 'kitchen' AND price > 300").await;
    assert_eq!(r["rows"][0][0], 0);
    assert_eq!(get("p0000003").await.unwrap()[2].as_i64().unwrap(), sold0 + 8);

    // Refused: a bad op, an unknown field.
    let (status, _) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "frobnicate"}]})).await;
    assert_eq!(status, 400);
    let (status, _) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "insert", "doc": {"id": "x", "nope": 1}}]})).await;
    assert_eq!(status, 400);
}

/// Concurrent writers, and merges committing beside them, all get through:
/// a write waits for the writer instead of failing as busy.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_writes_wait_for_the_writer() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(create(dir.path(), "c.strata", |i| i < 500));
    let schema = db.schema().clone();
    let svc = Arc::new(Service::new(Backend::Local(db.clone()), schema, agent_config("http://unused"), 16, Duration::from_secs(30)));
    let base = http(svc, None).await;
    let c = reqwest::Client::new();
    let sold = |r: &Json| r["rows"][0][0].as_i64().unwrap();
    let count = json!({"sql": "SELECT sold FROM products WHERE id = 'p0000003'"});
    let sold0 = sold(&post(&c, &format!("{base}/v1/query"), count.clone()).await.1);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let maintenance = {
        let (db, stop) = (db.clone(), stop.clone());
        std::thread::spawn(move || {
            let policy = strata::storage::MergePolicy { merge_factor: 2, floor_docs: 100_000, ..Default::default() };
            while !stop.load(Ordering::Relaxed) {
                let _ = db.storage().flush();
                let _ = db.storage().merge(&policy);
            }
        })
    };
    let writers: Vec<_> = (0..12)
        .map(|w| {
            let (c, base) = (c.clone(), base.clone());
            tokio::spawn(async move {
                for i in 0..10 {
                    let (status, r) = if (w + i) % 2 == 0 {
                        post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "add", "key": "p0000003", "field": "sold", "value": 1}]})).await
                    } else {
                        post(&c, &format!("{base}/v1/sql"), json!({"sql": "UPDATE products SET sold = sold + 1 WHERE id = 'p0000003'"})).await
                    };
                    assert_eq!((status, r["written"].as_u64()), (200, Some(1)), "{r}");
                }
            })
        })
        .collect();
    for w in writers {
        w.await.unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    maintenance.join().unwrap();
    assert_eq!(sold(&post(&c, &format!("{base}/v1/query"), count).await.1), sold0 + 120);
}

/// Write requests that wait together share one commit; one that fails is
/// not applied at all, and the others sharing its commit still are.
#[tokio::test(flavor = "multi_thread")]
async fn grouped_writes_fail_alone() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(create(dir.path(), "g.strata", |i| i < 200));
    let svc = Arc::new(Service::new(Backend::Local(db.clone()), db.schema().clone(), agent_config("http://unused"), 16, Duration::from_secs(30)));
    let base = http(svc, None).await;
    let c = reqwest::Client::new();
    let sold = |r: &Json| r["rows"][0][0].as_i64().unwrap();
    let count = json!({"sql": "SELECT sold FROM products WHERE id = 'p0000004'"});
    let sold0 = sold(&post(&c, &format!("{base}/v1/query"), count.clone()).await.1);
    let requests: Vec<_> = (0..64)
        .map(|i| {
            let (c, base) = (c.clone(), base.clone());
            tokio::spawn(async move {
                let mut ops = vec![json!({"op": "add", "key": "p0000004", "field": "sold", "value": 1})];
                if i % 8 == 0 {
                    // A key that is already live: this request fails as a whole.
                    ops.push(json!({"op": "insert", "doc": {"id": "p0000001", "title": "again"}}));
                }
                post(&c, &format!("{base}/v1/write"), json!({"ops": ops})).await.0
            })
        })
        .collect();
    let mut statuses = Vec::new();
    for r in requests {
        statuses.push(r.await.unwrap());
    }
    assert_eq!(statuses.iter().filter(|&&s| s == 200).count(), 56, "{statuses:?}");
    assert_eq!(statuses.iter().filter(|&&s| s == 400).count(), 8, "{statuses:?}");
    assert_eq!(sold(&post(&c, &format!("{base}/v1/query"), count).await.1), sold0 + 56);
}

/// A retried write with an idempotency key is applied once (DB-2), and an
/// `add` after an upsert of the same key in one request reads the upserted
/// value (bug 13).
#[tokio::test(flavor = "multi_thread")]
async fn retries_apply_once_and_adds_read_earlier_ops() {
    let dir = tempfile::tempdir().unwrap();
    let base = http(service(create(dir.path(), "i.strata", |i| i < 100), "http://unused"), None).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/write");
    let sold = |id: &'static str| {
        let (c, base) = (c.clone(), base.clone());
        async move { post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT sold FROM products WHERE id = :k", "params": {"k": id}})).await.1["rows"][0][0].clone() }
    };

    let ops = json!([
        {"op": "upsert", "doc": {"id": "p0000005", "title": "Five", "sold": 1}},
        {"op": "add", "key": "p0000005", "field": "sold", "value": 5},
    ]);
    let (status, r) = post(&c, &url, json!({"ops": ops})).await;
    assert_eq!((status, r["written"].clone()), (200, json!(2)), "{r}");
    assert_eq!(sold("p0000005").await, json!(6));

    // The same request three times, with a key: applied once, answered alike.
    let add = json!([{"op": "add", "key": "p0000005", "field": "sold", "value": 10}, {"op": "insert", "doc": {"id": "n9", "title": "New"}}]);
    for _ in 0..3 {
        let (status, r) = post(&c, &url, json!({"ops": add, "idempotency_key": "req-1"})).await;
        assert_eq!((status, r["written"].clone()), (200, json!(2)), "{r}");
    }
    assert_eq!(sold("p0000005").await, json!(16));
    // The key in a header works the same.
    for _ in 0..2 {
        let r = c.post(&url).header("Idempotency-Key", "req-2").json(&json!({"ops": [{"op": "add", "key": "p0000005", "field": "sold", "value": 1}]})).send().await.unwrap();
        assert_eq!(r.status(), 200);
    }
    assert_eq!(sold("p0000005").await, json!(17));
    // A key reused for other operations is refused.
    let (status, _) = post(&c, &url, json!({"ops": [{"op": "delete", "key": "n9"}], "idempotency_key": "req-1"})).await;
    assert_eq!(status, 422);
    // A request that failed is not remembered, so its retry runs.
    let bad = json!([{"op": "insert", "doc": {"id": "n9", "title": "again"}}]);
    assert_eq!(post(&c, &url, json!({"ops": bad, "idempotency_key": "req-3"})).await.0, 400);
    let (status, _) = post(&c, &url, json!({"ops": [{"op": "delete", "key": "n9"}], "idempotency_key": "req-4"})).await;
    assert_eq!(status, 200);
    let (status, r) = post(&c, &url, json!({"ops": bad, "idempotency_key": "req-3"})).await;
    assert_eq!((status, r["written"].clone()), (200, json!(1)), "{r}");

    // Integers stay exact past 2^53; an INT takes integer increments only.
    let big = 9007199254740993i64;
    let set = json!([{"op": "set", "key": "p0000005", "field": "sold", "value": big - 1}, {"op": "add", "key": "p0000005", "field": "sold", "value": 1}]);
    assert_eq!(post(&c, &url, json!({"ops": set})).await.0, 200);
    assert_eq!(sold("p0000005").await, json!(big));
    let (status, _) = post(&c, &url, json!({"ops": [{"op": "add", "key": "p0000005", "field": "sold", "value": 1.5}]})).await;
    assert_eq!(status, 400);
    let (status, _) = post(&c, &url, json!({"ops": [{"op": "add", "key": "p0000005", "field": "sold", "value": 1}], "idempotency_key": ""})).await;
    assert_eq!(status, 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_token_guards_the_api() {
    let dir = tempfile::tempdir().unwrap();
    let base = http(service(create(dir.path(), "t.strata", |i| i < 100), "http://unused"), Some("s3cret")).await;
    let c = reqwest::Client::new();
    let q = json!({"sql": "SELECT COUNT(*) FROM products"});
    assert_eq!(post(&c, &format!("{base}/v1/query"), q.clone()).await.0, 401);
    let r = c.post(format!("{base}/v1/query")).bearer_auth("wrong").json(&q).send().await.unwrap();
    assert_eq!(r.status(), 401);
    let r = c.post(format!("{base}/v1/query")).bearer_auth("s3cret").json(&q).send().await.unwrap();
    assert_eq!(r.status(), 200);
    // The UI page and health stay open, so the page can ask for the token.
    assert_eq!(c.get(format!("{base}/")).send().await.unwrap().status(), 200);
    assert_eq!(c.get(format!("{base}/v1/health")).send().await.unwrap().status(), 200);
}

/// OPS-1: the gRPC port took writes from anyone, a token on HTTP or not.
/// Every gRPC call now needs the token, and a coordinator sends it.
#[tokio::test(flavor = "multi_thread")]
async fn a_token_guards_grpc() {
    let dir = tempfile::tempdir().unwrap();
    let token = Some(Secret::new("sh4rd"));
    let url = grpc_with(service(create(dir.path(), "g.strata", |i| i < 100), "http://unused"), Access { token: token.clone(), ..Default::default() }).await;
    let mut c = pb::strata_client::StrataClient::connect(url.clone()).await.unwrap();
    let delete = pb::WriteRequest { sql: "DELETE FROM products".into(), ..Default::default() };
    let e = c.write(delete.clone()).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::Unauthenticated);
    let mut req = tonic::Request::new(delete);
    req.metadata_mut().insert("authorization", "Bearer wrong".parse().unwrap());
    assert_eq!(c.write(req).await.unwrap_err().code(), tonic::Code::Unauthenticated);
    assert!(c.health(pb::Empty {}).await.is_err());

    let err = Coordinator::connect(vec![url.clone()]).await.err().unwrap();
    assert!(err.message.contains("token"), "{}", err.message);
    let coord = Coordinator::connect_with(vec![url], CoordOptions { token, ..Default::default() }).await.unwrap();
    let svc = Arc::new(Service::new(Backend::Cluster(Arc::new(coord)), demo::schema(N), agent_config("http://unused"), 16, Duration::from_secs(10)));
    let rs = svc.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await.unwrap().rs;
    assert_eq!(rs.rows[0][0], Value::Int(100));
}

/// Bug 14 and OPS-2: with no token, any web page could write through the
/// visitor's browser: a text/plain POST to /v1/import needs no preflight,
/// and a page that rebinds its own name to 127.0.0.1 reads answers too.
#[tokio::test(flavor = "multi_thread")]
async fn pages_on_other_sites_are_kept_out() {
    let dir = tempfile::tempdir().unwrap();
    let hosts = auth::allowed_hosts("127.0.0.1".parse().unwrap(), &["db.internal".into()]);
    let base = http_with(service(create(dir.path(), "b.strata", |i| i < 100), "http://unused"), Access { hosts, ..Default::default() }).await;
    let host = base.trim_start_matches("http://").to_string();
    let c = reqwest::Client::new();
    let csv = "id,title\nx1,from a page\n";
    let import = format!("{base}/v1/import?format=csv");

    // The attack as the review sent it: refused for its origin, and for its
    // content type even without one.
    let r = c.post(&import).header("content-type", "text/plain").header("origin", "https://evil.example").body(csv).send().await.unwrap();
    assert_eq!(r.status(), 403);
    let r = c.post(&import).header("content-type", "text/plain").body(csv).send().await.unwrap();
    assert_eq!(r.status(), 415);
    let r = c.post(&import).body(csv).send().await.unwrap();
    assert_eq!(r.status(), 415);
    let count = json!({"sql": "SELECT COUNT(*) FROM products"});
    assert_eq!(post(&c, &format!("{base}/v1/query"), count.clone()).await.1["rows"][0][0], 100);
    // A JSON write from another origin, too.
    let w = json!({"ops": [{"op": "delete", "key": "p0000001"}]});
    let r = c.post(format!("{base}/v1/write")).header("origin", "https://evil.example").json(&w).send().await.unwrap();
    assert_eq!(r.status(), 403);

    // The UI's own requests, and clients that say what they send, pass.
    let r = c.post(&import).header("content-type", "text/csv").header("origin", format!("http://{host}")).body(csv).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = c.post(format!("{base}/v1/write")).header("origin", format!("http://{host}")).json(&w).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(post(&c, &format!("{base}/v1/query"), count).await.1["rows"][0][0], 100);

    // DNS rebinding: the page's own host name, on every route.
    let port = host.rsplit(':').next().unwrap();
    for path in ["/", "/v1/health", "/v1/schema"] {
        let r = c.get(format!("{base}{path}")).header("host", format!("evil.example:{port}")).send().await.unwrap();
        assert_eq!(r.status(), 403, "{path}");
    }
    for name in ["localhost", "db.internal", "DB.internal."] {
        let r = c.get(format!("{base}/v1/health")).header("host", format!("{name}:{port}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "{name}");
    }
}

/// OPS-1: TLS on both listeners, and a coordinator that checks its shards'
/// certificates against a CA.
#[tokio::test(flavor = "multi_thread")]
async fn tls_on_both_listeners() {
    let dir = tempfile::tempdir().unwrap();
    let (tls, ca) = test_tls();
    let svc = service(create(dir.path(), "s.strata", |i| i < 100), "http://unused");
    let access = Access { token: Some(Secret::new("t")), tls: Some(tls), ..Default::default() };
    let base = http_with(svc.clone(), access.clone()).await;
    let shard = grpc_with(svc, access).await;
    assert!(base.starts_with("https://") && shard.starts_with("https://"));

    let c = reqwest::Client::builder().add_root_certificate(reqwest::Certificate::from_pem(&ca).unwrap()).build().unwrap();
    let r = c.post(format!("{base}/v1/query")).bearer_auth("t").json(&json!({"sql": "SELECT COUNT(*) FROM products"})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Json>().await.unwrap()["rows"][0][0], 100);
    // Plain HTTP to the TLS port gets nothing, and a client without the CA
    // does not trust the certificate.
    assert!(reqwest::get(base.replace("https://", "http://") + "/v1/health").await.is_err());
    assert!(reqwest::get(format!("{base}/v1/health")).await.is_err());

    let opts = |ca| CoordOptions { token: Some(Secret::new("t")), ca, ..Default::default() };
    assert!(Coordinator::connect_with(vec![shard.clone()], opts(None)).await.is_err());
    let coord = Coordinator::connect_with(vec![shard], opts(Some(ca))).await.unwrap();
    assert_eq!(coord.schema.table, "products");
}

/// A coordinator over three shard servers answers like one database
/// holding every document, and routes writes by key.
#[tokio::test(flavor = "multi_thread")]
async fn coordinator_answers_like_one_database() {
    let dir = tempfile::tempdir().unwrap();
    let whole = create(dir.path(), "whole.strata", |_| true);
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let mut urls = Vec::new();
    let mut held = 0;
    for s in 0..3 {
        let db = create(dir.path(), &format!("s{s}.strata"), |i| shard_of(&key(i), 3) == s);
        held += db.snapshot().n_live_docs();
        urls.push(grpc(service(db, "http://unused")).await);
    }
    assert_eq!(held, N as u64);
    let c = Coordinator::connect(urls.clone()).await.unwrap();
    let schema = c.schema.clone();
    let coord = Arc::new(Service::new(Backend::Cluster(Arc::new(c)), schema, agent_config("http://unused"), 16, Duration::from_secs(10)));
    for (sql, params) in queries() {
        let p = if params["q"].is_object() { Params::new().set("q", vector_42(&whole)) } else { Params::new() };
        let want = js::result(&whole.query(sql, &p).unwrap(), 0.0);
        let params = coord.params(Some(&params)).await.unwrap();
        let got = coord.query(QueryReq { sql: sql.into(), params, ..Default::default() }).await.unwrap();
        same(sql, &want, &js::result(&got.rs, 0.0));
    }
    // A compound query's leaves reach the shards as their plans, `- -1` and
    // a column named like a keyword (`year`) included.
    for sql in [
        "SELECT a.id, b.id FROM products a JOIN products b ON a.brand = b.brand WHERE a.year >= - -2022 AND b.id = 'p0000042' ORDER BY a.price DESC, a.id LIMIT 5",
        "SELECT id, price FROM products WHERE id IN (SELECT id FROM products WHERE price > - -100 AND year >= 2020) ORDER BY price DESC LIMIT 10",
        // Rerank phases: a second phase each shard scores, a global one.
        "WITH top AS (SELECT * FROM products ORDER BY bm25(body, 'waterproof hiking') DESC LIMIT 60) SELECT id, score() FROM top ORDER BY bm25(body, 'boots') + coalesce(rating, 0) DESC LIMIT 10",
        "WITH a AS (SELECT * FROM products WHERE price < 300 ORDER BY bm25(body, 'leather') DESC LIMIT 80), b AS (SELECT * FROM a ORDER BY rating DESC LIMIT 30) SELECT id, price, score() FROM b ORDER BY minmax(bm25(body, 'jacket')) - 0.5 * minmax(price) DESC LIMIT 8 OFFSET 2",
        // Features: the fetch round scores BM25 with the query's
        // statistics, and so does one round.
        "SELECT id, title, features() FROM products ORDER BY bm25(body, 'waterproof hiking') * (1 + 0.1 * coalesce(rating, 0)) DESC LIMIT 10",
        "SELECT id, bm25(body, 'boots') AS b, ln(1 + bm25(body, 'leather')) - 0.01 * price AS x FROM products ORDER BY price DESC LIMIT 12",
        "WITH top AS (SELECT * FROM products ORDER BY bm25(body, 'waterproof hiking') DESC LIMIT 60) SELECT id, score(), features() FROM top ORDER BY bm25(body, 'boots') + coalesce(rating, 0) DESC LIMIT 10",
    ] {
        let want = js::result(&whole.query(sql, &Params::new()).unwrap(), 0.0);
        assert!(!want["rows"].as_array().unwrap().is_empty(), "{sql}");
        let got = coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap_or_else(|e| panic!("{sql}: {}", e.message));
        same(sql, &want, &js::result(&got.rs, 0.0));
        if sql.contains("features()") {
            assert!(got.rs.stats.plan.iter().any(|l| l.starts_with("fetch: ")), "{sql}: {:?}", got.rs.stats.plan);
        }
    }
    let h = coord.health().await;
    assert_eq!((h["status"].as_str(), h["docs"].as_u64()), (Some("ok"), Some(N as u64)), "{h}");
    // A coordinator runs a read's plan, a compound query's included, as it
    // runs the SQL: the shards take the plan's leaves. It takes writes as
    // SQL until its shards take write plans.
    for sql in [
        "SELECT id, price FROM products WHERE year >= 2020 ORDER BY price DESC, id LIMIT 7",
        "SELECT id, price FROM products WHERE id IN (SELECT id FROM products WHERE price > 100 AND year >= 2020) ORDER BY price DESC LIMIT 10",
        "SELECT a.id, b.id FROM products a JOIN products b ON a.brand = b.brand WHERE b.id = 'p0000042' ORDER BY a.price DESC, a.id LIMIT 5",
        "WITH top AS (SELECT * FROM products ORDER BY bm25(body, 'waterproof hiking') DESC LIMIT 60) SELECT id, score() FROM top ORDER BY bm25(body, 'boots') + coalesce(rating, 0) DESC LIMIT 10",
    ] {
        let plan = match strata::query::compound::is_compound(sql, &coord.schema) {
            true => strata::query::compound::plan(sql, &coord.schema, &Params::new()).unwrap(),
            false => strata::query::ir::Plan::bind(sql, &coord.schema, &Params::new()).unwrap(),
        };
        let want = js::result(&whole.query(sql, &Params::new()).unwrap(), 0.0);
        let Ok(strata_server::service::SqlAnswer::Rows(got)) = coord.run_plan(&Principal::open(), plan, After::Any, 0).await else { panic!("{sql}") };
        same(sql, &want, &js::result(&got.rs, 0.0));
    }
    let strata::query::Command::Write { write, .. } = strata::query::bind_command("DELETE FROM products WHERE id = 'p0000001'", &[&coord.schema], &Params::new()).unwrap() else { panic!() };
    let e = coord.run_plan(&Principal::open(), strata::query::ir::Plan::of_write(&coord.schema, &write), After::Any, 0).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("coordinator"), "{}", e.message);

    // Writes: inserts go to their key's shard, UPDATE and DELETE to all.
    let sql = |s: String| {
        let coord = coord.clone();
        async move { coord.sql(&Principal::open(), &s, &Params::new(), After::Any, 0).await.unwrap() }
    };
    let ids: Vec<String> = (0..30).map(|i| format!("new{i}")).collect();
    let values: Vec<String> = ids.iter().map(|id| format!("('{id}', 'Tent {id}', 'outdoor', 10.0, 0)")).collect();
    let n = match sql(format!("INSERT INTO products (id, title, category, price, sold) VALUES {}", values.join(", "))).await {
        strata_server::service::SqlAnswer::Written { w, .. } => w.rows,
        _ => panic!("rows"),
    };
    assert_eq!(n, 30);
    let mut per_shard = [0u64; 3];
    for id in &ids {
        per_shard[shard_of(&Value::Str(id.clone()), 3)] += 1;
    }
    let h = coord.health().await;
    for (s, shard) in h["shards"].as_array().unwrap().iter().enumerate() {
        let before = (0..N).filter(|&i| shard_of(&key(i), 3) == s).count() as u64;
        assert_eq!(shard["docs"].as_u64().unwrap(), before + per_shard[s], "shard {s}");
    }
    sql("UPDATE products SET sold = sold + 4 WHERE MATCH(body, 'tent') AND price = 10".into()).await;
    let count = |s: &str| {
        let coord = coord.clone();
        let s = s.to_string();
        async move { coord.query(QueryReq { sql: s, ..Default::default() }).await.unwrap().rs.rows[0][0].as_i64().unwrap() }
    };
    assert_eq!(count("SELECT COUNT(*) FROM products WHERE sold = 4 AND price = 10").await, 30);
    sql("DELETE FROM products WHERE price = 10".into()).await;
    assert_eq!(count("SELECT COUNT(*) FROM products").await, N as i64);

    // A coordinator given one shard finds the others in the cluster map
    // that shard holds, and routes keys as the first does.
    let one = cluster(Coordinator::connect(vec![urls[0].clone()]).await.unwrap());
    assert_eq!(one.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await.unwrap().rs.rows[0][0].as_i64(), Some(N as i64));
    // A shard that is down: connecting fails with its address.
    let err = Coordinator::connect(vec![urls[0].clone(), "http://127.0.0.1:1".into()]).await.err().unwrap();
    assert!(err.message.contains("127.0.0.1:1"), "{}", err.message);
}

/// A shard server that answers `Schema`, `Hello` and `Slots` and never
/// anything else, like one whose machine froze after the coordinator
/// started.
struct Stuck(String);

#[tonic::async_trait]
impl pb::strata_server::Strata for Stuck {
    async fn query(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        std::future::pending().await
    }
    async fn term_stats(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::TermStatsResponse>, tonic::Status> {
        std::future::pending().await
    }
    async fn write(&self, _: tonic::Request<pb::WriteRequest>) -> Result<tonic::Response<pb::WriteResponse>, tonic::Status> {
        std::future::pending().await
    }
    async fn schema(&self, _: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::SchemaResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::SchemaResponse { schema_json: self.0.clone(), ..Default::default() }))
    }
    async fn health(&self, _: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::HealthResponse>, tonic::Status> {
        std::future::pending().await
    }
    async fn hello(&self, _: tonic::Request<pb::HelloRequest>) -> Result<tonic::Response<pb::HelloResponse>, tonic::Status> {
        Ok(tonic::Response::new(strata_server::grpc::hello()))
    }
    async fn release(&self, _: tonic::Request<pb::ReleaseRequest>) -> Result<tonic::Response<pb::Empty>, tonic::Status> {
        Ok(tonic::Response::new(pb::Empty {}))
    }
    async fn slots(&self, _: tonic::Request<pb::SlotsRequest>) -> Result<tonic::Response<pb::SlotsResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::SlotsResponse { epoch: 1, ..Default::default() }))
    }
    async fn replicate(&self, _: tonic::Request<pb::ReplicateRequest>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn restore(&self, _: tonic::Request<tonic::Streaming<pb::RestoreChunk>>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn verify(&self, _: tonic::Request<pb::VerifyRequest>) -> Result<tonic::Response<pb::VerifyResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn logs(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no log store"))
    }
    async fn keys(&self, _: tonic::Request<pb::KeysRequest>) -> Result<tonic::Response<pb::KeysResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no keys"))
    }
    async fn profile(&self, _: tonic::Request<pb::ProfileRequest>) -> Result<tonic::Response<pb::ProfileResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no profiles"))
    }
    type WriteStreamStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<pb::WriteResponse, tonic::Status>> + Send>>;
    async fn write_stream(&self, _: tonic::Request<tonic::Streaming<pb::WriteRequest>>) -> Result<tonic::Response<Self::WriteStreamStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn sources(&self, _: tonic::Request<pb::SourcesRequest>) -> Result<tonic::Response<pb::SourcesResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn sid_next(&self, _: tonic::Request<pb::SidNextRequest>) -> Result<tonic::Response<pb::SidNextResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no semantic IDs here"))
    }
    async fn guard_counts(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::GuardCountsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no guards here"))
    }
    async fn configure(&self, _: tonic::Request<pb::ConfigureRequest>) -> Result<tonic::Response<pb::ConfigureResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    type ExportSlotsStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<pb::ExportSlotsChunk, tonic::Status>> + Send>>;
    async fn export_slots(&self, _: tonic::Request<pb::ExportSlotsRequest>) -> Result<tonic::Response<Self::ExportSlotsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn freeze_slots(&self, _: tonic::Request<pb::FreezeRequest>) -> Result<tonic::Response<pb::FreezeResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn cleanup_slots(&self, _: tonic::Request<pb::CleanupRequest>) -> Result<tonic::Response<pb::CleanupResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn vote(&self, _: tonic::Request<pb::VoteRequest>) -> Result<tonic::Response<pb::VoteResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
}

/// A shard server that refuses every query because too many are running.
struct Overloaded(String);

#[tonic::async_trait]
impl pb::strata_server::Strata for Overloaded {
    async fn query(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        Err(tonic::Status::resource_exhausted("too many queries running; try again"))
    }
    async fn term_stats(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::TermStatsResponse>, tonic::Status> {
        Err(tonic::Status::resource_exhausted("too many queries running; try again"))
    }
    async fn write(&self, _: tonic::Request<pb::WriteRequest>) -> Result<tonic::Response<pb::WriteResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("read only"))
    }
    async fn schema(&self, _: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::SchemaResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::SchemaResponse { schema_json: self.0.clone(), ..Default::default() }))
    }
    async fn health(&self, _: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::HealthResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::HealthResponse { status: "ok".into(), ..Default::default() }))
    }
    async fn hello(&self, _: tonic::Request<pb::HelloRequest>) -> Result<tonic::Response<pb::HelloResponse>, tonic::Status> {
        Ok(tonic::Response::new(strata_server::grpc::hello()))
    }
    async fn release(&self, _: tonic::Request<pb::ReleaseRequest>) -> Result<tonic::Response<pb::Empty>, tonic::Status> {
        Ok(tonic::Response::new(pb::Empty {}))
    }
    async fn slots(&self, _: tonic::Request<pb::SlotsRequest>) -> Result<tonic::Response<pb::SlotsResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::SlotsResponse { epoch: 1, ..Default::default() }))
    }
    async fn replicate(&self, _: tonic::Request<pb::ReplicateRequest>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn restore(&self, _: tonic::Request<tonic::Streaming<pb::RestoreChunk>>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn verify(&self, _: tonic::Request<pb::VerifyRequest>) -> Result<tonic::Response<pb::VerifyResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not a replica"))
    }
    async fn logs(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no log store"))
    }
    async fn keys(&self, _: tonic::Request<pb::KeysRequest>) -> Result<tonic::Response<pb::KeysResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no keys"))
    }
    async fn profile(&self, _: tonic::Request<pb::ProfileRequest>) -> Result<tonic::Response<pb::ProfileResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no profiles"))
    }
    type WriteStreamStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<pb::WriteResponse, tonic::Status>> + Send>>;
    async fn write_stream(&self, _: tonic::Request<tonic::Streaming<pb::WriteRequest>>) -> Result<tonic::Response<Self::WriteStreamStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn sources(&self, _: tonic::Request<pb::SourcesRequest>) -> Result<tonic::Response<pb::SourcesResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn sid_next(&self, _: tonic::Request<pb::SidNextRequest>) -> Result<tonic::Response<pb::SidNextResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no semantic IDs here"))
    }
    async fn guard_counts(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::GuardCountsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no guards here"))
    }
    async fn configure(&self, _: tonic::Request<pb::ConfigureRequest>) -> Result<tonic::Response<pb::ConfigureResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    type ExportSlotsStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<pb::ExportSlotsChunk, tonic::Status>> + Send>>;
    async fn export_slots(&self, _: tonic::Request<pb::ExportSlotsRequest>) -> Result<tonic::Response<Self::ExportSlotsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn freeze_slots(&self, _: tonic::Request<pb::FreezeRequest>) -> Result<tonic::Response<pb::FreezeResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn cleanup_slots(&self, _: tonic::Request<pb::CleanupRequest>) -> Result<tonic::Response<pb::CleanupResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
    async fn vote(&self, _: tonic::Request<pb::VoteRequest>) -> Result<tonic::Response<pb::VoteResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("a test shard"))
    }
}

async fn stuck(schema: &strata::Schema) -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let svc = pb::strata_server::StrataServer::new(Stuck(serde_json::to_string(schema).unwrap()));
    tokio::spawn(tonic::transport::Server::builder().add_service(svc).serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(l)));
    format!("http://{addr}")
}

fn cluster(c: Coordinator) -> Arc<Service> {
    let schema = c.schema.clone();
    Arc::new(Service::new(Backend::Cluster(Arc::new(c)), schema, agent_config("http://unused"), 16, Duration::from_secs(10)))
}

/// With two replicas, a shard whose first replica is down or hangs still
/// answers, as one database would, and quickly. A shard with no replica
/// answering fails the query, or, with partial answers allowed, is left
/// out and named (and not waited for again for a second). Writes reach
/// every replica.
#[tokio::test(flavor = "multi_thread")]
async fn replicas_cover_for_a_shard_down_or_hung() {
    let dir = tempfile::tempdir().unwrap();
    let whole = create(dir.path(), "whole.strata", |_| true);
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let mut good = Vec::new();
    for s in 0..2 {
        let mut replicas = Vec::new();
        for r in 0..2 {
            let db = create(dir.path(), &format!("s{s}r{r}.strata"), |i| shard_of(&key(i), 2) == s);
            replicas.push(grpc(service(db, "http://unused")).await);
        }
        good.push(replicas);
    }
    let dead = "http://127.0.0.1:1".to_string();
    let hung = stuck(whole.schema()).await;

    // Shard 0's first replica refuses connections; shard 1's never answers.
    let opts = CoordOptions { hedge: Duration::from_millis(50), ..Default::default() };
    let c = Coordinator::connect_with(vec![format!("{dead}|{}", good[0][0]), format!("{hung}|{}", good[1][0])], opts.clone()).await.unwrap();
    let coord = cluster(c);
    for (sql, params) in queries() {
        let p = if params["q"].is_object() { Params::new().set("q", vector_42(&whole)) } else { Params::new() };
        let want = js::result(&whole.query(sql, &p).unwrap(), 0.0);
        let params = coord.params(Some(&params)).await.unwrap();
        let t = std::time::Instant::now();
        let got = coord.query(QueryReq { sql: sql.into(), params, ..Default::default() }).await.unwrap();
        assert!(t.elapsed() < Duration::from_secs(2), "{sql}: {:?}", t.elapsed());
        assert!(got.missing.is_empty());
        same(sql, &want, &js::result(&got.rs, 0.0));
    }
    let h = coord.health().await;
    assert_eq!(h["status"], "degraded", "{h}");
    assert_eq!(h["docs"].as_u64(), Some(N as u64), "{h}");
    assert!(h["retries"].as_u64().unwrap() >= 1 && h["hedges"].as_u64().unwrap() >= 1, "{h}");

    // Shard 1 with no replica answering, given up after 200 ms.
    let sql = "SELECT id, score() FROM products ORDER BY bm25(body, 'waterproof hiking boots') DESC LIMIT 10";
    let opts = CoordOptions { shard_timeout: Some(Duration::from_millis(200)), ..opts };
    let c = Coordinator::connect_with(vec![good[0][1].clone(), hung.clone()], opts.clone()).await.unwrap();
    let strict = cluster(c);
    let t = std::time::Instant::now();
    let err = strict.query(QueryReq { sql: sql.into(), ..Default::default() }).await.err().unwrap();
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    assert_eq!(err.status, 504, "{}", err.message);
    assert!(err.message.contains(&hung), "{}", err.message);
    let a = strict.query(QueryReq { sql: sql.into(), partial: Some(true), ..Default::default() }).await.unwrap();
    assert_eq!(a.missing, vec![hung.clone()]);
    // For the next second, partial answers leave the shard out without waiting for it.
    let t = std::time::Instant::now();
    let again = strict.query(QueryReq { sql: sql.into(), partial: Some(true), ..Default::default() }).await.unwrap();
    assert!(t.elapsed() < Duration::from_millis(150), "{:?}", t.elapsed());
    assert_eq!(again.missing, vec![hung.clone()]);
    assert!(!a.rs.rows.is_empty());
    for row in &a.rs.rows {
        assert_eq!(shard_of(&row[0], 2), 0, "{row:?}");
    }
    // Scored over the shard that answered, as a database of its documents would.
    let only0 = create(dir.path(), "only0.strata", |i| shard_of(&key(i), 2) == 0);
    same(sql, &js::result(&only0.query(sql, &Params::new()).unwrap(), 0.0), &js::result(&a.rs, 0.0));
    let c = Coordinator::connect_with(vec![good[0][1].clone(), hung.clone()], CoordOptions { partial: true, ..opts }).await.unwrap();
    let lenient = cluster(c);
    assert_eq!(lenient.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap().missing, vec![hung.clone()]);
    assert!(lenient.query(QueryReq { sql: sql.into(), partial: Some(false), ..Default::default() }).await.is_err());

    // Writes go to every replica of their shard.
    let c = Coordinator::connect(vec![good[0].join("|"), good[1].join("|")]).await.unwrap();
    let both = cluster(c);
    let ids: Vec<String> = (0..20).map(|i| format!("rep{i}")).collect();
    let values: Vec<String> = ids.iter().map(|id| format!("('{id}', 'Tent {id}', 'outdoor', 10.0, 0)")).collect();
    both.sql(&Principal::open(), &format!("INSERT INTO products (id, title, category, price, sold) VALUES {}", values.join(", ")), &Params::new(), After::Any, 0).await.unwrap();
    let h = both.health().await;
    assert_eq!(h["status"], "ok", "{h}");
    for (s, shard) in h["shards"].as_array().unwrap().iter().enumerate() {
        let want = (0..N).filter(|&i| shard_of(&key(i), 2) == s).count() + ids.iter().filter(|id| shard_of(&Value::Str(id.to_string()), 2) == s).count();
        for r in shard["replicas"].as_array().unwrap() {
            assert_eq!(r["docs"].as_u64(), Some(want as u64), "shard {s}: {r}");
        }
    }
}

/// What a read brought a shard.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Seen {
    protocol: u32,
    typed: bool,
    hold: bool,
    release: bool,
    epoch: u64,
    min_seq_no: u64,
}

/// A shard server that records each read, as a server of protocol
/// `version` would take it: one of 1 is from before `Hello` (it answers
/// Unimplemented, and the fields it does not know, `protocol` and `plan`,
/// never reach it), one of 2 has no leases for a fetch (`hold`, `release`,
/// `Release`), one of 3 no slots (`epoch`, `Slots`), and one of 4 does not
/// wait for a write (`min_seq_no`). While `down`, it
/// fails every read, and while
/// `fail_fetch`, every fetch (a read that ends its lease). `after_hold` is
/// a write it applies once, after answering a read that held its snapshot.
struct Recording {
    inner: strata_server::grpc::Grpc,
    version: AtomicU32,
    down: AtomicBool,
    fail_fetch: AtomicBool,
    after_hold: std::sync::Mutex<Option<String>>,
    seen: std::sync::Mutex<Vec<Seen>>,
    /// Each `Replicate` call's transactions and whether it said the
    /// replica is behind.
    shipped: std::sync::Mutex<Vec<(usize, bool)>>,
    /// False: answers `Verify` Unimplemented, as a server from before it.
    verifies: AtomicBool,
    /// How long each `Replicate` call waits first, as on a slow link; a
    /// `Verify` waits five times that, as when hashing many rows.
    delay: std::sync::Mutex<Duration>,
    /// `Hello` and `Write` calls, answered or not.
    hellos: AtomicUsize,
    writes: AtomicUsize,
}

impl Recording {
    fn up(&self) -> Result<(), tonic::Status> {
        match self.down.load(Ordering::Relaxed) {
            true => Err(tonic::Status::unavailable("restarting")),
            false => Ok(()),
        }
    }

    fn read(&self, mut r: tonic::Request<pb::QueryRequest>) -> Result<tonic::Request<pb::QueryRequest>, tonic::Status> {
        self.up()?;
        let q = r.get_mut();
        self.seen.lock().unwrap().push(Seen { protocol: q.protocol, typed: q.plan.is_some(), hold: q.hold, release: q.release, epoch: q.epoch, min_seq_no: q.min_seq_no });
        let v = self.version.load(Ordering::Relaxed);
        if v < 5 {
            q.min_seq_no = 0;
        }
        if v < 4 {
            q.epoch = 0;
        }
        if v < 3 {
            (q.hold, q.release) = (false, false);
        }
        if v < 2 {
            q.protocol = 0;
            q.plan = None;
        }
        if self.fail_fetch.load(Ordering::Relaxed) && q.release && !q.hold {
            return Err(tonic::Status::unavailable("restarting"));
        }
        Ok(r)
    }

    fn last(&self) -> Option<Seen> {
        self.seen.lock().unwrap().last().copied()
    }

    fn leases(&self) -> usize {
        self.inner.0.sweep_leases()
    }
}

#[tonic::async_trait]
impl pb::strata_server::Strata for Recording {
    async fn query(&self, r: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        let r = self.read(r)?;
        let hold = r.get_ref().hold;
        let a = self.inner.query(r).await?;
        let write = self.after_hold.lock().unwrap().take_if(|_| hold);
        if let Some(sql) = write {
            self.inner.0.sql(&Principal::open(), &sql, &Params::new(), After::Any, 0).await.unwrap();
        }
        Ok(a)
    }
    async fn term_stats(&self, r: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::TermStatsResponse>, tonic::Status> {
        self.inner.term_stats(self.read(r)?).await
    }
    async fn write(&self, r: tonic::Request<pb::WriteRequest>) -> Result<tonic::Response<pb::WriteResponse>, tonic::Status> {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.up()?;
        self.inner.write(r).await
    }
    async fn schema(&self, r: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::SchemaResponse>, tonic::Status> {
        self.up()?;
        self.inner.schema(r).await
    }
    async fn health(&self, r: tonic::Request<pb::Empty>) -> Result<tonic::Response<pb::HealthResponse>, tonic::Status> {
        self.inner.health(r).await
    }
    async fn hello(&self, r: tonic::Request<pb::HelloRequest>) -> Result<tonic::Response<pb::HelloResponse>, tonic::Status> {
        self.hellos.fetch_add(1, Ordering::Relaxed);
        self.up()?;
        match self.version.load(Ordering::Relaxed) {
            1 => Err(tonic::Status::unimplemented("unknown method Hello")),
            v => {
                let h = self.inner.hello(r).await?.into_inner();
                Ok(tonic::Response::new(pb::HelloResponse { max_protocol: v, slot_epoch: if v >= 4 { h.slot_epoch } else { 0 }, ..h }))
            }
        }
    }
    async fn release(&self, r: tonic::Request<pb::ReleaseRequest>) -> Result<tonic::Response<pb::Empty>, tonic::Status> {
        match self.version.load(Ordering::Relaxed) {
            3.. => self.inner.release(r).await,
            _ => Err(tonic::Status::unimplemented("unknown method Release")),
        }
    }
    async fn slots(&self, r: tonic::Request<pb::SlotsRequest>) -> Result<tonic::Response<pb::SlotsResponse>, tonic::Status> {
        self.up()?;
        match self.version.load(Ordering::Relaxed) {
            4.. => self.inner.slots(r).await,
            _ => Err(tonic::Status::unimplemented("unknown method Slots")),
        }
    }
    async fn replicate(&self, r: tonic::Request<pb::ReplicateRequest>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        self.up()?;
        let delay = *self.delay.lock().unwrap();
        tokio::time::sleep(delay).await;
        self.shipped.lock().unwrap().push((r.get_ref().txns.len(), r.get_ref().behind));
        self.inner.replicate(r).await
    }
    async fn restore(&self, r: tonic::Request<tonic::Streaming<pb::RestoreChunk>>) -> Result<tonic::Response<pb::ReplicateResponse>, tonic::Status> {
        self.up()?;
        self.inner.restore(r).await
    }
    async fn profile(&self, _: tonic::Request<pb::ProfileRequest>) -> Result<tonic::Response<pb::ProfileResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no profiles"))
    }
    type WriteStreamStream = <strata_server::grpc::Grpc as pb::strata_server::Strata>::WriteStreamStream;
    async fn write_stream(&self, r: tonic::Request<tonic::Streaming<pb::WriteRequest>>) -> Result<tonic::Response<Self::WriteStreamStream>, tonic::Status> {
        self.up()?;
        self.inner.write_stream(r).await
    }
    async fn sources(&self, r: tonic::Request<pb::SourcesRequest>) -> Result<tonic::Response<pb::SourcesResponse>, tonic::Status> {
        self.up()?;
        match self.version.load(Ordering::Relaxed) {
            6.. => self.inner.sources(r).await,
            _ => Err(tonic::Status::unimplemented("unknown method Sources")),
        }
    }
    async fn logs(&self, _: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::QueryResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no log store"))
    }
    async fn keys(&self, _: tonic::Request<pb::KeysRequest>) -> Result<tonic::Response<pb::KeysResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("no keys"))
    }
    async fn verify(&self, r: tonic::Request<pb::VerifyRequest>) -> Result<tonic::Response<pb::VerifyResponse>, tonic::Status> {
        self.up()?;
        let delay = *self.delay.lock().unwrap();
        tokio::time::sleep(delay * 5).await;
        match self.verifies.load(Ordering::Relaxed) {
            true => self.inner.verify(r).await,
            false => Err(tonic::Status::unimplemented("unknown method Verify")),
        }
    }
    async fn sid_next(&self, r: tonic::Request<pb::SidNextRequest>) -> Result<tonic::Response<pb::SidNextResponse>, tonic::Status> {
        self.up()?;
        self.inner.sid_next(r).await
    }
    async fn guard_counts(&self, r: tonic::Request<pb::QueryRequest>) -> Result<tonic::Response<pb::GuardCountsResponse>, tonic::Status> {
        self.inner.guard_counts(self.read(r)?).await
    }
    async fn configure(&self, r: tonic::Request<pb::ConfigureRequest>) -> Result<tonic::Response<pb::ConfigureResponse>, tonic::Status> {
        self.up()?;
        self.inner.configure(r).await
    }
    type ExportSlotsStream = <strata_server::grpc::Grpc as pb::strata_server::Strata>::ExportSlotsStream;
    async fn export_slots(&self, r: tonic::Request<pb::ExportSlotsRequest>) -> Result<tonic::Response<Self::ExportSlotsStream>, tonic::Status> {
        self.up()?;
        self.inner.export_slots(r).await
    }
    async fn freeze_slots(&self, r: tonic::Request<pb::FreezeRequest>) -> Result<tonic::Response<pb::FreezeResponse>, tonic::Status> {
        self.up()?;
        self.inner.freeze_slots(r).await
    }
    async fn cleanup_slots(&self, r: tonic::Request<pb::CleanupRequest>) -> Result<tonic::Response<pb::CleanupResponse>, tonic::Status> {
        self.up()?;
        self.inner.cleanup_slots(r).await
    }
    async fn vote(&self, r: tonic::Request<pb::VoteRequest>) -> Result<tonic::Response<pb::VoteResponse>, tonic::Status> {
        self.up()?;
        self.inner.vote(r).await
    }
}

async fn recording(db: Db, version: u32) -> (String, Arc<Recording>) {
    recording_of(service(db, "http://unused"), version).await
}

async fn recording_of(svc: Arc<Service>, version: u32) -> (String, Arc<Recording>) {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let shard = Arc::new(Recording {
        inner: strata_server::grpc::Grpc(svc),
        version: AtomicU32::new(version),
        down: AtomicBool::new(false),
        fail_fetch: AtomicBool::new(false),
        after_hold: Default::default(),
        seen: Default::default(),
        shipped: Default::default(),
        verifies: AtomicBool::new(true),
        delay: std::sync::Mutex::new(Duration::ZERO),
        hellos: AtomicUsize::new(0),
        writes: AtomicUsize::new(0),
    });
    let svc = pb::strata_server::StrataServer::from_arc(shard.clone()).max_decoding_message_size(256 << 20);
    tokio::spawn(tonic::transport::Server::builder().add_service(svc).serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(l)));
    (format!("http://{addr}"), shard)
}

/// Waits for the shards to hold no snapshot lease: the coordinator ends
/// those of shards without rows on a page without waiting.
async fn leases_end(shards: &[Arc<Recording>]) {
    for _ in 0..100 {
        if shards.iter().all(|s| s.leases() == 0) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("leases held: {:?}", shards.iter().map(|s| s.leases()).collect::<Vec<_>>());
}

/// A coordinator asks each shard the protocol versions it speaks and
/// speaks the oldest any speaks, as in a rolling upgrade: with its slot
/// map's epoch (protocol 4) when every shard holds a map, query then fetch
/// with typed plans (3) while one does not, typed plans in one round (2)
/// while one holds no snapshots for a fetch, and SQL and JSON plans (1)
/// while one is from before `Hello`. The answers are the same, and from
/// protocol 3 no lease outlives its query.
#[tokio::test(flavor = "multi_thread")]
async fn shards_speak_the_oldest_protocol_among_them() {
    let dir = tempfile::tempdir().unwrap();
    let whole = create(dir.path(), "whole.strata", |_| true);
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let compound = [
        "SELECT a.id, b.id FROM products a JOIN products b ON a.brand = b.brand WHERE a.year >= - -2022 AND b.id = 'p0000042' ORDER BY a.price DESC, a.id LIMIT 5",
        "SELECT id, price FROM products WHERE id IN (SELECT id FROM products WHERE price > 100 AND year >= 2020) ORDER BY price DESC LIMIT 10",
    ];
    for (round, versions) in [[4, 4], [4, 3], [4, 2], [4, 1]].into_iter().enumerate() {
        let mut urls = Vec::new();
        let mut seen = Vec::new();
        for (s, version) in versions.into_iter().enumerate() {
            let db = create(dir.path(), &format!("r{round}s{s}.strata"), |i| shard_of(&key(i), 2) == s);
            let (url, log) = recording(db, version).await;
            urls.push(url);
            seen.push(log);
        }
        let c = Coordinator::connect(urls).await.unwrap();
        let want_protocol = versions[1];
        assert_eq!(c.protocol(), want_protocol);
        let coord = cluster(c);
        assert_eq!(coord.health().await["protocol"], want_protocol, "{versions:?}");
        // Writes go by slots once every shard has them.
        assert_eq!(coord.health().await["slot_epoch"], (want_protocol >= 4) as u64, "{versions:?}");
        for (sql, params) in queries() {
            let p = if params["q"].is_object() { Params::new().set("q", vector_42(&whole)) } else { Params::new() };
            let want = js::result(&whole.query(sql, &p).unwrap(), 0.0);
            let params = coord.params(Some(&params)).await.unwrap();
            let got = coord.query(QueryReq { sql: sql.into(), params, ..Default::default() }).await.unwrap_or_else(|e| panic!("{sql}: {}", e.message));
            same(sql, &want, &js::result(&got.rs, 0.0));
        }
        for sql in compound {
            let want = js::result(&whole.query(sql, &Params::new()).unwrap(), 0.0);
            let got = coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap_or_else(|e| panic!("{sql}: {}", e.message));
            same(sql, &want, &js::result(&got.rs, 0.0));
        }
        // Before protocol 3, a statistics round's leases last until they
        // expire.
        if want_protocol >= 3 {
            leases_end(&seen).await;
        }
        // Statistics alone, as a coordinator in front of this one asks.
        let (stats, _) = coord.term_stats("SELECT id FROM products ORDER BY bm25(body, 'waterproof') DESC LIMIT 3", None, &Params::new(), &After::Any, 0).await.unwrap();
        let (local, _) = service(create(dir.path(), &format!("r{round}w.strata"), |_| true), "http://unused").term_stats("SELECT id FROM products ORDER BY bm25(body, 'waterproof') DESC LIMIT 3", None, &Params::new(), &After::Any, 0).await.unwrap();
        assert_eq!(serde_json::to_value(&stats).unwrap(), serde_json::to_value(&local).unwrap());
        // Every read reached every shard in the protocol spoken, typed from
        // 2, and pages were fetched from 3.
        for shard in &seen {
            let log = shard.seen.lock().unwrap();
            assert!(log.len() > queries().len(), "{versions:?}: {}", log.len());
            assert!(log.iter().all(|s| s.protocol == want_protocol && s.typed == (want_protocol >= 2)), "{versions:?}: {log:?}");
            assert_eq!(log.iter().any(|s| s.hold), want_protocol >= 3, "{versions:?}: {log:?}");
            assert!(log.iter().all(|s| s.epoch == (want_protocol >= 4) as u64), "{versions:?}: {log:?}");
        }
    }
}

/// Guarded queries (`min_rows`, `max_rows`) on the demo's products.
const GUARDED: [&str; 3] = [
    "SELECT id, price FROM products WHERE price < 400 AND AT_LEAST(1, MATCH(body, 'waterproof'), tags = 'sale', rating >= 4, year >= 2022, max_rows => 300) ORDER BY price DESC, id LIMIT 10",
    "SELECT id, score() FROM products ORDER BY bm25(body, 'waterproof hiking boots leather', min => 1, min_rows => 40) DESC LIMIT 10",
    "SELECT id, score() FROM products WHERE MATCH(body, 'lightweight waterproof jacket', min => 1, max_rows => 150) ORDER BY bm25(body, 'jacket') DESC LIMIT 10",
];

/// A guard's level is the cluster's: the shards' counts, summed, choose
/// it as one database's would. A shard before protocol 8 cannot count.
#[tokio::test(flavor = "multi_thread")]
async fn a_guard_counts_the_rows_of_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let whole = create(dir.path(), "whole.strata", |_| true);
    let local = service(create(dir.path(), "local.strata", |_| true), "http://unused");
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let mut urls = Vec::new();
    for s in 0..3 {
        let db = create(dir.path(), &format!("s{s}.strata"), |i| shard_of(&key(i), 3) == s);
        urls.push(grpc(service(db, "http://unused")).await);
    }
    let coord = cluster(Coordinator::connect(urls).await.unwrap());
    let guard = |rs: &strata::query::exec::ResultSet| rs.stats.plan.iter().find(|l| l.starts_with("guard: ")).cloned();
    for sql in GUARDED {
        let want = whole.query(sql, &Params::new()).unwrap();
        assert!(!want.rows.is_empty(), "{sql}");
        let got = coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap_or_else(|e| panic!("{sql}: {}", e.message));
        same(sql, &js::result(&want, 0.0), &js::result(&got.rs, 0.0));
        assert!(guard(&want).is_some() && guard(&want) == guard(&got.rs), "{sql}: {:?} against {:?}", want.stats.plan, got.rs.stats.plan);
        // The counts alone, as a coordinator in front of this one asks.
        let counts = coord.guard_counts(sql, None, &Params::new(), &After::Any, 0).await.unwrap();
        assert_eq!(counts, local.guard_counts(sql, None, &Params::new(), &After::Any, 0).await.unwrap(), "{sql}");
        // A shard runs it at the level its coordinator chose, not its own.
        let e = local.query(QueryReq { sql: sql.into(), shard: true, ..Default::default() }).await.err().unwrap();
        assert!(e.message.contains("runs on shards at the level its coordinator chose"), "{sql}: {}", e.message);
    }
    let e = local.guard_counts("SELECT id FROM products WHERE price < 10", None, &Params::new(), &After::Any, 0).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("no guard"), "{}", e.message);
    let (old, _) = recording(create(dir.path(), "old.strata", |i| shard_of(&key(i), 2) == 0), 7).await;
    let (new, _) = recording(create(dir.path(), "new.strata", |i| shard_of(&key(i), 2) == 1), 8).await;
    let coord = cluster(Coordinator::connect(vec![old, new]).await.unwrap());
    let e = coord.query(QueryReq { sql: GUARDED[0].into(), ..Default::default() }).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("needs 8: upgrade it"), "{}", e.message);
    let sql = "SELECT id, price FROM products WHERE price < 400 AND AT_LEAST(1, MATCH(body, 'waterproof'), tags = 'sale') ORDER BY price DESC, id LIMIT 10";
    coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap_or_else(|e| panic!("{sql}: {}", e.message));
}

/// A replica that fails is asked its versions again before its next read,
/// as it may have restarted on another: one rolled back to a server from
/// before `Hello` gets SQL again.
#[tokio::test(flavor = "multi_thread")]
async fn a_replica_is_asked_its_protocol_again_after_failing() {
    let dir = tempfile::tempdir().unwrap();
    let (url, shard) = recording(create(dir.path(), "db.strata", |_| true), 4).await;
    let coord = cluster(Coordinator::connect(vec![url]).await.unwrap());
    let count = || {
        let coord = coord.clone();
        async move { coord.query(QueryReq { sql: "SELECT COUNT(*) FROM products WHERE year >= 2020".into(), ..Default::default() }).await }
    };
    let n = count().await.unwrap().rs.rows[0][0].clone();
    assert_eq!(shard.last().map(|s| (s.protocol, s.typed)), Some((4, true)));
    shard.down.store(true, Ordering::Relaxed);
    shard.version.store(1, Ordering::Relaxed);
    assert_eq!(count().await.err().unwrap().status, 503);
    shard.down.store(false, Ordering::Relaxed);
    assert_eq!(count().await.unwrap().rs.rows[0][0], n);
    assert_eq!(shard.last().map(|s| (s.protocol, s.typed)), Some((1, false)));
}

/// A read that must see a write goes only to a copy that waits for it. A
/// replica that speaks a protocol before 5 would answer what it holds, so
/// the coordinator sends such a read to the shard's primary until the
/// replica is upgraded, and other reads to either.
#[tokio::test(flavor = "multi_thread")]
async fn a_read_that_names_a_write_skips_a_replica_that_would_not_wait() {
    let dir = tempfile::tempdir().unwrap();
    // Numbered, and a primary's with a log to ship.
    let numbered = |name: &str| {
        let opts = DbOptions { publish_snapshots: true, ship_log_bytes: 64 << 20, ..DbOptions::default() };
        let db = Db::create(&dir.path().join(name), demo::schema(N), opts).unwrap();
        db.set_embedder("emb", Arc::new(strata::embed::HashEmbedder { dim: demo::DIM })).unwrap();
        db.load((0..300).map(|i| demo::product(i, 7))).unwrap();
        db.storage().start_seq_no().unwrap();
        Arc::new(db)
    };
    let (rdb, pdb) = (numbered("r.strata"), numbered("p.strata"));
    let follow = Arc::new(Follow::open(rdb.clone(), None).unwrap());
    let r = service_with(rdb.clone(), |s| s.set_replication(Replication::Replica(follow)));
    let (ru, replica) = recording_of(r.clone(), 4).await;
    // A transaction a batch, for the catch-up below; acknowledged by the
    // primary alone, so that it takes writes while the replica is down.
    let opts = PrimaryOptions { replicas: vec![ru.clone()], acks: Acks::Primary, heartbeat: Duration::from_millis(100), retry_max: Duration::from_millis(200), batch_bytes: 1, ..Default::default() };
    let p = service_with(pdb.clone(), |s| s.set_replication(Replication::Primary(Primary::start(pdb.clone(), None, opts).unwrap())));
    let pu = grpc(p.clone()).await;
    for _ in 0..200 {
        if r.health().await["replication"]["serves_reads"] == true {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // No hedges: each read goes to one copy, and to the next on a failure.
    let c = Coordinator::connect_with(vec![format!("{ru}|{pu}")], CoordOptions { hedge: Duration::from_secs(60), ..Default::default() }).await.unwrap();
    let coord = cluster(c);
    let sold = |after: Vec<u64>| {
        let coord = coord.clone();
        async move {
            let q = QueryReq { sql: "SELECT sold FROM products WHERE id = 'p0000003'".into(), after: After::Shards(after), ..Default::default() };
            coord.query(q).await.unwrap().rs.rows[0][0].as_i64().unwrap()
        }
    };
    let seen = |min: u64| replica.seen.lock().unwrap().iter().filter(|s| s.min_seq_no == min).count();
    // Each pair of reads starts once at each copy.
    for v in 1..=4 {
        let w = coord.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": v})], None).await.unwrap();
        for _ in 0..200 {
            if rdb.storage().seq_no().unwrap().seq >= w.seq_nos[0] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for after in [w.seq_nos.clone(), w.seq_nos.clone(), Vec::new(), Vec::new()] {
            assert_eq!(sold(after).await, v);
        }
        assert_eq!(seen(w.seq_nos[0]), 0, "a read that must see transaction {} never reaches a replica of protocol 4", w.seq_nos[0]);
    }
    assert!(seen(0) >= 4, "other reads do: {}", seen(0));
    // Passed over, it did not fail: it is not avoided.
    let h = coord.health().await;
    assert!(h["shards"][0]["replicas"][0].get("avoided").is_none() && h["retries"].as_u64() >= Some(4), "{h}");

    // Upgraded: it is asked its protocol again after failing a read.
    replica.down.store(true, Ordering::Relaxed);
    replica.version.store(5, Ordering::Relaxed);
    for _ in 0..2 {
        sold(Vec::new()).await;
    }
    replica.down.store(false, Ordering::Relaxed);
    // Avoided for a while after failing.
    tokio::time::sleep(CoordOptions::default().cool_off).await;
    let mut asked = 0;
    for v in 5..=8 {
        let w = coord.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": v})], None).await.unwrap();
        for _ in 0..2 {
            assert_eq!(sold(w.seq_nos.clone()).await, v);
        }
        asked += seen(w.seq_nos[0]);
    }
    assert!(asked >= 4, "reads that name a write reach it now: {asked}");

    // Each batch says whether it reaches the primary's last: a replica that
    // catches up is told it is behind until the last batch.
    replica.down.store(true, Ordering::Relaxed);
    for v in 9..=14 {
        coord.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": v})], None).await.unwrap();
    }
    replica.shipped.lock().unwrap().clear();
    replica.down.store(false, Ordering::Relaxed);
    let last = pdb.storage().seq_no().unwrap().seq;
    // The replica applies the last batch before its answer reaches the
    // primary, which counts it in sync (and so compares it) only then.
    for _ in 0..200 {
        if rdb.storage().seq_no().unwrap().seq == last
            && r.health().await["replication"]["serves_reads"] == true
            && p.health().await["replication"]["replicas"][0]["in_sync"] == true
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let shipped = replica.shipped.lock().unwrap().clone();
    let batches: Vec<bool> = shipped.iter().filter(|(n, _)| *n > 0).map(|&(_, behind)| behind).collect();
    assert!(batches.len() == 6 && batches[..5].iter().all(|&b| b) && !batches[5], "{shipped:?}");
    assert_eq!(r.health().await["replication"]["serves_reads"], true);

    // A replica from before `Verify` is not compared: the round says why,
    // and the replica stays in sync, with no copy.
    replica.verifies.store(false, Ordering::Relaxed);
    let v = p.verify_replicas(Duration::from_secs(10)).await.unwrap();
    let f = &v["replicas"][0];
    assert_eq!((&f["compared"], &f["in_sync"]), (&json!(true), &json!(true)), "{v}");
    assert!(f["error"].as_str().unwrap().contains("Unimplemented: unknown method Verify") && f.get("seq_no").is_none(), "{v}");
    let h = &p.health().await["replication"]["replicas"][0];
    assert_eq!((&h["in_sync"], &h["copies"], &h["verifications"], &h["verify_failures"]), (&json!(true), &json!(0), &json!(0), &json!(1)), "{h}");
    replica.verifies.store(true, Ordering::Relaxed);
    let v = p.verify_replicas(Duration::from_secs(10)).await.unwrap();
    assert_eq!((&v["replicas"][0]["seq_no"], &v["replicas"][0]["differing_slots"], v["replicas"][0].get("error")), (&json!(last), &json!(0), None), "{v}");
}

/// A replica a slow link keeps a few batches behind, under bursts of
/// writes, is compared all the same: a round ships the replica up to its
/// version in batches and not past it, and only the call that reaches the
/// version pins it; rounds due every 50 ms do not overlap, though a
/// `Verify` takes longer.
#[tokio::test(flavor = "multi_thread")]
async fn a_replica_behind_is_compared_under_a_steady_writer() {
    let dir = tempfile::tempdir().unwrap();
    let numbered = |name: &str| {
        let opts = DbOptions { publish_snapshots: true, ship_log_bytes: 64 << 20, ..DbOptions::default() };
        let db = Db::create(&dir.path().join(name), demo::schema(N), opts).unwrap();
        db.set_embedder("emb", Arc::new(strata::embed::HashEmbedder { dim: demo::DIM })).unwrap();
        db.load((0..300).map(|i| demo::product(i, 7))).unwrap();
        db.storage().start_seq_no().unwrap();
        Arc::new(db)
    };
    let (rdb, pdb) = (numbered("r.strata"), numbered("p.strata"));
    // Two writes a batch: the size of one, written before shipping starts.
    let set = |db: &Db, i: i64| {
        let mut tx = db.storage().begin().unwrap();
        assert!(tx.set(&Value::Str("p0000001".into()), "sold", Value::Int(1_000 + i)).unwrap());
        tx.commit_seq().unwrap();
    };
    set(&pdb, 0);
    set(&rdb, 0);
    let strata::storage::LogRead::Txns(t) = pdb.storage().log_from(1, usize::MAX) else { panic!("the log holds it") };
    let follow = Arc::new(Follow::open(rdb.clone(), None).unwrap());
    let r = service_with(rdb.clone(), |s| s.set_replication(Replication::Replica(follow)));
    let (ru, replica) = recording_of(r.clone(), 5).await;
    let opts = PrimaryOptions {
        replicas: vec![ru.clone()],
        acks: Acks::Primary,
        heartbeat: Duration::from_millis(100),
        retry_max: Duration::from_millis(200),
        batch_bytes: 2 * t[0].bytes.len() + 1,
        verify_every: Duration::from_millis(50),
        ..Default::default()
    };
    let p = service_with(pdb.clone(), |s| s.set_replication(Replication::Primary(Primary::start(pdb.clone(), None, opts).unwrap())));
    for _ in 0..200 {
        if p.health().await["replication"]["replicas"][0]["verifications"].as_u64().unwrap() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Five writes every 150 ms; a call takes 40 ms and carries two, and
    // a Verify 200 ms.
    *replica.delay.lock().unwrap() = Duration::from_millis(40);
    let writing = Arc::new(AtomicBool::new(true));
    let writer = {
        let (p, writing) = (p.clone(), writing.clone());
        tokio::spawn(async move {
            let mut i = 0;
            while writing.load(Ordering::Relaxed) {
                for _ in 0..5 {
                    p.write_ops(vec![json!({"op": "set", "key": format!("p{:07}", 10 + i % 200), "field": "sold", "value": i})], None).await.unwrap();
                    i += 1;
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            i
        })
    };
    let before = p.health().await["replication"]["replicas"][0]["verifications"].as_u64().unwrap();
    let mut seen = Vec::new();
    for _ in 0..4 {
        let v = p.verify_replicas(Duration::from_secs(3)).await.unwrap();
        let f = &v["replicas"][0];
        assert_eq!((&f["compared"], &f["differing_slots"], f.get("error")), (&json!(true), &json!(0), None), "{v}");
        seen.push(f["seq_no"].as_u64().unwrap());
    }
    tokio::time::sleep(Duration::from_millis(1200)).await;
    writing.store(false, Ordering::Relaxed);
    assert!(writer.await.unwrap() > 30);
    let shipped = replica.shipped.lock().unwrap().clone();
    let h = &p.health().await["replication"]["replicas"][0];
    assert!(h["verifications"].as_u64().unwrap() >= before + 6, "{h}");
    assert_eq!((&h["verify_failures"], &h["divergences"], &h["copies"], &h["in_sync"]), (&json!(0), &json!(0), &json!(0), &json!(true)), "{h}");
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    // The replica was behind: batches of two.
    assert!(shipped.iter().filter(|(n, _)| *n == 2).count() > 10, "{shipped:?}");
}

/// A replica down when the coordinator connected gets no slot map then; it
/// is asked its protocol and given the map before its first write, which
/// it would refuse (503) without one.
#[tokio::test(flavor = "multi_thread")]
async fn a_replica_down_at_connect_is_given_the_slot_map_before_it_writes() {
    let dir = tempfile::tempdir().unwrap();
    let (a, first) = recording(create(dir.path(), "a.strata", |_| true), 4).await;
    let (b, second) = recording(create(dir.path(), "b.strata", |_| true), 4).await;
    second.down.store(true, Ordering::Relaxed);
    let coord = cluster(Coordinator::connect(vec![format!("{a}|{b}")]).await.unwrap());
    assert_eq!((first.inner.0.slot_epoch(), second.inner.0.slot_epoch()), (1, 0));
    // Refused while it has no map, as a write that skipped the handshake
    // would be.
    second.down.store(false, Ordering::Relaxed);
    let e = second.inner.0.check_epoch(1).err().unwrap();
    assert_eq!(e.status, 503, "{}", e.message);
    coord.write_ops(vec![json!({"op": "upsert", "doc": {"id": "late", "title": "Lamp"}})], None).await.unwrap();
    assert_eq!(second.inner.0.slots(), Some((SlotMap::even(1, 1), 0)));
    for shard in [&first, &second] {
        let n = shard.inner.0.query(QueryReq { sql: "SELECT COUNT(*) FROM products WHERE id = 'late'".into(), ..Default::default() }).await.unwrap().rs.rows[0][0].clone();
        assert_eq!(n, Value::Int(1));
    }
}

/// Query then fetch: for a deep page, the shards rank their rows and
/// answer the keys, and those holding the page's rows fetch its columns
/// from the snapshot they ranked on, though a write came between the
/// rounds; then every lease ends. A shard that fails the fetch is left out
/// of a partial answer with its rows, which are counted, and fails a whole
/// one. A shallow page comes in one round.
#[tokio::test(flavor = "multi_thread")]
async fn a_page_is_fetched_from_the_snapshot_it_was_ranked_on() {
    let dir = tempfile::tempdir().unwrap();
    let whole = create(dir.path(), "whole.strata", |_| true);
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let pages = [
        // Without a statistics round, the query round makes the leases.
        "SELECT id, title, price FROM products WHERE year >= 2020 ORDER BY price DESC, id LIMIT 10 OFFSET 40",
        // With one, its leases are held on.
        "SELECT title, id, score() FROM products ORDER BY bm25(body, 'waterproof hiking boots') DESC LIMIT 10 OFFSET 30",
    ];
    for (n, sql) in pages.into_iter().enumerate() {
        let want = js::result(&whole.query(sql, &Params::new()).unwrap(), 0.0);
        let (mut urls, mut shards) = (Vec::new(), Vec::new());
        for s in 0..2 {
            let (url, shard) = recording(create(dir.path(), &format!("p{n}s{s}.strata"), |i| shard_of(&key(i), 2) == s), 4).await;
            *shard.after_hold.lock().unwrap() = Some("UPDATE products SET title = 'changed'".into());
            urls.push(url);
            shards.push(shard);
        }
        let coord = cluster(Coordinator::connect(urls.clone()).await.unwrap());
        let got = coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap();
        same(sql, &want, &js::result(&got.rs, 0.0));
        for shard in &shards {
            // The write came between the rounds.
            assert!(shard.after_hold.lock().unwrap().is_none());
            let log = shard.seen.lock().unwrap();
            assert!(log.iter().any(|s| s.hold) && log.iter().all(|s| !(s.hold && s.release)), "{log:?}");
        }
        assert!(shards.iter().any(|s| s.seen.lock().unwrap().iter().any(|s| s.release)));
        leases_end(&shards).await;
        let title = want["columns"].as_array().unwrap().iter().position(|c| c == "title").unwrap();
        let later = js::result(&coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap().rs, 0.0);
        assert!(later["rows"].as_array().unwrap().iter().all(|r| r[title] == "changed"), "{later}");
        leases_end(&shards).await;
        // In one round, the shards hold nothing: so asked, or for a page
        // the shards' rows of which are few.
        let one = cluster(Coordinator::connect_with(urls.clone(), CoordOptions { fetch: false, ..Default::default() }).await.unwrap());
        let before = shards[0].seen.lock().unwrap().len();
        same(sql, &later, &js::result(&one.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap().rs, 0.0));
        let shallow = sql.split(" OFFSET").next().unwrap();
        let by_default = js::result(&coord.query(QueryReq { sql: shallow.into(), ..Default::default() }).await.unwrap().rs, 0.0);
        same(shallow, &js::result(&one.query(QueryReq { sql: shallow.into(), ..Default::default() }).await.unwrap().rs, 0.0), &by_default);
        assert!(shards[0].seen.lock().unwrap()[before..].iter().all(|s| !s.hold));

        // A shard that fails the fetch.
        shards[1].fail_fetch.store(true, Ordering::Relaxed);
        let lenient = cluster(Coordinator::connect_with(urls.clone(), CoordOptions { partial: true, ..Default::default() }).await.unwrap());
        let a = lenient.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap();
        assert_eq!(a.missing, vec![urls[1].clone()]);
        let idc = want["columns"].as_array().unwrap().iter().position(|c| c == "id").unwrap();
        let id = |r: &Json| Value::Str(r[idc].as_str().unwrap().to_string());
        let kept: Vec<&Json> = later["rows"].as_array().unwrap().iter().filter(|r| shard_of(&id(r), 2) == 0).collect();
        let a = js::result(&a.rs, 0.0);
        assert_eq!(a["rows"].as_array().unwrap().iter().collect::<Vec<_>>(), kept, "{sql}");
        assert!(kept.len() < 10, "{sql}");
        let lost = 10 - kept.len();
        assert!(a["stats"]["plan"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().contains(&format!("fetch: {lost} rows of the page left out"))), "{a}");
        assert!(lenient.prometheus().await.contains(&format!("strata_fetch_lost_rows_total {lost}\n")));
        assert_eq!(coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.err().unwrap().status, 503);
    }
}

/// A server says which protocol versions and plans it takes, takes a typed
/// plan as its SQL, holds a query's snapshot for a fetch and ends it, and
/// refuses a protocol newer than its own and a plan that needs a feature
/// it lacks, by name.
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_takes_typed_plans_and_refuses_newer_ones() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "db.strata", |_| true);
    let schema = db.schema().clone();
    let svc = service(db, "http://unused");
    let url = grpc(svc.clone()).await;
    let mut c = pb::strata_client::StrataClient::connect(url).await.unwrap();
    let h = c.hello(pb::HelloRequest { min_protocol: 1, max_protocol: 10 }).await.unwrap().into_inner();
    assert_eq!((h.min_protocol, h.max_protocol, h.plan_version, h.slot_epoch), (1, 9, strata::query::ir::VERSION, 0));
    assert!(h.plan_features.iter().any(|f| f == "operators") && h.server.starts_with("strata-server "), "{h:?}");

    let sql = "SELECT id, price FROM products WHERE year >= 2020 AND MATCH(body, 'waterproof') ORDER BY price DESC, id LIMIT 7";
    let by_sql = c.query(pb::QueryRequest { sql: sql.into(), ..Default::default() }).await.unwrap().into_inner();
    let plan = strata_server::plan_pb::to_pb_plan(&strata::query::ir::Plan::bind(sql, &schema, &Params::new()).unwrap());
    // The SQL only names the plan.
    let typed = pb::QueryRequest { sql: "a label".into(), protocol: 2, plan: Some(plan.clone()), ..Default::default() };
    let by_plan = c.query(typed.clone()).await.unwrap().into_inner();
    assert!(!by_sql.rows.is_empty());
    assert_eq!((by_plan.columns, by_plan.rows), (by_sql.columns, by_sql.rows));

    // Protocol 3: a read holds the snapshot it read, a fetch reads it and
    // ends it, and Release ends one that no fetch needs.
    let typed = pb::QueryRequest { protocol: 3, ..typed };
    let held = c.query(pb::QueryRequest { hold: true, ..typed.clone() }).await.unwrap().into_inner();
    assert!(held.snapshot != 0 && svc.sweep_leases() == 1, "{}", held.snapshot);
    assert_eq!(c.query(typed.clone()).await.unwrap().into_inner().snapshot, 0);
    let fetched = c.query(pb::QueryRequest { snapshot: held.snapshot, release: true, ..typed.clone() }).await.unwrap().into_inner();
    assert_eq!((fetched.rows, svc.sweep_leases()), (held.rows, 0));
    let e = c.query(pb::QueryRequest { snapshot: held.snapshot, ..typed.clone() }).await.unwrap_err();
    assert!(e.message().contains("expired"), "{}", e.message());
    let unused = c.query(pb::QueryRequest { hold: true, ..typed.clone() }).await.unwrap().into_inner().snapshot;
    c.release(pb::ReleaseRequest { snapshots: vec![unused, 12345] }).await.unwrap();
    assert_eq!(svc.sweep_leases(), 0);

    let e = c.query(pb::QueryRequest { protocol: 10, ..typed.clone() }).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert!(e.message().contains("protocol 10"), "{}", e.message());
    let newer = pb::Plan { features: vec!["teleport".into()], ..plan };
    let e = c.term_stats(pb::QueryRequest { plan: Some(newer), ..typed }).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert!(e.message().contains("teleport"), "{}", e.message());
}

/// A shard serving `db`, keeping the slot map it is given beside `path`.
fn holding_slots(db: Db, path: &std::path::Path) -> Arc<Service> {
    let schema = db.schema().clone();
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema, agent_config("http://unused"), 16, Duration::from_secs(10));
    svc.keep_slots_in(slots::file_of(path)).unwrap();
    Arc::new(svc)
}

/// Keys live in slots. A coordinator gives each shard its slot map when it
/// connects and routes writes by it. A shard keeps the map beside its
/// database, refuses a key another shard holds and a coordinator with an
/// older map, and refuses a map while rows here are in other shards'
/// slots: rows placed by the hash before slots, or slots a new map moves.
#[tokio::test(flavor = "multi_thread")]
async fn keys_live_in_their_slots() {
    let dir = tempfile::tempdir().unwrap();
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let path = |s: usize| dir.path().join(format!("s{s}.strata"));
    let shards: Vec<Arc<Service>> = (0..2).map(|s| holding_slots(create(dir.path(), &format!("s{s}.strata"), |i| shard_of(&key(i), 2) == s), &path(s))).collect();
    let mut urls = Vec::new();
    for svc in &shards {
        urls.push(grpc(svc.clone()).await);
    }
    // Servers of protocol 9: the map names each shard's server.
    let named = |m: &SlotMap| m.clone().with_replicas(urls.iter().map(|u| vec![u.clone()]).collect()).unwrap();
    let even = named(&SlotMap::even(2, 1));
    let first = cluster(Coordinator::connect(urls.clone()).await.unwrap());
    assert_eq!(first.health().await["slot_epoch"], 1);
    for (s, svc) in shards.iter().enumerate() {
        assert_eq!(svc.slots(), Some((even.clone(), s)));
        assert_eq!(svc.health().await["slots"], json!({"epoch": 1, "shard": s, "of": 2}));
        assert!(slots::file_of(&path(s)).exists());
    }

    // Writes land on the shard holding their key's slot.
    let ids: Vec<String> = (0..40).map(|i| format!("new{i}")).collect();
    let values: Vec<String> = ids.iter().map(|id| format!("('{id}', 'Tent {id}', 'outdoor', 10.0, 0)")).collect();
    first.sql(&Principal::open(), &format!("INSERT INTO products (id, title, category, price, sold) VALUES {}", values.join(", ")), &Params::new(), After::Any, 0).await.unwrap();
    for id in &ids {
        let k = Value::Str(id.clone());
        for (s, svc) in shards.iter().enumerate() {
            let n = svc.query(QueryReq { sql: format!("SELECT COUNT(*) FROM products WHERE id = '{id}'"), ..Default::default() }).await.unwrap().rs.rows[0][0].clone();
            assert_eq!(n, Value::Int((even.shard_of(&k) == s) as i64), "{id} on shard {s}");
        }
    }
    // A shard refuses a key another holds, however it is written.
    let theirs = (0..).map(|i| format!("x{i}")).find(|id| even.shard_of(&Value::Str(id.clone())) == 1).unwrap();
    let e = shards[0].sql(&Principal::open(), &format!("INSERT INTO products (id, title) VALUES ('{theirs}', 'Lamp')"), &Params::new(), After::Any, 0).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("which shard 1 holds under slot map epoch 1"), "{}", e.message);
    let e = shards[0].write_ops(vec![json!({"op": "upsert", "doc": {"id": theirs, "title": "Lamp"}})], None).await.err().unwrap();
    assert_eq!(e.status, 400, "{}", e.message);
    shards[1].write_ops(vec![json!({"op": "upsert", "doc": {"id": theirs, "title": "Lamp"}})], None).await.unwrap();

    // A newer map of the same slots: the coordinator with the older one
    // still reads and writes, since each shard holds the slots it held
    // under it.
    let second_map = named(&SlotMap::from_json(&json!({"epoch": 2, "shards": [[[0, 2048]], [[2048, 4096]]]})).unwrap());
    let second = cluster(Coordinator::connect_with(urls.clone(), CoordOptions { slots: Some(second_map.clone()), ..Default::default() }).await.unwrap());
    assert_eq!(second.health().await["slot_epoch"], 2);
    assert_eq!(first.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await.unwrap().rs.rows[0][0], Value::Int((N + ids.len() + 1) as i64));
    first.write_ops(vec![json!({"op": "delete", "key": "new1"})], None).await.unwrap();
    assert_eq!(second.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await.unwrap().rs.rows[0][0], Value::Int((N + ids.len()) as i64));
    // A coordinator connecting takes up the newest map the shards hold,
    // given the servers of one shard only, too.
    let third = cluster(Coordinator::connect(urls.clone()).await.unwrap());
    assert_eq!(third.health().await["slot_epoch"], 2);
    let fourth = cluster(Coordinator::connect(vec![urls[1].clone()]).await.unwrap());
    assert_eq!(fourth.health().await["slot_epoch"], 2);
    assert_eq!(fourth.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await.unwrap().rs.rows[0][0], Value::Int((N + ids.len()) as i64));

    // The map outlives the server: a shard opened again reads it back.
    let mut again = Service::new(Backend::Local(Arc::new(create(dir.path(), "again.strata", |_| false))), demo::schema(N), agent_config("http://unused"), 16, Duration::from_secs(10));
    again.keep_slots_in(slots::file_of(&path(0))).unwrap();
    assert_eq!(again.slots(), Some((second_map.clone(), 0)));

    // A map that moves slots between shards is refused while rows are in
    // them, as nothing moves rows yet.
    let moved = SlotMap::from_json(&json!({"epoch": 3, "shards": [[[0, 1024], [2048, 4096]], [[1024, 2048]]]})).unwrap();
    let e = Coordinator::connect_with(urls.clone(), CoordOptions { slots: Some(moved), ..Default::default() }).await.err().unwrap();
    assert!(e.status == 409 && e.message.contains("rows here have keys in other shards' slots under slot map epoch 3"), "{}", e.message);
    assert_eq!(shards[0].slot_epoch(), 2);

    // Rows placed by the hash before slots are refused too, by count.
    let old_shard = |s: usize| move |i: usize| slots::hash_shard_of(&Value::Str(format!("p{i:07}")), 2) == s;
    let hashed = holding_slots(create(dir.path(), "h0.strata", old_shard(0)), &dir.path().join("h0.strata"));
    let misplaced = (0..N).filter(|&i| old_shard(0)(i) && shard_of(&key(i), 2) != 0).count();
    let e = Coordinator::connect(vec![grpc(hashed.clone()).await, grpc(holding_slots(create(dir.path(), "h1.strata", old_shard(1)), &dir.path().join("h1.strata"))).await]).await.err().unwrap();
    assert!(e.message.contains(&format!("{misplaced} rows here have keys in other shards' slots")), "{}", e.message);
    assert_eq!(hashed.slots(), None);
}

/// The agent's tool loop, against a stand-in for the Messages API that
/// asks for one query and then answers.
#[tokio::test(flavor = "multi_thread")]
async fn agent_runs_queries_and_answers() {
    use axum::{extract::State, routing::post as route_post, Json as AxJson, Router};
    #[derive(Clone)]
    struct Mock {
        calls: Arc<AtomicUsize>,
        seen: Arc<std::sync::Mutex<Vec<Json>>>,
    }
    async fn messages(State(m): State<Mock>, headers: axum::http::HeaderMap, AxJson(body): AxJson<Json>) -> AxJson<Json> {
        assert_eq!(headers["x-api-key"], "test-key");
        assert_eq!(headers["anthropic-version"], "2026-10-21");
        m.seen.lock().unwrap().push(body.clone());
        let n = m.calls.fetch_add(1, Ordering::SeqCst);
        let usage = json!({"input_tokens": 1000, "output_tokens": 50});
        AxJson(match n {
            0 => json!({"content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "SELECT brand, COUNT(*) FROM products WHERE MATCH(body, 'tent') GROUP BY brand ORDER BY 2 DESC LIMIT 3"}},
                {"type": "tool_use", "id": "t2", "name": "run_sql", "input": {"sql": "SELECT nope FROM products"}}
            ], "stop_reason": "tool_use", "usage": usage}),
            _ => json!({"content": [{"type": "text", "text": "Northpeak sells the most tents.\n\n```sql\nSELECT brand, COUNT(*) FROM products GROUP BY brand\n```"}], "stop_reason": "end_turn", "usage": usage}),
        })
    }
    let mock = Mock { calls: Arc::new(AtomicUsize::new(0)), seen: Arc::new(std::sync::Mutex::new(Vec::new())) };
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = l.local_addr().unwrap();
    let app = Router::new().route("/v1/messages", route_post(messages)).with_state(mock.clone());
    tokio::spawn(async move { axum::serve(l, app).await });

    let dir = tempfile::tempdir().unwrap();
    let base = http(service(create(dir.path(), "a.strata", |i| i < 1000), &format!("http://{addr}")), None).await;
    let c = reqwest::Client::new();
    let (status, r) = post(&c, &format!("{base}/v1/agent"), json!({"question": "Which brand sells the most tents?"})).await;
    assert_eq!(status, 200, "{r}");
    assert!(r["answer"].as_str().unwrap().starts_with("Northpeak"));
    assert_eq!(r["model"], "claude-haiku-5-5");
    assert_eq!(r["usage"]["input_tokens"], 2000);
    let steps = r["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["ok"], true);
    assert_eq!(steps[1]["ok"], false);

    {
        let seen = mock.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(seen[0]["system"].as_str().unwrap().contains("Table `products`, 1000 rows"));
        assert_eq!(seen[0]["tools"][0]["name"], "run_sql");
        // The second request carries the first answer unchanged (thinking
        // included) and both results in one user message.
        let msgs = seen[1]["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["content"][0]["type"], "thinking");
        let results = msgs[2]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["tool_use_id"], "t1");
        assert!(results[0]["content"].as_str().unwrap().contains("\"row_count\":3"));
        assert_eq!(results[1]["is_error"], true);
    }

    // Writes are refused; no key means a clear error.
    let svc = service(create(dir.path(), "b.strata", |i| i < 10), "http://unused");
    let mut cfg = svc.agent.clone();
    cfg.api_key = None;
    let svc = Arc::new(Service::new(Backend::Local(match &svc.backend {
        Backend::Local(db) => db.clone(),
        _ => unreachable!(),
    }), svc.schema.clone(), cfg, 4, Duration::from_secs(5)));
    let base = http(svc, None).await;
    let (status, r) = post(&c, &format!("{base}/v1/agent"), json!({"question": "hi"})).await;
    assert_eq!(status, 400);
    assert!(r["error"].as_str().unwrap().contains("API key"));
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_and_binary_vector_fields_over_http() {
    use strata::{Field, FieldType, Schema, VectorCodes, VectorIndexKind};
    let schema = Schema::new(
        "notes",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("splade", FieldType::Sparse),
            Field::new("emb", FieldType::Vector { dim: 8, index: VectorIndexKind::Flat }).codes(VectorCodes::Binary),
        ],
        Some("id"),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let opts = DbOptions { publish_snapshots: true, ..DbOptions::default() };
    let db = Db::create(&dir.path().join("s.strata"), schema, opts).unwrap();
    let base = http(service(db, "http://unused"), None).await;
    let c = reqwest::Client::new();
    let unit = |i: usize| (0..8).map(|j| if j == i % 8 { 1.0 } else { 0.1 }).collect::<Vec<f32>>();
    let docs: Vec<Json> = (0..40)
        .map(|i| json!({"id": format!("n{i:02}"), "splade": {"common": 0.5, format!("tok{}", i % 5): 1.0 + i as f64 / 100.0}, "emb": unit(i)}))
        .collect();
    let ops: Vec<Json> = docs.into_iter().map(|d| json!({"op": "insert", "doc": d})).collect();
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({ "ops": ops })).await;
    assert_eq!((status, r["written"].clone()), (200, json!(40)), "{r}");

    let (status, r) = post(
        &c,
        &format!("{base}/v1/query"),
        json!({"sql": "SELECT id, score() FROM notes ORDER BY similarity(splade, :w) DESC LIMIT 3", "params": {"w": {"tok3": 2.0, "common": 1.0}}}),
    )
    .await;
    assert_eq!(status, 200, "{r}");
    // tok3 is on n03, n08, ..., n38; the larger weight wins: n38 first.
    assert_eq!(r["keys"], json!(["n38", "n33", "n28"]));
    assert!((r["scores"][0].as_f64().unwrap() - (2.0 * 1.38 + 0.5)).abs() < 2e-3, "{r}");

    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id FROM notes ORDER BY similarity(emb, :v) DESC LIMIT 5", "params": {"v": unit(2)}})).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["keys"].as_array().unwrap().len(), 5);
    assert!(r["keys"].as_array().unwrap().iter().all(|k| k.as_str().unwrap()[1..].parse::<usize>().unwrap() % 8 == 2), "{r}");

    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "insert", "doc": {"id": "bad", "splade": {"x": -1}}}]})).await;
    assert_eq!(status, 400, "{r}");
    let metrics = c.get(format!("{base}/metrics")).send().await.unwrap().text().await.unwrap();
    assert!(metrics.contains(r#"kind="sparse",outcome="ok"} 1"#), "{metrics}");
}

/// A multi-vector field over HTTP: rows written as arrays of vectors, ranked
/// by `maxsim` with an array of query vectors, and read back as arrays.
#[tokio::test(flavor = "multi_thread")]
async fn multi_vector_fields_over_http() {
    use strata::{Field, FieldType, Schema, VectorIndexKind};
    let patches = Field::new("patches", FieldType::Vector { dim: 3, index: VectorIndexKind::Flat }).multi().keep_f32();
    let schema = Schema::new("pages", vec![Field::new("id", FieldType::Keyword), patches], Some("id")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("m.strata"), schema, DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let base = http(service(db, "http://unused"), None).await;
    let c = reqwest::Client::new();
    let docs = [
        json!({"id": "a", "patches": [[1, 0, 0], [0, 1, 0]]}),
        json!({"id": "b", "patches": [[0.0, 0.0, 1.0]]}),
        json!({"id": "c", "patches": [0.5, 0.5, 0.0]}),
        json!({"id": "d"}),
    ];
    let ops: Vec<Json> = docs.into_iter().map(|d| json!({"op": "insert", "doc": d})).collect();
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({ "ops": ops })).await;
    assert_eq!((status, r["written"].clone()), (200, json!(4)), "{r}");
    let sql = "SELECT id, patches FROM pages ORDER BY maxsim(patches, :qs) DESC LIMIT 5";
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "params": {"qs": [[1, 0, 0], [0, 1, 0]]}})).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["keys"], json!(["a", "c", "b"]), "{r}");
    assert_eq!(r["scores"], json!([2.0, 1.0, 0.0]), "{r}");
    assert_eq!(r["rows"][0][1], json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "{r}");
    assert_eq!(r["rows"][1][1], json!([[0.5, 0.5, 0.0]]), "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "insert", "doc": {"id": "bad", "patches": [[1, 0]]}}]})).await;
    assert_eq!(status, 400, "{r}");
}

/// `{"vector_of": key}` reads a row's vector whatever the fields are named:
/// the SQL it ran did not quote them, so a space broke it.
#[tokio::test(flavor = "multi_thread")]
async fn vector_of_any_field_name() {
    use strata::{Field, FieldType, Schema, VectorIndexKind};
    let schema = Schema::new("notes", vec![Field::new("doc id", FieldType::Keyword), Field::new("my vec", FieldType::Vector { dim: 2, index: VectorIndexKind::Flat })], Some("doc id")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("v.strata"), schema, DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let base = http(service(db, "http://unused"), None).await;
    let c = reqwest::Client::new();
    let ops: Vec<Json> = (0..4).map(|i| json!({"op": "insert", "doc": {"doc id": format!("n{i}"), "my vec": [1.0, i as f64]}})).collect();
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({ "ops": ops })).await;
    assert_eq!(status, 200, "{r}");
    let sql = "SELECT \"doc id\" FROM notes ORDER BY similarity(\"my vec\", :q) DESC LIMIT 1";
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "params": {"q": {"vector_of": "n3", "field": "my vec"}}})).await;
    assert_eq!((status, r["keys"].clone()), (200, json!(["n3"])), "{r}");
}

/// A field declared as the embedding of others: the server embeds the rows
/// writes send without it and the text of `{"embed": ...}` parameters with
/// the same model, and says so in /v1/schema.
#[tokio::test(flavor = "multi_thread")]
async fn embedded_fields_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("e.strata"), demo::schema(500), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let schema = db.schema().clone();
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema, agent_config("http://unused"), 16, Duration::from_secs(10));
    svc.open_models(dir.path());
    let base = http(Arc::new(svc), None).await;
    let c = reqwest::Client::new();
    let r: Json = c.get(format!("{base}/v1/schema")).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["embeddings"], json!([{"field": "emb", "from": ["title", "description"], "model": "hash:64", "loaded": true, "error": null, "usage": null}]));

    let docs: Vec<Json> = (0..500).map(|i| js::document_json(&demo::schema(500), &demo::product(i, 7))).collect();
    let mut ops: Vec<Json> = docs.iter().map(|d| json!({"op": "insert", "doc": d})).collect();
    ops.push(json!({"op": "insert", "doc": {"id": "zz1", "title": "Zebra striped umbrella", "description": "A zebra umbrella for rainy days."}}));
    let (status, w) = post(&c, &format!("{base}/v1/write"), json!({"ops": ops})).await;
    assert_eq!((status, w["written"].as_u64()), (200, Some(501)), "{w}");

    let find = |text: &str| json!({"sql": "SELECT id FROM products ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": {"embed": text, "field": "emb"}}});
    let (status, r) = post(&c, &format!("{base}/v1/query"), find("zebra umbrella")).await;
    assert_eq!((status, &r["rows"][0][0]), (200, &json!("zz1")), "{r}");
    // A rewrite of a source re-embeds the row.
    let (status, _) = post(&c, &format!("{base}/v1/sql"), json!({"sql": "UPDATE products SET title = 'Giraffe desk lamp', description = 'A tall giraffe lamp.' WHERE id = 'zz1'"})).await;
    assert_eq!(status, 200);
    let (_, r) = post(&c, &format!("{base}/v1/query"), find("giraffe lamp")).await;
    assert_eq!(r["rows"][0][0], json!("zz1"), "{r}");
    // Only an embedded field with a model embeds text.
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id FROM products ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": {"embed": "x", "field": "title"}}})).await;
    assert_eq!(status, 400, "{r}");

    // /v1/embed gives the vector an {"embed": ...} parameter stands for, for
    // a plan or a client that sends vectors: the same rows come back.
    let (status, e) = post(&c, &format!("{base}/v1/embed"), json!({"text": "giraffe lamp"})).await;
    assert_eq!((status, &e["field"]), (200, &json!("emb")), "{e}");
    assert_eq!(e["vector"].as_array().map(|v| v.len()), Some(64), "{e}");
    let (_, named) = post(&c, &format!("{base}/v1/embed"), json!({"text": "giraffe lamp", "field": "EMB"})).await;
    assert_eq!(named, e);
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id FROM products ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": e["vector"]}})).await;
    assert_eq!(r["rows"][0][0], json!("zz1"), "{r}");
    for bad in [json!({"text": "x", "field": "title"}), json!({"text": "x", "field": 3}), json!({"field": "emb"})] {
        let (status, r) = post(&c, &format!("{base}/v1/embed"), bad.clone()).await;
        assert_eq!(status, 400, "{bad}: {r}");
    }
}

const ITEMS: &str = r#"{"table": "items", "key": "id", "fields": [
    {"name": "id", "type": "keyword"},
    {"name": "title", "type": "text"},
    {"name": "tags", "type": "keyword", "multi": true},
    {"name": "price", "type": "float"},
    {"name": "emb", "type": "vector", "embed": {"from": "title", "model": "hash:16"}}
]}"#;

#[tokio::test(flavor = "multi_thread")]
async fn imports_files_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let schema = strata::Schema::from_json(ITEMS).unwrap();
    let db = Db::create(&dir.path().join("i.strata"), schema.clone(), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema, agent_config("http://unused"), 16, Duration::from_secs(10));
    svc.open_models(dir.path());
    let base = http(Arc::new(svc), None).await;
    let c = reqwest::Client::new();

    let jsonl = (0..2500).map(|i| json!({"id": format!("i{i}"), "title": format!("item number {i}"), "tags": ["a", "b"], "price": i, "colour": "red"}).to_string()).collect::<Vec<_>>().join("\n");
    let ndjson = "application/x-ndjson";
    let r = c.post(format!("{base}/v1/import?format=jsonl")).header("content-type", ndjson).body(jsonl).send().await.unwrap();
    let (status, r): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
    assert_eq!((status, &r["written"], &r["skipped_fields"]), (200, &json!(2500), &json!(["colour"])), "{r}");

    // CSV by its content type, replacing a row and adding one; the model
    // embeds the new titles.
    let csv = "id,title,tags,price\ni7,\"zebra striped, umbrella\",x|y,7.5\nnew,giraffe desk lamp,,1\n";
    let r = c.post(format!("{base}/v1/import")).header("content-type", "text/csv").body(csv).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT COUNT(*) FROM items"})).await;
    assert_eq!(r["rows"][0][0], json!(2501), "{r}");
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT title, tags, price FROM items WHERE id = 'i7'"})).await;
    assert_eq!(r["rows"][0], json!(["zebra striped, umbrella", ["x", "y"], 7.5]), "{r}");
    let find = |text: &str| json!({"sql": "SELECT id FROM items ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": {"embed": text, "field": "emb"}}});
    let (_, r) = post(&c, &format!("{base}/v1/query"), find("giraffe desk lamp")).await;
    assert_eq!(r["rows"][0][0], json!("new"), "{r}");

    // A bad row names its line; no format is an error.
    let r = c.post(format!("{base}/v1/import?format=jsonl")).header("content-type", ndjson).body("{\"id\": \"ok\"}\n{\"id\": \"bad\", \"price\": \"cheap\"}\n").send().await.unwrap();
    let (status, r): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
    assert_eq!(status, 400);
    assert!(r["error"].as_str().unwrap().contains("line 2"), "{r}");
    let r = c.post(format!("{base}/v1/import")).header("content-type", "application/octet-stream").body("id\nx\n").send().await.unwrap();
    assert_eq!(r.status().as_u16(), 400);
    // No content type at all is what a page's Blob upload sends.
    let r = c.post(format!("{base}/v1/import?format=csv")).body("id\nx\n").send().await.unwrap();
    assert_eq!(r.status().as_u16(), 415);

    // The schema in its short form, and a schema checked.
    let r: Json = c.get(format!("{base}/v1/schema")).send().await.unwrap().json().await.unwrap();
    let mut short: Json = serde_json::from_str(ITEMS).unwrap();
    short["fields"][4]["embed"]["from"] = json!(["title"]);
    assert_eq!(r["short"], short, "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/schema/check"), json!({"table": "t", "key": "id", "fields": [{"name": "id", "type": "keyword"}, {"name": "v", "type": "vector"}]})).await;
    assert_eq!(status, 400);
    assert!(r["error"].as_str().unwrap().contains("dim"), "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/schema/check"), json!({"table": "t", "key": "id", "fields": [{"name": "id", "type": "keyword"}]})).await;
    assert_eq!((status, &r["short"]["key"]), (200, &json!("id")), "{r}");
    assert_eq!(r["schema"]["primary_key"], json!(0), "{r}");
    // And the CREATE TABLE statement that makes it, which runs.
    assert_eq!(r["sql"], "CREATE TABLE t (\n  id KEYWORD PRIMARY KEY\n);", "{r}");
    let r: Json = c.get(format!("{base}/v1/tables")).send().await.unwrap().json().await.unwrap();
    let ddl = r["tables"][0]["sql"].as_str().unwrap().replacen("CREATE TABLE items", "CREATE TABLE items_copy", 1);
    let (status, r) = post(&c, &format!("{base}/v1/sql"), json!({ "sql": ddl })).await;
    assert_eq!(status, 200, "{ddl}: {r}");
    let r: Json = c.get(format!("{base}/v1/schema?table=items_copy")).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["short"]["fields"], short["fields"], "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_files_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let schema = strata::Schema::from_json(ITEMS).unwrap();
    let db = Db::create(&dir.path().join("i.strata"), schema.clone(), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema.clone(), agent_config("http://unused"), 16, Duration::from_secs(10));
    svc.open_models(dir.path());
    let base = http(Arc::new(svc), None).await;
    let c = reqwest::Client::new();

    // A SQLite file with rows for the server's table and a table it lacks,
    // made by an export without the Strata schemas, so that the server
    // makes the new table's from SQLite's declarations.
    let src = Db::create(&dir.path().join("src.strata"), schema, DbOptions::default()).unwrap();
    src.sql("INSERT INTO items (id, title, tags, price) VALUES ('a', 'red umbrella', ['rain'], 12.5), ('b', 'blue kettle', [], 30), ('c', 'green lamp', ['home', 'light'], 18)", &Params::new()).unwrap();
    src.sql("CREATE TABLE notes (id INT PRIMARY KEY, item KEYWORD REFERENCES items, body TEXT, at TIMESTAMP)", &Params::new()).unwrap();
    src.sql("INSERT INTO notes (id, item, body, at) VALUES (1, 'a', 'keeps the rain off on the walk to the station', '2026-10-01T12:00:00Z'), (2, 'c', 'a warm light for reading in the evening', '2026-10-02T08:30:00Z')", &Params::new()).unwrap();
    let file = dir.path().join("src.sqlite");
    strata::sqlite::export(&src, &file, &strata::sqlite::ExportOptions { meta: false, ..Default::default() }).unwrap();
    let bytes = std::fs::read(&file).unwrap();
    let tables = |c: reqwest::Client| {
        let base = base.clone();
        async move { c.get(format!("{base}/v1/tables")).send().await.unwrap().json::<Json>().await.unwrap()["tables"].as_array().unwrap().len() }
    };

    // The plan writes nothing.
    let r = c.post(format!("{base}/v1/import?format=sqlite&plan=true")).header("content-type", "application/vnd.sqlite3").body(bytes.clone()).send().await.unwrap();
    let (status, plan): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
    assert_eq!(status, 200, "{plan}");
    let summary: Vec<(String, bool)> = plan["tables"].as_array().unwrap().iter().map(|t| (t["table"].as_str().unwrap().to_string(), t["exists"].as_bool().unwrap())).collect();
    assert_eq!(summary, [("items".to_string(), true), ("notes".to_string(), false)], "{plan}");
    let notes = &plan["tables"][1]["fields"];
    assert_eq!(notes[1], json!({"name": "item", "type": "keyword", "references": "items"}), "{plan}");
    assert_eq!(notes[2]["type"], json!("text"), "{plan}");
    assert_eq!(notes[3]["type"], json!("timestamp"), "{plan}");
    assert_eq!(tables(c.clone()).await, 1);

    // Recognised by its first bytes; the new table is created, the rows
    // written through the writer, and the model embeds the titles.
    let r = c.post(format!("{base}/v1/import")).header("content-type", "application/octet-stream").body(bytes.clone()).send().await.unwrap();
    let (status, r): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
    assert_eq!((status, &r["written"]), (200, &json!(5)), "{r}");
    assert_eq!(r["tables"][1], json!({"source": "notes", "table": "notes", "rows": 2, "created": true, "bulk": false, "notes": []}), "{r}");
    assert_eq!(r["skipped"]["notes_fts"], json!("the fts5 index of notes (the columns it indexes are TEXT fields)"), "{r}");
    assert_eq!(tables(c.clone()).await, 2);
    let (_, q) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT n.id, i.title FROM notes n JOIN items i ON n.item = i.id WHERE MATCH(n.body, 'reading') "})).await;
    assert_eq!(q["rows"], json!([[2, "green lamp"]]), "{q}");
    let (_, q) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id FROM items ORDER BY similarity(emb, :q) DESC LIMIT 1", "params": {"q": {"embed": "blue kettle", "field": "emb"}}})).await;
    assert_eq!(q["rows"][0][0], json!("b"), "{q}");
    // Again: the rows are upserted by key, into tables that now exist.
    let r = c.post(format!("{base}/v1/import?format=sqlite&tables=notes")).header("content-type", "application/octet-stream").body(bytes.clone()).send().await.unwrap();
    let (status, r): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
    assert_eq!((status, &r["written"], &r["tables"][0]["created"]), (200, &json!(2), &json!(false)), "{r}");
    let (_, q) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT COUNT(*) FROM notes"})).await;
    assert_eq!(q["rows"][0][0], json!(2), "{q}");

    // What is refused: ?table= (for rows), a file that is not SQLite, a table it lacks.
    for (url, body, why) in [
        ("format=sqlite&table=items", bytes.clone(), "?tables="),
        ("format=sqlite", b"id,title\nx,y\n".to_vec(), "not a SQLite database"),
        ("format=sqlite&tables=nope", bytes.clone(), "no table nope"),
    ] {
        let r = c.post(format!("{base}/v1/import?{url}")).header("content-type", "application/octet-stream").body(body).send().await.unwrap();
        let (status, r): (u16, Json) = (r.status().as_u16(), r.json().await.unwrap());
        assert_eq!(status, 400, "{url}: {r}");
        assert!(r["error"].as_str().unwrap().contains(why), "{url}: {r}");
        // Review BUG-N11: the server's temporary path is not the client's business.
        assert!(!r["error"].as_str().unwrap().contains("import.sqlite"), "{url}: {r}");
    }

    // The export: a SQLite file of every table, which imports back as it was.
    let r = c.get(format!("{base}/v1/export")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(r.headers()["content-type"], "application/vnd.sqlite3");
    assert_eq!(r.headers()["content-disposition"], "attachment; filename=\"items.sqlite\"");
    assert_eq!(r.headers()["x-strata-tables"], "items,notes");
    let out = dir.path().join("out.sqlite");
    std::fs::write(&out, r.bytes().await.unwrap()).unwrap();
    let (back, done) = strata::sqlite::import_new(&out, &dir.path().join("back.strata"), DbOptions::default(), &Default::default()).unwrap();
    assert_eq!(done.tables.iter().map(|t| (t.table.as_str(), t.rows)).collect::<Vec<_>>(), [("items", 3), ("notes", 2)]);
    assert_eq!(back.storage().table("notes").unwrap().to_short_json()["fields"][1]["references"], json!("items"));
    // Some tables, bare; and only SQLite.
    let r = c.get(format!("{base}/v1/export?tables=notes&fts=false&indexes=0&schemas=no")).send().await.unwrap();
    assert_eq!(r.headers()["x-strata-tables"], "notes");
    let bare = dir.path().join("bare.sqlite");
    std::fs::write(&bare, r.bytes().await.unwrap()).unwrap();
    let (names, _) = strata::sqlite::Source::open(&bare).unwrap().tables().unwrap();
    assert_eq!(names, ["notes"]);
    let r = c.get(format!("{base}/v1/export?format=csv")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 400);
    let r = c.get(format!("{base}/v1/export?fts=maybe")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 400);
}

#[test]
fn imports_files_from_the_command_line() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, rows, more) = (dir.path().join("items.json"), dir.path().join("rows.jsonl"), dir.path().join("more.tsv"));
    std::fs::write(&schema, ITEMS).unwrap();
    std::fs::write(&rows, "{\"id\": \"a\", \"title\": \"first\"}\n{\"id\": \"b\", \"title\": \"second\", \"price\": 2}\n").unwrap();
    std::fs::write(&more, "id\ttitle\tprice\nb\tsecond again\t3\nc\tthird\t\n").unwrap();
    let db = dir.path().join("items.strata");
    server::import_files(&db, Some(&schema), None, dir.path(), None, false, &[rows.clone(), more], Default::default()).unwrap();
    let opened = Db::open(&db, DbOptions::default()).unwrap();
    let r = opened.query("SELECT id, title, price FROM items", &Params::new()).unwrap();
    let mut got: Vec<String> = r.rows.iter().map(|r| format!("{:?}", r)).collect();
    got.sort();
    assert_eq!(got.len(), 3, "{got:?}");
    assert!(got[1].contains("second again") && got[1].contains("3.0"), "{got:?}");
    drop(opened);
    // An existing database takes no schema.
    assert!(server::import_files(&db, Some(&schema), None, dir.path(), None, false, &[rows], Default::default()).unwrap_err().contains("exists"));
}

#[test]
fn sqlite_files_from_and_to_the_command_line() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, rows) = (dir.path().join("items.json"), dir.path().join("rows.jsonl"));
    std::fs::write(&schema, ITEMS).unwrap();
    std::fs::write(&rows, "{\"id\": \"a\", \"title\": \"first\"}\n{\"id\": \"b\", \"title\": \"second\", \"price\": 2}\n").unwrap();
    let db = dir.path().join("items.strata");
    server::import_files(&db, Some(&schema), None, dir.path(), None, false, std::slice::from_ref(&rows), Default::default()).unwrap();
    let out = dir.path().join("items.sqlite");
    server::export_sqlite(&db, &out, &Default::default()).unwrap();
    assert!(strata::sqlite::is_sqlite_file(&out));
    assert!(server::export_sqlite(&db, &out, &Default::default()).unwrap_err().contains("exists"));
    // --plan writes nothing, and is for SQLite files.
    let back = dir.path().join("back.strata");
    let plan = || server::SqliteImport { plan: true, ..Default::default() };
    server::import_files(&back, None, None, dir.path(), None, false, std::slice::from_ref(&out), plan()).unwrap();
    assert!(!back.exists());
    assert!(server::import_files(&back, None, None, dir.path(), None, false, std::slice::from_ref(&rows), plan()).unwrap_err().contains("--plan"));
    // A new database from the SQLite file: the schema it kept, and the rows.
    server::import_files(&back, None, None, dir.path(), None, false, &[out], Default::default()).unwrap();
    let (a, b) = (Db::open(&db, DbOptions::default()).unwrap(), Db::open(&back, DbOptions::default()).unwrap());
    assert_eq!(a.schema().to_short_json(), b.schema().to_short_json());
    assert_eq!(b.snapshot().n_live_docs(), 2);
    let r = b.query("SELECT price FROM items WHERE id = 'b'", &Params::new()).unwrap();
    assert_eq!(r.rows[0][0], Value::Float(2.0));
}

// OPS-3: a replica that refused a query because too many were running was
// retried on its sibling, which then carried both loads. The refusal (429,
// ResourceExhausted over gRPC) now fails the query instead.
#[tokio::test(flavor = "multi_thread")]
async fn an_overloaded_replica_is_not_relieved_by_its_sibling() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "db.strata", |_| true);
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let over = format!("http://{}", l.local_addr().unwrap());
    let fake = pb::strata_server::StrataServer::new(Overloaded(serde_json::to_string(db.schema()).unwrap()));
    tokio::spawn(tonic::transport::Server::builder().add_service(fake).serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(l)));
    let good = grpc(service(db, "http://unused")).await;
    let coord = cluster(Coordinator::connect(vec![format!("{over}|{good}")]).await.unwrap());
    // Reads take the replicas in turn: one of two asks the overloaded one first.
    let mut statuses = Vec::new();
    for _ in 0..2 {
        let r = coord.query(QueryReq { sql: "SELECT COUNT(*) FROM products".into(), ..Default::default() }).await;
        statuses.push(r.map_or_else(|e| e.status, |_| 200));
    }
    statuses.sort();
    assert_eq!(statuses, vec![200, 429]);
    assert_eq!(coord.health().await["retries"], 0);
}

// SCALE-2/OPS-3: a query whose caller went away (a client that hung up, a
// hedge that lost, a shard timeout) never took itself off the running
// count, so every later query searched its segments on fewer threads until
// a restart; and it gave its slot back while its work went on. The work now
// holds both, and stops at its next check.
#[tokio::test(flavor = "multi_thread")]
async fn dropped_queries_leave_the_running_count() {
    let dir = tempfile::tempdir().unwrap();
    let svc = service(create(dir.path(), "db.strata", |_| true), "http://unused");
    let sql = "SELECT id, score() FROM products ORDER BY bm25(body, 'waterproof hiking boots') DESC LIMIT 100";
    for i in 0..60u64 {
        let s = svc.clone();
        let t = tokio::spawn(async move { s.query(QueryReq { sql: sql.into(), ..Default::default() }).await });
        tokio::time::sleep(Duration::from_micros(50 * (i % 20))).await;
        t.abort();
    }
    let t0 = std::time::Instant::now();
    while svc.running() > 0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "{} queries still count as running", svc.running());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    svc.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap();
    assert_eq!(svc.running(), 0);
}

// SCALE-5: each BM25 query through a coordinator pinned a version on every
// shard for 60 s, released only when a later lease was made, and a client
// could make leases without limit. A statistics round's lease now lasts for
// its query's deadline and a second, expired ones are swept, and at most
// MAX_LEASES are held. A request's deadline is capped too.
#[tokio::test(flavor = "multi_thread")]
async fn snapshot_leases_end_with_their_query() {
    let dir = tempfile::tempdir().unwrap();
    let svc = service(create(dir.path(), "db.strata", |_| true), "http://unused");
    let sql = "SELECT id FROM products ORDER BY bm25(body, 'waterproof hiking boots') DESC LIMIT 5";
    let (_, lease) = svc.term_stats(sql, None, &Params::new(), &After::Any, 200).await.unwrap();
    let q = |snapshot| QueryReq { sql: sql.into(), snapshot, ..Default::default() };
    svc.query(q(lease)).await.unwrap();
    assert_eq!(svc.sweep_leases(), 1);
    tokio::time::sleep(Duration::from_millis(1300)).await;
    assert_eq!(svc.sweep_leases(), 0);
    let e = svc.query(q(lease)).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("expired"), "{}", e.message);

    for _ in 0..MAX_LEASES {
        svc.lease().unwrap();
    }
    assert_eq!(svc.lease().err().unwrap().status, 429);
    assert_eq!(svc.timeout_of(u64::MAX), svc.max_timeout);
}

// OPS-5/OPS-6: a database that could not write (a full disk) still said
// "ok", failed every write with a 500 nobody counted or logged, and went on
// merging every 5 s for nothing. Health now says "degraded" while flushes
// fail (writes still commit to the WAL) and "failing" once no write can
// succeed until a restart; /livez and /readyz, open without the token, then
// answer 503 so that the server is restarted. Failed writes are counted.
#[tokio::test(flavor = "multi_thread")]
async fn health_reports_a_database_that_cannot_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.strata");
    let opts = DbOptions { layout: Layout::Directory, publish_snapshots: true, ..DbOptions::default() };
    let db = Db::create(&path, demo::schema(10), opts).unwrap();
    db.load((0..10).map(|i| demo::product(i, 7))).unwrap();
    let svc = service(db, "http://unused");
    let db = svc.local_db().unwrap().clone();
    let url = http(svc.clone(), Some("tok")).await;
    let c = reqwest::Client::new();
    let get = |route: &str| {
        let r = c.get(format!("{url}{route}")).bearer_auth("tok").send();
        async move {
            let r = r.await.unwrap();
            (r.status().as_u16(), r.text().await.unwrap())
        }
    };
    let open = |route: &str| {
        let r = c.get(format!("{url}{route}")).send();
        async move { r.await.unwrap().status().as_u16() }
    };
    let write = |i: usize| svc.write_ops(vec![json!({"op": "upsert", "doc": js::document_json(&svc.schema, &demo::product(i, 7))})], None);
    let status = |h: &str| serde_json::from_str::<Json>(h).unwrap()["status"].clone();
    assert_eq!((open("/livez").await, open("/readyz").await), (200, 200));
    let (code, h) = get("/v1/health").await;
    assert_eq!((code, status(&h)), (200, json!("ok")), "{h}");

    // Flushes fail (directories where the segment files go): degraded.
    let blockers: Vec<_> = (0..64).map(|id| path.join(format!("seg-{id:010}.strata"))).filter(|p| !p.exists()).collect();
    for b in &blockers {
        std::fs::create_dir(b).unwrap();
    }
    write(20).await.unwrap();
    assert!(db.storage().flush().is_err());
    let (code, h) = get("/v1/health").await;
    assert_eq!((code, status(&h)), (200, json!("degraded")), "{h}");
    assert_eq!(open("/livez").await, 200);
    for b in &blockers {
        std::fs::remove_dir(b).unwrap();
    }

    // A commit that fails partway (here pointing CURRENT at the new
    // manifest) leaves the database refusing writes until it is reopened.
    std::fs::create_dir(path.join("CURRENT.tmp")).unwrap();
    assert!(db.storage().flush().is_err());
    assert!(db.storage().is_poisoned());
    let (code, h) = get("/v1/health").await;
    assert_eq!((code, status(&h)), (503, json!("failing")), "{h}");
    assert_eq!((open("/livez").await, open("/readyz").await), (503, 503));
    assert_eq!(write(21).await.err().unwrap().status, 500);
    let (_, m) = get("/metrics").await;
    assert!(m.contains("strata_write_errors_total{status=\"500\"} 1") && m.contains("strata_failing 1"), "{m}");
}


/// Checkpoints into `--backup-dir` (a write still in the memtable included,
/// a directory checkpoint linked to the one before), names that would leave
/// the directory refused, and the check and vacuum routes.
#[tokio::test]
async fn admin_routes_checkpoint_check_and_vacuum() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "db.strata", |_| true);
    let schema = db.schema().clone();
    let mut svc = Service::new(Backend::Local(Arc::new(db)), schema, agent_config("http://127.0.0.1:9"), 16, Duration::from_secs(10));
    let backups = dir.path().join("backups");
    svc.set_backup_dir(&backups).unwrap();
    let base = http(Arc::new(svc), None).await;
    let c = reqwest::Client::new();
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "delete", "key": "p0000001"}]})).await;
    assert_eq!(status, 200, "{r}");

    let (status, r) = post(&c, &format!("{base}/v1/admin/checkpoint"), json!({"name": "first"})).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["checkpoint"]["live_docs"], json!(N - 1));
    let copy = Db::open(&backups.join("first"), DbOptions::default()).unwrap();
    assert_eq!(copy.snapshot().n_live_docs(), (N - 1) as u64);
    assert!(copy.query("SELECT id FROM products WHERE id = 'p0000001'", &Params::new()).unwrap().rows.is_empty());
    drop(copy);

    let (status, r) = post(&c, &format!("{base}/v1/admin/checkpoint"), json!({"name": "d1", "layout": "dir"})).await;
    assert_eq!(status, 200, "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/admin/checkpoint"), json!({"name": "d2", "layout": "dir", "link_from": "d1"})).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["checkpoint"]["segments_linked"], r["checkpoint"]["segments"], "{r}");

    for body in [json!({"name": "../out"}), json!({"name": ".hidden"}), json!({"name": "a/b"}), json!({"name": "x", "layout": "zip"}), json!({}), json!({"name": "y", "layout": "dir", "link_from": "../d1"})] {
        let (status, r) = post(&c, &format!("{base}/v1/admin/checkpoint"), body.clone()).await;
        assert_eq!(status, 400, "{body}: {r}");
    }
    let (status, _) = post(&c, &format!("{base}/v1/admin/checkpoint"), json!({"name": "first"})).await;
    assert_eq!(status, 500, "an existing name is not overwritten");
    // (`first-wal` is the log opening `first` above created.)
    let mut names: Vec<String> = std::fs::read_dir(&backups).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n != "first-wal").collect();
    names.sort();
    assert_eq!(names, ["d1", "d2", "first"]);
    assert!(!dir.path().join("out").exists());

    let (status, r) = post(&c, &format!("{base}/v1/admin/check"), json!({})).await;
    assert_eq!((status, &r), (200, &json!({"ok": true})));
    let (status, r) = post(&c, &format!("{base}/v1/admin/vacuum"), json!({})).await;
    assert_eq!(status, 200, "{r}");
    assert!(r["bytes_freed"].is_u64(), "{r}");

    // Without --backup-dir there is nowhere to write.
    let db = create(dir.path(), "other.strata", |i| i < 10);
    let base = http(service(db, "http://127.0.0.1:9"), None).await;
    let (status, r) = post(&c, &format!("{base}/v1/admin/checkpoint"), json!({"name": "x"})).await;
    assert_eq!(status, 400, "{r}");
}

/// BOOL and TIMESTAMP fields answer as JSON booleans and ISO 8601 text,
/// take them as parameters and in writes, and come back the same from a
/// coordinator over shards reached by gRPC.
#[tokio::test(flavor = "multi_thread")]
async fn bool_and_timestamp_fields_over_http_and_grpc() {
    use strata::schema::{Field, FieldType, Schema};
    let schema = Schema::new("tasks", vec![Field::new("id", FieldType::Keyword), Field::boolean("done"), Field::timestamp("at"), Field::new("n", FieldType::Int)], Some("id")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let docs: Vec<Vec<Value>> = (0..40i64)
        .map(|i| {
            let done = if i % 4 == 3 { Value::Null } else { Value::Bool(i % 2 == 0) };
            vec![Value::Str(format!("t{i:02}")), done, Value::Str(format!("2026-10-{:02}T{:02}:00:00Z", 1 + i % 28, i % 24)), Value::Int(i)]
        })
        .collect();
    let open = |name: &str, keep: &dyn Fn(&Value) -> bool| {
        let opts = DbOptions { publish_snapshots: true, ..DbOptions::default() };
        let db = Db::create(&dir.path().join(name), schema.clone(), opts).unwrap();
        db.load(docs.iter().filter(|d| keep(&d[0])).cloned()).unwrap();
        db
    };
    let svc = |db: Db| Arc::new(Service::new(Backend::Local(Arc::new(db)), schema.clone(), agent_config("http://unused"), 16, Duration::from_secs(10)));

    let base = http(svc(open("whole.strata", &|_| true)), None).await;
    let c = reqwest::Client::new();
    let (query_url, write_url) = (format!("{base}/v1/query"), format!("{base}/v1/write"));
    let query = |sql: &str, params: Json| post(&c, &query_url, json!({"sql": sql, "params": params}));
    let (status, r) = query("SELECT id, done, at FROM tasks WHERE done = :d AND at >= :since ORDER BY n LIMIT 2", json!({"d": true, "since": "2026-10-20"})).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["rows"], json!([["t20", true, "2026-10-21T20:00:00Z"], ["t22", true, "2026-10-23T22:00:00Z"]]));
    let (_, r) = query("SELECT done, COUNT(*), MIN(at) FROM tasks WHERE done IS NOT NULL GROUP BY done", json!({})).await;
    let mut groups = r["rows"].as_array().unwrap().clone();
    groups.sort_by_key(|g| g[0].as_bool());
    assert_eq!(groups, [json!([false, 10, "2026-10-02T01:00:00Z"]), json!([true, 20, "2026-10-01T00:00:00Z"])]);

    // Writes take booleans and ISO 8601 text, and refuse what is neither.
    let write = |ops: Json| post(&c, &write_url, json!({"ops": ops}));
    let (status, r) = write(json!([
        {"op": "insert", "doc": {"id": "x1", "done": true, "at": "2026-11-01T02:00:00+02:00", "n": 100}},
        {"op": "set", "key": "x1", "field": "done", "value": false},
        {"op": "set", "key": "t00", "field": "at", "value": "2026-12-24"},
    ]))
    .await;
    assert_eq!((status, &r["written"]), (200, &json!(3)), "{r}");
    let (_, r) = query("SELECT id, done, at FROM tasks WHERE id IN ('x1', 't00') ORDER BY n", json!({})).await;
    assert_eq!(r["rows"], json!([["t00", true, "2026-12-24T00:00:00Z"], ["x1", false, "2026-11-01T00:00:00Z"]]));
    for bad in [json!({"op": "insert", "doc": {"id": "x2", "at": 1790812800}}), json!({"op": "set", "key": "x1", "field": "done", "value": 3}), json!({"op": "add", "key": "x1", "field": "at", "value": 1})] {
        let (status, r) = write(json!([bad])).await;
        assert_eq!(status, 400, "{bad}: {r}");
    }
    let (status, r) = post(&c, &format!("{base}/v1/sql"), json!({"sql": "UPDATE tasks SET done = TRUE, at = '2026-11-02' WHERE id = 'x1'"})).await;
    assert_eq!((status, &r["written"]), (200, &json!(1)), "{r}");
    let (_, r) = query("SELECT done, at FROM tasks WHERE id = 'x1'", json!({})).await;
    assert_eq!(r["rows"], json!([[true, "2026-11-02T00:00:00Z"]]));

    // A coordinator over two shards answers like the whole database, and
    // its shards carry the stored integers between them.
    let whole = open("whole2.strata", &|_| true);
    let mut urls = Vec::new();
    for s in 0..2 {
        urls.push(grpc(svc(open(&format!("s{s}.strata"), &|k| shard_of(k, 2) == s))).await);
    }
    let coord = Coordinator::connect(urls).await.unwrap();
    let coord = Arc::new(Service::new(Backend::Cluster(Arc::new(coord)), schema.clone(), agent_config("http://unused"), 16, Duration::from_secs(10)));
    for sql in [
        "SELECT done, COUNT(*), MIN(at), MAX(at) FROM tasks GROUP BY done",
        "SELECT id, done, at FROM tasks WHERE at < '2026-10-10' ORDER BY at DESC LIMIT 5",
        "SELECT id, done, at FROM tasks WHERE done IS NOT TRUE AND at >= '2026-10-03T12:00:00+02:00' ORDER BY n LIMIT 6",
    ] {
        let want = js::result(&whole.query(sql, &Params::new()).unwrap(), 0.0);
        let got = js::result(&coord.query(QueryReq { sql: sql.into(), ..Default::default() }).await.unwrap().rs, 0.0);
        let sorted = |j: &Json| {
            let mut rows = j["rows"].as_array().unwrap().clone();
            if sql.contains("GROUP BY") {
                rows.sort_by_key(|r| r.to_string());
            }
            rows
        };
        assert_eq!(sorted(&want), sorted(&got), "{sql}");
        let values: Vec<Json> = sorted(&got).iter().flat_map(|r| r.as_array().unwrap().clone()).collect();
        assert!(values.iter().any(Json::is_boolean) && values.iter().any(|v| v.as_str().is_some_and(|s| s.ends_with('Z'))), "{sql}: {got}");
    }
    // A parameter of each type reaches the shards.
    let params = coord.params(Some(&json!({"d": false, "since": "2026-10-25"}))).await.unwrap();
    let got = coord.query(QueryReq { sql: "SELECT id FROM tasks WHERE done = :d AND at >= :since ORDER BY n".into(), params, ..Default::default() }).await.unwrap();
    let want = whole.query("SELECT id FROM tasks WHERE done = FALSE AND at >= '2026-10-25' ORDER BY n", &Params::new()).unwrap();
    assert_eq!(got.rs.rows, want.rows);
    assert!(!want.rows.is_empty());
    // An INSERT through the coordinator reaches its key's shard as written.
    coord.sql(&Principal::open(), "INSERT INTO tasks (id, done, at, n) VALUES ('y1', TRUE, '2026-10-31T23:59:59.5-03:00', 200)", &Params::new(), After::Any, 0).await.unwrap();
    let got = coord.query(QueryReq { sql: "SELECT done, at FROM tasks WHERE id = 'y1'".into(), ..Default::default() }).await.unwrap();
    assert_eq!(js::result(&got.rs, 0.0)["rows"], json!([[true, "2026-11-01T02:59:59.500Z"]]));
}

/// NOT NULL and DEFAULT over HTTP writes and through a coordinator: a row
/// that leaves a field out takes its default, and a NOT NULL field left
/// NULL is refused with 400.
#[tokio::test(flavor = "multi_thread")]
async fn defaults_and_not_null_over_http_and_grpc() {
    use strata::schema::Schema;
    let schema = Schema::from_json(
        r#"{"table": "events", "key": "id", "fields": [
            {"name": "id", "type": "keyword"},
            {"name": "kind", "type": "keyword", "not_null": true, "default": "click"},
            {"name": "n", "type": "int", "default": 1},
            {"name": "at", "type": "timestamp", "not_null": true, "default": "now"}
        ]}"#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let open = |name: &str| Db::create(&dir.path().join(name), schema.clone(), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let svc = |db: Db| Arc::new(Service::new(Backend::Local(Arc::new(db)), schema.clone(), agent_config("http://unused"), 16, Duration::from_secs(10)));

    let base = http(svc(open("one.strata")), None).await;
    let c = reqwest::Client::new();
    let (status, r) = c.get(format!("{base}/v1/schema")).send().await.unwrap().json::<Json>().await.map(|j| (200, j)).unwrap();
    assert_eq!((status, &r["short"]["fields"][1]), (200, &json!({"name": "kind", "type": "keyword", "not_null": true, "default": "click"})));
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "insert", "doc": {"id": "a"}}, {"op": "insert", "doc": {"id": "b", "kind": "view", "n": null}}]})).await;
    assert_eq!((status, &r["written"]), (200, &json!(2)), "{r}");
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT id, kind, n FROM events WHERE at <= :t", "params": {"t": "2100-01-01"}})).await;
    let mut got = r["rows"].as_array().cloned().unwrap_or_default();
    got.sort_by_key(|r| r[0].to_string());
    assert_eq!(got, [json!(["a", "click", 1]), json!(["b", "view", null])], "{r}");
    let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT COUNT(*), MIN(at), MAX(at) FROM events"})).await;
    assert_eq!(r["rows"][0][0], json!(2), "{r}");
    assert_eq!(r["rows"][0][1], r["rows"][0][2], "one time for the request: {r}");
    for op in [json!({"op": "insert", "doc": {"id": "x", "kind": null}}), json!({"op": "set", "key": "a", "field": "at", "value": null})] {
        let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [op]})).await;
        assert_eq!(status, 400, "{op}: {r}");
        assert!(r["error"].as_str().unwrap_or("").contains("is NOT NULL"), "{r}");
    }

    // Through a coordinator, an INSERT's defaults are filled in once, so
    // its shards store the same time.
    let mut urls = Vec::new();
    for s in 0..2 {
        urls.push(grpc(svc(open(&format!("s{s}.strata")))).await);
    }
    let coord = Coordinator::connect(urls).await.unwrap();
    let coord = Arc::new(Service::new(Backend::Cluster(Arc::new(coord)), schema.clone(), agent_config("http://unused"), 16, Duration::from_secs(10)));
    let values: Vec<String> = (0..20).map(|i| format!("('k{i}')")).collect();
    coord.sql(&Principal::open(), &format!("INSERT INTO events (id) VALUES {}", values.join(", ")), &Params::new(), After::Any, 0).await.unwrap();
    let got = coord.query(QueryReq { sql: "SELECT kind, n, COUNT(*), MIN(at), MAX(at) FROM events GROUP BY kind, n".into(), ..Default::default() }).await.unwrap();
    let rows = js::result(&got.rs, 0.0)["rows"].clone();
    assert_eq!(rows.as_array().unwrap().len(), 1, "{rows}");
    assert_eq!((&rows[0][0], &rows[0][1], &rows[0][2]), (&json!("click"), &json!(1), &json!(20)));
    assert_eq!(rows[0][3], rows[0][4], "one time for the statement: {rows}");
    assert!(coord.sql(&Principal::open(), "INSERT INTO events (id, kind) VALUES ('y', NULL)", &Params::new(), After::Any, 0).await.is_err());
}

/// A TEXT field's analyzer over HTTP: the schema names it, and searches
/// of the field analyse their words with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_simple_analyzer_over_http() {
    use strata::schema::Schema;
    let schema = Schema::from_json(
        r#"{"table": "notes", "key": "id", "fields": [
            {"name": "id", "type": "keyword"},
            {"name": "title", "type": "text", "analyzer": "simple"},
            {"name": "body", "type": "text"}
        ]}"#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("n.strata"), schema.clone(), DbOptions { publish_snapshots: true, ..DbOptions::default() }).unwrap();
    let svc = Arc::new(Service::new(Backend::Local(Arc::new(db)), schema, agent_config("http://unused"), 16, Duration::from_secs(10)));
    let base = http(svc, None).await;
    let c = reqwest::Client::new();
    let r = c.get(format!("{base}/v1/schema")).send().await.unwrap().json::<Json>().await.unwrap();
    assert_eq!(r["short"]["fields"][1], json!({"name": "title", "type": "text", "analyzer": "simple"}));
    assert_eq!(r["schema"]["fields"][1]["analyzer"], json!({"name": "simple", "version": 1}));
    let (status, r) = post(&c, &format!("{base}/v1/schema/check"), json!({"table": "t", "fields": [{"name": "t", "type": "text", "analyzer": "klingon"}]})).await;
    assert_eq!(status, 400, "{r}");
    assert!(r["error"].as_str().unwrap_or("").contains("analyzer klingon"), "{r}");

    let doc = |id: &str, t: &str| json!({"op": "insert", "doc": {"id": id, "title": t, "body": t}});
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [doc("a", "The Running Cats"), doc("b", "they run")]})).await;
    assert_eq!(status, 200, "{r}");
    let ids = |r: &Json| {
        let mut v: Vec<String> = r["rows"].as_array().unwrap().iter().map(|x| x[0].as_str().unwrap().to_string()).collect();
        v.sort();
        v
    };
    for (sql, want) in [
        ("SELECT id FROM notes WHERE MATCH(title, 'run')", vec!["b"]),
        ("SELECT id FROM notes WHERE MATCH(body, 'run')", vec!["a", "b"]),
        ("SELECT id FROM notes WHERE MATCH(title, 'the')", vec!["a"]),
        ("SELECT id FROM notes ORDER BY bm25(title, 'the running') DESC LIMIT 1", vec!["a"]),
    ] {
        let (status, r) = post(&c, &format!("{base}/v1/query"), json!({ "sql": sql })).await;
        assert_eq!(status, 200, "{sql}: {r}");
        assert_eq!(ids(&r), want, "{sql}: {r}");
    }
}

/// `/v1/explain` gives the plan IR of a query, as JSON that reads back to
/// the query it was bound from, and as lines of text; a compound query's
/// plan has operators over single-table queries. `/v1/plan` runs a plan.
#[tokio::test(flavor = "multi_thread")]
async fn explain_gives_the_plan_ir() {
    use strata::query::ir::Plan;
    use strata::query::{bind_command, Command};
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "e.strata", |i| i < 300);
    let schema = db.schema().clone();
    let base = http(service(db, "http://unused"), None).await;
    let c = reqwest::Client::new();
    let sql = "SELECT id, price, score() FROM products WHERE category = 'kitchen' ORDER BY bm25(body, 'quiet kettle') DESC LIMIT 5";
    let (status, r) = post(&c, &format!("{base}/v1/explain"), json!({ "sql": sql })).await;
    assert_eq!(status, 200, "{r}");
    let select = strata::query::bind(sql, &schema, &Params::new()).unwrap();
    assert_eq!(Plan::from_json(&r["ir"]).unwrap().to_select(&schema).unwrap(), select, "{r}");
    let tree: Vec<&str> = r["tree"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    assert_eq!(
        tree,
        [
            "Project: id, price, score()",
            "  Limit: 5",
            "    Search products",
            "      filter: category = 'kitchen'",
            "      order: bm25(body, 'quiet kettle') DESC",
        ],
        "{r}"
    );
    assert!(r["plan"].as_array().is_some_and(|p| !p.is_empty()), "{r}");
    assert!(r.get("hits").is_none(), "{r}");
    // `hits`: the top rows with the values they were ranked by.
    let boosted = "SELECT id, price FROM products ORDER BY bm25(body, 'quiet kettle') * (1 + 0.1 * coalesce(rating, 0)) DESC LIMIT 5";
    let (status, h) = post(&c, &format!("{base}/v1/explain"), json!({ "sql": boosted, "hits": 3 })).await;
    assert_eq!(status, 200, "{h}");
    let hits = h["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3, "{h}");
    let (_, want) = post(&c, &format!("{base}/v1/query"), json!({ "sql": boosted })).await;
    for (i, hit) in hits.iter().enumerate() {
        assert_eq!(hit["id"], want["rows"][i][0], "{h}");
        let (score, bm25, rating) = (hit["score()"].as_f64().unwrap(), hit["bm25(body, 'quiet kettle')"].as_f64().unwrap(), hit["rating"].as_f64().unwrap_or(0.0));
        assert!((score - bm25 * (1.0 + 0.1 * rating)).abs() < 1e-9 * score.max(1.0), "{hit}");
        assert_eq!(hit["bm25(body, 'quiet kettle') * (1 + 0.1 * coalesce(rating, 0))"], hit["score()"], "{hit}");
    }
    let (status, e) = post(&c, &format!("{base}/v1/explain"), json!({ "sql": "SELECT id FROM products LIMIT 3", "hits": 3 })).await;
    assert!(status == 400 && e["error"].as_str().unwrap().contains("ORDER BY"), "{e}");
    // POST /v1/plan runs that plan as /v1/query runs the SQL, and a write
    // plan as /v1/sql runs the statement.
    let (status, want) = post(&c, &format!("{base}/v1/query"), json!({ "sql": sql })).await;
    assert_eq!(status, 200, "{want}");
    let (status, got) = post(&c, &format!("{base}/v1/plan"), json!({ "plan": r["ir"] })).await;
    assert_eq!(status, 200, "{got}");
    same(sql, &want, &got);
    let Command::Write { write, .. } = bind_command("DELETE FROM products WHERE category = 'kitchen'", &[&schema], &Params::new()).unwrap() else { panic!() };
    let kitchen = "SELECT COUNT(*) FROM products WHERE category = 'kitchen'";
    let (_, before) = post(&c, &format!("{base}/v1/query"), json!({ "sql": kitchen })).await;
    let (status, w) = post(&c, &format!("{base}/v1/plan"), json!({ "plan": Plan::of_write(&schema, &write).to_json().unwrap() })).await;
    assert_eq!(status, 200, "{w}");
    assert!(w["written"].as_u64().unwrap() > 0 && w["written"] == before["rows"][0][0], "{w} {before}");
    let (_, after) = post(&c, &format!("{base}/v1/query"), json!({ "sql": kitchen })).await;
    assert_eq!(after["rows"][0][0], 0, "{after}");
    // A plan this version cannot read is refused by name.
    let mut newer = r["ir"].clone();
    newer["features"] = json!(["hash_join"]);
    let (status, e) = post(&c, &format!("{base}/v1/plan"), json!({ "plan": newer })).await;
    assert!(status == 400 && e["error"].as_str().unwrap().contains("hash_join"), "{e}");
    // A compound query's plan is operators over single-table queries, and
    // runs as its SQL does.
    let compound = "SELECT id FROM products WHERE id IN (SELECT id FROM products WHERE price < 10) ORDER BY price DESC LIMIT 20";
    let (status, r) = post(&c, &format!("{base}/v1/explain"), json!({ "sql": compound })).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["ir"]["features"], json!(["operators"]), "{r}");
    let tree: Vec<&str> = r["tree"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    assert!(tree.iter().any(|l| l.contains("id IN values of (Project: id; Search products; filter: price <")), "{tree:?}");
    let (_, want) = post(&c, &format!("{base}/v1/query"), json!({ "sql": compound })).await;
    let (status, got) = post(&c, &format!("{base}/v1/plan"), json!({ "plan": r["ir"] })).await;
    assert_eq!(status, 200, "{got}");
    assert!(!want["rows"].as_array().unwrap().is_empty(), "{want}");
    same(compound, &want, &got);
}

/// Conditional writes: with `--seq-no` each row has the sequence number of
/// its last write (`seq_no()`), a write answers its own, and an operation
/// with `if_seq_no` applies only while the row still has that number,
/// else answers 409 and writes nothing. Of requests racing on one number,
/// exactly one applies, also when they share a commit.
#[tokio::test(flavor = "multi_thread")]
async fn conditional_writes_apply_while_the_row_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(dir.path(), "q.strata", |i| i < 200);
    db.storage().start_seq_no().unwrap();
    let svc = service(db, "http://unused");
    let base = http(svc.clone(), None).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/write");
    let row = |id: &'static str| {
        let (c, base) = (c.clone(), base.clone());
        async move {
            let (_, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": "SELECT seq_no(), sold FROM products WHERE id = :k", "params": {"k": id}})).await;
            r["rows"].get(0).map(|r| (r[0].as_u64().unwrap(), r[1].as_i64().unwrap()))
        }
    };
    let write = |ops: Json| {
        let (c, url) = (c.clone(), url.clone());
        async move { post(&c, &url, json!({"ops": ops})).await }
    };

    // Loaded rows have number 0; a write answers its number.
    assert_eq!(row("p0000003").await.unwrap().0, 0);
    let (status, r) = write(json!([{"op": "set", "key": "p0000003", "field": "sold", "value": 10, "if_seq_no": 0}])).await;
    assert_eq!((status, r["written"].clone(), r["seq_no"].clone()), (200, json!(1), json!(1)), "{r}");
    assert_eq!(row("p0000003").await, Some((1, 10)));
    // A stale number is refused, naming the row's.
    let (status, r) = write(json!([{"op": "add", "key": "p0000003", "field": "sold", "value": 1, "if_seq_no": 0}])).await;
    assert_eq!(status, 409, "{r}");
    assert!(r["error"].as_str().unwrap().contains("its sequence number is 1"), "{r}");
    let (status, r) = write(json!([{"op": "add", "key": "p0000003", "field": "sold", "value": 1, "if_seq_no": 1}])).await;
    assert_eq!((status, r["seq_no"].clone()), (200, json!(2)), "{r}");
    assert_eq!(row("p0000003").await, Some((2, 11)));

    // A refused operation takes the request's others with it.
    let (status, _) = write(json!([
        {"op": "set", "key": "p0000004", "field": "sold", "value": 99},
        {"op": "delete", "key": "p0000003", "if_seq_no": 1},
    ]))
    .await;
    assert_eq!(status, 409);
    assert_eq!(row("p0000004").await.unwrap().0, 0);
    // An operation after another on its row in the same request finds no
    // number to match.
    let (status, r) = write(json!([
        {"op": "set", "key": "p0000004", "field": "sold", "value": 99},
        {"op": "set", "key": "p0000004", "field": "sold", "value": 98, "if_seq_no": 0},
    ]))
    .await;
    assert_eq!(status, 409, "{r}");
    assert!(r["error"].as_str().unwrap().contains("an earlier operation of this request"), "{r}");
    // Upserts and deletes take a condition too; a missing row matches none.
    let (status, r) = write(json!([{"op": "upsert", "doc": {"id": "p0000004", "title": "Four", "sold": 4}, "if_seq_no": 0}])).await;
    assert_eq!((status, r["seq_no"].clone()), (200, json!(3)), "{r}");
    let (status, _) = write(json!([{"op": "delete", "key": "p0000004", "if_seq_no": 2}])).await;
    assert_eq!(status, 409);
    let (status, _) = write(json!([{"op": "delete", "key": "p0000004", "if_seq_no": 3}])).await;
    assert_eq!(status, 200);
    let (status, r) = write(json!([{"op": "set", "key": "p0000004", "field": "sold", "value": 1, "if_seq_no": 4}])).await;
    assert_eq!(status, 409);
    assert!(r["error"].as_str().unwrap().contains("no row has the key"), "{r}");
    // Refused as malformed: on an insert, or not a number.
    assert_eq!(write(json!([{"op": "insert", "doc": {"id": "n1", "title": "New"}, "if_seq_no": 0}])).await.0, 400);
    assert_eq!(write(json!([{"op": "add", "key": "p0000005", "field": "sold", "value": 1, "if_seq_no": "0"}])).await.0, 400);

    // SQL writes answer their number too; a retry with an idempotency key
    // answers the first one's.
    let (status, r) = post(&c, &format!("{base}/v1/sql"), json!({"sql": "UPDATE products SET sold = 0 WHERE id = 'p0000006'"})).await;
    assert_eq!((status, r["seq_no"].clone()), (200, json!(5)), "{r}");
    for _ in 0..2 {
        let (status, r) = post(&c, &url, json!({"ops": [{"op": "add", "key": "p0000006", "field": "sold", "value": 1}], "idempotency_key": "k1"})).await;
        assert_eq!((status, r["seq_no"].clone()), (200, json!(6)), "{r}");
    }

    // Requests racing on one number: exactly one applies. While the
    // writer waits on a write of another row, they queue for one commit.
    let (seq, sold) = row("p0000007").await.unwrap();
    let held = svc.local_db().unwrap().storage().begin().unwrap();
    let first = tokio::spawn(write(json!([{"op": "add", "key": "p0000009", "field": "sold", "value": 1}])));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let racing: Vec<_> = (0..32).map(|_| tokio::spawn(write(json!([{"op": "add", "key": "p0000007", "field": "sold", "value": 1, "if_seq_no": seq}])))).collect();
    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(held);
    assert_eq!(first.await.unwrap().0, 200);
    let mut statuses = Vec::new();
    for r in racing {
        statuses.push(r.await.unwrap().0);
    }
    assert_eq!((statuses.iter().filter(|&&s| s == 200).count(), statuses.iter().filter(|&&s| s == 409).count()), (1, 31), "{statuses:?}");
    assert_eq!(row("p0000007").await.unwrap().1, sold + 1);

    // gRPC answers the number as well.
    let mut g = pb::strata_client::StrataClient::connect(grpc(svc.clone()).await).await.unwrap();
    let ops = |n: u64| serde_json::to_vec(&json!([{"op": "add", "key": "p0000008", "field": "sold", "value": 1, "if_seq_no": n}])).unwrap();
    let w = g.write(pb::WriteRequest { ops_json: ops(0), ..Default::default() }).await.unwrap().into_inner();
    // 7 was the write of another row, 8 the one racer that applied: the
    // commits that failed took no number.
    assert_eq!((w.written, w.seq_no), (1, Some(9)));
    let e = g.write(pb::WriteRequest { ops_json: ops(0), ..Default::default() }).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition, "{e:?}");

    // A read may name the write it must see (`min_seq_no`). This server
    // holds every write it answered, and refuses a number past its last.
    let sql = "SELECT sold FROM products WHERE id = 'p0000008'";
    let plan = strata::query::ir::Plan::bind(sql, &svc.schema, &Params::new()).unwrap().to_json().unwrap();
    for (path, body) in [("query", json!({"sql": sql})), ("sql", json!({"sql": sql})), ("plan", json!({"plan": plan}))] {
        let url = format!("{base}/v1/{path}");
        let with = |extra: Json| {
            let mut b = body.clone();
            b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            b
        };
        let (status, r) = post(&c, &url, with(json!({"min_seq_no": 9}))).await;
        assert_eq!((status, r["rows"].as_array().map(Vec::len)), (200, Some(1)), "{path}: {r}");
        let (status, r) = post(&c, &url, with(json!({"min_seq_no": 10}))).await;
        assert!(status == 409 && r["error"].as_str().unwrap().contains("transaction 10 was not committed here"), "{path}: {r}");
        for bad in [json!({"min_seq_no": -1}), json!({"min_seq_no": "9"}), json!({"min_seq_nos": 9}), json!({"min_seq_nos": [9, "x"]}), json!({"min_seq_nos": [9]}), json!({"min_seq_no": 9, "min_seq_nos": [9]})] {
            let (status, r) = post(&c, &url, with(bad.clone())).await;
            assert_eq!(status, 400, "{path} {bad}: {r}");
        }
    }
    let (status, r) = post(&c, &format!("{base}/v1/sql"), json!({"sql": "UPDATE products SET sold = 1 WHERE id = 'p0000008'", "min_seq_no": 9})).await;
    assert!(status == 400 && r["error"].as_str().unwrap().contains("are for reads"), "{r}");
    let read = |min_seq_no: u64| pb::QueryRequest { sql: sql.into(), protocol: 5, min_seq_no, ..Default::default() };
    assert_eq!(g.query(read(9)).await.unwrap().into_inner().rows.len(), 1);
    assert_eq!(g.query(read(10)).await.unwrap_err().code(), tonic::Code::FailedPrecondition);
    let e = g.query(pb::QueryRequest { min_seq_nos: vec![9], ..read(0) }).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");

    // A database that does not number its writes refuses both, by name,
    // and answers no number.
    let plain = http(service(create(dir.path(), "p.strata", |i| i < 50), "http://unused"), None).await;
    let (status, r) = post(&c, &format!("{plain}/v1/write"), json!({"ops": [{"op": "add", "key": "p0000003", "field": "sold", "value": 1, "if_seq_no": 0}]})).await;
    assert_eq!(status, 400);
    assert!(r["error"].as_str().unwrap().contains("--seq-no"), "{r}");
    let (status, r) = post(&c, &format!("{plain}/v1/query"), json!({"sql": "SELECT id, seq_no() FROM products LIMIT 1"})).await;
    assert_eq!(status, 400);
    assert!(r["error"].as_str().unwrap().contains("does not number its writes"), "{r}");
    let (status, r) = post(&c, &format!("{plain}/v1/write"), json!({"ops": [{"op": "add", "key": "p0000003", "field": "sold", "value": 1}]})).await;
    assert_eq!(status, 200);
    assert!(r.get("seq_no").is_none(), "{r}");
    let (status, r) = post(&c, &format!("{plain}/v1/query"), json!({"sql": sql, "min_seq_no": 1})).await;
    assert!(status == 400 && r["error"].as_str().unwrap().contains("--seq-no"), "{r}");
    assert_eq!(post(&c, &format!("{plain}/v1/query"), json!({"sql": sql, "min_seq_no": 0})).await.0, 200);
}

/// Through a coordinator, `seq_no()` reads each shard's numbers and
/// `if_seq_no` reaches the row's shard, a conflict answering 409; with
/// more than one replica of a shard, whose numbers differ, both are
/// refused. `--seq-no` starts numbering on a server's database, and a
/// coordinator refuses it.
#[tokio::test(flavor = "multi_thread")]
async fn conditional_writes_through_a_coordinator() {
    let dir = tempfile::tempdir().unwrap();
    let key = |i: usize| Value::Str(format!("p{i:07}"));
    let mut urls = Vec::new();
    for s in 0..2 {
        let db = create(dir.path(), &format!("s{s}.strata"), |i| i < 300 && shard_of(&key(i), 2) == s);
        db.storage().start_seq_no().unwrap();
        urls.push(grpc(service(db, "http://unused")).await);
    }
    let coord = cluster(Coordinator::connect(urls.clone()).await.unwrap());
    let seq_of = |id: &'static str| {
        let coord = coord.clone();
        async move {
            let q = QueryReq { sql: format!("SELECT seq_no() FROM products WHERE id = '{id}'"), ..Default::default() };
            coord.query(q).await.unwrap().rs.rows[0][0].as_i64().unwrap()
        }
    };
    let set = |id: &str, n: u64| vec![json!({"op": "set", "key": id, "field": "sold", "value": 5, "if_seq_no": n})];
    assert_eq!(seq_of("p0000003").await, 0);
    let w = coord.write_ops(set("p0000003", 0), None).await.unwrap();
    assert_eq!((w.rows, w.seq_no), (1, None), "a coordinator answers no number: each shard numbers its own");
    let s3 = shard_of(&key(3), 2);
    assert_eq!(w.seq_nos, [(s3 == 0) as u64, (s3 == 1) as u64], "but one per shard: 0 where it wrote nothing");
    assert_eq!(seq_of("p0000003").await, 1);
    // Over HTTP and gRPC, a read through it names the write it must see
    // with each shard's number, as the write answered them.
    let base = http(coord.clone(), None).await;
    let c = reqwest::Client::new();
    let (status, r) = post(&c, &format!("{base}/v1/write"), json!({"ops": [{"op": "set", "key": "p0000004", "field": "sold", "value": 6}]})).await;
    assert_eq!(status, 200, "{r}");
    let nos = r["seq_nos"].clone();
    assert!(r.get("seq_no").is_none() && nos.as_array().is_some_and(|a| a.len() == 2), "{r}");
    let sql = "SELECT sold FROM products WHERE id = 'p0000004'";
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "min_seq_nos": nos})).await;
    assert_eq!((status, r["rows"][0][0].clone()), (200, json!(6)), "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "min_seq_no": 1})).await;
    assert!(status == 400 && r["error"].as_str().unwrap().contains("a coordinator takes min_seq_nos"), "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "min_seq_nos": [0, 0, 0]})).await;
    assert_eq!(status, 400, "{r}");
    let (status, r) = post(&c, &format!("{base}/v1/query"), json!({"sql": sql, "min_seq_nos": [1000, 1000]})).await;
    assert!(status == 409 && r["error"].as_str().unwrap().contains("was not committed here"), "{r}");
    let mut g = pb::strata_client::StrataClient::connect(grpc(coord.clone()).await).await.unwrap();
    let ops = serde_json::to_vec(&json!([{"op": "set", "key": "p0000004", "field": "sold", "value": 7}])).unwrap();
    let w = g.write(pb::WriteRequest { ops_json: ops, ..Default::default() }).await.unwrap().into_inner();
    assert!(w.seq_no.is_none() && w.seq_nos.len() == 2, "{w:?}");
    let a = g.query(pb::QueryRequest { sql: sql.into(), protocol: 5, min_seq_nos: w.seq_nos, ..Default::default() }).await.unwrap().into_inner();
    assert_eq!(a.rows.len(), 1);
    let e = coord.write_ops(set("p0000003", 0), None).await.unwrap_err();
    assert_eq!(e.status, 409, "{}", e.message);

    // Shards of two copies that each take the shard's writes (copies of
    // the shards above, but servers of their own: the shards above hold
    // a cluster map naming one server each).
    let mut copies = Vec::new();
    for s in 0..2 {
        let mut group = Vec::new();
        for c in 0..2 {
            let db = create(dir.path(), &format!("c{s}{c}.strata"), |i| i < 300 && shard_of(&key(i), 2) == s);
            db.storage().start_seq_no().unwrap();
            group.push(grpc(service(db, "http://unused")).await);
        }
        copies.push(group.join("|"));
    }
    let replicated = cluster(Coordinator::connect(copies).await.unwrap());
    let e = replicated.write_ops(set("p0000003", 1), None).await.unwrap_err();
    assert!(e.status == 400 && e.message.contains("one replica per shard"), "{}", e.message);
    let e = replicated.query(QueryReq { sql: "SELECT id, seq_no() FROM products LIMIT 3".into(), ..Default::default() }).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("one replica per shard"), "{}", e.message);
    // Each copy numbers what it takes, so a write answers no number, and
    // a read cannot name one.
    let w = replicated.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": 8})], None).await.unwrap();
    assert!(w.seq_no.is_none() && w.seq_nos.is_empty(), "{w:?}");
    let read = |after: Vec<u64>| QueryReq { sql: "SELECT sold FROM products WHERE id = 'p0000003'".into(), after: After::Shards(after), ..Default::default() };
    let e = replicated.query(read(vec![0, 1])).await.err().unwrap();
    assert!(e.status == 400 && e.message.contains("whose replicas each number their own writes"), "{}", e.message);
    assert!(replicated.query(read(vec![0, 0])).await.is_ok());

    // --seq-no.
    let numbered = |name: &str, seq_no: bool| server::Options { db: Some(dir.path().join(name)), demo: Some(20), seq_no, ..Default::default() };
    let svc = server::service(&numbered("on.strata", true)).await.unwrap();
    assert!(svc.local_db().unwrap().storage().seq_no().is_some());
    let svc = server::service(&numbered("off.strata", false)).await.unwrap();
    assert!(svc.local_db().unwrap().storage().seq_no().is_none());
    let e = server::service(&server::Options { shards: urls, seq_no: true, ..Default::default() }).await.err().unwrap();
    assert!(e.contains("start each shard server with it"), "{e}");
}

/// Once a replica is promoted, a coordinator sends the shard's writes to
/// it. A write the old primary refuses as replaced, or that finds it down,
/// goes once to the copy that took its place; a conditional write's
/// refusal is not retried.
#[tokio::test(flavor = "multi_thread")]
async fn a_coordinator_writes_to_the_promoted_primary() {
    let dir = tempfile::tempdir().unwrap();
    let numbered = |name: &str| {
        let opts = DbOptions { publish_snapshots: true, ship_log_bytes: 64 << 20, ..DbOptions::default() };
        let db = Db::create(&dir.path().join(name), demo::schema(N), opts).unwrap();
        db.set_embedder("emb", Arc::new(strata::embed::HashEmbedder { dim: demo::DIM })).unwrap();
        db.load((0..300).map(|i| demo::product(i, 7))).unwrap();
        db.storage().start_seq_no().unwrap();
        Arc::new(db)
    };
    let fast = |replicas: Vec<String>| PrimaryOptions { replicas, acks: Acks::All, heartbeat: Duration::from_millis(100), retry_max: Duration::from_millis(200), ..Default::default() };
    let sold = |db: &Db| db.query("SELECT sold, seq_no() FROM products WHERE id = 'p0000003'", &Params::new()).unwrap().rows[0].clone();
    let set = |v: i64| vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": v})];

    // The old primary is up, and learns it was replaced when it next ships.
    let (pdb, rdb) = (numbered("p.strata"), numbered("r.strata"));
    let r = service_with(rdb.clone(), |s| s.set_replication(Replication::Replica(Arc::new(Follow::open(rdb.clone(), None).unwrap()))));
    let ru = grpc(r.clone()).await;
    let p = service_with(pdb.clone(), |s| s.set_replication(Replication::Primary(Primary::start(pdb.clone(), None, fast(vec![ru.clone()])).unwrap())));
    let (pu, old) = recording_of(p.clone(), 5).await;
    let coord = cluster(Coordinator::connect_with(vec![format!("{pu}|{ru}")], CoordOptions { hedge: Duration::from_secs(60), ..Default::default() }).await.unwrap());
    assert_eq!(coord.write_ops(set(1), None).await.unwrap().seq_nos, vec![1]);
    // The old primary votes for the replica, and steps down: it becomes a
    // replica of the new one.
    let a = r.promote(&Principal::open(), vec![pu.clone()], Acks::All, false).await.unwrap();
    assert_eq!(a["replicas"][0]["role"], "primary", "{a}");
    assert_eq!(a["votes"][0]["granted"], true, "{a}");
    assert_eq!(p.role(), pb::Role::Replica);
    // A conditional write the new primary refuses is refused once.
    let e = coord.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": 9, "if_seq_no": 7})], None).await.err().unwrap();
    assert_eq!(e.status, 409, "{}", e.message);
    let w = coord.write_ops(set(2), None).await.unwrap();
    assert_eq!(w.seq_nos, vec![2]);
    assert_eq!(sold(&rdb), vec![Value::Int(2), Value::Int(2)]);
    assert_eq!(rdb.storage().numbering().unwrap().term_at(2), 2);
    let h = coord.health().await;
    assert_eq!((&h["shards"][0]["replicas"][0]["role"], &h["shards"][0]["replicas"][1]["role"]), (&json!("replica"), &json!("primary")), "{h}");
    // And the next write goes there at once; a conditional write the new
    // primary refuses asks it again, and no other copy.
    let (hellos, writes) = (old.hellos.load(Ordering::Relaxed), old.writes.load(Ordering::Relaxed));
    assert_eq!(coord.write_ops(set(3), None).await.unwrap().seq_nos, vec![3]);
    let e = coord.write_ops(vec![json!({"op": "set", "key": "p0000003", "field": "sold", "value": 9, "if_seq_no": 1})], None).await.err().unwrap();
    assert_eq!(e.status, 409, "{}", e.message);
    assert_eq!((old.hellos.load(Ordering::Relaxed), old.writes.load(Ordering::Relaxed)), (hellos, writes), "the old primary was not asked");
    assert_eq!(last_of(&pdb), 3, "the old primary holds the new one's writes, which ships to it");

    // The old primary is down, listed before the replica or after it.
    for old_first in [true, false] {
        let (pdb, rdb) = (numbered(&format!("p-{old_first}.strata")), numbered(&format!("r-{old_first}.strata")));
        let r = service_with(rdb.clone(), |s| s.set_replication(Replication::Replica(Arc::new(Follow::open(rdb.clone(), None).unwrap()))));
        let ru = grpc(r.clone()).await;
        let p = service_with(pdb.clone(), |s| s.set_replication(Replication::Primary(Primary::start(pdb.clone(), None, fast(vec![ru.clone()])).unwrap())));
        let (pu, old) = recording_of(p.clone(), 5).await;
        let copies = if old_first { format!("{pu}|{ru}") } else { format!("{ru}|{pu}") };
        let coord = cluster(Coordinator::connect_with(vec![copies], CoordOptions { hedge: Duration::from_secs(60), ..Default::default() }).await.unwrap());
        assert_eq!(coord.write_ops(set(1), None).await.unwrap().seq_nos, vec![1]);
        old.down.store(true, Ordering::Relaxed);
        // One copy of two is no majority, and a replica that knows no other
        // copy has none to ask: promoted only when forced.
        let e = r.promote(&Principal::open(), vec![pu.clone()], Acks::All, false).await.unwrap_err();
        assert!(e.status == 409 && e.message.contains("and it needs 2"), "{}", e.message);
        let e = r.promote(&Principal::open(), Vec::new(), Acks::All, false).await.unwrap_err();
        assert!(e.status == 409 && e.message.contains("knows no other copy"), "{}", e.message);
        r.promote(&Principal::open(), Vec::new(), Acks::All, true).await.unwrap();
        let w = coord.write_ops(set(2), None).await.unwrap();
        assert_eq!(w.seq_nos, vec![2]);
        assert_eq!(sold(&rdb), vec![Value::Int(2), Value::Int(2)]);
        // The coordinator flags the two copies that last said they were
        // primaries: the old one has not heard of the new one.
        let h = coord.health().await;
        assert_eq!(h["shards"][0]["primaries"].as_array().map(Vec::len), Some(2), "{h}");
        assert!(h["shards"][0]["warning"].as_str().unwrap().contains("2 copies say they are the shard's primary"), "{h}");
        // The next goes to the new primary, of the newer term, at once:
        // both say they are primaries, and the old one is not asked.
        let (hellos, writes) = (old.hellos.load(Ordering::Relaxed), old.writes.load(Ordering::Relaxed));
        assert_eq!(coord.write_ops(set(3), None).await.unwrap().seq_nos, vec![3]);
        assert_eq!((old.hellos.load(Ordering::Relaxed), old.writes.load(Ordering::Relaxed)), (hellos, writes), "the old primary was not asked");
        assert_eq!(last_of(&pdb), 1);
    }
}

fn last_of(db: &Db) -> u64 {
    db.storage().numbering().unwrap().seq
}

/// The next codes of semantic-ID prefixes over HTTP, and from a
/// coordinator, which sums its shards': as the embedded database answers.
/// A shard of an older protocol is named in the error.
#[tokio::test(flavor = "multi_thread")]
async fn next_semantic_id_codes_over_http_and_shards() {
    use std::collections::BTreeMap;
    use strata::schema::{Field, FieldType, Schema};
    use strata::sid::Codebook;
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    // A vector near (10a + 1-or-minus-1, 10b + ...) has code a.b of a 2 x 4
    // grid codebook in two dimensions.
    let l0: Vec<f32> = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]].concat();
    let l1: Vec<f32> = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]].concat();
    Codebook::new(2, 4, vec![l0.clone(), l1.clone()]).unwrap().save("grid", &models).unwrap();
    let schema = Schema::new(
        "items",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("tag", FieldType::Keyword),
            Field::new("emb", FieldType::Vector { dim: 2, index: strata::VectorIndexKind::Flat }),
            Field::semantic_id("sid", "emb", "grid"),
        ],
        Some("id"),
    )
    .unwrap();
    let rows: Vec<Vec<Value>> = (0..400)
        .map(|i| {
            let (a, b) = ((i * 5 + i / 7) % 4, (i * 3 + i / 11) % 4);
            let v = vec![l0[2 * a] + l1[2 * b] + 0.01 * (i % 5) as f32, l0[2 * a + 1] + l1[2 * b + 1]];
            vec![Value::Str(format!("i{i:04}")), Value::Str(["red", "blue"][i % 2].into()), Value::Vector(v.into()), Value::Null]
        })
        .collect();
    let make = |name: &str, keep: &dyn Fn(&Value) -> bool| {
        let db = Db::create_in(&dir.path().join(name), schema.clone(), DbOptions::default(), &models).unwrap();
        db.load(rows.iter().filter(|r| keep(&r[0])).cloned()).unwrap();
        db
    };
    let whole = make("whole.strata", &|_| true);
    let prefixes: Vec<String> = ["", "0", "1", "2.3", "3.1", "1.2.4", "7"].map(String::from).to_vec();
    let want = |filter: Option<&str>| whole.sid_next("items", "sid", &prefixes, filter, &Params::new()).unwrap();
    assert!(want(None)[0].len() == 4 && want(None)[2].len() == 4, "{:?}", want(None));
    let as_json = |next: Vec<Vec<(String, u64)>>| -> Json {
        prefixes.iter().zip(next).map(|(p, cs)| json!({"prefix": p, "next": cs.into_iter().map(|(code, count)| json!({"code": code, "count": count})).collect::<Vec<_>>()})).collect()
    };

    // Over HTTP.
    let url = http(service(make("http.strata", &|_| true), "http://unused"), None).await;
    let c = reqwest::Client::new();
    let (status, got) = post(&c, &format!("{url}/v1/sid/next"), json!({"field": "sid", "prefixes": prefixes, "where": "tag = :t", "params": {"t": "red"}})).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["next"], as_json(want(Some("tag = 'red'"))));
    let (status, got) = post(&c, &format!("{url}/v1/sid/next"), json!({"table": "items", "field": "sid", "prefixes": prefixes})).await;
    assert_eq!((status, &got["next"]), (200, &as_json(want(None))));
    for (body, says) in [
        (json!({"field": "sid"}), "prefixes"),
        (json!({"field": "tag", "prefixes": [""]}), "not a semantic-ID field"),
        (json!({"field": "sid", "prefixes": ["1..2"]}), "not a prefix"),
        (json!({"field": "sid", "prefixes": [""], "where": "id IN (SELECT id FROM items)"}), "subqueries"),
    ] {
        let (status, got) = post(&c, &format!("{url}/v1/sid/next"), body.clone()).await;
        assert_eq!(status, 400, "{body}: {got}");
        assert!(got["error"].as_str().unwrap().contains(says), "{body}: {got}");
    }

    // A coordinator sums its shards'. Each shard numbers its own ties, so
    // the levels' counts are the whole table's and a tie's are the sum of
    // the shards' with that tie.
    let mut urls = Vec::new();
    let mut shards = Vec::new();
    for s in 0..3 {
        let db = make(&format!("s{s}.strata"), &|k| shard_of(k, 3) == s);
        shards.push(make(&format!("c{s}.strata"), &|k| shard_of(k, 3) == s));
        urls.push(recording(db, strata_server::grpc::PROTOCOL).await.0);
    }
    let coord = cluster(Coordinator::connect(urls.clone()).await.unwrap());
    for filter in [None, Some("tag = 'blue'"), Some("sid = '2'")] {
        let mut sums: Vec<BTreeMap<u32, (String, u64)>> = vec![BTreeMap::new(); prefixes.len()];
        for db in &shards {
            for (sum, next) in sums.iter_mut().zip(db.sid_next("items", "sid", &prefixes, filter, &Params::new()).unwrap()) {
                for (code, k) in next {
                    sum.entry(code.rsplit('.').next().unwrap().parse().unwrap()).or_insert((code, 0)).1 += k;
                }
            }
        }
        let summed: Vec<Vec<(String, u64)>> = sums.into_iter().map(|m| m.into_values().collect()).collect();
        assert_eq!(summed[..3], want(filter)[..3], "{filter:?}");
        let sql = format!("SELECT COUNT(*) FROM items{}", filter.map(|f| format!(" WHERE {f}")).unwrap_or_default());
        let got = coord.sid_next(&sql, None, &Params::new(), "sid", &prefixes, &After::Any, 0).await.unwrap_or_else(|e| panic!("{filter:?}: {}", e.message));
        assert_eq!(got, summed, "{filter:?}");
    }
    // One shard from before SidNext.
    let (old, _) = recording(make("old.strata", &|k| shard_of(k, 3) == 2), 5).await;
    let coord = cluster(Coordinator::connect(vec![urls[0].clone(), urls[1].clone(), old]).await.unwrap());
    let e = coord.sid_next("SELECT COUNT(*) FROM items", None, &Params::new(), "sid", &prefixes, &After::Any, 0).await.unwrap_err();
    assert!(e.message.contains("protocol 5") && e.message.contains("needs 7"), "{}", e.message);
}
