//! VECTOR fields declared as the embedding of other fields: the schema
//! records them, writes that leave the vector out get it computed, rewrites
//! of a source recompute it, merges and copies do not re-embed, a copy
//! into a new schema embeds only what changed, and SQL writes embed before
//! they take the writer lock.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use strata::db::Db;
use strata::embed::{Embedder, HashEmbedder};
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::{DbOptions, MergePolicy};
use strata::value::Value;
use strata::VectorIndexKind;

const DIM: usize = 32;

/// The hash model, counting the texts it embeds.
struct Counting(HashEmbedder, AtomicUsize);

impl Embedder for Counting {
    fn name(&self) -> String {
        self.0.name()
    }
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        self.1.fetch_add(texts.len(), Ordering::Relaxed);
        self.0.embed(texts)
    }
}

fn counting() -> Arc<Counting> {
    Arc::new(Counting(HashEmbedder { dim: DIM }, AtomicUsize::new(0)))
}

fn vector(dim: usize) -> FieldType {
    FieldType::Vector { dim, index: VectorIndexKind::Flat }
}

/// id, title, text, stock, emb (embedding of title + text).
fn schema() -> Schema {
    Schema::new(
        "docs",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("title", FieldType::Text { positions: false }),
            Field::new("text", FieldType::Text { positions: false }),
            Field::new("stock", FieldType::Int),
            Field::new("emb", vector(DIM)).not_stored().embedded("hash:32", &["title", "text"]),
        ],
        Some("id"),
    )
    .unwrap()
}

fn doc(i: usize, title: &str, text: &str) -> Vec<Value> {
    vec![Value::Str(format!("d{i:04}")), Value::Str(title.into()), Value::Str(text.into()), Value::Int(i as i64), Value::Null]
}

fn emb_of(db: &Db, key: &str) -> Option<Vec<f32>> {
    let r = db.query("SELECT emb FROM docs WHERE id = :k", &Params::new().set("k", Value::Str(key.into()))).unwrap();
    r.rows.first().and_then(|row| row[0].as_vector().map(|v| v.to_vec()))
}

fn want(text: &str) -> Vec<f32> {
    HashEmbedder { dim: DIM }.embed(&[text]).unwrap().pop().unwrap()
}

#[test]
fn schema_checks_embedded_fields() {
    let bad = |fields: Vec<Field>| Schema::new("t", fields, Some("id")).is_err();
    let id = || Field::new("id", FieldType::Keyword);
    let title = || Field::new("title", FieldType::Text { positions: false });
    // Only a VECTOR field, from existing stored text.
    assert!(bad(vec![id(), title(), Field::new("x", FieldType::Text { positions: false }).embedded("hash:8", &["title"])]));
    assert!(bad(vec![id(), title(), Field::new("e", vector(8)).embedded("hash:8", &["nope"])]));
    assert!(bad(vec![id(), title().not_stored(), Field::new("e", vector(8)).embedded("hash:8", &["title"])]));
    assert!(bad(vec![id(), Field::new("n", FieldType::Int), Field::new("e", vector(8)).embedded("hash:8", &["n"])]));
    assert!(bad(vec![id(), title(), Field::new("e", vector(8)).embedded("hash:8", &[])]));
    // A generated source is fine.
    let body = Field::new("body", FieldType::Text { positions: false }).not_stored().generated(&["title"]);
    assert!(!bad(vec![id(), title(), body, Field::new("e", vector(8)).embedded("hash:8", &["body"])]));
}

#[test]
fn writes_embed_what_they_leave_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("e.strata");
    let db = Db::create(&path, schema(), DbOptions::default()).unwrap();
    let m = counting();
    db.set_embedder("emb", m.clone()).unwrap();
    // A model of the wrong size, or for a field that is not embedded, is refused.
    assert!(db.set_embedder("emb", Arc::new(HashEmbedder { dim: 8 })).is_err());
    assert!(db.set_embedder("title", m.clone()).is_err());

    let mut docs: Vec<_> = (0..200).map(|i| doc(i, &format!("red mug {i}"), "ceramic, large")).collect();
    // A vector the writer supplies is kept; a row with no source text gets none.
    docs[1][4] = Value::Vector(vec![1.0; DIM].into());
    docs[2] = doc(2, "", "");
    db.load(docs).unwrap();
    assert_eq!(m.1.load(Ordering::Relaxed), 198);
    assert_eq!(emb_of(&db, "d0000").unwrap(), want("red mug 0\nceramic, large"));
    assert_eq!(emb_of(&db, "d0001").unwrap(), vec![1.0; DIM]);
    // (A row without a vector reads back as zeros.)
    assert!(emb_of(&db, "d0002").is_none_or(|v| v.iter().all(|&x| x == 0.0)));

    // SQL INSERT, and UPDATE that rewrites a source.
    db.sql("INSERT INTO docs (id, title, text, stock) VALUES ('n1', 'blue kettle', 'steel', 3)", &Params::new()).unwrap();
    assert_eq!(emb_of(&db, "n1").unwrap(), want("blue kettle\nsteel"));
    db.sql("UPDATE docs SET title = 'green teapot' WHERE id = 'd0000'", &Params::new()).unwrap();
    assert_eq!(emb_of(&db, "d0000").unwrap(), want("green teapot\nceramic, large"));
    // A supplied vector survives an UPDATE of numbers only.
    db.sql("UPDATE docs SET stock = 9 WHERE id = 'd0001'", &Params::new()).unwrap();
    assert_eq!(emb_of(&db, "d0001").unwrap(), vec![1.0; DIM]);

    // Text search by meaning finds the rewritten row.
    let q = Params::new().set("q", Value::Vector(m.embed_query("green teapot\nceramic, large").unwrap().into()));
    let r = db.query("SELECT id FROM docs ORDER BY similarity(emb, :q) DESC LIMIT 1", &q).unwrap();
    assert_eq!(r.rows[0][0], Value::Str("d0000".into()));

    // Merges rebuild rows from what is stored: no model calls.
    let before = m.1.load(Ordering::Relaxed);
    db.storage().flush().unwrap();
    let stats = db.storage().merge(&MergePolicy { merge_factor: 2, ..MergePolicy::default() }).unwrap();
    assert!(stats.docs_merged > 0, "{stats:?}");
    assert_eq!(m.1.load(Ordering::Relaxed), before);
    assert_eq!(emb_of(&db, "d0000").unwrap(), want("green teapot\nceramic, large"));
    drop(db);

    // The schema keeps the embedding; a reopened database needs its model again.
    let db = Db::open(&path, DbOptions::default()).unwrap();
    let e = db.schema().field("emb").unwrap().embed.clone().unwrap();
    assert_eq!((e.from, e.model), (vec!["title".to_string(), "text".to_string()], "hash:32".to_string()));
    assert!(db.embedder("emb").is_none());
    assert!(db.open_embedders(dir.path()).is_empty());
    assert!(db.embedder("emb").is_some());
}

#[test]
fn a_copy_into_a_new_schema_embeds_only_what_changed() {
    let dir = tempfile::tempdir().unwrap();
    let old = Db::create(&dir.path().join("old.strata"), schema(), DbOptions::default()).unwrap();
    old.set_embedder("emb", counting()).unwrap();
    old.load((0..300).map(|i| doc(i, &format!("lamp {i}"), "brass and glass"))).unwrap();
    old.sql("DELETE FROM docs WHERE id = 'd0007'", &Params::new()).unwrap();

    // The same `emb`, plus a vector of the title alone; `stock` dropped.
    let new_schema = Schema::new(
        "docs",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("title", FieldType::Text { positions: false }),
            Field::new("text", FieldType::Text { positions: false }),
            Field::new("emb", vector(DIM)).not_stored().embedded("hash:32", &["title", "text"]),
            Field::new("emb_title", vector(DIM)).not_stored().embedded("hash:32", &["title"]),
        ],
        Some("id"),
    )
    .unwrap();
    let new = Db::create(&dir.path().join("new.strata"), new_schema, DbOptions::default()).unwrap();
    let m = counting();
    new.set_embedder("emb", m.clone()).unwrap();
    new.set_embedder("emb_title", m.clone()).unwrap();
    assert_eq!(new.load_from(&old).unwrap(), 299);
    // Only the new field was embedded.
    assert_eq!(m.1.load(Ordering::Relaxed), 299);
    let r = new.query("SELECT id, emb, emb_title FROM docs WHERE id = 'd0005'", &Params::new()).unwrap();
    assert_eq!(r.rows[0][1].as_vector().unwrap(), &want("lamp 5\nbrass and glass")[..]);
    assert_eq!(r.rows[0][2].as_vector().unwrap(), &want("lamp 5")[..]);
    assert_eq!(new.query("SELECT COUNT(*) FROM docs WHERE id = 'd0007'", &Params::new()).unwrap().rows[0][0], Value::Int(0));
}

/// The hash model; once armed, its next call says it started and waits to
/// be let go, so a write is stopped while it embeds.
struct Gated {
    inner: HashEmbedder,
    armed: AtomicBool,
    texts: AtomicUsize,
    started: Mutex<mpsc::Sender<()>>,
    go: Mutex<mpsc::Receiver<()>>,
}

impl Embedder for Gated {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        self.texts.fetch_add(texts.len(), Ordering::Relaxed);
        if self.armed.swap(false, Ordering::SeqCst) {
            self.started.lock().unwrap().send(()).unwrap();
            self.go.lock().unwrap().recv().unwrap();
        }
        self.inner.embed(texts)
    }
}

/// Review ARCH-9: an INSERT, an UPDATE that rewrites a source and an `ON
/// CONFLICT DO UPDATE` compute their vectors before they take the writer
/// lock, so other writes commit while the model works, and nothing is
/// embedded again under the lock.
#[test]
fn sql_writes_embed_before_taking_the_writer_lock() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("e.strata"), schema(), DbOptions::default()).unwrap();
    let (started_tx, started) = mpsc::channel();
    let (go, go_rx) = mpsc::channel();
    let m = Arc::new(Gated { inner: HashEmbedder { dim: DIM }, armed: AtomicBool::new(false), texts: AtomicUsize::new(0), started: Mutex::new(started_tx), go: Mutex::new(go_rx) });
    db.set_embedder("emb", m.clone()).unwrap();
    db.sql("INSERT INTO docs (id, title, text, stock) VALUES ('a', 'red mug', 'ceramic', 1)", &Params::new()).unwrap();
    let db = &db;
    for sql in [
        "INSERT INTO docs (id, title, text, stock) VALUES ('b', 'blue kettle', 'steel', 2)",
        "UPDATE docs SET title = 'green teapot' WHERE id = 'a'",
        "INSERT INTO docs (id, title, text, stock) VALUES ('a', 'red cup', 'glass', 3) ON CONFLICT (id) DO UPDATE SET title = excluded.title",
    ] {
        let texts = m.texts.load(Ordering::Relaxed);
        m.armed.store(true, Ordering::SeqCst);
        std::thread::scope(|s| {
            let write = s.spawn(move || db.sql(sql, &Params::new()).unwrap());
            started.recv_timeout(Duration::from_secs(30)).expect("the write embeds");
            // Another write commits while the model works.
            let (done_tx, done) = mpsc::channel();
            s.spawn(move || {
                db.sql("UPDATE docs SET stock = stock + 10 WHERE id = 'a'", &Params::new()).unwrap();
                done_tx.send(()).unwrap();
            });
            let other = done.recv_timeout(Duration::from_secs(10));
            go.send(()).unwrap();
            write.join().unwrap();
            assert!(other.is_ok(), "{sql}: another write could not commit while the model worked");
        });
        // One text, embedded ahead; none again under the lock.
        assert_eq!(m.texts.load(Ordering::Relaxed), texts + 1, "{sql}");
    }
    assert_eq!(emb_of(db, "b").unwrap(), want("blue kettle\nsteel"));
    // The row read under the lock has the other writes' stock and keeps its text.
    assert_eq!(emb_of(db, "a").unwrap(), want("red cup\nceramic"));
    let r = db.query("SELECT stock FROM docs WHERE id = 'a'", &Params::new()).unwrap();
    assert_eq!(r.rows[0][0], Value::Int(31));
}

/// A write prepared ahead ([`Db::prepare_write`]) embeds, under the lock,
/// only the rows whose sources changed after it was prepared.
#[test]
fn rows_changed_after_a_write_was_prepared_are_embedded_again() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("e.strata"), schema(), DbOptions::default()).unwrap();
    let m = counting();
    db.set_embedder("emb", m.clone()).unwrap();
    db.sql("INSERT INTO docs (id, title, text, stock) VALUES ('a', 'red mug', 'ceramic', 1), ('b', 'blue kettle', 'steel', 2)", &Params::new()).unwrap();
    let calls = || m.1.load(Ordering::Relaxed);
    let n = calls();
    let p = db.prepare_write("UPDATE docs SET title = 'green teapot' WHERE id IN ('a', 'b')", &Params::new()).unwrap().unwrap();
    assert_eq!(calls(), n + 2);
    db.sql("UPDATE docs SET text = 'porcelain' WHERE id = 'a'", &Params::new()).unwrap();
    assert_eq!(calls(), n + 3);
    assert_eq!(db.write_prepared(&p).unwrap(), 2);
    // Only a's new text, "green teapot\nporcelain", was embedded again.
    assert_eq!(calls(), n + 4);
    assert_eq!(emb_of(&db, "a").unwrap(), want("green teapot\nporcelain"));
    assert_eq!(emb_of(&db, "b").unwrap(), want("green teapot\nsteel"));

    // Numbers set in place embed nothing, ahead or under the lock; a
    // SELECT is not a write.
    let p = db.prepare_write("UPDATE docs SET stock = 5", &Params::new()).unwrap().unwrap();
    assert_eq!(db.write_prepared(&p).unwrap(), 2);
    assert_eq!(calls(), n + 4);
    assert!(db.prepare_write("SELECT id FROM docs", &Params::new()).unwrap().is_none());
    // INSERT OR IGNORE embeds only the rows it writes.
    db.sql("INSERT OR IGNORE INTO docs (id, title, text, stock) VALUES ('a', 'x', 'y', 1), ('c', 'yellow bowl', 'clay', 4)", &Params::new()).unwrap();
    assert_eq!(calls(), n + 5);
    assert_eq!(emb_of(&db, "c").unwrap(), want("yellow bowl\nclay"));
}
