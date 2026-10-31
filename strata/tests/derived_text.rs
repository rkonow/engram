//! Text made from media (feature `vision`): a TEXT field derived from a
//! MEDIA field of images by a model, made before the writer through JSON
//! rows, loads and SQL INSERT and UPDATE, kept when a write gives it, made
//! again when the file changes, and read by BM25, generated fields and
//! embedded vectors as any text is; and what is refused. The `colors`
//! model names an image's colours, so a red image's text is "red".
#![cfg(feature = "vision")]

use std::io::Cursor;
use std::sync::Arc;

use base64::Engine;
use serde_json::json;
use strata::db::{Db, SqlResult};
use strata::derive::{ColorNamer, Deriver};
use strata::embed::{Embedder, HashEmbedder, Input, Modality};
use strata::media::{Allowlist, MediaKind, Resolver};
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

fn photos(model: &str) -> Schema {
    let text = || FieldType::Text { positions: true };
    let fields = vec![
        Field::new("id", FieldType::Keyword),
        Field::new("title", text()),
        Field::media("image", MediaKind::Image),
        Field::new("caption", text()).derived(model, "image"),
        Field::new("about", text()).not_stored().generated(&["title", "caption"]),
        Field::new("emb", FieldType::Vector { dim: 16, index: VectorIndexKind::Flat }).embedded("hash:16", &["caption"]),
    ];
    Schema::new("photos", fields, Some("id")).unwrap()
}

fn opts() -> DbOptions {
    DbOptions { sync: false, ..DbOptions::default() }
}

fn db(path: &std::path::Path) -> Db {
    let db = Db::create(path, photos("colors"), opts()).unwrap();
    db.set_deriver_in("photos", "caption", Arc::new(ColorNamer)).unwrap();
    db.set_embedder("emb", Arc::new(HashEmbedder { dim: 16 })).unwrap();
    db
}

/// A PNG of an 8 by 8 image, its left half `left` and its right half
/// `right`.
fn png(left: [u8; 3], right: [u8; 3]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbImage::from_fn(8, 8, |x, _| image::Rgb(if x < 4 { left } else { right })).write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

const RED: [u8; 3] = [250, 10, 10];
const GREEN: [u8; 3] = [10, 250, 10];
const BLUE: [u8; 3] = [10, 10, 250];

fn file(rgb: [u8; 3]) -> String {
    json!({"data": base64::engine::general_purpose::STANDARD.encode(png(rgb, rgb)), "uri": format!("mem:{rgb:?}")}).to_string()
}

fn doc(id: &str, image: &str) -> Vec<Value> {
    vec![Value::Str(id.into()), Value::Str(format!("photo {id}")), Value::Str(strata::media::input_text(&serde_json::from_str(image).unwrap()).unwrap()), Value::Null, Value::Null, Value::Null]
}

fn sql(db: &Db, q: &str) -> Vec<Vec<Value>> {
    match db.sql(q, &Params::new()).unwrap_or_else(|e| panic!("{q}: {e}")) {
        SqlResult::Rows(r) => r.rows,
        SqlResult::Written(_) => Vec::new(),
    }
}

/// Each row's id, caption and vector, by id.
fn rows(db: &Db) -> Vec<(String, Option<String>, Vec<f32>)> {
    let mut v: Vec<_> = sql(db, "SELECT id, caption, emb FROM photos LIMIT 100")
        .into_iter()
        .map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_str().map(String::from), r[2].as_vector().map(<[f32]>::to_vec).unwrap_or_default()))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn caption(db: &Db, id: &str) -> Option<String> {
    rows(db).into_iter().find(|r| r.0 == id).and_then(|r| r.1)
}

fn ids(db: &Db, q: &str) -> Vec<String> {
    let mut v: Vec<String> = sql(db, q).into_iter().map(|r| r[0].as_str().unwrap().to_string()).collect();
    v.sort();
    v
}

fn vector_of(text: &str) -> Vec<f32> {
    HashEmbedder { dim: 16 }.embed(&[text]).unwrap().pop().unwrap()
}

#[test]
fn rows_get_the_text_of_their_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.strata");
    let db = db(&path);
    // JSON rows (what a server does before the writer), a load, SQL.
    let mut docs = vec![doc("r", &file(RED)), doc("b", &file(BLUE))];
    db.fill_in("photos", &mut docs).unwrap();
    assert_eq!(docs[0][3], Value::Str("red".into()), "made before the writer");
    db.load(docs).unwrap();
    db.load(vec![doc("g", &file(GREEN))]).unwrap();
    sql(&db, &format!("INSERT INTO photos (id, title, image) VALUES ('g2', 'again', '{}')", file(GREEN)));
    let two = json!({"data": base64::engine::general_purpose::STANDARD.encode(png(RED, BLUE)), "uri": "mem:two"}).to_string();
    sql(&db, &format!("INSERT INTO photos (id, title, image) VALUES ('rb', 'two', '{two}')"));
    let got: Vec<(String, Option<String>)> = rows(&db).into_iter().map(|r| (r.0, r.1)).collect();
    let want = [("b", "blue"), ("g", "green"), ("g2", "green"), ("r", "red"), ("rb", "red blue")];
    assert_eq!(got, want.map(|(i, c)| (i.to_string(), Some(c.to_string()))));
    // BM25, a generated field and an embedded vector read the text.
    assert_eq!(ids(&db, "SELECT id FROM photos WHERE MATCH(caption, 'green')"), ["g", "g2"]);
    assert_eq!(ids(&db, "SELECT id FROM photos WHERE MATCH(about, 'blue')"), ["b", "rb"]);
    assert_eq!(ids(&db, "SELECT id FROM photos WHERE MATCH(about, 'again')"), ["g2"]);
    assert!(rows(&db).iter().all(|r| r.2 == vector_of(r.1.as_deref().unwrap())));
    let top = sql(&db, "SELECT id FROM photos ORDER BY bm25(caption, 'red') DESC LIMIT 1");
    assert_eq!(top[0][0], Value::Str("r".into()));

    // A text the write gives is kept as given, and embedded.
    let mut given = vec![doc("m", &file(RED))];
    given[0][3] = Value::Str("my own words".into());
    db.fill_in("photos", &mut given).unwrap();
    db.load(given).unwrap();
    sql(&db, &format!("INSERT INTO photos (id, image, caption) VALUES ('m2', '{}', 'said by the writer')", file(BLUE)));
    assert_eq!((caption(&db, "m").as_deref(), caption(&db, "m2").as_deref()), (Some("my own words"), Some("said by the writer")));
    assert_eq!(rows(&db).into_iter().find(|r| r.0 == "m").unwrap().2, vector_of("my own words"));
    // A row without a file gets no text.
    sql(&db, "INSERT INTO photos (id, title) VALUES ('none', 'no file')");
    assert_eq!(caption(&db, "none"), None);

    // An UPDATE of the file makes the text and its vector again; one that
    // sets the text keeps it; one of another field leaves both.
    sql(&db, &format!("UPDATE photos SET image = '{}' WHERE id = 'r'", file(GREEN)));
    assert_eq!(caption(&db, "r").as_deref(), Some("green"));
    assert_eq!(rows(&db).into_iter().find(|r| r.0 == "r").unwrap().2, vector_of("green"));
    assert_eq!(ids(&db, "SELECT id FROM photos WHERE MATCH(about, 'green')"), ["g", "g2", "r"]);
    sql(&db, &format!("UPDATE photos SET image = '{}', caption = 'chosen' WHERE id = 'b'", file(RED)));
    assert_eq!(caption(&db, "b").as_deref(), Some("chosen"));
    sql(&db, "UPDATE photos SET title = 'renamed' WHERE id = 'g'");
    assert_eq!(caption(&db, "g").as_deref(), Some("green"));
    // ON CONFLICT DO UPDATE of the file makes it again too.
    sql(&db, &format!("INSERT INTO photos (id, image) VALUES ('g2', '{}') ON CONFLICT (id) DO UPDATE SET image = excluded.image", file(BLUE)));
    assert_eq!(caption(&db, "g2").as_deref(), Some("blue"));

    // Stored: a database opened without the model reads it, and merges
    // keep it.
    let before = rows(&db);
    db.storage().merge(&strata::storage::MergePolicy::default()).unwrap();
    drop(db);
    let db = Db::open(&path, opts()).unwrap();
    assert_eq!(rows(&db), before);
    assert_eq!(ids(&db, "SELECT id FROM photos WHERE MATCH(caption, 'blue')"), ["g2", "rb"]);
    // Without the model, a write that leaves the text out stores none.
    db.load(vec![doc("later", &file(RED))]).unwrap();
    assert_eq!(caption(&db, "later"), None);
    // The models directory opens it by name.
    assert!(db.open_embedders(dir.path()).is_empty());
    sql(&db, &format!("INSERT INTO photos (id, image) VALUES ('later2', '{}')", file(RED)));
    assert_eq!(caption(&db, "later2").as_deref(), Some("red"));
}

/// Shouts: a second model, to tell which made a text.
struct Shout;

impl Deriver for Shout {
    fn name(&self) -> String {
        "shout".into()
    }
    fn modality(&self) -> Modality {
        Modality::Image
    }
    fn derive(&self, inputs: &[Input<'_>]) -> strata::Result<Vec<String>> {
        Ok(ColorNamer.derive(inputs)?.into_iter().map(|t| t.to_uppercase()).collect())
    }
}

#[test]
fn another_model_makes_the_text_again_when_rows_are_copied() {
    let dir = tempfile::tempdir().unwrap();
    // Files the databases may read again.
    let resolver = || Resolver { allow: Allowlist::parse(&[dir.path().display().to_string()]).unwrap(), ..Resolver::default() };
    let at = |name: &str, rgb: [u8; 3]| {
        std::fs::write(dir.path().join(name), png(rgb, rgb)).unwrap();
        json!(format!("file:{}", dir.path().join(name).display())).to_string()
    };
    let old = db(&dir.path().join("old.strata"));
    old.set_media_resolver(resolver());
    old.load(vec![doc("r", &at("r.png", RED)), doc("b", &at("b.png", BLUE))]).unwrap();
    // Same model: the text is copied, not made again (none is set here).
    let same = Db::create(&dir.path().join("same.strata"), photos("colors"), opts()).unwrap();
    same.load_from(&old).unwrap();
    assert_eq!(caption(&same, "r").as_deref(), Some("red"));
    // Another model: made anew, from the files.
    let new = Db::create(&dir.path().join("new.strata"), photos("shout"), opts()).unwrap();
    new.set_media_resolver(resolver());
    new.set_deriver_in("photos", "caption", Arc::new(Shout)).unwrap();
    new.load_from(&old).unwrap();
    assert_eq!((caption(&new, "r").as_deref(), caption(&new, "b").as_deref()), (Some("RED"), Some("BLUE")));
}

#[test]
fn what_a_derived_field_takes() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(&dir.path().join("p.strata"));
    // An image that cannot be decoded fails the write whole.
    let mut bad = png(RED, RED);
    bad.truncate(40);
    bad.extend([7u8; 40]);
    let bad = json!({"data": base64::engine::general_purpose::STANDARD.encode(&bad), "uri": "bad.png"}).to_string();
    let mut docs = vec![doc("ok", &file(RED)), doc("x", &bad)];
    let e = db.fill_in("photos", &mut docs).unwrap_err().to_string();
    assert!(e.contains("field image") && e.contains("bad.png") && e.contains("could not be decoded"), "{e}");
    let e = db.sql(&format!("INSERT INTO photos (id, image) VALUES ('ok', '{}'), ('x', '{bad}')", file(RED)), &Params::new()).unwrap_err().to_string();
    assert!(e.contains("could not be decoded"), "{e}");
    assert_eq!(sql(&db, "SELECT COUNT(*) FROM photos")[0][0], Value::Int(0));

    // A model of images for a field of any file, and fields that are not
    // derived.
    sql(&db, "CREATE TABLE files (id KEYWORD PRIMARY KEY, f MEDIA, words TEXT OPTIONS(derive = 'f', model = 'colors'))");
    let e = db.set_deriver_in("files", "words", Arc::new(ColorNamer)).unwrap_err().to_string();
    assert!(e.contains("makes text of images") && e.contains("kind = 'image'"), "{e}");
    let e = db.set_deriver_in("photos", "title", Arc::new(ColorNamer)).unwrap_err().to_string();
    assert!(e.contains("not a derived TEXT field"), "{e}");
    assert!(db.open_embedders(dir.path()).iter().any(|(f, e)| f == "files.words" && e.to_string().contains("kind = 'image'")));
    // Models by name.
    let e = strata::derive::open("no-such-model", dir.path()).err().unwrap().to_string();
    assert!(e.contains("no-such-model") && e.contains("fetch it first"), "{e}");
    std::fs::create_dir_all(dir.path().join("bert")).unwrap();
    std::fs::write(dir.path().join("bert/config.json"), r#"{"model_type": "bert"}"#).unwrap();
    let e = strata::derive::open("bert", dir.path()).err().unwrap().to_string();
    assert!(e.contains("does not make text of media"), "{e}");
    let e = ColorNamer.derive(&[Input::Text("words")]).unwrap_err().to_string();
    assert!(e.contains("makes text of images"), "{e}");
    // The table needs the manifest feature.
    assert!(db.schema().features().any(|f| f == "derive"));
}
