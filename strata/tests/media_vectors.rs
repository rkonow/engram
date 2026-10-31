//! Vectors of images (feature `vision`): a VECTOR field embedding a MEDIA
//! field of images, computed before the writer from the bytes the write
//! gave or read (or, for a value written whole, from where the value
//! says), through JSON rows, loads, SQL INSERT and UPDATE; queries by an
//! image file; and what is refused: an embedding of images and text at
//! once, of a MEDIA field that may hold other files, by a model of text
//! only, and an image that cannot be read or decoded. The `hash:16` model
//! embeds an image by its colours, so equal images get equal vectors and
//! an image's nearest row is its own.
#![cfg(feature = "vision")]

use std::io::Cursor;
use std::sync::Arc;

use base64::Engine;
use serde_json::json;
use strata::db::{Db, SqlResult};
use strata::embed::{embed_file, Embedder, HashEmbedder};
use strata::media::{Allowlist, Media, MediaKind, Resolver};
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

fn photos() -> Schema {
    let emb = Field::new("emb", FieldType::Vector { dim: 16, index: VectorIndexKind::Flat }).embedded("hash:16", &["image"]);
    Schema::new("photos", vec![Field::new("id", FieldType::Keyword), Field::new("title", FieldType::Text { positions: true }), Field::media("image", MediaKind::Image), emb], Some("id")).unwrap()
}

fn db(dir: &std::path::Path) -> Db {
    let db = Db::create(&dir.join("p.strata"), photos(), DbOptions { sync: false, ..DbOptions::default() }).unwrap();
    db.set_embedder("emb", Arc::new(HashEmbedder { dim: 16 })).unwrap();
    db
}

/// A PNG of an 8 by 8 image of one colour.
fn png(rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(8, 8, image::Rgb(rgb)).write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn doc(id: &str, image: serde_json::Value) -> Vec<Value> {
    vec![Value::Str(id.into()), Value::Str(format!("photo {id}")), Value::Str(strata::media::input_text(&image).unwrap()), Value::Null]
}

fn sql(db: &Db, q: &str) -> Vec<Vec<Value>> {
    match db.sql(q, &Params::new()).unwrap_or_else(|e| panic!("{q}: {e}")) {
        SqlResult::Rows(r) => r.rows,
        SqlResult::Written(_) => Vec::new(),
    }
}

/// Each row's id and vector, by id.
fn vectors(db: &Db) -> Vec<(String, Vec<f32>)> {
    let mut v: Vec<_> = sql(db, "SELECT id, emb FROM photos LIMIT 100").into_iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_vector().map(<[f32]>::to_vec).unwrap_or_default())).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// The ids of the rows nearest to an image file, nearest first.
fn nearest(db: &Db, file: &[u8]) -> Vec<String> {
    let q = embed_file(db.embedder("emb").unwrap().as_ref(), file).unwrap();
    let r = db.query("SELECT id FROM photos ORDER BY similarity(emb, :q) DESC LIMIT 3", &Params::new().set("q", Value::Vector(q.into()))).unwrap();
    r.rows.into_iter().map(|r| r[0].as_str().unwrap().to_string()).collect()
}

#[test]
fn rows_get_the_vectors_of_their_images() {
    let dir = tempfile::tempdir().unwrap();
    let (root, store) = (dir.path().join("files"), dir.path().join("store"));
    std::fs::create_dir_all(&root).unwrap();
    let (red, green, blue) = (png([250, 10, 10]), png([10, 250, 10]), png([10, 10, 250]));
    std::fs::write(root.join("green.png"), &green).unwrap();
    let db = db(dir.path());
    db.set_media_resolver(Resolver { allow: Allowlist::parse(&[root.display().to_string()]).unwrap(), store: Some(store.clone()), ..Resolver::default() });

    // Bytes with a URI, bytes kept in the store, a file, through fill_in.
    let mut docs = vec![
        doc("r", json!({"uri": "red.png", "data": b64(&red)})),
        doc("b", json!({"data": b64(&blue)})),
        doc("g", json!(format!("file:{}", root.join("green.png").display()))),
    ];
    db.fill_in("photos", &mut docs).unwrap();
    db.load(docs).unwrap();
    // A value written whole is embedded from where it says: the store.
    let whole = Media::parse(sql(&db, "SELECT image FROM photos WHERE id = 'b'")[0][0].as_str().unwrap()).unwrap();
    let mut again = vec![doc("b2", serde_json::from_str(&whole.to_text()).unwrap())];
    db.fill_in("photos", &mut again).unwrap();
    db.load(again).unwrap();
    // SQL: an INSERT by URI and by bytes.
    let green_uri = format!("file:{}", root.join("green.png").display());
    sql(&db, &format!("INSERT INTO photos (id, title, image) VALUES ('g2', 'again', '{green_uri}')"));
    sql(&db, &format!("INSERT INTO photos (id, title, image) VALUES ('r2', 'again', '{}')", json!({"data": b64(&red)})));

    let v = vectors(&db);
    assert_eq!(v.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), ["b", "b2", "g", "g2", "r", "r2"]);
    let of = |id: &str| v.iter().find(|r| r.0 == id).unwrap().1.clone();
    assert!(of("r").len() == 16 && of("g").len() == 16 && of("b").len() == 16);
    assert_eq!((of("b"), of("g"), of("r")), (of("b2"), of("g2"), of("r2")), "the same image, the same vector");
    assert_ne!(of("r"), of("g"));
    assert_eq!(nearest(&db, &green)[..2].iter().map(String::as_str).collect::<Vec<_>>(), ["g", "g2"]);

    // An UPDATE that changes the image recomputes the vector.
    sql(&db, &format!("UPDATE photos SET image = '{green_uri}' WHERE id = 'r'"));
    assert_eq!(vectors(&db).iter().find(|r| r.0 == "r").unwrap().1, of("g"));
    // A vector the write gives is kept as given.
    let mine = vec![1.0f32; 16];
    let mut given = vec![doc("m", json!({"uri": "m.png", "data": b64(&red)}))];
    given[0][3] = Value::Vector(mine.clone().into());
    db.fill_in("photos", &mut given).unwrap();
    db.load(given).unwrap();
    assert_eq!(vectors(&db).iter().find(|r| r.0 == "m").unwrap().1, mine);
}

#[test]
fn images_that_cannot_be_read_or_decoded_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    // Its bytes are neither held nor anywhere the database may read.
    let gone = Media { uri: "https://cdn.example.com/a.png".into(), sha256: "ab".repeat(32), mime: "image/png".into(), bytes: 10, width: Some(8), height: Some(8), ..Media::default() };
    let mut docs = vec![doc("a", json!({"uri": "ok.png", "data": b64(&png([1, 2, 3]))})), doc("x", serde_json::from_str(&gone.to_text()).unwrap())];
    let e = db.fill_in("photos", &mut docs).unwrap_err().to_string();
    assert!(e.contains("field image") && e.contains("not on the media fetch list"), "{e}");
    let e = db.sql(&format!("INSERT INTO photos (id, image) VALUES ('x', '{}')", gone.to_text()), &Params::new()).unwrap_err().to_string();
    assert!(e.contains("not on the media fetch list"), "{e}");
    // A PNG header over bytes that are not one.
    let mut bad = png([1, 2, 3]);
    bad.truncate(40);
    bad.extend([7u8; 40]);
    let mut docs = vec![doc("x", json!({"uri": "bad.png", "data": b64(&bad)}))];
    let e = db.fill_in("photos", &mut docs).unwrap_err().to_string();
    assert!(e.contains("bad.png") && e.contains("could not be decoded"), "{e}");
    assert!(embed_file(&HashEmbedder { dim: 16 }, b"not an image").is_err());
    assert_eq!(sql(&db, "SELECT COUNT(*) FROM photos")[0][0], Value::Int(0));
}

/// A model of text only.
struct Words;

impl Embedder for Words {
    fn name(&self) -> String {
        "words".into()
    }
    fn dim(&self) -> usize {
        16
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        HashEmbedder { dim: 16 }.embed(texts)
    }
}

#[test]
fn what_an_embedding_of_images_takes() {
    let field = |from: &[&str]| Field::new("emb", FieldType::Vector { dim: 16, index: VectorIndexKind::Flat }).embedded("hash:16", from);
    let base = || vec![Field::new("id", FieldType::Keyword), Field::new("title", FieldType::Text { positions: true })];
    let with = |media: Field, emb: Field| Schema::new("t", [base(), vec![media, emb]].concat(), Some("id"));
    let e = with(Field::media("image", MediaKind::Image), field(&["image", "title"])).unwrap_err().to_string();
    assert!(e.contains("one MEDIA field (image), or text fields, not both"), "{e}");
    let e = with(Field::media("file", MediaKind::Any), field(&["file"])).unwrap_err().to_string();
    assert!(e.contains("is of kind image"), "{e}");
    let e = with(Field::media("doc", MediaKind::Pdf), field(&["doc"])).unwrap_err().to_string();
    assert!(e.contains("is of kind image"), "{e}");
    // A MEDIA field is still not the source of a generated text field.
    let gen = Field::new("all", FieldType::Text { positions: true }).not_stored().generated(&["title", "image"]);
    assert!(with(Field::media("image", MediaKind::Image), gen).is_err());

    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    let e = db.set_embedder("emb", Arc::new(Words)).unwrap_err().to_string();
    assert!(e.contains("does not embed images") && e.contains("images of image"), "{e}");
    let e = Words.embed_inputs(&[strata::embed::Input::Image(&strata::media::decode_image(&png([0, 0, 0])).unwrap())]).unwrap_err().to_string();
    assert!(e.contains("does not embed images"), "{e}");
    // SQL declares it as JSON does.
    sql(&db, "CREATE TABLE clips (id KEYWORD PRIMARY KEY, still MEDIA OPTIONS(kind = 'image'), v VECTOR(16) OPTIONS(embed = 'still', model = 'hash:16'))");
    let clips = db.storage().table("clips").unwrap();
    assert_eq!(clips.embeds_media(2), Some(1));
}
