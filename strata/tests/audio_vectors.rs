//! Vectors of sound (feature `audio`): an embedded VECTOR field of a MEDIA
//! field of kind audio, whose files are decoded as for text made from
//! sound (tests/derived_audio.rs), and queries by a sound file and by
//! text. The test model puts a tone on a quarter circle by its pitch, and
//! the text "tone 440" where a 440 Hz tone is.
#![cfg(feature = "audio")]

use std::sync::Arc;

use base64::Engine;
use serde_json::json;
use strata::db::{Db, SqlResult};
use strata::embed::{Audio, Embedder, Input, Modality};
use strata::media::MediaKind;
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::DbOptions;
use strata::value::Value;
use strata::VectorIndexKind;

/// A 16-bit PCM WAV file of one channel of `seconds` of a tone of `hz` at
/// `rate`.
fn wav(hz: f32, rate: u32, seconds: f32) -> Vec<u8> {
    let pcm: Vec<u8> = (0..(rate as f32 * seconds) as u32).flat_map(|i| (((2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5 * 32767.0) as i16).to_le_bytes()).collect();
    let mut out = Vec::new();
    out.extend(b"RIFF");
    out.extend((36 + pcm.len() as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((pcm.len() as u32).to_le_bytes());
    out.extend(pcm);
    out
}

fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// The pitch of a tone, to the nearest 10 Hz, from how often it crosses
/// zero.
fn pitch(a: &Audio) -> f32 {
    let crossings = a.samples.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
    (crossings as f32 / 2.0 / (a.samples.len() as f32 / a.rate as f32) / 10.0).round() * 10.0
}

/// A pitch from 0 to 2 kHz as a point on a quarter circle.
fn at(hz: f32) -> Vec<f32> {
    let a = hz / 2000.0 * std::f32::consts::FRAC_PI_2;
    vec![a.cos(), a.sin()]
}

/// A model of sound and text: a tone by its pitch, "tone 440" as 440 Hz.
struct Tones(&'static [Modality]);

impl Embedder for Tones {
    fn name(&self) -> String {
        "tones".into()
    }
    fn dim(&self) -> usize {
        2
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| at(t.trim_start_matches("tone ").parse().unwrap_or(0.0))).collect())
    }
    fn modalities(&self) -> &'static [Modality] {
        self.0
    }
    fn embed_inputs(&self, inputs: &[Input<'_>]) -> strata::Result<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|i| match i {
                Input::Audio(a) => at(pitch(a)),
                Input::Text(t) => self.embed(&[t]).unwrap().remove(0),
                Input::Image(_) => panic!("tones hears sound and reads text"),
            })
            .collect())
    }
}

const SOUND: &[Modality] = &[Modality::Text, Modality::Audio];
const TEXT: &[Modality] = &[Modality::Text];
const IMAGES: &[Modality] = &[Modality::Text, Modality::Image];

fn schema() -> Schema {
    let fields = vec![
        Field::new("id", FieldType::Keyword),
        Field::media("sound", MediaKind::Audio),
        Field::new("emb", FieldType::Vector { dim: 2, index: VectorIndexKind::Flat }).embedded("tones", &["sound"]),
    ];
    Schema::new("clips", fields, Some("id")).unwrap()
}

fn file(hz: f32, rate: u32) -> String {
    json!({"data": b64(&wav(hz, rate, 0.5)), "uri": format!("mem:{hz}@{rate}")}).to_string()
}

fn nearest(db: &Db, q: Vec<f32>) -> Vec<String> {
    let r = db.query("SELECT id FROM clips ORDER BY similarity(emb, :q) DESC LIMIT 2", &Params::new().set("q", Value::Vector(q.into()))).unwrap();
    r.rows.iter().map(|r| r[0].as_str().unwrap().to_string()).collect()
}

#[test]
fn rows_get_the_vectors_of_their_sounds() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("c.strata"), schema(), DbOptions { sync: false, ..DbOptions::default() }).unwrap();
    db.set_embedder_in("clips", "emb", Arc::new(Tones(SOUND))).unwrap();
    // More sounds than a batch, at 8 to 96 kHz.
    let tones = [(220.0, 16_000), (440.0, 44_100), (880.0, 96_000), (1320.0, 22_050), (330.0, 8_000), (1760.0, 48_000)];
    let value = |f: &str| strata::media::input_text(&serde_json::from_str(f).unwrap()).unwrap();
    db.load(tones.iter().enumerate().map(|(i, (hz, rate))| vec![Value::Str(format!("c{i}")), Value::Str(value(&file(*hz, *rate))), Value::Null]).collect::<Vec<_>>()).unwrap();
    match db.sql(&format!("INSERT INTO clips (id, sound) VALUES ('s', '{}')", file(1000.0, 44_100)), &Params::new()).unwrap() {
        SqlResult::Written(n) => assert_eq!(n, 1),
        SqlResult::Rows(_) => panic!("an insert writes"),
    }
    let r = db.query("SELECT id, emb FROM clips LIMIT 10", &Params::new()).unwrap();
    for row in &r.rows {
        let hz = match row[0].as_str().unwrap() {
            "s" => 1000.0,
            id => tones[id[1..].parse::<usize>().unwrap()].0,
        };
        let v = row[1].as_vector().unwrap();
        assert!(v.iter().zip(at(hz)).all(|(x, y)| (x - y).abs() < 1e-3), "{row:?}");
    }
    // A query by a sound file, at its own rate, and by text.
    let e = db.embedder_in("clips", "emb").unwrap();
    let (m, data) = strata::embed::query_file("audio", &b64(&wav(870.0, 11_025, 0.5))).unwrap();
    assert_eq!(m, Modality::Audio);
    assert_eq!(nearest(&db, strata::embed::embed_file(e.as_ref(), &data).unwrap()), ["c2", "s"]);
    assert_eq!(nearest(&db, e.embed_query("tone 1300").unwrap()), ["c3", "s"]);
    // A file given as the other kind than it is.
    let png = b"\x89PNG\r\n\x1a\n not really";
    let e = strata::embed::query_file("image", &b64(&data)).unwrap_err().to_string();
    assert!(e.contains("is a sound file"), "{e}");
    let e = strata::embed::query_file("audio", &b64(png)).unwrap_err().to_string();
    assert!(e.contains("not a sound file"), "{e}");
    let e = strata::embed::query_file("video", &b64(png)).unwrap_err().to_string();
    assert!(e.contains("\"video\""), "{e}");
}

#[test]
fn a_model_that_does_not_hear_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("c.strata"), schema(), DbOptions::default()).unwrap();
    for m in [TEXT, IMAGES] {
        let e = db.set_embedder_in("clips", "emb", Arc::new(Tones(m))).unwrap_err().to_string();
        assert!(e.contains("does not embed sound"), "{e}");
    }
}

#[test]
fn an_embedded_field_embeds_media_of_kind_image_or_audio() {
    for kind in [MediaKind::Any, MediaKind::Video, MediaKind::Pdf] {
        let fields = vec![
            Field::new("id", FieldType::Keyword),
            Field::media("file", kind),
            Field::new("emb", FieldType::Vector { dim: 2, index: VectorIndexKind::Flat }).embedded("tones", &["file"]),
        ];
        let e = Schema::new("clips", fields, Some("id")).unwrap_err().to_string();
        assert!(e.contains("of kind image or audio"), "{e}");
    }
}
