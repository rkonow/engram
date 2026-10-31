//! A store written by the Strata from before field ids, which this Strata
//! refuses to open, is copied into the current format the first time
//! `Store::open` opens it, the old file kept beside it. The fixture
//! (`tests/fixtures/store-before-field-ids`) was written at 6854657 by the
//! program in its README: 12 turns of three sessions, each a prompt, an
//! answer naming a file and a decision about an entity, the first eight
//! flushed and the rest left in the WAL.

use std::path::Path;
use std::sync::Arc;

use engram::embed::HashEmbedder;
use engram::record::{Draft, Kind};
use engram::store::{Store, StoreOptions};
use strata::db::Db;
use strata::query::Params;
use strata::storage::DbOptions;
use strata::value::Value;

/// Every row of the store and its vector, in order of `seq`.
fn rows(db: &Db) -> Vec<Vec<Value>> {
    let mut rows = db.query("SELECT * FROM memory ORDER BY seq LIMIT 100000", &Params::new()).unwrap().rows;
    let embs = db.query("SELECT emb FROM memory ORDER BY seq LIMIT 100000", &Params::new()).unwrap().rows;
    for (r, e) in rows.iter_mut().zip(embs) {
        r.extend(e);
    }
    rows
}

#[test]
fn a_store_from_before_field_ids_is_copied_on_first_open() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store-before-field-ids");
    for name in ["m.strata", "m.strata-wal"] {
        std::fs::copy(fixture.join(name), dir.path().join(name)).unwrap();
    }
    let path = dir.path().join("m.strata");
    assert!(strata::upgrade::needs_upgrade(&path).unwrap());
    let before = {
        let old = Db::open(&path, DbOptions { open_old: true, ..DbOptions::default() }).unwrap();
        rows(&old)
    };
    assert!(before.len() >= 36, "{}", before.len());

    let open = || Store::open(&path, Arc::new(HashEmbedder { dim: 32 }), StoreOptions { sync: false, ..Default::default() }).unwrap();
    let s = open();
    assert!(!strata::upgrade::needs_upgrade(&path).unwrap());
    assert!(strata::upgrade::old_path(&path).exists());
    assert_eq!(s.len(), before.len() as u64);
    assert_eq!(rows(&s.db), before);
    // Its words and its vectors are searched as before.
    let hits = s.query("SELECT text FROM memory WHERE MATCH(body, 'semicolons') AND kind = 'decision' ORDER BY seq LIMIT 100").unwrap().rows;
    assert_eq!(hits.len(), 4, "{hits:?}");
    // And its links: the decisions about an entity are one hop in.
    let about = s.query("SELECT text FROM memory WHERE REACHABLE(links, 'e:export', hops => 1, dir => 'in') AND kind = 'decision' ORDER BY seq LIMIT 100").unwrap().rows;
    assert_eq!(about, hits);
    let v = s.embedder.embed_query("Deploys happen on Tuesdays only (decided in turn 5)").unwrap();
    let near = s.query_with("SELECT text FROM memory ORDER BY similarity(emb, :v) DESC LIMIT 1", &Params::new().set("v", Value::Vector(v.into()))).unwrap().rows;
    assert_eq!(near, [vec![Value::Str("Deploys happen on Tuesdays only (decided in turn 5)".into())]]);
    // It takes writes, and is not copied again.
    s.append("s4", 12, 1_790_813_520, vec![Draft::new(Kind::Note, "a note written after the upgrade")]).unwrap();
    s.flush().unwrap();
    let n = s.len();
    drop(s);
    let s = open();
    assert_eq!(s.len(), n);
    assert_eq!(s.query("SELECT COUNT(*) FROM memory WHERE MATCH(body, 'upgrade')").unwrap().rows[0][0], Value::Int(1));
}
