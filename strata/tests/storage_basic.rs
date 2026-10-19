//! Storage engine: commits, recovery, deletes, snapshots, freelist, merges.

mod storage_util;

use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use storage_util::*;
use strata::storage::pager::{Meta, PAGE};
use strata::storage::{Database, DbOptions, MergePolicy, Snapshot};
use strata::{Error, Field, FieldType, Schema, Value};

fn keys(r: std::ops::Range<u64>) -> BTreeSet<String> {
    r.map(key).collect()
}

fn image_crcs(s: &Snapshot) -> Vec<u32> {
    s.segments().iter().map(|v| crc32fast::hash(v.file.image())).collect()
}

#[test]
fn types_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Database>();
    check::<Snapshot>();
}

#[test]
fn commit_reopen_and_flush_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let want = keys(0..100);
    {
        let db = Database::create(&path, schema(), opts(&src)).unwrap();
        assert!(matches!(Database::create(&path, schema(), opts(&src)), Err(Error::Io(_))));
        assert_eq!(db.snapshot().n_live_docs(), 0);
        for t in 0..10 {
            let mut tx = db.begin().unwrap();
            for i in t * 10..t * 10 + 10 {
                tx.insert(doc(i, 0)).unwrap();
            }
            tx.commit().unwrap();
        }
        let s = db.snapshot();
        assert_keys(&s, &want);
        assert_eq!(s.segments().len(), 1);
        assert!(s.segments()[0].in_memory && s.segments()[0].id >> 63 == 1);
        // Cached until the next write.
        let again = db.snapshot();
        assert_eq!((again.txn(), again.segments()[0].id), (s.txn(), s.segments()[0].id));
        let mut tx = db.begin().unwrap();
        tx.insert(doc(100, 0)).unwrap();
        tx.rollback();
        assert_eq!(db.snapshot().segments()[0].id, s.segments()[0].id);
    }
    {
        // Recovery replays the WAL into the memtable.
        let db = Database::open(&path, opts(&src)).unwrap();
        let s = db.snapshot();
        assert_keys(&s, &want);
        assert!(s.txn() >= 11);
        db.flush().unwrap();
        let s = db.snapshot();
        assert_eq!(s.segments().len(), 1);
        assert!(!s.segments()[0].in_memory);
        assert_keys(&s, &want);
        assert_eq!(db.file_stats().wal_bytes, 16);
        db.check().unwrap();
    }
    let db = Database::open(&path, opts(&src)).unwrap();
    assert_keys(&db.snapshot(), &want);
    assert_eq!(db.file_stats().segments.len(), 1);
    db.check().unwrap();
}

/// The memtable is split into chunks; a write rebuilds only the last
/// chunk's in-memory segment, and deletes only change live filters.
#[test]
fn memtable_chunks_are_built_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Database::create(&path, schema(), DbOptions { memtable_chunk_docs: 50, ..opts(&src) }).unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..300 {
        tx.insert(doc(i, 0)).unwrap();
    }
    tx.commit().unwrap();
    // Chunks hold 64 documents (rounded up to a word of the delete bitset).
    let s1 = db.snapshot();
    let ids = |s: &Snapshot| s.segments().iter().map(|v| v.id).collect::<Vec<_>>();
    assert_eq!(s1.segments().iter().map(|v| v.file.n_docs()).collect::<Vec<_>>(), vec![64, 64, 64, 64, 44]);
    assert!(s1.segments().iter().all(|v| v.in_memory));
    assert_keys(&s1, &keys(0..300));

    let mut tx = db.begin().unwrap();
    tx.insert(doc(300, 0)).unwrap();
    assert!(tx.delete(&Value::Str(key(10))).unwrap());
    tx.commit().unwrap();
    let s2 = db.snapshot();
    let (a, b) = (ids(&s1), ids(&s2));
    assert_eq!(a[..4], b[..4], "full chunks are not rebuilt");
    assert_ne!(a[4], b[4]);
    assert_eq!(s2.segments()[0].n_live, 63);
    let mut want = keys(0..301);
    want.remove(&key(10));
    assert_keys(&s2, &want);
    // Deleting every document of a chunk drops its segment.
    let mut tx = db.begin().unwrap();
    for i in 64..128 {
        tx.delete(&Value::Str(key(i))).unwrap();
        want.remove(&key(i));
    }
    tx.commit().unwrap();
    let s3 = db.snapshot();
    assert_eq!(s3.segments().len(), 4);
    assert_keys(&s3, &want);
    // The old snapshots are unaffected.
    assert_keys(&s1, &keys(0..300));
    db.flush().unwrap();
    assert_keys(&db.snapshot(), &want);
    assert_eq!(db.snapshot().segments().len(), 1);
}

#[test]
fn wal_replay_ignores_torn_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let o = DbOptions { sync: true, ..opts(&src) };
    {
        let db = Database::create(&path, schema(), o.clone()).unwrap();
        for t in 0..3 {
            let mut tx = db.begin().unwrap();
            for i in t * 5..t * 5 + 5 {
                tx.insert(doc(i, 0)).unwrap();
            }
            tx.commit().unwrap();
        }
    }
    let mut wal = OpenOptions::new().append(true).open(dir.path().join("db.strata-wal")).unwrap();
    wal.write_all(&[40, 0, 0, 0, 1, 2, 3, 4, 1, 9, 9]).unwrap();
    drop(wal);
    {
        let db = Database::open(&path, o.clone()).unwrap();
        assert_keys(&db.snapshot(), &keys(0..15));
        let mut tx = db.begin().unwrap();
        tx.insert(doc(15, 0)).unwrap();
        tx.commit().unwrap();
    }
    let db = Database::open(&path, o).unwrap();
    assert_keys(&db.snapshot(), &keys(0..16));
}

#[test]
fn deletes_and_upserts_visible_and_durable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let o = DbOptions { segment_max_docs: 400, ..opts(&src) };
    let mut want = keys(100..1050);
    {
        let db = Database::create(&path, schema(), o.clone()).unwrap();
        db.bulk_load((0..1000).map(|i| doc(i, 0))).unwrap();
        assert_eq!(db.snapshot().segments().len(), 3);
        let mut tx = db.begin().unwrap();
        for i in 0..100 {
            assert!(tx.delete(&Value::Str(key(i))).unwrap());
        }
        assert!(!tx.delete(&Value::Str(key(5))).unwrap(), "already deleted in this txn");
        assert!(!tx.delete(&Value::Str(key(5000))).unwrap());
        assert!(matches!(tx.delete(&Value::Int(5)), Err(Error::Schema(_))));
        for i in 100..200 {
            tx.upsert(doc(i, 1)).unwrap();
        }
        for i in 1000..1050 {
            tx.insert(doc(i, 0)).unwrap();
        }
        assert!(matches!(tx.insert(doc(150, 2)), Err(Error::Schema(_))));
        tx.commit().unwrap();
        let s = db.snapshot();
        assert_eq!(s.segments().len(), 4);
        assert_keys(&s, &want);
        assert!(s.lookup(&Value::Str(key(50))).is_none());
        // The upserted key lives in the memtable segment now.
        assert_eq!(s.lookup(&Value::Str(key(150))).unwrap().0, 3);

        // A deleted key can be inserted again; a memtable key deleted and
        // reinserted leaves two versions in the memtable segment.
        let mut tx = db.begin().unwrap();
        tx.insert(doc(5, 3)).unwrap();
        assert!(tx.delete(&Value::Str(key(1000))).unwrap());
        tx.insert(doc(1000, 3)).unwrap();
        tx.commit().unwrap();
        want.insert(key(5));
        assert_keys(&db.snapshot(), &want);
    }
    {
        let db = Database::open(&path, o.clone()).unwrap();
        assert_keys(&db.snapshot(), &want);
        db.flush().unwrap();
        let st = db.file_stats();
        assert_eq!(st.segments.iter().map(|s| s.n_deleted).sum::<u32>(), 200);
        assert_eq!(st.pending_deletes, 0);
        assert_eq!(st.memtable_docs, 0);
        assert_keys(&db.snapshot(), &want);
        db.check().unwrap();
    }
    let db = Database::open(&path, o).unwrap();
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
}

#[test]
fn snapshot_isolation_pins_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let o = DbOptions { segment_max_docs: 500, ..opts(&src) };
    let db = Database::create(&path, schema(), o).unwrap();
    let all: Vec<_> = (0..2000).map(|i| doc(i, 0)).collect();
    all.iter().for_each(|d| src.put(d));
    db.bulk_load(all).unwrap();
    let s1 = db.snapshot();
    let crcs = image_crcs(&s1);
    assert_eq!(crcs.len(), 4);

    let mut want = keys(0..2000);
    let policy = MergePolicy { merge_factor: 2, floor_docs: 100, ..MergePolicy::default() };
    for round in 0..6u64 {
        let mut tx = db.begin().unwrap();
        for i in (round..2000).step_by(7) {
            if want.remove(&key(i)) {
                tx.delete(&Value::Str(key(i))).unwrap();
            }
        }
        tx.commit().unwrap();
        db.flush().unwrap();
        db.merge(&policy).unwrap();
        let fresh: Vec<_> = (10_000 + 100 * round..10_100 + 100 * round).map(|i| doc(i, 0)).collect();
        fresh.iter().for_each(|d| src.put(d));
        want.extend(fresh.iter().map(|d| d[0].as_str().unwrap().to_string()));
        db.bulk_load(fresh).unwrap();
    }
    // The old snapshot still sees its version, byte for byte.
    assert_keys(&s1, &keys(0..2000));
    assert_eq!(image_crcs(&s1), crcs);
    assert_keys(&db.snapshot(), &want);
    let held = db.file_stats();
    assert!(held.pending_pages > 0);

    drop(s1);
    // Two commits: the first releases what only s1 held, except pages
    // freed by the newest commit (the older meta slot still names them).
    db.bulk_load([doc(50_000, 0)]).unwrap();
    db.bulk_load([doc(50_001, 0)]).unwrap();
    let after = db.file_stats();
    // Left pending: the manifests and freelists of the last two commits.
    assert!(after.pending_pages <= 8 && after.pending_pages < held.pending_pages, "{} -> {}", held.pending_pages, after.pending_pages);
    assert!(after.free_pages > 0);
    db.check().unwrap();
}

#[test]
fn freed_pages_are_reused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Database::create(&path, schema(), opts(&src)).unwrap();
    let docs: Vec<_> = (0..3000).map(|i| doc(i, 0)).collect();
    docs.iter().for_each(|d| src.put(d));
    db.bulk_load(docs).unwrap();
    let policy = MergePolicy { merge_factor: 2, floor_docs: 1000, max_deleted_ratio: 0.2, ..MergePolicy::default() };
    let mut sizes = Vec::new();
    for round in 1..=25 {
        let mut tx = db.begin().unwrap();
        for i in 0..3000 {
            let d = doc(i, round);
            src.put(&d);
            tx.upsert(d).unwrap();
        }
        tx.commit().unwrap();
        db.flush().unwrap();
        db.merge(&policy).unwrap();
        sizes.push(db.file_stats().file_bytes);
    }
    // Every round rewrites the whole table; without reuse the file would
    // grow linearly.
    let (early, late) = (sizes[4], *sizes.last().unwrap());
    assert!(late <= early * 2, "file grew from {early} to {late}: {sizes:?}");
    assert_keys(&db.snapshot(), &keys(0..3000));
    db.check().unwrap();
}

/// After deletes and merges the free space sits below the newest segments;
/// vacuum moves them down and truncates the file, but never touches pages
/// a live snapshot can reach.
#[test]
fn vacuum_returns_space_and_spares_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Database::create(&path, schema(), DbOptions { segment_max_docs: 1000, ..opts(&src) }).unwrap();
    let docs: Vec<_> = (0..6000).map(|i| doc(i, 0)).collect();
    docs.iter().for_each(|d| src.put(d));
    db.bulk_load(docs).unwrap();
    let mut want = keys(0..6000);
    let mut tx = db.begin().unwrap();
    for i in (0..6000).filter(|i| i % 3 != 0) {
        tx.delete(&Value::Str(key(i))).unwrap();
        want.remove(&key(i));
    }
    tx.commit().unwrap();
    db.flush().unwrap();
    let policy = MergePolicy { max_deleted_ratio: 0.5, ..MergePolicy::default() };
    assert_eq!(db.merge(&policy).unwrap().merges, 6);
    let held = db.snapshot();
    let held_crcs = image_crcs(&held);
    let before = db.file_stats();
    // With the merged version pinned, its pages stay where they are.
    db.vacuum().unwrap();
    assert_eq!(image_crcs(&held), held_crcs);
    drop(held);
    let shrunk = db.vacuum().unwrap();
    let after = db.file_stats();
    assert!(shrunk > 0 && after.file_bytes + shrunk <= before.file_bytes, "{} -> {}", before.file_bytes, after.file_bytes);
    // What is left beyond live data: the last commit's pending frees.
    assert!(after.free_pages + after.pending_pages <= 8, "{after:?}");
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
    drop(db);
    let db = Database::open(&path, opts(&src)).unwrap();
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
}

#[test]
fn merge_with_document_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let mut want = BTreeSet::new();
    {
        let db = Database::create(&path, schema(), opts(&src)).unwrap();
        for r in 0..10 {
            let mut tx = db.begin().unwrap();
            for i in r * 100..r * 100 + 100 {
                let d = doc(i, 0);
                src.put(&d);
                tx.insert(d).unwrap();
                want.insert(key(i));
            }
            tx.commit().unwrap();
            db.flush().unwrap();
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut tx = db.begin().unwrap();
        for i in 0..1000 {
            if rng.gen_bool(0.3) {
                assert!(tx.delete(&Value::Str(key(i))).unwrap());
                want.remove(&key(i));
            }
        }
        tx.commit().unwrap();
        // Deletes alone add no memtable segment.
        assert_eq!(db.snapshot().segments().len(), 10);
        let deleted = 1000 - want.len() as u64;
        let policy = MergePolicy { merge_factor: 4, floor_docs: 50, ..MergePolicy::default() };
        let stats = db.merge(&policy).unwrap();
        // Ten tier-0 segments: two merges of four leave two behind.
        assert_eq!((stats.merges, stats.segments_in), (2, 8), "{stats:?}");
        let s = db.snapshot();
        assert_eq!(s.segments().len(), 4);
        assert_keys(&s, &want);
        // Merged segments hold only documents live at the merge; pending
        // deletes of their inputs are dropped with the documents.
        let st = db.file_stats();
        let stored: u64 = st.segments.iter().map(|s| s.n_docs as u64).sum();
        assert_eq!(stored, 1000 - stats.docs_dropped);
        assert!(stats.docs_dropped > deleted / 2 && stats.docs_dropped <= deleted);
        assert_eq!(st.pending_deletes as u64, deleted - st.segments.iter().map(|s| s.n_deleted as u64).sum::<u64>() - stats.docs_dropped);
        db.check().unwrap();
    }
    let db = Database::open(&path, opts(&src)).unwrap();
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
    let no_source = DbOptions { doc_source: None, ..opts(&src) };
    drop(db);
    let db = Database::open(&path, no_source).unwrap();
    assert!(matches!(db.merge(&MergePolicy::default()), Err(Error::Schema(_))));
}

/// Writes that land between a merge's snapshot and its commit (deletes
/// still pending, deletes flushed into the inputs' delete sets, upserts)
/// are carried over to the merged segment.
#[test]
fn merge_folds_in_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Arc::new(Database::create(&path, schema(), DbOptions { segment_max_docs: 250, ..opts(&src) }).unwrap());
    let docs: Vec<_> = (0..1000).map(|i| doc(i, 0)).collect();
    docs.iter().for_each(|d| src.put(d));
    db.bulk_load(docs).unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..50 {
        tx.delete(&Value::Str(key(i))).unwrap();
    }
    tx.commit().unwrap();

    let hook_db = db.clone();
    let hook_src = src.clone();
    *src.hook.lock().unwrap() = Some(Box::new(move || {
        let mut tx = hook_db.begin().unwrap();
        for i in 100..150 {
            tx.delete(&Value::Str(key(i))).unwrap();
        }
        tx.commit().unwrap();
        hook_db.flush().unwrap();
        let mut tx = hook_db.begin().unwrap();
        for i in 300..350 {
            tx.delete(&Value::Str(key(i))).unwrap();
        }
        for i in 500..550 {
            let d = doc(i, 1);
            hook_src.put(&d);
            tx.upsert(d).unwrap();
        }
        tx.commit().unwrap();
    }));
    let policy = MergePolicy { merge_factor: 4, floor_docs: 100, ..MergePolicy::default() };
    let stats = db.merge(&policy).unwrap();
    assert_eq!(stats.merges, 1);
    assert_eq!(stats.docs_dropped, 50);
    let mut want = keys(50..1000);
    for i in (100..150).chain(300..350) {
        want.remove(&key(i));
    }
    let s = db.snapshot();
    assert_keys(&s, &want);
    // One merged segment holding the concurrent deletes, then the memtable.
    assert_eq!(s.segments().len(), 2);
    assert_eq!(s.segments()[0].n_live, 950 - 100 - 50);
    db.check().unwrap();
    drop(s);
    drop(db);
    let db = Database::open(&path, opts(&src)).unwrap();
    assert_keys(&db.snapshot(), &want);
    db.flush().unwrap();
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
}

fn newest_slot(path: &std::path::Path) -> u64 {
    let f = std::fs::File::open(path).unwrap();
    let mut best = (0, 0);
    for slot in 0..2u64 {
        let mut page = vec![0u8; PAGE as usize];
        f.read_exact_at(&mut page, slot * PAGE).unwrap();
        if let Some(m) = Meta::decode(&page).unwrap() {
            if m.txn > best.1 {
                best = (slot, m.txn);
            }
        }
    }
    best.0
}

#[test]
fn damaged_meta_slot_falls_back_to_previous_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    {
        let db = Database::create(&path, schema(), opts(&src)).unwrap();
        let docs: Vec<_> = (0..300).map(|i| doc(i, 0)).collect();
        docs.iter().for_each(|d| src.put(d));
        db.bulk_load(docs).unwrap();
        let mut tx = db.begin().unwrap();
        for i in 0..100 {
            tx.delete(&Value::Str(key(i))).unwrap();
        }
        tx.commit().unwrap();
        db.flush().unwrap();
        // The newest version: a merge that frees everything the previous
        // version uses.
        let policy = MergePolicy { max_deleted_ratio: 0.2, ..MergePolicy::default() };
        assert_eq!(db.merge(&policy).unwrap().merges, 1);
        let mut tx = db.begin().unwrap();
        tx.insert(doc(1000, 0)).unwrap();
        tx.commit().unwrap();
        db.flush().unwrap();
    }
    // Tear the newest meta slot.
    let slot = newest_slot(&path);
    let f = OpenOptions::new().write(true).open(&path).unwrap();
    f.write_all_at(&[0xAB; 40], slot * PAGE + 20).unwrap();
    drop(f);
    {
        let db = Database::open(&path, opts(&src)).unwrap();
        // The merge's version: deleted documents dropped, the last insert lost.
        let s = db.snapshot();
        assert_keys(&s, &keys(100..300));
        assert_eq!(s.segments().len(), 1);
        assert_eq!(s.segments()[0].file.n_docs(), 200);
        db.check().unwrap();
        // And the database keeps working from there.
        let mut tx = db.begin().unwrap();
        tx.insert(doc(2000, 0)).unwrap();
        tx.commit().unwrap();
        db.flush().unwrap();
        db.check().unwrap();
    }
    let f = OpenOptions::new().write(true).open(&path).unwrap();
    f.write_all_at(&vec![0u8; 2 * PAGE as usize], 0).unwrap();
    drop(f);
    assert!(matches!(Database::open(&path, opts(&src)), Err(Error::Corrupt(_))));
}

#[test]
fn single_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Database::create(&path, schema(), opts(&src)).unwrap();
    assert!(matches!(Database::open(&path, opts(&src)), Err(Error::Busy(_))));
    let tx = db.begin().unwrap();
    assert!(matches!(db.begin(), Err(Error::Busy(_))));
    assert!(matches!(db.flush(), Err(Error::Busy(_))));
    assert!(matches!(db.bulk_load(vec![doc(1, 0)]), Err(Error::Busy(_))));
    assert!(matches!(db.check(), Err(Error::Busy(_))));
    let _ = db.snapshot();
    drop(tx);
    db.begin().unwrap().commit().unwrap();
}

#[test]
fn bulk_load_splits_segments_and_rejects_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let db = Database::create(&path, schema(), DbOptions { segment_max_docs: 1000, ..opts(&src) }).unwrap();
    db.bulk_load((0..2500).map(|i| doc(i, 0))).unwrap();
    let sizes: Vec<_> = db.file_stats().segments.iter().map(|s| s.n_docs).collect();
    assert_eq!(sizes, vec![1000, 1000, 500]);
    let before = db.file_stats();
    assert!(matches!(db.bulk_load([doc(3000, 0), doc(3000, 0)]), Err(Error::Schema(_))));
    assert!(matches!(db.bulk_load([doc(5, 0)]), Err(Error::Schema(_))));
    assert!(matches!(db.bulk_load([vec![Value::Int(1)]]), Err(Error::Schema(_))));
    assert_eq!(db.file_stats().segments.len(), before.segments.len());
    // Memtable contents are flushed first.
    let mut tx = db.begin().unwrap();
    tx.insert(doc(4000, 0)).unwrap();
    tx.commit().unwrap();
    db.bulk_load([doc(4001, 0)]).unwrap();
    let st = db.file_stats();
    assert_eq!((st.memtable_docs, st.segments.len()), (0, 5));
    let mut want = keys(0..2500);
    want.insert(key(4000));
    want.insert(key(4001));
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
    let s = db.snapshot();
    s.evict_from_cache().unwrap();
    assert_keys(&s, &want);
}

#[test]
fn table_without_primary_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let schema = Schema::new("t", vec![Field::new("n", FieldType::Int)], None).unwrap();
    let db = Database::create(&path, schema, DbOptions { sync: false, ..DbOptions::default() }).unwrap();
    let mut tx = db.begin().unwrap();
    tx.insert(vec![Value::Int(1)]).unwrap();
    tx.insert(vec![Value::Int(1)]).unwrap();
    assert!(matches!(tx.delete(&Value::Int(1)), Err(Error::Schema(_))));
    assert!(matches!(tx.upsert(vec![Value::Int(1)]), Err(Error::Schema(_))));
    tx.commit().unwrap();
    db.bulk_load([vec![Value::Int(2)]]).unwrap();
    let s = db.snapshot();
    assert_eq!(s.n_live_docs(), 3);
    assert!(s.lookup(&Value::Int(1)).is_none() && s.pk(0).is_none());
    db.check().unwrap();
}

/// Readers take snapshots while one thread writes (with automatic flushes)
/// and another merges; every snapshot is a committed state.
#[test]
fn readers_see_consistent_snapshots_during_writes_and_merges() {
    readers_during_writes_and_merges(false);
}

/// The same with snapshots built by the writer and published.
#[test]
fn readers_see_published_snapshots_during_writes_and_merges() {
    readers_during_writes_and_merges(true);
}

fn readers_during_writes_and_merges(publish_snapshots: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let o = DbOptions { memtable_max_docs: 64, publish_snapshots, ..opts(&src) };
    let db = Arc::new(Database::create(&path, schema(), o).unwrap());
    let done = Arc::new(AtomicBool::new(false));
    const TXNS: u64 = 150;
    std::thread::scope(|sc| {
        sc.spawn(|| {
            for t in 0..TXNS {
                let mut tx = loop {
                    match db.begin() {
                        Ok(tx) => break tx,
                        Err(Error::Busy(_)) => std::thread::yield_now(),
                        Err(e) => panic!("{e}"),
                    }
                };
                for i in t * 10..t * 10 + 10 {
                    let d = doc(i, 0);
                    src.put(&d);
                    tx.insert(d).unwrap();
                }
                tx.commit().unwrap();
            }
            done.store(true, Ordering::Release);
        });
        sc.spawn(|| {
            let policy = MergePolicy { merge_factor: 3, floor_docs: 64, ..MergePolicy::default() };
            while !done.load(Ordering::Acquire) {
                db.merge(&policy).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        for _ in 0..2 {
            sc.spawn(|| {
                let mut last = 0;
                let mut seen = 0;
                while !done.load(Ordering::Acquire) || seen < 3 {
                    let s = db.snapshot();
                    let n = s.n_live_docs();
                    assert!(n.is_multiple_of(10) && n >= last, "{n} after {last}");
                    assert_keys(&s, &keys(0..n));
                    assert!(s.lookup(&Value::Str(key(n))).is_none());
                    last = n;
                    seen += 1;
                }
            });
        }
    });
    assert_keys(&db.snapshot(), &keys(0..TXNS * 10));
    db.check().unwrap();
}

/// With published snapshots, every write returns with its version built
/// and handed to readers: a reader gets that one (the same one each time)
/// and does not wait for a transaction in progress.
#[test]
fn published_snapshots_follow_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let o = DbOptions { publish_snapshots: true, memtable_chunk_docs: 64, ..opts(&src) };
    let db = Database::create(&path, schema(), o.clone()).unwrap();
    assert_eq!(db.snapshot().n_live_docs(), 0);
    let ids = |s: &Snapshot| s.segments().iter().map(|v| v.id).collect::<Vec<_>>();
    let mut tx = db.begin().unwrap();
    for i in 0..100 {
        src.put(&doc(i, 0));
        tx.insert(doc(i, 0)).unwrap();
    }
    tx.commit().unwrap();
    let s1 = db.snapshot();
    assert_keys(&s1, &keys(0..100));
    assert_eq!(ids(&db.snapshot()), ids(&s1));
    assert_eq!(db.snapshot().txn(), s1.txn());
    // A reader during a transaction sees the last commit, without waiting.
    let mut tx = db.begin().unwrap();
    src.put(&doc(100, 0));
    tx.insert(doc(100, 0)).unwrap();
    std::thread::scope(|sc| sc.spawn(|| assert_keys(&db.snapshot(), &keys(0..100))).join().unwrap());
    tx.commit().unwrap();
    let s2 = db.snapshot();
    assert_keys(&s2, &keys(0..101));
    assert_eq!(ids(&s1)[0], ids(&s2)[0], "a full chunk is not rebuilt");
    assert_keys(&s1, &keys(0..100));
    // Flushes, bulk loads, merges and vacuums publish too.
    db.flush().unwrap();
    assert!(db.snapshot().segments().iter().all(|v| !v.in_memory));
    db.bulk_load((200..300).map(|i| {
        src.put(&doc(i, 0));
        doc(i, 0)
    }))
    .unwrap();
    let mut want = keys(0..101);
    want.extend(keys(200..300));
    assert_keys(&db.snapshot(), &want);
    let mut tx = db.begin().unwrap();
    for i in 0..50 {
        tx.delete(&Value::Str(key(i))).unwrap();
        want.remove(&key(i));
    }
    tx.commit().unwrap();
    db.flush().unwrap();
    db.merge(&MergePolicy { merge_factor: 2, floor_docs: 1000, ..MergePolicy::default() }).unwrap();
    assert_eq!(db.snapshot().segments().len(), 1);
    assert_keys(&db.snapshot(), &want);
    db.vacuum().unwrap();
    assert_keys(&db.snapshot(), &want);
    db.check().unwrap();
    // Reopening publishes the version recovered from the WAL.
    let mut tx = db.begin().unwrap();
    src.put(&doc(400, 0));
    tx.insert(doc(400, 0)).unwrap();
    tx.commit().unwrap();
    want.insert(key(400));
    drop(db);
    let db = Database::open(&path, o).unwrap();
    assert_keys(&db.snapshot(), &want);
}

/// Random transactions, flushes, merges and reopens against a model.
#[test]
fn random_operations_match_model() {
    for seed in 0..6 {
        random_operations(seed, false);
    }
}

/// The same with published snapshots.
#[test]
fn random_operations_match_model_with_published_snapshots() {
    for seed in 0..3 {
        random_operations(seed, true);
    }
}

fn random_operations(seed: u64, publish_snapshots: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.strata");
    let src = Arc::new(MapSource::default());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let o = DbOptions { memtable_max_docs: 150, memtable_chunk_docs: 64, segment_max_docs: 120, publish_snapshots, ..opts(&src) };
    let policy = MergePolicy { merge_factor: 3, floor_docs: 40, max_deleted_ratio: 0.4, ..MergePolicy::default() };
    let mut db = Some(Database::create(&path, schema(), o.clone()).unwrap());
    let mut model: BTreeSet<String> = BTreeSet::new();
    let mut version = 0i64;
    for step in 0..400 {
        let d = db.as_ref().unwrap();
        match rng.gen_range(0..100) {
            0..=69 => {
                let commit = rng.gen_bool(0.85);
                let mut local = model.clone();
                let mut tx = d.begin().unwrap();
                for _ in 0..rng.gen_range(1..40) {
                    let i = rng.gen_range(0..400u64);
                    let k = key(i);
                    version += 1;
                    match rng.gen_range(0..3) {
                        0 => {
                            let r = tx.insert(doc(i, version));
                            assert_eq!(r.is_ok(), !local.contains(&k));
                            if r.is_ok() {
                                src.put(&doc(i, version));
                            }
                            local.insert(k);
                        }
                        1 => {
                            src.put(&doc(i, version));
                            tx.upsert(doc(i, version)).unwrap();
                            local.insert(k);
                        }
                        _ => assert_eq!(tx.delete(&Value::Str(k.clone())).unwrap(), local.remove(&k)),
                    }
                }
                if commit {
                    tx.commit().unwrap();
                    model = local;
                } else {
                    tx.rollback();
                }
            }
            70..=77 => d.flush().unwrap(),
            78..=86 => {
                d.merge(&policy).unwrap();
            }
            87..=89 => {
                d.vacuum().unwrap();
            }
            90..=94 => d
                .bulk_load((10_000 + step * 3..10_003 + step * 3).map(|i| {
                    model.insert(key(i));
                    let x = doc(i, 0);
                    src.put(&x);
                    x
                }))
                .unwrap(),
            _ => {
                // Close before reopening: the file lock admits one handle.
                drop(db.take());
                db = Some(Database::open(&path, o.clone()).unwrap());
            }
        }
        assert_keys(&db.as_ref().unwrap().snapshot(), &model);
        if step % 50 == 0 {
            db.as_ref().unwrap().check().unwrap();
        }
    }
    db.as_ref().unwrap().check().unwrap();
}
