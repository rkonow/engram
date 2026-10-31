//! Read-only views of the store for the demo inspector (`demo/`): SQL with
//! structured results and text-embedded parameters, the table's schema and
//! physical layout, and stored vectors.

use std::time::Instant;

use serde_json::{json, Map, Value as Json};
use strata::query::Params;
use strata::Value;

use crate::store::Store;
use crate::{err, Result};

/// A Strata value as JSON. Vectors are returned whole unless `max_dims`
/// cuts them (with their length kept in `dim`).
pub fn value_json(v: &Value, max_dims: Option<usize>) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Int(x) => json!(x),
        Value::Float(x) => json!(x),
        Value::Str(s) => json!(s),
        Value::Vector(v) => {
            let n = max_dims.unwrap_or(v.len()).min(v.len());
            json!({"dim": v.len(), "values": v[..n].to_vec()})
        }
        Value::List(xs) => Json::Array(xs.iter().map(|x| value_json(x, max_dims)).collect()),
        Value::Bool(b) => json!(b),
        Value::Timestamp(t) => json!(strata::timestamp::format(*t)),
    }
}

/// Runs one read-only SELECT. `params` binds placeholders to plain values;
/// `embed` binds placeholders to the store embedder's vector of a text, so
/// `ORDER BY similarity(emb, :q)` works from a sentence.
pub fn query(store: &Store, req: &Json) -> Result<Json> {
    let sql = req.get("sql").and_then(|v| v.as_str()).unwrap_or("").trim();
    if !sql.get(..6).is_some_and(|s| s.eq_ignore_ascii_case("select")) {
        return Err(err("only SELECT statements are allowed"));
    }
    let mut params = Params::new();
    if let Some(m) = req.get("params").and_then(|v| v.as_object()) {
        for (k, v) in m {
            let val = match v {
                Json::Number(n) if n.is_i64() => Value::Int(n.as_i64().unwrap()),
                Json::Number(n) => Value::Float(n.as_f64().unwrap_or(0.0)),
                Json::String(s) => Value::Str(s.clone()),
                Json::Bool(b) => Value::Int(*b as i64),
                Json::Array(a) => Value::Vector(a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>().into()),
                Json::Null | Json::Object(_) => Value::Null,
            };
            params = params.set(k, val);
        }
    }
    if let Some(m) = req.get("embed").and_then(|v| v.as_object()) {
        for (k, v) in m {
            let text = v.as_str().unwrap_or("");
            params = params.set(k, Value::Vector(store.embedder.embed_query(text)?.into()));
        }
    }
    let max_dims = req.get("max_dims").and_then(|v| v.as_u64()).map(|n| n as usize);
    let limit = req.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
    let t0 = Instant::now();
    let rs = store.query_with(sql, &params)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let rows: Vec<Json> = rs.rows.iter().take(limit).map(|r| Json::Array(r.iter().map(|v| value_json(v, max_dims)).collect())).collect();
    Ok(json!({
        "columns": rs.columns, "rows": rows, "total_rows": rs.rows.len(), "ms": ms,
        "plan": rs.stats.plan, "filter_docs": rs.stats.filter_docs, "scored": rs.stats.scored,
    }))
}

/// The table's schema and how it is laid out on disk: segments (with their
/// sections' sizes), the memtable, the WAL and the file.
pub fn layout(store: &Store) -> Result<Json> {
    let snap = store.db.snapshot();
    let schema = serde_json::to_value(snap.schema())?;
    let segments: Vec<Json> = snap
        .segments()
        .iter()
        .map(|v| {
            let mut sections: Vec<(String, usize)> = v.file.section_sizes().into_iter().map(|(k, n)| (strata::segment::label(snap.schema(), &k), n)).collect();
            sections.sort_by(|a, b| b.1.cmp(&a.1));
            let sections: Map<String, Json> = sections.into_iter().map(|(k, n)| (k, json!(n))).collect();
            json!({"id": v.id, "in_memory": v.in_memory, "n_docs": v.file.n_docs(), "n_live": v.n_live, "sections": sections})
        })
        .collect();
    let fs = store.db.storage().file_stats();
    Ok(json!({
        "schema": schema, "segments": segments, "live_docs": snap.n_live_docs(), "embedder": store.embedder.name(), "dim": store.embedder.dim(),
        "file": {"file_bytes": fs.file_bytes, "page_count": fs.page_count, "live_pages": fs.live_pages, "free_pages": fs.free_pages,
                 "wal_bytes": fs.wal_bytes, "memtable_docs": fs.memtable_docs, "pending_deletes": fs.pending_deletes},
    }))
}

/// The store embedder's vector of a text (what `similarity(emb, :q)` compares
/// records to).
pub fn embed(store: &Store, req: &Json) -> Result<Json> {
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let v = store.embedder.embed_query(text)?;
    Ok(json!({"embedder": store.embedder.name(), "dim": v.len(), "values": v}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Draft, Kind};
    use crate::store::StoreOptions;
    use std::sync::Arc;

    #[test]
    fn query_layout_and_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let emb: Arc<dyn crate::embed::Embedder> = Arc::from(crate::embed::open("hash:64").unwrap());
        let s = Store::open(&dir.path().join("m.strata"), emb, StoreOptions::default()).unwrap();
        s.append("sess", 1, 10, vec![Draft::new(Kind::Decision, "Prices are integer cents").entities(vec!["prices".into()]), Draft::new(Kind::Prompt, "add a refund endpoint")])
            .unwrap();
        let r = query(&s, &json!({"sql": "SELECT id, kind, emb FROM memory WHERE kind = 'decision' ORDER BY similarity(emb, :q) DESC LIMIT 5", "embed": {"q": "cents"}, "max_dims": 4}))
            .unwrap();
        assert_eq!(r["columns"], json!(["id", "kind", "emb"]));
        assert_eq!(r["rows"][0][1], json!("decision"));
        assert_eq!(r["rows"][0][2]["dim"], json!(64));
        assert_eq!(r["rows"][0][2]["values"].as_array().unwrap().len(), 4);
        assert!(query(&s, &json!({"sql": "DELETE FROM memory"})).is_err());
        let l = layout(&s).unwrap();
        assert_eq!(l["schema"]["table"], json!("memory"));
        // Sections are named by field id on disk, by field name here.
        s.flush().unwrap();
        let l = layout(&s).unwrap();
        assert!(l["segments"].as_array().unwrap().iter().any(|g| g["sections"].get("text.body.dict").is_some()), "{}", l["segments"]);
        assert_eq!(l["live_docs"], json!(3)); // two records and one entity
        assert_eq!(embed(&s, &json!({"text": "x"})).unwrap()["dim"], json!(64));
    }
}
