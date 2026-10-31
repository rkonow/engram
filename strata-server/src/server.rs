//! Opening the database, and serving HTTP and gRPC.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use strata::db::Db;
use strata::ingest::Format;
use strata::schema::Schema;
use strata::segment::build::BuildOptions;
use strata::storage::{DbOptions, Layout, MergePolicy, MergerOptions};
use tokio::net::TcpListener;

use crate::auth::{self, Access};
use crate::coord::{CoordOptions, Coordinator};
use crate::replication::{self, Acks, Follow, Primary, PrimaryOptions};
use crate::service::{shard_of, AgentConfig, Backend, Replication, Service};
use crate::alerts::Alerts;
use crate::keys::{Audit, Keys};
use crate::{demo, grpc, http, pb};

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The database to serve; created if missing (with `schema` or `demo`).
    pub db: Option<PathBuf>,
    /// A JSON schema for a new database, in either form ([`Schema::from_json`]).
    pub schema: Option<PathBuf>,
    /// Fill a new database with the rows of this one ([`Db::load_from`]).
    pub copy_from: Option<PathBuf>,
    /// Create the demo catalogue with this many products.
    pub demo: Option<usize>,
    /// Keep only the demo products of shard `i` of `n`.
    pub demo_shard: Option<(usize, usize)>,
    /// A new database as a directory of segment files.
    pub directory: bool,
    /// Open a database whose WAL was damaged after it was durable
    /// ([`DbOptions::wal_drop_damaged`]).
    pub wal_drop_damaged: bool,
    /// Check every segment's checksum on opening
    /// ([`DbOptions::verify_checksums`]).
    pub verify_checksums: bool,
    /// Number the database's writes ([`strata::storage::Database::start_seq_no`]),
    /// for `seq_no()` and conditional writes (`if_seq_no`); once started,
    /// for good.
    pub seq_no: bool,
    /// Ship the database's writes to these replicas (gRPC URLs of servers
    /// started with `replica`), as their primary
    /// ([`crate::replication`]); the replicas' token and CA are
    /// `coord.token` and `coord.ca`. With `replica`: the shard's other
    /// copies, which it ships to once promoted, and the only ones a
    /// promotion asked for with a key may name.
    pub replicas: Vec<String>,
    /// Apply the writes a primary ships, and take none from clients.
    pub replica: bool,
    /// Start as the primary of a new term, shipping to `replicas` (none:
    /// to no replica), though the database was a replica or another
    /// primary replaced it ([`PrimaryOptions::promote`]).
    pub promote: bool,
    /// With `replicas`: which copies hold a write before it is answered.
    pub write_acks: Acks,
    /// On a primary or a replica: the latest transactions kept to ship, in
    /// MiB (0: 64).
    pub ship_log_mb: usize,
    /// With `replicas`: how often each replica's rows are compared with
    /// the primary's, in seconds (0: only when asked, `POST
    /// /v1/admin/verify`).
    pub verify_every_s: u64,
    /// On a primary or a replica: how long a replica hears nothing from
    /// its primary before it asks its shard's copies to elect it, in
    /// milliseconds (0: never; [`Service::start_failover`]).
    pub failover_after_ms: u64,
    /// Coordinate these shard servers (gRPC URLs; replicas of a shard
    /// joined by `|`) instead of serving a database.
    pub shards: Vec<String>,
    pub coord: CoordOptions,
    /// With `shards`: the `StrataCluster` (`namespace/name`) that runs this
    /// coordinator on Kubernetes, which the command center shows and scales.
    pub kube_cluster: Option<String>,
    pub timeout_ms: u64,
    /// The longest deadline a request may ask for (0: the larger of
    /// `timeout_ms` and 60 s).
    pub max_timeout_ms: u64,
    pub max_concurrent: usize,
    /// The most rows a query may reach, OFFSET included (0: no limit).
    pub max_result_window: usize,
    /// The most groups a query may make (0: no limit).
    pub max_groups: usize,
    /// The most rows a cross-encoder's rerank phase may score (0: no
    /// limit).
    pub max_rerank: usize,
    /// Memory the running queries may hold together, in MiB (`None`: a
    /// third of [`strata::storage::memory_limit`]; 0: no limit).
    pub query_memory_mb: Option<u64>,
    /// Write requests held at once before more are refused (0: no limit).
    pub max_queued_writes: usize,
    /// The largest write or import body, in MiB (0: 256).
    pub max_body_mb: usize,
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: String,
    /// Where embedded fields' models are ([`strata::embed::open`]).
    pub models: PathBuf,
    /// The synonym list for `synonyms => 1` searches (Solr format), read at
    /// start if it exists and written by `PUT /v1/synonyms`.
    pub synonyms: Option<PathBuf>,
    /// Where `POST /v1/admin/checkpoint` writes checkpoints.
    pub backup_dir: Option<PathBuf>,
    /// What writes may fetch for MEDIA fields: `file:` roots and hosts
    /// ([`strata::media::Allowlist::parse`]).
    pub media_fetch: Vec<String>,
    /// The largest media file a write fetches or gives, in bytes (0:
    /// [`strata::media::DEFAULT_MAX_BYTES`]).
    pub media_max_bytes: u64,
    /// Where bytes that writes give a MEDIA field without a URI are kept.
    pub media_store: Option<PathBuf>,
    /// How long a media fetch may take, in seconds (0:
    /// [`crate::media::DEFAULT_TIMEOUT`]).
    pub media_timeout_s: u64,
    /// The background merger ([`strata::storage::Database::spawn_merger`]).
    pub merger: MergerOptions,
    /// Log queries and writes that take this many milliseconds or more
    /// (0: none).
    pub slow_ms: u64,
    /// The API keys' file (default: `<db>.keys.json` beside a database; a
    /// coordinator's keys are in memory without it).
    pub keys: Option<PathBuf>,
    /// Append every audited request here, as JSON lines.
    pub audit_log: Option<PathBuf>,
    /// The alert rules' file (default rules, in memory, without it).
    pub alerts: Option<PathBuf>,
    /// Post alerts that fire or resolve to this URL.
    pub alert_webhook: Option<String>,
    /// The log store ([`crate::logstore`]): `None` beside the database
    /// (`<db>.logs`), and none on a coordinator; `Some(None)` none.
    pub log_store: Option<Option<PathBuf>>,
    /// The name this server's rows carry in log stores (empty: the host's
    /// name and the HTTP port, [`crate::logstore::hostname`]).
    pub node_name: String,
    /// Log store rows kept this long (zero: 7 days).
    pub log_retain: Duration,
    /// The most bytes the log store takes (0: no limit).
    pub log_store_bytes: u64,
    /// Queries a second the log store keeps besides slow and failed ones.
    pub log_queries_per_sec: u64,
}

/// How writes resolve MEDIA values, from `--media-fetch`,
/// `--media-max-bytes`, `--media-store` and `--media-timeout-s`.
fn media_resolver(o: &Options) -> Result<strata::media::Resolver, String> {
    let allow = strata::media::Allowlist::parse(&o.media_fetch).map_err(|e| e.to_string())?;
    if let Some(dir) = &o.media_store {
        std::fs::create_dir_all(dir).map_err(|e| format!("--media-store {}: {e}", dir.display()))?;
    }
    let timeout = if o.media_timeout_s > 0 { Duration::from_secs(o.media_timeout_s) } else { crate::media::DEFAULT_TIMEOUT };
    let max_bytes = if o.media_max_bytes > 0 { o.media_max_bytes } else { strata::media::DEFAULT_MAX_BYTES };
    let http = crate::media::HttpFetcher::new(allow.clone(), timeout)?;
    tracing::info!(fetch = %allow.entries().join(", "), max_bytes, store = ?o.media_store, timeout_s = timeout.as_secs(), "media fields: what writes may fetch");
    Ok(strata::media::Resolver { allow, max_bytes, store: o.media_store.clone(), http: Some(std::sync::Arc::new(http)) })
}

/// Segment builds (flushes, merges, imports): vector indexes on every core.
fn build_options() -> BuildOptions {
    BuildOptions::default().set("vector.threads", std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// How the server opens a database. Each write publishes its snapshot,
/// rebuilding the newest in-memory chunk (up to 1,024 rows): those chunks
/// keep their vectors flat, since a graph over so few rows costs more to
/// build at every write than an exact scan of them costs a query. Segments
/// build their vector indexes on every core.
pub fn db_options(directory: bool) -> DbOptions {
    DbOptions {
        publish_snapshots: true,
        build: build_options(),
        mem_build: Some(BuildOptions::default().set("vector.index", "flat")),
        layout: if directory { Layout::Directory } else { Layout::SingleFile },
        ..DbOptions::default()
    }
}

/// Opens or creates the database, or connects to the shards.
pub async fn service(o: &Options) -> Result<Arc<Service>, String> {
    let agent = AgentConfig { api_key: o.api_key.clone(), model: o.model.clone(), base_url: o.api_base.clone() };
    let timeout = Duration::from_millis(o.timeout_ms.max(1));
    // What a new database is filled with, once its models are open.
    enum Fill {
        Demo(usize),
        Copy(PathBuf),
    }
    let mut fill = None;
    let mut slots_file = None;
    if o.seq_no && !o.shards.is_empty() {
        return Err("--seq-no numbers a database's writes: start each shard server with it, not the coordinator".into());
    }
    let replicating = !o.replicas.is_empty() || o.replica || o.promote;
    if replicating && !o.shards.is_empty() {
        return Err("--replicas, --replica and --promote are for the servers holding a shard's copies, not the coordinator".into());
    }
    if o.promote && o.replica {
        return Err("a server is a primary (--replicas, --promote) or a replica (--replica), not both".into());
    }
    // A server that numbers its writes keeps a log of them to ship, so that
    // it can become a primary while it serves (`Configure`), and so that
    // rows moving to another shard catch up from it.
    let ship_log_bytes = match replicating || o.seq_no {
        true => (if o.ship_log_mb > 0 { o.ship_log_mb } else { 64 }) << 20,
        false => 0,
    };
    let mut replication_file = None;
    let (backend, schema) = if !o.shards.is_empty() {
        // A shard with no replica answering yet (the cluster is starting, or
        // one is down) is waited for; a wrong URL or schema is not.
        let c = loop {
            match Coordinator::connect_with(o.shards.clone(), o.coord.clone()).await {
                Ok(c) => break c,
                // A shard that refuses the request, or the slot map, will
                // not change its mind.
                Err(e) if e.status != 400 && e.status != 409 => {
                    tracing::warn!(error = %e.message, "waiting for the shards");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => return Err(e.message),
            }
        };
        let schema = c.schema.clone();
        let c = Arc::new(c);
        // Follows the cluster map as admin calls (here or at another
        // coordinator) change it.
        c.watch();
        (Backend::Cluster(c), schema)
    } else {
        let path = o.db.clone().ok_or("pass --db PATH (or --shards to coordinate shard servers)")?;
        slots_file = Some(crate::slots::file_of(&path));
        replication_file = Some(replication::file_of(&path));
        // A part in replication kept beside the database is taken up again
        // (`Configure` set it, or a promotion): such a database numbers
        // its writes, and keeps a log of them.
        let kept = if path.exists() && !replicating { replication::kept_role(&path)?.0 } else { None };
        let ship_log_bytes = if kept.is_some() && ship_log_bytes == 0 { (if o.ship_log_mb > 0 { o.ship_log_mb } else { 64 }) << 20 } else { ship_log_bytes };
        if o.copy_from.is_some() && (path.exists() || o.schema.is_none()) {
            return Err("--copy-from fills a new database: pass a --db that does not exist and its --schema".into());
        }
        let db = if path.exists() {
            let db = Db::open(&path, DbOptions { wal_drop_damaged: o.wal_drop_damaged, verify_checksums: o.verify_checksums, ship_log_bytes, ..db_options(o.directory) })
                .map_err(|e| format!("open {}: {e}", path.display()))?;
            if let Some(what) = db.storage().damaged() {
                tracing::warn!(db = %path.display(), damage = %what, "opened the database past damage");
            }
            db
        } else if let Some(n) = o.demo {
            fill = Some(Fill::Demo(n));
            Db::create_in(&path, demo::schema(n), DbOptions { ship_log_bytes, ..db_options(o.directory) }, &o.models).map_err(|e| e.to_string())?
        } else if let Some(s) = &o.schema {
            let text = std::fs::read_to_string(s).map_err(|e| format!("read {}: {e}", s.display()))?;
            let schema = Schema::from_json(&text).map_err(|e| format!("{}: {e}", s.display()))?;
            fill = o.copy_from.clone().map(Fill::Copy);
            Db::create_in(&path, schema, DbOptions { ship_log_bytes, ..db_options(o.directory) }, &o.models).map_err(|e| e.to_string())?
        } else {
            return Err(format!("{} does not exist: pass --schema FILE or --demo N to create it", path.display()));
        };
        for (field, why) in db.open_codebooks(&o.models) {
            tracing::warn!(field = %field, error = %why, "cannot give this field semantic IDs: writes must carry them");
        }
        if o.seq_no {
            db.storage().start_seq_no().map_err(|e| format!("--seq-no on {}: {e}", path.display()))?;
        }
        if replicating && db.storage().seq_no().is_none() {
            return Err(format!("{} does not number its writes, which replication ships by: pass --seq-no", path.display()));
        }
        let db = Arc::new(db);
        // Writes go on while a full memtable becomes segments, and while
        // segments are merged.
        db.spawn_flusher();
        db.spawn_merger(o.merger.clone());
        if let Some(b) = db.storage().merge_status().memory_budget {
            tracing::info!(threads = o.merger.threads.max(1), memory_mib = b >> 20, "merging in the background");
        }
        let schema = db.schema().clone();
        (Backend::Local(db), schema)
    };
    let mut svc = Service::new(backend, schema, agent, o.max_concurrent.max(1), timeout);
    if let Some(path) = slots_file {
        svc.keep_slots_in(path)?;
        if let Some((m, shard)) = svc.slots() {
            tracing::info!(epoch = m.epoch, shard, of = m.shards(), "holds slots");
        }
    }
    if let Some(c) = o.kube_cluster.as_deref().filter(|_| !o.shards.is_empty()) {
        match crate::operator::kube::Link::new(c) {
            Ok(l) => {
                tracing::info!(cluster = %c, "run by the Kubernetes operator: the command center shows and scales the StrataCluster");
                svc.kube = Some(l);
            }
            Err(e) => tracing::warn!(cluster = %c, error = %e, "cannot reach the StrataCluster that runs this coordinator"),
        }
    }
    if o.max_timeout_ms > 0 {
        svc.max_timeout = Duration::from_millis(o.max_timeout_ms).max(timeout);
    }
    svc.slow = (o.slow_ms > 0).then(|| Duration::from_millis(o.slow_ms));
    svc.max_rows = (o.max_result_window > 0).then_some(o.max_result_window);
    svc.max_groups = (o.max_groups > 0).then_some(o.max_groups);
    svc.max_rerank = (o.max_rerank > 0).then_some(o.max_rerank);
    let query_memory = match o.query_memory_mb {
        Some(mb) => (mb > 0).then(|| mb.saturating_mul(1 << 20)),
        None => strata::storage::memory_limit().map(|l| l / 3),
    };
    if let Some(b) = query_memory {
        tracing::info!(memory_mib = b >> 20, "queries share a memory budget");
    }
    svc.query_memory = query_memory.map(strata::query::memory::MemoryBudget::new);
    svc.max_queued_writes = if o.max_queued_writes > 0 { o.max_queued_writes } else { usize::MAX };
    if o.max_body_mb > 0 {
        svc.max_body = o.max_body_mb.saturating_mul(1 << 20);
    }
    let keys = o.keys.clone().or_else(|| o.db.as_deref().filter(|_| o.shards.is_empty()).map(Keys::file_of));
    if let Some(file) = keys {
        svc.keys = Arc::new(Keys::open(file)?);
    }
    svc.audit = Audit::new(o.audit_log.clone())?;
    svc.alerts = Alerts::new(o.alerts.clone(), o.alert_webhook.clone())?;
    svc.open_models(&o.models);
    for m in svc.models().iter() {
        match &m.model {
            Ok(e) => tracing::info!(table = %m.table, field = %m.field, model = %e.name(), from = %m.spec.from.join(" + "), "embedded field"),
            Err(why) => tracing::warn!(table = %m.table, field = %m.field, model = %m.spec.model, from = %m.spec.from.join(" + "), error = %why, "cannot run this embedded field's model: writes must supply its vectors"),
        }
    }
    if let (Some(fill), Backend::Local(db)) = (fill, &svc.backend) {
        let t0 = std::time::Instant::now();
        let n = match fill {
            Fill::Demo(n) => {
                let keep = |i: usize| match o.demo_shard {
                    Some((s, of)) => shard_of(&strata::Value::Str(format!("p{i:07}")), of) == s,
                    None => true,
                };
                db.load((0..n).filter(|&i| keep(i)).map(|i| demo::product(i, 7))).map_err(|e| e.to_string())?;
                db.snapshot().n_live_docs()
            }
            Fill::Copy(from) => {
                // A database from before field ids opens to be copied too.
                let old = Db::open(&from, DbOptions { open_old: true, ..DbOptions::default() }).map_err(|e| format!("open {}: {e}", from.display()))?;
                db.load_from(&old).map_err(|e| format!("copy from {}: {e}", from.display()))?
            }
        };
        tracing::info!(rows = n, secs = t0.elapsed().as_secs_f64(), "loaded the new database");
    }
    if let Some(path) = &o.synonyms {
        svc.open_synonyms(path).map_err(|e| e.message)?;
    }
    if let Some(dir) = &o.backup_dir {
        svc.set_backup_dir(dir).map_err(|e| e.message)?;
    }
    if !o.media_fetch.is_empty() || o.media_store.is_some() || o.media_max_bytes > 0 {
        svc.set_media(media_resolver(o)?).map_err(|e| e.message)?;
    }
    // Once loaded: bulk loads are not shipped, so every copy loads the same.
    if let (Backend::Local(_), Some(file)) = (&svc.backend, &replication_file) {
        svc.promote_options = Some(PrimaryOptions { token: o.coord.token.clone(), ca: o.coord.ca.clone(), verify_every: Duration::from_secs(o.verify_every_s), acks: o.write_acks, ..Default::default() });
        svc.failover_after = Duration::from_millis(o.failover_after_ms);
        svc.keep_replication_in(file.clone());
    }
    if let (Backend::Local(db), Some(file)) = (&svc.backend, &replication_file) {
        // How it ships as a primary, but for its replicas and acks.
        let ships = PrimaryOptions { token: o.coord.token.clone(), ca: o.coord.ca.clone(), verify_every: Duration::from_secs(o.verify_every_s), ..Default::default() };
        let ships = crate::service::failover_tuned(ships, Duration::from_millis(o.failover_after_ms));
        let kept = match replicating {
            true => (None, false),
            false => replication::kept_role(db.storage().path())?,
        };
        if kept.0.is_some() && db.storage().seq_no().is_none() {
            return Err(format!("{} keeps a part in replication, but does not number its writes", db.storage().path().display()));
        }
        // With automatic failover, a primary that starts again is a replica
        // until its shard's copies elect one: another may have been
        // elected while it was down.
        let elects = o.failover_after_ms > 0 && svc.slots().is_some_and(|(m, s)| m.replicas().get(s).is_some_and(|c| c.len() > 1));
        if let (Some(replication::KeptRole::Primary { .. }), false, true) = (&kept.0, kept.1, elects) {
            svc.set_replication(Replication::Replica(Arc::new(Follow::open(db.clone(), Some(file))?)));
            tracing::info!("was its shard's primary: starts as a replica, until its shard's copies elect one (--failover-after-ms)");
        } else if let (Some(replication::KeptRole::Primary { replicas, acks }), false) = &kept {
            let opts = PrimaryOptions { replicas: replicas.clone(), acks: *acks, ..ships };
            let p = Primary::start(db.clone(), Some(file), opts)?;
            tracing::info!(replicas = %replicas.join(", "), write_acks = acks.name(), "primary, as kept beside the database: ships its writes to its replicas");
            svc.set_replication(Replication::Primary(p));
        } else if kept.0.is_some() {
            svc.set_replication(Replication::Replica(Arc::new(Follow::open(db.clone(), Some(file))?)));
            match kept.1 {
                true => tracing::info!("replica: a primary of a newer term replaced this one, which follows it"),
                false => tracing::info!("replica, as kept beside the database: applies the writes its primary ships"),
            }
        } else if (!o.replicas.is_empty() && !o.replica) || o.promote {
            let opts = PrimaryOptions { replicas: o.replicas.clone(), acks: o.write_acks, promote: o.promote, ..ships };
            let p = Primary::start(db.clone(), Some(file), opts)?;
            tracing::info!(replicas = %o.replicas.join(", "), write_acks = o.write_acks.name(), "primary: ships its writes to its replicas");
            svc.set_replication(Replication::Primary(p));
        } else if o.replica {
            svc.set_replication(Replication::Replica(Arc::new(Follow::open(db.clone(), Some(file))?)));
            // Its --replicas: the shard's other copies, the ones a
            // promotion ships to (Service::known_copies).
            svc.promote_options = Some(PrimaryOptions { replicas: o.replicas.clone(), acks: o.write_acks, ..ships });
            tracing::info!(copies = %o.replicas.join(", "), "replica: applies the writes its primary ships, and takes none from clients");
        }
    }
    open_log_store(&mut svc, o)?;
    let svc = Arc::new(svc);
    spawn_maintenance(&svc);
    spawn_sampler(&svc);
    svc.start_failover();
    // The cluster's keys: a coordinator takes them from its shards before
    // it serves, and a coordinator or primary passes changes on.
    crate::keysync::start(&svc).await;
    Ok(svc)
}

/// Opens the server's log store ([`crate::logstore`]), beside the database
/// unless `o.log_store` says where or that there is none, and hands it the
/// service's queries; a coordinator keeps one only where told. Without one
/// the lines kept for it are let go ([`obs::Sink::stop`]).
fn open_log_store(svc: &mut Service, o: &Options) -> Result<(), String> {
    let path = match &o.log_store {
        Some(p) => p.clone(),
        None => o.db.as_deref().filter(|_| o.shards.is_empty()).map(crate::logstore::file_of),
    };
    let Some(path) = path else {
        crate::obs::sink().stop();
        return Ok(());
    };
    let node = match o.node_name.is_empty() {
        true => crate::logstore::hostname(),
        false => o.node_name.clone(),
    };
    let mut lo = crate::logstore::LogOptions::new(path, node);
    if !o.log_retain.is_zero() {
        lo.retain = o.log_retain;
    }
    lo.max_bytes = o.log_store_bytes;
    lo.queries_per_sec = o.log_queries_per_sec;
    let store = crate::logstore::LogStore::open(lo)?;
    tracing::info!(path = %store.options().path.display(), node = %store.options().node, retain_s = store.options().retain.as_secs(), max_mib = store.options().max_bytes >> 20, "keeps its logs in a log store");
    let keep = store.clone();
    let _ = svc.query_log.keep.set(Box::new(move |q, slow| keep.keep_query(q, slow)));
    svc.log_store = Some(store);
    Ok(())
}

/// Asks the replica serving HTTP at `url` to become the primary of a new
/// term, shipping to `replicas` under `write_acks` (`POST
/// /v1/admin/promote`), and answers what it said.
pub async fn promote(url: &str, replicas: &[String], write_acks: &str, force: bool, token: Option<&str>) -> Result<serde_json::Value, String> {
    let mut req = reqwest::Client::new()
        .post(format!("{}/v1/admin/promote", url.trim_end_matches('/')))
        .json(&serde_json::json!({"replicas": replicas, "write_acks": write_acks, "force": force}));
    if let Some(t) = token.filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);
    }
    let r = req.send().await.map_err(|e| format!("{url}: {e}"))?;
    let status = r.status();
    let j: serde_json::Value = r.json().await.map_err(|e| format!("{url}: {status}: {e}"))?;
    match status.is_success() {
        true => Ok(j),
        false => Err(format!("{url}: {status}: {}", j["error"].as_str().map_or_else(|| j.to_string(), String::from))),
    }
}

/// Downloads model `name` ([`strata::embed::KNOWN_MODELS`],
/// [`strata::derive::KNOWN_MODELS`], a cross-encoder of
/// [`cross_encoder::KNOWN_MODELS`], or a Hugging Face repository
/// `owner/name` of an embedding model or a cross-encoder) into `dir`;
/// returns its directory.
pub async fn fetch_model(name: &str, dir: &std::path::Path) -> Result<PathBuf, String> {
    use strata::query::cross_encoder;
    let derives = strata::derive::KNOWN_MODELS.iter().any(|m| m.0.eq_ignore_ascii_case(name));
    let known = || strata::embed::KNOWN_MODELS.iter().chain(strata::derive::KNOWN_MODELS).chain(cross_encoder::KNOWN_MODELS);
    let (local, repo) = match known().find(|m| m.0.eq_ignore_ascii_case(name)) {
        Some(m) => (m.0.to_string(), m.1.to_string()),
        None if name.contains('/') => (name.rsplit('/').next().unwrap_or(name).to_string(), name.to_string()),
        None => {
            let names: Vec<&str> = known().map(|m| m.0).collect();
            return Err(format!("unknown model {name}: one of {}, or a Hugging Face repository owner/name", names.join(", ")));
        }
    };
    let out = dir.join(&local);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let client = reqwest::Client::new();
    let files = if derives { strata::derive::model_files(&local) } else { strata::embed::model_files(&local) };
    for file in files {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let r = client.get(&url).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("{url}: {e}"))?;
        let bytes = r.bytes().await.map_err(|e| format!("{url}: {e}"))?;
        // Written aside and renamed, so that an open model is never
        // overwritten in place.
        let tmp = out.join(format!("{file}.part"));
        std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, out.join(file))).map_err(|e| format!("{}: {e}", out.join(file).display()))?;
        tracing::info!(file, bytes = bytes.len(), "downloaded");
    }
    if derives {
        strata::derive::open(&local, dir).map_err(|e| e.to_string())?;
    } else if !cross_encoder::is_cross_encoder(&out) {
        strata::embed::open(&local, dir).map_err(|e| e.to_string())?;
    } else if let Err(e) = cross_encoder::CrossEncoder::open(&local, &out) {
        match cfg!(feature = "bert") {
            true => return Err(e.to_string()),
            false => tracing::warn!(error = %e, "downloaded a cross-encoder this build cannot run"),
        }
    }
    Ok(out)
}

/// What `strata-server import` does with SQLite files.
#[derive(Default)]
pub struct SqliteImport {
    /// Every file is a SQLite database (`--format sqlite`); otherwise a
    /// file is one when its first bytes say so.
    pub format: bool,
    /// Only these tables.
    pub tables: Vec<String>,
    /// BLOB columns as base64 text ([`strata::sqlite::ImportOptions::blobs_base64`]).
    pub blobs_base64: bool,
    /// Print the plan as JSON, and write nothing.
    pub plan: bool,
}

/// `strata-server import`: loads `files` into the database at `path`, rows
/// from JSON Lines, JSON, CSV or TSV files (`format`, or each file's
/// extension) into its table `table` (the first if `None`), and the tables
/// of SQLite files into the tables of their names, and says what each
/// wrote. A database that does not exist is created with `schema`, or from
/// the first SQLite file's tables; with SQLite files, `schema` holds
/// schemas for the tables they create ([`strata::sqlite::schemas_from_json`]).
#[allow(clippy::too_many_arguments)]
pub fn import_files(path: &Path, schema: Option<&Path>, table: Option<&str>, models: &Path, format: Option<Format>, directory: bool, files: &[PathBuf], sqlite: SqliteImport) -> Result<(), String> {
    let opts = || DbOptions { layout: if directory { Layout::Directory } else { Layout::SingleFile }, build: build_options(), ..DbOptions::default() };
    let is_sqlite = |f: &Path| sqlite.format || (format.is_none() && strata::sqlite::is_sqlite_file(f));
    let any_sqlite = files.iter().any(|f| is_sqlite(f));
    if let Some(f) = files.iter().find(|f| table.is_some() && is_sqlite(f)) {
        return Err(format!("{}: --table is for rows; a SQLite file's tables go to the tables of their names (--tables picks them)", f.display()));
    }
    let text = schema.map(|s| std::fs::read_to_string(s).map_err(|e| format!("read {}: {e}", s.display()))).transpose()?;
    let given = match (&text, schema, any_sqlite) {
        (Some(t), Some(s), true) => strata::sqlite::schemas_from_json(t).map_err(|e| format!("{}: {e}", s.display()))?,
        _ => Vec::new(),
    };
    let sq = strata::sqlite::ImportOptions { tables: sqlite.tables, schemas: given, blobs_base64: sqlite.blobs_base64 };
    let created = |db: Db| {
        for (field, why) in db.open_embedders(models) {
            tracing::warn!(field = %field, error = %why, "cannot embed this field: its vectors must be in the files");
        }
        db
    };
    let mut db = if path.exists() {
        if schema.is_some() && !any_sqlite {
            return Err(format!("{} exists: --schema is only for a new database", path.display()));
        }
        Some(created(Db::open(path, opts()).map_err(|e| format!("open {}: {e}", path.display()))?))
    } else {
        None
    };
    if sqlite.plan {
        let mut plans = Vec::new();
        for file in files {
            if !is_sqlite(file) {
                return Err(format!("{}: --plan is for SQLite files", file.display()));
            }
            plans.push(strata::sqlite::plan(file, db.as_ref(), &sq).map_err(|e| format!("{}: {e}", file.display()))?.to_json());
        }
        let out = if plans.len() == 1 { plans.remove(0) } else { serde_json::Value::Array(plans) };
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        return Ok(());
    }
    for file in files {
        let t0 = std::time::Instant::now();
        if is_sqlite(file) {
            let at = |e: strata::Error| format!("{}: {e}", file.display());
            let src = strata::sqlite::Source::open(file).map_err(at)?;
            let plan = src.plan(db.as_ref(), &sq).map_err(at)?;
            let d = match db.take() {
                Some(d) => d,
                None => {
                    let first = plan.tables.first().ok_or_else(|| format!("{}: no table to import", file.display()))?;
                    created(Db::create_in(path, first.schema.clone(), opts(), models).map_err(|e| format!("create {}: {e}", path.display()))?)
                }
            };
            let r = src.load(&d, &plan);
            db = Some(d);
            let r = r.map_err(at)?;
            for (name, why) in &r.skipped {
                tracing::info!(file = %file.display(), source = %name, "left out: {why}");
            }
            for t in &r.tables {
                let how = if t.bulk { "loaded" } else { "upserted" };
                tracing::info!(file = %file.display(), source = %t.source, table = %t.table, created = t.created, rows = t.rows, how, "imported");
                for n in &t.notes {
                    tracing::warn!(table = %t.table, "{n}");
                }
            }
            tracing::info!(file = %file.display(), tables = r.tables.len(), secs = t0.elapsed().as_secs_f64(), "imported the SQLite file");
            continue;
        }
        let d = match db.take() {
            Some(d) => d,
            None => {
                let (s, text) = schema.zip(text.as_deref()).ok_or_else(|| format!("{} does not exist: pass --schema FILE to create it", path.display()))?;
                let schema = Schema::from_json(text).map_err(|e| format!("{}: {e}", s.display()))?;
                created(Db::create_in(path, schema, opts(), models).map_err(|e| format!("create {}: {e}", path.display()))?)
            }
        };
        let into = table.map_or_else(|| d.schema().table.clone(), String::from);
        let r = strata::ingest::import_file_into(&d, &into, file, format).map_err(|e| format!("{}: {e}", file.display()));
        db = Some(d);
        let r = r?;
        let how = if r.bulk { "loaded" } else { "upserted" };
        tracing::info!(file = %file.display(), rows = r.rows, how, secs = t0.elapsed().as_secs_f64(), "imported");
        if !r.unknown.is_empty() {
            tracing::warn!(file = %file.display(), fields = %r.unknown.join(", "), "skipped fields the schema lacks");
        }
    }
    let db = db.ok_or("no files to import")?;
    db.storage().merge(&MergePolicy::default()).map_err(|e| format!("merge: {e}"))?;
    tracing::info!(db = %path.display(), rows = db.snapshot().n_live_docs(), "imported every file");
    Ok(())
}

/// Writes the database at `path` to a new SQLite file `out`
/// ([`strata::sqlite::export`]).
pub fn export_sqlite(path: &Path, out: &Path, opts: &strata::sqlite::ExportOptions) -> Result<(), String> {
    let db = Db::open(path, DbOptions::default()).map_err(|e| format!("open {}: {e}", path.display()))?;
    let t0 = std::time::Instant::now();
    let r = strata::sqlite::export(&db, out, opts).map_err(|e| format!("{}: {e}", out.display()))?;
    for t in &r.tables {
        tracing::info!(table = %t.table, rows = t.rows, indexes = t.indexes.len(), fts = t.fts.as_deref().unwrap_or("none"), "exported");
        for n in &t.notes {
            tracing::info!(table = %t.table, "{n}");
        }
    }
    tracing::info!(out = %out.display(), tables = r.tables.len(), secs = t0.elapsed().as_secs_f64(), "exported to SQLite");
    Ok(())
}

/// Releases expired snapshot leases, and reports merges that fail (the
/// database's merger threads merge as writes add segments; a database
/// that cannot write, [`Service::failing`], merges nothing).
pub fn spawn_maintenance(svc: &Arc<Service>) {
    let Some(db) = svc.local_db().cloned() else { return };
    let svc = Arc::downgrade(svc);
    tokio::spawn(async move {
        let mut reported = None;
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let Some(svc) = svc.upgrade() else { return };
            svc.sweep_leases();
            let why = svc.failing().map(|w| format!("{w}; merges stop until then")).or_else(|| db.storage().merge_status().error.map(|e| format!("merge failed: {e}")));
            if why.is_some() && why != reported {
                tracing::error!(error = why.as_deref().unwrap_or_default(), "the database cannot merge");
            }
            reported = why;
        }
    });
}

/// Samples the service's counters and gauges every interval of its
/// series ([`crate::telemetry::Series`]), for the UI's charts, until the
/// service is dropped.
pub fn spawn_sampler(svc: &Arc<Service>) {
    let every = svc.series.interval;
    let svc = Arc::downgrade(svc);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(every);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Some(svc) = svc.upgrade() else { return };
            let s = svc.sample();
            if let Some(point) = svc.series.add(s) {
                let changes = svc.alerts.check(&point);
                if !changes.is_empty() {
                    let svc = svc.clone();
                    tokio::spawn(async move { svc.alerts.notify(&changes).await });
                }
            }
        }
    });
}

/// Serves the HTTP API and UI, over TLS with `access.tls`.
pub async fn serve_http(svc: Arc<Service>, listener: TcpListener, access: Access) -> std::io::Result<()> {
    let router = http::router(svc, &access);
    let Some(tls) = &access.tls else {
        return axum::serve(listener, router).with_graceful_shutdown(shutdown()).await;
    };
    let config = auth::server_config(tls).map_err(std::io::Error::other)?;
    let acceptor = tokio_rustls::TlsAcceptor::from(config);
    let service = hyper_util::service::TowerToHyperService::new(router);
    let stop = shutdown();
    tokio::pin!(stop);
    loop {
        let (tcp, _) = tokio::select! {
            r = listener.accept() => match r {
                Ok(c) => c,
                // Out of descriptors, say: wait rather than spin.
                Err(_) => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            _ = &mut stop => return Ok(()),
        };
        let _ = tcp.set_nodelay(true);
        let (acceptor, service) = (acceptor.clone(), service.clone());
        tokio::spawn(async move {
            // A client that never finishes its handshake holds a task, not
            // the accept loop.
            let Ok(Ok(stream)) = tokio::time::timeout(Duration::from_secs(10), acceptor.accept(tcp)).await else { return };
            let io = hyper_util::rt::TokioIo::new(stream);
            let _ = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new()).serve_connection_with_upgrades(io, service).await;
        });
    }
}

/// Serves gRPC: every call needs `access.cluster`, `access.token` or a key
/// of the scope its method needs ([`crate::keys`]), over TLS with
/// `access.tls`.
pub async fn serve_grpc(svc: Arc<Service>, listener: TcpListener, access: Access) -> Result<(), tonic::transport::Error> {
    use tokio_stream::StreamExt;
    // A gRPC answer is several small writes (headers, data, trailers); with
    // Nagle's algorithm on, the last waits for the client's delayed ACK and
    // every call to a shard took 40 ms longer.
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener).map(|s| {
        if let Ok(s) = &s {
            let _ = s.set_nodelay(true);
        }
        s
    });
    let (token, cluster) = (access.token.clone(), access.cluster.clone());
    let counted = svc.clone();
    // Who the call is from, for each method to check its scope.
    let check = move |mut req: tonic::Request<()>| match counted.keys.resolve(token.as_ref(), cluster.as_ref(), req.metadata().get("authorization").and_then(|v| v.to_str().ok())) {
        Some(who) => {
            req.extensions_mut().insert(who);
            Ok(req)
        }
        None => {
            counted.metrics.grpc("any", "Unauthenticated", 0.0);
            // The method is not known yet: the path is the service's.
            counted.audit.add(crate::keys::Entry {
                ts_ms: crate::telemetry::now_ms(),
                who: "unknown".into(),
                key: None,
                scope: None,
                method: "gRPC".into(),
                route: "any".into(),
                status: 401,
                ms: 0.0,
                request_id: None,
                detail: None,
            });
            Err(tonic::Status::unauthenticated("missing or wrong token"))
        }
    };
    // A Write message carries rows, as an HTTP write's body does.
    let max_body = svc.max_body;
    let service = pb::strata_server::StrataServer::new(grpc::Grpc(svc)).max_decoding_message_size(max_body).max_encoding_message_size(256 << 20);
    let mut builder = tonic::transport::Server::builder();
    if let Some(tls) = &access.tls {
        builder = builder.tls_config(tonic::transport::ServerTlsConfig::new().identity(tonic::transport::Identity::from_pem(&tls.cert, &tls.key)))?;
    }
    builder
        .add_service(tonic::service::interceptor::InterceptedService::new(service, check))
        .serve_with_incoming_shutdown(incoming, shutdown())
        .await
}

/// Ctrl-C, or SIGTERM: how Docker and Kubernetes stop a container. As a
/// container's first process the server would otherwise ignore SIGTERM, and
/// be killed only when the grace period ran out (30 s in Kubernetes).
async fn shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}
