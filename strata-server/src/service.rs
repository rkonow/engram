//! The operations both front ends serve, over a local database or, as a
//! coordinator, over shard servers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value as Json};
use strata::db::{Db, PreparedWrite, Source};
use strata::embed::Embedder;
use strata::ingest::{Format, Rows};
use strata::query::exec::{ExecOptions, ResultSet};
use strata::query::memory::MemoryBudget;
use strata::query::stats::TermStats;
use strata::query::ir::Plan;
use strata::query::model::Models;
use strata::query::sql::{ident, quoted};
use strata::query::{bind_command, compound, ops, profile, steps, Command, Params, Select};
use strata::rank_log::{self, Log};
use strata::schema::Schema;
use strata::storage::{CheckpointOptions, CheckpointStats, Layout, RankProfile, Snapshot, TableTxn, WriteTxn};
use strata::text::Synonyms;
use strata::value::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Snapshot leases held at once at most; a request for one more is
/// refused (429) until some expire.
pub const MAX_LEASES: usize = 10_000;

use crate::coord::Coordinator;
use crate::idem;
use crate::replication::{Acks, Follow, Primary, PrimaryOptions, READ_WAIT};
use crate::slots::{self, SlotMap};
use crate::metrics::{self, Family, Metrics};
use crate::alerts::Alerts;
use crate::keys::{self, Audit, Keys, Principal, Scope};
use crate::telemetry::{self, LiveQueries, QueryLog, Sample, Series};
use crate::{json as js, obs, pb, ApiError};

mod failover;
mod moving;

pub use failover::tuned as failover_tuned;
pub(crate) use moving::Moving;
pub use moving::FREEZE_TTL;

tokio::task_local! {
    /// The cluster map epoch the request being served routes by, as
    /// [`Service::check_epoch`] checked it (0: none): a read takes its
    /// version and a write checks its keys against the map then.
    pub static ROUTED: std::cell::Cell<u64>;
}

/// The epoch the request being served routes by ([`ROUTED`]).
fn routed() -> u64 {
    ROUTED.try_with(std::cell::Cell::get).unwrap_or(0)
}

/// Binds one SELECT into its plan, and gives back what the executor runs
/// of it: every query runs as a plan ([`strata::query::ir`]).
fn bind(sql: &str, schema: &Schema, params: &Params) -> strata::Result<Select> {
    Plan::bind(sql, schema, params)?.to_select(schema)
}

pub enum Backend {
    Local(Arc<Db>),
    Cluster(Arc<Coordinator>),
}

/// A SELECT to run.
#[derive(Clone, Default)]
pub struct QueryReq {
    pub sql: String,
    pub params: Params,
    /// 0: the server's default.
    pub timeout_ms: u64,
    /// Return results a coordinator can merge ([`ExecOptions::shard`]).
    pub shard: bool,
    /// Statistics summed over shards, for global BM25 scores.
    pub stats: Option<TermStats>,
    /// A snapshot lease; 0 for the latest version.
    pub snapshot: u64,
    /// From a coordinator: answer without shards that cannot answer
    /// (`None`: its default, [`crate::coord::CoordOptions::partial`]).
    pub partial: Option<bool>,
    /// A plan to run instead of `sql` (which then only names it in logs):
    /// a leaf of a coordinator's compound query.
    pub plan: Option<Arc<Plan>>,
    /// Keep the snapshot read, `snapshot` or a new lease, for a fetch that
    /// follows ([`Answer::lease`]).
    pub hold: bool,
    /// End the lease `snapshot` once answered.
    pub release: bool,
    /// The writes the query must see ([`After`]).
    pub after: After,
    /// Measure each step's CPU time ([`ExecOptions::profile`]).
    pub profile: bool,
    /// Not kept among the latest queries, nor in the statements'
    /// statistics: a run of a CPU profile's query ([`crate::profiler`]),
    /// which runs it thousands of times. Still counted in the metrics.
    pub quiet: bool,
}

/// The writes a read must see, so that a client reads its own writes
/// from any copy: the sequence number a server answered its last write
/// with (`min_seq_no`), or through a coordinator, the number each shard
/// answered (`min_seq_nos`, 0 for a shard not written). A replica waits
/// until it has applied that write, up to [`crate::replication::READ_WAIT`],
/// and refuses the read otherwise (503), so that a coordinator asks the
/// shard's primary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum After {
    #[default]
    Any,
    Seq(u64),
    Shards(Vec<u64>),
}

impl After {
    /// From a request's `min_seq_no` and `min_seq_nos` (0 and empty: not
    /// given), which are not both given.
    pub fn of(min_seq_no: u64, min_seq_nos: Vec<u64>) -> Result<After, ApiError> {
        match (min_seq_no, min_seq_nos.is_empty()) {
            (0, true) => Ok(After::Any),
            (n, true) => Ok(After::Seq(n)),
            (0, false) => Ok(After::Shards(min_seq_nos)),
            _ => Err(ApiError::bad("a read takes min_seq_no (a server's sequence number) or min_seq_nos (a coordinator's, one per shard), not both")),
        }
    }

    /// The number a read on a server must see (0: none).
    fn seq(&self) -> Result<u64, ApiError> {
        match self {
            After::Any => Ok(0),
            After::Seq(n) => Ok(*n),
            After::Shards(_) => Err(ApiError::bad("min_seq_nos is a coordinator's, one number per shard; a server takes min_seq_no, the number its write answered")),
        }
    }
}

pub struct Answer {
    pub rs: ResultSet,
    pub took_ms: f64,
    pub kind: &'static str,
    /// Shards a coordinator answered without (see [`QueryReq::partial`]).
    pub missing: Vec<String>,
    /// The lease on the snapshot read, with [`QueryReq::hold`]; else 0.
    pub lease: u64,
}

/// An embedded VECTOR field ([`strata::Embed`]) and its model, which
/// computes the vectors writes leave out and embeds `{"embed": "text"}`
/// parameters.
#[derive(Clone)]
pub struct FieldModel {
    pub table: String,
    pub field: String,
    pub spec: strata::Embed,
    /// The model, or why it did not open (then writes must supply the
    /// vectors, and text cannot be searched against the field).
    pub model: Result<Arc<dyn Embedder>, String>,
}

/// The agent's model settings ([`crate::agent`]).
#[derive(Clone)]
pub struct AgentConfig {
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: String,
}

pub struct Service {
    pub backend: Backend,
    pub schema: Schema,
    pub metrics: Metrics,
    pub agent: AgentConfig,
    /// The `StrataCluster` that runs this coordinator on Kubernetes, if one
    /// does ([`crate::operator::kube::Link`]).
    pub kube: Option<crate::operator::kube::Link>,
    /// The queries running now ([`telemetry::LiveQueries`]).
    pub live: Arc<LiveQueries>,
    /// The latest and slow queries, and statistics per statement.
    pub query_log: QueryLog,
    /// Rates and gauges every few seconds ([`Self::sample`]).
    pub series: Series,
    /// When the server started.
    pub started: std::time::SystemTime,
    /// The API keys ([`crate::keys`]), the audit log, and the alert rules
    /// the sampler checks.
    pub keys: Arc<Keys>,
    pub audit: Audit,
    pub alerts: Alerts,
    /// The server's log store ([`crate::logstore`]): its log lines and the
    /// queries it answered, in a Strata database of their own (`None`:
    /// `--log-store off`, or a coordinator without one).
    pub log_store: Option<Arc<crate::logstore::LogStore>>,
    /// The embedded VECTOR fields' models, of every table
    /// ([`Service::open_models`], [`Service::models`]).
    models: RwLock<Arc<Vec<FieldModel>>>,
    /// Where [`Service::open_models`] opened them, for tables created
    /// later, and where ranking models are.
    models_dir: Option<PathBuf>,
    /// A coordinator's ranking models ([`Service::open_ranking_models`]);
    /// a database keeps its own.
    ranking: Models,
    /// Default query deadline.
    pub timeout: Duration,
    /// The longest deadline a request may ask for (`timeout_ms`).
    pub max_timeout: Duration,
    /// Queries and writes that take this long or longer are logged as
    /// slow (`None`: none are), unless changed while the server runs
    /// ([`Self::slow_after`]).
    pub slow: Option<Duration>,
    /// Profiling settings changed at run time: the slow threshold, and
    /// whether every query's CPU time is measured.
    pub profiling: telemetry::Profiling,
    /// The most rows a query may reach, OFFSET included
    /// ([`ExecOptions::max_rows`]; `None`: no limit).
    pub max_rows: Option<usize>,
    /// The most groups a query may make ([`ExecOptions::max_groups`]).
    pub max_groups: Option<usize>,
    /// The most rows a cross-encoder's rerank phase may score
    /// (`--max-rerank`; `None`: no limit).
    pub max_rerank: Option<usize>,
    /// The memory queries may hold at once ([`ExecOptions::memory`];
    /// `None`: not counted).
    pub query_memory: Option<Arc<MemoryBudget>>,
    /// The most write requests in the write path at once (embedding,
    /// waiting for the writer or being applied); more are refused with 429.
    pub max_queued_writes: usize,
    /// The largest body `/v1/write` and `/v1/import` (and a gRPC message)
    /// may have, in bytes; other routes take [`SMALL_BODY`].
    pub max_body: usize,
    pub lease_ttl: Duration,
    leases: Mutex<HashMap<u64, (Snapshot, Instant)>>,
    next_lease: AtomicU64,
    /// Queries allowed to run at once; the rest wait until their deadline.
    admission: Arc<Semaphore>,
    pub max_concurrent: usize,
    /// Texts embedded at once (one per core): writes' and queries'.
    embedding: Arc<Semaphore>,
    cores: usize,
    /// Queries running, counted until their work ends ([`Running`]).
    running: Arc<AtomicUsize>,
    /// The queue of a local database's writer thread ([`writer`]).
    writes: Option<mpsc::Sender<Queued>>,
    writer_state: Arc<WriterState>,
    /// Idempotency keys of the writes it applied ([`crate::idem`]).
    idempotency: Arc<idem::Keys>,
    /// The synonym list `synonyms => 1` searches use, and its file.
    synonyms: RwLock<Option<Arc<Synonyms>>>,
    synonyms_file: Option<PathBuf>,
    /// Where `POST /v1/admin/checkpoint` writes checkpoints.
    backup_dir: Option<PathBuf>,
    /// The slot map a coordinator gave this shard and its place in it
    /// ([`Service::set_slots`]), and the file that keeps them (a new map
    /// is set while its lock is held).
    slots: RwLock<Option<(SlotMap, usize)>>,
    /// The oldest map epoch under which this shard holds the slots it
    /// holds now: a coordinator routing by a map that old still reads and
    /// writes here correctly ([`Service::check_epoch`]).
    slots_since: AtomicU64,
    slots_file: tokio::sync::Mutex<Option<PathBuf>>,
    /// Its part in moving slots between shards ([`moving`]).
    pub(crate) moving: Moving,
    /// Its part in replication ([`crate::replication`]), which a replica's
    /// promotion changes ([`Self::promote`]).
    replication: RwLock<Replication>,
    /// How a replica ships once promoted, but for its replicas and
    /// `--write-acks` (none: as [`PrimaryOptions::default`]).
    pub promote_options: Option<PrimaryOptions>,
    /// Held while a promotion or a change of its part runs.
    promoting: tokio::sync::Mutex<()>,
    /// Where its part in replication is kept (`<db>.replication`).
    replication_file: Option<PathBuf>,
    /// How long a replica hears nothing from its primary before it asks
    /// its shard's other copies to elect it (`--failover-after-ms`; zero:
    /// never), as [`Self::start_failover`] does.
    pub failover_after: Duration,
}

/// A server's part in replication.
#[derive(Clone, Default)]
pub enum Replication {
    #[default]
    None,
    /// Ships its writes to its replicas (`--replicas`).
    Primary(Arc<Primary>),
    /// Applies its primary's writes and takes none from clients
    /// (`--replica`).
    Replica(Arc<Follow>),
}

impl Service {
    pub fn new(backend: Backend, schema: Schema, agent: AgentConfig, max_concurrent: usize, timeout: Duration) -> Service {
        // Steps count from this moment, which must come before any
        // request's start for them to line up with its trace.
        steps::epoch();
        // With the counting allocator, a profiled query's steps carry the
        // bytes they allocated, and the heap's peak is kept.
        if crate::heap::installed() {
            steps::count_allocations(crate::heap::thread_allocated);
            crate::heap::watch();
        }
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let idempotency = Arc::new(idem::Keys::default());
        let writer_state = Arc::new(WriterState::default());
        let writes = match &backend {
            Backend::Local(db) => {
                let (tx, rx) = mpsc::channel();
                let (db, state) = (db.clone(), writer_state.clone());
                std::thread::Builder::new().name("strata-writer".into()).spawn(move || writer(&db, rx, &state)).expect("spawn the writer thread");
                Some(tx)
            }
            Backend::Cluster(_) => None,
        };
        Service {
            backend,
            schema,
            metrics: Metrics::default(),
            agent,
            kube: None,
            live: Arc::new(LiveQueries::default()),
            query_log: QueryLog::default(),
            series: Series::default(),
            started: std::time::SystemTime::now(),
            keys: Arc::new(Keys::memory()),
            audit: Audit::new(None).expect("an audit log in memory"),
            alerts: Alerts::new(None, None).expect("the default alert rules"),
            log_store: None,
            models: RwLock::new(Arc::new(Vec::new())),
            models_dir: None,
            ranking: Models::default(),
            timeout,
            max_timeout: timeout.max(Duration::from_secs(60)),
            slow: None,
            profiling: telemetry::Profiling::default(),
            max_rows: None,
            max_rerank: None,
            max_groups: None,
            query_memory: None,
            max_queued_writes: usize::MAX,
            max_body: 256 << 20,
            lease_ttl: Duration::from_secs(60),
            leases: Mutex::new(HashMap::new()),
            next_lease: AtomicU64::new(1),
            admission: Arc::new(Semaphore::new(max_concurrent.max(1))),
            max_concurrent: max_concurrent.max(1),
            embedding: Arc::new(Semaphore::new(cores)),
            cores,
            running: Arc::new(AtomicUsize::new(0)),
            writes,
            writer_state,
            idempotency,
            synonyms: RwLock::new(None),
            synonyms_file: None,
            backup_dir: None,
            slots: RwLock::new(None),
            slots_since: AtomicU64::new(0),
            slots_file: tokio::sync::Mutex::new(None),
            moving: Moving::default(),
            replication: RwLock::new(Replication::None),
            promote_options: None,
            promoting: tokio::sync::Mutex::new(()),
            replication_file: None,
            failover_after: Duration::ZERO,
        }
    }

    /// Its part in replication now.
    pub fn replication(&self) -> Replication {
        self.replication.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Sets its part in replication, before it serves.
    pub fn set_replication(&mut self, r: Replication) {
        *self.replication.get_mut().unwrap_or_else(|e| e.into_inner()) = r;
    }

    /// Its role in replication, as `Hello` says it.
    pub fn role(&self) -> pb::Role {
        match &self.replication() {
            Replication::None => pb::Role::Standalone,
            Replication::Primary(_) => pb::Role::Primary,
            Replication::Replica(_) => pb::Role::Replica,
        }
    }

    /// The newer term of a primary that replaced this one, as a primary
    /// (0: none).
    pub fn replaced_by(&self) -> u64 {
        match &self.replication() {
            Replication::Primary(p) => p.replaced_by(),
            _ => 0,
        }
    }

    /// Refuses a client's write on a replica (409), and one a primary may
    /// not apply now ([`Self::admit_write`]).
    fn writable(&self) -> Result<(), ApiError> {
        self.takes_writes()?;
        self.admit_write()
    }

    /// Refuses a client's write on a replica (409).
    fn takes_writes(&self) -> Result<(), ApiError> {
        match &self.replication() {
            Replication::Replica(_) => Err(conflict("this server is a replica: it applies its primary's writes and takes none from clients; write to the primary".into())),
            _ => Ok(()),
        }
    }

    /// Refuses a write a primary may not apply now ([`Primary::admit`]).
    fn admit_write(&self) -> Result<(), ApiError> {
        match &self.replication() {
            Replication::Primary(p) => p.admit(),
            _ => Ok(()),
        }
    }

    /// Answers a committed write once the copies `--write-acks` asks for
    /// hold it, waiting up to `timeout` ([`Primary::acknowledged`]).
    async fn replicated(&self, w: Written, timeout: Duration) -> Result<Written, ApiError> {
        if let (Replication::Primary(p), Some(seq)) = (&self.replication(), w.seq_no) {
            p.acknowledged(seq, Instant::now() + timeout).await?;
        }
        Ok(w)
    }

    /// Applies a batch its primary shipped (`Replicate`), on a replica.
    pub async fn replicate(&self, r: pb::ReplicateRequest) -> Result<pb::ReplicateResponse, ApiError> {
        match &self.replication() {
            Replication::Replica(f) => f.replicate(r).await,
            Replication::Primary(p) => p.shipped_to(&r.primary, r.term),
            Replication::None => Err(conflict("this server is not a replica (--replica): it takes no shipped transactions".into())),
        }
    }

    /// Installs a copy of its primary (`Restore`), on a replica.
    pub async fn restore(&self, chunks: tonic::Streaming<pb::RestoreChunk>) -> Result<pb::ReplicateResponse, ApiError> {
        match &self.replication() {
            Replication::Replica(f) => f.restore(chunks).await,
            Replication::Primary(p) => p.copied_to(chunks).await,
            Replication::None => Err(conflict("this server is not a replica (--replica): it takes no copy of a primary".into())),
        }
    }

    /// A replica's digests of its rows at the version its primary pinned
    /// (`Verify`).
    pub async fn verify_for(&self, r: pb::VerifyRequest) -> Result<pb::VerifyResponse, ApiError> {
        match &self.replication() {
            Replication::Replica(f) => f.verify(r).await,
            Replication::Primary(p) => {
                p.shipped_to(&r.primary, r.term)?;
                Err(conflict("this server is a primary: it has no rows of another primary's to compare".into()))
            }
            Replication::None => Err(conflict("this server is not a replica (--replica): it has no rows of a primary's to compare".into())),
        }
    }

    /// Compares each replica's rows with this primary's now
    /// ([`Primary::verify`]), waiting up to `timeout`.
    pub async fn verify_replicas(&self, timeout: Duration) -> Result<Json, ApiError> {
        match &self.replication() {
            Replication::Primary(p) => Ok(p.verify(Instant::now() + timeout).await),
            _ => Err(conflict("this server is not a primary (--replicas): it has no replicas to compare".into())),
        }
    }

    /// The shard's copies, as this replica knows them: its `--replicas`,
    /// and the servers its cluster map names for its shard (which only the
    /// cluster's token sets). The only servers a promotion asked for with
    /// a key calls, and so sends the cluster's token to.
    pub fn known_copies(&self) -> Vec<String> {
        let mut known = self.promote_options.as_ref().map(|o| o.replicas.clone()).unwrap_or_default();
        if let Some((m, shard)) = self.slots() {
            for url in m.replicas().get(shard).into_iter().flatten() {
                if !known.contains(url) {
                    known.push(url.clone());
                }
            }
        }
        known
    }

    /// Makes this replica the primary of a new term, shipping to
    /// `replicas` under `acks` ([`Follow::promote`]), once a majority of
    /// its shard's copies voted for it: the servers its cluster map names
    /// for its shard, or else `replicas`. With no `replicas`, it ships to
    /// the shard's other copies, or without a cluster map to its
    /// `--replicas`. With `force`, it asks for no votes, and is promoted
    /// even when a copy holds transactions it does not. Promotion calls
    /// each copy with the cluster's token, so a key's promotion may name
    /// only [`Self::known_copies`]; the token's holder, who could send the
    /// token anywhere anyway, may name others.
    pub async fn promote(&self, who: &Principal, replicas: Vec<String>, acks: Acks, force: bool) -> Result<Json, ApiError> {
        who.may(Scope::Admin, "a promotion")?;
        let known = self.known_copies();
        let unknown: Vec<&String> = replicas.iter().filter(|u| !known.contains(u)).collect();
        if !who.root && !unknown.is_empty() {
            let known = match known.is_empty() {
                true => "none: start the replica with --replicas naming them, or add it to a cluster".to_string(),
                false => known.join(", "),
            };
            return Err(ApiError {
                status: 403,
                message: format!(
                    "{} {} not among this shard's copies, which promotion calls with the cluster's token; known copies: {known}",
                    unknown.iter().map(|u| u.as_str()).collect::<Vec<_>>().join(", "),
                    if unknown.len() == 1 { "is" } else { "are" }
                ),
            });
        }
        let _one = self.promoting.lock().await;
        let Replication::Replica(f) = self.replication() else {
            return Err(conflict("this server is not a replica (--replica): only a replica is promoted".into()));
        };
        let voters = self.slots().and_then(|(m, shard)| m.replicas().get(shard).cloned()).filter(|v| v.len() > 1);
        // None named, and no cluster map: the copies it was started with.
        let replicas = match (replicas.is_empty(), &voters) {
            (true, None) => self.promote_options.as_ref().map(|o| o.replicas.clone()).unwrap_or_default(),
            _ => replicas,
        };
        keys::note(format!(
            "promote, shipping to {}",
            match (replicas.is_empty(), &voters) {
                (false, _) => replicas.join(", "),
                (true, Some(_)) => "the shard's other copies".to_string(),
                (true, None) => "no replica".to_string(),
            }
        ));
        let opts = PrimaryOptions { replicas, acks, ..self.primary_options() };
        let (p, answer) = f.promote(opts, voters, force).await?;
        *self.replication.write().unwrap_or_else(|e| e.into_inner()) = Replication::Primary(p);
        Ok(answer)
    }

    /// How this server ships once promoted ([`Self::promote_options`]),
    /// with the lease on reads automatic failover needs.
    fn primary_options(&self) -> PrimaryOptions {
        failover::tuned(self.promote_options.clone().unwrap_or_default(), self.failover_after)
    }

    /// Sets this server's part in replication while it serves
    /// (`Configure`), kept beside its database: `Primary` ships to
    /// `replicas` under `acks` (as now when `None`, else all) from now on,
    /// starting to ship on a standalone server; `Replica` makes a
    /// standalone server wait for a primary, whose copy replaces its rows
    /// (refused while it holds rows, unless `force`), or a primary another
    /// replaced (any primary with `force`) follow the new one;
    /// `Standalone` ends replication (a replica's only with `force`).
    /// A replica is made primary by promotion ([`Self::promote`]), not
    /// here.
    pub async fn configure(&self, role: pb::Role, replicas: Vec<String>, acks: Option<Acks>, force: bool) -> Result<pb::ConfigureResponse, ApiError> {
        let _one = self.promoting.lock().await;
        let db = self.local().cloned().ok_or_else(|| ApiError::bad("a coordinator has no part in replication: configure its shards' servers"))?;
        let file = self.replication_file.clone();
        let now = self.replication();
        let next = match (role, &now) {
            (pb::Role::Primary, Replication::Primary(p)) => {
                let acks = acks.unwrap_or(p.acks());
                match p.replicas() == replicas.as_slice() && p.acks() == acks {
                    true => None,
                    false => Some(Replication::Primary(p.reconfigure(replicas, acks)?)),
                }
            }
            (pb::Role::Primary, Replication::None) => {
                if db.storage().numbering().is_none() {
                    return Err(ApiError::bad("a primary numbers its writes, and this server's database does not: start it with --seq-no"));
                }
                let opts = PrimaryOptions { replicas, acks: acks.unwrap_or_default(), joining: true, ..self.primary_options() };
                let p = Primary::start(db.clone(), file.as_deref(), opts).map_err(|e| conflict(e))?;
                tracing::info!(replicas = %p.replicas().join(", "), write_acks = p.acks().name(), "configured: the primary of its shard");
                Some(Replication::Primary(p))
            }
            (pb::Role::Primary, Replication::Replica(_)) => {
                return Err(conflict("this server is a replica: promote it to make it the primary of a new term (POST /v1/admin/promote)".into()));
            }
            (pb::Role::Replica, Replication::Replica(_)) => None,
            (pb::Role::Replica, Replication::Primary(p)) if p.replaced_by() == 0 && !force => {
                return Err(conflict("this server is its shard's primary, and no primary replaced it: promote another copy first (or pass force)".into()));
            }
            (pb::Role::Replica, _) => {
                if db.storage().numbering().is_none() {
                    return Err(ApiError::bad("a replica numbers its writes as its primary does: start it with --seq-no"));
                }
                let rows = db.snapshot().n_live_docs();
                if matches!(now, Replication::None) && rows > 0 && !force {
                    return Err(conflict(format!("this server holds {rows} rows, which its primary's copy would replace: pass force to drop them")));
                }
                let f = Follow::open(db.clone(), file.as_deref()).map_err(ApiError::bad)?;
                tracing::info!("configured: a replica, which applies what its primary ships");
                Some(Replication::Replica(Arc::new(f)))
            }
            (pb::Role::Standalone, Replication::None) => None,
            (pb::Role::Standalone, Replication::Replica(_)) if !force => {
                return Err(conflict("this server is a replica: on its own it would take writes its primary does not hold; pass force".into()));
            }
            (pb::Role::Standalone, _) => {
                crate::replication::forget_role(file.as_deref()).map_err(|e| ApiError::internal(format!("keep the replication role: {e}")))?;
                tracing::info!("configured: standalone, replicating no more");
                Some(Replication::None)
            }
        };
        if let Some(r) = next {
            *self.replication.write().unwrap_or_else(|e| e.into_inner()) = r;
        }
        Ok(self.configured())
    }

    /// Its part in replication, as `Configure` answers it.
    pub fn configured(&self) -> pb::ConfigureResponse {
        let n = self.local().and_then(|db| db.storage().numbering()).unwrap_or_default();
        let (replicas, write_acks) = match &self.replication() {
            Replication::Primary(p) => (p.replicas().to_vec(), p.acks().name().to_string()),
            _ => (Vec::new(), String::new()),
        };
        pb::ConfigureResponse { role: self.role() as i32, term: n.term, seq_no: n.seq, replicas, write_acks }
    }

    /// Keeps its part in replication in `path` (`<db>.replication`), for
    /// [`Self::configure`].
    pub fn keep_replication_in(&mut self, path: PathBuf) {
        self.replication_file = Some(path);
    }

    /// Keeps the slot map a coordinator gives this shard in `path`, and
    /// reads the one kept there, if any. Without a file, the map lasts
    /// until the server stops.
    pub fn keep_slots_in(&mut self, path: PathBuf) -> Result<(), String> {
        if path.exists() {
            let j: Json = std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())).map_err(|e| format!("{}: {e}", path.display()))?;
            let map = SlotMap::from_json(&j["map"]).map_err(|e| format!("{}: {e}", path.display()))?;
            let shard = j["shard"].as_u64().map(|s| s as usize).filter(|&s| s < map.shards()).ok_or_else(|| format!("{}: no shard of the map", path.display()))?;
            *self.slots_since.get_mut() = j["since"].as_u64().unwrap_or(map.epoch).min(map.epoch);
            self.hide_residue(&map, shard);
            *self.slots.get_mut().unwrap_or_else(|e| e.into_inner()) = Some((map, shard));
        }
        *self.slots_file.get_mut() = Some(path);
        Ok(())
    }

    /// The epoch of this shard's slot map; 0 without one.
    pub fn slot_epoch(&self) -> u64 {
        self.slots.read().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |(m, _)| m.epoch)
    }

    /// The coordinator this server is, for the calls that change the
    /// cluster; a database refuses them (400).
    pub fn coordinator(&self) -> Result<&Arc<Coordinator>, ApiError> {
        match &self.backend {
            Backend::Cluster(c) => Ok(c),
            Backend::Local(_) => Err(ApiError::bad("this server is a database, not a coordinator (--shards): change the cluster through its coordinator")),
        }
    }

    /// This shard's slot map and its place in it, if a coordinator gave one.
    pub fn slots(&self) -> Option<(SlotMap, usize)> {
        self.slots.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Takes `map`, with this server as its shard `shard`, from a
    /// coordinator: a map of an older epoch than this shard's is refused
    /// (409), as is another map of the same epoch. Every row here must be
    /// in the shard's slots, as no rows move between shards yet: a shard
    /// loaded by another map, or by the hash before slots, refuses the map
    /// (409) until it is loaded again. A map that leaves the shard its
    /// slots (another shard's, or servers, changed) keeps serving
    /// coordinators that route by the one before. Returns the epoch now
    /// held.
    pub async fn set_slots(&self, map: SlotMap, shard: usize) -> Result<u64, ApiError> {
        if shard >= map.shards() {
            return Err(ApiError::bad(format!("shard {shard} of a slot map of {} shards", map.shards())));
        }
        let db = self.local().cloned().ok_or_else(|| ApiError::bad("a coordinator holds no slots"))?;
        let file = self.slots_file.lock().await;
        let held = self.slots();
        if let Some((m, s)) = &held {
            if map.epoch < m.epoch {
                return Err(conflict(format!("slot map epoch {} is older than this shard's ({}): the coordinator must load the newer map (--slot-map)", map.epoch, m.epoch)));
            }
            if map.epoch == m.epoch {
                return match *m == map && *s == shard {
                    true => Ok(map.epoch),
                    false => Err(conflict(format!(
                        "this shard holds a different slot map of epoch {}, as shard {s} of {}: list the shards as when it was given, or give a map of a higher epoch (--slot-map)",
                        m.epoch,
                        m.shards()
                    ))),
                };
            }
        }
        let same = held.as_ref().is_some_and(|(m, s)| map.same_slots_as(shard, m, *s));
        let since = match same {
            true => self.slots_since.load(Ordering::Relaxed).min(map.epoch),
            false => map.epoch,
        };
        // A move's next step: the coordinator moved the rows itself.
        let stepped = held.as_ref().is_some_and(|(m, s)| *s == shard && m.shards() == map.shards() && m.moves_to(&map));
        if !same && !stepped {
            let m = map.clone();
            let elsewhere = tokio::task::spawn_blocking(move || rows_elsewhere(&db, &m, shard)).await.map_err(ApiError::internal)??;
            if let Some((n, key)) = elsewhere {
                let other = map.shard_of(&key);
                return Err(conflict(format!(
                    "{n} rows here have keys in other shards' slots under slot map epoch {} (key {} is in slot {}, which shard {other} holds, not {shard}): load the collection again through the coordinator (or with --demo-shard), then connect",
                    map.epoch,
                    js::to_json(&key),
                    slots::slot(&key)
                )));
            }
        }
        if let Some(path) = &*file {
            save_slots(path, &map, shard, since).map_err(|e| ApiError::internal(format!("keep the slot map in {}: {e}", path.display())))?;
        }
        let epoch = map.epoch;
        // The rows a read sees change with the map: a read takes its
        // version under this lock ([`Self::latest`]).
        let mut held = self.slots.write().unwrap_or_else(|e| e.into_inner());
        self.slots_since.store(since, Ordering::Relaxed);
        self.hide_residue(&map, shard);
        *held = Some((map, shard));
        Ok(epoch)
    }

    /// Checks a request's slot map epoch against this shard's: 0 (a
    /// request not from a coordinator with a map) passes, as does an
    /// older epoch under which this shard held the slots it holds;
    /// another older epoch is refused (409), as the coordinator routes by
    /// a map the shards left; a newer one, or one this shard lacks, is
    /// refused as unavailable (503), so that the coordinator asks again
    /// and sends its map.
    pub fn check_epoch(&self, epoch: u64) -> Result<(), ApiError> {
        let _ = ROUTED.try_with(|r| r.set(epoch));
        let held = self.slots.read().unwrap_or_else(|e| e.into_inner());
        self.epoch_ok(held.as_ref().map_or(0, |(m, _)| m.epoch), epoch)
    }

    /// [`Self::check_epoch`] against a map of epoch `held` (0: none).
    fn epoch_ok(&self, held: u64, epoch: u64) -> Result<(), ApiError> {
        if epoch == 0 {
            return Ok(());
        }
        match held {
            e if e == epoch => Ok(()),
            e if e > epoch && epoch >= self.slots_since.load(Ordering::Relaxed) => Ok(()),
            e if e > epoch => Err(conflict(format!("slot map epoch {epoch} is older than this shard's ({e}): the coordinator must load the newer map (--slot-map)"))),
            0 => Err(ApiError { status: 503, message: format!("{NO_MAP_YET}; the request's is epoch {epoch}") }),
            e => Err(ApiError { status: 503, message: format!("this shard has slot map epoch {e}, {OLDER_THAN_REQUEST} {epoch}") }),
        }
    }

    /// Refuses (400) to write a row placed in a slot another shard holds,
    /// when this shard has a slot map: by its key, or by its table's
    /// shard_by value ([`op_place`]), each named by `what` (`key`, or the
    /// shard_by field's name).
    /// Rows of slots moving to or from the shard pass with `moved`, as
    /// the coordinator moving them writes them. A request routed by an
    /// older map that this one changed since it was checked is refused
    /// for its coordinator to route it again (409).
    fn check_keys<'a>(&self, keys: impl IntoIterator<Item = (&'a str, &'a Value)>, moved: bool) -> Result<(), ApiError> {
        let held = self.slots.read().unwrap_or_else(|e| e.into_inner());
        let Some((map, shard)) = &*held else { return Ok(()) };
        for (what, key) in keys {
            let s = slots::slot(key);
            if map.owner(s) == *shard || (moved && map.takes_moved(*shard, s)) {
                continue;
            }
            let e = routed();
            if e > 0 && e < map.epoch {
                return Err(conflict(format!("slot map epoch {e} is older than this shard's ({}), under which slot {s} is shard {}'s: the coordinator must load the newer map", map.epoch, map.owner(s))));
            }
            return Err(ApiError::bad(format!(
                "{what} {} is in slot {s}, which shard {} holds under slot map epoch {}, not this shard ({shard}): write through the coordinator",
                js::to_json(key),
                map.owner(s),
                map.epoch
            )));
        }
        Ok(())
    }

    /// Opens the model of each embedded VECTOR field, of every table, from
    /// `dir` ([`strata::embed::open`]); a local database computes the
    /// vectors its writes leave out with them, and opens those of tables
    /// created later ([`Db::open_embedders`]).
    pub fn open_models(&mut self, dir: &std::path::Path) {
        self.models_dir = Some(dir.to_path_buf());
        if let Backend::Local(db) = &self.backend {
            db.open_embedders(dir);
        }
        self.refresh_models();
    }

    /// `plan` with its ranking models opened ([`strata::query::model`]):
    /// a database's from its models directory, and a coordinator's from
    /// its own, since it scores the shards' rows with them too. The plan
    /// then names each model's hash, which every shard checks against its
    /// own file. A `shard` leaves cross-encoders to its coordinator, which
    /// refuses a phase that would score more rows with one than
    /// `--max-rerank` lets it ([`Service::max_rerank`]), as a database
    /// does.
    pub fn open_ranking_models(&self, plan: Arc<Plan>, shard: bool) -> Result<Arc<Plan>, ApiError> {
        if !plan.root.has_models() {
            return Ok(plan);
        }
        if let Some(max) = self.max_rerank.filter(|_| !shard) {
            let mut over = None;
            plan.root.each_search(&mut |s| {
                for p in &s.phases {
                    let mut cross = false;
                    p.expr.visit(&mut |x| cross |= matches!(x, strata::query::ir::Expr::CrossEncoder { .. }));
                    if cross && p.count > max as u64 {
                        over.get_or_insert(p.count);
                    }
                }
            });
            if let Some(n) = over {
                return Err(ApiError::bad(format!("a cross-encoder scores at most {max} rows of a query here (--max-rerank), not {n}: rerank fewer rows by it")));
            }
        }
        let mut p = (*plan).clone();
        match &self.backend {
            Backend::Local(db) => db.open_ranking_models(&mut p, shard)?,
            Backend::Cluster(_) => self.ranking.open(&self.models_dir.clone().unwrap_or_else(strata::embed::models_dir), &mut p, true)?,
        }
        Ok(Arc::new(p))
    }

    /// The embedded fields' models ([`Service::open_models`]).
    pub fn models(&self) -> Arc<Vec<FieldModel>> {
        self.models.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Lists the model of each embedded field of every table now, keeping
    /// those already listed: after `CREATE TABLE`, and when a coordinator
    /// learns of new tables. Nothing before [`Service::open_models`].
    fn refresh_models(&self) {
        let Some(dir) = &self.models_dir else { return };
        let old = self.models();
        let mut list = Vec::new();
        for t in self.catalog() {
            for f in &t.fields {
                let Some(spec) = f.embed.clone() else { continue };
                if let Some(m) = old.iter().find(|m| m.table.eq_ignore_ascii_case(&t.table) && m.field == f.name && m.spec == spec) {
                    list.push(m.clone());
                    continue;
                }
                let model = match &self.backend {
                    Backend::Local(db) => match db.embedder_in(&t.table, &f.name) {
                        Some(m) => Ok(m),
                        None => strata::embed::open(&spec.model, dir).and_then(|m| db.set_embedder_in(&t.table, &f.name, m.clone()).map(|_| m)),
                    },
                    Backend::Cluster(_) => strata::embed::open(&spec.model, dir),
                };
                list.push(FieldModel { table: t.table.clone(), field: f.name.clone(), spec, model: model.map_err(|e| e.to_string()) });
            }
        }
        *self.models.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(list);
    }

    /// Whether this server embeds any field's text.
    fn embeds(&self) -> bool {
        self.models().iter().any(|m| m.model.is_ok())
    }

    /// The model of embedded field `field` of `table`, or of the first
    /// embedded field that has one: of `table` when it is given, else of
    /// the first table that has such a field, in the database's order.
    pub fn model(&self, table: Option<&str>, field: Option<&str>) -> Result<(String, String, Arc<dyn Embedder>), ApiError> {
        let mut models = self.models();
        // A table created since the models were listed.
        if table.is_some_and(|t| !models.iter().any(|m| m.table.eq_ignore_ascii_case(t))) {
            self.refresh_models();
            models = self.models();
        }
        let of_table = |m: &&FieldModel| table.is_none_or(|t| m.table.eq_ignore_ascii_case(t));
        let named = match table {
            Some(t) => format!("{t}.{}", field.unwrap_or("")),
            None => field.unwrap_or("").to_string(),
        };
        let m = match field {
            Some(f) => models.iter().filter(of_table).find(|m| m.field.eq_ignore_ascii_case(f)).ok_or_else(|| ApiError::bad(format!("{named} is not an embedded VECTOR field; pass a vector")))?,
            None => models.iter().filter(of_table).find(|m| m.model.is_ok()).ok_or_else(|| ApiError::bad("no table has a field this server can embed text for; pass a vector"))?,
        };
        match &m.model {
            Ok(e) => Ok((m.table.clone(), m.field.clone(), e.clone())),
            Err(why) => Err(ApiError::bad(format!("{}'s model {} is not available here ({why}); pass a vector", m.field, m.spec.model))),
        }
    }

    /// Sets how a local database's writes resolve the MEDIA values they
    /// give (`--media-fetch`, `--media-max-bytes`, `--media-store`,
    /// [`strata::db::Db::set_media_resolver`]). A coordinator refuses: its
    /// shards fetch for the writes they take.
    pub fn set_media(&self, r: strata::media::Resolver) -> Result<(), ApiError> {
        match &self.backend {
            Backend::Local(db) => {
                db.set_media_resolver(r);
                Ok(())
            }
            Backend::Cluster(_) => Err(ApiError::bad("--media-fetch, --media-max-bytes and --media-store go to each shard server, which fetches the files of the writes it takes")),
        }
    }

    /// Whether a table has a MEDIA field, whose values writes resolve
    /// before the writer ([`Service::embed_ops`]).
    fn has_media(&self) -> bool {
        self.catalog().iter().any(|t| t.fields.iter().any(|f| f.media.is_some()))
    }

    /// Does the work the documents of `ops` leave to the server before
    /// they queue for the writer: resolves their MEDIA values (fetching
    /// files, [`strata::db::Db::resolve_media_in`]), makes the derived text
    /// and computes the embedded vectors they leave out, so that models run
    /// on many cores rather than in the writer thread, and the rows logged
    /// (and shipped to replicas) hold the results.
    fn embed_ops(&self, db: &Db, ops: &mut [Json]) -> Result<(), ApiError> {
        let models = self.models();
        let writes_doc = |op: &Json| matches!(op.get("op").and_then(|o| o.as_str()), Some("insert" | "upsert" | "upsert_if_newer"));
        let now = strata::timestamp::now();
        let t0 = Instant::now();
        let mut n = 0;
        let first = db.schema().table.clone();
        for schema in db.catalog() {
            let wanted: Vec<usize> = models.iter().filter(|m| m.model.is_ok() && m.table.eq_ignore_ascii_case(&schema.table)).filter_map(|m| schema.field_index(&m.field)).collect();
            let media: Vec<usize> = (0..schema.fields.len()).filter(|&f| schema.fields[f].media.is_some()).collect();
            let derived: Vec<usize> = (0..schema.fields.len()).filter(|&f| schema.fields[f].derive.is_some()).collect();
            if wanted.is_empty() && media.is_empty() {
                continue;
            }
            let mut at = Vec::new();
            let mut docs = Vec::new();
            for (i, op) in ops.iter().enumerate() {
                if writes_doc(op) && op.get("doc").is_some_and(|d| d.is_object()) && op_table(op, &first)?.eq_ignore_ascii_case(&schema.table) {
                    docs.push(js::document(&schema, &op["doc"], now)?);
                    at.push(i);
                }
            }
            let before: Vec<Vec<bool>> = docs.iter().map(|d| wanted.iter().map(|&f| matches!(d[f], Value::Null)).collect()).collect();
            let untold: Vec<Vec<bool>> = docs.iter().map(|d| derived.iter().map(|&f| matches!(d[f], Value::Null)).collect()).collect();
            db.fill_in(&schema.table, &mut docs)?;
            n += before.iter().flatten().filter(|&&m| m).count() as u64;
            for (((i, d), missing), untold) in at.into_iter().zip(&docs).zip(before).zip(untold) {
                for (&f, was_missing) in wanted.iter().zip(missing) {
                    if let (true, Value::Vector(v)) = (was_missing, &d[f]) {
                        ops[i]["doc"][schema.fields[f].name.as_str()] = json!(v.as_ref());
                    }
                }
                for (&f, was_untold) in derived.iter().zip(untold) {
                    if let (true, Value::Str(t)) = (was_untold, &d[f]) {
                        ops[i]["doc"][schema.fields[f].name.as_str()] = json!(t);
                    }
                }
                for &f in &media {
                    if let Value::Str(t) = &d[f] {
                        ops[i]["doc"][schema.fields[f].name.as_str()] = json!(t);
                    }
                }
            }
        }
        if n > 0 {
            self.metrics.embedded("write", n, t0.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Reads the synonym list at `path` if it exists; `PUT /v1/synonyms`
    /// writes it there.
    pub fn open_synonyms(&mut self, path: &Path) -> Result<(), ApiError> {
        if let Backend::Cluster(_) = self.backend {
            return Err(ApiError::bad("--synonyms goes to each shard server, which expands its own searches"));
        }
        self.synonyms_file = Some(path.to_path_buf());
        match std::fs::read_to_string(path) {
            Ok(text) => self.set_synonyms_text(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(ApiError::bad(format!("read {}: {e}", path.display()))),
        }
    }

    /// The synonym list, if one is loaded.
    pub fn synonyms(&self) -> Option<Arc<Synonyms>> {
        self.synonyms.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_synonyms_text(&self, text: &str) -> Result<(), ApiError> {
        let s = Synonyms::parse(text)?;
        *self.synonyms.write().unwrap_or_else(|e| e.into_inner()) = (!s.is_empty() || !text.trim().is_empty()).then(|| Arc::new(s));
        Ok(())
    }

    /// Replaces the synonym list (and its file, if the server has one).
    pub fn put_synonyms(&self, text: &str) -> Result<usize, ApiError> {
        if let Backend::Cluster(_) = self.backend {
            return Err(ApiError::bad("a coordinator has no synonym list: each shard server takes --synonyms FILE"));
        }
        let parsed = Synonyms::parse(text)?;
        if let Some(path) = &self.synonyms_file {
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| ApiError::internal(format!("write {}: {e}", path.display())))?;
        }
        let rules = parsed.rules();
        *self.synonyms.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(parsed));
        Ok(rules)
    }

    /// Where the synonym list is saved, if anywhere.
    pub fn synonyms_file(&self) -> Option<&Path> {
        self.synonyms_file.as_deref()
    }

    /// Sets where `POST /v1/admin/checkpoint` writes checkpoints, creating
    /// the directory if need be.
    pub fn set_backup_dir(&mut self, dir: &Path) -> Result<(), ApiError> {
        if let Backend::Cluster(_) = self.backend {
            return Err(ApiError::bad("--backup-dir goes to each shard server, which checkpoints its own database"));
        }
        std::fs::create_dir_all(dir).map_err(|e| ApiError::bad(format!("create {}: {e}", dir.display())))?;
        self.backup_dir = Some(dir.to_path_buf());
        Ok(())
    }

    /// Writes a checkpoint of the database ([`strata::storage::Database::checkpoint`])
    /// named `name` in the backup directory, hard-linking the segments an
    /// earlier checkpoint there (`link_from`) already holds.
    pub async fn checkpoint(&self, name: &str, layout: Layout, link_from: Option<&str>) -> Result<CheckpointStats, ApiError> {
        let dir = self.backup_dir.as_ref().ok_or_else(|| ApiError::bad("checkpoints need the server started with --backup-dir"))?;
        let db = self.admin_db()?;
        let dest = dir.join(checkpoint_name(name)?);
        let link_from = link_from.map(|n| checkpoint_name(n).map(|n| dir.join(n))).transpose()?;
        let opts = CheckpointOptions { layout, link_from };
        tokio::task::spawn_blocking(move || db.storage().checkpoint(&dest, &opts)).await.map_err(ApiError::internal)?.map_err(Into::into)
    }

    /// Checks every byte of the database against its checksums
    /// ([`strata::storage::Database::check`]). Writes wait meanwhile.
    pub async fn check(&self) -> Result<(), ApiError> {
        let db = self.admin_db()?;
        let deadline = Instant::now() + self.max_timeout;
        tokio::task::spawn_blocking(move || wait_for_writer(deadline, || Ok(db.storage().check()?))).await.map_err(ApiError::internal)?
    }

    /// Returns free space to the file system ([`strata::storage::Database::vacuum`]);
    /// the bytes the file shrank by.
    pub async fn vacuum(&self) -> Result<u64, ApiError> {
        let db = self.admin_db()?;
        let deadline = Instant::now() + self.max_timeout;
        tokio::task::spawn_blocking(move || wait_for_writer(deadline, || Ok(db.storage().vacuum()?))).await.map_err(ApiError::internal)?
    }

    fn admin_db(&self) -> Result<Arc<Db>, ApiError> {
        self.local_db().cloned().ok_or_else(|| ApiError::bad("a coordinator has no database of its own: ask each shard server"))
    }

    /// A place in the write path, held until the request is answered;
    /// 429 when [`Self::max_queued_writes`] requests hold one.
    fn enqueue(&self) -> Result<InQueue, ApiError> {
        InQueue::start(&self.writer_state, self.max_queued_writes)
            .ok_or_else(|| ApiError { status: 429, message: format!("{} writes are queued, the most this server takes; try again", self.max_queued_writes) })
    }

    /// Hands a write to the writer thread and waits for its commit. The
    /// writer records `key`'s outcome, even if this future is dropped.
    /// Writes marked `moved` are rows of moving slots, from the
    /// coordinator moving them ([`Self::check_keys`]); while slots move
    /// from this shard, a write to them, or an UPDATE or DELETE statement,
    /// holds the move's gate until it commits ([`Self::pass_gate`]).
    async fn write(&self, job: WriteJob, deadline: Instant, key: Option<String>, queued: InQueue, moved: bool) -> Result<Written, ApiError> {
        let tx = self.writes.as_ref().ok_or_else(|| ApiError::internal("no writer for a coordinator"))?;
        // Counted in before it looks at the map, which a move may change.
        let entry = self.moving.enter();
        let pass = if self.slots().is_none() {
            moving::Pass::Entry(entry)
        } else {
            let catalog = self.catalog();
            let places: Vec<(&str, Value)> = match &job {
                WriteJob::Ops(ops) => ops.iter().filter(|op| !is_mark(op)).map(|op| op_schema(&catalog, op).and_then(|t| Ok((place_name(t), op_place(t, op)?)))).collect::<Result<Vec<_>, _>>()?,
                WriteJob::Sql(p) => {
                    let what = place_name(p.table());
                    p.inserted_places().into_iter().map(|v| (what, v.clone())).collect()
                }
            };
            let statement = matches!(&job, WriteJob::Sql(_)) && places.is_empty();
            let pass = match moved {
                true => moving::Pass::Entry(entry),
                false => self.pass_gate(entry, &places, statement).await?,
            };
            self.check_keys(places.iter().map(|(w, v)| (*w, v)), moved)?;
            pass
        };
        let (reply, rx) = tokio::sync::oneshot::channel();
        let keys = self.idempotency.clone();
        tx.send(Queued { job, deadline, reply: Some(reply), key, keys, queued: Some(queued), pass: Some(pass) }).map_err(|_| ApiError::internal("the writer thread stopped"))?;
        rx.await.map_err(|_| ApiError::internal("the writer thread stopped"))?
    }

    /// The database, unless this is a coordinator.
    pub fn local_db(&self) -> Option<&Arc<Db>> {
        self.local()
    }

    fn local(&self) -> Option<&Arc<Db>> {
        match &self.backend {
            Backend::Local(db) => Some(db),
            Backend::Cluster(_) => None,
        }
    }

    /// The database's tables, the first first: a local database's as they
    /// are now, a coordinator's as its shards last named them
    /// ([`Coordinator::catalog`]).
    pub fn catalog(&self) -> Vec<Arc<Schema>> {
        match &self.backend {
            Backend::Local(db) => db.catalog(),
            Backend::Cluster(c) => c.catalog(),
        }
    }

    /// The table named `name` (any case). A coordinator that does not know
    /// it asks its shards again first.
    pub async fn table(&self, name: &str) -> Result<Arc<Schema>, ApiError> {
        let find = |c: Vec<Arc<Schema>>| c.into_iter().find(|t| t.table.eq_ignore_ascii_case(name));
        if let Some(t) = find(self.catalog()) {
            return Ok(t);
        }
        if self.learn_tables(false).await {
            if let Some(t) = find(self.catalog()) {
                return Ok(t);
            }
        }
        Err(ApiError::bad(format!("query error: no such table: {name}")))
    }

    /// A coordinator asks its shards for their tables and rank profiles
    /// again ([`Coordinator::refresh`]; at most once a second unless
    /// `now`): whether it learned of a change.
    async fn learn_tables(&self, now: bool) -> bool {
        let Backend::Cluster(c) = &self.backend else { return false };
        let changed = c.refresh(now).await;
        if changed {
            self.refresh_models();
        }
        changed
    }

    /// Every table's rank profiles, by table, then name: a local
    /// database's as they are now, a coordinator's as its shards last held
    /// them alike ([`Coordinator::rank_profiles`]).
    pub fn rank_profiles(&self) -> Vec<(String, RankProfile)> {
        match &self.backend {
            Backend::Local(db) => db.catalog().into_iter().flat_map(|t| db.storage().rank_profiles(&t.table).unwrap_or_default().into_iter().map(move |p| (t.table.clone(), p))).collect(),
            Backend::Cluster(c) => c.rank_profiles().to_vec(),
        }
    }

    /// Table `table`'s rank profile named `name`, both in any case.
    pub fn rank_profile(&self, table: &str, name: &str) -> Option<RankProfile> {
        let named = |p: &RankProfile| p.name.eq_ignore_ascii_case(name);
        match &self.backend {
            Backend::Local(db) => db.storage().rank_profiles(table).ok()?.into_iter().find(named),
            Backend::Cluster(c) => c.rank_profiles().iter().find(|(t, p)| t.eq_ignore_ascii_case(table) && named(p)).map(|(_, p)| p.clone()),
        }
    }

    /// Runs `bind` over the database's tables; a coordinator whose tables
    /// lack one the statement names, or a rank profile it ranks by, asks
    /// its shards again, then binds once more.
    async fn bound<T>(&self, bind: impl Fn(&[&Schema]) -> strata::Result<T>) -> Result<T, ApiError> {
        let run = || {
            let c = self.catalog();
            let refs: Vec<&Schema> = c.iter().map(|t| &**t).collect();
            bind(&refs)
        };
        match run() {
            Err(e) if (e.to_string().contains("no such table") || e.to_string().contains("has no rank profile named")) && self.learn_tables(false).await => Ok(run()?),
            r => Ok(r?),
        }
    }

    /// Pins the current version ([`Self::latest`]) for `lease_ttl`;
    /// returns the lease and the version's txn.
    pub fn lease(&self) -> Result<(u64, u64), ApiError> {
        let db = self.local().ok_or_else(|| ApiError::bad("a coordinator holds no snapshots"))?;
        let snap = self.latest(db)?;
        let txn = snap.txn();
        Ok((self.lease_of(snap, self.lease_ttl)?, txn))
    }

    /// Keeps `snap` for `ttl`; returns the lease. At most [`MAX_LEASES`]
    /// are held at once.
    fn lease_of(&self, snap: Snapshot, ttl: Duration) -> Result<u64, ApiError> {
        let mut l = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        l.retain(|_, (_, t)| *t > now);
        if l.len() >= MAX_LEASES {
            return Err(ApiError { status: 429, message: format!("{MAX_LEASES} snapshot leases are held; try again when some expire") });
        }
        let id = self.next_lease.fetch_add(1, Ordering::Relaxed);
        l.insert(id, (snap, now + ttl));
        Ok(id)
    }

    /// Ends `leases` before they expire, and releases the versions only
    /// they held. Leases unknown or expired are passed over.
    pub fn release(&self, leases: &[u64]) {
        let mut l = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        for id in leases {
            l.remove(id);
        }
    }

    /// The lease a query reads with `hold` ([`QueryReq::hold`]): the one
    /// it names, or a new one on `snap`, the version it reads, lasting as
    /// a [`Service::term_stats`] lease.
    fn held(&self, req: &QueryReq, snap: &Snapshot, timeout: Duration) -> Result<u64, ApiError> {
        match (req.hold, req.snapshot) {
            (true, 0) => self.lease_of(snap.clone(), self.lease_ttl.min(timeout + Duration::from_secs(1))),
            (true, l) => Ok(l),
            (false, _) => Ok(0),
        }
    }

    /// Releases expired leases, and the versions only they held; returns
    /// how many are held still. The maintenance loop calls it, so that a
    /// version is not kept until the next lease is made.
    pub fn sweep_leases(&self) -> usize {
        let mut l = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        l.retain(|_, (_, t)| *t > now);
        l.len()
    }

    /// A request's deadline: its `timeout_ms` up to [`Self::max_timeout`],
    /// or the default.
    pub fn timeout_of(&self, timeout_ms: u64) -> Duration {
        if timeout_ms > 0 {
            Duration::from_millis(timeout_ms).min(self.max_timeout)
        } else {
            self.timeout
        }
    }

    /// The version lease `lease` keeps.
    fn leased(&self, lease: u64) -> Result<Snapshot, ApiError> {
        let l = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        match l.get(&lease) {
            Some((s, t)) if *t > Instant::now() => Ok(s.clone()),
            _ => Err(ApiError::bad(format!("snapshot lease {lease} is unknown or expired"))),
        }
    }

    /// The latest version, for a read that does not name a lease: refused
    /// on a replica its primary has not found caught up (503,
    /// [`Follow::readable`]).
    ///
    /// The version is taken under the cluster map it is read by: rows of
    /// moving slots this shard holds without serving them are hidden
    /// ([`Self::hide_residue`]), and a read routed by a map this shard has
    /// left since its epoch was checked is refused (409), as the rows it
    /// would read are another map's.
    fn latest(&self, db: &Db) -> Result<Snapshot, ApiError> {
        match &self.replication() {
            Replication::Replica(f) => f.readable()?,
            Replication::Primary(p) => p.readable()?,
            Replication::None => {}
        }
        let held = self.slots.read().unwrap_or_else(|e| e.into_inner());
        self.epoch_ok(held.as_ref().map_or(0, |(m, _)| m.epoch), routed())?;
        let snap = db.snapshot();
        let mask = self.moving.mask();
        drop(held);
        match mask {
            Some(m) => Ok(snap.masked(&m)?),
            None => Ok(snap),
        }
    }

    /// The version a read reads: lease `lease`, or the latest
    /// ([`Self::latest`]), holding the write `after` names. A replica waits
    /// until it has applied that write, until `deadline` and
    /// [`READ_WAIT`] at most, and refuses the read then (503: read from
    /// the primary). A primary, or a server without replicas, holds every
    /// write it answered, so it refuses a number past its last (409), and
    /// a lease keeps its version, so one from before the write is refused
    /// too (409).
    async fn read_snapshot(&self, db: &Db, lease: u64, after: &After, deadline: Instant) -> Result<Snapshot, ApiError> {
        let min = after.seq()?;
        if min > 0 && db.storage().numbering().is_none() {
            return Err(ApiError::bad("min_seq_no needs a database that numbers its writes: start the server with --seq-no"));
        }
        let snap = match lease {
            0 => self.latest(db)?,
            l => self.leased(l)?,
        };
        let at = snap.seq_no().unwrap_or(0);
        if at >= min {
            return Ok(snap);
        }
        match (&self.replication(), lease) {
            (Replication::Replica(f), 0) => f.holding(min, deadline.min(Instant::now() + READ_WAIT)).await,
            (_, 0) => Err(conflict(format!("this server's last transaction is {at}: transaction {min} was not committed here"))),
            (_, l) => Err(conflict(format!("snapshot lease {l} keeps a version holding transactions up to {at}, before {min}"))),
        }
    }


    /// Parameters from JSON. Besides plain values (see [`js::param`]), a
    /// parameter may be `{"vector_of": key, "field": "emb"}`, the vector of
    /// a stored document (field: the first VECTOR field if left out), or
    /// `{"embed": "text", "field": "emb"}`, the vector of the text by the
    /// model of that embedded field (the first one with a model if left
    /// out), so that the text and the rows are embedded alike, or
    /// `{"embed": {"image": "<base64>"}, "field": "emb"}` (or `"audio"`),
    /// the vector of an image or sound file by a model of that and text.
    pub async fn params(self: &Arc<Self>, j: Option<&Json>) -> Result<Params, ApiError> {
        let mut p = Params::new();
        let Some(j) = j.filter(|j| !j.is_null()) else { return Ok(p) };
        let m = j.as_object().ok_or_else(|| ApiError::bad("params must be an object"))?;
        for (k, v) in m {
            let value = match v {
                Json::Object(o) if o.contains_key("vector_of") && o.keys().all(|k| ["vector_of", "field", "table"].contains(&k.as_str())) => {
                    let table = o.get("table").and_then(|f| f.as_str());
                    let (table, field) = self.vector_field(table, o.get("field").and_then(|f| f.as_str()))?;
                    self.vector_of(&table, &o["vector_of"], &field).await?
                }
                Json::Object(o) if o.get("embed").is_some_and(|e| e.is_string() || e.get("image").is_some() || e.get("audio").is_some()) && o.keys().all(|k| ["embed", "field", "table"].contains(&k.as_str())) => {
                    let (_, field, e) = self.model(o.get("table").and_then(|t| t.as_str()), o.get("field").and_then(|f| f.as_str()))?;
                    let file = match &o["embed"] {
                        Json::String(_) => None,
                        Json::Object(i) if i.len() == 1 => {
                            let (kind, b64) = i.iter().next().expect("one key");
                            let b64 = b64.as_str().ok_or_else(|| ApiError::bad(format!("{{\"embed\": {{\"{kind}\": ...}}}} is a file in base64")))?;
                            Some(strata::embed::query_file(kind, b64)?)
                        }
                        _ => return Err(ApiError::bad("\"embed\" is a text, {\"image\": \"<base64>\"} or {\"audio\": \"<base64>\"}")),
                    };
                    if let Some((m, _)) = file.as_ref().filter(|(m, _)| !e.modalities().contains(m)) {
                        return Err(ApiError::bad(format!("the model of {field} ({}) does not embed {}", e.name(), strata::embed::modality_name(*m))));
                    }
                    let text = o["embed"].as_str().unwrap_or_default().to_string();
                    let t0 = Instant::now();
                    let core = match tokio::time::timeout(self.timeout, self.embedding.clone().acquire_owned()).await {
                        Ok(p) => p.map_err(ApiError::internal)?,
                        Err(_) => return Err(ApiError { status: 429, message: "every core is embedding; try again".into() }),
                    };
                    let v = tokio::task::spawn_blocking(move || {
                        let _core = core;
                        match file {
                            Some((_, data)) => strata::embed::embed_file(e.as_ref(), &data),
                            None => Ok(e.embed_query(&text)),
                        }
                    })
                    .await
                    .map_err(ApiError::internal)??;
                    self.metrics.embedded("query", 1, t0.elapsed().as_secs_f64());
                    Value::Vector(v.into())
                }
                v => js::param(v)?,
            };
            p = p.set(k, value);
        }
        Ok(p)
    }

    /// The table and VECTOR field `{"vector_of": key}` reads: `field` of
    /// `table`, or of the first table that has a field so named; with no
    /// `field`, the first VECTOR field of `table` (or of the first table).
    pub fn vector_field(&self, table: Option<&str>, field: Option<&str>) -> Result<(String, String), ApiError> {
        let catalog = self.catalog();
        let tables: Vec<&Arc<Schema>> = match table {
            Some(t) => vec![catalog.iter().find(|s| s.table.eq_ignore_ascii_case(t)).ok_or_else(|| ApiError::bad(format!("query error: no such table: {t}")))?],
            None => match field {
                Some(_) => catalog.iter().collect(),
                None => catalog.iter().take(1).collect(),
            },
        };
        for t in tables {
            let found = match field {
                Some(f) => t.field(f),
                None => t.fields.iter().find(|f| matches!(f.ty, strata::FieldType::Vector { .. })),
            };
            if let Some(f) = found {
                return Ok((t.table.clone(), f.name.clone()));
            }
        }
        Err(match field {
            Some(f) => ApiError::bad(format!("no field {f}")),
            None => ApiError::bad("the table has no VECTOR field"),
        })
    }

    /// The vector `field` of the document `key` of `table`.
    pub async fn vector_of(self: &Arc<Self>, table: &str, key: &Json, field: &str) -> Result<Value, ApiError> {
        let schema = self.table(table).await?;
        let pk = schema.primary_key.ok_or_else(|| ApiError::bad("vector_of needs a primary key"))?;
        let f = schema.field_index(field).ok_or_else(|| ApiError::bad(format!("no field {field}")))?;
        if !matches!(schema.fields[f].ty, strata::FieldType::Vector { .. }) {
            return Err(ApiError::bad(format!("{field} is not a VECTOR field")));
        }
        // Quoted: a field may be named like a keyword (`order`).
        let sql = format!("SELECT {} FROM {} WHERE {} = :k LIMIT 1", quoted(field), ident(&schema.table), quoted(&schema.fields[pk].name));
        let params = Params::new().set("k", js::field_value(&schema, pk, key)?);
        let a = self.query(QueryReq { sql, params, ..Default::default() }).await?;
        match a.rs.rows.into_iter().next().and_then(|r| r.into_iter().next()) {
            Some(v @ Value::Vector(_)) => Ok(v),
            Some(_) => Err(ApiError::bad(format!("document {key} has no {field}"))),
            None => Err(ApiError::bad(format!("no document {key}"))),
        }
    }

    /// The type and bytes of the file whose SHA-256 is `sha256`, found
    /// through the first row of a MEDIA field that names it and read as
    /// writes read files (the media store, or a place `--media-fetch`
    /// lists), checked against the hash. Only files rows name can be
    /// read. A coordinator refuses: its shards hold the files.
    pub async fn media_file(self: &Arc<Self>, sha256: &str) -> Result<(String, Vec<u8>), ApiError> {
        if !strata::media::is_sha256(sha256) {
            return Err(ApiError::bad("a file is named by the SHA-256 of its bytes: 64 lowercase hex digits"));
        }
        let Backend::Local(db) = &self.backend else {
            return Err(ApiError::bad("a coordinator holds no files: ask the shard server whose rows name it"));
        };
        let catalog = self.catalog();
        for schema in catalog.iter() {
            for f in schema.fields.iter().filter(|f| f.media.is_some()) {
                let sql = format!("SELECT {} FROM {} WHERE {} = :h LIMIT 1", quoted(&f.name), ident(&schema.table), quoted(&f.name));
                let params = Params::new().set("h", Value::Str(sha256.into()));
                let a = self.query(QueryReq { sql, params, ..Default::default() }).await?;
                let Some(Value::Str(text)) = a.rs.rows.into_iter().next().and_then(|r| r.into_iter().next()) else { continue };
                let m = strata::media::Media::parse(&text).map_err(ApiError::internal)?;
                let resolver = db.media_resolver();
                let mime = m.mime.clone();
                let bytes = tokio::task::spawn_blocking(move || resolver.bytes_of(&m)).await.map_err(ApiError::internal)??;
                return Ok((mime, bytes));
            }
        }
        Err(ApiError { status: 404, message: format!("no row names a file of sha256 {sha256}") })
    }

    /// Runs a query, or its plan when it has one. A coordinator plans a
    /// query by a rank profile with the profile its shards held when it
    /// last asked; when a shard refuses the plan because its profile has
    /// changed since (through another coordinator, or on that shard), the
    /// coordinator asks again and plans once more.
    pub async fn query(self: &Arc<Self>, req: QueryReq) -> Result<Answer, ApiError> {
        let again = (matches!(self.backend, Backend::Cluster(_)) && req.plan.is_none() && req.sql.to_ascii_lowercase().contains("rank_profile")).then(|| req.clone());
        match (self.query_once(req).await, again) {
            (Err(e), Some(req)) if profile::is_stale(&e.message) => {
                self.learn_tables(true).await;
                self.query_once(req).await
            }
            (r, _) => r,
        }
    }

    async fn query_once(self: &Arc<Self>, mut req: QueryReq) -> Result<Answer, ApiError> {
        let t0 = Instant::now();
        let timeout = self.timeout_of(req.timeout_ms);
        // Bound against every table: a query of one table is the
        // executor's, any other a compound plan.
        let tp = Instant::now();
        let plan = match &req.plan {
            Some(p) => Ok(p.clone()),
            None => self.bound(|tables| compound::plan_read(&req.sql, tables, &req.params, &|t, n| self.rank_profile(t, n))).await.map(Arc::new),
        };
        // A plan made with a rank profile runs only on a database whose
        // profile of that name is the same.
        let plan = match (plan, &self.backend) {
            (Ok(p), Backend::Local(_)) if p.root.has_profiles() => profile::check_plan(&p, &|t, n| self.rank_profile(t, n).map(|p| p.hash())).map(|_| p).map_err(ApiError::from),
            (p, _) => p,
        };
        let plan = plan.and_then(|p| self.open_ranking_models(p, req.shard));
        telemetry::record("plan", tp, if plan.is_ok() { "ok" } else { "error" });
        let bound = match plan {
            Ok(plan) if plan.root.uses_operators() => {
                // A compound query runs as its plan; a coordinator sends the
                // shards the plan's leaves.
                req.plan = Some(plan.clone());
                return self.compound(req, plan, t0, timeout).await;
            }
            Ok(plan) => match self.table(plan.table()).await {
                Ok(schema) => {
                    // A coordinator sends the shards a plan by a rank
                    // profile or a model as is, naming the profile or the
                    // model's hash they check.
                    if plan.root.has_profiles() || plan.root.has_models() {
                        req.plan = Some(plan.clone());
                    }
                    plan.to_select(&schema).map(|select| (schema, select)).map_err(ApiError::from)
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match bound {
            Ok((schema, select)) => self.select(req, schema, select, t0, timeout).await,
            Err(e) => {
                let secs = t0.elapsed().as_secs_f64();
                self.metrics.query("invalid", "error", secs);
                // Kept with its error, as a query that ran and failed is.
                self.query_log.add(&req.sql, "invalid", "error", secs, 0, req.shard, Some(&e.message), false, None, false);
                Err(e)
            }
        }
    }

    /// Runs a query and logs what it served ([`strata::rank_log`]): the
    /// answer has the query's columns, and the log's rows are written as
    /// [`Self::write_ops`] writes them, through a coordinator to the shards
    /// their keys place them on. The first query logged to a table makes
    /// it, on every shard through a coordinator. A replica refuses it, the
    /// log being a write. Returns the request's id and the log's write.
    /// A logged query needs a write key, and the first one to a table,
    /// which creates it, an admin key.
    pub async fn query_logged(self: &Arc<Self>, who: &Principal, mut req: QueryReq, log: Log) -> Result<(Answer, String, Written), ApiError> {
        who.may(Scope::Write, "a logged query (it writes the rank log)")?;
        self.takes_writes()?;
        let schema = match self.table(&log.table).await {
            Ok(s) => s,
            Err(_) => {
                let what = format!("the first query logged to {} (it creates the table; log once with an admin key, or create it with CREATE TABLE)", log.table);
                who.may(Scope::Admin, &what)?;
                self.sql(who, &rank_log::create_sql(&log.table)?, &Params::new(), After::Any, req.timeout_ms).await?;
                self.table(&log.table).await?
            }
        };
        rank_log::check(&schema)?;
        let p = rank_log::prepare(&req.sql, &req.params, &log)?;
        req.sql = p.sql.clone();
        let mut a = self.query(req).await?;
        let rs = std::mem::replace(&mut a.rs, ResultSet { columns: Vec::new(), rows: Vec::new(), keys: Vec::new(), scores: Vec::new(), parts: None, stats: Default::default() });
        let served = rank_log::serve(rs, &p, &log)?;
        a.rs = served.rs;
        let ops: Vec<Json> = served.entries.iter().map(|e| json!({"op": "upsert", "table": log.table, "doc": e.to_json()})).collect();
        let w = match ops.is_empty() {
            true => Written { rows: 0, seq_no: None, seq_nos: Vec::new(), skipped: 0, stamps: Vec::new() },
            false => self.write_ops(ops, None).await?,
        };
        Ok((a, served.request, w))
    }

    /// Runs a plan ([`strata::query::ir`]): a read answers as
    /// [`Service::query`] does, and a write as [`Service::sql`] does (on any
    /// table). A coordinator takes writes as SQL until its shards take write
    /// plans. A read sees the writes `after` names.
    pub async fn run_plan(self: &Arc<Self>, who: &Principal, plan: Plan, after: After, timeout_ms: u64) -> Result<SqlAnswer, ApiError> {
        if plan.writes() {
            who.may(Scope::Write, "a write plan")?;
            if !plan.is_write() {
                return Err(ApiError::bad("a plan writes at its root only: an INSERT, UPDATE or DELETE inside a read is not run"));
            }
        }
        let t0 = Instant::now();
        // What the slow-query log shows of the plan.
        let catalog = self.catalog();
        let text = plan.explain(&catalog.iter().map(|t| &**t).collect::<Vec<_>>()).join(" / ");
        if plan.is_write() {
            if after != After::Any {
                return Err(ApiError::bad(FOR_READS));
            }
            let Backend::Local(db) = &self.backend else {
                return Err(ApiError::bad("a coordinator takes writes as SQL; its shards do not take write plans yet"));
            };
            let w = self.write_local(db, move |db| db.prepare_plan(&plan).map(Some), timeout_ms).await;
            let w = self.wrote(w, t0, Some(&text))?;
            return Ok(SqlAnswer::Written { w, took_ms: t0.elapsed().as_secs_f64() * 1e3 });
        }
        let req = QueryReq { sql: text, timeout_ms, plan: Some(Arc::new(plan)), after, ..Default::default() };
        self.query(req).await.map(SqlAnswer::Rows)
    }

    /// Runs a bound query of table `schema`, on the database or over the
    /// shards.
    async fn select(self: &Arc<Self>, mut req: QueryReq, schema: Arc<Schema>, select: Select, t0: Instant, timeout: Duration) -> Result<Answer, ApiError> {
        let kind = metrics::kind(&select);
        let mut lease = 0;
        req.profile |= self.profiling.cpu_all.load(Ordering::Relaxed);
        let profiled = req.profile;
        let live = self.live.start(&req.sql, kind, req.shard);
        let res = match &self.backend {
            Backend::Local(db) => {
                let stats = req.stats.take().map(Arc::new);
                LiveQueries::run(&live, async {
                    // A read on a replica may wait for a write it must see,
                    // before it takes a query slot.
                    let tw = Instant::now();
                    let snap = self.read_snapshot(db, req.snapshot, &req.after, t0 + timeout).await?;
                    if req.after != After::Any {
                        telemetry::record("wait for writes", tw, "ok");
                    }
                    let ta = Instant::now();
                    let permit = self.admit(timeout).await.ok_or_else(too_many);
                    telemetry::record("admission", ta, if permit.is_ok() { "ok" } else { "rejected" });
                    let permit = permit?;
                    lease = self.held(&req, &snap, timeout)?;
                    let view = snap.table(&schema.table).ok_or_else(|| ApiError::bad(format!("query error: no such table: {}", schema.table)))?;
                    let (running, n) = Running::start(&self.running);
                    let cancel = Cancel::default();
                    // Each running query gets its share of the cores for its
                    // segments (as engram's recall does).
                    let opts = ExecOptions {
                        stats,
                        shard: req.shard,
                        threads: (self.cores / n).max(1),
                        deadline: Some(t0 + timeout),
                        cancel: Some(cancel.0.clone()),
                        synonyms: self.synonyms().or_else(|| db.exec.synonyms.clone()),
                        // A coordinator's compound queries ask a shard for
                        // as many rows as a compound query's leaf may hold.
                        max_rows: if req.shard { self.max_rows.map(|m| m.max(compound::MAX_ROWS + 1)) } else { self.max_rows },
                        max_groups: self.max_groups,
                        memory: self.query_memory.clone(),
                        profile: req.profile,
                        ..db.exec.clone()
                    };
                    let db = db.clone();
                    // The work holds the slot and the count until it ends,
                    // and stops at its next check if this future is dropped.
                    let te = Instant::now();
                    let r = tokio::task::spawn_blocking(move || {
                        let _held = (permit, running);
                        let started = Instant::now();
                        let r = db.execute(&view, &select, &opts);
                        (r, started, Instant::now())
                    })
                    .await;
                    let back = Instant::now();
                    drop(cancel);
                    telemetry::record("execute", te, if matches!(r, Ok((Ok(_), ..))) { "ok" } else { "error" });
                    match r {
                        Ok((r, started, ended)) => r
                            .map(|mut rs| {
                                steps::around(&mut rs.stats.steps, te, started, ended, back);
                                (rs, Vec::new())
                            })
                            .map_err(ApiError::from),
                        Err(e) => Err(ApiError::internal(e)),
                    }
                })
                .await
            }
            Backend::Cluster(c) => {
                LiveQueries::run(&live, async {
                    let ta = Instant::now();
                    let permit = self.admit(timeout).await;
                    telemetry::record("admission", ta, if permit.is_some() { "ok" } else { "rejected" });
                    let permit = permit.ok_or_else(too_many)?;
                    let reach = select.limit.unwrap_or(0).saturating_add(select.offset);
                    let r = match self.max_rows.filter(|&m| reach > m) {
                        Some(m) => Err(ApiError::bad(format!("query error: LIMIT and OFFSET reach row {reach}; a query returns at most {m} rows"))),
                        None => c.query(&req, &schema, &select, t0 + timeout).await.map(|m| (m.rs, m.missing)),
                    };
                    drop(permit);
                    r
                })
                .await
            }
        };
        drop(live);
        // A shard's answer to a coordinator is not a client's.
        let a = self.done(&req.sql, kind, t0, self.max_rows.filter(|_| !req.shard), req.shard, (profiled, req.quiet), res);
        // A lease made for a query that failed holds nothing to fetch.
        let made = (lease != req.snapshot).then_some(lease).filter(|_| a.is_err());
        if let Some(l) = (req.release && req.snapshot != 0).then_some(req.snapshot).or(made) {
            self.release(&[l]);
        }
        a.map(|a| Answer { lease, ..a })
    }

    /// Counts a query's outcome, logs it if it was slow, and makes its
    /// answer. One with more rows than `window` (a compound query's, or a
    /// coordinator's without a LIMIT) fails.
    #[allow(clippy::too_many_arguments)]
    fn done(&self, sql: &str, kind: &'static str, t0: Instant, window: Option<usize>, shard: bool, (profiled, quiet): (bool, bool), res: Result<(ResultSet, Vec<String>), ApiError>) -> Result<Answer, ApiError> {
        let mut res = match (res, window) {
            (Ok((rs, _)), Some(m)) if rs.rows.len() > m => Err(ApiError::bad(format!("query error: the query matches more than {m} rows, the most a query returns; add a LIMIT"))),
            (r, _) => r,
        };
        // The steps count from the request's start, as its trace does: a
        // shard's, from the start of the call the coordinator made.
        if let Ok((rs, _)) = &mut res {
            let start = telemetry::trace().map_or(t0, |t| t.t0());
            steps::rebase(&mut rs.stats.steps, steps::us_at(start));
        }
        let secs = t0.elapsed().as_secs_f64();
        let outcome = match &res {
            Ok((_, missing)) if !missing.is_empty() => "partial",
            Ok(_) => "ok",
            Err(e) if e.status == 504 => "timeout",
            // Out of query memory, or a shard was.
            Err(e) if e.status == 429 => "rejected",
            Err(e) if e.status == 499 => "cancelled",
            Err(_) => "error",
        };
        self.metrics.query(kind, outcome, secs);
        let rows = res.as_ref().map_or(0, |(rs, _)| rs.rows.len());
        let slow = self.slow_after().is_some_and(|s| t0.elapsed() >= s);
        if slow {
            self.metrics.slow("query");
            tracing::warn!(kind, outcome, secs, rows, sql = %obs::truncate(sql, 2000), "slow query");
        }
        if !quiet {
            self.query_log.add(sql, kind, outcome, secs, rows, shard, res.as_ref().err().map(|e| e.message.as_str()), slow, res.as_ref().ok().map(|(rs, _)| &rs.stats), profiled);
        }
        res.map(|(rs, missing)| Answer { rs, took_ms: secs * 1e3, kind, missing, lease: 0 })
    }

    /// A compound query ([`compound`]): its plan's leaves run on one
    /// snapshot here, or each through the coordinator, and its operators
    /// here.
    async fn compound(self: &Arc<Self>, mut req: QueryReq, plan: Arc<Plan>, t0: Instant, timeout: Duration) -> Result<Answer, ApiError> {
        let kind = "compound";
        req.profile |= self.profiling.cpu_all.load(Ordering::Relaxed);
        let (profiled, quiet) = (req.profile, req.quiet);
        let catalog = self.catalog();
        let deadline = t0 + timeout;
        let req_sql = req.sql.clone();
        let live = self.live.start(&req.sql, kind, req.shard);
        let r = LiveQueries::run(&live, self.compound_run(req, plan, catalog, t0, timeout)).await;
        drop(live);
        let res = match r {
            Err(_) if Instant::now() > deadline => Err(ApiError { status: 504, message: format!("the query ran past its {} ms deadline", timeout.as_millis()) }),
            r => r,
        };
        self.done(&req_sql, kind, t0, self.max_rows, false, (profiled, quiet), res)
    }

    /// The work of [`Self::compound`]; [`Self::done`] counts it.
    async fn compound_run(self: &Arc<Self>, req: QueryReq, plan: Arc<Plan>, catalog: Vec<Arc<Schema>>, t0: Instant, timeout: Duration) -> Result<(ResultSet, Vec<String>), ApiError> {
        let deadline = t0 + timeout;
        // Each leaf checks it: the caller may have stopped waiting.
        let cancel = Cancel::default();
        let r = match &self.backend {
            Backend::Local(db) => {
                // A read on a replica may wait for a write it must see,
                // before it takes a query slot.
                let snap = self.read_snapshot(db, req.snapshot, &req.after, deadline).await?;
                let Some(permit) = self.admit(timeout).await else { return Err(too_many()) };
                let (running, n) = Running::start(&self.running);
                // Leaves hold up to compound::MAX_ROWS rows; the answer is
                // held to the window below.
                let opts = ExecOptions {
                    threads: (self.cores / n).max(1),
                    deadline: Some(deadline),
                    cancel: Some(cancel.0.clone()),
                    synonyms: self.synonyms().or_else(|| db.exec.synonyms.clone()),
                    max_groups: self.max_groups,
                    memory: self.query_memory.clone(),
                    profile: req.profile,
                    ..db.exec.clone()
                };
                let db = db.clone();
                let te = Instant::now();
                let r = tokio::task::spawn_blocking(move || {
                    let _held = (permit, running);
                    let started = Instant::now();
                    let tables: Vec<&Schema> = catalog.iter().map(|t| &**t).collect();
                    let r = ops::run_in(
                        &plan,
                        &tables,
                        &mut |leaf| {
                            let view = snap.table(leaf.table()).ok_or_else(|| strata::Error::Query(format!("no such table: {}", leaf.table())))?;
                            db.execute(&view, &leaf.to_select(view.schema())?, &opts)
                        },
                        &opts,
                    );
                    (r, started, Instant::now())
                })
                .await;
                let back = Instant::now();
                r.map(|(r, started, ended)| {
                    r.map(|mut rs| {
                        steps::around(&mut rs.stats.steps, te, started, ended, back);
                        (rs, Vec::new())
                    })
                })
            }
            Backend::Cluster(c) => {
                let Some(permit) = self.admit(timeout).await else { return Err(too_many()) };
                let c = c.clone();
                let rt = tokio::runtime::Handle::current();
                let stop = cancel.0.clone();
                let memory = self.query_memory.clone();
                // The leaves' calls to the shards carry the request's id.
                let id = obs::current();
                let trace = telemetry::trace();
                tokio::task::spawn_blocking(move || {
                    let _held = permit;
                    // Shards any leaf was answered without (partial answers).
                    let mut missing: Vec<String> = Vec::new();
                    let checks = ExecOptions { deadline: Some(deadline), cancel: Some(stop.clone()), memory, profile: req.profile, ..ExecOptions::default() };
                    let tables: Vec<&Schema> = catalog.iter().map(|t| &**t).collect();
                    let rs = ops::run_in(&plan, &tables, &mut |leaf| {
                        checks.check()?;
                        let schema = catalog.iter().find(|t| t.table.eq_ignore_ascii_case(leaf.table())).ok_or_else(|| strata::Error::Query(format!("no such table: {}", leaf.table())))?;
                        let select = leaf.to_select(schema)?;
                        // The shards run the leaf's plan; its text names it in their logs.
                        let sql = leaf.explain(&tables).join(" / ");
                        let step = QueryReq { sql, plan: Some(Arc::new(leaf.clone())), partial: req.partial, after: req.after.clone(), profile: req.profile, ..Default::default() };
                        let m = rt.block_on(telemetry::within(trace.clone(), obs::scope(id.clone(), c.query(&step, schema, &select, deadline)))).map_err(|e| strata::Error::Query(e.message))?;
                        for s in m.missing {
                            if !missing.contains(&s) {
                                missing.push(s);
                            }
                        }
                        Ok(m.rs)
                    }, &checks);
                    rs.map(|rs| (rs, missing))
                })
                .await
            }
        };
        drop(cancel);
        match r {
            Ok(r) => r.map_err(ApiError::from),
            Err(e) => Err(ApiError::internal(e)),
        }
    }

    /// Counts a query refused for want of a slot, and says so (429).
    fn rejected(&self, kind: &'static str, t0: Instant) -> ApiError {
        self.metrics.query(kind, "rejected", t0.elapsed().as_secs_f64());
        too_many()
    }

    /// A query slot, waited for until `timeout`; `None` (counted as
    /// rejected) if none came free.
    async fn admit(&self, timeout: Duration) -> Option<OwnedSemaphorePermit> {
        let t0 = Instant::now();
        match tokio::time::timeout(timeout, self.admission.clone().acquire_owned()).await {
            Ok(Ok(p)) => {
                self.metrics.admitted(t0.elapsed().as_secs_f64());
                Some(p)
            }
            _ => {
                self.metrics.rejected();
                None
            }
        }
    }

    /// The slow threshold now: as set while running, else as started.
    pub fn slow_after(&self) -> Option<Duration> {
        self.profiling.slow(self.slow)
    }

    /// Queries running now, including those whose callers stopped waiting
    /// and which have not reached a check yet.
    pub fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }

    /// The statistics `sql` scores with, and a lease on the snapshot they
    /// came from (0 from a coordinator, which sums its shards'). The lease
    /// lasts for the query round that follows: the request's deadline
    /// (`timeout_ms`, or the default) and a second. The snapshot holds the
    /// writes `after` names ([`After`]).
    pub async fn term_stats(self: &Arc<Self>, sql: &str, plan: Option<&Plan>, params: &Params, after: &After, timeout_ms: u64) -> Result<(TermStats, u64), ApiError> {
        let bound = match plan {
            Some(p) => p.clone(),
            None => self.bound(|tables| compound::plan_read(sql, tables, params, &|t, n| self.rank_profile(t, n))).await?,
        };
        // Statistics are a shard's: its coordinator scores with the
        // cross-encoders.
        let bound = Arc::unwrap_or_clone(self.open_ranking_models(Arc::new(bound), true)?);
        if bound.root.uses_operators() {
            return Err(ApiError::bad("term statistics are a single-table query's"));
        }
        let schema = self.table(bound.table()).await?;
        let select = bound.to_select(&schema)?;
        let timeout = self.timeout_of(timeout_ms);
        match &self.backend {
            Backend::Local(db) => {
                let snap = self.read_snapshot(db, 0, after, Instant::now() + timeout).await?;
                let lease = self.lease_of(snap.clone(), self.lease_ttl.min(timeout + Duration::from_secs(1)))?;
                let view = snap.table(&schema.table).ok_or_else(|| ApiError::bad(format!("query error: no such table: {}", schema.table)))?;
                let db = db.clone();
                let opts = ExecOptions { synonyms: self.synonyms().or_else(|| db.exec.synonyms.clone()), ..db.exec.clone() };
                let stats = tokio::task::spawn_blocking(move || db.term_stats_with(&view, &select, &opts)).await.map_err(ApiError::internal)??;
                Ok((stats, lease))
            }
            Backend::Cluster(c) => {
                // A query by a rank profile reaches the shards written
                // out, and one by a model with the model's hash.
                let given = plan.filter(|_| !bound.root.has_models()).or((bound.root.has_profiles() || bound.root.has_models()).then_some(&bound));
                Ok((c.term_stats(sql, &schema, given, params, after, Instant::now() + timeout).await?.0, 0))
            }
        }
    }

    /// The loosest level of the guard of `sql` (`min_rows`, `max_rows`)
    /// and how many rows each level from it up to the strictest matches,
    /// summed over the shards of a coordinator, as the query would read
    /// them seeing the writes `after` names.
    pub async fn guard_counts(self: &Arc<Self>, sql: &str, plan: Option<&Plan>, params: &Params, after: &After, timeout_ms: u64) -> Result<(usize, Vec<u64>), ApiError> {
        let t0 = Instant::now();
        let bound = match plan {
            Some(p) => p.clone(),
            None => self.bound(|tables| compound::plan_read(sql, tables, params, &|t, n| self.rank_profile(t, n))).await?,
        };
        if bound.root.uses_operators() {
            return Err(ApiError::bad("query error: a guard's levels are counted over one table's rows: a query without subqueries"));
        }
        let schema = self.table(bound.table()).await?;
        let select = bound.to_select(&schema)?;
        let timeout = self.timeout_of(timeout_ms);
        let deadline = t0 + timeout;
        match &self.backend {
            Backend::Local(db) => {
                let snap = self.read_snapshot(db, 0, after, deadline).await?;
                let view = snap.table(&schema.table).ok_or_else(|| ApiError::bad(format!("query error: no such table: {}", schema.table)))?;
                let Some(permit) = self.admit(timeout).await else { return Err(self.rejected("guard_counts", t0)) };
                let db = db.clone();
                let synonyms = self.synonyms().or_else(|| db.exec.synonyms.clone());
                let opts = ExecOptions { deadline: Some(deadline), memory: self.query_memory.clone(), synonyms, ..db.exec.clone() };
                let r = tokio::task::spawn_blocking(move || db.guard_counts(&view, &select, &opts)).await.map_err(ApiError::internal)?;
                drop(permit);
                Ok(r?)
            }
            Backend::Cluster(c) => c.guard_counts(sql, &schema, plan, params, &select, after, deadline).await,
        }
    }

    /// The codes that can follow each of `prefixes` in semantic-ID field
    /// `field`, among the rows `sql` (`SELECT COUNT(*) FROM t WHERE ...`)
    /// or `plan` filters, seeing the writes `after` names: per prefix, each
    /// next code and its rows ([`strata::db::Db::sid_next`]), summed over
    /// the shards on a coordinator. It takes a query slot.
    #[allow(clippy::too_many_arguments)]
    pub async fn sid_next(self: &Arc<Self>, sql: &str, plan: Option<&Plan>, params: &Params, field: &str, prefixes: &[String], after: &After, timeout_ms: u64) -> Result<Vec<Vec<(String, u64)>>, ApiError> {
        let t0 = Instant::now();
        let bound = match plan {
            Some(p) => p.clone(),
            None => self.bound(|tables| compound::plan_read(sql, tables, params, &|t, n| self.rank_profile(t, n))).await?,
        };
        if bound.root.uses_operators() {
            return Err(ApiError::bad("query error: the next codes count one table's rows: a condition without subqueries"));
        }
        let schema = self.table(bound.table()).await?;
        let select = bound.to_select(&schema)?;
        let timeout = self.timeout_of(timeout_ms);
        let deadline = t0 + timeout;
        match &self.backend {
            Backend::Local(db) => {
                let snap = self.read_snapshot(db, 0, after, deadline).await?;
                let view = snap.table(&schema.table).ok_or_else(|| ApiError::bad(format!("query error: no such table: {}", schema.table)))?;
                let Some(permit) = self.admit(timeout).await else { return Err(self.rejected("sid_next", t0)) };
                let db = db.clone();
                let opts = ExecOptions { deadline: Some(deadline), memory: self.query_memory.clone(), ..db.exec.clone() };
                let (field, prefixes) = (field.to_string(), prefixes.to_vec());
                let r = tokio::task::spawn_blocking(move || db.sid_next_with(&view, &select, &field, &prefixes, &opts)).await.map_err(ApiError::internal)?;
                drop(permit);
                Ok(r?)
            }
            Backend::Cluster(c) => c.sid_next(sql, &schema, plan, params, field, prefixes, after, deadline).await,
        }
    }

    /// Any statement, on any table: rows for a SELECT, which sees the
    /// writes `after` names; the documents an INSERT, UPDATE or DELETE
    /// wrote; none for `CREATE TABLE`, `DROP TABLE` and the `RANK PROFILE`
    /// statements ([`Self::ddl`]).
    /// The statement needs the scope of its kind ([`command_scope`]), which
    /// `who` must have; a table's or rank profile's statement is audited.
    pub async fn sql(self: &Arc<Self>, who: &Principal, sql: &str, params: &Params, after: After, timeout_ms: u64) -> Result<SqlAnswer, ApiError> {
        self.sql_in(who, sql, params, after, timeout_ms, None).await
    }

    /// [`Self::sql`]; with `slots`, an UPDATE or DELETE over the rows of
    /// those slots alone, as a coordinator sends one again to the shard
    /// slots moved to while it ran.
    pub async fn sql_in(self: &Arc<Self>, who: &Principal, sql: &str, params: &Params, after: After, timeout_ms: u64, slots: Option<Vec<usize>>) -> Result<SqlAnswer, ApiError> {
        let command = self.bound(|tables| bind_command(sql, tables, params)).await?;
        let need = command_scope(&command);
        who.may(need, &format!("a {} statement", match need { Scope::Read => "query", Scope::Write => "write", Scope::Admin => "table or rank profile" }))?;
        if need == Scope::Admin {
            keys::note(obs::truncate(sql, 500).into_owned());
        }
        let ddl = match command {
            Command::Query => {
                let a = self.query(QueryReq { sql: sql.into(), params: params.clone(), timeout_ms, after, ..Default::default() }).await?;
                return Ok(SqlAnswer::Rows(a));
            }
            Command::Explain { .. } => return Err(ApiError::bad("query error: EXPLAIN runs through POST /v1/explain")),
            Command::CreateTable { .. } | Command::DropTable { .. } => Some(Ddl::Table),
            Command::CreateRankProfile { .. } | Command::DropRankProfile { .. } => Some(Ddl::RankProfile),
            Command::Write { .. } => None,
        };
        if after != After::Any {
            return Err(ApiError::bad(FOR_READS));
        }
        let t0 = Instant::now();
        let r = match (&self.backend, ddl) {
            (_, Some(ddl)) => self.ddl(sql, params, ddl, timeout_ms).await,
            (Backend::Local(db), None) => {
                let (text, p) = (sql.to_string(), params.clone());
                let only = slots.map(|s| moving::mask_outside(&s));
                let refused = Arc::new(Mutex::new(None));
                let guard = self.statement_guard(routed(), refused.clone());
                let prepare = move |db: &Db| {
                    let Some(w) = db.prepare_write(&text, &p)? else { return Ok(None) };
                    let w = match only {
                        Some(m) => w.masked(m)?,
                        None => w,
                    };
                    Ok(Some(w.guarded(guard)))
                };
                let r = self.write_local(db, prepare, timeout_ms).await;
                let refusal = refused.lock().unwrap_or_else(|e| e.into_inner()).take();
                match refusal {
                    Some(e) if r.is_err() => Err(e),
                    _ => r,
                }
            }
            (Backend::Cluster(c), None) => c.sql(sql, params).await,
        };
        let w = self.wrote(r, t0, Some(sql))?;
        Ok(SqlAnswer::Written { w, took_ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// `CREATE TABLE` or `DROP TABLE`, `CREATE RANK PROFILE` or `DROP RANK
    /// PROFILE`: on a local database, once no write holds the writer;
    /// through a coordinator, on every shard ([`Coordinator::ddl`]). A
    /// replica refuses it, as it refuses every write. A primary refuses a
    /// table's: the tables a primary has are not shipped to its replicas
    /// yet, so each copy takes them before replication starts. A rank
    /// profile is logged as a write is, so a primary ships it, and answers
    /// once the copies its acks name hold it.
    async fn ddl(self: &Arc<Self>, sql: &str, params: &Params, ddl: Ddl, timeout_ms: u64) -> Result<Written, ApiError> {
        let r = match &self.backend {
            Backend::Local(db) => {
                self.takes_writes()?;
                if let (Replication::Primary(_), Ddl::Table) = (self.replication(), ddl) {
                    return Err(conflict("CREATE TABLE and DROP TABLE do not reach replicas yet: give each copy its tables before it replicates".into()));
                }
                let timeout = self.timeout_of(timeout_ms);
                let (db, sql, params) = (db.clone(), sql.to_string(), params.clone());
                let w = tokio::task::spawn_blocking(move || {
                    wait_for_writer(Instant::now() + timeout, || {
                        db.sql(&sql, &params)?;
                        // Its number, or a later one.
                        Ok(Written { seq_no: db.storage().seq_no().map(|n| n.seq).filter(|_| ddl == Ddl::RankProfile), ..Default::default() })
                    })
                })
                .await
                .map_err(ApiError::internal)?;
                match w {
                    Ok(w) => self.replicated(w, timeout).await,
                    e => e,
                }
            }
            Backend::Cluster(c) => c.ddl(sql, params).await,
        };
        // The models of the tables it made.
        self.refresh_models();
        r
    }

    /// A write `prepare` binds (`None`: not a write), queued for the writer
    /// thread. It is bound, and its vectors computed, before it queues:
    /// models run on a core of their own, not in the writer thread.
    async fn write_local<F>(self: &Arc<Self>, db: &Arc<Db>, prepare: F, timeout_ms: u64) -> Result<Written, ApiError>
    where
        F: FnOnce(&Db) -> strata::Result<Option<PreparedWrite>> + Send + 'static,
    {
        self.writable()?;
        self.backpressure(db).await?;
        let q = self.enqueue()?;
        let core = match self.embeds() {
            true => Some(self.embedding.clone().acquire_owned().await.map_err(ApiError::internal)?),
            false => None,
        };
        let db = db.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            let _core = core;
            prepare(&db)
        })
        .await
        .map_err(ApiError::internal)??
        .ok_or_else(|| ApiError::bad("not a write statement"))?;
        // The deadline bounds the wait for the writer, not the embedding.
        let deadline = Instant::now() + self.timeout_of(timeout_ms);
        let w = self.write(WriteJob::Sql(Box::new(prepared)), deadline, None, q, false).await?;
        self.replicated(w, self.timeout_of(timeout_ms)).await
    }

    /// Applies write operations (see the proto's WriteRequest) in one
    /// transaction per server, which other requests waiting at the same
    /// time may share (see [`writer`]). With an idempotency `key`, a retry
    /// of a request that committed returns its answer unapplied.
    pub async fn write_ops(self: &Arc<Self>, ops: Vec<Json>, key: Option<String>) -> Result<Written, ApiError> {
        self.write_ops_as(ops, key, false).await
    }

    /// [`Self::write_ops`], of rows of moving slots from the coordinator
    /// moving them when `moved` ([`Self::write`]).
    pub async fn write_ops_as(self: &Arc<Self>, mut ops: Vec<Json>, key: Option<String>, moved: bool) -> Result<Written, ApiError> {
        let t0 = Instant::now();
        let n = match &self.backend {
            Backend::Local(db) => {
                // Before any work: a replica, a database behind on its
                // flushes, or a full queue refuses the request.
                let admitted = match self.takes_writes() {
                    Ok(()) => self.backpressure(db).await.and_then(|_| self.enqueue()),
                    Err(e) => Err(e),
                };
                let queued = match admitted {
                    Ok(q) => q,
                    Err(e) => return self.wrote(Err(e), t0, None),
                };
                if let Some(k) = &key {
                    match self.idempotency.begin(k, idem::hash_ops(&ops)) {
                        idem::Begin::New => {}
                        idem::Begin::Done(n) => {
                            self.metrics.replay();
                            // A retry after a write its copies did not all
                            // acknowledge waits for them again.
                            return self.replicated(n, self.timeout).await;
                        }
                        idem::Begin::Busy => return Err(ApiError { status: 409, message: "a write with this idempotency key is still running".into() }),
                        idem::Begin::Mismatch => return Err(ApiError { status: 422, message: "this idempotency key was used for different operations".into() }),
                    }
                }
                let r = async {
                    // Not a retry of a write applied: a primary may refuse
                    // it before applying it.
                    self.admit_write()?;
                    if self.embeds() || self.has_media() {
                        // Queued writes wait for a core to embed on.
                        let core = self.embedding.clone().acquire_owned().await.map_err(ApiError::internal)?;
                        let (svc, db) = (self.clone(), db.clone());
                        ops = tokio::task::spawn_blocking(move || {
                            let _core = core;
                            svc.embed_ops(&db, &mut ops).map(|_| ops)
                        })
                        .await
                        .map_err(ApiError::internal)??;
                    }
                    // The deadline bounds the wait for the writer, not the embedding.
                    let w = self.write(WriteJob::Ops(ops), Instant::now() + self.timeout, key.clone(), queued, moved).await?;
                    // Visible now: how long after its event each row is.
                    let now = strata::timestamp::now();
                    for &ts in &w.stamps {
                        self.metrics.lag(now.saturating_sub(ts) as f64 / 1e6);
                    }
                    Ok(w)
                }
                .await;
                if let (Err(_), Some(k)) = (&r, &key) {
                    // Not committed (the writer reports a commit as Ok).
                    self.idempotency.finish(k, None);
                }
                // Committed: a retry with its key answers it, whatever its
                // copies say.
                match r {
                    Ok(w) => self.replicated(w, self.timeout).await,
                    e => e,
                }
            }
            Backend::Cluster(c) => c.write_ops(ops, key).await,
        };
        self.wrote(n, t0, None)
    }

    /// Writes one batch of a stream ([`Self::write_ops`]), which `s` took
    /// ([`Stream::admit`]). A local database's stream waits while the
    /// database is behind on its flushes or the writer's queue is full,
    /// rather than being refused, and reads nothing more from its client
    /// meanwhile. A coordinator's sends a batch a shard refused for its
    /// load (429) again a second later, under the batch's idempotency key,
    /// so that the shards that took it do not apply it twice.
    pub async fn stream_write(self: &Arc<Self>, s: &mut Stream, ops: Vec<Json>) -> Result<Written, ApiError> {
        let give_up = Instant::now() + self.max_timeout;
        let n = s.batches;
        s.batches += 1;
        let w = match &self.backend {
            Backend::Local(db) => {
                let busy = || db.storage().backlogged() || self.writer_state.queued.load(Ordering::Relaxed) >= self.max_queued_writes;
                while busy() && Instant::now() < give_up {
                    tokio::time::sleep(STREAM_POLL).await;
                }
                self.write_ops(ops, None).await?
            }
            Backend::Cluster(_) => {
                let key = Some(format!("stream-{:016x}-{n}", s.id));
                loop {
                    match self.write_ops(ops.clone(), key.clone()).await {
                        Err(e) if e.status == 429 && Instant::now() + RETRY_AFTER < give_up => tokio::time::sleep(RETRY_AFTER).await,
                        r => break r?,
                    }
                }
            }
        };
        s.written += w.rows;
        s.skipped += w.skipped;
        Ok(w)
    }

    /// Waits up to [`BACKLOG_WAIT`] while `db` is behind on its flushes
    /// ([`strata::storage::Database::backlogged`]); then refuses the write
    /// (429), so that clients slow down rather than queue.
    async fn backpressure(&self, db: &Db) -> Result<(), ApiError> {
        let until = Instant::now() + BACKLOG_WAIT;
        while db.storage().backlogged() {
            if Instant::now() >= until {
                return Err(ApiError { status: 429, message: "the database is behind on flushing writes to disk; try again in a second".into() });
            }
            tokio::time::sleep(STREAM_POLL).await;
        }
        Ok(())
    }

    /// The stream sources written from, with the offset of the last event
    /// each wrote: a coordinator's are the least over its shards, where a
    /// stream resumes ([`Coordinator::sources`]).
    pub async fn sources(&self) -> Result<Vec<Source>, ApiError> {
        match &self.backend {
            Backend::Local(db) => {
                let db = db.clone();
                tokio::task::spawn_blocking(move || db.sources()).await.map_err(ApiError::internal)?.map_err(Into::into)
            }
            Backend::Cluster(c) => c.sources().await,
        }
    }

    /// Counts a write's outcome (the request logs a failure on the
    /// server's side), and logs it if it was slow.
    fn wrote(&self, r: Result<Written, ApiError>, t0: Instant, sql: Option<&str>) -> Result<Written, ApiError> {
        let secs = t0.elapsed().as_secs_f64();
        match &r {
            Ok(w) => self.metrics.write(w.rows, secs),
            Err(e) => self.metrics.write_error(e.status),
        }
        if self.slow_after().is_some_and(|s| t0.elapsed() >= s) {
            self.metrics.slow("write");
            let (rows, status) = (r.as_ref().map_or(0, |w| w.rows), r.as_ref().map_or_else(|e| e.status, |_| 200));
            match sql {
                Some(sql) => tracing::warn!(secs, rows, status, sql = %obs::truncate(sql, 2000), "slow write"),
                None => tracing::warn!(secs, rows, status, "slow write"),
            }
        }
        r
    }

    /// Imports rows of `format` from `body` ([`strata::ingest`]): upserted
    /// through the writer (or routed to the shards) [`IMPORT_OPS`] rows a
    /// request, so the rows before a failed request stay written. Every
    /// row is read once first, so that a bad one writes nothing; then read
    /// again a request's worth at a time, the next while the last is
    /// written, so that an import holds its body and two requests' rows,
    /// not every row. Returns the rows written and the fields the schema
    /// lacks (skipped).
    pub async fn import(self: &Arc<Self>, table: Option<&str>, body: axum::body::Bytes, format: Format) -> Result<(u64, Vec<String>), ApiError> {
        let schema = match table {
            Some(t) => self.table(t).await?,
            None => Arc::new(self.schema.clone()),
        };
        // Rows of the first table name none, as before there were tables.
        let named = table.map(|_| schema.table.clone());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<strata::Result<Vec<Json>>>(1);
        let reader = tokio::task::spawn_blocking(move || -> strata::Result<Vec<String>> {
            let mut check = Rows::new(&schema, std::io::Cursor::new(body.clone()), format)?;
            check.by_ref().try_for_each(|d| d.map(drop))?;
            let mut rows = Rows::new(&schema, std::io::Cursor::new(body), format)?;
            loop {
                let op = |d: strata::value::Document| {
                    let mut op = json!({"op": "upsert", "doc": js::document_json(&schema, &d)});
                    if let Some(t) = &named {
                        op["table"] = json!(t);
                    }
                    op
                };
                let chunk: strata::Result<Vec<Json>> = rows.by_ref().take(IMPORT_OPS).map(|d| d.map(op)).collect();
                let last = chunk.as_ref().map_or(true, |c| c.len() < IMPORT_OPS);
                // The writing side stopped: a write failed.
                if tx.blocking_send(chunk).is_err() || last {
                    break;
                }
            }
            Ok(rows.unknown.into_iter().collect())
        });
        let mut n = 0;
        while let Some(chunk) = rx.recv().await {
            let ops = chunk?;
            if !ops.is_empty() {
                n += self.write_ops(ops, None).await?.rows;
            }
        }
        let unknown = reader.await.map_err(ApiError::internal)??;
        Ok((n, unknown))
    }

    /// Imports the tables of the SQLite file in `body`
    /// ([`strata::sqlite`]): each into the table of its name, created from
    /// the SQLite table when the database lacks it, its rows upserted
    /// through the writer (or routed to the shards) [`IMPORT_OPS`] a
    /// request. The plan is made and every row read once before anything
    /// is written, so that a value that does not fit writes nothing. With
    /// `plan_only`, returns the plan ([`strata::sqlite::Plan::to_json`])
    /// and writes nothing; otherwise what was written
    /// ([`strata::sqlite::Imported::to_json`]).
    /// Rows need a write key, and an import that creates tables an admin
    /// key, refused before anything is written.
    pub async fn import_sqlite(self: &Arc<Self>, who: &Principal, body: axum::body::Bytes, opts: strata::sqlite::ImportOptions, plan_only: bool, timeout_ms: u64) -> Result<Json, ApiError> {
        who.may(Scope::Write, "an import")?;
        let catalog = self.catalog();
        let checked = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
            let dir = tempfile::tempdir().map_err(ApiError::internal)?;
            let path = dir.path().join("import.sqlite");
            std::fs::write(&path, &body).map_err(ApiError::internal)?;
            // The server's temporary path is no business of the client's.
            let shown = path.display().to_string();
            let src = strata::sqlite::Source::open(&path).map_err(|e| ApiError::bad(e.to_string().replace(&shown, "the uploaded file")))?;
            let plan = src.plan_over(&catalog, &opts)?;
            if !plan_only {
                for t in &plan.tables {
                    src.documents(t, |docs| {
                        for d in docs {
                            d?;
                        }
                        Ok(())
                    })?;
                }
            }
            Ok((dir, src, plan))
        })
        .await
        .map_err(ApiError::internal)??;
        let (dir, src, plan) = checked;
        if plan_only {
            return Ok(plan.to_json());
        }
        let creates: Vec<&str> = plan.tables.iter().filter(|t| !t.exists).map(|t| t.schema.table.as_str()).collect();
        if !creates.is_empty() {
            who.may(Scope::Admin, &format!("an import that creates tables ({})", creates.join(", ")))?;
            keys::note(format!("SQLite import created tables {}", creates.join(", ")));
        }
        enum Step {
            Table(usize),
            Ops(Vec<Json>),
        }
        let tables = plan.tables.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<strata::Result<Step>>(1);
        let reader = tokio::task::spawn_blocking(move || {
            let _dir = dir;
            for (i, t) in tables.iter().enumerate() {
                if tx.blocking_send(Ok(Step::Table(i))).is_err() {
                    return;
                }
                let op = if t.schema.primary_key.is_some() { "upsert" } else { "insert" };
                let r = src.documents(t, |docs| {
                    loop {
                        let chunk: strata::Result<Vec<Json>> =
                            (&mut *docs).take(IMPORT_OPS).map(|d| d.map(|d| json!({"op": op, "table": t.schema.table, "doc": js::document_json(&t.schema, &d)}))).collect();
                        let last = chunk.as_ref().map_or(true, |c| c.len() < IMPORT_OPS);
                        // The writing side stopped: a write failed.
                        if tx.blocking_send(chunk.map(Step::Ops)).is_err() {
                            return Err(strata::Error::Query("stopped".into()));
                        }
                        if last {
                            return Ok(());
                        }
                    }
                });
                if r.is_err() {
                    return;
                }
            }
        });
        let mut out = strata::sqlite::Imported { tables: Vec::new(), skipped: plan.skipped.clone() };
        while let Some(step) = rx.recv().await {
            match step? {
                Step::Table(i) => {
                    let t = &plan.tables[i];
                    if !t.exists {
                        self.create_table(t.schema.clone(), timeout_ms).await?;
                    }
                    out.tables.push(strata::sqlite::TableImport { source: t.source.clone(), table: t.schema.table.clone(), rows: 0, created: !t.exists, bulk: false, notes: t.notes.clone() });
                }
                Step::Ops(ops) => {
                    if !ops.is_empty() {
                        let n = self.write_ops(ops, None).await?.rows;
                        if let Some(t) = out.tables.last_mut() {
                            t.rows += n;
                        }
                    }
                }
            }
        }
        reader.await.map_err(ApiError::internal)?;
        Ok(out.to_json())
    }

    /// Creates a table, as `CREATE TABLE` does ([`Self::ddl`]): on a local
    /// database, once no write holds the writer. A coordinator creates
    /// tables only through `CREATE TABLE`.
    async fn create_table(self: &Arc<Self>, schema: Schema, timeout_ms: u64) -> Result<(), ApiError> {
        let Backend::Local(db) = &self.backend else {
            return Err(ApiError::bad(format!("a coordinator imports into the tables its shards have: create table {} first (CREATE TABLE through the coordinator), or import on each shard", schema.table)));
        };
        self.takes_writes()?;
        if let Replication::Primary(_) = self.replication() {
            return Err(conflict(format!("the import would create table {}, and CREATE TABLE does not reach replicas yet: give each copy its tables before it replicates", schema.table)));
        }
        let deadline = Instant::now() + self.timeout_of(timeout_ms);
        let db = db.clone();
        tokio::task::spawn_blocking(move || wait_for_writer(deadline, || Ok(db.create_table(schema.clone())?))).await.map_err(ApiError::internal)??;
        self.refresh_models();
        Ok(())
    }

    /// Writes the tables (all, or `opts.tables`) to a new SQLite file, from
    /// one snapshot ([`strata::sqlite::export`]), in a directory that is
    /// removed when the returned [`tempfile::TempDir`] is dropped. Only a
    /// server with the database: a coordinator holds no rows.
    pub async fn export_sqlite(self: &Arc<Self>, opts: strata::sqlite::ExportOptions) -> Result<(tempfile::TempDir, PathBuf, strata::sqlite::Exported), ApiError> {
        let Backend::Local(db) = &self.backend else {
            return Err(ApiError::bad("a coordinator holds no rows: export from each shard's server, or with strata-server export"));
        };
        let db = db.clone();
        tokio::task::spawn_blocking(move || {
            let dir = tempfile::tempdir().map_err(ApiError::internal)?;
            let path = dir.path().join("export.sqlite");
            let r = strata::sqlite::export(&db, &path, &opts)?;
            Ok((dir, path, r))
        })
        .await
        .map_err(ApiError::internal)?
    }

    /// "ok"; for a database, "failing" when it cannot write until it
    /// restarts ([`Self::failing`]) and "degraded" while its flushes fail
    /// (writes still commit, to the WAL, until the memtable holds twice its
    /// limit); for a coordinator, its shards' state ([`Coordinator::health`]).
    pub async fn health(self: &Arc<Self>) -> Json {
        match &self.backend {
            Backend::Local(db) => {
                let s = db.snapshot();
                let mut h = json!({"status": "ok", "role": "shard", "docs": s.n_live_docs(), "segments": s.segments().len(), "txn": s.txn()});
                if let Some((m, shard)) = self.slots() {
                    h["slots"] = json!({"epoch": m.epoch, "shard": shard, "of": m.shards()});
                }
                match &self.replication() {
                    Replication::None => {}
                    Replication::Primary(p) => h["replication"] = p.health(),
                    Replication::Replica(f) => h["replication"] = f.health().await,
                }
                if let Some(why) = self.failing() {
                    h["status"] = json!("failing");
                    h["error"] = json!(why);
                } else if let Some(e) = db.storage().flush_error() {
                    h["status"] = json!("degraded");
                    h["error"] = json!(format!("flushes fail: {e}"));
                }
                h
            }
            Backend::Cluster(c) => c.health().await,
        }
    }

    /// Why this server cannot write until it restarts, if it cannot: a
    /// write failed partway (a full disk, an I/O error), after which the
    /// database refuses every write until it is reopened, or the writer
    /// thread stopped. Liveness (`/livez`) fails then, so that the server
    /// is restarted. A coordinator is never failing; its shards may be.
    pub fn failing(&self) -> Option<String> {
        match &self.backend {
            Backend::Local(db) if db.storage().is_poisoned() => Some("a write failed partway, so the database refuses writes until it is reopened: restart the server".into()),
            Backend::Local(_) if !self.writer_state.alive.load(Ordering::Relaxed) => Some("the writer thread stopped: restart the server".into()),
            _ => None,
        }
    }

    /// Storage statistics and what the merges are doing (local), and
    /// query counters.
    pub async fn stats(self: &Arc<Self>) -> Json {
        let (storage, merges) = match &self.backend {
            Backend::Local(db) => (
                serde_json::to_value(db.storage().file_stats()).unwrap_or(Json::Null),
                serde_json::to_value(db.storage().merge_status()).unwrap_or(Json::Null),
            ),
            Backend::Cluster(_) => (Json::Null, Json::Null),
        };
        json!({"health": self.health().await, "storage": storage, "merges": merges, "metrics": self.metrics.summary()})
    }

    /// The counters and gauges of this moment, for the series
    /// ([`telemetry::Series`]). Reads no shard: a coordinator's own.
    pub fn sample(&self) -> Sample {
        let mut s = Sample { at: Some(Instant::now()), ts_ms: telemetry::now_ms(), ..Default::default() };
        self.metrics.totals(&mut s);
        let (rss, cpu) = telemetry::process_usage();
        s.cpu_s = cpu;
        let heap = crate::heap::stats();
        (s.alloc_bytes, s.allocs) = (heap.allocated_bytes, heap.allocs);
        let g = &mut s.gauges;
        if let Some(r) = rss {
            g.insert("rss_bytes", r as f64);
        }
        if heap.installed {
            g.insert("heap_bytes", heap.in_use_bytes as f64);
            g.insert("heap_peak_bytes", heap.peak_bytes as f64);
        }
        g.insert("running", self.running() as f64);
        g.insert("live", self.live.len() as f64);
        g.insert("slots_free", self.admission.available_permits() as f64);
        if let Some(b) = &self.query_memory {
            g.insert("query_memory_bytes", b.used() as f64);
            g.insert("query_memory_limit_bytes", b.limit() as f64);
            if b.limit() > 0 {
                g.insert("query_memory_ratio", b.used() as f64 / b.limit() as f64);
            }
        }
        if let Backend::Local(db) = &self.backend {
            let snap = db.snapshot();
            let f = db.storage().file_stats();
            let m = db.storage().merge_status();
            let deleted: u64 = f.segments.iter().map(|s| s.n_deleted as u64).sum::<u64>() + f.pending_deletes as u64;
            for (k, v) in [
                ("docs", snap.n_live_docs() as f64),
                ("segments", snap.segments().len() as f64),
                ("deleted_docs", deleted as f64),
                ("memtable_docs", f.memtable_docs as f64),
                ("file_bytes", f.file_bytes as f64),
                ("wal_bytes", f.wal_bytes as f64),
                ("reusable_bytes", (f.free_pages * strata::storage::pager::PAGE) as f64),
                ("merges_running", m.running as f64),
                ("merge_memory_bytes", m.memory as f64),
                ("write_queue", self.writer_state.queued.load(Ordering::Relaxed) as f64),
                ("leases", self.leases.lock().unwrap_or_else(|e| e.into_inner()).len() as f64),
                ("failing", self.failing().is_some() as u8 as f64),
                ("flush_failing", db.storage().flush_error().is_some() as u8 as f64),
                ("merge_failing", m.error.is_some() as u8 as f64),
                ("pool_queued", strata::query::pool::Pool::global().queued() as f64),
            ] {
                g.insert(k, v);
            }
            if let Some((total, _, avail)) = telemetry::disk(db.storage().path()) {
                g.insert("disk_total_bytes", total as f64);
                g.insert("disk_free_bytes", avail as f64);
                if total > 0 {
                    g.insert("disk_used_ratio", 1.0 - avail as f64 / total as f64);
                }
            }
            match &self.replication() {
                Replication::None => {}
                Replication::Primary(p) => {
                    let h = p.health();
                    let rs = h["replicas"].as_array().cloned().unwrap_or_default();
                    g.insert("replicas", rs.len() as f64);
                    let in_sync = rs.iter().filter(|r| r["in_sync"] == true).count();
                    g.insert("replicas_in_sync", in_sync as f64);
                    g.insert("replicas_out_of_sync", (rs.len() - in_sync) as f64);
                    g.insert("replica_behind_max", rs.iter().filter_map(|r| r["behind"].as_f64()).fold(0.0, f64::max));
                    g.insert("seq_no", h["seq_no"].as_f64().unwrap_or(0.0));
                }
                Replication::Replica(f) => {
                    g.insert("seq_no", db.storage().numbering().map_or(0, |n| n.seq) as f64);
                    g.insert("serves_reads", f.serves_reads() as u8 as f64);
                }
            }
        }
        s
    }

    /// The process, its limits and where its database is
    /// (`GET /v1/system`).
    pub fn system(&self) -> Json {
        let (rss, cpu) = telemetry::process_usage();
        let started = self.started.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        let uptime = self.started.elapsed().map_or(0.0, |d| d.as_secs_f64());
        let db = match &self.backend {
            Backend::Local(db) => {
                let path = db.storage().path();
                let disk = telemetry::disk(path).map(|(total, free, avail)| json!({"total_bytes": total, "free_bytes": free, "available_bytes": avail}));
                json!({"path": path.display().to_string(), "disk": disk, "tables": self.catalog().len(), "seq_no": db.storage().numbering().map(|n| json!({"term": n.term, "seq": n.seq}))})
            }
            Backend::Cluster(_) => Json::Null,
        };
        json!({
            "version": env!("CARGO_PKG_VERSION"), "pid": std::process::id(), "started_ms": started, "uptime_s": uptime,
            "cores": self.cores, "os": std::env::consts::OS, "arch": std::env::consts::ARCH, "simd": strata::simd::describe(),
            "rss_bytes": rss, "cpu_seconds": cpu, "memory_limit_bytes": strata::storage::memory_limit(),
            "backend": match self.backend { Backend::Local(_) => "local", Backend::Cluster(_) => "coordinator" },
            "role": self.role().as_str_name().to_lowercase(),
            "db": db,
            "limits": {
                "timeout_ms": self.timeout.as_millis() as u64, "max_timeout_ms": self.max_timeout.as_millis() as u64,
                "max_concurrent": self.max_concurrent, "max_rows": self.max_rows, "max_groups": self.max_groups,
                "query_memory_bytes": self.query_memory.as_ref().map(|b| b.limit()), "max_queued_writes": self.max_queued_writes,
                "max_body_bytes": self.max_body, "slow_ms": self.slow_after().map(|d| d.as_millis() as u64), "lease_ttl_s": self.lease_ttl.as_secs(),
            },
            "features": {
                "synonyms": self.synonyms().map_or(0, |s| s.rules()), "backup_dir": self.backup_dir.as_ref().map(|p| p.display().to_string()),
                "models": self.models().len(), "agent_key": self.agent.api_key.is_some(),
            },
            "sampling": {"interval_ms": self.series.interval.as_millis() as u64, "points": telemetry::POINTS},
        })
    }

    /// A profile of `node`, a server of this coordinator's cluster
    /// ([`Coordinator::profile`]); a server that is not a coordinator has
    /// no other node to profile.
    pub async fn profile_node(&self, node: &str, a: crate::profiling::Asked) -> Result<crate::profiling::Answer, ApiError> {
        match &self.backend {
            Backend::Local(_) => Err(ApiError::bad("node names a server of a coordinator's cluster: this server is not a coordinator, so leave node out to profile it")),
            Backend::Cluster(c) => c.profile(node, a).await,
        }
    }

    /// Where this server is in its cluster, and how each copy is doing
    /// (`GET /v1/cluster`): a server's own health and latest point; a
    /// coordinator asks every replica of every shard
    /// ([`Coordinator::cluster`]).
    pub async fn cluster(self: &Arc<Self>) -> Json {
        match &self.backend {
            Backend::Local(_) => json!({"role": self.role().as_str_name().to_lowercase(), "node": self.detail().await}),
            Backend::Cluster(c) => {
                let mut j = c.cluster().await;
                j["coordinator"] = json!({"load": self.series.last().map(|p| (*p).clone()), "health": {"retries": j["retries"], "hedges": j["hedges"]}});
                j
            }
        }
    }

    /// This server's health and latest point, as a coordinator's
    /// `/v1/cluster` shows each shard's copy.
    pub async fn detail(self: &Arc<Self>) -> Json {
        let health = self.health().await;
        let load = self.series.last().map(|p| (*p).clone());
        let log_store = self.log_store.as_ref().map(|s| s.status());
        json!({"health": health, "load": load, "system": {"version": env!("CARGO_PKG_VERSION"), "uptime_s": self.started.elapsed().map_or(0.0, |d| d.as_secs_f64()), "cores": self.cores}, "running": self.live.len(), "log_store": log_store})
    }

    /// The metrics (`GET /metrics`). A coordinator does not ask its shards
    /// (each serves its own); it shows how its calls to them went.
    pub async fn prometheus(self: &Arc<Self>) -> String {
        let one = Family::one;
        let mut more = vec![
            one("strata_queries_running", "gauge", "Queries running now, those whose callers stopped waiting included.", self.running() as f64),
            Family { name: "strata_alerts_firing", ty: "gauge", help: "Alerts firing now, by severity (GET /v1/alerts).", values: self.alerts.firing() },
            one("strata_api_keys", "gauge", "API keys the server holds (GET /v1/admin/keys).", self.keys.list().len() as f64),
        ];
        if let Some(s) = &self.log_store {
            more.extend(s.metrics());
        }
        if let Some(b) = &self.query_memory {
            more.extend([
                one("strata_query_memory_bytes", "gauge", "Memory the running queries hold, as counted against their budget.", b.used() as f64),
                one("strata_query_memory_limit_bytes", "gauge", "Memory the running queries may hold together (--query-memory-mb).", b.limit() as f64),
                one("strata_query_memory_trips_total", "counter", "Queries refused because the running queries' memory budget was spent.", b.trips() as f64),
            ]);
        }
        match &self.backend {
            Backend::Local(db) => {
                let s = db.snapshot();
                let f = db.storage().file_stats();
                let m = db.storage().merge_status();
                let w = &self.writer_state;
                let deleted: u64 = f.segments.iter().map(|s| s.n_deleted as u64).sum::<u64>() + f.pending_deletes as u64;
                let pool = strata::query::pool::Pool::global();
                more.extend([
                    one("strata_query_pool_threads", "gauge", "Threads in the pool queries borrow from to search segments in parallel.", pool.threads() as f64),
                    one("strata_query_pool_queued", "gauge", "Threads queries asked the pool for that none was free to give yet.", pool.queued() as f64),
                    one("strata_docs", "gauge", "Rows in the database.", s.n_live_docs() as f64),
                    one("strata_segments", "gauge", "Segments on disk.", s.segments().len() as f64),
                    one("strata_deleted_docs", "gauge", "Deleted rows that merges have not dropped yet.", deleted as f64),
                    one("strata_memtable_docs", "gauge", "Rows written but not yet flushed to a segment.", f.memtable_docs as f64),
                    one("strata_file_bytes", "gauge", "Bytes the database takes on disk, without its write-ahead log.", f.file_bytes as f64),
                    one("strata_free_bytes", "gauge", "Bytes of the database file free for reuse (0 in the directory layout).", (f.free_pages * strata::storage::pager::PAGE) as f64),
                    one("strata_wal_bytes", "gauge", "Bytes of the write-ahead log.", f.wal_bytes as f64),
                    one("strata_snapshot_leases", "gauge", "Snapshot leases held.", self.sweep_leases() as f64),
                    one("strata_failing", "gauge", "1 when the database cannot write until the server restarts.", self.failing().is_some() as u8 as f64),
                    one("strata_flush_failing", "gauge", "1 while flushes fail (a full disk).", db.storage().flush_error().is_some() as u8 as f64),
                    one("strata_write_queue_requests", "gauge", "Write requests in the write path: embedding, waiting for the writer, or being applied.", w.queued.load(Ordering::Relaxed) as f64),
                    one("strata_write_batches_total", "counter", "Groups of write requests the writer applied in one transaction.", w.batches.load(Ordering::Relaxed) as f64),
                    one("strata_writer_panics_total", "counter", "Groups of writes whose application panicked.", w.panics.load(Ordering::Relaxed) as f64),
                    one("strata_merges_running", "gauge", "Merges running now.", m.running as f64),
                    one("strata_merge_memory_bytes", "gauge", "Memory the running merges are estimated to use.", m.memory as f64),
                    one("strata_merges_total", "counter", "Merges finished.", m.merges as f64),
                    one("strata_merges_limited_total", "counter", "Merges cut, or put off, to fit the memory budget or an interrupted merge's.", m.merges_limited as f64),
                    one("strata_merges_interrupted", "gauge", "Groups of segments whose merge a crash interrupted.", m.interrupted as f64),
                    one("strata_merge_failing", "gauge", "1 while merges fail.", m.error.is_some() as u8 as f64),
                ]);
                match &self.replication() {
                    Replication::None => {}
                    Replication::Primary(p) => more.extend(p.metrics()),
                    Replication::Replica(f) => more.extend(f.metrics()),
                }
            }
            Backend::Cluster(c) => more.extend(c.metrics()),
        }
        self.metrics.render(&more)
    }
}

pub enum SqlAnswer {
    Rows(Answer),
    Written { w: Written, took_ms: f64 },
}

/// What a statement that changes the catalog changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ddl {
    Table,
    RankProfile,
}

/// What a write did: the rows it wrote and, in a database that numbers
/// its writes (`--seq-no`), its transaction's sequence number. Through a
/// coordinator, whose shards number their own writes, each shard's
/// instead (`seq_nos`, 0 for a shard not written or whose replicas each
/// number their own); empty when none is numbered. A read that passes
/// them on sees the write ([`After`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Written {
    pub rows: u64,
    pub seq_no: Option<u64>,
    pub seq_nos: Vec<u64>,
    /// Operations not applied because their source offset was written
    /// already ([`op_source`]): a replay's.
    pub skipped: u64,
    /// When the events the write applied happened (their `source.ts`, in
    /// microseconds), for the lag metric; not part of the answer.
    pub stamps: Vec<i64>,
}

/// Rows one write request of an import carries.
pub const IMPORT_OPS: usize = 1000;

/// Operations a batch of a stream (`/v1/stream`) takes at most; fewer when
/// the client has sent no more yet.
pub const STREAM_OPS: usize = 4096;

/// How often a stream, or a write held back by a backlog, looks again.
const STREAM_POLL: Duration = Duration::from_millis(5);

/// How long a write waits for a backlog of flushes to clear before 429.
const BACKLOG_WAIT: Duration = Duration::from_secs(1);

/// When a client should send a write refused for the server's load (429)
/// again, as `Retry-After` says.
pub const RETRY_AFTER: Duration = Duration::from_secs(1);

/// The largest body a route other than `/v1/write` and `/v1/import` takes:
/// a query, its parameters (vectors among them) or a synonym list.
pub const SMALL_BODY: usize = 16 << 20;

/// The shard of `key` among `n` under an even slot map
/// ([`slots::shard_of`]), as a coordinator without a map of its own routes
/// writes. Loaders must split documents the same way.
pub fn shard_of(key: &Value, n: usize) -> usize {
    slots::shard_of(key, n.max(1))
}

fn conflict(message: String) -> ApiError {
    ApiError { status: 409, message }
}

/// What a shard says of a request routed by a newer cluster map than the
/// one it holds (503): its coordinator gives it the map and asks again.
pub const OLDER_THAN_REQUEST: &str = "older than the request's";

/// What a shard without a cluster map says of a request routed by one
/// (503): it is being given one, or its coordinator gives it one.
pub const NO_MAP_YET: &str = "this shard has no slot map";

/// In the message of a request a shard refuses (409) as routed by a map
/// older than its own: one the cluster changed since.
pub const OLDER_THAN_SHARD: &str = "is older than this shard's";

/// Why a write refuses `min_seq_no` and `min_seq_nos`.
const FOR_READS: &str = "min_seq_no and min_seq_nos are for reads: a write sees every write before it";

/// A query refused for want of a slot ([`Service::admit`]).
fn too_many() -> ApiError {
    ApiError { status: 429, message: "too many queries running; try again".into() }
}

/// How many rows of `db`, of any table, are placed outside shard
/// `shard`'s slots under `map`, by their key or their table's shard_by
/// value ([`Schema::with_shard_by`]), and one of those values; `None` if
/// every row is in them. Reads every row's.
fn rows_elsewhere(db: &Db, map: &SlotMap, shard: usize) -> Result<Option<(usize, Value)>, ApiError> {
    let snap = db.snapshot();
    let opts = ExecOptions { max_rows: None, ..db.exec.clone() };
    let mut out: Option<(usize, Value)> = None;
    for schema in db.catalog() {
        let Some(place) = schema.shard_by_field().or(schema.primary_key) else { continue };
        let view = snap.table(&schema.table).ok_or_else(|| ApiError::internal(format!("no table {} in its own snapshot", schema.table)))?;
        let sql = format!("SELECT {} FROM {}", quoted(&schema.fields[place].name), ident(&schema.table));
        let select = bind(&sql, &schema, &Params::new())?;
        let rs = db.execute(&view, &select, &opts)?;
        let values: Vec<Value> = match schema.shard_by_field() {
            Some(_) => rs.rows.into_iter().filter_map(|r| r.into_iter().next()).collect(),
            None => rs.keys,
        };
        // Rows of slots moving that the shard holds without serving are
        // in place too.
        let residue = map.residue(shard).unwrap_or_default();
        for v in values.into_iter().filter(|v| map.shard_of(v) != shard && residue.binary_search(&slots::slot(v)).is_err()) {
            match &mut out {
                Some((n, _)) => *n += 1,
                None => out = Some((1, v)),
            }
        }
    }
    Ok(out)
}

/// Writes `map`, `shard` and the epoch since which the shard holds its
/// slots to `path` ([`save_json`]).
fn save_slots(path: &Path, map: &SlotMap, shard: usize, since: u64) -> std::io::Result<()> {
    save_json(path, &json!({"shard": shard, "since": since, "map": map.to_json()}))
}

/// Writes `j` to `path` and syncs it, through a new file renamed over the
/// old one, so that a crash leaves one or the other.
pub(crate) fn save_json(path: &Path, j: &Json) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let body = serde_json::to_vec_pretty(j).map_err(std::io::Error::other)?;
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&body)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)?;
    match path.parent().filter(|d| !d.as_os_str().is_empty()) {
        Some(dir) => std::fs::File::open(dir)?.sync_all(),
        None => Ok(()),
    }
}

/// The table an operation writes: its `"table"`, else `first`.
fn op_table<'a>(op: &'a Json, first: &'a str) -> Result<&'a str, ApiError> {
    match op.get("table") {
        None | Some(Json::Null) => Ok(first),
        Some(Json::String(t)) => Ok(t),
        Some(_) => Err(ApiError::bad("an operation's \"table\" is a table's name")),
    }
}

/// The table of `catalog` (the database's tables, the first first) that an
/// operation writes ([`op_table`]).
pub fn op_schema<'a>(catalog: &'a [Arc<Schema>], op: &Json) -> Result<&'a Arc<Schema>, ApiError> {
    let first = catalog.first().ok_or_else(|| ApiError::internal("a database without tables"))?;
    let name = op_table(op, &first.table)?;
    catalog.iter().find(|t| t.table.eq_ignore_ascii_case(name)).ok_or_else(|| ApiError::bad(format!("query error: no such table: {name}")))
}

/// What places the row an operation writes on a shard: its key, or for a
/// table placed by a column ([`Schema::with_shard_by`]) that column's
/// value, from the operation's document or, for an operation by key, its
/// `"routing"`.
pub fn op_place(schema: &Schema, op: &Json) -> Result<Value, ApiError> {
    let Some(f) = schema.shard_by_field() else { return op_key(schema, op) };
    let name = schema.fields[f].name.as_str();
    let given = op.get("doc").and_then(|d| d.get(name)).filter(|v| !v.is_null()).or_else(|| op.get("routing").filter(|v| !v.is_null()));
    match given {
        Some(v) => js::field_value(schema, f, v),
        None => Err(ApiError::bad(format!("{} places its rows on shards by {name} (shard_by): the operation needs the row's {name}, in its doc or as \"routing\"", schema.table))),
    }
}

/// What places a row of `schema`'s table on a shard, as messages name
/// it: `key`, or the shard_by field.
fn place_name(schema: &Schema) -> &str {
    schema.shard_by.as_deref().unwrap_or("key")
}

/// The primary key an operation names (`key`, or the document's).
pub fn op_key(schema: &Schema, op: &Json) -> Result<Value, ApiError> {
    let pk = schema.primary_key.ok_or_else(|| ApiError::bad("writes by key need a primary key"))?;
    match (op.get("key"), op.get("doc")) {
        (Some(k), _) => js::field_value(schema, pk, k),
        (None, Some(d)) => js::field_value(schema, pk, d.get(&schema.fields[pk].name).unwrap_or(&Json::Null)),
        _ => Err(ApiError::bad("an operation needs a key or a doc")),
    }
}

/// A write request for the writer thread.
enum WriteJob {
    /// Operations of `/v1/write` (or gRPC `Write`).
    Ops(Vec<Json>),
    /// An `INSERT`, `UPDATE` or `DELETE`, its vectors computed.
    Sql(Box<PreparedWrite>),
}

struct Queued {
    job: WriteJob,
    deadline: Instant,
    /// `None` once answered.
    reply: Option<tokio::sync::oneshot::Sender<Result<Written, ApiError>>>,
    /// The request's idempotency key, whose outcome the writer records.
    key: Option<String>,
    keys: Arc<idem::Keys>,
    /// Counts the request as queued until it is answered (`None` then).
    queued: Option<InQueue>,
    /// Held until the write commits, for a hold on writes to moving slots
    /// to wait for ([`Service::pass_gate`]).
    pass: Option<moving::Pass>,
}

impl Queued {
    /// Records the outcome under the request's key and answers it.
    fn answer(mut self, r: Result<Written, ApiError>) {
        if let Some(k) = &self.key {
            self.keys.finish(k, r.as_ref().ok().cloned());
        }
        // Uncounted before the answer, which its caller may follow with a
        // look at the metrics.
        self.queued = None;
        self.pass = None;
        if let Some(reply) = self.reply.take() {
            let _ = reply.send(r);
        }
    }
}

impl Drop for Queued {
    /// A request dropped unanswered: the writer panicked applying it.
    fn drop(&mut self) {
        if let Some(reply) = self.reply.take() {
            if let Some(k) = &self.key {
                self.keys.finish(k, None);
            }
            let _ = reply.send(Err(ApiError::internal("the writer failed applying this write (see the server's log)")));
        }
    }
}

/// The writer thread's state, for health checks and metrics.
pub struct WriterState {
    /// False once the thread has stopped.
    pub alive: AtomicBool,
    /// Groups of write requests whose application panicked.
    pub panics: AtomicU64,
    /// Groups of write requests applied, each in one transaction (or
    /// each request alone, after the group's failed).
    pub batches: AtomicU64,
    /// Write requests queued for the writer or being applied.
    pub queued: Arc<AtomicUsize>,
}

impl Default for WriterState {
    fn default() -> WriterState {
        WriterState { alive: AtomicBool::new(true), panics: AtomicU64::new(0), batches: AtomicU64::new(0), queued: Arc::default() }
    }
}

/// Counts a write request as queued until dropped.
struct InQueue(Arc<AtomicUsize>);

impl InQueue {
    /// Counts a request as queued, unless `max` already are.
    fn start(state: &WriterState, max: usize) -> Option<InQueue> {
        let before = state.queued.fetch_add(1, Ordering::Relaxed);
        // Dropped at once when refused, which uncounts it.
        let q = InQueue(state.queued.clone());
        (before < max).then_some(q)
    }
}

impl Drop for InQueue {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Marks the writer stopped when its thread ends, however it ends.
struct Alive<'a>(&'a WriterState);

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::Relaxed);
    }
}

/// Counts a query among those running until dropped. The blocking work
/// holds it, so a query whose caller stopped waiting still counts while
/// it runs.
struct Running(Arc<AtomicUsize>);

impl Running {
    /// Starts counting; returns the guard and how many run now.
    fn start(n: &Arc<AtomicUsize>) -> (Running, usize) {
        let now = n.fetch_add(1, Ordering::Relaxed) + 1;
        (Running(n.clone()), now)
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A query's cancellation flag ([`ExecOptions::cancel`]), set when this
/// is dropped: when the query's future ends, finished or dropped by its
/// caller (a client that went away, a hedge that lost, a shard timeout).
#[derive(Default)]
struct Cancel(Arc<AtomicBool>);

impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Operations one shared transaction takes at most.
const GROUP_OPS: usize = 4096;

/// The writer of a local database: one thread applying write requests in
/// the order they arrive. Requests of operations that wait together share
/// one transaction, so one commit and one disk flush serve them all (group
/// commit); if one of them fails, each runs again alone, so that only it
/// fails. SQL writes find their rows by reading the database, which does
/// not show a transaction's own changes, so they commit alone. A group
/// that panics is answered with errors and the writer goes on.
fn writer(db: &Db, rx: mpsc::Receiver<Queued>, state: &WriterState) {
    let _alive = Alive(state);
    let mut next = None;
    loop {
        let Some(first) = next.take().or_else(|| rx.recv().ok()) else { return };
        let mut group = vec![first];
        if matches!(group[0].job, WriteJob::Ops(_)) {
            let mut n = group[0].ops().len();
            while n < GROUP_OPS {
                match rx.try_recv() {
                    Ok(q) if matches!(q.job, WriteJob::Ops(_)) => {
                        n += q.ops().len();
                        group.push(q);
                    }
                    Ok(q) => {
                        next = Some(q);
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        // The requests a panic leaves unanswered are answered as they drop.
        state.batches.fetch_add(1, Ordering::Relaxed);
        let n = group.len();
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| apply_group(db, group))).is_err() {
            state.panics.fetch_add(1, Ordering::Relaxed);
            tracing::error!(requests = n, "the writer panicked applying a group of writes; they were answered with errors");
        }
    }
}

/// Applies one group of write requests and answers each.
fn apply_group(db: &Db, group: Vec<Queued>) {
    // A request still queued at its deadline is refused unapplied.
    let now = Instant::now();
    let (late, group): (Vec<Queued>, Vec<Queued>) = group.into_iter().partition(|q| q.deadline <= now);
    for q in late {
        q.answer(Err(ApiError { status: 503, message: "the deadline passed while waiting for the writer".into() }));
    }
    let Some(deadline) = group.iter().map(|q| q.deadline).max() else { return };
    if let [Queued { job: WriteJob::Sql(p), .. }] = &group[..] {
        let r = wait_for_writer(deadline, || db.write_prepared_seq(p).map(|(rows, seq_no)| Written { rows, seq_no, ..Default::default() }).map_err(ApiError::from));
        group.into_iter().next().expect("one").answer(r);
        return;
    }
    let requests: Vec<&[Json]> = group.iter().map(|q| q.ops()).collect();
    match wait_for_writer(deadline, || apply_requests(db, &requests)) {
        Ok((applied, seq_no)) => {
            for (q, a) in group.into_iter().zip(applied) {
                q.answer(Ok(a.written(seq_no)));
            }
        }
        Err(e) if group.len() == 1 => {
            group.into_iter().next().expect("one").answer(Err(e));
        }
        // A failed commit wrote nothing (Strata reports a durable commit
        // as Ok), so each request can run again alone.
        Err(_) => {
            for q in group {
                let r = wait_for_writer(q.deadline, || apply_ops(db, q.ops()));
                q.answer(r);
            }
        }
    }
}

impl Queued {
    fn ops(&self) -> &[Json] {
        match &self.job {
            WriteJob::Ops(ops) => ops,
            WriteJob::Sql(..) => &[],
        }
    }
}

/// A checkpoint's name in the backup directory: letters, digits, '.', '_'
/// and '-', not starting with '.', so that it stays in the directory.
fn checkpoint_name(name: &str) -> Result<&str, ApiError> {
    let ok = !name.is_empty() && name.len() <= 200 && !name.starts_with('.') && name.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    ok.then_some(name).ok_or_else(|| ApiError::bad(format!("bad checkpoint name {name:?}: use letters, digits, '.', '_' and '-', not starting with '.'")))
}

/// Runs `write` until it gets the writer or `deadline` passes: Strata
/// answers `Busy` (503) rather than queueing while another write, or a
/// background merge's commit, holds it. A busy write has changed nothing.
pub(crate) fn wait_for_writer<T>(deadline: Instant, mut write: impl FnMut() -> Result<T, ApiError>) -> Result<T, ApiError> {
    loop {
        match write() {
            Err(e) if e.status == 503 && Instant::now() < deadline => std::thread::sleep(Duration::from_millis(2)),
            r => return r,
        }
    }
}

/// Applies `ops` to `db` in one transaction.
pub fn apply_ops(db: &Db, ops: &[Json]) -> Result<Written, ApiError> {
    let (mut applied, seq_no) = apply_requests(db, &[ops])?;
    Ok(applied.remove(0).written(seq_no))
}

/// What one request's operations did in a transaction ([`apply_into`]).
#[derive(Default)]
struct Applied {
    rows: u64,
    skipped: u64,
    stamps: Vec<i64>,
}

impl Applied {
    fn written(self, seq_no: Option<u64>) -> Written {
        Written { rows: self.rows, seq_no, skipped: self.skipped, stamps: self.stamps, ..Default::default() }
    }
}

/// Applies several requests' operations in one transaction; returns what
/// each did, and the transaction's sequence number in a database that
/// numbers its writes.
fn apply_requests(db: &Db, requests: &[&[Json]]) -> Result<(Vec<Applied>, Option<u64>), ApiError> {
    let catalog = db.catalog();
    // Made outside the transaction, which holds the writer.
    if requests.iter().flat_map(|ops| ops.iter()).any(has_source) {
        db.storage().ensure_sources()?;
    }
    let mut t = db.storage().begin()?;
    let mut applied = Vec::with_capacity(requests.len());
    for ops in requests {
        applied.push(apply_into(&catalog, &mut t, ops)?);
    }
    let seq_no = t.commit_seq()?;
    Ok((applied, seq_no))
}

/// Where an operation's event came from (its `"source"`): a stream's
/// source, the event's offset in it, and when the event happened
/// (microseconds since the epoch), if the operation says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mark {
    pub name: String,
    pub offset: i64,
    pub ts: Option<i64>,
}

/// The longest source name, in bytes.
const SOURCE_NAME: usize = 256;

pub fn has_source(op: &Json) -> bool {
    op.get("source").is_some_and(|s| !s.is_null())
}

/// The scope a SQL statement needs ([`crate::keys`]): a query read, a
/// write write, and a table's or rank profile's statement admin. Every
/// kind is named, so that a new one must say what it needs.
pub fn command_scope(c: &Command) -> Scope {
    match c {
        Command::Query | Command::Explain { .. } => Scope::Read,
        Command::Write { .. } => Scope::Write,
        Command::CreateTable { .. } | Command::DropTable { .. } | Command::CreateRankProfile { .. } | Command::DropRankProfile { .. } => Scope::Admin,
    }
}

/// Whether an operation only marks a source's offset (`"op": "source"`),
/// writing no row.
pub fn is_mark(op: &Json) -> bool {
    op.get("op").and_then(|o| o.as_str()) == Some("source")
}

/// An operation's `"source": {"name": ..., "offset": n, "ts": ...}`, if it
/// has one; `ts` is ISO 8601 text, or milliseconds since the epoch.
pub fn op_source(op: &Json) -> Result<Option<Mark>, ApiError> {
    let s = match op.get("source") {
        None | Some(Json::Null) => return Ok(None),
        Some(s) => s,
    };
    let name = s
        .get("name")
        .and_then(|n| n.as_str())
        .filter(|n| !n.is_empty() && n.len() <= SOURCE_NAME)
        .ok_or_else(|| ApiError::bad(format!("an operation's source has a \"name\" of 1 to {SOURCE_NAME} bytes")))?;
    let offset = s.get("offset").and_then(|o| o.as_i64()).ok_or_else(|| ApiError::bad(format!("source {name}: the \"offset\" is a whole number")))?;
    let ts = match s.get("ts") {
        None | Some(Json::Null) => None,
        Some(Json::String(t)) => Some(strata::timestamp::parse(t).ok_or_else(|| ApiError::bad(format!("source {name}: ts {t:?} is not ISO 8601 text")))?),
        Some(v) => Some(v.as_i64().ok_or_else(|| ApiError::bad(format!("source {name}: ts is ISO 8601 text or whole milliseconds since the epoch")))?.saturating_mul(1000)),
    };
    Ok(Some(Mark { name: name.to_string(), offset, ts }))
}

/// A stream's state from one batch to the next ([`Service::stream_write`]).
pub struct Stream {
    /// Each source's highest offset so far, which the next must pass.
    last: HashMap<String, i64>,
    /// The sources admitted since [`Stream::offsets`] last took them.
    fresh: Vec<String>,
    /// Names the stream's batches in their idempotency keys.
    id: u64,
    pub batches: u64,
    pub written: u64,
    pub skipped: u64,
}

impl Default for Stream {
    fn default() -> Stream {
        Stream { last: HashMap::new(), fresh: Vec::new(), id: rand_id(), batches: 0, written: 0, skipped: 0 }
    }
}

impl Stream {
    /// Takes the operations of one event or of several (a line of
    /// `/v1/stream`, a message of gRPC `WriteStream`), which a batch never
    /// splits: each source's offsets must pass those before them in the
    /// stream, since an offset written already marks a replay and is
    /// skipped.
    pub fn admit(&mut self, ops: &[Json]) -> Result<(), ApiError> {
        let mut top: Vec<(String, i64)> = Vec::new();
        for op in ops {
            let Some(m) = op_source(op)? else {
                if is_mark(op) {
                    return Err(ApiError::bad("a \"source\" operation needs a source: {\"op\": \"source\", \"source\": {\"name\": ..., \"offset\": n}}"));
                }
                continue;
            };
            if let Some(&last) = self.last.get(&m.name) {
                if m.offset <= last {
                    return Err(ApiError::bad(format!(
                        "source {}: offset {} comes after {last} in this stream: offsets increase along a stream, and one event's operations go together (one line, as {{\"ops\": [...]}}, or one message)",
                        m.name, m.offset
                    )));
                }
            }
            match top.iter_mut().find(|(n, _)| *n == m.name) {
                Some((_, t)) => *t = (*t).max(m.offset),
                None => top.push((m.name, m.offset)),
            }
        }
        for (name, offset) in top {
            if !self.fresh.contains(&name) {
                self.fresh.push(name.clone());
            }
            self.last.insert(name, offset);
        }
        Ok(())
    }

    /// The last offset of each source admitted since the last call: once
    /// a batch has committed, where each of its sources stands, which a
    /// consumer may commit to its log.
    pub fn offsets(&mut self) -> Vec<(String, i64)> {
        let fresh = std::mem::take(&mut self.fresh);
        fresh.into_iter().map(|n| { let o = self.last[&n]; (n, o) }).collect()
    }
}

fn rand_id() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    h.finish()
}

/// Refuses an operation with `if_seq_no` unless the row with `key` is
/// live with that sequence number, as this transaction finds it (a row an
/// earlier operation of the transaction wrote has none yet). Checked
/// under the writer lock, so no other write comes between it and the
/// operation.
fn check_seq_no(t: &TableTxn<'_, '_>, op: &Json, key: &Value) -> Result<(), ApiError> {
    let want = match op.get("if_seq_no") {
        None | Some(Json::Null) => return Ok(()),
        Some(v) => v.as_u64().ok_or_else(|| ApiError::bad("if_seq_no is a sequence number: a whole number from 0"))?,
    };
    let now = t.seq_of(key).map_err(|e| match e {
        strata::Error::Schema(_) => ApiError::bad("if_seq_no needs a database that numbers its writes: start the server with --seq-no"),
        e => e.into(),
    })?;
    if now == Some(want) {
        return Ok(());
    }
    let found = match now {
        Some(n) => format!("its sequence number is {n}"),
        None => "no row has the key, or an earlier operation of this request wrote it".to_string(),
    };
    Err(ApiError { status: 409, message: format!("if_seq_no {want} does not hold for key {}: {found}", key_text(key)) })
}

fn key_text(key: &Value) -> String {
    match key {
        Value::Str(s) => format!("{s:?}"),
        v => format!("{v:?}"),
    }
}

/// Applies one request's `ops`, each to the table it names ([`op_table`]);
/// a TIMESTAMP defaulting to `"now"` takes the same time in all of them.
///
/// An operation with a source ([`op_source`]) whose offset the source has
/// reached already, as the transaction finds it when the request starts,
/// is a replay's and is skipped; afterwards each source's highest offset
/// is kept, in the same transaction ([`WriteTxn::set_source`]). So a
/// stream resumed from its sources' offsets applies each event once.
fn apply_into(catalog: &[Arc<Schema>], txn: &mut WriteTxn<'_>, ops: &[Json]) -> Result<Applied, ApiError> {
    let mut a = Applied::default();
    let now = strata::timestamp::now();
    // Each source's offset when the request started, and its highest here.
    let mut sources: Vec<(String, Option<i64>, i64)> = Vec::new();
    // The event last stamped: its operations follow one another.
    let mut stamped = None;
    for op in ops {
        if let Some(m) = op_source(op)? {
            let i = match sources.iter().position(|(n, ..)| *n == m.name) {
                Some(i) => i,
                None => {
                    let stored = txn.source_offset(&m.name)?;
                    sources.push((m.name.clone(), stored, i64::MIN));
                    sources.len() - 1
                }
            };
            let (_, stored, top) = &mut sources[i];
            if stored.is_some_and(|s| m.offset <= s) {
                a.skipped += !is_mark(op) as u64;
                continue;
            }
            *top = (*top).max(m.offset);
            if m.ts.is_some() && stamped != Some((i, m.offset)) {
                a.stamps.extend(m.ts);
                stamped = Some((i, m.offset));
            }
        }
        if is_mark(op) {
            if !has_source(op) {
                return Err(ApiError::bad("a \"source\" operation needs a source: {\"op\": \"source\", \"source\": {\"name\": ..., \"offset\": n}}"));
            }
            continue;
        }
        let schema = &**op_schema(catalog, op)?;
        let mut tt = txn.table(&schema.table)?;
        let t = &mut tt;
        let kind = op.get("op").and_then(|o| o.as_str()).unwrap_or("");
        let doc = || js::document(schema, op.get("doc").unwrap_or(&Json::Null), now);
        let field = || op.get("field").and_then(|f| f.as_str()).ok_or_else(|| ApiError::bad(format!("{kind} needs a field")));
        let checked = |t: &TableTxn<'_, '_>| -> Result<(), ApiError> {
            match op.get("if_seq_no") {
                None => Ok(()),
                Some(_) if kind == "insert" => Err(ApiError::bad("insert takes no if_seq_no: it fails if the row exists")),
                Some(_) => check_seq_no(t, op, &op_key(schema, op)?),
            }
        };
        checked(t)?;
        let wrote = match kind {
            "insert" => {
                t.insert(doc()?)?;
                true
            }
            "upsert" => {
                t.upsert(doc()?)?;
                true
            }
            "upsert_if_newer" => {
                let v = op.get("version").and_then(|v| v.as_str()).ok_or_else(|| ApiError::bad("upsert_if_newer needs a version field"))?;
                t.upsert_if_newer(doc()?, v)?
            }
            "delete" => t.delete(&op_key(schema, op)?)?,
            "set" => {
                let key = op_key(schema, op)?;
                let f = field()?;
                let fi = schema.field_index(f).ok_or_else(|| ApiError::bad(format!("no field {f}")))?;
                t.set(&key, f, js::field_value(schema, fi, op.get("value").unwrap_or(&Json::Null))?)?
            }
            // The transaction's own writes count, other requests' in the
            // same group commit included. With a "doc", a key no row has
            // inserts it, its field plus the value.
            "add" => {
                let raw = op.get("value").unwrap_or(&Json::Null);
                let delta = match (raw.as_i64(), raw.as_f64()) {
                    (Some(x), _) => Value::Int(x),
                    (None, Some(x)) => Value::Float(x),
                    _ => return Err(ApiError::bad("add needs a numeric value")),
                };
                match op.get("doc") {
                    Some(_) => {
                        let d = doc()?;
                        let pk = schema.primary_key.ok_or_else(|| ApiError::bad("writes by key need a primary key"))?;
                        if op.get("key").is_some() && op_key(schema, op)? != d[pk] {
                            return Err(ApiError::bad("add's key and its doc's key differ: give one"));
                        }
                        t.add_or_insert(d, field()?, &delta)?;
                        true
                    }
                    None => t.add(&op_key(schema, op)?, field()?, &delta)?,
                }
            }
            other => return Err(ApiError::bad(format!("unknown write op {other:?}"))),
        };
        a.rows += wrote as u64;
    }
    for (name, stored, top) in sources {
        if top > i64::MIN && stored.is_none_or(|s| top > s) {
            txn.set_source(&name, top, now)?;
        }
    }
    Ok(a)
}


#[cfg(test)]
mod tests {
    use super::*;

    // OPS-6: a panic in the writer thread ended it, and every later write
    // failed with health still "ok". The writer now catches it; the
    // requests of the batch it was applying are answered as they drop.
    #[test]
    fn a_write_dropped_unanswered_gets_an_error() {
        let keys = Arc::new(idem::Keys::default());
        assert_eq!(keys.begin("k", 1), idem::Begin::New);
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let state = WriterState::default();
        let q = Queued { job: WriteJob::Ops(Vec::new()), deadline: Instant::now(), reply: Some(tx), key: Some("k".into()), keys: keys.clone(), queued: InQueue::start(&state, 1), pass: None };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _held = q;
            panic!("a bug while applying a write");
        }));
        assert!(r.is_err());
        assert_eq!(rx.try_recv().unwrap().unwrap_err().status, 500);
        // It did not commit, so a retry with its key applies it.
        assert_eq!(keys.begin("k", 1), idem::Begin::New);
        assert_eq!(state.queued.load(Ordering::Relaxed), 0);
    }
}
