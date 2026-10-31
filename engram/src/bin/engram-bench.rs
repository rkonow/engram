//! Evaluation of Engram's retrieval on long-conversation memory benchmarks
//! (LongMemEval, LoCoMo) converted by scripts/convert_memory_benchmarks.py.
//!
//!   engram-bench retrieval --data locomo.jsonl --embedder static --ctx round --out results/retrieval
//!
//! Each unit (one conversation) becomes its own store: one record per turn
//! chunk. Each question retrieves once with BM25 and once with vectors; the
//! fusion configurations are then scored on those lists. Gate features are
//! written for every question, and for a "foreign" question from another
//! unit (a negative: nothing relevant exists).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

use engram::benchdata::{parse_time, Question, Unit};
use engram::capture::chunk;
use engram::embed::{self, Embedder};
use engram::recall::{self, PackParams, RankParams};
use engram::record::{Kind, Record};
use engram::store::{Filter, Store, StoreOptions};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Retrieval {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "static")]
        embedder: String,
        /// none | round (the other side of the user/assistant exchange as context)
        #[arg(long, default_value = "round")]
        ctx: String,
        #[arg(long, default_value = "results/retrieval")]
        out: PathBuf,
        /// Only every n-th unit (subsampling).
        #[arg(long, default_value_t = 1)]
        every: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Embedding cache file (for slow embedders).
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long, default_value = "")]
        tag: String,
        /// Write texts missing from the cache to this JSONL file and exit.
        #[arg(long)]
        dump_missing: Option<PathBuf>,
    },
    /// Query understanding (memory_ask) on a memory benchmark: per question a
    /// model plans searches, they run over the unit's store, and a second
    /// call picks the records. Scores evidence found in the pool and in the
    /// picks against plain hybrid search, and writes the picked records as
    /// reader contexts (scripts/reader_eval.py).
    Deep {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "results/deep")]
        out: PathBuf,
        #[arg(long, default_value_t = 1)]
        every: usize,
        /// Only every n-th question of each unit.
        #[arg(long, default_value_t = 1)]
        qevery: usize,
        #[arg(long)]
        limit: Option<usize>,
        /// Questions in flight at once (each waits on two model calls).
        #[arg(long, default_value_t = 8)]
        threads: usize,
        /// Planner and ranker model.
        #[arg(long, default_value = "haiku")]
        model: String,
        /// Ranker model for questions the planner calls hard (empty: --model).
        #[arg(long, default_value = "")]
        hard_model: String,
        #[arg(long, default_value_t = 8)]
        max_picks: usize,
        /// `model` (planned searches) or `none` (the question's own searches only).
        #[arg(long, default_value = "model")]
        planner: String,
        /// `model` or `local` (scored without a model call).
        #[arg(long, default_value = "model")]
        ranker: String,
        /// Also writes each candidate's local-ranker features and whether it
        /// held evidence, for fitting the local ranker (scripts/local_ranker.py).
        #[arg(long)]
        features: Option<PathBuf>,
        #[arg(long, default_value = "")]
        tag: String,
    },
    /// Distils each exchange of a test set with Engram's prompt, several
    /// times, and writes every output (graded by scripts/distill_eval.py).
    Distill {
        /// JSON lines: id, request, answer, commands, known ([id, kind, text]).
        #[arg(long, default_value = "eval/distill/cases.jsonl")]
        cases: PathBuf,
        #[arg(long, default_value = "haiku")]
        model: String,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 6)]
        threads: usize,
        #[arg(long, default_value = "results/distill/outputs.jsonl")]
        out: PathBuf,
    },
    /// Ingest, storage and latency as the store grows (LongMemEval turns as records).
    Scale {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "results/scale")]
        out: PathBuf,
        /// Queries timed per size.
        #[arg(long, default_value_t = 300)]
        queries: usize,
        /// Hook invocations through the engram binary per size.
        #[arg(long, default_value_t = 200)]
        hooks: usize,
        /// Pause between hook invocations (prompts arrive seconds apart in
        /// real sessions; background work from one should not count against
        /// the next).
        #[arg(long, default_value_t = 200)]
        hook_gap_ms: u64,
    },
    /// Where a prompt's time goes, the write-then-query stall under memtable
    /// settings, and hooks through a process against an in-process client.
    Latency {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "results/latency")]
        out: PathBuf,
        #[arg(long, default_value_t = 94)]
        queries: usize,
        /// Append-then-query rounds per memtable configuration.
        #[arg(long, default_value_t = 600)]
        rounds: usize,
        /// Hook invocations per path.
        #[arg(long, default_value_t = 100)]
        hooks: usize,
        /// Share of records stored as knowledge (decision, preference,
        /// fact), which the per-prompt knowledge query searches.
        #[arg(long, default_value_t = 0.01)]
        knowledge: f64,
    },
    /// Reads while records are written: reader threads run the prompt's
    /// searches (knowledge and hybrid search side by side, then packing, as
    /// the prompt hook does) for a fixed time, alone and while a writer
    /// appends a tool call and its result at a steady rate; then the same
    /// through a daemon, with clients sending prompt hooks while another
    /// sends tool hooks.
    Concurrency {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "100000,1000000")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "results/concurrency")]
        out: PathBuf,
        #[arg(long, default_value_t = 94)]
        queries: usize,
        /// Reader threads (and prompt clients) per configuration.
        #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
        readers: Vec<usize>,
        /// Seconds per configuration.
        #[arg(long, default_value_t = 10)]
        secs: u64,
        /// Pause between the writer's appends (and the tool client's hooks).
        #[arg(long, default_value_t = 10)]
        write_every_ms: u64,
        #[arg(long, default_value_t = 0.01)]
        knowledge: f64,
    },
    /// The session tail's cost on a prompt (`ENGRAM_SESSION_CONTEXT=1`):
    /// knowledge and hybrid search side by side, then packing, as the prompt
    /// hook runs them, without the tail and with it (the session's last
    /// records read, a digest of their words searched by BM25 and their
    /// vectors by maxsim), alternating, on one
    /// warm store per size, one prompt at a time.
    Tail {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1000000")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "results/tail")]
        out: PathBuf,
        #[arg(long, default_value_t = 94)]
        queries: usize,
        /// Passes over the queries.
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(long, default_value_t = 0.01)]
        knowledge: f64,
    },
    /// Appends tool calls back to back from one or more clients, as tool
    /// hooks do, to an empty store, and reports how long each append took:
    /// any wait for a flush or a merge shows in the tail.
    Writes {
        #[arg(long, default_value = "hash:256")]
        embedder: String,
        /// Records per configuration (an append stores two).
        #[arg(long, default_value_t = 200_000)]
        records: usize,
        /// Clients appending at once, per configuration.
        #[arg(long, value_delimiter = ',', default_value = "1,4")]
        clients: Vec<usize>,
        #[arg(long, default_value = "results/writes")]
        out: PathBuf,
    },
    /// Embeds a text field of every JSON line of a file into raw float32
    /// vectors (`<out>`, n × dim, little-endian), for other tools (Strata's
    /// graph benchmark embeds arXiv papers with it).
    Embed {
        #[arg(long)]
        input: PathBuf,
        /// Fields joined by a newline to make the text.
        #[arg(long, value_delimiter = ',', default_value = "title,abstract")]
        fields: Vec<String>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Prints embeddings (JSON lines) for comparing implementations.
    EmbedCheck {
        #[arg(long)]
        embedder: String,
        #[arg(long)]
        query: bool,
        texts: Vec<String>,
    },
    /// Questions about this repository's own session transcript.
    Probes {
        #[arg(long)]
        transcript: PathBuf,
        #[arg(long)]
        probes: PathBuf,
        /// Evaluation points `window_line:ask_line` (comma-separated). Records
        /// from lines before ask_line are stored; those from window_line on are
        /// in the live context and excluded.
        #[arg(long, value_delimiter = ',')]
        points: Vec<String>,
        #[arg(long, default_value = "static")]
        embedder: String,
        #[arg(long, default_value = "/home/user/codex-test-2")]
        project: String,
        #[arg(long, default_value = "results/probes")]
        out: PathBuf,
        /// Embedding cache file (for slow embedders).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Write texts (records and probe questions) missing from the cache
        /// to this JSONL file and exit. Takes every probe file to cover.
        #[arg(long)]
        dump_missing: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        dump_probes: Vec<PathBuf>,
        /// Override the packing caps (records per turn, records in all).
        #[arg(long)]
        per_turn: Option<usize>,
        #[arg(long)]
        max_records: Option<usize>,
    },
}

/// Answers only from the embedding cache (vectors computed elsewhere).
struct CachedOnly {
    dim: usize,
    name: String,
}

impl Embedder for CachedOnly {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        eprintln!("warning: {} texts missing from the embedding cache", texts.len());
        Ok(texts.iter().map(|_| vec![0f32; self.dim]).collect())
    }
}

/// Serves vectors from an embedding cache (documents by text, queries by
/// the query key); anything missing goes to `inner` or becomes zeros.
struct CacheEmbedder {
    inner: Arc<dyn Embedder>,
    cache: Arc<Mutex<HashMap<u64, Vec<f32>>>>,
    missing: AtomicUsize,
}

fn query_key(q: &str) -> u64 {
    embed::fnv(&format!("\u{1}query\u{1}{q}"))
}

impl Embedder for CacheEmbedder {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, texts: &[&str]) -> strata::Result<Vec<Vec<f32>>> {
        let c = self.cache.lock().unwrap();
        texts
            .iter()
            .map(|t| match c.get(&embed::fnv(t)) {
                Some(v) => Ok(v.clone()),
                None => {
                    self.missing.fetch_add(1, Ordering::Relaxed);
                    Ok(self.inner.embed(&[t])?.pop().unwrap_or_default())
                }
            })
            .collect()
    }
    fn embed_query(&self, q: &str) -> strata::Result<Vec<f32>> {
        if let Some(v) = self.cache.lock().unwrap().get(&query_key(q)) {
            return Ok(v.clone());
        }
        self.missing.fetch_add(1, Ordering::Relaxed);
        self.inner.embed_query(q)
    }
}

fn open_embedder(spec: &str) -> Arc<dyn Embedder> {
    if let Some(rest) = spec.strip_prefix("cached:") {
        let (dim, name) = rest.split_once(':').unwrap_or((rest, "cached"));
        return Arc::new(CachedOnly { dim: dim.parse().expect("cached:<dim>:<name>"), name: format!("cached:{name}") });
    }
    Arc::from(embed::open(spec).expect("embedder"))
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Records of a unit: one per turn chunk; `tag` holds the turn id.
fn records(u: &Unit, ctx_mode: &str) -> Vec<Record> {
    let locomo = u.dataset == "locomo";
    let mut out = Vec::new();
    let mut seq = 1i64;
    for (i, t) in u.turns.iter().enumerate() {
        let mut text = if locomo { format!("{}: {}", t.role, t.text) } else { t.text.clone() };
        if let Some(c) = &t.image_caption {
            text += &format!(" [shares an image: {c}]");
        }
        let same = |j: usize| u.turns.get(j).filter(|o| o.session_id == t.session_id);
        let ctx = match ctx_mode {
            "round" => {
                let other = if t.role == "user" { same(i + 1) } else if i > 0 { same(i - 1) } else { None };
                other.map(|o| clip(&if locomo { format!("{}: {}", o.role, o.text) } else { o.text.clone() }, 300)).unwrap_or_default()
            }
            // What Engram does: responses carry the request, requests carry nothing.
            "request" => {
                let other = if t.role == "user" { None } else if i > 0 { same(i - 1) } else { None };
                other.map(|o| clip(&if locomo { format!("{}: {}", o.role, o.text) } else { o.text.clone() }, 300)).unwrap_or_default()
            }
            _ => String::new(),
        };
        let kind = if t.role == "assistant" { Kind::Answer } else { Kind::Prompt };
        let ts = t.timestamp.as_deref().map(parse_time).unwrap_or(0);
        for c in chunk(&text, engram::capture::CHUNK_CHARS) {
            out.push(Record {
                id: format!("r{seq}"),
                session: t.session_id.clone(),
                kind,
                tag: t.turn_id.clone(),
                turn: i as i64,
                seq,
                ts,
                importance: kind.prior(),
                files: String::new(),
                text: c,
                ctx: ctx.clone(),
                links: Vec::new(),
            });
            seq += 1;
        }
    }
    out
}

/// Embeddings, cached by text hash for slow embedders.
struct Embeds {
    e: Arc<dyn Embedder>,
    cache: Option<Mutex<HashMap<u64, Vec<f32>>>>,
}

impl Embeds {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let Some(cache) = &self.cache else {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            return self.e.embed(&refs).expect("embedding");
        };
        let keys: Vec<u64> = texts.iter().map(|t| embed::fnv(t)).collect();
        let missing: Vec<usize> = {
            let c = cache.lock().unwrap();
            (0..texts.len()).filter(|&i| !c.contains_key(&keys[i])).collect()
        };
        if !missing.is_empty() {
            let refs: Vec<&str> = missing.iter().map(|&i| texts[i].as_str()).collect();
            let vs = self.e.embed(&refs).expect("embedding");
            let mut c = cache.lock().unwrap();
            for (&i, v) in missing.iter().zip(vs) {
                c.insert(keys[i], v);
            }
        }
        let c = cache.lock().unwrap();
        keys.iter().map(|k| c[k].clone()).collect()
    }
    fn query(&self, q: &str) -> Vec<f32> {
        let Some(cache) = &self.cache else { return self.e.embed_query(q).expect("embedding") };
        let k = embed::fnv(&format!("\u{1}query\u{1}{q}"));
        if let Some(v) = cache.lock().unwrap().get(&k) {
            return v.clone();
        }
        let v = self.e.embed_query(q).expect("embedding");
        cache.lock().unwrap().insert(k, v.clone());
        v
    }
}

fn load_cache(path: &Path, dim: usize) -> HashMap<u64, Vec<f32>> {
    let mut m = HashMap::new();
    if let Ok(b) = std::fs::read(path) {
        let rec = 8 + 4 * dim;
        for c in b.chunks_exact(rec) {
            let k = u64::from_le_bytes(c[..8].try_into().unwrap());
            m.insert(k, c[8..].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect());
        }
    }
    m
}

fn save_cache(path: &Path, m: &HashMap<u64, Vec<f32>>) {
    let mut b = Vec::new();
    for (k, v) in m {
        b.extend_from_slice(&k.to_le_bytes());
        v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    }
    let _ = std::fs::write(path, b);
}

/// Fusion configurations scored on the same candidate lists.
fn configs() -> Vec<(&'static str, RankParams)> {
    let base = RankParams::default();
    let np = RankParams { lambda_imp: 0.0, lambda_rec: 0.0, ..base.clone() };
    vec![
        ("bm25", RankParams { use_vectors: false, ..np.clone() }),
        ("dense", RankParams { use_bm25: false, ..np.clone() }),
        ("hybrid_0.3", RankParams { w_bm25: 0.3, ..np.clone() }),
        ("hybrid_0.5", RankParams { w_bm25: 0.5, ..np.clone() }),
        ("hybrid_0.7", RankParams { w_bm25: 0.7, ..np.clone() }),
        ("hybrid_0.5_priors", base.clone()),
        ("hybrid_0.3_priors", RankParams { w_bm25: 0.3, ..base.clone() }),
    ]
}

#[derive(Default, Clone, Serialize)]
struct QMetrics {
    r_any: [f64; 4],
    r_all: [f64; 4],
    ndcg10: f64,
    s_any: [f64; 2],
    s_all: [f64; 2],
    pack4k: f64,
    pack8k: f64,
    pack4k_chars: f64,
    pack_brief: f64,
}

const KS: [usize; 4] = [1, 5, 10, 20];

fn metrics(hits: &[engram::store::Hit], q: &Question, query: &str) -> QMetrics {
    let ev: HashSet<&str> = q.evidence_turn_ids.iter().map(|s| s.as_str()).collect();
    let evs: HashSet<&str> = q.evidence_session_ids.iter().map(|s| s.as_str()).collect();
    let mut turns: Vec<&str> = Vec::new();
    let mut sessions: Vec<&str> = Vec::new();
    for h in hits {
        if !turns.contains(&h.record.tag.as_str()) {
            turns.push(h.record.tag.as_str());
        }
        if !sessions.contains(&h.record.session.as_str()) {
            sessions.push(h.record.session.as_str());
        }
    }
    let mut m = QMetrics::default();
    for (i, &k) in KS.iter().enumerate() {
        let got = turns.iter().take(k).filter(|t| ev.contains(*t)).count();
        m.r_any[i] = (got > 0) as u8 as f64;
        m.r_all[i] = (got == ev.len()) as u8 as f64;
    }
    let dcg: f64 = turns.iter().take(10).enumerate().filter(|(_, t)| ev.contains(*t)).map(|(i, _)| 1.0 / ((i + 2) as f64).log2()).sum();
    let idcg: f64 = (0..ev.len().min(10)).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    m.ndcg10 = if idcg > 0.0 { dcg / idcg } else { 0.0 };
    for (i, k) in [5usize, 10].iter().enumerate() {
        let got = sessions.iter().take(*k).filter(|s| evs.contains(*s)).count();
        m.s_any[i] = (got > 0) as u8 as f64;
        m.s_all[i] = (got == evs.len()) as u8 as f64;
    }
    let brief = recall::pack(hits, query, &HashSet::new(), &PackParams::brief());
    m.pack_brief = brief.iter().any(|(r, _)| ev.contains(r.tag.as_str())) as u8 as f64;
    for (budget, slot) in [(4000usize, 0), (8000, 1)] {
        // As many records per character as the product default (10 per 4,000).
        let p = PackParams { budget_chars: budget, max_records: budget * PackParams::default().max_records / PackParams::default().budget_chars, ..PackParams::default() };
        let items = recall::pack(hits, query, &HashSet::new(), &p);
        let cov = items.iter().any(|(r, _)| ev.contains(r.tag.as_str())) as u8 as f64;
        if slot == 0 {
            m.pack4k = cov;
            m.pack4k_chars = items.iter().map(|(_, s)| s.len() as f64).sum();
        } else {
            m.pack8k = cov;
        }
    }
    m
}

#[derive(Serialize)]
struct Row {
    qid: String,
    unit: String,
    qtype: String,
    abstention: bool,
    n_evidence: usize,
    foreign: bool,
    configs: HashMap<String, QMetrics>,
    evidence: recall::Evidence,
    evidence_dense: recall::Evidence,
}

/// Content hash of each session of a unit.
fn session_hashes(u: &Unit) -> HashMap<String, u64> {
    let mut acc: HashMap<String, String> = HashMap::new();
    for t in &u.turns {
        let e = acc.entry(t.session_id.clone()).or_default();
        e.push_str(&t.role);
        e.push('\u{1}');
        e.push_str(&t.text);
        e.push('\u{2}');
    }
    acc.into_iter().map(|(k, v)| (k, embed::fnv(&v))).collect()
}

fn run_unit(u: &Unit, foreign: Option<&(Question, HashSet<u64>)>, emb: &Embeds, ctx: &str, tmp: &Path, cfgs: &[(&'static str, RankParams)]) -> Vec<Row> {
    let recs = records(u, ctx);
    let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
    let vecs = emb.embed(&texts);
    let _ = std::fs::remove_dir_all(tmp);
    std::fs::create_dir_all(tmp).unwrap();
    let store = Store::open(&tmp.join("m.strata"), emb.e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
    store.bulk_insert(&recs, vecs).unwrap();
    let hashes: HashSet<u64> = session_hashes(u).into_values().collect();
    let mut rows = Vec::new();
    let mut qs: Vec<(Question, bool)> = u.questions.iter().map(|q| (q.clone(), false)).collect();
    if let Some((fq, fh)) = foreign {
        if fh.is_disjoint(&hashes) {
            qs.push((fq.clone(), true));
        }
    }
    for (q, is_foreign) in qs {
        let terms = strata::text::analyze_query(&q.question);
        let bm = if terms.is_empty() { Vec::new() } else { store.bm25_top(&q.question, 100, &Filter::default()).unwrap_or_default() };
        let qv = emb.query(&q.question);
        let dv = store.vector_top(&qv, 100, &Filter::default()).unwrap_or_default();
        let mut cm = HashMap::new();
        let mut ev_main = recall::Evidence::default();
        let mut ev_dense = recall::Evidence::default();
        for (name, p) in cfgs {
            let b = if p.use_bm25 { bm.clone() } else { Vec::new() };
            let d = if p.use_vectors { dv.clone() } else { Vec::new() };
            let (hits, ev) = recall::fuse(&store, &q.question, &terms, b, d, p);
            if *name == "hybrid_0.5_priors" {
                ev_main = ev.clone();
            }
            if *name == "dense" {
                ev_dense = ev;
            }
            if !is_foreign {
                cm.insert(name.to_string(), metrics(&hits, &q, &q.question));
            } else if *name == "hybrid_0.5_priors" {
                cm.insert(name.to_string(), QMetrics::default());
            }
        }
        rows.push(Row {
            qid: q.qid.clone(),
            unit: u.unit_id.clone(),
            qtype: q.question_type.clone(),
            abstention: q.abstention,
            n_evidence: q.evidence_turn_ids.len(),
            foreign: is_foreign,
            configs: cm,
            evidence: ev_main,
            evidence_dense: ev_dense,
        });
    }
    drop(store);
    let _ = std::fs::remove_dir_all(tmp);
    rows
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Embed { input, fields, embedder, out } => {
            let e = open_embedder(&embedder);
            let t0 = Instant::now();
            let text = std::fs::read_to_string(&input).expect("read input");
            let texts: Vec<String> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let v: Json = serde_json::from_str(l).expect("JSON line");
                    fields.iter().filter_map(|f| v.get(f).and_then(|x| x.as_str())).collect::<Vec<_>>().join("\n")
                })
                .collect();
            drop(text);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&out).expect("create output"));
            for chunk in texts.chunks(4096) {
                let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                let embs: Vec<Vec<f32>> = std::thread::scope(|sc| {
                    let parts: Vec<&[&str]> = refs.chunks(refs.len().div_ceil(8).max(1)).collect();
                    let hs: Vec<_> = parts.into_iter().map(|p| { let e = e.clone(); sc.spawn(move || e.embed(p).expect("embedding")) }).collect();
                    hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
                });
                for v in embs {
                    for x in v {
                        std::io::Write::write_all(&mut w, &x.to_le_bytes()).unwrap();
                    }
                }
            }
            eprintln!("{} texts, {} dimensions ({}), {:.1} s", texts.len(), e.dim(), e.name(), t0.elapsed().as_secs_f64());
        }
        Cmd::EmbedCheck { embedder, query, texts } => {
            let e = open_embedder(&embedder);
            let t0 = Instant::now();
            for t in &texts {
                let v = if query { e.embed_query(t).expect("embedding") } else { e.embed(&[t.as_str()]).expect("embedding").pop().unwrap() };
                println!("{}", serde_json::to_string(&v).unwrap());
            }
            eprintln!("{} texts in {:.1} ms", texts.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        Cmd::Distill { cases, model, repeats, threads, out } => distill_cmd(&cases, &model, repeats, threads, &out),
        Cmd::Deep { data, embedder, out, every, qevery, limit, threads, model, hard_model, max_picks, planner, ranker, features, tag } => {
            deep_cmd(&data, &embedder, &out, every, qevery, limit, threads, &model, &hard_model, max_picks, &planner, &ranker, features.as_deref(), &tag)
        }
        Cmd::Scale { data, sizes, embedder, out, queries, hooks, hook_gap_ms } => scale_cmd(&data, &sizes, &embedder, &out, queries, hooks, hook_gap_ms),
        Cmd::Latency { data, sizes, embedder, out, queries, rounds, hooks, knowledge } => latency_cmd(&data, &sizes, &embedder, &out, queries, rounds, hooks, knowledge),
        Cmd::Tail { data, sizes, embedder, out, queries, rounds, knowledge } => tail_cmd(&data, &sizes, &embedder, &out, queries, rounds, knowledge),
        Cmd::Concurrency { data, sizes, embedder, out, queries, readers, secs, write_every_ms, knowledge } => {
            concurrency_cmd(&data, &sizes, &embedder, &out, queries, &readers, secs, write_every_ms, knowledge)
        }
        Cmd::Writes { embedder, records, clients, out } => writes_cmd(&embedder, records, &clients, &out),
        Cmd::Probes { transcript, probes, points, embedder, project, out, cache, dump_missing, dump_probes, per_turn, max_records } => {
            let d = PackParams::default();
            let pack = PackParams { per_turn: per_turn.unwrap_or(d.per_turn), max_records: max_records.unwrap_or(d.max_records), ..d };
            probes_cmd(&transcript, &probes, &points, &embedder, &project, &out, cache.as_deref(), dump_missing.as_deref(), &dump_probes, &pack)
        }
        Cmd::Retrieval { data, embedder, ctx, out, every, limit, threads, cache, tag, dump_missing } => {
            let t0 = Instant::now();
            let e: Arc<dyn Embedder> = open_embedder(&embedder);
            let text = std::fs::read_to_string(&data).expect("read data");
            let mut units: Vec<Unit> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("unit")).collect();
            drop(text);
            units = units.into_iter().enumerate().filter(|(i, _)| i % every == 0).map(|(_, u)| u).collect();
            if let Some(n) = limit {
                units.truncate(n);
            }
            // A foreign question for each unit: one from the unit half-way round,
            // with the content hashes of its evidence sessions.
            let foreign: Vec<Option<(Question, HashSet<u64>)>> = (0..units.len())
                .map(|i| {
                    let j = (i + units.len() / 2) % units.len().max(1);
                    if j == i {
                        return None;
                    }
                    let fu = &units[j];
                    let q = fu.questions.iter().find(|q| !q.abstention && !q.evidence_turn_ids.is_empty())?.clone();
                    let hs = session_hashes(fu);
                    let ev: HashSet<u64> = q.evidence_session_ids.iter().filter_map(|s| hs.get(s).copied()).collect();
                    Some((q, ev))
                })
                .collect();
            let dataset = units.first().map(|u| u.dataset.clone()).unwrap_or_default();
            if let Some(path) = dump_missing {
                let have = cache.as_ref().map(|p| load_cache(p, e.dim())).unwrap_or_default();
                let mut seen = HashSet::new();
                let mut jl = String::new();
                for u in &units {
                    for r in records(u, &ctx) {
                        let t = r.embed_text();
                        let k = embed::fnv(&t);
                        if !have.contains_key(&k) && seen.insert(k) {
                            jl += &serde_json::to_string(&json!({"key": k.to_string(), "text": t, "query": false})).unwrap();
                            jl.push('\n');
                        }
                    }
                    for q in &u.questions {
                        let k = embed::fnv(&format!("\u{1}query\u{1}{}", q.question));
                        if !have.contains_key(&k) && seen.insert(k) {
                            jl += &serde_json::to_string(&json!({"key": k.to_string(), "text": q.question, "query": true})).unwrap();
                            jl.push('\n');
                        }
                    }
                }
                std::fs::write(&path, jl).unwrap();
                eprintln!("wrote {} missing texts to {}", seen.len(), path.display());
                return;
            }
            let ename = embedder.split(':').next().unwrap_or("x").to_string();
            let ename = if ename == "cached" { embedder.rsplit(':').next().unwrap_or("cached").to_string() } else { ename };
            let emb = Embeds {
                e: e.clone(),
                cache: cache.as_ref().map(|p| Mutex::new(load_cache(p, e.dim()))),
            };
            let cfgs = configs();
            let next = AtomicUsize::new(0);
            let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
            let done = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for t in 0..threads.max(1) {
                    let (units, foreign, emb, cfgs, next, rows, done, ctx) = (&units, &foreign, &emb, &cfgs, &next, &rows, &done, &ctx);
                    s.spawn(move || loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= units.len() {
                            break;
                        }
                        let tmp = std::env::temp_dir().join(format!("engram-bench-{}-{t}", std::process::id()));
                        let r = run_unit(&units[i], foreign[i].as_ref(), emb, ctx, &tmp, cfgs);
                        rows.lock().unwrap().extend(r);
                        let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                        if d % 25 == 0 || d == units.len() {
                            eprintln!("  {d}/{} units, {:.0} s", units.len(), t0.elapsed().as_secs_f64());
                        }
                    });
                }
            });
            if let (Some(p), Some(c)) = (&cache, &emb.cache) {
                save_cache(p, &c.lock().unwrap());
            }
            let rows = rows.into_inner().unwrap();
            // Aggregate over answerable, non-foreign questions with evidence.
            let eligible: Vec<&Row> = rows.iter().filter(|r| !r.foreign && !r.abstention && r.n_evidence > 0).collect();
            let agg = |filter: &dyn Fn(&Row) -> bool| -> Json {
                let sel: Vec<&&Row> = eligible.iter().filter(|r| filter(r)).collect();
                let mut o = serde_json::Map::new();
                o.insert("n".into(), json!(sel.len()));
                for (name, _) in &cfgs {
                    let ms: Vec<&QMetrics> = sel.iter().filter_map(|r| r.configs.get(*name)).collect();
                    let n = ms.len().max(1) as f64;
                    let mean = |f: &dyn Fn(&QMetrics) -> f64| ms.iter().map(|m| f(m)).sum::<f64>() / n;
                    o.insert(
                        name.to_string(),
                        json!({
                            "recall_any@1": mean(&|m| m.r_any[0]), "recall_any@5": mean(&|m| m.r_any[1]), "recall_any@10": mean(&|m| m.r_any[2]), "recall_any@20": mean(&|m| m.r_any[3]),
                            "recall_all@5": mean(&|m| m.r_all[1]), "recall_all@10": mean(&|m| m.r_all[2]),
                            "ndcg@10": mean(&|m| m.ndcg10),
                            "session_any@5": mean(&|m| m.s_any[0]), "session_all@5": mean(&|m| m.s_all[0]), "session_all@10": mean(&|m| m.s_all[1]),
                            "pack4k": mean(&|m| m.pack4k), "pack8k": mean(&|m| m.pack8k), "pack4k_chars": mean(&|m| m.pack4k_chars), "pack_brief": mean(&|m| m.pack_brief),
                        }),
                    );
                }
                Json::Object(o)
            };
            let mut by_type = serde_json::Map::new();
            let types: HashSet<String> = eligible.iter().map(|r| r.qtype.clone()).collect();
            let mut types: Vec<String> = types.into_iter().collect();
            types.sort();
            for ty in &types {
                by_type.insert(ty.clone(), agg(&|r: &Row| &r.qtype == ty));
            }
            let summary = json!({
                "dataset": dataset, "embedder": e.name(), "ctx": ctx, "units": units.len(), "questions": rows.iter().filter(|r| !r.foreign).count(),
                "eligible": eligible.len(), "foreign": rows.iter().filter(|r| r.foreign).count(), "seconds": t0.elapsed().as_secs_f64(),
                "all": agg(&|_| true), "by_type": by_type,
            });
            std::fs::create_dir_all(&out).unwrap();
            let base = format!("{dataset}_{ename}_{ctx}{}", if tag.is_empty() { String::new() } else { format!("_{tag}") });
            std::fs::write(out.join(format!("{base}.json")), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
            let mut jl = String::new();
            for r in &rows {
                jl += &serde_json::to_string(r).unwrap();
                jl.push('\n');
            }
            std::fs::write(out.join(format!("{base}.rows.jsonl")), jl).unwrap();
            let a = &summary["all"];
            eprintln!("{base}: {} eligible questions in {:.0} s", eligible.len(), t0.elapsed().as_secs_f64());
            for (name, _) in &cfgs {
                let c = &a[*name];
                eprintln!(
                    "  {name:<20} R@5 {:.3}  R@10 {:.3}  all@10 {:.3}  nDCG@10 {:.3}  sess_all@5 {:.3}  pack4k {:.3}  pack8k {:.3}",
                    c["recall_any@5"].as_f64().unwrap_or(0.0),
                    c["recall_any@10"].as_f64().unwrap_or(0.0),
                    c["recall_all@10"].as_f64().unwrap_or(0.0),
                    c["ndcg@10"].as_f64().unwrap_or(0.0),
                    c["session_all@5"].as_f64().unwrap_or(0.0),
                    c["pack4k"].as_f64().unwrap_or(0.0),
                    c["pack8k"].as_f64().unwrap_or(0.0)
                );
            }
        }
    }
}

// ---------------------------------------------------------------- distill

fn distill_cmd(cases: &Path, model: &str, repeats: usize, threads: usize, out: &Path) {
    use engram::distill::{self, DistillConfig};
    let t0 = Instant::now();
    let cases: Vec<Json> = std::fs::read_to_string(cases).expect("read cases").lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("case")).collect();
    let cfg = DistillConfig { model: model.into(), ..Default::default() };
    let jobs: Vec<(usize, usize)> = (0..cases.len()).flat_map(|i| (0..repeats).map(move |r| (i, r))).collect();
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Json>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            let (cases, jobs, next, rows, cfg) = (&cases, &jobs, &next, &rows, &cfg);
            s.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= jobs.len() {
                    break;
                }
                let (i, rep) = jobs[k];
                let c = &cases[i];
                let st = |k: &str| c[k].as_str().unwrap_or("").to_string();
                let commands: Vec<String> = c["commands"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
                let known: Vec<(String, Kind, String)> = c["known"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| {
                                let kind = match x[1].as_str()? {
                                    "rule" => Kind::Preference,
                                    "requirement" => Kind::Todo,
                                    k => Kind::parse(k)?,
                                };
                                Some((x[0].as_str()?.to_string(), kind, x[2].as_str()?.to_string()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let row = match distill::distill(cfg, &st("request"), &st("answer"), &commands, &known) {
                    Ok(d) => json!({
                        "id": st("id"), "repeat": rep, "cost_usd": d.cost_usd, "tokens": d.usage.json(),
                        "items": d.drafts.iter().map(|x| json!({"kind": distill::label(x.kind), "text": x.text, "replaces": x.links.iter().filter(|l| l.label == "supersedes").map(|l| l.to.as_str()).collect::<Vec<_>>(), "entities": x.entities, "when": x.when})).collect::<Vec<_>>(),
                        "done": d.done, "error": null,
                    }),
                    Err(e) => json!({"id": st("id"), "repeat": rep, "items": [], "done": [], "error": e.to_string()}),
                };
                rows.lock().unwrap().push(row);
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| (a["id"].as_str(), a["repeat"].as_u64()).cmp(&(b["id"].as_str(), b["repeat"].as_u64())));
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(out, rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect::<String>()).unwrap();
    let cost: f64 = rows.iter().filter_map(|r| r["cost_usd"].as_f64()).sum();
    eprintln!("{} outputs ({} cases × {repeats}) in {:.0} s, ${cost:.4}: {}", rows.len(), cases.len(), t0.elapsed().as_secs_f64(), out.display());
}

// ---------------------------------------------------------------- deep

/// A record as a reader context shows it (the layout of engram-structure's dumps).
fn reader_block(r: &Record) -> String {
    let role = if r.kind == Kind::Answer { "assistant" } else { "user" };
    format!("[{} · session {}]\n{role}: {}", recall::fmt_time(r.ts), r.session, r.text)
}

#[allow(clippy::too_many_arguments)]
fn deep_cmd(
    data: &Path, embedder: &str, out: &Path, every: usize, qevery: usize, limit: Option<usize>, threads: usize, model: &str, hard_model: &str, max_picks: usize, planner: &str, ranker: &str,
    features: Option<&Path>, tag: &str,
) {
    use engram::deep::{self, DeepConfig};
    let t0 = Instant::now();
    let e: Arc<dyn Embedder> = open_embedder(embedder);
    let text = std::fs::read_to_string(data).expect("read data");
    let mut units: Vec<Unit> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("unit")).collect();
    drop(text);
    units = units.into_iter().enumerate().filter(|(i, _)| i % every.max(1) == 0).map(|(_, u)| u).collect();
    if let Some(n) = limit {
        units.truncate(n);
    }
    let dataset = units.first().map(|u| u.dataset.clone()).unwrap_or_default();
    let cfg = DeepConfig {
        mode: "tool".into(),
        planner_model: model.into(),
        ranker_model: model.into(),
        hard_model: hard_model.into(),
        max_picks,
        planner: planner.into(),
        ranker: ranker.into(),
        ..Default::default()
    };
    let cli = engram::distill::DistillConfig::default();
    let jobs: Vec<(usize, usize)> = units.iter().enumerate().flat_map(|(i, u)| (0..u.questions.len()).filter(move |j| j % qevery.max(1) == 0).map(move |j| (i, j))).collect();
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Json>> = Mutex::new(Vec::new());
    let ctxs: Mutex<Vec<Json>> = Mutex::new(Vec::new());
    let feats: Mutex<Vec<Json>> = Mutex::new(Vec::new());
    let want_features = features.is_some();
    std::thread::scope(|s| {
        for t in 0..threads.max(1) {
            let (units, jobs, next, rows, ctxs, feats, e, cfg, cli) = (&units, &jobs, &next, &rows, &ctxs, &feats, &e, &cfg, &cli);
            s.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= jobs.len() {
                    break;
                }
                let (i, j) = jobs[k];
                let (u, q) = (&units[i], &units[i].questions[j]);
                let tmp = std::env::temp_dir().join(format!("engram-deep-{}-{t}", std::process::id()));
                let _ = std::fs::remove_dir_all(&tmp);
                std::fs::create_dir_all(&tmp).unwrap();
                let recs = records(u, "request");
                let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let vecs = e.embed(&refs).expect("embedding");
                let store = Store::open(&tmp.join("m.strata"), e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
                store.bulk_insert(&recs, vecs).unwrap();
                let now = q.question_date.as_deref().map(parse_time).unwrap_or_else(|| recs.iter().map(|r| r.ts).max().unwrap_or(0) + 86_400);
                let (hits, _) = recall::search(&store, &q.question, &Filter::default(), &RankParams { n: 30, ..Default::default() }).unwrap();
                let base = metrics(&hits, q, &q.question);
                let res = deep::run(&store, cfg, cli, &q.question, &[], &Filter::default(), None, now);
                let ev: HashSet<&str> = q.evidence_turn_ids.iter().map(|s| s.as_str()).collect();
                let cover = |tags: &[&str]| -> (f64, f64) {
                    let got: HashSet<&str> = tags.iter().copied().filter(|t| ev.contains(t)).collect();
                    ((!got.is_empty()) as u8 as f64, (!ev.is_empty() && got.len() == ev.len()) as u8 as f64)
                };
                if want_features {
                    let fs = deep::features(&store, &q.question, &res.candidates);
                    let picked: HashSet<&str> = res.picks.iter().map(|(r, _)| r.id.as_str()).collect();
                    let cands: Vec<Json> = res.candidates.iter().zip(fs).map(|(c, f)| json!({"f": f.to_vec(), "y": ev.contains(c.record.tag.as_str()) as u8, "picked": picked.contains(c.record.id.as_str()) as u8})).collect();
                    feats.lock().unwrap().push(json!({"qid": q.qid, "qtype": q.question_type, "abstention": q.abstention, "n_evidence": ev.len(), "cands": cands}));
                }
                let pool: Vec<&str> = res.candidates.iter().map(|c| c.record.tag.as_str()).collect();
                let picks: Vec<&str> = res.picks.iter().map(|(r, _)| r.tag.as_str()).collect();
                let base10: Vec<&str> = hits.iter().take(10).map(|h| h.record.tag.as_str()).collect();
                let base_pool: Vec<&str> = hits.iter().take(res.candidates.len().max(1)).map(|h| h.record.tag.as_str()).collect();
                let (pool_any, pool_all) = cover(&pool);
                let (pick_any, pick_all) = cover(&picks);
                let (b10_any, b10_all) = cover(&base10);
                let (bp_any, bp_all) = cover(&base_pool);
                let mut picked = String::new();
                for (r, _) in &res.picks {
                    picked += &reader_block(r);
                    picked += "\n\n";
                }
                let reading = if res.answer.is_empty() { picked.clone() } else { format!("A first reading of these records (by a smaller model; check it against them): {}\n\n{picked}", res.answer) };
                let row = json!({
                    "qid": q.qid, "unit": u.unit_id, "qtype": q.question_type, "abstention": q.abstention, "n_evidence": ev.len(),
                    "base_r_any@10": b10_any, "base_r_all@10": b10_all, "base_pool_any": bp_any, "base_pool_all": bp_all, "base_pack4k": base.pack4k,
                    "pool_any": pool_any, "pool_all": pool_all, "pick_any": pick_any, "pick_all": pick_all,
                    "pool": pool.len(), "picks": picks.len(), "picked_chars": picked.len(),
                    "queries": res.plan.queries.iter().map(|q| q.label()).collect::<Vec<_>>(), "hard": res.plan.hard as u8, "need": res.plan.need,
                    "confidence": res.confidence, "answer": res.answer, "gold": q.answer_text(), "ranker_model": res.ranker_model,
                    "cost_usd": res.cost_usd, "tokens": res.usage.json(), "plan_ms": res.plan.ms, "ms": res.ms, "error": res.error.clone().or(res.plan.error.clone()),
                });
                // The plain hybrid search's top 10, for reading without the query engine.
                let top10: String = hits.iter().take(10).map(|h| reader_block(&h.record) + "\n\n").collect();
                ctxs.lock().unwrap().push(json!({"qid": q.qid, "context": picked.trim_end(), "reading": reading.trim_end(), "chars": picked.len(), "base10": top10.trim_end()}));
                let n = {
                    let mut g = rows.lock().unwrap();
                    g.push(row);
                    g.len()
                };
                if n % 10 == 0 || n == jobs.len() {
                    eprintln!("  {n}/{} questions, {:.0} s", jobs.len(), t0.elapsed().as_secs_f64());
                }
                drop(store);
                let _ = std::fs::remove_dir_all(&tmp);
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let eligible: Vec<&Json> = rows.iter().filter(|r| !r["abstention"].as_bool().unwrap_or(false) && r["n_evidence"].as_u64().unwrap_or(0) > 0).collect();
    let mean = |sel: &[&Json], k: &str| sel.iter().map(|r| r[k].as_f64().unwrap_or(0.0)).sum::<f64>() / sel.len().max(1) as f64;
    let keys = ["base_r_any@10", "base_r_all@10", "base_pool_any", "base_pool_all", "base_pack4k", "pool_any", "pool_all", "pick_any", "pick_all", "pool", "picks", "picked_chars"];
    let agg = |sel: &[&Json]| -> Json {
        let mut o = serde_json::Map::new();
        o.insert("n".into(), json!(sel.len()));
        for k in keys {
            o.insert(k.into(), json!(mean(sel, k)));
        }
        Json::Object(o)
    };
    let mut types: Vec<String> = eligible.iter().filter_map(|r| r["qtype"].as_str().map(String::from)).collect::<HashSet<_>>().into_iter().collect();
    types.sort();
    let by_type: serde_json::Map<String, Json> = types.iter().map(|ty| (ty.clone(), agg(&eligible.iter().copied().filter(|r| r["qtype"] == ty.as_str()).collect::<Vec<_>>()))).collect();
    let mut ms: Vec<f64> = rows.iter().filter_map(|r| r["ms"].as_f64()).collect();
    let all: Vec<&Json> = rows.iter().collect();
    let tok = |k: &str| rows.iter().filter_map(|r| r["tokens"][k].as_u64()).sum::<u64>();
    let summary = json!({
        "dataset": dataset, "embedder": e.name(), "model": model, "hard_model": hard_model, "max_picks": max_picks, "planner": planner, "ranker": ranker,
        "questions": rows.len(), "eligible": eligible.len(), "seconds": t0.elapsed().as_secs_f64(),
        "cost_usd": rows.iter().filter_map(|r| r["cost_usd"].as_f64()).sum::<f64>(),
        "tokens": {"input": tok("input"), "cache_read": tok("cache_read"), "cache_write": tok("cache_write"), "output": tok("output")},
        "ms_p50": pct(&mut ms, 0.5), "ms_p90": pct(&mut ms, 0.9),
        "errors": rows.iter().filter(|r| !r["error"].is_null()).count(),
        "hard": mean(&all, "hard"), "all": agg(&eligible), "by_type": by_type,
    });
    std::fs::create_dir_all(out).unwrap();
    let base = format!("{dataset}_deep{}", if tag.is_empty() { String::new() } else { format!("_{tag}") });
    std::fs::write(out.join(format!("{base}.json")), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    let lines = |v: &[Json]| v.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect::<String>();
    std::fs::write(out.join(format!("{base}.rows.jsonl")), lines(&rows)).unwrap();
    std::fs::write(out.join(format!("{base}.contexts.jsonl")), lines(&ctxs.into_inner().unwrap())).unwrap();
    if let Some(p) = features {
        std::fs::write(p, lines(&feats.into_inner().unwrap())).unwrap();
    }
    let a = &summary["all"];
    let f = |k: &str| a[k].as_f64().unwrap_or(0.0);
    eprintln!("{base}: {} questions ({} with evidence) in {:.0} s, ${:.3}", rows.len(), eligible.len(), t0.elapsed().as_secs_f64(), summary["cost_usd"].as_f64().unwrap_or(0.0));
    eprintln!("  hybrid top-10   any {:.3} all {:.3}", f("base_r_any@10"), f("base_r_all@10"));
    eprintln!("  hybrid top-pool any {:.3} all {:.3}", f("base_pool_any"), f("base_pool_all"));
    eprintln!("  deep pool       any {:.3} all {:.3} ({:.1} records)", f("pool_any"), f("pool_all"), f("pool"));
    eprintln!("  deep picks      any {:.3} all {:.3} ({:.1} records, {:.0} chars)", f("pick_any"), f("pick_all"), f("picks"), f("picked_chars"));
}

// ---------------------------------------------------------------- probes

#[derive(Deserialize)]
struct Probe {
    q: String,
    a: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn probes_cmd(transcript: &Path, probes: &Path, points: &[String], embedder: &str, project: &str, out: &Path, cache: Option<&Path>, dump: Option<&Path>, dump_probes: &[PathBuf], pack: &PackParams) {
    use engram::recall::Gate;
    use engram::transcript as tr;
    let probes: Vec<Probe> = serde_json::from_str(&std::fs::read_to_string(probes).expect("probes")).expect("probes json");
    let pats: Vec<Vec<regex::Regex>> = probes.iter().map(|p| p.a.iter().map(|a| regex::Regex::new(a).expect("regex")).collect()).collect();
    let inner = open_embedder(embedder);
    let cached = Arc::new(Mutex::new(cache.map(|p| load_cache(p, inner.dim())).unwrap_or_default()));
    let e: Arc<dyn Embedder> = if cache.is_some() { Arc::new(CacheEmbedder { inner: inner.clone(), cache: cached.clone(), missing: AtomicUsize::new(0) }) } else { inner.clone() };
    let events = tr::read(transcript).expect("transcript");
    let batches = tr::drafts(&events, project);
    if let Some(path) = dump {
        let have = cached.lock().unwrap();
        let mut seen = HashSet::new();
        let mut jl = String::new();
        let mut push = |key: u64, text: &str, query: bool| {
            if !have.contains_key(&key) && seen.insert(key) {
                jl += &serde_json::to_string(&json!({"key": key.to_string(), "text": text, "query": query})).unwrap();
                jl.push('\n');
            }
        };
        for b in &batches {
            for d in &b.drafts {
                let r = Record::from_draft(d.clone(), &b.session, b.turn, 0, b.ts);
                let t = r.embed_text();
                push(embed::fnv(&t), &t, false);
            }
        }
        for p in &probes {
            push(query_key(&p.q), &p.q, true);
        }
        for pf in dump_probes {
            let ps: Vec<Probe> = serde_json::from_str(&std::fs::read_to_string(pf).expect("probes")).expect("probes json");
            for p in ps {
                push(query_key(&p.q), &p.q, true);
            }
        }
        std::fs::write(path, jl).unwrap();
        eprintln!("wrote {} missing texts to {}", seen.len(), path.display());
        return;
    }
    let gate = Gate::for_embedder(embedder);
    let rank = RankParams::default();
    let mut rows = Vec::new();
    for pt in points {
        let (w, a) = pt.split_once(':').expect("window:ask");
        let (w, a): (usize, usize) = (w.parse().unwrap(), a.parse().unwrap());
        for fresh in [false, true] {
            let mut recs = Vec::new();
            let mut seq = 1i64;
            let mut window_seq = i64::MAX;
            let mut session = String::new();
            for b in batches.iter().filter(|b| b.line < a) {
                if b.line >= w && window_seq == i64::MAX {
                    window_seq = seq;
                }
                session = b.session.clone();
                for d in &b.drafts {
                    recs.push(Record::from_draft(d.clone(), &b.session, b.turn, seq, b.ts));
                    seq += 1;
                }
            }
            let tmp = std::env::temp_dir().join(format!("engram-probes-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
            let store = Store::open(&tmp.join("m.strata"), e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
            let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
            let t0 = Instant::now();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embs = e.embed(&refs).expect("embedding");
            let embed_s = t0.elapsed().as_secs_f64();
            store.bulk_insert(&recs, embs).unwrap();
            let filter = if fresh { Filter::default() } else { Filter { exclude_window: Some((session.clone(), window_seq)), ..Default::default() } };
            let eligible = recs.iter().filter(|r| fresh || r.seq < window_seq).count();
            eprintln!("point {pt} fresh={fresh}: {} records ({eligible} outside the window), embedded in {embed_s:.1} s", recs.len());
            for (i, p) in probes.iter().enumerate() {
                let t1 = Instant::now();
                let (hits, ev) = recall::search(&store, &p.q, &filter, &rank).unwrap();
                let ms = t1.elapsed().as_secs_f64() * 1e3;
                let hit = |t: &str| pats[i].iter().any(|r| r.is_match(t));
                let items = recall::pack(&hits, &p.q, &HashSet::new(), pack);
                let inject = recall::fmt_recall(&items, 0);
                let decision = gate.decide(&ev);
                let auto = match decision {
                    recall::Decision::Inject => inject.clone(),
                    recall::Decision::Brief => recall::fmt_recall_as(&recall::pack(&hits, &p.q, &HashSet::new(), &PackParams::brief()), 0, true),
                    recall::Decision::Skip => String::new(),
                };
                let terms: HashSet<String> = strata::text::analyze_query(&p.q).into_iter().collect();
                let pull: String = hits.iter().take(8).map(|h| recall::snippet(&h.record.text, &terms, 700)).collect::<Vec<_>>().join("\n");
                let first_rank = hits.iter().position(|h| hit(&h.record.text)).map(|r| r + 1);
                let eligible_has = recs.iter().any(|r| (fresh || r.seq < window_seq) && hit(&r.text));
                rows.push(json!({
                    "point": pt, "fresh": fresh, "probe": i, "q": p.q, "stored": eligible_has,
                    "inject_hit": hit(&inject), "auto_hit": hit(&auto), "auto_chars": auto.len(), "pull8_hit": hit(&pull), "first_rank": first_rank,
                    "gate": gate.score(&ev), "decision": format!("{decision:?}"), "evidence": ev,
                    "inject_chars": inject.len(), "ms": ms,
                }));
            }
            drop(store);
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
    std::fs::create_dir_all(out).unwrap();
    let name = format!("session_probes_{}.json", embedder.split(':').next().unwrap_or("x"));
    std::fs::write(out.join(&name), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    for pt in points {
        for fresh in [false, true] {
            let sel: Vec<&Json> = rows.iter().filter(|r| r["point"] == json!(pt) && r["fresh"] == json!(fresh) && r["stored"] == json!(true)).collect();
            let n = sel.len().max(1) as f64;
            let c = |k: &str| sel.iter().filter(|r| r[k] == json!(true)).count() as f64 / n;
            let r1 = sel.iter().filter(|r| r["first_rank"].as_u64() == Some(1)).count() as f64 / n;
            eprintln!("point {pt} fresh={fresh}: {} probes answerable from the store; injected {:.2}, gated {:.2}, pull@8 {:.2}, rank 1 {:.2}", sel.len(), c("inject_hit"), c("auto_hit"), c("pull8_hit"), r1);
        }
    }
}

// ---------------------------------------------------------------- scale

/// Needle in a haystack: the evidence ("answer_") sessions of every fifth
/// unit's question, mixed at random positions into generic filler sessions
/// (no evidence) repeated until there are `n` records. Returns the records,
/// the questions with their evidence texts, and the evidence record count.
#[allow(clippy::type_complexity)]
fn haystack(units: &[Unit], n: usize, n_queries: usize) -> (Vec<Record>, Vec<(Question, HashSet<String>)>, usize) {
    let mut evidence: Vec<Vec<Record>> = Vec::new();
    let mut qs: Vec<(Question, HashSet<String>)> = Vec::new();
    let mut filler: Vec<Vec<Record>> = Vec::new();
    let mut seen_sessions: HashSet<String> = HashSet::new();
    for (ui, u) in units.iter().enumerate() {
        let ur = records(u, "round");
        let mut by_session: Vec<(String, Vec<Record>)> = Vec::new();
        for r in ur {
            match by_session.last_mut() {
                Some((sid, v)) if *sid == r.session => v.push(r),
                _ => by_session.push((r.session.clone(), vec![r])),
            }
        }
        if ui % 5 == 0 {
            if let Some(q) = u.questions.iter().find(|q| !q.abstention && !q.evidence_turn_ids.is_empty()) {
                let ids: HashSet<&str> = q.evidence_turn_ids.iter().map(|s| s.as_str()).collect();
                let mut ev = HashSet::new();
                for (sid, v) in &by_session {
                    if q.evidence_session_ids.contains(sid) {
                        ev.extend(v.iter().filter(|r| ids.contains(r.tag.as_str())).map(|r| r.text.clone()));
                        let mut v = v.clone();
                        v.iter_mut().for_each(|r| r.session = format!("u{ui}:{}", r.session));
                        evidence.push(v);
                    }
                }
                if !ev.is_empty() {
                    qs.push((q.clone(), ev));
                }
            }
        }
        for (sid, v) in by_session {
            if !sid.starts_with("answer_") && seen_sessions.insert(sid) {
                filler.push(v);
            }
        }
    }
    qs.truncate(n_queries);
    let ev_records: usize = evidence.iter().map(|v| v.len()).sum();
    // Deterministic shuffle of session order (evidence among filler).
    let mut sessions: Vec<Vec<Record>> = evidence;
    let mut copy = 0usize;
    let mut total = ev_records;
    'fill: loop {
        for f in &filler {
            if total >= n {
                break 'fill;
            }
            let mut v = f.clone();
            if copy > 0 {
                v.iter_mut().for_each(|r| r.session = format!("c{copy}:{}", r.session));
            }
            total += v.len();
            sessions.push(v);
        }
        copy += 1;
    }
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    for i in (1..sessions.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        sessions.swap(i, (rng % (i as u64 + 1)) as usize);
    }
    let mut recs: Vec<Record> = Vec::with_capacity(total);
    for v in sessions {
        for mut r in v {
            let seq = recs.len() as i64 + 1;
            r.seq = seq;
            r.id = format!("r{seq}");
            recs.push(r);
        }
    }
    (recs, qs, ev_records)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn scale_cmd(data: &Path, sizes: &[usize], embedder: &str, out: &Path, n_queries: usize, n_hooks: usize, hook_gap_ms: u64) {
    let e = open_embedder(embedder);
    let text = std::fs::read_to_string(data).expect("data");
    let units: Vec<Unit> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("unit")).collect();
    drop(text);
    let mut results = Vec::new();
    for &n in sizes {
        eprintln!("== {n} records");
        let (recs, qs, ev_records) = haystack(&units, n, n_queries);
        let n = recs.len();
        let chars: usize = recs.iter().map(|r| r.text.len() + r.ctx.len()).sum();

        let t0 = Instant::now();
        let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let chunks: Vec<&[&str]> = refs.chunks((refs.len() / 4).max(1)).collect();
            let hs: Vec<_> = chunks.into_iter().map(|c| { let e = e.clone(); s.spawn(move || e.embed(c).expect("embedding")) }).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let embed_s = t0.elapsed().as_secs_f64();
        drop(texts);

        // The store lives where the daemon will look for it.
        let home = std::env::temp_dir().join(format!("engram-scale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let project = home.join("proj");
        std::fs::create_dir_all(&project).unwrap();
        std::env::set_var("ENGRAM_HOME", &home);
        // Hooks and daemons time the hot path only: no model calls.
        std::env::set_var("ENGRAM_DEEP", "off");
        std::env::set_var("ENGRAM_DISTILL", "0");
        let dir = engram::paths::store_dir(&project);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_string(&json!({"embedder": embedder})).unwrap()).unwrap();
        let db = dir.join("memory.strata");
        let t1 = Instant::now();
        {
            let store = Store::open(&db, e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
            store.bulk_insert(&recs, embs).unwrap();
        }
        let load_s = t1.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&db).map(|m| m.len()).unwrap_or(0);

        // Reopen as the daemon would, then appends and queries.
        let t2 = Instant::now();
        let store = Store::open(&db, e.clone(), StoreOptions::default()).unwrap();
        let open_ms = t2.elapsed().as_secs_f64() * 1e3;
        let mut append_ms = Vec::new();
        for i in 0..200 {
            let drafts = vec![
                engram::record::Draft::new(Kind::ToolCall, format!("$ cargo test --package p{i}")),
                engram::record::Draft::new(Kind::ToolResult, format!("test result: ok. {i} passed; 0 failed")),
                engram::record::Draft::new(Kind::Answer, format!("All {i} tests pass after the fix to the parser.")),
            ];
            let t = Instant::now();
            store.append("bench-live", i, 0, drafts).unwrap();
            append_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let rank = RankParams::default();
        let gate = recall::Gate::default();
        let filter = Filter { exclude_window: Some(("bench-live".into(), 0)), ..Default::default() };
        let mut q_ms = Vec::new();
        let mut hit10 = 0usize;
        let mut pack_hit = 0usize;
        for (q, ev) in &qs {
            let t = Instant::now();
            let (hits, evd) = recall::search(&store, &q.question, &filter, &rank).unwrap();
            let _ = gate.decide(&evd);
            let items = recall::pack(&hits, &q.question, &HashSet::new(), &PackParams::default());
            q_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let mut seen = Vec::new();
            for h in &hits {
                if !seen.contains(&h.record.tag) {
                    seen.push(h.record.tag.clone());
                }
                if seen.len() > 10 {
                    break;
                }
                if ev.contains(&h.record.text) {
                    hit10 += 1;
                    break;
                }
            }
            pack_hit += items.iter().any(|(r, _)| ev.contains(&r.text)) as usize;
        }
        drop(store);

        // Hooks through the real binary and daemon (process start included).
        let exe = std::env::current_exe().unwrap().with_file_name("engram");
        let mut hook_ms = Vec::new();
        let mut first_ms = 0.0;
        for i in 0..n_hooks.min(qs.len().max(1)) {
            std::thread::sleep(std::time::Duration::from_millis(hook_gap_ms));
            let prompt = qs.get(i).map(|q| q.0.question.clone()).unwrap_or_else(|| "what did we decide".into());
            let payload = json!({"session_id": "bench-hook", "hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": project});
            let t = Instant::now();
            let mut child = std::process::Command::new(&exe)
                .args(["hook", "UserPromptSubmit"])
                .env("ENGRAM_HOME", &home)
                .env("CLAUDE_PROJECT_DIR", &project)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            {
                use std::io::Write;
                child.stdin.take().unwrap().write_all(payload.to_string().as_bytes()).unwrap();
            }
            child.wait().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if i == 0 {
                first_ms = ms;
            } else {
                hook_ms.push(ms);
            }
        }
        let _ = engram::daemon::request(&dir, &project.to_string_lossy(), &json!({"op": "shutdown"}), std::time::Duration::from_secs(5));
        std::thread::sleep(std::time::Duration::from_millis(300));
        let r = json!({
            "records": n, "evidence_records": ev_records, "chars": chars, "embed_s": embed_s, "embed_records_per_s": n as f64 / embed_s,
            "bulk_load_s": load_s, "file_bytes": bytes, "bytes_per_record": bytes as f64 / n as f64,
            "open_ms": open_ms,
            "append_ms_p50": pct(&mut append_ms.clone(), 0.5), "append_ms_p99": pct(&mut append_ms, 0.99),
            "query_ms_p50": pct(&mut q_ms.clone(), 0.5), "query_ms_p99": pct(&mut q_ms, 0.99), "queries": qs.len(),
            "recall10": hit10 as f64 / qs.len().max(1) as f64, "pack4k": pack_hit as f64 / qs.len().max(1) as f64,
            "hook_gap_ms": hook_gap_ms, "hook_first_ms": first_ms, "hook_ms_p50": pct(&mut hook_ms.clone(), 0.5), "hook_ms_p99": pct(&mut hook_ms, 0.99),
        });
        eprintln!("{}", serde_json::to_string(&r).unwrap());
        results.push(r);
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(out).unwrap();
        std::fs::write(out.join(format!("scale_{}.json", embedder.split(':').next().unwrap_or("x"))), serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
}

// ---------------------------------------------------------------- latency

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn stats(v: &[f64]) -> Json {
    let mut v = v.to_vec();
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({"p50": pct(&mut v, 0.5), "p90": pct(&mut v, 0.9), "p99": pct(&mut v, 0.99), "mean": mean, "n": v.len()})
}

/// The IDF lookup Engram used before: one COUNT query per term.
fn idf_by_count(store: &Store, terms: &[String]) -> Vec<f64> {
    let n = store.len().max(1) as f64;
    terms
        .iter()
        .map(|t| {
            let rs = store.query(&format!("SELECT COUNT(*) FROM memory WHERE MATCH(body, '{}')", t.replace(['\'', '"', '*', '(', ')', '-'], "")));
            let df = rs.ok().and_then(|r| r.rows.first().and_then(|x| x[0].as_i64())).unwrap_or(0).max(0) as f64;
            (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
        })
        .collect()
}

fn run_hook(exe: &Path, home: &Path, project: &Path, event: &str, payload: &Json) -> f64 {
    let t = Instant::now();
    let mut child = std::process::Command::new(exe)
        .args(["hook", event])
        .env("ENGRAM_HOME", home)
        .env("CLAUDE_PROJECT_DIR", project)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        child.stdin.take().unwrap().write_all(payload.to_string().as_bytes()).unwrap();
    }
    child.wait().unwrap();
    ms_since(t)
}

#[allow(clippy::too_many_arguments)]
fn latency_cmd(data: &Path, sizes: &[usize], embedder: &str, out: &Path, n_queries: usize, rounds: usize, n_hooks: usize, knowledge: f64) {
    let e = open_embedder(embedder);
    let units = engram::benchdata::read_units(data, 1, None).expect("data");
    let mut results = Vec::new();
    std::fs::create_dir_all(out).unwrap();
    for &n in sizes {
        eprintln!("== {n} records");
        let (mut recs, qs, _) = haystack(&units, n, n_queries);
        // A share of the records as knowledge, spread evenly, so the
        // per-prompt knowledge query has a realistic set to search.
        if knowledge > 0.0 {
            let every = (1.0 / knowledge).round().max(1.0) as usize;
            let kinds = [Kind::Decision, Kind::Preference, Kind::Fact];
            for (i, r) in recs.iter_mut().enumerate().filter(|(i, _)| i % every == 0) {
                r.kind = kinds[(i / every) % 3];
                r.tag = "distilled".into();
            }
        }
        let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let chunks: Vec<&[&str]> = refs.chunks((refs.len() / 4).max(1)).collect();
            let hs: Vec<_> = chunks.into_iter().map(|c| { let e = e.clone(); s.spawn(move || e.embed(c).expect("embedding")) }).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        drop(texts);
        let home = std::env::temp_dir().join(format!("engram-latency-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let project = home.join("proj");
        std::fs::create_dir_all(&project).unwrap();
        std::env::set_var("ENGRAM_HOME", &home);
        // Hooks and daemons time the hot path only: no model calls.
        std::env::set_var("ENGRAM_DEEP", "off");
        std::env::set_var("ENGRAM_DISTILL", "0");
        let dir = engram::paths::store_dir(&project);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_string(&json!({"embedder": embedder})).unwrap()).unwrap();
        let db = dir.join("memory.strata");
        {
            let store = Store::open(&db, e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
            store.bulk_insert(&recs, embs).unwrap();
        }
        let n = recs.len();
        drop(recs);
        let mut row = serde_json::Map::new();
        row.insert("records".into(), json!(n));
        row.insert("knowledge_share".into(), json!(knowledge));

        // 0. What a daemon start pays for the embedder: opening it now (the
        //    table is mapped), against reading and converting the whole
        //    table as Engram did before.
        if embedder == "static" {
            let dir = embed::models_dir().join("potion-retrieval-32M");
            let t = Instant::now();
            let fresh = embed::open("static").unwrap();
            let _ = fresh.embed_query("warm up the first rows").expect("embedding");
            let open_ms = ms_since(t);
            let t = Instant::now();
            let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
            let floats: Vec<f32> = bytes[bytes.len() % 4..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let read_ms = ms_since(t);
            std::hint::black_box(floats.len());
            eprintln!("  embedder open {open_ms:.1} ms (mapped); reading the table {read_ms:.1} ms");
            row.insert("embedder_open_ms".into(), json!(open_ms));
            row.insert("table_read_ms".into(), json!(read_ms));
        }

        // 1. Where a prompt's time goes (warm store, nothing pending).
        {
            let store = Store::open(&db, e.clone(), StoreOptions::default()).unwrap();
            store.warm();
            let filter = Filter { exclude_window: Some(("bench-live".into(), i64::MAX)), ..Default::default() };
            let rank = RankParams::default();
            let mut ph: HashMap<&str, Vec<f64>> = HashMap::new();
            for (q, _) in qs.iter().cycle().take(qs.len() * 2) {
                let terms = strata::text::analyze_query(&q.question);
                let t = Instant::now();
                let v = store.embedder.embed_query(&q.question).expect("embedding");
                ph.entry("embed_query").or_default().push(ms_since(t));
                let t = Instant::now();
                let bm = store.bm25_top(&q.question, rank.n, &filter).unwrap();
                ph.entry("bm25_top50").or_default().push(ms_since(t));
                let t = Instant::now();
                let dv = store.vector_top(&v, rank.n, &filter).unwrap();
                ph.entry("vector_top50").or_default().push(ms_since(t));
                let t = Instant::now();
                let _ = store.idf(&terms).unwrap();
                ph.entry("idf_dictionary").or_default().push(ms_since(t));
                let t = Instant::now();
                let _ = idf_by_count(&store, &terms);
                ph.entry("idf_count_queries").or_default().push(ms_since(t));
                let t = Instant::now();
                let (hits, _) = recall::fuse(&store, &q.question, &terms, bm, dv, &rank);
                ph.entry("fuse").or_default().push(ms_since(t));
                let t = Instant::now();
                let items = recall::pack(&hits, &q.question, &HashSet::new(), &PackParams::default());
                let _ = recall::fmt_recall(&items, 0);
                ph.entry("pack_format").or_default().push(ms_since(t));
                let t = Instant::now();
                let _ = recall::search(&store, &q.question, &filter, &rank).unwrap();
                ph.entry("search_total").or_default().push(ms_since(t));
            }
            let m: serde_json::Map<String, Json> = ph.iter().map(|(k, v)| (k.to_string(), stats(v))).collect();
            eprintln!("  phases: {}", m.iter().map(|(k, v)| format!("{k} {:.2}", v["p50"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(", "));
            row.insert("phases".into(), Json::Object(m));
        }

        // 2. A write, then a query right away (a tool call followed by the
        //    next hook), under memtable settings. Each configuration starts
        //    from the bulk-loaded file: its appends stay in the memtable and
        //    the WAL (fewer than a flush's worth), and the WAL is deleted
        //    afterwards, which leaves the file as it was.
        let wal = { let mut w = db.as_os_str().to_owned(); w.push("-wal"); PathBuf::from(w) };
        let mut stall = serde_json::Map::new();
        for (name, chunk, flat) in [("chunk1024_hnsw", 1024usize, false), ("chunk1024_flat", 1024, true), ("chunk256_flat", 256, true), ("chunk64_flat", 64, true)] {
            let _ = std::fs::remove_file(&wal);
            assert!(rounds * 2 < StoreOptions::default().flush_docs, "the appends must stay below a flush");
            let store = Store::open(&db, e.clone(), StoreOptions { sync: true, mem_chunk_docs: chunk, mem_flat: flat, ..Default::default() }).unwrap();
            store.warm();
            let filter = Filter { exclude_window: Some(("bench-live".into(), i64::MAX)), ..Default::default() };
            let rank = RankParams::default();
            let (mut app, mut first, mut second, mut rebuild) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for i in 0..rounds {
                let drafts = vec![
                    engram::record::Draft::new(Kind::ToolCall, format!("$ cargo test --package p{i} -- --nocapture")).ctx("Turn 3 request: fix the parser"),
                    engram::record::Draft::new(Kind::ToolResult, format!("running 12 tests\ntest parser::tests::case_{i} ... ok\ntest result: ok. {i} passed; 0 failed")).ctx("Turn 3 request: fix the parser"),
                ];
                let t = Instant::now();
                store.append("bench-live", i as i64, 0, drafts).unwrap();
                app.push(ms_since(t));
                let q = &qs[i % qs.len()].0.question;
                if i % 2 == 0 {
                    // Back to back: the query pays for the rebuild.
                    let t = Instant::now();
                    let (hits, _) = recall::search(&store, q, &filter, &rank).unwrap();
                    let _ = recall::pack(&hits, q, &HashSet::new(), &PackParams::default());
                    first.push(ms_since(t));
                } else {
                    // With a pause: the rebuild alone, then a query.
                    let t = Instant::now();
                    store.warm();
                    rebuild.push(ms_since(t));
                    let t = Instant::now();
                    let (hits, _) = recall::search(&store, q, &filter, &rank).unwrap();
                    let _ = recall::pack(&hits, q, &HashSet::new(), &PackParams::default());
                    second.push(ms_since(t));
                }
            }
            drop(store);
            let _ = std::fs::remove_file(&wal);
            eprintln!(
                "  {name}: append {:.2}, rebuild {:.2}, query after write {:.2} (p99 {:.2}), query after rebuild {:.2}",
                pct(&mut app.clone(), 0.5),
                pct(&mut rebuild.clone(), 0.5),
                pct(&mut first.clone(), 0.5),
                pct(&mut first.clone(), 0.99),
                pct(&mut second.clone(), 0.5)
            );
            stall.insert(name.into(), json!({"chunk": chunk, "flat": flat, "append": stats(&app), "rebuild": stats(&rebuild), "query_after_write": stats(&first), "query_after_rebuild": stats(&second)}));
        }
        row.insert("stall".into(), Json::Object(stall));

        // 3. Hooks: a process per event (command hook) against a request
        //    from a client that is already running (what an HTTP hook pays,
        //    less its TCP and HTTP framing).
        {
            let exe = std::env::current_exe().unwrap().with_file_name("engram");
            // The first hook starts the daemon (through the engram binary, as
            // in Claude Code).
            let first = json!({"session_id": "bench-hook", "hook_event_name": "Notification", "cwd": project});
            row.insert("daemon_start_ms".into(), json!(run_hook(&exe, &home, &project, "Notification", &first)));
            let mut paths: HashMap<&str, Vec<f64>> = HashMap::new();
            for i in 0..n_hooks {
                let q = &qs[i % qs.len()].0.question;
                let prompt = json!({"session_id": "bench-hook", "hook_event_name": "UserPromptSubmit", "prompt": q, "cwd": project});
                let tool = json!({"session_id": "bench-hook", "hook_event_name": "PostToolUse", "tool_name": "Bash", "cwd": project,
                    "tool_input": {"command": format!("cargo test case_{i}")}, "tool_response": {"stdout": format!("test result: ok. {i} passed"), "stderr": ""}});
                let nothing = json!({"session_id": "bench-hook", "hook_event_name": "Notification", "cwd": project});
                std::thread::sleep(std::time::Duration::from_millis(150));
                paths.entry("process_noop").or_default().push(run_hook(&exe, &home, &project, "Notification", &nothing));
                std::thread::sleep(std::time::Duration::from_millis(150));
                paths.entry("process_prompt").or_default().push(run_hook(&exe, &home, &project, "UserPromptSubmit", &prompt));
                std::thread::sleep(std::time::Duration::from_millis(150));
                paths.entry("process_tool").or_default().push(run_hook(&exe, &home, &project, "PostToolUse", &tool));
                std::thread::sleep(std::time::Duration::from_millis(150));
                let t = Instant::now();
                let _ = engram::daemon::request(&dir, &project.to_string_lossy(), &json!({"op": "hook", "event": "Notification", "payload": nothing}), std::time::Duration::from_secs(5));
                paths.entry("client_noop").or_default().push(ms_since(t));
                std::thread::sleep(std::time::Duration::from_millis(150));
                let t = Instant::now();
                let _ = engram::daemon::request(&dir, &project.to_string_lossy(), &json!({"op": "hook", "event": "UserPromptSubmit", "payload": prompt}), std::time::Duration::from_secs(5));
                paths.entry("client_prompt").or_default().push(ms_since(t));
                std::thread::sleep(std::time::Duration::from_millis(150));
                let t = Instant::now();
                let _ = engram::daemon::request(&dir, &project.to_string_lossy(), &json!({"op": "hook", "event": "PostToolUse", "payload": tool}), std::time::Duration::from_secs(5));
                paths.entry("client_tool").or_default().push(ms_since(t));
            }
            let _ = engram::daemon::request(&dir, &project.to_string_lossy(), &json!({"op": "shutdown"}), std::time::Duration::from_secs(5));
            std::thread::sleep(std::time::Duration::from_millis(300));
            let m: serde_json::Map<String, Json> = paths.iter().map(|(k, v)| (k.to_string(), stats(v))).collect();
            eprintln!("  hooks: {}", m.iter().map(|(k, v)| format!("{k} {:.2}", v["p50"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(", "));
            row.insert("hooks".into(), Json::Object(m));
        }
        let _ = std::fs::remove_dir_all(&home);
        results.push(Json::Object(row));
        std::fs::write(out.join(format!("latency_{}.json", embedder.split(':').next().unwrap_or("x"))), serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
}

/// The prompt hook's searches for one prompt: knowledge and hybrid search
/// side by side when a core is free (as the daemon runs them), then packing.
fn prompt_searches(store: &Store, q: &str, filter: &Filter, rank: &RankParams, kp: &recall::KnowledgeParams) {
    let skip = HashSet::new();
    let (known, searched) = recall::both(|| recall::knowledge(store, q, filter, &skip, kp), || recall::search(store, q, filter, rank));
    known.unwrap();
    let (hits, _) = searched.unwrap();
    let _ = recall::pack(&hits, q, &skip, &PackParams::default());
}

/// The prompt's searches as the hook runs them with the session tail on:
/// the session's last records read, then [`prompt_searches`] with them.
/// Returns the time the read took.
fn prompt_searches_tail(store: &Store, q: &str, session: &str, filter: &Filter, rank: &RankParams, kp: &recall::KnowledgeParams) -> f64 {
    let t = Instant::now();
    let tail = recall::Tail::of(&store.tail(session, rank.tail.records).unwrap(), &rank.tail);
    let read = ms_since(t);
    let skip = HashSet::new();
    let (known, searched) = recall::both(|| recall::knowledge(store, q, filter, &skip, kp), || recall::search_with(store, q, Some(&tail), filter, rank));
    known.unwrap();
    let (hits, _) = searched.unwrap();
    let _ = recall::pack(&hits, q, &skip, &PackParams::default());
    read
}

fn tail_cmd(data: &Path, sizes: &[usize], embedder: &str, out: &Path, n_queries: usize, rounds: usize, knowledge: f64) {
    let e = open_embedder(embedder);
    let units = engram::benchdata::read_units(data, 1, None).expect("data");
    std::fs::create_dir_all(out).unwrap();
    let mut results = Vec::new();
    for &n in sizes {
        eprintln!("== {n} records");
        let (mut recs, qs, _) = haystack(&units, n, n_queries);
        if knowledge > 0.0 {
            let every = (1.0 / knowledge).round().max(1.0) as usize;
            let kinds = [Kind::Decision, Kind::Preference, Kind::Fact];
            for (i, r) in recs.iter_mut().enumerate().filter(|(i, _)| i % every == 0) {
                r.kind = kinds[(i / every) % 3];
                r.tag = "distilled".into();
            }
        }
        // The prompts' sessions: sessions of at least 20 records, spread
        // over the store, as a long session's tail would be.
        let mut by: HashMap<&str, usize> = HashMap::new();
        for r in &recs {
            *by.entry(r.session.as_str()).or_default() += 1;
        }
        let mut long: Vec<String> = by.into_iter().filter(|(_, c)| *c >= 20).map(|(s, _)| s.to_string()).collect();
        long.sort();
        assert!(!long.is_empty(), "no session of 20 records");
        let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let chunks: Vec<&[&str]> = refs.chunks((refs.len() / 4).max(1)).collect();
            let hs: Vec<_> = chunks.into_iter().map(|c| { let e = e.clone(); s.spawn(move || e.embed(c).expect("embedding")) }).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        drop(texts);
        let dir = std::env::temp_dir().join(format!("engram-tail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("memory.strata");
        {
            let store = Store::open(&db, e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
            store.bulk_insert(&recs, embs).unwrap();
        }
        let n = recs.len();
        drop(recs);
        // The bulk store's merger can still hold the file for a while after
        // a large load; wait for it.
        let t0 = Instant::now();
        let store = loop {
            match Store::open(&db, e.clone(), StoreOptions::default()) {
                Ok(s) => break s,
                Err(err) if err.to_string().contains("busy") && t0.elapsed().as_secs() < 600 => std::thread::sleep(std::time::Duration::from_millis(200)),
                Err(err) => panic!("{err}"),
            }
        };
        store.warm();
        let off = RankParams::default();
        let mut on = RankParams::default();
        on.tail.enabled = true;
        let kp = recall::KnowledgeParams::default();
        let (mut t_off, mut t_on, mut t_read) = (Vec::new(), Vec::new(), Vec::new());
        // One warm-up pass, then alternate the two.
        for (i, (q, _)) in qs.iter().enumerate() {
            let session = &long[i * 7919 % long.len()];
            let filter = Filter { exclude_window: Some((session.clone(), i64::MAX)), ..Default::default() };
            prompt_searches(&store, &q.question, &filter, &off, &kp);
            prompt_searches_tail(&store, &q.question, session, &filter, &on, &kp);
        }
        for round in 0..rounds {
            for (i, (q, _)) in qs.iter().enumerate() {
                let session = &long[i * 7919 % long.len()];
                let filter = Filter { exclude_window: Some((session.clone(), i64::MAX)), ..Default::default() };
                let time_off = || {
                    let t = Instant::now();
                    prompt_searches(&store, &q.question, &filter, &off, &kp);
                    ms_since(t)
                };
                let time_on = || {
                    let t = Instant::now();
                    let read = prompt_searches_tail(&store, &q.question, session, &filter, &on, &kp);
                    (ms_since(t), read)
                };
                if (round + i) % 2 == 0 {
                    t_off.push(time_off());
                    let (a, r) = time_on();
                    t_on.push(a);
                    t_read.push(r);
                } else {
                    let (a, r) = time_on();
                    t_on.push(a);
                    t_read.push(r);
                    t_off.push(time_off());
                }
            }
        }
        let (so, sn, sr) = (stats(&t_off), stats(&t_on), stats(&t_read));
        eprintln!("  off: p50 {:.2} ms, mean {:.2}; on: p50 {:.2} ms, mean {:.2}; tail read p50 {:.2} ms", so["p50"].as_f64().unwrap(), so["mean"].as_f64().unwrap(), sn["p50"].as_f64().unwrap(), sn["mean"].as_f64().unwrap(), sr["p50"].as_f64().unwrap());
        results.push(json!({"records": n, "sessions_used": long.len().min(qs.len()), "tail": on.tail, "off_ms": so, "on_ms": sn, "tail_read_ms": sr}));
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }
    let machine = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
        .unwrap_or_default();
    let report = json!({"machine": {"cpu": machine, "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)}, "embedder": embedder, "queries": n_queries, "rounds": rounds, "results": results});
    std::fs::write(out.join(format!("tail_{}.json", embedder.split(':').next().unwrap_or("x"))), serde_json::to_string_pretty(&report).unwrap()).unwrap();
}

/// A tool call and its result, as a tool hook stores them.
fn tool_drafts(i: i64) -> Vec<engram::record::Draft> {
    vec![
        engram::record::Draft::new(Kind::ToolCall, format!("$ cargo test --package p{i} -- --nocapture")).ctx("Turn 3 request: fix the parser"),
        engram::record::Draft::new(Kind::ToolResult, format!("running 12 tests\ntest parser::tests::case_{i} ... ok\ntest result: ok. {i} passed; 0 failed")).ctx("Turn 3 request: fix the parser"),
    ]
}

fn writes_cmd(embedder: &str, records: usize, clients: &[usize], out: &Path) {
    let e = open_embedder(embedder);
    std::fs::create_dir_all(out).unwrap();
    let mut rows = Vec::new();
    for &k in clients {
        let dir = std::env::temp_dir().join(format!("engram-writes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open(&dir.join("memory.strata"), e.clone(), StoreOptions::default()).unwrap();
        let (appends, next) = (records / 2, AtomicUsize::new(0));
        let t0 = Instant::now();
        let mut lat: Vec<f64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..k.max(1))
                .map(|_| {
                    s.spawn(|| {
                        let mut lat = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= appends {
                                break lat;
                            }
                            let t = Instant::now();
                            store.append("bench-writes", i as i64, i as i64, tool_drafts(i as i64)).unwrap();
                            lat.push(ms_since(t));
                        }
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let secs = t0.elapsed().as_secs_f64();
        let segments = store.snapshot().segments().len();
        // Closing waits for a merge under way.
        drop(store);
        let closed = t0.elapsed().as_secs_f64() - secs;
        let over = |ms: f64| lat.iter().filter(|&&l| l >= ms).count();
        let (over_100, over_1000) = (over(100.0), over(1000.0));
        let row = json!({
            "clients": k, "records": appends * 2, "secs": secs, "appends_per_sec": appends as f64 / secs,
            "p50_ms": pct(&mut lat, 0.5), "p99_ms": pct(&mut lat, 0.99), "p999_ms": pct(&mut lat, 0.999), "max_ms": pct(&mut lat, 1.0),
            "over_100ms": over_100, "over_1s": over_1000, "segments": segments, "close_secs": closed,
        });
        eprintln!("{row}");
        rows.push(row);
        let _ = std::fs::remove_dir_all(&dir);
    }
    std::fs::write(out.join("writes.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn concurrency_cmd(data: &Path, sizes: &[usize], embedder: &str, out: &Path, n_queries: usize, readers: &[usize], secs: u64, every_ms: u64, knowledge: f64) {
    use std::sync::atomic::AtomicBool;
    let e = open_embedder(embedder);
    let units = engram::benchdata::read_units(data, 1, None).expect("data");
    let every = std::time::Duration::from_millis(every_ms);
    let run_for = std::time::Duration::from_secs(secs);
    let mut results = Vec::new();
    std::fs::create_dir_all(out).unwrap();
    for &n in sizes {
        eprintln!("== {n} records");
        let (mut recs, qs, _) = haystack(&units, n, n_queries);
        if knowledge > 0.0 {
            let every = (1.0 / knowledge).round().max(1.0) as usize;
            let kinds = [Kind::Decision, Kind::Preference, Kind::Fact];
            for (i, r) in recs.iter_mut().enumerate().filter(|(i, _)| i % every == 0) {
                r.kind = kinds[(i / every) % 3];
                r.tag = "distilled".into();
            }
        }
        let texts: Vec<String> = recs.iter().map(|r| r.embed_text()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let chunks: Vec<&[&str]> = refs.chunks((refs.len() / 4).max(1)).collect();
            let hs: Vec<_> = chunks.into_iter().map(|c| { let e = e.clone(); s.spawn(move || e.embed(c).expect("embedding")) }).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        drop(texts);
        let home = std::env::temp_dir().join(format!("engram-concurrency-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let project = home.join("proj");
        std::fs::create_dir_all(&project).unwrap();
        std::env::set_var("ENGRAM_HOME", &home);
        std::env::set_var("ENGRAM_DEEP", "off");
        std::env::set_var("ENGRAM_DISTILL", "0");
        let dir = engram::paths::store_dir(&project);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_string(&json!({"embedder": embedder})).unwrap()).unwrap();
        let db = dir.join("memory.strata");
        {
            let store = Store::open(&db, e.clone(), StoreOptions { sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
            store.bulk_insert(&recs, embs).unwrap();
        }
        let n = recs.len();
        drop(recs);
        let wal = { let mut w = db.as_os_str().to_owned(); w.push("-wal"); PathBuf::from(w) };
        let _ = std::fs::remove_file(&wal);
        let mut row = serde_json::Map::new();
        row.insert("records".into(), json!(n));
        row.insert("secs".into(), json!(secs));
        row.insert("write_every_ms".into(), json!(every_ms));

        // 1. In this process: reader threads on one store, with and without
        //    a writer. Each configuration starts from the bulk-loaded file
        //    (its appends stay in the memtable and the WAL, deleted after).
        let mut runs = Vec::new();
        for &k in readers {
            for writer in [false, true] {
                let _ = std::fs::remove_file(&wal);
                let store = Store::open(&db, e.clone(), StoreOptions::default()).unwrap();
                store.warm();
                let filter = Filter { exclude_window: Some(("bench-live".into(), i64::MAX)), ..Default::default() };
                let (rank, kp) = (RankParams::default(), recall::KnowledgeParams::default());
                let stop = AtomicBool::new(false);
                let (store, qs, filter, rank, kp, stop) = (&store, &qs, &filter, &rank, &kp, &stop);
                let (lat, app) = std::thread::scope(|sc| {
                    let w = writer.then(|| {
                        sc.spawn(move || {
                            let mut app = Vec::new();
                            let mut i = 0i64;
                            while !stop.load(Ordering::Acquire) {
                                let t = Instant::now();
                                store.append("bench-live", i, 0, tool_drafts(i)).unwrap();
                                app.push(ms_since(t));
                                i += 1;
                                std::thread::sleep(every);
                            }
                            app
                        })
                    });
                    let rs: Vec<_> = (0..k)
                        .map(|r| {
                            sc.spawn(move || {
                                let mut lat = Vec::new();
                                let mut i = r * 31;
                                while !stop.load(Ordering::Acquire) {
                                    let q = &qs[i % qs.len()].0.question;
                                    let t = Instant::now();
                                    prompt_searches(store, q, filter, rank, kp);
                                    lat.push(ms_since(t));
                                    i += 1;
                                }
                                lat
                            })
                        })
                        .collect();
                    std::thread::sleep(run_for);
                    stop.store(true, Ordering::Release);
                    let lat: Vec<f64> = rs.into_iter().flat_map(|h| h.join().unwrap()).collect();
                    (lat, w.map(|h| h.join().unwrap()).unwrap_or_default())
                });
                let _ = std::fs::remove_file(&wal);
                eprintln!(
                    "  store: {k} readers{}: read p50 {:.2} p99 {:.2} ms, {:.0} reads/s; {} appends, p50 {:.2} p99 {:.2} ms",
                    if writer { " + writer" } else { "" },
                    pct(&mut lat.clone(), 0.5),
                    pct(&mut lat.clone(), 0.99),
                    lat.len() as f64 / secs as f64,
                    app.len(),
                    pct(&mut app.clone(), 0.5),
                    pct(&mut app.clone(), 0.99)
                );
                runs.push(json!({"readers": k, "writer": writer, "read": stats(&lat), "reads_per_s": lat.len() as f64 / secs as f64, "append": stats(&app), "appends": app.len()}));
            }
        }
        row.insert("store".into(), Json::Array(runs));

        // 2. Through a daemon: prompt clients back to back, with and without
        //    a client sending tool hooks. Each configuration gets a fresh
        //    daemon on a copy of the bulk-loaded file (stopped with SIGKILL,
        //    so it does not flush its memtable into the file).
        let exe = std::env::current_exe().unwrap().with_file_name("engram");
        let backup = dir.join("memory.strata.bak");
        std::fs::copy(&db, &backup).unwrap();
        let proj = project.to_string_lossy().to_string();
        let mut runs = Vec::new();
        for &k in readers {
            for writer in [false, true] {
                for f in [&wal, &dir.join("state.json")] {
                    let _ = std::fs::remove_file(f);
                }
                std::fs::copy(&backup, &db).unwrap();
                let first = json!({"session_id": "bench-start", "hook_event_name": "Notification", "cwd": project});
                run_hook(&exe, &home, &project, "Notification", &first);
                let pid = engram::daemon::request(&dir, &proj, &json!({"op": "ping"}), std::time::Duration::from_secs(30)).unwrap()["pid"].as_i64().unwrap();
                let stop = AtomicBool::new(false);
                let (dir, proj, qs, stop) = (&dir, &proj, &qs, &stop);
                let (lat, tools) = std::thread::scope(|sc| {
                    let w = writer.then(|| {
                        sc.spawn(move || {
                            let mut lat = Vec::new();
                            let mut i = 0;
                            while !stop.load(Ordering::Acquire) {
                                let tool = json!({"session_id": "bench-tool", "hook_event_name": "PostToolUse", "tool_name": "Bash",
                                    "tool_input": {"command": format!("cargo test case_{i}")}, "tool_response": {"stdout": format!("test result: ok. {i} passed"), "stderr": ""}});
                                let t = Instant::now();
                                let _ = engram::daemon::request(dir, proj, &json!({"op": "hook", "event": "PostToolUse", "payload": tool}), std::time::Duration::from_secs(10));
                                lat.push(ms_since(t));
                                i += 1;
                                std::thread::sleep(every);
                            }
                            lat
                        })
                    });
                    let rs: Vec<_> = (0..k)
                        .map(|r| {
                            sc.spawn(move || {
                                let mut lat = Vec::new();
                                let mut i = r * 31;
                                while !stop.load(Ordering::Acquire) {
                                    let q = &qs[i % qs.len()].0.question;
                                    let prompt = json!({"session_id": format!("bench-c{r}"), "hook_event_name": "UserPromptSubmit", "prompt": q});
                                    let t = Instant::now();
                                    let resp = engram::daemon::request(dir, proj, &json!({"op": "hook", "event": "UserPromptSubmit", "payload": prompt}), std::time::Duration::from_secs(10));
                                    lat.push(ms_since(t));
                                    assert!(resp.is_ok_and(|r| r.get("error").is_none()));
                                    i += 1;
                                }
                                lat
                            })
                        })
                        .collect();
                    std::thread::sleep(run_for);
                    stop.store(true, Ordering::Release);
                    let lat: Vec<f64> = rs.into_iter().flat_map(|h| h.join().unwrap()).collect();
                    (lat, w.map(|h| h.join().unwrap()).unwrap_or_default())
                });
                let stats_resp = engram::daemon::request(dir, proj, &json!({"op": "stats"}), std::time::Duration::from_secs(30)).unwrap_or(Json::Null);
                // SAFETY: kill has no memory-safety preconditions.
                unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                std::thread::sleep(std::time::Duration::from_millis(300));
                eprintln!(
                    "  daemon: {k} prompt clients{}: prompt p50 {:.2} p99 {:.2} ms, {:.0} prompts/s; {} tool hooks, p50 {:.2} p99 {:.2} ms; {} records",
                    if writer { " + tool hooks" } else { "" },
                    pct(&mut lat.clone(), 0.5),
                    pct(&mut lat.clone(), 0.99),
                    lat.len() as f64 / secs as f64,
                    tools.len(),
                    pct(&mut tools.clone(), 0.5),
                    pct(&mut tools.clone(), 0.99),
                    stats_resp["records"]
                );
                runs.push(json!({"clients": k, "tool_hooks": writer, "prompt": stats(&lat), "prompts_per_s": lat.len() as f64 / secs as f64,
                    "tool": stats(&tools), "tool_hooks_sent": tools.len(), "records_after": stats_resp["records"]}));
            }
        }
        row.insert("daemon".into(), Json::Array(runs));
        let _ = std::fs::remove_dir_all(&home);
        results.push(Json::Object(row));
        std::fs::write(out.join(format!("concurrency_{}.json", embedder.split(':').next().unwrap_or("x"))), serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
}
