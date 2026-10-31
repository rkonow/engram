//! Hybrid retrieval with priors, the injection gate, snippets and the
//! text Engram injects.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::anchor::Checker;
use crate::capture::past_reference;
use crate::record::{Kind, Record};
use crate::store::{Filter, Hit, Reach, Store};
use crate::Result;

/// Ranking parameters (see DESIGN.md §5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankParams {
    /// Candidates from each of BM25 and vector search.
    pub n: usize,
    /// Weight of normalised BM25 against normalised cosine.
    pub w_bm25: f64,
    /// Importance prior: score × (1 + λ_ι ι).
    pub lambda_imp: f64,
    /// Recency prior weight λ_t and scale τ (in records).
    pub lambda_rec: f64,
    pub tau: f64,
    /// Use BM25 / vectors at all.
    pub use_bm25: bool,
    pub use_vectors: bool,
    /// Records linked to the best hits (see [`GraphParams`]).
    #[serde(default)]
    pub graph: GraphParams,
    /// The session's last records as context (see [`TailParams`]).
    #[serde(default)]
    pub tail: TailParams,
}

impl Default for RankParams {
    fn default() -> Self {
        RankParams { n: 50, w_bm25: 0.5, lambda_imp: 0.2, lambda_rec: 0.05, tau: 5000.0, use_bm25: true, use_vectors: true, graph: GraphParams::default(), tail: TailParams::default() }
    }
}

/// Graph recall: the best hits, and the entities nearest the query, start a
/// walk over the records' links that passes through entities and notes; the
/// records it reaches, best by similarity to the query, are a second ranked
/// list fused into the hybrid ranking by reciprocal rank,
/// score = 1/(k + rank_hybrid) + weight/(k + rank_graph).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphParams {
    pub enabled: bool,
    /// Top hits whose links start the walk.
    pub seeds: usize,
    /// Entities nearest the query that start it too, at this cosine or more.
    pub entity_seeds: usize,
    pub min_entity_cos: f64,
    pub hops: usize,
    /// Records the walk returns.
    pub n: usize,
    pub k: f64,
    pub weight: f64,
    pub max_nodes: usize,
}

impl Default for GraphParams {
    /// Off unless `ENGRAM_GRAPH=1` (or the config turns it on) until the
    /// evaluations in eval/graph_eval_plan.md show it helps.
    fn default() -> Self {
        GraphParams { enabled: std::env::var("ENGRAM_GRAPH").is_ok_and(|v| v == "1"), seeds: 5, entity_seeds: 3, min_entity_cos: 0.5, hops: 2, n: 20, k: 10.0, weight: 0.5, max_nodes: 5000 }
    }
}

/// The session's last records as context for a prompt's recall: what the
/// work in progress touches, which the prompt may not say ("add the
/// endpoint" while editing the billing module). Their files and failures
/// join the prompt on the BM25 side; their vectors rank records by `maxsim`
/// (the best cosine to any of them), a third list fused beside BM25 and
/// cosine; and records about their files get a boost. The gate's evidence
/// stays the prompt's own. Off unless `ENGRAM_SESSION_CONTEXT=1` (or the
/// config turns it on) until the A/B harness shows it helps.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TailParams {
    pub enabled: bool,
    /// The session's newest records read.
    pub records: usize,
    /// Of those, the newest with vectors (prompts, answers, tool calls and
    /// errors; not tool output) that rank by `maxsim`.
    pub vectors: usize,
    /// Candidates of each of its lists (BM25 over the digest, `maxsim`).
    pub n: usize,
    /// Weight of a record's tail score (the better of its normalised scores
    /// in the two lists), beside BM25's and cosine's (which sum to 1).
    pub weight: f64,
    /// Score × (1 + boost) for a record about one of the tail's files.
    pub file_boost: f64,
    /// Words of the digest searched with BM25, at most.
    pub digest_words: usize,
}

impl Default for TailParams {
    fn default() -> Self {
        TailParams { enabled: std::env::var("ENGRAM_SESSION_CONTEXT").is_ok_and(|v| v == "1"), records: 20, vectors: 4, n: 20, weight: 0.5, file_boost: 0.15, digest_words: 24 }
    }
}

/// What a session's last records say about the work in progress.
#[derive(Clone, Debug, Default)]
pub struct Tail {
    /// The files they name.
    pub files: HashSet<String>,
    /// Their files and failures, as words for BM25.
    pub digest: String,
    /// The newest records' vectors, for `maxsim`.
    pub vectors: Vec<Vec<f32>>,
}

impl Tail {
    /// The tail of `records`, newest first (as [`Store::tail`] returns them).
    pub fn of(records: &[(Record, Option<Vec<f32>>)], p: &TailParams) -> Tail {
        let mut files = HashSet::new();
        let mut words: Vec<String> = Vec::new();
        let add = |w: &str, words: &mut Vec<String>| {
            if !w.is_empty() && words.len() < p.digest_words && !words.iter().any(|x| x == w) {
                words.push(w.to_string());
            }
        };
        // A file by its name and its directory's: "billing money" for
        // src/billing/money.rs, not the "src" and "rs" every path has.
        for (r, _) in records {
            for f in r.files.split_whitespace() {
                if files.insert(f.to_string()) {
                    let path = std::path::Path::new(f);
                    let dir = path.parent().and_then(|d| d.file_name()).and_then(|d| d.to_str()).filter(|d| !matches!(*d, "src" | "lib" | "test" | "tests" | "." | ".."));
                    for w in [dir, path.file_stem().and_then(|s| s.to_str())].into_iter().flatten() {
                        add(w, &mut words);
                    }
                }
            }
        }
        // A failure's first line names what failed: a test, a command.
        for (r, _) in records.iter().filter(|(r, _)| r.kind == Kind::Error) {
            for w in r.text.lines().next().unwrap_or("").split_whitespace().take(8) {
                add(w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'), &mut words);
            }
        }
        let vectors = records
            .iter()
            .filter(|(r, _)| matches!(r.kind, Kind::Prompt | Kind::Answer | Kind::ToolCall | Kind::Error))
            .filter_map(|(_, v)| v.clone())
            .take(p.vectors)
            .collect();
        Tail { files, digest: words.join(" "), vectors }
    }

    /// Whether `r` is about one of the tail's files: it names one, or
    /// links to it.
    fn touches(&self, r: &Record) -> bool {
        !self.files.is_empty()
            && (r.files.split_whitespace().any(|f| self.files.contains(f)) || r.links.iter().any(|l| l.to.strip_prefix("f:").is_some_and(|f| self.files.contains(f))))
    }
}

/// Kinds a graph walk passes through: entities and the notes about them.
pub const THROUGH: [Kind; 6] = [Kind::Entity, Kind::Preference, Kind::Decision, Kind::Fact, Kind::Todo, Kind::Note];

/// The records linked to the best hits and to the entities nearest the
/// query, best first by cosine similarity to `qv`.
pub fn expand(store: &Store, qv: &[f32], hits: &[Hit], filter: &Filter, p: &GraphParams) -> Result<Vec<(Record, f64)>> {
    let mut seeds: Vec<String> = hits.iter().take(p.seeds).map(|h| h.record.id.clone()).collect();
    if p.entity_seeds > 0 {
        let f = Filter { kinds: vec![Kind::Entity], ..Default::default() };
        for (r, c) in store.vector_top(qv, p.entity_seeds, &f)? {
            if c >= p.min_entity_cos {
                seeds.push(r.id);
            }
        }
    }
    let reach = Reach { hops: p.hops, through: THROUGH.to_vec(), max_nodes: p.max_nodes };
    store.reach(&seeds, qv, p.n, filter, &reach)
}

/// Fuses the graph list into the hybrid ranking (see [`GraphParams`]).
/// Graph records keep their cosine; their score is the fused one.
pub fn merge_graph(hits: Vec<Hit>, graph: Vec<(Record, f64)>, p: &GraphParams) -> Vec<Hit> {
    if graph.is_empty() {
        return hits;
    }
    let mut by: HashMap<String, (Hit, f64)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (i, h) in hits.into_iter().enumerate() {
        order.push(h.record.id.clone());
        by.insert(h.record.id.clone(), (h, 1.0 / (p.k + 1.0 + i as f64)));
    }
    for (j, (r, c)) in graph.into_iter().enumerate() {
        let add = p.weight / (p.k + 1.0 + j as f64);
        match by.get_mut(&r.id) {
            Some(e) => e.1 += add,
            None => {
                order.push(r.id.clone());
                by.insert(r.id.clone(), (Hit { record: r, bm25: None, cos: Some(c), score: 0.0 }, add));
            }
        }
    }
    let mut out: Vec<(Hit, f64)> = order.into_iter().filter_map(|id| by.remove(&id)).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out.into_iter()
        .map(|(mut h, s)| {
            h.score = s;
            h
        })
        .collect()
}

/// Absolute evidence about the best candidates, for the gate.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evidence {
    /// Highest cosine similarity among vector candidates.
    pub max_cos: f64,
    /// Highest IDF-weighted query-term coverage among the top 5.
    pub max_cov: f64,
    /// The top candidate is in both the BM25 and the vector top 10.
    pub agree: bool,
    /// The prompt refers back to earlier work.
    pub cue: bool,
    /// How far the best vector candidate stands out: cos₁ − cos₁₀.
    #[serde(default)]
    pub cos_gap: f64,
    /// How far the best BM25 candidate stands out: 1 − bm25₁₀ / bm25₁.
    #[serde(default)]
    pub bm_gap: f64,
}

/// The injection gate, a linear score over the evidence:
/// G = a·max_cos + b·max_cov + c·cue + d·agree + e·cos_gap + f·bm_gap.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Gate {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
    /// Full injection at G ≥ hi; a brief one (two records) at lo ≤ G < hi.
    pub hi: f64,
    pub lo: f64,
}

impl Default for Gate {
    fn default() -> Self {
        // Logistic fit on LongMemEval-S with the static embedder and
        // 400-byte records (scripts/fit_gate.py, results/gate_static_c400.json):
        // hi holds precision 0.9, lo precision 0.7.
        Gate { a: 4.868, b: 4.794, c: 0.595, d: 0.981, e: 1.959, f: -1.448, hi: 6.650, lo: 4.508 }
    }
}

impl Gate {
    /// The gate fitted for an embedder spec (cosine scales differ).
    pub fn for_embedder(spec: &str) -> Gate {
        // BGE-small vectors (in the daemon, or from a benchmark cache).
        if spec.starts_with("bert") || spec.contains("bge") {
            Gate::bert()
        } else {
            Gate::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Decision {
    /// Strong evidence: the full budget.
    Inject,
    /// Weak evidence: the top two records in a small budget.
    Brief,
    Skip,
}

impl Gate {
    /// Logistic fit on every fifth LongMemEval-S haystack with BGE-small
    /// vectors (scripts/fit_gate.py); hi holds precision 0.9, lo 0.7.
    pub fn bert() -> Gate {
        Gate { a: 3.803, b: 4.311, c: 0.326, d: 1.583, e: 2.778, f: 0.617, hi: 9.711, lo: 6.374 }
    }

    pub fn score(&self, e: &Evidence) -> f64 {
        self.a * e.max_cos + self.b * e.max_cov + self.c * e.cue as u8 as f64 + self.d * e.agree as u8 as f64 + self.e * e.cos_gap + self.f * e.bm_gap
    }
    pub fn decide(&self, e: &Evidence) -> Decision {
        let g = self.score(e);
        if g >= self.hi {
            Decision::Inject
        } else if g >= self.lo {
            Decision::Brief
        } else {
            Decision::Skip
        }
    }
}

fn analyzed(text: &str) -> HashSet<String> {
    strata::text::analyze(text).into_iter().collect()
}

/// Requests running [`both`] now.
static REQUESTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The threads this thread's part of a request may use; 0 outside one.
    static SHARE: Cell<usize> = const { Cell::new(0) };
}

fn cores() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// A request's share of the cores: divided among the requests running when
/// it starts.
struct Request;

impl Request {
    fn start() -> Request {
        let n = REQUESTS.fetch_add(1, Ordering::AcqRel) + 1;
        SHARE.set((cores() / n).max(1));
        Request
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        SHARE.set(0);
        REQUESTS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Runs `a` and `b` side by side (`a` on another thread) when the request's
/// share of the cores allows, else one after the other. A lone request
/// splits across every core; with every core busy, splitting a request
/// would only slow the others down.
pub fn both<A: Send, B>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B) -> (A, B) {
    let _request = (SHARE.get() == 0).then(Request::start);
    let share = SHARE.get();
    if share < 2 {
        return (a(), b());
    }
    let theirs = share / 2;
    SHARE.set(share - theirs);
    let out = std::thread::scope(|s| {
        let h = s.spawn(move || {
            SHARE.set(theirs);
            a()
        });
        let b = b();
        (h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)), b)
    });
    SHARE.set(share);
    out
}

/// Hybrid search: BM25 and vector candidates, min-max fused, with priors.
/// The two searches run on one snapshot, side by side when a core is free.
pub fn search(store: &Store, query: &str, filter: &Filter, p: &RankParams) -> Result<(Vec<Hit>, Evidence)> {
    search_with(store, query, None, filter, p)
}

/// [`search`] with the session's last records as context ([`TailParams`]),
/// when `p.tail` is on.
pub fn search_with(store: &Store, query: &str, tail: Option<&Tail>, filter: &Filter, p: &RankParams) -> Result<(Vec<Hit>, Evidence)> {
    let store = &store.pinned();
    let terms = strata::text::analyze_query(query);
    let tail = tail.filter(|_| p.tail.enabled);
    let ((bm, td), (vq, dv, tv)) = both(
        || {
            let bm = if p.use_bm25 && !terms.is_empty() { store.bm25_top(query, p.n, filter) } else { Ok(Vec::new()) };
            let td = match tail {
                Some(t) if p.use_bm25 && !strata::text::analyze_query(&t.digest).is_empty() => store.bm25_top(&t.digest, p.tail.n, filter),
                _ => Ok(Vec::new()),
            };
            (bm, td)
        },
        || {
            let vq = p.use_vectors.then(|| store.embedder.embed_query(query)).transpose();
            let dv = match &vq {
                Ok(Some(v)) => store.vector_top(v, p.n, filter),
                _ => Ok(Vec::new()),
            };
            let tv = match tail {
                Some(t) if p.use_vectors => store.maxsim_top(&t.vectors, p.tail.n, filter),
                _ => Ok(Vec::new()),
            };
            (vq, dv, tv)
        },
    );
    let (mut bm, mut dv, mut td, mut tv, vq) = (bm?, dv?, td?, tv?, vq?);
    // Earlier copies of the same question and answers that admitted not
    // knowing match a repeated question best and carry nothing: they would
    // inflate the gate and repeat the failure.
    let qterms: HashSet<String> = terms.iter().cloned().collect();
    let useful = |r: &Record| r.tag != "uncertain" && r.tag != "superseded" && !same_question(r, &qterms);
    bm.retain(|(r, _)| useful(r));
    dv.retain(|(r, _)| useful(r));
    td.retain(|(r, _)| useful(r));
    tv.retain(|(r, _)| useful(r));
    let (hits, ev) = fuse_with(store, query, &terms, bm, dv, [td, tv], tail, p);
    if !p.graph.enabled || hits.is_empty() {
        return Ok((hits, ev));
    }
    let qv = match vq {
        Some(v) => v,
        None => store.embedder.embed_query(query)?,
    };
    let mut graph = expand(store, &qv, &hits, filter, &p.graph)?;
    graph.retain(|(r, _)| useful(r));
    Ok((merge_graph(hits, graph, &p.graph), ev))
}

/// Fuses BM25 and vector candidate lists (exposed for benchmarks that run
/// the two searches themselves).
pub fn fuse(store: &Store, query: &str, terms: &[String], bm: Vec<(Record, f64)>, dv: Vec<(Record, f64)>, p: &RankParams) -> (Vec<Hit>, Evidence) {
    fuse_with(store, query, terms, bm, dv, [Vec::new(), Vec::new()], None, p)
}

/// [`fuse`] with the session tail's two lists (BM25 over its digest and
/// `maxsim` over its vectors) and its files. A record's tail score is the
/// better of its two normalised scores. The evidence is the prompt's own:
/// read from the order without the tail, so the tail reorders what the gate
/// lets through but does not open or close it.
#[allow(clippy::too_many_arguments)]
fn fuse_with(store: &Store, query: &str, terms: &[String], bm: Vec<(Record, f64)>, dv: Vec<(Record, f64)>, tails: [Vec<(Record, f64)>; 2], tail: Option<&Tail>, p: &RankParams) -> (Vec<Hit>, Evidence) {
    let top_ids = |l: &[(Record, f64)]| l.iter().take(10).map(|(r, _)| r.id.clone()).collect::<HashSet<_>>();
    let (bm_top, dv_top) = (top_ids(&bm), top_ids(&dv));
    let max_cos = dv.iter().map(|x| x.1).fold(0.0, f64::max);
    let nth = |l: &[(Record, f64)], i: usize| l.get(i.min(l.len().saturating_sub(1))).map_or(0.0, |x| x.1);
    let cos_gap = if dv.is_empty() { 0.0 } else { nth(&dv, 0) - nth(&dv, 9) };
    let bm_gap = if bm.is_empty() || nth(&bm, 0) <= 0.0 { 0.0 } else { 1.0 - nth(&bm, 9) / nth(&bm, 0) };
    let range = |l: &[(Record, f64)]| {
        let lo = l.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        let hi = l.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let (blo, bhi) = range(&bm);
    let (vlo, vhi) = range(&dv);
    let norm = |x: f64, lo: f64, hi: f64| if hi > lo { (x - lo) / (hi - lo) } else { 1.0 };
    let mut by: HashMap<String, Hit> = HashMap::new();
    for (r, s) in bm {
        by.insert(r.id.clone(), Hit { record: r, bm25: Some(s), cos: None, score: 0.0 });
    }
    for (r, s) in dv {
        by.entry(r.id.clone()).and_modify(|h| h.cos = Some(s)).or_insert(Hit { record: r, bm25: None, cos: Some(s), score: 0.0 });
    }
    let mut near: HashMap<String, f64> = HashMap::new();
    for l in tails {
        let (lo, hi) = range(&l);
        for (r, s) in l {
            let t = near.entry(r.id.clone()).or_insert(0.0);
            *t = t.max(norm(s, lo, hi));
            by.entry(r.id.clone()).or_insert(Hit { record: r, bm25: None, cos: None, score: 0.0 });
        }
    }
    let wt = if near.is_empty() { 0.0 } else { p.tail.weight };
    let max_seq = by.values().map(|h| h.record.seq).max().unwrap_or(0).max(store.last_seq());
    let (wb, wv) = match (bhi.is_finite(), vhi.is_finite()) {
        (true, true) => (p.w_bm25, 1.0 - p.w_bm25),
        (true, false) => (1.0, 0.0),
        _ => (0.0, 1.0),
    };
    // Each hit with its score without the tail, which orders the evidence.
    let mut hits: Vec<(Hit, f64)> = by
        .into_values()
        .map(|mut h| {
            let b = h.bm25.map_or(0.0, |x| norm(x, blo, bhi));
            let c = h.cos.map_or(0.0, |x| norm(x, vlo, vhi));
            let t = near.get(&h.record.id).copied().unwrap_or(0.0);
            let age = (max_seq - h.record.seq).max(0) as f64;
            let score = |rel: f64| rel * (1.0 + p.lambda_imp * h.record.importance) + p.lambda_rec * (-age / p.tau).exp();
            let plain = score(wb * b + wv * c);
            h.score = score(wb * b + wv * c + wt * t);
            if tail.is_some_and(|t| t.touches(&h.record)) {
                h.score *= 1.0 + p.tail.file_boost;
            }
            (h, plain)
        })
        .collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.record.seq.cmp(&a.0.record.seq)));
    let mut ev = Evidence { max_cos, cue: past_reference(query), cos_gap, bm_gap, ..Default::default() };
    if let Some((top, _)) = hits.first() {
        ev.agree = bm_top.contains(&top.record.id) && dv_top.contains(&top.record.id);
    }
    if !terms.is_empty() {
        let idf = store.idf(terms).unwrap_or_else(|_| vec![1.0; terms.len()]);
        let total: f64 = idf.iter().sum();
        for (h, _) in hits.iter().take(5) {
            let have = analyzed(&format!("{}\n{}", h.record.text, h.record.ctx));
            let got: f64 = terms.iter().zip(&idf).filter(|(t, _)| have.contains(*t)).map(|(_, w)| w).sum();
            ev.max_cov = ev.max_cov.max(if total > 0.0 { got / total } else { 0.0 });
        }
    }
    let mut hits: Vec<Hit> = hits.into_iter().map(|(h, _)| h).collect();
    if tail.is_some() {
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.record.seq.cmp(&a.record.seq)));
    }
    (hits, ev)
}

/// The part of `text` (at most `max` bytes) with the most query terms.
pub fn snippet(text: &str, terms: &HashSet<String>, max: usize) -> String {
    let (s, e) = snippet_range(text, terms, max);
    if s == 0 && e == text.len() {
        return text.to_string();
    }
    format!("{}{}{}", if s > 0 { "…" } else { "" }, text[s..e].trim(), if e < text.len() { "…" } else { "" })
}

/// The byte range [`snippet`] shows.
pub fn snippet_range(text: &str, terms: &HashSet<String>, max: usize) -> (usize, usize) {
    if text.len() <= max {
        return (0, text.len());
    }
    let mut hits = Vec::new();
    let mut pos = 0;
    for w in text.split(|c: char| !c.is_alphanumeric()) {
        if !w.is_empty() {
            if let Some(t) = strata::text::analyzer::term_of(&w.to_lowercase()) {
                if terms.contains(t.as_ref()) {
                    hits.push(pos);
                }
            }
        }
        pos += w.len() + 1;
    }
    let mut best = (0usize, 0usize);
    for &h in &hits {
        let start = h.saturating_sub(max / 5);
        let n = hits.iter().filter(|&&x| x >= start && x < start + max).count();
        if n > best.1 {
            best = (start, n);
        }
    }
    let floor = |mut i: usize| {
        i = i.min(text.len());
        while !text.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    (floor(best.0), floor(best.0 + max))
}

/// How retrieved records become injected text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackParams {
    pub budget_chars: usize,
    pub snippet_chars: usize,
    /// Records taken from one session turn at most.
    pub per_turn: usize,
    /// Records at most.
    pub max_records: usize,
}

impl Default for PackParams {
    fn default() -> Self {
        // Records are at most CHUNK_CHARS (400) bytes, so about ten fit the
        // budget with their headers. A coding turn spans many records (each
        // tool call and result), so the per-turn cap is loose: at 3, the
        // answer was often a fourth record of a turn already shown.
        PackParams { budget_chars: 4000, snippet_chars: 600, per_turn: 8, max_records: 12 }
    }
}

impl PackParams {
    /// The weak-evidence tier: three records, about 300 tokens.
    pub fn brief() -> PackParams {
        PackParams { budget_chars: 1200, snippet_chars: 500, per_turn: 1, max_records: 3 }
    }
}

/// Whether a stored prompt is essentially the query asked again: it matches
/// best and says nothing new, so it only takes up the budget.
fn same_question(r: &Record, terms: &HashSet<String>) -> bool {
    // Only stored questions: a statement that repeats the question's words
    // ("my dog is called Max") is the answer, not a repeat.
    if r.kind != Kind::Prompt || terms.is_empty() || !r.text.contains('?') {
        return false;
    }
    let other: HashSet<String> = strata::text::analyze_query(&r.text).into_iter().collect();
    let inter = terms.intersection(&other).count() as f64;
    let union = terms.union(&other).count() as f64;
    union > 0.0 && inter / union >= 0.7
}

/// Selects hits within the budget; returns them oldest first with snippets.
/// Answers marked uncertain and earlier copies of the same question are skipped.
pub fn pack(hits: &[Hit], query: &str, skip: &HashSet<String>, p: &PackParams) -> Vec<(Record, String)> {
    let terms: HashSet<String> = strata::text::analyze_query(query).into_iter().collect();
    let mut per: HashMap<(String, i64), usize> = HashMap::new();
    let mut used = 0usize;
    let mut out: Vec<(Record, String)> = Vec::new();
    for h in hits {
        if out.len() >= p.max_records {
            break;
        }
        if skip.contains(&h.record.id) || h.record.tag == "uncertain" || same_question(&h.record, &terms) {
            continue;
        }
        let k = (h.record.session.clone(), h.record.turn);
        if *per.get(&k).unwrap_or(&0) >= p.per_turn {
            continue;
        }
        let snip = snippet(&h.record.text, &terms, p.snippet_chars);
        if used + snip.len() + 80 > p.budget_chars && !out.is_empty() {
            continue;
        }
        used += snip.len() + 80;
        *per.entry(k).or_default() += 1;
        out.push((h.record.clone(), snip));
    }
    out.sort_by_key(|(r, _)| r.seq);
    out
}

/// `2026-10-11 07:43 UTC` for a Unix time.
pub fn fmt_time(ts: i64) -> String {
    let days = ts.div_euclid(86_400);
    let secs = ts.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", secs / 3600, (secs % 3600) / 60)
}

/// `3 h ago`, `2 days ago`.
pub fn fmt_ago(ts: i64, now: i64) -> String {
    let d = (now - ts).max(0);
    match d {
        0..=119 => "just now".into(),
        120..=7199 => format!("{} min ago", d / 60),
        7200..=172_799 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86_400),
    }
}

fn short(session: &str) -> String {
    session.chars().take(8).collect()
}

/// One injected record: a provenance line and its snippet.
pub fn fmt_record(r: &Record, snip: &str, now: i64) -> String {
    fmt_record_labeled(r, snip, now, "")
}

/// One injected record with a freshness label after its provenance line.
pub fn fmt_record_labeled(r: &Record, snip: &str, now: i64, label: &str) -> String {
    let mut head = format!("[{}] {} · session {} turn {} · {}", r.id, fmt_time(r.ts), short(&r.session), r.turn, r.kind.as_str());
    if r.tag != "-" && !r.tag.is_empty() {
        head += &format!(" ({})", r.tag);
    }
    head += &format!(" · {}{label}", fmt_ago(r.ts, now));
    format!("{head}\n{}", snip.trim())
}

/// The recall block for a prompt (`weak`: a low-confidence match).
pub fn fmt_recall_as(items: &[(Record, String)], now: i64, weak: bool) -> String {
    fmt_recall_checked(items, now, weak, None)
}

/// The recall block, notes labelled when their code changed since.
pub fn fmt_recall_checked(items: &[(Record, String)], now: i64, weak: bool, mut check: Option<&mut Checker>) -> String {
    let mut s = if weak {
        format!(
            "Engram found {} possibly related record{} from earlier work in this project (a weak match, outside the current context window). They may be irrelevant or outdated.\n",
            items.len(),
            if items.len() == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "Engram recalled {} record{} from earlier work in this project, outside the current context window (oldest first). They may be outdated; newer records and the current files take precedence.\n",
            items.len(),
            if items.len() == 1 { "" } else { "s" }
        )
    };
    for (r, snip) in items {
        let label = match check.as_deref_mut() {
            Some(c) if r.kind.is_knowledge() => c.check_record(&r).label(),
            _ => String::new(),
        };
        s += "\n";
        s += &fmt_record_labeled(r, snip, now, &label);
        s += "\n";
    }
    s += "\nThe memory_timeline tool shows what surrounded a record id; memory_search finds more.";
    s
}

pub fn fmt_recall(items: &[(Record, String)], now: i64) -> String {
    fmt_recall_as(items, now, false)
}

/// One line per further candidate (id, kind, opening words), for the model
/// to follow up with `memory_timeline`.
pub fn fmt_more(hits: &[Hit], shown: &HashSet<String>, n: usize) -> String {
    let more: Vec<String> = hits
        .iter()
        .filter(|h| !shown.contains(&h.record.id) && h.record.tag != "uncertain")
        .take(n)
        .map(|h| {
            let first: String = h.record.text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(90).collect();
            format!("- [{}] {}: {}…", h.record.id, h.record.kind.as_str(), first)
        })
        .collect();
    if more.is_empty() {
        String::new()
    } else {
        format!("\nFurther candidates (memory_timeline with an id shows the full record and its surroundings):\n{}", more.join("\n"))
    }
}

/// When a prompt gets the knowledge written down earlier (decisions, rules,
/// requirements, facts) that bears on it, independently of the gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeParams {
    pub enabled: bool,
    /// At most this many records per prompt.
    pub max_records: usize,
    /// A record qualifies with this cosine similarity to the prompt…
    pub min_cos: f64,
    /// …or with this share of the prompt's IDF weight among its terms.
    pub min_cov: f64,
    /// Candidates per search.
    pub n: usize,
}

impl Default for KnowledgeParams {
    fn default() -> Self {
        KnowledgeParams { enabled: true, max_records: 4, min_cos: 0.40, min_cov: 0.30, n: 20 }
    }
}

pub const KNOWLEDGE: [Kind; 5] = [Kind::Preference, Kind::Decision, Kind::Fact, Kind::Todo, Kind::Note];

/// Knowledge records that match a prompt well enough on their own text,
/// best first, leaving out `skip` and records closed as done or superseded.
pub fn knowledge(store: &Store, query: &str, filter: &Filter, skip: &HashSet<String>, p: &KnowledgeParams) -> Result<Vec<(Record, f64, f64)>> {
    let store = &store.pinned();
    let mut f = filter.clone();
    f.kinds = KNOWLEDGE.to_vec();
    let terms = strata::text::analyze_query(query);
    let (bm, dv) = both(
        || if terms.is_empty() { Ok(Vec::new()) } else { store.bm25_top(query, p.n, &f) },
        || store.vector_top(&store.embedder.embed_query(query)?, p.n, &f),
    );
    let mut by: HashMap<String, (Record, f64)> = HashMap::new();
    for (r, c) in dv? {
        by.insert(r.id.clone(), (r, c));
    }
    for (r, _) in bm? {
        by.entry(r.id.clone()).or_insert((r, 0.0));
    }
    let idf = if terms.is_empty() { Vec::new() } else { store.idf(&terms).unwrap_or_else(|_| vec![1.0; terms.len()]) };
    let total: f64 = idf.iter().sum();
    let todo_open = |r: &Record| r.kind != Kind::Todo || r.tag == "planned";
    let mut out: Vec<(Record, f64, f64)> = by
        .into_values()
        .filter(|(r, _)| !skip.contains(&r.id) && r.tag != "superseded" && r.tag != "done" && todo_open(r))
        .map(|(r, c)| {
            // A note's `When:` line counts: it names the requests it bears on.
            let have = analyzed(&r.key_text());
            let got: f64 = terms.iter().zip(&idf).filter(|(t, _)| have.contains(*t)).map(|(_, w)| w).sum();
            let cov = if total > 0.0 { got / total } else { 0.0 };
            (r, c, cov)
        })
        .filter(|(_, c, cov)| *c >= p.min_cos || *cov >= p.min_cov)
        .collect();
    out.sort_by(|a, b| (b.1 + b.2).total_cmp(&(a.1 + a.2)));
    out.truncate(p.max_records);
    Ok(out)
}

/// The live notes most related to `text` (by vector and BM25, no
/// thresholds), best first: what the distiller is shown, so that it can
/// update an old note instead of only the newest ones.
pub fn related(store: &Store, text: &str, n: usize) -> Result<Vec<Record>> {
    let store = &store.pinned();
    let f = Filter { kinds: KNOWLEDGE.to_vec(), ..Default::default() };
    let live = |r: &Record| r.tag != "superseded" && r.tag != "done" && r.tag != "cue" && (r.kind != Kind::Todo || r.tag == "planned");
    let (bm, dv) = both(
        || if strata::text::analyze_query(text).is_empty() { Ok(Vec::new()) } else { store.bm25_top(text, n * 2, &f) },
        || store.vector_top(&store.embedder.embed_query(text)?, n * 2, &f),
    );
    let mut score: HashMap<String, (Record, f64)> = HashMap::new();
    for (i, (r, _)) in dv?.into_iter().enumerate() {
        score.entry(r.id.clone()).or_insert((r, 0.0)).1 += 1.0 / (10.0 + i as f64);
    }
    for (i, (r, _)) in bm?.into_iter().enumerate() {
        score.entry(r.id.clone()).or_insert((r, 0.0)).1 += 1.0 / (10.0 + i as f64);
    }
    let mut v: Vec<(Record, f64)> = score.into_values().filter(|(r, _)| live(r)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(v.into_iter().take(n).map(|x| x.0).collect())
}

fn note_label(r: &Record) -> &'static str {
    match r.kind {
        Kind::Todo => "planned",
        Kind::Preference => "rule",
        k => k.as_str(),
    }
}

/// The knowledge block of a prompt's context.
pub fn fmt_knowledge(items: &[(Record, f64, f64)], now: i64) -> String {
    fmt_knowledge_checked(items, now, None)
}

/// The knowledge block, each note labelled when the code it is anchored to
/// changed since.
pub fn fmt_knowledge_checked(items: &[(Record, f64, f64)], now: i64, mut check: Option<&mut Checker>) -> String {
    let mut s = String::from("Engram notes from earlier sessions that may bear on this request (decisions, rules and requirements agreed before; check them before deciding):\n");
    for (r, _, _) in items {
        let t = r.text.split_whitespace().collect::<Vec<_>>().join(" ");
        let fresh = check.as_deref_mut().map(|c| c.check_record(&r).label()).unwrap_or_default();
        s += &format!("- [{}] {t} ({}, {}){fresh}\n", note_label(r), r.id, fmt_ago(r.ts, now));
    }
    s
}

/// The briefing given at the start of a session: the knowledge written
/// down earlier (planned work, rules, decisions, facts), where the last
/// session left off, and how to search the rest. Returns the text and the
/// ids it shows, so that per-prompt recall does not repeat them.
pub fn briefing(store: &Store, current_session: &str, max_chars: usize, now: i64) -> Result<(String, Vec<String>)> {
    briefing_checked(store, current_session, max_chars, now, None)
}

/// The briefing, with notes whose code changed since moved to a section of
/// their own and labelled, and an index of what else the notes are about.
pub fn briefing_checked(store: &Store, current_session: &str, max_chars: usize, now: i64, mut check: Option<&mut Checker>) -> Result<(String, Vec<String>)> {
    let store = &store.pinned();
    let total = store.len();
    if total == 0 {
        return Ok((String::new(), Vec::new()));
    }
    let none = Filter::default();
    let sessions = store.query("SELECT session, COUNT(*) FROM memory GROUP BY session ORDER BY 2 DESC LIMIT 1000")?.rows.len();
    let prev = Filter { exclude_window: Some((current_session.to_string(), 0)), ..Default::default() };
    let last = store.latest(&[], 1, &prev)?;
    let mut s = format!("Engram memory for this project: {total} records from {sessions} session{}", if sessions == 1 { "" } else { "s" });
    if let Some(r) = last.first() {
        s += &format!(", last activity {}", fmt_ago(r.ts, now));
    }
    s += ". Requirements, decisions and conventions agreed in earlier conversations are often not in the code: \
          before a design decision, and whenever a task refers to earlier work, search memory (memory_ask answers a question about earlier work by running several searches; \
          memory_search is one quick search; memory_timeline shows a record's surroundings; memory_sql runs structured queries). memory_remember stores a new decision or fact.\n";
    let clip = |t: &str, n: usize| -> String {
        let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if t.chars().count() > n {
            format!("{}…", t.chars().take(n).collect::<String>())
        } else {
            t
        }
    };
    let norm = |t: &str| t.split_whitespace().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" ");
    // Once knowledge is distilled, the cue-word preferences (sentences
    // picked from prompts by their wording) only repeat it in fragments.
    let distilled = !store.query("SELECT id FROM memory WHERE tag = 'distilled' LIMIT 1")?.rows.is_empty();
    let mut seen: HashSet<String> = HashSet::new();
    let mut ids = Vec::new();
    let mut stale: Vec<(Record, String)> = Vec::new();
    let mut unshown: Vec<Record> = Vec::new();
    // Most recent first, oldest shown first; one line per record, while the budget lasts.
    let mut section = |title: &str, recs: Vec<Record>, n: usize, keep: usize, s: &mut String, stale: &mut Vec<(Record, String)>, unshown: &mut Vec<Record>| {
        let mut lines = Vec::new();
        for r in recs {
            if r.tag == "superseded" || r.tag == "done" || (distilled && r.tag == "cue") || !seen.insert(norm(&r.text)) {
                continue;
            }
            let fresh = match check.as_deref_mut() {
                Some(c) if r.kind.is_knowledge() => c.check_record(&r),
                _ => crate::anchor::Freshness::Current,
            };
            if fresh.is_stale() {
                stale.push((r, fresh.label()));
                continue;
            }
            if lines.len() == keep {
                unshown.push(r);
                continue;
            }
            lines.push((r.id.clone(), format!("- {} ({}, {}){}\n", clip(&r.text, n), r.id, fmt_ago(r.ts, now), fresh.label())));
        }
        lines.reverse();
        let head = format!("\n{title}\n");
        let mut added = false;
        for (id, line) in lines {
            if s.len() + line.len() + if added { 0 } else { head.len() } > max_chars {
                continue;
            }
            if !added {
                *s += &head;
                added = true;
            }
            *s += &line;
            ids.push(id);
        }
    };
    let planned: Vec<Record> = store.latest(&[Kind::Todo], 60, &none)?.into_iter().filter(|r| r.tag == "planned").collect();
    section("Planned work, with the requirements agreed for it:", planned, 400, 10, &mut s, &mut stale, &mut unshown);
    section("Rules and conventions:", store.latest(&[Kind::Preference], 80, &none)?, 260, 14, &mut s, &mut stale, &mut unshown);
    section("Decisions:", store.latest(&[Kind::Decision], 80, &none)?, 260, 10, &mut s, &mut stale, &mut unshown);
    section("Facts about the project:", store.latest(&[Kind::Fact, Kind::Note], 80, &none)?, 260, 10, &mut s, &mut stale, &mut unshown);
    let mut stale_ids: Vec<String> = Vec::new();
    if !stale.is_empty() {
        let head = "\nNotes that may be out of date (the code they describe has changed since they were written; check it before relying on them):\n";
        let mut added = false;
        for (r, label) in stale.iter().take(6) {
            let line = format!("- {} ({}, {}){label}\n", clip(&r.text, 220), r.id, fmt_ago(r.ts, now));
            if s.len() + line.len() + if added { 0 } else { head.len() } > max_chars {
                continue;
            }
            if !added {
                s += head;
                added = true;
            }
            s += &line;
            stale_ids.push(r.id.clone());
        }
    }
    let checkpoint: Vec<Record> = store.latest(&[Kind::Todo], 10, &none)?.into_iter().filter(|r| r.tag != "planned" && r.tag != "done").take(1).collect();
    section("Open tasks at the last checkpoint:", checkpoint, 500, 1, &mut s, &mut stale, &mut unshown);
    section("Where the most recent earlier session left off:", store.latest(&[Kind::Episode], 3, &prev)?, 300, 3, &mut s, &mut stale, &mut unshown);
    ids.extend(stale_ids);
    // What the notes left out are about, so the model knows what to ask for.
    let unshown: Vec<Record> = unshown.into_iter().filter(|r| r.kind.is_knowledge()).collect();
    if !unshown.is_empty() {
        let mut count: HashMap<String, usize> = HashMap::new();
        for r in &unshown {
            for e in r.linked("about") {
                if let Some(name) = e.strip_prefix("e:") {
                    *count.entry(name.to_string()).or_default() += 1;
                }
            }
        }
        let mut top: Vec<(String, usize)> = count.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let names: Vec<String> = top.iter().take(20).map(|(n, c)| if *c > 1 { format!("{n} ({c})") } else { n.clone() }).collect();
        let mut line = format!("\n{} older note{} not shown here", unshown.len(), if unshown.len() == 1 { " is" } else { "s are" });
        if !names.is_empty() {
            line += &format!(", about: {}", names.join(", "));
        }
        line += ". memory_ask or memory_search finds them.\n";
        if s.len() + line.len() <= max_chars + 400 {
            s += &line;
        }
    }
    Ok((s, ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of a request divide its share of the cores between them,
    /// and a request (even one that panics) gives its share back.
    #[test]
    fn requests_divide_their_share() {
        let ((sa, (x, y)), sb) = both(|| (SHARE.get(), both(|| SHARE.get(), || SHARE.get())), || SHARE.get());
        assert!(sa >= 1 && sb >= 1 && sa + sb <= cores().max(2), "{sa} + {sb}");
        if sa >= 2 {
            assert_eq!(x + y, sa);
        } else {
            assert_eq!((x, y), (1, 1));
        }
        assert_eq!(SHARE.get(), 0);
        assert!(std::panic::catch_unwind(|| both(|| panic!("a search failed"), || 1)).is_err());
        assert_eq!(SHARE.get(), 0);
    }

    /// The session's last records bring back what the prompt does not
    /// say: a rule about the file being edited.
    #[test]
    fn the_session_tail_recalls_what_the_work_touches() {
        use crate::embed::HashEmbedder;
        use crate::record::Draft;
        use crate::store::StoreOptions;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("m.strata"), std::sync::Arc::new(HashEmbedder { dim: 64 }), StoreOptions { sync: false, ..Default::default() }).unwrap();
        let money = "src/billing/money.rs".to_string();
        store
            .append(
                "old",
                1,
                100,
                vec![
                    Draft::new(Kind::Decision, "Amounts in the billing module are integer cents, never floats").files(vec![money.clone()]),
                    Draft::new(Kind::Decision, "A new endpoint goes in src/api/routes.rs and gets an integration test"),
                    Draft::new(Kind::Decision, "Endpoints return JSON errors with a code and a message"),
                    Draft::new(Kind::Preference, "Keep pull requests small, one endpoint each"),
                ],
            )
            .unwrap();
        let first = store
            .append(
                "now",
                1,
                200,
                vec![
                    Draft::new(Kind::Prompt, "Refactor how invoice totals are rounded"),
                    Draft::new(Kind::ToolCall, "Edit src/billing/money.rs").tag("Edit").files(vec![money.clone()]),
                    Draft::new(Kind::Error, "test billing::rounding_of_totals failed: left 10.01 right 10.00").tag("Bash").files(vec![money.clone()]),
                ],
            )
            .unwrap()[0]
            .seq;
        let filter = Filter { exclude_window: Some(("now".into(), first)), ..Default::default() };
        let mut p = RankParams::default();
        p.tail = TailParams { enabled: true, ..TailParams::default() };
        let tail = Tail::of(&store.tail("now", p.tail.records).unwrap(), &p.tail);
        assert!(tail.files.contains(&money) && tail.digest.contains("rounding_of_totals"), "{tail:?}");
        assert_eq!(tail.vectors.len(), 3);
        let prompt = "now add a discount endpoint";
        let rank = |hits: &[Hit]| hits.iter().position(|h| h.record.text.contains("integer cents"));
        let (with, ev_with) = search_with(&store, prompt, Some(&tail), &filter, &p).unwrap();
        let (without, ev) = search_with(&store, prompt, None, &filter, &p).unwrap();
        // The gate reads the prompt's own evidence either way.
        assert_eq!(format!("{ev_with:?}"), format!("{ev:?}"));
        let show = |hits: &[Hit]| hits.iter().map(|h| format!("{:.3} {}", h.score, h.record.text)).collect::<Vec<_>>();
        // The billing rule the work touches gains about the tail's weight
        // (it heads the digest's list) and the file boost; an endpoint rule,
        // which the prompt names, stays first.
        let score = |hits: &[Hit]| hits.iter().find(|h| h.record.text.contains("integer cents")).map_or(0.0, |h| h.score);
        assert_eq!(rank(&without), Some(3), "{:?}", show(&without));
        assert!(score(&with) > score(&without) + p.tail.weight, "{:?} against {:?}", show(&with), show(&without));
        assert!(with[0].record.text.to_lowercase().contains("endpoint"), "{:?}", show(&with));
        // Off, the tail changes nothing.
        p.tail.enabled = false;
        let (off, _) = search_with(&store, prompt, Some(&tail), &filter, &p).unwrap();
        assert_eq!(off.iter().map(|h| &h.record.id).collect::<Vec<_>>(), without.iter().map(|h| &h.record.id).collect::<Vec<_>>());
    }

    #[test]
    fn time_formats() {
        assert_eq!(fmt_time(0), "1970-01-01 00:00 UTC");
        assert_eq!(fmt_time(1_791_788_086), "2026-10-12 06:54 UTC");
        assert_eq!(fmt_ago(100, 100 + 3 * 3600), "3 h ago");
    }

    #[test]
    fn snippet_finds_terms() {
        let text = format!("{} the planner probes local selectivity {}", "x ".repeat(500), "y ".repeat(500));
        let terms: HashSet<String> = strata::text::analyze_query("planner selectivity").into_iter().collect();
        let s = snippet(&text, &terms, 200);
        assert!(s.contains("planner probes local selectivity"), "{s}");
        assert!(s.len() <= 210);
    }
}
