//! Query understanding: a model turns a request into several searches over
//! memory, they all run, and a second model call picks, from everything
//! they found, the records that bear on the request.
//!
//! One search per request misses what a request needs but does not say: a
//! question like "where does the export run and why is the night job
//! responsible for it?" has two parts with different words, and a task like
//! "read the depot config from TOML" is bound by a fact (the depot PCs run
//! an old Python) that shares no word with it. The planner writes keyword,
//! semantic, note, entity (graph), file and SQL searches, several of one
//! kind if useful; their results are fused by reciprocal rank, and the
//! ranker reads the pooled candidates and keeps those that matter, with a
//! line on why and a short reading of what they say.
//!
//! The models are parameters: a small one by default, and a stronger one
//! for requests the planner calls hard when `hard_model` is set.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use strata::Value;

use crate::anchor::Checker;
use crate::distill::DistillConfig;
use crate::llm::{self, clip};
use crate::recall::{self, fmt_ago, KnowledgeParams, RankParams, THROUGH};
use crate::record::{entity_id, file_entity_id, Kind, Record, COLUMNS};
use crate::store::{row_record, Filter, Reach, Store};
use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DeepConfig {
    /// `off`; `tool` (only through memory_ask); `async` (after each prompt,
    /// in the background: shown at the next tool call, or at the end of the
    /// turn); `sync` (the prompt waits for it). `ENGRAM_DEEP` overrides it.
    pub mode: String,
    /// Model that writes the searches (`ENGRAM_DEEP_PLANNER_MODEL`, or
    /// `ENGRAM_DEEP_MODEL` for both).
    pub planner_model: String,
    /// Model that picks the records (`ENGRAM_DEEP_RANKER_MODEL`).
    pub ranker_model: String,
    /// Model that picks the records when the planner calls a request hard;
    /// empty: the ranker model always (`ENGRAM_DEEP_HARD_MODEL`).
    pub hard_model: String,
    /// Per model call.
    pub timeout_secs: u64,
    pub max_queries: usize,
    /// Results kept from each search.
    pub per_query: usize,
    /// Candidates the ranker reads.
    pub candidates: usize,
    /// Records the ranker may pick.
    pub max_picks: usize,
    /// Not below this many records in the store.
    pub min_records: u64,
    /// Not for prompts shorter than this many words.
    pub min_words: usize,
    /// In `async` mode: how long the end of a turn waits for a search still
    /// running, and whether it then asks the model to reconsider once when
    /// it found something the turn did not see.
    pub stop_wait_secs: u64,
    pub stop_recheck: bool,
    /// `model`: a model writes the searches; `none`: only the request's own
    /// keyword-and-vector and notes searches run, deeper (`ENGRAM_DEEP_PLAN`).
    pub planner: String,
    /// `model`: a model picks the records and writes a reading of them;
    /// `local`: they are scored here, with no model call (`ENGRAM_DEEP_RANK`).
    pub ranker: String,
}

impl Default for DeepConfig {
    fn default() -> Self {
        DeepConfig {
            mode: "async".into(),
            planner_model: "haiku".into(),
            ranker_model: "haiku".into(),
            hard_model: String::new(),
            timeout_secs: 60,
            max_queries: 8,
            per_query: 12,
            candidates: 30,
            max_picks: 8,
            min_records: 20,
            min_words: 4,
            stop_wait_secs: 25,
            stop_recheck: true,
            planner: "model".into(),
            ranker: "model".into(),
        }
    }
}

impl DeepConfig {
    /// The configuration with environment overrides applied.
    pub fn resolved(&self) -> DeepConfig {
        let mut c = self.clone();
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        if let Some(m) = env("ENGRAM_DEEP") {
            c.mode = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_MODEL") {
            c.planner_model = m.clone();
            c.ranker_model = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_PLANNER_MODEL") {
            c.planner_model = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_RANKER_MODEL") {
            c.ranker_model = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_HARD_MODEL") {
            c.hard_model = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_PLAN") {
            c.planner = m;
        }
        if let Some(m) = env("ENGRAM_DEEP_RANK") {
            c.ranker = m;
        }
        c
    }

    pub fn automatic(&self) -> bool {
        matches!(self.mode.as_str(), "async" | "sync")
    }

    /// Whether a prompt is worth a search (and its model calls).
    pub fn wants(&self, prompt: &str, records: u64) -> bool {
        let p = prompt.trim();
        self.automatic() && records >= self.min_records && !p.starts_with('/') && p.split_whitespace().count() >= self.min_words
    }
}

/// One search the planner asked for.
#[derive(Clone, Debug, Serialize)]
pub struct Query {
    /// text | keywords | semantic | notes | entity | file | sql
    pub kind: String,
    pub q: String,
    pub names: Vec<String>,
    pub path: String,
    pub sql_where: String,
    pub sql_order: String,
    pub kinds: Vec<Kind>,
}

impl Query {
    pub fn label(&self) -> String {
        match self.kind.as_str() {
            "entity" => format!("entity:{}", self.names.join("|")),
            "file" => format!("file:{}", self.path),
            "sql" => format!("sql:{}", clip(&self.sql_where, 60)),
            k => format!("{k}:{}", clip(&self.q, 60)),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Plan {
    pub need: bool,
    pub hard: bool,
    pub queries: Vec<Query>,
    pub cost_usd: f64,
    pub usage: llm::Usage,
    pub ms: u64,
    pub error: Option<String>,
}

/// A record found by one or more searches.
#[derive(Clone, Debug, Serialize)]
pub struct Cand {
    pub record: Record,
    pub score: f64,
    pub sources: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DeepResult {
    pub plan: Plan,
    pub candidates: Vec<Cand>,
    /// Picked records, most important first, with why.
    pub picks: Vec<(Record, String)>,
    pub answer: String,
    pub confidence: String,
    pub ranker_model: String,
    pub cost_usd: f64,
    pub usage: llm::Usage,
    pub ms: u64,
    pub error: Option<String>,
}

impl DeepResult {
    pub fn ids(&self) -> Vec<String> {
        self.picks.iter().map(|(r, _)| r.id.clone()).collect()
    }
}

const PLAN_SYSTEM: &str = "You turn a user's request into searches over an assistant's memory of earlier sessions with that user (usually work sessions in a software project). Output one JSON object and nothing else.";

const PLAN_PROMPT: &str = r#"An assistant is about to work on this request (usually in a software project; sometimes it is a question about earlier conversations with the user):
<request>
{request}
</request>
{recent}
Memory holds records of earlier sessions: user prompts, assistant answers, tool calls and their results, errors, one episode summary per turn, compaction summaries, and notes written down from conversations: decisions with their reasons, rules, requirements for planned work, and facts. It now holds {overview}.
Things the notes are about: {entities}
Files the notes mention: {files}

Write the searches that would find what earlier sessions said that bears on this request:
- the facts it needs, and for a question about earlier work or conversations, the records that answer each part of it (one search per part; for a question about how things changed or how many times something happened, search for every occurrence, not only the latest);
- decisions, rules and requirements that constrain how to do it, including ones the request does not mention: think about what the task touches (the machines and environment it runs on, file formats, people and systems involved, conventions, limits, earlier failed attempts) and search for those too, in the words earlier sessions would have used.
Several searches of the same type are fine. Types:
- {"type": "text", "q": "<natural language>"}: keyword and semantic search together, over everything.
- {"type": "keywords", "q": "<distinctive words, identifiers, error text>"}: keyword (BM25) search.
- {"type": "semantic", "q": "<a sentence phrased like the record you hope to find>"}: vector search.
- {"type": "notes", "q": "<what a decision, rule, requirement or fact would say>"}: only the written-down notes.
- {"type": "entity", "names": ["<a thing notes are about>", ...], "q": "<what you want about them>"}: records linked to those things, and their neighbours.
- {"type": "file", "path": "<file path>", "q": "<what you want about it>"}: records about a file.
- {"type": "sql", "where": "<condition>", "order": "<ordering>"}: a filter over memory(id, session, kind, tag, turn, seq, ts, importance, files, text, ctx); body is the full-text index over text and ctx; conditions may use MATCH(body, 'word AND word'), MATCH(files, 'name'), kind IN ('error', 'decision'), ts >= <unix seconds>; order is e.g. seq DESC or bm25(body, 'words') DESC.
Any search may add "kinds": [...] (prompt, answer, tool_call, tool_result, error, episode, summary, decision, preference, fact, todo, note).

Output {"need_memory": true or false, "difficulty": "easy" or "hard", "queries": [1 to {max} searches]}. need_memory is false only when earlier sessions cannot matter (a greeting, thanks, "commit this"). difficulty is hard when the request needs several facts combined or reasoning across sessions."#;

const RANK_SYSTEM: &str = "You judge which records from an assistant's memory of earlier sessions (usually work sessions in a software project) bear on a user's request. Output one JSON object and nothing else.";

const RANK_PROMPT: &str = r#"<request>
{request}
</request>

Records found in the memory of earlier sessions (id in brackets; a label in square brackets after a note says the code it describes changed since):

{cands}

Pick the records an assistant working on this request should see: those that state facts, decisions, rules, requirements or earlier results it needs, including constraints that apply although the request does not mention them. When the request asks for suggestions, recommendations or advice, also pick what the user said about their own tastes, habits, experience, plans and things they have, even when those records do not answer the request: the assistant should tailor its answer to them. Leave out records that are merely on the same topic. When two records disagree, the later one usually holds; pick it and say so. Order the picks by importance, at most {k}.

Output {"relevant": [{"id": "<id>", "why": "<under 20 words: what it tells the assistant>"}], "answer": "<for a question about earlier work or conversations, 1 to 3 sentences answering it from the picked records, citing ids; otherwise the one or two things the assistant must keep in mind; empty if nothing applies>", "confidence": "high" or "medium" or "low" or "none"}"#;

fn kinds_of(v: &Json) -> Vec<Kind> {
    v.get("kinds").and_then(|k| k.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().and_then(Kind::parse)).filter(|k| *k != Kind::Entity).collect()).unwrap_or_default()
}

/// Parses the planner's reply.
pub fn parse_plan(text: &str, max: usize) -> Plan {
    let Some(v) = llm::json_object(text) else {
        return Plan { need: true, error: Some("no JSON in the planner's reply".into()), ..Default::default() };
    };
    let s = |v: &Json, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
    let mut queries = Vec::new();
    for q in v.get("queries").and_then(|q| q.as_array()).into_iter().flatten() {
        let kind = s(q, "type");
        let mut query = Query { kind: kind.clone(), q: s(q, "q"), names: crate::distill::strings(q, "names", 6, 80), path: s(q, "path"), sql_where: s(q, "where"), sql_order: s(q, "order"), kinds: kinds_of(q) };
        let ok = match kind.as_str() {
            "text" | "keywords" | "semantic" | "notes" => !query.q.is_empty(),
            "entity" => !query.names.is_empty(),
            "file" => !query.path.is_empty(),
            "sql" => !query.sql_where.is_empty() && safe_sql(&query.sql_where) && safe_sql(&query.sql_order),
            _ => false,
        };
        if ok {
            if query.q.len() > 500 {
                query.q = clip(&query.q, 500);
            }
            queries.push(query);
        }
        if queries.len() >= max {
            break;
        }
    }
    Plan { need: v.get("need_memory").and_then(|b| b.as_bool()).unwrap_or(true), hard: s(&v, "difficulty") == "hard", queries, ..Default::default() }
}

/// A WHERE or ORDER BY fragment that cannot end the statement or comment out the rest.
/// Full-text calls name the indexed field `body`; the planner sometimes names
/// the stored columns it covers (`text`, `ctx`), which have no index.
fn text_to_body(sql: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(match|bm25)\s*\(\s*(?:text|ctx)\s*,").unwrap()).replace_all(sql, "$1(body,").into_owned()
}

fn safe_sql(s: &str) -> bool {
    let l = s.to_lowercase();
    s.len() <= 600 && !s.contains(';') && !s.contains("--") && !s.contains("/*") && !l.contains(" limit ") && !l.contains("select ")
}

/// What memory holds, for the planner: size, time span, entity and file names.
pub fn overview(store: &Store, now: i64) -> (String, String, String) {
    let n = store.len();
    let sessions = store.query("SELECT session, COUNT(*) FROM memory WHERE kind != 'entity' GROUP BY session ORDER BY 2 DESC LIMIT 1000").map(|r| r.rows.len()).unwrap_or(0);
    let notes = store
        .query("SELECT kind, COUNT(*) FROM memory WHERE kind IN ('decision', 'preference', 'fact', 'todo', 'note') GROUP BY kind ORDER BY 2 DESC LIMIT 10")
        .map(|r| r.rows.iter().filter_map(|x| x.get(1).and_then(|v| v.as_i64())).sum::<i64>())
        .unwrap_or(0);
    let span = |order: &str| store.query(&format!("SELECT ts FROM memory WHERE kind != 'entity' ORDER BY ts {order} LIMIT 1")).ok().and_then(|r| r.rows.first().and_then(|x| x[0].as_i64()));
    let mut ov = format!("{n} records from {sessions} session{} ({notes} written-down notes)", if sessions == 1 { "" } else { "s" });
    if let (Some(a), Some(b)) = (span("ASC"), span("DESC")) {
        ov += &format!(", from {} to {}, now {}", recall::fmt_time(a), recall::fmt_time(b), recall::fmt_time(now));
    }
    let names = store.query("SELECT id, text FROM memory WHERE kind = 'entity' ORDER BY seq DESC LIMIT 160").map(|r| r.rows).unwrap_or_default();
    let mut ents = Vec::new();
    let mut files = Vec::new();
    for r in names {
        let id = r.first().and_then(|v| v.as_str()).unwrap_or("");
        let t = r.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string();
        if id.starts_with("f:") {
            if files.len() < 40 {
                files.push(t);
            }
        } else if ents.len() < 80 {
            ents.push(t);
        }
    }
    let or_none = |v: Vec<String>| if v.is_empty() { "(none yet)".to_string() } else { v.join(", ") };
    (ov, or_none(ents), or_none(files))
}

/// Asks the planner for searches.
pub fn plan(store: &Store, cfg: &DeepConfig, cli: &DistillConfig, request: &str, recent: &[String], now: i64) -> Plan {
    let (ov, ents, files) = overview(store, now);
    let recent = if recent.is_empty() {
        String::new()
    } else {
        format!("Earlier requests in the current session, for context:\n<recent>\n{}\n</recent>\n", recent.iter().map(|r| clip(r, 400)).collect::<Vec<_>>().join("\n---\n"))
    };
    let prompt = PLAN_PROMPT
        .replace("{request}", &clip(request, 4000))
        .replace("{recent}", &recent)
        .replace("{overview}", &ov)
        .replace("{entities}", &ents)
        .replace("{files}", &files)
        .replace("{max}", &cfg.max_queries.to_string());
    match llm::complete(&cli.cli(&cfg.planner_model, cfg.timeout_secs), PLAN_SYSTEM, &prompt) {
        Ok(r) => {
            let mut p = parse_plan(&r.text, cfg.max_queries);
            p.cost_usd = r.cost_usd;
            p.usage = r.usage;
            p.ms = r.ms;
            p
        }
        Err(e) => Plan { need: true, error: Some(e.to_string()), ..Default::default() },
    }
}

/// The records one search finds, best first.
pub fn run_query(store: &Store, q: &Query, filter: &Filter, n: usize) -> Result<Vec<Record>> {
    let mut f = filter.clone();
    f.kinds = q.kinds.clone();
    let useful = |r: &Record| r.kind != Kind::Entity && r.tag != "uncertain" && r.tag != "superseded";
    let out: Vec<Record> = match q.kind.as_str() {
        "text" => recall::search(store, &q.q, &f, &RankParams { n: n.max(20), ..Default::default() })?.0.into_iter().map(|h| h.record).collect(),
        "keywords" => {
            if strata::text::analyze_query(&q.q).is_empty() {
                Vec::new()
            } else {
                store.bm25_top(&q.q, n, &f)?.into_iter().map(|x| x.0).collect()
            }
        }
        "semantic" => store.vector_top(&store.embedder.embed_query(&q.q)?, n, &f)?.into_iter().map(|x| x.0).collect(),
        "notes" => {
            let p = KnowledgeParams { enabled: true, max_records: n, min_cos: 0.0, min_cov: 0.0, n: n.max(20) };
            recall::knowledge(store, &q.q, filter, &HashSet::new(), &p)?.into_iter().map(|x| x.0).collect()
        }
        "entity" => {
            let text = if q.q.is_empty() { q.names.join(", ") } else { q.q.clone() };
            let qv = store.embedder.embed_query(&text)?;
            let mut seeds: Vec<String> = Vec::new();
            for name in &q.names {
                if crate::capture::paths_in(name).first().is_some_and(|p| p == name) {
                    seeds.push(file_entity_id(name));
                }
                seeds.extend(entity_id(name));
                let ef = Filter { kinds: vec![Kind::Entity], ..Default::default() };
                for (r, c) in store.vector_top(&store.embedder.embed_query(name)?, 2, &ef)? {
                    if c >= 0.5 && !seeds.contains(&r.id) {
                        seeds.push(r.id);
                    }
                }
                // Spelling variants ("depot PCs" for "depot PC") match by stem.
                if !strata::text::analyze_query(name).is_empty() {
                    for (r, _) in store.bm25_top(name, 2, &ef)? {
                        if !seeds.contains(&r.id) {
                            seeds.push(r.id);
                        }
                    }
                }
            }
            let reach = Reach { hops: 2, through: THROUGH.to_vec(), max_nodes: 5000 };
            store.reach(&seeds, &qv, n, &f, &reach)?.into_iter().map(|x| x.0).collect()
        }
        "file" => {
            let terms: Vec<String> = strata::text::analyze(&q.path);
            if terms.is_empty() {
                Vec::new()
            } else {
                let mut conds = f.conds();
                conds.push("MATCH(files, :files)".into());
                let order = if strata::text::analyze_query(&q.q).is_empty() { "seq DESC" } else { "bm25(body, :q, syntax => 'plain') DESC" };
                let params = f.params().set("files", Value::Str(terms.join(" AND "))).set("q", Value::Str(q.q.clone()));
                let rs = store.query_with(&format!("SELECT {COLUMNS} FROM memory WHERE {} ORDER BY {order} LIMIT {n}", conds.join(" AND ")), &params)?;
                rs.rows.iter().filter_map(|r| row_record(r)).collect()
            }
        }
        "sql" => {
            let mut conds = f.conds();
            conds.push(format!("({})", text_to_body(&q.sql_where)));
            let order = if q.sql_order.is_empty() { "seq DESC".to_string() } else { text_to_body(&q.sql_order) };
            let rs = store.query_with(&format!("SELECT {COLUMNS} FROM memory WHERE {} ORDER BY {order} LIMIT {n}", conds.join(" AND ")), &f.params())?;
            rs.rows.iter().filter_map(|r| row_record(r)).collect()
        }
        _ => Vec::new(),
    };
    Ok(out.into_iter().filter(useful).take(n).collect())
}

/// Runs every search (and two baseline ones on the request itself) and
/// fuses the lists by reciprocal rank.
pub fn gather(store: &Store, cfg: &DeepConfig, request: &str, plan: &Plan, filter: &Filter) -> (Vec<Cand>, Vec<String>) {
    // The searches run side by side, all on one snapshot.
    let store = &store.pinned();
    let mut queries: Vec<Query> = vec![
        Query { kind: "text".into(), q: clip(request, 2000), names: vec![], path: String::new(), sql_where: String::new(), sql_order: String::new(), kinds: vec![] },
        Query { kind: "notes".into(), q: clip(request, 2000), names: vec![], path: String::new(), sql_where: String::new(), sql_order: String::new(), kinds: vec![] },
    ];
    queries.extend(plan.queries.iter().cloned());
    // Without planned searches the request's own two fill the pool.
    let depth = if plan.queries.is_empty() { cfg.candidates.max(cfg.per_query) } else { cfg.per_query };
    let lists: Vec<(String, Result<Vec<Record>>)> = std::thread::scope(|s| {
        let hs: Vec<_> = queries.iter().map(|q| (q.label(), s.spawn(move || run_query(store, q, filter, depth)))).collect();
        hs.into_iter().map(|(l, h)| (l, h.join().unwrap_or_else(|_| Err(crate::err("search panicked"))))).collect()
    });
    let mut by: HashMap<String, Cand> = HashMap::new();
    let mut errors = Vec::new();
    for (label, res) in lists {
        match res {
            Ok(recs) => {
                for (i, r) in recs.into_iter().enumerate() {
                    let add = 1.0 / (60.0 + 1.0 + i as f64);
                    let e = by.entry(r.id.clone()).or_insert(Cand { record: r, score: 0.0, sources: Vec::new() });
                    e.score += add;
                    if !e.sources.contains(&label) {
                        e.sources.push(label.clone());
                    }
                }
            }
            Err(e) => errors.push(format!("{label}: {e}")),
        }
    }
    let mut cands: Vec<Cand> = by.into_values().collect();
    cands.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.record.seq.cmp(&a.record.seq)));
    cands.truncate(cfg.candidates);
    (cands, errors)
}

/// The turn's request a fragment came from, from its context.
fn request_of(r: &Record) -> Option<String> {
    let line = r.ctx.lines().next()?;
    let rest = line.strip_prefix("Turn ")?;
    let (_, req) = rest.split_once(" request: ")?;
    Some(clip(req, 160))
}

fn describe(r: &Record, terms: &HashSet<String>, now: i64, label: &str, max: usize) -> String {
    let kind = match r.kind {
        Kind::Todo if r.tag == "planned" => "requirement",
        Kind::Preference => "rule",
        k => k.as_str(),
    };
    let mut s = format!("[{}] {kind} · {} · {}{}", r.id, recall::fmt_time(r.ts), fmt_ago(r.ts, now), label);
    if !r.kind.is_knowledge() {
        if let Some(req) = request_of(r) {
            s += &format!("\n  (during the request: {req})");
        }
    }
    let t = recall::snippet(&r.text, terms, max);
    s += &format!("\n  {}", t.split_whitespace().collect::<Vec<_>>().join(" "));
    s
}

/// What the ranker picked, most important first, with why, its reading of
/// them, and what it cost.
#[derive(Clone, Debug, Default)]
pub struct Ranked {
    pub picks: Vec<(String, String)>,
    pub answer: String,
    pub confidence: String,
    pub model: String,
    pub cost_usd: f64,
    pub usage: llm::Usage,
    pub error: Option<String>,
}

/// Picks from the candidates: by a model, or with `ranker = "local"` by
/// [`local_scores`].
pub fn rank(store: &Store, cfg: &DeepConfig, cli: &DistillConfig, request: &str, cands: &[Cand], labels: &HashMap<String, String>, hard: bool, now: i64) -> Ranked {
    if cfg.ranker == "local" {
        return rank_local(store, cfg, request, cands);
    }
    let mut terms: HashSet<String> = strata::text::analyze_query(request).into_iter().collect();
    for c in cands {
        for s in &c.sources {
            if let Some((_, q)) = s.split_once(':') {
                terms.extend(strata::text::analyze_query(q));
            }
        }
    }
    let text: Vec<String> = cands.iter().map(|c| describe(&c.record, &terms, now, labels.get(&c.record.id).map(|s| s.as_str()).unwrap_or(""), 600)).collect();
    let prompt = RANK_PROMPT.replace("{request}", &clip(request, 4000)).replace("{cands}", &text.join("\n\n")).replace("{k}", &cfg.max_picks.to_string());
    let model = if hard && !cfg.hard_model.is_empty() { cfg.hard_model.clone() } else { cfg.ranker_model.clone() };
    let mut out = Ranked { model: model.clone(), confidence: "none".into(), ..Default::default() };
    match llm::complete(&cli.cli(&model, cfg.timeout_secs), RANK_SYSTEM, &prompt) {
        Ok(r) => {
            out.cost_usd = r.cost_usd;
            out.usage = r.usage;
            let Some(v) = llm::json_object(&r.text) else {
                out.error = Some("no JSON in the ranker's reply".into());
                return out;
            };
            let known: HashSet<&str> = cands.iter().map(|c| c.record.id.as_str()).collect();
            for p in v.get("relevant").and_then(|x| x.as_array()).into_iter().flatten() {
                let (id, why) = match p {
                    Json::String(s) => (s.trim().trim_matches(|c| c == '[' || c == ']').to_string(), String::new()),
                    _ => (
                        p.get("id").and_then(|x| x.as_str()).unwrap_or("").trim().trim_matches(|c| c == '[' || c == ']').to_string(),
                        p.get("why").and_then(|x| x.as_str()).unwrap_or("").trim().to_string(),
                    ),
                };
                if known.contains(id.as_str()) && !out.picks.iter().any(|(i, _)| *i == id) {
                    out.picks.push((id, clip(&why, 200)));
                }
                if out.picks.len() >= cfg.max_picks {
                    break;
                }
            }
            out.answer = clip(v.get("answer").and_then(|x| x.as_str()).unwrap_or("").trim(), 800);
            out.confidence = v.get("confidence").and_then(|x| x.as_str()).unwrap_or("low").trim().to_lowercase();
        }
        Err(e) => out.error = Some(e.to_string()),
    }
    out
}

/// Features the local ranker scores a candidate by, in [`FEATURES`] order.
pub const FEATURES: [&str; 12] =
    ["fused", "rank", "cos", "cover_idf", "cover", "note", "prompt", "answer", "tool", "len", "both", "newer"];

/// Weights of [`FEATURES`] and the intercept, fitted by logistic regression
/// on which candidates held a question's evidence in LongMemEval-S
/// (`scripts/local_ranker.py`; LoCoMo held out). Those histories have no
/// notes or tool records, so their weights are 0.
pub const LOCAL_WEIGHTS: [f64; 13] = [5.308, 1.224, 2.521, 1.815, -0.761, 0.000, 1.151, -1.151, 0.000, 4.442, 0.000, -0.612, -10.910];

/// How many candidates the local ranker picks. Its probabilities are low
/// for every candidate (few hold evidence), so a fixed count beats a
/// threshold: the top 5 hold all of a question's evidence 67% of the time
/// against 49% for the fused order's top 5 (cross-validated).
pub const LOCAL_PICKS: usize = 5;

/// Each candidate's features (see [`FEATURES`]), in candidate order.
pub fn features(store: &Store, request: &str, cands: &[Cand]) -> Vec<[f64; 12]> {
    let terms = strata::text::analyze_query(request);
    let idf = if terms.is_empty() { Vec::new() } else { store.idf(&terms).unwrap_or_else(|_| vec![1.0; terms.len()]) };
    let idf_total: f64 = idf.iter().sum();
    let qv = store.embedder.embed_query(request).unwrap_or_else(|_| vec![0.0; store.embedder.dim()]);
    let texts: Vec<String> = cands.iter().map(|c| c.record.embed_text()).collect();
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let vecs = if refs.is_empty() { Vec::new() } else { store.embedder.embed(&refs).unwrap_or_else(|_| vec![vec![0.0; store.embedder.dim()]; refs.len()]) };
    let top = cands.iter().map(|c| c.score).fold(0.0, f64::max);
    let mut seqs: Vec<i64> = cands.iter().map(|c| c.record.seq).collect();
    seqs.sort_unstable();
    cands
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let r = &c.record;
            let have: HashSet<String> = strata::text::analyze(&format!("{}\n{}", r.text, r.ctx)).into_iter().collect();
            let got: f64 = terms.iter().zip(&idf).filter(|(t, _)| have.contains(*t)).map(|(_, w)| w).sum();
            let n_got = terms.iter().filter(|t| have.contains(*t)).count();
            let cos: f64 = vecs.get(i).map(|v| v.iter().zip(&qv).map(|(a, b)| (*a as f64) * (*b as f64)).sum()).unwrap_or(0.0);
            let newer = seqs.iter().position(|s| *s == r.seq).unwrap_or(0) as f64 / (seqs.len().max(2) - 1) as f64;
            [
                if top > 0.0 { c.score / top } else { 0.0 },
                1.0 / (1.0 + i as f64),
                cos,
                if idf_total > 0.0 { got / idf_total } else { 0.0 },
                if terms.is_empty() { 0.0 } else { n_got as f64 / terms.len() as f64 },
                r.kind.is_knowledge() as u8 as f64,
                (r.kind == Kind::Prompt) as u8 as f64,
                (r.kind == Kind::Answer) as u8 as f64,
                matches!(r.kind, Kind::ToolCall | Kind::ToolResult | Kind::Error) as u8 as f64,
                ((r.text.len() + 1) as f64).ln() / 10.0,
                (c.sources.len() > 1) as u8 as f64,
                newer,
            ]
        })
        .collect()
}

/// Probability the local model gives each candidate.
pub fn local_scores(store: &Store, request: &str, cands: &[Cand]) -> Vec<f64> {
    features(store, request, cands)
        .iter()
        .map(|f| {
            let z: f64 = f.iter().zip(&LOCAL_WEIGHTS).map(|(x, w)| x * w).sum::<f64>() + LOCAL_WEIGHTS[FEATURES.len()];
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

/// The local ranker: its top [`LOCAL_PICKS`] candidates (at most
/// `max_picks`), best first. No model call, so no reading of them.
fn rank_local(store: &Store, cfg: &DeepConfig, request: &str, cands: &[Cand]) -> Ranked {
    let scores = local_scores(store, request, cands);
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    order.truncate(LOCAL_PICKS.min(cfg.max_picks));
    let best = order.first().map(|i| scores[*i]).unwrap_or(0.0);
    Ranked {
        picks: order.iter().map(|i| (cands[*i].record.id.clone(), String::new())).collect(),
        confidence: if best >= 0.5 { "high" } else if best >= 0.2 { "medium" } else { "low" }.into(),
        model: "local".into(),
        ..Default::default()
    }
}

/// Plans, searches and ranks for one request. `recent` holds earlier
/// requests of the session; `project` (for anchored notes) labels notes
/// whose code changed. `now` is the time of the request.
pub fn run(store: &Store, cfg: &DeepConfig, cli: &DistillConfig, request: &str, recent: &[String], filter: &Filter, project: Option<&Path>, now: i64) -> DeepResult {
    let t0 = Instant::now();
    let plan = if cfg.planner == "none" { Plan { need: true, ..Default::default() } } else { plan(store, cfg, cli, request, recent, now) };
    let mut out = DeepResult { plan, ..Default::default() };
    out.cost_usd = out.plan.cost_usd;
    out.usage = out.plan.usage;
    if !out.plan.need {
        out.confidence = "none".into();
        out.ms = t0.elapsed().as_millis() as u64;
        return out;
    }
    let (cands, errors) = gather(store, cfg, request, &out.plan, filter);
    if !errors.is_empty() && out.plan.error.is_none() {
        out.plan.error = Some(errors.join("; "));
    }
    let mut labels = HashMap::new();
    if let Some(p) = project {
        let mut ck = Checker::new(p);
        for c in &cands {
            if c.record.kind.is_knowledge() {
                let l = ck.check_record(&c.record).label();
                if !l.is_empty() {
                    labels.insert(c.record.id.clone(), l);
                }
            }
        }
    }
    if cands.is_empty() {
        out.confidence = "none".into();
        out.ms = t0.elapsed().as_millis() as u64;
        return out;
    }
    let r = rank(store, cfg, cli, request, &cands, &labels, out.plan.hard, now);
    out.adopt(r, cands);
    out.ms = t0.elapsed().as_millis() as u64;
    out
}

impl DeepResult {
    /// Takes the ranker's picks (records from `cands`), reading and cost.
    pub fn adopt(&mut self, r: Ranked, cands: Vec<Cand>) {
        let by: HashMap<&str, &Record> = cands.iter().map(|c| (c.record.id.as_str(), &c.record)).collect();
        self.picks = r.picks.into_iter().filter_map(|(id, why)| by.get(id.as_str()).map(|rec| ((*rec).clone(), why))).collect();
        self.answer = r.answer;
        self.confidence = r.confidence;
        self.ranker_model = r.model;
        self.cost_usd += r.cost_usd;
        self.usage.add(r.usage);
        self.error = r.error;
        self.candidates = cands;
    }
}

/// The context block for a result: the picked records not shown already
/// (`shown`), why each matters, and the ranker's reading. Empty when there
/// is nothing new.
pub fn format(res: &DeepResult, shown: &HashSet<String>, now: i64, project: Option<&Path>) -> String {
    let new: Vec<&(Record, String)> = res.picks.iter().filter(|(r, _)| !shown.contains(&r.id)).collect();
    if new.is_empty() || res.confidence == "none" {
        return String::new();
    }
    let mut ck = project.map(Checker::new);
    let mut s = format!(
        "Engram searched the memory of earlier sessions for this request ({} searches) and judged these records relevant, most important first. They may be outdated; the current files take precedence.\n",
        res.plan.queries.len() + 2
    );
    let terms: HashSet<String> = HashSet::new();
    for (r, why) in &new {
        let label = match (&mut ck, r.kind.is_knowledge()) {
            (Some(c), true) => c.check_record(&r).label(),
            _ => String::new(),
        };
        s += &format!("\n{}", describe(r, &terms, now, &label, 700));
        if !why.is_empty() {
            s += &format!("\n  Why it matters: {why}");
        }
        s += "\n";
    }
    let already: Vec<&str> = res.picks.iter().filter(|(r, _)| shown.contains(&r.id)).map(|(r, _)| r.id.as_str()).collect();
    if !already.is_empty() {
        s += &format!("\nAlso relevant, shown earlier: {}.\n", already.join(", "));
    }
    if !res.answer.is_empty() && matches!(res.confidence.as_str(), "high" | "medium") {
        s += &format!("\nEngram's reading of these records (a model's summary; check it against them): {}\n", res.answer.split_whitespace().collect::<Vec<_>>().join(" "));
    }
    s += "\nmemory_timeline shows what surrounded a record id; memory_ask runs this search for another question.";
    s
}

/// The log line for a result.
pub fn log_json(res: &DeepResult) -> Json {
    json!({
        "need": res.plan.need, "hard": res.plan.hard, "queries": res.plan.queries.iter().map(|q| q.label()).collect::<Vec<_>>(),
        "candidates": res.candidates.len(), "picks": res.ids(), "confidence": res.confidence, "ranker_model": res.ranker_model,
        "cost_usd": res.cost_usd, "tokens": res.usage.json(), "plan_ms": res.plan.ms, "ms": res.ms, "error": res.error.clone().or(res.plan.error.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;
    use crate::record::Draft;
    use crate::store::StoreOptions;
    use std::sync::Arc;

    #[test]
    fn plans_are_parsed_and_checked() {
        let text = r#"Here: {"need_memory": true, "difficulty": "hard", "queries": [
            {"type": "keywords", "q": "tomllib python version"},
            {"type": "entity", "names": ["depot PC", "config.toml"], "q": "python version"},
            {"type": "sql", "where": "kind = 'decision'; DROP", "order": "seq DESC"},
            {"type": "sql", "where": "kind IN ('decision', 'fact') AND MATCH(body, 'python')", "order": "seq DESC"},
            {"type": "file", "path": "src/config.py"},
            {"type": "magic", "q": "x"},
            {"type": "notes", "q": "which Python the client machines run", "kinds": ["fact", "entity"]}
        ]}"#;
        let p = parse_plan(text, 8);
        assert!(p.need && p.hard);
        let kinds: Vec<&str> = p.queries.iter().map(|q| q.kind.as_str()).collect();
        assert_eq!(kinds, vec!["keywords", "entity", "sql", "file", "notes"]);
        assert_eq!(p.queries[4].kinds, vec![Kind::Fact]);
        assert_eq!(parse_plan("{\"need_memory\": false, \"queries\": []}", 8).need, false);
        assert!(parse_plan("nothing", 8).error.is_some());
    }

    #[test]
    fn searches_of_every_type_run() {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(&dir.path().join("m.strata"), Arc::new(HashEmbedder { dim: 64 }), StoreOptions { sync: false, ..Default::default() }).unwrap();
        s.append("a", 1, 100, vec![Draft::new(Kind::Prompt, "Please read the depot config from config.toml")]).unwrap();
        s.append("a", 1, 101, vec![Draft::new(Kind::Fact, "The depot PCs run Python 3.8, so no tomllib").entities(vec!["depot PC".into()]).files(vec!["src/config.py".into()])]).unwrap();
        s.append("b", 1, 200, vec![Draft::new(Kind::Error, "ModuleNotFoundError: tomllib").files(vec!["src/config.py".into()])]).unwrap();
        let f = Filter::default();
        let q = |kind: &str| Query { kind: kind.into(), q: "depot python".into(), names: vec!["depot PCs".into()], path: "src/config.py".into(), sql_where: "kind = 'error'".into(), sql_order: "seq DESC".into(), kinds: vec![] };
        for k in ["text", "keywords", "semantic", "notes", "entity", "file", "sql"] {
            let got = run_query(&s, &q(k), &f, 5).unwrap();
            assert!(!got.is_empty(), "{k}");
            assert!(got.iter().all(|r| r.kind != Kind::Entity), "{k}");
        }
        assert_eq!(run_query(&s, &q("sql"), &f, 5).unwrap()[0].kind, Kind::Error);
        // The live window reaches the file and SQL searches as a parameter,
        // and a text with quotes is a value, not SQL.
        let window = Filter { exclude_window: Some(("a".into(), 1)), ..Default::default() };
        for k in ["file", "sql"] {
            let got = run_query(&s, &Query { q: "it's 'tomllib'".into(), ..q(k) }, &window, 5).unwrap();
            assert_eq!(got.iter().map(|r| r.session.as_str()).collect::<Vec<_>>(), ["b"], "{k}");
        }
        // A full-text call on the stored column searches the index instead.
        let on_text = Query { sql_where: "MATCH(text, 'tomllib')".into(), sql_order: "bm25(text, 'tomllib') DESC".into(), ..q("sql") };
        assert_eq!(run_query(&s, &on_text, &f, 5).unwrap().len(), 2);
        assert_eq!(text_to_body("kind = 'x' AND match ( ctx , 'a')"), "kind = 'x' AND match(body, 'a')");
        let plan = Plan { need: true, queries: vec![q("entity"), q("sql")], ..Default::default() };
        let (cands, errors) = gather(&s, &DeepConfig::default(), "read the depot config", &plan, &f);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(cands.len(), 3);
        // Found by several searches, so ranked first.
        assert!(cands[0].sources.len() >= 2);
    }

    #[test]
    fn results_format_only_new_picks() {
        let r = Record::from_draft(Draft::new(Kind::Fact, "The depot PCs run Python 3.8"), "s", 1, 1, 100);
        let res = DeepResult { picks: vec![(r.clone(), "rules out tomllib".into())], answer: "Use configparser; tomllib needs 3.11 [s:1].".into(), confidence: "high".into(), ..Default::default() };
        let t = format(&res, &HashSet::new(), 200, None);
        assert!(t.contains("[s:1] fact") && t.contains("Why it matters: rules out tomllib") && t.contains("Engram's reading"), "{t}");
        let shown: HashSet<String> = [r.id.clone()].into_iter().collect();
        assert!(format(&res, &shown, 200, None).is_empty());
    }
}
