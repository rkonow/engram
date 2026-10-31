//! What should a memory record be? Keys, values and queries compared on the
//! long-conversation benchmarks.
//!
//!   engram-structure --data longmemeval_s.jsonl --out results/structure
//!
//! Each unit becomes one store per *layout*. A layout decides the keys: how
//! a turn is cut into indexed rows (the window), what context each row's
//! lexical key carries (nothing, the request it answers, or the other side
//! of the exchange), and what the row's vector embeds. A *presentation*
//! decides the value: the text a retrieved row puts into the context. Every
//! combination is packed into fixed character budgets and scored on whether
//! an evidence turn, and the answer itself, reach the model.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;
use serde_json::{json, Value as Json};

use engram::benchdata::{parse_time, read_units, Question, Unit};
use engram::capture::chunk_spans;
use engram::embed::{self, Embedder};
use engram::recall::{self, GraphParams, RankParams};
use engram::record::{entity_id, Kind, Link, Record};
use engram::store::{Filter, Hit, Store, StoreOptions};
use strata::VectorIndexKind;

#[derive(Parser)]
struct Cli {
    /// A converted benchmark (LongMemEval, LoCoMo).
    #[arg(long)]
    data: Option<PathBuf>,
    /// Instead: a Claude Code transcript, asked the probe questions.
    #[arg(long)]
    transcript: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    probes: Vec<PathBuf>,
    /// `window_line:ask_line` points (see engram-bench probes).
    #[arg(long, value_delimiter = ',')]
    points: Vec<String>,
    #[arg(long, default_value = "/home/user/codex-test-2")]
    project: String,
    #[arg(long, default_value = "static")]
    embedder: String,
    #[arg(long, default_value = "results/structure")]
    out: PathBuf,
    /// Only every n-th unit.
    #[arg(long, default_value_t = 1)]
    every: usize,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Layouts to run (default: all).
    #[arg(long, value_delimiter = ',')]
    layouts: Vec<String>,
    #[arg(long, default_value = "")]
    tag: String,
    /// Facts per session (scripts/extract_facts.py) for the fact layouts.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// Write the injected context of `layout/presentation/budget` for every
    /// question to `<out>/context_<...>.jsonl`.
    #[arg(long, value_delimiter = ',')]
    dump: Vec<String>,
    /// Graph recall variants run on the graph layout (default: all).
    #[arg(long, value_delimiter = ',')]
    graphs: Vec<String>,
}

/// What the lexical key carries besides the row's own text.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ctx {
    None,
    /// A response carries the request it answers.
    Request,
    /// Both sides carry the other.
    Partner,
}

/// How turns become indexed rows.
#[derive(Clone, Debug)]
struct Layout {
    name: &'static str,
    /// Row size in bytes; 0 makes one row per session.
    window: usize,
    ctx: Ctx,
    ctx_chars: usize,
    /// The vector embeds the context too (else the row's text only).
    dense_ctx: bool,
    /// Only user turns are indexed.
    user_only: bool,
    /// Also index rows of this second window size.
    also: Option<usize>,
    facts: Facts,
}

/// How extracted facts take part (see `--facts`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Facts {
    No,
    /// Facts are the only rows; each fact is its own value.
    Only,
    /// Fact rows next to the turn rows.
    WithTurns,
    /// A turn's facts join its key; the turn stays the value.
    Keys,
    /// [`Facts::WithTurns`], with links: each fact to the turn it comes
    /// from and to the entities it is about, which are rows of their own.
    /// Graph recall walks them (see [`graph_cfgs`]).
    Graph,
}

/// Facts by session content hash (see [`session_hash`]): (turn index within
/// the session, sentence, entities). Session ids repeat across LoCoMo
/// conversations, so the content picks the facts.
type SessionFacts = Vec<(usize, String, Vec<String>)>;
static FACTS: std::sync::OnceLock<HashMap<String, SessionFacts>> = std::sync::OnceLock::new();

/// Graph recall variants, each named; the first is the default.
fn graph_cfgs() -> Vec<(&'static str, GraphParams)> {
    let b = GraphParams::default();
    vec![
        ("g", b.clone()),
        ("g-h3", GraphParams { hops: 3, ..b.clone() }),
        ("g-w0.25", GraphParams { weight: 0.25, ..b.clone() }),
        ("g-w1", GraphParams { weight: 1.0, ..b.clone() }),
        ("g-s3", GraphParams { seeds: 3, ..b.clone() }),
        ("g-s10", GraphParams { seeds: 10, ..b.clone() }),
        ("g-e0", GraphParams { entity_seeds: 0, ..b.clone() }),
        ("g-e5", GraphParams { entity_seeds: 5, ..b.clone() }),
        ("g-n40", GraphParams { n: 40, ..b.clone() }),
        ("g-k60", GraphParams { k: 60.0, ..b.clone() }),
        ("g-h3-w1", GraphParams { hops: 3, weight: 1.0, ..b.clone() }),
        ("g-h3-n40", GraphParams { hops: 3, n: 40, ..b }),
    ]
}
/// The graph variants asked for (`--graphs`).
static GRAPHS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
/// Characters each injected fact costs besides its text (a date and an id).
const FACT_HEADER: usize = 40;

fn layouts() -> Vec<Layout> {
    let l = |name, window, ctx, ctx_chars| Layout { name, window, ctx, ctx_chars, dense_ctx: true, user_only: false, also: None, facts: Facts::No };
    vec![
        l("turn", 1500, Ctx::None, 0),
        l("turn+req", 1500, Ctx::Request, 300),
        l("turn+partner", 1500, Ctx::Partner, 300),
        l("win800+req", 800, Ctx::Request, 200),
        l("win400+req", 400, Ctx::Request, 200),
        l("win200+req", 200, Ctx::Request, 200),
        l("win400", 400, Ctx::None, 0),
        Layout { dense_ctx: false, ..l("win400+req/vec-text", 400, Ctx::Request, 200) },
        l("win400+partner", 400, Ctx::Partner, 200),
        Layout { also: Some(400), ..l("turn+win400+req", 1500, Ctx::Request, 300) },
        Layout { user_only: true, ..l("user-turns", 1500, Ctx::None, 0) },
        l("session", 0, Ctx::None, 0),
        Layout { facts: Facts::Only, ..l("facts", 1500, Ctx::None, 0) },
        Layout { facts: Facts::WithTurns, ..l("turn+facts", 1500, Ctx::None, 0) },
        Layout { facts: Facts::Keys, ..l("turn/fact-keys", 1500, Ctx::None, 0) },
        Layout { facts: Facts::Graph, ..l("turn+facts+graph", 1500, Ctx::None, 0) },
    ]
}

/// What a retrieved row puts into the context.
#[derive(Clone, Copy, Debug)]
enum Pres {
    /// The part of the row's own text with the most query terms (Engram today).
    Snip(usize),
    /// The row's span, widened within its turn to this many bytes.
    Span(usize),
    /// [`Pres::Span`] plus the head of the other side of the exchange.
    SpanPartner(usize, usize),
    /// The whole turn, or its best part when longer.
    Turn(usize),
}

impl Pres {
    fn name(&self) -> String {
        match self {
            Pres::Snip(n) => format!("snip{n}"),
            Pres::Span(n) => format!("span{n}"),
            Pres::SpanPartner(n, p) => format!("span{n}+p{p}"),
            Pres::Turn(n) => format!("turn{n}"),
        }
    }
}

fn presentations() -> Vec<Pres> {
    vec![Pres::Snip(600), Pres::Span(400), Pres::Span(600), Pres::SpanPartner(600, 150), Pres::SpanPartner(400, 150), Pres::Turn(1500)]
}

const BUDGETS: [usize; 4] = [1000, 2000, 4000, 8000];
/// Characters each injected record costs besides its text (id, time, kind).
const HEADER: usize = 80;

/// An indexed row: a span of one turn, or a whole session.
struct Row {
    /// Turns the row covers (one, or a session's).
    turns: Vec<usize>,
    /// The row's span in the text of `turns` joined by newlines.
    s: usize,
    e: usize,
    text: String,
    ctx: String,
    dense: String,
    /// A fact row: the sentence is the row's text and its value.
    fact: Option<String>,
    /// A fact row's entities.
    entities: Vec<String>,
    /// An entity row (never a result; walks pass through it).
    entity: Option<String>,
}

/// The text of `turns` joined by newlines, with each turn's start offset.
fn doc(texts: &[String], turns: &[usize]) -> (String, Vec<(usize, usize)>) {
    let mut s = String::new();
    let mut map = Vec::with_capacity(turns.len());
    for &t in turns {
        if !s.is_empty() {
            s.push('\n');
        }
        map.push((s.len(), t));
        s.push_str(&texts[t]);
    }
    (s, map)
}

fn clip_bytes(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut i = n;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    &s[..i]
}

fn rows(u: &Unit, texts: &[String], l: &Layout) -> Vec<Row> {
    let mut out = Vec::new();
    if l.window == 0 {
        let mut order: Vec<String> = Vec::new();
        let mut by: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, t) in u.turns.iter().enumerate() {
            if !by.contains_key(&t.session_id) {
                order.push(t.session_id.clone());
            }
            by.entry(t.session_id.clone()).or_default().push(i);
        }
        for sid in order {
            let turns = by.remove(&sid).unwrap();
            let (text, _) = doc(texts, &turns);
            out.push(Row { s: 0, e: text.len(), dense: text.clone(), text, ctx: String::new(), turns, fact: None, entities: Vec::new(), entity: None });
        }
        return out;
    }
    // Each turn's facts, when the layout uses them.
    let mut turn_facts: Vec<Vec<(&str, &[String])>> = vec![Vec::new(); texts.len()];
    if l.facts != Facts::No {
        let facts = FACTS.get().expect("--facts is needed for the fact layouts");
        let mut by: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, t) in u.turns.iter().enumerate() {
            by.entry(t.session_id.as_str()).or_default().push(i);
        }
        for idx in by.values() {
            let h = session_hash(idx.iter().map(|&i| (u.turns[i].role.as_str(), u.turns[i].text.as_str())));
            for (k, f, ents) in facts.get(&h).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Some(&i) = idx.get(*k) {
                    turn_facts[i].push((f.as_str(), ents.as_slice()));
                }
            }
        }
    }
    if matches!(l.facts, Facts::Only | Facts::WithTurns | Facts::Graph) {
        for (i, fs) in turn_facts.iter().enumerate() {
            for (f, ents) in fs {
                let entities = if l.facts == Facts::Graph { ents.to_vec() } else { Vec::new() };
                out.push(Row { turns: vec![i], s: 0, e: 0, text: f.to_string(), ctx: String::new(), dense: f.to_string(), fact: Some(f.to_string()), entities, entity: None });
            }
        }
        if l.facts == Facts::Only {
            return out;
        }
    }
    let windows: Vec<usize> = std::iter::once(l.window).chain(l.also).collect();
    for (i, text) in texts.iter().enumerate() {
        if l.user_only && u.turns[i].role != "user" {
            continue;
        }
        let other = match l.ctx {
            Ctx::None => None,
            Ctx::Request => u.request_of(i),
            Ctx::Partner => u.partner_of(i),
        };
        let mut ctx = other.map(|j| clip_bytes(&texts[j], l.ctx_chars).to_string()).unwrap_or_default();
        if l.facts == Facts::Keys && !turn_facts[i].is_empty() {
            let fs: Vec<&str> = turn_facts[i].iter().map(|f| f.0).collect();
            ctx = format!("{ctx}\n{}", fs.join("\n")).trim().to_string();
        }
        for &w in &windows {
            for (s, e) in chunk_spans(text, w) {
                let t = text[s..e].to_string();
                let dense = if l.dense_ctx && !ctx.is_empty() { format!("{ctx}\n{t}") } else { t.clone() };
                out.push(Row { turns: vec![i], s, e, text: t, ctx: ctx.clone(), dense, fact: None, entities: Vec::new(), entity: None });
            }
        }
    }
    if l.facts == Facts::Graph {
        let mut seen = HashSet::new();
        let names: Vec<String> = out.iter().flat_map(|r| r.entities.iter().cloned()).collect();
        for n in names {
            if let Some(id) = entity_id(&n) {
                if seen.insert(id) {
                    out.push(Row { turns: vec![0], s: 0, e: 0, text: n.clone(), ctx: String::new(), dense: n.clone(), fact: None, entities: Vec::new(), entity: Some(n) });
                }
            }
        }
    }
    out
}

/// What one retrieved turn contributes to the context: spans of its text,
/// and the head of the other side of the exchange.
#[derive(Clone)]
struct Piece {
    turn: usize,
    /// Spans of the turn's text, sorted and disjoint.
    spans: Vec<(usize, usize)>,
    /// The other side of the exchange and how many bytes of its head show.
    partner: Option<(usize, usize)>,
    /// Spans over several turns (a session row): (turn, start, end).
    multi: Vec<(usize, usize, usize)>,
    /// A fact (shows no turn text; it stands for its source turn).
    fact: Option<String>,
}

impl Piece {
    fn segs(&self) -> Vec<(usize, usize, usize)> {
        let mut v: Vec<(usize, usize, usize)> = self.spans.iter().map(|&(s, e)| (self.turn, s, e)).collect();
        v.extend(self.partner.map(|(j, n)| (j, 0, n)));
        v.extend(self.multi.iter().copied());
        v
    }

    fn add_span(&mut self, s: usize, e: usize) {
        self.spans.push((s, e));
        self.spans.sort();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(a, b) in &self.spans {
            match merged.last_mut() {
                Some(last) if a <= last.1 + 1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        self.spans = merged;
    }

    /// The text shown, each part labelled with its speaker.
    fn text(&self, u: &Unit, texts: &[String]) -> String {
        if let Some(f) = &self.fact {
            return format!("fact: {f}");
        }
        let part = |t: usize, spans: &[(usize, usize)]| -> String {
            let text = &texts[t];
            let mut s = String::new();
            for (k, &(a, b)) in spans.iter().enumerate() {
                if a > 0 {
                    s.push('…');
                } else if k > 0 {
                    s.push(' ');
                }
                s.push_str(text[a..b].trim());
                if b < text.len() && spans.get(k + 1).is_none() {
                    s.push('…');
                }
            }
            format!("{}: {s}", u.turns[t].role)
        };
        if !self.multi.is_empty() {
            return self.multi.iter().map(|&(t, a, b)| part(t, &[(a, b)])).collect::<Vec<_>>().join("\n");
        }
        let main = part(self.turn, &self.spans);
        match self.partner {
            Some((j, n)) if j < self.turn => format!("{}\n{main}", part(j, &[(0, n)])),
            Some((j, n)) => format!("{main}\n{}", part(j, &[(0, n)])),
            None => main,
        }
    }
}

/// Maps a byte range of a joined doc to per-turn spans.
fn map_range(map: &[(usize, usize)], texts: &[String], s: usize, e: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for &(off, t) in map {
        let (ts, te) = (off, off + texts[t].len());
        let (a, b) = (s.max(ts), e.min(te));
        if a < b {
            out.push((t, a - off, b - off));
        }
    }
    out
}

fn floor(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Widens `[s, e)` in `text` to `size` bytes, centred, cut at word boundaries.
fn widen(text: &str, s: usize, e: usize, size: usize) -> (usize, usize) {
    if e - s >= size {
        return (floor(text, s), floor(text, s + size));
    }
    let need = size - (e - s);
    let mut a = s.saturating_sub(need / 2);
    let mut b = (e + (need - (s - a))).min(text.len());
    a = a.saturating_sub(size.saturating_sub(b - a));
    a = floor(text, a);
    b = floor(text, b);
    if a > 0 {
        if let Some((k, c)) = text[a..s.max(a)].char_indices().find(|(_, c)| c.is_whitespace()) {
            a += k + c.len_utf8();
        }
    }
    if b < text.len() {
        if let Some(k) = text[e.min(b)..b].rfind(char::is_whitespace) {
            b = e.min(b) + k;
        }
    }
    (a, b.max(a))
}

fn render(u: &Unit, texts: &[String], row: &Row, pres: Pres, terms: &HashSet<String>) -> Piece {
    let turn = row.turns[0];
    if row.turns.len() > 1 {
        // A session row: its best part, wherever the presentation would look.
        let (dtext, map) = doc(texts, &row.turns);
        let n = match pres {
            Pres::Snip(n) | Pres::Span(n) | Pres::SpanPartner(n, _) | Pres::Turn(n) => n,
        };
        let (a, b) = recall::snippet_range(&dtext, terms, n);
        let multi = map_range(&map, texts, a, b);
        return Piece { turn: multi.first().map_or(turn, |x| x.0), spans: Vec::new(), partner: None, multi, fact: None };
    }
    if let Some(f) = &row.fact {
        return Piece { turn, spans: Vec::new(), partner: None, multi: Vec::new(), fact: Some(f.clone()) };
    }
    let one = |(a, b): (usize, usize)| Piece { turn, spans: vec![(a, b)], partner: None, multi: Vec::new(), fact: None };
    match pres {
        Pres::Snip(n) => {
            let (a, b) = recall::snippet_range(&row.text, terms, n);
            one((row.s + a, row.s + b))
        }
        Pres::Span(n) => one(widen(&texts[turn], row.s, row.e, n)),
        Pres::Turn(n) => one(recall::snippet_range(&texts[turn], terms, n)),
        Pres::SpanPartner(n, p) => {
            let mut piece = one(widen(&texts[turn], row.s, row.e, n));
            piece.partner = u.partner_of(turn).map(|j| (j, clip_bytes(&texts[j], p).len()));
            piece
        }
    }
}

/// Greedy packing in rank order within `budget` bytes (each piece also pays
/// [`HEADER`]). `Snip` packs every hit as its own record, at most two per
/// turn, as Engram does today. The other presentations merge hits on the
/// same turn into one piece (at most three spans), drop a partner head that
/// is already shown, and skip a span mostly shown already.
fn pack(u: &Unit, texts: &[String], rows: &[Row], hits: &[Hit], pres: Pres, terms: &HashSet<String>, budget: usize) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut used = 0;
    let merge = !matches!(pres, Pres::Snip(_));
    let shown = |out: &[Piece], t: usize, s: usize, e: usize| -> usize { out.iter().flat_map(|q| q.segs()).filter(|x| x.0 == t).map(|x| x.2.min(e).saturating_sub(x.1.max(s))).sum() };
    for h in hits {
        if out.len() >= 24 || used + 60 > budget {
            break;
        }
        let row = &rows[(h.record.seq - 1) as usize];
        let mut p = render(u, texts, row, pres, terms);
        if p.fact.is_some() {
            // At most four facts per turn, no repeats.
            if out.iter().any(|q| q.fact == p.fact) || out.iter().filter(|q| q.fact.is_some() && q.turn == p.turn).count() >= 4 {
                continue;
            }
            let cost = p.text(u, texts).len() + FACT_HEADER;
            if used + cost <= budget {
                used += cost;
                out.push(p);
            }
            continue;
        }
        let segs = p.segs();
        let Some(&(t, s, e)) = segs.first() else { continue };
        if shown(&out, t, s, e) * 2 > e - s {
            continue;
        }
        if merge && p.multi.is_empty() {
            if let Some(k) = out.iter().position(|q| q.turn == p.turn && q.multi.is_empty() && q.fact.is_none()) {
                if out[k].spans.len() >= 3 {
                    continue;
                }
                let mut trial = out[k].clone();
                trial.add_span(s, e);
                let delta = trial.text(u, texts).len().saturating_sub(out[k].text(u, texts).len());
                if used + delta > budget {
                    continue;
                }
                used += delta;
                out[k] = trial;
                continue;
            }
            if let Some((j, n)) = p.partner {
                if out.iter().any(|q| q.turn == j) || shown(&out, j, 0, n) * 2 > n {
                    p.partner = None;
                }
            }
        } else if out.iter().filter(|q| q.turn == p.turn && q.fact.is_none()).count() >= 2 {
            continue;
        }
        let cost = p.text(u, texts).len() + HEADER;
        if used + cost > budget {
            continue;
        }
        used += cost;
        // A turn shown in full no longer needs to be another piece's head.
        for q in out.iter_mut().filter(|q| q.partner.is_some_and(|(j, _)| j == p.turn)) {
            let before = q.text(u, texts).len();
            q.partner = None;
            used -= before - q.text(u, texts).len();
        }
        out.push(p);
    }
    out
}

/// The key scripts/extract_facts.py gives a session: the first 16 hex digits
/// of the SHA-1 of `json.dumps([[role, text], ...])` (Python's defaults:
/// ", " between items, non-ASCII escaped as \uXXXX).
fn session_hash<'a>(turns: impl Iterator<Item = (&'a str, &'a str)>) -> String {
    fn esc(out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                ' '..='~' => out.push(c),
                _ => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        let _ = write!(out, "\\u{:04x}", u);
                    }
                }
            }
        }
        out.push('"');
    }
    let mut j = String::from("[");
    for (k, (role, text)) in turns.enumerate() {
        if k > 0 {
            j.push_str(", ");
        }
        j.push('[');
        esc(&mut j, role);
        j.push_str(", ");
        esc(&mut j, text);
        j.push(']');
    }
    j.push(']');
    let d = sha1(j.as_bytes());
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// Lowercase words separated by single spaces, padded with spaces.
fn norm(s: &str) -> String {
    let w: Vec<String> = s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect();
    format!(" {} ", w.join(" "))
}

/// The answer, normalised, when it is long enough to test for and appears
/// in an evidence turn.
fn verbatim_answer(q: &Question, texts: &[String], ids: &HashMap<&str, usize>) -> Option<String> {
    let a = norm(q.answer_text().trim_end_matches('.'));
    if a.trim().len() < 4 {
        return None;
    }
    let ev: String = q.evidence_turn_ids.iter().filter_map(|t| ids.get(t.as_str())).map(|&i| norm(&texts[i])).collect();
    ev.contains(&a).then_some(a)
}

#[derive(Default, Clone, Serialize)]
struct PackM {
    /// Some evidence turn is shown (at least 80 bytes of it, or half).
    ev: f64,
    /// Every evidence session is shown.
    all: f64,
    /// The answer string is in the context (verbatim-answer questions).
    ans: Option<f64>,
    chars: f64,
}

#[derive(Default, Clone, Serialize)]
struct RankM {
    r1: f64,
    r5: f64,
    r10: f64,
    mrr: f64,
}

#[derive(Serialize)]
struct QRow {
    qid: String,
    qtype: String,
    n_evidence: usize,
    n_ev_sessions: usize,
    verbatim: bool,
    rank: HashMap<String, RankM>,
    pack: HashMap<String, PackM>,
}

fn rank_metrics(hits: &[Hit], rows: &[Row], ev: &HashSet<usize>) -> RankM {
    let mut seen: Vec<usize> = Vec::new();
    let mut first = None;
    for h in hits {
        let row = &rows[(h.record.seq - 1) as usize];
        let key = row.turns[0];
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        if first.is_none() && row.turns.iter().any(|t| ev.contains(t)) {
            first = Some(seen.len());
        }
    }
    let at = |k: usize| first.is_some_and(|r| r <= k) as u8 as f64;
    RankM { r1: at(1), r5: at(5), r10: at(10), mrr: first.map_or(0.0, |r| 1.0 / r as f64) }
}

fn pack_metrics(pieces: &[Piece], u: &Unit, texts: &[String], ev: &HashSet<usize>, ans: Option<&str>) -> PackM {
    let shown = |t: usize| -> bool {
        let need = 80.min(texts[t].len().div_ceil(2)).max(1);
        pieces.iter().flat_map(|p| p.segs()).filter(|s| s.0 == t).map(|s| s.2 - s.1).sum::<usize>() >= need
    };
    let fact_turns: HashSet<usize> = pieces.iter().filter(|p| p.fact.is_some()).map(|p| p.turn).collect();
    let ev_shown: Vec<usize> = ev.iter().copied().filter(|&t| shown(t) || fact_turns.contains(&t)).collect();
    let sessions: HashSet<&str> = ev.iter().map(|&t| u.turns[t].session_id.as_str()).collect();
    let got: HashSet<&str> = ev_shown.iter().map(|&t| u.turns[t].session_id.as_str()).collect();
    let all_text: String = pieces.iter().map(|p| norm(&p.text(u, texts))).collect();
    PackM {
        ev: (!ev_shown.is_empty()) as u8 as f64,
        all: (got.len() == sessions.len()) as u8 as f64,
        ans: ans.map(|a| all_text.contains(a) as u8 as f64),
        chars: pieces.iter().map(|p| p.text(u, texts).len() + if p.fact.is_some() { FACT_HEADER } else { HEADER }).sum::<usize>() as f64,
    }
}

fn fmt_ts(ts: i64) -> String {
    let s = recall::fmt_time(ts);
    s.trim_end_matches(" UTC").to_string()
}

/// The context as Engram injects it: one header per record, oldest first.
fn render_context(u: &Unit, texts: &[String], pieces: &[Piece]) -> String {
    let mut ps: Vec<&Piece> = pieces.iter().collect();
    ps.sort_by_key(|p| p.turn);
    let mut s = String::new();
    for p in ps {
        let t = &u.turns[p.turn];
        let ts = t.timestamp.as_deref().map(parse_time).unwrap_or(0);
        let _ = writeln!(s, "[{} · session {}]\n{}\n", fmt_ts(ts), clip_bytes(&t.session_id, 12), p.text(u, texts));
    }
    s
}

/// The context grouped by session, sessions in time order, each with its date.
fn render_grouped(u: &Unit, texts: &[String], pieces: &[Piece]) -> String {
    let mut by: Vec<(usize, Vec<&Piece>)> = Vec::new();
    let mut ps: Vec<&Piece> = pieces.iter().collect();
    ps.sort_by_key(|p| p.turn);
    for p in ps {
        let sid = &u.turns[p.turn].session_id;
        match by.iter_mut().find(|(t, _)| &u.turns[*t].session_id == sid) {
            Some((_, v)) => v.push(p),
            None => by.push((p.turn, vec![p])),
        }
    }
    let mut s = String::new();
    for (first, v) in by {
        let t = &u.turns[first];
        let ts = t.timestamp.as_deref().map(parse_time).unwrap_or(0);
        let _ = writeln!(s, "### Conversation on {}", fmt_ts(ts));
        for p in v {
            let _ = writeln!(s, "{}", p.text(u, texts));
        }
        s.push('\n');
    }
    s
}

struct Ctxs {
    rows: Vec<Json>,
}

#[allow(clippy::too_many_arguments)]
fn run_unit(u: &Unit, e: &Arc<dyn Embedder>, lays: &[Layout], press: &[Pres], tmp: &Path, dumps: &[(String, String, usize)], ctxs: &Mutex<HashMap<String, Ctxs>>) -> Vec<QRow> {
    let texts: Vec<String> = (0..u.turns.len()).map(|i| u.turn_text(i)).collect();
    let ids: HashMap<&str, usize> = u.turns.iter().enumerate().map(|(i, t)| (t.turn_id.as_str(), i)).collect();
    let qs: Vec<&Question> = u.questions.iter().filter(|q| !q.evidence_turn_ids.is_empty() && !q.abstention).collect();
    let all_qs: Vec<&Question> = u.questions.iter().collect();
    let mut out: Vec<QRow> = qs
        .iter()
        .map(|q| {
            let ev: HashSet<usize> = q.evidence_turn_ids.iter().filter_map(|t| ids.get(t.as_str()).copied()).collect();
            QRow {
                qid: q.qid.clone(),
                qtype: q.question_type.clone(),
                n_evidence: ev.len(),
                n_ev_sessions: ev.iter().map(|&t| u.turns[t].session_id.clone()).collect::<HashSet<_>>().len(),
                verbatim: verbatim_answer(q, &texts, &ids).is_some(),
                rank: HashMap::new(),
                pack: HashMap::new(),
            }
        })
        .collect();
    let gcfgs: Vec<(&str, GraphParams)> = graph_cfgs().into_iter().filter(|(n, _)| GRAPHS.get().is_none_or(|g| g.is_empty() || g.iter().any(|x| x == n))).collect();
    for l in lays {
        let rs = rows(u, &texts, l);
        // The first row of each turn, which a fact links to.
        let mut first_row: HashMap<usize, usize> = HashMap::new();
        for (k, r) in rs.iter().enumerate() {
            if r.fact.is_none() && r.entity.is_none() {
                first_row.entry(r.turns[0]).or_insert(k);
            }
        }
        let recs: Vec<Record> = rs
            .iter()
            .enumerate()
            .map(|(k, r)| {
                let t = &u.turns[r.turns[0]];
                let kind = if r.entity.is_some() {
                    Kind::Entity
                } else if r.fact.is_some() {
                    Kind::Fact
                } else if t.role == "assistant" {
                    Kind::Answer
                } else {
                    Kind::Prompt
                };
                let mut links = Vec::new();
                if r.fact.is_some() && l.facts == Facts::Graph {
                    if let Some(&j) = first_row.get(&r.turns[0]) {
                        links.push(Link::new("from", format!("r{}", j + 1)));
                    }
                    for n in &r.entities {
                        if let Some(id) = entity_id(n) {
                            if !links.iter().any(|x: &Link| x.to == id) {
                                links.push(Link::new("about", id));
                            }
                        }
                    }
                }
                Record {
                    id: if r.entity.is_some() { entity_id(&r.text).unwrap() } else { format!("r{}", k + 1) },
                    session: if r.entity.is_some() { "-".into() } else { t.session_id.clone() },
                    kind,
                    tag: t.turn_id.clone(),
                    turn: r.turns[0] as i64,
                    seq: k as i64 + 1,
                    ts: if r.entity.is_some() { 0 } else { t.timestamp.as_deref().map(parse_time).unwrap_or(0) },
                    importance: kind.prior(),
                    files: String::new(),
                    text: r.text.clone(),
                    ctx: r.ctx.clone(),
                    links,
                }
            })
            .collect();
        let dense: Vec<&str> = rs.iter().map(|r| r.dense.as_str()).collect();
        let vecs = e.embed(&dense).expect("embedding");
        let _ = std::fs::remove_dir_all(tmp);
        std::fs::create_dir_all(tmp).unwrap();
        let store = Store::open(&tmp.join("m.strata"), e.clone(), StoreOptions { vector_index: VectorIndexKind::Flat, sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
        store.bulk_insert(&recs, vecs).unwrap();
        let hybrid = RankParams::default();
        let n = 100;
        for q in &all_qs {
            let terms = strata::text::analyze_query(&q.question);
            let tset: HashSet<String> = terms.iter().cloned().collect();
            let bm = if terms.is_empty() { Vec::new() } else { store.bm25_top(&q.question, n, &Filter::default()).unwrap_or_default() };
            let qv = e.embed_query(&q.question).expect("embedding");
            let dv = store.vector_top(&qv, n, &Filter::default()).unwrap_or_default();
            let (hits, _) = recall::fuse(&store, &q.question, &terms, bm.clone(), dv.clone(), &hybrid);
            // The hybrid ranking, and on the graph layout each graph variant.
            let mut rankings: Vec<(String, Vec<Hit>)> = vec![(l.name.to_string(), hits.clone())];
            if l.facts == Facts::Graph {
                for (gname, gp) in &gcfgs {
                    let g = recall::expand(&store, &qv, &hits, &Filter::default(), gp).unwrap_or_default();
                    rankings.push((format!("{}:{gname}", l.name), recall::merge_graph(hits.clone(), g, gp)));
                }
            }
            if let Some(qr) = out.iter_mut().find(|r| r.qid == q.qid) {
                let ev: HashSet<usize> = q.evidence_turn_ids.iter().filter_map(|t| ids.get(t.as_str()).copied()).collect();
                let ans = verbatim_answer(q, &texts, &ids);
                let only = |b: bool| -> Vec<Hit> {
                    let p = RankParams { use_bm25: b, use_vectors: !b, lambda_imp: 0.0, lambda_rec: 0.0, ..RankParams::default() };
                    recall::fuse(&store, &q.question, &terms, if b { bm.clone() } else { Vec::new() }, if b { Vec::new() } else { dv.clone() }, &p).0
                };
                qr.rank.insert(format!("{}|bm25", l.name), rank_metrics(&only(true), &rs, &ev));
                qr.rank.insert(format!("{}|dense", l.name), rank_metrics(&only(false), &rs, &ev));
                for (name, hs) in &rankings {
                    qr.rank.insert(format!("{name}|hybrid"), rank_metrics(hs, &rs, &ev));
                    for p in press {
                        for &b in &BUDGETS {
                            let pieces = pack(u, &texts, &rs, hs, *p, &tset, b);
                            qr.pack.insert(format!("{name}|{}|{b}", p.name()), pack_metrics(&pieces, u, &texts, &ev, ans.as_deref()));
                        }
                    }
                }
            }
            for (lname, pname, b) in dumps {
                let Some((_, hits)) = rankings.iter().find(|(n, _)| n == lname) else { continue };
                let Some(p) = press.iter().find(|p| &p.name() == pname) else { continue };
                let pieces = pack(u, &texts, &rs, hits, *p, &tset, *b);
                let key = format!("{lname}/{pname}/{b}");
                let row = json!({
                    "qid": q.qid, "unit": u.unit_id, "question": q.question, "question_date": q.question_date, "answer": q.answer_text(),
                    "qtype": q.question_type, "abstention": q.abstention,
                    "context": render_context(u, &texts, &pieces), "context_grouped": render_grouped(u, &texts, &pieces),
                    "chars": pieces.iter().map(|p| p.text(u, &texts).len() + HEADER).sum::<usize>(),
                });
                ctxs.lock().unwrap().entry(key).or_insert(Ctxs { rows: Vec::new() }).rows.push(row);
            }
        }
        drop(store);
        let _ = std::fs::remove_dir_all(tmp);
    }
    out
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

fn aggregate(rows: &[&QRow]) -> Json {
    let mut keys: Vec<String> = rows.first().map(|r| r.pack.keys().cloned().collect()).unwrap_or_default();
    keys.sort();
    let mut pack = serde_json::Map::new();
    for k in &keys {
        let ms: Vec<&PackM> = rows.iter().filter_map(|r| r.pack.get(k)).collect();
        let multi: Vec<&PackM> = rows.iter().filter(|r| r.n_ev_sessions > 1).filter_map(|r| r.pack.get(k)).collect();
        pack.insert(
            k.clone(),
            json!({
                "ev": mean(ms.iter().map(|m| m.ev)),
                "ans": mean(ms.iter().filter_map(|m| m.ans)),
                "all_multi": mean(multi.iter().map(|m| m.all)),
                "chars": mean(ms.iter().map(|m| m.chars)),
            }),
        );
    }
    let mut rkeys: Vec<String> = rows.first().map(|r| r.rank.keys().cloned().collect()).unwrap_or_default();
    rkeys.sort();
    let mut rank = serde_json::Map::new();
    for k in &rkeys {
        let ms: Vec<&RankM> = rows.iter().filter_map(|r| r.rank.get(k)).collect();
        rank.insert(k.clone(), json!({"r1": mean(ms.iter().map(|m| m.r1)), "r5": mean(ms.iter().map(|m| m.r5)), "r10": mean(ms.iter().map(|m| m.r10)), "mrr": mean(ms.iter().map(|m| m.mrr))}));
    }
    json!({
        "n": rows.len(),
        "n_verbatim": rows.iter().filter(|r| r.verbatim).count(),
        "n_multi": rows.iter().filter(|r| r.n_ev_sessions > 1).count(),
        "pack": pack, "rank": rank,
    })
}

fn main() {
    let cli = Cli::parse();
    let t0 = Instant::now();
    let e: Arc<dyn Embedder> = Arc::from(embed::open(&cli.embedder).expect("embedder"));
    if cli.transcript.is_some() {
        return transcript_mode(&cli, &e);
    }
    if let Some(path) = &cli.facts {
        let mut m: HashMap<String, SessionFacts> = HashMap::new();
        for line in std::fs::read_to_string(path).expect("facts").lines() {
            let v: Json = serde_json::from_str(line).expect("facts line");
            let sid = v["hash"].as_str().unwrap_or("").to_string();
            let ents = |f: &Json| -> Vec<String> { f["entities"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default() };
            let fs: SessionFacts = v["facts"].as_array().map(|a| a.iter().filter_map(|f| Some((f["turn"].as_u64()? as usize, f["fact"].as_str()?.to_string(), ents(f)))).collect()).unwrap_or_default();
            m.entry(sid).or_insert(fs);
        }
        eprintln!("facts for {} distinct sessions", m.len());
        FACTS.set(m).unwrap();
    }
    GRAPHS.set(cli.graphs.clone()).unwrap();
    let units = read_units(cli.data.as_deref().expect("--data or --transcript"), cli.every, cli.limit).expect("read data");
    let lays: Vec<Layout> = layouts()
        .into_iter()
        .filter(|l| if cli.layouts.is_empty() { l.facts == Facts::No || cli.facts.is_some() } else { cli.layouts.iter().any(|n| n == l.name) })
        .collect();
    let press = presentations();
    let dumps: Vec<(String, String, usize)> = cli
        .dump
        .iter()
        .map(|d| {
            // Layout names may contain '/', so split from the right.
            let p: Vec<&str> = d.rsplitn(3, '/').collect();
            assert_eq!(p.len(), 3, "--dump layout/presentation/budget");
            (p[2].to_string(), p[1].to_string(), p[0].parse().expect("budget"))
        })
        .collect();
    let dataset = units.first().map(|u| u.dataset.clone()).unwrap_or_default();
    eprintln!("{} units of {dataset}, {} layouts × {} presentations × {} budgets", units.len(), lays.len(), press.len(), BUDGETS.len());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let rows: Mutex<Vec<QRow>> = Mutex::new(Vec::new());
    let ctxs: Mutex<HashMap<String, Ctxs>> = Mutex::new(HashMap::new());
    std::thread::scope(|s| {
        for t in 0..cli.threads.max(1) {
            let (units, e, lays, press, next, done, rows, ctxs, dumps) = (&units, &e, &lays, &press, &next, &done, &rows, &ctxs, &dumps);
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= units.len() {
                    break;
                }
                let tmp = std::env::temp_dir().join(format!("engram-structure-{}-{t}", std::process::id()));
                let r = run_unit(&units[i], e, lays, press, &tmp, dumps, ctxs);
                rows.lock().unwrap().extend(r);
                let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                if d % 25 == 0 || d == units.len() {
                    eprintln!("  {d}/{} units, {:.0} s", units.len(), t0.elapsed().as_secs_f64());
                }
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let all: Vec<&QRow> = rows.iter().collect();
    let mut by_type = serde_json::Map::new();
    let mut types: Vec<String> = rows.iter().map(|r| r.qtype.clone()).collect::<HashSet<_>>().into_iter().collect();
    types.sort();
    for ty in &types {
        let sel: Vec<&QRow> = rows.iter().filter(|r| &r.qtype == ty).collect();
        by_type.insert(ty.clone(), aggregate(&sel));
    }
    let layout_info: Vec<Json> = lays
        .iter()
        .map(|l| json!({"name": l.name, "window": l.window, "ctx": format!("{:?}", l.ctx), "ctx_chars": l.ctx_chars, "dense_ctx": l.dense_ctx, "user_only": l.user_only, "also": l.also}))
        .collect();
    let mean_rows: serde_json::Map<String, Json> = lays
        .iter()
        .map(|l| {
            let n: usize = units.iter().map(|u| rows_count(u, l)).sum();
            (l.name.to_string(), json!(n as f64 / units.len().max(1) as f64))
        })
        .collect();
    let summary = json!({
        "dataset": dataset, "embedder": e.name(), "units": units.len(), "questions": rows.len(), "seconds": t0.elapsed().as_secs_f64(),
        "budgets": BUDGETS, "header_chars": HEADER, "layouts": layout_info, "presentations": press.iter().map(|p| p.name()).collect::<Vec<_>>(),
        "rows_per_unit": mean_rows,
        "all": aggregate(&all), "by_type": by_type,
    });
    std::fs::create_dir_all(&cli.out).unwrap();
    let base = format!("{dataset}{}", if cli.tag.is_empty() { String::new() } else { format!("_{}", cli.tag) });
    std::fs::write(cli.out.join(format!("{base}.json")), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    let mut jl = String::new();
    for r in &rows {
        jl += &serde_json::to_string(r).unwrap();
        jl.push('\n');
    }
    std::fs::write(cli.out.join(format!("{base}.rows.jsonl")), jl).unwrap();
    for (k, c) in ctxs.into_inner().unwrap() {
        let mut jl = String::new();
        for r in &c.rows {
            jl += &serde_json::to_string(r).unwrap();
            jl.push('\n');
        }
        let name = k.replace(['/', '+'], "_");
        std::fs::write(cli.out.join(format!("context_{base}_{name}.jsonl")), jl).unwrap();
    }
    eprintln!("{} questions in {:.0} s", rows.len(), t0.elapsed().as_secs_f64());
    let a = &summary["all"];
    for l in &lays {
        let r = &a["rank"][format!("{}|hybrid", l.name)];
        let mut line = format!("{:<22} rows {:>6.0} R@10 {:.3} MRR {:.3} |", l.name, summary["rows_per_unit"][l.name].as_f64().unwrap_or(0.0), r["r10"].as_f64().unwrap_or(0.0), r["mrr"].as_f64().unwrap_or(0.0));
        for p in &press {
            let k = format!("{}|{}|4000", l.name, p.name());
            let m = &a["pack"][&k];
            line += &format!(" {} ev {:.3} ans {:.3}", p.name(), m["ev"].as_f64().unwrap_or(0.0), m["ans"].as_f64().unwrap_or(0.0));
        }
        eprintln!("{line}");
    }
}

fn rows_count(u: &Unit, l: &Layout) -> usize {
    let texts: Vec<String> = (0..u.turns.len()).map(|i| u.turn_text(i)).collect();
    rows(u, &texts, l).len()
}

#[derive(serde::Deserialize)]
struct Probe {
    q: String,
    a: Vec<String>,
}

/// The probe questions about a Claude Code transcript: records as the hooks
/// would have stored them, cut and presented by each layout, and scored on
/// whether the answer pattern reaches the context.
fn transcript_mode(cli: &Cli, e: &Arc<dyn Embedder>) {
    use engram::benchdata::Turn;
    use engram::transcript as tr;
    let t0 = Instant::now();
    let events = tr::read(cli.transcript.as_deref().unwrap()).expect("transcript");
    let batches = tr::drafts(&events, &cli.project);
    let sets: Vec<(String, Vec<Probe>)> = cli
        .probes
        .iter()
        .map(|p| (p.file_stem().unwrap().to_string_lossy().to_string(), serde_json::from_str(&std::fs::read_to_string(p).expect("probes")).expect("probes json")))
        .collect();
    let lays: Vec<Layout> = layouts().into_iter().filter(|l| l.facts == Facts::No && (cli.layouts.is_empty() || cli.layouts.iter().any(|n| n == l.name))).collect();
    let press = presentations();
    let mut results = Vec::new();
    for pt in &cli.points {
        let (w, a) = pt.split_once(':').expect("window:ask");
        let (w, a): (usize, usize) = (w.parse().unwrap(), a.parse().unwrap());
        for fresh in [false, true] {
            // Stored before the question; after a compaction (not fresh),
            // what follows the window line is still in context.
            let mut turns = Vec::new();
            let mut requests = Vec::new();
            let mut prompt_of: HashMap<(String, i64), usize> = HashMap::new();
            for b in batches.iter().filter(|b| b.line < a && (fresh || b.line < w)) {
                for d in &b.drafts {
                    let idx = turns.len();
                    let role = if d.kind == Kind::Prompt { "user" } else { d.kind.as_str() };
                    turns.push(Turn { turn_id: format!("t{idx}"), session_id: b.session.clone(), role: role.into(), text: d.text.clone(), timestamp: None, image_caption: None });
                    if d.kind == Kind::Prompt {
                        prompt_of.insert((b.session.clone(), b.turn), idx);
                        requests.push(None);
                    } else {
                        requests.push(prompt_of.get(&(b.session.clone(), b.turn)).copied());
                    }
                }
            }
            let u = Unit { dataset: "transcript".into(), unit_id: pt.clone(), turns, questions: Vec::new(), requests };
            let texts: Vec<String> = (0..u.turns.len()).map(|i| u.turn_text(i)).collect();
            eprintln!("point {pt} fresh={fresh}: {} records", texts.len());
            let tmp = std::env::temp_dir().join(format!("engram-structure-tr-{}", std::process::id()));
            for l in &lays {
                let rs = rows(&u, &texts, l);
                let recs: Vec<Record> = rs
                    .iter()
                    .enumerate()
                    .map(|(k, r)| {
                        let t = &u.turns[r.turns[0]];
                        let kind = Kind::parse(&t.role).unwrap_or(Kind::Prompt);
                        Record { id: format!("r{}", k + 1), session: t.session_id.clone(), kind, tag: t.turn_id.clone(), turn: r.turns[0] as i64, seq: k as i64 + 1, ts: 0, importance: kind.prior(), files: String::new(), text: r.text.clone(), ctx: r.ctx.clone(), links: Vec::new() }
                    })
                    .collect();
                let dense: Vec<&str> = rs.iter().map(|r| r.dense.as_str()).collect();
                let vecs = e.embed(&dense).expect("embedding");
                let _ = std::fs::remove_dir_all(&tmp);
                std::fs::create_dir_all(&tmp).unwrap();
                let store = Store::open(&tmp.join("m.strata"), e.clone(), StoreOptions { vector_index: VectorIndexKind::Flat, sync: false, flush_docs: usize::MAX / 8, ..Default::default() }).unwrap();
                store.bulk_insert(&recs, vecs).unwrap();
                for (set, probes) in &sets {
                    let mut agg: HashMap<String, (usize, usize)> = HashMap::new();
                    let mut rank1 = 0usize;
                    let mut rank10 = 0usize;
                    let mut eligible = 0usize;
                    for p in probes {
                        let pats: Vec<regex::Regex> = p.a.iter().map(|x| regex::Regex::new(x).expect("regex")).collect();
                        let hit = |t: &str| pats.iter().any(|r| r.is_match(t));
                        if !texts.iter().any(|t| hit(t)) {
                            continue;
                        }
                        eligible += 1;
                        let terms = strata::text::analyze_query(&p.q);
                        let tset: HashSet<String> = terms.iter().cloned().collect();
                        let bm = if terms.is_empty() { Vec::new() } else { store.bm25_top(&p.q, 100, &Filter::default()).unwrap_or_default() };
                        let dv = store.vector_top(&e.embed_query(&p.q).expect("embedding"), 100, &Filter::default()).unwrap_or_default();
                        let (hits, _) = recall::fuse(&store, &p.q, &terms, bm, dv, &RankParams::default());
                        let first = hits.iter().position(|h| hit(&rs[(h.record.seq - 1) as usize].text));
                        rank1 += (first == Some(0)) as usize;
                        rank10 += first.is_some_and(|r| r < 10) as usize;
                        for pr in &press {
                            for &b in &BUDGETS {
                                let pieces = pack(&u, &texts, &rs, &hits, *pr, &tset, b);
                                let shown: String = pieces.iter().map(|x| x.text(&u, &texts)).collect::<Vec<_>>().join("\n");
                                let ent = agg.entry(format!("{}|{}|{b}", l.name, pr.name())).or_default();
                                ent.0 += hit(&shown) as usize;
                                ent.1 += 1;
                            }
                        }
                    }
                    let n = eligible.max(1) as f64;
                    let mut pack = serde_json::Map::new();
                    for (k, (got, _)) in agg {
                        pack.insert(k, json!(got as f64 / n));
                    }
                    results.push(json!({"point": pt, "fresh": fresh, "set": set, "layout": l.name, "eligible": eligible, "rows": rs.len(), "rank1": rank1 as f64 / n, "rank10": rank10 as f64 / n, "pack": pack}));
                }
                drop(store);
                let _ = std::fs::remove_dir_all(&tmp);
            }
        }
    }
    std::fs::create_dir_all(&cli.out).unwrap();
    let name = format!("transcript{}.json", if cli.tag.is_empty() { String::new() } else { format!("_{}", cli.tag) });
    std::fs::write(cli.out.join(&name), serde_json::to_string_pretty(&json!({"layouts": lays.iter().map(|l| l.name).collect::<Vec<_>>(), "presentations": press.iter().map(|p| p.name()).collect::<Vec<_>>(), "budgets": BUDGETS, "results": results})).unwrap()).unwrap();
    eprintln!("wrote {} in {:.0} s", cli.out.join(&name).display(), t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_hash_matches_extract_facts() {
        let hex: String = sha1(b"abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
        // hashlib.sha1(json.dumps([["user", "héllo \"q\" \\ \n\t😀 \x01 \x7f"]]).encode()).hexdigest()[:16]
        assert_eq!(session_hash([("user", "héllo \"q\" \\ \n\t😀 \u{1} \u{7f}")].into_iter()), "0904fecd007f4f0f");
    }
}
