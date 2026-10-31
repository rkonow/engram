//! Keeps the written-down knowledge small as a project grows: notes about
//! the same thing are merged, now and then, into fewer notes that keep
//! every detail still valid, and what changed wins over what it replaced.
//!
//! Distillation adds a note or two per turn and replaces only the notes it
//! is shown, so over weeks the same requirement appears in four wordings
//! and an old value lingers next to its replacement. Consolidation groups
//! the live notes by meaning (vector similarity, or a shared entity and a
//! looser similarity), and a model rewrites each group as edits: notes kept
//! as they are, and new notes that replace a listed set. Replaced notes are
//! kept, marked superseded, and linked from their replacement, so the
//! history stays queryable. A merged note keeps the code anchors of the
//! notes it replaces, so it is not taken for current when the code changed.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::anchor::Anchor;
use crate::distill::{strings, DistillConfig};
use crate::llm::{self, clip};
use crate::recall::{fmt_time, KNOWLEDGE};
use crate::record::{Draft, Kind, Record};
use crate::store::{Filter, Store};
use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsolidateConfig {
    pub enabled: bool,
    /// Run after this many distilled turns since the last run…
    pub every_turns: usize,
    /// …when at least this many notes are live.
    pub min_live: usize,
    /// Notes at this cosine similarity or more are grouped…
    pub cos: f64,
    /// …and so are notes about a shared entity at this similarity or more.
    pub entity_cos: f64,
    pub max_group: usize,
    /// Groups rewritten per run.
    pub max_groups: usize,
}

impl Default for ConsolidateConfig {
    fn default() -> Self {
        ConsolidateConfig { enabled: true, every_turns: 8, min_live: 30, cos: 0.80, entity_cos: 0.55, max_group: 10, max_groups: 8 }
    }
}

impl ConsolidateConfig {
    /// Enabled, and not switched off with `ENGRAM_CONSOLIDATE=0`.
    pub fn active(&self) -> bool {
        self.enabled && std::env::var("ENGRAM_CONSOLIDATE").map_or(true, |v| v != "0")
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub live_before: usize,
    pub groups: usize,
    pub replaced: usize,
    pub written: usize,
    pub cost_usd: f64,
    pub usage: llm::Usage,
    pub errors: Vec<String>,
}

const SYSTEM: &str = "You maintain the long-term notes of a software project. Output JSON lines only.";

const PROMPT: &str = r#"These notes from a software project's memory are about related things. They are listed oldest first, with their ids and when each was written:

{notes}

Rewrite them as the smallest set of notes that keeps every detail, number, name, format and reason that still holds:
- when a later note changes, reverses or replaces an earlier one, keep only the current version, and say briefly what it replaced if that helps (e.g. "30 days, changed from 14");
- merge notes that say the same thing or parts of one thing; keep notes on different subjects separate;
- never drop a rule, a requirement for planned work, a constraint or a reason;
- each note is one or two self-contained sentences that name their subject;
- a note that merges with nothing stays as it is: keep it rather than rewording it.

Output one JSON object per line:
{"keep": "<id>"} for a note that stays exactly as it is;
{"kind": "decision" | "rule" | "requirement" | "fact", "text": "<note>", "entities": ["<entity>", ...], "when": ["<situation in which it matters>", ...], "replaces": ["<id>", ...]} for a new note that replaces the listed ones.
Every id above must appear exactly once: in a keep line or in one replaces list."#;

fn label(r: &Record) -> &'static str {
    match r.kind {
        Kind::Todo => "requirement",
        Kind::Preference => "rule",
        Kind::Decision => "decision",
        _ => "fact",
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// The live notes: not superseded or done, requirements still planned.
pub fn live_notes(store: &Store, limit: usize) -> Result<Vec<Record>> {
    Ok(store
        .latest(&KNOWLEDGE, limit, &Filter::default())?
        .into_iter()
        .filter(|r| r.tag != "superseded" && r.tag != "done" && r.tag != "cue" && (r.kind != Kind::Todo || r.tag == "planned"))
        .collect())
}

/// Groups of related notes (indices into `notes`), largest first, each of
/// at least two and at most `max_group`.
pub fn groups(store: &Store, notes: &[Record], cfg: &ConsolidateConfig) -> Vec<Vec<usize>> {
    let texts: Vec<String> = notes.iter().map(|r| r.key_text()).collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let Ok(vecs) = store.embedder.embed(&refs) else { return Vec::new() };
    let ents: Vec<HashSet<&str>> = notes.iter().map(|r| r.linked("about").collect()).collect();
    let n = notes.len();
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let c = cosine(&vecs[i], &vecs[j]);
            if c >= cfg.cos || (c >= cfg.entity_cos && !ents[i].is_disjoint(&ents[j])) {
                edges.push((c, i, j));
            }
        }
    }
    // Strongest links first, never growing a group past max_group.
    edges.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut parent: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut x = x;
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (_, i, j) in edges {
        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
        if a != b && size[a] + size[b] <= cfg.max_group {
            parent[b] = a;
            size[a] += size[b];
        }
    }
    let mut by: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        by.entry(r).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = by.into_values().filter(|g| g.len() >= 2).collect();
    for g in &mut out {
        g.sort_by_key(|&i| notes[i].seq);
    }
    out.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    out
}

/// One group's edits: (new drafts, the ids each replaces), as the model wrote them.
pub fn parse(text: &str, ids: &[String]) -> Vec<(Draft, Vec<String>)> {
    let known: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.starts_with('{') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v.get("keep").is_some() {
            continue;
        }
        let replaces: Vec<String> = match v.get("replaces") {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        }
        .into_iter()
        .map(|s| s.trim().trim_matches(|c| c == '[' || c == ']').to_string())
        .filter(|s| known.contains(s.as_str()) && !used.contains(s))
        .collect();
        let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("").trim();
        if replaces.is_empty() || text.len() < 12 {
            continue;
        }
        let (k, tag) = match v.get("kind").and_then(|k| k.as_str()).unwrap_or("fact") {
            "decision" => (Kind::Decision, "distilled"),
            "rule" => (Kind::Preference, "distilled"),
            "requirement" => (Kind::Todo, "planned"),
            _ => (Kind::Fact, "distilled"),
        };
        used.extend(replaces.iter().cloned());
        let d = Draft::new(k, clip(&crate::redact::redact(text), 900))
            .tag(tag)
            .files(crate::capture::paths_in(text))
            .entities(strings(&v, "entities", 6, 60))
            .when(strings(&v, "when", 4, 160));
        out.push((d, replaces));
    }
    out
}

/// Merges groups of related live notes. Writes the new notes and marks the
/// ones they replace superseded.
pub fn run(store: &Store, cfg: &ConsolidateConfig, cli: &DistillConfig, now: i64) -> Result<Report> {
    let notes = live_notes(store, 2000)?;
    let mut rep = Report { live_before: notes.len(), ..Default::default() };
    let gs = groups(store, &notes, cfg);
    let model = cli.model();
    for g in gs.into_iter().take(cfg.max_groups) {
        rep.groups += 1;
        let members: Vec<&Record> = g.iter().map(|&i| &notes[i]).collect();
        let ids: Vec<String> = members.iter().map(|r| r.id.clone()).collect();
        let listed: Vec<String> = members.iter().map(|r| format!("[{}] {} ({}): {}", r.id, label(r), fmt_time(r.ts), r.text.split_whitespace().collect::<Vec<_>>().join(" "))).collect();
        let prompt = PROMPT.replace("{notes}", &listed.join("\n"));
        let reply = match llm::complete(&cli.cli(&model, cli.timeout_secs), SYSTEM, &prompt) {
            Ok(r) => r,
            Err(e) => {
                rep.errors.push(e.to_string());
                continue;
            }
        };
        rep.cost_usd += reply.cost_usd;
        rep.usage.add(reply.usage);
        for (draft, replaces) in parse(&reply.text, &ids) {
            // A note rewritten on its own only churns: keep the original.
            if replaces.len() < 2 {
                continue;
            }
            let olds: Vec<&&Record> = members.iter().filter(|r| replaces.contains(&r.id)).collect();
            // The newest replaced note's session and turn; its context and the
            // anchors of all of them.
            let newest = olds.iter().max_by_key(|r| r.seq).unwrap();
            let anchors: Vec<Anchor> = olds.iter().filter_map(|r| Anchor::parse(&r.ctx)).collect();
            let mut ctx = format!("Merged from {} notes, {}", olds.len(), fmt_time(now));
            if !anchors.is_empty() {
                ctx += "\n";
                ctx += &Anchor::merge(&anchors).to_line();
            }
            let mut d = draft.ctx(ctx);
            d.importance = Some(olds.iter().map(|r| r.importance).fold(0.0, f64::max));
            for id in &replaces {
                d = d.link("supersedes", id.clone());
            }
            store.append(&newest.session, newest.turn, now, vec![d])?;
            for id in &replaces {
                let _ = store.retag(id, "superseded");
            }
            rep.replaced += replaces.len();
            rep.written += 1;
        }
    }
    store.warm();
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;
    use crate::store::StoreOptions;
    use std::sync::Arc;

    #[test]
    fn edits_are_parsed() {
        let ids = vec!["a:1".to_string(), "a:2".to_string(), "a:3".to_string()];
        let text = "{\"keep\": \"a:3\"}\n{\"kind\": \"rule\", \"text\": \"Refunds are allowed within 30 days (changed from 14).\", \"replaces\": [\"a:1\", \"[a:2]\", \"zz:9\"], \"when\": [\"refund code\"]}\n{\"kind\": \"fact\", \"text\": \"A second note that reuses a:1.\", \"replaces\": [\"a:1\"]}";
        let p = parse(text, &ids);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].1, vec!["a:1".to_string(), "a:2".to_string()]);
        assert_eq!(p[0].0.kind, Kind::Preference);
        assert_eq!(p[0].0.when, vec!["refund code".to_string()]);
    }

    #[test]
    fn related_notes_are_grouped() {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(&dir.path().join("m.strata"), Arc::new(HashEmbedder { dim: 256 }), StoreOptions { sync: false, ..Default::default() }).unwrap();
        let notes = [
            ("Refunds are allowed within 14 days of purchase", "refund"),
            ("Refunds are allowed within 30 days of purchase now", "refund"),
            ("The export file uses semicolons as separators", "export"),
            ("Deploys happen on Tuesdays only", "deploy"),
        ];
        for (t, e) in notes {
            s.append("a", 1, 100, vec![Draft::new(Kind::Preference, t).tag("distilled").entities(vec![e.into()])]).unwrap();
        }
        let live = live_notes(&s, 100).unwrap();
        assert_eq!(live.len(), 4);
        let g = groups(&s, &live, &ConsolidateConfig { cos: 0.99, entity_cos: 0.2, ..Default::default() });
        assert_eq!(g.len(), 1, "{g:?}");
        let texts: Vec<&str> = g[0].iter().map(|&i| live[i].text.as_str()).collect();
        assert!(texts.iter().all(|t| t.starts_with("Refunds")), "{texts:?}");
    }
}
