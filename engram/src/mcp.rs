//! `engram mcp`: a Model Context Protocol server over stdio (newline-
//! delimited JSON-RPC 2.0) exposing the memory tools. Each call is forwarded
//! to the project's daemon, on its own thread: a quick search is answered
//! while a slow `memory_ask` is still running.

use std::io::{BufRead, Write};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value as Json};

use crate::{daemon, paths};

const INSTRUCTIONS: &str = "Engram is this project's long-term memory: every earlier prompt, answer, tool call, error and compaction summary, across sessions. \
Relevant records are injected automatically when a prompt clearly matches them; these tools are for everything else. \
memory_ask is the way to answer a question about earlier work (what was decided, why, where something happens and why, which error, which numbers): it plans several searches, runs them and picks the records that answer it. \
memory_search is one quick hybrid search. \
memory_timeline shows what happened around a record id. memory_sql answers structured questions (counts, by file, by kind, by time). \
memory_remember stores a decision, preference, fact or open task worth keeping beyond this session, ideally one sentence that will make sense later.";

fn tools() -> Json {
    json!([
        {
            "name": "memory_ask",
            "description": "Answer a question about this project's earlier work from memory. A model rewrites the question into several searches (keyword, semantic, notes, entity graph, file, SQL), all of them run, and a second model call picks the records that bear on it and says why, with a short answer that cites record ids. Use it for questions with several parts or that need reasoning (\"where does X happen and why is Y responsible for it?\", \"what constrains how we build Z?\") and before design decisions. Takes several seconds.",
            "inputSchema": {"type": "object", "properties": {
                "question": {"type": "string", "description": "The question or task, in natural language."},
                "model": {"type": "string", "description": "Model for planning and ranking (default: the configured one, usually haiku); a stronger model for hard questions, e.g. sonnet."},
                "include_current": {"type": "boolean", "default": false}
            }, "required": ["question"]}
        },
        {
            "name": "memory_search",
            "description": "Search this project's history (earlier prompts, answers, tool calls and results, errors, compaction summaries, remembered decisions) with hybrid BM25 + vector search. Returns record ids with time, session, turn, kind and a snippet. Records already in the current context window are excluded unless include_current is true.",
            "inputSchema": {"type": "object", "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language or keywords."},
                "kinds": {"type": "array", "items": {"type": "string", "enum": ["prompt", "answer", "tool_call", "tool_result", "error", "episode", "summary", "decision", "preference", "fact", "todo", "note"]}, "description": "Only these record kinds."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "since_days": {"type": "number", "description": "Only records from the last N days."},
                "include_current": {"type": "boolean", "default": false}
            }, "required": ["query"]}
        },
        {
            "name": "memory_timeline",
            "description": "The records just before and after a record id in its session, with full text: what was asked, what was run, what came out.",
            "inputSchema": {"type": "object", "properties": {
                "id": {"type": "string"}, "before": {"type": "integer", "default": 8}, "after": {"type": "integer", "default": 8}
            }, "required": ["id"]}
        },
        {
            "name": "memory_sql",
            "description": "Read-only SQL over the table memory(id, session, kind, tag, turn, seq, ts, importance, files, text, ctx, links). Full-text search uses MATCH(body, 'a AND b') or ORDER BY bm25(body, 'words') DESC; paths use MATCH(files, 'exec AND rs'); ts is Unix seconds; kind is one of prompt, answer, tool_call, tool_result, error, episode, summary, decision, preference, fact, todo, note, entity (a thing notes are about, id 'e:<name>' or 'f:<path>'); links holds a note's edges (about an entity, from its turn, supersedes an older note) and REACHABLE(links, 'e:count sheet', hops => 1, dir => 'in') finds the records about an entity; tag holds the tool name or summary section. Supports WHERE with AND/OR/NOT/IN/BETWEEN, ORDER BY a column or score, GROUP BY with COUNT(*), LIMIT. Example: SELECT ts, text FROM memory WHERE kind = 'error' AND MATCH(files, 'planner') ORDER BY ts DESC LIMIT 5",
            "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        },
        {
            "name": "memory_remember",
            "description": "Store a durable decision, preference, fact, open task or note for future sessions. Use a self-contained sentence (what, why, where). Use supersedes with an older record id when this replaces it.",
            "inputSchema": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["decision", "preference", "fact", "todo", "note"]},
                "text": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "supersedes": {"type": "string"}
            }, "required": ["kind", "text"]}
        },
        {
            "name": "memory_forget",
            "description": "Delete a record that is wrong or must not be kept (for example a leaked secret).",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
        }
    ])
}

fn call(dir: &std::path::Path, project: &str, name: &str, args: &Json) -> (String, bool) {
    let op = match name {
        "memory_ask" => "ask",
        "memory_search" => "search",
        "memory_timeline" => "timeline",
        "memory_sql" => "sql",
        "memory_remember" => "remember",
        "memory_forget" => "forget",
        _ => return (format!("unknown tool {name}"), true),
    };
    let mut req = args.clone();
    if !req.is_object() {
        req = json!({});
    }
    req["op"] = json!(op);
    let wait = Duration::from_secs(if op == "ask" { 300 } else { 20 });
    match daemon::request(dir, project, &req, wait) {
        Ok(r) => match r.get("error") {
            Some(e) => (e.as_str().unwrap_or("error").to_string(), true),
            None => (r.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(), false),
        },
        Err(e) => (e.to_string(), true),
    }
}

pub fn run() -> i32 {
    let project = paths::project_dir(None);
    let dir = paths::store_dir(&project);
    let proj = project.to_string_lossy().to_string();
    serve(std::io::stdin().lock(), &Mutex::new(std::io::stdout()), |name, args| call(&dir, &proj, name, args));
    0
}

/// Answers the requests read from `input` on `out`. Tool calls run side by
/// side, each on its own thread, and their responses (which carry their
/// request's id) are written as they finish; at the end of the input the
/// calls still running are waited for.
pub fn serve<W: Write + Send>(input: impl BufRead, out: &Mutex<W>, call: impl Fn(&str, &Json) -> (String, bool) + Sync) {
    let call = &call;
    std::thread::scope(|sc| {
        for line in input.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let Ok(msg) = serde_json::from_str::<Json>(&line) else { continue };
            let id = msg.get("id").cloned();
            let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let result = match method {
                "initialize" => {
                    let version = msg.pointer("/params/protocolVersion").and_then(|v| v.as_str()).unwrap_or("2025-06-18");
                    Some(json!({
                        "protocolVersion": version,
                        "capabilities": {"tools": {"listChanged": false}},
                        "serverInfo": {"name": "engram", "version": env!("CARGO_PKG_VERSION")},
                        "instructions": INSTRUCTIONS
                    }))
                }
                "ping" => Some(json!({})),
                "tools/list" => Some(json!({"tools": tools()})),
                "tools/call" => {
                    let name = msg.pointer("/params/name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let args = msg.pointer("/params/arguments").cloned().unwrap_or(json!({}));
                    sc.spawn(move || {
                        let (text, is_error) = call(&name, &args);
                        respond(out, id, method_result(Some(json!({"content": [{"type": "text", "text": text}], "isError": is_error})), "tools/call"));
                    });
                    continue;
                }
                _ => None,
            };
            if !respond(out, id, method_result(result, method)) {
                break;
            }
        }
    });
}

/// A result, or the error for a method that has none.
fn method_result(result: Option<Json>, method: &str) -> std::result::Result<Json, Json> {
    result.ok_or_else(|| json!({"code": -32601, "message": format!("method not found: {method}")}))
}

/// Writes one response (none for a notification); false when the output is gone.
fn respond<W: Write>(out: &Mutex<W>, id: Option<Json>, result: std::result::Result<Json, Json>) -> bool {
    let Some(id) = id else { return true };
    let resp = match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    };
    let mut out = out.lock().unwrap_or_else(|e| e.into_inner());
    writeln!(out, "{resp}").is_ok() && out.flush().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slow call does not hold up a quick one sent after it.
    #[test]
    fn tool_calls_run_side_by_side() {
        let input = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "memory_ask", "arguments": {"question": "why?"}}}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "memory_search", "arguments": {"query": "planner"}}}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 4, "method": "nope"}),
        ]
        .map(|m| m.to_string())
        .join("\n");
        let out = Mutex::new(Vec::new());
        serve(std::io::Cursor::new(input), &out, |name, args| {
            if name == "memory_ask" {
                std::thread::sleep(Duration::from_millis(300));
            }
            (format!("{name} {args}"), false)
        });
        let out = String::from_utf8(out.into_inner().unwrap()).unwrap();
        let resps: Vec<Json> = out.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let ids: Vec<i64> = resps.iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert_eq!(ids.len(), 4, "{out}");
        assert_eq!(ids.last(), Some(&1), "the slow call answers last: {out}");
        assert!(resps.iter().any(|r| r["id"] == 2 && r["result"]["content"][0]["text"].as_str().unwrap().starts_with("memory_search")));
        assert!(resps.iter().any(|r| r["id"] == 4 && r["error"]["code"] == -32601));
    }
}
