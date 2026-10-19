# Engram: automatic long-term context for coding agents, on Strata

Engram gives a coding agent (Claude Code first) a memory that outlives its
context window. It records every prompt, answer, tool call and compaction
summary of a project into one Strata file. It brings the relevant pieces
back at the moments they matter, without the model having to ask, and gives
the model tools to search its own history when it decides to.

## 1. The problem

A long session on a complex project outgrows any context window. Claude
Code then *compacts*: it replaces the history with a summary of about
15,000 characters. Summaries keep the plan and lose the particulars:
- exact numbers, error messages and file paths;
- the reason an approach was rejected;
- a preference the user stated once, three hours earlier.

A new session starts with only `CLAUDE.md` and the auto-memory index. The
session that built Strata ran for 32 hours and wrote 43.7 MB of transcript
across two compactions; its summaries hold under 0.1% of that text.

Three requirements follow:

1. **Store without being asked.** Capture must not depend on the model
   remembering to save. Every event is stored, losslessly and cheaply.
2. **Load without being asked, but only when it helps.** Injected context
   costs tokens and attention, and irrelevant context hurts accuracy ("lost
   in the middle", context rot). Injection must be gated, budgeted and must
   skip what the model can already see.
3. **Let the model ask precisely.** When the model knows it needs history
   ("what did we decide about X?", "which commit broke Y?"), it gets search,
   timeline and SQL tools over the same store.

## 2. Architecture

```
 Claude Code ──hooks──▶ engram hook <event>  ─┐  (thin client, ~1 ms + process start)
     │                                         │ unix socket, JSON lines
     └──MCP (stdio)──▶ engram mcp ────────────┤
                                               ▼
                         engram serve (one daemon per project)
                         ├─ capture: payload → records (chunk, redact, salience)
                         ├─ embedder (static Model2Vec, or BGE-small)
                         ├─ recall: hybrid search → priors → gate → pack
                         ├─ session state: live window, injected ids
                         └─ Strata file: memory.strata (+ WAL)
```

Strata holds an exclusive lock on its file, and hooks are short-lived
processes that run concurrently with the MCP server. So one daemon per
project owns the database. Hooks and the MCP server are clients over a Unix
socket.
- **Lazy start.** The first client to find no socket spawns the daemon under
  a spawn lock, and the daemon exits after 30 idle minutes.
- **Warm state.** The embedder, segment readers and the memtable stay in
  memory between hooks. The embedding table is memory-mapped, so a daemon
  starts without reading it.
- **Reads in parallel, one writer.** Each client connection gets a thread,
  and reads run on it against a snapshot, any number at once. Every write
  goes to one writer thread, which commits whatever is queued as one
  transaction (one WAL sync), numbers new records in commit order, and
  flushes and merges between transactions. Callers embed on their own
  threads and return once their write is visible, so a session reads its
  own writes. The writer also builds each new version before the write
  returns (Strata's `publish_snapshots`), including the last memtable
  chunk's in-memory segment, so a reader never builds an index or waits for
  one. Chunks hold 256 records and scan their vectors instead of building a
  graph (Strata's `mem_build` option), so that build stays cheap. In the
  cloud container at 1M records, with four prompt clients and a tool hook
  every 10 ms, the daemon answers 146–149 prompts a second at a 56–57 ms
  p99, against 87–112 prompts and 81–120 ms when readers built the segment
  (`results/concurrency/`).
- **One snapshot per request.** The searches behind one prompt (BM25 and
  vectors, main and knowledge queries, the query engine's searches) share a
  snapshot, so they agree however much is written meanwhile. They run side
  by side while cores are free, and one after the other once every core is
  searching, where splitting a request would only slow the others. IDF for the gate comes from the segments' term dictionaries,
  not from a `COUNT` query per term.
- **Never in the way.** A client that cannot reach the daemon within its
  deadline prints nothing and exits 0. A memory failure never blocks a
  prompt or a tool call.

The store lives outside the repository, in
`${ENGRAM_HOME:-~/.local/share/engram}/<project-slug>/`, because
transcripts can contain secrets. A redaction pass removes common key and
token formats before anything is stored.

## 3. Data model

One Strata table, `memory`:

| field | type | meaning |
|---|---|---|
| `id` | KEYWORD, primary key | `<session-prefix>:<seq>` |
| `session` | KEYWORD | Claude Code session id |
| `kind` | KEYWORD | `prompt`, `answer`, `tool_call`, `tool_result`, `error`, `episode`, `summary`, `decision`, `preference`, `fact`, `todo`, `note` |
| `tag` | KEYWORD | tool name, summary section, or `-` |
| `turn` | INT | user turn within the session |
| `seq` | INT | project-wide sequence number (order of events) |
| `ts` | INT | Unix seconds |
| `importance` | FLOAT | salience prior in [0, 1] |
| `files` | TEXT | paths the record touches (path components are terms) |
| `text` | TEXT, stored | the value: one chunk of at most 400 bytes, injected as is |
| `ctx` | TEXT, stored | the turn's request (≤ 200 characters) or a summary's section: part of the lexical key, never shown |
| `body` | TEXT, generated from `text` and `ctx`, positions | the lexical key that `bm25` and `MATCH` search |
| `emb` | VECTOR(d), HNSW | the dense key: embedding of `text` alone |

Each record is one key–value pair with attributes. The keys are small and
precise: a 400-byte window matches a question better than a 1,500-byte
chunk in which the answer is one fact among many, and a context budget holds
about ten such windows instead of five snippets. The turn's request joins
the lexical key, so that a terse tool result ("test result: ok") is found by
the words of the task it served; it stays out of the vector, where it would
pull every response towards its request and crowd out the request itself.
`report/structure.html` has the comparison of layouts behind these choices.

Records come in three levels:

- **L0, events.** User prompts, assistant answers, tool calls and results.
  Text is cut into records of at most 400 bytes at line, sentence or word
  boundaries. Large results keep only their head (three units of 1,500
  bytes) and tail (one unit), then are cut the same way.
  `Read` results are stored as stubs, since the file is on disk and may
  have changed. Edits keep the new text, because the history of a file's
  changes is the valuable part.
- **L1, episodes.** One card per user turn, built at `Stop` with no model
  call: the prompt, files touched, commands run, errors seen, and the head
  of the final answer.
- **L2, knowledge.** Durable statements:
  - decisions and facts the model saves with `memory_remember`;
  - preferences extracted from user prompts by cue phrases ("always",
    "never", "don't", "from now on", "make sure", "remember");
  - Claude Code's own compaction summaries, split by section.

The kind sets the salience prior \(\iota\): preference 1.0, decision 0.9,
summary 0.8, prompt 0.7, error and episode 0.6, answer 0.5, tool call and
result 0.3.

## 4. When to store

Always. Storage is cheap (text compresses; a durable commit costs about 4 ms
on macOS and 0.3 ms on the Linux VM, off the hook's path), and a missed record
cannot be recovered later. Each hook's payload is the source:

| hook | stored |
|---|---|
| `SessionStart` | session registered; live window opened |
| `UserPromptSubmit` | `prompt`, plus `preference` records for cue sentences (stored after recall, off the critical path) |
| `PostToolUse` | `tool_call` and `tool_result` (or `error`), asynchronously |
| `Stop` | `answer` (from `last_assistant_message`), the commentary written between tool calls (read from the transcript tail since the last offset), the turn's `episode`, then the distilled knowledge (below) |
| `PreCompact` | the commentary not yet captured, before the context is summarised |
| `PostCompact` | `summary` sections; the live window restarts here |
| `SessionEnd` | flush |

`engram ingest-transcript` back-fills from Claude Code's JSONL transcripts
on a best-effort basis; the format is internal and versioned.

### Distilled knowledge

Raw records hold what was said, but a later session needs what was
*agreed*: a decision and its reason, a client's rule, the requirements of
work announced for later. Much of it never reaches the code, and a
request that depends on it rarely sounds like earlier work (a request to
make saving faster says nothing about why storage is JSON), so search
alone misses it. After each `Stop`, once the hook has answered, the
daemon gives the turn (request, commands run, reply) to a small model
through the `claude` CLI in print mode, with no tools, from a scratch
directory and with Engram disabled for that call. The model also sees the
knowledge already noted (the latest 40 items, with ids) and returns JSON
lines:

| item | stored as | tag |
|---|---|---|
| decision, with the reason | `decision` | `distilled` |
| rule, constraint, convention (kept even if the code already follows it) | `preference` | `distilled` |
| requirement of work planned but not done | `todo` | `planned` |
| fact about the project or its environment, failed approaches | `fact` | `distilled` |
| `replaces: id` on any of the above | the old item is retagged | `superseded` |
| `done: id` for a planned item completed | the item is retagged | `done` |

It is told to leave out what is already noted, status that goes stale
(test counts, branches, what is committed) and the assistant's own
working conventions. An item whose analysed terms overlap an existing
one by a Jaccard index of 0.6 or more is dropped. Distillations run one
at a time, so each sees the last one's notes, and shutdown waits for any
in flight. On the Tally history, ten turns gave 25 items at about $0.001
and 8 s each (before the prompt saw existing notes, it wrote 71, most of
them repeats). `engram distill` back-fills a store recorded before;
`ENGRAM_DISTILL=0` or `"distill": {"enabled": false}` turns it off, and
`distill.model` or `ENGRAM_DISTILL_MODEL` picks the model.

## 5. When to load

Records still inside the model's context are never injected. For each
session the daemon keeps the window start \(w_s\): the sequence number at
session start or at the last compaction. Every query excludes

\[ \{\, r : r.\text{session} = s \;\wedge\; r.\text{seq} \ge w_s \,\} \;\cup\; \text{injected}_s, \]

where \(\text{injected}_s\) is the set of records already injected since \(w_s\).

There are three push points and one pull path:

1. **`SessionStart`: briefing, always, at most 6,000 characters.** In
   order: planned work with its requirements, rules and conventions,
   decisions, facts (the latest of each, deduplicated, never superseded or
   done items), the open tasks of the latest checkpoint, and where the
   previous session left off. It opens with an instruction to search
   memory before a design decision and whenever a task refers to earlier
   work. Sections fill in that order while the budget lasts; the ids shown
   join \(\text{injected}_s\). After a compaction (`source = compact`), the
   briefing also recalls the records most relevant to the last two
   prompts, which the summary just squeezed out.
2. **`UserPromptSubmit`: knowledge, then gated recall.** First, up to four
   knowledge records (decision, preference, fact, note, planned todo) not
   yet shown that match the prompt on their own text, with
   \(\cos \ge 0.40\) or \(\mathrm{cov} \ge 0.30\), labelled as what was
   agreed before. These bypass the gate: a request can be bound by an
   earlier decision without sounding like earlier work. Then the search
   with the prompt and a three-way gate:
   - strong evidence: inject the top records, at most 4,000 characters;
   - weak evidence: inject the top three records, at most 1,200 characters,
     marked as a weak match;
   - otherwise: nothing.
3. **`PreToolUse` on `Edit`/`Write`: file notes.** Inject up to two older
   records anchored to that path (errors, decisions, preferences), once per
   file per window.
4. **Pull: MCP tools.**
   - `memory_search(query, kinds, limit)`;
   - `memory_timeline(id, before, after)`, the neighbourhood of a hit;
   - `memory_sql(select)`: read-only SQL over `memory`, so the model can
     ask questions like "the last five errors in `exec.rs`";
   - `memory_remember(kind, text, files)` and `memory_forget(id)`.

   Tool descriptions and a short `CLAUDE.md` note say when to use each,
   phrased as facts. The server answers calls side by side, so a quick
   search is not held up by a `memory_ask` still running.

### Ranking

BM25 and vector search each return their top \(n = 50\) under the exclusion
filter (two Strata queries on one snapshot). Their scores are min-max
normalised over the union \(C\), with a missing score counting as 0:

\[ \mathrm{rel}(r) = w\,\widehat{\mathrm{bm25}}(r) + (1 - w)\,\widehat{\cos}(r). \]

Priors then break near-ties without overriding relevance:

\[ S(r) = \mathrm{rel}(r)\,\bigl(1 + \lambda_\iota\,\iota(r)\bigr) + \lambda_t\, e^{-(\mathrm{seq}_{\max} - \mathrm{seq}(r))/\tau}. \]

Candidates exclude answers that admitted not knowing (tagged `uncertain`
at capture) and stored prompts that are the same question again; both match
a repeated question best and would only replay a failure. Packing is greedy
by \(S\), at most twelve records and eight per user turn (a coding turn
spans many records; a tighter cap dropped answers that were a fourth
record of a turn already shown). Records are at most 400 bytes, so each is
shown whole. Each injected
line carries its provenance: kind, session, turn, relative time and id. Up to five further
candidates follow as one line each (id and opening words), for the model to
open with `memory_timeline`.

### The gate

Min-max scores cannot gate, since the best candidate always scores 1. The
gate uses absolute evidence about the best candidates: the highest cosine
and IDF-weighted term coverage among the top five, how far the best BM25
and cosine scores stand above the tenth, whether BM25 and vectors agree on
the top record, and whether the prompt refers back:

\[ G = a\,\cos_{\max} + b\,\mathrm{cov}_{\max} + c\,\mathbb{1}_{\text{cue}} + d\,\mathbb{1}_{\text{agree}} + e\,(\cos_1 - \cos_{10}) + f\,\Bigl(1 - \frac{\mathrm{bm25}_{10}}{\mathrm{bm25}_1}\Bigr), \qquad \mathrm{cov}(q, r) = \frac{\sum_{t \in q \cap r} \mathrm{idf}(t)}{\sum_{t \in q} \mathrm{idf}(t)}. \]

Past-reference cues include "earlier", "we decided", "last time", "again"
and "remember". The full budget is injected when \(G \ge \theta_{\text{hi}}\),
and three records when \(\theta_{\text{lo}} \le G < \theta_{\text{hi}}\). The
weights come from a logistic fit on LongMemEval-S (`scripts/fit_gate.py`):
- **positives:** questions whose evidence the packed injection contains;
- **negatives:** questions asked against another conversation's store,
  abstentions, and misses.

\(\theta_{\text{hi}}\) holds precision 0.9 on the fit and \(\theta_{\text{lo}}\)
precision 0.7. The weights depend on the embedder, because cosine scales
differ.

## 6. Costs and latency targets

| path | target |
|---|---|
| `PostToolUse`, `Stop` client | < 10 ms (queued, embedded and committed in the background) |
| `UserPromptSubmit` | < 100 ms p99 at 1M records (measured on an Apple M2: 24.9 ms p50, 38.6 ms p99 as a hook process, with 1% knowledge records; the knowledge query runs beside the main search; about 10 ms of it is process start) |
| `SessionStart` | < 150 ms (daemon already up), < 1 s with daemon spawn |
| background | the writer thread commits queued writes together and builds the new in-memory segment (at most 256 records, no graph) before they return, so no query does |
| ingest | > 1,000 records per second with the static embedder |

The default embedder is static (Model2Vec potion-retrieval-32M: token
lookup and mean, 512 dimensions, microseconds per record). BGE-small (384
dimensions) is optional and roughly 50 ms per record on the cloud VM's CPU; it suits
a background re-embedding pass, not the hook path.

## 7. Evaluation plan

1. **Retrieval quality** on LongMemEval-S and LoCoMo: evidence recall@k and
   NDCG@10 for BM25, dense, hybrid, and hybrid with priors, at turn and
   round granularity, with both embedders.
2. **Context-loss probes** on this repository's 32-hour session transcript:
   questions whose answers were settled before a compaction. Compare what
   the compaction summary still contains against what Engram injects
   within a budget.
3. **Gate:** injection precision against evidence coverage across
   thresholds, and injected tokens per prompt.
4. **Systems:** hook latency by store size, ingest rate, bytes per record,
   daemon cold start.
5. **Live run:** headless `claude -p` sessions in a scratch project with
   Engram installed. A fact established in one session must be recalled in
   a fresh one, and after a compaction.
