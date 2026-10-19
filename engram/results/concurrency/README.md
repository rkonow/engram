# Concurrent reads, one writer

What changed (branch `claude/concurrent-reads-single-writer-ojc9d3`):

- **Strata** can build each new version's snapshot in the writer, before a
  commit, flush, bulk load, merge or vacuum returns (`DbOptions::publish_snapshots`).
  Readers clone the published snapshot: they never build the last memtable
  chunk's in-memory segment, and never queue behind another reader building it.
- **Engram** sends every write to one writer thread. It commits whatever is
  queued as one transaction (one WAL sync), numbers new records in commit
  order, and flushes and merges between transactions. Callers embed on their
  own threads and return once their write is visible.
- The searches of one request share a snapshot. A request gets a share of the
  cores (the cores divided by the requests running when it starts), and its
  BM25, vector and knowledge searches split across that share: a lone prompt
  uses every core, two prompts get two threads each, four or more run unsplit.
- The daemon writes `state.json` outside the session lock, and the MCP server
  answers tool calls side by side.

## Method

`engram-bench concurrency --data longmemeval_s.jsonl --sizes 100000,1000000`
(defaults: 10 s per configuration, 1, 2 and 4 readers, a write every 10 ms,
1% knowledge records). A store is bulk-loaded with LongMemEval turns and
filler, as in the scale benchmark, then measured two ways:

- **daemon:** a real `engram serve` on a copy of the file. 1, 2 or 4 clients
  send `UserPromptSubmit` hooks back to back over the socket, alone or while
  another client sends a `PostToolUse` hook every 10 ms. Each prompt is also
  stored, so prompt clients write too. A prompt's time is the hook round trip
  (process start not included).
- **store, in process:** reader threads run a prompt's searches (knowledge
  and hybrid search, then packing) back to back on one `Store`, alone or while
  a writer thread appends a tool call and its result every 10 ms.

The benchmark's `prompt_searches` mirrors each side's daemon: on `main` the
knowledge search always runs on a thread of its own, on the branch both
halves go through `recall::both`. Everything else in `engram-bench` is the
same file on both sides.

Machine: the cloud container (Intel Xeon 2.10 GHz, 4 cores, 15 GiB, Linux
6.18), `rustc 1.97.0`, release builds in separate checkouts. Runs went one at
a time, each starting only when no build or benchmark was running and the
1-minute load average was under 1.0, alternating sides. Each side ran twice;
ranges below are over the two runs. `machine.txt` has the load averages
before and after every run.

- main: `01f9562` (main after PR #10), with the `concurrency` subcommand added
  to `engram-bench`
- branch: `ee4a61e`

## Through the daemon

Prompt latency (p50, p99) and prompts answered per second, main → branch.

**100,000 records**

| prompt clients | tool hooks | prompt p50 ms | prompt p99 ms | prompts/s | tool hook p99 ms |
|---|---|---|---|---|---|
| 1 | no | 7.2–7.6 → 4.4–4.5 | 15.0–16.0 → 8.1–8.4 | 124–132 → 211–220 | |
| 1 | yes | 8.2–8.6 → 4.8–5.0 | 15.1–18.8 → 8.9–9.4 | 112–119 → 192–200 | 2.8–3.3 → 2.9–3.4 |
| 2 | no | 9.7–10.1 → 6.6 | 19.9–20.7 → 12.4–13.3 | 191–198 → 292 | |
| 2 | yes | 11.4–11.5 → 7.0–7.5 | 27.5–29.7 → 13.7–15.9 | 156–161 → 254–274 | 4.2–5.2 → 6.2–7.1 |
| 4 | no | 19.1–19.4 → 10.9–11.0 | 48.8–51.7 → 26.1–26.5 | 189–192 → 339–342 | |
| 4 | yes | 24.1–26.5 → 11.7–12.8 | 68.4–92.1 → 27.1–30.9 | 134–148 → 292–321 | 7.5–7.9 → 11.6–12.9 |

**1,000,000 records**

| prompt clients | tool hooks | prompt p50 ms | prompt p99 ms | prompts/s | tool hook p99 ms |
|---|---|---|---|---|---|
| 1 | no | 14.6–15.1 → 10.5 | 33.8–34.9 → 28.4–28.8 | 62–66 → 87 | |
| 1 | yes | 16.9–17.5 → 11.0–11.3 | 38.1–38.2 → 27.5–29.6 | 54–56 → 82–84 | 2.0–4.3 → 4.3 |
| 2 | no | 16.7–18.6 → 15.1–15.2 | 35.0–36.7 → 32.7–33.4 | 102–115 → 125–127 | |
| 2 | yes | 18.9–21.7 → 16.3–17.5 | 45.8–50.5 → 33.9–37.8 | 86–99 → 107–118 | 7.3–7.6 → 6.5–8.1 |
| 4 | no | 28.7–30.6 → 23.5–24.4 | 59.6–67.2 → 52.9–55.1 | 123–133 → 154–157 | |
| 4 | yes | 33.2–41.5 → 25.3–25.5 | 81.0–120.1 → 55.6–56.9 | 87–112 → 146–149 | 10.9–15.8 → 8.2 |

On `main`, every stored prompt or tool call made the next prompt rebuild the
in-memory segment, with other readers queued behind that build. On the
branch the writer has built it before the write returns, so writes barely
move prompt latency (4 clients at 1M: p99 53–55 ms without tool hooks,
56–57 ms with them, against 60–67 and 81–120 ms on `main`).

A tool hook answers before its record is stored, on both sides. With 4
prompt clients at 100K it got slower (p50 0.4 → 1.9–2.0 ms, p99 7.5–7.9 →
11.6–12.9 ms): the daemon was answering twice as many prompts, and storing
each one, on the same 4 cores, and the writer thread takes one of them while
it builds versions. At 1M, where prompts are slower, it did not (p99 10.9–15.8
→ 8.2 ms).

## Store, in process

Read latency and reads per second, main → branch; append latency for the
writer.

**100,000 records**

| readers | writer | read p50 ms | read p99 ms | reads/s | append p50 / p99 ms |
|---|---|---|---|---|---|
| 1 | no | 3.2–3.4 → 2.7–2.8 | 5.9–6.0 → 5.1–5.6 | 285–299 → 340–349 | |
| 1 | yes | 4.0–4.2 → 2.9 | 8.4–8.6 → 5.4–6.2 | 220–227 → 319–335 | 0.9 / 2.2–3.2 → 2.8 / 6.1–6.8 |
| 2 | no | 3.3–3.4 → 3.3–3.4 | 6.1–6.2 → 6.5–7.0 | 564–576 → 559–569 | |
| 2 | yes | 3.9–4.7 → 3.6 | 8.4–11.4 → 7.3–7.7 | 391–470 → 506–519 | 1.0–1.2 / 6.9–8.0 → 3.9–4.1 / 12.3–14.4 |
| 4 | no | 5.3–7.9 → 4.9–5.0 | 10.9–16.6 → 8.6–10.0 | 474–706 → 751–773 | |
| 4 | yes | 6.7–10.7 → 5.3 | 15.4–24.1 → 10.1–10.4 | 356–558 → 703–710 | 1.0–1.1 / 8.3–9.6 → 2.7–6.0 / 9.5–11.7 |

**1,000,000 records**

| readers | writer | read p50 ms | read p99 ms | reads/s | append p50 / p99 ms |
|---|---|---|---|---|---|
| 1 | no | 12.1–12.2 → 8.2–8.4 | 29.0–30.2 → 24.2–24.8 | 79–80 → 108–112 | |
| 1 | yes | 13.1–13.3 → 8.5–8.8 | 30.2–34.8 → 26.0–27.1 | 71–72 → 102–106 | 0.9–1.0 / 2.9–3.2 → 3.0 / 8.0–10.0 |
| 2 | no | 12.1–12.2 → 11.9–12.2 | 29.4–30.4 → 30.0–31.6 | 157 → 156–158 | |
| 2 | yes | 12.8–13.1 → 12.4–12.5 | 30.5–30.9 → 30.7–31.4 | 145–147 → 152–156 | 0.9 / 7.0–7.3 → 3.2–3.3 / 10.3–11.9 |
| 4 | no | 18.3–19.5 → 17.7–18.7 | 46.1–47.0 → 40.1–42.5 | 192–202 → 201–215 | |
| 4 | yes | 21.6–22.0 → 18.6–19.1 | 46.8–48.1 → 42.6–44.2 | 172–174 → 196–203 | 0.7–0.8 / 9.4–9.7 → 2.6–2.8 / 9.5–11.2 |

With a writer, `main` loses 6–31% of its reads (the first read after a
write rebuilds the segment, and the others wait for it); the branch loses
1–11%. Two readers without a writer are level, since both sides then run
two threads per reader.

**The cost: an append returns later,** 2.6–4.1 ms at the median (6.0 ms in
one run with four busy readers) against 0.7–1.2 ms, because it now waits for its group's commit and for the new
version to be built. In the daemon, appends run after the hook has answered,
so no client waits for them.

## One client at a million records

`engram-bench latency --sizes 1000000`, the benchmark behind the scale and
latency numbers in the docs: one prompt at a time, as a hook process (the
`engram` binary, process start included) and over the socket, plus appends
with a durable commit and a query right after each one. Two runs per side.

| | main | branch |
|---|---|---|
| prompt hook, p50 / p99 | 18.7–19.3 / 31.0–37.3 ms | 14.0–14.1 / 27.5–27.9 ms |
| prompt over the socket, p50 / p99 | 14.7–14.9 / 28.0–35.1 ms | 10.3–10.4 / 27.8–30.1 ms |
| tool hook, p50 | 3.7–3.8 ms | 3.6–3.7 ms |
| hybrid search, p50 | 11.0–11.3 ms | 7.8–8.7 ms |
| query right after a write, p50 / p99 | 13.4–13.7 / 33.3–36.3 ms | 8.3–9.3 / 28.2–30.0 ms |
| durable append (256-record chunks), p50 / p99 | 0.9–1.0 / 1.8–2.7 ms | 2.6–3.2 / 4.9–6.0 ms |

The search is faster because its BM25 and vector halves now run side by side;
a query right after a write costs what any query costs, because the append
has built the in-memory segment (the benchmark's separate "rebuild" step
drops from 1.5 ms to nothing). Today's `main` prompt hook (18.7–19.3 ms) is
slower than the 16.0–17.7 ms measured on this container on 2026-10-15
(`../v2/cloud/`); both sides here ran the same day, alternating.

## Variants tried

Kept in `branch/variants/`, one run each:

| build | how a request's searches split | 100K store reads/s, 2 / 4 readers | 100K daemon prompts/s, 4 clients + tool hooks |
|---|---|---|---|
| `33e7d63` | always, three extra threads per prompt | 466 / 588 | 289 |
| `f08c352` | while fewer threads than cores were searching | 480 / 744 | 319 |
| `a91b09f` | on rayon's pool (stopped after the 100K store part) | 445 / 536 | |
| `ee4a61e` (final) | by the request's share of the cores | 559–569 / 751–773 | 292–321 |

Always splitting oversubscribed the 4 cores once several readers ran.
Counting searching threads left cores idle while a thread waited for its
other half. Rayon's pool was the slowest with every core busy.

## Files

- `main/`, `branch/`: `concurrency_r1.json`, `concurrency_r2.json` and their
  logs; `latency_1m_r1.json`, `latency_1m_r2.json` and their logs.
- `branch/variants/`: the earlier builds above.
- `machine.txt`: the machine, the builds and every run's start and end with
  load averages.
