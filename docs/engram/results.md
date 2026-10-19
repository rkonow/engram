---
title: Results
parent: Engram
nav_order: 2
---

# Engram: results
{: .no_toc }

Engram works as a long-term memory for Claude Code. Every hook event is stored without the model's help. The details a compaction throws away come back when a prompt asks for them, and a new session starts out knowing what earlier sessions decided. Its weak point is the decision to inject: the gate keeps costs down but cannot reliably tell a related record from a relevant one. The query engine added in October 2026 lets a small model plan the searches and pick what matters, for about a tenth of a cent a request. It raises LongMemEval answers from 72% to 82%, helps on three newer benchmarks, and catches requirements a request implies without naming them.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

## Summary

| requirement | met | missed |
|---|---|---|
| **Store without being asked** | Prompts, tool calls and results, errors, answers, commentary and compaction summaries are captured from hooks. A hook costs 14–26 ms, most of it starting the hook process; a durable append takes about 4.5 ms. The 26-hour session becomes 2,204 records, back-filled in 6 s (on the cloud VM). | The model's reasoning is redacted in transcripts, so the "why" survives only where it was written out. |
| **Load without being asked, only when it helps** | 13 of 15 lost details came back automatically after a compaction; 8 of 8 questions about the 26-hour history answered in live sessions, against 0 of 4 without Engram. | The gate's precision is modest: 59% against 42% for always injecting on LongMemEval, no gain on LoCoMo. |
| **Let the model ask precisely** | In a live session the model followed an injected record id with `memory_timeline` and quoted the exact command and output of a previous session. | Whether the model searches depends on the model and the prompt. |
| **Connect a request to what it implies** | The query engine plans several searches and a second call picks what matters. With its picks added, LongMemEval answers rise from 72.2% to **81.8%**; on BEAM and LongMemEval-M, over histories of 100,000 tokens to more than a million, the picks beat plain search by 4–10 points. On coding tasks that imply a requirement without naming it (a school's nut-free rule, nuts in a product), the new builds solved 12–14 of 16 runs, `main` 8. | It costs $0.0012 and about 12 s a request with Haiku, so it runs in the background. Questions that chain two facts stay hard: 17% against 71% with the whole history in context. |
| **Do better than the alternatives** | Over 208 follow-up runs per system on four projects, Engram solved 94–97% (by version), against 44% with no memory, 74% with Claude Code's own memory, 78% with grep over past transcripts, 60% with claude-mem and 86% with mem0, and as many as with the whole history in context (93%) at about a quarter of its cost. | The October changes are not measurably better than `main` overall (196–201 runs against 195); their gain is on the tasks that imply a requirement. Given the same reader, mem0's own retrieval is level with the query engine on LongMemEval-S and BEAM 100K (79.0% against 79.0%, 41.9 against 43.1), after spending $0.13–0.22 in model calls to store each history. Hindsight is ahead (88.0% and 46.7) by handing the reader 38–75 times more text, about what the whole history in context scores on LongMemEval (87.0%), after about $1 of model calls per history; with a context the size of Engram's it is level or behind (78.0%, 39.8). |
| **Write down only what holds** | On a fixed set of 12 exchanges, the note-writer's outputs keep rules, decisions and failures with their reasons and leave out open questions, the assistant's habits, stale status and secrets: 58 of 60 pass, up from 54. | Stale notes misled no system in the comparison, because the agent read the code first, so code anchors showed no effect there. |
| **Scale far beyond a context window** | At a million records (about 200 sessions of this repository's size) a recall query takes 5.7 ms at the median and the file holds 3.1 KB per record. | Finding one conversation's needle among generic filler gets harder: recall@10 0.80 at 10k records, 0.44 at 1M. |

## Claude Code with and without Engram

Everything else on this page measures what reaches the model. The direct test is Claude Code itself, with and without Engram, on work that depends on earlier sessions.

**Method.** A *history* of headless working sessions on a small Python project is recorded by Engram. Along the way the user states rules, decisions with their reasons, failed approaches, and requirements for work planned later; some end up in the code, others are only said. Then each *follow-up task* runs in a fresh headless session on a clone of the repository as the history left it, with Engram and without (no hooks, no memory tools). Both conditions have every normal tool, including files, shell and git history. A script grades each result with one check per fact, so a task is solved only when every remembered requirement holds. The same small, fast model runs everything.

**Held out.** Two projects were written by a separate agent that never saw Engram's code or design, *before* Engram's changes were tested on them: stock keeping for a hardware wholesaler (3 sessions, 13 turns) and appointment booking for a physiotherapy practice (3 sessions, 14 turns). Each has ten tasks: six need something said only in conversation, two need a convention that is in the repository, and two are controls. Engram was frozen first; each task ran five times per condition, 200 runs in all, and nothing was changed after.

![Follow-up tasks solved with and without Engram on two held-out projects](../assets/img/chart-engram-ab.png)

| tasks | without Engram | with Engram |
|---|---|---|
| need a fact said only in conversation | 20 / 60 | **60 / 60** |
| need a convention in the repository | 20 / 20 | 20 / 20 |
| controls (need nothing earlier) | 20 / 20 | 20 / 20 |
| **all** | **60 / 100** | **100 / 100** |

A permutation test that shuffles the condition labels within each task 20,000 times never reaches the observed difference, so p < 1/20,001; Fisher's exact test on the pooled counts gives p ≈ 10<sup>−14</sup>.

Without Engram, the failed checks are exactly the facts that were only said: purchase suggestions that did not order up to three times the reorder point, a count sheet not in walking order by bin, `tomllib` recommended for a client machine that runs Python 3.8, a day plan that overran the 42-character receipt printer, a wrong cancellation fee, and a calendar feed that nobody knew had already been tried and abandoned.

The 6,000-character briefing did most of the work: it reached every Engram run. The model called the memory tools in 9 of 100 runs. Engram runs took fewer steps (6.3 against 7.5, and 7.0 against 9.0), read 5–6% fewer input tokens, and cost about the same ($0.007–0.009 a run).

**How it got there.** The first A/B, on a development project (Tally, an expense-splitting library), found almost no difference: 26 of 30 against 25 of 30 (p = 0.33). Replaying its prompts showed that decisions given with a reason never reached the briefing, and the gate scored the prompts that needed them just under its threshold. That led to distilled knowledge records and the per-prompt knowledge block ([how it works](index.md#distilled-knowledge)). On Tally they took Engram to 30 of 30 (p = 0.02), but Tally was used for development, which is why the verdict rests on the held-out projects.

**Limits.** Three small projects, one history each, one small model. The tasks were written alongside the histories and aimed at facts they state; real work rarely depends on earlier sessions this often, so the 40-point gain is a gain on such tasks, not on work in general.

{: .note }
A longer A/B on multi-day scenarios was run twice. The first run (`engram/results/long/*_v1_leaky`) is kept for the record only: Claude Code's own auto-memory, which lives outside the repository, was not reset between runs, so the runs within a lane were not independent. The rerun with the isolated harness (`engram/results/long/stockroom` and `slotwise`, 240 runs) does not show Engram recovering what compaction dropped. After a hand review of the regex-graded advice answers, stockroom was solved 26/30 with compaction alone, 27/30 with Engram, 30/30 with Engram and graph recall, and 30/30 with the full history; slotwise was 30/30 in every lane. Each compacted lane missed the one conversation-only fact its own last compaction summary dropped. In both Engram lanes the recall gate skipped those prompts and no run called a memory tool, so the differences between lanes come from their summaries. `stockroom/NOTE.md` has the review.

## Memory systems compared

The A/B above compares Engram with nothing. This one puts Claude Code next to the memory systems a user could pick instead, on four projects, and compares Engram's `main` with the October changes (notes kept current, request-aware recall, code anchors and the query engine).

**Method.** Each *lane* is one memory system. It records the project's history with that system on: each block of prompts is a new headless session, as separate working days are, and in one project a teammate's commits land between sessions without the agent being told. The lane's repository and whole home directory (which holds `~/.claude`, Claude Code's own memory, plugins and each system's data) are then frozen. Every follow-up task restores that state and starts a new session; a script grades the result, one check per fact. Each task runs four times per lane. Every `claude` process (and every hook, daemon and worker it starts) sees only `PATH`, its lane's `HOME` and the API address, so lanes cannot see each other or the machine's own setup. The agent and every memory system's own model calls use Haiku. The harness is [`engram/scripts/memory_bench.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/memory_bench.py) and the competitors' setup is in [`engram/eval/competitors`](https://github.com/rkonow/engram/tree/main/engram/eval/competitors).

| lane | what the agent has |
|---|---|
| no memory | nothing: Claude Code's auto memory off, no plugins, no hooks |
| Claude Code auto memory | Claude Code's built-in memory, on as by default: the agent writes notes into `~/.claude/projects/…/memory/` and `MEMORY.md` is loaded into each session |
| past transcripts | the history's sessions as markdown files in a folder the agent is told it can grep |
| claude-mem | the [claude-mem](https://github.com/thedotmack/claude-mem) plugin (13.39.0): hooks, a model call per tool use and per turn that writes observations and summaries, SQLite and Chroma search, injected context and MCP search tools |
| mem0 | [mem0](https://github.com/mem0ai/mem0) (2.2.1) as MCP tools to add and search memories, with a system prompt asking the agent to use them; a model call extracts facts on each add |
| Engram main | Engram as on `main` |
| Engram new (default) | the October changes as shipped, with the query engine in its default mode: planner and ranker, in the background |
| Engram new, earlier ranker prompt | the same, with the longer ranker prompt that was cut back after LongMemEval ([variants](#the-query-engine)) |
| Engram new, ranker only, earlier prompt | that, with `ENGRAM_DEEP_PLAN=none`: one model call per request instead of two |
| whole history in context | the history's one long session, forked for each task: an upper bound |

![Follow-up runs solved with each memory system, per project](../assets/img/chart-engram-v2-lanes.png)

| lane | bakehouse | stockroom | slotwise | kestrel | all | needs an earlier session | $ per run, agent + memory |
|---|---|---|---|---|---|---|---|
| no memory | 31/72 | 18/40 | 16/40 | 27/56 | 92/208 (44%) | 40/156 | 0.005 + 0 |
| Claude Code auto memory | 52/72 | 30/40 | 33/40 | 39/56 | 154/208 (74%) | 102/156 | 0.006 + 0 |
| past transcripts | 48/72 | 34/40 | 32/40 | 48/56 | 162/208 (78%) | 110/156 | 0.006 + 0 |
| claude-mem | 31/72 | 33/40 | 28/40 | 33/56 | 125/208 (60%) | 73/156 | 0.007 + 0.008 |
| mem0 | 52/72 | 40/40 | 36/40 | 50/56 | 178/208 (86%) | 126/156 | 0.006 + 0.0003 |
| Engram main | 63/72 | 40/40 | 40/40 | 52/56 | 195/208 (94%) | 143/156 | 0.006 + 0 |
| Engram new (default) | 67/72 | 40/40 | 40/40 | 49/56 | 196/208 (94%) | 144/156 | 0.007 + 0.002 |
| Engram new, earlier ranker prompt | 69/72 | 40/40 | 40/40 | 52/56 | **201/208 (97%)** | **149/156** | 0.007 + 0.002 |
| Engram new, ranker only, earlier prompt | **70/72** | 40/40 | 40/40 | 51/56 | **201/208 (97%)** | **149/156** | 0.007 + 0.001 |
| whole history in context | 62/72 | 40/40 | 40/40 | 52/56 | 194/208 (93%) | 142/156 | 0.032 + 0 |

*"Needs an earlier session" leaves out the controls and the tasks whose convention is in the repository. Bakehouse has 18 tasks, kestrel 14, the others 10, each run four times per lane: 2,080 runs. Every lane's history and runs are in `engram/results/memory_bench/<project>/`, the pooled tests in `pooled.md`. The account hit its usage limit twice during these runs; the runs that failed for it were dropped and run again (`NOTE.md`).*

**Any memory helps, and Engram leads.** Every Engram lane solved more runs than any other memory system and about as many as the whole history in context or more, at about a quarter of its average cost per run. Against no memory, the shipped default solves 196 of 208 runs against 92 (p < 10<sup>−30</sup>). Of the others, mem0 comes closest (178/208; p = 0.009 against Engram `main`); it depends on the agent calling its tools, and it solved one of bakehouse's 16 runs whose requirement the task implies without naming it. Claude Code's own memory and grep over past transcripts recover about two thirds of what was only said. claude-mem was the weakest: no better than nothing on bakehouse (31 against 31 of 72), while its observer made by far the most model calls (146,000 tokens a run, against 10,000 for Engram's query engine).

**The October changes against `main`.** Three lanes ran the new code: the shipped default, the same with the ranker prompt before its last change, and that with ranker only. They solve 196, 201 and 201 of 208 runs against `main`'s 195: together 598 of 624 (96%) against 94%, Fisher p = 0.25, and the shipped default alone is level with `main`. The spread between the three new lanes (p = 0.35 for the widest pair) is this test's run-to-run noise: each lane records its own history and writes its own notes, and an agent run is not deterministic. The default lost three runs of a kestrel task by leaving out a 15-minute buffer after assessments that its notes do hold, which the other lanes kept.

Where all three new lanes beat `main` is on bakehouse's tasks that imply a requirement without naming it, and the query engine is why: 12, 13 and 14 of 16 such runs, against 8 for `main`, 10 for the whole history and 1 for mem0. One asks to enter a school's order that includes Bakewell slices; the history said months earlier that the school is nut-free, and the catalog lists nuts in the slice. The new lanes left the slice out in 4, 3 and 4 of 4 runs, saying why; `main` did in none, and Claude Code with the whole history in context in 2.

Dropping the planner cost nothing here: ranker only solves the same 201 runs as its planner lane for a third less in memory model calls ($0.0013 a run against $0.0019). On LongMemEval it loses temporal questions (below), so the planner stays on by default and `ENGRAM_DEEP_PLAN=none` is the cheaper setting.

**Stale notes misled no one.** Bakehouse's two quick questions ask for something a teammate's commit changed after the history's notes were written: the order cutoff moved from 14:00 to 13:00 and the closed day from Sunday to Monday. Every lane, `main` and no memory included, answered both correctly in all four runs: the agent read the code before answering. With the three other tasks about code that changed under a note, this is 20 of 20 runs for every lane except the whole history (17 of 20), so code anchors did not show up in these grades. Revised decisions are solved by every Engram lane (15 or 16 of 16 runs; `main` 15).

**What memory costs a run.** From the runs that logged where memory reached the agent (the fourth repeat of every lane, and all runs of the three new lanes; [`efficiency.md`](https://github.com/rkonow/engram/blob/main/engram/results/memory_bench/efficiency.md)):

| lane | memory context per run | agent input per run (p50) | memory system's model calls per run | $ per run, agent + memory |
|---|---|---|---|---|
| no memory | 0 | 124,500 tokens | 0 | 0.0048 + 0 |
| Claude Code auto memory | 1,100 chars | 148,300 | 0 | 0.0054 + 0 |
| past transcripts | 4,000 chars | 141,300 | 0 | 0.0060 + 0 |
| claude-mem | 17,000 chars | 170,800 | 145,600 tokens | 0.0067 + 0.0082 |
| mem0 | 3,200 chars | 158,200 | 4,000 tokens | 0.0058 + 0.0003 |
| Engram main | 9,800 chars | 141,700 | 0 | 0.0056 + 0 |
| Engram new (default) | 12,500 chars | 158,500 | 10,600 tokens | 0.0068 + 0.0019 |
| Engram new, ranker only, earlier prompt | 12,300 chars | 144,000 | 7,500 tokens | 0.0066 + 0.0013 |
| whole history in context | the history | 178,900 | 0 | 0.0712 + 0 |

Agent input counts every token the agent read over its steps, most of it Claude Code's own system prompt and tools, served from cache. Engram's 12,000 characters (about 3,000 tokens) are 2% of it. The whole history costs $0.016 a run while it is in the prompt cache and $0.075 when it is not, as in this fourth repeat, which ran hours after the history was recorded: about twice Engram's default ($0.0087) with the cache and nine times without. Wall times (p50 25–37 s) are not compared, because the lanes ran side by side on a 4-core container.

**The first build.** Before these four-repeat runs, the query engine as first proposed in the same change ran three repeats of the 50 tasks that existed then: 144 of 150 in the background, 139 in front of each prompt, 137 with it off (`pooled.md`). Running it before the model starts cost 10 seconds a run and solved fewer. On slotwise it lost all three runs of one task: it picked a note the distiller had written from the assistant's own unanswered questions ("these cancellation fee details are still undecided and must be settled before the fee is built"), and the agent asked instead of building. The distill prompt now leaves such questions out ([check](https://github.com/rkonow/engram/tree/main/engram/results/v2/distill_open_questions): 6 of 6 outputs carried one before, 0 of 6 after), and the build above has the fix.

## This repository's own 26-hour session

The transcript of the session that built Strata and Engram is the most realistic test available: 26.5 hours, 43.6 MB, three projects, and two compactions whose summaries keep 14,202 and 17,998 characters. Replaying it through Engram's capture gives the records the hooks would have stored; questions are asked at the point just before Engram's own work began.

Three probe sets:

- **Recurring facts (40):** decisions and results that later work kept restating.
- **Lost details (18):** errors, failed commands and measurements that no later text repeats.
- **Lost details, vague (12):** the same details, asked the way someone would who no longer remembers the identifiers.

![Probe questions about the 26-hour session, after a compaction and in a new session](../assets/img/chart-engram-probes.png)

**After a compaction**, the summary held none of the lost details. Of the 15 the model no longer had, automatic injection brought back **13**, and one `memory_search` 14. Asked vaguely, injection recovered 8 of 10.

**In a new session**, which starts with nothing, the gated injection alone answered 83% of the lost-detail probes, 58% of their vague versions and 54% of the recurring facts; search raised these to 94%, 67% and 78%. After the record-structure changes below, the gated injection answers 50 of 67 probes unprompted (42 before), and the full injection 58.

## Live Claude Code runs

Claude Code itself, headless (`claude -p`, a small model), in a scratch repository after `engram install`:

- **Run A** states two decisions and writes code. The hooks stored 12 records, including a preference picked up from "always", the commentary, the answer and an episode card.
- **Runs B and C** are new sessions with every file tool disabled, asking for the two decisions. With Engram, the session answered from the briefing; without, it said it did not know.
- **Run D** asks for a detail the briefing does not carry. The model looked up the record id named in the injected context, called `memory_timeline` on it, and quoted the exact command and output from the earlier session.

The final live test back-fills the 26-hour transcript into a fresh project (1,835 records) and asks four questions, each in a new headless session with file, shell and web tools disabled, twice with Engram and once without. **With Engram all eight answers were right**; two came from automatic injection alone, and in the other six the injection pointed the model at the right records before it searched. **Without Engram it answered none of the four.**

Two earlier versions of this test failed in instructive ways. A prompt beginning "Without reading any files or running commands" made the model skip memory search too. And "I don't know" answers were stored and later recalled as evidence that the answer was unknown; Engram now tags such answers and never injects them.

## Retrieval on long-conversation benchmarks

Each benchmark conversation becomes its own store, with one record per turn chunk. A question succeeds when its annotated evidence turn comes back.

- **LongMemEval-S:** 500 questions, each over its own history of about 115,000 tokens; the 470 with evidence are scored.
- **LoCoMo:** 10 conversations of about 600 turns and 1,536 answerable questions.

![Turn-level recall of the annotated evidence on LongMemEval-S, by method](../assets/img/chart-engram-retrieval.png)

On LongMemEval-S, hybrid search with priors puts the evidence in the top 10 for **90%** of questions, against 88% for BM25 or static vectors alone, and the default 4,000-character injection contains it for 87%. LoCoMo's short turns between two people favour exact words: BM25 alone reaches 0.651 at rank 5 and static vectors only 0.382, so the static hybrid (0.607) falls below BM25; BGE vectors reverse that (0.678 hybrid). By question type, hybrid search helps most on multi-session questions (recall@5 0.760 → 0.835 over BM25); preference questions are hardest, at 0.50.

### What a record should be

What a record should hold (its key, its value, its size, and whether a model should write it) was studied separately, by what reaches the model and by the answers a model gives from it.

![What reaches the model as the context budget grows, by record layout](../assets/img/chart-engram-budget.png)

- **Coding sessions want small keys.** In a new session a 4,000-character injection held the answer to 72% of 67 probe questions with 1,500-byte records and **85%** with 400-byte windows.
- **Context belongs in the lexical key only.** Adding the turn's request to the BM25 key helps short, elliptical turns (LoCoMo answer-in-context 74% → 83%); adding it to the vector hurts (MRR 0.664 → 0.603).
- **A session is the wrong unit** for a value: it routes well (R@10 0.985) but its best snippet holds the evidence for only 64% of questions, against 91% for turns.
- **Facts written by a model help aggregation.** A small model wrote 66,864 facts for 4,559 sessions at $0.0019 per session. Multi-session questions answered correctly rose from 40% to 60%.

![Correct answers from each context on 500 LongMemEval-S questions](../assets/img/chart-engram-answers.png)

**Does it change the answers?** A small model answered all 500 LongMemEval questions from each context, graded with LongMemEval's own prompts. Memory takes it from 7% (it can only decline) to **71–75%**; an oracle retriever reaches 92%. When the answer string was in the context, 92–94% of answers were right; when it was not, 0–4%. Reading the *whole* 170,000-token history answers 84% on a subset, against 72% for Engram's 4,000-character context, but costs $0.17 per question against $0.0007, about 240 times more.

## The query engine

The [query engine](index.md#query-understanding) has a small model turn a request into several typed searches, fuses what they return, and has a second call pick the records that matter. It was measured on all 500 LongMemEval-S questions, each against its own store, with Haiku as planner, ranker, reader and judge (LongMemEval's grading prompts), in the cloud container.

![Questions answered correctly on LongMemEval-S, by the context the reader was given](../assets/img/chart-engram-v2-answers.png)

| context given to the reader | correct | characters |
|---|---|---|
| none | 8.6% | 0 |
| Engram main: its 4,000-character recall | 72.2% | 3,527 |
| the query engine's picks alone | 79.0% | 941 |
| Engram's recall plus the query engine's picks | **81.8%** | 4,535 |
| the evidence sessions, whole (oracle) | 89.8% | 26,738 |

Adding the picks to the recall fixed 61 answers and broke 13 (McNemar's exact test, p = 1 × 10<sup>−8</sup>). The picks alone beat the recall with a quarter of its characters (57 fixed, 23 broken, p = 0.0002), and the recall still adds to them (p = 0.03). The gain is largest where an answer needs several turns:

| question type (n) | Engram main | + query engine | oracle |
|---|---|---|---|
| temporal reasoning (127) | 66.9% | **81.1%** | 93.7% |
| multi-session (121) | 52.9% | **62.8%** | 78.5% |
| single-session, assistant said it (56) | 75.0% | **96.4%** | 98.2% |
| single-session, user said it (64) | 84.4% | **93.8%** | 95.3% |
| knowledge update (72) | 86.1% | 83.3% | 91.7% |
| preference (30) | 83.3% | **90.0%** | 80.0% |
| abstention (30) | 96.7% | 96.7% | 96.7% |

**Variants.** Each row is one query engine run over the 500 questions, read twice: picks alone, and added to Engram's recall. Evidence is the share of the 470 questions with annotated evidence whose every evidence turn is among the picks.

| query engine | model calls | + recall | picks alone | all evidence in picks | $ per question |
|---|---|---|---|---|---|
| first ranker prompt | planner, ranker | 81.6% / 81.2% (two runs) | 78.2% / 80.2% | 73.6% | 0.0011 |
| **+ pick stated tastes for advice requests (default)** | planner, ranker | **81.8%** | 79.0% | **75.5%** | 0.0012 |
| + that, no age warnings, a reading for advice | planner, ranker | 79.4% | 77.2% | 72.1% | |
| ranker only (`ENGRAM_DEEP_PLAN=none`) | ranker | 80.6% | 79.4% | 73.4% | 0.0008 |
| local ranker (`ENGRAM_DEEP_RANK=local`), top 5 | none | 74.4% | 68.6% | 66.2% | 0 |

- **Stated preferences.** The first ranker prompt left out the turns where a taste was stated, because they do not look like answers: preference questions fell from 83% to 73% when its picks were added. One sentence asking it to pick what the user said about their tastes, habits, experience and plans when a request asks for suggestions or advice brings them to 90% (27 of 30) and costs nothing elsewhere (81.8% against 81.2% and 81.6%; the two runs of the first prompt show the run-to-run spread). A longer change that also stopped age warnings and asked for a reading of the user's circumstances scored lower (79.4%, p = 0.11 against the default) and was dropped.
- **The planner earns its call on time.** Without it the ranker reads the 30 best of the request's own two searches. It keeps the evidence about as often (73% against 76%) for two thirds of the cost and half the wait (6.6 s at the median against 12 s), and answers drop 1.2 points (not significant, p = 0.5), mostly on temporal questions (74.0% against 81.1%), where the planner's date-filtered searches find the turns. On the coding projects above it made no difference. Questions about when something happened are common in coding work ("what did we change before the release"), so the planner stays on by default and `ENGRAM_DEEP_PLAN=none` is the cheaper, faster setting.
- **A local ranker.** A logistic model over twelve features of each candidate (fused rank, similarity, term coverage, kind, length, recency), fitted on LongMemEval-S with five-fold cross-validation ([`local_ranker.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/local_ranker.py), [fit](https://github.com/rkonow/engram/blob/main/engram/results/v2/local_ranker/fit.txt)), takes the top 5 with no model call. On held-out LoCoMo its top 5 hold all the evidence for 53% of questions, against 48% for the fused order's top 5. Added to the recall it gives +2.2 points (p = 0.14): a free option, well short of the model ranker.

**Where the gain comes from.** The planner's searches add depth more than reach: its 27 candidates hold all of a question's evidence for 79% of the 470 questions with evidence, against 78% for plain hybrid search taken to the same depth and 62% for its top 10. What changes the answers is the ranker. It keeps 2.8 records (960 characters) that hold all the evidence for 76% of questions and some of it for 94%.

**LoCoMo.** On every fourth LoCoMo question (500, of which 388 have annotated evidence) the picks, 3.3 records and 790 characters, hold all the evidence for 68% of questions, against 57% for plain hybrid search's top 10 records and 68% for its top 28. The gain is in the questions that need more than one turn: multi-hop 13% → 28%, open-domain 15% → 31%, single-hop 73% → 82%, temporal 64% → 73% (top 10 → picks). Ranker only holds all the evidence for 66%, losing most on open-domain questions (19%). LoCoMo was measured for evidence only; the reader study's scripts are written for LongMemEval.

**Cost and time.** Planner and ranker read about 5,800 tokens and write 1,150 per question: $0.0012 with Haiku ($0.58 for 500), with prompt caching off (each call's prompt is new, and a cache write costs more than plain input). They took 12 s at the median with six questions at a time. That is why the engine runs in the background by default and its picks join the session at the next tool call.

*The first runs of the query engine, published with it, measured $0.003 and 16 s a question. Those model calls had inherited the cloud session's own memory (about 7,000 tokens of unrelated notes per call, through environment variables Claude Code passes to child processes) and wrote every prompt to the cache. Model calls now drop both variables and turn caching off; the accuracy of the first runs (81.0%) is within the spread of the clean ones.*

**A stronger ranker for hard questions.** The planner called 49% of the questions hard. On every fifth question (100), sending those to `sonnet` instead of Haiku raised the picks holding all the evidence from 66% to 72%, but the answers hardly moved: 73% → 74% from the picks alone, 77% → 76% with the recall, at about five times the cost. Haiku stays the default; `hard_model` is there for users who want the stronger ranker. *(Measured with the first ranker prompt, with the inherited memory described above.)*

## Three newer memory benchmarks

LongMemEval and LoCoMo are what memory systems usually report. Three newer benchmarks test what they leave out: facts that are later overwritten, histories ten to forty times longer, and abilities beyond looking something up. Each conversation becomes its own store and a reader (Haiku) answers from each context. Each benchmark is scored its own way: FactConsolidation by whether an accepted answer appears in the reply, BEAM by a judge model against its rubric (0, 0.5 or 1 per item, averaged), LongMemEval-M by LongMemEval's judge prompts. The converters and scorers are [`convert_new_benchmarks.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/convert_new_benchmarks.py) and [`bench_eval.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/bench_eval.py); every number below is in `engram/results/v2/benchmarks/`.

| benchmark | what it tests | questions | history per question |
|---|---|---|---|
| [MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) FactConsolidation | a numbered list of facts in which later ones overwrite earlier ones; questions about the current state, in one hop or two | 800, over 8 histories | 6,000 to 262,000 tokens |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | ten abilities over long chats: extraction, temporal reasoning, event order, contradictions, knowledge updates, multi-session reasoning, instruction and preference following, summaries, abstention | 400, 700, 700 | 100K, 500K, 1M tokens |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval)-M | LongMemEval's questions over about 500 sessions instead of 50 | every fifth: 100 | about 1.2M tokens |

The contexts: none; Engram's hybrid search, its top 10 records; the query engine's picks; and the whole history, where it fits in 600,000 characters (about 150,000 tokens).

| | no memory | hybrid search, top 10 | query engine | whole history |
|---|---|---|---|---|
| FactConsolidation, one hop | 0% | **95.5%** | 85.3% | 85.3% (up to 64K) |
| FactConsolidation, two hops | 0% | 11.5% | 16.8% | **71.3%** (up to 64K) |
| BEAM 100K | 10.4% | 36.3% | **43.2%** | 48.1% on a sample of 60 (query engine on them: 41.0%) |
| BEAM 500K | 10.2% | 33.5% | **40.2%** | does not fit |
| BEAM 1M | 9.8% | 32.2% | **35.7%** | does not fit |
| LongMemEval-M | 6% | 57% | **67%** | does not fit (oracle 90%) |
| the same 100 questions on LongMemEval-S | | 66% | **77%** | |

**Overwritten facts.** For a question one fact answers, hybrid search finds its latest version 92–98% of the time at every length. The query engine loses ten points, and the planner is why: for 43 of the 400 one-hop questions ("What is the capital of Japan?") it judged that memory was not needed, since the question reads as general knowledge, and the engine returned nothing. The benchmark's facts are deliberately counterfactual (in one history the official language of Japan is Swedish); in Engram the per-prompt recall still runs beside the engine, so such a miss costs nothing there. No other benchmark here had more than one such question. Two-hop questions chain facts ("the citizenship of X's spouse"), and the second search depends on what the first finds: one-shot retrieval answers 8–19%, the query engine 10–27%. The whole history answers 60–82% where it fits, at about 30 times the cost per question ($0.041 against $0.0014), and loses one-hop questions as it grows (100% at 6K tokens, 68% at 64K). A second search after a first reading of the picks is the step missing here.

**BEAM.** Without memory the reader is right only where it should decline (abstention), which sets a floor of 10%. Both kinds of retrieval do best where one or two messages hold the answer, and that is where the query engine gains most: information extraction (hybrid search 50% and 44% at 100K and 500K, the query engine 63% and 66%), event order (12% → 29%, 9% → 19%), temporal reasoning (31% → 39%, 29% → 36%) and preferences (58% → 64%, 57% → 66%). What neither does well is anything that needs many messages at once: at every length summaries (11–23%), contradiction resolution (14–26%) and multi-session reasoning (26–37%) stay low, and on the last the engine is no better than plain search. Retrieval also costs some abstention: with records in front of it the reader answers some questions it should decline (100% → 80% at 100K). The whole history fits 12 of the 20 100K conversations; it was read for every fourth of their questions, 60 that happen to cover five abilities, and scores 48% on them against the query engine's 41%, at $0.19 a question against $0.003, and its lead is in summaries (40% against 21%) and multi-session reasoning (38% against 25%). At a million tokens the query engine's lead narrows to 3.5 points (35.7% against 32.2%; better on 204 questions, worse on 144, sign test p = 0.002, against p < 10<sup>−4</sup> at 100K and 500K). It still gains in temporal reasoning (14% → 26%), extraction (38% → 47%) and instruction following (28% → 38%), and loses more abstention (81% → 71%). Its picks hold some of a question's evidence for 54% of questions, against 50% for plain search's top 10.

**Ten times the sessions.** On the same 100 LongMemEval questions, going from about 50 sessions to 500 costs both contexts about ten points (hybrid search 66% → 57%, query engine 77% → 67%), and the query engine keeps its lead. Its picks hold all of a question's evidence for 64% of questions, against 54% for the top 10, in 960 characters against 3,400. Over histories of about 4,900 turns a request still costs $0.0012 and 15 s at the median.

## Other memory systems on the same benchmarks

The lanes above compare memory systems inside Claude Code. Here a memory system's own retrieval is put on LongMemEval-S (every fifth question, 100) and BEAM 100K (400 questions) with the same reader and judge as Engram's contexts, and Engram's committed contexts are read again in the same run. Each system's own model is Haiku too, and each is run as vectorize-io's agent-memory-benchmark (AMB) runs it:

- [mem0](https://github.com/mem0ai/mem0) 2.2.1 stores each session (on BEAM, each message group) with a model call and returns its top 20 memories. Its default install searches by vector alone; a second run turns on its optional keyword search and entity boosts (spaCy, BM25) over the same stored memories.
- [Hindsight](https://github.com/vectorize-io/hindsight) 1.0.0, a server on Postgres, extracts facts and entities from each session with several model calls and recalls with budget `high`: up to 32,768 tokens of facts and 16,384 of source text a question on LongMemEval (12,288 and 8,192 on BEAM), reranked by a cross-encoder. A second recall from the same banks is capped at 640 tokens of facts and no source text, about the size of Engram's context.

The runner is [`competitor_contexts.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/competitor_contexts.py), and every context, answer and grade, with the method in full, is in [`engram/results/v3/competitors/`](https://github.com/rkonow/engram/tree/main/engram/results/v3/competitors) (cloud container, 4 CPUs).

| | LongMemEval-S, 100 q | BEAM 100K, 400 q | context per question, characters (LME / BEAM) | model calls to store a history | per question |
|---|---|---|---|---|---|
| Hindsight, AMB's budgets | **88.0%** [81, 94] | **46.7** [42.7, 50.7] | 169,900 / 120,900 | about $1.00 (LME), $1.20 (BEAM) | no model call; reading costs $0.011–0.015 |
| Hindsight, Engram's context size | 78.0% [70, 86] | 39.8 [35.8, 43.7] | 4,607 / 4,340 | the same | no model call |
| Engram, query engine | 79.0% [71, 87] | 43.1 [39.0, 47.1] | 4,510 / 1,604 | none | $0.0012–0.0014 |
| mem0 | 79.0% [71, 87] | 41.9 [37.8, 46.0] | 4,421 / 4,875 | $0.13 (LME), $0.22 (BEAM) | none |
| mem0 with its search extras | 78.0% [70, 86] | 41.4 [37.4, 45.5] | 4,455 / 4,913 | the same | none |
| Engram, plain search | 69.0% | 36.6 | 3,527 / 3,383 | none | none |
| whole history in context | 87.0% [80, 93] | – | 497,600 / – | none | $0.17 to read |
| only the sessions that hold the answer | 88.0% [81, 94] | – | 27,500 / – | none | |

*95% bootstrap intervals. Paired against the query engine on the same questions: Hindsight at AMB's budgets +9.0 points on LongMemEval (+2 to +16, McNemar p = 0.02) and +3.7 on BEAM (−0.4 to +7.8, sign-flip p = 0.09); at Engram's context size −1.0 (p = 1) and −3.3 (−7.3 to +0.7, p = 0.11); mem0 ±0 (18 questions differ, 9 each way, p = 1) and −1.2 (−5.3 to +3.0, p = 0.58); mem0 with its extras −1.0 (p = 1) and −1.6 (p = 0.46); the whole history +8.0 (p = 0.06). Hindsight's model costs are its own token estimates priced at Haiku's list prices.*

**Hindsight is ahead because it hands the reader 38 to 75 times more text.** At AMB's budgets its recall returns 170,000 characters a question on LongMemEval and 121,000 on BEAM, against the query engine's 4,500 and 1,600. On LongMemEval that reaches this reader's ceiling: the whole history in context scores 87%, and only the sessions that hold the answer 88%. Hindsight gains most where an answer is spread over many messages: LongMemEval's multi-session questions (72% against 56%) and BEAM's summaries (37.5 against 17.0), where the whole history in context also leads (above). It loses abstention (62.5 against 73.8 on BEAM): with that much in front of it, the reader answers questions it should decline. Given a context the size of Engram's it is level on LongMemEval (78% against 79%) and behind on BEAM (39.8 against 43.1, not significant), and those gains are gone. Its facts are paid for at write time: 26,170 model calls for the 100 LongMemEval histories, about $100, eight times mem0's, and a recall took about 30 s on this container's CPUs, most of it in the cross-encoder.

**mem0 is level with the query engine, and pays at write time.** Its extraction call per session turns a history into short dated facts ahead of time, where Engram stores the raw records and spends its model calls on each question instead: mem0 spent $0.13 per LongMemEval history and $0.22 per BEAM conversation, against Engram's $0 to store and $0.0012–0.0014 a question. Which costs less depends on how many questions a history gets: on BEAM, at 20 a conversation, mem0 spends $0.011 a question. By ability, mem0 is better on BEAM's knowledge updates (56 against 40), preferences (72 against 66) and summaries (24 against 17), the kinds of question that facts rewritten at write time should help; the query engine is better on abstention (74 against 58), instruction following (54 against 43) and extraction (64 against 60), which depend more on the exact message. Turning on its optional search extras does not change that: 78% and 41.4.

## The note-writer

After each turn a small model writes down what the turn settled: decisions with their reasons, rules, requirements, facts, failed approaches. A note it should not have written misleads every later session; the first build's slow mode lost a task to one (above). Twelve exchanges in [`eval/distill/cases.jsonl`](https://github.com/rkonow/engram/blob/main/engram/eval/distill/cases.jsonl) each say what correct notes must and must not contain:

- kept: a rule with its reason, a decision with its reason, a change that replaces a noted rule, a fact about the environment, an approach that failed and why, planned work the exchange finished (marked done);
- left out: the assistant's unanswered questions, its own working habits, a suggestion the user turned down, status that will soon be stale, small talk, and any secret.

`engram-bench distill` runs each exchange five times with Haiku and [`distill_eval.py`](https://github.com/rkonow/engram/blob/main/engram/scripts/distill_eval.py) checks every output ([results](https://github.com/rkonow/engram/tree/main/engram/results/distill)).

| prompt | outputs passing | cost of the 60 |
|---|---|---|
| before | 54 / 60 | $0.035 |
| after | **58 / 60** | $0.032 |

Before, the note-writer recorded which branch the work had been pushed to in 4 of 5 outputs, once wrote the assistant's commit-message habit as a project rule, and once missed that planned work was done. The prompt now says that only the user, the team or a client make a project rule, and names stale status and the assistant's own habits ("as usual I'll …") among what to leave out. Two failures remain: one more note naming the branch, and one decision that says the user rejected Redis caching, which the strict check counts as keeping the suggestion.

## The gate

Compared with injecting the full budget on every prompt, the gate fitted on LongMemEval:

| | always inject | gate |
|---|---|---|
| prompts with an injection | 100% | 55% |
| characters per prompt | ~3,000 | ~930 |
| injections that contain evidence (precision) | 42% | 59% |
| questions whose evidence reaches the model | 87% | 67% |

After the move to 400-byte records and a refit, the gate brings evidence to 77% of LongMemEval questions and 65% of its injections hold evidence, for about 1,000 characters per prompt. On held-out LoCoMo it saves the same tokens but loses precision. Its features (similarity and term coverage) cannot tell "related" from "relevant" well; a cross-encoder reranker on the candidates that pass it is the next step.

## Scale and latency

A store is filled with LongMemEval turns: 94 needle questions whose evidence sessions are kept, and generic filler to reach each size, shuffled. Hooks run through the real `engram` binary and daemon, one prompt every 200 ms.

| records | recall query p50 / p99 | full hook p50 / p99 | append p99 | bytes per record | needles in top 10 |
|---|---|---|---|---|---|
| 10,000 | 1.3 / 3.3 ms | 14.4 / 17.2 ms | 6.2 ms | 3.1 KB | 0.80 |
| 100,000 | 1.7 / 5.0 ms | 15.6 / 19.8 ms | 5.6 ms | 3.1 KB | 0.65 |
| 1,000,000 | 5.7 / 16 ms | 26.1 / 42.7 ms | 8.0 ms | 3.1 KB | 0.44 |

*Source: `engram/results/scale/scale_static.json`, on the M2 MacBook Air. A million records is about 3.1 GB, mostly 512-dimension vectors. The static embedder processes 48,000–52,000 records a second. Records are cut at 400 characters, so the needles' evidence spans 8,762 records. A full hook is mostly process start (about 10 ms on macOS, against 3 ms on Linux), and an append is a durable commit, which on macOS flushes the drive's write cache (about 4.5 ms).*

Later work removed the biggest remaining cost, which was not the search at all. A write used to make the next query rebuild an HNSW graph over the last memtable chunk (12 ms); memtable segments now scan their vectors instead. A query right after a write fell from 18 to **6.4 ms** at a million records (p99 45 → 16 ms), and the prompt hook, with the knowledge query running beside the main search, takes **24.9 ms** at the median and 38.6 ms at p99.

**What the October changes cost.** `main` (787387d) and the final build of the October changes were compiled in separate checkouts and run one at a time in the cloud container (Intel Xeon 2.10 GHz, 4 cores, 15 GiB, Linux), each run starting only when no build, benchmark or model call was running and the 1-minute load average was under 1.0. Distillation and the query engine were off, as in every timing here. One scale run per side:

| records | recall query p50 / p99, main → changed | full hook p50 / p99, main → changed | needles in top 10 |
|---|---|---|---|
| 10,000 | 2.54 / 4.2 → 2.09 / 3.6 ms | 6.8 / 12.1 → 6.6 / 11.4 ms | 0.80 → 0.80 |
| 100,000 | 2.97 / 5.0 → 3.32 / 4.8 ms | 8.1 / 12.4 → 8.1 / 17.9 ms | 0.65 → 0.65 |
| 1,000,000 | 14.7 / 34.5 → 11.2 / 22.6 ms | 19.4 / 39.4 → 16.9 / 30.1 ms | 0.44 → 0.44 |

At a million records the latency benchmark then ran three times per side, alternating. The prompt hook's median is 16.0–17.7 ms on `main` and 15.8–17.5 ms changed (16.9 against 16.8 on average), its p99 27–40 against 30–35 ms; appends (p50 0.6–0.8 ms) and a query right after a write (11–15 ms) overlap as well. The changes cost nothing measurable here. An earlier comparison on the M2 MacBook Air, with a build from before the last query engine changes, found the prompt hook about 1 ms (4%) slower at the median in each of three pairs; the container's run-to-run spread, about 2 ms, is larger than that. Against the M2 numbers above, the container searches more slowly (a recall query at a million records takes 11–15 ms against 5.7) and starts hook processes faster (3 ms against 10), and its appends do not wait for a drive cache flush (0.7 ms against 4.5). The files are in `engram/results/v2/cloud/` and `engram/results/v2/m2/`, with the machine's load before and after each run.

**Concurrent reads, one writer.** Reads now run side by side on a snapshot and never wait for a write: one writer thread commits everything queued as one transaction and builds the new in-memory segment before the write returns, instead of the next query building it while other readers queue behind. A lone prompt also splits its BM25, vector and knowledge searches across the cores. Measured in the same container on 2026-10-19, `main` (01f9562) against the change, two runs per side, alternating; prompt clients send prompt hooks back to back to one daemon while another client sends a tool hook every 10 ms:

| records, prompt clients | prompt p50, main → changed | prompt p99, main → changed | prompts per second, main → changed |
|---|---|---|---|
| 100,000, 1 | 8.2–8.6 → 4.8–5.0 ms | 15.1–18.8 → 8.9–9.4 ms | 112–119 → 192–200 |
| 100,000, 4 | 24.1–26.5 → 11.7–12.8 ms | 68.4–92.1 → 27.1–30.9 ms | 134–148 → 292–321 |
| 1,000,000, 1 | 16.9–17.5 → 11.0–11.3 ms | 38.1–38.2 → 27.5–29.6 ms | 54–56 → 82–84 |
| 1,000,000, 4 | 33.2–41.5 → 25.3–25.5 ms | 81.0–120.1 → 55.6–56.9 ms | 87–112 → 146–149 |

One prompt at a time at a million records, the prompt hook (process start included) fell from 18.7–19.3 to 14.0–14.1 ms at the median. The cost is on the write side: an append returns after 2.6–3.2 ms instead of 0.9–1.0, because it waits for the new version to be built; the daemon stores records after the hook has answered, so no hook waits for that. With four prompt clients at 100,000 records the tool hook's own p99 rose from 8 to 12–13 ms while the daemon answered twice as many prompts. Every table and the variants that were tried and dropped are in [`engram/results/concurrency/`](https://github.com/rkonow/engram/blob/main/engram/results/concurrency/README.md).

Quality is what degrades with size: the filler here is chat on every subject, while real project history is more homogeneous, so this is a stress test rather than a forecast. Strata's filters (session, time, file, kind) are what keep large stores precise, and `memory_sql` gives the model those filters directly.

## Reproducing the recorded numbers

Before anything changed, every committed benchmark that needs no session transcript was rerun on `main` (787387d) on the cloud VM; [`engram/results/v2/baseline/REPORT.md`](https://github.com/rkonow/engram/blob/main/engram/results/v2/baseline/REPORT.md) has every table. Retrieval, the gate and the record-structure study reproduced exactly or within 0.007 (the docs' own MacBook rerun matches to the last digit; the committed `_c400` files are slightly older than the code), and the scale benchmark's quality numbers were identical at all three sizes. The reader study reproduces within 0.006 when the same model judges, but `haiku` now resolves to a newer model than the recorded runs used, which answers better: 8.6% / 72.2% / 89.8% with no memory, Engram's context and the oracle, against 11.0% / 68.2% / 81.0% recorded. New numbers on this page are compared with these reruns, not with the recorded runs.

On the changed build, the retrieval benchmarks return the same ranking as `main` for every question of both datasets: the October changes leave the per-prompt recall as it was, and their effect shows only where notes, anchors and the query engine are used.

## Limitations

- **The benchmarks are chats, not coding sessions.** LongMemEval, LoCoMo, BEAM and MemoryAgentBench test assistant memory over conversations; the coding evidence is one long session, a handful of live runs and six small projects.
- **The session probes were written after seeing the answers.** They favour distinctive terms, which is why the vague set exists.
- **The gate was fitted on chat data,** and its weights depend on the embedder.
- **The static embedder trades quality for speed.** BGE is better on paraphrases but costs about 50 ms per record.
- **Redaction is pattern-based.** It catches common key and token formats, not every secret.
- **One small model everywhere.** In the memory-system comparison the agent, every memory system's own calls, and the query engine are all Haiku; the readers and judges on every benchmark are too. A stronger agent may need memory less, or use it better.
- **Lanes of the same code differ by a few runs.** Each lane records its own history and writes its own notes, so three lanes of the new code solved 196 to 201 of 208 runs. Differences of that size between lanes are noise, which is why the comparison pools tasks and reports tests within them.
- **One of the four comparison projects was written alongside the changes.** Bakehouse was written to exercise what they target (revised decisions, code that changed under a note, requirements a task implies); stockroom, slotwise and kestrel are older. Results are reported per project for that reason.
- **The competitors ran close to their defaults, with two changes this container forced.** mem0 searches with vectors only (MiniLM), because its optional keyword search and lemmatiser could not be downloaded at the time (turning them on later changed its LongMemEval-S and BEAM 100K scores by a point or less), and calls Haiku through the `claude` CLI; claude-mem's Stop hook was made synchronous, because Claude Code cancels asynchronous Stop hooks when a headless session exits. Neither was tuned.
- **Model-calling runs shared a 4-core container.** Their wall times include load from the runs beside them, so only costs, steps and grades are compared across lanes, and the query engine's latency is quoted with how many requests ran at once.
- **The newer benchmarks ran an intermediate build.** Its ranker prompt was later cut back (79.4% against 81.8% on LongMemEval-S), and a planner search that names a stored column instead of the full-text index still failed (fixed since, about 1.5% of questions), so their query engine numbers are, if anything, slightly low.

## Next steps

- Two-hop questions. When the answer to one search names what to search next ("the citizenship of X's spouse"), a second round of searches from the first picks is the missing step: FactConsolidation's two-hop questions are at 17% with the query engine and 71% with the whole history.
- A planner that knows memory can contradict the world. It skipped memory for questions that read as general knowledge (43 of FactConsolidation's 400 one-hop questions); a project's own conventions can do the same.
- Answers that need many records at once. BEAM's summaries, contradiction resolution and multi-session reasoning stay at 12–38% with any retrieval, Hindsight's included.
- A stronger local ranker. The logistic ranker keeps about a quarter of the model ranker's gain for free; a small cross-encoder over the 30 candidates might keep most of it.
- Two-speed embeddings: capture with the static model, re-embed with BGE when the machine is idle.
- Hooks over HTTP to a localhost daemon, saving the process start per event: about 10 ms on macOS, 3 ms on Linux.
- Shared project memory: one store per repository, synced between teammates' agents.

The long-form reports have every chart and table: [Engram](../reports/engram.html) and [record structure](../reports/structure.html).
