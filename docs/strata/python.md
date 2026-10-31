---
title: Python
parent: Strata
nav_order: 5
---

# Strata from Python
{: .no_toc }

The `strata` package runs a Strata database inside a Python process, the way `sqlite3` runs SQLite, or talks to a running [strata-server](server.md). Both have the same methods, so a script written against a local file works against a server.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

## Install

The package is in [`strata-py/`](https://github.com/rkonow/strata/tree/main/strata-py): a Rust extension built with [maturin](https://www.maturin.rs), for CPython 3.10 and later. It needs Rust (stable) and a C compiler.

```sh
pip install ./strata-py                    # from the repository's root: builds and installs a wheel
```

or, while working on it:

```sh
cd strata-py
python3 -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release                  # builds the extension into the virtualenv
pytest tests                               # the client tests also need strata-server built
```

Model2Vec embedding models need nothing more. For the BERT models, build with `--features bert` (`--features accelerate` on a Mac): `maturin develop --release --features bert`, or `MATURIN_PEP517_ARGS="--features bert" pip install ./strata-py`. The extension runs on any CPU of its architecture: Strata picks its SIMD kernels (AVX2, AVX-512, the aarch64 dot product instructions) at run time. `RUSTFLAGS="-C target-cpu=native"` builds one for the building machine only.

## A database in your process

```python
import strata

db = strata.open("books.strata", schema={
    "table": "books",
    "key": "id",
    "fields": [
        {"name": "id", "type": "keyword"},
        {"name": "title", "type": "text"},
        {"name": "description", "type": "text"},
        {"name": "genres", "type": "keyword", "multi": True},
        {"name": "year", "type": "int"},
        {"name": "embedding", "type": "vector",
         "embed": {"from": ["title", "description"], "model": "potion-retrieval-32M"}},
    ],
})

db.insert([
    {"id": "b1", "title": "The river and the dragon", "description": "A fantasy of rivers.", "genres": ["fantasy"], "year": 1999},
    {"id": "b2", "title": "Winter garden", "description": "A quiet mystery.", "genres": ["mystery"], "year": 2005},
])

for row in db.search("dragon", limit=5):
    print(row["id"], row["title"])

db.query("SELECT year, COUNT(*) FROM books WHERE genres = :g GROUP BY year", g="fantasy").rows
db.close()
```

`strata.open(path, schema)` opens the database at `path`, or creates it with `schema` if it does not exist. The schema is a dict, JSON text or the path of a JSON file, in the [form the server takes](server.md#schemas). `Database.create(path, schema)` and `Database.open(path)` do one or the other. `models=` says where embedding and [ranking](query-language.md#ranking-models-gbdtmodel-features) models, and the captioners of [text made from media](server.md#text-from-media), are (default `$STRATA_MODELS`, else `~/.local/share/strata/models`; `strata-server fetch-model NAME` downloads one there), and `directory=True` creates the database as a directory of segment files.

A database written by a version of Strata from before [field ids](index.md#wal-memtable-and-snapshots) does not open: `strata.upgrade(path)` copies it into this version's format in its place, keeps the old one as `<path>.before-field-ids`, and returns `(old_path, tables, rows)`, or `None` when the database needs no upgrade. Nothing may have it open meanwhile.

A database file can be open in one process at a time. Within that process, threads can query at once: every call releases the GIL while it runs. To share a database between processes, serve it with `strata-server` and use the client below. A `with` block, or `close()`, merges the small segments that writes left and closes the file.

## Writing

Rows are dicts of field name to value: `str`, `int`, `float`, `None`, `bool` for a `bool` field, a `datetime` or `date` (or ISO 8601 text) for a `timestamp` field, a list for a `multi` field, a list of floats or a numpy array for a vector, a dict of term weights for a `sparse` field, a URI or a dict for a `media` field (in-process, the whole value or `{"uri": …, "data": <base64>}`: nothing is fetched, see [media fields](server.md#media-fields)). A naive `datetime` is taken as UTC. A field the schema does not have is an error. Results give a `bool` field's values as `bool` and a `timestamp` field's as ISO 8601 text in UTC (`"2026-10-01T12:30:00Z"`), in-process and from the client alike; `datetime.fromisoformat` reads them.

| method | what it does |
|---|---|
| `insert(rows)` | adds a row (a dict) or rows (any iterable of dicts) whose keys must be new; all or none are written. From 1,000 rows on, they are bulk-loaded straight into segments. |
| `upsert(rows)` | adds rows, replacing the rows with their keys, in transactions of 10,000 |
| `delete(keys)` | deletes the rows with a key or keys; returns how many there were |
| `sql("UPDATE ...")` | `INSERT`, `UPDATE` and `DELETE` statements; returns the rows written. It also runs `CREATE TABLE` and `DROP TABLE` ([tables](query-language.md#tables)) and `CREATE` and `DROP RANK PROFILE` ([rank profiles](query-language.md#rank-profiles-a-ranking-kept-with-its-table)), which return 0, and writes any table |
| `import_file(path, format=None)` | loads a JSON Lines, JSON, CSV or TSV file ([how files are read](server.md#loading-rows)); returns `{rows, bulk, skipped_fields}` |
| `write(ops)` | applies write operations in one transaction, as the server's `/v1/write` takes them: `insert`, `upsert` and `delete`, and `set` or `add` of a numeric field in place (`{"op": "add", "key": k, "field": "n", "value": 1}`; with a `doc` instead of a `key`, an `add` inserts the row when no row has its key); each names its `table` or writes the first; returns the rows written |
| `import_sqlite(path, tables=None, schemas=None)` | imports a SQLite database's tables, each into the table of its name, created when missing ([SQLite in and out](sqlite.md)); returns what it did and what did not carry over |
| `export_sqlite(path, tables=None, fts=True, indexes=True)` | writes the tables to a new SQLite file, with indexes, an FTS5 index over the text and the Strata schemas |

`strata.from_sqlite("shop.sqlite", "shop.strata")` creates a database from a SQLite file's tables, and `strata.plan_sqlite("shop.sqlite")` says what it would make of them without writing; edit the plan and pass it to either as `schemas`.

`insert`, `upsert`, `delete` and `import_file` write the database's first table, the one `schema` describes, or the one `table=` names (`db.insert(rows, table="events")`); `sql` names the table it writes, and `tables()` lists every table's schema. Through a coordinator, `delete` on a table placed by a column takes that column's value as `routing=` ([tables on shards](server.md#tables-on-shards)). A key a row leaves out gets its field's `default`, and a field with `not_null` refuses a row without a value ([schema options](server.md#schemas)); a `timestamp` defaulting to `"now"` takes one time for all the rows of a call.

Vectors of embedded fields that a row leaves out are computed by the field's model when it is open, here or behind an API: a field naming `openai/text-embedding-3-small`, `cohere/embed-v4.0`, `voyage/voyage-3.5`, `gemini/gemini-embedding-001`, `ollama/nomic-embed-text` or another [hosted model](server.md#hosted-models) gets its vectors from the provider, with the key in the process's environment (`OPENAI_API_KEY`, ...), so inserting plain text is enough. A write whose vectors the provider will not give (after its retries) raises `StrataError` and writes nothing. `db.embed_errors` lists the embedded fields whose model did not open (a hosted model without its key says which variable to set), whose vectors rows must then bring, and the [semantic-ID fields](#semantic-id-codebooks) whose codebook did not open.

## Querying

`query(sql, params)` runs a `SELECT` in [Strata's SQL](query-language.md). Parameters bind `:name` placeholders, from a dict, keyword arguments, or both:

```python
db.query("SELECT id, title FROM books WHERE year >= :y ORDER BY bm25(title, :q) DESC LIMIT 10", {"y": 2000}, q="dragon")
db.query("SELECT id FROM books ORDER BY similarity(embedding, :v) DESC LIMIT 10", v={"embed": "a quiet mystery"})
db.query("SELECT id FROM books ORDER BY similarity(embedding, :v) DESC LIMIT 10", v={"vector_of": "b1"})
```

A parameter is a number, a string, `None`, a list of numbers or a numpy array (a vector), a list, a dict of term weights, `{"embed": text}` (the text's vector by an embedded field's model; add `"field"` to pick one) or `{"vector_of": key}` (a stored row's vector).

The answer is a `Result`:

| | |
|---|---|
| `for row in result`, `result[i]` | rows as dicts of column to value |
| `result.rows`, `result.columns` | rows as lists, in column order |
| `result.keys`, `result.scores` | each row's primary key and the score it was ranked by |
| `result.column("title")`, `result.scalar()` | one column; the first value (a count, a sum) |
| `result.stats` | what the engine did |
| `result.to_pandas()` | a DataFrame (with pandas installed) |

`search(text, limit=10, where=None, columns="*", mode="auto")` writes the query for the common case. It ranks by BM25 over the text fields (the generated ones if the table has any, such as a combined `body`, else every indexed text field) fused by reciprocal rank with the similarity of the first embedded vector field whose model is open; `mode="text"` or `"vector"` uses one side only. `where` adds a condition, [built in Python](#queries-built-in-python) or in SQL with `params` for its placeholders:

```python
db.search("dragon", where=col("year") >= 1990, columns=["id", "title"])
db.search("dragon", where="year >= :y", params={"y": 1990})
```

`embed(text, field=None)` returns the query vector of a text (`embed(image=data, field="emb")` of an image file's bytes, by a [model of images](server.md#vectors-of-images), with the package built with `--features vision`, and `embed(audio=data)` of a sound file's, by a [model of sound](server.md#vectors-of-sound), built with `--features audio`), and `len(db)` the number of rows.

`plan(sql, params)` returns a statement's [plan](query-language.md#explain) as a dict without running it, and `run(plan)` runs a plan: a read's `Result`, or the number of rows a write wrote.

```python
p = db.plan("SELECT id FROM books WHERE year >= :y LIMIT 3", y=2000)
db.run(p)
```

## Queries built in Python

`table()` starts a query of the table built from Python values instead of SQL text, so nothing is quoted or escaped: fields are named as the schema names them (`col("order")`, `col("doc no")`), and values keep their types (a `str` is text whatever quotes it holds, a `date` or `datetime` a timestamp, UTC unless it says otherwise). Each method returns a new query; iterating one, or `run()`, runs it and gives its `Result`:

```python
from strata import col, match, bm25, similarity, rrf, count, score

q = (db.table()
     .select("id", "title", score())
     .where((col("year") >= 2019) & match("body", "graph neural") & ~col("tags").isin(["draft"]))
     .order_by(bm25("body", "graph neural"))
     .limit(10))
for row in q:
    print(row["id"], row["score()"])

db.table().select("genre", count().alias("n"), col("rating").avg().alias("avg")).group_by("genre").having(count() > 10).order_by(col("genre"))
db.table().order_by(rrf(bm25("title", text), similarity("emb", db.embed(text)))).limit(10)
# each video's best clip, the 10 best videos
db.table().select("video", "id", score()).distinct_on("video").order_by(bm25("transcript", text)).limit(10)
```

| | |
|---|---|
| `col(name)`, `lit(value)` | a field; a value where an expression is wanted |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | comparisons with a value or another expression |
| `&`, `\|`, `~` | and, or, not (Python's `and`, `or`, `not` and chained comparisons such as `1 < col("x") < 5` raise `TypeError`) |
| `.isin(values)`, `.between(lo, hi)`, `.is_null()`, `.is_not_null()` | as SQL's `IN`, `BETWEEN` (a bound of `None` is open), `IS NULL` |
| `match(field, query, min=None, min_rows=None, max_rows=None)` | the full-text condition, in [MATCH's syntax](query-language.md#full-text-match); `min=2` or `min="75%"` asks for that many of its items, and with `max_rows` (or `min_rows`) for [as many as fit](query-language.md#as-many-as-fit-min_rows-and-max_rows) |
| `at_least(n, *conds, min_rows=None, max_rows=None)`, `count_true(*conds)` | at least `n` of the conditions hold (`n` a count, or a percentage such as `"75%"`), as SQL's [`AT_LEAST`](query-language.md#some-of-the-conditions-at_least), or with `max_rows` (or `min_rows`) [as many as fit](query-language.md#as-many-as-fit-min_rows-and-max_rows); how many hold, a number to rank by |
| `bm25(field, text, syntax=None, min=None, min_rows=None, max_rows=None)`, `similarity(field, vector)`, `rrf(*scores, k=60, depth=100)`, `ranked(keys)` | scores; `bm25` reads [weights, required and excluded words](query-language.md#weights-required-and-excluded-words) (`"+mrna vaccine^2 -animal"`) unless `syntax="plain"`, and in `order_by` with `max_rows` ranks the rows with [as many of its words as fit](query-language.md#as-many-as-fit-min_rows-and-max_rows); `similarity` takes a vector (`embed(text)` gives a text's), or `{term: weight}` for a SPARSE field; `ranked` is a [ranking from elsewhere](query-language.md#rankedkeys-a-ranking-from-elsewhere), primary keys best first |
| `+`, `-`, `*`, `/`, `%`, `**`, `abs()`, `round()`, `ln`, `log`, `exp`, `sqrt`, `sigmoid`, `clamp`, `least`, `greatest`, `coalesce`, `minmax`, `rank`, `case((cond, value), ..., otherwise=None)` | a [ranking expression](query-language.md#ranking-expressions) of scores, fields and numbers: `bm25("body", q) * (1 + 0.2 * coalesce(col("rating"), 0))`. `+` adds raw values; `0.7 * minmax(a) + 0.3 * minmax(b)` is weighted fusion |
| `gbdt(model, *features)` | a [ranking model](query-language.md#ranking-models-gbdtmodel-features)'s score of a row from its features: `gbdt("docs_ltr", bm25("body", q), col("rating"))`, the LightGBM or XGBoost model `<models>/docs_ltr/gbdt.json` |
| `reachable(field, seeds, hops=1, ...)`, `path(field, seeds, to, ...)`, `ppr(field, seeds, ...)` | [graph](graphs.md) conditions and PageRank as a score, with the SQL functions' options as keyword arguments |
| `score()`, `hops()` | the score a row was ranked by; its hops from a traversal's seeds |
| `features()` | in `select`, a column for each input of the ranking ([features](query-language.md#features-the-values-a-ranking-read)), named as Python writes it; any expression of scores that scores a row by itself is a column too |
| `count()`, `col(f).count(distinct=True)`, `.sum()`, `.avg()`, `.min()`, `.max()` | aggregates, for `select`, `having` and `order_by` over groups |
| `.alias(name)`, `.asc()`, `.desc(nulls_first=False)` | a column's name; an `order_by` direction |

`select(...)` names the columns (without it, the stored fields but vectors, as `SELECT *`), `where(...)` keeps rows, `order_by(...)` ranks by a score best first or orders by fields ascending, `rerank(expr, count)` ranks its top `count` rows again ([rerank phases](query-language.md#rerank-phases)), `group_by(...)` and `having(...)` group, `distinct_on(...)` keeps the [first row of each group](query-language.md#the-first-row-of-each-group-distinct-on) in the query's order, `max_per(k, field)` keeps [at most k rows of each value](query-language.md#at-most-k-rows-of-a-value-max-k-per), and `limit(n)` and `offset(n)` page (groups, after `distinct_on`). A query is a [plan](query-language.md#explain), the same JSON that `plan(sql)` gives: `q.plan()` shows it, and a `Client` runs it on a server the same way.

## The rank log

`query_logged(sql, params, context=..., variant=..., candidates=...)` runs a ranked query and writes what it served to a [rank log](query-language.md#the-rank-log-what-each-query-served): a row per row answered, with its position, its score and the values its ranking read, as they were computed. `candidates` logs that many rows from the query's `OFFSET`, those past its `LIMIT` as not shown, and `ts` (a `datetime` or microseconds) says when it was served, now by default. The `Result` is the query's, with `result.request`, the id its log rows carry, for labelling them later. `export_rank_log(...)` reads the log back as a list of dicts for training, each with its context's history and its label as of the row's time, from a table of events:

```python
r = db.query_logged("SELECT id, title FROM books ORDER BY bm25(title, :q) DESC LIMIT 10",
                    {"q": "dragon"}, context="u42", variant="b", candidates=50)
db.sql("UPDATE rank_log SET label = 3 WHERE request = :r AND row_key = :k", {"r": r.request, "k": r.keys[0]})

rows = db.export_rank_log(events={"table": "events", "user": "usr", "item": "item", "time": "ts", "action": "action"},
                          history=20, window="30 minutes", grades={"click": 1, "buy": 3})
rows[0]["features"], rows[0]["label"], rows[0]["history_items"]
```

## Semantic-ID codebooks

A [semantic-ID field](server.md#semantic-ids) gives each row a code from its vector by a codebook, which `strata.train.codebook` trains and saves in the models directory before a table names it:

```python
h = strata.train.codebook(db, "items-rq", field="emb")       # a table's vectors
h = strata.train.codebook(vectors, "items-rq")               # a list of lists, or a 2-D numpy array

items = strata.open("items.strata", schema={
    "table": "items", "key": "id",
    "fields": [
        {"name": "id", "type": "keyword"},
        {"name": "emb", "type": "vector", "dim": 384},
        {"name": "sid", "type": "keyword", "multi": True,
         "semantic_id": {"from": "emb", "codebook": "items-rq"}},
    ],
})
items.upsert(rows)                                            # each row gets its sid
items.query("SELECT id FROM items WHERE sid = '12.7'")
```

It returns the SHA-256 of the file, which the table records. The options are `levels` (1 to 4, default 3), `k` centroids a level (up to 256), `iterations` of k-means a level (25), `sample`, the vectors it trains on (100 a centroid by default, drawn at random), `seed` (the same vectors and options give the same codebook), `table` (a database's first by default) and `models` (the default models directory, as for `Database.create`). Pass the same `models=` to `Database.create` and `Database.open`: a codebook that does not open is listed in `db.embed_errors`, and writes that need a code then fail.

## Next items

`strata.ngrams` keeps counts of what came after the last one or two items in a table, and `strata.next_items(db, history, k=10)` reads them with backoff ([Next items](next-items.md)):

```python
from strata import ngrams

db = strata.open("next.strata", ngrams.schema())
db.write(ngrams.ops(["a", "b"], "c"))       # after each event: "c" after [..., "a", "b"]
strata.next_items(db, ["a", "b"], k=10)     # [("c", 1.0), ...]
```

## Sequence models

[`encode('model', history)`](query-language.md#sequence-models-encodemodel-history) turns a user's events into a vector of an item field, by a model that `strata.train.sequence` trains with PyTorch (`pip install torch`) and saves in the models directory:

```python
import strata.train
db = strata.Database.open("shop.strata", models="models")
out = strata.train.sequence(db, "next", table="items", field="emb",
                            events={"table": "events", "user": "usr", "item": "item", "action": "action", "time": "ts"},
                            learn_items=True, negatives="all", holdout=True, models="models")
strata.train.write_vectors(db, "next", models="models")      # the learned item vectors, into items.emb
db.query("""SELECT id FROM items ORDER BY similarity(emb, encode('next',
            (SELECT item, action, ts FROM events WHERE usr = :u ORDER BY ts DESC LIMIT 200))) DESC LIMIT 10""", {"u": "u42"})
```

- **The histories** come from one of `events` (a table of the database: its columns of users, items, and optionally actions, times, and `order` for events with equal times), `sequences` (a dict from user to a list of `(item, action, time in seconds)`, oldest first) or `records` (the [rank log's export](#the-rank-log) with `history`: each record's history is a training example, and its row the target when its label is positive or missing). The item vectors are the field's, or `vectors` (a dict from key to vector).
- **What it learns**: every event is the target of the events before it. Each item is scored as `similarity` scores it, the dot product of the model's unit vector with the item's vector, over `temperature` (0.05), and the loss is cross-entropy against the other targets of the batch (`negatives="in_batch"`, the default) or every item (`"all"`, for catalogs that fit). With `learn_items=True` the item vectors are trained too, starting from the field's, and returned normalized in `out["vectors"]`; without it the field keeps its vectors and the model learns to map histories into their space, so a new item is found as soon as it has a vector.
- **The item vectors go with the model**: it is saved with the vectors it was trained with (`items.safetensors`, part of its hash), and `encode` refuses a history whose items have other vectors in the field. `strata.train.write_vectors(db, name, models=None)` writes them into the field the model's config names, by an `UPDATE` of that field alone for each item (a `Database` or a `Client`); returns the rows written. `strata._sequence.read_items(path)` reads them back.
- **Options**: `width` (64), `layers` (2), `heads` (1), `ffn` (4 × width), `max_len` (200 events), `gap_buckets` (32), `actions` (the actions seen by default; others count as one more), `epochs` (50), `batch` (128), `lr` (1e-3), `dropout` (0.2), `seed`. With `holdout=True` each user's last event is kept out of training, and training stops when NDCG@10 on those has not risen for `patience` epochs (5), keeping the best epoch. `max_seconds` stops it after the first epoch that ends past that many seconds. `log=print` prints each epoch's loss and metrics, which `out["history"]` also holds.
- It returns `{"path", "history", "vectors"}`. Pass the same `models=` to `Database.open`, or start the server with `--models`: a coordinator needs the model, its shards don't.

## A server from Python

`strata.connect(url, token=None)` returns a `Client` of a running strata-server, using only the standard library. It has `schema`, `tables`, `query`, `sql`, `search`, `insert`, `upsert`, `delete`, `import_file` (an upload to [`/v1/import`](server.md#loading-rows), up to the server's `--max-body-mb`, 256 MiB unless set), `import_sqlite(path, tables=None)` and `plan_sqlite(path)` (a SQLite file's tables, uploaded the same way; [SQLite in and out](sqlite.md#over-http-and-in-the-ui)), `export_sqlite(path, tables=None)` (the server's tables downloaded as a SQLite file) and `len()`, as above, plus `write(ops)` for raw [write operations](server.md#http-api), `explain`, `run(plan)` (a plan from `explain(...)["ir"]` or a `table()` query, run by [`/v1/plan`](server.md#http-api)), `table()` ([queries built in Python](#queries-built-in-python)), `embed(text, field=None)` (a text's vector by the server's model, from [`/v1/embed`](server.md#http-api), or `embed(image=data)` an image's and `embed(audio=data)` a sound's), `query_logged` and `export_rank_log` ([the rank log](#the-rank-log)), `health`, `stats` and `ask(question)` for the [agent](server.md#the-agent).

```python
client = strata.connect("http://127.0.0.1:7700")   # token= or $STRATA_TOKEN for a server with --token
client.upsert({"id": "b3", "title": "Star letters", "year": 2024})
client.search("letters", limit=3).dicts()
```

The server embeds `{"embed": text}` parameters and the rows it is sent with its own models. Errors the server reports raise `strata.StrataError`, with the HTTP status as `.status`; a server that cannot be reached raises `urllib.error.URLError`.

### Streams and Kafka

`client.stream(events)` writes an iterable of [stream lines](server.md#streams) through `/v1/stream` as it is read, each an operation or one event's operations as `{"ops": [...], "source": {"name", "offset", "ts"}}`, and returns the stream's last answer (`lines`, `written`, `skipped`). `client.source(name)` is the offset of the last event a source wrote (None for none), and `client.sources()` lists them all. A server skips the operations of an offset it has reached, so a stream resumed from `source(name) + 1`, or from anywhere before it, writes each event once.

`strata.ingest.kafka` is a Kafka (or Redpanda) consumer built on it, which needs `confluent-kafka` (`pip install confluent-kafka`, or `pip install "./strata-py[kafka]"`). Each partition is a source named `<topic>/<partition>` (after `prefix=`), and a partition given to the consumer is read from after the offset the server has, not from what the consumer group committed to Kafka, so each event is written once across consumer restarts, server restarts and rebalances. Within a partition the events are written in order.

```python
import json
import strata
from strata import ingest

def click(e):                                   # an ingest.Event: topic, partition, offset, timestamp, key, value
    c = json.loads(e.value)
    return [
        {"op": "insert", "table": "events", "doc": {"id": c["id"], "usr": c["usr"], "item": c["item"]}},
        {"op": "add", "table": "users", "key": c["usr"], "field": "n_events", "value": 1},
    ]

ingest.kafka(strata.connect("http://127.0.0.1:7700"), "clicks", click, brokers="127.0.0.1:9092", group="strata")
```

The function gives an event's operations, which commit together with its offset; None writes the offset alone (for events to leave out). Without one, each value is JSON: an operation, or `{"ops": [...]}`. The consumer runs until interrupted, or until `max_events=` events were read or none came for `idle=` seconds, and returns counts (`events`, `written`, `skipped`, `streams`, `retries`). It keeps a stream open for `seconds_per_stream=` (1 by default), then commits the stream's offsets to Kafka for Kafka's own lag tools and opens the next. When the server cannot be reached, or answers 429 or 5xx, it waits and reads every partition again from the server's offsets, for up to `retry_for=` seconds (60). An event the function fails on stops it after the events before it are written, so a restart begins with that event. `config=` adds librdkafka settings, and consumers of one `group` share the partitions. `ingest.consume(client, log, to_ops)` runs the same loop over another log, through the small `ingest.Log` interface.
