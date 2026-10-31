---
title: Query language
parent: Strata
nav_order: 1
---

# Strata's query language
{: .no_toc }

Strata speaks standard SQL over the tables of a database: `SELECT` with functions for full text, vectors, rank fusion and graphs, aggregates, typo-tolerant and synonym searches, and subqueries, `WITH` and joins of a table with itself; `INSERT`, `UPDATE` and `DELETE`; and `CREATE TABLE` and `DROP TABLE`. Text, structured and vector conditions mix freely in one statement, and the planner decides how to evaluate them together.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

## A first look

```sql
-- boolean full text and structured filters, ranked by BM25
SELECT id, title, score() FROM papers
WHERE MATCH(body, 'vaccine AND (mrna OR "messenger rna") NOT animal')
  AND year BETWEEN 2019 AND 2021 AND category IN ('virology', 'immunology')
ORDER BY bm25(body, 'mrna vaccine efficacy') DESC LIMIT 10;

-- filtered vector search
SELECT id FROM papers WHERE rating >= 4
ORDER BY similarity(emb, :q) DESC LIMIT 10;

-- hybrid: a weighted sum of normalised scores, or reciprocal rank fusion
SELECT id FROM papers WHERE year >= 2015
ORDER BY 0.2 * tmm(bm25(body, :text)) + 0.8 * tmm(similarity(emb, :q)) DESC LIMIT 10;

SELECT id FROM papers WHERE year >= 2015
ORDER BY rrf(bm25(body, :text), similarity(emb, :q)) DESC LIMIT 10;

-- any expression of scores and columns
SELECT id FROM papers
ORDER BY ln(1 + bm25(body, :text)) * (1 + 0.2 * coalesce(rating, 0)) DESC LIMIT 10;

-- facets and aggregates over any result
SELECT category, COUNT(*), AVG(rating), MAX(year) FROM papers WHERE MATCH(body, 'vaccine')
GROUP BY category HAVING COUNT(*) >= 10 ORDER BY 2 DESC LIMIT 5;

-- typos and synonyms
SELECT id FROM papers WHERE MATCH(body, 'vacine~ OR mrna', synonyms => 1) LIMIT 10;

-- a join along citations, ranked by one side
SELECT a.id, a.title, b.id AS cited, b.title AS cited_title
FROM papers a JOIN papers b ON a.cites = b.id
WHERE a.year >= 2020 ORDER BY bm25(a.body, 'mrna vaccine') DESC LIMIT 20;
```

## Statement shape

```text
[WITH name AS (SELECT ...), ...]
SELECT [DISTINCT | DISTINCT ON (<expressions>)] <columns | * | score() | hops() | seq_no() | features() | ranking expressions | aggregates>
FROM <table> [JOIN <table> ON ...] | <WITH name> | (SELECT ...) AS t
[WHERE <predicate>]
[GROUP BY <columns> [HAVING <predicate over aggregates>]]
[ORDER BY <score, column or aggregate> [ASC | DESC] [NULLS FIRST | LAST] [, <tie-break> [ASC | DESC]] [MAX k PER <field>]]
[LIMIT n [OFFSET m]]
```

- **Tables by name.** A database holds one or more [tables](#tables), and a query reads the tables it names (in any case): one, or several joined, read by subqueries or named in `WITH` ([below](#subqueries-with-and-joins)).
- **Quoted names.** A table or column name in double quotes is the name inside them, so any name can be written: `"order"` (a word SQL reserves), `"doc no"`, `"say ""hi"""` (a quote doubled), and `FROM "papers"` reads papers. Python's [`table()`](python.md#queries-built-in-python) builds queries with no quoting at all.
- **Strings** are single-quoted, `'sale'`. Where a value goes, a double-quoted `"sale"` is a string too, as SQLite and MySQL read it, and a bare name is an error rather than a string: `WHERE tag = sale` says `no such column: sale`.
- **Clauses Strata does not run are errors**, never ignored: `FETCH`, `FOR UPDATE`, `TOP`, `SELECT INTO`, `WINDOW`, `QUALIFY`, `GROUP BY ... WITH ROLLUP` and the like.
- **`SELECT *`** returns every stored, non-vector column.
- **`score()`** is the value of the `ORDER BY` score for each row (over a `WITH` result read without an `ORDER BY`, the score the `WITH` ranked its rows by); **`hops()`** is a row's distance in a graph traversal ([below](#graphs)); **`seq_no()`** is the sequence number of the write that last changed the row, in a database that numbers its writes, for [conditional writes](server.md#conditional-writes). It is a column of a query on one table, and cannot be aggregated or ordered by. A [ranking expression](#features-the-values-a-ranking-read) that scores each row by itself is a column too (`bm25(title, :q) AS t`), and **`features()`** stands for the inputs of the query's ranking.
- **`ORDER BY`** takes one key, plus an optional tie-break that must be the primary key (or, in a facet query, the group column). Ranking by a score is always best first; a `KEYWORD` or stored `TEXT` column first sorts the rows by its text, `NULL` last either way, which the row evaluator does after reading every row the `WHERE` keeps; `ORDER BY distance(…) ASC` ranks like `similarity(…) DESC`. `MAX k PER field` at its end keeps at most k rows of each value ([below](#at-most-k-rows-of-a-value-max-k-per)).
- **Parameters** are `:name` placeholders, bound from the API or from the shell's `--param name=value`. A vector parameter is a JSON array.
- **Limits.** An expression nests at most 1,000 levels, counting parentheses and the operators of a chain such as `a OR b OR …` (SQLite has the same default); a long list of values belongs in `IN (…)`, which has no such limit. A `LIMIT` larger than the table returns every row that qualifies. A query ranked by a score without a `LIMIT` returns 100 rows, counted after its `OFFSET`, on one database and through a coordinator alike.

## Columns and types

| type | indexed as | filters |
|---|---|---|
| `INT`, `FLOAT` (finite: no NaN or infinity) | frame-of-reference column with zone maps, range histogram | `=`, `!=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IS [NOT] NULL` |
| `KEYWORD` | dictionary codes and one docID list per value | `=`, `!=`, `IN`, `NOT IN`, `IS [NOT] NULL`, [prefixes](#prefixes-and-autocomplete) (`LIKE 'p%'`, `PREFIX`); `GROUP BY` |
| `BOOL` | an `INT` column of 1 and 0 | the field alone (`WHERE done`, `NOT done`), `=`, `!=`, `IS [NOT] TRUE`, `IS [NOT] FALSE`, `IS [NOT] NULL`; `GROUP BY` |
| `TIMESTAMP` | an `INT` column of microseconds since 1970-01-01 UTC | as `INT`, with times written as text: `at >= '2026-10-01'` |
| `TEXT` | inverted index (optionally with positions); may be generated from other text fields | `MATCH` |
| `VECTOR(d)` | float32 rows plus 8-bit codes (or 1-bit codes, `"codes": "binary"`), with HNSW, IVF or flat search; may be declared as a model's embedding of other fields ([embedded vector fields](server.md#embedded-vector-fields)) | ordering by `similarity` / `distance`; a similarity graph |
| `VECTOR(d)[]` | several vectors per row (a page's patches, a song's frames): 1-bit codes searched, 8-bit codes to score (float32 too with `keep_f32`), and maps from rows to their vectors and back | ordering by `maxsim` |
| `SPARSE` | inverted index of learned term weights (SPLADE and similar) | ordering by `similarity` |
| `MEDIA` | a file by reference: URI, SHA-256, type, size; a `KEYWORD` index of the hashes | `=`, `IN` on the hash; `GROUP BY` |
| `EDGES` | out-edge CSR and reverse index per segment | `REACHABLE`, `PATH`, `ppr` |

Predicates combine with `AND`, `OR`, `NOT`, [`AT_LEAST`](#some-of-the-conditions-at_least) and parentheses, across all of these, and [`max_rows`](#as-many-as-fit-min_rows-and-max_rows) lets an `AT_LEAST` choose its count.

A `KEYWORD` compares with text: `k = 1` is an error that points to `k = '1'`, as `n = '2'` is on an `INT` (a write still stores the number 1 in a `KEYWORD` as `'1'`). `INT` values are 64-bit integers and compare exactly: `id = 9007199254740993` finds that row and not 9007199254740992, which a double cannot tell apart. A fraction bounds integers by rounding inward (`n > 2.5` is `n >= 3`, `n = 2.5` matches nothing).

`NULL` follows SQL's three-valued logic. A comparison with a missing value is neither true nor false, and so is its negation: `rating != 2` and `NOT rating > 2` leave out the rows without a rating. Ask for them with `rating IS NULL` (`IS NOT NULL` for the others). `MATCH` and graph predicates are always true or false, so `NOT MATCH(body, 'x')` includes rows without a body. A filter on a multi-valued field holds when some value matches, and its negation when none does, so `tags != 'red'` includes rows without tags (`tags IS NULL` finds those). `rating = NULL` is an error that points to `IS NULL`. `rating IN (1, NULL)` is TRUE where the rating is 1 and UNKNOWN elsewhere, so `rating NOT IN (1, NULL)`, like `NOT IN` a subquery that returns a `NULL`, keeps no row; a multi-valued field's `IN` is TRUE or FALSE (the `NULL` matches none of its values). `LIKE '%'` and `PREFIX(f, '')` hold for every row with a value and are UNKNOWN for the others, so their `NOT` keeps no row of a single-valued field. A scalar subquery that finds no row is `NULL`, and a comparison with it keeps no row either way.

A `BOOL` is `TRUE` or `FALSE`; a write also takes 1 and 0 and the text `'true'`, `'false'`, `'t'`, `'f'`, `'1'` and `'0'`, and refuses anything else. A field named alone is a condition (`WHERE done AND NOT archived`), with `NULL` neither true nor false: `NOT done` and `done != TRUE` leave out the rows without a value, `done IS NOT TRUE` takes them in. `SUM(done)` counts the trues, and answers come back as booleans.

A `TIMESTAMP` is a moment, kept to the microsecond in UTC and written as ISO 8601 text: a date (`'2026-10-01'`, midnight UTC) or a date and time (`'2026-10-01T12:30:00Z'`, `'2026-10-01 14:30:05.25+02:00'`), with an offset or none (UTC). A number is refused, so seconds are not mistaken for microseconds. Comparisons, `BETWEEN`, `IN`, `ORDER BY`, `MIN`, `MAX` and `COUNT` work on it (`SUM` and `AVG` do not), and answers are ISO 8601 text in UTC, such as `2026-10-01T12:30:05.250Z`. In `decay(at, origin, scale)` on a `TIMESTAMP`, the origin is a time or `now()`, and the scale and offset are seconds: `decay(published, now(), 2592000)` halves a row's weight for each 30 days of age. A database whose tables use `BOOL` or `TIMESTAMP` is written in [file format 3](index.md#wal-memtable-and-snapshots), which versions of Strata before these types refuse.

`KEYWORD` and `INT` fields may be multi-valued (`"multi": true` in the schema): a row holds a list, such as `ARRAY['red', 'blue']`, and a filter matches a row if any of its values matches. A facet counts each distinct value once per row.

A `KEYWORD[]` field may hold the [semantic ID](server.md#semantic-ids) of a vector field: every prefix of a code such as `12.7.201.0`, filled in by the writer, so `sid = '12.7'` finds the rows under a prefix. `sid_at(sid, n)` is the first `n` numbers of a code (`sid_at(sid, 2)` is `'12.7'`), in `SELECT`, `WHERE` and `GROUP BY`.

A `SPARSE` value is a JSON object of terms and non-negative weights, `{"tok": 1.25, "##en": 0.4}`, as a learned sparse model (SPLADE, BGE-M3's sparse output, OpenSearch's neural sparse models) writes it. Terms are taken as given, not analysed, and weights are stored in steps of 1/1024.

A `MEDIA` value names a file, `{"uri": "https://cdn.example.com/a.png", "sha256": "…", "mime": "image/png", "bytes": 48213, "width": 640, "height": 480}`, and a write may give only its URI, `'https://cdn.example.com/a.png'`, or its bytes, `'{"data": "<base64>"}'`, for the server to fill in ([media fields](server.md#media-fields)). Its index holds the hash: `image = '9f86d0…'` finds the rows of that file. `OPTIONS(kind = 'image')` takes only images (`'audio'`, `'video'`, `'pdf'`), whose vectors a `VECTOR` field can hold: `emb VECTOR OPTIONS(embed = 'image', model = 'siglip-base-patch16-224')` ([vectors of images](server.md#vectors-of-images)), `sound_emb VECTOR OPTIONS(embed = 'recording', model = 'clap-htsat-unfused')` ([vectors of sound](server.md#vectors-of-sound)), and whose captions or transcripts a `TEXT` field can: `caption TEXT OPTIONS(derive = 'image', model = 'blip-image-captioning-base')`, `transcript TEXT OPTIONS(derive = 'recording', model = 'whisper-small')` ([text from media](server.md#text-from-media)).

## Some of the conditions: `AT_LEAST`

`AT_LEAST(n, c1, c2, …)` holds where at least `n` of the conditions do. A condition is anything `WHERE` takes: a comparison, `IN`, `IS NULL`, `MATCH`, a graph predicate, a subquery condition, a group in parentheses, another `AT_LEAST`.

```sql
-- two of three signals, ranked by the text
SELECT id, title FROM papers
WHERE AT_LEAST(2, MATCH(title, 'covid'), year >= 2020, venue IN ('nejm', 'lancet'))
ORDER BY bm25(body, :q) DESC LIMIT 20;

-- rows meeting more of the conditions rank higher
SELECT id FROM papers
ORDER BY bm25(body, :q) * (1 + 0.1 * count_true(year >= 2021, venue IN ('nejm', 'lancet', 'jama'))) DESC LIMIT 20;
```

- **`n`** is a count, a percentage of the conditions (`'75%'`, rounded down) or a parameter (`:n`), 1 or more (`0` and `'0%'` are errors: none of the conditions is no condition). Either is clamped to between 1 and the number of conditions, as [`MATCH`'s groups](#full-text-match) are: `AT_LEAST(1, …)` is the `OR` of the conditions, and a count past the end is their `AND`.
- **`NULL`.** A condition that is UNKNOWN (a comparison with a `NULL`) is not met. Under `NOT`, `AT_LEAST` follows SQL's three-valued logic: `NOT AT_LEAST(n, …)` holds where fewer than `n` conditions would be TRUE even if every UNKNOWN one were, so it never returns a row whose answer depends on a `NULL`. With `a = 1` and `c` `NULL`, `AT_LEAST(2, a > 0, c = 'x')` leaves the row out, and so does `NOT AT_LEAST(2, a > 0, c = 'x')`.
- **`count_true(c1, c2, …)`** is how many of the conditions hold, a number for [ranking expressions](#ranking-expressions) and `SELECT`, exactly the sum of `CASE WHEN c THEN 1 ELSE 0 END` over them: a condition on a `NULL` counts 0. In a `SELECT` over rows in memory (a join, a `WITH` result) its conditions cannot use `MATCH` or graph predicates, which only a table's indexes answer.
- **How it runs.** Each condition becomes the set of rows it holds for, as `WHERE` evaluates it, and a counter adds the sets up 64 rows to a machine word, in ⌈log2(y + 1)⌉ words for each 64 rows; sets with few rows are merged instead. A ranking that checks its candidates one at a time tests their conditions in turn, until the count is reached or out of reach. The cost grows with the number of conditions, not with `n`: over 1,000,000 rows where each condition holds for an independent 20% of them, counting the rows took 3.0 ms for 2 of 4 conditions, 4.1 ms for 4 of 8 and 9.3 ms for 8 of 16, against 0.37 ms for one condition. The same conditions written as an `OR` of every `AND` of `n` took 8.2 ms for 2 of 4 (6 `AND`s) and 114 ms for 4 of 8 (70), and 8 of 16 would need 12,870 (`strata-at-least-bench` on a 4-core cloud VM, Intel Xeon @ 2.10GHz, `strata/results/strata/at_least.json`).
- A plan with `AT_LEAST` or `count_true` needs the plan feature `at_least`. From Python, [`at_least(2, match("title", "covid"), col("year") >= 2020)`](python.md#queries-built-in-python) and `count_true(…)` build them.

## As many as fit: `min_rows` and `max_rows`

An "x of y" condition can choose its own x. With `max_rows` or `min_rows`, an `AT_LEAST`, a `MATCH(…, min => n)` or a `bm25(…, min => n)` asks for all y of its items first, and relaxes towards `n` while the rows are too few, stopping before a level that would match more than `max_rows`:

```sql
-- as many of the signals as keep the candidates to 2,000 or fewer
SELECT id, title FROM papers
WHERE year >= 2015 AND AT_LEAST(2, MATCH(title, 'covid'), venue IN ('nejm', 'lancet'), cited > 100, max_rows => 2000)
ORDER BY bm25(body, :q) DESC LIMIT 20;

-- all four words if 50 rows have them, else three, else two
SELECT id, score() FROM docs
ORDER BY bm25(body, 'mrna vaccine trial phase', min => 2, min_rows => 50) DESC LIMIT 10;
```

- **The bounds** are counts of rows, 1 or more, and `min_rows` is at most `max_rows` (else no level could meet both, an error).
- **The level.** From the strictest level, all y items, a guard relaxes one item at a time while the level matches fewer than `min_rows` rows and the next looser one no more than `max_rows`, down to `n`. With `max_rows` alone that is the loosest level that fits, and with `min_rows` alone the strictest with enough rows. If even all y match more than `max_rows`, the query runs at y. The rows counted are those the rest of `WHERE` keeps; `LIMIT` does not cut them.
- **Where.** A guard is an `AT_LEAST` or a `MATCH` joined to the rest of `WHERE` by `AND` (not one under `OR`, `NOT` or another condition), or a `bm25` of the first ranking of `ORDER BY`. `MATCH` and `bm25` need a `min`, the loosest level; `AT_LEAST`'s is its `n`. A query has one guard at most, and a [rerank phase](#rerank-phases) none. A `SELECT` over a `WITH` result that is not a rerank phase counts that result's rows.
- **The plan line** gives each level's rows and the level chosen: `guard: AT_LEAST 3 of 4 (4: 12, 3: 340, 2: 5120, 1: 23000; max_rows 2000)`.
- **Its cost** is counting the rows of every level, over all the rows the rest of `WHERE` keeps, before running at one: a guarded query costs about what counting its matches does, whatever its `LIMIT`. Over 1,000,000 rows where each condition holds for an independent 20% of them, `AT_LEAST(1, …, max_rows => 10000) LIMIT 100` took 2.5 ms over 4 conditions, 6.1 ms over 8 and 11.6 ms over 16, against 0.25, 0.51 and 0.98 ms for the same query written at the level chosen. For four words of different frequencies, over rows of 12 words drawn by Zipf's law from 5,000, `bm25(body, …, min => 1, max_rows => 1000)` took 2.4 ms at the median of 50 queries (`LIMIT 10`), against 1.4 ms at the level chosen and 0.75 ms without `min`, and a `MATCH` guarded alike 2.3 ms against 1.3 ms. A text's words are counted once each, as `AT_LEAST`'s conditions are, unless it has required or excluded words, when each level is counted in turn (`strata-guard-bench` on a 4-core cloud VM, Intel Xeon @ 2.10GHz, `strata/results/strata/guard.json`).
- **What it keeps.** BEIR's queries are sentences, and few rows hold all of a sentence's words. On five BEIR collections the top 100 rows by `bm25(body, :q, min => '100%')` held 0.03 to 0.40 of the relevant rows (recall@100), against 0.13 to 0.97 for any of the words; `min => 1, min_rows => 100`, all the words while 100 rows have them, else as many as 100 rows have, held 0.10 to 0.96, within 0.4 to 7.7 points of any word, with nDCG@10 within 0.013 of it, at a median of 0.15 to 2.05 ms a query, 0.8 to 1.6 times the any-word query's (`strata-bm25-eval`, the `_r100` variants, 3 runs each on a 4-core cloud VM, Intel Xeon @ 2.10GHz, `strata/results/text/bm25_guard.json`).
- **Over shards**, a coordinator first asks every shard for its counts (protocol 8, so a shard of an older server is refused by name), adds them up, chooses the level as one database would, and sends the shards the query at that level: one round more. A shard answering a coordinator runs the level it is sent, never one of its own. With rows like those above split over 4 shards on one machine, the guarded `AT_LEAST` of 4, 8 and 16 conditions took 4.3, 7.4 and 12.6 ms through the coordinator (1.6, 2.6 and 1.8 ms at the level chosen) against 2.3, 3.9 and 7.7 ms on one database, and the guarded `bm25` 4.3 ms against 1.7 ms, medians of 50 queries (`strata-guard-shards-bench`, the same VM, `strata-server/results/guard/shards-4.json`).
- A plan with a guard needs the plan feature `guard`. From Python, `at_least(2, …, max_rows=2000)`, `match(field, query, min=2, max_rows=500)` and `bm25(field, text, min=2, min_rows=50)`.

## Full text: `MATCH`

`MATCH(field, query)` is a boolean predicate. The query grammar follows SQLite FTS5:

```text
query   := and ("OR" and)*
and     := unary (["AND"] unary)*          juxtaposition is AND
unary   := ("NOT" | "-") unary | primary    "a NOT b" is "a AND NOT b"
primary := "(" query ")" ["@" n | "@" p "%"] | ("\"phrase\"" | word | word* | word~ | word~N) ["^" weight]
```

| query | matches |
|---|---|
| `vaccine mrna` | both words (juxtaposition is `AND`) |
| `vaccine OR vaccination` | either |
| `vaccine -animal`, `vaccine NOT animal` | the first without the second |
| `"messenger rna"` | the phrase (needs positions on the field) |
| `vacc*` | every indexed term with that prefix |
| `vacine~`, `vacine~1` | the word and the indexed terms an edit or two from it ([typos](#typos-and-synonyms)) |
| `vaccine AND (mrna OR "messenger rna") -animal` | any combination, with parentheses |
| `(mrna vaccine trial phase)@2`, `(…)@50%` | at least 2 of the group's items, or half of them |

Operators are uppercase; lowercase `and`, `or` and `not` are ordinary words (and stopwords). Parentheses and `NOT`s nest at most 100 deep. Words and phrases are analysed like the field's documents, by its [analyzer](#tables): a word that analyses to several terms (`e-mail`) becomes a phrase, and one that analyses to none (a stopword) is dropped. A prefix is lowercased and folded but not stemmed, and matches the stemmed terms that start with it. A weight (`vaccine^2`, `"messenger rna"^0.5`) is read and ignored, since `MATCH` only says whether a row matches: `vaccine^2 mrna` matches what `vaccine mrna` does, so one text can go to both `MATCH` and [`bm25`](#weights-required-and-excluded-words). A weight of 0 is read and ignored too, while `bm25` scores nothing for that word. A `^` before something that starts like a number but is not a weight (`vaccine^1e308`, `vaccine^-1`, `vaccine^2x`), and a `~` with more than 2 typos (`apple~9`), are errors in `MATCH`, where they would otherwise be searched as words; `bm25` reads its text leniently, and there they are words.

A group with `@n` matches a row with at least `n` of its items: its operands, whether juxtaposed, joined by `AND` or by `OR`, each a word, a phrase, a prefix, a typo-tolerant word or a parenthesised query (`vaccine AND (mrna "messenger rna" trial~ (phase OR stage))@2`). A negated item (`-animal`) is not counted: the row must not have it. An item written twice is one item, as in `bm25`'s `min` (`(apple apple cherry)@2` needs both words), and so are words that analyse to the same term. A stopword is not an item, and a word's synonyms and typo neighbours count as it does. `@p%` asks for p percent of the items, rounded down; a count or percentage of 0 is an error, and either is clamped to between 1 and the number of items, as Lucene's `minimum_should_match` is, so `@1` is any of them and a count past the end is all of them. `MATCH(body, 'a b c d', min => 2)` (or `min => '75%'`) treats the whole query as one group. A group nests anywhere a parenthesised query does. A group reads each item's matches once, so it costs about what an `OR` of its items does.

## Prefixes and autocomplete

`kw LIKE 'p%'` and `PREFIX(kw, 'p')` select the rows of a `KEYWORD` field whose value starts with `p`, byte for byte (no case folding). A field's values are sorted in its dictionary, so the values with a prefix are one run of it, found by two binary searches per segment: a short run unions its values' lists, and one of more than 1,024 values of a single-valued field scans the code column, as a range of numbers does. A multi-valued field matches a row when one of its values starts with the prefix, and `NOT` when none does.

```sql
-- autocomplete: the ten most asked queries that start with what was typed
SELECT id, n FROM queries WHERE PREFIX(id, :typed) ORDER BY n DESC, id LIMIT 10;

-- a user's own history first: keys of the form '<user>|<query>'
SELECT id, n FROM user_queries WHERE PREFIX(id, 'u42|how to') ORDER BY n DESC LIMIT 3;

-- with typos: 'hwo to ins' still finds 'how to install'
SELECT id, n FROM queries WHERE PREFIX(id, :typed, fuzzy => 'auto') ORDER BY n DESC, id LIMIT 10;

-- the same, exact completions first, then those one edit away, then two
SELECT id, n FROM queries WHERE PREFIX(id, :typed, fuzzy => 2)
ORDER BY CASE WHEN PREFIX(id, :typed) THEN 2 WHEN PREFIX(id, :typed, fuzzy => 1) THEN 1 ELSE 0 END * 1000000 + n DESC LIMIT 10;

SELECT id FROM files WHERE path LIKE 'src/query/%' AND path NOT LIKE 'src/query/ir/%';
```

- `LIKE` on a table's own `KEYWORD` field takes a value (no wildcard: `=`) or text followed by `%`. `_` anywhere, or `%` before the end, is refused there, since the index cannot answer it; over joined or derived rows, `LIKE` takes any pattern.
- `ESCAPE 'c'` makes the character after `c` literal: `LIKE 'snake!_%' ESCAPE '!'` is the prefix `snake_`. `PREFIX` needs no escaping.
- `ILIKE` is refused on a table's own field: store the value lowercased (and a display form beside it) and use `LIKE`.
- `PREFIX(x, p)` also works over joined or derived rows, as a test of each value; `PREFIX(kw, '')` and `LIKE '%'` are `IS NOT NULL`.
- **`PREFIX(kw, p, fuzzy => 1 | 2 | 'auto')`** also selects the values that start with a prefix within that many edits of `p` whose first letter is `p`'s own, with the edits `MATCH` counts ([typos](#typos-and-synonyms)); `'auto'` goes by `p`'s length, as it does for a word: none up to 2 letters, 1 up to 5, else 2. The dictionary is walked as a trie, one row of the edit-distance table per letter, jumping past every value whose start is already too far from `p` and past the whole run of a start close enough, so the cost follows the values near what was typed rather than the dictionary's size. The values found are a few runs of the dictionary, read as an exact prefix's one is. A value's edits are the fewest of any of its prefixes, so a completion of `p` itself has none even when a shorter prefix is also close enough. Typo matches rank by the `ORDER BY` like exact ones; to rank completions by their edits, exact ones first and each tier by its count, put the tiers in the ranking expression: `ORDER BY CASE WHEN PREFIX(id, :typed) THEN 2 WHEN PREFIX(id, :typed, fuzzy => 1) THEN 1 ELSE 0 END * 1000000 + n DESC` (the factor larger than any count). `LIKE` takes no typos.
- Counts kept with `UPDATE ... SET n = n + 1` or a write's `add` change in place, so a query log's popularity stays current without rewriting rows.

A plan with a prefix needs the plan feature `prefix`, and one with typos also `prefix_edits`, so a shard of an older version refuses it rather than misreading it. The same run of the dictionary answers [semantic-ID next codes](server.md#semantic-ids).

**Cluster a completion table by its key.** `ORDER BY n DESC LIMIT 10` reads the count column's blocks best first ([ordering by a column](#a-column)), and with `OPTIONS(cluster_by = 'id')` a prefix's rows sit together in each segment, so the ten most asked queries under a prefix read about ten blocks whatever the prefix's length:

```sql
CREATE TABLE queries (id KEYWORD PRIMARY KEY, n INT) OPTIONS(cluster_by = 'id');
```

Over ORCAS's 10.4 million queries, loaded in no particular order and merged by the default policy (11 segments), 2,000 typed queries drawn by clicks, one thread on a 4-core cloud VM (Intel Xeon @ 2.10GHz), p99 in ms:

| characters typed | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows matching (median) | 560,354 | 64,887 | 9,461 | 2,418 | 415 | 68 | 6 |
| every match read | 46.7 | 25.4 | 18.2 | 17.1 | 9.9 | 9.1 | 2.6 |
| blocks best first | 9.5 | 13.0 | 8.8 | 9.7 | 6.3 | 6.5 | 2.6 |
| blocks, clustered by `id` | 0.27 | 0.53 | 0.48 | 0.38 | 0.35 | 0.33 | 0.28 |

The full runs, with medians and the 105 segments straight after the load, are [`results/text/complete-orcas.json` and `complete-orcas-clustered.json`](https://github.com/rkonow/strata/tree/main/strata/results/text) (`strata-complete-bench`).

Typos cost the walk of each segment's dictionary, which finds each value's fewest edits. On the same clustered table, the median top 10 with `fuzzy => 'auto'` takes 1.9 ms at 3 letters and 2.7 ms at 4, where it allows one edit, and 8.2 to 11.9 ms from 6 letters, where it allows two (p99 up to 24 ms); with `fuzzy => 1` it takes 3.3 to 4.1 ms from 6 letters (p99 up to 8.5 ms). Up to 2 letters `'auto'` allows none, so it costs what an exact prefix does. Straight after a load, with 105 segments, the same queries take 2.7 to 3.9 times as long. Ranked exact completions first (the `CASE` above), the median is 5.1 to 15 ms from 2 letters and 28 ms at 1 (p99 up to 62 ms), since a ranking expression is not a column whose blocks can be read best first: every completion is ranked. Results: `complete-orcas-clustered-fuzzy.json`, `complete-orcas-clustered-fuzzy1.json` and `complete-orcas-clustered-fuzzy-exact-first.json` (one thread of the 4-core cloud VM, Intel Xeon @ 2.10GHz).

## Scores

### `bm25(field, text)`

BM25 of the analysed terms of `text`, disjunctively: a row scores if it contains any of them, unless the text [requires or excludes words](#weights-required-and-excluded-words). Ranked queries drop English function words (NLTK's 153-word list), except that its single letters (`d`, `m`, `o`, `s`, `t`, `y`) are kept unless they end a contraction: "vitamin d" and "T cell" keep their letter, "don't" and "it's" drop it. A term written twice weighs twice. Used in `ORDER BY` without a `MATCH`, it is a ranked search; with one, it ranks the matching rows.

| option | default | effect |
|---|---|---|
| `k1`, `b` | 1.2, 0.75 | BM25 parameters (`k1 >= 0`, `0 <= b <= 1`) |
| `algo` | planner's choice (MaxScore) | `'maxscore'`, `'bmw'` (Block-Max WAND) or `'exhaustive'` |
| `stopwords` | 1 (0 on a `simple` field) | `0` keeps English function words in the query, `1` drops them |
| `fuzzy` | 0 | `1`, `2` or `'auto'`: also score the indexed terms that many edits from each word ([typos](#typos-and-synonyms)) |
| `synonyms` | 0 | `1`: also score each word's synonyms |
| `syntax` | `'query'` | `'plain'` reads `+`, `-`, quotes and `^` as punctuation: the text is words alone |
| `min` | none | `n` or `'p%'`: rank only the rows with at least that many of the text's optional words and phrases |

Keeping function words is better for short questions: on Quora it gains 0.055 nDCG@10, while on TREC-COVID, whose queries are full questions, dropping them gains the same amount.

#### Weights, required and excluded words

The text may weigh a word, require it or exclude it, as a search box does:

| text | ranks |
|---|---|
| `vaccine mrna` | the rows with either word |
| `vaccine^2 mrna` | the same rows, `vaccine` weighing twice: as `vaccine vaccine mrna` |
| `+mrna vaccine` | the rows with `mrna`, by both words |
| `vaccine -animal` | the rows with `vaccine` and without `animal` |
| `+"messenger rna" vaccine^0.5` | the rows with the phrase (with both words, on a field without positions), by its words and half of `vaccine` |
| `+covid^0 treatment` | the rows with `covid`, by `treatment` alone |

- **`^w`** after a word or a closing quote multiplies its terms' BM25 by `w`, a number of at least 0 (`2`, `0.5`, `.5`). Weights add up, as a word written twice does. A word weighing 0 scores nothing, and a text whose words all weigh 0 ranks no row.
- **`+`** before a word or a quoted phrase requires it, **`-`** excludes it: a row must contain every required one and none of the excluded ones, and ranks by the others too. An excluded word does not score, and a word the query drops as a stopword is neither required nor excluded. A sign counts only at the start of a word and before a letter, digit or quote: `covid-19`, a `-` alone, `+-x` and `x^` are words. Quotes alone change nothing: `"messenger rna"` scores what `messenger rna` does.
- With `fuzzy` or `synonyms`, a word's typo neighbours and synonyms take the highest weight the word is written with (writing it twice weighs the word twice, not them), and a required or excluded word counts as present when one of them is, as in `MATCH`.
- A required or excluded word is a filter on the ranked rows: the top-k loop (all three `algo`s), a [rank-safe boost](#rank-safe-boosts), a [rerank phase](#rerank-phases) (where a row the filter drops scores 0, as one without the words does) and shards all apply it, so the rows and scores are those of the rows the text allows.
- Weights and signs cost little. With each query's first word weighed 2 (`word^2`) or required (`+word`), or its last word excluded (`-word`), the median query took 0.27, 0.28 and 0.23 ms on Quora's 10,000 queries against 0.25 ms for the plain text, and 0.24, 0.18 and 0.22 ms on FiQA against 0.22 ms; a required word can make a query faster, as the filter drops rows before they are scored. Text written without operators ranks and times as the plain text does on all five collections measured (`strata/results/text/bm25_term_operators.json`, one core of a 4-core cloud VM while another job held the other three). The operators were put on words mechanically to time them, so these queries rank worse than the plain text: an operator helps when it says what the person searching means.
- **`min => n`** (or `'p%'`) ranks only the rows with at least `n` of the text's optional words and phrases: those without a sign, each counted once however often it is written, whatever its weight (`^0` counts too). Required and excluded words are not counted; a text without optional words ignores `min`. Counts are clamped to between 1 and the number of optional words, and a percentage rounds down, as in `MATCH`'s groups: `bm25(body, 'vaccine mrna trial phase', min => 2)` ranks the rows with two of the four words, by all of them. With `syntax => 'plain'`, each word is an optional word.
- `min` filters the ranked rows, as a required word does, so its cost depends on how many rows hold one of the words. With `min => 2` the median query took 0.24 ms on Quora's 10,000 queries (0.25 ms plain) and 0.41 ms on FiQA (0.23 ms plain). On TREC-COVID, whose query words appear in most of its documents, it took 1.6 ms (0.5 ms plain), because the filter counts every row that holds one of the words (`strata/results/text/bm25_min_match.json`, one core of a 4-core cloud VM while another job held the other three). `min` is for leaving rows out, not for ranking better. Asking every BEIR query for two of its words, or half of them, left nDCG@10 unchanged on TREC-COVID and lowered it elsewhere, by as much as 0.077 (SciFact, half the words).
- **`syntax => 'plain'`** reads the text as words alone, as `bm25` did before: for text not written for a search box, such as a user's message passed through, or a benchmark's recorded queries. BEIR's hold `+1`, `-word` or `x^2` here and there, so the evaluations and benchmarks in this repository pass `syntax => 'plain'`, and so does Engram.
- A plan that ranks with weights, signs, or a `MATCH` with weights needs the plan feature `text_operators`, and one with `min` or a `MATCH` group `(…)@n` needs `min_match`. From Python, `bm25("body", "+mrna vaccine^2")` reads them, `bm25("body", text, syntax="plain")` does not, and `bm25("body", text, min=2)` and `match("body", text, min="75%")` ask for some of the words.

**Changed:** `bm25` read `+`, `-`, `"` and `^` as punctuation before; text with them now ranks as above, and `syntax => 'plain'` keeps the old reading. In `MATCH`, `^` separated words: `vaccine^2` matched `vaccine` and `2`, and now matches `vaccine`; and `)@2` closes a group a row needs 2 of, where `@2` was a word before.

### `similarity(field, vector)` and `distance(field, vector)`

The dot product (cosine for unit vectors) and the distance between a row's vector and the query vector, which is a bound parameter or a JSON array of the field's dimension.

| option | effect |
|---|---|
| `strategy` | `'auto'` (default: the planner decides), `'prefilter'` (evaluate the filter, scan its members exactly), `'index'` (search HNSW or IVF with the filter inside), `'postfilter'` (search with a larger *k* and drop non-matching rows), `'flat'` (exact scan) |
| `ef` | HNSW search breadth (default 64) |
| `nprobe` | IVF clusters probed |
| `rerank` | how many candidates are re-scored in float32 |

A `VECTOR` field keeps, besides its float32 rows, a compact code per row that the search scans: 8 bits per dimension by default, or 1 bit per dimension with `"codes": "binary"` in the field's schema (or the build option `vector.codes = 'binary'`). Binary codes are 32 times smaller than the vectors and 8 times smaller than 8-bit codes; to make up for the coarser scores, the search re-scores 8 times as many candidates in float32. On synthetic 768-dimensional data that kept recall@10 at 0.985–0.994 with flat, IVF and HNSW search, at an eighth of the memory ([results](results.md#compressed-vectors)); check recall on your own embeddings before choosing them, since some models lose more to one-bit codes than others.

### `similarity(field, weights)` on a `SPARSE` field

The dot product of a row's term weights with the query's, `Σ q_t · w_t` over the terms both have. The query is a JSON object of term weights, bound as a parameter or written as a string:

```sql
SELECT id, score() FROM docs
ORDER BY similarity(splade, '{"vaccine": 1.4, "mrna": 2.1, "efficacy": 0.6}') DESC LIMIT 10;
```

It runs on the same machinery as `bm25` (MaxScore, Block-Max WAND, the filter strategies), so `algo => 'maxscore' | 'bmw' | 'exhaustive'` applies, and it fuses with other scores in `rrf`, weighted sums and products. `distance` is not defined on a `SPARSE` field.

### `maxsim(field, vectors)`: several query vectors

The best similarity of a row's vector with any of several query vectors, $\max_i q_i \cdot v$: a row near any one of a user's recent items, or of a query's several phrasings. The vectors are a bound list of vectors, a tuple of vector parameters, a JSON array of arrays, or a subquery that returns one column of vectors, which runs first:

```sql
-- items like any of the user's last 20, by similarity
SELECT id, score() FROM items
ORDER BY maxsim(emb, (SELECT emb FROM history WHERE usr = :u ORDER BY ts DESC LIMIT 20)) DESC LIMIT 10;

-- and fused with text and a row's rating
SELECT id FROM items
ORDER BY rrf(bm25(title, :t), maxsim(emb, :recent)) * (1 + 0.1 * coalesce(rating, 0)) DESC LIMIT 10;
```

It runs one search per query vector, with the filter and the options of `similarity` (`strategy`, `ef`, `nprobe`, `rerank`), and scores the union of their top *k* exactly against every query vector. The union holds the exact top *k*: a row among them is among the top *k* of the query vector it is closest to. Each added query vector costs one more search ([what that costs](results.md#maxsim)). A row without a vector scores `NULL`, a subquery with no rows ranks nothing, and more than 64 vectors is an error unless `max_vectors => n` raises the limit. It is a score wherever `similarity` is one: in `rrf`, `minmax()` sums and ranking expressions, on one database or over shards.

On a **multi-vector field** (`VECTOR(d)[]`), `maxsim` is late interaction as ColBERT and ColPali score it: the sum over the query vectors of each one's best match among the row's vectors, $\sum_i \max_j q_i \cdot v_j$. Each query vector's search returns its best `depth` vectors (64), which name candidate rows; the `candidates` rows (200, at least the `LIMIT`) with the largest sum of the matches found are scored over all their vectors, with their 8-bit codes, or float32 when the field keeps them. A row without vectors ranks nowhere. `depth => n` and `candidates => n` trade time for recall; with both at least the number of vectors, the answer is exact. `similarity` refuses a multi-vector field, and so do graph functions.

```sql
-- pages by their patches' best matches with the query's token vectors
SELECT id, score() FROM pages ORDER BY maxsim(patches, :tokens) DESC LIMIT 5;
```

`dot(a, b)` is the dot product of two vectors as a value, for rows in memory: `SELECT id, dot(emb, :q) FROM h` over a `WITH` result, a join, or a table's own rows. It equals `similarity` to the bit, and in a ranking expression over a table (`ORDER BY bm25(title, :q) + 2 * coalesce(dot(emb, :q), 0) DESC`) it is `similarity(emb, :q)`: `NULL` for a row without a vector.

### `rrf(s1, s2, …)`: reciprocal rank fusion

Each input ranks the rows that pass `WHERE`, to a depth of 100 by default, and a row's fused score is

$$
\mathrm{rrf}(d) = \sum_{i} \frac{1}{k + \mathrm{rank}_i(d)}, \qquad k = 60
$$

Options: `k => 60`, `depth => 100`. The depth is raised to `LIMIT + OFFSET` when the page asks for more rows, as in weighted sums and products.

### `ranked(keys)`: a ranking from elsewhere

`ranked(:keys)` is a ranking computed outside Strata, given as primary keys, best first: a recommender's candidates, a reranker's order, or the rows a [semantic-ID](server.md#semantic-ids) decoder generated. It is a score like any other, so it fuses with Strata's own rankings:

```sql
SELECT id FROM items WHERE in_stock ORDER BY rrf(bm25(title, :q), ranked(:keys)) DESC LIMIT 20;
```

A listed row scores `-(its position)`, from `-1` for the first key; a key listed twice ranks where it is first. A row the list leaves out has `NULL`, so in an expression it ranks last unless `coalesce` gives it a value, and in a weighted sum it takes the input's worst value, as a missing column does. Times another score (`ranked(:keys) * rating`) it multiplies as arithmetic, not as a product of scores, which would count its negative values as 0. A key no live row has is skipped. On its own, `ORDER BY ranked(:keys) DESC` answers the listed rows that `WHERE` keeps, in list order. The keys are a parameter holding a list (an array of keys in the HTTP API), a tuple (`ranked(('a', 'b'))`) or an array (`ranked(ARRAY['a', 'b'])`). Over shards, each shard scores its own rows from the list and the coordinator merges them, so no extra round is needed. A plan that uses it needs the plan feature `ranked`.

### Weighted sums: `tmm()`, `minmax()` and `zscore()`

A weighted sum of normalised scores is weighted (convex) fusion: `0.2 * tmm(bm25(body, :t)) + 0.8 * tmm(similarity(emb, :q))`. Weights are numeric constants; `+` and `-` combine any number of terms. Candidates are the union of each input's top 100, re-scored exactly by every input and normalised over that union ($C$). Three normalisers put a score on one scale:

| normaliser | a row's value | a row without a value |
|---|---|---|
| `tmm(x)`, theoretical min-max | $\dfrac{x - lo}{\max_C x - lo}$, `lo` the least value `x` can take | 0 |
| `minmax(x)` | $\dfrac{x - \min_C x}{\max_C x - \min_C x}$, 1 where all are equal | 0 |
| `zscore(x)` | $\dfrac{x - \mathrm{mean}_C x}{\mathrm{sd}_C x}$, 0 where all are equal | the least of the candidates' |

Use `tmm` unless your own judged queries say otherwise. Its lower end is fixed by what the score is, not by the candidates: a row's value depends only on the best candidate, which is the same at any depth, while `minmax` moves every row whenever the weakest candidate changes, which it does with `depth` and with what each input happens to return. Bruch, Gai and Ingber ([TOIS 2023](https://arxiv.org/abs/2210.11934)) find this convex combination beats RRF in and out of domain once its one weight is tuned, and that a few dozen judged queries tune it; the [relevance lab](server.md) tunes it on yours ([what it gives on BEIR](results.md#score-fusion)).

`lo` is the score's own: 0 for `bm25`, a `SPARSE` similarity (weights at least 0, as learned models give them), `rrf`, `ppr` and `decay`; -1 for `similarity` and a one-vector `maxsim`, the least cosine of unit vectors (the vectors of an embedded field, and of most models; for vectors of other lengths, give `lo`); -n for `maxsim` of n vectors on a multi-vector field. Anything else, a column or an expression, needs it written: `tmm(rating, lo => 0)`. With `hi` too, `tmm(x, lo => a, hi => b)` is $(x - a) / (b - a)$, each row's own value (0 without one), fixed bounds that read nothing of the other candidates.

Options, given on any input: `depth => 100` (candidates per input), `exact => 1` (score every row that passes `WHERE` instead of the top-100 union). The top-100 shortcut recovers 89–95% of the exact fused top 10 ([results](results.md#hybrid-ranking)).

A row without a value of a column input (`0.7 * minmax(bm25(body, :t)) + 0.3 * minmax(rating)` for a row whose `rating` is `NULL`) takes that input's worst value among the candidates: its lowest when the weight is positive and its highest when the weight is negative (0 and 1 after min-max), so a missing value never lifts a row. Under `tmm` and `zscore` it takes the value in the table above. A plan with `tmm` or `zscore` needs the plan feature `score_norms`, which a version of Strata from before them refuses by name.

**Changed:** a sum of bare scores, `0.2 * bm25(body, :t) + 0.8 * similarity(emb, :q)`, was this fusion before ranking expressions; `+` now adds raw values, as in any expression below. Write `minmax()` around each term for the fusion. For one release, `EXPLAIN` and a query's plan lines add a note when an expression adds raw relevance scores. `norm => 'none'`, which kept raw scores in a sum, is accepted and changes nothing.

### Ranking expressions

`ORDER BY` takes any expression of scores, numeric columns and numbers, ranked best first (`DESC`; negate it to rank the smallest first):

```sql
-- relevance boosted by a rating, which may be missing
SELECT id FROM products ORDER BY ln(1 + bm25(body, :t)) * (1 + 0.2 * coalesce(rating, 0)) DESC LIMIT 10;

-- a business rule on top of similarity
SELECT id FROM products
ORDER BY similarity(emb, :q) + CASE WHEN in_stock THEN 0.1 WHEN category = 'outlet' THEN -0.2 ELSE 0 END DESC LIMIT 10;

-- Borda-style fusion of two rankings
SELECT id FROM papers ORDER BY -rank(bm25(body, :t)) - rank(similarity(emb, :q)) DESC LIMIT 10;
```

| part | meaning |
|---|---|
| `+ - * / %`, unary `-` | arithmetic on raw values |
| `bm25`, `similarity`, `ppr`, `decay`, `rrf`, `ranked`, a numeric column, `hops()`, `seq_no()` | the leaves: a row's value of each |
| `ln(x)`, `log(x)` (base 10), `log(b, x)`, `exp(x)`, `sqrt(x)`, `pow(x, y)`, `abs(x)`, `round(x[, digits])`, `sigmoid(x)`, `clamp(x, lo, hi)` | numeric functions |
| `least(…)`, `greatest(…)`, `coalesce(…)` | over their arguments that are not `NULL` |
| `tmm(x[, lo => a][, hi => b])` | `x` from its least possible value to the best row's, put on [0, 1]: 0 where `x` is `NULL` ([above](#weighted-sums-tmm-minmax-and-zscore)) |
| `minmax(x)` | `x` over the rows ranked, put on [0, 1]: 1 where all are equal, 0 where `x` is `NULL` |
| `zscore(x)` | `x` less the rows' mean, over their deviation; `NULL` takes the least of the rows' |
| `rank(x)` | 1 plus the number of rows ranked with a greater `x`; `NULL` places after every other |
| `CASE WHEN c THEN x … ELSE y END` | the first branch whose condition holds (any `WHERE` condition), else `y`, else `NULL` |
| `count_true(c, …)` | how many of the conditions hold ([`AT_LEAST`](#some-of-the-conditions-at_least)) |

`NULL` is a missing value (a `NULL` column, a row without a vector, one a traversal does not reach) or a result that is not a finite number (`ln(0)`, a division by 0, an overflow). Arithmetic and functions of `NULL` are `NULL`, except `coalesce`, `least` and `greatest`, which skip it. A row whose score is `NULL` ranks after every other, and its `score()` is `NULL`.

Candidates are the top `depth` (100) rows of each relevance leaf (`bm25`, `similarity`, a graph's, `ranked`), and of every input of `rrf`; an expression with no such leaf (`rating * 2 + year / 1000`), or `exact => 1` on any leaf, ranks every row that passes `WHERE`. A [rank-safe boost](#rank-safe-boosts) ranks every row its text score matches. Each leaf scores every candidate exactly, and `tmm()`, `minmax()`, `zscore()` and `rank()` are taken over the candidates. Over shards, each shard returns its candidates' leaf values and the coordinator runs the expression, so a split collection ranks like the whole one.

#### Rank-safe boosts

An expression with one text score (`bm25`, or `similarity` on a `SPARSE` field) whose other parts are numeric columns, `decay`, `CASE` conditions, numbers and `seq_no()` finds its exact top rows: the rows that scoring every row the text matches would rank first, not only the best of the text's top 100. Set functions (`tmm()`, `minmax()`, `zscore()`, `rank()`), `rrf` and `exact => 1` take the candidates above instead.

```sql
-- the 10 best by BM25 boosted by importance, even a row 5,000th by BM25 alone
SELECT id FROM docs ORDER BY bm25(body, :t) * (1 + 0.2 * importance) DESC LIMIT 10;
```

The text score's top-k loop (MaxScore, Block-Max WAND or exhaustive, as `algo` says) ranks by the whole expression. From the k-th best value so far and each column's smallest and largest value, it works out the lowest text score a row needs, and skips the rows below it. A row the text does not match is not a candidate. When the expression does not rise with the text score (`rating - bm25(body, :t)`), no text score can be ruled out: the loop scores every matching row, and the plan line says `every match scored`. Over shards, each shard returns its own exact top rows, so a split collection ranks like the whole one.

A plan that ranks by an expression needs the plan feature `rank_expressions`; a version of Strata from before it refuses the plan by that name. Weighted sums of `minmax()`, products of scores and `rrf` are written as before, so older shards still run them.

#### Rerank phases

A ranking can go on in phases, as a search engine's rank profile does: a cheap first phase over every candidate, then a costlier one over its top rows. SQL writes a phase as a `WITH` that ranks a table and cuts it, ranked again by the next `WITH` or the query itself:

```sql
-- BM25's top 100, ranked again by a mix of title and body relevance and rating
WITH top AS (SELECT * FROM docs WHERE year >= 2020 ORDER BY bm25(body, :t) DESC LIMIT 100)
SELECT id, title, score() FROM top
ORDER BY bm25(title, :t) + 0.5 * bm25(body, :t) + 0.1 * coalesce(rating, 0) DESC LIMIT 10;

-- three phases: similarity's top 500, its best 50 by a rating, then a fusion of the 50
WITH a AS (SELECT * FROM products ORDER BY similarity(emb, :q) DESC LIMIT 500),
     b AS (SELECT * FROM a ORDER BY coalesce(rating, 0) DESC LIMIT 50)
SELECT id, score() FROM b ORDER BY minmax(bm25(body, :t)) + minmax(similarity(emb, :q)) DESC LIMIT 10;
```

The planner makes one search of them, with phases. Each phase ranks the rows of the one before by an expression, best first, and may read any of the table's columns, since the rows are the table's. Every row a phase reads goes on: one the expression gives no value ranks last with a `NULL` score (and a row a `bm25` phase's terms miss scores 0), and rows tied keep the order the phase before gave them. `LIMIT` and `OFFSET` on the query cut the last phase, and `score()` is the last phase's score. The first `WITH` may filter (`WHERE`) and rank as any query does; each later step reads the step before alone, with no `WHERE` or grouping, ranks by one expression `DESC` (which may hold [correlated subqueries](#correlated-subqueries-a-ranking-that-reads-other-rows)), and each `WITH` keeps the table's columns (`*`) and has a `LIMIT`. `WITH` results before the first phase, a user's history say, are computed first, for the phases' subqueries to read. Other `WITH` queries plan as before: the table restricted to the rows of the `WITH`.

A phase whose expression is row-local is a second phase: over shards, each shard scores it on its own top rows by the first phase. A phase with a set function (`tmm()`, `minmax()`, `zscore()`, `rank()`), `rrf` or a weighted sum reads every row it reranks, so it is a global phase: it runs once over the merged rows (on the coordinator over shards), and so does every phase after it. The coordinator keeps the top rows of the whole collection by the first phase before it ranks them by the next, so a split collection answers exactly as the whole one does. `EXPLAIN` shows each phase and which kind it is, and a query's plan lines say how many rows each reranked. The first phase's `LIMIT` counts against the rows a query may reach (`max_rows`), and the phases against its deadline.

From Python, `rerank(expr, count)` adds a phase: `t.select("id", score()).order_by(bm25("body", q)).rerank(col("rating"), 100).limit(10)`. A plan with phases needs the plan feature `rank_phases`.

#### Correlated subqueries: a ranking that reads other rows

A scalar subquery in a ranking may name a column of the row it ranks (`first.cat`). It is then correlated, and gives a number for each row: how often the user met the row's category, how long ago, how close the row is to what they bought.

```sql
-- the user's last 50 events with their items' categories and embeddings, then
-- BM25's top 1,000 reranked by what the user did with each one's category
WITH h AS (SELECT e.item AS item, i.cat AS cat, i.emb AS emb, e.action AS action, e.ts AS ts
           FROM events e JOIN items i ON e.item = i.id WHERE e.usr = :u ORDER BY e.ts DESC LIMIT 50),
     first AS (SELECT * FROM items ORDER BY bm25(title, :q) DESC LIMIT 1000)
SELECT id, title, score() FROM first
ORDER BY bm25(title, :q)
       + 0.5 * (SELECT COUNT(*) FROM h WHERE h.cat = first.cat)
       - 0.0001 * coalesce((SELECT MIN(:now - h.ts) FROM h WHERE h.cat = first.cat), 100000)
       + coalesce((SELECT MAX(dot(h.emb, first.emb)) FROM h WHERE h.action = 'buy'), 0) DESC
LIMIT 20;

-- a table read through its indexes: the user's events in the row's category that cost less
SELECT id FROM items WHERE MATCH(title, :q)
ORDER BY bm25(title, :q) + 0.1 * (SELECT COUNT(*) FROM events e WHERE e.usr = :u AND e.cat = items.cat AND e.price < items.price) DESC
LIMIT 20;

-- as a column of the rows ranked
WITH first AS (SELECT * FROM items ORDER BY bm25(title, :q) DESC LIMIT 100)
SELECT id, (SELECT COUNT(*) FROM events e WHERE e.usr = :u AND e.item = first.id) AS seen FROM first
ORDER BY bm25(title, :q) DESC LIMIT 20;
```

- **Where.** The rows ranked must be a bounded set: a [rerank phase](#rerank-phases), whose rows are the phase before's top rows; an `ORDER BY` over a table, or over a `WITH` result, of at most 10,000 rows (more is an error that suggests a phase); or the columns of a ranked query, alone or in an expression (`coalesce(price, 0) + (SELECT ...)`). A subquery naming the outer row in `WHERE`, `IN` or `EXISTS` is an error.
- **What it names.** `name.column`, where `name` is the outer query's table or `WITH` result (or its alias) and the column one of its columns. The subquery reads a `WITH` result in memory, compared any way an expression over rows in memory can be, or a table through its indexes, whose conditions compare one of its columns with the outer row's (`e.cat = items.cat`, `e.price < items.price`) as they would a value.
- **Its value** is a number (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG` of numbers, or a boolean as 0 or 1). One of no rows is `NULL` (`COUNT` gives 0), which ranks last unless `coalesce`d. A `NULL` of the outer row is compared as SQL compares it: `e.price < first.price` holds for no event when the row has no price.
- **How it runs.** First the rows ranked (the phases before, or the rows a bounded `ORDER BY` reads), with the columns the subqueries name. The parts of a subquery that do not read the outer row run once (`h` above, and its rows where `h.action = 'buy'`). A subquery whose outer columns meet only equalities ANDed into its `WHERE` (`h.cat = first.cat`) then runs once, grouped by them over the values the rows hold (`h.cat IN (...)`), and a value no group has takes the subquery's value over no rows; any other runs once for each distinct value of the columns it names. Then the ranking sorts the rows, with each row's values, and the plan lines say how much ran: `correlated: 3 subqueries over 1000 rows, 597 values, 3 runs`.
- **Over shards** a coordinator runs the steps before, and the subqueries over the shards' rows, and sends each row's values to the shards inside the plan, by key, so that every shard ranks with the same ones. A plan with them needs the plan feature `correlated`.
- **Cost** at 1,000 candidates, measured with `strata-correlated-bench` on a cloud VM (4-core Intel Xeon @ 2.10GHz, 16.9 GB, Linux 6.18): 100,000 items in 200 categories, 400,000 events of 2,000 users, a history of 50 events, one thread, 300 requests of each shape. BM25's top 1,000 reranked to 20 took 1.29 ms (mean) without subqueries, and reading the history added 1.67 ms. A subquery added, over the rerank alone (with the history's read):

  | subquery | values | runs | added, mean |
  |---|---|---|---|
  | by category, over `h` | 199 | 2 | 4.90 ms |
  | three by category, over `h` | 596 | 6 | 8.14 ms |
  | by category, over the events table | 199 | 2 | 4.09 ms |
  | by item, over the events table | 992 | 2 | 21.6 ms |
  | by embedding (`dot(h.emb, first.emb)`), over `h` | 992 | 992 | 10.5 ms |

  A grouped run costs about what reading the candidates' values does. A subquery that runs once per value costs about 10 µs a run over rows in memory, and one grouped by a key reads an `IN` list as long as the candidates (`e.item IN (...)`), which is most of its cost. A second run gave the same numbers within 0.4 ms (`strata/results/correlated/`).

#### Features: the values a ranking read

A ranking expression that scores each row by itself is a column, with the value the ranking computes for the row, and `features()` lists the inputs of the query's ranking, to log beside what users chose and train a model on (Vespa's match-features):

```sql
SELECT id, score(), features() FROM docs
ORDER BY bm25(body, :t) * (1 + 0.2 * coalesce(rating, 0)) DESC LIMIT 10;
-- columns: id, score(), bm25(body, :t), rating, bm25(body, :t) * (1 + 0.2 * coalesce(rating, 0))

SELECT id, bm25(title, :t) AS title_bm25, ln(1 + rating) AS r, year / 1000 FROM docs
ORDER BY similarity(emb, :q) DESC LIMIT 10;
```

- **`features()`** stands for one column per input, each once, named by its SQL text: every leaf of the ranking (a score such as `bm25(body, :t)` or `similarity(emb, :q)`, a column, `decay(..)`, `hops()`), the first phase's first, then each phase's expression that scores a row by itself and is not a leaf. With [rerank phases](#rerank-phases) it lists every phase's. An expression with a set function (`tmm()`, `minmax()`, `zscore()`, `rank()`), `rrf` or a weighted sum ranks rows against each other, so it is not a column: its inputs are, and `score()` is its value. A ranking by `distance(..)` lists the `similarity(..)` it ranks by. `features()` takes no alias and needs an `ORDER BY`.
- **A selected expression** may use anything a ranking expression does but the set functions: `bm25`, `similarity`, `decay`, columns, numbers, arithmetic, `ln` and the other functions, `CASE`. Its value is the ranking's, bit for bit: a row a `bm25`'s terms miss scores 0, and one without a value is `NULL`. `ORDER BY 2` ranks by the second column when it is one.
- **Over shards** a feature is one database's value: a coordinator sums the shards' BM25 statistics of the terms any column scores, and its fetch round computes the columns for the page alone, with those statistics.
- A plan with such a column needs the plan feature `rank_features`. From Python, `features()` goes in `select`, and so does any expression of scores (`select("id", bm25("title", q).alias("t"))`); Python names a feature as it writes the expression.

#### Rank profiles: a ranking kept with its table

A ranking used by many queries can be kept with its table under a name, as a search engine keeps a rank profile, and queries rank by the name:

```sql
CREATE RANK PROFILE recall ON memories AS
  WITH top AS (SELECT * FROM memories ORDER BY bm25(body, :q) DESC LIMIT 100)
  SELECT * FROM top ORDER BY bm25(body, :q) * (1 + :imp * importance) DESC
  WITH (imp => 0.2);

SELECT id, score() FROM memories WHERE kind = 'fact'
ORDER BY rank_profile('recall', q => :q) LIMIT 10;

CREATE OR REPLACE RANK PROFILE recall ON memories AS ...;   -- replaces it
DROP RANK PROFILE IF EXISTS recall ON memories;            -- ON may go when one table has it
```

- **The ranking** is a query of the table that ranks every row: `SELECT * FROM table ORDER BY expression` (any [ranking expression](#ranking-expressions), or a numeric column with `ASC` or `DESC`; `CREATE` checks that every column it names is the table's), or [rerank phases](#rerank-phases) over it. It has no `WHERE`, and no `LIMIT` after its last ranking: the query that ranks by it filters and cuts. `WITH (name => value, ...)` after it gives its parameters defaults: numbers, strings, `TRUE` and `FALSE`. Names are letters, digits and `_`, in any case; `CREATE` refuses a name the table has already, and `CREATE OR REPLACE` replaces it.
- **`ORDER BY rank_profile('name', parameter => value, ...)`** is the whole `ORDER BY` of a query of the profile's table, without `ASC`, a `WITH`, a join or `GROUP BY`. The query ranks as if it were written out: its `WHERE` filters the rows the first ranking reads, and its columns, `LIMIT` and `OFFSET` are the last ranking's, so `score()` and [`features()`](#features-the-values-a-ranking-read) work as with the ranking written out. Each parameter of the ranking is the call's argument (a value, or a parameter of the query: `q => :q`), else the profile's default, else the query's parameter of the same name.
- **Kept and replicated as writes are**: a profile is logged in its own transaction (numbered where writes are, shipped to replicas), and dropped with its table. A database with a profile needs the format 3 feature `rank_profiles`, which a version of Strata from before profiles refuses by name; once the last profile is dropped and flushed, the database no longer needs it.
- **A plan names its profile** and a hash of its definition (`EXPLAIN` shows `rank profile: recall (hash ...)`) and needs the plan feature `rank_profiles`. A database whose profile of that name is another (replaced since, or a shard that missed the change) refuses the plan rather than rank otherwise. Through a coordinator, `CREATE` and `DROP RANK PROFILE` run on every shard, and the coordinator serves the profiles every replica holds alike.

#### Ranking models: `gbdt('model', features…)`

A gradient-boosted tree model, trained by LightGBM or XGBoost on the [features](#features-the-values-a-ranking-read) a query logged, ranks rows as a ranking expression of them:

```sql
SELECT id, score() FROM docs
ORDER BY gbdt('docs_ltr', bm25(title, :t), bm25(body, :t), coalesce(rating, 0), year % 6) DESC LIMIT 10;

-- as a second phase over BM25's top 200, and a binary model's probability
WITH top AS (SELECT * FROM docs ORDER BY bm25(body, :t) DESC LIMIT 200)
SELECT id, score() FROM top ORDER BY sigmoid(gbdt('clicks', bm25(title, :t), bm25(body, :t), rating)) DESC LIMIT 10;
```

- **The model** is the file `<models>/<model>/gbdt.json` in the models directory, the one embedded fields' models come from (`--models` on a server, `models=` in Python, else `~/.local/share/strata/models`): LightGBM's `booster.dump_model()` written with `json.dump` (numerical and categorical splits, any missing-value handling, a random forest too), or XGBoost's `booster.save_model("gbdt.json")` (a `gbtree`; categorical splits, multiclass and `dart` are refused). A name is letters, digits, `_`, `-` and `.`. A model is read when a query first uses it and again when its file changes.
- **Its arguments are its features**, in the order it was trained on, each any ranking expression of the row (a score, a column, arithmetic); as many as the model reads. A `NULL` feature is a missing value, which goes where the model sends missing values (LightGBM: `NaN`; or 0 under `zero_as_missing`).
- **The score is the model's raw one**, the sum of its trees, bit for bit what LightGBM's `predict(raw_score=True)` and XGBoost's `predict(output_margin=True)` give (XGBoost's in `f32`, from its base score): `sigmoid(gbdt(…))` is a binary model's probability. A random forest's score is the sum of its trees, as LightGBM's raw score is.
- **Rank it over the top rows of a cheaper ranking.** In a [rerank phase](#rerank-phases) a model is a second phase, which each shard scores on its own rows, one tree at a time over them: a LambdaMART model of 300 trees over BM25's top 100 takes 0.7 to 1.0 ms a query at the median on SciFact, NFCorpus and FiQA, against 0.02 to 0.2 ms for BM25 alone (`strata/results/text/bm25_gbdt.json`, a 4-core cloud VM). `ORDER BY gbdt(bm25(…), columns…)` alone is exact, a [rank-safe boost](#rank-safe-boosts) bounded by the leaves each tree can reach from the text score's bound and each column's range; but summed over hundreds of trees such a bound sits well above what rows score, so the top-k loop skips few matches and scores the rest with every tree: 5.5 to 70 ms on the same collections.
- **A plan names the model and the hash of its file** (`EXPLAIN` shows `ranking model: docs_ltr (hash …)`) and needs the plan feature `rank_models`. A database or shard whose file of that name is another refuses the plan rather than rank with another model; a plan without a hash runs with the file there. A coordinator scores the shards' rows with the model too, so it needs the same file in its own models directory.
- `features()` lists a model's inputs' leaves and then the model's score, and from Python `gbdt("docs_ltr", bm25("title", t), col("rating"))` builds the same ranking.

#### Cross-encoders: `cross_encoder('model', fields…, query)`

A cross-encoder reads the query and a row's text together and scores how well the text answers it. It costs thousands of times what BM25 does per row, so it reranks the top rows of a cheaper ranking, in a rerank phase of its own:

```sql
WITH top AS (SELECT * FROM docs ORDER BY bm25(body, :q) DESC LIMIT 100)
SELECT id, title, score() FROM top
ORDER BY cross_encoder('ms-marco-MiniLM-L-6-v2', title, body, :q) DESC LIMIT 10;

-- mixed with other scores of the row, and a phase after it
WITH top AS (SELECT * FROM docs ORDER BY bm25(body, :q) DESC LIMIT 100),
     ce AS (SELECT * FROM top ORDER BY cross_encoder('bge-reranker-base', body, :q) + 0.1 * coalesce(rating, 0) DESC LIMIT 20)
SELECT id, score() FROM ce ORDER BY bm25(title, :q) + decay(published, now(), 2592000) DESC LIMIT 10;
```

- **The model** is a Hugging Face directory `<models>/<model>` holding `config.json`, `tokenizer.json` and `model.safetensors`: a `BertForSequenceClassification` (such as `cross-encoder/ms-marco-MiniLM-L-6-v2`) or an `XLMRobertaForSequenceClassification` (such as `BAAI/bge-reranker-base`), with one label. It runs through candle in `f32` on the CPU, in a server or Python package built with `--features bert` (`accelerate` on a Mac), as BERT embedding models do; without it, a query by such a model is refused. A model is read when a query first uses it and again when one of its files changes. `builtin:overlap` needs no files: the share of the query's distinct words the text holds, for tests and demos.
- **The text** is the values of the fields named, stored TEXT or KEYWORD fields, joined by a space (`NULL`s left out, each value of a list), and the query is the last argument, a string or a parameter. Query and text together are cut to the model's length (512 tokens at most), the longer of the two first, as Hugging Face's tokenizers cut a pair.
- **The score is the label's logit**, what sentence-transformers' `CrossEncoder.predict` gives with an identity activation (`sigmoid()` of it is the probability for a model trained with one), equal to PyTorch's within 1e-5. A phase may mix it with other scores of the row (`cross_encoder(…) + 0.1 * rating`), and phases after it rerank the rows it kept.
- **It runs once per query, in a global phase**: on a database over the rows the phase before it kept, and through a coordinator once on the coordinator, which reads the text of the rows its shards ranked best in the fetch round of query then fetch and scores them; shards need no model. A cross-encoder is refused as the first ranking (rerank a cheaper ranking's top rows), as a column, and among `features()`.
- **Its cost is per row and per token**: `ms-marco-MiniLM-L-6-v2` takes about 50 ms a row of a few hundred tokens on four cores, so BM25's top 20 takes 0.9 to 1.0 s at the median on SciFact, NFCorpus and ArguAna, and the top 100 4.5 to 5.2 s, where BM25 alone takes 0.04 to 1.1 ms (`strata/results/text/bm25_ce_latency.json`, a 4-core cloud VM). `--max-rerank` on a server (default 20, a third of the default deadline at p99) refuses a phase that would score more rows, and the query's deadline (`--timeout-ms`, 5 s) is checked between batches of 16 rows. Its layers run on the threads the queries share (one per core), so reranks running at once take turns on the cores rather than each starting a thread per core, and queries that ask for a model not yet open wait for one of them to open it.
- **Whether it helps depends on the collection**: reranking BM25's top 100 by `ms-marco-MiniLM-L-6-v2`, a model trained on web questions, raises NFCorpus's nDCG@10 from 0.329 to 0.352, leaves SciFact's where it was (0.687 against 0.691) and lowers ArguAna's from 0.362 to 0.312; see [the results](results.md#cross-encoder-reranking). Measure it on your own queries before paying for it.
- **A plan names the model and the hash of its files** (FNV-1a of the three; `EXPLAIN` shows `ranking model: … (hash …)`), with the plan feature `rank_models`, as for [`gbdt`](#ranking-models-gbdtmodel-features). From Python, `.order_by(bm25("body", q)).rerank(cross_encoder("ms-marco-MiniLM-L-6-v2", ["title", "body"], q), 100)` builds the same query.

#### The rank log: what each query served

A ranking is trained on what it served. A query run with a log writes one row per row it answered to a log table, with the values its ranking read as it computed them, so a model trained later sees what serving saw, however the tables change in between: new rows that move BM25's statistics, counters that grow, rows deleted.

```sql
-- what the first query logged to it makes (rank_log unless the log names another)
CREATE TABLE IF NOT EXISTS rank_log (
  id KEYWORD PRIMARY KEY,                 -- '<request>#<position>'
  request KEYWORD NOT NULL, ts TIMESTAMP NOT NULL, context KEYWORD,
  row_key KEYWORD NOT NULL, position INT NOT NULL, shown BOOL NOT NULL,
  score FLOAT, features KEYWORD OPTIONS(indexed = false),
  variant KEYWORD, label FLOAT
) OPTIONS(cluster_by = 'context');

-- a label written later: a judge's grade, an agent citing the row
UPDATE rank_log SET label = 3 WHERE request = :r AND row_key = :k;
```

```sh
curl -s localhost:7700/v1/query -d '{"sql": "SELECT id, title FROM docs ORDER BY bm25(body, :t) * (1 + 0.1 * coalesce(rating, 0)) DESC LIMIT 10",
  "params": {"t": "mrna vaccine"}, "log": {"context": "u42", "variant": "b", "candidates": 50}}'
# {"columns": ["id", "title"], "rows": [...], ..., "request": "9f0c...", "logged": {"rows": 50, "seq_no": 812}}
```

- **A logged query** is one `SELECT` ranked by an `ORDER BY` ([rerank phases](#rerank-phases) and [rank profiles](#rank-profiles-a-ranking-kept-with-its-table) included) of a table with a primary key: `"log"` in the body of `POST /v1/query` or `/v1/sql` (`true`, or an object of options), `db.query_logged(sql, params, context=...)` in [Python](python.md#the-rank-log), `Db::query_logged` in Rust. It runs with `score()` and [`features()`](#features-the-values-a-ranking-read) added to its columns, answers with the columns it asked for and the request's id (`request`), and writes its rows to the log in one transaction once it has answered: a logged query costs a write.
- **Options**: `context`, whom the request served (a user, a session, a query class), which the export reads events by; `request`, the request's id (a random one when left out; logging a request's id again overwrites its rows at the same positions); `variant`, the client's name for the ranking that served (an A/B arm); `candidates`, a number of rows from the query's `OFFSET` to log, more than its `LIMIT` returns, those past the `LIMIT` logged as not shown (unshown negatives); `ts`, when it was served (ISO 8601 text or microseconds; now when left out), for replaying past traffic; `table`, another log table.
- **A row** holds its request, time and context, the key of the row served (`row_key`, as text), its `position` from 0 (the `OFFSET` included), whether it was `shown`, its `score()`, and `features`: a JSON object of each `features()` column, named by its SQL text, whose numbers read back to the bits the ranking computed. A table made by hand serves as a log if it has these columns, of these types (with `retain = '90 days'`, say).
- **Through a coordinator** the query is the coordinator's, so the values are one database's ([features over shards](#features-the-values-a-ranking-read)), and the rows go to the shards their keys place them on; the first logged query makes the table on every shard. A replica refuses a logged query, the log being a write, and so does a primary until its copies have the table (`CREATE TABLE` does not reach replicas yet). A shard's query (`"shard": true`) is not logged.
- **Cost.** Computing the values is nearly free, since the ranking computes them anyway; the write is the cost. On a 4-core cloud VM over HTTP, with 100,000 documents and one client, a query took 1.87 ms at the median plain, 1.85 ms with its values added and 4.33 ms logged (7.81 ms logging 100 candidates); writing the same ten rows alone through `/v1/write` took 1.97 ms. Without the disk's flush (`--no-sync`) the logged query took 3.50 ms: the sync is a third of the log's cost here, and the rest is the write itself. Four clients got 1,620 plain and 670 logged queries a second. A row took 152 bytes in the segments (an Intel Xeon @ 2.10GHz, 16.9 GB, Linux 6.18; `strata-server/results/rank_log/vm.json` and `vm-no-sync.json`, from `strata-rank-log-bench`). The export read 70,000 records a second, each with a history of 20 events and a label.

#### Exporting the log for training

The export reads a log table back from one snapshot, one record per logged row, with what happened around it as of its time, from a table of events (a user's views, clicks, purchases): the context's last events before the row (its history, for a sequence model), and a label from the events on the row's key after it, within a window.

```sh
curl -s localhost:7700/v1/rank_log/export -d '{"events": {"table": "events", "user": "usr", "item": "item", "time": "ts", "action": "action"},
  "history": 20, "window": "30 minutes", "grades": {"click": 1, "buy": 3}, "since": "2026-10-01"}'
# {"id": "9f0c...#0", "request": "9f0c...", "ts": 1793338200000000, "context": "u42", "row_key": "d17", "position": 0, "shown": true,
#  "score": 7.31, "variant": "b", "label": 3.0,
#  "features": {"bm25(body, :t)": 6.09, "bm25(body, :t) * (1 + 0.1 * coalesce(rating, 0))": 7.31, "rating": 2.0},
#  "history_items": ["d3", "d9", ...], "history_times": [...], "history_actions": ["view", "click", ...]}
```

- **Records** come as NDJSON from `POST /v1/rank_log/export`, as a list of dicts from `db.export_rank_log(...)` in Python, and from `Db::export_rank_log` in Rust, context by context, each context's rows in the order they were served. Times are microseconds since the epoch. `since` and `until` keep the rows served in `[since, until)`, and `table` names another log.
- **`events`** names a table, its column of contexts (`user`, `KEYWORD` or `INT`), of the rows the events were about (`item`), of when they happened (`time`, a `TIMESTAMP` or an `INT` of microseconds), and optionally of what they were (`action`). `OPTIONS(shard_by = 'usr', cluster_by = 'usr')` keeps each user's events together, on one shard and in few pages.
- **History**: with `history` set to n, the context's last n events before the row's time, oldest first. **Label**: with `window`, a period (`'30 minutes'`) or seconds, the best grade among the context's events on the row's key in the window after the row's time, `grades` giving each action's (an event whose action has none is not a label; without `grades` every event grades 1), and 0 when there is none. A label written on the row wins. A row without a context has no history and no label but its own.
- **As of the row's time**, by the events' own times: an event timed before a row is in its history even when it arrived after the row was served, which a query at the time could not have seen. The features are the values served, never recomputed.
- **A coordinator does not export**: export from each shard's server. Logs placed by context (a hand-made log with `context KEYWORD NOT NULL` and `shard_by = 'context'`) and events placed by user then give each shard every row and event of its contexts.

#### Sequence models: `encode('model', history)`

A sequence model reads what a user did, newest first, and gives a vector in the space of an item field, so that the items it ranks first are the ones the user is likely to want next:

```sql
-- the next item for a user, from their last 200 events
SELECT id, score() FROM items
WHERE NOT id IN (SELECT item FROM events WHERE usr = :u)
ORDER BY similarity(emb, encode('next', (SELECT item, action, ts FROM events WHERE usr = :u ORDER BY ts DESC LIMIT 200), now => :now)) DESC
LIMIT 10;

-- as a feature of a ranking, next to the text's
WITH top AS (SELECT * FROM items ORDER BY bm25(title, :q) DESC LIMIT 200)
SELECT id, score() FROM top
ORDER BY bm25(title, :q) + 2 * coalesce(dot(emb, encode('next', (SELECT item, ts FROM events WHERE usr = :u ORDER BY ts DESC LIMIT 200))), 0) DESC
LIMIT 10;
```

- **The history** is a subquery of the user's events, newest first: its first column is an item's primary key in the model's table, then optionally what the event was (a text column: `view`, `buy`, a rating) and when (a `TIMESTAMP`, or a number of microseconds since the epoch, as the [rank log's export](#exporting-the-log-for-training) writes them). With two columns the second is the time when it is a `TIMESTAMP` and the action when it is text; a number there is refused, since it could be either, so give microseconds as the third column (`SELECT item, NULL, micros`). Order it by time, descending: a row newer than one above it is an error rather than a history read backwards. Of its rows whose item has a vector, the model reads the newest, up to the length it was trained on, as training reads a history. `now =>` is the time of the request (a `TIMESTAMP`, microseconds or ISO 8601 text; the clock's by default): the time since the newest event is part of what the model reads.
- **Where it goes**: `similarity(field, encode(…))` ranks the field's nearest items (the dot product with the model's vector: a search of the field's index, its best scored exactly, as [`maxsim`](#maxsimfield-vectors-several-query-vectors) runs one), `maxsim(field, encode(…))` the same as one query vector, and `dot(field, encode(…))` is a number in any ranking expression. A user with no events has no vector: `similarity` and `maxsim` then return no rows, and `dot` is `NULL`. The history runs once a query, however often `encode` appears in it.
- **The model** is a small causal transformer (SASRec's): each event is its item's vector, projected, plus a learned vector for its action, for the time until the next event (in powers of two of seconds) and for its position; two layers of attention over the events, each seeing only the ones before it, then the last event's output, projected into the item field's space and normalized. It is the directory `<models>/<model>/` (`config.json`, `model.safetensors` and `items.safetensors`), in the models directory [`gbdt`](#ranking-models-gbdtmodel-features)'s files come from. The config names the table and the vector field the model was trained on and their dimension: a database without that field, or with one of another dimension, refuses the model, and so do `similarity` and `maxsim` over another field. `items.safetensors` holds the item vectors it was trained with, and is part of the model's version (its hash): a history whose items have other vectors in the field is refused, naming one, since the model's vector would be in a space it never saw. An item the file does not list (one added since) is read with the field's vector.
- **Training** is `strata.train.sequence` in Python, with PyTorch (`pip install torch`), from a table of events, the rank log's export, or lists of `(item, action, time)`:

  ```python
  import strata, strata.train
  db = strata.Database.open("shop.strata")
  out = strata.train.sequence(db, "next", table="items", field="emb",
                              events={"table": "events", "user": "usr", "item": "item", "action": "action", "time": "ts"},
                              learn_items=True)
  strata.train.write_vectors(db, "next")   # with learn_items, the items' learned vectors into items.emb
  ```

  It trains next-item prediction over each user's events, with every item in the field (`negatives="all"`) or the other targets of a batch (`"in_batch"`, for many items) as the negatives, and writes the model where `models=` says (the models directory by default), with the item vectors it was trained with. With `learn_items=True` the item vectors are trained too, starting from the field's: `strata.train.write_vectors` writes them into the field, changing nothing else of each row, and until then `encode` refuses the model. Otherwise the field's vectors are kept as they are and only the model learns.
- **Equal to PyTorch's**: Strata's vector is the one the PyTorch module gives for the same history, within 2·10⁻⁵ (`strata/tests/sequence.rs`), and the items a query ranks are the ones PyTorch's vector ranks.
- **How well it ranks**: on MovieLens-1M and Amazon Beauty, leave-one-out over every item as SASRec is measured, a model of two layers that reads items only has Recall@10 0.278 and 0.066 (NDCG@10 0.159 and 0.041). That is above the original SASRec's published 0.250 and 0.061, but below SASRec trained the same way as this model, with cross-entropy over every item (SASRec+, 0.315 on MovieLens), and on Beauty below counted transitions from the same events ([`strata.next_items`](next-items.md), 0.069): good, not the best published. Reading times too, it has 0.319 and 0.077. Both MovieLens runs stopped at a time limit while still improving. Strata's answers equal PyTorch's on every user ([`results/sequence`](https://github.com/rkonow/strata/tree/main/strata/results/sequence), with the protocol and the published numbers), and CI checks ranking quality on MovieLens-1M against popularity and item-kNN baselines and Strata's other sources (`strata/results/sequence/gate.py`): there a model trained for an hour has Recall@10 0.306, `ppr` over what came after what 0.242, counted transitions 0.211 and item-kNN 0.069.
- **What it costs**: the model alone, on one thread of a cloud VM's Xeon at 2.80 GHz, takes 0.09 ms for 10 events, 0.28 ms for 50 and 1.5 ms for 200 (p50); the whole query, reading the history and searching the field, 0.6 to 0.7 ms on Beauty's 50 events and 2 to 3 ms on MovieLens's 200. Keep `max_len` as short as the ranking allows.
- **A plan names the model and the hash of its files** and needs the plan feature `encode`; a database or shard whose files of that name are others refuses it. A coordinator runs `encode` itself, from the history its shards return, so it needs the model; the shards don't.

### `decay(field, origin, scale, …)`: closeness to a value

A numeric column's closeness to `origin`, as in Elasticsearch's decay functions. With `d = max(0, |x − origin| − offset)`, the score is 1 at `d = 0` and `decay` at `d = scale`:

| `shape` | score |
|---|---|
| `'exp'` (default) | `decay^(d / scale)` |
| `'gauss'` | `decay^((d / scale)²)` |
| `'linear'` | `max(0, 1 − (1 − decay) · d / scale)`, reaching 0 at `scale / (1 − decay)` |

Options: `offset => 0`, `decay => 0.5`, `shape => 'exp'`. Rows without a value score 0. `now()` as the origin is the current Unix time in seconds, so `decay(published, now(), 86400)` halves a row's score for each day of age. `decay` ranks on its own, inside `rrf` and weighted sums, and in products.

### Products: boosting relevance by a row's values

Multiplying scores boosts relevance by what a row holds:

```sql
-- fresh news first: BM25 halved at 30 days old
SELECT id FROM news ORDER BY bm25(body, :t) * decay(published, now(), 2592000) DESC LIMIT 10;

-- similar products, weighted by their rating
SELECT id FROM products ORDER BY similarity(emb, :q) * rating DESC LIMIT 10;
```

A product's factors are scores (`bm25`, `similarity`, `ppr`, `decay`) and numeric columns, one of them a relevance score (`bm25`, `similarity`, a graph's); a product of a row's own values alone (`rating * clicks`, `decay(..) * rating`) is arithmetic, and ranks as the numbers multiply. Candidates are the top `depth` (100) rows of each relevance factor, scored exactly by every factor; a product of one text score and row values (`bm25(..) * decay(..)`, `bm25(..) * rating`) is a [rank-safe boost](#rank-safe-boosts) and ranks every row the text matches. A factor below 0 (a negative similarity) or missing (a `NULL` column) counts as 0, so the product of two negatives cannot rank high. Over shards, each shard returns its factors' raw scores and the coordinator multiplies them, so a split collection ranks like the whole one.

### A column

`ORDER BY year DESC` orders by a numeric column. Rows without a value come last in either direction (as in Elasticsearch); `ORDER BY year DESC NULLS FIRST` puts them first. Their `score()` is `NULL`. With no `ORDER BY`, rows come in storage order.

With a `LIMIT`, the rows can come from the column's blocks of 128 rows, best block first by its zone map (its largest value for `DESC`, smallest for `ASC`), stopping at the first block whose best value is worse than the k-th row found. A row's filter is checked only when its value can still make the cut, and blocks the filter's own zone maps rule out are never read. So the ten most clicked of the 560,000 ORCAS queries starting with `a` read about 150 of 81,000 blocks, and about 10 when the table is clustered by query (`OPTIONS(cluster_by = 'id')`), where each prefix's rows sit together. The planner keeps the plain scan of the matches when they are a few thousand rows, or under 1 in 64 of the rows the zone maps admit (a rare prefix of an unclustered table, whose k-th best value is low enough that most blocks could hold a better one), and switches to it when the blocks have cost half of what it would (a filter that rejects the best values, `WHERE n < 3 ORDER BY n DESC`); `NULLS FIRST` always scans. Values written in place are read first, so counters kept with `UPDATE ... SET n = n + 1` rank by their current value.

## Typos and synonyms

`MATCH` and `bm25` take two options that widen a search beyond the words written:

```sql
SELECT id FROM papers WHERE MATCH(body, 'nueral netwrok', fuzzy => 'auto') LIMIT 10;
SELECT id, score() FROM papers ORDER BY bm25(body, 'nueral netwrok', fuzzy => 1) DESC LIMIT 10;
SELECT id FROM papers WHERE MATCH(body, 'gnn', synonyms => 1) LIMIT 10;
```

- **`fuzzy => 1` or `2`** also matches the indexed terms within that many edits of each word, counting an insertion, deletion, substitution or swap of two neighbouring letters as one edit (optimal string alignment). **`fuzzy => 'auto'`** allows no edits for words of 1 or 2 letters, 1 for 3 to 5 letters and 2 for longer ones. In the `MATCH` grammar, `word~` is `'auto'` for one word and `word~1`, `word~2` choose. A ranked query adds the 50 nearest terms of each word, each weighted `1 − edits / length` (at least 0.1), as Lucene's fuzzy queries do. A word and its typo neighbours share one document frequency, the largest among them, so a rare misspelling scores below the word rather than above it; the expansion is found once per query over every segment's dictionary, so all segments, and the plan line `bm25 expanded with N terms`, agree. Each dictionary is walked as a trie, skipping every run of terms whose start is already too many edits away, rather than tested term by term: for a word of a typed query looked up in the 797,472 analysed terms of ORCAS's 10.4 million queries, the median lookup takes 2.1 ms with one edit and 16 ms with two, against 79 and 100 ms testing every term (one thread of a 4-core cloud VM, Intel Xeon @ 2.10GHz; [`results/text/fuzzy-orcas.json`](https://github.com/rkonow/strata/tree/main/strata/results/text), `strata-fuzzy-bench`).
- **`synonyms => 1`** also matches what the server's synonym list says a word or phrase means. The list is in Solr's format: `laptop, notebook, portable computer` makes each find the others, and `tv => television` makes `tv` find `television` but not the reverse. Both sides are analysed like documents, so `laptops` finds `notebook`. A one-word synonym of a word shares that word's document frequency, as Lucene's synonym queries do; the words of a longer synonym score on their own. [strata-server](server.md) loads it from `--synonyms FILE` and edits it at `/v1/synonyms`; from Rust, set `ExecOptions::synonyms`. Without a list, `synonyms => 1` is an error that says so.

## Aggregates

```sql
SELECT category, COUNT(*) FROM docs
WHERE MATCH(body, 'india') AND year >= 2000
GROUP BY category ORDER BY 2 DESC LIMIT 5;

SELECT category, topic, COUNT(*) AS n, AVG(rating), MIN(year), MAX(year), COUNT(DISTINCT source)
FROM docs WHERE year >= 2000
GROUP BY category, topic HAVING n > 20 ORDER BY AVG(rating) DESC LIMIT 10;

SELECT COUNT(*), SUM(clicks), AVG(rating) FROM docs WHERE MATCH(body, 'india');
```

- **`GROUP BY`** takes one or more indexed `KEYWORD`, `INT` or `FLOAT` columns, by name, position or alias. A multi-valued column puts a row in one group per value. The rows without a value form the group `NULL`, as in SQL, and `SELECT DISTINCT` lists `NULL` once.
- **Aggregates** are `COUNT(*)`, `COUNT(f)`, `COUNT(DISTINCT f)`, `SUM(f)` and `AVG(f)` over numeric columns, and `MIN(f)` and `MAX(f)` over numeric or keyword columns. Every selected column is a group column or inside an aggregate. Without `GROUP BY` they summarize every row that passes `WHERE`, in one row (`COUNT(*)` 0 and the others `NULL` when none does).
- **`HAVING`** filters groups by their aggregates, written out or by alias, with comparisons, `BETWEEN`, `AND`, `OR` and `NOT`, in three-valued logic: an aggregate of no values (`AVG(rating)` of a group without ratings) is `NULL`, so neither `AVG(rating) < 3` nor `NOT (AVG(rating) < 3)` keeps that group.
- **`ORDER BY`** a group column, an aggregate or a position, with a tie-break; without one, the largest groups come first. `LIMIT` and `OFFSET` apply to groups.

A coordinator over shards merges partial aggregates (sums and counts for `AVG`, the distinct values for `COUNT(DISTINCT)`), so the answers are the same as on one server.

## Subqueries, WITH and joins

```sql
-- IN (SELECT ...): rows of the categories that 2001's surveys are in
SELECT id FROM papers WHERE category IN (SELECT category FROM papers WHERE year = 2001 AND kind = 'survey');

-- a scalar subquery
SELECT id, year FROM papers WHERE year > (SELECT AVG(year) FROM papers) ORDER BY year DESC LIMIT 10;

-- WITH: name a query and read it like the table
WITH top AS (SELECT id FROM papers ORDER BY bm25(body, 'graph neural network') DESC LIMIT 50)
SELECT category, COUNT(*) FROM top GROUP BY category;

-- a WITH result as graph seeds
WITH top AS (SELECT id FROM papers ORDER BY bm25(body, 'graph neural network') DESC LIMIT 20)
SELECT id FROM papers WHERE REACHABLE(cites, (SELECT id FROM top), dir => 'in') LIMIT 100;

-- FROM (SELECT ...) and DISTINCT
SELECT category, n FROM (SELECT category, COUNT(*) AS n FROM papers GROUP BY category) AS t WHERE n > 50 ORDER BY n DESC;
SELECT DISTINCT category FROM papers WHERE MATCH(body, 'graph');

-- joins of two tables: on a key, a multi-valued column (each of its values), or three tables
SELECT p.id, v.name FROM papers p JOIN venues v ON p.venue = v.id WHERE v.rank >= 4 AND MATCH(p.body, 'graph');
SELECT a.name, COUNT(*) AS n FROM papers p JOIN authors a ON p.authors = a.id GROUP BY a.name ORDER BY n DESC LIMIT 10;
SELECT p.id, a.name, v.name FROM papers p JOIN authors a ON p.authors = a.id LEFT JOIN venues v ON p.venue = v.id WHERE p.year = 2024;
SELECT id FROM papers WHERE venue IN (SELECT id FROM venues WHERE field = 'ir');

-- joins of the table with itself: along an EDGES field, or on a shared value
SELECT a.id, b.id AS cited FROM papers a JOIN papers b ON a.cites = b.id WHERE a.year >= 2020;
SELECT a.id, b.id AS cited_by FROM papers a LEFT JOIN papers b ON b.cites = a.id WHERE a.id IN ('p1', 'p2');
SELECT b.id, COUNT(*) AS n FROM papers a JOIN papers b ON a.cites = b.id GROUP BY b.id ORDER BY n DESC LIMIT 10;
SELECT a.id, b.id FROM papers a JOIN papers b ON a.category = b.category AND b.id != a.id WHERE a.id = 'p1';
```

Each plans into operators over the single-table queries above ([`EXPLAIN`](#explain) shows them), all run on one snapshot, so every single-table query uses the indexes. Each is bound as it is built, never printed as SQL and parsed again:

- **`IN (SELECT ...)`** and **`NOT IN`** run the subquery first and filter with its values (a multi-valued column contributes each value). **`EXISTS`** is true or false, and a **scalar subquery** must return one column and at most one row. A subquery's values reach the outer query's filter as values, so its indexes serve it: `year > (SELECT AVG(year) ...)` is a range of `year`. A scalar subquery that names a column of the outer row is [correlated](#correlated-subqueries-a-ranking-that-reads-other-rows): it ranks, and is not a condition.
- **`WITH`** results and **`FROM (SELECT ...)`** are read like a table. A `WITH` whose rows are rows of the table (no aggregates) keeps their keys, so an outer query over it reads the table restricted to those rows, in the `WITH`'s order unless it orders them itself, and `(SELECT id FROM top)` can seed a graph. `WITH RECURSIVE` is not supported: `REACHABLE` and `PATH` walk links.
- **Joins** are `JOIN` (inner) and `LEFT JOIN`, of a table with another, with itself or with a `WITH` result; name each side (`papers a JOIN papers b`). `ON` needs an equality between the sides, of two columns of one kind: `KEYWORD`, or `INT` and `FLOAT` compared as numbers (`p.venue = a.id`, an `INT` with a `KEYWORD`, is an error). A multi-valued column joins each of its values. Comparing an `EDGES` field with the key follows its links (`a.cites = b.id`: *b* is a row *a* links to; `b.cites = a.id`: *b* links to *a*); links name keys of their own table, so they join a table with itself. Each `WHERE` or `ON` condition on one side filters that side's scan, and the second side reads only the rows whose join values the first side holds (`b.id IN (...)`, or for links the rows linking to them), before a hash join in memory. Text, vector and graph functions name one side (`MATCH(a.body, ...)`); `ORDER BY` a score ranks by that side's scan, and when the order is the first side's, scans of a growing number of its rows stop once `LIMIT` rows join. Joined columns are named by their expression (`b.title`) unless aliased.
- Over joined or derived rows, `WHERE`, `GROUP BY`, aggregates, `HAVING`, `ORDER BY`, `DISTINCT` and `LIMIT` also take arithmetic, `||`, `LIKE`, `ILIKE`, `CASE`, `COALESCE`, `LOWER`, `UPPER`, `LENGTH`, `ABS`, `ROUND`, `DOT` and the numeric functions of ranking expressions (`ln`, `pow`, `least`, ...). So does the `SELECT` list over one table (`SELECT id, rating * 2, lower(category), dot(emb, :q) FROM papers WHERE ...`): the table's query finds the rows and the columns are computed in memory, typed as over derived rows. A `SELECT` that also selects a ranking's value (`bm25(..) * 2`, `hops()`) computes its columns as ranking expressions do, as `FLOAT`s. Their conditions follow the same three-valued logic as a single table's, multi-valued columns included (`NOT (b.tags = 'red')` keeps rows without tags). Integer arithmetic is exact, and a result too big for an integer is a `FLOAT`; `/` gives a `FLOAT`, and dividing by zero gives `NULL`. `||` skips `NULL`s, and `LIKE` with a `NULL` side is `NULL`. `ORDER BY` puts rows without a value last in either direction unless `NULLS FIRST`, as over a single table.
- Each single-table query, `WITH` result and side of a join holds at most 1,000,000 rows. `UNION`, `INTERSECT`, `EXCEPT`, window functions, and `RIGHT`, `FULL` and `CROSS` joins are not supported.

The executor's plan lines (`--plan`, `/v1/explain`'s `plan`) list each single-table query that ran, with its rows and its own plan.

## The first row of each group: DISTINCT ON

`SELECT DISTINCT ON (keys) ...` keeps, of the rows with equal keys, the first in the query's order, as PostgreSQL does: a row per group, the groups in the order of their first rows, and `LIMIT` and `OFFSET` count groups. With rows of a long video or document in a table of segments, it answers each video once, by its best segment:

```sql
-- each video's best clip, the 10 best videos
SELECT DISTINCT ON (video) video, id, start_ms, score() FROM clips
WHERE MATCH(transcript, 'penalty kick') ORDER BY bm25(transcript, 'penalty kick') DESC LIMIT 10;

-- each user's latest event
SELECT DISTINCT ON (usr) usr, item, ts FROM events ORDER BY ts DESC, id;

-- the best page of each PDF, by an image query's similarity
SELECT DISTINCT ON (doc) doc, page FROM pages ORDER BY similarity(emb, :q) DESC LIMIT 5;

-- over groups: each country's top seller
SELECT DISTINCT ON (country) country, seller, SUM(price) AS total FROM sales GROUP BY country, seller ORDER BY country, total DESC;
```

- **Keys** are expressions of the rows (`video`, `lower(lang)`, `year % 10`), or columns of the answer by name or number (`DISTINCT ON (2)`), and need not be selected. Keys are equal when their values are: `NULL` with `NULL`, and a multi-valued column's list with the same list.
- **The order.** PostgreSQL requires `ORDER BY` to start with the keys; Strata does not, so `ORDER BY` a score gives the best row of each group, the best groups first. When `ORDER BY` does start with the keys, the answer is PostgreSQL's. Without `ORDER BY`, which row of a group is kept is not defined.
- **Reads.** When the query's order is the scan's (a score, or one numeric column), a `LIMIT` reads the scan in growing slices: four times the rows asked for (at least 32), then four times as many, until that many groups have a row or the scan runs out. A group with many rows near the top costs more reads, never a wrong answer. `DISTINCT ON` one keyword or `INT` field of a table ranked by `bm25` reads less: each segment keeps each group's first row as it scores ([`MAX k PER`](#at-most-k-rows-of-a-value-max-k-per) does the same with k rows), so the scan returns the rows asked for. Ordered by several columns, the rows are sorted in memory, and the scan holds at most 1,000,000 rows.
- **Shards.** A coordinator keeps the first rows over its shards' merged rows, so three shards give what one database does, whether or not the table is placed by the keys (`shard_by`).

The plan's node is `DistinctOn` (`Distinct on: video` in `EXPLAIN`), and its JSON needs the plan features `operators` and `distinct_on`; a version of Strata from before it refuses the plan by that name.

## At most k rows of a value: `MAX k PER`

`ORDER BY ... MAX k PER field` keeps, of the rows with equal values of `field` (a `KEYWORD` or a number of one value: a `TEXT` field is an error, as rows rarely share a whole text), the first `k` in the query's order, and the rows kept stay in that order: a ranking with at most two clips of a video, three products of a brand or two arguments of a debate. `LIMIT` and `OFFSET` count the rows kept.

```sql
-- the 20 best clips, at most two of a video
SELECT id, video, score() FROM clips ORDER BY bm25(transcript, :q) DESC MAX 2 PER video LIMIT 20;

-- the newest products in stock, at most three of a brand
SELECT id, brand, title FROM products WHERE in_stock ORDER BY added DESC, id MAX 3 PER brand LIMIT 30;

-- in a subquery: of each video's two most viewed clips, those about a dog
SELECT id FROM clips WHERE MATCH(transcript, 'dog')
  AND id IN (SELECT id FROM clips ORDER BY views DESC, id MAX 2 PER video LIMIT 1000);
```

- **It ends `ORDER BY`**, and needs one: the rows a value keeps are its first in that order. `k` is a whole number, 1 or more, and `MAX 1 PER x` is `DISTINCT ON (x)`, with the same plan.
- **A field of one value.** `field` is a field of the rows (`video`, `t.brand` in a join), and `NULL` is one value of its own. A multi-valued field is refused, since a row with several values would be in several groups.
- **Not with `DISTINCT`** or `DISTINCT ON`, which already keep one row of each.
- **Reads.** A table's own rows ranked by `bm25` and capped by a keyword or `INT` field keep the cap as they are scored: each segment keeps each value's best k rows (and the rows tied with its k-th) and skips the rows that can no longer reach the `LIMIT`, so the query reads its rows once. Other rankings, and caps over joined or derived rows or by other fields, read growing slices of the ranking until `LIMIT` rows are kept or it runs out, as [`DISTINCT ON`](#the-first-row-of-each-group-distinct-on) does. A field whose values cannot fill the `LIMIT`, such as one row of each of 15 topics for a `LIMIT` of 20, reads every row that matches either way, and returns 15.
- **Shards.** A coordinator keeps each value's first rows over its shards' merged rows, so three shards give what one database does.

The plan's node is `DistinctOn` with `per` (`At most 2 per: video` in `EXPLAIN`). With `k` above 1 its JSON needs the plan feature `per_cap`, so a version of Strata from before it refuses the plan rather than keeping one row of each value. Python's builder writes it `.max_per(2, "video")`.

On ArguAna (BEIR's 8,674 arguments from 592 debates on 15 topics; a document's debate and topic are in its id), every one of the 1,406 queries ranked by `bm25(body, :q)`, its own argument left out as BEIR scores it, top 20. Times are the median over queries of each query's median of 3 runs, one thread on a 4-core cloud VM (Intel Xeon @ 2.10GHz):

| top 20 | nDCG@10 | debates | topics | p50 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| uncapped | 0.489 | 7.5 | 3.8 | 1.6 | 4.1 |
| `MAX 1 PER debate` | 0.307 | 20.0 | 7.2 | 2.4 | 5.4 |
| `MAX 2 PER debate` | 0.379 | 14.5 | 5.9 | 2.3 | 5.0 |
| `MAX 3 PER debate` | 0.419 | 12.6 | 5.4 | 2.3 | 5.5 |
| `MAX 1 PER topic` | 0.301 | 14.9 | 15.0 | 2.8 | 5.7 |
| `MAX 2 PER topic` | 0.376 | 16.3 | 11.4 | 2.6 | 5.4 |
| `MAX 3 PER topic` | 0.416 | 14.2 | 9.1 | 2.4 | 5.4 |

Debates and topics are the mean number of each in a query's answer. Each segment keeps the cap as it scores, so a cap costs about 1.5 times the uncapped query whatever the field: BM25 can skip fewer rows, since the 20th row kept scores lower than the 20th of all. One row of each of the 15 topics cannot fill 20 rows, so that query scores every argument that matches, once, and returns 15. Before the scan kept the cap, the same queries read growing slices of the ranking: 6.0 to 6.1 ms by debate, 7.1 and 18.8 ms for three and two per topic, and 549 ms for one per topic, which read every row again at each slice (with the uncapped query at 3.3 ms in that run).

Caps trade relevance for spread. ArguAna's answer to a query is the argument against it, in the query's own debate, so a cap on debates also caps how many of the right debate's arguments the ranking can show: nDCG@10 falls from 0.489 to 0.307 with one per debate, and to 0.419 with three. The run is `strata-per-cap-bench` ([`results/text/per_cap.json`](https://github.com/rkonow/strata/blob/main/strata/results/text/per_cap.json)).

## Graphs

Rows with an `EDGES` field are nodes of a graph. Graph predicates and scores take **seeds**: a key (`'p1'`), a parameter, a tuple of keys (`('p1', 'p2')`), or a subquery, whose filters, ranking and `LIMIT` are those of any query. A subquery over the same table seeds with its rows; one over another table returns one column, whose values are the keys to start from, such as the items a user's recent events name: `ppr(cooc, (SELECT item FROM events WHERE usr = :u ORDER BY ts DESC LIMIT 20))`. With `weighted => 1`, a seed's restart weight is its row's score, summed over the rows that name it: `ppr(cooc, (SELECT item FROM events WHERE usr = :u ORDER BY w DESC), weighted => 1)` starts from each item in proportion to the sum of its events' `w`, and ordering the events by `decay(ts, ...)` instead weighs recent ones more.

```sql
-- papers within two citations of p1, either way, nearest first
SELECT id, title, hops() FROM papers
WHERE REACHABLE(cites, 'p1', hops => 2, dir => 'both') ORDER BY hops(), id LIMIT 20;

-- one shortest path, only through surveys
SELECT id, hops() FROM papers
WHERE PATH(cites, 'p1', 'p9', max_hops => 6, through => kind = 'survey') ORDER BY hops();

-- vector search as the way in: what the two nearest papers cite, by similarity
SELECT id FROM papers
WHERE REACHABLE(cites, (SELECT id FROM papers ORDER BY similarity(emb, :q) DESC LIMIT 2))
  AND NOT REACHABLE(cites, (SELECT id FROM papers ORDER BY similarity(emb, :q) DESC LIMIT 2), hops => 0, min_hops => 0)
ORDER BY similarity(emb, :q) DESC LIMIT 20;

-- personalized PageRank from the same seeds, fused with BM25
SELECT id FROM papers
ORDER BY rrf(bm25(body, :t), ppr(cites, (SELECT id FROM papers ORDER BY similarity(emb, :q) DESC LIMIT 5))) DESC LIMIT 20;
```

| function | what it is | options (defaults) |
|---|---|---|
| `REACHABLE(f, seeds, …)` | rows *v* with `min_hops` ≤ *d*(seeds, *v*) ≤ `hops` | `hops 1`, `min_hops 1`, `dir 'out'` (or `'in'`, `'both'`), `label 'a,b'`, `through <predicate>`, `max_nodes 100000` |
| `PATH(f, from, to, …)` | the rows of one shortest (unweighted) path | `max_hops 6`, `dir`, `label`, `through`, `max_nodes` |
| `hops()` | a row's distance from the seeds, or its position on the path; `ORDER BY hops()` is nearest first | |
| `ppr(f, seeds, …)` | personalized PageRank, usable wherever a score is, including `rrf` and weighted sums | `alpha 0.15`, `eps 1e-4`, `dir 'both'`, `label`, `max_pushes`, `weighted 0` (`1`: the seed subquery's scores are the restart weights) |
| a `VECTOR` field as `f` | `REACHABLE` and `PATH` follow similarity edges: a row's neighbours are its `k` most similar rows | `k 10`, `min_sim` |

The [query console](server.md#the-query-console)'s Graph tab builds these queries (seeds from keys, a search or a condition; links or similarity; neighbourhood, path or PageRank) and draws the answer.

`through` restricts the rows a traversal may pass through (seeds and end points are exempt); the rest of `WHERE` filters the rows it ends at. `max_nodes` bounds the rows a traversal meets besides its seeds (and a path's ends): a `REACHABLE` that would meet more, or a `PATH` that meets more before it finds a path, is an error saying so, never a part of the answer. `max_hops => 0` is a path of no hops (a seed that is also a target); `max_nodes` and `k` are 1 or more. `REACHABLE(f, seeds, hops => 0, min_hops => 0)` is the seed set itself, so `NOT` of it removes the seeds from a result. `hops()` refers to the first `REACHABLE` or `PATH` in `WHERE`. The [Graphs](graphs.md) page explains how edges are stored and what these queries cost.

Edges are written as part of their row: a JSON array of target keys, each optionally with a label and a positive weight.

```text
["p2", ["cites", "p7"], ["extends", "p9", 0.5], {"to": "p11", "label": "refutes"}]
```

### Edge tables

An `EDGES` field belongs to its source row, so adding one edge rewrites the row, its text and its vectors. A graph kept up as events arrive, such as items seen together, goes in an **edge table** instead: a table whose rows are edges, declared with `OPTIONS(edges = 'src, dst, w')`. A walk over the table whose keys it holds names the edge table where it would name a field:

```sql
CREATE TABLE cooc (
  id KEYWORD PRIMARY KEY,                  -- 'src>dst'
  src KEYWORD NOT NULL REFERENCES items,
  dst KEYWORD NOT NULL REFERENCES items,
  w FLOAT DEFAULT 0
) OPTIONS(edges = 'src, dst, w');

SELECT id FROM items
ORDER BY ppr(cooc, (SELECT item FROM events WHERE usr = :u ORDER BY ts DESC LIMIT 20)) DESC LIMIT 200;
```

- Each row is an edge from the row whose key is in `src` to the row whose key is in `dst`, weighted by `w`, or by 1 without a weight column (`edges = 'src, dst'`). `src` and `dst` are stored, single-valued `KEYWORD` or `INT` columns of the type of the other table's key, and the weight is a stored `FLOAT` or `INT`. If `src` references a table, walks are over that table.
- Writing an edge writes its row and nothing else: insert it, delete it, or add to its weight with an `add` operation, whose `doc` creates the row on the edge's first event ([writes](server.md#writes)).
- Edges are directed, from `src` to `dst`. For a click on item *i* after the user's last items *h*₁ … *h*₅ (*h*₁ the most recent), adding 1/*g* to the edge *h*_g → *i* records what comes after what: `ppr(cooc, :x, dir => 'out')` and `REACHABLE(cooc, :x, dir => 'out')` follow the items that came after *x*, and `dir => 'both'`, `ppr`'s default, reads them as seen together. That is five `add` operations per event. For the next item alone, counts of what came after the last one or two items are cheaper and exact ([Next items](next-items.md)).
- An edge is followed while its weight, as it is now, is above 0. `UPDATE cooc SET w = w * 0.9` decays every weight in place, and an edge whose weight is 0 or `NULL` stays in the table without being followed. `src` and `dst` are never changed in place: an `UPDATE` of a `KEYWORD` one rewrites the row, and `set`, `add` or an `UPDATE` of an `INT` one is refused, as for `shard_by` (delete the row and insert it again).
- `REACHABLE`, `PATH`, `hops()` and `ppr` take an edge table with every option but `label`, since its edges have none. A walk over `cooc` and a walk over an `EDGES` field holding the same edges and weights find the same rows at the same hops and the same PageRank, up to rounding.
- Each segment of the edge table indexes its rows by the keys they link, which flushes and merges build, so a walk finds a row's edges with one lookup of its key in each of the edge table's segments ([how, and what it costs](graphs.md#edge-tables)).

## Running SQL

The `strata` shell runs one statement, or reads statements from standard input:

```sh
cd strata && cargo build --release
target/release/strata db.strata "SELECT category, COUNT(*) FROM docs WHERE MATCH(body, 'india') GROUP BY category ORDER BY 2 DESC LIMIT 5" --plan
target/release/strata db.strata --param t='rapid covid tests' < queries.sql
```

| flag | effect |
|---|---|
| `--param name=value` | binds `:name`: a number, a string, or a JSON array for a vector |
| `--plan` | prints the plan lines the executor recorded for each statement |
| `--repeat n` | runs each statement *n* times and reports the median time |

From Rust, `Db::query` takes a `SELECT` and a `Params` map, and `Db::sql` takes any statement, on any table, writes and `CREATE TABLE` included. Over the network, [strata-server](server.md) serves the same SQL over every table of a database.

## Tables

A database starts with the table it was created with, and `CREATE TABLE` adds more. Each table has its own columns, primary key, indexes and merges; the tables share the file, its write-ahead log and transactions.

```sql
CREATE TABLE authors (
  id KEYWORD PRIMARY KEY,
  name TEXT OPTIONS(analyzer = 'simple'),
  born INT,
  topics KEYWORD[],
  bio TEXT OPTIONS(positions = false),
  emb VECTOR OPTIONS(embed = 'name, bio', model = 'potion-retrieval-32M')
);
CREATE TABLE IF NOT EXISTS wrote (
  id INT PRIMARY KEY,
  author KEYWORD NOT NULL REFERENCES authors,
  paper KEYWORD REFERENCES papers (id),
  role KEYWORD NOT NULL DEFAULT 'author',
  added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE events (
  id INT PRIMARY KEY,
  usr KEYWORD NOT NULL,
  item KEYWORD,
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) OPTIONS(shard_by = 'usr', cluster_by = 'usr', retain = '400 days');
DROP TABLE IF EXISTS wrote, authors;
```

- **Types** are `TEXT`, `KEYWORD`, `INT` (or `INTEGER`, `BIGINT`), `FLOAT` (or `DOUBLE`, `REAL`), `BOOL` (or `BOOLEAN`), `TIMESTAMP` (or `TIMESTAMPTZ`, `TIMESTAMP WITH TIME ZONE`, `DATETIME`; a `DATE` is a `TIMESTAMP` at midnight UTC), `VECTOR(dimensions)`, `SPARSE`, `MEDIA` and `EDGES`; `KEYWORD[]` and `INT[]` hold lists, and `VECTOR(dimensions)[]` several vectors per row ([multi-vector fields](#maxsimfield-vectors-several-query-vectors)). A `BOOL` or `TIMESTAMP` column holds one value and is neither the primary key nor a reference.
- **`OPTIONS(...)`** takes the options of the [JSON schema](server.md#schemas): `positions`, `stored` and `indexed` (true or false; an indexed `TEXT` or `SPARSE` field is stored unless generated, since merges rebuild its index from the stored values), `from` (the fields a generated `TEXT` field joins), `analyzer` (`'english'`, the default: stopwords dropped and words stemmed, so `running` finds `run`; or `'simple'`: every word kept as written, for names, codes and other languages), `index` (`'hnsw'` with `m` and `ef_construction`, `'flat'`, or `'ivf'` with `nlist`), `codes` (`'sq8'` or `'binary'`; a multi-vector field searches `'binary'` by default), `keep_f32` (a multi-vector field keeps its float32 vectors to score with, four times the bytes of its 8-bit codes), `pool` (a multi-vector field merges a row's vectors whose cosine similarity is at least this, such as `0.9`, into their mean when it is written), `kind` of a `MEDIA` field, `derive` (a `MEDIA` field) with `model` for [text made from media](server.md#text-from-media), `embed` with `model` for an [embedded vector field](server.md#embedded-vector-fields), which may leave out its dimensions when Strata knows the model's (`VECTOR` alone), and on a `KEYWORD[]` `semantic_id` (the vector field) with `codebook` for a [semantic-ID field](server.md#semantic-ids): `sid KEYWORD[] OPTIONS(semantic_id = 'emb', codebook = 'items-rq')`. Field lists are a string, `'name, bio'`, or `ARRAY['name', 'bio']`.
- **The primary key** is one column: `PRIMARY KEY` on it, or a `PRIMARY KEY (column)` constraint. A table without one only takes plain `INSERT`.
- **`REFERENCES table`**, or `FOREIGN KEY (column) REFERENCES table`, records that a `KEYWORD` or `INT` column holds primary keys of that table, which must exist (or be the table being created) and have a primary key of the column's type. References are not enforced: a row may name a key that is not there, and an inner join along the reference leaves that row out (`LEFT JOIN` keeps it). Joins along them are [ordinary joins](#subqueries-with-and-joins) (`FROM wrote w JOIN authors a ON w.author = a.id`).
- **`NOT NULL`** refuses a row without a value in that column, from any write: SQL, JSON, a file or the Rust API. **`DEFAULT value`** is the value a write that leaves the column out gets: a constant of the column's type on a single-valued `TEXT`, `KEYWORD`, `INT`, `FLOAT`, `BOOL` or `TIMESTAMP` column that is not the key, or `CURRENT_TIMESTAMP` (or `now()`) on a `TIMESTAMP`, the time of the statement, the same in all its rows. An `INSERT` that leaves a column out, a JSON row without its key and a CSV file without its column give it the default; a `NULL` that a write gives stays `NULL`, and a column with `NOT NULL` refuses it. In SQL the keyword `DEFAULT` stands for a column's default, in `VALUES` and in `SET`, and `CURRENT_TIMESTAMP` (or `now()`, `LOCALTIMESTAMP`) is the time of the statement there and in conditions (`SET seen = CURRENT_TIMESTAMP`, also in `ON CONFLICT ... DO UPDATE`). The Rust API writes a row as it is given; `Schema::defaults` returns a row of the defaults to start from. The JSON schema's options are `"not_null": true` and `"default"` (`"now"` for the time of the write).
- **`OPTIONS(shard_by = 'column')`** after the columns places the table's rows on [shards](server.md#tables-on-shards) by that column instead of by the primary key, so that a user's events sit on one shard together. The column is a stored, single-valued `KEYWORD` or `INT` with `NOT NULL`, other than the key, and a row keeps it: `UPDATE` does not change it (delete the row and insert it again). In JSON it is `"shard_by": "usr"`. One database places nothing, so the option only matters to a coordinator.
- **`OPTIONS(cluster_by = 'column')`** stores each segment's rows sorted by that column, rows of one value in the order they were written: a flush sorts the rows it writes, and a merge sorts the rows of the segments it joins. A user's history then sits in one run of each segment instead of scattered across it, so reading it touches a few pages rather than one per row. The column is an indexed, single-valued `KEYWORD`, `INT` or `TIMESTAMP`, and a row keeps it (`UPDATE` does not change it). Queries return the same rows either way; only `ORDER BY` fixes their order. In JSON it is `"cluster_by": "usr"`.
- **`OPTIONS(retain = '400 days')`** drops rows once their time is older than the period: a flush does not write them, and a merge leaves them out of the segment it writes, or drops a segment whose rows are all past it without rewriting anything; the merger looks for such segments on its own. Until then a query still finds them, so filter on the time where that matters (`WHERE ts > '2025-10-01'`). The time is the table's only `TIMESTAMP` column, or the one `retain_by = 'column'` names (indexed, single-valued, and not changed by `UPDATE`); a row without one is kept. A period is a whole number and a unit: seconds, minutes, hours, days or weeks (`'90 s'`, `'12h'`, `'2 weeks'`). In JSON it is `"retain": "400 days"` with `"retain_by"`.
- **`OPTIONS(edges = 'src, dst, w')`** makes each row an edge of a graph over another table's rows, which walks over that table follow ([edge tables](#edge-tables)). In JSON it is `"edges": "src, dst, w"`.
- **`DROP TABLE`** drops tables other than the first. A table that another table references goes with it (`DROP TABLE wrote, authors`) or with `CASCADE`, which removes those references.
- Every other clause (`UNIQUE`, `CHECK`, `ON DELETE`, constraint names, a `DEFAULT` that is an expression, `CREATE TABLE ... AS SELECT`) is an error, and there is no `ALTER TABLE` yet.

`CREATE TABLE` and `DROP TABLE` run through `Db::sql`, Python's in-process [`strata.Database`](python.md), and [strata-server](server.md#schemas)'s `POST /v1/sql`, where a coordinator runs them on every shard.

## Writes

```sql
INSERT INTO products (id, title, price, tags, emb) VALUES ('p1', 'red mug', 12.5, ARRAY['kitchen', 'gift'], :v);
INSERT OR REPLACE INTO products (id, title, price) VALUES ('p1', 'red mug, large', 14.0);
INSERT INTO products (id, title, price) VALUES ('p2', 'blue mug', 11.0) ON CONFLICT (id) DO NOTHING;
INSERT INTO products (id, title, price) VALUES ('p1', 'red mug', 12.0)
  ON CONFLICT (id) DO UPDATE SET price = excluded.price, stock = stock + 10;
UPDATE products SET stock = stock - 1, price = 11.0 WHERE id = 'p1';
UPDATE products SET stock = DEFAULT WHERE id = 'p2';
UPDATE products SET price = round(price * 0.9, 2), title = title || ' (sale)' WHERE tags = 'gift';
INSERT INTO products (id, stock) VALUES ('p1', 5)
  ON CONFLICT (id) DO UPDATE SET stock = coalesce(stock, 0) + excluded.stock;
DELETE FROM products WHERE MATCH(title, 'discontinued');
DELETE FROM products WHERE stock = 0 ORDER BY price LIMIT 100;
```

- A write names its table, any table of the database.
- **`INSERT`** fails on a primary key that already exists. **`INSERT OR REPLACE`** (or `REPLACE INTO`) replaces that row, an upsert. **`ON CONFLICT (key) DO NOTHING`** (or `INSERT OR IGNORE`) writes only the rows whose key is new, and the first of a key the statement repeats. **`ON CONFLICT (key) DO UPDATE SET ...`** inserts the new keys and sets fields of the rows already there, where `excluded.column` is the value the statement gave; a key that comes twice in one such statement is an error. Lists are `ARRAY[…]`; a vector is a parameter, `ARRAY[…]` or a JSON array string; a multi-vector value is a list of vectors (`ARRAY[ARRAY[…], ARRAY[…]]`, a parameter holding a list of vectors, or `'[[…], […]]'`), and one vector given to it is a list of one; a `SPARSE` value is a JSON object string, and a `MEDIA` value a URI or a JSON object string.
- **`UPDATE`** takes any `WHERE`, and a `LIMIT`. Setting only single-valued `INT` and `FLOAT` fields (`SET clicks = clicks + 1`) changes them in place, without rewriting the row, which suits counters and prices that change often. Setting any other field rewrites the row, which needs every text field stored or generated.
- **`SET`** takes a value or any expression of the row: arithmetic, `||`, `CASE`, `NOT`, comparisons, `lower`, `upper`, `length`, `abs`, `round` and `coalesce` over the row's columns, and in `ON CONFLICT DO UPDATE` over `excluded.column` too. Every expression reads the row as it was, so `SET a = b, b = a` swaps two fields. The result is stored as the field's type: a whole number fits an `INT` (`clicks = price * 2`), but `7 / 2` is 3.5 and an error there (`round(clicks / 2)` fits); one value set to a multi-valued field becomes a list of it. NULL in gives NULL out, as in SQL, except `field + number` and `field - number`, which count a NULL as zero, as a counter does; `coalesce(field, 0) + other` does the same elsewhere. A multi-valued field without values is NULL to `coalesce` and `IS NULL`. Searches, scores and aggregates (`MATCH`, `bm25`, `score()`, `COUNT`) are errors in `SET`. A statement that cannot set one of its rows sets none.
- **`DELETE`** takes any `WHERE`, and `ORDER BY` with `LIMIT` to delete the first rows in that order.
- `RETURNING`, `UPDATE ... FROM`, `DELETE ... USING`, `INSERT ... SELECT`, `ON DUPLICATE KEY UPDATE` and `DO UPDATE ... WHERE` are errors.

Each statement is one transaction, and returns the number of rows it wrote. From Rust, the transaction API (`Database::begin`, `WriteTxn::insert`, `upsert`, `set`, `delete`, `commit`, and `WriteTxn::table` for the other tables) groups several writes into one, across tables too; `WriteTxn::upsert_if_newer` keeps whichever version of a row has the highest version number, for replaying a feed out of order.

## EXPLAIN

`EXPLAIN <statement>` shows the plan of a `SELECT`, `INSERT`, `UPDATE` or `DELETE` without running it: a row per line, in a column named `plan`, the root first and each operator's input indented under it.

```sql
EXPLAIN SELECT category, COUNT(*) AS n FROM papers WHERE published >= '2026-10-01'
GROUP BY category HAVING COUNT(*) >= 2 ORDER BY n DESC, category LIMIT 5;
```

```text
Project: category, COUNT(*) AS n
  Limit: 5
    Sort: COUNT(*) DESC, category
      Filter: COUNT(*) >= 2.0
        Aggregate by category: COUNT(*)
          Search papers
            filter: published >= TIMESTAMP '2026-10-01T00:00:00Z'
```

A `Search` is the rows of one table that its `filter` holds for, ranked by its `order`; the operators above it group, keep, sort, cut and name them, and a write's operator (`Insert`, `Update`, `Delete`) sits on top. The plan is the statement as bound: values have their field's type, and a `NOT` over a field that can be null also says `x IS NOT NULL`, since a row without a value matches neither a comparison nor its negation.

`EXPLAIN FORMAT JSON` (or `EXPLAIN (FORMAT JSON)`) gives the plan as one row of JSON: fields by their position in the table, typed values (`{"int": 5}`, `{"text": "5"}`), a version and the features the plan needs. Strata refuses a plan from a newer version by name. `EXPLAIN ANALYZE` and other formats are refused.

A compound query's plan has operators over rows in memory above its `Search`es, and its JSON needs the feature `operators`:

```sql
EXPLAIN SELECT a.id, b.title FROM papers a JOIN papers b ON a.cites = b.id WHERE a.year >= 2019 ORDER BY a.year DESC LIMIT 5;
```
```text
Project: id AS "a.id", title AS "b.title"
  Limit: 5
    Join on targets(cites) = id
      Project: id, year, cites
        Search papers
          filter: year >= 2019
          order: year DESC
      Project: title, id
        Search papers
```

`Join` (or `Left join`) matches its two inputs on the keys after `on`, `targets(f)` being the rows an `EDGES` field links to, with any other `ON` condition after `AND`; `With` names its results, which `Ref` reads; `Key order of top` puts rows in the order of the `WITH` result `top`; `Distinct` keeps each row once, and `Distinct on: video` the [first row of each group](#the-first-row-of-each-group-distinct-on). A subquery sits in its condition: `IN values of (...)` (its values), `IN keys of (...)` (its rows' keys), `EXISTS (...)` and a scalar `(...)`, each shown on one line with `;` between its operators. `SELECT DISTINCT` of the table alone is a `GROUP BY`, a single-table query.

Every statement runs as its plan, and a plan in JSON runs on its own: `Db::run` in Rust, `Database.run` in [Python](python.md#querying) and [`POST /v1/plan`](server.md#http-api) over HTTP take one, and check it against the table's schema as the SQL binder checks a statement.

## Reading a plan

`--plan` shows what the planner decided, segment by segment. The same statement can get different plans as the data changes. In the evaluation, an `OR` filter matching about 15% of documents was scanned exactly on SciFact's 5,183 documents, but on the larger collections the planner measured 11% near the query and post-filtered the HNSW result. An RRF query's topic filter matched 11% of Quora globally but only 3.4% near the query, so the planner switched to an exact scan of its 58,767 members, at 3.1 ms. [Planning and execution](index.md#planning-and-execution) describes the choices.
