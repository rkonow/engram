"""Queries built from Python values (`strata.build`): each runs as the SQL
it stands for, names any field, and takes values as they are."""

import datetime

import pytest

import strata
from strata import at_least, bm25, case, clamp, coalesce, col, count, count_true, exp, features, greatest, least, lit, ln, log, match, maxsim, minmax, rank, ranked, rrf, score, similarity, sigmoid, sqrt, tmm, zscore
from conftest import BOOKS, book


def same(db, q, sql, **params):
    """The builder's query answers as its SQL does."""
    got, want = db.run(q), db.query(sql, params or None)
    assert got.columns == want.columns, sql
    assert got.rows == want.rows, sql
    assert list(q) == list(want), sql
    return got


def test_builder_runs_as_its_sql(db):
    t = db.table()
    r = same(db, t.select("id", "year").where((col("year") >= 1950) & (col("rating") < 3)).order_by(col("year").desc()).limit(5),
             "SELECT id, year FROM books WHERE year >= 1950 AND rating < 3 ORDER BY year DESC LIMIT 5")
    assert len(r) == 5
    r = same(db, t.select("id", score()).where(match("description", "dragon ship")).order_by(bm25("description", "dragon ship")).limit(5),
             "SELECT id, score() FROM books WHERE MATCH(description, 'dragon ship') ORDER BY bm25(description, 'dragon ship') DESC LIMIT 5")
    assert len(r) == 5
    same(db, t.select("id").where(col("genres").isin(["classic"]) & ~col("year").between(1950, 1999)).order_by(col("year")),
         "SELECT id FROM books WHERE genres IN ('classic') AND NOT (year BETWEEN 1950 AND 1999) ORDER BY year")
    same(db, t.select("id").where((col("year") < 1910) | (col("year") > 2015)).order_by(col("year").desc()).limit(4).offset(2),
         "SELECT id FROM books WHERE year < 1910 OR year > 2015 ORDER BY year DESC LIMIT 4 OFFSET 2")
    same(db, t.select("id").order_by(rrf(bm25("title", "dragon"), bm25("description", "dragon"), k=30)).limit(5),
         "SELECT id FROM books ORDER BY rrf(bm25(title, 'dragon'), bm25(description, 'dragon'), k => 30) DESC LIMIT 5")
    same(db, t.select("id").order_by(0.7 * minmax(bm25("title", "river")) + 0.3 * minmax(bm25("description", "river"))).limit(5),
         "SELECT id FROM books ORDER BY 0.7 * minmax(bm25(title, 'river')) + 0.3 * minmax(bm25(description, 'river')) DESC LIMIT 5")
    # Prefixes of a KEYWORD field and of a multi-valued one.
    r = same(db, t.select("id").where(col("id").startswith("b000") & ~col("genres").startswith("cl")).order_by(col("year")),
             "SELECT id FROM books WHERE PREFIX(id, 'b000') AND NOT PREFIX(genres, 'cl') ORDER BY year")
    assert len(r) > 0
    # With typos: 'calssic' is a swap away from 'classic'.
    r = same(db, t.select("id").where(col("genres").startswith("calssic", fuzzy="auto")).order_by(col("year")),
             "SELECT id FROM books WHERE PREFIX(genres, 'calssic', fuzzy => 'auto') ORDER BY year")
    assert len(r) > 0 and r.rows == db.query("SELECT id FROM books WHERE genres IN ('classic') ORDER BY year").rows
    # Without select: the stored fields but vectors, as SELECT * gives.
    same(db, t.where(col("id") == "b0007"), "SELECT * FROM books WHERE id = 'b0007'")


def test_bm25_weighs_requires_and_excludes_words(db):
    t = db.table()
    q = '+dragon ship^2 -river "secret garden"'
    r = same(db, t.select("id", "description", score()).order_by(bm25("description", q)).limit(20),
             f"SELECT id, description, score() FROM books ORDER BY bm25(description, '{q}') DESC LIMIT 20")
    assert len(r) > 0 and all("dragon" in d and "river" not in d for _, d, _ in r.rows)
    plain = same(db, t.select("id", score()).order_by(bm25("description", q, syntax="plain")).limit(20),
                 f"SELECT id, score() FROM books ORDER BY bm25(description, '{q}', syntax => 'plain') DESC LIMIT 20")
    assert plain.rows != r.rows
    with pytest.raises(strata.StrataError, match="'query' or 'plain'"):
        db.run(t.select("id").order_by(bm25("description", q, syntax="lucene")).limit(5))


def test_min_asks_for_some_of_the_words(db):
    t = db.table()
    q = "dragon ship river secret"
    r = same(db, t.select("id", score()).order_by(bm25("description", q, min=3)).limit(20),
             f"SELECT id, score() FROM books ORDER BY bm25(description, '{q}', min => 3) DESC LIMIT 20")
    assert len(r) > 0 and all(sum(w in d for w in q.split()) >= 3 for d in (book(int(i[1:]))["description"] for i, _ in r.rows))
    same(db, t.select("id").order_by(bm25("description", q, min="50%", syntax="plain")).limit(20),
         f"SELECT id FROM books ORDER BY bm25(description, '{q}', min => '50%', syntax => 'plain') DESC LIMIT 20")
    same(db, t.select("id").where(match("description", q, min=3)).order_by(col("year")).limit(100),
         f"SELECT id FROM books WHERE MATCH(description, '{q}', min => 3) ORDER BY year LIMIT 100")
    same(db, t.select("id").where(match("description", "(dragon ship river)@2 -secret")).order_by(col("year")).limit(100),
         "SELECT id FROM books WHERE MATCH(description, '(dragon ship river)@2 -secret') ORDER BY year LIMIT 100")
    with pytest.raises(strata.StrataError, match="count or a percentage"):
        bm25("description", q, min=1.5)


def test_at_least_and_count_true_run_as_their_sql(db):
    t = db.table()
    conds = [match("description", "dragon"), col("year") >= 1990, col("genres").isin(["classic"]), col("rating") < 3]
    sql = "MATCH(description, 'dragon'), year >= 1990, genres IN ('classic'), rating < 3"
    for n, text in [(2, "2"), (3, "3"), ("50%", "'50%'"), (9, "9")]:
        r = same(db, t.select("id").where(at_least(n, *conds)).order_by(col("year")).limit(100),
                 f"SELECT id FROM books WHERE AT_LEAST({text}, {sql}) ORDER BY year LIMIT 100")
        assert len(r) > 0, text
    # A count past the conditions' needs every one of them.
    every = "MATCH(description, 'dragon') AND year >= 1990 AND genres IN ('classic') AND rating < 3"
    assert r.keys == db.query(f"SELECT id FROM books WHERE {every} ORDER BY year LIMIT 100").keys
    same(db, t.select("id").where(~at_least(2, *conds)).order_by(col("year")).limit(100),
         f"SELECT id FROM books WHERE NOT AT_LEAST(2, {sql}) ORDER BY year LIMIT 100")
    r = same(db, t.select("id", count_true(*conds).alias("k")).order_by(bm25("description", "river") * (1 + count_true(*conds))).limit(20),
             f"SELECT id, count_true({sql}) AS k FROM books ORDER BY bm25(description, 'river') * (1 + count_true({sql})) DESC LIMIT 20")
    assert len(r) == 20 and all(0 <= k <= 4 for _, k in r.rows) and len({k for _, k in r.rows}) > 1
    with pytest.raises(strata.StrataError, match="a count .2. or a percentage"):
        at_least("120%", *conds)
    with pytest.raises(strata.StrataError, match="one or more conditions"):
        count_true()


def test_guards_run_as_their_sql(db):
    t = db.table()
    conds = [match("description", "dragon"), col("year") >= 1990, col("genres").isin(["classic"]), col("rating") < 3]
    sql = "MATCH(description, 'dragon'), year >= 1990, genres IN ('classic'), rating < 3"
    for opts, text in [(dict(max_rows=40), "max_rows => 40"), (dict(min_rows=30), "min_rows => 30"), (dict(min_rows=10, max_rows=90), "min_rows => 10, max_rows => 90")]:
        q = t.select("id").where(col("year") > 1910, at_least(1, *conds, **opts)).order_by(col("year")).limit(200)
        r = same(db, q, f"SELECT id FROM books WHERE year > 1910 AND AT_LEAST(1, {sql}, {text}) ORDER BY year LIMIT 200")
        assert "guard" in q.plan()["root"]["input"]["input"]
        q = t.select("id").where(match("description", "dragon ship river winter", min="25%", **opts)).order_by(col("year")).limit(200)
        same(db, q, f"SELECT id FROM books WHERE MATCH(description, 'dragon ship river winter', min => '25%', {text}) ORDER BY year LIMIT 200")
        q = t.select("id", score()).order_by(bm25("description", "dragon ship river winter", min=1, **opts)).limit(10)
        same(db, q, f"SELECT id, score() FROM books ORDER BY bm25(description, 'dragon ship river winter', min => 1, {text}) DESC LIMIT 10")
    assert len(r) > 0
    with pytest.raises(strata.StrataError, match="relax its min, which it needs"):
        match("description", "dragon ship", max_rows=10)
    with pytest.raises(strata.StrataError, match="a count of rows"):
        at_least(1, *conds, max_rows=2.5)
    guarded = at_least(1, *conds, max_rows=40)
    for q, msg in [
        (t.select("id").where(guarded | (col("year") > 2000)), "joined to the others by AND"),
        (t.select("id").where(~guarded), "joined to the others by AND"),
        (t.select("id").where(guarded, match("description", "dragon ship", min=1, max_rows=5)), "one guard"),
        (t.select("id").order_by(bm25("description", "dragon")).rerank(bm25("description", "ship river", min=1, max_rows=5), 20), "not a rerank phase"),
    ]:
        with pytest.raises(strata.StrataError, match=msg):
            db.run(q)


def test_a_ranking_given_by_keys_runs_as_its_sql(db):
    t = db.table()
    keys = ["b0007", "b0003", "none", "b0011", "b0003"]
    r = same(db, t.select("id", score()).order_by(ranked(keys)).limit(10),
             "SELECT id, score() FROM books ORDER BY ranked(:keys) DESC LIMIT 10", keys=keys)
    assert r.rows == [["b0007", -1.0], ["b0003", -2.0], ["b0011", -4.0]]
    same(db, t.select("id").where(col("year") >= 1950).order_by(rrf(bm25("description", "dragon"), ranked(keys))).limit(5),
         "SELECT id FROM books WHERE year >= 1950 ORDER BY rrf(bm25(description, 'dragon'), ranked(:keys)) DESC LIMIT 5", keys=keys)


def test_ranking_expressions_run_as_their_sql(db):
    """Arithmetic, functions, minmax(), rank() and case() rank as SQL's
    expressions do: `+` adds raw values, and NULL ranks last."""
    t = db.table()
    for q, sql in [
        (0.5 * minmax(bm25("title", "river")) - 0.5 * minmax(col("rating")),
         "0.5 * minmax(bm25(title, 'river')) - 0.5 * minmax(rating)"),
        (bm25("title", "river") + bm25("description", "river"),
         "bm25(title, 'river') + bm25(description, 'river')"),
        (bm25("description", "dragon") * (1 + 0.2 * coalesce(col("rating"), 0)),
         "bm25(description, 'dragon') * (1 + 0.2 * coalesce(rating, 0))"),
        (ln(1 + bm25("description", "dragon")) - 0.1 * col("rating"),
         "ln(1 + bm25(description, 'dragon')) - 0.1 * rating"),
        (case((col("year") > 2000, 1), (col("year") > 1950, 0.5), otherwise=0) + minmax(bm25("description", "ship")),
         "CASE WHEN year > 2000 THEN 1 WHEN year > 1950 THEN 0.5 ELSE 0 END + minmax(bm25(description, 'ship'))"),
        (-rank(bm25("title", "dragon")) - rank(bm25("description", "dragon")),
         "-rank(bm25(title, 'dragon')) - rank(bm25(description, 'dragon'))"),
        (round(abs(col("rating") - 3), 1) + sqrt(col("year") / 1000) + clamp(col("rating"), 1, 4) ** 2,
         "round(abs(rating - 3), 1) + sqrt(year / 1000) + pow(clamp(rating, 1, 4), 2)"),
        (greatest(sigmoid(col("rating")), least(log(col("year")), exp(1))) % 2,
         "greatest(sigmoid(rating), least(log(year), exp(1))) % 2"),
        (0.6 * tmm(bm25("title", "river")) + 0.4 * tmm(bm25("description", "river")),
         "0.6 * tmm(bm25(title, 'river')) + 0.4 * tmm(bm25(description, 'river'))"),
        (tmm(bm25("description", "dragon")) - 0.1 * tmm(col("rating"), lo=0),
         "tmm(bm25(description, 'dragon')) - 0.1 * tmm(rating, lo => 0)"),
        (tmm(bm25("description", "dragon"), lo=0, hi=10) + tmm(col("rating"), lo=0, hi=5),
         "tmm(bm25(description, 'dragon'), lo => 0, hi => 10) + tmm(rating, lo => 0, hi => 5)"),
        (zscore(bm25("title", "river")) + 0.5 * zscore(col("year")),
         "zscore(bm25(title, 'river')) + 0.5 * zscore(year)"),
    ]:
        same(db, t.select("id", score()).order_by(q).limit(8), f"SELECT id, score() FROM books ORDER BY {sql} DESC LIMIT 8")
    with pytest.raises(TypeError):
        col("rating") + "x"
    with pytest.raises(strata.StrataError):
        tmm(col("rating"), hi=5)
    with pytest.raises(strata.StrataError):
        tmm(col("rating"), lo=5, hi=5)
    with pytest.raises(strata.StrataError, match="lo"):
        db.run(t.select("id").order_by(tmm(col("rating"))).limit(3))


def test_rerank_phases_run_as_their_sql(db):
    """`rerank` ranks the top rows again as the SQL WITH form does: a
    second phase, a global one, and a chain."""
    t = db.table()
    first = t.select("id", score()).order_by(bm25("description", "dragon ship river"))
    for q, sql in [
        (first.rerank(col("rating"), 40).limit(10),
         "WITH top AS (SELECT * FROM books ORDER BY bm25(description, 'dragon ship river') DESC LIMIT 40) SELECT id, score() FROM top ORDER BY rating DESC LIMIT 10"),
        (first.rerank(minmax(bm25("title", "river")) + 0.5 * minmax(col("year")), 30).limit(5).offset(2),
         "WITH top AS (SELECT * FROM books ORDER BY bm25(description, 'dragon ship river') DESC LIMIT 30) SELECT id, score() FROM top ORDER BY minmax(bm25(title, 'river')) + 0.5 * minmax(year) DESC LIMIT 5 OFFSET 2"),
        (first.rerank(col("year"), 60).rerank(coalesce(col("rating"), 0) - col("year") / 1000, 20).limit(8),
         "WITH a AS (SELECT * FROM books ORDER BY bm25(description, 'dragon ship river') DESC LIMIT 60), b AS (SELECT * FROM a ORDER BY year DESC LIMIT 20) SELECT id, score() FROM b ORDER BY coalesce(rating, 0) - year / 1000 DESC LIMIT 8"),
    ]:
        same(db, q, sql)
    phases = first.rerank(col("rating"), 40).rerank(rank(col("year")), 20).rerank(col("rating"), 10).plan()["root"]["input"]["phases"]
    assert [p.get("global", False) for p in phases] == [False, True, True]
    with pytest.raises(strata.StrataError, match="at least one row"):
        first.rerank(col("rating"), 0)
    with pytest.raises(strata.StrataError, match="follow an order_by"):
        t.select("id").rerank(col("rating"), 10).plan()


def test_features_run_as_their_sql(db):
    """`features()` and ranking expressions in `select` give the values
    SQL's do, named as Python writes them."""
    t = db.table()
    dragon, river = bm25("description", "dragon"), bm25("title", "river")
    boost = dragon * (1 + 0.2 * coalesce(col("rating"), 0))
    later = ln(1 + river) - 0.1 * col("rating")
    for q, sql, names in [
        (t.select("id", score(), features()).order_by(boost).limit(10),
         "SELECT id, score(), features() FROM books ORDER BY bm25(description, 'dragon') * (1 + 0.2 * coalesce(rating, 0)) DESC LIMIT 10",
         ["id", "score()", repr(dragon), "rating", repr(boost)]),
        (t.select("id", features()).order_by(rrf(river, dragon)).limit(8),
         "SELECT id, features() FROM books ORDER BY rrf(bm25(title, 'river'), bm25(description, 'dragon')) DESC LIMIT 8",
         ["id", repr(river), repr(dragon)]),
        (t.select("id", score(), features()).order_by(bm25("description", "dragon ship river")).rerank(later, 30).limit(8),
         "WITH top AS (SELECT * FROM books ORDER BY bm25(description, 'dragon ship river') DESC LIMIT 30) SELECT id, score(), features() FROM top ORDER BY ln(1 + bm25(title, 'river')) - 0.1 * rating DESC LIMIT 8",
         ["id", "score()", "bm25('description', 'dragon ship river')", repr(river), "rating", repr(later)]),
        (t.select("id", river.alias("r"), col("year") / 1000).order_by(col("rating").desc()).limit(10),
         "SELECT id, bm25(title, 'river') AS r, year / 1000 FROM books ORDER BY rating DESC LIMIT 10",
         ["id", "r", repr(col("year") / 1000)]),
    ]:
        got, want = db.run(q), db.query(sql)
        assert got.columns == names, sql
        assert got.rows == want.rows, sql
    with pytest.raises(strata.StrataError, match="no order_by"):
        t.select("id", features()).plan()
    with pytest.raises(strata.StrataError, match="not of groups"):
        t.select("genres", count(), features()).group_by("genres").plan()
    with pytest.raises(strata.StrataError, match="against each other"):
        db.run(t.select("id", minmax(col("rating")).alias("m")).order_by(col("rating").desc()).limit(3))


def test_groups(db):
    t = db.table()
    r = same(db, t.select("genres", count().alias("n"), col("rating").avg().alias("r")).group_by("genres").having(count() > 50).order_by(col("genres")),
             "SELECT genres, COUNT(*) AS n, AVG(rating) AS r FROM books GROUP BY genres HAVING COUNT(*) > 50 ORDER BY genres")
    assert r.column("genres") == ["fantasy", "mystery", "new"]
    same(db, t.select("genres", count()).group_by("genres").having(count() != 100, col("year").max() <= 2018).order_by(count().desc()),
         "SELECT genres, COUNT(*) FROM books GROUP BY genres HAVING COUNT(*) != 100 AND MAX(year) <= 2018 ORDER BY COUNT(*) DESC")
    assert t.select(count()).run().scalar() == 200
    same(db, t.select(count(), col("year").count(distinct=True)).where(col("rating") >= 4),
         "SELECT COUNT(*), COUNT(DISTINCT year) FROM books WHERE rating >= 4")


def test_distinct_on(db):
    """The first row of each group, in the query's order, as SQL's
    DISTINCT ON keeps it."""
    t = db.table()
    q = t.select("id", "year").distinct_on("rating").order_by(col("year").desc()).limit(5)
    r = same(db, q, "SELECT DISTINCT ON (rating) id, year FROM books ORDER BY year DESC LIMIT 5")
    assert len(r) == 5
    assert q.plan()["features"] == ["operators", "distinct_on"]
    r = same(db, t.select("id", "year", score()).where(match("description", "dragon")).distinct_on("year").order_by(bm25("description", "dragon")).limit(4).offset(1),
             "SELECT DISTINCT ON (year) id, year, score() FROM books WHERE MATCH(description, 'dragon') ORDER BY bm25(description, 'dragon') DESC LIMIT 4 OFFSET 1")
    assert len({row[1] for row in r.rows}) == 4
    # Two keys, one an expression; without a limit, every group.
    r = same(db, t.select("id").distinct_on("rating", col("year") % 2).order_by(col("year")),
             "SELECT DISTINCT ON (rating, year % 2) id FROM books ORDER BY year")
    assert len(r) == 40  # a rating fixes the parity of the year
    with pytest.raises(strata.StrataError, match="distinct_on"):
        t.select("genres", count()).group_by("genres").distinct_on("genres").plan()


def test_max_per(db):
    """At most k rows of each value, the first k in the query's order, as
    SQL's MAX k PER keeps them."""
    t = db.table()
    # Five books have each rating: two of each of the three best.
    q = t.select("id", "rating").order_by(col("rating").desc()).max_per(2, "rating").limit(6)
    r = same(db, q, "SELECT id, rating FROM books ORDER BY rating DESC MAX 2 PER rating LIMIT 6")
    assert [row[1] for row in r.rows] == [4.9, 4.9, 4.8, 4.8, 4.7, 4.7]
    assert q.plan()["features"] == ["operators", "distinct_on", "per_cap"]
    same(db, t.select("id", "year", score()).where(match("description", "dragon")).order_by(bm25("description", "dragon")).max_per(3, "year").limit(6).offset(1),
         "SELECT id, year, score() FROM books WHERE MATCH(description, 'dragon') ORDER BY bm25(description, 'dragon') DESC MAX 3 PER year LIMIT 6 OFFSET 1")
    # One a value is DISTINCT ON, which needs no per_cap.
    assert t.select("id").order_by(col("year")).max_per(1, "rating").limit(3).plan()["features"] == ["operators", "distinct_on"]
    for bad, msg in [(lambda: t.select("id").max_per(2, "rating").limit(3), "order_by"), (lambda: t.select("id").order_by(col("year")).max_per(2, "genres"), "multi-valued"),
                     (lambda: t.select("id").order_by(col("year")).max_per(0, "rating"), "1 or more")]:
        with pytest.raises(strata.StrataError, match=msg):
            bad().plan()


def test_values_need_no_quoting(tmp_path):
    """Any field name, and any text, with no escaping: a name SQL reserves,
    one with a space or a quote, text with quotes."""
    schema = {"table": "notes", "key": "id", "fields": [
        {"name": "id", "type": "keyword"},
        {"name": "order", "type": "int"},
        {"name": "doc no", "type": "keyword"},
        {"name": 'say "hi"', "type": "text"},
        {"name": "when", "type": "timestamp"},
        {"name": "done", "type": "bool"},
    ]}
    with strata.Database.create(tmp_path / "n.strata", schema) as db:
        db.insert([
            {"id": "a", "order": 1, "doc no": "O'Brien \"7\"", "say \"hi\"": "it's a 'quoted' word", "when": "2026-10-01T00:00:00Z", "done": True},
            {"id": "b", "order": 2, "doc no": "x", "say \"hi\"": "plain words", "when": "2026-10-20T12:00:00Z", "done": False},
            {"id": "c", "order": 3, "doc no": "y", "say \"hi\"": "more plain words"},
        ])
        ids = lambda q: [r["id"] for r in q]
        t = db.table()
        assert ids(t.select("id").where(col("doc no") == "O'Brien \"7\"")) == ["a"]
        assert ids(t.select("id").where(match('say "hi"', "quoted"))) == ["a"]
        assert ids(t.select("id").where(col("order") > 1).order_by(col("order").desc())) == ["c", "b"]
        assert ids(t.select("id").where(col("when") >= datetime.date(2026, 10, 10))) == ["b"]
        assert ids(t.select("id").where(col("when") < datetime.datetime(2026, 10, 10, tzinfo=datetime.timezone.utc))) == ["a"]
        assert ids(t.select("id").where(col("done") == True).order_by(col("order"))) == ["a"]  # noqa: E712
        assert ids(t.select("id").where(col("done").is_null())) == ["c"]
        assert ids(t.select("id").where(col("when").is_not_null()).order_by(col("order"))) == ["a", "b"]
        # Names in another case find the field; a column is named as
        # written, as in SQL.
        assert [r["ID"] for r in t.select("ID").where(col("ORDER") == 2)] == ["b"]
        # The SQL condition of search() names fields quoted too.
        assert ids(db.search("words", where=col("order") >= 3)) == ["c"]
        assert ids(db.search("words", where='"order" >= 3')) == ["c"]


def test_search_takes_a_condition(db):
    a = db.search("dragon", limit=5, where=col("year") >= 1950)
    b = db.search("dragon", limit=5, where="year >= 1950")
    assert a.column("id") == b.column("id") and len(a) == 5
    assert all(r["year"] >= 1950 for r in a)
    assert db.search("dragon", where="").column("id") == db.search("dragon").column("id")
    a = db.search("dragon", limit=5, where=col("year") >= 1950, columns=["id", "year", "score()"])
    b = db.search("dragon", limit=5, where="year >= 1950", columns=["id", "year", "score()"])
    assert a.columns == b.columns == ["id", "year", "score()"] and a.rows == b.rows
    v = db.embed("dragon ship", "emb")
    near = db.table().select("id").order_by(similarity("emb", v)).limit(3)
    assert near.run().rows == db.query("SELECT id FROM books ORDER BY similarity(emb, :v) DESC LIMIT 3", {"v": {"embed": "dragon ship"}}).rows


def test_maxsim_runs_as_its_sql(db):
    """maxsim() of several vectors ranks as its SQL does, and of one
    vector as similarity()."""
    texts = ["dragon ship", "river town", "old war"]
    vs = [db.embed(t, "emb") for t in texts]
    t = db.table()
    got = t.select("id", score()).order_by(maxsim("emb", vs)).limit(5).run()
    want = db.query("SELECT id, score() FROM books ORDER BY maxsim(emb, (:a, :b, :c)) DESC LIMIT 5", {k: {"embed": x} for k, x in zip("abc", texts)})
    assert got.rows == want.rows and len(got) == 5
    one = t.select("id").order_by(maxsim("emb", vs[:1], max_vectors=1)).limit(3).run()
    assert one.rows == t.select("id").order_by(similarity("emb", vs[0])).limit(3).run().rows
    fused = t.select("id").order_by(rrf(bm25("description", "dragon"), maxsim("emb", vs))).limit(5)
    assert fused.run().rows == db.query("SELECT id FROM books ORDER BY rrf(bm25(description, 'dragon'), maxsim(emb, (:a, :b, :c))) DESC LIMIT 5", {k: {"embed": x} for k, x in zip("abc", texts)}).rows
    with pytest.raises(strata.StrataError):
        t.order_by(maxsim("title", vs)).plan()


def test_mistakes_are_errors(db):
    t = db.table()
    with pytest.raises(strata.StrataError, match="no field 'nope'"):
        t.where(col("nope") == 1).plan()
    with pytest.raises(TypeError, match="&, \\| and ~"):
        t.where(col("year") > 1950 and col("year") < 2000)
    with pytest.raises(TypeError):
        t.where(1950 < col("year") < 2000)
    with pytest.raises(strata.StrataError, match="is_null"):
        t.where(col("year") == None).plan()  # noqa: E711
    with pytest.raises(strata.StrataError, match="group_by"):
        t.select("id").having(count() > 1).plan()
    with pytest.raises(strata.StrataError, match="not a group key"):
        t.select("id", count()).group_by("genres").plan()
    with pytest.raises(strata.StrataError, match="embed the text"):
        t.order_by(similarity("emb", "dragon")).plan()
    with pytest.raises(strata.StrataError, match="no table"):
        db.table("other")
    with pytest.raises(TypeError):
        t.where(col("year") == object()).plan()
    # A value where an expression is wanted.
    assert t.select("id").where(col("year") == lit(1950)).run().column("id") == db.query("SELECT id FROM books WHERE year = 1950").column("id")
