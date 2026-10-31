"""Queries built from Python values instead of SQL text.

A `Query` makes a plan, the JSON that `Database.plan` and the server's
`/v1/explain` give, and `Database.run` and `Client.run` run it. Fields are
named by their names, whatever they are (`col("order")`, `col("doc no")`),
and values keep their types, so nothing is quoted or escaped:

    from strata import col, match, bm25

    q = (db.table()
         .select("id", "title")
         .where((col("year") >= 2019) & match("body", "graph neural"))
         .order_by(bm25("body", "graph neural"))
         .limit(10))
    for row in q:
        ...

Conditions combine with `&`, `|` and `~` (not `and`, `or` and `not`).
Scores rank best first; fields and aggregates rank ascending unless given
with `.desc()`.

A ranking is any expression of scores, fields and numbers, as in SQL:
`bm25("body", q) * (1 + 0.2 * coalesce(col("rating"), 0))`. `+` adds raw
values; `0.7 * minmax(bm25(...)) + 0.3 * minmax(similarity(...))` puts
each score on [0, 1] first, the weighted fusion of the two.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Iterable, Mapping, Sequence

from ._strata import StrataError

VERSION = 1

# Rows a scan whose groups `distinct_on` keeps reads at most (Strata's
# `compound::MAX_ROWS`).
_MAX_ROWS = 1_000_000

__all__ = [
    "Query",
    "Expr",
    "col",
    "lit",
    "match",
    "at_least",
    "bm25",
    "similarity",
    "maxsim",
    "rrf",
    "ranked",
    "minmax",
    "tmm",
    "zscore",
    "rank",
    "gbdt",
    "cross_encoder",
    "ln",
    "log",
    "exp",
    "sqrt",
    "sigmoid",
    "clamp",
    "least",
    "greatest",
    "coalesce",
    "case",
    "count_true",
    "score",
    "hops",
    "features",
    "count",
    "reachable",
    "path",
    "ppr",
]


class _Ctx:
    """What expressions compile against: the table's fields by name and,
    over groups, the outputs of the Aggregate."""

    def __init__(self, schema: Mapping[str, Any]):
        self.table = schema["table"]
        self.key = schema.get("key")
        self.fields = list(schema["fields"])
        self.outputs: dict | None = None

    def field(self, name: str) -> tuple[int, dict]:
        for i, f in enumerate(self.fields):
            if f["name"] == name:
                return i, f
        for i, f in enumerate(self.fields):
            if f["name"].lower() == name.lower():
                return i, f
        raise StrataError(f"{self.table} has no field {name!r}")


def _micros(t: _dt.date) -> int:
    if not isinstance(t, _dt.datetime):
        t = _dt.datetime(t.year, t.month, t.day, tzinfo=_dt.timezone.utc)
    elif t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    d = t - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    return (d.days * 86400 + d.seconds) * 1_000_000 + d.microseconds


def _value(v: Any, field: Mapping[str, Any] | None = None) -> Any:
    """A typed literal: `v` as `field`'s type when it is compared with a
    field (an int as a KEYWORD's text, as SQL reads it)."""
    t = field["type"] if field else None
    if v is None:
        return "null"
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, _dt.date):
        return {"timestamp": _micros(v)}
    if isinstance(v, int):
        if t == "keyword":
            return {"text": str(v)}
        if t == "float":
            return {"float": float(v)}
        return {"int": v}
    if isinstance(v, float):
        return {"float": v}
    if isinstance(v, str):
        return {"text": v}
    if isinstance(v, (list, tuple)):
        return {"list": [_value(x, field) for x in v]}
    raise TypeError(f"a query value is None, a bool, number, str, date, datetime or a list of them, not {type(v).__name__}")


class Expr:
    """A field, value, condition or score of a query."""

    # `==` builds a condition, so expressions are not hashable.
    __hash__ = None  # type: ignore[assignment]

    def _ir(self, ctx: _Ctx) -> Any:
        raise NotImplementedError

    def _field(self) -> str | None:
        """The field this names, if it is one."""
        return None

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return _Cmp("eq", self, other)

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return _Cmp("ne", self, other)

    def __lt__(self, other: Any) -> Expr:
        return _Cmp("lt", self, other)

    def __le__(self, other: Any) -> Expr:
        return _Cmp("le", self, other)

    def __gt__(self, other: Any) -> Expr:
        return _Cmp("gt", self, other)

    def __ge__(self, other: Any) -> Expr:
        return _Cmp("ge", self, other)

    def __and__(self, other: Expr) -> Expr:
        return _Bool("and", [self, _expr(other)])

    def __rand__(self, other: Expr) -> Expr:
        return _Bool("and", [_expr(other), self])

    def __or__(self, other: Expr) -> Expr:
        return _Bool("or", [self, _expr(other)])

    def __ror__(self, other: Expr) -> Expr:
        return _Bool("or", [_expr(other), self])

    def __invert__(self) -> Expr:
        return _Not(self)

    def __bool__(self) -> bool:
        raise TypeError("a query condition has no truth value here: combine conditions with &, | and ~, and compare one bound at a time")

    def isin(self, values: Iterable[Any]) -> Expr:
        """The expression equals one of `values` (a multi-valued field: one
        of its values does)."""
        if isinstance(values, (str, bytes)):
            raise TypeError("isin takes a list of values")
        return _In(self, list(values))

    def between(self, lo: Any, hi: Any) -> Expr:
        """`lo <= expr <= hi`; a bound of None is open."""
        return _Between(self, lo, hi)

    def startswith(self, prefix: str, fuzzy: int | str = 0) -> Expr:
        """The text starts with `prefix` (a multi-valued field: one of its
        values does); on a KEYWORD field, one run of its index, as SQL's
        `LIKE 'p%'` and `PREFIX(field, 'p')` are. `fuzzy` (1, 2 or
        "auto", by the prefix's length as MATCH does: none up to 2
        letters, 1 up to 5, else 2) also takes a prefix that many edits
        away whose first letter is the same, as `PREFIX(field, 'p', fuzzy
        => 1)` does."""
        if not isinstance(prefix, str):
            raise TypeError("startswith takes a str")
        if isinstance(fuzzy, str) and fuzzy.lower() == "auto":
            n = len(prefix)
            fuzzy = 0 if n <= 2 else 1 if n <= 5 else 2
        if fuzzy not in (0, 1, 2) or isinstance(fuzzy, bool):
            raise StrataError("fuzzy is 0, 1, 2 or 'auto'")
        return _Prefix(self, prefix, int(fuzzy))

    def is_null(self) -> Expr:
        """The expression has no value (a multi-valued field: holds none)."""
        return _IsNull(self)

    def is_not_null(self) -> Expr:
        return _Not(_IsNull(self))

    def alias(self, name: str) -> Expr:
        """The expression as a column named `name`."""
        return _Alias(self, name)

    def asc(self, nulls_first: bool = False) -> _Key:
        """Ascending, as an `order_by` key; rows without a value last
        unless `nulls_first`."""
        return _Key(self, False, nulls_first)

    def desc(self, nulls_first: bool = False) -> _Key:
        """Descending, as an `order_by` key."""
        return _Key(self, True, nulls_first)

    # Arithmetic, for a ranking: `+` adds raw values (minmax() normalises
    # a score first). abs(), round() and ** work as on numbers.
    def __add__(self, other: Any) -> Expr:
        return _Arith("add", self, other)

    def __radd__(self, other: Any) -> Expr:
        return _Arith("add", other, self)

    def __sub__(self, other: Any) -> Expr:
        return _Arith("sub", self, other)

    def __rsub__(self, other: Any) -> Expr:
        return _Arith("sub", other, self)

    def __mul__(self, other: Any) -> Expr:
        return _Arith("mul", self, other)

    def __rmul__(self, other: Any) -> Expr:
        return _Arith("mul", other, self)

    def __truediv__(self, other: Any) -> Expr:
        return _Arith("div", self, other)

    def __rtruediv__(self, other: Any) -> Expr:
        return _Arith("div", other, self)

    def __mod__(self, other: Any) -> Expr:
        return _Arith("mod", self, other)

    def __rmod__(self, other: Any) -> Expr:
        return _Arith("mod", other, self)

    def __neg__(self) -> Expr:
        return _Ranked("neg", [self], lambda ctx, a: {"neg": a[0]}, f"-{self!r}")

    def __pos__(self) -> Expr:
        return self

    def __abs__(self) -> Expr:
        return _call("abs", self)

    def __round__(self, digits: int | None = None) -> Expr:  # type: ignore[override]
        return _call("round", self) if digits is None else _call("round", self, digits)

    def __pow__(self, other: Any) -> Expr:
        return _call("pow", self, other)

    def __rpow__(self, other: Any) -> Expr:
        return _call("pow", other, self)

    # Aggregates of a field, for `select` and `having` over groups.
    def count(self, distinct: bool = False) -> Expr:
        """The rows with a value (`distinct`: the different values)."""
        return _Agg("count_distinct" if distinct else "count", self)

    def sum(self) -> Expr:
        return _Agg("sum", self)

    def avg(self) -> Expr:
        return _Agg("avg", self)

    def min(self) -> Expr:
        return _Agg("min", self)

    def max(self) -> Expr:
        return _Agg("max", self)

    def _name(self) -> str:
        """The column's name when it has no alias."""
        raise StrataError(f"{self!r} is not a column: select fields, score(), hops(), features(), aggregates or ranking expressions")

    def _is_score(self) -> bool:
        return False


def _expr(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    raise TypeError(f"{x!r} is not a query condition: build one with col(...) and the functions of strata")


def _num(x: Any) -> Expr:
    """An operand of arithmetic: an expression, or a number."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _Lit(float(x))
    raise TypeError(f"arithmetic takes expressions and numbers, not {x!r}")


def _operand(x: Any, ctx: _Ctx, field: Mapping[str, Any] | None) -> Any:
    return x._ir(ctx) if isinstance(x, Expr) else {"lit": _value(x, field)}


def _field_of(x: Any, ctx: _Ctx) -> Mapping[str, Any] | None:
    name = x._field() if isinstance(x, Expr) else None
    return ctx.field(name)[1] if name is not None else None


class _Col(Expr):
    def __init__(self, name: str):
        self.name = name

    def _field(self) -> str | None:
        return self.name

    def _ir(self, ctx: _Ctx) -> Any:
        i, _ = ctx.field(self.name)
        if ctx.outputs is not None:
            if ("field", i) not in ctx.outputs:
                raise StrataError(f"{self.name} is not a group key: name it in group_by, or aggregate it")
            return {"output": ctx.outputs[("field", i)]}
        return {"field": i}

    def _name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"col({self.name!r})"


class _Lit(Expr):
    def __init__(self, v: Any):
        self.v = v

    def _ir(self, ctx: _Ctx) -> Any:
        return {"lit": _value(self.v)}

    def __repr__(self) -> str:
        return f"lit({self.v!r})"


class _Cmp(Expr):
    def __init__(self, op: str, left: Any, right: Any):
        self.op, self.left, self.right = op, left, right

    def _ir(self, ctx: _Ctx) -> Any:
        if not isinstance(self.left, Expr):
            raise TypeError("a comparison names an expression first")
        if self.right is None:
            raise StrataError("a comparison with None is never true: use is_null() or is_not_null()")
        f = _field_of(self.left, ctx) if ctx.outputs is None else None
        return {"cmp": {"op": self.op, "left": self.left._ir(ctx), "right": _operand(self.right, ctx, f)}}

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


class _Bool(Expr):
    def __init__(self, op: str, parts: list[Expr]):
        flat: list[Expr] = []
        for p in parts:
            flat += p.parts if isinstance(p, _Bool) and p.op == op else [p]
        self.op, self.parts = op, flat

    def _ir(self, ctx: _Ctx) -> Any:
        return {self.op: [p._ir(ctx) for p in self.parts]}

    def __repr__(self) -> str:
        return f" {self.op} ".join(map(repr, self.parts)).join("()")


class _Not(Expr):
    def __init__(self, e: Expr):
        self.e = _expr(e)

    def _ir(self, ctx: _Ctx) -> Any:
        return {"not": self.e._ir(ctx)}

    def __repr__(self) -> str:
        return f"~{self.e!r}"


class _IsNull(Expr):
    def __init__(self, e: Expr):
        self.e = e

    def _ir(self, ctx: _Ctx) -> Any:
        return {"is_null": self.e._ir(ctx)}


class _Prefix(Expr):
    def __init__(self, e: Expr, prefix: str, edits: int = 0):
        self.e, self.prefix, self.edits = e, prefix, edits

    def _ir(self, ctx: _Ctx) -> Any:
        ir = {"expr": self.e._ir(ctx), "prefix": _operand(self.prefix, ctx, _field_of(self.e, ctx))}
        if self.edits:
            ir["edits"] = self.edits
        return {"prefix": ir}


class _In(Expr):
    def __init__(self, e: Expr, values: list[Any]):
        self.e, self.values = e, values

    def _ir(self, ctx: _Ctx) -> Any:
        f = _field_of(self.e, ctx)
        if any(v is None for v in self.values):
            raise StrataError("isin with None: None equals nothing; use is_null()")
        return {"in": {"expr": self.e._ir(ctx), "list": [_operand(v, ctx, f) for v in self.values]}}


class _Between(Expr):
    def __init__(self, e: Expr, lo: Any, hi: Any):
        self.e, self.lo, self.hi = e, lo, hi

    def _ir(self, ctx: _Ctx) -> Any:
        f = _field_of(self.e, ctx) if ctx.outputs is None else None
        b: dict[str, Any] = {"expr": self.e._ir(ctx)}
        if self.lo is not None:
            b["lo"] = _operand(self.lo, ctx, f)
        if self.hi is not None:
            b["hi"] = _operand(self.hi, ctx, f)
        return {"between": b}


class _Alias(Expr):
    def __init__(self, e: Expr, name: str):
        self.e, self.name = e, name

    def _ir(self, ctx: _Ctx) -> Any:
        return self.e._ir(ctx)

    def _name(self) -> str:
        return self.name

    def _is_score(self) -> bool:
        return self.e._is_score()


class _Key:
    """An `order_by` key: an expression and its direction."""

    def __init__(self, e: Expr, desc: bool, nulls_first: bool):
        self.e, self.desc, self.nulls_first = _expr(e), desc, nulls_first

    def _ir(self, ctx: _Ctx) -> dict:
        k: dict[str, Any] = {"expr": self.e._ir(ctx)}
        if self.desc:
            k["desc"] = True
        if self.nulls_first:
            k["nulls_first"] = True
        return k


class _Named(Expr):
    """`score()` or `hops()`."""

    def __init__(self, what: str):
        self.what = what

    def _ir(self, ctx: _Ctx) -> Any:
        return self.what

    def _name(self) -> str:
        return f"{self.what}()"

    def __repr__(self) -> str:
        return f"{self.what}()"


class _Fn(Expr):
    """A function of the row: a score (BM25, a similarity, RRF) or a
    condition (MATCH, a traversal). `inputs` are the scores RRF fuses."""

    def __init__(self, build, text: str, score: bool = True, inputs: list[Expr] | None = None, guard=None):
        self.build, self.text, self.score, self.inputs = build, text, score, inputs
        # The plan's guard, from a context, when the function has one.
        self.guard = guard

    def _ir(self, ctx: _Ctx) -> Any:
        return self.build(ctx)

    def _is_score(self) -> bool:
        return self.score

    def _name(self) -> str:
        if not self.score:
            return super()._name()
        return self.text

    def __repr__(self) -> str:
        return self.text


class _Ranked(Expr):
    """Part of a ranking expression: arithmetic, a function, minmax(),
    rank() or a case. It ranks best first, as a score does, unless it
    is over aggregates, which order groups. `inputs` are the parts whose
    values it reads (a case's values, not its conditions)."""

    def __init__(self, what: str, parts: list[Any], build, text: str, inputs: list[Any] | None = None):
        self.what, self.parts, self.build, self.text = what, parts, build, text
        self.inputs = parts if inputs is None else inputs

    def _ir(self, ctx: _Ctx) -> Any:
        return self.build(ctx, [p._ir(ctx) for p in self.parts])

    def _is_score(self) -> bool:
        return not any(isinstance(_bare(p), _Agg) for p in self.parts)

    def _name(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


def _Arith(op: str, left: Any, right: Any) -> Expr:
    left, right = _num(left), _num(right)
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[op]
    return _Ranked(op, [left, right], lambda ctx, a: {"arith": {"op": op, "left": a[0], "right": a[1]}}, f"({left!r} {sym} {right!r})")


def _call(func: str, *args: Any) -> Expr:
    parts = [_num(a) for a in args]
    return _Ranked(func, parts, lambda ctx, a: {"call": {"func": func, "args": a}}, f"{func}({', '.join(map(repr, parts))})")


class _Agg(Expr):
    def __init__(self, func: str, arg: Expr | None):
        self.func, self.arg = func, arg

    def key(self, ctx: _Ctx) -> tuple:
        return ("agg", self.func, None if self.arg is None else ctx.field(_arg_field(self.arg))[0])

    def _ir(self, ctx: _Ctx) -> Any:
        if ctx.outputs is None:
            raise StrataError(f"{self!r} is an aggregate: select it, or use it in having or order_by")
        return {"output": ctx.outputs[self.key(ctx)]}

    def agg(self, ctx: _Ctx) -> dict:
        a: dict[str, Any] = {"func": self.func}
        if self.arg is not None:
            a["arg"] = {"field": ctx.field(_arg_field(self.arg))[0]}
        return a

    def _name(self) -> str:
        if self.arg is None:
            return "COUNT(*)"
        f = _arg_field(self.arg)
        if self.func == "count_distinct":
            return f"COUNT(DISTINCT {f})"
        return f"{self.func.upper()}({f})"

    def __repr__(self) -> str:
        return self._name()


def _arg_field(e: Expr) -> str:
    name = e._field()
    if name is None:
        raise StrataError(f"an aggregate takes a field, not {e!r}")
    return name


def col(name: str) -> Expr:
    """The field `name`, as the table's schema names it."""
    return _Col(name)


def lit(value: Any) -> Expr:
    """A value, where an expression is wanted."""
    return _Lit(value)


def score() -> Expr:
    """The score the query ranked a row by."""
    return _Named("score")


def hops() -> Expr:
    """A row's hops from the seeds of the query's first `reachable` or
    `path`."""
    return _Named("hops")


def count() -> Expr:
    """The rows of a group (`COUNT(*)`); `col(f).count()` counts the rows
    with a value."""
    return _Agg("count", None)


def features() -> Expr:
    """In `select`, a column for each input of the query's ranking: each
    score and field it reads, then each ranking (`order_by`, `rerank`) that
    scores a row by itself, each once, named as Python writes it; the
    values the query ranked by, to train a model on."""
    return _Features()


class _Features(Expr):
    def _ir(self, ctx: _Ctx) -> Any:
        raise StrataError("features() stands for several columns: select it by itself")

    def __repr__(self) -> str:
        return "features()"


def _need(min: int | str) -> str:
    """A count or a percentage (`2`, `"75%"`), as MATCH's `@` writes it."""
    if isinstance(min, bool) or not isinstance(min, (int, str)):
        raise StrataError(f"min is a count or a percentage ('75%'), not {min!r}")
    return str(min)


def _rows(what: str, min: int | str | None, min_rows: int | None, max_rows: int | None) -> dict[str, int]:
    """`min_rows` and `max_rows` of a guard of `what`, which relaxes its
    `min`: only the ones given."""
    rows: dict[str, int] = {}
    for k, v in (("min_rows", min_rows), ("max_rows", max_rows)):
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise StrataError(f"{k} is a count of rows (500), not {v!r}")
            rows[k] = v
    if rows and min is None:
        raise StrataError(f"min_rows and max_rows on {what} relax its min, which it needs")
    return rows


def _need_ir(need: int | str) -> int | str:
    """A guard's loosest level: a count, or a percentage as text."""
    return need.strip() if isinstance(need, str) and need.strip().endswith("%") else int(need)


def _guarded(rows: dict[str, int], site, need: int | str):
    """The guard of a plan, from a context, if `rows` asks for one."""
    if not rows:
        return None
    return lambda ctx: {"site": site(ctx), "need": _need_ir(need), **rows}


def match(field: str, query: str, *, min: int | str | None = None, min_rows: int | None = None, max_rows: int | None = None) -> Expr:
    """`field` (an indexed TEXT field) matches the full-text `query`, in
    MATCH's syntax (`a b` both words, `a OR b` either, `"a b"` the phrase,
    `-a` not, `a~` with typos, `(a b c)@2` two of them). `min=2` or
    `min="75%"` asks for that many of the query's items. With
    `max_rows=500` (or `min_rows`), in `where`, it asks for as many items
    as it can, all of them down to `min`, while the rows fit (or are too
    few), as SQL's `MATCH(..., min => n, max_rows => 500)` does."""
    q = query if min is None else f"({query})@{_need(min)}"
    rows = _rows("match", min, min_rows, max_rows)
    site = lambda ctx: {"match": {"field": ctx.field(field)[0], "text": query}}
    extra = "".join(f", {k}={v!r}" for k, v in rows.items())
    return _Fn(lambda ctx: {"match": {"field": ctx.field(field)[0], "query": q}}, f"match({field!r}, {q!r}{extra})", score=False, guard=_guarded(rows, site, min))


def _of(need: int | str, y: int) -> int:
    """`need` of `y` items: a count, or a percentage (`"75%"`) rounded
    down, clamped to between 1 and `y`, as SQL's AT_LEAST reads it."""
    s = str(need).strip() if isinstance(need, (int, str)) and not isinstance(need, bool) else ""
    p = s[:-1] if s.endswith("%") else None
    if not (p if p is not None else s).isdigit() or (p is not None and int(p) > 100):
        raise StrataError(f"at_least takes a count (2) or a percentage ('75%'), not {need!r}")
    n = y * int(p) // 100 if p is not None else int(s)
    return max(1, min(n, y))


class _AtLeast(Expr):
    def __init__(self, need: int | str, conds: list[Expr], rows: dict[str, int]):
        self.text, self.conds, self.rows = need, conds, rows
        self.need = _of(need, len(conds))
        site = lambda ctx: {"at_least": {"conds": [c._ir(ctx) for c in self.conds]}}
        self.guard = _guarded(rows, site, need if isinstance(need, str) and need.strip().endswith("%") else self.need)

    def _ir(self, ctx: _Ctx) -> Any:
        return {"at_least": {"need": self.need, "conds": [c._ir(ctx) for c in self.conds]}}

    def __repr__(self) -> str:
        extra = "".join(f", {k}={v!r}" for k, v in self.rows.items())
        return f"at_least({self.text!r}, {', '.join(map(repr, self.conds))}{extra})"


def at_least(n: int | str, *conds: Expr, min_rows: int | None = None, max_rows: int | None = None) -> Expr:
    """At least `n` of the conditions hold, as SQL's `AT_LEAST(n, ...)`:
    `at_least(2, match("title", "covid"), col("year") >= 2020,
    col("venue").isin(["nejm", "lancet"]))`. `n` is a count or a
    percentage (`"75%"`, rounded down), clamped to between 1 and the
    number of conditions. A condition on a NULL is not met, and
    `~at_least(...)` keeps only the rows with too few met whatever their
    NULLs are. With `max_rows=500` (or `min_rows`), in `where`, it asks
    for as many of the conditions as it can, all of them down to `n`,
    while the rows fit (or are too few)."""
    if not conds:
        raise StrataError("at_least(n, *conds) takes one or more conditions")
    return _AtLeast(n, [_expr(c) for c in conds], _rows("at_least", n, min_rows, max_rows))


def bm25(field: str, text: str, *, syntax: str | None = None, min: int | str | None = None, min_rows: int | None = None, max_rows: int | None = None) -> Expr:
    """BM25 of a TEXT field for `text`, as a score. `text` may weigh a
    word (`word^2`), require one (`+word`, `+"a phrase"`) or exclude one
    (`-word`); `syntax="plain"` reads it as plain words. `min=2` or
    `min="75%"` ranks only the rows with that many of its optional words.
    With `max_rows=500` (or `min_rows`), in `order_by`, it ranks the rows
    with as many of them as it can, all of them down to `min`, while the
    rows fit (or are too few)."""
    if min is not None:
        _need(min)
    rows = _rows("bm25", min, min_rows, max_rows)

    def build(ctx: _Ctx) -> Any:
        m: dict[str, Any] = {"field": ctx.field(field)[0], "text": text}
        opts: dict[str, Any] = {}
        if syntax is not None:
            opts["syntax"] = {"text": syntax}
        if min is not None:
            opts["min"] = {"text": min} if isinstance(min, str) else {"int": min}
        if opts:
            m["opts"] = opts
        return {"bm25": m}

    extra = "".join(f", {k}={v!r}" for k, v in (("syntax", syntax), ("min", min), *rows.items()) if v is not None)
    site = lambda ctx: {"bm25": {"field": ctx.field(field)[0], "text": text}}
    return _Fn(build, f"bm25({field!r}, {text!r}{extra})", guard=_guarded(rows, site, min))


def similarity(field: str, query: Sequence[float] | Mapping[str, float]) -> Expr:
    """The similarity of a VECTOR field to a vector, or of a SPARSE field
    to `{term: weight}`, as a score. `Database.embed` and `Client.embed`
    give a text's vector."""

    def build(ctx: _Ctx) -> Any:
        i, f = ctx.field(field)
        if f["type"] == "sparse":
            if not isinstance(query, Mapping):
                raise StrataError(f"{field} is SPARSE: give {{term: weight}}")
            return {"sparse": {"field": i, "terms": [[t, float(w)] for t, w in query.items()]}}
        if isinstance(query, (str, Mapping)):
            raise StrataError(f"{field} takes a vector: embed the text with embed(text, {field!r})")
        return {"similarity": {"field": i, "vector": [float(x) for x in query]}}

    return _Fn(build, f"similarity({field!r}, ...)")


def maxsim(field: str, vectors: Iterable[Sequence[float]], *, max_vectors: int | None = None) -> Expr:
    """The best similarity of a VECTOR field to any of `vectors` (a
    user's recent items, a query's phrasings), as a score: one search
    per vector, at most 64 of them unless `max_vectors` says more. Over
    a multi-vector field (VECTOR(d)[]), the sum over `vectors` of each
    one's best match among a row's vectors."""
    vs = [[float(x) for x in v] for v in vectors]

    def build(ctx: _Ctx) -> Any:
        i, f = ctx.field(field)
        if f["type"] != "vector":
            raise StrataError(f"maxsim takes a VECTOR field, and {field} is not one")
        m: dict[str, Any] = {"field": i, "vectors": {"lit": {"list": [{"vector": v} for v in vs]}}}
        if max_vectors is not None:
            m["opts"] = {"max_vectors": {"int": int(max_vectors)}}
        return {"maxsim": m}

    return _Fn(build, f"maxsim({field!r}, {len(vs)} vectors)")


def rrf(*scores: Expr, k: float = 60, depth: int = 100) -> Expr:
    """Reciprocal rank fusion of rankings: each row scores Σ 1 / (k +
    its rank), over each ranking's best `depth` rows."""
    if not scores:
        raise StrataError("rrf takes rankings")
    return _Fn(lambda ctx: {"rrf": {"inputs": [s._ir(ctx) for s in scores], "k": float(k), "depth": int(depth)}}, f"rrf({', '.join(map(repr, scores))})", inputs=list(scores))


def ranked(keys: Iterable[Any]) -> Expr:
    """A ranking computed outside Strata, as a score: the rows of `keys`
    (primary keys, best first), each scoring -(its position) from 1, and
    rows not listed none. An input of `rrf`, or of an expression."""
    keys = [keys] if isinstance(keys, (str, int)) else list(keys)

    def build(ctx: _Ctx) -> Any:
        if not ctx.key:
            raise StrataError(f"ranked() ranks rows by their primary keys, and {ctx.table} has none")
        key = ctx.field(ctx.key)[1]
        return {"ranked": {"keys": [_value(k, key) for k in keys]}}

    return _Fn(build, f"ranked({keys!r})")


def minmax(x: Expr) -> Expr:
    """`x` put on [0, 1] over the rows it ranks: (x - min) / (max - min),
    1 where all are equal and 0 where `x` is NULL. A weighted sum of
    minmax() of scores is their weighted fusion."""
    x = _num(x)
    return _Ranked("minmax", [x], lambda ctx, a: {"norm": {"kind": "min_max", "input": a[0]}}, f"minmax({x!r})")


def tmm(x: Expr, lo: float | None = None, hi: float | None = None) -> Expr:
    """Theoretical min-max: (x - lo) / (max - lo), where `lo` is the least
    value `x` can take (0 for bm25(), -1 for similarity(); give it for any
    other) and `max` the largest over the rows it ranks; 0 where `x` is
    NULL. A row's value does not depend on how deep the candidates go,
    as minmax()'s does. With `hi` too the bounds are fixed: (x - lo) /
    (hi - lo), each row's own value. `0.3 * tmm(bm25(...)) + 0.7 *
    tmm(similarity(...))` is the convex fusion Bruch et al. (TOIS 2023)
    find best."""
    x = _num(x)
    for name, b in (("lo", lo), ("hi", hi)):
        if b is not None and not (isinstance(b, (int, float)) and math.isfinite(b)):
            raise StrataError(f"tmm's {name} is a finite number, not {b!r}")
    shown = f"tmm({x!r}{'' if lo is None else f', lo={lo!r}'}{'' if hi is None else f', hi={hi!r}'})"
    if hi is not None:
        if lo is None:
            raise StrataError("tmm(x, hi=...) needs lo too: the bounds are fixed together")
        if hi <= lo:
            raise StrataError(f"tmm's hi ({hi}) must be above its lo ({lo})")
        return coalesce((x - lo) / (hi - lo), 0)

    def build(ctx: _Ctx, a: list) -> Any:
        n = {"kind": "tmm", "input": a[0]}
        if lo is not None:
            n["lo"] = float(lo)
        return {"norm": n}

    return _Ranked("tmm", [x], build, shown)


def zscore(x: Expr) -> Expr:
    """`x` centred and scaled over the rows it ranks: (x - mean) / sd, 0
    where all are equal; a row where `x` is NULL takes the least value."""
    x = _num(x)
    return _Ranked("zscore", [x], lambda ctx, a: {"norm": {"kind": "z_score", "input": a[0]}}, f"zscore({x!r})")


def rank(x: Expr) -> Expr:
    """A row's place in `x`'s ranking of the rows it ranks: 1 plus the
    rows with a greater value; NULL places after every other."""
    x = _num(x)
    return _Ranked("rank", [x], lambda ctx, a: {"norm": {"kind": "rank", "input": a[0]}}, f"rank({x!r})")


def gbdt(model: str, *features: Any) -> Expr:
    """A ranking model's score of a row from its features (scores, fields,
    any ranking expression of the row), in the order it was trained on:
    the gradient-boosted trees of `<models>/<model>/gbdt.json`, saved by
    LightGBM (`json.dump(booster.dump_model(), f)`) or XGBoost
    (`booster.save_model("gbdt.json")`). The raw score, as LightGBM's
    `predict(raw_score=True)` and XGBoost's `output_margin=True` give it:
    `sigmoid(gbdt(...))` is a binary model's probability. A plan names the
    hash of the model's file, and a database whose file differs refuses
    it."""
    if not features:
        raise StrataError("gbdt takes a model's name and the features it reads")
    parts = [_num(f) for f in features]
    return _Ranked("gbdt", parts, lambda ctx, a: {"gbdt": {"model": model, "args": a}}, f"gbdt({model!r}, {', '.join(map(repr, parts))})")


def cross_encoder(model: str, fields: str | Sequence[str], query: str) -> Expr:
    """A cross-encoder's score of how well a row's text answers `query`:
    the model `<models>/<model>` (a BERT or XLM-RoBERTa sequence
    classifier saved by Hugging Face, such as `ms-marco-MiniLM-L-6-v2`)
    reads the query and the text of `fields` (stored TEXT or KEYWORD
    fields, their values joined by a space) together. Milliseconds a row,
    so it reranks the top rows of a cheaper ranking:
    `.order_by(bm25(...)).rerank(cross_encoder(...), 100)`, a global
    phase that runs once per query. A plan names the hash of the model's
    files; `builtin:overlap` needs none (the share of the query's words a
    text holds)."""
    names = [fields] if isinstance(fields, str) else list(fields)
    if not names:
        raise StrataError("cross_encoder reads the text of at least one field")
    build = lambda ctx: {"cross_encoder": {"model": model, "fields": [ctx.field(f)[0] for f in names], "query": str(query)}}
    return _Fn(build, f"cross_encoder({model!r}, {', '.join(names)})")


def ln(x: Any) -> Expr:
    """The natural logarithm; NULL for x <= 0."""
    return _call("ln", x)


def log(base: Any, x: Any = None) -> Expr:
    """`log(x)` is the base 10 logarithm, `log(base, x)` the base `base`
    one."""
    return _call("log", base) if x is None else _call("log", base, x)


def exp(x: Any) -> Expr:
    return _call("exp", x)


def sqrt(x: Any) -> Expr:
    """The square root; NULL for x < 0."""
    return _call("sqrt", x)


def sigmoid(x: Any) -> Expr:
    """1 / (1 + e^-x)."""
    return _call("sigmoid", x)


def clamp(x: Any, lo: Any, hi: Any) -> Expr:
    """`x` within [lo, hi]."""
    return _call("clamp", x, lo, hi)


def least(*xs: Any) -> Expr:
    """The smallest of `xs` that is not NULL."""
    return _call("least", *xs)


def greatest(*xs: Any) -> Expr:
    """The largest of `xs` that is not NULL."""
    return _call("greatest", *xs)


def coalesce(*xs: Any) -> Expr:
    """The first of `xs` that is not NULL."""
    return _call("coalesce", *xs)


def case(*whens: tuple[Expr, Any], otherwise: Any = None) -> Expr:
    """The value of the first `(condition, value)` whose condition holds,
    else `otherwise`, else NULL: `case((col("year") > 2020, 1), otherwise=0)`."""
    if not whens:
        raise StrataError("case takes (condition, value) pairs")
    conds = [_expr(c) for c, _ in whens]
    values = [_num(v) for _, v in whens]
    rest = [] if otherwise is None else [_num(otherwise)]
    n = len(whens)

    def build(ctx: _Ctx, a: list) -> Any:
        c: dict[str, Any] = {"whens": [{"when": a[i], "then": a[n + i]} for i in range(n)]}
        if rest:
            c["otherwise"] = a[2 * n]
        return {"case": c}

    text = " ".join(f"WHEN {c!r} THEN {v!r}" for c, v in zip(conds, values))
    return _Ranked("case", conds + values + rest, build, f"CASE {text}{f' ELSE {rest[0]!r}' if rest else ''} END", inputs=values + rest)


def count_true(*conds: Expr) -> Expr:
    """How many of the conditions hold, a number to rank by:
    `bm25("body", q) * (1 + 0.1 * count_true(col("year") >= 2021,
    col("venue").isin(["nejm", "lancet"])))`. A condition on a NULL counts
    as not met."""
    if not conds:
        raise StrataError("count_true(*conds) takes one or more conditions")
    parts = [_expr(c) for c in conds]
    return _Fn(lambda ctx: {"call": {"func": "count_true", "args": [c._ir(ctx) for c in parts]}}, f"count_true({', '.join(map(repr, parts))})")


def _graph(field: str, seeds: Any, kind: dict, dir: str, labels: Iterable[str], through: Expr | None, max_nodes: int, k: int, min_sim: float | None):
    def build(ctx: _Ctx) -> dict:
        key = ctx.field(ctx.key)[1] if ctx.key else None
        g: dict[str, Any] = {"field": ctx.field(field)[0], "seeds": _seeds(seeds, key), "kind": kind(key) if callable(kind) else kind, "dir": dir}
        ls = [labels] if isinstance(labels, str) else list(labels)
        if ls:
            g["labels"] = ls
        if through is not None:
            g["through"] = through._ir(ctx)
        g["max_nodes"] = int(max_nodes)
        g["k"] = int(k)
        if min_sim is not None:
            g["min_sim"] = float(min_sim)
        return g

    return build


def _seeds(seeds: Any, key: Mapping[str, Any] | None) -> dict:
    keys = [seeds] if isinstance(seeds, (str, int)) else list(seeds)
    return {"lit": {"list": [_value(s, key) for s in keys]}}


def reachable(field: str, seeds: Any, hops: int = 1, *, min_hops: int | None = None, dir: str = "out", labels: Iterable[str] = (), through: Expr | None = None, max_nodes: int = 100_000, k: int = 10, min_sim: float | None = None) -> Expr:
    """The rows `min_hops` to `hops` steps from `seeds` (a key or keys)
    along an EDGES field, or a VECTOR field's `k` nearest rows; `dir` is
    "out", "in" or "both", and `through` a condition the rows passed
    through hold."""
    lo = min(1, hops) if min_hops is None else min_hops
    b = _graph(field, seeds, {"kind": "reachable", "min_hops": int(lo), "max_hops": int(hops)}, dir, labels, through, max_nodes, k, min_sim)
    return _Fn(lambda ctx: {"graph": b(ctx)}, f"reachable({field!r}, {seeds!r})", score=False)


def path(field: str, seeds: Any, to: Any, *, max_hops: int = 6, dir: str = "out", labels: Iterable[str] = (), through: Expr | None = None, max_nodes: int = 100_000, k: int = 10, min_sim: float | None = None) -> Expr:
    """The rows on the shortest paths from `seeds` to `to`."""
    b = _graph(field, seeds, lambda key: {"kind": "path", "to": _seeds(to, key), "max_hops": max(1, int(max_hops))}, dir, labels, through, max_nodes, k, min_sim)
    return _Fn(lambda ctx: {"graph": b(ctx)}, f"path({field!r}, {seeds!r}, {to!r})", score=False)


def ppr(field: str, seeds: Any, *, alpha: float = 0.15, eps: float = 1e-4, max_pushes: int = 1_000_000, weighted: bool = False, dir: str = "both", labels: Iterable[str] = ()) -> Expr:
    """Personalized PageRank from `seeds`, as a score."""
    b = _graph(field, seeds, {"kind": "ppr", "alpha": float(alpha), "eps": float(eps), "max_pushes": int(max_pushes), "weighted": bool(weighted)}, dir, labels, None, 100_000, 10, None)
    return _Fn(lambda ctx: {"ppr": b(ctx)}, f"ppr({field!r}, {seeds!r})")


class Query:
    """A query of one table, built a clause at a time; each method returns
    a new query. `plan()` gives its plan, and `run()` (or iterating it)
    runs it."""

    def __init__(self, source: Any, schema: Mapping[str, Any]):
        self._source = source
        self._schema = schema
        self._select: list[Expr] | None = None
        self._where: list[Expr] = []
        self._order: list[_Key] = []
        self._group: list[Expr] = []
        self._having: list[Expr] = []
        self._phases: list[tuple[Expr, int]] = []
        self._limit: int | None = None
        self._offset = 0
        self._distinct_on: list[Expr] = []
        self._per = 1
        self._per_field: str | None = None

    def _copy(self, **kw: Any) -> Query:
        q = Query.__new__(Query)
        q.__dict__.update(self.__dict__)
        for k, v in kw.items():
            setattr(q, "_" + k, v)
        return q

    def select(self, *columns: str | Expr) -> Query:
        """The columns of the answer: field names, `col(...)`, `score()`,
        `hops()`, and over groups aggregates (`count()`,
        `col(f).avg()`), each renamed with `.alias(name)`. Without it, the
        stored fields but vectors."""
        return self._copy(select=[col(c) if isinstance(c, str) else _expr(c) for c in columns])

    def where(self, *conditions: Expr) -> Query:
        """Keeps the rows every condition holds for."""
        return self._copy(where=self._where + [_expr(c) for c in conditions])

    def order_by(self, *keys: Expr | _Key) -> Query:
        """Ranks by a score (best first), or orders by a field or an
        aggregate (ascending; `.desc()` for descending), then possibly by
        the primary key or the first group key."""
        out = []
        for k in keys:
            if isinstance(k, _Key):
                out.append(k)
            else:
                e = _expr(k)
                out.append(_Key(e, e._is_score(), False))
        return self._copy(order=out)

    def rerank(self, expr: Expr, count: int) -> Query:
        """Ranks the top `count` rows of the ranking before (`order_by`, or
        the last `rerank`) again by `expr`, best first, rows tied keeping
        their order; `limit` and `offset` cut the last one. A phase with
        `minmax`, `rank` or `rrf`, and any after it, runs once over every
        row it reranks (a global phase); any other on each shard over its
        own rows."""
        if int(count) < 1:
            raise StrataError("a rerank phase reranks at least one row")
        return self._copy(phases=self._phases + [(_expr(expr), int(count))])

    def group_by(self, *fields: str | Expr) -> Query:
        """Groups the rows by the values of fields (each value of a
        multi-valued field its own group)."""
        return self._copy(group=[col(f) if isinstance(f, str) else _expr(f) for f in fields])

    def distinct_on(self, *keys: str | Expr) -> Query:
        """Keeps the first row of each group of rows with equal keys
        (fields or expressions), in the query's order, as SQL's
        `SELECT DISTINCT ON (...)`: `.distinct_on("video")` with
        `.order_by(score)` gives each video's best row, the best videos
        first, and `limit` counts groups."""
        return self._copy(distinct_on=[col(k) if isinstance(k, str) else _expr(k) for k in keys], per=1, per_field=None)

    def max_per(self, k: int, field: str) -> Query:
        """Keeps at most `k` rows of each value of `field`, the first `k`
        in the query's order, as SQL's `MAX k PER field`:
        `.order_by(score).max_per(2, "venue").limit(20)` gives the best 20
        rows with no more than 2 of a venue. NULL is one value; a
        multi-valued field is refused."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise StrataError(f"max_per keeps a whole number of rows, 1 or more ({k!r})")
        if not isinstance(field, str):
            raise StrataError("max_per takes a field's name")
        return self._copy(distinct_on=[col(field)], per=k, per_field=field)

    def having(self, *conditions: Expr) -> Query:
        """Keeps the groups every condition on their aggregates holds for."""
        return self._copy(having=self._having + [_expr(c) for c in conditions])

    def limit(self, n: int | None) -> Query:
        return self._copy(limit=None if n is None else int(n))

    def offset(self, n: int) -> Query:
        return self._copy(offset=int(n))

    def plan(self) -> dict:
        """The query's plan, as `Database.plan` gives a statement's."""
        ctx = _Ctx(self._schema)
        search: dict[str, Any] = {"node": "search", "table": ctx.table}
        # A guard is a condition of where() joined to the rest by AND, or
        # a bm25 of the ranking, taken out of them into the search.
        where = [p for c in self._where for p in (c.parts if isinstance(c, _Bool) and c.op == "and" else [c])]
        guards = [c for c in where if getattr(c, "guard", None) is not None]
        where = [c for c in where if getattr(c, "guard", None) is None]
        if any(_guards(c) for c in where):
            raise StrataError("min_rows and max_rows guard a condition of where() joined to the others by AND, not one inside |, ~ or another condition")
        guards += [g for k in self._order for g in _guards(k.e)]
        if any(_guards(e) for e, _ in self._phases):
            raise StrataError("min_rows and max_rows guard a search's first ranking, not a rerank phase")
        if any(_guards(c) for c in self._select or []):
            raise StrataError("min_rows and max_rows on bm25 guard the ranking of order_by")
        if len(guards) > 1:
            raise StrataError("a query has one guard (min_rows, max_rows) at most")
        if guards:
            search["guard"] = guards[0].guard(ctx)
        if where:
            cond = where[0] if len(where) == 1 else _Bool("and", where)
            search["filter"] = cond._ir(ctx)
        columns = self._select
        if columns is None:
            stored = [f for f in ctx.fields if f.get("stored", f["type"] != "vector" and not f.get("from")) and f["type"] != "vector"]
            columns = [col(f["name"]) for f in stored]
        grouped = bool(self._group) or any(isinstance(_bare(c), _Agg) for c in columns)
        if any(isinstance(c, _Features) for c in columns):
            if grouped:
                raise StrataError("features() lists the inputs of a ranking of rows, not of groups")
            columns = [x for c in columns for x in (self._features(ctx) if isinstance(c, _Features) else [c])]
        node: dict[str, Any] = search
        if not grouped:
            if self._having:
                raise StrataError("having keeps groups: group the rows with group_by")
            if self._order:
                search["order"] = [k._ir(ctx) for k in self._order]
            if self._phases:
                if len(self._order) != 1:
                    raise StrataError("rerank phases follow an order_by of one key")
                phases, after = [], False
                for e, n in self._phases:
                    expr = e._ir(ctx)
                    after = after or _reads_every_row(expr) or _has_cross_encoder(expr)
                    phases.append({"count": n, "expr": expr, **({"global": True} if after else {})})
                search["phases"] = phases
        else:
            if self._phases:
                raise StrataError("rerank phases rank rows, not groups")
            if self._order and any(k.e._is_score() for k in self._order):
                raise StrataError("groups are ordered by a group key or an aggregate, not a score")
            keys = [ctx.field(_arg_field(g))[0] for g in self._group]
            outputs: dict[tuple, int] = {("field", f): n for n, f in enumerate(keys)}
            aggs: list[dict] = []
            for e in list(columns) + self._having + [k.e for k in self._order]:
                for a in _aggs(e):
                    k = a.key(ctx)
                    if k not in outputs:
                        outputs[k] = len(keys) + len(aggs)
                        aggs.append(a.agg(ctx))
            node = {"node": "aggregate", "input": search, "keys": [{"field": f} for f in keys], "aggs": aggs}
            ctx.outputs = outputs
            if self._having:
                cond = self._having[0] if len(self._having) == 1 else _Bool("and", self._having)
                node = {"node": "filter", "input": node, "cond": cond._ir(ctx)}
            if self._order:
                node = {"node": "sort", "input": node, "keys": [k._ir(ctx) for k in self._order]}
        features: list[str] = []
        if self._distinct_on:
            if grouped:
                raise StrataError("distinct_on keeps rows of the table: not with group_by or aggregates")
            if self._per_field is not None:
                if not self._order:
                    raise StrataError("max_per needs an order_by: it keeps each value's first rows in its order")
                if ctx.field(self._per_field)[1].get("multi"):
                    raise StrataError(f"max_per: {self._per_field} is multi-valued, and a row with several values is in several groups")
            # The scan's rows with the answer's columns and the fields of
            # the keys, which are computed over them, the first row of each
            # group kept in their order. A LIMIT reads the scan in growing
            # slices; the executor's own cap is lifted.
            n = len(columns)
            fields: list[int] = []
            for k in self._distinct_on:
                for name in _cols(k):
                    i = ctx.field(name)[0]
                    if i not in fields:
                        fields.append(i)
            scan = {"node": "limit", "input": node, "limit": _MAX_ROWS + 1}
            exprs = [c._ir(ctx) for c in columns] + [{"field": i} for i in fields]
            leaf = {"node": "project", "input": scan, "columns": [{"name": f"#{i}", "expr": e} for i, e in enumerate(exprs)]}
            ctx.outputs = {("field", f): n + j for j, f in enumerate(fields)}
            node = {"node": "distinct_on", "input": leaf, "keys": [k._ir(ctx) for k in self._distinct_on]}
            ctx.outputs = None
            features = ["operators", "distinct_on"]
            if self._per > 1:
                node["per"] = self._per
                features.append("per_cap")
        if self._limit is not None or self._offset:
            node = {"node": "limit", "input": node}
            if self._offset:
                node["offset"] = self._offset
            if self._limit is not None:
                node["limit"] = self._limit
        if features:
            out = [{"name": c._name(), "expr": {"output": i}} for i, c in enumerate(columns)]
        else:
            out = [{"name": c._name(), "expr": c._ir(ctx)} for c in columns]
        root = {"node": "project", "input": node, "columns": out}
        plan: dict[str, Any] = {"version": VERSION, "root": root}
        if features:
            plan["features"] = features
        return plan

    def _features(self, ctx: _Ctx) -> list[Expr]:
        """What `features()` stands for: the leaves of every ranking, then
        each ranking that scores a row by itself and is not a leaf."""
        ranks = [k.e for k in self._order[:1]] + [e for e, _ in self._phases]
        if not ranks:
            raise StrataError("features() lists the inputs of the query's ranking, and it has no order_by")

        def leaves(e: Any, out: list[Expr]) -> None:
            e = _bare(e)
            if isinstance(e, _Lit):
                return
            if isinstance(e, _Ranked) or (isinstance(e, _Fn) and e.inputs is not None):
                for x in e.inputs:
                    leaves(x, out)
            elif not _has_cross_encoder(e._ir(ctx)):
                out.append(e)

        items: list[Expr] = []
        for e in ranks:
            leaves(e, items)
        for e in ranks:
            b = _bare(e)
            inner = isinstance(b, _Ranked) or (isinstance(b, _Fn) and b.inputs is not None)
            if inner and not _reads_every_row(b._ir(ctx)) and not _has_cross_encoder(b._ir(ctx)):
                items.append(b)
        out, seen = [], set()
        for e in items:
            if e._name() not in seen:
                seen.add(e._name())
                out.append(e)
        return out

    def run(self) -> Any:
        """Runs the query: its `Result`."""
        return self._source.run(self.plan())

    def __iter__(self):
        return iter(self.run())

    def __repr__(self) -> str:
        return f"<Query of {self._schema['table']}>"


def _reads_every_row(ir: Any) -> bool:
    """Whether a ranking's plan holds a set function (`minmax`, `rank`),
    RRF or a weighted sum, which read every row ranked."""
    if isinstance(ir, dict):
        return any(k in ("norm", "rrf", "weighted") or _reads_every_row(v) for k, v in ir.items())
    if isinstance(ir, list):
        return any(_reads_every_row(v) for v in ir)
    return False


def _has_cross_encoder(ir: Any) -> bool:
    """Whether a ranking's plan holds a cross-encoder, which ranks in a
    global phase and is no feature."""
    if isinstance(ir, dict):
        return any(k == "cross_encoder" or _has_cross_encoder(v) for k, v in ir.items())
    if isinstance(ir, list):
        return any(_has_cross_encoder(v) for v in ir)
    return False


def _bare(e: Expr) -> Expr:
    return e.e if isinstance(e, _Alias) else e


def _cols(e: Any) -> list[str]:
    """The fields an expression names."""
    if isinstance(e, _Col):
        return [e.name]
    out: list[str] = []
    for v in getattr(e, "__dict__", {}).values():
        for x in v if isinstance(v, list) else [v]:
            if isinstance(x, Expr):
                out += _cols(x)
    return out


def _guards(e: Any) -> list[Expr]:
    """The guarded conditions and scores in an expression."""
    if getattr(e, "guard", None) is not None:
        return [e]
    out: list[Expr] = []
    for k, v in getattr(e, "__dict__", {}).items():
        for x in v if isinstance(v, list) else [v]:
            if isinstance(x, Expr):
                out += _guards(x)
    return out


def _aggs(e: Any) -> list[_Agg]:
    """The aggregates in an expression."""
    if isinstance(e, _Agg):
        return [e]
    out: list[_Agg] = []
    for v in getattr(e, "__dict__", {}).values():
        for x in v if isinstance(v, list) else [v]:
            if isinstance(x, Expr):
                out += _aggs(x)
    return out
