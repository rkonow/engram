//! Ranking expressions: a score made of others (RRF, a weighted sum, a
//! product, arithmetic and functions of scores and row values) as one
//! expression over its leaves, and the one evaluator of that expression,
//! which a database and a coordinator both run.
//!
//! A database ranks by an expression in three steps
//! ([`crate::query::exec`]):
//!
//! 1. **Candidates.** The top `depth` rows of each leaf that generates them
//!    (a relevance score: BM25, similarity, a graph's; every input of RRF
//!    and of a weighted sum; a product's relevance factors, or all its
//!    factors when none is), or every row that passes `WHERE` with
//!    `exact => 1` or where no leaf generates any ([`Rank::candidates`]).
//! 2. **Values.** Each leaf's exact value of every candidate, where the
//!    expression reads values and not only rankings ([`Rank::reads_values`]).
//! 3. **The expression** over those values and the leaves' rankings
//!    ([`Rank::eval`]).
//!
//! A shard answers every candidate with each leaf's value and whether the row
//! is in the leaf's top `depth` ([`crate::query::exec::RankParts`]). The
//! coordinator takes each leaf's top `depth` over all shards and runs steps 1
//! and 3 over those rows with the same code ([`crate::cluster`]), so a split
//! collection ranks like the whole one.
//!
//! NULL is a value that is not a finite number (NaN inside the evaluator):
//! a row a leaf has no value for (a NULL column, a row without a vector, one
//! a traversal does not reach), and the result of `ln(0)`, a zero divisor
//! or an overflow. Arithmetic and functions of NULL are NULL (`coalesce`,
//! `least` and `greatest` skip it), and a row whose score is NULL ranks
//! last. Set functions and the fused shapes keep the rule from before
//! expressions: a missing value counts as the input's worst, so a row never
//! gains from one.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::query::fusion::{self, Norm};
use crate::query::ir::{ArithOp, Func, NormKind};
use crate::query::logical::{Pred, Score};
use crate::query::model::Gbdt;
use crate::schema::Schema;

/// A ranking expression, ready to run.
#[derive(Clone, Debug)]
pub struct Rank<'s> {
    /// The scores the expression reads, each of one row by itself, each
    /// once.
    pub leaves: Vec<&'s Score>,
    /// Whether each leaf's top rows are candidates.
    pub generates: Vec<bool>,
    /// The top rows each generating leaf gives, before the query raises it
    /// to the rows its page needs.
    pub depth: usize,
    /// Every row that passes `WHERE` is a candidate.
    pub exact: bool,
    pub expr: Expr,
}

/// An expression over a [`Rank`]'s leaves and their rankings, computed for a
/// set of candidates.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A leaf's exact value; NULL where the row has none.
    Leaf(usize),
    /// `Σ wᵢ · eᵢ`, each term min-max normalised over the candidates
    /// ([`Norm::MinMax`]) or raw ([`Norm::None`]). A missing value counts as
    /// the worst of its term over the candidates ([`fusion::combine`]).
    Sum { terms: Vec<(f64, Expr)>, norm: Norm },
    /// `Π eᵢ`; a factor below 0, or missing, counts as 0.
    Product(Vec<Expr>),
    /// `Σ 1 / (k + rank)` over the leaves' rankings of their top rows, ranks
    /// from 1; a row gets nothing from a ranking it is not in.
    Rrf { leaves: Vec<usize>, k: f64 },
    Num(f64),
    Arith { op: ArithOp, left: Box<Expr>, right: Box<Expr> },
    Neg(Box<Expr>),
    Call { func: Func, args: Vec<Expr> },
    /// A set function over the candidates ([`NormKind`]); `lo` is the
    /// least value of `input`, which `tmm` reads.
    Norm { kind: NormKind, input: Box<Expr>, lo: f64 },
    /// The value of the first branch whose test is a number other than 0,
    /// else `otherwise`, else NULL.
    Case { whens: Vec<(Expr, Expr)>, otherwise: Option<Box<Expr>> },
    /// A ranking model's score of the row from its features `args`; NULL
    /// until the model is opened ([`crate::query::model::Models::open`]),
    /// which a plan is before it runs.
    Gbdt { name: String, model: Option<Arc<Gbdt>>, args: Vec<Expr> },
}

/// Builds a [`Rank`] from a [`Score`].
struct Compile<'s> {
    leaves: Vec<&'s Score>,
    generates: Vec<bool>,
    /// An `rrf(.., depth => n)`'s depth, and a `depth => n` given on any
    /// leaf.
    rrf_depth: Option<usize>,
    depth: Option<usize>,
    exact: bool,
}

impl<'s> Compile<'s> {
    /// The index of leaf `s`, added once; `generates` if its top rows are
    /// candidates.
    fn leaf(&mut self, s: &'s Score, generates: bool) -> usize {
        let i = match self.leaves.iter().position(|l| *l == s) {
            Some(i) => i,
            None => {
                self.leaves.push(s);
                self.generates.push(false);
                if let Some(o) = s.opts() {
                    if self.depth.is_none() {
                        self.depth = o.get("depth").and_then(|v| v.as_f64()).map(|d| d as usize);
                    }
                    self.exact |= o.get("exact").and_then(|v| v.as_i64()).unwrap_or(0) != 0;
                }
                self.leaves.len() - 1
            }
        };
        self.generates[i] |= generates;
        i
    }

    fn expr(&mut self, s: &'s Score) -> Expr {
        match s {
            Score::Rrf { inputs, k, depth } => {
                self.rrf_depth.get_or_insert(*depth);
                Expr::Rrf { leaves: inputs.iter().map(|x| self.leaf(x, true)).collect(), k: *k }
            }
            Score::Weighted(terms) => {
                // An option holds for the whole sum, given on any term.
                let norm = match terms.iter().find_map(|(_, s)| s.opts().and_then(|o| o.get("norm")).and_then(|v| v.as_str())) {
                    Some("none") => Norm::None,
                    _ => Norm::MinMax,
                };
                Expr::Sum { terms: terms.iter().map(|(w, s)| (*w, Expr::Leaf(self.leaf(s, true)))).collect(), norm }
            }
            Score::Product(factors) => {
                let any_relevance = factors.iter().any(Score::is_relevance);
                Expr::Product(factors.iter().map(|f| Expr::Leaf(self.leaf(f, f.is_relevance() || !any_relevance))).collect())
            }
            Score::Num(x) => Expr::Num(*x),
            Score::Arith { op, left, right } => Expr::Arith { op: *op, left: Box::new(self.expr(left)), right: Box::new(self.expr(right)) },
            Score::Neg(x) => Expr::Neg(Box::new(self.expr(x))),
            Score::Call { func, args } => Expr::Call { func: *func, args: args.iter().map(|a| self.expr(a)).collect() },
            Score::Norm { kind, input, lo } => Expr::Norm { kind: *kind, input: Box::new(self.expr(input)), lo: lo.unwrap_or(0.0) },
            Score::Gbdt { model, args, open, .. } => Expr::Gbdt { name: model.clone(), model: open.0.clone(), args: args.iter().map(|a| self.expr(a)).collect() },
            Score::Case { whens, otherwise } => Expr::Case {
                whens: whens.iter().map(|(c, x)| (self.expr(c), self.expr(x))).collect(),
                otherwise: otherwise.as_ref().map(|x| Box::new(self.expr(x))),
            },
            s => Expr::Leaf(self.leaf(s, s.is_relevance())),
        }
    }
}

impl<'s> Rank<'s> {
    /// The ranking expression `score`, or `None` for a leaf the executor
    /// ranks by directly (BM25, similarity, a column...). `default_depth`
    /// is the depth where neither an `rrf()` nor a leaf sets one.
    pub fn of(score: &'s Score, default_depth: usize) -> Option<Rank<'s>> {
        if score.is_leaf() && !matches!(score, Score::SeqNo | Score::Cond(_)) {
            return None;
        }
        let mut c = Compile { leaves: Vec::new(), generates: Vec::new(), rrf_depth: None, depth: None, exact: false };
        let expr = c.expr(score);
        // Without a leaf to take candidates from, every row is one.
        let exact = c.exact || !c.generates.contains(&true);
        Some(Rank { depth: c.rrf_depth.or(c.depth).unwrap_or(default_depth), exact, expr, leaves: c.leaves, generates: c.generates })
    }

    /// The ranking of a rerank phase ([`crate::query::logical::Phase`]):
    /// `score`'s expression, or the leaf alone, every row it reranks a
    /// candidate.
    pub fn of_phase(score: &'s Score) -> Rank<'s> {
        match Rank::of(score, usize::MAX) {
            Some(r) => Rank { exact: true, ..r },
            None => Rank { leaves: vec![score], generates: vec![score.is_relevance()], depth: usize::MAX, exact: true, expr: Expr::Leaf(0) },
        }
    }

    /// The score of each of `n` rows that a phase reranks, best first:
    /// `values[j]` is leaf `j`'s value of each row (NaN where it has none),
    /// rows in the previous phase's order. Rows tied on score keep that
    /// order (under RRF, the order they first appear in a ranking); RRF
    /// ranks the rows by each of its inputs, a row without a value out of
    /// that ranking.
    pub fn rerank(&self, n: usize, values: &[Vec<f64>]) -> Vec<(usize, f64)> {
        let lists: Vec<Vec<usize>> = (0..self.leaves.len())
            .map(|j| match self.generates[j] && self.reads_rankings() {
                true => {
                    let mut l: Vec<usize> = (0..n).filter(|&i| !values[j][i].is_nan()).collect();
                    l.sort_by(|&a, &b| values[j][b].total_cmp(&values[j][a]));
                    l.truncate(self.depth);
                    l
                }
                false => Vec::new(),
            })
            .collect();
        self.eval(n, values, &lists, n)
    }

    /// Whether a rerank phase by `score` reads every row it reranks (a set
    /// function, RRF or a weighted sum), or scores with a cross-encoder, so
    /// that it runs once over all of them: a global phase
    /// ([`crate::query::logical::Phase::global`]).
    pub fn is_global(score: &Score) -> bool {
        score.has_cross_encoder() || Rank::of(score, 0).is_some_and(|r| !r.is_row_local())
    }

    /// [`Rank::rerank`] of rows whose phase scores a shard computed:
    /// `scores` (NaN for NULL), best first, rows tied keeping their order.
    pub fn by_scores(scores: Vec<f64>) -> Vec<(usize, f64)> {
        let n = scores.len();
        Rank { leaves: Vec::new(), generates: Vec::new(), depth: n, exact: true, expr: Expr::Leaf(0) }.eval(n, &[scores], &[], n)
    }

    /// Whether the expression reads the leaves' values; RRF reads only
    /// their rankings.
    pub fn reads_values(&self) -> bool {
        let mut reads = false;
        visit(&self.expr, &mut |e| reads |= matches!(e, Expr::Leaf(_)));
        reads
    }

    /// Whether the expression reads the leaves' rankings (RRF does).
    pub fn reads_rankings(&self) -> bool {
        let mut reads = false;
        visit(&self.expr, &mut |e| reads |= matches!(e, Expr::Rrf { .. }));
        reads
    }

    /// Whether the expression adds raw relevance scores (`bm25(..) +
    /// similarity(..)`), which meant min-max fusion before ranking
    /// expressions: plan lines say so, for one release.
    pub fn adds_raw_scores(&self) -> bool {
        // The leaves an expression reads raw: not through a set function or
        // a fused shape.
        fn raw(e: &Expr, out: &mut Vec<usize>) {
            match e {
                Expr::Leaf(j) => out.push(*j),
                Expr::Arith { left, right, .. } => {
                    raw(left, out);
                    raw(right, out);
                }
                Expr::Neg(x) => raw(x, out),
                Expr::Call { args, .. } => args.iter().for_each(|a| raw(a, out)),
                Expr::Case { whens, otherwise } => whens.iter().map(|(_, x)| x).chain(otherwise.as_deref()).for_each(|x| raw(x, out)),
                _ => {}
            }
        }
        let mut found = false;
        visit(&self.expr, &mut |e| {
            if let Expr::Arith { op: ArithOp::Add | ArithOp::Sub, left, right } = e {
                let (mut l, mut r) = (Vec::new(), Vec::new());
                raw(left, &mut l);
                raw(right, &mut r);
                let relevance = |j: &usize| self.leaves[*j].is_relevance();
                found |= !l.is_empty() && !r.is_empty() && l != r && (l.iter().any(relevance) || r.iter().any(relevance));
            }
        });
        found
    }

    /// The candidates of an expression that is not `exact`: the rows in the
    /// top rows of a generating leaf (`lists`, one per leaf), each once, in
    /// order.
    pub fn candidates<K: Copy + Ord>(&self, lists: &[Vec<K>]) -> Vec<K> {
        let mut cands: Vec<K> = lists.iter().zip(&self.generates).filter(|(_, g)| **g).flat_map(|(l, _)| l.iter().copied()).collect();
        cands.sort_unstable();
        cands.dedup();
        cands
    }

    /// The score of the best `keep` of `n` candidates and of every
    /// candidate tied on score with the last of them, best first:
    /// `values[j]` is leaf `j`'s value of each candidate (empty when the
    /// expression [reads](Self::reads_values) none), and `lists[j]` the
    /// candidates in leaf `j`'s top rows, best first. Rows tied on score
    /// come in the order they first appear in a ranking under RRF, in
    /// candidate order otherwise. A NULL score is −∞, −0 is 0 (so the two
    /// tie: which one an `f64::min` or a product gives is up to the
    /// compiler), and every other is finite.
    pub fn eval(&self, n: usize, values: &[Vec<f64>], lists: &[Vec<usize>], keep: usize) -> Vec<(usize, f64)> {
        let mut out: Vec<(usize, f64)> = self.values(&self.expr, n, values, lists).iter().map(|&x| if x.is_nan() { f64::NEG_INFINITY } else { x + 0.0 }).enumerate().collect();
        if let Expr::Rrf { leaves, .. } = &self.expr {
            let mut first = vec![usize::MAX; n];
            let mut seen = 0;
            for &c in leaves.iter().flat_map(|&j| &lists[j]) {
                if first[c] == usize::MAX {
                    first[c] = seen;
                    seen += 1;
                }
            }
            best_first(&mut out, keep, |a, b| b.1.total_cmp(&a.1).then(first[a.0].cmp(&first[b.0])));
        } else {
            // A sum over every row `WHERE` keeps can tie on most of them:
            // no lookup per comparison.
            best_first(&mut out, keep, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        }
        out
    }

    /// The value of `e` for each candidate, NaN for NULL; a leaf's are
    /// borrowed where it has a value for every row.
    fn values<'v>(&self, e: &Expr, n: usize, values: &'v [Vec<f64>], lists: &[Vec<usize>]) -> Cow<'v, [f64]> {
        let each = |x: Cow<[f64]>, f: &dyn Fn(f64) -> f64| -> Vec<f64> { x.iter().map(|&v| finite(f(v))).collect() };
        match e {
            Expr::Leaf(j) => match values[*j].iter().all(|x| x.is_finite()) {
                true => Cow::Borrowed(&values[*j]),
                false => Cow::Owned(values[*j].iter().map(|&x| finite(x)).collect()),
            },
            Expr::Sum { terms, norm } => {
                let weights: Vec<f64> = terms.iter().map(|(w, _)| *w).collect();
                let inputs: Vec<Cow<[f64]>> = terms.iter().map(|(_, e)| self.values(e, n, values, lists)).collect();
                Cow::Owned(fusion::combine(&weights, &inputs, *norm))
            }
            Expr::Product(fs) => {
                let inputs: Vec<Cow<[f64]>> = fs.iter().map(|e| self.values(e, n, values, lists)).collect();
                Cow::Owned(fusion::combine(&vec![1.0; fs.len()], &inputs, Norm::Product))
            }
            Expr::Rrf { leaves, k } => {
                let mut out = vec![0.0; n];
                for &j in leaves {
                    for (i, &c) in lists[j].iter().enumerate() {
                        out[c] += 1.0 / (k + (i + 1) as f64);
                    }
                }
                Cow::Owned(out)
            }
            Expr::Num(x) => Cow::Owned(vec![finite(*x); n]),
            Expr::Arith { op, left, right } => {
                let (a, b) = (self.values(left, n, values, lists), self.values(right, n, values, lists));
                Cow::Owned(a.iter().zip(b.iter()).map(|(&x, &y)| arith(*op, x, y)).collect())
            }
            Expr::Neg(x) => Cow::Owned(each(self.values(x, n, values, lists), &|v| -v)),
            Expr::Call { func, args } => {
                let args: Vec<Cow<[f64]>> = args.iter().map(|a| self.values(a, n, values, lists)).collect();
                let mut row = vec![0.0; args.len()];
                Cow::Owned(
                    (0..n)
                        .map(|i| {
                            for (r, a) in row.iter_mut().zip(&args) {
                                *r = a[i];
                            }
                            func.eval_f64(&row)
                        })
                        .collect(),
                )
            }
            Expr::Norm { kind: NormKind::Tmm, input, lo } => {
                let x = self.values(input, n, values, lists);
                let lo = *lo;
                let hi = x.iter().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, |hi, &v| hi.max(v));
                Cow::Owned(each(x, &|v| match v {
                    v if v.is_nan() => 0.0,
                    _ if hi > lo => (v - lo) / (hi - lo),
                    _ => 0.0,
                }))
            }
            Expr::Norm { kind: NormKind::ZScore, input, .. } => {
                let x = self.values(input, n, values, lists);
                let (count, sum) = x.iter().filter(|v| !v.is_nan()).fold((0usize, 0.0), |(c, s), &v| (c + 1, s + v));
                let mean = sum / count.max(1) as f64;
                let var = x.iter().filter(|v| !v.is_nan()).map(|&v| (v - mean) * (v - mean)).sum::<f64>() / count.max(1) as f64;
                let sd = var.sqrt();
                let z = |v: f64| if sd > 0.0 { (v - mean) / sd } else { 0.0 };
                // A missing value is the least of the candidates'.
                let least = x.iter().filter(|v| !v.is_nan()).map(|&v| z(v)).fold(f64::INFINITY, f64::min);
                let least = if least.is_finite() { least } else { 0.0 };
                Cow::Owned(each(x, &|v| if v.is_nan() { least } else { z(v) }))
            }
            Expr::Norm { kind: NormKind::MinMax, input, .. } => {
                let x = self.values(input, n, values, lists);
                let (lo, hi) = x.iter().filter(|v| !v.is_nan()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                Cow::Owned(each(x, &|v| match v {
                    v if v.is_nan() => 0.0,
                    _ if hi > lo => (v - lo) / (hi - lo),
                    _ => 1.0,
                }))
            }
            Expr::Norm { kind: NormKind::Rank, input, .. } => {
                let x = self.values(input, n, values, lists);
                // Best first, NULL last: a row's rank is 1 plus the rows
                // before its run of equal values.
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_unstable_by(|&a, &b| match (x[a].is_nan(), x[b].is_nan()) {
                    (false, false) => x[b].total_cmp(&x[a]),
                    (a, b) => a.cmp(&b),
                });
                let mut out = vec![0.0; n];
                let mut start = 0;
                for (i, &r) in order.iter().enumerate() {
                    let same = |p: f64, q: f64| (p.is_nan() && q.is_nan()) || p == q;
                    if i > 0 && !same(x[order[i - 1]], x[r]) {
                        start = i;
                    }
                    out[r] = (start + 1) as f64;
                }
                Cow::Owned(out)
            }
            Expr::Gbdt { model, args, .. } => {
                let args: Vec<Cow<[f64]>> = args.iter().map(|a| self.values(a, n, values, lists)).collect();
                let columns: Vec<&[f64]> = args.iter().map(|a| a.as_ref()).collect();
                Cow::Owned(match model {
                    Some(m) => m.predict_columns(&columns, n).into_iter().map(finite).collect(),
                    None => vec![f64::NAN; n],
                })
            }
            Expr::Case { whens, otherwise } => {
                let branches: Vec<(Cow<[f64]>, Cow<[f64]>)> = whens.iter().map(|(c, x)| (self.values(c, n, values, lists), self.values(x, n, values, lists))).collect();
                let otherwise = otherwise.as_ref().map(|x| self.values(x, n, values, lists));
                Cow::Owned(
                    (0..n)
                        .map(|i| match branches.iter().find(|(c, _)| !c[i].is_nan() && c[i] != 0.0) {
                            Some((_, x)) => x[i],
                            None => otherwise.as_ref().map_or(f64::NAN, |o| o[i]),
                        })
                        .collect(),
                )
            }
        }
    }

    /// Whether each row's value comes from its own leaves alone, with no
    /// set function, fusion or ranking over the candidates: what
    /// [`eval_row`](Self::eval_row) and [`upper`](Self::upper) take.
    pub fn is_row_local(&self) -> bool {
        let mut local = true;
        visit(&self.expr, &mut |e| local &= !matches!(e, Expr::Sum { .. } | Expr::Rrf { .. } | Expr::Norm { .. }));
        local
    }

    /// The leaf whose top-k loop can rank by the whole expression, so that
    /// its top k is exact without scoring every row (rank-safe boosts,
    /// `bm25(body, :q) * (1 + 0.2 * importance)`): the one text score (BM25
    /// or sparse) of a row-local expression whose other leaves are row
    /// values (columns, decays, conditions, `seq_no()`), not under
    /// `exact => 1`, and the only leaf whose top rows are candidates (a
    /// product of row values alone, `decay(..) * rating`, takes its own).
    /// The text score's bounds then bound the expression
    /// ([`upper`](Self::upper)), and a row it does not match is not a
    /// candidate, as with its top `depth`.
    pub fn driver(&self) -> Option<usize> {
        if self.exact || !self.is_row_local() {
            return None;
        }
        let mut text = None;
        for (j, l) in self.leaves.iter().enumerate() {
            match l {
                Score::Bm25 { .. } | Score::Sparse { .. } if text.is_none() => text = Some(j),
                Score::Column { .. } | Score::Decay { .. } | Score::Cond(_) | Score::SeqNo if !self.generates[j] => {}
                _ => return None,
            }
        }
        text
    }

    /// The value of a row-local expression ([`is_row_local`](Self::is_row_local))
    /// for a row whose leaves have `leaves` (NaN for NULL): the score
    /// [`eval`](Self::eval) gives it, bit for bit; NaN for NULL.
    pub fn eval_row(&self, leaves: &[f64]) -> f64 {
        row(&self.expr, leaves)
    }

    /// An upper bound of a row-local expression's value over the rows whose
    /// leaves lie in `leaves`: no such row's [`eval_row`](Self::eval_row) is
    /// greater (NULL ranks below every number). `f64::MIN` where every such
    /// row is NULL, and +∞ where nothing bounds it.
    pub fn upper(&self, leaves: &[Interval]) -> f64 {
        let i = interval(&self.expr, leaves);
        if i.is_empty() { f64::MIN } else { i.hi.max(f64::MIN) }
    }

    /// The expression as SQL would write it, normalisation explicit and
    /// leaves by field (`0.3 * minmax(bm25(title)) + 0.7 *
    /// minmax(similarity(vec))`), for plan lines.
    pub fn explain(&self, schema: &Schema) -> String {
        self.show(&self.expr, schema)
    }

    fn show(&self, e: &Expr, schema: &Schema) -> String {
        // An operand in parentheses where it binds less tightly than its
        // operator: a sum within a product, `a - (b + c)`, `a / (b * c)`.
        let tight = |e: &Expr| match e {
            Expr::Sum { .. } => 0,
            Expr::Arith { op: ArithOp::Add | ArithOp::Sub, .. } => 1,
            Expr::Arith { .. } | Expr::Product(_) => 2,
            _ => 3,
        };
        let operand = |e: &Expr, at_least: u8| match tight(e) < at_least {
            true => format!("({})", self.show(e, schema)),
            false => self.show(e, schema),
        };
        match e {
            Expr::Leaf(j) => leaf_name(self.leaves[*j], schema),
            Expr::Sum { terms, norm } => terms
                .iter()
                .map(|(w, e)| match norm {
                    Norm::MinMax => format!("{w} * minmax({})", self.show(e, schema)),
                    _ => format!("{w} * {}", operand(e, 2)),
                })
                .collect::<Vec<_>>()
                .join(" + "),
            Expr::Product(fs) => fs.iter().map(|e| operand(e, 2)).collect::<Vec<_>>().join(" * "),
            Expr::Rrf { leaves, k } => format!("rrf({}, k => {k})", leaves.iter().map(|&j| leaf_name(self.leaves[j], schema)).collect::<Vec<_>>().join(", ")),
            Expr::Num(x) => format!("{x}"),
            Expr::Arith { op, left, right } => {
                let (sym, l, r) = match op {
                    ArithOp::Add => ("+", 1, 1),
                    ArithOp::Sub => ("-", 1, 2),
                    ArithOp::Mul => ("*", 2, 2),
                    ArithOp::Div => ("/", 2, 3),
                    ArithOp::Mod => ("%", 2, 3),
                };
                format!("{} {sym} {}", operand(left, l), operand(right, r))
            }
            Expr::Neg(x) => format!("-{}", operand(x, 3)),
            Expr::Call { func, args } => format!("{}({})", func.name(), args.iter().map(|a| self.show(a, schema)).collect::<Vec<_>>().join(", ")),
            Expr::Gbdt { name, args, .. } => format!("gbdt({})", std::iter::once(format!("'{name}'")).chain(args.iter().map(|a| self.show(a, schema))).collect::<Vec<_>>().join(", ")),
            Expr::Norm { kind: NormKind::Tmm, input, lo } => format!("tmm({}, lo => {lo})", self.show(input, schema)),
            Expr::Norm { kind, input, .. } => format!("{}({})", kind.name(), self.show(input, schema)),
            Expr::Case { whens, otherwise } => {
                let mut s = "CASE".to_string();
                for (c, x) in whens {
                    s += &format!(" WHEN {} THEN {}", self.show(c, schema), self.show(x, schema));
                }
                if let Some(x) = otherwise {
                    s += &format!(" ELSE {}", self.show(x, schema));
                }
                s + " END"
            }
        }
    }
}

/// NaN, NULL in the evaluator, for a value that is not a finite number.
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

/// `x op y`, NULL where it is not a finite number.
#[inline]
fn arith(op: ArithOp, x: f64, y: f64) -> f64 {
    finite(match op {
        ArithOp::Add => x + y,
        ArithOp::Sub => x - y,
        ArithOp::Mul => x * y,
        ArithOp::Div => x / y,
        ArithOp::Mod => x % y,
    })
}

/// One row's value of `e`, as [`Rank::values`] computes it for every row.
fn row(e: &Expr, v: &[f64]) -> f64 {
    match e {
        Expr::Leaf(j) => finite(v[*j]),
        Expr::Num(x) => finite(*x),
        Expr::Arith { op, left, right } => arith(*op, row(left, v), row(right, v)),
        Expr::Neg(x) => finite(-row(x, v)),
        Expr::Call { func, args } => {
            let mut buf = [0.0; 8];
            match args.len() {
                n if n <= buf.len() => {
                    for (b, a) in buf.iter_mut().zip(args) {
                        *b = row(a, v);
                    }
                    func.eval_f64(&buf[..n])
                }
                _ => func.eval_f64(&args.iter().map(|a| row(a, v)).collect::<Vec<_>>()),
            }
        }
        // As fusion::combine multiplies: a factor below 0, or missing,
        // counts as 0, and an overflow ranks last.
        Expr::Product(fs) => {
            let p = fs.iter().fold(1.0, |p, f| {
                let x = row(f, v);
                p * if x.is_finite() { x.max(0.0) } else { 0.0 }
            });
            if p.is_finite() { p } else { f64::MIN }
        }
        Expr::Gbdt { model: None, .. } => f64::NAN,
        Expr::Gbdt { model: Some(m), args, .. } => {
            let mut buf = [0.0; 16];
            match args.len() {
                n if n <= buf.len() => {
                    for (b, a) in buf.iter_mut().zip(args) {
                        *b = row(a, v);
                    }
                    finite(m.predict(&buf[..n]))
                }
                _ => finite(m.predict(&args.iter().map(|a| row(a, v)).collect::<Vec<_>>())),
            }
        }
        Expr::Case { whens, otherwise } => match whens.iter().find(|(c, _)| {
            let c = row(c, v);
            !c.is_nan() && c != 0.0
        }) {
            Some((_, x)) => row(x, v),
            None => otherwise.as_ref().map_or(f64::NAN, |x| row(x, v)),
        },
        Expr::Sum { .. } | Expr::Rrf { .. } | Expr::Norm { .. } => f64::NAN,
    }
}

/// The values an expression takes over a set of rows: each finite value
/// lies in `[lo, hi]` (none does when `lo > hi`), and NULL is possible when
/// `null`. A bound may be infinite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    pub null: bool,
}

impl Interval {
    /// Any value, or NULL.
    pub const ANY: Interval = Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY, null: true };
    /// NULL only.
    const NULL: Interval = Interval { lo: f64::INFINITY, hi: f64::NEG_INFINITY, null: true };

    pub fn new(lo: f64, hi: f64, null: bool) -> Interval {
        Interval { lo, hi, null }
    }

    fn point(x: f64) -> Interval {
        if x.is_finite() { Interval::new(x, x, false) } else { Interval::NULL }
    }

    /// No finite value.
    pub fn is_empty(&self) -> bool {
        !(self.lo <= self.hi)
    }

    /// Both sets of values.
    fn hull(self, o: Interval) -> Interval {
        Interval { lo: self.lo.min(o.lo), hi: self.hi.max(o.hi), null: self.null || o.null }
    }

    /// The values of `f`, non-decreasing, over this interval: `f` of its
    /// ends, widened by a few ulps for a libm function that is not exactly
    /// monotone; unbounded on a side where `f` of the end is NULL.
    fn rising(self, f: impl Fn(f64) -> f64) -> Interval {
        if self.is_empty() {
            return Interval::NULL;
        }
        let (lo, hi) = (f(self.lo), f(self.hi));
        let wide = |x: f64| x.abs() * 1e-12 + f64::MIN_POSITIVE;
        Interval {
            lo: if lo.is_nan() { f64::NEG_INFINITY } else { lo - wide(lo) },
            hi: if hi.is_nan() { f64::INFINITY } else { hi + wide(hi) },
            null: self.null || lo.is_nan() || hi.is_nan(),
        }
    }

    /// The values of `f`, non-increasing, over this interval.
    fn falling(self, f: impl Fn(f64) -> f64) -> Interval {
        let r = self.rising(|x| -f(x));
        if r.is_empty() { r } else { Interval::new(-r.hi, -r.lo, r.null) }
    }
}

/// The values of `e` over rows whose leaves lie in `leaves`: an interval
/// that holds every row's [`row`] value, wider where a tight one would cost
/// more than it gives.
fn interval(e: &Expr, leaves: &[Interval]) -> Interval {
    let any = |null: bool| Interval { null, ..Interval::ANY };
    match e {
        Expr::Leaf(j) => leaves[*j],
        Expr::Num(x) => Interval::point(*x),
        Expr::Neg(x) => {
            let x = interval(x, leaves);
            if x.is_empty() { x } else { Interval::new(-x.hi, -x.lo, x.null) }
        }
        Expr::Arith { op, left, right } => {
            let (a, b) = (interval(left, leaves), interval(right, leaves));
            if a.is_empty() || b.is_empty() {
                return Interval::NULL;
            }
            let null = a.null || b.null;
            // A result off the ends' results is NULL (an overflow).
            let corners = |f: fn(f64, f64) -> f64| {
                let c = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
                match c.iter().any(|x| x.is_nan()) {
                    true => any(true),
                    false => Interval::new(c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max), null || c.iter().any(|x| x.is_infinite())),
                }
            };
            match op {
                ArithOp::Add => {
                    let (lo, hi) = (a.lo + b.lo, a.hi + b.hi);
                    if lo.is_nan() || hi.is_nan() { any(true) } else { Interval::new(lo, hi, null || lo.is_infinite() || hi.is_infinite()) }
                }
                ArithOp::Sub => {
                    let (lo, hi) = (a.lo - b.hi, a.hi - b.lo);
                    if lo.is_nan() || hi.is_nan() { any(true) } else { Interval::new(lo, hi, null || lo.is_infinite() || hi.is_infinite()) }
                }
                ArithOp::Mul => corners(|x, y| x * y),
                // A divisor near 0 makes any quotient.
                ArithOp::Div if b.lo <= 0.0 && b.hi >= 0.0 => any(true),
                ArithOp::Div => corners(|x, y| x / y),
                // |x % y| < |y| and |x % y| <= |x|, with the sign of x.
                ArithOp::Mod => {
                    let m = b.lo.abs().max(b.hi.abs());
                    Interval::new(if a.lo < 0.0 { a.lo.max(-m) } else { 0.0 }, if a.hi > 0.0 { a.hi.min(m) } else { 0.0 }, true)
                }
            }
        }
        Expr::Call { func, args } => {
            let a: Vec<Interval> = args.iter().map(|x| interval(x, leaves)).collect();
            let one = |f: Func| move |x: f64| f.eval_f64(&[x]);
            match func {
                Func::Abs => {
                    let x = a[0];
                    match () {
                        _ if x.is_empty() => x,
                        _ if x.lo >= 0.0 => x,
                        _ if x.hi <= 0.0 => Interval::new(-x.hi, -x.lo, x.null),
                        _ => Interval::new(0.0, x.hi.max(-x.lo), x.null),
                    }
                }
                Func::Ln | Func::Exp | Func::Sqrt | Func::Sigmoid => a[0].rising(one(*func)),
                Func::Log if a.len() == 1 => a[0].rising(one(Func::Log)),
                Func::Round if a.len() == 1 => a[0].rising(one(Func::Round)),
                Func::Round => match a[1] {
                    d if !d.is_empty() && d.lo == d.hi => {
                        let r = a[0].rising(|x| Func::Round.eval_f64(&[x, d.lo]));
                        Interval { null: r.null || d.null, ..r }
                    }
                    _ => any(true),
                },
                // x^c for a constant c and x >= 0, and c^y for a constant
                // c > 0, are monotone.
                Func::Pow => match (a[0], a[1]) {
                    (x, c) if !x.is_empty() && !c.is_empty() && c.lo == c.hi && x.lo >= 0.0 => {
                        let f = |x: f64| Func::Pow.eval_f64(&[x, c.lo]);
                        let r = if c.lo >= 0.0 { x.rising(f) } else { x.falling(f) };
                        Interval { null: r.null || c.null, ..r }
                    }
                    (c, y) if !c.is_empty() && !y.is_empty() && c.lo == c.hi && c.lo > 0.0 => {
                        let f = |y: f64| Func::Pow.eval_f64(&[c.lo, y]);
                        let r = if c.lo >= 1.0 { y.rising(f) } else { y.falling(f) };
                        Interval { null: r.null || c.null, ..r }
                    }
                    _ => any(true),
                },
                // Non-decreasing in each argument.
                Func::Clamp => {
                    if a.iter().any(Interval::is_empty) {
                        return Interval::NULL;
                    }
                    let lo = Func::Clamp.eval_f64(&[a[0].lo, a[1].lo, a[2].lo]);
                    let hi = Func::Clamp.eval_f64(&[a[0].hi, a[1].hi, a[2].hi]);
                    if lo.is_nan() || hi.is_nan() { any(true) } else { Interval::new(lo, hi, a.iter().any(|x| x.null)) }
                }
                // NULL arguments are skipped: the result is one of the
                // others, or NULL when every one is.
                Func::Least | Func::Greatest | Func::Coalesce => {
                    let present: Vec<&Interval> = a.iter().filter(|x| !x.is_empty()).collect();
                    let null = a.iter().all(|x| x.null);
                    if present.is_empty() {
                        return Interval::NULL;
                    }
                    let (lo, hi) = (present.iter().map(|x| x.lo).fold(f64::INFINITY, f64::min), present.iter().map(|x| x.hi).fold(f64::NEG_INFINITY, f64::max));
                    // An argument that is never NULL is always among them.
                    let sure = a.iter().filter(|x| !x.null && !x.is_empty());
                    match func {
                        Func::Least => Interval::new(lo, sure.map(|x| x.hi).fold(hi, f64::min), null),
                        Func::Greatest => Interval::new(sure.map(|x| x.lo).fold(lo, f64::max), hi, null),
                        _ => Interval::new(lo, hi, null),
                    }
                }
                // Conditions, 0 or 1 each.
                Func::CountTrue => Interval::new(0.0, a.len() as f64, false),
                _ => any(true),
            }
        }
        // Each factor counts as max(x, 0), 0 when missing; an overflow is
        // f64::MIN.
        Expr::Product(fs) => {
            let (mut lo, mut hi) = (1.0f64, 1.0f64);
            for f in fs {
                let x = interval(f, leaves);
                let (flo, fhi) = match x.is_empty() {
                    true => (0.0, 0.0),
                    false => (if x.null { 0.0 } else { x.lo.max(0.0) }, x.hi.max(0.0)),
                };
                lo *= flo;
                hi *= fhi;
            }
            match hi.is_finite() && lo.is_finite() {
                true => Interval::new(lo, hi, false),
                false => Interval::new(f64::MIN, f64::INFINITY, false),
            }
        }
        // The leaves the features' values reach, tree by tree.
        Expr::Gbdt { model: None, .. } => Interval::ANY,
        Expr::Gbdt { model: Some(m), args, .. } => {
            let a: Vec<(f64, f64, bool)> = args.iter().map(|x| interval(x, leaves)).map(|i| (i.lo, i.hi, i.null)).collect();
            let (lo, hi) = m.bounds(&a);
            if lo > hi { Interval::NULL } else { Interval::new(lo, hi, false) }
        }
        Expr::Case { whens, otherwise } => {
            let mut out = otherwise.as_ref().map_or(Interval::NULL, |x| interval(x, leaves));
            for (_, x) in whens {
                out = out.hull(interval(x, leaves));
            }
            out
        }
        Expr::Sum { .. } | Expr::Rrf { .. } | Expr::Norm { .. } => Interval::ANY,
    }
}

/// Calls `f` on `e` and each expression within it.
fn visit(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Sum { terms, .. } => terms.iter().for_each(|(_, x)| visit(x, f)),
        Expr::Product(xs) | Expr::Call { args: xs, .. } | Expr::Gbdt { args: xs, .. } => xs.iter().for_each(|x| visit(x, f)),
        Expr::Arith { left, right, .. } => {
            visit(left, f);
            visit(right, f);
        }
        Expr::Neg(x) | Expr::Norm { input: x, .. } => visit(x, f),
        Expr::Case { whens, otherwise } => {
            for (c, x) in whens {
                visit(c, f);
                visit(x, f);
            }
            if let Some(x) = otherwise {
                visit(x, f);
            }
        }
        Expr::Leaf(_) | Expr::Rrf { .. } | Expr::Num(_) => {}
    }
}

/// A leaf as plan lines name it: its function and field.
fn leaf_name(s: &Score, schema: &Schema) -> String {
    let name = |f: &usize| schema.fields.get(*f).map_or("?", |f| f.name.as_str());
    match s {
        Score::Bm25 { field, .. } => format!("bm25({})", name(field)),
        Score::Similarity { field, .. } | Score::Sparse { field, .. } => format!("similarity({})", name(field)),
        Score::MaxSim { field, .. } => format!("maxsim({})", name(field)),
        Score::Column { field } => name(field).to_string(),
        Score::Decay { field, .. } => format!("decay({})", name(field)),
        Score::Ppr { .. } => "ppr()".into(),
        Score::Hops { .. } => "hops()".into(),
        Score::SeqNo => "seq_no()".into(),
        Score::CrossEncoder { model, fields, .. } => format!("cross_encoder('{model}', {})", fields.iter().map(name).collect::<Vec<_>>().join(", ")),
        Score::Ranked { keys } => format!("ranked({} keys)", keys.len()),
        Score::Cond(p) => {
            fn fields(p: &Pred, out: &mut Vec<usize>) {
                match p {
                    Pred::Match { field, .. } | Pred::Range { field, .. } | Pred::IntRange { field, .. } | Pred::In { field, .. } | Pred::Prefix { field, .. } | Pred::Present { field } => out.push(*field),
                    Pred::And(ps) | Pred::Or(ps) | Pred::AtLeast { items: ps, .. } => ps.iter().for_each(|p| fields(p, out)),
                    Pred::Not(p) => fields(p, out),
                    Pred::True | Pred::False | Pred::Graph { .. } | Pred::Sub(_) => {}
                }
            }
            let mut fs = Vec::new();
            fields(p, &mut fs);
            fs.sort_unstable();
            fs.dedup();
            format!("<a condition on {}>", fs.iter().map(name).collect::<Vec<_>>().join(", "))
        }
        _ => "(an expression)".into(),
    }
}

/// Orders `rows` by `cmp`, a total order that ranks by score first, and
/// keeps its best `keep` rows and every row tied on score with the last of
/// them: the prefix of the whole order that a page of `keep` rows can reach,
/// without ordering the rest.
fn best_first(rows: &mut Vec<(usize, f64)>, keep: usize, cmp: impl Fn(&(usize, f64), &(usize, f64)) -> Ordering) {
    if keep == 0 {
        rows.clear();
        return;
    }
    if keep < rows.len() {
        rows.select_nth_unstable_by(keep - 1, &cmp);
        let cut = rows[keep - 1].1;
        let mut end = keep;
        for i in keep..rows.len() {
            if rows[i].1.total_cmp(&cut).is_eq() {
                rows.swap(i, end);
                end += 1;
            }
        }
        rows.truncate(end);
    }
    rows.sort_unstable_by(&cmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::logical::Opts;
    use crate::schema::{Field, FieldType};

    fn bm25(field: usize) -> Score {
        Score::Bm25 { field, text: "x".into(), opts: Opts::default() }
    }

    #[test]
    fn today_s_shapes_compile_to_their_explicit_forms() {
        let schema = Schema::new(
            "t",
            vec![Field::new("id", FieldType::Keyword), Field::new("body", FieldType::Text { positions: false }), Field::new("rating", FieldType::Float)],
            Some("id"),
        )
        .unwrap();
        let w = Score::Weighted(vec![(0.3, bm25(1)), (0.7, Score::Column { field: 2 })]);
        let r = Rank::of(&w, 100).unwrap();
        assert_eq!(r.explain(&schema), "0.3 * minmax(bm25(body)) + 0.7 * minmax(rating)");
        assert_eq!((r.depth, r.exact, r.generates.clone()), (100, false, vec![true, true]));
        assert!(r.reads_values());

        let p = Score::Product(vec![bm25(1), Score::Column { field: 2 }]);
        let r = Rank::of(&p, 100).unwrap();
        assert_eq!(r.explain(&schema), "bm25(body) * rating");
        // A product's candidates come from its relevance factors.
        assert_eq!(r.generates, vec![true, false]);

        let f = Score::Rrf { inputs: vec![bm25(1), Score::Column { field: 2 }], k: 60.0, depth: 30 };
        let r = Rank::of(&f, 100).unwrap();
        assert_eq!(r.explain(&schema), "rrf(bm25(body), rating, k => 60)");
        assert_eq!(r.depth, 30);
        assert!(!r.reads_values());
        assert!(Rank::of(&bm25(1), 100).is_none());
    }

    #[test]
    fn eval_matches_the_fusion_functions() {
        // Weighted sums and products: fusion::weighted's scores and order.
        let s0 = vec![1.0f32, 3.0, 2.0, 3.0];
        let s1 = vec![10.0f32, f32::NEG_INFINITY, 20.0, 5.0];
        let values: Vec<Vec<f64>> = [&s0, &s1].iter().map(|s| s.iter().map(|&x| f64::from(x)).collect()).collect();
        for (score, norm) in [
            (Score::Weighted(vec![(0.7, bm25(0)), (0.3, bm25(1))]), Norm::MinMax),
            (Score::Weighted(vec![(1.0, bm25(0)), (-0.1, bm25(1))]), Norm::MinMax),
            (Score::Product(vec![bm25(0), bm25(1)]), Norm::Product),
        ] {
            let r = Rank::of(&score, 100).unwrap();
            let weights: Vec<f64> = match &score {
                Score::Weighted(t) => t.iter().map(|(w, _)| *w).collect(),
                _ => vec![1.0, 1.0],
            };
            assert_eq!(r.eval(4, &values, &[vec![], vec![]], 4), fusion::weighted(&weights, &[s0.clone(), s1.clone()], norm), "{score:?}");
        }
        // RRF: fusion::rrf's scores and order, ties by first appearance.
        let lists = vec![vec![2, 0, 1], vec![1, 3, 2]];
        let f = Score::Rrf { inputs: vec![bm25(0), bm25(1)], k: 60.0, depth: 3 };
        assert_eq!(Rank::of(&f, 100).unwrap().eval(4, &[], &lists, 4), fusion::rrf(&lists, 60.0));
    }

    /// Row `i`'s value of `s` by the rules in the module docs, read straight
    /// off the expression: `None` is NULL.
    fn reference(s: &Score, cols: &[Vec<f64>], i: usize) -> Option<f64> {
        let fin = |x: f64| Some(x).filter(|x| x.is_finite());
        let all = |e: &Score| (0..cols[0].len()).map(|r| reference(e, cols, r)).collect::<Vec<_>>();
        match s {
            Score::Column { field } => fin(cols[*field][i]),
            Score::Num(x) => fin(*x),
            Score::Neg(x) => reference(x, cols, i).map(|v| -v),
            Score::Arith { op, left, right } => {
                let (a, b) = (reference(left, cols, i)?, reference(right, cols, i)?);
                fin(match op {
                    ArithOp::Add => a + b,
                    ArithOp::Sub => a - b,
                    ArithOp::Mul => a * b,
                    ArithOp::Div => a / b,
                    ArithOp::Mod => a % b,
                })
            }
            Score::Call { func, args } => {
                let v: Vec<Option<f64>> = args.iter().map(|a| reference(a, cols, i)).collect();
                let some: Vec<f64> = v.iter().flatten().copied().collect();
                match func {
                    Func::Coalesce => some.first().copied(),
                    Func::Least => some.into_iter().reduce(f64::min),
                    Func::Greatest => some.into_iter().reduce(f64::max),
                    Func::Abs => v[0].map(f64::abs),
                    Func::Ln => v[0].filter(|x| *x > 0.0).map(f64::ln),
                    Func::Log if v.len() == 1 => v[0].filter(|x| *x > 0.0).map(f64::log10),
                    Func::Log => match (v[0], v[1]) {
                        (Some(b), Some(x)) if b > 0.0 && b != 1.0 && x > 0.0 => fin(x.ln() / b.ln()),
                        _ => None,
                    },
                    Func::Exp => fin(v[0]?.exp()),
                    Func::Sqrt => v[0].filter(|x| *x >= 0.0).map(f64::sqrt),
                    Func::Pow => fin(v[0]?.powf(v[1]?)),
                    Func::Sigmoid => fin(1.0 / (1.0 + (-v[0]?).exp())),
                    Func::Clamp => fin(v[0]?.max(v[1]?).min(v[2]?)),
                    Func::Round => {
                        let m = 10f64.powi(v.get(1).map_or(Some(0.0), |d| *d)?.clamp(-30.0, 30.0).trunc() as i32);
                        fin((v[0]? * m).round() / m)
                    }
                    Func::CountTrue => Some(some.iter().filter(|&&x| x != 0.0).count() as f64),
                    Func::Lower | Func::Upper | Func::Length | Func::Dot | Func::SidAt => None,
                }
            }
            Score::Norm { kind: NormKind::Tmm, input, lo } => {
                let xs = all(input);
                let lo = lo.unwrap();
                let hi = xs.iter().flatten().fold(f64::NEG_INFINITY, |hi, &v| hi.max(v));
                Some(match xs[i] {
                    Some(v) if hi > lo => (v - lo) / (hi - lo),
                    _ => 0.0,
                })
            }
            Score::Norm { kind: NormKind::ZScore, input, .. } => {
                let xs: Vec<f64> = all(input).into_iter().flatten().collect();
                let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
                let sd = (xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xs.len().max(1) as f64).sqrt();
                let z = |v: f64| if sd > 0.0 { (v - mean) / sd } else { 0.0 };
                Some(match all(input)[i] {
                    Some(v) => z(v),
                    None => xs.iter().map(|&v| z(v)).reduce(f64::min).unwrap_or(0.0),
                })
            }
            Score::Norm { kind: NormKind::MinMax, input, .. } => {
                let xs = all(input);
                let (lo, hi) = xs.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                Some(match xs[i] {
                    None => 0.0,
                    Some(v) if hi > lo => (v - lo) / (hi - lo),
                    Some(_) => 1.0,
                })
            }
            Score::Norm { kind: NormKind::Rank, input, .. } => {
                let xs = all(input);
                Some(1.0 + xs.iter().filter(|x| match (x, xs[i]) {
                    (Some(x), Some(v)) => *x > v,
                    (Some(_), None) => true,
                    (None, _) => false,
                }).count() as f64)
            }
            Score::Case { whens, otherwise } => match whens.iter().find(|(c, _)| reference(c, cols, i).is_some_and(|c| c != 0.0)) {
                Some((_, x)) => reference(x, cols, i),
                None => otherwise.as_ref().and_then(|x| reference(x, cols, i)),
            },
            s => panic!("{s:?}"),
        }
    }

    /// A random ranking expression over three columns, `depth` deep.
    fn random_score(rng: &mut u64, depth: u32) -> Score {
        let mut next = |n: u64| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng % n
        };
        if depth == 0 || next(5) == 0 {
            return match next(4) {
                0 => Score::Num([0.0, 1.0, -2.5, 0.5, 3.0][next(5) as usize]),
                _ => Score::Column { field: next(3) as usize },
            };
        }
        let (a, b, c) = (next(1 << 62), next(1 << 62), next(1 << 62));
        let sub = |seed: u64| {
            let mut r = seed | 1;
            random_score(&mut r, depth - 1)
        };
        let bx = |s: Score| Box::new(s);
        match next(9) {
            0..=2 => Score::Arith {
                op: [ArithOp::Add, ArithOp::Sub, ArithOp::Mul, ArithOp::Div, ArithOp::Mod][next(5) as usize],
                left: bx(sub(a)),
                right: bx(sub(b)),
            },
            3 => Score::Neg(bx(sub(a))),
            4 => {
                let func = [Func::Abs, Func::Ln, Func::Log, Func::Exp, Func::Sqrt, Func::Pow, Func::Sigmoid, Func::Clamp, Func::Round, Func::Coalesce, Func::Least, Func::Greatest][next(12) as usize];
                let n = match func {
                    Func::Pow => 2,
                    Func::Clamp => 3,
                    Func::Log | Func::Round => 1 + next(2) as usize,
                    Func::Coalesce | Func::Least | Func::Greatest => 1 + next(3) as usize,
                    _ => 1,
                };
                Score::Call { func, args: [a, b, c].iter().take(n).map(|&s| sub(s)).collect() }
            }
            5 => match [NormKind::MinMax, NormKind::Rank, NormKind::Tmm, NormKind::ZScore][next(4) as usize] {
                NormKind::Tmm => Score::Norm { kind: NormKind::Tmm, input: bx(sub(a)), lo: Some([0.0, -1.0, 2.5][next(3) as usize]) },
                kind => Score::Norm { kind, input: bx(sub(a)), lo: None },
            },
            6 => Score::Case { whens: vec![(sub(a), sub(b))], otherwise: (next(2) == 0).then(|| bx(sub(c))) },
            _ => Score::Arith { op: ArithOp::Add, left: bx(sub(a)), right: bx(sub(b)) },
        }
    }

    /// The evaluator gives every row the reference's value and ranks rows by
    /// it, NULL last, over random expressions of columns with NULLs.
    #[test]
    fn eval_matches_a_reference_with_nulls() {
        let n = 12;
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut col = |nulls: u64| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    match rng % 10 {
                        x if x < nulls => f64::NAN,
                        // Ties, zeros and negatives.
                        x => [0.0, 1.0, 2.0, -1.5, 0.25, 7.0, 2.0, 1e300, -0.0, 3.5][x as usize] * if rng % 3 == 0 { -1.0 } else { 1.0 },
                    }
                })
                .collect()
        };
        let cols = vec![col(0), col(2), col(5)];
        let mut seen_null = 0;
        for seed in 1..3000u64 {
            let mut r = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
            let s = random_score(&mut r, 4);
            let Some(rank) = Rank::of(&s, 100) else { continue };
            let values: Vec<Vec<f64>> = rank.leaves.iter().map(|l| match l {
                Score::Column { field } => cols[*field].clone(),
                l => panic!("{l:?}"),
            }).collect();
            let lists = vec![Vec::new(); values.len()];
            let mut want: Vec<(usize, f64)> = (0..n).map(|i| (i, reference(&s, &cols, i).map_or(f64::NEG_INFINITY, |v| v + 0.0))).collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            seen_null += want.iter().filter(|w| w.1 == f64::NEG_INFINITY).count();
            let got = rank.eval(n, &values, &lists, n);
            assert_eq!(got.len(), n);
            for (g, w) in got.iter().zip(&want) {
                assert!(g.0 == w.0 && g.1.total_cmp(&w.1).is_eq(), "{s:?}\n got {got:?}\nwant {want:?}");
            }
        }
        assert!(seen_null > 1000, "{seen_null}");
    }

    /// A random row-local expression: [`random_score`] without set
    /// functions, sometimes times a product of columns.
    fn random_local(rng: &mut u64, depth: u32) -> Score {
        loop {
            let s = random_score(rng, depth);
            let mut r = (*rng).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            *rng = r;
            r ^= r >> 29;
            let s = match r % 5 {
                0 => Score::Arith {
                    op: ArithOp::Mul,
                    left: Box::new(s),
                    right: Box::new(Score::Product(vec![Score::Column { field: (r % 3) as usize }, Score::Column { field: (r / 3 % 3) as usize }])),
                },
                _ => s,
            };
            if Rank::of(&s, 100).is_some_and(|k| k.is_row_local()) {
                return s;
            }
        }
    }

    /// Rows of three columns with NULLs, ties, zeros, negatives and 1e300.
    fn columns(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = seed | 1;
        let mut col = |nulls: u64| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    match rng % 10 {
                        x if x < nulls => f64::NAN,
                        x => [0.0, 1.0, 2.0, -1.5, 0.25, 7.0, 2.0, 1e300, -0.0, 3.5][x as usize] * if rng % 3 == 0 { -1.0 } else { 1.0 },
                    }
                })
                .collect()
        };
        vec![col(0), col(2), col(5)]
    }

    /// The leaves' values of each row, in a Rank's leaf order.
    fn leaf_values(rank: &Rank, cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
        rank.leaves
            .iter()
            .map(|l| match l {
                Score::Column { field } => cols[*field].clone(),
                l => panic!("{l:?}"),
            })
            .collect()
    }

    /// eval_row gives each row the value the batch evaluator gives it, bit
    /// for bit, so a top-k loop that scores rows one at a time keeps the
    /// rows the evaluator ranks first.
    #[test]
    fn eval_row_matches_the_batch_evaluator() {
        let n = 16;
        let cols = columns(n, 0x2545_f491_4f6c_dd1d);
        let mut rng = 0x1234_5678_9abc_def1u64;
        for _ in 0..3000 {
            let s = random_local(&mut rng, 4);
            let rank = Rank::of(&s, 100).unwrap();
            let values = leaf_values(&rank, &cols);
            let batch = rank.values(&rank.expr, n, &values, &vec![Vec::new(); values.len()]);
            for i in 0..n {
                let leaves: Vec<f64> = values.iter().map(|v| v[i]).collect();
                let one = rank.eval_row(&leaves);
                assert!(one.to_bits() == batch[i].to_bits() || (one.is_nan() && batch[i].is_nan()), "{s:?} row {i}: {one} vs {}", batch[i]);
            }
        }
    }

    /// No row whose leaves lie in a box scores above the box's upper bound,
    /// over random expressions, boxes and rows in them.
    #[test]
    fn upper_bounds_every_row_in_the_box() {
        let mut rng = 0x0bad_5eed_1234_5678u64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let pick = |r: u64| -> f64 { [0.0, 1.0, 2.0, -1.5, 0.25, 7.0, -3.0, 1e300, -0.0, 3.5, 1e-300, 0.5][(r % 12) as usize] };
        let mut tight = 0;
        for _ in 0..4000 {
            let mut r = next(u64::MAX) | 1;
            let s = random_local(&mut r, 4);
            let rank = Rank::of(&s, 100).unwrap();
            // A box per leaf, and rows inside it: its ends, points
            // between them, and NULL where the box allows it.
            let boxes: Vec<Interval> = rank
                .leaves
                .iter()
                .map(|_| {
                    let (a, b) = (pick(next(1 << 20)), pick(next(1 << 20)));
                    Interval::new(a.min(b), a.max(b), next(3) == 0)
                })
                .collect();
            let upper = rank.upper(&boxes);
            for _ in 0..24 {
                let leaves: Vec<f64> = boxes
                    .iter()
                    .map(|b| match next(6) {
                        0 if b.null => f64::NAN,
                        1 => b.lo,
                        2 => b.hi,
                        _ => b.lo + (b.hi - b.lo) * (next(1000) as f64 / 1000.0),
                    })
                    .map(|x| if x.is_finite() { x } else { f64::NAN })
                    .collect();
                let v = rank.eval_row(&leaves);
                assert!(v.is_nan() || v <= upper, "{s:?}\nboxes {boxes:?}\nrow {leaves:?}: {v} > {upper}");
                tight += (v == upper) as usize;
            }
        }
        // The bounds are reached, not only safe.
        assert!(tight > 1000, "{tight}");
    }

    #[test]
    fn minus_zero_ties_with_zero() {
        // -x gives 0 for row 1 and -0 for rows 0 and 2: the three tie, so
        // they come in row order, each with the score 0.
        let s = Score::Neg(Box::new(Score::Column { field: 0 }));
        let r = Rank::of(&s, 100).unwrap();
        let got = r.eval(4, &[vec![0.0, -0.0, 0.0, 1.0]], &[vec![]], 4);
        assert_eq!(got.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert!(got[..3].iter().all(|x| x.1.to_bits() == 0), "{got:?}");
    }

    #[test]
    fn eval_keeps_the_page_and_the_tie_at_its_cut() {
        // Scores 5, 4, 4, 4, 3, 2 in a scrambled candidate order.
        let v = vec![4.0, 3.0, 4.0, 5.0, 2.0, 4.0];
        let w = Score::Weighted(vec![(1.0, bm25(0)), (0.0, bm25(1))]);
        let r = Rank::of(&w, 100).unwrap();
        let values = vec![v.clone(), vec![0.0; 6]];
        let all = r.eval(6, &values, &[vec![], vec![]], 6);
        assert_eq!(all.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 0, 2, 5, 1, 4]);
        for (keep, want) in [(0, 0), (1, 1), (2, 4), (3, 4), (4, 4), (5, 5), (6, 6), (9, 6)] {
            assert_eq!(r.eval(6, &values, &[vec![], vec![]], keep), all[..want], "keep {keep}");
        }
    }
}
