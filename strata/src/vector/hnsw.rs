//! HNSW: hierarchical navigable small-world graph (Malkov & Yashunin).
//!
//! **Construction** inserts documents in docID order (in a seeded random
//! order with several threads) with the neighbour-selection heuristic, `M`
//! links per node on the upper layers and `2M` on layer 0. Several threads
//! insert at once; each node's link list has its own lock, as in hnswlib.
//! Levels come from a seeded hash of the docID, so the layer structure does
//! not depend on the thread count (the links do, slightly, through insertion
//! races). Similarities are those a search uses: the inserted vector's
//! float32 values against the others' SQ8 codes ([`BuildCodes`]), a quarter
//! of the memory traffic of float32 rows; a field with binary codes builds on
//! float32.
//!
//! **Merges** keep the graph of their largest input that lost no documents
//! (Lucene 9.9) and join the other whole inputs' graphs to it as Lucene 10.2
//! does: a greedy cover of each one's nodes (its *join set*, about a quarter
//! of each node's neighbours) is inserted as usual, and every other node
//! starts its layer-0 search from its old neighbours in the join set and
//! their new neighbours, with a narrower beam.
//!
//! **Layout** (sections `vec.<field>.hnsw` and `vec.<field>.hnsw_upper`):
//! layer 0 is `n × 2M` u32 neighbour ids, padded with `u32::MAX`, so a node's
//! list is at a fixed offset (128 bytes for M = 16, two cache lines). Each
//! upper layer `l` is a sorted array of the nodes at level ≥ l plus their
//! `n_l × M` links; a node's slot is found by binary search (upper layers
//! hold ~1/M of the nodes each and a search touches a handful of them).
//!
//! **Search** runs on SQ8 codes (1 KiB per 1024-d vector instead of 4 KiB) and
//! the caller re-ranks the best candidates with float32. A filter is applied
//! *in* the traversal, in one of two ways:
//!
//! * traverse (default): every reached node is scored and expanded as a
//!   stepping stone, only members enter the result set, and the search ends
//!   when the closest unexpanded candidate is worse than the `ef`-th best
//!   member. The graph stays connected for any filter; the work grows like
//!   `1/selectivity`.
//! * two-hop (ACORN-1): only members are scored and expanded; a node's
//!   non-member neighbours are looked through to their own neighbours. The
//!   work no longer grows with `1/selectivity`, but on a graph built with the
//!   ordinary `M` the members' two-hop graph falls apart below ~10%
//!   selectivity, or when the members lie away from the query, and recall
//!   collapses (ACORN builds a denser graph for this; see `notes/vector.md`).

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::postings::bytes::{ByteReader, ByteWriter, OwnedBytes, Slice};

use crate::error::{corrupt, Result};
use crate::segment::DocId;
use crate::vector::build::Rows;
use crate::vector::kernels::{self, dot_f32};
use crate::vector::codes::QueryCode;
use crate::vector::sq8::{Sq8, Sq8Query};
use crate::vector::topk::{Members, TopK};
use crate::vector::{io, Codes, SearchStats};

const NONE: u32 = u32::MAX;
const MAX_LEVEL: usize = 16;

/// Construction parameters.
#[derive(Clone, Debug)]
pub(crate) struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub threads: usize,
    pub seed: u64,
}

/// A built graph in its on-disk shape.
pub(crate) struct Graph {
    pub m: usize,
    pub entry: u32,
    pub max_level: usize,
    pub layer0: Vec<u32>,
    /// Per upper layer 1..=max_level: (sorted node ids, ids.len() × m links).
    pub upper: Vec<(Vec<u32>, Vec<u32>)>,
}

impl Graph {
    /// The `hnsw_upper` section.
    pub fn upper_bytes(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.u64(self.upper.len() as u64);
        for (ids, adj) in &self.upper {
            w.array(ids);
            w.array(adj);
        }
        w.finish()
    }

    pub fn layer0_bytes(&self) -> Vec<u8> {
        self.layer0.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Scored(f32, u32);

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}

impl Ord for Scored {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
    }
}

/// Level of node `i`: floor(-ln(U) / ln(M)) with U from a seeded hash.
fn level_of(i: u32, seed: u64, m: usize) -> usize {
    let mut z = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    ((-u.ln() / (m as f64).ln()).floor() as usize).min(MAX_LEVEL)
}

/// SQ8 codes to build on: `codes` holds row `v`'s at `v * stride`.
pub(crate) struct BuildCodes<'a> {
    pub sq8: &'a Sq8,
    pub codes: &'a [u8],
    pub stride: usize,
}

impl BuildCodes<'_> {
    fn row(&self, v: u32) -> &[u8] {
        &self.codes[v as usize * self.stride..(v as usize + 1) * self.stride]
    }
}

/// A vector prepared for scoring others during construction.
enum Probe<'a> {
    F32(&'a [f32]),
    Sq8(Sq8Query),
}

struct Builder<'a> {
    rows: &'a Rows<'a>,
    codes: Option<&'a BuildCodes<'a>>,
    m: usize,
    m0: usize,
    ef: usize,
    /// The beam of a layer-0 search started from a joined node's old
    /// neighbours: half the beam (Lucene caps it at 3M, which lost recall
    /// on TREC-COVID, notes/vector.md).
    ef_seeded: usize,
    levels: Vec<u8>,
    /// links[v][l]: node v's neighbours on layer l (empty for absent docs).
    links: Vec<Vec<Mutex<Vec<u32>>>>,
    entry: Mutex<(u32, usize)>,
    /// Held by an insert whose level exceeds the current top level.
    grow: Mutex<()>,
    /// Whether each node's insert has finished.
    done: Vec<AtomicBool>,
}

struct Scratch {
    visited: Vec<u32>,
    epoch: u32,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch { visited: vec![0; n], epoch: 0 }
    }
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }
}

impl Builder<'_> {
    fn row(&self, v: u32) -> &[f32] {
        self.rows.row(v as usize)
    }

    /// `v` prepared for scoring the others.
    fn probe(&self, v: u32) -> Probe<'_> {
        match self.codes {
            None => Probe::F32(self.row(v)),
            Some(c) => Probe::Sq8(c.sq8.prepare_int8(self.row(v), c.stride)),
        }
    }

    /// Similarity of `p` with `v`: exact, or of `p`'s float32 values with
    /// `v`'s codes.
    fn score(&self, p: &Probe, v: u32) -> f32 {
        match (p, self.codes) {
            (Probe::Sq8(q), Some(c)) => q.score(q.dot(c.row(v))),
            (Probe::F32(q), _) => dot_f32(q, self.row(v)),
            (Probe::Sq8(_), None) => unreachable!("probes match the builder"),
        }
    }

    /// Vectors are 4 KiB at 1024 dimensions (1 KiB as codes) and scattered:
    /// callers start fetching the next one while scoring the current one.
    fn prefetch(&self, v: u32) {
        match self.codes {
            None => {
                let r = self.row(v);
                kernels::prefetch(r.as_ptr() as *const u8, r.len() * 4);
            }
            Some(c) => {
                let r = c.row(v);
                kernels::prefetch(r.as_ptr(), r.len());
            }
        }
    }

    fn neighbours(&self, v: u32, l: usize) -> Vec<u32> {
        self.links[v as usize][l].lock().unwrap().clone()
    }

    fn greedy(&self, q: &Probe, mut cur: Scored, l: usize) -> Scored {
        loop {
            let mut changed = false;
            for nb in self.neighbours(cur.1, l) {
                let s = self.score(q, nb);
                if s > cur.0 {
                    cur = Scored(s, nb);
                    changed = true;
                }
            }
            if !changed {
                return cur;
            }
        }
    }

    /// The `ef` nodes closest to `q` on layer `l` reachable from `eps`, best first.
    fn search_layer(&self, q: &Probe, eps: &[Scored], ef: usize, l: usize, sc: &mut Scratch) -> Vec<Scored> {
        let epoch = sc.next_epoch();
        let mut cand: BinaryHeap<Scored> = BinaryHeap::new();
        let mut res: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        for &e in eps {
            sc.visited[e.1 as usize] = epoch;
            cand.push(e);
            res.push(Reverse(e));
        }
        while res.len() > ef {
            res.pop();
        }
        while let Some(c) = cand.pop() {
            if res.len() >= ef && c < res.peek().unwrap().0 {
                break;
            }
            let mut fresh = self.neighbours(c.1, l);
            fresh.retain(|&nb| std::mem::replace(&mut sc.visited[nb as usize], epoch) != epoch);
            for (i, &nb) in fresh.iter().enumerate() {
                if let Some(&next) = fresh.get(i + 1) {
                    self.prefetch(next);
                }
                let s = Scored(self.score(q, nb), nb);
                if res.len() < ef || s > res.peek().unwrap().0 {
                    cand.push(s);
                    res.push(Reverse(s));
                    if res.len() > ef {
                        res.pop();
                    }
                }
            }
        }
        let mut v: Vec<Scored> = res.into_iter().map(|r| r.0).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }

    /// Keeps a candidate (best first) only if it is closer to the base than
    /// to every neighbour already kept, which spreads links across directions.
    fn select(&self, cands: &[Scored], m: usize) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(m);
        let mut kept: Vec<Probe> = Vec::with_capacity(m);
        for (i, c) in cands.iter().enumerate() {
            if out.len() >= m {
                break;
            }
            if let Some(next) = cands.get(i + 1) {
                self.prefetch(next.1);
            }
            if kept.iter().all(|r| self.score(r, c.1) <= c.0) {
                out.push(c.1);
                if out.len() < m {
                    kept.push(self.probe(c.1));
                }
            }
        }
        out
    }

    fn connect(&self, e: u32, new: u32, l: usize) {
        let cap = if l == 0 { self.m0 } else { self.m };
        let mut g = self.links[e as usize][l].lock().unwrap();
        if g.len() < cap {
            g.push(new);
            return;
        }
        let ev = self.probe(e);
        let ids: Vec<u32> = g.iter().copied().chain([new]).collect();
        let mut cands: Vec<Scored> = Vec::with_capacity(ids.len());
        for (i, &x) in ids.iter().enumerate() {
            if let Some(&next) = ids.get(i + 1) {
                self.prefetch(next);
            }
            cands.push(Scored(self.score(&ev, x), x));
        }
        cands.sort_unstable_by(|a, b| b.cmp(a));
        *g = self.select(&cands, cap);
    }

    /// Inserts `i`; a non-empty `seeds` starts its layer-0 search instead of
    /// the layer above's result, with the narrower beam.
    fn insert(&self, i: u32, seeds: &[u32], sc: &mut Scratch) {
        let q = self.probe(i);
        let level = self.levels[i as usize] as usize;
        let (mut ep, mut top) = *self.entry.lock().unwrap();
        let _grow = if level > top {
            let g = self.grow.lock().unwrap();
            (ep, top) = *self.entry.lock().unwrap();
            Some(g)
        } else {
            None
        };
        let mut cur = Scored(self.score(&q, ep), ep);
        for l in (level + 1..=top).rev() {
            cur = self.greedy(&q, cur, l);
        }
        let mut eps = vec![cur];
        for l in (0..=level.min(top)).rev() {
            let w = if l == 0 && !seeds.is_empty() {
                let mut from = Vec::with_capacity(seeds.len());
                for (j, &s) in seeds.iter().enumerate() {
                    if let Some(&next) = seeds.get(j + 1) {
                        self.prefetch(next);
                    }
                    from.push(Scored(self.score(&q, s), s));
                }
                self.search_layer(&q, &from, self.ef_seeded, 0, sc)
            } else {
                self.search_layer(&q, &eps, self.ef, l, sc)
            };
            let sel = self.select(&w, self.m);
            *self.links[i as usize][l].lock().unwrap() = sel.clone();
            for e in sel {
                self.connect(e, i, l);
            }
            eps = w;
        }
        if level > top {
            *self.entry.lock().unwrap() = (i, level);
        }
    }
}

/// A graph to start a build from: `view`'s node `v` is the build's node
/// `base + v`, with the same vector (a merge's input that lost no
/// documents). The first one given that fits keeps its links; the others
/// that fit are joined to it.
pub(crate) struct StartGraph<'a> {
    pub view: &'a HnswView,
    pub base: u32,
    pub n: usize,
}

impl StartGraph<'_> {
    /// Whether the graph's nodes are exactly the build's present rows in its
    /// range (a node of a graph of several has links), so its links can be
    /// kept as they are.
    fn fits(&self, present: &(dyn Fn(u32) -> bool + Sync), n: usize, m: usize) -> bool {
        self.view.m == m
            && self.base as usize + self.n <= n
            && (self.view.entry == NONE || (self.view.entry as usize) < self.n)
            && (0..self.n as u32).all(|v| present(self.base + v) == (v == self.view.entry || !self.view.neighbours(v, 0).is_empty()))
    }

    /// Node `v`'s level: the highest upper layer holding it.
    fn levels(&self) -> Vec<u8> {
        let mut levels = vec![0u8; self.n];
        for (l, (ids, _)) in self.view.upper.iter().enumerate() {
            for &v in ids.iter() {
                levels[v as usize] = levels[v as usize].max(l as u8 + 1);
            }
        }
        levels
    }

    fn range(&self) -> std::ops::Range<u32> {
        self.base..self.base + self.n as u32
    }

    /// Lucene's join set (10.2, `UpdateGraphsUtils.computeJoinSet`): nodes
    /// taken greedily by how much they cover, until each node is in the set
    /// or has `k` of its layer-0 neighbours in it, `k` a quarter of its
    /// neighbours (two below nine). Gains are recomputed lazily, as Lucene
    /// does, so the cover is close to that, not exact.
    fn join_set(&self) -> Vec<bool> {
        let n = self.n;
        let nbrs = |v: u32| self.view.neighbours(v, 0);
        let need: Vec<u32> = (0..n as u32)
            .map(|v| {
                let d = nbrs(v).len();
                (if d < 9 { 2 } else { d.div_ceil(4) }).min(d) as u32
            })
            .collect();
        let goal: u64 = need.iter().map(|&k| k as u64).sum();
        let (mut counts, mut j, mut stale) = (vec![0u32; n], vec![false; n], vec![false; n]);
        let mut heap: BinaryHeap<(u32, Reverse<u32>)> = (0..n as u32).map(|v| (need[v as usize] + nbrs(v).len() as u32, Reverse(v))).collect();
        let mut got = 0u64;
        while got < goal {
            let Some((gain, Reverse(v))) = heap.pop() else { break };
            let vi = v as usize;
            if stale[vi] {
                stale[vi] = false;
                let g = need[vi].saturating_sub(counts[vi]) + nbrs(v).iter().filter(|&&u| !j[u as usize] && counts[u as usize] < need[u as usize]).count() as u32;
                if g > 0 {
                    heap.push((g, Reverse(v)));
                }
                continue;
            }
            j[vi] = true;
            got += gain as u64;
            for &u in nbrs(v) {
                let ui = u as usize;
                stale[ui] = true;
                counts[ui] += 1;
                if counts[ui] == need[ui] {
                    for &w in nbrs(u) {
                        stale[w as usize] = true;
                    }
                }
            }
        }
        j
    }
}

/// A graph joined to the one the build starts from, and its join set.
struct Joined<'a> {
    start: &'a StartGraph<'a>,
    join: Vec<bool>,
}

impl Joined<'_> {
    /// Build node `i`'s seeds: its old neighbours already in the graph
    /// being built (the join set and the nodes inserted before it, as in
    /// Lucene) and their neighbours there.
    fn seeds(&self, b: &Builder, i: u32) -> Vec<u32> {
        let s = self.start;
        let mut out = Vec::new();
        for &v in s.view.neighbours(i - s.base, 0) {
            if self.join[v as usize] || b.done[(s.base + v) as usize].load(Ordering::Acquire) {
                out.push(s.base + v);
                out.extend(b.neighbours(s.base + v, 0));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Builds the graph over the present rows, scoring with `codes` if given.
/// Started from graphs that fit, the first one's nodes keep their levels
/// and links, as Lucene does when merging (from 9.9), and the others are
/// joined to it (Lucene 10.2): their join sets are inserted as usual, their
/// other nodes from seeds.
pub(crate) fn build(rows: &Rows, present: &(dyn Fn(u32) -> bool + Sync), cfg: &HnswConfig, starts: &[StartGraph], codes: Option<&BuildCodes>) -> Graph {
    let n = rows.n;
    let (m, m0) = (cfg.m.max(2), 2 * cfg.m.max(2));
    let mut fitting: Vec<&StartGraph> = Vec::new();
    for s in starts {
        if s.view.entry != NONE && s.fits(present, n, m) && fitting.iter().all(|f| f.range().end <= s.base || s.range().end <= f.base) {
            fitting.push(s);
        }
    }
    let start = fitting.first().copied();
    let joined: Vec<Joined> = fitting.iter().skip(1).map(|&s| Joined { start: s, join: s.join_set() }).collect();
    let started = |i: u32| start.is_some_and(|s| s.range().contains(&i));
    // Build node `i`'s joined graph and whether it is in that graph's join set.
    let joined_at = |i: u32| joined.iter().find(|j| j.start.range().contains(&i)).map(|j| (j, j.join[(i - j.start.base) as usize]));
    let mut levels: Vec<u8> = (0..n as u32).map(|i| if present(i) { level_of(i, cfg.seed, m) as u8 } else { 0 }).collect();
    if let Some(s) = start {
        levels[s.base as usize..s.base as usize + s.n].copy_from_slice(&s.levels());
    }
    let links: Vec<Vec<Mutex<Vec<u32>>>> = (0..n as u32)
        .map(|i| if present(i) { (0..=levels[i as usize]).map(|l| Mutex::new(Vec::with_capacity(if l == 0 { m0 + 1 } else { m + 1 }))).collect() } else { Vec::new() })
        .collect();
    if let Some(s) = start {
        for v in 0..s.n as u32 {
            for (l, g) in links[(s.base + v) as usize].iter().enumerate() {
                g.lock().unwrap().extend(s.view.neighbours(v, l).iter().map(|&x| x + s.base));
            }
        }
    }
    let first = match start {
        Some(s) => Some(s.base + s.view.entry),
        None => (0..n as u32).find(|&i| present(i)),
    };
    let Some(first) = first else {
        return Graph { m, entry: NONE, max_level: 0, layer0: vec![NONE; n * m0], upper: Vec::new() };
    };
    let ef = cfg.ef_construction.max(m);
    let b = Builder {
        rows,
        codes,
        m,
        m0,
        ef,
        ef_seeded: (ef / 2).max(m),
        levels,
        links,
        entry: Mutex::new((first, 0)),
        grow: Mutex::new(()),
        done: (0..n).map(|_| AtomicBool::new(false)).collect(),
    };
    *b.entry.lock().unwrap() = (first, b.levels[first as usize] as usize);
    // Concurrent inserts cannot see each other. In docID order, threads
    // insert neighbouring docIDs together, and where docIDs follow the
    // source (consecutive passages of one document are each other's nearest
    // neighbours) those pairs end up unlinked: 4 threads cut recall@10 at ef
    // 64 from 0.86 to 0.77 on MS MARCO passages (notes/scale.md). A seeded
    // permutation makes concurrent inserts unrelated.
    // The joined graphs' nodes outside their join sets go in last, from
    // seeds among the nodes already in.
    let (mut order, mut seeded): (Vec<u32>, Vec<u32>) =
        (0..n as u32).filter(|&i| i != first && present(i) && !started(i)).partition(|&i| joined_at(i).is_none_or(|(_, in_join)| in_join));
    if cfg.threads > 1 {
        use rand::{seq::SliceRandom, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        order.shuffle(&mut rng);
        seeded.shuffle(&mut rng);
    }
    for (pass, ids) in [order, seeded].iter().enumerate() {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..cfg.threads.max(1) {
                s.spawn(|| {
                    let mut sc = Scratch::new(n);
                    loop {
                        let j = next.fetch_add(1, Ordering::Relaxed);
                        if j >= ids.len() {
                            break;
                        }
                        let i = ids[j];
                        let seeds = match (pass, joined_at(i)) {
                            (1, Some((g, _))) => g.seeds(&b, i),
                            _ => Vec::new(),
                        };
                        b.insert(i, &seeds, &mut sc);
                        b.done[i as usize].store(true, Ordering::Release);
                    }
                });
            }
        });
    }
    let (entry, max_level) = *b.entry.lock().unwrap();
    let mut layer0 = vec![NONE; n * m0];
    for (v, l) in b.links.iter().enumerate() {
        if let Some(l0) = l.first() {
            let g = l0.lock().unwrap();
            layer0[v * m0..v * m0 + g.len()].copy_from_slice(&g);
        }
    }
    let mut upper = Vec::with_capacity(max_level);
    for l in 1..=max_level {
        let ids: Vec<u32> = (0..n as u32).filter(|&v| present(v) && b.levels[v as usize] as usize >= l).collect();
        let mut adj = vec![NONE; ids.len() * m];
        for (j, &v) in ids.iter().enumerate() {
            let g = b.links[v as usize][l].lock().unwrap();
            adj[j * m..j * m + g.len()].copy_from_slice(&g);
        }
        upper.push((ids, adj));
    }
    Graph { m, entry, max_level, layer0, upper }
}

/// Zero-copy view of a stored graph.
pub(crate) struct HnswView {
    m: usize,
    m0: usize,
    entry: u32,
    layer0: Slice<u32>,
    upper: Vec<(Slice<u32>, Slice<u32>)>,
}

impl HnswView {
    pub fn open(m: usize, entry: u32, layer0: OwnedBytes, upper: OwnedBytes, n: usize) -> Result<Self> {
        let m0 = 2 * m;
        let layer0: Slice<u32> = Slice::view(layer0).map_err(corrupt)?;
        if layer0.len() != n * m0 {
            return Err(corrupt("hnsw layer 0 has the wrong size"));
        }
        let mut r = ByteReader::new(upper);
        let levels = r.u64().map_err(corrupt)? as usize;
        let mut up = Vec::with_capacity(levels);
        for _ in 0..levels {
            let ids: Slice<u32> = r.array().map_err(corrupt)?;
            let adj: Slice<u32> = r.array().map_err(corrupt)?;
            if adj.len() != ids.len() * m {
                return Err(corrupt("hnsw upper layer has the wrong size"));
            }
            up.push((ids, adj));
        }
        if entry != NONE && entry as usize >= n {
            return Err(corrupt("hnsw entry point out of range"));
        }
        Ok(HnswView { m, m0, entry, layer0, upper: up })
    }

    pub fn is_shared(&self) -> bool {
        matches!(self.layer0, Slice::Shared { .. }) && self.upper.iter().all(|(a, b)| matches!(a, Slice::Shared { .. }) && matches!(b, Slice::Shared { .. }))
    }

    fn neighbours(&self, v: u32, l: usize) -> &[u32] {
        let s: &[u32] = if l == 0 {
            &self.layer0[v as usize * self.m0..(v as usize + 1) * self.m0]
        } else {
            let (ids, adj) = &self.upper[l - 1];
            match ids.binary_search(&v) {
                Ok(j) => &adj[j * self.m..(j + 1) * self.m],
                Err(_) => &[],
            }
        };
        &s[..s.iter().position(|&x| x == NONE).unwrap_or(s.len())]
    }

    /// The `ef` best members by SQ8 score, best first.
    ///
    /// With a filter, `two_hop` picks the traversal: without it every reached
    /// node is scored and expanded and only members are collected; with it
    /// (ACORN-1 style) only members are scored and expanded, and a node's
    /// non-member neighbours are looked through to their own neighbours, up
    /// to `2M` members per expansion. With `hints`, the rows about to be
    /// scored are announced to the kernel first.
    #[allow(clippy::too_many_arguments)]
    pub fn search(&self, codes: &Codes, q: &QueryCode, ef: usize, filter: Option<Members>, two_hop: bool, hints: bool, stats: &mut SearchStats) -> Vec<(i32, DocId)> {
        if self.entry == NONE || ef == 0 {
            return Vec::new();
        }
        let mut cur = self.entry;
        let mut cur_s = q.dot(codes.row(cur));
        stats.sq8_distances += 1;
        let announce = |ids: &[u32]| {
            if hints {
                io::will_need(&mut ids.iter().map(|&d| (codes.row(d).as_ptr() as usize, codes.stride())).collect::<Vec<_>>());
            }
        };
        for l in (1..=self.upper.len()).rev() {
            loop {
                let mut changed = false;
                announce(self.neighbours(cur, l));
                for &nb in self.neighbours(cur, l) {
                    let s = q.dot(codes.row(nb));
                    stats.sq8_distances += 1;
                    if s > cur_s || (s == cur_s && nb < cur) {
                        (cur, cur_s) = (nb, s);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let n = codes.n();
        let mut visited = vec![0u64; n.div_ceil(64)];
        let seen = |v: &mut Vec<u64>, d: u32| {
            let (w, b) = (d as usize / 64, d % 64);
            let was = v[w] >> b & 1 == 1;
            v[w] |= 1 << b;
            was
        };
        let peek = |v: &Vec<u64>, d: u32| v[d as usize / 64] >> (d % 64) & 1 == 1;
        let two_hop = two_hop && filter.is_some();
        let is_member = |d: u32, stats: &mut SearchStats| match &filter {
            None => true,
            Some(f) => {
                stats.filter_checks += 1;
                f.contains(d)
            }
        };
        // Candidates: max-heap by (score, smaller docID); results: TopK of members.
        let key = |s: i32, d: u32| ((s as i64) << 32) | (u32::MAX - d) as i64;
        let mut cand: BinaryHeap<i64> = BinaryHeap::new();
        let mut res = TopK::new(ef);
        seen(&mut visited, cur);
        cand.push(key(cur_s, cur));
        if is_member(cur, stats) {
            res.push(cur_s, cur);
        }
        let mut fresh: Vec<u32> = Vec::with_capacity(self.m0);
        let mut hop2: Vec<u32> = Vec::with_capacity(self.m0);
        while let Some(c) = cand.pop() {
            let (cs, cd) = ((c >> 32) as i32, u32::MAX - c as u32);
            if res.is_full() && cs < res.floor() {
                break;
            }
            stats.nodes_visited += 1;
            fresh.clear();
            if two_hop {
                hop2.clear();
                for &nb in self.neighbours(cd, 0) {
                    if !peek(&visited, nb) {
                        if is_member(nb, stats) {
                            seen(&mut visited, nb);
                            fresh.push(nb);
                        } else {
                            hop2.push(nb);
                        }
                    }
                }
                // A non-member is marked only once looked through, so one cut
                // short by the 2M cap stays a bridge for later expansions.
                'look: for &v in &hop2 {
                    if fresh.len() >= self.m0 {
                        break;
                    }
                    seen(&mut visited, v);
                    for &nb in self.neighbours(v, 0) {
                        if fresh.len() >= self.m0 {
                            break 'look;
                        }
                        if !peek(&visited, nb) && is_member(nb, stats) {
                            seen(&mut visited, nb);
                            fresh.push(nb);
                        }
                    }
                }
            } else {
                for &nb in self.neighbours(cd, 0) {
                    if !seen(&mut visited, nb) {
                        fresh.push(nb);
                    }
                }
            }
            for &nb in &fresh {
                kernels::prefetch(codes.row(nb).as_ptr(), codes.stride());
            }
            announce(&fresh);
            let mut i = 0;
            while i < fresh.len() {
                let mut scores = [0i32; 4];
                let take = (fresh.len() - i).min(4);
                if take == 4 {
                    scores = q.dot4([codes.row(fresh[i]), codes.row(fresh[i + 1]), codes.row(fresh[i + 2]), codes.row(fresh[i + 3])]);
                } else {
                    for j in 0..take {
                        scores[j] = q.dot(codes.row(fresh[i + j]));
                    }
                }
                stats.sq8_distances += take;
                for j in 0..take {
                    let (s, nb) = (scores[j], fresh[i + j]);
                    if !res.is_full() || s > res.floor() {
                        cand.push(key(s, nb));
                        if two_hop || is_member(nb, stats) {
                            res.push(s, nb);
                        }
                    }
                }
                i += take;
            }
        }
        res.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_geometric() {
        let n = 200_000u32;
        let mut count = [0usize; 4];
        for i in 0..n {
            let l = level_of(i, 42, 16);
            for c in count.iter_mut().take(l.min(3) + 1) {
                *c += 1;
            }
        }
        // P(level >= l) = 16^-l.
        assert_eq!(count[0], n as usize);
        let p1 = count[1] as f64 / n as f64;
        assert!((p1 - 1.0 / 16.0).abs() < 0.005, "p1 {p1}");
        let p2 = count[2] as f64 / n as f64;
        assert!((p2 - 1.0 / 256.0).abs() < 0.001, "p2 {p2}");
    }
}
