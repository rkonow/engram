//! Bounded top-k selection and the filter adapter used by every search path.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::segment::{DocFilter, DocId, DocSet};

/// Keeps the `k` best `(score, doc)` pairs by integer SQ8 score; ties go to
/// the smaller docID so results are deterministic.
#[derive(Default)]
pub(crate) struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<i64>>,
}

/// Orders by score, then by smaller docID; unique per document.
#[inline]
fn pack(score: i32, doc: DocId) -> i64 {
    ((score as i64) << 32) | (u32::MAX - doc) as i64
}

#[inline]
fn unpack(key: i64) -> (i32, DocId) {
    ((key >> 32) as i32, u32::MAX - (key as u32))
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k, heap: BinaryHeap::with_capacity(k.saturating_add(1).min(4096)) }
    }

    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// The worst kept score once full (anything below cannot enter).
    #[inline]
    pub fn floor(&self) -> i32 {
        if self.is_full() {
            self.heap.peek().map_or(i32::MIN, |r| unpack(r.0).0)
        } else {
            i32::MIN
        }
    }

    #[inline]
    pub fn push(&mut self, score: i32, doc: DocId) {
        if self.k == 0 {
            return;
        }
        let key = pack(score, doc);
        if self.heap.len() < self.k {
            self.heap.push(Reverse(key));
        } else if key > self.heap.peek().unwrap().0 {
            self.heap.pop();
            self.heap.push(Reverse(key));
        }
    }

    /// [`Self::push`], giving back the pair it does not keep: the one it
    /// pushed out, or this one.
    #[inline]
    pub fn push_out(&mut self, score: i32, doc: DocId) -> Option<(i32, DocId)> {
        let key = pack(score, doc);
        if self.heap.len() < self.k {
            self.heap.push(Reverse(key));
            None
        } else if self.k > 0 && key > self.heap.peek().unwrap().0 {
            let out = self.heap.pop().unwrap().0;
            self.heap.push(Reverse(key));
            Some(unpack(out))
        } else {
            Some((score, doc))
        }
    }

    /// How many it keeps.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Best first, kept.
    pub fn sorted(&self) -> Vec<(i32, DocId)> {
        let mut v: Vec<i64> = self.heap.iter().map(|r| r.0).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.into_iter().map(unpack).collect()
    }

    /// Best first.
    pub fn into_sorted(self) -> Vec<(i32, DocId)> {
        let mut v: Vec<i64> = self.heap.into_iter().map(|r| r.0).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.into_iter().map(unpack).collect()
    }
}

/// A filter seen through its cheapest access path: materialised sets are
/// enumerated and intersected directly, anything else is probed per document.
#[derive(Clone, Copy)]
pub(crate) enum Members<'a> {
    Sorted(&'a [DocId]),
    Bits(&'a [u64]),
    Probe(&'a dyn DocFilter),
}

impl<'a> Members<'a> {
    pub fn of(f: &'a dyn DocFilter) -> Self {
        match f.as_doc_set() {
            Some(DocSet::Sorted { docs, .. }) => Members::Sorted(docs),
            Some(DocSet::Bits { words, .. }) => Members::Bits(words),
            None => Members::Probe(f),
        }
    }

    #[inline]
    pub fn contains(&self, d: DocId) -> bool {
        match self {
            Members::Sorted(s) => s.binary_search(&d).is_ok(),
            Members::Bits(w) => w.get(d as usize / 64).is_some_and(|x| x >> (d % 64) & 1 == 1),
            Members::Probe(f) => f.contains(d),
        }
    }

    /// Calls `f` with the members below `n` in ascending order, in chunks.
    pub fn for_each_chunk(&self, n: u32, mut f: impl FnMut(&[DocId])) {
        const CHUNK: usize = 256;
        match self {
            Members::Sorted(s) => {
                let end = s.partition_point(|&d| d < n);
                for c in s[..end].chunks(CHUNK) {
                    f(c);
                }
            }
            Members::Bits(words) => {
                let mut buf = Vec::with_capacity(CHUNK + 64);
                for (i, &w) in words.iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        let d = i as u32 * 64 + w.trailing_zeros();
                        if d >= n {
                            break;
                        }
                        buf.push(d);
                        w &= w - 1;
                    }
                    if buf.len() >= CHUNK {
                        f(&buf);
                        buf.clear();
                    }
                }
                if !buf.is_empty() {
                    f(&buf);
                }
            }
            Members::Probe(p) => {
                let mut buf = Vec::with_capacity(CHUNK);
                for d in 0..n {
                    if p.contains(d) {
                        buf.push(d);
                        if buf.len() == CHUNK {
                            f(&buf);
                            buf.clear();
                        }
                    }
                }
                if !buf.is_empty() {
                    f(&buf);
                }
            }
        }
    }

    /// Positions `i` with `ids[i]` a member (`ids` ascending), appended to
    /// `out`. Sorted sets are intersected by galloping, so a short cluster
    /// list costs O(|ids| · log(gap)) against a long filter list.
    pub fn select(&self, ids: &[DocId], out: &mut Vec<u32>) {
        match self {
            Members::Sorted(s) => {
                let mut lo = 0usize;
                for (i, &d) in ids.iter().enumerate() {
                    // Gallop forward from `lo` to bracket `d`, then binary search.
                    let mut step = 1;
                    let mut hi = lo;
                    while hi < s.len() && s[hi] < d {
                        lo = hi + 1;
                        hi += step;
                        step *= 2;
                    }
                    let hi = hi.min(s.len());
                    lo += s[lo..hi].partition_point(|&x| x < d);
                    if lo < s.len() && s[lo] == d {
                        out.push(i as u32);
                    }
                    if lo >= s.len() {
                        break;
                    }
                }
            }
            Members::Bits(w) => {
                for (i, &d) in ids.iter().enumerate() {
                    if w.get(d as usize / 64).is_some_and(|x| x >> (d % 64) & 1 == 1) {
                        out.push(i as u32);
                    }
                }
            }
            Members::Probe(f) => {
                for (i, &d) in ids.iter().enumerate() {
                    if f.contains(d) {
                        out.push(i as u32);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn topk_keeps_best_with_docid_ties() {
        let mut t = TopK::new(3);
        for (s, d) in [(5, 9), (7, 1), (5, 2), (1, 0), (7, 0), (6, 3)] {
            t.push(s, d);
        }
        assert_eq!(t.into_sorted(), vec![(7, 0), (7, 1), (6, 3)]);
        let mut t = TopK::new(2);
        t.push(-5, 4);
        t.push(i32::MIN, 0);
        assert_eq!(t.floor(), i32::MIN);
        t.push(-1, 1);
        assert_eq!(t.floor(), -5);
        assert_eq!(t.into_sorted(), vec![(-1, 1), (-5, 4)]);
    }

    struct Odd;
    impl DocFilter for Odd {
        fn contains(&self, d: DocId) -> bool {
            d % 2 == 1
        }
        fn cardinality(&self) -> usize {
            0
        }
    }

    #[test]
    fn member_access_paths_agree() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let n = 5000u32;
        for density in [0.001, 0.02, 0.3] {
            let docs: Vec<u32> = (0..n).filter(|_| rng.gen_bool(density)).collect();
            let set = DocSet::from_sorted(docs.clone(), n);
            let m = Members::of(&set);
            let mut seen = Vec::new();
            m.for_each_chunk(n, |c| seen.extend_from_slice(c));
            assert_eq!(seen, docs);
            let ids: Vec<u32> = (0..n).filter(|_| rng.gen_bool(0.05)).collect();
            let mut pos = Vec::new();
            m.select(&ids, &mut pos);
            let want: Vec<u32> = (0..ids.len() as u32).filter(|&i| docs.binary_search(&ids[i as usize]).is_ok()).collect();
            assert_eq!(pos, want, "density {density}");
        }
        let probe = Members::of(&Odd);
        let mut seen = Vec::new();
        probe.for_each_chunk(10, |c| seen.extend_from_slice(c));
        assert_eq!(seen, vec![1, 3, 5, 7, 9]);
    }
}
