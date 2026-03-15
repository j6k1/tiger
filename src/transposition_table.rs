use std::mem;
use std::num::Wrapping;
use std::ops::{Add, BitXor, Deref, DerefMut, Index, IndexMut, Neg, Sub};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use usiagent::hash::{InitialHash, KyokumenHash};
use usiagent::rule::{LegalMove, AppliedMove, AtomicLegalMove};
use usiagent::shogi::{Banmen, Mochigoma, MochigomaCollections, MochigomaKind, Teban};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Bound {
    None,
    UpperBound,
    LowerBound,
    Exact,
}
pub trait ExactScoreBound {
    fn exact_score_bound(&self) -> bool;
}
pub trait ToBucketIndex {
    fn to_bucket_index(self) -> usize;
}
impl ToBucketIndex for u128 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u64 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u32 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u16 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u8 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Score {
    NEGINFINITE,
    MAYBENEGINFINITE,
    Value(i32),
    MAYBEINFINITE,
    INFINITE,
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::Value(v) => Score::Value(-v),
            Score::INFINITE => Score::NEGINFINITE,
            Score::NEGINFINITE => Score::INFINITE,
            Score::MAYBEINFINITE => Score::MAYBENEGINFINITE,
            Score::MAYBENEGINFINITE => Score::MAYBEINFINITE,
        }
    }
}
impl Add<i32> for Score {
    type Output = Self;

    fn add(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v + other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
            Score::MAYBEINFINITE => Score::MAYBEINFINITE,
            Score::MAYBENEGINFINITE => Score::MAYBENEGINFINITE,
        }
    }
}
impl Sub<i32> for Score {
    type Output = Self;

    fn sub(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v - other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
            Score::MAYBEINFINITE => Score::MAYBEINFINITE,
            Score::MAYBENEGINFINITE => Score::MAYBENEGINFINITE,
        }
    }
}
impl Default for Score {
    fn default() -> Self {
        Score::NEGINFINITE
    }
}
impl ExactScoreBound for Score {
    fn exact_score_bound(&self) -> bool {
        *self == Score::INFINITE
    }
}
#[derive(Debug,Clone)]
pub struct ZobristHash<T>
    where T: Add + Sub + BitXor<Output = T> + Copy + InitialHash,
             Wrapping<T>: Add<Output = Wrapping<T>> + Sub<Output = Wrapping<T>> + BitXor<Output = Wrapping<T>> + Copy,
             Standard: Distribution<T> {
    mhash:T,
    shash:T,
    teban:Teban
}
impl<T> ZobristHash<T>
    where T: Add + Sub + BitXor<Output = T> + Copy + InitialHash,
             Wrapping<T>: Add<Output = Wrapping<T>> + Sub<Output = Wrapping<T>> + BitXor<Output = Wrapping<T>> + Copy,
             Standard: Distribution<T> {
    pub fn new(hasher:&KyokumenHash<T>,teban:Teban,banmen:&Banmen,ms:&Mochigoma,mg:&Mochigoma) -> ZobristHash<T> {
        let (mhash,shash) = hasher.calc_initial_hash(&banmen, &ms, &mg);

        ZobristHash {
            mhash:mhash,
            shash:shash,
            teban:teban
        }
    }

    pub fn updated(&self,hasher:&KyokumenHash<T>,teban:Teban,banmen:&Banmen,mc:&MochigomaCollections,m:AppliedMove,obtained:&Option<MochigomaKind>)
        -> ZobristHash<T> {
        let mhash = hasher.calc_main_hash(self.mhash,teban,banmen,mc,m,obtained);
        let shash = hasher.calc_sub_hash(self.shash,teban,banmen,mc,m,obtained);

        ZobristHash {
            mhash:mhash,
            shash:shash,
            teban:teban.opposite()
        }
    }

    pub fn teban_fliped(&self) -> ZobristHash<T> {
        ZobristHash {
            mhash:self.mhash,
            shash:self.shash,
            teban:self.teban.opposite()
        }
    }
    
    pub fn keys(&self) -> (T,T) {
        (self.mhash,self.shash)
    }

    pub fn teban(&self) -> Teban {
        self.teban
    }
}
#[derive(Debug,Clone)]
pub struct TTPartialEntry {
    pub depth:i8,
    pub score:Score,
    pub bound:Bound,
    pub best_move:Option<LegalMove>
}
impl Default for TTPartialEntry {
    fn default() -> Self {
        TTPartialEntry {
            depth:-1,
            score:Score::default(),
            bound:Bound::None,
            best_move: None
        }
    }
}
#[repr(C)]
#[derive(Debug)]
pub struct TTEntry {
    /*
    used:bool,
    mhash:K,
    shash:K,
    teban:Teban,
    entry:TTPartialEntry<T>
     */
    key:AtomicU64,
    payload:AtomicU64,
    best_move:AtomicLegalMove,
    version:AtomicU32,
    reserved:AtomicU64
}
impl TTEntry {
    pub fn unpack(&self) -> (Teban,bool,i8,u16,Score,Bound) {
        let mut payload = self.payload.load(Ordering::Acquire);

        let teban = if payload & 1 == 0 { Teban::Sente } else { Teban::Gote };

        payload >>= 1;

        let used = (payload & 1) == 1;

        payload >>= 1;

        let depth = (payload & 0xff) as u8 as i8;

        payload >>= 8;

        let generation = (payload & 0xffff) as u16;

        payload >>= 16;

        let score = match payload {
            0x10000 => Score::NEGINFINITE,
            0x10011 => Score::INFINITE,
            0x10010 => Score::MAYBEINFINITE,
            0x10001 => Score::MAYBENEGINFINITE,
            v => Score::Value(v as i32)
        };

        payload >>= 17;

        let bound = match payload {
            0 => Bound::None,
            1 => Bound::LowerBound,
            2 => Bound::UpperBound,
            3 => Bound::Exact,
            _ => unreachable!()
        };

        (teban,used,depth,generation,score,bound)
    }
}
impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key:AtomicU64::new(0),
            payload:AtomicU64::new(0),
            best_move:AtomicLegalMove::default(),
            version:AtomicU32::new(0),
            reserved:AtomicU64::new(0)
        }
    }
}
const fn support_fast_mod(v:usize) -> bool {
    v != 0 && v & (v - 1) == 0
}
pub struct TT<const S:usize,const N:usize> {
    buckets:Vec<[TTEntry;N]>,
    generation:AtomicU16,
}
impl<const S:usize,const N:usize> TT<S,N> {
    pub fn new() -> TT<S, N> {
        let mut buckets = Vec::with_capacity(S);

        buckets.resize_with(S, || {
            (0..N).map(|_| TTEntry::default()).collect::<Vec<_>>().try_into().unwrap()
        });

        TT {
            buckets: buckets,
            generation: AtomicU16::new(0),
        }
    }

    pub fn clear(&mut self) {
        self.buckets.fill_with(|| {
            (0..N).map(|_| TTEntry::default()).collect::<Vec<_>>().try_into().unwrap()
        });
    }

    fn bucket_index(&self, zh: &ZobristHash<u64>) -> usize {
        if support_fast_mod(S) {
            zh.mhash.to_bucket_index() & (S - 1)
        } else {
            zh.mhash.to_bucket_index() % S
        }
    }

    pub fn generational_shift(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }
    pub fn pack(&self,teban:Teban,used:bool,depth:i8,generation:u16,score:Score,bound:Bound) -> u64 {
        let teban = teban as u64;

        let used:u64 = if used { 1 } else { 0 };

        let score:u64 = match score {
            Score::NEGINFINITE => 0x10000,
            Score::INFINITE => 0x10011,
            Score::MAYBEINFINITE => 0x10010,
            Score::MAYBENEGINFINITE => 0x10001,
            Score::Value(v) => v as u64
        };

        let payload:u64 = teban |
            (used << 1) |
            ((depth as u8 as u64) << 2) |
            ((generation as u64) << 10) | (score << 20) | ((bound as u8 as u64) << 37);

        payload
    }

    pub fn pack_keys(&self,mhash:u64,shash:u64) -> u64 {
        mhash ^ shash.rotate_left(13)
    }

    pub fn get(&self, zh: &ZobristHash<u64>, thread_index: usize) -> Option<TTPartialEntry> {
        let index = self.bucket_index(zh);
        let key = self.pack_keys(zh.mhash, zh.shash);

        let mut find_index = thread_index % N;
        let mut search_count = 0;
        let mut found = false;

        match &self.buckets[index] {
            bucket => {
                while search_count < N {
                    search_count += 1;

                    let v = bucket[find_index].version.load(Ordering::Acquire);

                    if v & 1 == 0 {
                        let bucket = &bucket[find_index];

                        if key == bucket.key.load(Ordering::Acquire) {
                            let (teban, used, depth, _, score, bound) = bucket.unpack();

                            if zh.teban == teban && used {
                                found = true;

                                let mv = bucket.best_move.load(Ordering::Acquire);

                                if v == bucket.version.load(Ordering::Acquire) {
                                    return Some(TTPartialEntry {
                                        depth: depth,
                                        score: score,
                                        bound: bound,
                                        best_move: mv,
                                    })
                                }
                            }
                        }
                    }

                    if found {
                        break;
                    }

                    find_index += 1;;
                    find_index = find_index % N;
                }
            }
        }

        None
    }

    pub fn update<'a>(&self,
                      zh: &'a ZobristHash<u64>,
                      thread_index: usize,
                      depth: i8,
                      score: Score,
                      bound: Bound,
                      best_move: Option<LegalMove>) {
        let payload = self.pack(zh.teban, true, depth, self.generation.load(Ordering::Acquire), score, bound);
        let key = self.pack_keys(zh.mhash, zh.shash);

        let index = self.bucket_index(zh);

        let mut find_index = thread_index % N;
        let mut search_count = 0;
        let mut priority = u32::MAX;
        let mut primary_index = find_index;

        match &self.buckets[index] {
            bucket => {
                while search_count < N {
                    search_count += 1;

                    let bucket = &bucket[find_index];

                    if key == bucket.key.load(Ordering::Acquire) {
                        if bucket.version.fetch_or(1, Ordering::SeqCst) & 1 == 0 {
                            bucket.payload.store(payload, Ordering::Release);
                            bucket.best_move.store(best_move, Ordering::Release);
                            bucket.version.fetch_add(1, Ordering::SeqCst);
                        }

                        return;
                    }

                    find_index += 1;
                    find_index = find_index % N;

                    let (_, used, depth, generation, _, _) = bucket.unpack();

                    if !used {
                        if bucket.version.fetch_or(1, Ordering::SeqCst) & 1 == 0 {
                            bucket.payload.store(payload, Ordering::Release);
                            bucket.best_move.store(best_move, Ordering::Release);
                            bucket.key.store(key, Ordering::Release);
                            bucket.version.fetch_add(1, Ordering::SeqCst);

                            return;
                        }
                    }

                    let pri = ((generation as u32) << 16) | depth.max(0) as u32;

                    if pri < priority {
                        priority = pri;
                        primary_index = find_index;
                    }
                }

                let bucket = &bucket[primary_index];

                if bucket.version.fetch_or(1, Ordering::SeqCst) & 1 == 0 {
                    bucket.payload.store(payload, Ordering::Release);
                    bucket.best_move.store(best_move, Ordering::Release);
                    bucket.key.store(key, Ordering::Release);
                    bucket.version.fetch_add(1, Ordering::SeqCst);

                    return;
                }
            }
        }
    }
}