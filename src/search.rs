use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::{Add, Deref, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex, LazyLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use nncombinator::arr::Arr;
use nncombinator::layer::{ForwardAll, PreTrain};
use parking_lot::RwLock;
use rand::Rng;
use rand::rngs::ThreadRng;
use rayon::ThreadPool;
use usiagent::bitboard::BitBoard;
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::consts::{FU_SCORE, PIECE_SCORE_MAP};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher, UsiGoTimeLimit};
use usiagent::hash::KyokumenHash;
use usiagent::logger::Logger;
use usiagent::math::Prng;
use usiagent::move_orderer::{MoveOrderer, UnusedQuietSee};
use usiagent::movepick::{MovePicker, RandomPicker};
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{CaptureOrPawnPromotions, Evasions, LegalMove, NonEvasions, QuietsWithoutPawnPromotions, Rule, Square, SquareToPoint, State, OU_SURROUNDING_BOTTOM_MASK, OU_SURROUNDING_MASK, OU_SURROUNDING_TOP_MASK, POSSIBLE_OU_CAPTURES_MASK_OF_GOTE, POSSIBLE_OU_CAPTURES_MASK_OF_SENTE};
use usiagent::see::calc_see;
use usiagent::shogi::{KomaKind, MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use usiagent::shogi::KomaKind::Blank;
use crate::error::ApplicationError;
use crate::features::HalfKP;
use crate::nn::{Evalutor, FEATURES_NUM};
use crate::transposition_table::{TT, ZobristHash, TTPartialEntry, Bound, ExactScoreBound};

pub const TURN_LIMIT:u32 = 10000;
pub const BASE_DEPTH:u32 = 14;
pub const MAX_DEPTH:u32 = 14;
pub const MAX_THREADS:u32 = 8;
pub const NODES_PER_LEAF_NODE:u16 = 5;
pub const GAMMA:u8 = 100;
pub const QUIET_SEE_FACTOR:i64 = 128;

const DEPTH_LIMIT: u32 = 64;
const MAX_MOVES: usize = 64;

fn generate_lmr_table() -> [[u8; MAX_MOVES]; DEPTH_LIMIT as usize] {
    let mut table = [[0; MAX_MOVES]; DEPTH_LIMIT as usize];

    for depth in 1..DEPTH_LIMIT {
        for mv in 1..MAX_MOVES {
            let base = (2809. / 128.) * (mv as f32).ln();
            let scaled = base * (depth as f32 / DEPTH_LIMIT as f32);
            let r = (scaled / 12.).floor() as i32;
            let r = r.clamp(0,4) as u8;

            table[depth as usize][mv] = r;
        }
    }

    table
}

static LMR_TABLE: LazyLock<[[u8; MAX_MOVES]; DEPTH_LIMIT as usize]> = LazyLock::new(|| generate_lmr_table());
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
pub struct Environment<L,S> where L: Logger, S: InfoSender {
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub search_id:Arc<AtomicUsize>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub teban:Teban,
    pub limit:Option<UsiGoTimeLimit>,
    pub turn_limit:Option<u32>,
    pub timelimit_margin:u64,
    pub current_limit:Arc<RwLock<(Option<Instant>,Option<Instant>)>>,
    pub base_depth:u32,
    pub max_nodes:Option<u64>,
    pub max_threads:u32,
    pub nodes_per_leaf_node:u16,
    pub gamma:u8,
    pub abort:Arc<AtomicBool>,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub history:HashSet<(Teban,u64,u64)>,
    pub transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
    pub move_orderer:MoveOrderer<UnusedQuietSee>,
    pub nodes:Arc<AtomicU64>
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            search_id:Arc::clone(&self.search_id),
            hasher:Arc::clone(&self.hasher),
            teban:self.teban.clone(),
            limit:self.limit.clone(),
            turn_limit:self.turn_limit.clone(),
            timelimit_margin:self.timelimit_margin.clone(),
            current_limit:self.current_limit.clone(),
            base_depth:self.base_depth,
            max_nodes:self.max_nodes.clone(),
            max_threads:self.max_threads,
            nodes_per_leaf_node:self.nodes_per_leaf_node,
            gamma:self.gamma,
            abort:Arc::clone(&self.abort),
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            history:self.history.clone(),
            transposition_table:self.transposition_table.clone(),
            move_orderer:self.move_orderer.clone(),
            nodes:Arc::clone(&self.nodes),
        }
    }
}
#[derive(Debug,Clone)]
pub enum EvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>),
    NodeLimits,
    Timeout,
    Stop,
    Repetition
}
#[derive(Debug)]
pub enum RootEvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32, usize, MoveOrderer<UnusedQuietSee>),
    NodeLimits,
    Timeout,
    Stop,
    Repetition
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               search_id:Arc<AtomicUsize>,
               teban:Teban,
               limit:Option<UsiGoTimeLimit>,
               turn_limit:Option<u32>,
               timelimit_margin:u64,
               current_limit:(Option<Instant>,Option<Instant>),
               base_depth:u32,
               max_nodes:Option<u64>,
               max_threads:u32,
               nodes_per_leaf_node:u16,
               gamma:u8,
               history:HashSet<(Teban,u64,u64)>,
               transposition_table: &Arc<TT<u64,Score,{1 << 20},4>>
    ) -> Environment<L,S> {
        let abort = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            hasher:hasher,
            search_id:search_id,
            teban:teban,
            limit:limit,
            turn_limit:turn_limit,
            timelimit_margin:timelimit_margin,
            current_limit:Arc::new(RwLock::new(current_limit)),
            base_depth:base_depth,
            max_nodes:max_nodes,
            max_threads:max_threads,
            nodes_per_leaf_node:nodes_per_leaf_node,
            gamma:gamma,
            abort:abort,
            stop:stop,
            quited:quited,
            history:history,
            transposition_table:Arc::clone(transposition_table),
            move_orderer:MoveOrderer::<UnusedQuietSee>::new(base_depth as usize + 2),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a> {
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub rng:&'a mut ThreadRng,
    pub alpha:Score,
    pub beta:Score,
    pub search_offset:usize,
    pub best_score:Score,
    pub m:Option<LegalMove>,
    pub prev_kind:KomaKind,
    pub move_history:&'a mut Vec<Option<(u8,u8)>>,
    pub pv:&'a VecDeque<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub zh:ZobristHash<u64>,
    pub prev_self_ss:Option<i32>,
    pub prev_opponent_ss:Option<i32>,
    pub depth:u32,
    pub current_depth:u32,
    pub current_max_ply:usize,
    pub base_depth:u32,
    pub extend_depth:u32,
    pub extend_check:u32,
    pub extend_threatmate:u32

}
pub struct Root<L,S,M> where L: Logger + Send + 'static,
                             S: InfoSender,
                             M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                PreTrain<f32> + Send + Sync + 'static,
                             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>,
    receiver:Receiver<Result<RootEvaluationResult, ApplicationError>>,
    sender:Sender<Result<RootEvaluationResult, ApplicationError>>,
    thread_pool:ThreadPool
}
pub const TIMELIMIT_MARGIN:u64 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MoveOrder {
    Quiet,
    PawnPromotions,
    Captures,
    ThreatPromotions,
    ThreatCaptures,
    Check
}

pub trait Search<L,S,M>: Sized where L: Logger + Send + 'static,
                                     S: InfoSender,
                                     M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                        PreTrain<f32> + Send + Sync + 'static,
                                     <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError>;
    fn qsearch<'b>(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
               env:&mut Environment<L,S>,
               event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
               zh: &ZobristHash<u64>,
               history:&mut HashSet<(Teban,u64,u64)>,
               mut alpha:Score,beta:Score,depth:usize,mut extend_depth:usize,
               evalutor: &Arc<Evalutor<M>>,rng:&mut ThreadRng) -> Result<Score,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            self.timelimit_reached(env)? || history.contains(&(teban,mk,sk)) {
            let score = Score::Value(evalutor.evalute(teban, state, mc)?);
            return Ok(score);
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        let in_check = Rule::in_check(teban,state);

        if in_check {
            Rule::generate_moves::<Evasions>(teban, state, mc, &mut picker)?;
        } else {
            Rule::generate_moves_by_banmen::<CaptureOrPawnPromotions>(teban,state,&mut picker)?;
        }

        if in_check && picker.len() == 0 {
            return Ok(Score::NEGINFINITE);
        }

        if picker.len() == 0 {
            return Ok(Score::Value(evalutor.evalute(teban,state,mc)?));
        }

        let mvs = picker.filter(|m| m.obtained().is_some()).collect::<Vec<_>>();

        if mvs.len() == 0 {
            return Ok(Score::Value(evalutor.evalute(teban,state,mc)?));
        }

        let (mk,sk) = zh.keys();

        history.insert((teban,mk,sk));

        let stand_pat = Score::Value(evalutor.evalute(teban,state,mc)?);

        if stand_pat >= beta {
            return Ok(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }
        
        let mut bestscore = Score::NEGINFINITE;

        let mut opponent_surrounding_mask = BitBoard::from(OU_SURROUNDING_MASK);

        {
            let p = Rule::ou_square(teban.opposite(),state) as u32;

            let (_,y) = p.square_to_point();

            if y == 0 {
                opponent_surrounding_mask &= OU_SURROUNDING_TOP_MASK;
            } else if y == 8 {
                opponent_surrounding_mask &= OU_SURROUNDING_BOTTOM_MASK;
            }

            if p >= 10 {
                opponent_surrounding_mask = opponent_surrounding_mask << (p - 10) as u128;
            } else {
                opponent_surrounding_mask = opponent_surrounding_mask >> (10 - p) as u128;
            }

            opponent_surrounding_mask = opponent_surrounding_mask << 1;
        }

        for m in mvs {
            /*
            if !in_check && !m.is_nari() && !Rule::is_oute_move(state,teban,m) {
                if let Some(o) = m.obtained() {
                    if calc_see(teban,state,m) < -PIECE_SCORE_MAP[o as usize] / 2 {
                        continue;
                    }
                }
            }
            */
            if let Some(ObtainKind::Ou) = match m {
                LegalMove::To(m) => m.obtained(),
                _ => None
            } {
                history.remove(&(teban,mk,sk));

                return Ok(Score::INFINITE);
            }

            let o = match m {
                LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                _ => None
            };

            let is_pawn_move = match m {
                LegalMove::To(m) if teban == Teban::Sente => {
                    state.get_part().sente_self_board & (1 << (m.src() + 1)) != 0
                },
                LegalMove::To(m) if teban == Teban::Gote => {
                    state.get_part().sente_opponent_board & (1 << (m.src() + 1)) != 0
                },
                _ => false
            };

            let is_nari = match m {
                LegalMove::To(m) => m.is_nari(),
                _ => false
            };

            let zh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

            let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

            let expand = extend_depth > 0 && !Rule::in_check(teban.opposite(),&next) && (
                    m.obtained().is_some() && (opponent_surrounding_mask & (1 << (m.dst() + 1)) != 0)
                );

            let extend_depth = if expand {
                extend_depth - 1
            } else {
                extend_depth
            };

            let score = if expand {
                -self.qsearch_threatmate(teban.opposite(),
                              &next,
                              &nmc,
                              env,
                              event_dispatcher,
                              &zh,
                              history,
                              -beta,
                              -alpha,
                              depth+1,
                              extend_depth,
                              evalutor,
                              rng)?
            } else {
                -self.qsearch(teban.opposite(),
                              &next,
                              &nmc,
                              env,
                              event_dispatcher,
                              &zh,
                              history,
                              -beta,
                              -alpha,
                              depth+1,
                              extend_depth,
                              evalutor,
                              rng)?
            };
            /*

            let score = -self.qsearch(teban.opposite(),
                                              &next,
                                              &nmc,
                                              env,
                                              event_dispatcher,
                                              &zh,
                                              history,
                                              -beta,
                                              -alpha,
                                              depth+1,
                                              extend_depth,
                                              evalutor,
                                              rng)?;
            */
            if score >= beta {
                return Ok(score);
            }

            if score > bestscore {
                bestscore = score;
            }

            if score > alpha {
                alpha = score;
            }

            if self.timelimit_reached(env)? {
                break;
            }
        }

        history.remove(&(teban,mk,sk));

        if bestscore == Score::NEGINFINITE {
            Ok(stand_pat)
        } else {
            Ok(bestscore)
        }
    }
    fn qsearch_threatmate<'b>(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
                   env:&mut Environment<L,S>,
                   event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                   zh: &ZobristHash<u64>,
                   history:&mut HashSet<(Teban,u64,u64)>,
                   mut alpha:Score,beta:Score,depth:usize,mut extend_depth:usize,
                   evalutor: &Arc<Evalutor<M>>,rng:&mut ThreadRng) -> Result<Score,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            self.timelimit_reached(env)? || history.contains(&(teban,mk,sk)) {
            let score = Score::Value(evalutor.evalute(teban, state, mc)?);
            return Ok(score);
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        Rule::generate_moves_by_banmen::<NonEvasions>(teban,state,&mut picker)?;

        let mut opponent_surrounding_mask = BitBoard::from(OU_SURROUNDING_MASK);

        {
            let p = Rule::ou_square(teban.opposite(),state) as u32;

            let (_,y) = p.square_to_point();

            if y == 0 {
                opponent_surrounding_mask &= OU_SURROUNDING_TOP_MASK;
            } else if y == 8 {
                opponent_surrounding_mask &= OU_SURROUNDING_BOTTOM_MASK;
            }

            if p >= 10 {
                opponent_surrounding_mask = opponent_surrounding_mask << (p - 10) as u128;
            } else {
                opponent_surrounding_mask = opponent_surrounding_mask >> (10 - p) as u128;
            }

            opponent_surrounding_mask = opponent_surrounding_mask << 1;
        }

        let mut mvs = (&mut picker).map(|m| {
            let is_pawn_move = match m {
                LegalMove::To(m) if teban == Teban::Sente => {
                    state.get_part().sente_self_board & (1 << (m.src() + 1)) != 0
                },
                LegalMove::To(m) if teban == Teban::Gote => {
                    state.get_part().sente_opponent_board & (1 << (m.src() + 1)) != 0
                },
                _ => false
            };

            let is_nari = match m {
                LegalMove::To(m) => m.is_nari(),
                _ => false
            };

            if Rule::is_oute_move(state,teban,m) {
                (MoveOrder::Check, m)
            } else if m.obtained().is_some() && (
                opponent_surrounding_mask & (1 << (m.dst() + 1)) != 0
            ) {
                (MoveOrder::ThreatCaptures,m)
            } else if is_pawn_move && is_nari && (
                opponent_surrounding_mask & (1 << (m.dst() + 1)) != 0
            ) {
                (MoveOrder::ThreatPromotions, m)
            } else if m.obtained().is_some() {
                (MoveOrder::Captures, m)
            } else if is_pawn_move && is_nari {
                (MoveOrder::PawnPromotions, m)
            } else {
                (MoveOrder::Quiet,m)
            }
        }).collect::<Vec<(MoveOrder,LegalMove)>>();

        mvs.sort_by(|a,b| b.0.cmp(&a.0));

        if mvs.len() == 0 {
            return Ok(Score::Value(evalutor.evalute(teban,state,mc)?));
        }

        let (mk,sk) = zh.keys();

        history.insert((teban,mk,sk));

        let stand_pat = Score::Value(evalutor.evalute(teban,state,mc)?);

        if stand_pat >= beta {
            return Ok(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut bestscore = Score::NEGINFINITE;

        for (mo,m) in mvs {
            if let Some(ObtainKind::Ou) = match m {
                LegalMove::To(m) => m.obtained(),
                _ => None
            } {
                history.remove(&(teban,mk,sk));

                return Ok(Score::INFINITE);
            }

            let o = match m {
                LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                _ => None
            };

            let zh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

            let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

            let expand = match mo {
                MoveOrder::ThreatCaptures => {
                    extend_depth > 0 && !Rule::in_check(teban.opposite(),&next)
                },
                _ => false
            };

            let extend_depth = if expand {
                extend_depth - 1
            } else {
                extend_depth
            };

            let score = if expand {
                -self.qsearch_threatmate(teban.opposite(),
                              &next,
                              &nmc,
                              env,
                              event_dispatcher,
                              &zh,
                              history,
                              -beta,
                              -alpha,
                              depth+1,
                              extend_depth,
                              evalutor,
                              rng)?
            } else {
                -self.qsearch(teban.opposite(),
                              &next,
                              &nmc,
                              env,
                              event_dispatcher,
                              &zh,
                              history,
                              -beta,
                              -alpha,
                              depth+1,
                              extend_depth,
                              evalutor,
                              rng)?
            };

            if score >= beta {
                return Ok(score);
            }

            if score > bestscore {
                bestscore = score;
            }

            if score > alpha {
                alpha = score;
            }

            if self.timelimit_reached(env)? {
                break;
            }
        }

        history.remove(&(teban,mk,sk));

        Ok(bestscore)
    }

    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> Result<bool,ApplicationError> {
        let mut reached;
        let timelimit_margin = env.timelimit_margin;

        match *env.current_limit.read() {
            (current_turn_lmit,current_limit) => {
                reached = current_turn_lmit.map(|l| l - Instant::now() <= Duration::from_millis(timelimit_margin)).unwrap_or(false);
                reached = reached || current_limit.map(|l| l - Instant::now() <= Duration::from_millis(timelimit_margin)).unwrap_or(false);
            }
        }

        Ok(reached)
    }

    fn is_important_move(&self, env:&mut Environment<L,S>,
                depth:u32,
                current_depth:u32,
                teban: Teban,
                state: &State,
                m:LegalMove,
                pv:Option<&LegalMove>)
        -> Result<bool,ApplicationError> {
        Ok(m.obtained().is_some() ||
            m.is_nari() ||
            pv.map(|pm| pm == &m).unwrap_or(false) ||
            env.move_orderer.is_killer(current_depth,m)? ||
            Rule::is_oute_move(state,teban,m)
        )
    }
    fn calc_lmr(&self, env:&mut Environment<L,S>,
                       depth:u32,
                       current_depth:u32,
                       index:usize,
                       teban: Teban,
                       state: &State,
                       m:LegalMove,
                       tt_move:Option<&LegalMove>,
                       pv:Option<&LegalMove>,
                       prev_move:Option<&LegalMove>,
                       prev_kind:KomaKind,
                       see:i32,
                       zh:&ZobristHash<u64>,
                       prev_eval: Option<i32>,
                       static_eval: i32) -> Result<u32,ApplicationError> {
        if depth < 3 ||
            Rule::in_check(teban,state) ||
            tt_move.map(|&tm| tm == m).unwrap_or(false) ||
            self.is_important_move(env,depth,current_depth,
                                   teban,state,m,pv)? {
            Ok(0)
        } else if index <= 1 {
            Ok(0)
        } else {
            let r = LMR_TABLE[depth as usize][index.min(63)];

            let h = env.move_orderer.look_up_history(teban,state,m)?;

            if h > depth as i64 * 6 {
                Ok(r.saturating_sub(1) as u32)
            } else if h < -(depth as i64) * 6 {
                Ok(r.saturating_add(1) as u32)
            } else {
                Ok(r as u32)
            }
        }
    }

    fn in_danger(&self, teban: Teban, state:&State, m: LegalMove) -> bool {
        /*
        match teban {
            Teban::Sente => {
                let p = Rule::ou_square(Teban::Sente, state) as u32;

                let possible_move_mask = Rule::gen_candidate_bits(teban,state.get_part().sente_self_board,p,KomaKind::SOu);

                let possible_move_count = possible_move_mask.bitcount();

                let mut danger_mask = BitBoard::from(OU_SURROUNDING_MASK);

                let (_,y) = p.square_to_point();

                if y == 0 {
                    danger_mask &= OU_SURROUNDING_TOP_MASK;
                } else if y == 8 {
                    danger_mask &= OU_SURROUNDING_BOTTOM_MASK;
                }

                let possible_move_count = possible_move_count - Rule::sente_danger_count(
                    state.get_part(),danger_mask,p as i32 - 10,
                    BitBoard::from(POSSIBLE_OU_CAPTURES_MASK_OF_GOTE), p as i32 - 20
                );

                if p >= 10 {
                    danger_mask = danger_mask << (p - 10) as u128;
                } else {
                    danger_mask = danger_mask >> (10 - p) as u128;
                }

                /*
                possible_move_mask.iter().fold(0i32,|acc,p| {
                    acc + Rule::control_count(teban.opposite(),state.get_part(),p as Square) as i32 -
                          Rule::control_count(teban,state.get_part(),p as Square) as i32
                }) >= 2 ||
                    (possible_move_count <= 2 &&
                        (m.obtained() == Some(ObtainKind::Kin) ||
                         m.obtained() == Some(ObtainKind::Gin)
                        ) &&
                        (danger_mask << 1) & (1 << (m.dst() + 1)) != 0
                    )

                 */
            },
            Teban::Gote => {
                let p = Rule::ou_square(Teban::Gote, state) as u32;

                let possible_move_mask = Rule::gen_candidate_bits(teban,state.get_part().gote_self_board,p,KomaKind::GOu).reverse();

                let possible_move_count = possible_move_mask.bitcount();

                let mut danger_mask = BitBoard::from(OU_SURROUNDING_MASK);

                let (_,y) = p.square_to_point();

                if y == 0 {
                    danger_mask &= OU_SURROUNDING_TOP_MASK;
                } else if y == 8 {
                    danger_mask &= OU_SURROUNDING_BOTTOM_MASK;
                }

                let possible_move_count = possible_move_count - Rule::gote_danger_count(
                    state.get_part(),danger_mask,p as i32 - 10,
                    BitBoard::from(POSSIBLE_OU_CAPTURES_MASK_OF_SENTE), p as i32 - 19
                );

                if p >= 10 {
                    danger_mask = danger_mask << (p - 10) as u128;
                } else {
                    danger_mask = danger_mask >> (10 - p) as u128;
                }

                /*
                possible_move_mask.iter().fold(0i32,|acc,p| {
                    acc + Rule::control_count(teban.opposite(),state.get_part(),p as Square) as i32 -
                          Rule::control_count(teban,state.get_part(),p as Square) as i32
                }) >= 2 ||
                    (possible_move_count <= 2 &&
                        (m.obtained() == Some(ObtainKind::Kin) ||
                            m.obtained() == Some(ObtainKind::Gin)
                        ) &&
                        (danger_mask << 1) & (1 << (m.dst() + 1)) != 0
                    )

                 */
            }
        }

         */
        false
    }

    fn is_threat(&self, teban: Teban, state:&State, m:LegalMove) -> bool {
        /*
        const ZONE_LARGE:u128 = 0b000011111_000011111_000011011_000011111_000011111;

        let ps = state.get_part();

        match teban {
            Teban::Sente => {
                let ou_square = Rule::ou_square(Teban::Gote,state) as u32;

                let (_,oy) = ou_square.square_to_point();

                let mask = ps.sente_gin_board |
                           ps.sente_kin_board | ps.sente_nari_board |
                           ps.sente_kaku_board | ps.sente_hisha_board;

                match m {
                    LegalMove::To(m) => {
                        let mut zone_mask = ZONE_LARGE;

                        if oy == 0 {
                            zone_mask &= 0b111111100_111111100_111111100_111111100_111111100;
                        } else if oy == 1 {
                            zone_mask &= 0b111111110_111111110_111111110_111111110_111111110;
                        } else if oy == 8 {
                            zone_mask &= 0b000000111_000000111_000000111_000000111_000000111;
                        } else if oy == 7 {
                            zone_mask &= 0b000001111_000001111_000001111_000001111_000001111;
                        }

                        let zone_mask = if m.dst() >= 20 {
                            zone_mask << m.dst() - 20
                        } else {
                            zone_mask << 20 - m.dst()
                        };

                        (m.obtained().is_some() &&
                         Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0 &&
                         (zone_mask << 1) & (1 << (m.dst() + 1)) != 0) ||
                        (BitBoard::from(1 << (m.src() + 1)) & BitBoard::from(zone_mask << 1) == 0 &&
                            BitBoard::from(1 << (m.dst() + 1)) & BitBoard::from(zone_mask << 1) != 0 &&
                            BitBoard::from(1 << (m.dst() + 1)) & mask != 0 &&
                            Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0
                        )
                    },
                    LegalMove::Put(m) => {
                        let mut zone_mask = ZONE_LARGE;

                        if oy == 0 {
                            zone_mask &= 0b111111100_111111100_111111100_111111100_111111100;
                        } else if oy == 1 {
                            zone_mask &= 0b111111110_111111110_111111110_111111110_111111110;
                        } else if oy == 8 {
                            zone_mask &= 0b000000111_000000111_000000111_000000111_000000111;
                        } else if oy == 7 {
                            zone_mask &= 0b000001111_000001111_000001111_000001111_000001111;
                        }

                        let zone_mask = if m.dst() >= 20 {
                            zone_mask << m.dst() - 20
                        } else {
                            zone_mask << 20 - m.dst()
                        };

                        match m.kind() {
                            MochigomaKind::Gin | MochigomaKind::Kin | MochigomaKind::Kaku | MochigomaKind::Hisha => {
                                BitBoard::from(1 << (m.dst() + 1)) & BitBoard::from(zone_mask << 1) != 0 &&
                                Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0
                            },
                            _ => false
                        }
                    }
                }
            },
            Teban::Gote => {
                let ou_square = Rule::ou_square(Teban::Sente,state) as u32;

                let (_,oy) = ou_square.square_to_point();

                let mask = ps.gote_gin_board |
                           ps.gote_kin_board | ps.gote_nari_board |
                           ps.gote_kaku_board | ps.gote_hisha_board;
                match m {
                    LegalMove::To(m) => {
                        let mut zone_mask = ZONE_LARGE;

                        if oy == 0 {
                            zone_mask &= 0b111111100_111111100_111111100_111111100_111111100;
                        } else if oy == 1 {
                            zone_mask &= 0b111111110_111111110_111111110_111111110_111111110;
                        } else if oy == 8 {
                            zone_mask &= 0b000000111_000000111_000000111_000000111_000000111;
                        } else if oy == 7 {
                            zone_mask &= 0b000001111_000001111_000001111_000001111_000001111;
                        }

                        let zone_mask = if m.dst() >= 20 {
                            zone_mask << m.dst() - 20
                        } else {
                            zone_mask << 20 - m.dst()
                        };

                        (m.obtained().is_some() &&
                          Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0 &&
                          (zone_mask << 1) & (1 << (m.dst() + 1)) != 0) ||
                        (BitBoard::from(1 << (m.src() + 1)) & BitBoard::from(zone_mask << 1) == 0 &&
                            BitBoard::from(1 << (m.dst() + 1)) & BitBoard::from(zone_mask << 1) != 0 &&
                            BitBoard::from(1 << (m.dst() + 1)) & mask != 0 &&
                            Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0
                        )
                    },
                    LegalMove::Put(m) => {
                        let mut zone_mask = ZONE_LARGE;

                        if oy == 0 {
                            zone_mask &= 0b111111100_111111100_111111100_111111100_111111100;
                        } else if oy == 1 {
                            zone_mask &= 0b111111110_111111110_111111110_111111110_111111110;
                        } else if oy == 8 {
                            zone_mask &= 0b000000111_000000111_000000111_000000111_000000111;
                        } else if oy == 7 {
                            zone_mask &= 0b000001111_000001111_000001111_000001111_000001111;
                        }

                        let zone_mask = if m.dst() >= 20 {
                            zone_mask << m.dst() - 20
                        } else {
                            zone_mask << 20 - m.dst()
                        };

                        match m.kind() {
                            MochigomaKind::Gin | MochigomaKind::Kin | MochigomaKind::Kaku | MochigomaKind::Hisha => {
                                BitBoard::from(1 << (m.dst() + 1)) & BitBoard::from(zone_mask << 1) != 0 &&
                                Rule::control_count(teban.opposite(),ps,m.dst() as Square) == 0
                            },
                            _ => false
                        }
                    }
                }
            }
        }

         */
        false
    }
    fn send_info(&self, env:&mut Environment<L,S>,
                 depth:u32, seldepth:u32, pv:&VecDeque<LegalMove>, score:&Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            Score::INFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Plus)))
            },
            Score::NEGINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Minus)))
            },
            Score::MAYBEINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(1000000)))
            },
            Score::MAYBENEGINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(-1000000)))
            },
            Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(*s as i64)))
            }
        }

        commands.push(UsiInfoSubCommand::Depth(depth));

        if depth < seldepth {
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }

        Ok(env.info_sender.send(commands)?)
    }

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands)?)
    }
}
pub trait PartialSearch<L,S,M>: Sized where L: Logger + Send + 'static,
                                     S: InfoSender,
                                     M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                     PreTrain<f32> + Send + Sync + 'static,
                                     <M as PreTrain<f32>>::OutStack: Send + Sync + 'static
{
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      evalutor: &Arc<Evalutor<M>>,
                      mvs:&Vec<LegalMove>) -> Result<EvaluationResult, ApplicationError>;
}
impl<L,S,M> Root<L,S,M> where L: Logger + Send + 'static,
                              S: InfoSender,
                              M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                 PreTrain<f32> + Send + Sync + 'static,
                              <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn new(thread_pool:ThreadPool) -> Root<L,S,M> {
        let(s,r) = mpsc::channel();

        Root {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            m:PhantomData::<M>,
            receiver:r,
            sender:s,
            thread_pool:thread_pool
        }
    }

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                                         stop:&Arc<AtomicBool>,
                                         quited:&Arc<AtomicBool>,
                                         teban:Teban,
                                         limit:&'a Option<UsiGoTimeLimit>,
                                         turn_limit:&'a Option<u32>,
                                         current_limit:&Arc<RwLock<(Option<Instant>,Option<Instant>)>>)
                                         -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler);

        {
            let stop = Arc::clone(stop);

            event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
                match e {
                    &UserEvent::Stop => {
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        {
            let stop = Arc::clone(stop);
            let quited = Arc::clone(quited);

            event_dispatcher.add_handler(UserEventKind::Quit, move |_,e| {
                match e {
                    &UserEvent::Quit => {
                        quited.store(true,atomic::Ordering::Release);
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        {
            let current_limit = Arc::clone(current_limit);
            let limit = limit.clone();
            let teban = teban.clone();

            event_dispatcher.add_handler(UserEventKind::PonderHit, move |_,e| {
                match e {
                    &UserEvent::PonderHit(think_start_time) => {
                        {
                            let mut l = current_limit.write();
                            l.0 = turn_limit.map(move |l| think_start_time + Duration::from_millis(l as u64));
                            let teban = teban.clone();
                            l.1 = limit.and_then(move |l| l.to_instant(teban,think_start_time));
                        }
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        event_dispatcher
    }

    fn start_thread<'a,'b>(&self,
                           pending_results:Arc<AtomicUsize>,
                           env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           mvs: Arc<Vec<LegalMove>>,
                           search_offset:usize,
                           pv:VecDeque<LegalMove>,
                           //prev_score:Score,
                           evalutor: &Arc<Evalutor<M>>,move_orderer: MoveOrderer<UnusedQuietSee>) {
        let sender = self.sender.clone();
        let teban = gs.teban;
        let state = Arc::clone(&gs.state);
        let mut env = env.clone();
        let mut move_orderer = move_orderer;
        move_orderer.on_start_search(gs.depth);
        env.move_orderer = move_orderer;

        let evalutor = Arc::clone(&evalutor);
        let mc = Arc::clone(&gs.mc);
        let zh = gs.zh.clone();
        let depth = gs.depth;
        let current_depth = 0;
        let current_max_ply = gs.current_max_ply;
        let base_depth = gs.base_depth;
        let extend_depth = gs.extend_depth;
        let best_score = gs.best_score;

        self.thread_pool.spawn(move || {
            let mut rng = rand::thread_rng();

            let r = /*if let Score::Value(prev_score) = prev_score {
                let delta = Self::compute_aspiration_window_delta(depth);

                let mut alpha = Score::Value(prev_score - delta);
                let mut beta = Score::Value(prev_score + delta);

                let mut gs = GameState {
                    teban: teban,
                    state: &state,
                    alpha: alpha,
                    beta: beta,
                    search_offset: search_offset,
                    best_score: best_score,
                    m: None,
                    prev_kind: KomaKind::Blank,
                    pv:&pv,
                    mc: &mc,
                    zh: zh,
                    depth: depth,
                    current_depth: current_depth,
                    current_max_ply: current_max_ply,
                    base_depth: base_depth,
                    extend_depth: extend_depth,
                    extend_check: 1,
                    extend_threatmate: 1,
                    rng:&mut rng
                };

                let strategy = Inter::new();

                let mut r = strategy.search(&mut env, &mut gs, &evalutor, &mvs);

                r = match r {
                    Ok(EvaluationResult::Immediate(score,rmvs,zh)) => {
                        if score <= alpha {
                            gs.alpha = Score::NEGINFINITE;
                            strategy.search(&mut env, &mut gs, &evalutor, &mvs)
                        } else if score >= beta {
                            gs.beta = Score::INFINITE;
                            strategy.search(&mut env, &mut gs, &evalutor, &mvs)
                        } else {
                            Ok(EvaluationResult::Immediate(score,rmvs,zh))
                        }
                    },
                    r => {
                        r
                    }
                };

                r
            } else */{
                let mut gs = GameState {
                    teban: teban,
                    state: &state,
                    alpha: Score::NEGINFINITE,
                    beta: Score::INFINITE,
                    search_offset: search_offset,
                    best_score: best_score,
                    m: None,
                    prev_kind: KomaKind::Blank,
                    move_history: &mut Vec::new(),
                    pv:&pv,
                    mc: &mc,
                    zh: zh,
                    prev_self_ss: None,
                    prev_opponent_ss: None,
                    depth: depth,
                    current_depth: current_depth,
                    current_max_ply: current_max_ply,
                    base_depth: base_depth,
                    extend_depth: extend_depth,
                    extend_check: 1,
                    extend_threatmate: 1,
                    rng:&mut rng
                };

                let strategy = Inter::new();

                strategy.search(&mut env, &mut gs, &evalutor, &mvs)
            };

            pending_results.fetch_add(1, atomic::Ordering::SeqCst);

            match r {
                Ok(EvaluationResult::Immediate(score,mvs,zh)) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Immediate(score,mvs,zh,depth,search_offset,env.move_orderer)));
                },
                Ok(EvaluationResult::NodeLimits) => {
                    let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                },
                Ok(EvaluationResult::Timeout) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                },
                Ok(EvaluationResult::Repetition) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Repetition));
                },
                Ok(EvaluationResult::Stop) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Stop));
                },
                Err(e) => {
                    let _ = sender.send(Err(e));
                }
            }
        });
    }

    fn termination(&self,env:&mut Environment<L,S>,mut busy_threads:u32, pending_results:&Arc<AtomicUsize>) -> Result<(),ApplicationError> {
        env.abort.store(true,Ordering::Release);

        let mut last_error = None;

        while busy_threads > 0 {
            if let Err(e) = self.receiver.recv().map_err(|e| ApplicationError::from(e))? {
                last_error = Some(e);
            }

            busy_threads -= 1;
            pending_results.fetch_sub(1, Ordering::SeqCst);
        }

        env.info_sender.flush()?;

        match last_error {
            Some(e) => Err(e),
            None => Ok(())
        }
    }

    pub fn compute_aspiration_window_delta(depth:u32) -> i32 {
        let a = 12.0f64;
        let b = 25.0f64;

        let delta = a * (depth as f64).sqrt() + b;

        (delta * 1.4) as i32
    }
}
impl<L,S,M> Search<L,S,M> for Root<L,S,M> where L: Logger + Send + 'static,
                                            S: InfoSender,
                                            M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                               PreTrain<f32> + Send + Sync + 'static,
                                            <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     _:&mut UserEventDispatcher<'b,Root<L,S,M>,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError> {
        env.search_id.fetch_add(1, Ordering::Release);

        let base_depth = gs.depth.min(env.base_depth);
        let mut current_depth = 1;
        let max_depth = base_depth as usize + 2;
        let mut already_started = vec![false;max_depth+1];
        let mut workings = vec![0;max_depth+1];
        let pending_results = Arc::new(AtomicUsize::new(0));

        let mut move_orderer_quque = VecDeque::<MoveOrderer<UnusedQuietSee>>::new();
        let mut busy_threads = 0;
        let mut remaining_threads = 0;
        let mut last_depth = false;
        let mut search_done_threads = vec![0;max_depth+1];
        let mut result = vec![None;max_depth+1];
        let mut decided_depth = 0;
        let mut extend_depth = 0;

        let mut picker = RandomPicker::new(Prng::new(gs.rng.gen()));

        let mut mvs = Vec::new();
        let nodes_per_leaf_node = env.nodes_per_leaf_node as u128;
        let mut search_space:u128 = mvs.len() as u128 / env.max_threads as u128;
        let mut leafnodes_seacch_space:u128 = mvs.len() as u128;
        let gamma = env.gamma;
        let mut prev_score = Score::NEGINFINITE;

        let mut evasions_count;

        if Rule::in_check(gs.teban,&gs.state) {
            Rule::generate_moves::<Evasions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            evasions_count = Some(picker.len());
            mvs = picker.collect::<Vec<LegalMove>>();
        } else {
            {
                Rule::generate_moves::<CaptureOrPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
                let mut v = (&mut picker).collect::<Vec<LegalMove>>();
                mvs.append(&mut v);
            }

            {
                Rule::generate_moves::<QuietsWithoutPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
                let mut v = (&mut picker).collect::<Vec<LegalMove>>();
                mvs.append(&mut v);
            }

            evasions_count = None;
        };

        let mvs = Arc::new(mvs);

        env.abort.store(false,Ordering::Release);

        loop {
            if busy_threads > 0 && (busy_threads == env.max_threads || remaining_threads > 0 || pending_results.load(Ordering::SeqCst) > 0) {
                while pending_results.load(Ordering::Acquire) > 0 {
                    match self.receiver.recv().map_err(|e| ApplicationError::from(e))? {
                        Ok(RootEvaluationResult::Immediate(s, mvs, zh, depth,search_offset, move_orderer)) => {
                            busy_threads -= 1;
                            workings[depth as usize] -= 1;
                            search_done_threads[depth as usize] += 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            if let Err(e) = env.info_sender.flush() {
                                let _ = env.on_error_handler.lock().map(|h| h.call(&e));
                            }

                            match result[depth as usize] {
                                Some(EvaluationResult::Immediate(bs, _, _)) if (depth > decided_depth && s >= bs)
                                    || search_offset == 0 => {

                                    if search_offset == 0 {
                                        prev_score = s;
                                    }

                                    result[depth as usize] = Some(EvaluationResult::Immediate(s, mvs, zh));
                                },
                                None => {
                                    result[depth as usize] = Some(EvaluationResult::Immediate(s, mvs, zh));
                                },
                                _ => ()
                            }

                            // If you're examining the root node with search_offset=0,
                            // you should have already exhausted all legal moves at this depth.
                            // Therefore, once the result of exploring nodes up to max_depth is returned,
                            // it can be considered finalized.
                            if search_offset == 0 && depth == base_depth {
                                decided_depth = depth;

                                self.termination(env, busy_threads, &pending_results)?;

                                return Ok(result[decided_depth.max(1) as usize].take().unwrap_or(EvaluationResult::Timeout));
                            // When search_offset=0,
                            // all legal moves at the root node should have been examined at this search depth,
                            // so it is considered searched.
                            } else if search_offset == 0 {
                                extend_depth = 2;

                                if depth > decided_depth {
                                    decided_depth = depth;
                                }

                                if depth >= current_depth {
                                    while current_depth <= depth {
                                        current_depth += 1;

                                        leafnodes_seacch_space = leafnodes_seacch_space * nodes_per_leaf_node;

                                        search_space = search_space + leafnodes_seacch_space;
                                        search_space = search_space * gamma as u128 / 100;
                                    }
                                }
                            }

                            move_orderer_quque.push_back(move_orderer);

                            self.send_message(env, format!("search_id = {}, decided_depth = {}, {}",
                                                           env.search_id.load(Ordering::Acquire), decided_depth, result[decided_depth as usize].is_some()
                            ).as_str())?;
                        },
                        Ok(RootEvaluationResult::NodeLimits) => {
                            busy_threads -= 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            self.termination(env, busy_threads, &pending_results)?;

                            return Ok(result[decided_depth.max(1) as usize].take().unwrap_or(EvaluationResult::NodeLimits));
                        },
                        Ok(RootEvaluationResult::Timeout) => {
                            busy_threads -= 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            self.termination(env, busy_threads, &pending_results)?;

                            return Ok(result[decided_depth.max(1) as usize].take().unwrap_or(EvaluationResult::Timeout));
                        },
                        Ok(RootEvaluationResult::Stop) => {
                            busy_threads -= 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            self.termination(env, busy_threads, &pending_results)?;

                            return Ok(result[decided_depth.max(1) as usize].take().unwrap_or(EvaluationResult::Stop));
                        },
                        Ok(RootEvaluationResult::Repetition) => {
                            busy_threads -= 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            self.termination(env, busy_threads, &pending_results)?;

                            return Err(ApplicationError::LogicError(String::from(
                                "A Repetition was returned at the root node."
                            )));
                        },
                        Err(e) => {
                            busy_threads -= 1;

                            pending_results.fetch_sub(1, Ordering::SeqCst);

                            self.termination(env, busy_threads, &pending_results)?;

                            return Err(e);
                        }
                    }
                }
            } else if busy_threads == 0 && (remaining_threads > 0 || last_depth) {
                return Ok(result[decided_depth.max(1) as usize].take().unwrap_or(EvaluationResult::Timeout));
            } else {
                if env.nodes.load(Ordering::Acquire) as u128 >= search_space {
                    current_depth += 1;
                    leafnodes_seacch_space = leafnodes_seacch_space * nodes_per_leaf_node;

                    search_space = search_space + leafnodes_seacch_space;
                    search_space = search_space * gamma as u128 / 100;
                }

                gs.depth = current_depth;
                gs.base_depth = current_depth;
                gs.current_max_ply = current_depth as usize;
                gs.extend_depth = extend_depth;

                if current_depth <= base_depth {
                    let search_offset = if !already_started[current_depth as usize] {
                        already_started[current_depth as usize] = true;
                        0
                    } else {
                  gs.rng.gen::<usize>() % (mvs.len() / 4).max(1)
                    };

                    workings[current_depth as usize] += 1;

                    let mvs = Arc::clone(&mvs);

                    let pending_results = Arc::clone(&pending_results);

                    let pv = if decided_depth == 0 || search_offset != 0 {
                        VecDeque::new()
                    } else if let Some(EvaluationResult::Immediate(_, ref mvs, _)) = result[decided_depth as usize].as_ref() {
                        mvs.clone()
                    } else {
                        VecDeque::new()
                    };

                    self.start_thread(pending_results,
                                      env,gs,mvs,
                                      search_offset,
                                      pv,
                                      //prev_score,
                                      evalutor,
                                      move_orderer_quque
                                          .pop_front()
                                          .unwrap_or(MoveOrderer::new(max_depth)));

                    busy_threads += 1;
                }

                if current_depth >= base_depth {
                    remaining_threads = busy_threads;
                    last_depth = true;
                }
            }
        }
    }
}
pub struct Recursive<L,S,M> where L: Logger + Send + 'static,
                                  S: InfoSender,
                                  M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                     PreTrain<f32> + Send + Sync + 'static,
                                  <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>
}
impl<L,S,M> Recursive<L,S,M> where L: Logger + Send + 'static,
                                   S: InfoSender,
                                   M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                      PreTrain<f32> + Send + Sync + 'static,
                                   <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn new() -> Recursive<L,S,M> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            m:PhantomData::<M>
        }
    }

    pub fn is_obtained_ou(&self,m:LegalMove) -> Result<bool,ApplicationError> {
        Ok(Some(ObtainKind::Ou) == m.obtained())
    }

    pub fn search_child_node<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                     m:LegalMove,pv:&VecDeque<LegalMove>,
                                     alpha:Score,
                                     depth:u32,
                                     static_eval: i32,
                                     evasions_count: Option<usize>,
                                     event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        let o = match m {
            LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
            _ => None
        };

        let mut depth = depth;
        let mut extend_depth = gs.extend_depth;
        let mut extend_check = gs.extend_check;
        let mut extend_threatmate = gs.extend_threatmate;

        let piece_index = env.move_orderer.calc_piece_index(gs.teban,gs.state,m)?;

        let zh = gs.zh.updated(&env.hasher, gs.teban, gs.state.get_banmen(), gs.mc, m.to_applied_move(), &o);

        let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

        match next {
            (state, mc, _) => {
                if extend_depth > 0 {
                    if extend_check > 0 && Rule::in_check(gs.teban.opposite(),&state) {
                        depth += 1;
                        extend_depth -= 1;
                        extend_check -= 1;
                    } else if extend_threatmate > 0 &&
                        (self.in_danger(gs.teban.opposite(),&state, m) || self.is_threat(gs.teban,&state,m)) {
                        depth += 1;
                        extend_depth -= 1;
                        extend_threatmate -= 1;
                    }
                }

                let state = Arc::new(state);

                let mc = Arc::new(mc);

                let prev_kind = match m {
                    LegalMove::To(mv) => {
                        let (x,y) = mv.src().square_to_point();

                        gs.state.get_banmen().0[y as usize][x as usize]
                    },
                    _ => KomaKind::Blank
                };

                gs.move_history.push(Some((piece_index as u8,m.dst() as u8)));

                let mut gs = GameState {
                    teban: gs.teban.opposite(),
                    state: &state,
                    rng: gs.rng,
                    alpha: -gs.beta,
                    beta: -alpha,
                    search_offset: 0,
                    best_score: gs.best_score,
                    m: Some(m),
                    prev_kind: prev_kind,
                    pv:pv,
                    move_history: gs.move_history,
                    mc: &mc,
                    zh: zh.clone(),
                    prev_self_ss: gs.prev_opponent_ss,
                    prev_opponent_ss: Some(static_eval),
                    depth: depth - 1,
                    current_depth: gs.current_depth + 1,
                    current_max_ply: gs.current_max_ply,
                    base_depth: gs.base_depth,
                    extend_depth: extend_depth,
                    extend_check: extend_check,
                    extend_threatmate: extend_threatmate,
                };

                let strategy = Recursive::new();

                let r = strategy.search(env, &mut gs, event_dispatcher, evalutor);

                gs.move_history.pop();

                r
            }
        }
    }

    pub fn search_null_move<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                   alpha:Score,
                                   beta:Score,
                                   depth:u32,
                                   static_eval: i32,
                                   event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                   evalutor: &Arc<Evalutor<M>>)
        -> Result<EvaluationResult, ApplicationError> {
        let state = gs.state;
        let mc = gs.mc;
        let zh = gs.zh.teban_fliped();

        gs.move_history.push(None);

        let mut gs = GameState {
            teban: gs.teban.opposite(),
            state: state,
            rng: gs.rng,
            alpha: alpha,
            beta: beta,
            search_offset: 0,
            best_score: gs.best_score,
            m: None,
            prev_kind: Blank,
            pv:&VecDeque::new(),
            move_history: gs.move_history,
            mc: &mc,
            zh: zh.clone(),
            prev_self_ss: gs.prev_opponent_ss,
            prev_opponent_ss: Some(static_eval),
            depth: depth - 1,
            current_depth: gs.current_depth + 1,
            current_max_ply: gs.current_max_ply,
            base_depth: gs.base_depth,
            extend_depth: gs.extend_depth,
            extend_check: gs.extend_check,
            extend_threatmate: gs.extend_threatmate,
        };

        let strategy = Recursive::new();

        strategy.search(env, &mut gs, event_dispatcher, evalutor)
    }
}
impl<L,S,M> Search<L,S,M> for Recursive<L,S,M> where L: Logger + Send + 'static,
                                                     S: InfoSender,
                                                     M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                                        PreTrain<f32> + Send + Sync + 'static,
                                                     <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                      evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        env.nodes.fetch_add(1,Ordering::Release);

        if env.max_nodes.map(|n| {
            env.nodes.load(Ordering::Acquire) >= n
        }).unwrap_or(false) {
            return Ok(EvaluationResult::NodeLimits);
        }

        let (mk,sk) = gs.zh.keys();

        if env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Stop);
        }

        if self.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        let prev_eval = gs.prev_self_ss;

        let static_eval = 0;//evalutor.evalute(gs.teban,&gs.state,&gs.mc)?;

        if self.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        if env.history.contains(&(gs.teban,mk,sk)) {
            return Ok(EvaluationResult::Repetition);
        }

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Stop);
        }

        if self.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        if let Some(prev_move) = gs.m.clone() {
            let r = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: d,
                            score: s,
                            beta: _,
                            alpha: _,
                            bound,
                            best_move: _
                        }) = r {

                if (bound == Bound::Exact && d as u32 >= gs.depth) ||
                   (bound == Bound::LowerBound && d as u32 >= gs.depth && s >= gs.beta) ||
                   (bound == Bound::UpperBound && d as u32 >= gs.depth && s <= gs.alpha) {
                    let mut mvs = VecDeque::new();

                    mvs.push_front(prev_move);

                    return Ok(EvaluationResult::Immediate(s,mvs,gs.zh.clone()));
                }
            }
        }

        let prev_move = gs.m.clone();

        if Rule::in_check(gs.teban.opposite(),&gs.state) {
            if let Some(m) = prev_move.clone() {
                env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE,gs.beta,gs.alpha,Bound::Exact,None);

                let mut mvs = VecDeque::new();

                mvs.push_front(m);

                return Ok(EvaluationResult::Immediate(Score::INFINITE, mvs, gs.zh.clone()));
            }
        }

        /*
        {
            const R:u32 = 2;

            let ou_surrounding_threats_count = match gs.teban {
                Teban::Sente => Rule::sente_ou_surrounding_threats_count(&gs.state.get_part()),
                Teban::Gote => Rule::gote_ou_surrounding_threats_count(&gs.state.get_part())
            };

            let mcount = match &gs.mc.deref() {
                &MochigomaCollections::Empty => 0,
                &MochigomaCollections::Pair(ms,mg) => {
                    ms.iter().fold(0,|acc,(_,c)| acc + c) +
                    mg.iter().fold(0,|acc,(_,c)| acc + c)
                }
            };

            if gs.depth >= 4 &&
                gs.beta < Score::MAYBEINFINITE &&
                gs.beta > Score::MAYBENEGINFINITE &&
                !Rule::in_check(gs.teban,&gs.state) &&
                ((gs.state.get_part().sente_self_board | gs.state.get_part().sente_opponent_board).bitcount() > 12 ||
                  mcount > 6) && ou_surrounding_threats_count < 3 {

                let depth = gs.depth - R;

                match self.search_null_move(env, gs,-gs.beta,-gs.beta+1, depth, event_dispatcher, evalutor)? {
                    EvaluationResult::Immediate(s, _, _) => {
                        let s = -s;

                        if s >= gs.beta {
                            env.transposition_table.update(&gs.zh,depth as i8,s,gs.beta,gs.alpha,Bound::LowerBound,None);

                            let mut best_moves = VecDeque::new();

                            gs.m.map(|m| best_moves.push_front(m));

                            return Ok(EvaluationResult::Immediate(s, best_moves, gs.zh.clone()));
                        }
                    },
                    EvaluationResult::Timeout => {
                        return Ok(EvaluationResult::Timeout);
                    },
                    EvaluationResult::Stop => {
                        return Ok(EvaluationResult::Stop);
                    },
                    EvaluationResult::Repetition => {

                    }
                }
            }
        }

         */

        /*
        if gs.depth >= 1 && gs.depth <= 3 &&
            !Rule::in_check(gs.teban,&gs.state) &&
            !Rule::in_check(gs.teban.opposite(),&gs.state) &&
            gs.m.map(|m| {
                m.obtained().is_none() && !m.is_nari()
            }).unwrap_or(true) {
            let static_eval = evalutor.evalute(gs.teban,gs.state,gs.mc)?;

            let boundary = Score::Value(static_eval + FU_SCORE * (gs.depth as i32 + 1));

            // Futility Pruning
            if gs.depth <= 2 && boundary <= gs.alpha {
                let mut mvs = VecDeque::new();

                prev_move.map(|m| mvs.push_front(m));

                return Ok(EvaluationResult::Immediate(Score::MAYBENEGINFINITE, mvs, gs.zh.clone()));
            // Razoring
            } else if gs.depth >= 2 && boundary <= gs.alpha {
                let s = self.qsearch(gs.teban,
                                     &gs.state,
                                     &gs.mc,
                                     env,
                                     event_dispatcher,
                                     &gs.zh,
                                     &mut HashSet::new(),
                                     gs.alpha,
                                     gs.beta,
                                     0,
                                     1,
                                     evalutor,
                                     gs.rng)?;

                if s <= gs.alpha {
                    let mut mvs = VecDeque::new();

                    prev_move.map(|m| mvs.push_front(m));

                    if env.stop.load(Ordering::Acquire) {
                        return Ok(EvaluationResult::Stop);
                    } else {
                        return Ok(EvaluationResult::Immediate(s, mvs, gs.zh.clone()));
                    }
                }
            }
        }
        */
        if gs.depth == 0 {
            let s = self.qsearch(gs.teban,
                                       &gs.state,
                                       &gs.mc,
                                       env,
                                       event_dispatcher,
                                       &gs.zh,
                                       &mut HashSet::new(),
                                       gs.alpha,
                                       gs.beta,
                                 0,
                                       1,
                                       evalutor,
                                       gs.rng)?;

            let mut mvs = VecDeque::new();

            prev_move.map(|m| mvs.push_front(m));

            if env.stop.load(Ordering::Acquire) {
                return Ok(EvaluationResult::Stop);
            } else {
                return Ok(EvaluationResult::Immediate(s, mvs, gs.zh.clone()));
            }
        }

        let start_alpha = gs.alpha;
        let mut alpha = gs.alpha;
        let mut quiet_alpha = gs.alpha;

        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        let mut picker = RandomPicker::new(Prng::new(gs.rng.gen()));

        let count = if Rule::in_check(gs.teban,&gs.state) {
            2
        } else {
            3
        };

        env.history.insert((gs.teban,mk,sk));

        let mut pv_move = None;
        let mut tt_move = None;

        let pv_non = VecDeque::new();

        #[allow(unused)]
        let mut search_count = 0;
        #[allow(unused)]
        let mut mvs_count = 0;

        let mut pruned_count = 0;

        let mut evasions_count = None;

        let mut quiet_moves = Vec::with_capacity(593);

        for i in 0..count {
            if i == 0 {
                tt_move = if let Some(TTPartialEntry {
                                              depth: _,
                                              score: _,
                                              beta: _,
                                              alpha: _,
                                              bound: _,
                                              best_move: m
                                          }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
                    m
                } else {
                    None
                };

                pv_move = if gs.pv.len() > gs.current_depth as usize {
                    Some(gs.pv[gs.current_depth as usize])
                } else {
                    None
                };

                {
                    let mvs = if let Some(pv) = pv_move {
                        if let Some(m) = tt_move {
                            vec![pv, m]
                        } else {
                            vec![pv]
                        }
                    } else {
                        if let Some(m) = tt_move {
                            vec![m]
                        } else {
                            vec![]
                        }
                    };

                    for (j,m) in mvs.into_iter().enumerate() {
                        if m.obtained().is_none() {
                            quiet_moves.push(m);
                        }

                        let mut r = self.calc_lmr(env,gs.depth,
                                                       gs.current_depth,
                                                       j,
                                                       gs.teban,
                                                       gs.state,
                                                       m,
                                                       tt_move.as_ref(),
                                                       pv_move.as_ref(),
                                                       prev_move.as_ref(),
                                                       gs.prev_kind,
                                                       calc_see(gs.teban,gs.state,m),
                                                       &gs.zh,
                                                       prev_eval,
                                                       static_eval)?;

                        let pv = pv_move.map(|pv| {
                            if pv == m {
                                gs.pv
                            } else {
                                &pv_non
                            }
                        }).unwrap_or(&pv_non);

                        for k in 0..2 {
                            let depth = if k == 0 {
                                gs.depth - r
                            } else {
                                gs.depth
                            };

                            if self.is_obtained_ou(m)? {
                                env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE,beta,alpha,Bound::Exact,Some(m));

                                let mut mvs = VecDeque::new();

                                mvs.push_front(m);
                                prev_move.map(|m| mvs.push_front(m));
                                env.history.remove(&(gs.teban,mk,sk));

                                return Ok(EvaluationResult::Immediate(Score::INFINITE, mvs, gs.zh.clone()));
                            }

                            match self.search_child_node(env, gs, m, pv, alpha, depth, static_eval, evasions_count, event_dispatcher, evalutor)? {
                                EvaluationResult::Immediate(s, mvs, _) => {
                                    let s = -s;

                                    if r > 0 && (s >= beta || s > start_alpha) {
                                        r = 0;
                                        continue;
                                    }

                                    if s > scoreval {
                                        scoreval = s;

                                        best_moves = mvs;

                                        if scoreval >= beta {
                                            if scoreval == Score::INFINITE {
                                                env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::Exact,Some(m));
                                            } else {
                                                env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::LowerBound,Some(m));
                                            }

                                            match m {
                                                LegalMove::To(mv) if mv.obtained().is_none() => {
                                                    if !mv.is_nari() {
                                                        env.move_orderer.update_killer(gs.current_depth, m)?;

                                                        let _ = prev_move.map(|prev_move| {
                                                            env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                        }).unwrap_or(Ok(()))?;
                                                    }

                                                    env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                                },
                                                LegalMove::Put(_) => {
                                                    env.move_orderer.update_killer(gs.current_depth, m)?;

                                                    let _ = prev_move.map(|prev_move| {
                                                        env.move_orderer.update_counter_move(m,gs.teban,prev_move,gs.prev_kind)
                                                    }).unwrap_or(Ok(()))?;

                                                    env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                                },
                                                _ => ()
                                            };

                                            env.history.remove(&(gs.teban,mk,sk));

                                            prev_move.map(|m| best_moves.push_front(m));

                                            return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                        }
                                    }

                                    if m.obtained().is_none() && quiet_alpha < s {
                                        quiet_alpha = s;
                                    }

                                    if alpha < s {
                                        alpha = s;
                                    }

                                    break;
                                },
                                EvaluationResult::NodeLimits => {
                                    env.history.remove(&(gs.teban,mk,sk));

                                    return Ok(EvaluationResult::NodeLimits);
                                },
                                EvaluationResult::Timeout => {
                                    env.history.remove(&(gs.teban,mk,sk));

                                    return Ok(EvaluationResult::Timeout);
                                },
                                EvaluationResult::Stop => {
                                    env.history.remove(&(gs.teban,mk,sk));

                                    return Ok(EvaluationResult::Stop);
                                },
                                EvaluationResult::Repetition => {
                                }
                            }
                        }

                        search_count += 1;
                    }
                }

                continue;
            } else if i == 1 && Rule::in_check(gs.teban,&gs.state) {
                Rule::generate_moves::<Evasions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
                evasions_count = Some(picker.len());
            } else if i == 1 {
                Rule::generate_moves::<CaptureOrPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
                evasions_count = None;
            } else {
                Rule::generate_moves::<QuietsWithoutPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
                evasions_count = None;
            }

            mvs_count += picker.len();

            for (j,(m,see)) in env.move_orderer.ordering(
                &mut picker, gs.current_depth, gs.teban, &gs.state, gs.m, gs.prev_kind,gs.move_history)?.enumerate() {

                if pv_move.map(|pv | pv == m).unwrap_or(false) {
                    continue;
                }

                if tt_move.map(|tt_move | tt_move == m).unwrap_or(false) {
                    continue;
                }

                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                let is_nari = match m {
                    LegalMove::To(mv) => mv.is_nari(),
                    _ => false
                };

                /*
                if let Some(o) = m.obtained() {
                    if !is_nari && !Rule::is_oute_move(gs.state,gs.teban,m) &&
                        see < -PIECE_SCORE_MAP[o as usize] / 2 {
                        pruned_count += 1;
                        continue;
                    }
                }

                 */

                let mut r  = self.calc_lmr(env,
                                                gs.depth,
                                                gs.current_depth,
                                          j+gs.search_offset,
                                                gs.teban,
                                                gs.state,
                                                m,
                                                tt_move.as_ref(),
                                                pv_move.as_ref(),
                                                prev_move.as_ref(),
                                                gs.prev_kind,
                                                see,
                                                &gs.zh,
                                                prev_eval,
                                                static_eval)?;

                for k in 0..2 {
                    let depth = if k == 0 {
                        gs.depth - r
                    } else {
                        gs.depth
                    };

                    if self.is_obtained_ou(m)? {
                        env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,beta,alpha,Bound::Exact,Some(m));

                        let mut mvs = VecDeque::new();

                        mvs.push_front(m);
                        prev_move.map(|m| mvs.push_front(m));
                        env.history.remove(&(gs.teban,mk,sk));

                        return Ok(EvaluationResult::Immediate(Score::INFINITE,mvs,gs.zh.clone()));
                    }

                    match self.search_child_node(env,gs,m,&pv_non,alpha,depth,static_eval,evasions_count,event_dispatcher,evalutor)? {
                        EvaluationResult::Immediate(s, mvs, _) => {
                            let s = -s;

                            if r > 0 && (s >= beta || s > start_alpha) {
                                r = 0;
                                continue
                            }

                            if s > scoreval {
                                scoreval = s;

                                best_moves = mvs;

                                if scoreval >= beta {
                                    if Score::INFINITE == scoreval {
                                        env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::Exact,Some(m));
                                    } else {
                                        env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::LowerBound,Some(m));
                                    }

                                    match m {
                                        LegalMove::To(mv) if mv.obtained().is_none() => {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;

                                                let _ = prev_move.map(|prev_move| {
                                                    env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                }).unwrap_or(Ok(()))?;
                                            }

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        LegalMove::Put(_) => {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;

                                            let _ = prev_move.map(|prev_move| {
                                                env.move_orderer.update_counter_move(m,gs.teban,prev_move,gs.prev_kind)
                                            }).unwrap_or(Ok(()))?;

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        _ => ()
                                    };
                                    env.history.remove(&(gs.teban,mk,sk));

                                    prev_move.map(|m| best_moves.push_front(m));

                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                }
                            }

                            if m.obtained().is_none() && quiet_alpha < s {
                                quiet_alpha = s;
                            }

                            if alpha < s {
                                alpha = s;
                            }

                            break;
                        },
                        EvaluationResult::NodeLimits => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::NodeLimits);
                        },
                        EvaluationResult::Timeout => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Timeout);
                        },
                        EvaluationResult::Stop => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Stop);
                        },
                        EvaluationResult::Repetition => {
                        }
                    }
                }

                search_count += 1;
            }
        }

        if quiet_alpha == start_alpha {
            for m in quiet_moves {
                env.move_orderer.update_degrade_history(gs.teban,&gs.state,m,gs.depth)?;
            }
        }

        if pruned_count > 0 && scoreval == Score::NEGINFINITE {
            scoreval = Score::MAYBENEGINFINITE;
        }

        if scoreval <= start_alpha {
            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval, beta, alpha, Bound::UpperBound, best_moves.front().map(|m| m.clone()));
        } else if scoreval != Score::MAYBEINFINITE && scoreval != Score::MAYBENEGINFINITE {
            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval, beta, alpha, Bound::Exact, best_moves.front().map(|m| m.clone()));
        }

        env.history.remove(&(gs.teban,mk,sk));

        prev_move.map(|m| best_moves.push_front(m));

        Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()))
    }
}
pub struct Inter<L,S,M> where L: Logger + Send + 'static,
                              S: InfoSender,
                              M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                 PreTrain<f32> + Send + Sync + 'static,
                              <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>
}
impl<L,S,M> Inter<L,S,M> where L: Logger + Send + 'static,
                               S: InfoSender,
                               M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                  PreTrain<f32> + Send + Sync + 'static,
                               <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn new() -> Inter<L,S,M> {
        Inter {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            m:PhantomData::<M>
        }
    }

    pub fn is_obtained_ou(&self,m:LegalMove) -> Result<bool,ApplicationError> {
        Ok(Some(ObtainKind::Ou) == m.obtained())
    }

    pub fn search_child_node<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                    m:LegalMove,
                                    pv:&VecDeque<LegalMove>,
                                    alpha:Score,
                                    depth:u32,
                                    static_eval: i32,
                                    evasions_count: Option<usize>,
                                    event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                    evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        let search = Recursive::new();

        Ok(search.search_child_node(env,gs,m,pv,alpha,depth,static_eval,evasions_count,event_dispatcher,evalutor)?)
    }
}
impl<L,S,M> PartialSearch<L,S,M> for Inter<L,S,M> where L: Logger + Send + 'static,
                                                        S: InfoSender,
                                                        M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                                           PreTrain<f32> + Send + Sync + 'static,
                                                        <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      evalutor: &Arc<Evalutor<M>>,
                      mvs:&Vec<LegalMove>) -> Result<EvaluationResult, ApplicationError> {
        let recur = Recursive::new();

        let limit = env.limit.clone();
        let turn_limit = env.turn_limit.clone();

        let (mk,sk) = gs.zh.keys();

        if env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Stop);
        }

        if recur.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        let prev_eval = gs.prev_self_ss;

        let static_eval = 0;//evalutor.evalute(gs.teban,&gs.state,&gs.mc)?;

        if recur.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        let mut event_dispatcher = Root::<L,S,M>::create_event_dispatcher::<Recursive<L,S,M>>(
            &env.on_error_handler, &env.stop, &env.quited, env.teban.clone(), &limit,&turn_limit, &env.current_limit
        );

        event_dispatcher.dispatch_events(&recur,&env.event_queue)?;

        if env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Stop);
        }

        if recur.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        let prev_move = gs.m.clone();

        let start_alpha = gs.alpha;
        let mut alpha = gs.alpha;
        let mut quiet_alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        env.history.insert((gs.teban,mk,sk));

        let tt_move = if let Some(TTPartialEntry {
                        depth: _,
                        score: _,
                        beta: _,
                        alpha: _,
                        bound: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            m
        } else {
            None
        };

        let pv_move = if gs.pv.len() > gs.current_depth as usize {
            Some(gs.pv[gs.current_depth as usize])
        } else {
            None
        };

        let pv_non = VecDeque::new();

        #[allow(unused)]
        let mut search_count =  0;

        let mut pruned_count = 0;
        let mut enable_pruning_by_see = true;

        let mut evasions_count = None;

        let mut quiet_moves = Vec::with_capacity(593);

        {
            let mvs = if let Some(pv) = pv_move {
                if let Some(m) = tt_move {
                    vec![pv, m]
                } else {
                    vec![pv]
                }
            } else {
                if let Some(m) = tt_move {
                    vec![m]
                } else {
                    vec![]
                }
            };

            for (i,m) in mvs.into_iter().enumerate() {
                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                let mut r = recur.calc_lmr(env,
                                                gs.depth,
                                                gs.current_depth,
                                                i,
                                                gs.teban,
                                                gs.state,
                                                m,
                                                tt_move.as_ref(),
                                                pv_move.as_ref(),
                                                prev_move.as_ref(),
                                                gs.prev_kind,
                                                calc_see(gs.teban,gs.state,m),
                                                &gs.zh,
                                                prev_eval,
                                                static_eval)?;

                for j in 0..2 {
                    let depth = if j == 0 {
                        gs.depth - r
                    } else {
                        gs.depth
                    };

                    if self.is_obtained_ou(m)? {
                        env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE,beta,alpha,Bound::Exact,Some(m));

                        let mut mvs = VecDeque::new();

                        mvs.push_front(m);
                        env.history.remove(&(gs.teban,mk,sk));

                        return Ok(EvaluationResult::Immediate(Score::INFINITE, mvs, gs.zh.clone()));
                    }

                    let pv = pv_move.map(|pv| {
                        if pv == m {
                            gs.pv
                        } else {
                            &pv_non
                        }
                    }).unwrap_or(&pv_non);

                    match recur.search_child_node(env, gs, m, pv, alpha, depth, static_eval, evasions_count, &mut event_dispatcher, evalutor)? {
                        EvaluationResult::Immediate(s, mvs, _) => {
                            let s = -s;

                            if r > 0 && (s >= beta || s > start_alpha) {
                                r = 0;
                                continue;
                            }

                            if s > scoreval {
                                scoreval = s;

                                best_moves = mvs;

                                if gs.search_offset == 0 {
                                    assert_eq!(gs.search_offset,0);
                                    recur.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
                                }

                                if scoreval >= beta {
                                    if Score::INFINITE == scoreval {
                                        env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::Exact,Some(m));
                                    } else {
                                        env.transposition_table.update(&gs.zh,depth as i8,scoreval,beta,alpha,Bound::LowerBound,Some(m));
                                    }

                                    match m {
                                        LegalMove::To(mv) if mv.obtained().is_none() => {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;

                                                let _ = prev_move.map(|prev_move| {
                                                    env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                }).unwrap_or(Ok(()))?;
                                            }

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        LegalMove::Put(_) => {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;

                                            let _ = prev_move.map(|prev_move| {
                                                env.move_orderer.update_counter_move(m,gs.teban,prev_move,gs.prev_kind)
                                            }).unwrap_or(Ok(()))?;

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        _ => ()
                                    };

                                    env.history.remove(&(gs.teban,mk,sk));

                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                }
                            }

                            if m.obtained().is_none() && quiet_alpha < s {
                                quiet_alpha = s;
                            }

                            if alpha < s {
                                alpha = s;
                            }

                            break;
                        },
                        EvaluationResult::NodeLimits => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::NodeLimits);
                        },
                        EvaluationResult::Timeout => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Timeout);
                        },
                        EvaluationResult::Stop => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Stop);
                        },
                        EvaluationResult::Repetition => {
                        }
                    }
                }

                search_count += 1;
            }
        }

        for i in 0..2 {
            for (j,(m,see)) in env.move_orderer.ordering(
                mvs.iter().cloned(), gs.current_depth, gs.teban, &gs.state, gs.m, gs.prev_kind, gs.move_history
            )?.skip(gs.search_offset).enumerate() {
                if pv_move.map(|pv | pv == m).unwrap_or(false) {
                    continue;
                }

                if tt_move.map(|tt_move | tt_move == m).unwrap_or(false) {
                    continue;
                }

                if i == 0 && m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                let is_nari = match m {
                    LegalMove::To(mv) => mv.is_nari(),
                    _ => false
                };

                /*
                if let Some(o) = m.obtained() {
                    if enable_pruning_by_see && !is_nari &&
                        !Rule::is_oute_move(gs.state,gs.teban,m) &&
                        see < -PIECE_SCORE_MAP[o as usize] / 2 {
                        pruned_count += 1;
                        continue;
                    }
                }

                 */

                let mut r = recur.calc_lmr(env,
                                               gs.depth,
                                               gs.current_depth,
                                         j+gs.search_offset,
                                               gs.teban,
                                               gs.state,
                                               m,
                                               tt_move.as_ref(),
                                               pv_move.as_ref(),
                                               prev_move.as_ref(),
                                               gs.prev_kind,
                                               see,
                                               &gs.zh,
                                               prev_eval,
                                               static_eval)?;

                for j in 0..2 {
                    let depth = if j == 0 {
                        gs.depth - r
                    } else {
                        gs.depth
                    };

                    if self.is_obtained_ou(m)? {
                        env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE,beta,alpha,Bound::Exact,Some(m));

                        let mut mvs = VecDeque::new();

                        mvs.push_front(m);
                        env.history.remove(&(gs.teban,mk,sk));

                        return Ok(EvaluationResult::Immediate(Score::INFINITE,mvs,gs.zh.clone()));
                    }

                    match recur.search_child_node(env,gs,m,&pv_non,alpha,depth,static_eval,evasions_count,&mut event_dispatcher,evalutor)? {
                        EvaluationResult::Immediate(s, mvs, _) => {
                            let s = -s;

                            if r > 0 && (s >= beta || s > start_alpha) {
                                r = 0;
                                continue;
                            }

                            if s > scoreval {
                                scoreval = s;

                                best_moves = mvs;

                                if gs.search_offset == 0 {
                                    assert_eq!(gs.search_offset,0);
                                    recur.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
                                }

                                if scoreval >= beta {
                                    if Score::INFINITE == scoreval {
                                        env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,beta,alpha,Bound::Exact,Some(m));
                                    } else {
                                        env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,beta,alpha,Bound::LowerBound,Some(m));
                                    }

                                    match m {
                                        LegalMove::To(mv) if mv.obtained().is_none() => {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;
                                            }
                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        LegalMove::Put(_) => {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;
                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,depth,gs.move_history)?;
                                        },
                                        _ => ()
                                    };

                                    env.history.remove(&(gs.teban,mk,sk));

                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                }
                            }

                            if m.obtained().is_none() && quiet_alpha < s {
                                quiet_alpha = s;
                            }

                            if alpha < s {
                                alpha = s;
                            }

                            break;
                        },
                        EvaluationResult::NodeLimits => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::NodeLimits);
                        },
                        EvaluationResult::Timeout => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Timeout);
                        },
                        EvaluationResult::Stop => {
                            env.history.remove(&(gs.teban,mk,sk));

                            return Ok(EvaluationResult::Stop);
                        },
                        EvaluationResult::Repetition => {
                        }
                    }
                }

                search_count += 1;
            }

            if pruned_count < mvs.len() {
                break;
            } else {
                enable_pruning_by_see = false;
            }
        }

        if quiet_alpha == start_alpha {
            for m in quiet_moves {
                env.move_orderer.update_degrade_history(gs.teban, &gs.state, m, gs.depth)?;
            }
        }

        if pruned_count > 0 && scoreval == Score::NEGINFINITE {
            scoreval = Score::MAYBENEGINFINITE;
        }

        if gs.search_offset == 0 && scoreval <= start_alpha {
            env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,beta,alpha,Bound::UpperBound,best_moves.front().map(|m| m.clone()));
        } else if gs.search_offset == 0 && scoreval != Score::MAYBEINFINITE && scoreval != Score::MAYBENEGINFINITE  {
            env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,beta,alpha,Bound::Exact,best_moves.front().map(|m| m.clone()));
        }

        env.history.remove(&(gs.teban,mk,sk));

        Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()))
    }
}