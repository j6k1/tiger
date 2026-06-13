use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::{Deref};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use nncombinator::arr::Arr;
use nncombinator::layer::{ContinueForward, ForwardAll, PartialForward, PreTrain};
use parking_lot::RwLock;
use rand::Rng;
use rand::rngs::ThreadRng;
use rayon::ThreadPool;
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::consts::{CAPTURED_SCORE_MAP, FU_SCORE};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher, UsiGoTimeLimit};
use usiagent::hash::KyokumenHash;
use usiagent::logger::Logger;
use usiagent::math::Prng;
use usiagent::move_orderer::{MoveOrderer, UnusedQuietSee};
use usiagent::movepick::{MovePicker, RandomPicker};
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{CaptureOrPawnPromotions, Evasions, LegalMove, NonEvasions, QuietsWithoutPawnPromotions, Rule, SquareToPoint, State, OU_SURROUNDING_BOTTOM_MASK, OU_SURROUNDING_MASK, OU_SURROUNDING_TOP_MASK};
use usiagent::see::calc_see;
use usiagent::shogi::{KomaKind, MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use usiagent::shogi::KomaKind::Blank;
use crate::error::ApplicationError;
use crate::features::{HalfKP, HalfKPDiff};
use crate::math::SignFloat;
use crate::nn::{Evalutor, FEATURES_NUM};
use crate::transposition_table::{TT, ZobristHash, TTPartialEntry, Bound, Score, NormalizeMate, LocalizeMate};

pub const TURN_LIMIT:u32 = 1000;
pub const BASE_DEPTH:u32 = 20;
pub const MAX_THREADS:u32 = 2;
pub const THREATMATE_DEPTH:u32 = 7;

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub enum MateDepth {
    Mate(i32),
    Nomate
}
#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub struct LazyEval {
    pub static_eval:Option<i32>,
}
impl LazyEval {
    pub fn new() -> LazyEval {
        LazyEval {
            static_eval:None
        }
    }

    pub fn get_or_insert_with<F: FnOnce() -> Result<i32,ApplicationError>>(&mut self, f:F) -> Result<i32,ApplicationError> {
        match self.static_eval {
            Some(e) => Ok(e),
            None => {
                let static_eval = f()?;
                Ok(*self.static_eval.get_or_insert(static_eval))
            }
        }
    }
}
pub struct Environment<L,S> where L: Logger, S: InfoSender {
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub teban:Teban,
    pub limit:Option<UsiGoTimeLimit>,
    pub turn_limit:Option<u32>,
    pub timelimit_margin:u64,
    pub current_limit:Arc<RwLock<(Option<Instant>,Option<Instant>)>>,
    pub base_depth:u32,
    pub qsearch_max_depth:Option<u32>,
    pub threatmate_depth:u32,
    pub max_nodes:Option<u64>,
    pub max_threads:u32,
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
            hasher:Arc::clone(&self.hasher),
            teban:self.teban.clone(),
            limit:self.limit.clone(),
            turn_limit:self.turn_limit.clone(),
            timelimit_margin:self.timelimit_margin.clone(),
            current_limit:self.current_limit.clone(),
            base_depth:self.base_depth,
            qsearch_max_depth:self.qsearch_max_depth.clone(),
            threatmate_depth:self.threatmate_depth,
            max_nodes:self.max_nodes.clone(),
            max_threads:self.max_threads,
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
    Exact(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32),
    Cut,
    NodeLimits,
    Timeout,
    Stop,
    Repetition
}
impl EvaluationResult {
    pub fn best_score(&self) -> Option<Score> {
        match self {
            EvaluationResult::Exact(s, _, _, _) => Some(*s),
            _ => None
        }
    }
}
#[derive(Debug)]
pub enum RootEvaluationResult {
    Exact(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32, u32, usize),
    NodeLimits,
    Timeout,
    Stop,
    Repetition,
    Quit(MoveOrderer<UnusedQuietSee>,usize)
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               teban:Teban,
               limit:Option<UsiGoTimeLimit>,
               turn_limit:Option<u32>,
               timelimit_margin:u64,
               current_limit:(Option<Instant>,Option<Instant>),
               base_depth:u32,
               qsearch_max_depth:Option<u32>,
               threatmate_depth:u32,
               max_nodes:Option<u64>,
               max_threads:u32,
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
            teban:teban,
            limit:limit,
            turn_limit:turn_limit,
            timelimit_margin:timelimit_margin,
            current_limit:Arc::new(RwLock::new(current_limit)),
            base_depth:base_depth,
            qsearch_max_depth:qsearch_max_depth,
            threatmate_depth:threatmate_depth,
            max_nodes:max_nodes,
            max_threads:max_threads,
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
    pub static_eval: LazyEval,
    pub prev_kind:KomaKind,
    pub move_history:&'a mut Vec<Option<(u8,u8)>>,
    pub self_partial_output: Arc<Arr<f32,{256*2}>>,
    pub opponent_partial_output: Arc<Arr<f32,{256*2}>>,
    pub thread_index:usize,
    pub pv:&'a VecDeque<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub zh:ZobristHash<u64>,
    pub depth:u32,
    pub current_depth:u32,
    pub cut_node:bool,
    pub already_reduced_lmr:bool,
    pub nmp_min_ply:Option<u32>,
    pub base_depth:u32,
    pub extend_depth:u32,
    pub extend_check:u32,
    pub extend_threatmate:u32

}
pub struct Root<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>,
    receiver:Receiver<Result<RootEvaluationResult, ApplicationError>>,
    sender:Sender<Result<RootEvaluationResult, ApplicationError>>,
    thread_pool:ThreadPool
}
pub const TIMELIMIT_MARGIN:u64 = 50;

/*
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MoveOrder {
    Quiet,
    PawnPromotions,
    Captures,
    ThreatMate,
    ThreatCaptures,
    Check
}
 */
pub trait SendInfo<L,S,M>: Sized
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn send_info(&self, env:&mut Environment<L,S>,
                 depth:u32, seldepth:u32, pv:&VecDeque<LegalMove>, score:&Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            &Score::INFINITE(depth) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Num(-(depth as i64)))))
            },
            &Score::NEGINFINITE(depth) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Num(-(depth as i64)))))
            },
            &Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(s as i64)))
            }
        }

        commands.push(UsiInfoSubCommand::Depth(depth));
        commands.push(UsiInfoSubCommand::SelDepth(seldepth));

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }

        commands.push(UsiInfoSubCommand::Nodes(env.nodes.load(Ordering::Acquire)));

        Ok(env.info_sender.send(commands)?)
    }

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands)?)
    }
}
pub trait Search<L,S,M>: SendInfo<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError>;
    fn qsearch<'b>(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
               env:&mut Environment<L,S>,
               event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
               zh: &ZobristHash<u64>,
               history:&mut HashSet<(Teban,u64,u64)>,
               self_partial_output: Arc<Arr<f32,{256*2}>>,
               opponent_partial_output: Arc<Arr<f32,{256*2}>>,
               mut alpha:Score,beta:Score,
               depth:usize,current_depth:usize,
               prev_move:Option<LegalMove>,
               evalutor: &Arc<Evalutor<M>>,rng:&mut ThreadRng)
        -> Result<Score,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            env.qsearch_max_depth.map(|d| depth >= d as usize).unwrap_or(false) ||
            self.timelimit_reached(env)? || history.contains(&(teban,mk,sk)) {
            let score = Score::Value(evalutor.evalute(&self_partial_output)?);
            return Ok(score);
        }

        {
            let r = env.transposition_table.get(&zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: _,
                            score: s,
                            bound,
                            best_move: _
                        }) = r {

                if bound == Bound::Exact ||
                   (bound == Bound::LowerBound && s >= beta) ||
                   (bound == Bound::UpperBound && s <= alpha) {

                    return Ok(s.normalize_mate(current_depth as i32));
                }
            }
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        let in_check = Rule::in_check(teban,state);

        if in_check {
            Rule::generate_moves::<Evasions>(teban, state, mc, &mut picker)?;

            if picker.len() == 0 {
                env.transposition_table.update(&zh,0,Score::NEGINFINITE(0),Bound::Exact,None);
                return Ok(Score::NEGINFINITE(current_depth as i32));
            }

            let start_alpha = alpha;

            let mut bestscore = Score::default();

            let mut best_move = None;

            history.insert((teban,mk,sk));

            for m in picker {
                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh,0,Score::INFINITE(0),Bound::Exact,Some(m));

                    return Ok(Score::INFINITE(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

                let self_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban,&state,&mc,&next,&nmc,m,Arc::clone(&self_partial_output))?);
                let opponent_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban.opposite(),&state,&mc,&next,&nmc,m,Arc::clone(&opponent_partial_output))?);

                let score = -self.qsearch(teban.opposite(),
                                          &next,
                                          &nmc,
                                          env,
                                          event_dispatcher,
                                          &nzh,
                                          history,
                                          opponent_partial_output,
                                          self_partial_output,
                                          -beta,
                                          -alpha,
                                          depth+1,
                                          current_depth+1,
                                          Some(m),
                                          evalutor,
                                          rng)?;

                if score >= beta {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh, 0, score.localize_mate(current_depth as i32), Bound::LowerBound, Some(m));

                    return Ok(score);
                }

                if score > bestscore {
                    best_move = Some(m);

                    bestscore = score;
                }

                if score > alpha {
                    alpha = score;
                }

                if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
                    self.timelimit_reached(env)? {
                    break;
                }
            }

            history.remove(&(teban,mk,sk));

            let bs = bestscore.localize_mate(current_depth as i32);

            if alpha > start_alpha {
                env.transposition_table.update(&zh, 0, bs, Bound::Exact, best_move);
            } else {
                env.transposition_table.update(&zh, 0, bs, Bound::UpperBound, best_move);
            }

            Ok(bestscore)
        } else {
            let stand_pat = Score::Value(evalutor.evalute(&self_partial_output)?);

            if stand_pat >= beta {
                return Ok(stand_pat);
            }

            Rule::generate_moves_by_banmen::<CaptureOrPawnPromotions>(teban,state,&mut picker)?;

            let mvs = picker.filter(|m| m.obtained().is_some()).collect::<Vec<_>>();

            if mvs.len() == 0 {
                return Ok(stand_pat);
            }

            if stand_pat > alpha {
                alpha = stand_pat;
            }

            let mut bestscore = stand_pat;

            let mut best_move = None;

            history.insert((teban,mk,sk));

            for m in mvs {
                if !m.is_nari() {
                    if let Some(o) = m.obtained() {
                        if !prev_move.map(|pm| {
                            pm.obtained().is_some() && m.dst() == pm.dst()
                        }).unwrap_or(false) && !Rule::is_oute_move(state,teban,m) {
                            if calc_see(teban,state,m) < -CAPTURED_SCORE_MAP[o as usize] * 4 / 3 {
                                continue;
                            }
                        }
                    }
                }

                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh,0,Score::INFINITE(0),Bound::Exact,Some(m));

                    return Ok(Score::INFINITE(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

                let self_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban,&state,&mc,&next,&nmc,m,Arc::clone(&self_partial_output))?);
                let opponent_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban.opposite(),&state,&mc,&next,&nmc,m,Arc::clone(&opponent_partial_output))?);

                let score = -self.qsearch(teban.opposite(),
                                &next,
                                &nmc,
                                env,
                                event_dispatcher,
                                &nzh,
                                history,
                                opponent_partial_output,
                                self_partial_output,
                                -beta,
                                -alpha,
                                depth+1,
                                current_depth+1,
                                Some(m),
                                evalutor,
                                rng)?;

                if score >= beta {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh, 0, score.localize_mate(current_depth as i32), Bound::LowerBound, Some(m));

                    return Ok(score);
                }

                if score > bestscore {
                    bestscore = score;
                    best_move = Some(m);
                }

                if score > alpha {
                    alpha = score;
                }

                if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
                    self.timelimit_reached(env)? {
                    break;
                }
            }

            history.remove(&(teban,mk,sk));

            let bs = bestscore.localize_mate(current_depth as i32);

            env.transposition_table.update(&zh, 0, bs, Bound::UpperBound, best_move);

            Ok(bestscore)
        }
    }

    /*
    fn qsearch_threatmate<'b>(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
                   env:&mut Environment<L,S>,
                   event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                   zh: &ZobristHash<u64>,
                   history:&mut HashSet<(Teban,u64,u64)>,
                   self_partial_output: Arc<Arr<f32,{256*2}>>,
                   opponent_partial_output: Arc<Arr<f32,{256*2}>>,
                   mut alpha:Score,beta:Score,depth:usize, _:usize,
                   evalutor: &Arc<Evalutor<M>>,rng:&mut ThreadRng)
        -> Result<Score,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            env.qsearch_max_depth.map(|d| depth >= d as usize).unwrap_or(false) ||
            self.timelimit_reached(env)? || history.contains(&(teban,mk,sk)) {
            let score = Score::Value(evalutor.evalute(&self_partial_output)?);
            return Ok(score);
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        let mut bestscore = Score::NEGINFINITE;
        let mut stand_pat = Score::NEGINFINITE;

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

        'outer: for i in 0..2 {
            if i == 0 {
                Rule::generate_moves::<CaptureOrPawnPromotions>(teban,state,mc,&mut picker)?;
            } else {
                Rule::generate_moves::<QuietsWithoutPawnPromotions>(teban,state,mc,&mut picker)?;
            }

            let mut mvs = (&mut picker).map(|m| {
                let is_pawn_move = match m {
                    LegalMove::To(m) if teban == Teban::Sente => {
                        state.get_part().sente_fu_board & (1 << (m.src() + 1)) != 0
                    },
                    LegalMove::To(m) if teban == Teban::Gote => {
                        state.get_part().gote_fu_board & (1 << (m.src() + 1)) != 0
                    },
                    _ => false
                };

                let is_kin_move = match m {
                    LegalMove::Put(m) if m.kind() == MochigomaKind::Kin => true,
                    LegalMove::To(m) => {
                        if teban == Teban::Sente {
                            state.get_part().sente_kin_board & (1 << (m.src() + 1)) != 0
                        } else {
                            state.get_part().gote_kin_board & (1 << (m.src() + 1)) != 0
                        }
                    },
                    _ => false
                };

                let is_gin_move = match m {
                    LegalMove::Put(m) if m.kind() == MochigomaKind::Gin => true,
                    LegalMove::To(m) => {
                        if teban == Teban::Sente {
                            state.get_part().sente_gin_board & (1 << (m.src() + 1)) != 0
                        } else {
                            state.get_part().gote_gin_board & (1 << (m.src() + 1)) != 0
                        }
                    },
                    _ => false
                };

                let is_nari = m.is_nari();

                let dst_mask = 1 << (m.dst() + 1);

                if Rule::is_oute_move(state,teban,m) {
                    (MoveOrder::Check, m)
                } else if m.obtained().is_some() && (
                    opponent_surrounding_mask & dst_mask != 0
                ) {
                    (MoveOrder::ThreatCaptures,m)
                } else if ((is_nari && is_pawn_move) || is_kin_move || is_gin_move) && (
                    opponent_surrounding_mask & dst_mask != 0
                ) {
                    (MoveOrder::ThreatMate, m)
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
                return Ok(Score::NEGINFINITE);
            }

            history.insert((teban,mk,sk));

            stand_pat = Score::Value(evalutor.evalute(&self_partial_output)?);

            if stand_pat >= beta {
                return Ok(stand_pat);
            }

            if stand_pat > alpha {
                alpha = stand_pat;
            }

            for (mo,m) in mvs {
                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    return Ok(Score::INFINITE);
                }

                if mo < MoveOrder::Captures {
                    continue;
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let zh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

                let self_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban,&state,&mc,&next,&nmc,m,Arc::clone(&self_partial_output))?);
                let opponent_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(teban, teban.opposite(),&state,&mc,&next,&nmc,m,Arc::clone(&opponent_partial_output))?);

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
                                             opponent_partial_output,
                                             self_partial_output,
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
                                  opponent_partial_output,
                                  self_partial_output,
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

                if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
                    self.timelimit_reached(env)? {
                    break 'outer;
                }
            }
        }

        history.remove(&(teban,mk,sk));

        if bestscore == Score::NEGINFINITE {
            Ok(stand_pat)
        } else {
            Ok(bestscore)
        }
    }

     */

    fn satisfy_threatmate_search(&self, attacker:Teban, state:&State, m:LegalMove, depth:u32, evasions_count:usize) -> bool {
        if evasions_count > 4 || depth < 4 || depth > 6 {
            return false;
        }

        let ps = state.get_part();

        match attacker {
            Teban::Sente => {
                if ps.sente_checked_board & (
                    (ps.sente_kyou_board & !ps.sente_nari_board) |
                     ps.sente_kaku_board | ps.sente_hisha_board
                ) == 0 && !m.is_nari() && match m {
                    LegalMove::To(_) => true,
                    LegalMove::Put(_) => false
                } {
                    return false;
                }

                let mut count = 0;

                if (ps.sente_self_board | ps.sente_opponent_board).bitcount() <= 18 {
                    count += 1;
                }

                if (ps.sente_kaku_board | ps.sente_hisha_board |
                    ps.gote_kaku_board | ps.gote_hisha_board).bitcount() <= 2 {
                    count += 1;
                }

                if (ps.gote_nari_board | ps.sente_nari_board).bitcount() >= 2 {
                    count += 1;
                }

                if count >= 3 {
                    return true;
                }

                let p = Rule::ou_square(attacker.opposite(), state) as u32;

                let (_,y) = p.square_to_point();

                let mut mask = OU_SURROUNDING_MASK;

                if y == 0 {
                    mask = mask & OU_SURROUNDING_TOP_MASK;
                } else if y == 8 {
                    mask = mask & OU_SURROUNDING_BOTTOM_MASK;
                }

                if p < 10 {
                    mask = mask >> (10 - p) as u128;
                } else {
                    mask = mask << (p - 10) as u128;
                }

                if ((mask << 1) & (ps.sente_opponent_board | ps.sente_self_board)).bitcount() <= 4 {
                    count += 1;
                }

                count >= 3
            },
            Teban::Gote => {
                if ps.gote_checked_board & (
                    (ps.gote_kyou_board & !ps.gote_nari_board) |
                        ps.gote_kaku_board | ps.gote_hisha_board
                ) == 0 && !m.is_nari() && match m {
                    LegalMove::To(_) => true,
                    LegalMove::Put(_) => false
                } {
                    return false;
                }

                let mut count = 0;

                if (ps.gote_self_board | ps.gote_opponent_board).bitcount() <= 18 {
                    count += 1;
                }

                if (ps.gote_kaku_board | ps.gote_hisha_board |
                    ps.sente_kaku_board | ps.sente_hisha_board).bitcount() <= 2 {
                    count += 1;
                }

                if (ps.gote_nari_board | ps.sente_nari_board).bitcount() >= 2 {
                    count += 1;
                }

                if count >= 3 {
                    return true;
                }

                let p = Rule::ou_square(attacker.opposite(), state) as u32;

                let (_,y) = p.square_to_point();

                let mut mask = OU_SURROUNDING_MASK;

                if y == 0 {
                    mask = mask & OU_SURROUNDING_TOP_MASK;
                } else if y == 8 {
                    mask = mask & OU_SURROUNDING_BOTTOM_MASK;
                }

                if p < 10 {
                    mask = mask >> (10 - p) as u128;
                } else {
                    mask = mask << (p - 10) as u128;
                }

                if ((mask << 1) & (ps.sente_self_board | ps.sente_opponent_board)).bitcount() <= 4 {
                    count += 1;
                }

                count >= 3
            }
        }
    }

    fn threatmate_search<'b,MP>(&self,
                         attacker:Teban,
                         teban:Teban,
                         state:&State,
                         mc:&MochigomaCollections,
                         env:&mut Environment<L,S>,
                         event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                         zh: &ZobristHash<u64>,
                         history:&mut HashSet<(Teban,u64,u64)>,
                         depth:usize,
                         current_depth:usize,
                         mut picker:MP,
                         rng:&mut ThreadRng) -> Result<MateDepth,ApplicationError> where MP: MovePicker<LegalMove> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if depth == 0 || env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            self.timelimit_reached(env)? || history.contains(&(teban,mk,sk)) {
            return Ok(MateDepth::Nomate);
        }

        let ps = state.get_part();

        let mut count = 0;

        if (ps.sente_self_board | ps.sente_opponent_board).bitcount() > 18 {
            count += 1;
        }

        if (ps.sente_kaku_board | ps.sente_hisha_board |
            ps.gote_kaku_board | ps.gote_hisha_board).bitcount() > 2 {
            count += 1;
        }

        if (ps.sente_nari_board | ps.gote_nari_board).bitcount() < 2 {
            count += 1;
        }

        if count >= 2 {
            return Ok(MateDepth::Nomate);
        }

        let p = Rule::ou_square(attacker.opposite(),state) as u32;

        let (_,y) = p.square_to_point();

        let mut mask = OU_SURROUNDING_MASK;

        if y == 0 {
            mask = mask & OU_SURROUNDING_TOP_MASK;
        } else if y == 8 {
            mask = mask & OU_SURROUNDING_BOTTOM_MASK;
        }

        if p < 10 {
            mask = mask >> (10 - p) as u128;
        } else {
            mask = mask << (p - 10) as u128;
        }

        if ((mask << 1) & (ps.sente_opponent_board | ps.sente_self_board)).bitcount() > 4 {
            count += 1;
        }

        if count >= 2 {
            return Ok(MateDepth::Nomate);
        }

        {
            let r = env.transposition_table.get(&zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: _,
                            score: s,
                            bound,
                            best_move: _
                        }) = r {

                if bound == Bound::Exact {
                    match s {
                        Score::INFINITE(d) if attacker == teban => {
                            return Ok(MateDepth::Mate(-d + current_depth as i32));
                        },
                        Score::NEGINFINITE(d) if attacker != teban => {
                            return Ok(MateDepth::Mate(d + current_depth as i32));
                        },
                        Score::NEGINFINITE(_) => {
                            return Ok(MateDepth::Nomate);
                        },
                        _ => {}
                    }
                }
            }
        }

        let in_check = Rule::in_check(teban,state);

        if in_check {
            if picker.len() == 0 {
                Rule::generate_moves::<Evasions>(teban, state, mc, &mut picker)?;
            }

            if teban != attacker && picker.len() >= 8 {
                return Ok(MateDepth::Nomate);
            }
        } else {
            if picker.len() == 0 {
                Rule::generate_moves::<NonEvasions>(teban, state, mc, &mut picker)?;
            }
        }

        if picker.len() == 0 {
            if teban != attacker {
                return Ok(MateDepth::Nomate);
            } else {
                return Ok(MateDepth::Mate(current_depth as i32));
            }
        }

        history.insert((teban,mk,sk));

        let mut checkmate = MateDepth::Mate(0);

        for m in picker {
            if let Some(ObtainKind::Ou) = match m {
                LegalMove::To(m) => m.obtained(),
                _ => None
            } {
                history.remove(&(teban,mk,sk));

                env.transposition_table.update(&zh,depth as i8,Score::INFINITE(0),Bound::Exact,Some(m));

                if attacker == teban {
                    return Ok(MateDepth::Mate(current_depth as i32));
                }
            }

            if in_check || Rule::is_oute_move(state,teban,m) {
                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

                let picker = RandomPicker::new(Prng::new(rng.gen()));

                let mate_depth = self.threatmate_search(attacker,
                                                       teban.opposite(),
                                                       &next,
                                                       &nmc,
                                                       env,
                                                       event_dispatcher,
                                                       &nzh,
                                                       history,
                                                       depth - 1,
                                                       current_depth + 1,
                                                       picker,
                                                       rng)?;

                if let MateDepth::Mate(_) = mate_depth {
                    history.remove(&(teban, mk, sk));

                    if attacker == teban {
                        env.transposition_table.update(&zh, depth as i8, Score::INFINITE(-(depth as i32)), Bound::Exact, Some(m));

                        return Ok(mate_depth);
                    } else if mate_depth > checkmate {
                        checkmate = mate_depth;
                    }
                } else if attacker != teban {
                    history.remove(&(teban, mk, sk));

                    return Ok(MateDepth::Nomate);
                }

                if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
                    self.timelimit_reached(env)? {
                    break;
                }
            }
        }

        history.remove(&(teban,mk,sk));

        Ok(checkmate)
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
                current_depth:u32,
                teban: Teban,
                state: &State,
                m:LegalMove) -> Result<bool,ApplicationError> {
        Ok(m.obtained().is_some() ||
            m.is_nari() ||
            env.move_orderer.is_killer(current_depth,m)? ||
            Rule::is_oute_move(state,teban,m)
        )
    }
    fn calc_lmr(&self, env:&mut Environment<L,S>,
                       index:&mut usize,
                       depth:u32,
                       current_depth:u32,
                       teban: Teban,
                       state: &State,
                       m:LegalMove,
                       tt_move:Option<&LegalMove>,
                       pv:Option<&LegalMove>) -> Result<u32,ApplicationError> {
        if depth < 3 ||
            Rule::in_check(teban,state) ||
            tt_move.map(|&tm| tm == m).unwrap_or(false) ||
            pv.map(|pm| pm == &m).unwrap_or(false) ||
            self.is_important_move(env,current_depth,
                                   teban,state,m)? {
            Ok(0)
        } else if *index < 1 {
            *index += 1;
            Ok(0)
        } else {
            let move_index = *index + 1;
            let r = ((depth as f32).ln() * (move_index as f32).ln() / 2.2).floor() as i32;
            let r = r.clamp(0, depth as i32 - 1) as u32;

            let h = env.move_orderer.look_up_history(teban,state,m)?;

            *index += 1;

            if h > depth as i32 * 6 * 256 {
                Ok(r.saturating_sub(1) as u32)
            } else if h < -(depth as i32) * 6 * 256 {
                Ok(r.saturating_add(1).min(depth - 1))
            } else {
                Ok(r as u32)
            }
        }
    }

    /*
    fn in_danger(&self, teban: Teban, state:&State, m: LegalMove) -> bool {
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

        false
    }

    fn is_threat(&self, teban: Teban, state:&State, m:LegalMove) -> bool {
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

        false
    }
    */
}
pub trait PartialSearch<L,S,M>: Sized
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      evalutor: &Arc<Evalutor<M>>,
                      mvs:&Vec<LegalMove>) -> Result<EvaluationResult, ApplicationError>;
}
impl<L,S,M> Root<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
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
                           thread_index:usize,
                           shared_depth:&Arc<AtomicUsize>,
                           env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           mvs: Arc<Vec<LegalMove>>,
                           evalutor:&Arc<Evalutor<M>>,
                           move_orderer: MoveOrderer<UnusedQuietSee>) {
        let sender = self.sender.clone();
        let teban = gs.teban;
        let state = Arc::clone(&gs.state);
        let mut env = env.clone();

        env.move_orderer = move_orderer;

        let evalutor = Arc::clone(&evalutor);
        let mc = Arc::clone(&gs.mc);
        let zh = gs.zh.clone();
        let current_depth = 0;
        let base_depth = gs.base_depth;
        let extend_depth = gs.extend_depth;

        let mut best_score = Score::default();

        let self_partial_output = Arc::clone(&gs.self_partial_output);
        let opponent_partial_output = Arc::clone(&gs.opponent_partial_output);

        let shared_depth = Arc::clone(shared_depth);

        if thread_index == 0 {
            self.thread_pool.spawn(move || {
                let mut pv = VecDeque::new();
                let mut rng = rand::thread_rng();
                let search_offset = 0;

                let mut prev_score = Score::default();

                'ounter: for depth in 1..=base_depth {
                    if let Score::Value(ps) = prev_score {
                        let delta = Self::compute_aspiration_window_delta(depth);
                        //let delta = FU_SCORE * 2;

                        let mut alpha = Score::Value(ps - delta);
                        let mut beta = Score::Value(ps + delta);

                        let strategy = Inter::new();

                        for i in 0..2 {
                            let mut gs = GameState {
                                teban: teban,
                                state: &state,
                                alpha: alpha,
                                beta: beta,
                                search_offset: search_offset,
                                best_score: best_score,
                                m: None,
                                static_eval: LazyEval::new(),
                                prev_kind: KomaKind::Blank,
                                move_history: &mut Vec::new(),
                                self_partial_output:Arc::clone(&self_partial_output),
                                opponent_partial_output:Arc::clone(&opponent_partial_output),
                                thread_index:thread_index,
                                pv:&pv,
                                mc: &mc,
                                zh: zh.clone(),
                                depth: depth,
                                current_depth: current_depth,
                                cut_node: false,
                                already_reduced_lmr: false,
                                nmp_min_ply: None,
                                base_depth: base_depth,
                                extend_depth: extend_depth,
                                extend_check: 1,
                                extend_threatmate: 1,
                                rng:&mut rng
                            };

                            match strategy.search(&mut env, &mut gs, &evalutor, &mvs) {
                                Ok(EvaluationResult::Exact(score, mvs, zh, seldepth)) => {
                                    if i == 0 && score <= alpha {
                                        alpha = Score::default();
                                    } else if i == 0 && score >= beta {
                                        beta = Score::INFINITE(0);
                                    } else {
                                        prev_score = score;

                                        pv = mvs.clone();

                                        if score > best_score {
                                            best_score = score;
                                        }

                                        let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
                                        break;
                                    }
                                },
                                Ok(EvaluationResult::NodeLimits) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                                    break 'ounter;
                                },
                                Ok(EvaluationResult::Timeout) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                                    break 'ounter;
                                },
                                Ok(EvaluationResult::Repetition) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Repetition));
                                    break 'ounter;
                                },
                                Ok(EvaluationResult::Stop) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Stop));
                                    break 'ounter;
                                },
                                Ok(EvaluationResult::Cut) => {
                                    let _ = sender.send(Err(ApplicationError::LogicError(String::from("The root node has been pruned."))));
                                    break 'ounter;
                                },
                                Err(e) => {
                                    let _ = sender.send(Err(e));
                                    break 'ounter;
                                }
                            };
                        }
                    } else {
                        let mut gs = GameState {
                            teban: teban,
                            state: &state,
                            alpha: Score::default(),
                            beta: Score::INFINITE(0),
                            search_offset: search_offset,
                            best_score: best_score,
                            m: None,
                            static_eval: LazyEval::new(),
                            prev_kind: KomaKind::Blank,
                            move_history: &mut Vec::new(),
                            self_partial_output: Arc::clone(&self_partial_output),
                            opponent_partial_output: Arc::clone(&opponent_partial_output),
                            thread_index: thread_index,
                            pv: &pv,
                            mc: &mc,
                            zh: zh.clone(),
                            depth: depth,
                            current_depth: current_depth,
                            cut_node: false,
                            already_reduced_lmr: false,
                            nmp_min_ply: None,
                            base_depth: base_depth,
                            extend_depth: extend_depth,
                            extend_check: 1,
                            extend_threatmate: 1,
                            rng: &mut rng
                        };

                        let strategy = Inter::new();

                        match strategy.search(&mut env, &mut gs, &evalutor, &mvs) {
                            Ok(EvaluationResult::Exact(score, mvs, zh, seldepth)) => {
                                pv = mvs.clone();

                                if score > best_score {
                                    best_score = score;
                                }

                                let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
                            },
                            Ok(EvaluationResult::NodeLimits) => {
                                let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                                break;
                            },
                            Ok(EvaluationResult::Timeout) => {
                                let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                                break;
                            },
                            Ok(EvaluationResult::Repetition) => {
                                let _ = sender.send(Ok(RootEvaluationResult::Repetition));
                                break;
                            },
                            Ok(EvaluationResult::Stop) => {
                                let _ = sender.send(Ok(RootEvaluationResult::Stop));
                                break;
                            },
                            Ok(EvaluationResult::Cut) => {
                                let _ = sender.send(Err(ApplicationError::LogicError(String::from("The root node has been pruned."))));
                                break;
                            }
                            Err(e) => {
                                let _ = sender.send(Err(e));
                                break;
                            }
                        }
                    }

                    if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) {
                        break;
                    }

                    shared_depth.fetch_add(1, Ordering::Release);
                }

                let _ = sender.send(Ok(RootEvaluationResult::Quit(env.move_orderer,thread_index)));

                env.abort.store(true,Ordering::Release);
            });
        } else {
            let pv = VecDeque::new();

            let self_partial_output = gs.self_partial_output.clone();
            let opponent_partial_output = gs.opponent_partial_output.clone();

            self.thread_pool.spawn(move || {
                let mut rng = rand::thread_rng();

                let mut depth = shared_depth.load(Ordering::Acquire) as u32;

                while depth <= base_depth {
                    let len = mvs.len();

                    let search_offset = if len == 0 {
                        0
                    } else {
                        (thread_index * 7 + (rng.gen::<usize>() % len)) % len
                    };

                    let mut gs = GameState {
                        teban: teban,
                        state: &state,
                        alpha: Score::default(),
                        beta: Score::INFINITE(0),
                        search_offset: search_offset,
                        best_score: best_score,
                        m: None,
                        static_eval: LazyEval::new(),
                        prev_kind: KomaKind::Blank,
                        move_history: &mut Vec::new(),
                        self_partial_output:Arc::clone(&self_partial_output),
                        opponent_partial_output:Arc::clone(&opponent_partial_output),
                        thread_index:thread_index,
                        pv:&pv,
                        mc: &mc,
                        zh: zh.clone(),
                        depth: depth,
                        current_depth: current_depth,
                        cut_node: false,
                        already_reduced_lmr: false,
                        nmp_min_ply: None,
                        base_depth: base_depth,
                        extend_depth: extend_depth,
                        extend_check: 1,
                        extend_threatmate: 1,
                        rng:&mut rng
                    };

                    let strategy = Inter::new();

                    let r = strategy.search(&mut env, &mut gs, &evalutor, &mvs);

                    match r {
                        Ok(EvaluationResult::Exact(score, mvs, zh, seldepth)) => {
                            if score > best_score {
                                best_score = score;
                            }

                            let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
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
                        Ok(EvaluationResult::Cut) => {
                            let _ = sender.send(Err(ApplicationError::LogicError(String::from("The root node has been pruned."))));
                        },
                        Err(e) => {
                            let _ = sender.send(Err(e));
                        }
                    }

                    if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) {
                        break;
                    }

                    std::thread::yield_now();

                    depth = (depth + 1).max(shared_depth.load(Ordering::Acquire) as u32);
                }

                let _ = sender.send(Ok(RootEvaluationResult::Quit(env.move_orderer,thread_index)));
            });
        }
    }

    fn termination(&self,env:&mut Environment<L,S>,mut busy_threads:u32,move_orderers: &mut Vec<MoveOrderer<UnusedQuietSee>>) -> Result<(),ApplicationError> {
        env.abort.store(true,Ordering::Release);

        let mut last_error = None;

        while busy_threads > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e))? {
                Err(e) => {
                    last_error = Some(e);
                },
                Ok(RootEvaluationResult::Quit(move_orderer,thread_index)) => {
                    busy_threads -= 1;
                    move_orderers[thread_index] = move_orderer;
                },
                _ => ()
            }
        }

        env.info_sender.flush()?;

        match last_error {
            Some(e) => Err(e),
            None => Ok(())
        }
    }

    pub fn compute_aspiration_window_delta(depth:u32) -> i32 {
        FU_SCORE * (2 + depth as i32 / 2)
    }

    pub fn choose_result(&self, pv_result:&mut [Option<EvaluationResult>], pv_depth: usize,
                         _: &mut [Option<EvaluationResult>], _: usize) -> Option<EvaluationResult> {
        //if worker_depth > pv_depth && worker_result[worker_depth].as_ref().and_then(|wr| {
        //    pv_result[pv_depth].as_ref().map(|pr| {
        //        wr.best_score().and_then(|ws| pr.best_score().map(|ps| ws >= ps)).unwrap_or(false)
        //    }).or(Some(true))
        //}).unwrap_or(false) {
        //    worker_result[worker_depth].take()
        //} else {
            pv_result[pv_depth].take()
        //}
    }
    pub fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     _:&mut UserEventDispatcher<'b,Root<L,S,M>,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>,move_orderers: &mut Vec<MoveOrderer<UnusedQuietSee>>) -> Result<EvaluationResult,ApplicationError> {
        let base_depth = gs.base_depth;
        let max_depth = base_depth as usize + 2;
        let mut pv_result = vec![None;max_depth+1];
        let mut worker_result = vec![None;max_depth+1];
        let mut pv_depth = 0;
        let mut worker_depth = 0;
        let mut pv_best_score = vec![Score::default();max_depth+1];
        let mut worker_best_score = vec![Score::default();max_depth+1];

        let shared_depth = Arc::new(AtomicUsize::new(1));

        let mut picker = RandomPicker::new(Prng::new(gs.rng.gen()));

        let mut mvs = Vec::new();

        if Rule::in_check(gs.teban,&gs.state) {
            Rule::generate_moves::<Evasions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
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
        };

        let mvs = Arc::new(mvs);

        env.abort.store(false,Ordering::Release);

        for i in 0..env.max_threads {
            let mvs = Arc::clone(&mvs);

            self.start_thread(i as usize,
                              &shared_depth,
                              env,gs,mvs,
                              evalutor,
                              move_orderers[i as usize].clone());

        }

        let mut busy_threads = env.max_threads;

        while busy_threads > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e))? {
                Ok(RootEvaluationResult::Exact(s, mvs, zh, depth, seldepth, thread_index)) => {
                    if let Err(e) = env.info_sender.flush() {
                        let _ = env.on_error_handler.lock().map(|h| h.call(&e));
                    }

                    if thread_index == 0 {
                        match pv_result[depth as usize] {
                            Some(EvaluationResult::Exact(bs, _, _, _)) if bs > pv_best_score[depth as usize] => {
                                pv_best_score[depth as usize] = bs;
                                pv_result[depth as usize] = Some(EvaluationResult::Exact(s, mvs, zh, seldepth));
                            },
                            None => {
                                pv_result[depth as usize] = Some(EvaluationResult::Exact(s, mvs, zh, seldepth));
                            },
                            _ => ()
                        }

                        if depth > pv_depth {
                            pv_depth = depth;
                        }
                    } else {
                        match worker_result[depth as usize] {
                            Some(EvaluationResult::Exact(bs, _, _, _)) if bs > worker_best_score[depth as usize] => {
                                worker_best_score[depth as usize] = bs;
                                worker_result[depth as usize] = Some(EvaluationResult::Exact(s, mvs, zh, seldepth));
                            },
                            None => {
                                worker_result[depth as usize] = Some(EvaluationResult::Exact(s, mvs, zh, seldepth));
                            },
                            _ => ()
                        }

                        if depth > worker_depth {
                            worker_depth = depth;
                        }
                    }

                    self.send_message(env, format!("pv_depth = {}, worker_depth = {}", pv_depth, worker_depth).as_str())?;
                },
                Ok(RootEvaluationResult::NodeLimits) => {
                    self.termination(env, busy_threads, move_orderers)?;

                    return Ok(self.choose_result(&mut pv_result, pv_depth as usize, &mut worker_result, worker_depth as usize).unwrap_or(EvaluationResult::NodeLimits));
                },
                Ok(RootEvaluationResult::Timeout) => {
                    self.termination(env, busy_threads, move_orderers)?;

                    return Ok(self.choose_result(&mut pv_result, pv_depth as usize, &mut worker_result, worker_depth as usize).unwrap_or(EvaluationResult::Timeout));
                },
                Ok(RootEvaluationResult::Stop) => {
                    self.termination(env, busy_threads, move_orderers)?;

                    return Ok(self.choose_result(&mut pv_result, pv_depth as usize, &mut worker_result, worker_depth as usize).unwrap_or(EvaluationResult::Stop));
                },
                Ok(RootEvaluationResult::Repetition) => {
                    self.termination(env, busy_threads, move_orderers)?;

                    return Err(ApplicationError::LogicError(String::from(
                        "A Repetition was returned at the root node."
                    )));
                },
                Ok(RootEvaluationResult::Quit(move_orderer,thread_index)) => {
                    busy_threads -= 1;
                    move_orderers[thread_index] = move_orderer;
                },
                Err(e) => {
                    self.termination(env, busy_threads, move_orderers)?;

                    return Err(e);
                }
            }
        }

        self.choose_result(&mut pv_result, pv_depth as usize, &mut worker_result, worker_depth as usize).ok_or(
            ApplicationError::LogicError(String::from(
                "No search results"
            ))
        )
    }
}
impl<L,S,M> SendInfo<L,S,M> for Root<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {}
pub struct Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>,
}
impl<L,S,M> Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
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

    pub fn futility_margin(&self,depth:u32,m:LegalMove) -> i32 {
        if m.is_nari() {
            81 + 40 * (depth as i32)
        } else {
            160 + 40 * (depth as i32)
        }
    }

    pub fn search_child_node<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                     m:LegalMove,pv:&VecDeque<LegalMove>,
                                     alpha:Score,
                                     depth:u32,
                                     cut_node:bool,
                                     lmr_reduced:bool,
                                     nmp_min_ply:Option<u32>,
                                     event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        let o = match m {
            LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
            _ => None
        };

        let mut depth = depth;
        let mut extend_depth = gs.extend_depth;
        let mut extend_check = gs.extend_check;
        let extend_threatmate = gs.extend_threatmate;

        let piece_index = env.move_orderer.calc_piece_us_index(gs.teban,env.move_orderer.calc_piece_index(gs.teban,gs.state,m)?)?;

        let zh = gs.zh.updated(&env.hasher, gs.teban, gs.state.get_banmen(), gs.mc, m.to_applied_move(), &o);

        let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

        match next {
            (state, mc, _) => {
                let self_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(gs.teban, gs.teban,&gs.state,gs.mc,&state,&mc,m,Arc::clone(&gs.self_partial_output))?);
                let opponent_partial_output = Arc::new(evalutor.prepare_evalute_by_diff(gs.teban, gs.teban.opposite(),&gs.state,gs.mc,&state,&mc,m,Arc::clone(&gs.opponent_partial_output))?);

                let static_eval = LazyEval::new();

                if extend_depth > 0 {
                    if extend_check > 0 && Rule::in_check(gs.teban.opposite(),&state) {
                        depth += 1;
                        extend_depth -= 1;
                        extend_check -= 1;
                    }/* else if extend_threatmate > 0 &&
                        (self.in_danger(gs.teban.opposite(),&state, m) || self.is_threat(gs.teban,&state,m)) {
                        depth += 1;
                        extend_depth -= 1;
                        extend_threatmate -= 1;
                    }*/
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
                    static_eval:static_eval,
                    prev_kind: prev_kind,
                    thread_index:gs.thread_index,
                    pv:pv,
                    move_history: gs.move_history,
                    self_partial_output:Arc::clone(&opponent_partial_output),
                    opponent_partial_output:Arc::clone(&self_partial_output),
                    mc: &mc,
                    zh: zh.clone(),
                    depth: depth - 1,
                    current_depth: gs.current_depth + 1,
                    cut_node: cut_node,
                    already_reduced_lmr: lmr_reduced,
                    nmp_min_ply: nmp_min_ply,
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
            static_eval: LazyEval::new(),
            prev_kind: Blank,
            thread_index:gs.thread_index,
            pv:&VecDeque::new(),
            move_history: gs.move_history,
            self_partial_output:Arc::clone(&gs.opponent_partial_output),
            opponent_partial_output:Arc::clone(&gs.self_partial_output),
            mc: &mc,
            zh: zh.clone(),
            depth: depth,
            current_depth: gs.current_depth + 1,
            cut_node: false,
            already_reduced_lmr:gs.already_reduced_lmr,
            nmp_min_ply: gs.nmp_min_ply,
            base_depth: gs.base_depth,
            extend_depth: gs.extend_depth,
            extend_check: gs.extend_check,
            extend_threatmate: gs.extend_threatmate,
        };

        let strategy = Recursive::new();

        strategy.search(env, &mut gs, event_dispatcher, evalutor)
    }
}
impl<L,S,M> SendInfo<L,S,M> for Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {}
impl<L,S,M> Search<L,S,M> for Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
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

        let mut static_eval = gs.static_eval;

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

        let prev_move = gs.m.clone();

        if Rule::in_check(gs.teban.opposite(),&gs.state) {
            if let Some(m) = prev_move.clone() {
                env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE(0),Bound::Exact,None);

                let mut mvs = VecDeque::new();

                mvs.push_front(m);

                return Ok(EvaluationResult::Exact(Score::INFINITE(-(gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
            }
        }

        if let Some(prev_move) = gs.m.clone() {
            let r = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: d,
                            score: s,
                            bound,
                            best_move: _
                        }) = r {

                if (bound == Bound::Exact && d as u32 >= gs.depth) ||
                   (bound == Bound::LowerBound && d as u32 >= gs.depth && s >= gs.beta) ||
                   (bound == Bound::UpperBound && d as u32 >= gs.depth && s <= gs.alpha) {
                    let mut mvs = VecDeque::new();

                    mvs.push_front(prev_move);

                    return Ok(EvaluationResult::Exact(s.normalize_mate(gs.current_depth as i32), mvs, gs.zh.clone(), gs.current_depth));
                }
            }
        }

        let in_check = Rule::in_check(gs.teban,&gs.state);

        if gs.depth == 0 && gs.current_depth >= 5 && !in_check {
            let picker = RandomPicker::new(Prng::new(gs.rng.gen()));

            let checkmate = self.threatmate_search(gs.teban,
                                                   gs.teban,
                                                   gs.state,
                                                   &gs.mc,
                                                   env,
                                                   event_dispatcher,
                                                   &gs.zh,
                                                   &mut HashSet::new(),
                                                   env.threatmate_depth as usize,
                                                   gs.current_depth as usize,
                                                   picker,
                                                   gs.rng)?;

            if let MateDepth::Mate(ply) = checkmate {
                env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE(-(ply - gs.current_depth as i32)),Bound::Exact,None);

                let mut mvs = VecDeque::new();

                gs.m.map(|m| mvs.push_front(m));

                return Ok(EvaluationResult::Exact(Score::INFINITE(-ply), mvs, gs.zh.clone(),gs.current_depth));
            }
        }

        if gs.depth == 0 {
            let s = self.qsearch(gs.teban,
                                 &gs.state,
                                 &gs.mc,
                                 env,
                                 event_dispatcher,
                                 &gs.zh,
                                 &mut HashSet::new(),
                                 Arc::clone(&gs.self_partial_output),
                                 Arc::clone(&gs.opponent_partial_output),
                                 gs.alpha,
                                 gs.beta,
                                 0,
                                 gs.current_depth as usize,
                                 prev_move.clone(),
                                 evalutor,
                                 gs.rng)?;

            let mut mvs = VecDeque::new();

            prev_move.map(|m| mvs.push_front(m));

            if env.stop.load(Ordering::Acquire) {
                return Ok(EvaluationResult::Stop);
            } else {
                return Ok(EvaluationResult::Exact(s, mvs, gs.zh.clone(), gs.current_depth));
            }
        }

        // Razoring
        if gs.depth >= 1 && gs.pv.is_empty() {
            if Score::Value(static_eval.get_or_insert_with(|| {
                evalutor.evalute(&gs.self_partial_output)
            })?) < gs.alpha - 514 - 294 * gs.depth as i32 * gs.depth as i32 {
                let s = self.qsearch(gs.teban,
                                     &gs.state,
                                     &gs.mc,
                                     env,
                                     event_dispatcher,
                                     &gs.zh,
                                     &mut HashSet::new(),
                                     Arc::clone(&gs.self_partial_output),
                                     Arc::clone(&gs.opponent_partial_output),
                                     gs.alpha,
                                     gs.beta,
                                     0,
                                     gs.current_depth as usize,
                                     prev_move.clone(),
                                     evalutor,
                                     gs.rng)?;

                if s <= gs.alpha {
                    let mut mvs = VecDeque::new();

                    prev_move.map(|m| mvs.push_front(m));

                    if env.stop.load(Ordering::Acquire) {
                        return Ok(EvaluationResult::Stop);
                    } else {
                        return Ok(EvaluationResult::Exact(s, mvs, gs.zh.clone(), gs.current_depth));
                    }
                }
            }
        }

        if let Score::Value(beta) = gs.beta {
            if gs.pv.is_empty() && gs.cut_node && static_eval.get_or_insert_with(|| {
                evalutor.evalute(&gs.self_partial_output)
            })? >= beta - 18 * gs.depth as i32 + 390 && gs.current_depth >= gs.nmp_min_ply.unwrap_or(0) {
                //let r = 7 + gs.depth / 3;
                let r = 3 + gs.depth / 3;

                match self.search_null_move(env, gs, -gs.beta, -gs.beta + 1, gs.depth.saturating_sub(r), event_dispatcher, evalutor)? {
                    EvaluationResult::Exact(s, _, zh, _) => {
                        let s = -s;

                        let null_value = s;

                        let mut best_moves = VecDeque::new();

                        gs.m.map(|m| best_moves.push_front(m));

                        if s >= gs.beta {
                            if let Score::Value(_) = s {
                                if gs.nmp_min_ply.unwrap_or(0) == 0 || gs.depth < 16 {
                                    return Ok(EvaluationResult::Exact(s, best_moves, zh, gs.current_depth));
                                }

                                let nmp_min_ply = (gs.current_depth as i32 + 3 * (gs.depth as i32 - r as i32) / 4).max(0) as u32;

                                let mut gs = GameState {
                                    teban: gs.teban,
                                    state: &gs.state,
                                    rng: gs.rng,
                                    alpha: Score::Value(beta - 1),
                                    beta: Score::Value(beta),
                                    search_offset: gs.search_offset,
                                    best_score: gs.best_score,
                                    m: gs.m,
                                    static_eval: static_eval.clone(),
                                    prev_kind: gs.prev_kind,
                                    thread_index: gs.thread_index,
                                    pv: &VecDeque::new(),
                                    move_history: gs.move_history,
                                    self_partial_output: Arc::clone(&gs.self_partial_output),
                                    opponent_partial_output: Arc::clone(&gs.opponent_partial_output),
                                    mc: gs.mc,
                                    zh: gs.zh.clone(),
                                    depth: gs.depth.saturating_sub(r),
                                    current_depth: gs.current_depth,
                                    cut_node: false,
                                    already_reduced_lmr: gs.already_reduced_lmr,
                                    nmp_min_ply: Some(nmp_min_ply),
                                    base_depth: gs.base_depth,
                                    extend_depth: gs.extend_depth,
                                    extend_check: gs.extend_check,
                                    extend_threatmate: gs.extend_threatmate,
                                };

                                let strategy = Recursive::new();

                                match strategy.search(env, &mut gs, event_dispatcher, evalutor)? {
                                    EvaluationResult::Exact(s, _, _, _) => {
                                        if s >= gs.beta {
                                            return Ok(EvaluationResult::Exact(null_value, best_moves, gs.zh, gs.current_depth));
                                        }
                                    },
                                    EvaluationResult::NodeLimits => {
                                        env.history.remove(&(gs.teban, mk, sk));

                                        return Ok(EvaluationResult::NodeLimits);
                                    },
                                    EvaluationResult::Timeout => {
                                        env.history.remove(&(gs.teban, mk, sk));

                                        return Ok(EvaluationResult::Timeout);
                                    },
                                    EvaluationResult::Stop => {
                                        env.history.remove(&(gs.teban, mk, sk));

                                        return Ok(EvaluationResult::Stop);
                                    },
                                    EvaluationResult::Repetition | EvaluationResult::Cut => {}
                                }
                            }
                        }
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
                    EvaluationResult::Repetition | EvaluationResult::Cut => {
                    }
                }
            }
        }

        let start_alpha = gs.alpha;
        let mut alpha = gs.alpha;
        let mut quiet_alpha = gs.alpha;

        let beta = gs.beta;

        let mut scoreval = Score::default();
        let mut best_moves = VecDeque::new();

        let mut picker = RandomPicker::new(Prng::new(gs.rng.gen()));

        let count = if Rule::in_check(gs.teban,&gs.state) {
            1
        } else {
            2
        };

        env.history.insert((gs.teban,mk,sk));

        let pv_non = VecDeque::new();

        let mut quiet_moves = Vec::with_capacity(593);

        let tt_move = if let Some(TTPartialEntry {
                                  depth: d,
                                  score: _,
                                  bound: _,
                                  best_move: m
                              }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            if d as u32 >= gs.depth.saturating_sub(2) {
                m
            } else {
                None
            }
        } else {
            None
        };

        let pv_move = if gs.pv.len() > gs.current_depth as usize {
            Some(gs.pv[gs.current_depth as usize])
        } else {
            None
        };

        let mut lmr_index = 0;

        let mut max_seldepth = gs.current_depth;

        for i in 0..count {
            if i == 0 && in_check {
                Rule::generate_moves::<Evasions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            } else if i == 0 {
                Rule::generate_moves::<CaptureOrPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            } else {
                Rule::generate_moves::<QuietsWithoutPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            }

            for (m,_) in env.move_orderer.ordering(
                &mut picker, gs.current_depth, gs.teban, &gs.state, tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history)? {

                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                // Futility Pruning
                if !gs.cut_node && gs.depth >= 2 && gs.depth <= 4 &&
                    m.obtained().is_none() &&
                    !pv_move.map(|pm| pm == m).unwrap_or(false) &&
                    !Rule::in_check(gs.teban,&gs.state) &&
                    !Rule::is_oute_move(&gs.state,gs.teban,m) &&
                    Score::Value(gs.static_eval.get_or_insert_with(|| {
                        evalutor.evalute(&gs.self_partial_output)
                    })? + self.futility_margin(gs.depth,m)) <= alpha {

                    continue;
                }

                /*
                if !in_check && !m.is_nari() {
                    if let Some(o) = m.obtained() {
                        if !prev_move.map(|pm| {
                            pm.obtained().is_some() && m.dst() == pm.dst()
                        }).unwrap_or(false) && !Rule::is_oute_move(gs.state,gs.teban,m) {
                            if see < -CAPTURED_SCORE_MAP[o as usize] * 3 / 4 {
                                continue;
                            }
                        }
                    }
                }

                 */

                let mut r = self.calc_lmr(env,
                              &mut lmr_index,
                              gs.depth,
                              gs.current_depth,
                              gs.teban,
                              gs.state,
                              m,
                              tt_move.as_ref(),
                              pv_move.as_ref())?;

                if gs.already_reduced_lmr {
                    r = r.saturating_sub(1);
                }

                let mut lmr_reduced = gs.already_reduced_lmr || r > 0;

                for k in 0..2 {
                    let depth = if k == 0 {
                        gs.depth - r
                    } else {
                        gs.depth
                    };

                    if self.is_obtained_ou(m)? {
                        env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE(0),Bound::Exact,Some(m));

                        let mut mvs = VecDeque::new();

                        mvs.push_front(m);
                        prev_move.map(|m| mvs.push_front(m));
                        env.history.remove(&(gs.teban,mk,sk));

                        return Ok(EvaluationResult::Exact(Score::INFINITE(-(gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
                    }

                    let pv = if pv_move.map(|pm| pm == m).unwrap_or(false) {
                        gs.pv
                    } else {
                        &pv_non
                    };

                    match self.search_child_node(env,gs,m,pv,alpha,depth,gs.cut_node,lmr_reduced,gs.nmp_min_ply,event_dispatcher,evalutor)? {
                        EvaluationResult::Exact(s, mvs, _, seldepth) => {
                            let s = -s;

                            if r > 0 && s > alpha {
                                r = 0;
                                lmr_reduced = gs.already_reduced_lmr;
                                continue
                            }

                            max_seldepth = max_seldepth.max(seldepth);

                            if s > scoreval {
                                scoreval = s;

                                best_moves = mvs;

                                if scoreval >= beta {
                                    match scoreval {
                                        Score::INFINITE(_) => {
                                            env.transposition_table.update(&gs.zh,depth as i8,Score::INFINITE(-(gs.depth as i32)),Bound::Exact,Some(m));
                                        },
                                        Score::NEGINFINITE(_) => {
                                            env.transposition_table.update(&gs.zh,depth as i8,Score::NEGINFINITE(gs.depth as i32),Bound::Exact,Some(m));
                                        },
                                        _ => {
                                            env.transposition_table.update(&gs.zh,depth as i8,scoreval,Bound::LowerBound,Some(m));
                                        }
                                    }

                                    match m {
                                        LegalMove::To(mv) if mv.obtained().is_none() => {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;

                                                let _ = prev_move.map(|prev_move| {
                                                    env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                }).unwrap_or(Ok(()))?;
                                            }

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,gs.depth,gs.current_depth,gs.move_history)?;
                                        },
                                        LegalMove::Put(_) => {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;

                                            let _ = prev_move.map(|prev_move| {
                                                env.move_orderer.update_counter_move(m,gs.teban,prev_move,gs.prev_kind)
                                            }).unwrap_or(Ok(()))?;

                                            env.move_orderer.update_improve_history(gs.teban,&gs.state,m,gs.depth,gs.current_depth,gs.move_history)?;
                                        },
                                        _ => ()
                                    };
                                    env.history.remove(&(gs.teban,mk,sk));

                                    prev_move.map(|m| best_moves.push_front(m));

                                    return Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth));
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
                        EvaluationResult::Cut => {
                        },
                        EvaluationResult::Repetition => {
                        }
                    }
                }
            }
        }

        if quiet_alpha == start_alpha {
            for m in quiet_moves {
                env.move_orderer.update_degrade_history(gs.teban,&gs.state,m,gs.depth)?;
            }
        }

        let bs = scoreval.localize_mate(gs.current_depth as i32);

        if scoreval <= start_alpha {
            env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::UpperBound, None);
        } else {
            env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::Exact, best_moves.front().map(|m| m.clone()));
        }

        env.history.remove(&(gs.teban,mk,sk));

        prev_move.map(|m| best_moves.push_front(m));

        Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
    }
}
pub struct Inter<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
             PreTrain<f32> + Send + Sync + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>
}
impl<L,S,M> Inter<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
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
                                    cut_node:bool,
                                    lmr_reduced:bool,
                                    nmp_min_ply:Option<u32>,
                                    event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                    evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        let search = Recursive::new();

        Ok(search.search_child_node(env,gs,m,pv,alpha,depth,cut_node,lmr_reduced,nmp_min_ply,event_dispatcher,evalutor)?)
    }
}
impl<L,S,M> PartialSearch<L,S,M> for Inter<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
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

        let start_alpha = gs.alpha;
        let mut alpha = gs.alpha;
        let mut quiet_alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::default();
        let mut best_moves = VecDeque::new();

        env.history.insert((gs.teban,mk,sk));

        let tt_move = if let Some(TTPartialEntry {
                        depth: d,
                        score: _,
                        bound: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            if d as u32 >= gs.depth.saturating_sub(2) {
                m
            } else {
                None
            }
        } else {
            None
        };

        let pv_move = if gs.pv.len() > gs.current_depth as usize {
            Some(gs.pv[gs.current_depth as usize])
        } else {
            None
        };

        let pv_non = VecDeque::new();

        let mut quiet_moves = Vec::with_capacity(593);

        let mut quiet_index = 0;

        let mut max_seldepth = 0;

        for (m,_) in env.move_orderer.ordering(
            mvs.iter().cloned(), gs.current_depth, gs.teban, &gs.state, tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history
        )?.skip(gs.search_offset) {
            if m.obtained().is_none() {
                quiet_moves.push(m);
            }

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
                               &mut quiet_index,
                               gs.depth,
                               gs.current_depth,
                               gs.teban,
                               gs.state,
                               m,
                               tt_move.as_ref(),
                               pv_move.as_ref())?;

            if gs.already_reduced_lmr {
                r = r.saturating_sub(1);
            }

            let mut lmr_reduced = gs.already_reduced_lmr || r > 0;

            for j in 0..2 {
                let depth = if j == 0 {
                    gs.depth - r
                } else {
                    gs.depth
                };

                if self.is_obtained_ou(m)? {
                    env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE(0),Bound::Exact,Some(m));

                    let mut mvs = VecDeque::new();

                    mvs.push_front(m);
                    env.history.remove(&(gs.teban,mk,sk));

                    return Ok(EvaluationResult::Exact(Score::INFINITE(-(gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
                }

                let pv = if pv_move.map(|pm| pm == m).unwrap_or(false) {
                    gs.pv
                } else {
                    &pv_non
                };

                match recur.search_child_node(env,gs,m,pv,alpha,depth,gs.cut_node,lmr_reduced,gs.nmp_min_ply,&mut event_dispatcher,evalutor)? {
                    EvaluationResult::Exact(s, mvs, _, seldepth) => {
                        let s = -s;

                        if r > 0 && s > alpha {
                            r = 0;
                            lmr_reduced = gs.already_reduced_lmr;
                            continue;
                        }

                        max_seldepth = max_seldepth.max(seldepth);

                        if s > scoreval {
                            scoreval = s;

                            best_moves = mvs;

                            if gs.thread_index == 0 {
                                recur.send_info(env, gs.depth, max_seldepth, &best_moves, &scoreval)?;
                            }

                            if scoreval >= beta {
                                match scoreval {
                                    Score::INFINITE(_) => {
                                        env.transposition_table.update(&gs.zh,gs.depth as i8,Score::INFINITE(-(gs.depth as i32)),Bound::Exact,Some(m));
                                    },
                                    Score::NEGINFINITE(_) => {
                                        env.transposition_table.update(&gs.zh,gs.depth as i8,Score::NEGINFINITE(gs.depth as i32),Bound::Exact,Some(m));
                                    },
                                    _ => {
                                        env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval,Bound::LowerBound,Some(m));
                                    }
                                }

                                match m {
                                    LegalMove::To(mv) if mv.obtained().is_none() => {
                                        if !mv.is_nari() {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;
                                        }
                                        env.move_orderer.update_improve_history(gs.teban,&gs.state,m,gs.depth,gs.current_depth,gs.move_history)?;
                                    },
                                    LegalMove::Put(_) => {
                                        env.move_orderer.update_killer(gs.current_depth, m)?;
                                        env.move_orderer.update_improve_history(gs.teban,&gs.state,m,gs.depth,gs.current_depth,gs.move_history)?;
                                    },
                                    _ => ()
                                };

                                env.history.remove(&(gs.teban,mk,sk));

                                return Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth));
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
                    EvaluationResult::Cut => {
                    },
                    EvaluationResult::Repetition => {
                    }
                }
            }
        }

        if quiet_alpha == start_alpha {
            for m in quiet_moves {
                env.move_orderer.update_degrade_history(gs.teban, &gs.state, m, gs.depth)?;
            }
        }

        let bs = scoreval.localize_mate(gs.current_depth as i32);

        if gs.search_offset != 0 || scoreval <= start_alpha {
            env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::UpperBound,best_moves.front().map(|m| m.clone()));
        } else {
            env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::Exact,best_moves.front().map(|m| m.clone()));
        }

        env.history.remove(&(gs.teban,mk,sk));

        Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
    }
}