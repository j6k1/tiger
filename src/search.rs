use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::{Deref, Neg};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use nncombinator::arr::Arr;
use nncombinator::layer::{ContinueForward, ForwardAll, PartialForward, PreTrain};
use parking_lot::RwLock;
use rand::Rng;
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
use usiagent::position::Position;
use usiagent::rule::{CaptureOrPawnPromotions, Checks, Evasions, LegalMove, QuietsWithoutPawnPromotions, Rule, SquareToPoint, State};
use usiagent::see::calc_see;
use usiagent::shogi::{KomaKind, MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use usiagent::shogi::KomaKind::{Blank, GHishaN, GKakuN, SHishaN, SKakuN};
use crate::error::ApplicationError;
use crate::features::{HalfKP, HalfKPDiff};
use crate::math::SignFloat;
use crate::nn::{Evalutor, FEATURES_NUM};
use crate::transposition_table::{TT, ZobristHash, TTPartialEntry, Bound, Score, LocalizeScore, NormalizeScore, ExactScoreBound, TTScore};

pub const TURN_LIMIT:u32 = 1000;
pub const BASE_DEPTH:u32 = 20;
pub const MAX_THREADS:u32 = 2;
pub const THREATMATE_DEPTH:u32 = 7;

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub enum ThreatMateSearchResult {
    Checkmated(i32),
    Unknown,
    Repetition,
    Checkmate(i32)
}
#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub enum ThreatMateSearchResultRelative {
    Checkmated(i32),
    Unknown,
    Checkmate(i32)
}
impl Neg for ThreatMateSearchResult {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            ThreatMateSearchResult::Checkmate(ply) => ThreatMateSearchResult::Checkmated(-ply),
            ThreatMateSearchResult::Unknown => ThreatMateSearchResult::Unknown,
            ThreatMateSearchResult::Repetition => ThreatMateSearchResult::Repetition,
            ThreatMateSearchResult::Checkmated(ply) => ThreatMateSearchResult::Checkmate(-ply)
        }
    }
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
    pub lazy_abort:Arc<AtomicBool>,
    pub history:HashSet<(Teban,u64,u64)>,
    pub transposition_table:Arc<TT<u64,TTScore,{1<<20},4>>,
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
            lazy_abort:Arc::clone(&self.lazy_abort),
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
    Repetition(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32)
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
    LazyAbort(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32, u32, usize),
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
               transposition_table: &Arc<TT<u64,TTScore,{1 << 20},4>>
    ) -> Environment<L,S> {
        let abort = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));
        let lazy_abort = Arc::new(AtomicBool::new(false));

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
            lazy_abort:lazy_abort,
            history:history,
            transposition_table:Arc::clone(transposition_table),
            move_orderer:MoveOrderer::<UnusedQuietSee>::new(base_depth as usize + 2),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub const UNDO_BUFFER_SIZE:usize = 200;

pub struct GameState<'a> {
    pub teban:Teban,
    pub pos:&'a mut Position<UNDO_BUFFER_SIZE>,
    pub rng:&'a mut Prng,
    pub alpha:Score,
    pub beta:Score,
    pub search_offset:usize,
    pub best_score:Score,
    pub m:Option<LegalMove>,
    pub static_eval: LazyEval,
    pub gives_check_us:bool,
    pub gives_check_them:bool,
    pub prev_kind:KomaKind,
    pub move_history:&'a mut Vec<Option<(u8,u8)>>,
    pub threatmate_cache:&'a mut HashMap<(Teban,u64,u64),(ThreatMateSearchResultRelative,u32)>,
    pub self_partial_output: &'a Arr<f32,{256*2}>,
    pub opponent_partial_output: &'a Arr<f32,{256*2}>,
    pub thread_index:usize,
    pub pv:&'a VecDeque<LegalMove>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChecksMoveOrder {
    Promote,
    Capture,
    Major,
    Seal,
    Keep,
    Block(u8)
}
pub struct ChecksMoveOrderer {

}
impl ChecksMoveOrderer {
    pub fn new() -> ChecksMoveOrderer {
        ChecksMoveOrderer {}
    }

    pub fn ordering<I: Iterator<Item=LegalMove>>(&self,teban:Teban,
                                                 state:&State,
                                                 mc:&MochigomaCollections,
                                                 it:I) -> impl Iterator<Item=LegalMove> {
        let mut mvs = Vec::new();

        for m in it {
            if m.is_nari() {
                mvs.push((ChecksMoveOrder::Promote,m));
            } else if self.is_capture_order(teban,state,m) {
                mvs.push((ChecksMoveOrder::Capture,m));
            } else if self.is_major_order(teban,state,m) {
                mvs.push((ChecksMoveOrder::Major,m));
            } else if self.is_seal_order(teban,state,m) {
                mvs.push((ChecksMoveOrder::Seal,m));
            //} else if let Some(c) = self.blocker_count(teban,state,mc,m) {
            //    mvs.push((ChecksMoveOrder::Block(c),m));
            } else {
                mvs.push((ChecksMoveOrder::Keep,m));
            }
        }
        mvs.sort_by(|a,b| a.0.cmp(&b.0));

        mvs.into_iter().map(|(_,m)| m)
    }

    fn is_capture_order(&self,teban:Teban,state:&State,m:LegalMove) -> bool {
        if m.obtained().is_none() {
            false
        } else if let LegalMove::To(mv) = m {
            let part = state.get_part();

            let to_mask = 1 << (mv.dst() + 1);

            if teban == Teban::Sente {
                let surrounding_mask = Rule::gen_ou_surrounding_mask(teban.opposite(),part);

                if to_mask & (part.gote_kyou_board & !part.gote_nari_board) != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_kyou(part.sente_opponent_board,part.sente_self_board,mv.dst()).reverse() != 0
                } else if to_mask & part.gote_kaku_board != 0 && to_mask & part.gote_nari_board != 0 {
                    surrounding_mask & (Rule::gen_control_bits_by_kaku(
                        part.gote_self_board,part.gote_opponent_board,
                        part.sente_opponent_board,part.sente_self_board,80 - mv.dst()) |
                        Rule::gen_control_bits(80 - mv.dst(),GKakuN)).reverse() != 0
                } else if to_mask & part.gote_kaku_board != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_kaku(
                        part.gote_self_board,part.gote_opponent_board,
                        part.sente_opponent_board,part.sente_self_board,80 - mv.dst()).reverse() != 0
                } else if to_mask & part.gote_hisha_board != 0 && to_mask & part.gote_nari_board != 0 {
                    surrounding_mask & (Rule::gen_control_bits_by_hisha(
                        part.gote_self_board,part.gote_opponent_board,
                        part.sente_opponent_board,part.sente_self_board,80 - mv.dst()) |
                        Rule::gen_control_bits(80 - mv.dst(),GHishaN)).reverse() != 0
                } else if to_mask & part.gote_hisha_board != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_hisha(
                        part.gote_self_board,part.gote_opponent_board,
                        part.sente_opponent_board,part.sente_self_board,80 - mv.dst()).reverse() != 0
                } else {
                    let (x,y) = mv.dst().square_to_point();

                    surrounding_mask & Rule::gen_control_bits(
                        80 - mv.dst(), state.get_banmen()[y as usize][x as usize]
                    ).reverse() != 0
                }
            } else {
                let surrounding_mask = Rule::gen_ou_surrounding_mask(teban.opposite(), part);

                if to_mask & (part.sente_kyou_board & !part.sente_nari_board) != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_kyou(part.gote_opponent_board,part.gote_self_board,80 - mv.dst()) != 0
                } else if to_mask & part.sente_kaku_board != 0 && to_mask & part.sente_nari_board != 0 {
                    surrounding_mask & (Rule::gen_control_bits_by_kaku(
                        part.sente_self_board, part.sente_opponent_board,
                        part.gote_opponent_board, part.gote_self_board, mv.dst()) |
                        Rule::gen_control_bits(mv.dst(), SKakuN)) != 0
                } else if to_mask & part.sente_kaku_board != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_kaku(
                        part.sente_self_board, part.sente_opponent_board,
                        part.gote_opponent_board, part.gote_self_board, mv.dst()) != 0
                } else if to_mask & part.gote_hisha_board != 0 && to_mask & part.sente_nari_board != 0 {
                    surrounding_mask & (Rule::gen_control_bits_by_hisha(
                        part.sente_self_board,part.sente_opponent_board,
                        part.gote_opponent_board,part.gote_self_board,mv.dst()) |
                        Rule::gen_control_bits(mv.dst(),SHishaN)) != 0
                } else if to_mask & part.gote_hisha_board != 0 {
                    surrounding_mask & Rule::gen_control_bits_by_hisha(
                        part.sente_self_board,part.sente_opponent_board,
                        part.gote_opponent_board,part.gote_self_board,mv.dst()) != 0
                } else {
                    let (x,y) = mv.dst().square_to_point();

                    surrounding_mask & Rule::gen_control_bits(
                        mv.dst(), state.get_banmen()[y as usize][x as usize]
                    ) != 0
                }
            }
        } else {
            false
        }
    }

    fn is_major_order(&self,teban:Teban,state:&State,m:LegalMove) -> bool {
        match m {
            LegalMove::Put(mv) if mv.kind() == MochigomaKind::Kyou ||
                mv.kind() == MochigomaKind::Kaku ||
                mv.kind() == MochigomaKind::Hisha=> true,
            LegalMove::To(mv) if teban == Teban::Sente => {
                let part = state.get_part();

                let from_mask = 1 << (mv.src() + 1);

                ((part.sente_kyou_board & !part.sente_nari_board) & from_mask) != 0 ||
                 (part.sente_kaku_board & from_mask) != 0 ||
                 (part.sente_hisha_board & from_mask) != 0
            },
            LegalMove::To(mv) => {
                let part = state.get_part();

                let from_mask = 1 << (mv.src() + 1);

                ((part.gote_kyou_board & !part.gote_nari_board) & from_mask) != 0 ||
                 (part.gote_kaku_board & from_mask) != 0 ||
                 (part.gote_hisha_board & from_mask) != 0
            },
            _ => false
        }
    }

    fn is_seal_order(&self,teban:Teban,state:&State,m:LegalMove) -> bool {
        let p = Rule::ou_square(teban.opposite(),state);
        let to = m.dst();

        if p == -1 {
            false
        } else {
            let (tx,ty) = to.square_to_point();
            let (dx,dy) = p.square_to_point();

            if tx == 4 && ty == 4 {
                true
            } else if tx >= 4 && ty >= 4 {
                tx >= dx && ty >= dy
            } else if tx >= 4 && ty <= 4 {
                tx >= dx && ty <= dy
            } else if tx <= 4 && ty <= 4 {
                tx <= dx && ty <= dy
            } else {
                tx <= dx && ty >= dy
            }
        }
    }

    fn blocker_count(&self,teban:Teban,state:&State,mc:&MochigomaCollections,m:LegalMove) -> Option<u8> {
        let blocking_count = Rule::can_blocking_count(teban,state,mc,m);

        if blocking_count == 0 {
            None
        } else {
            Some(blocking_count as u8)
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvasionsMoveOrder {
    Block,
    Drop,
    Capture,
    Escape
}
pub struct EvasionsMoveOrderer {

}
impl EvasionsMoveOrderer {
    pub fn new() -> Self {
        EvasionsMoveOrderer {}
    }
    pub fn ordering<I: Iterator<Item=LegalMove>>(&self,teban:Teban,
                                                 state:&State,
                                                 it:I) -> impl Iterator<Item=LegalMove> {
        let mut mvs = Vec::new();

        for m in it {
            if let LegalMove::Put(_) = m {
                mvs.push((EvasionsMoveOrder::Drop,m));
            } else if m.obtained().is_some() {
                mvs.push((EvasionsMoveOrder::Capture,m));
            } else if self.is_escape(teban,state,m) {
                mvs.push((EvasionsMoveOrder::Escape,m));
            } else {
                mvs.push((EvasionsMoveOrder::Block,m));
            }
        }
        mvs.sort_by(|a,b| a.0.cmp(&b.0));

        mvs.into_iter().map(|(_,m)| m)
    }

    fn is_escape(&self,teban:Teban,state:&State,m:LegalMove) -> bool {
        if let LegalMove::To(mv) = m {
            Rule::ou_square(teban,state) == mv.src() as i32
        } else {
            false
        }
    }
}
pub struct Root<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError>;
    fn qsearch<'a,'b>(&self,teban:Teban,
               pos:&mut Position<UNDO_BUFFER_SIZE>,
               env:&mut Environment<L,S>,
               event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
               zh: &ZobristHash<u64>,
               history:&mut HashSet<(Teban,u64,u64)>,
               self_partial_output: &'a Arr<f32,{256*2}>,
               opponent_partial_output: &'a Arr<f32,{256*2}>,
               mut alpha:Score,beta:Score,
               depth:usize,current_depth:usize,
               prev_move:Option<LegalMove>,
               evalutor: &Arc<Evalutor<M>>,rng:&mut Prng)
        -> Result<Score,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.lazy_abort.load(Ordering::Acquire) {
            let score = Score::Value(evalutor.evalute(&self_partial_output)?);
            return Ok(score);
        }

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

                let s = s.localize_score(current_depth as i32);

                if bound == Bound::Exact ||
                   (bound == Bound::LowerBound && s >= beta) ||
                   (bound == Bound::UpperBound && s < alpha) {

                    return Ok(s);
                }
            }
        }

        let mut picker = RandomPicker::new(Prng::new(rng.rnd64()));

        let in_check = Rule::in_check(teban,pos.get_state());

        if in_check {
            Rule::generate_moves::<Evasions>(teban, pos.get_state(), pos.get_mc(), &mut picker)?;

            if picker.len() == 0 {
                env.transposition_table.update(&zh,0,TTScore::NEGINFINITE(0),Bound::Exact,None);

                return Ok(Score::NEGINFINITE(current_depth as i32));
            }

            let move_orderer = EvasionsMoveOrderer::new();

            let start_alpha = alpha;

            let mut bestscore = Score::default();

            let mut best_move = None;

            history.insert((teban,mk,sk));

            for m in move_orderer.ordering(teban,pos.get_state(),&mut picker) {
                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh,0,TTScore::INFINITE(0),Bound::Exact,Some(m));

                    return Ok(Score::INFINITE(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, pos.get_state().get_banmen(), pos.get_mc(), m.to_applied_move(), &o);

                let use_diff = match m {
                    LegalMove::To(m) => m.src() != Rule::ou_square(teban,pos.get_state()) as u32,
                    _ => false
                };

                let (self_partial_output,opponent_partial_output) = if use_diff {
                    let self_partial_output = evalutor.prepare_evalute_by_diff(teban, teban,pos.get_state(),pos.get_mc(),m,self_partial_output)?;
                    let opponent_partial_output = evalutor.prepare_evalute_by_diff(teban, teban.opposite(),pos.get_state(),pos.get_mc(),m,opponent_partial_output)?;

                    pos.apply_move(teban,m)?;

                    (self_partial_output,opponent_partial_output)
                } else {
                    pos.apply_move(teban,m)?;

                    let self_partial_output = evalutor.prepare_evalute(teban,pos.get_state(),pos.get_mc())?;
                    let opponent_partial_output = evalutor.prepare_evalute(teban.opposite(),pos.get_state(),pos.get_mc())?;

                    (self_partial_output,opponent_partial_output)
                };

                let score = -self.qsearch(teban.opposite(),
                                          pos,
                                          env,
                                          event_dispatcher,
                                          &nzh,
                                          history,
                                          &opponent_partial_output,
                                          &self_partial_output,
                                          -beta,
                                          -alpha,
                                          depth+1,
                                          current_depth+1,
                                          Some(m),
                                          evalutor,
                                          rng)?;

                pos.undo_move()?;

                if score.is_infinite() {
                    history.remove(&(teban,mk,sk));

                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&zh, 0, score.normalize_score(current_depth as i32), Bound::Exact, Some(m));
                    }

                    return Ok(score);
                } else if score >= beta {
                    history.remove(&(teban,mk,sk));

                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&zh, 0, score.normalize_score(current_depth as i32), Bound::LowerBound, Some(m));
                    }

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

            let bs = bestscore.normalize_score(current_depth as i32);

            if alpha > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&zh, 0, bs, Bound::Exact, best_move);
            } else if !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&zh, 0, bs, Bound::UpperBound, best_move);
            }

            Ok(bestscore)
        } else {
            let start_alpha = alpha;

            let stand_pat = Score::Value(evalutor.evalute(&self_partial_output)?);

            if stand_pat >= beta {
                return Ok(stand_pat);
            }

            Rule::generate_moves_by_banmen::<CaptureOrPawnPromotions>(teban,pos.get_state(),&mut picker)?;

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
                        }).unwrap_or(false) && !Rule::is_oute_move(pos.get_state(),teban,m) {
                            if calc_see(teban,pos.get_state(),m) < -CAPTURED_SCORE_MAP[o as usize] * 4 / 3 {
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

                    env.transposition_table.update(&zh,0,TTScore::INFINITE(0),Bound::Exact,Some(m));

                    return Ok(Score::INFINITE(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, pos.get_state().get_banmen(), pos.get_mc(), m.to_applied_move(), &o);


                let use_diff = match m {
                    LegalMove::To(m) => m.src() != Rule::ou_square(teban,pos.get_state()) as u32,
                    _ => false
                };

                let (self_partial_output,opponent_partial_output) = if use_diff {
                    let self_partial_output = evalutor.prepare_evalute_by_diff(teban, teban,pos.get_state(),pos.get_mc(),m,self_partial_output)?;
                    let opponent_partial_output = evalutor.prepare_evalute_by_diff(teban, teban.opposite(),pos.get_state(),pos.get_mc(),m,opponent_partial_output)?;

                    pos.apply_move(teban,m)?;

                    (self_partial_output,opponent_partial_output)
                } else {
                    pos.apply_move(teban,m)?;

                    let self_partial_output = evalutor.prepare_evalute(teban,pos.get_state(),pos.get_mc())?;
                    let opponent_partial_output = evalutor.prepare_evalute(teban.opposite(),pos.get_state(),pos.get_mc())?;

                    (self_partial_output,opponent_partial_output)
                };

                let score = -self.qsearch(teban.opposite(),
                                pos,
                                env,
                                event_dispatcher,
                                &nzh,
                                history,
                                &opponent_partial_output,
                                &self_partial_output,
                                -beta,
                                -alpha,
                                depth+1,
                                current_depth+1,
                                Some(m),
                                evalutor,
                                rng)?;

                pos.undo_move()?;

                if score.is_infinite() {
                    history.remove(&(teban,mk,sk));

                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&zh, 0, score.normalize_score(current_depth as i32), Bound::Exact, Some(m));
                    }

                    return Ok(score);
                } else if score >= beta {
                    history.remove(&(teban,mk,sk));

                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&zh, 0, score.normalize_score(current_depth as i32), Bound::LowerBound, Some(m));
                    }

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

            let bs = bestscore.normalize_score(current_depth as i32);

            if bestscore > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&zh, 0, bs, Bound::Exact, best_move);
            } else if !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&zh, 0, bs, Bound::UpperBound, best_move);
            }

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

    /*
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
    */
    fn satisfy_threatmate_search(&self, attacker:Teban, state:&State) -> bool {
        let mut score = 0;

        let ps = state.get_part();

        if (ps.sente_self_board | ps.sente_opponent_board).bitcount() <= 18 {
            score += 1;
        }

        if (ps.sente_kaku_board | ps.sente_hisha_board | ps.gote_kaku_board | ps.gote_hisha_board).bitcount() <= 2 {
            score += 1;
        }

        if (ps.sente_nari_board | ps.gote_nari_board).bitcount() >= 2 {
            score += 1;
        }

        if score >= 2 {
            return true;
        }

        let mask = Rule::gen_ou_surrounding_mask(attacker.opposite(), state.get_part());

        if (mask & (ps.sente_opponent_board | ps.sente_self_board)).bitcount() <= 4 {
            score += 1;
        }

        score >= 2
    }
    fn threatmate_search<'b>(&self,
                         attacker:Teban,
                         teban:Teban,
                         pos:&mut Position<UNDO_BUFFER_SIZE>,
                         env:&mut Environment<L,S>,
                         event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                         zh: &ZobristHash<u64>,
                         history:&mut HashSet<(Teban,u64,u64)>,
                         threatmate_cache: &mut HashMap<(Teban,u64,u64),(ThreatMateSearchResultRelative,u32)>,
                         depth:usize,
                         current_depth:usize,
                         rng:&mut Prng) -> Result<ThreatMateSearchResult,ApplicationError> {
        let (mk,sk) = zh.keys();

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        match threatmate_cache.get(&(teban,mk,sk)) {
            Some(&(r,d)) if d >= depth as u32 => {
                match r {
                    ThreatMateSearchResultRelative::Unknown => return Ok(ThreatMateSearchResult::Unknown),
                    ThreatMateSearchResultRelative::Checkmate(d) => return Ok(ThreatMateSearchResult::Checkmate(d - current_depth as i32)),
                    ThreatMateSearchResultRelative::Checkmated(d) => return Ok(ThreatMateSearchResult::Checkmated(d + current_depth as i32)),
                }
            },
            _ => {}
        }

        if depth == 0 ||
            env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
            env.lazy_abort.load(Ordering::Acquire) || self.timelimit_reached(env)? {
            return Ok(ThreatMateSearchResult::Unknown);
        }

        if history.contains(&(teban,mk,sk)) {
            return Ok(ThreatMateSearchResult::Repetition);
        }

        {
            let ps = pos.get_state().get_part();

            let mut count = 0;

            if (ps.sente_self_board | ps.sente_opponent_board).bitcount() > 24 {
                count += 1;
            }

            if (ps.sente_kaku_board | ps.sente_hisha_board |
                ps.gote_kaku_board | ps.gote_hisha_board).bitcount() > 3 {
                count += 1;
            }

            if (ps.sente_nari_board | ps.gote_nari_board).bitcount() < 1 {
                count += 1;
            }

            if count >= 3 {
                threatmate_cache.insert((teban, mk, sk), (ThreatMateSearchResultRelative::Unknown, depth as u32));
                return Ok(ThreatMateSearchResult::Unknown);
            }

            let mask = Rule::gen_ou_surrounding_mask(attacker.opposite(), ps);

            if (mask & (ps.sente_opponent_board | ps.sente_self_board)).bitcount() > 5 {
                count += 1;
            }

            if count >= 3 {
                threatmate_cache.insert((teban, mk, sk), (ThreatMateSearchResultRelative::Unknown, depth as u32));
                return Ok(ThreatMateSearchResult::Unknown);
            }
        }

        {
            let r = env.transposition_table.get(&zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: d,
                            score: s,
                            bound,
                            best_move: _
                        }) = r {

                if bound == Bound::Exact {
                    match s {
                        TTScore::INFINITE(d) => {
                            threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(d),depth as u32));
                            return Ok(ThreatMateSearchResult::Checkmate(d - current_depth as i32));
                        },
                        TTScore::NEGINFINITE(d) => {
                            threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmated(d),depth as u32));
                            return Ok(ThreatMateSearchResult::Checkmated(d + current_depth as i32));
                        },
                        _ if d as usize >= depth => {
                            threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Unknown,depth as u32));
                            return Ok(ThreatMateSearchResult::Unknown);
                        },
                        _ => ()
                    }
                }
            }
        }

        let mut picker = RandomPicker::new(Prng::new(rng.rnd64()));

        if attacker == teban {
            Rule::generate_moves::<Checks>(teban, pos.get_state(), pos.get_mc(), &mut picker)?;

            if picker.len() == 0 {
                threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Unknown,depth as u32));
                return Ok(ThreatMateSearchResult::Unknown);
            }

            history.insert((teban,mk,sk));

            let mut best_score = ThreatMateSearchResult::Unknown;

            for m in &mut picker {
                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh,depth as i8,TTScore::INFINITE(0),Bound::Exact,Some(m));

                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(0),depth as u32));
                    return Ok(ThreatMateSearchResult::Checkmate(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, pos.get_state().get_banmen(), pos.get_mc(), m.to_applied_move(), &o);

                pos.apply_move(teban,m)?;

                let s = -self.threatmate_search(attacker,
                                                teban.opposite(),
                                                pos,
                                                env,
                                                event_dispatcher,
                                                &nzh,
                                                history,
                                                threatmate_cache,
                                                depth - 1,
                                                current_depth + 1,
                                                rng)?;
                pos.undo_move()?;

                if let ThreatMateSearchResult::Checkmate(ply) = s {
                    history.remove(&(teban, mk, sk));

                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&zh, depth as i8, TTScore::INFINITE(ply + (current_depth as i32)), Bound::Exact, Some(m));
                    }

                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(ply + current_depth as i32),depth as u32));
                    return Ok(ThreatMateSearchResult::Checkmate(ply));
                }

                if env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) ||
                    self.timelimit_reached(env)? {
                    break;
                }

                if s > best_score {
                    best_score = s;
                }
            }

            history.remove(&(teban,mk,sk));

            match best_score {
                ThreatMateSearchResult::Checkmate(d) => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(d + current_depth as i32),depth as u32));
                },
                ThreatMateSearchResult::Checkmated(d) => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmated(d - current_depth as i32),depth as u32));
                },
                ThreatMateSearchResult::Unknown => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Unknown,depth as u32));
                },
                _ => ()
            }

            Ok(best_score)
        } else {
            Rule::generate_moves::<Evasions>(teban, pos.get_state(), pos.get_mc(), &mut picker)?;

            if picker.len() == 0 {
                threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmated(0),depth as u32));
                return Ok(ThreatMateSearchResult::Checkmated(current_depth as i32));
            }

            history.insert((teban,mk,sk));

            let mut best_score = ThreatMateSearchResult::Checkmated(current_depth as i32);

            for m in &mut picker {
                if let Some(ObtainKind::Ou) = match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                } {
                    history.remove(&(teban,mk,sk));

                    env.transposition_table.update(&zh,depth as i8,TTScore::INFINITE(0),Bound::Exact,Some(m));

                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(0),depth as u32));

                    return Ok(ThreatMateSearchResult::Checkmate(-(current_depth as i32)));
                }

                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let nzh = zh.updated(&env.hasher, teban, pos.get_state().get_banmen(), pos.get_mc(), m.to_applied_move(), &o);

                pos.apply_move(teban,m)?;

                let s = -self.threatmate_search(attacker,
                                                teban.opposite(),
                                                pos,
                                                env,
                                                event_dispatcher,
                                                &nzh,
                                                history,
                                                threatmate_cache,
                                                depth - 1,
                                                current_depth + 1,
                                                rng)?;
                pos.undo_move()?;

                match s {
                    ThreatMateSearchResult::Unknown | ThreatMateSearchResult::Checkmate(_) => {
                        history.remove(&(teban, mk, sk));

                        threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Unknown,depth as u32));

                        return Ok(ThreatMateSearchResult::Unknown);
                    },
                    ThreatMateSearchResult::Repetition => {
                        history.remove(&(teban, mk, sk));

                        return Ok(ThreatMateSearchResult::Repetition);
                    }
                    _ => ()
                }

                if s > best_score {
                    best_score = s;
                }
            }

            history.remove(&(teban,mk,sk));

            match best_score {
                ThreatMateSearchResult::Checkmated(d) if !env.lazy_abort.load(Ordering::Acquire) => {
                    env.transposition_table.update(&zh,depth as i8,TTScore::NEGINFINITE(d - current_depth as i32),Bound::Exact,None);
                },
                _ => ()
            }

            match best_score {
                ThreatMateSearchResult::Checkmate(d) => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmate(d + current_depth as i32),depth as u32));
                },
                ThreatMateSearchResult::Checkmated(d) => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Checkmated(d - current_depth as i32),depth as u32));
                },
                ThreatMateSearchResult::Unknown => {
                    threatmate_cache.insert((teban,mk,sk),(ThreatMateSearchResultRelative::Unknown,depth as u32));
                },
                _ => ()
            }

            Ok(best_score)
        }
    }
    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> Result<bool,ApplicationError> {
        let reached;
        let timelimit_margin = env.timelimit_margin;

        match *env.current_limit.read() {
            (current_turn_lmit,current_limit) => {
                let reached_lazy_abort = current_turn_lmit.map(|l| l - Instant::now() <= Duration::from_millis(timelimit_margin)).unwrap_or(false);

                if reached_lazy_abort {
                    env.lazy_abort.store(true,Ordering::Release);
                }

                reached = current_limit.map(|l| l - Instant::now() <= Duration::from_millis(timelimit_margin)).unwrap_or(false);
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
                       _:i32,
                       tt_move:Option<&LegalMove>,
                       pv:Option<&LegalMove>,
                       _: &mut LazyEval,
                       _: &Arr<f32,{256*2}>,
                       _: &Arc<Evalutor<M>>) -> Result<u32,ApplicationError> {
        if depth <= 1 ||
            Rule::in_check(teban,state) ||
            tt_move.map(|&tm| tm == m).unwrap_or(false) ||
            pv.map(|pm| pm == &m).unwrap_or(false) ||
            self.is_important_move(env,current_depth,
                                   teban,state,m)? {
            Ok(0)
        } else if *index == 0 {
            *index += 1;
            Ok(0)
        } else {
            let move_index = *index + 1;
            let mut r = 1.05 * (depth as f32 + 1.).ln() * 0.85 *(move_index as f32).ln() / 2.35;

            let h = env.move_orderer.look_up_history(teban,state,m)?;

            *index += 1;

            let threshold = (depth as i32 + 1) * 5 * 256;

            if h > threshold {
                r -= 0.6;
            } else if h < -threshold {
                r += 0.6;
            }

            let r = r.clamp(0., depth as f32 - 1.) as u32;

            Ok(r)
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
        let mut env = env.clone();

        env.move_orderer = move_orderer;

        let evalutor = Arc::clone(&evalutor);
        let zh = gs.zh.clone();

        let gives_check_us = gs.gives_check_us;
        let gives_check_them = gs.gives_check_them;

        let current_depth = 0;
        let base_depth = gs.base_depth;
        let extend_depth = gs.extend_depth;

        let mut best_score = Score::default();

        let self_partial_output = gs.self_partial_output.clone();
        let opponent_partial_output = gs.opponent_partial_output.clone();

        let shared_depth = Arc::clone(shared_depth);

        let mut pos = gs.pos.clone();

        if thread_index == 0 {
            self.thread_pool.spawn(move || {
                env.move_orderer.startup();

                let mut pv = VecDeque::new();
                let mut rng = rand::thread_rng();
                let mut rng = Prng::new(rng.gen());

                let search_offset = 0;

                let mut prev_score = Score::default();

                let mut threatmate_cache = HashMap::new();

                'outer: for depth in 1..=base_depth {
                    if let Score::Value(ps) = prev_score {
                        let delta = Self::compute_aspiration_window_delta(depth);

                        let mut alpha = Score::Value(ps - delta);
                        let mut beta = Score::Value(ps + delta);

                        let strategy = Inter::new();

                        for i in 0..2 {
                            let mut gs = GameState {
                                teban: teban,
                                pos: &mut pos,
                                alpha: alpha,
                                beta: beta,
                                search_offset: search_offset,
                                best_score: best_score,
                                m: None,
                                static_eval: LazyEval::new(),
                                gives_check_us:gives_check_us,
                                gives_check_them:gives_check_them,
                                prev_kind: KomaKind::Blank,
                                move_history: &mut Vec::new(),
                                threatmate_cache:&mut threatmate_cache,
                                self_partial_output:&self_partial_output,
                                opponent_partial_output:&opponent_partial_output,
                                thread_index:thread_index,
                                pv:&pv,
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

                                        if env.lazy_abort.load(atomic::Ordering::Acquire) {
                                            let _ = sender.send(Ok(RootEvaluationResult::LazyAbort(score, mvs, zh, depth, seldepth, thread_index)));
                                        } else {
                                            let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
                                        }

                                        break;
                                    }
                                },
                                Ok(EvaluationResult::NodeLimits) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                                    break 'outer;
                                },
                                Ok(EvaluationResult::Timeout) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                                    break 'outer;
                                },
                                Ok(EvaluationResult::Repetition(_,_,_,_)) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Repetition));
                                    break 'outer;
                                },
                                Ok(EvaluationResult::Stop) => {
                                    let _ = sender.send(Ok(RootEvaluationResult::Stop));
                                    break 'outer;
                                },
                                Ok(EvaluationResult::Cut) => {
                                    let _ = sender.send(Err(ApplicationError::LogicError(String::from("The root node has been pruned."))));
                                    break 'outer;
                                },
                                Err(e) => {
                                    if let Err(e) = pos.rewind() {
                                        let _ = sender.send(Err(ApplicationError::from(e)));
                                    }

                                    let _ = sender.send(Err(e));
                                    break 'outer;
                                }
                            };
                        }
                    } else {
                        let mut gs = GameState {
                            teban: teban,
                            pos: &mut pos,
                            alpha: Score::default(),
                            beta: Score::INFINITE(0),
                            search_offset: search_offset,
                            best_score: best_score,
                            m: None,
                            static_eval: LazyEval::new(),
                            gives_check_us:gives_check_us,
                            gives_check_them:gives_check_them,
                            prev_kind: KomaKind::Blank,
                            move_history: &mut Vec::new(),
                            threatmate_cache:&mut threatmate_cache,
                            self_partial_output: &self_partial_output,
                            opponent_partial_output: &opponent_partial_output,
                            thread_index: thread_index,
                            pv: &pv,
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

                                if env.lazy_abort.load(atomic::Ordering::Acquire) {
                                    let _ = sender.send(Ok(RootEvaluationResult::LazyAbort(score, mvs, zh, depth, seldepth, thread_index)));
                                } else {
                                    let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
                                }
                            },
                            Ok(EvaluationResult::NodeLimits) => {
                                let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                                break;
                            },
                            Ok(EvaluationResult::Timeout) => {
                                let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                                break;
                            },
                            Ok(EvaluationResult::Repetition(_,_,_,_)) => {
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
                                if let Err(e) = pos.rewind() {
                                    let _ = sender.send(Err(ApplicationError::from(e)));
                                }

                                let _ = sender.send(Err(e));

                                break;
                            }
                        }
                    }

                    if env.abort.load(Ordering::Acquire) ||
                        env.stop.load(Ordering::Acquire) ||
                        env.lazy_abort.load(Ordering::Acquire) {
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

            let gives_check_us = gs.gives_check_us;
            let gives_check_them = gs.gives_check_them;

            self.thread_pool.spawn(move || {
                env.move_orderer.startup();

                let mut rng = rand::thread_rng();
                let mut rng = Prng::new(rng.gen());

                let mut threatmate_cache = HashMap::new();

                let mut depth = shared_depth.load(Ordering::Acquire) as u32;

                while depth <= base_depth {
                    let len = mvs.len();

                    let search_offset = if len == 0 {
                        0
                    } else {
                        (thread_index * 7 + (rng.rnd64() as usize % len)) % len
                    };

                    let mut gs = GameState {
                        teban: teban,
                        pos: &mut pos,
                        alpha: Score::default(),
                        beta: Score::INFINITE(0),
                        search_offset: search_offset,
                        best_score: best_score,
                        m: None,
                        static_eval: LazyEval::new(),
                        gives_check_us:gives_check_us,
                        gives_check_them:gives_check_them,
                        prev_kind: KomaKind::Blank,
                        move_history: &mut Vec::new(),
                        threatmate_cache:&mut threatmate_cache,
                        self_partial_output:&self_partial_output,
                        opponent_partial_output:&opponent_partial_output,
                        thread_index:thread_index,
                        pv:&pv,
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

                            if env.lazy_abort.load(atomic::Ordering::Acquire) {
                                let _ = sender.send(Ok(RootEvaluationResult::LazyAbort(score, mvs, zh, depth, seldepth, thread_index)));
                            } else {
                                let _ = sender.send(Ok(RootEvaluationResult::Exact(score, mvs, zh, depth, seldepth, thread_index)));
                            }
                        },
                        Ok(EvaluationResult::NodeLimits) => {
                            let _ = sender.send(Ok(RootEvaluationResult::NodeLimits));
                        },
                        Ok(EvaluationResult::Timeout) => {
                            let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                        },
                        Ok(EvaluationResult::Repetition(_,_,_,_)) => {
                            let _ = sender.send(Ok(RootEvaluationResult::Repetition));
                        },
                        Ok(EvaluationResult::Stop) => {
                            let _ = sender.send(Ok(RootEvaluationResult::Stop));
                        },
                        Ok(EvaluationResult::Cut) => {
                            let _ = sender.send(Err(ApplicationError::LogicError(String::from("The root node has been pruned."))));
                        },
                        Err(e) => {
                            if let Err(e) = pos.rewind() {
                                let _ = sender.send(Err(ApplicationError::from(e)));
                            }

                            let _ = sender.send(Err(e));

                            break;
                        }
                    }

                    if env.abort.load(Ordering::Acquire) ||
                        env.stop.load(Ordering::Acquire) ||
                        env.lazy_abort.load(Ordering::Acquire) {
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
                    let _ = env.on_error_handler.lock().map(|h| h.call(&e));
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

        let mut picker = RandomPicker::new(Prng::new(gs.rng.rnd64()));

        let mut mvs = Vec::new();

        if Rule::in_check(gs.teban,gs.pos.get_state()) {
            Rule::generate_moves::<Evasions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;
            mvs = picker.collect::<Vec<LegalMove>>();
        } else {
            {
                Rule::generate_moves::<CaptureOrPawnPromotions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;
                let mut v = (&mut picker).collect::<Vec<LegalMove>>();
                mvs.append(&mut v);
            }

            {
                Rule::generate_moves::<QuietsWithoutPawnPromotions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;
                let mut v = (&mut picker).collect::<Vec<LegalMove>>();
                mvs.append(&mut v);
            }
        };

        let mvs = Arc::new(mvs);

        env.abort.store(false,Ordering::Release);
        env.lazy_abort.store(false,Ordering::Release);

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
                Ok(RootEvaluationResult::LazyAbort(s, mvs, zh, _, seldepth, thread_index)) => {
                    if let Err(e) = env.info_sender.flush() {
                        let _ = env.on_error_handler.lock().map(|h| h.call(&e));
                    }

                    if thread_index == 0 {
                        let r = EvaluationResult::Exact(s, mvs, zh, seldepth);

                        self.termination(env, busy_threads, move_orderers)?;

                        return Ok(self.choose_result(&mut pv_result, pv_depth as usize, &mut worker_result, worker_depth as usize).unwrap_or(r));
                    }
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
                    let _ = env.on_error_handler.lock().map(|h| h.call(&e));

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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {}
pub struct Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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

        let piece_index = env.move_orderer.calc_piece_us_index(gs.teban,env.move_orderer.calc_piece_index(gs.teban,gs.pos.get_state(),m)?)?;

        let zh = gs.zh.updated(&env.hasher, gs.teban, gs.pos.get_state().get_banmen(), gs.pos.get_mc(), m.to_applied_move(), &o);

        let use_diff = match m {
            LegalMove::To(m) => m.src() != Rule::ou_square(gs.teban,gs.pos.get_state()) as u32,
            _ => false
        };

        let prev_kind = match m {
            LegalMove::To(mv) => {
                let (x,y) = mv.src().square_to_point();

                gs.pos.get_state().get_banmen()[y as usize][x as usize]
            },
            _ => KomaKind::Blank
        };

        let (self_partial_output,opponent_partial_output) = if use_diff {
            let self_partial_output = evalutor.prepare_evalute_by_diff(gs.teban, gs.teban,gs.pos.get_state(),gs.pos.get_mc(),m,gs.self_partial_output)?;
            let opponent_partial_output = evalutor.prepare_evalute_by_diff(gs.teban, gs.teban.opposite(),gs.pos.get_state(),gs.pos.get_mc(),m,gs.opponent_partial_output)?;

            gs.pos.apply_move(gs.teban, m)?;

            (self_partial_output,opponent_partial_output)
        } else {
            gs.pos.apply_move(gs.teban, m)?;

            let self_partial_output = evalutor.prepare_evalute(gs.teban,gs.pos.get_state(),gs.pos.get_mc())?;
            let opponent_partial_output = evalutor.prepare_evalute(gs.teban.opposite(),gs.pos.get_state(),gs.pos.get_mc())?;

            (self_partial_output,opponent_partial_output)
        };

        let static_eval = LazyEval::new();

        if extend_depth > 0 {
            if extend_check > 0 && Rule::in_check(gs.teban.opposite(),gs.pos.get_state()) {
                depth += 1;
                extend_depth -= 1;
                extend_check -= 1;
            }
        }

        gs.move_history.push(Some((piece_index as u8,m.dst() as u8)));

        let in_check = Rule::in_check(gs.teban.opposite(),gs.pos.get_state());

        let mut gs = GameState {
            teban: gs.teban.opposite(),
            rng: gs.rng,
            pos: gs.pos,
            alpha: -gs.beta,
            beta: -alpha,
            search_offset: 0,
            best_score: gs.best_score,
            m: Some(m),
            static_eval:static_eval,
            gives_check_us:gs.gives_check_them,
            gives_check_them:in_check,
            prev_kind: prev_kind,
            thread_index:gs.thread_index,
            pv:pv,
            move_history: gs.move_history,
            threatmate_cache: gs.threatmate_cache,
            self_partial_output:&opponent_partial_output,
            opponent_partial_output:&self_partial_output,
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

        gs.pos.undo_move()?;

        r
    }

    pub fn search_null_move<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                   alpha:Score,
                                   beta:Score,
                                   depth:u32,
                                   event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                   evalutor: &Arc<Evalutor<M>>)
        -> Result<EvaluationResult, ApplicationError> {
        let zh = gs.zh.teban_fliped();

        gs.move_history.push(None);

        let mut gs = GameState {
            teban: gs.teban.opposite(),
            pos: gs.pos,
            rng: gs.rng,
            alpha: alpha,
            beta: beta,
            search_offset: 0,
            best_score: gs.best_score,
            m: None,
            static_eval: LazyEval::new(),
            gives_check_us:gs.gives_check_them,
            gives_check_them:false,
            prev_kind: Blank,
            thread_index:gs.thread_index,
            pv:&VecDeque::new(),
            move_history: gs.move_history,
            threatmate_cache: gs.threatmate_cache,
            self_partial_output:gs.opponent_partial_output,
            opponent_partial_output:gs.self_partial_output,
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

        let r = strategy.search(env, &mut gs, event_dispatcher, evalutor);

        gs.move_history.pop();

        r
    }
}
impl<L,S,M> SendInfo<L,S,M> for Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {}
impl<L,S,M> Search<L,S,M> for Recursive<L,S,M>
    where L: Logger + Send + 'static,
          S: InfoSender,
          for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
            let mut mvs = VecDeque::new();

            gs.m.map(|m| mvs.push_front(m));

            return Ok(EvaluationResult::Repetition(Score::Value(400),mvs,gs.zh.clone(),gs.current_depth));
        }

        event_dispatcher.dispatch_events(&self,&env.event_queue)?;

        if env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Stop);
        }

        if self.timelimit_reached(env)? || env.abort.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        let prev_move = gs.m.clone();

        if Rule::in_check(gs.teban.opposite(),gs.pos.get_state()) {
            if let Some(m) = prev_move.clone() {
                env.transposition_table.update(&gs.zh,gs.depth as i8,TTScore::INFINITE(0),Bound::Exact,None);

                let mut mvs = VecDeque::new();

                mvs.push_front(m);

                return Ok(EvaluationResult::Exact(Score::INFINITE(-(gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
            }
        }

        if let Some(prev_move) = gs.m.clone() {
            match gs.threatmate_cache.get(&(gs.teban,mk,sk)) {
                Some((ThreatMateSearchResultRelative::Checkmate(d),_)) => {
                    let mut mvs = VecDeque::new();

                    mvs.push_front(prev_move);

                    return Ok(EvaluationResult::Exact(Score::INFINITE(d - (gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
                },
                Some((ThreatMateSearchResultRelative::Checkmated(d),_)) => {
                    let mut mvs = VecDeque::new();

                    mvs.push_front(prev_move);

                    return Ok(EvaluationResult::Exact(Score::NEGINFINITE(d + (gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
                },
                _ => {}
            }

            let r = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: d,
                            score: s,
                            bound,
                            best_move: _
                        }) = r {

                let s = s.localize_score(gs.current_depth as i32);

                if (bound == Bound::Exact && s.exact_score_bound()) ||
                   (bound == Bound::Exact && d as u32 >= gs.depth) ||
                   (bound == Bound::LowerBound && d as u32 >= gs.depth && s >= gs.beta) ||
                   (bound == Bound::UpperBound && d as u32 >= gs.depth && s < gs.alpha) {
                    let mut mvs = VecDeque::new();

                    mvs.push_front(prev_move);

                    return Ok(EvaluationResult::Exact(s, mvs, gs.zh.clone(), gs.current_depth));
                }
            }
        }

        let in_check = Rule::in_check(gs.teban,gs.pos.get_state());

        if in_check {
            if gs.depth == 0 {
                let s = self.qsearch(gs.teban,
                                     gs.pos,
                                     env,
                                     event_dispatcher,
                                     &gs.zh,
                                     &mut HashSet::new(),
                                     gs.self_partial_output,
                                     gs.opponent_partial_output,
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

            let start_alpha = gs.alpha;
            let mut alpha = gs.alpha;
            let mut quiet_alpha = gs.alpha;

            let beta = gs.beta;

            let mut scoreval = Score::default();
            let mut best_moves = VecDeque::new();

            let mut picker = RandomPicker::new(Prng::new(gs.rng.rnd64()));

            env.history.insert((gs.teban,mk,sk));

            let pv_non = VecDeque::new();

            let mut quiet_moves = Vec::with_capacity(593);

            let pv_move = if gs.pv.len() > gs.current_depth as usize {
                Some(gs.pv[gs.current_depth as usize])
            } else {
                None
            };

            let mut max_seldepth = gs.current_depth;

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

            Rule::generate_moves::<Evasions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;

            for (m, _) in env.move_orderer.ordering(
                &mut picker, gs.current_depth, gs.teban, gs.pos.get_state(), tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history)? {                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                if self.is_obtained_ou(m)? {
                    env.transposition_table.update(&gs.zh,gs.depth as i8,TTScore::INFINITE(0),Bound::Exact,Some(m));

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

                assert!(gs.depth > 0);

                match self.search_child_node(env,gs,m,pv,alpha,gs.depth,gs.cut_node,gs.already_reduced_lmr,gs.nmp_min_ply,event_dispatcher,evalutor)? {
                    EvaluationResult::Exact(s, mvs, _, seldepth) |
                    EvaluationResult::Repetition(s, mvs, _, seldepth) => {
                        let s = -s;

                        max_seldepth = max_seldepth.max(seldepth);

                        if s > scoreval {
                            scoreval = s;

                            best_moves = mvs;

                            if scoreval >= beta || scoreval.is_infinite() {
                                if !env.lazy_abort.load(Ordering::Acquire) {
                                    match scoreval {
                                        Score::INFINITE(_) => {
                                            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                        },
                                        Score::NEGINFINITE(_) => {
                                            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                        },
                                        _ => {
                                            env.transposition_table.update(&gs.zh, gs.depth as i8,scoreval.normalize_score(gs.current_depth as i32),Bound::LowerBound,Some(m));
                                        }
                                    }
                                }

                                match m {
                                    LegalMove::To(mv) if mv.obtained().is_none() => {
                                        if !env.lazy_abort.load(Ordering::Acquire) {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;

                                                let _ = prev_move.map(|prev_move| {
                                                    env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                }).unwrap_or(Ok(()))?;
                                            }

                                            if !gs.already_reduced_lmr {
                                                env.move_orderer.update_improve_history(gs.teban, gs.pos.get_state(), m, gs.depth, gs.current_depth, gs.move_history)?;
                                            }
                                        }
                                    },
                                    LegalMove::Put(_) => {
                                        if !env.lazy_abort.load(Ordering::Acquire) {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;

                                            let _ = prev_move.map(|prev_move| {
                                                env.move_orderer.update_counter_move(m,gs.teban,prev_move,gs.prev_kind)
                                            }).unwrap_or(Ok(()))?;

                                            if !gs.already_reduced_lmr {
                                                env.move_orderer.update_improve_history(gs.teban,gs.pos.get_state(),m,gs.depth,gs.current_depth,gs.move_history)?;
                                            }
                                        }
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

                        if env.lazy_abort.load(Ordering::Acquire) && !scoreval.is_neginfinite() {
                            break;
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
                    EvaluationResult::Cut => {
                    }
                }
            }

            if quiet_alpha == start_alpha && !gs.already_reduced_lmr && !env.lazy_abort.load(Ordering::Acquire) {
                for m in quiet_moves {
                    env.move_orderer.update_degrade_history(gs.teban,gs.pos.get_state(),m,gs.depth)?;
                }
            }

            let bs = scoreval.normalize_score(gs.current_depth as i32);

            if scoreval > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::Exact, best_moves.front().map(|m| m.clone()));
            } else if !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::UpperBound, best_moves.front().map(|m| m.clone()));
            }

            env.history.remove(&(gs.teban,mk,sk));

            prev_move.map(|m| best_moves.push_front(m));

            Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
        } else {
            if env.threatmate_depth > 0 && gs.depth >= 4 && gs.gives_check_us {
                let checkmate = self.threatmate_search(gs.teban,
                                                       gs.teban,
                                                       gs.pos,
                                                       env,
                                                       event_dispatcher,
                                                       &gs.zh,
                                                       &mut HashSet::new(),
                                                       gs.threatmate_cache,
                                                       env.threatmate_depth as usize,
                                                       gs.current_depth as usize,
                                                       gs.rng)?;

                if let ThreatMateSearchResult::Checkmate(ply) = checkmate {
                    if !env.lazy_abort.load(Ordering::Acquire) {
                        env.transposition_table.update(&gs.zh, gs.depth as i8, TTScore::INFINITE(ply + (gs.current_depth as i32)), Bound::Exact, None);
                    }

                    let mut mvs = VecDeque::new();

                    gs.m.map(|m| mvs.push_front(m));

                    return Ok(EvaluationResult::Exact(Score::INFINITE(ply), mvs, gs.zh.clone(), gs.current_depth));
                }
            }

            if gs.depth == 0 {
                let s = self.qsearch(gs.teban,
                                     gs.pos,
                                     env,
                                     event_dispatcher,
                                     &gs.zh,
                                     &mut HashSet::new(),
                                     gs.self_partial_output,
                                     gs.opponent_partial_output,
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
                                         gs.pos,
                                         env,
                                         event_dispatcher,
                                         &gs.zh,
                                         &mut HashSet::new(),
                                         gs.self_partial_output,
                                         gs.opponent_partial_output,
                                         gs.alpha,
                                         gs.beta,
                                         0,
                                         gs.current_depth as usize,
                                         prev_move.clone(),
                                         evalutor,
                                         gs.rng)?;

                    if s < gs.alpha {
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

                    env.history.insert((gs.teban, mk, sk));

                    match self.search_null_move(env, gs, -gs.beta, -gs.beta + 1, gs.depth.saturating_sub(r), event_dispatcher, evalutor)? {
                        EvaluationResult::Exact(s, _, zh, _) | EvaluationResult::Repetition(s, _, zh, _) => {
                            let s = -s;

                            let null_value = s;

                            let mut best_moves = VecDeque::new();

                            gs.m.map(|m| best_moves.push_front(m));

                            if s >= gs.beta {
                                if let Score::Value(_) = s {
                                    if gs.nmp_min_ply.unwrap_or(0) == 0 || gs.depth < 16 {
                                        env.history.remove(&(gs.teban, mk, sk));

                                        return Ok(EvaluationResult::Exact(s, best_moves, zh, gs.current_depth));
                                    }

                                    let nmp_min_ply = (gs.current_depth as i32 + 3 * (gs.depth as i32 - r as i32) / 4).max(0) as u32;

                                    env.history.remove(&(gs.teban, mk, sk));

                                    let mut gs = GameState {
                                        teban: gs.teban,
                                        pos: gs.pos,
                                        rng: gs.rng,
                                        alpha: Score::Value(beta - 1),
                                        beta: Score::Value(beta),
                                        search_offset: gs.search_offset,
                                        best_score: gs.best_score,
                                        m: gs.m,
                                        static_eval: static_eval.clone(),
                                        gives_check_us:gs.gives_check_us,
                                        gives_check_them:gs.gives_check_them,
                                        prev_kind: gs.prev_kind,
                                        thread_index: gs.thread_index,
                                        pv: &VecDeque::new(),
                                        move_history: gs.move_history,
                                        threatmate_cache: gs.threatmate_cache,
                                        self_partial_output: gs.self_partial_output,
                                        opponent_partial_output: gs.opponent_partial_output,
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
                                        EvaluationResult::Exact(s, _, _, _) | EvaluationResult::Repetition(s, _, _, _) => {
                                            if s >= gs.beta {
                                                return Ok(EvaluationResult::Exact(null_value, best_moves, gs.zh, gs.current_depth));
                                            }
                                        },
                                        EvaluationResult::NodeLimits => {
                                            return Ok(EvaluationResult::NodeLimits);
                                        },
                                        EvaluationResult::Timeout => {
                                            return Ok(EvaluationResult::Timeout);
                                        },
                                        EvaluationResult::Stop => {
                                            return Ok(EvaluationResult::Stop);
                                        },
                                        EvaluationResult::Cut => {}
                                    }
                                }
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
                        EvaluationResult::Cut => {
                            env.history.remove(&(gs.teban, mk, sk));
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

            let mut picker = RandomPicker::new(Prng::new(gs.rng.rnd64()));

            let count = 2;

            env.history.insert((gs.teban, mk, sk));

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
                if i == 0 {
                    Rule::generate_moves::<CaptureOrPawnPromotions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;
                } else {
                    Rule::generate_moves::<QuietsWithoutPawnPromotions>(gs.teban, gs.pos.get_state(), gs.pos.get_mc(), &mut picker)?;
                }

                'outer: for (m, see) in env.move_orderer.ordering(
                    &mut picker, gs.current_depth, gs.teban, gs.pos.get_state(), tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history)? {
                    if m.obtained().is_none() {
                        quiet_moves.push(m);
                    }

                    // Futility Pruning
                    if !gs.cut_node && gs.depth >= 2 && gs.depth <= 4 &&
                        m.obtained().is_none() &&
                        !pv_move.map(|pm| pm == m).unwrap_or(false) &&
                        !Rule::in_check(gs.teban, gs.pos.get_state()) &&
                        !Rule::is_oute_move(gs.pos.get_state(), gs.teban, m) &&
                        Score::Value(gs.static_eval.get_or_insert_with(|| {
                            evalutor.evalute(&gs.self_partial_output)
                        })? + self.futility_margin(gs.depth, m)) <= alpha {
                        continue;
                    }

                    let mut r = self.calc_lmr(env,
                                              &mut lmr_index,
                                              gs.depth - 1,
                                              gs.current_depth,
                                              gs.teban,
                                              gs.pos.get_state(),
                                              m,
                                              see,
                                              tt_move.as_ref(),
                                              pv_move.as_ref(),
                                              &mut gs.static_eval,
                                              gs.self_partial_output,
                                              evalutor)?;

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
                            env.transposition_table.update(&gs.zh, gs.depth as i8, TTScore::INFINITE(0), Bound::Exact, Some(m));

                            let mut mvs = VecDeque::new();

                            mvs.push_front(m);
                            prev_move.map(|m| mvs.push_front(m));
                            env.history.remove(&(gs.teban, mk, sk));

                            return Ok(EvaluationResult::Exact(Score::INFINITE(-(gs.current_depth as i32)), mvs, gs.zh.clone(), gs.current_depth));
                        }

                        let pv = if pv_move.map(|pm| pm == m).unwrap_or(false) {
                            gs.pv
                        } else {
                            &pv_non
                        };

                        assert!(depth > 0);

                        match self.search_child_node(env, gs, m, pv, alpha, depth, gs.cut_node, lmr_reduced, gs.nmp_min_ply, event_dispatcher, evalutor)? {
                            EvaluationResult::Exact(s, mvs, _, seldepth) |
                            EvaluationResult::Repetition(s, mvs, _, seldepth) => {
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

                                    if scoreval >= beta || scoreval.is_infinite() {
                                        if !env.lazy_abort.load(Ordering::Acquire) {
                                            match scoreval {
                                                Score::INFINITE(_) => {
                                                    env.transposition_table.update(&gs.zh, depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                                },
                                                Score::NEGINFINITE(_) => {
                                                    env.transposition_table.update(&gs.zh, depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                                },
                                                _ => {
                                                    env.transposition_table.update(&gs.zh, depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::LowerBound, Some(m));
                                                }
                                            }
                                        }

                                        match m {
                                            LegalMove::To(mv) if mv.obtained().is_none() => {
                                                if !env.lazy_abort.load(Ordering::Acquire) {
                                                    if !mv.is_nari() {
                                                        env.move_orderer.update_killer(gs.current_depth, m)?;

                                                        let _ = prev_move.map(|prev_move| {
                                                            env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                        }).unwrap_or(Ok(()))?;
                                                    }

                                                    if !lmr_reduced {
                                                        env.move_orderer.update_improve_history(gs.teban, gs.pos.get_state(), m, gs.depth, gs.current_depth, gs.move_history)?;
                                                    }
                                                }
                                            },
                                            LegalMove::Put(_) => {
                                                if !env.lazy_abort.load(Ordering::Acquire) {
                                                    env.move_orderer.update_killer(gs.current_depth, m)?;

                                                    let _ = prev_move.map(|prev_move| {
                                                        env.move_orderer.update_counter_move(m, gs.teban, prev_move, gs.prev_kind)
                                                    }).unwrap_or(Ok(()))?;

                                                    if !lmr_reduced {
                                                        env.move_orderer.update_improve_history(gs.teban, gs.pos.get_state(), m, gs.depth, gs.current_depth, gs.move_history)?;
                                                    }
                                                }
                                            },
                                            _ => ()
                                        };
                                        env.history.remove(&(gs.teban, mk, sk));

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

                                if env.lazy_abort.load(Ordering::Acquire) && !scoreval.is_neginfinite() {
                                    break 'outer;
                                }

                                break;
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
                            EvaluationResult::Cut => {}
                        }
                    }
                }
            }

            if quiet_alpha == start_alpha && !gs.already_reduced_lmr && !env.lazy_abort.load(Ordering::Acquire) {
                for m in quiet_moves {
                    env.move_orderer.update_degrade_history(gs.teban, gs.pos.get_state(), m, gs.depth)?;
                }
            }

            let bs = scoreval.normalize_score(gs.current_depth as i32);

            if scoreval > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::Exact, best_moves.front().map(|m| m.clone()));
            } else if !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh, gs.depth as i8, bs, Bound::UpperBound, best_moves.front().map(|m| m.clone()));
            }

            env.history.remove(&(gs.teban, mk, sk));

            prev_move.map(|m| best_moves.push_front(m));

            Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
        }
    }
}
pub struct Inter<L, S, M>
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
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
                     PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                     ContinueForward + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
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

        let in_check = Rule::in_check(gs.teban, gs.pos.get_state());

        if in_check {
            for (m,_) in env.move_orderer.ordering(
                mvs.iter().cloned(), gs.current_depth, gs.teban, gs.pos.get_state(), tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history
            )?.skip(gs.search_offset) {
                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                if self.is_obtained_ou(m)? {
                    env.transposition_table.update(&gs.zh,gs.depth as i8,TTScore::INFINITE(0),Bound::Exact,Some(m));

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

                assert!(gs.depth > 0);

                match recur.search_child_node(env,gs,m,pv,alpha,gs.depth,gs.cut_node,gs.already_reduced_lmr,gs.nmp_min_ply,&mut event_dispatcher,evalutor)? {
                    EvaluationResult::Exact(s, mvs, _, seldepth) |
                    EvaluationResult::Repetition(s, mvs, _, seldepth) => {
                        let s = -s;

                        max_seldepth = max_seldepth.max(seldepth);

                        if s > scoreval {
                            scoreval = s;

                            best_moves = mvs;

                            if gs.thread_index == 0 {
                                recur.send_info(env, gs.depth, max_seldepth, &best_moves, &scoreval)?;
                            }

                            if scoreval >= beta || scoreval.is_infinite() {
                                if !env.lazy_abort.load(Ordering::Acquire) {
                                    match scoreval {
                                        Score::INFINITE(_) => {
                                            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                        },
                                        Score::NEGINFINITE(_) => {
                                            env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                        },
                                        _ => {
                                            env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval.normalize_score(gs.current_depth as i32),Bound::LowerBound,Some(m));
                                        }
                                    }
                                }

                                match m {
                                    LegalMove::To(mv) if mv.obtained().is_none() => {
                                        if !env.lazy_abort.load(Ordering::Acquire) {
                                            if !mv.is_nari() {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;
                                            }

                                            if !gs.already_reduced_lmr {
                                                env.move_orderer.update_improve_history(gs.teban,gs.pos.get_state(),m,gs.depth,gs.current_depth,gs.move_history)?;
                                            }
                                        }
                                    },
                                    LegalMove::Put(_) => {
                                        if !env.lazy_abort.load(Ordering::Acquire) {
                                            env.move_orderer.update_killer(gs.current_depth, m)?;

                                            if !gs.already_reduced_lmr {
                                                env.move_orderer.update_improve_history(gs.teban,gs.pos.get_state(),m,gs.depth,gs.current_depth,gs.move_history)?;
                                            }
                                        }
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

                        if env.lazy_abort.load(Ordering::Acquire) && !scoreval.is_neginfinite(){
                            break;
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
                    EvaluationResult::Cut => {
                    }
                }
            }

            if quiet_alpha == start_alpha && !gs.already_reduced_lmr && !env.lazy_abort.load(Ordering::Acquire) {
                for m in quiet_moves {
                    env.move_orderer.update_degrade_history(gs.teban, gs.pos.get_state(), m, gs.depth)?;
                }
            }

            let bs = scoreval.normalize_score(gs.current_depth as i32);

            if gs.search_offset == 0 && scoreval > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::Exact,best_moves.front().map(|m| m.clone()));
            } else if gs.search_offset == 0 && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::UpperBound,best_moves.front().map(|m| m.clone()));
            }

            env.history.remove(&(gs.teban,mk,sk));

            Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
        } else {
            for (m,see) in env.move_orderer.ordering(
                mvs.iter().cloned(), gs.current_depth, gs.teban, gs.pos.get_state(), tt_move, pv_move, gs.m, gs.prev_kind, gs.move_history
            )?.skip(gs.search_offset) {
                if m.obtained().is_none() {
                    quiet_moves.push(m);
                }

                let mut r = recur.calc_lmr(env,
                                           &mut quiet_index,
                                           gs.depth - 1,
                                           gs.current_depth,
                                           gs.teban,
                                           gs.pos.get_state(),
                                           m,
                                           see,
                                           tt_move.as_ref(),
                                           pv_move.as_ref(),
                                           &mut gs.static_eval,
                                           gs.self_partial_output,
                                           evalutor)?;

                if gs.already_reduced_lmr {
                    r = r.saturating_sub(1);
                }

                let mut lmr_reduced = gs.already_reduced_lmr || r > 0;

                'outer: for j in 0..2 {
                    let depth = if j == 0 {
                        gs.depth - r
                    } else {
                        gs.depth
                    };

                    if self.is_obtained_ou(m)? {
                        env.transposition_table.update(&gs.zh,gs.depth as i8,TTScore::INFINITE(0),Bound::Exact,Some(m));

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

                    assert!(depth > 0);

                    match recur.search_child_node(env,gs,m,pv,alpha,depth,gs.cut_node,lmr_reduced,gs.nmp_min_ply,&mut event_dispatcher,evalutor)? {
                        EvaluationResult::Exact(s, mvs, _, seldepth) |
                        EvaluationResult::Repetition(s, mvs, _, seldepth) => {
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

                                if scoreval >= beta || scoreval.is_infinite() {
                                    if !env.lazy_abort.load(Ordering::Acquire) {
                                        match scoreval {
                                            Score::INFINITE(_) => {
                                                env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                            },
                                            Score::NEGINFINITE(_) => {
                                                env.transposition_table.update(&gs.zh, gs.depth as i8, scoreval.normalize_score(gs.current_depth as i32), Bound::Exact, Some(m));
                                            },
                                            _ => {
                                                env.transposition_table.update(&gs.zh,gs.depth as i8,scoreval.normalize_score(gs.current_depth as i32),Bound::LowerBound,Some(m));
                                            }
                                        }
                                    }

                                    match m {
                                        LegalMove::To(mv) if mv.obtained().is_none() => {
                                            if !env.lazy_abort.load(Ordering::Acquire) {
                                                if !mv.is_nari() {
                                                    env.move_orderer.update_killer(gs.current_depth, m)?;
                                                }

                                                if !lmr_reduced {
                                                    env.move_orderer.update_improve_history(gs.teban,gs.pos.get_state(),m,gs.depth,gs.current_depth,gs.move_history)?;
                                                }
                                            }
                                        },
                                        LegalMove::Put(_) => {
                                            if !env.lazy_abort.load(Ordering::Acquire) {
                                                env.move_orderer.update_killer(gs.current_depth, m)?;

                                                if !lmr_reduced {
                                                    env.move_orderer.update_improve_history(gs.teban,gs.pos.get_state(),m,gs.depth,gs.current_depth,gs.move_history)?;
                                                }
                                            }
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

                            if env.lazy_abort.load(Ordering::Acquire) && !scoreval.is_neginfinite() {
                                break 'outer;
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
                        }
                    }
                }
            }

            if quiet_alpha == start_alpha && !gs.already_reduced_lmr && !env.lazy_abort.load(Ordering::Acquire) {
                for m in quiet_moves {
                    env.move_orderer.update_degrade_history(gs.teban, gs.pos.get_state(), m, gs.depth)?;
                }
            }

            let bs = scoreval.normalize_score(gs.current_depth as i32);

            if gs.search_offset == 0 && scoreval > start_alpha && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::Exact,best_moves.front().map(|m| m.clone()));
            } else if gs.search_offset == 0 && !env.lazy_abort.load(Ordering::Acquire) {
                env.transposition_table.update(&gs.zh,gs.depth as i8,bs,Bound::UpperBound,best_moves.front().map(|m| m.clone()));
            }

            env.history.remove(&(gs.teban,mk,sk));

            Ok(EvaluationResult::Exact(scoreval, best_moves, gs.zh.clone(), max_seldepth))
        }
    }
}