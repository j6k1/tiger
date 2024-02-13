use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use nncombinator::arr::Arr;
use nncombinator::layer::{ForwardAll, PreTrain};
use rand::Rng;
use rand::rngs::ThreadRng;
use rayon::ThreadPool;
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::KyokumenHash;
use usiagent::logger::Logger;
use usiagent::math::Prng;
use usiagent::movepick::{MovePicker, RandomPicker};
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{CaptureOrPawnPromotions, LegalMove, QuietsWithoutPawnPromotions, Rule, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::ApplicationError;
use crate::nn::Evalutor;
use crate::transposition_table::{TT, TTPartialEntry, ZobristHash};

pub const TURN_LIMIT:u32 = 10000;
pub const BASE_DEPTH:u32 = 14;
pub const MAX_DEPTH:u32 = 14;
pub const MAX_THREADS:u32 = 8;
pub const FACTOR_FOR_NUMBER_OF_NODES_PER_THREAD:u8 = 14;
pub const NODES_PER_LEAF_NODE:u16 = 5;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Score {
    NEGINFINITE,
    Value(i32),
    INFINITE,
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::Value(v) => Score::Value(-v),
            Score::INFINITE => Score::NEGINFINITE,
            Score::NEGINFINITE => Score::INFINITE,
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
        }
    }
}
impl Default for Score {
    fn default() -> Self {
        Score::NEGINFINITE
    }
}
pub struct Environment<L,S> where L: Logger, S: InfoSender {
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub limit:Option<Instant>,
    pub turn_limit:Option<Instant>,
    pub base_depth:u32,
    pub max_depth:u32,
    pub max_threads:u32,
    pub factor_nodes_per_thread:u8,
    pub nodes_per_leaf_node:u16,
    pub abort:Arc<AtomicBool>,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
    pub nodes:Arc<AtomicU64>
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            hasher:Arc::clone(&self.hasher),
            limit:self.limit.clone(),
            turn_limit:self.turn_limit.clone(),
            base_depth:self.base_depth,
            max_depth:self.max_depth,
            max_threads:self.max_threads,
            factor_nodes_per_thread:self.factor_nodes_per_thread,
            nodes_per_leaf_node:self.nodes_per_leaf_node,
            abort:Arc::clone(&self.abort),
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            transposition_table:self.transposition_table.clone(),
            nodes:Arc::clone(&self.nodes),
        }
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>),
    Timeout
}
#[derive(Debug)]
pub enum RootEvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>, u32),
    Timeout
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               limit:Option<Instant>,
               turn_limit:Option<Instant>,
               base_depth:u32,
               max_depth:u32,
               max_threads:u32,
               factor_nodes_per_thread:u8,
               nodes_per_leaf_node:u16,
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
            limit:limit,
            turn_limit:turn_limit,
            base_depth:base_depth,
            max_depth:max_depth,
            max_threads:max_threads,
            factor_nodes_per_thread:factor_nodes_per_thread,
            nodes_per_leaf_node:nodes_per_leaf_node,
            abort:abort,
            stop:stop,
            quited:quited,
            transposition_table:Arc::clone(transposition_table),
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
    pub best_score:Score,
    pub m:Option<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub zh:ZobristHash<u64>,
    pub depth:u32,
    pub current_depth:u32,
    pub base_depth:u32,
    pub max_depth:u32
}
pub struct Root<L,S,M> where L: Logger + Send + 'static,
                             S: InfoSender,
                             M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                PreTrain<f32> + Send + Sync + 'static,
                             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>,
    receiver:Receiver<Result<RootEvaluationResult, ApplicationError>>,
    sender:Sender<Result<RootEvaluationResult, ApplicationError>>,
    thread_pool:ThreadPool
}
const TIMELIMIT_MARGIN:u64 = 50;

pub trait Search<L,S,M>: Sized where L: Logger + Send + 'static,
                                     S: InfoSender,
                                     M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                        PreTrain<f32> + Send + Sync + 'static,
                                     <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError>;
    fn qsearch(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
               mut alpha:Score,beta:Score,evalutor: &Arc<Evalutor<M>>,rng:&mut ThreadRng) -> Result<Score,ApplicationError> {
        let mut score = Score::Value(evalutor.evalute(teban,state.get_banmen(),mc)?);

        if score >= beta {
            return Ok(score);
        }

        if score > alpha {
            alpha = score;
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        Rule::legal_moves_from_banmen_by_strategy::<CaptureOrPawnPromotions>(teban,state,&mut picker)?;

        if picker.len() == 0 {
            return Ok(alpha);
        }

        let mut bestscore = Score::NEGINFINITE;

        for m in picker {
            if let Some(ObtainKind::Ou) = match m {
                LegalMove::To(m) => m.obtained(),
                _ => None
            } {
                return Ok(Score::INFINITE);
            }

            let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

            score = -self.qsearch(teban.opposite(),&next,&nmc,-beta,-alpha,evalutor,rng)?;

            if score >= beta {
                return Ok(score);
            }

            if score > bestscore {
                bestscore = score;
            }

            if score > alpha {
                alpha = score;
            }
        }

        Ok(bestscore)
    }

    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> bool {
        env.turn_limit.map(|l| l - Instant::now() <= Duration::from_millis(TIMELIMIT_MARGIN)).unwrap_or(false)
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

        env.info_sender.send(commands)?;
        Ok(env.info_sender.flush()?)
    }

    fn update_tt<'a>(&self, env: &mut Environment<L, S>,
                     zh: &'a ZobristHash<u64>,
                     depth: u32,
                     score: Score,
                     beta: Score,
                     alpha:Score) {
        let mut tte = env.transposition_table.entry(&zh);
        let tte = tte.or_default();

        if tte.depth == -1 || score == Score::INFINITE ||
            (tte.beta >= beta && tte.alpha <= alpha && tte.depth < depth as i8 - 1) ||
            (tte.depth == depth as i8 - 1 && tte.score < score) {
            tte.depth = depth as i8 - 1;
            tte.score = score;
            tte.beta = beta;
            tte.alpha = alpha;
        }
    }

    fn update_best_move<'a>(&self, env: &mut Environment<L, S>,
                            zh: &'a ZobristHash<u64>,
                            depth: u32,
                            score:Score,
                            beta:Score,
                            alpha:Score,
                            m: Option<LegalMove>) {
        let mut tte = env.transposition_table.entry(zh);
        let tte = tte.or_default();

        if tte.depth == -1 || score == Score::INFINITE ||
            (tte.beta >= beta && tte.alpha <= alpha && tte.depth < depth as i8) ||
            (tte.depth == depth as i8 && tte.score < score) {
            tte.depth = depth as i8;
            tte.score = score;
            tte.beta = beta;
            tte.alpha = alpha;
            tte.best_move = m;
        }
    }
}
impl<L,S,M> Root<L,S,M> where L: Logger + Send + 'static,
                              S: InfoSender,
                              M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
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

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
                                         -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler);

        {
            let stop = stop.clone();

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
            let stop = stop.clone();
            let quited = quited.clone();

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

        event_dispatcher
    }

    fn start_thread<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           evalutor: &Arc<Evalutor<M>>) {
        let sender = self.sender.clone();
        let teban = gs.teban;
        let state = Arc::clone(&gs.state);
        let mut env = env.clone();
        let evalutor = Arc::clone(&evalutor);
        let mc = Arc::clone(&gs.mc);
        let zh = gs.zh.clone();
        let depth = gs.depth;
        let current_depth = 0;
        let base_depth = gs.base_depth;
        let max_depth = gs.max_depth;
        let best_score = gs.best_score;

        self.thread_pool.spawn(move || {
            let mut event_dispatcher = Self::create_event_dispatcher::<Recursive<L,S,M>>(&env.on_error_handler, &env.stop, &env.quited);

            let mut rng = rand::thread_rng();

            let mut gs = GameState {
                teban: teban,
                state: &state,
                alpha: Score::NEGINFINITE,
                beta: Score::INFINITE,
                best_score: best_score,
                m: None,
                mc: &mc,
                zh: zh,
                depth: depth,
                current_depth: current_depth + 1,
                base_depth: base_depth,
                max_depth: max_depth,
                rng:&mut rng
            };

            let strategy = Recursive::new();

            let r = strategy.search(&mut env, &mut gs, &mut event_dispatcher, &evalutor);

            match r {
                Ok(EvaluationResult::Immediate(score,mvs,zh)) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Immediate(score,mvs,zh,depth)));
                },
                Ok(EvaluationResult::Timeout) => {
                    let _ = sender.send(Ok(RootEvaluationResult::Timeout));
                },
                Err(e) => {
                    let _ = sender.send(Err(e));
                }
            }
        });
    }

    fn termination(&self,env:&mut Environment<L,S>,mut busy_threads:u32) -> Result<(),ApplicationError> {
        env.abort.store(true,Ordering::Release);

        while busy_threads > 0 {
            self.receiver.recv().map_err(|e| ApplicationError::from(e))??;

            busy_threads -= 1;
        }

        Ok(())
    }
}
impl<L,S,M> Search<L,S,M> for Root<L,S,M> where L: Logger + Send + 'static,
                                            S: InfoSender,
                                            M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                               PreTrain<f32> + Send + Sync + 'static,
                                            <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     _:&mut UserEventDispatcher<'b,Root<L,S,M>,ApplicationError,L>,
                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult,ApplicationError> {
        let base_depth = gs.depth.min(env.base_depth);
        let mut depth = 1;
        let mut thread_index = 0;
        let nodes_per_leaf_node = env.nodes_per_leaf_node as u128;
        let nodes_per_thread:u128 = nodes_per_leaf_node.pow(env.factor_nodes_per_thread as u32) as u128;
        let mut search_space:u128 = nodes_per_leaf_node as u128 * 4;
        let mut busy_threads = 0;
        let mut last_depth = depth;
        let mut result = None;

        env.abort.store(false,Ordering::Release);

        loop {
            if busy_threads == env.max_threads {
                match self.receiver.recv().map_err(|e| ApplicationError::from(e))? {
                    Ok(RootEvaluationResult::Immediate(s, mvs, zh,depth)) if base_depth <= depth => {
                        busy_threads -= 1;

                        self.termination(env,busy_threads)?;

                        return Ok(EvaluationResult::Immediate(s, mvs, zh));
                    },
                    Ok(RootEvaluationResult::Immediate(s, mvs, zh,depth)) => {
                        busy_threads -= 1;

                        if depth >= last_depth {
                            last_depth = depth;

                            if s > gs.best_score {
                                gs.best_score = s;
                            }

                            result = Some(EvaluationResult::Immediate(s, mvs, zh));
                        }
                    },
                    Ok(RootEvaluationResult::Timeout) => {
                        busy_threads -= 1;

                        self.termination(env,busy_threads)?;

                        return Ok(result.unwrap_or(EvaluationResult::Timeout));
                    },
                    Err(e) => {
                        busy_threads -= 1;

                        self.termination(env,busy_threads)?;

                        return Err(e);
                    }
                }
            } else {
                if depth < base_depth && thread_index * nodes_per_thread >= search_space {
                    depth += 1;
                    search_space = search_space * nodes_per_leaf_node;
                }

                gs.depth = depth;
                gs.base_depth = depth;
                gs.max_depth = env.max_depth - (base_depth - depth);

                self.start_thread(env,gs,evalutor);

                busy_threads += 1;
                thread_index += 1;
            }
        }
    }
}
pub struct Recursive<L,S,M> where L: Logger + Send + 'static,
                                  S: InfoSender,
                                  M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                     PreTrain<f32> + Send + Sync + 'static,
                                  <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    l:PhantomData<L>,
    s:PhantomData<S>,
    m:PhantomData<M>
}
impl<L,S,M> Recursive<L,S,M> where L: Logger + Send + 'static,
                                 S: InfoSender,
                                 M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                    PreTrain<f32> + Send + Sync + 'static,
                                 <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn new() -> Recursive<L,S,M> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            m:PhantomData::<M>
        }
    }

    pub fn search_child_node<'a,'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                                     m:LegalMove,alpha:Score,
                                     event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                                     evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        let o = match m {
            LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
            _ => None
        };

        let mut depth = gs.depth;

        if o.is_some() {
            depth += 1;
        }

        let zh = gs.zh.updated(&env.hasher, gs.teban, gs.state.get_banmen(), gs.mc, m.to_applied_move(), &o);

        let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

        match next {
            (state, mc, _) => {
                let state = Arc::new(state);

                if Rule::is_mate(gs.teban.opposite(),&state) {
                    let mut mvs = VecDeque::new();
                    mvs.push_front(m);

                    return Ok(EvaluationResult::Immediate(Score::INFINITE,mvs,zh));
                }

                let mc = Arc::new(mc);

                let mut gs = GameState {
                    teban: gs.teban.opposite(),
                    state: &state,
                    rng: gs.rng,
                    alpha: -gs.beta,
                    beta: -alpha,
                    best_score: gs.best_score,
                    m: Some(m),
                    mc: &mc,
                    zh: zh.clone(),
                    depth: depth - 1,
                    current_depth: gs.current_depth + 1,
                    base_depth: gs.base_depth,
                    max_depth: gs.max_depth
                };

                let strategy = Recursive::new();

                strategy.search(env, &mut gs, event_dispatcher, evalutor)
            }
        }
    }
}
impl<L,S,M> Search<L,S,M> for Recursive<L,S,M> where L: Logger + Send + 'static,
                                                     S: InfoSender,
                                                     M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                                        PreTrain<f32> + Send + Sync + 'static,
                                                     <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L,S,M>, ApplicationError, L>,
                      evalutor: &Arc<Evalutor<M>>) -> Result<EvaluationResult, ApplicationError> {
        env.nodes.fetch_add(1,Ordering::Release);

        if self.timelimit_reached(env) || env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout);
        }

        if let Some(prev_move) = gs.m.clone() {
            let r = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone());

            if let Some(TTPartialEntry {
                            depth: d,
                            score: s,
                            beta,
                            alpha,
                            best_move: _
                        }) = r {

                match s {
                    Score::INFINITE => {
                        let mut mvs = VecDeque::new();

                        mvs.push_front(prev_move);

                        return Ok(EvaluationResult::Immediate(Score::INFINITE,mvs,gs.zh.clone()));
                    },
                    Score::NEGINFINITE => {
                        let mut mvs = VecDeque::new();

                        mvs.push_front(prev_move);

                        return Ok(EvaluationResult::Immediate(Score::NEGINFINITE,mvs,gs.zh.clone()));
                    },
                    Score::Value(s) if d as u32 >= gs.depth && beta >= gs.beta && alpha <= gs.alpha => {
                        let mut mvs = VecDeque::new();

                        mvs.push_front(prev_move);

                        return Ok(EvaluationResult::Immediate(Score::Value(s),mvs,gs.zh.clone()));
                    },
                    _ => ()
                }
            }
        }

        let prev_move = gs.m.clone();

        let obtained = match prev_move {
            Some(m) => {
                match m {
                    LegalMove::To(m) => m.obtained(),
                    _ => None
                }
            },
            None => None
        };

        if let Some(ObtainKind::Ou) = obtained {
            let mut mvs = VecDeque::new();

            prev_move.map(|m| mvs.push_front(m));

            return Ok(EvaluationResult::Immediate(Score::NEGINFINITE,mvs,gs.zh.clone()));
        }

        if gs.depth == 0 || gs.current_depth >= gs.max_depth {
            let s = self.qsearch(gs.teban,&gs.state,&gs.mc,gs.alpha,gs.beta,evalutor,gs.rng)?;

            let mut mvs = VecDeque::new();

            prev_move.map(|m| mvs.push_front(m));

            return Ok(EvaluationResult::Immediate(s,mvs,gs.zh.clone()))
        }

        let start_alpha = gs.alpha;
        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        let mut picker = RandomPicker::new(Prng::new(gs.rng.gen()));

        for i in 0..3 {
            if i == 0 {
                if let Some(TTPartialEntry {
                                depth: _,
                                score: _,
                                beta: _,
                                alpha: _,
                                best_move: m
                            }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
                    if let Some(m) = m {
                        match self.search_child_node(env, gs, m, alpha, event_dispatcher, evalutor)? {
                            EvaluationResult::Immediate(s, mvs, zh) => {
                                self.update_tt(env, &zh, gs.depth, s, -alpha, -beta);

                                let s = -s;

                                if s > scoreval {
                                    scoreval = s;

                                    best_moves = mvs;

                                    if gs.current_depth == 1 && s > gs.best_score {
                                        self.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
                                    }

                                    self.update_best_move(env, &gs.zh, gs.depth, scoreval, beta, start_alpha, Some(m));

                                    if scoreval >= beta {
                                        prev_move.map(|m| best_moves.push_front(m));
                                        return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                    }
                                }

                                if alpha < s {
                                    alpha = s;
                                }
                            },
                            EvaluationResult::Timeout => {
                                return Ok(EvaluationResult::Timeout);
                            }
                        }
                    }
                }

                continue;
            } else if i == 1 {
                Rule::legal_moves_all_by_strategy::<CaptureOrPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            } else {
                Rule::legal_moves_all_by_strategy::<QuietsWithoutPawnPromotions>(gs.teban, &gs.state, &gs.mc, &mut picker)?;
            }

            for m in &mut picker {
                match self.search_child_node(env,gs,m,alpha,event_dispatcher,evalutor)? {
                    EvaluationResult::Immediate(s, mvs, zh) => {
                        self.update_tt(env, &zh, gs.depth, s, -alpha, -beta);

                        let s = -s;

                        if s > scoreval {
                            scoreval = s;

                            best_moves = mvs;

                            self.update_best_move(env, &gs.zh, gs.depth, scoreval, beta, start_alpha, Some(m));

                            if gs.current_depth == 1 && s > gs.best_score {
                                self.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
                            }

                            if scoreval >= beta {
                                prev_move.map(|m| best_moves.push_front(m));
                                return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                            }
                        }

                        if alpha < s {
                            alpha = s;
                        }
                    },
                    EvaluationResult::Timeout => {
                        return Ok(EvaluationResult::Timeout);
                    }
                }
            }
        }

        prev_move.map(|m| best_moves.push_front(m));

        Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()))
    }
}