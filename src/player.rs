use std::collections::{BTreeMap, HashSet, VecDeque};
use std::{fmt, fs};
use std::fs::DirEntry;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use nncombinator::arr::Arr;
use nncombinator::layer::{ForwardAll, PreTrain};
use rayon::ThreadPoolBuilder;
use usiagent::command::{BestMove, CheckMate, UsiInfoSubCommand, UsiOptType};
use usiagent::error::{PlayerError, UsiProtocolError};
use usiagent::event::{GameEndState, SysEventOption, SysEventOptionKind, UserEvent, UserEventQueue, UsiGoMateTimeLimit, UsiGoTimeLimit};
use usiagent::hash::{KyokumenHash};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::output::USIOutputWriter;
use usiagent::player::{InfoSender, OnKeepAlive, PeriodicallyInfo, USIPlayer};
use usiagent::rule::{AppliedMove, Kyokumen, State};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MochigomaCollections, Move, Teban};
use crate::error::ApplicationError;
use crate::features::HalfKP;
use crate::nn::{Evalutor, FEATURES_NUM};
use crate::search::{BASE_DEPTH, Environment, EvaluationResult, GameState, MAX_THREADS, NODES_PER_LEAF_NODE, Root, Score, Search, TURN_LIMIT, GAMMA, TIMELIMIT_MARGIN};
use crate::transposition_table::{TT, ZobristHash};

pub trait FromOption {
    fn from_option(option:SysEventOption) -> Option<Self> where Self: Sized;
}
impl FromOption for i64 {
    fn from_option(option: SysEventOption) -> Option<i64> {
        match option {
            SysEventOption::Num(v) => Some(v),
            _ => None
        }
    }
}
impl FromOption for u64 {
    fn from_option(option: SysEventOption) -> Option<u64> {
        match option {
            SysEventOption::Num(v) => Some(v as u64),
            _ => None
        }
    }
}
impl FromOption for u32 {
    fn from_option(option: SysEventOption) -> Option<u32> {
        match option {
            SysEventOption::Num(v) => Some(v as u32),
            _ => None
        }
    }
}
impl FromOption for u16 {
    fn from_option(option: SysEventOption) -> Option<u16> {
        match option {
            SysEventOption::Num(v) => Some(v as u16),
            _ => None
        }
    }
}
impl FromOption for u8 {
    fn from_option(option: SysEventOption) -> Option<u8> {
        match option {
            SysEventOption::Num(v) => Some(v as u8),
            _ => None
        }
    }
}
impl FromOption for usize {
    fn from_option(option: SysEventOption) -> Option<usize> {
        match option {
            SysEventOption::Num(v) => Some(v as usize),
            _ => None
        }
    }
}
impl FromOption for bool {
    fn from_option(option: SysEventOption) -> Option<bool> {
        match option {
            SysEventOption::Bool(b) => Some(b),
            _ => None
        }
    }
}
impl FromOption for String {
    fn from_option(option: SysEventOption) -> Option<String> {
        match option {
            SysEventOption::Str(s) => Some(s),
            _ => None
        }
    }
}
pub struct Tiger<M> where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                             PreTrain<f32> + Send + Sync + 'static,
                          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    evalutor_creator: Box<dyn Fn(String) -> Result<Evalutor<M>,ApplicationError> + Send + 'static>,
    evalutor: Option<Arc<Evalutor<M>>>,
    kyokumen:Option<Kyokumen>,
    zh:Option<ZobristHash<u64>>,
    hasher:Arc<KyokumenHash<u64>>,
    transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
    search_id:Arc<AtomicUsize>,
    base_depth:u32,
    max_nodes:Option<u64>,
    max_threads:u32,
    nodes_per_leaf_node:u16,
    gamma:u8,
    turn_limit:Option<u32>,
    timelimit_margin:u64,
    model_name:String
}
impl<M> fmt::Debug for Tiger<M> where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                         PreTrain<f32> + Send + Sync + 'static,
                                      <M as PreTrain<f32>>::OutStack: Send + Sync + 'static{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tiger")
    }
}
impl<M> Tiger<M> where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                          PreTrain<f32> + Send + Sync + 'static,
                       <M as PreTrain<f32>>::OutStack: Send + Sync + 'static{
    pub fn new<C: Fn(String) -> Result<Evalutor<M>,ApplicationError> + Send + Sync + 'static>(evalutor_creator:C) -> Tiger<M> {
        Tiger {
            evalutor_creator:Box::new(evalutor_creator),
            evalutor:None,
            kyokumen:None,
            zh:None,
            hasher:Arc::new(KyokumenHash::new()),
            transposition_table:Arc::new(TT::new()),
            search_id:Arc::new(AtomicUsize::new(0)),
            base_depth:BASE_DEPTH,
            max_nodes:None,
            max_threads:MAX_THREADS,
            nodes_per_leaf_node:NODES_PER_LEAF_NODE,
            gamma:GAMMA,
            turn_limit:None,
            timelimit_margin:TIMELIMIT_MARGIN,
            model_name: String::from("nn.bin")
        }
    }

    fn think_common<L,S,P>(&mut self,
                           mut env:Environment<L,S>,
                           periodically_info:P,
                           on_error_handler:Arc<Mutex<OnErrorHandler<L>>>) -> Result<BestMove,ApplicationError>
    where L: Logger + Send + 'static,
          S: InfoSender,
          P: PeriodicallyInfo {
        let (teban, state, mc) = self.kyokumen.as_ref().map(|k| (k.teban, &k.state, &k.mc)).ok_or(
            UsiProtocolError::InvalidState(
                String::from("Position information is not initialized."))
        )?;

        let base_depth = env.base_depth;

        let turn_limit = env.turn_limit.clone();
        let limit = env.limit.clone();

        let mut event_dispatcher = Root::<L,S,M>::create_event_dispatcher(
            &on_error_handler,&env.stop,&env.quited,teban,&limit,&turn_limit,&env.current_limit
        );

        match self.evalutor {
            Some(ref evalutor) => {
                let _pinfo_sender = {
                    let nodes = env.nodes.clone();
                    let mut prev_time = Instant::now();
                    let mut prev_nodes = 0;
                    let on_error_handler = env.on_error_handler.clone();

                    periodically_info.start(1000,move || {
                        let mut commands = vec![];
                        commands.push(UsiInfoSubCommand::Nodes(nodes.load(Ordering::Acquire)));

                        let now = Instant::now();
                        let current_nodes = nodes.load(Ordering::Acquire);

                        let msec = (now - prev_time).as_millis();

                        if msec > 0 {
                            commands.push(UsiInfoSubCommand::Nps(
                                ((current_nodes - prev_nodes) as u128 * 1000 / msec) as u64
                            ));

                            prev_time = now;
                            prev_nodes = current_nodes;
                        }

                        commands
                    }, &on_error_handler)
                };

                let zh = match self.zh.as_ref() {
                    Some(zh) => zh.clone(),
                    None => {
                        return Err(ApplicationError::InvalidStateError(format!("ZobristHash is not initialized!")))
                    }
                };

                let mut rng = rand::thread_rng();

                let mut gs = GameState {
                    teban: teban,
                    state: &Arc::new(state.clone()),
                    rng:&mut rng,
                    alpha: Score::NEGINFINITE,
                    beta: Score::INFINITE,
                    search_offset: 0,
                    best_score: Score::NEGINFINITE,
                    m:None,
                    prev_kind: KomaKind::Blank,
                    pv:&VecDeque::new(),
                    move_history:&mut Vec::new(),
                    mc: &Arc::new(mc.clone()),
                    zh:zh,
                    prev_self_ss: None,
                    prev_opponent_ss: None,
                    depth:base_depth,
                    current_depth:0,
                    current_max_ply:1,
                    base_depth:base_depth,
                    extend_depth:2,
                    extend_check:1,
                    extend_threatmate:1,
                };

                let strategy  = Root::new(ThreadPoolBuilder::new()
                    .num_threads(self.max_threads as usize)
                    .stack_size(1024 * 1024 * 200).build()?);

                let result = strategy.search(&mut env,&mut gs, &mut event_dispatcher, &evalutor);

                let bestmove = match result {
                    Err(ref e) => {
                        strategy.send_message(&mut env,format!("error {}",&e).as_str())?;
                        env.info_sender.flush()?;

                        let _ = env.on_error_handler.lock().map(|h| h.call(e));
                        BestMove::Resign
                    },
                    Ok(EvaluationResult::NodeLimits) => {
                        strategy.send_message(&mut env,"node limits!")?;
                        env.info_sender.flush()?;

                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Timeout) => {
                        strategy.send_message(&mut env,"timeout!")?;
                        env.info_sender.flush()?;

                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Stop) => {
                        strategy.send_message(&mut env,"stop!")?;
                        env.info_sender.flush()?;

                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Repetition) => {
                        strategy.send_message(&mut env,"repetition!")?;
                        env.info_sender.flush()?;

                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Immediate(Score::NEGINFINITE,_,_)) => {
                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Immediate(_,mvs,_)) if mvs.len() == 0 => {
                        BestMove::Resign
                    },
                    Ok(EvaluationResult::Immediate(_,mvs,_)) if mvs.len() >= 2 => {
                        BestMove::Move(mvs[0].to_move(),Some(mvs[1].to_move()))
                    },
                    Ok(EvaluationResult::Immediate(_,mvs,_)) => {
                        BestMove::Move(mvs[0].to_move(),None)
                    }
                };

                Ok(bestmove)
            },
            None =>  {
                Err(ApplicationError::InvalidStateError(format!("evalutor is not initialized!")))
            }
        }
    }
}
impl<M> USIPlayer<ApplicationError> for Tiger<M> where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                                          PreTrain<f32> + Send + Sync + 'static,
                                                      <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    const ID: &'static str = "tiger";
    const AUTHOR: &'static str = "j6k1";

    fn get_option_kinds(&mut self) -> Result<BTreeMap<String,SysEventOptionKind>,ApplicationError> {
        let mut kinds:BTreeMap<String,SysEventOptionKind> = BTreeMap::new();

        kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
        kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
        kinds.insert(String::from("Threads"),SysEventOptionKind::Num);
        kinds.insert(String::from("BaseDepth"),SysEventOptionKind::Num);
        kinds.insert(String::from("MaxNodes"),SysEventOptionKind::Num);
        kinds.insert(String::from("TurnLimit"),SysEventOptionKind::Num);
        kinds.insert(String::from("TIMELIMIT_MARGIN"),SysEventOptionKind::Num);
        kinds.insert(String::from("NodesPerLeafNodes"),SysEventOptionKind::Num);
        kinds.insert(String::from("gamma"),SysEventOptionKind::Num);
        kinds.insert(String::from("ModelFile"),SysEventOptionKind::Str);

        Ok(kinds)
    }

    fn get_options(&mut self) -> Result<BTreeMap<String,UsiOptType>,ApplicationError> {
        let mut options:BTreeMap<String,UsiOptType> = BTreeMap::new();
        let mut paths = fs::read_dir(Path::new("data"))?.into_iter()
            .collect::<Vec<Result<DirEntry,_>>>();

        paths.sort_by(|a,b| {
            match (a,b) {
                (Ok(a),Ok(b)) => {
                    let a = a.file_name();
                    let b = b.file_name();
                    a.cmp(&b)
                },
                _ => {
                    std::cmp::Ordering::Equal
                }
            }
        });

        let paths = paths.into_iter().filter(|ent| {
            ent.as_ref().map(|e| e.path().as_path().extension().map(|ext| ext == "bin").unwrap_or(false)).unwrap_or(false)
        }).map(|ent| {
            ent.as_ref().map(|e| e.path().as_path().file_name().map(|s| {
                s.to_string_lossy().to_string()
            }).unwrap_or(String::from(""))).unwrap_or(String::from(""))
        }).filter(|f| !f.is_empty()).collect::<Vec<String>>();

        options.insert(String::from("BaseDepth"),UsiOptType::Spin(1,100,Some(BASE_DEPTH as i64)));
        options.insert(String::from("MaxNodes"),UsiOptType::Spin(0,i64::MAX,Some(0)));
        options.insert(String::from("Threads"),UsiOptType::Spin(1,1024,Some(MAX_THREADS as i64)));
        options.insert(String::from("TurnLimit"),UsiOptType::Spin(1,3600000,Some(TURN_LIMIT as i64)));
        options.insert(String::from("TIMELIMIT_MARGIN"),UsiOptType::Spin(0,60000,Some(TIMELIMIT_MARGIN as i64)));
        options.insert(String::from("NodesPerLeafNodes"), UsiOptType::Spin(1,593,Some(NODES_PER_LEAF_NODE as i64)));
        options.insert(String::from("gamma"), UsiOptType::Spin(1,500,Some(GAMMA as i64)));
        options.insert(String::from("ModelFile"),UsiOptType::Combo(Some(String::from("nn.bin")),paths));

        Ok(options)
    }

    fn take_ready<W,L>(&mut self, _:OnKeepAlive<W,L>)
                       -> Result<(),ApplicationError> where W: USIOutputWriter + Send + 'static,
                                                            L: Logger + Send + 'static {
        match self.evalutor {
            Some(_) => (),
            None => {
                self.evalutor = Some(Arc::new((self.evalutor_creator)(self.model_name.clone())?))
            }
        }
        Ok(())
    }

    fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),ApplicationError> {
        match &*name {
            "BaseDepth" => {
                self.base_depth = u32::from_option(value).unwrap_or(BASE_DEPTH);
            },
            "MaxNodes" => {
                self.max_nodes = u64::from_option(value).and_then(|n| {
                    if n == 0 {
                        None
                    } else {
                        Some(n)
                    }
                });
            },
            "Threads" => {
                self.max_threads = u32::from_option(value).unwrap_or(MAX_THREADS);
            },
            "TurnLimit" => {
                self.turn_limit = u32::from_option(value);
            },
            "TIMELIMIT_MARGIN" => {
                self.timelimit_margin = u64::from_option(value).unwrap_or(TIMELIMIT_MARGIN);
            },
            "NodesPerLeafNodes" => {
                self.nodes_per_leaf_node = u16::from_option(value).unwrap_or(NODES_PER_LEAF_NODE);
            },
            "gamma" => {
                self.gamma = u8::from_option(value).unwrap_or(GAMMA);
            },
            "ModelFile" => {
                self.model_name = String::from_option(value).unwrap_or(String::from("nn.bin"));
            },
            _ => ()
        }

        Ok(())
    }

    fn newgame(&mut self) -> Result<(),ApplicationError> {
        self.kyokumen = None;

        match Arc::get_mut(&mut self.transposition_table) {
            Some(transposition_table) => {
                transposition_table.clear();
            },
            None => {
                return Err(ApplicationError::InvalidStateError(String::from(
                    "Failed to get mutable reference for transposition_table."
                )));
            }
        }
        Ok(())
    }
    fn set_position(&mut self,teban:Teban,banmen:Banmen,
                    ms:Mochigoma,mg:Mochigoma,_:u32,m:Vec<Move>)
                    -> Result<(),ApplicationError> {
        let zh = ZobristHash::new(&self.hasher,teban,&banmen,&ms,&mg);

        let teban = teban;
        let state = State::new(banmen);

        let mc = MochigomaCollections::new(ms,mg);

        let (t,state,mc,r) = self.apply_moves(state,teban, mc,&m.into_iter()
            .map(|m| m.to_applied_move())
            .collect::<Vec<AppliedMove>>(),
                                              zh,
                                              |_,t,banmen,mc,m,o,r| {
                                                  let mut zh = r;

                                                  let zh = match m {
                                                      &Some(m) => {
                                                          zh = zh.updated(&self.hasher,t,&banmen,&mc,m,&o);
                                                          zh
                                                      },
                                                      &None => {
                                                          zh
                                                      }
                                                  };
                                                  zh
                                              });

        let zh = r;

        self.kyokumen = Some(Kyokumen {
            state:state,
            mc:mc,
            teban:t
        });
        self.zh = Some(zh);
        Ok(())
    }

    fn think<L,S,P>(&mut self,think_start_time:Instant,
                    limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
                    info_sender:S,periodically_info:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
                    -> Result<BestMove,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        let env = {
            let limit = Some(limit.clone());

            let transposition_table = Arc::clone(&self.transposition_table);
            let hasher = Arc::clone(&self.hasher);

            let (teban, _, _) = self.kyokumen.as_ref().map(|k| (k.teban, &k.state, &k.mc)).ok_or(
                UsiProtocolError::InvalidState(
                    String::from("Position information is not initialized."))
            )?;

            let env = Environment::new(
                Arc::clone(&event_queue),
                info_sender.clone(),
                Arc::clone(&on_error_handler),
                hasher,
                Arc::clone(&self.search_id),
                teban,
                limit,
                self.turn_limit,
                self.timelimit_margin,
                (
                    self.turn_limit.map(|l| think_start_time + Duration::from_millis(l as u64)),
                    limit.and_then(move |l| l.to_instant(teban,think_start_time))
                ),
                self.base_depth,
                self.max_nodes.clone(),
                self.max_threads,
                self.nodes_per_leaf_node,
                self.gamma,
                HashSet::new(),
                &transposition_table
            );

            env
        };

        Ok(self.think_common(env,
                                    periodically_info,
                                    on_error_handler)?)
    }

    fn think_ponder<L,S,P>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
                           info_sender:S,periodically_info:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
                           -> Result<BestMove,ApplicationError> where L: Logger + Send + 'static, S: InfoSender,
                                                                      P: PeriodicallyInfo + Send + 'static {
        let env = {
            let limit = Some(limit.clone());

            let transposition_table = Arc::clone(&self.transposition_table);
            let hasher = Arc::clone(&self.hasher);

            let (teban, _, _) = self.kyokumen.as_ref().map(|k| (k.teban, &k.state, &k.mc)).ok_or(
                UsiProtocolError::InvalidState(
                    String::from("Position information is not initialized."))
            )?;

            let env = Environment::new(
                Arc::clone(&event_queue),
                info_sender.clone(),
                Arc::clone(&on_error_handler),
                hasher,
                Arc::clone(&self.search_id),
                teban,
                limit,
                self.turn_limit,
                self.timelimit_margin,
                (None,None),
                self.base_depth,
                self.max_nodes.clone(),
                self.max_threads,
                self.nodes_per_leaf_node,
                self.gamma,
                HashSet::new(),
                &transposition_table
            );

            env
        };

        Ok(self.think_common(env,
                                    periodically_info,
                                    on_error_handler)?)
    }

    fn think_mate<L,S,P>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<UserEventQueue>>,
                         _:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
                         -> Result<CheckMate,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        Ok(CheckMate::NotiImplemented)
    }

    fn on_stop(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn gameover<L>(&mut self,_:&GameEndState,
                   _:Arc<Mutex<UserEventQueue>>, _:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),ApplicationError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        Ok(())
    }

    fn on_ponderhit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn on_quit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn quit(&mut self) -> Result<(),ApplicationError> {
        Ok(())
    }
}
