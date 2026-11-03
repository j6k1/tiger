#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]

extern crate libc;
extern crate rand;
extern crate rand_distr;
extern crate rand_xorshift;
extern crate statrs;
extern crate getopts;
extern crate toml;
extern crate rayon;
extern crate crossbeam_channel;
#[cfg(feature = "cuda")]
extern crate cuda_runtime_sys;
#[cfg(feature = "cuda")]
extern crate rcublas_sys;
#[cfg(feature = "cuda")]
extern crate rcublas;
#[cfg(feature = "cuda")]
extern crate rcudnn;
#[cfg(feature = "cuda")]
extern crate rcudnn_sys;

#[macro_use]
extern crate serde_derive;

extern crate usiagent;
extern crate nncombinator;
extern crate packedsfen;
extern crate shogi_dataloader;
extern crate core;

use std::{env};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc};
#[cfg(feature = "cuda")]
use std::sync::{Mutex};
use getopts::Options;
#[cfg(feature = "cuda")]
use nncombinator::cuda::allocator::{DeviceAlloc, MemoryPoolAllocator, MemoryPoolAllocatorInstantiation};
#[cfg(feature = "cuda")]
use usiagent::logger::FileLogger;
#[cfg(feature = "cuda")]
use usiagent::{OnErrorHandler};
use usiagent::{UsiAgent};
use usiagent::output::USIStdErrorWriter;
use crate::error::ApplicationError;
use crate::nn::{EvalutorCreator};
use crate::player::Tiger;
#[cfg(feature = "cuda")]
use crate::nn::{TrainerCreator};
#[cfg(feature = "cuda")]
use crate::learning::Learnener;

pub mod device;
pub mod features;
pub mod nn;
pub mod layer;
#[cfg(feature = "cuda")]
pub mod learning;
pub mod transposition_table;
pub mod player;
pub mod search;
pub mod error;
#[cfg(feature = "cuda")]
pub mod kernel;
pub mod evalutor;
pub mod math;
const LEAN_SFEN_READ_SIZE:usize = 1024 * 10000 * 10;
#[cfg(feature = "cuda")]
const LEAN_BATCH_SIZE:usize = 8192;
const EVAL_TEST_MAX_THREADS:usize = 16;
const EVAL_TEST_SAMPLES:usize = 10000;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub learn_sfen_read_size:Option<usize>,
    pub learn_batch_size:Option<usize>,
    pub batches_per_epoch:Option<usize>,
    pub lambda:Option<f32>,
    pub step_count:Option<usize>,
    pub gamma:Option<f32>,
    pub eta_min:Option<f32>,
    pub warmup_steps:Option<usize>,
    pub warmup_epochs:Option<usize>,
    pub start_factor:Option<f32>,
    pub save_batch_count:Option<usize>,
    pub learning_rate:Option<f32>,
    pub eval_test_max_threads:Option<usize>,
    pub verbose:Option<bool>,
}
pub struct ConfigLoader {
    reader:BufReader<File>,
}
impl ConfigLoader {
    pub fn new<P: AsRef<Path>>(file:P) -> Result<ConfigLoader, ApplicationError> {
        match Path::new(file.as_ref()).exists() {
            true => {
                Ok(ConfigLoader {
                    reader:BufReader::new(OpenOptions::new().read(true).create(false).open(file.as_ref())?),
                })
            },
            false => {
                Err(ApplicationError::StartupError(String::from(
                    "Configuration file does not exists."
                )))
            }
        }
    }
    pub fn load(&mut self) -> Result<Config,ApplicationError> {
        let mut buf = String::new();
        self.reader.read_to_string(&mut buf)?;
        match toml::from_str(buf.as_str()) {
            Ok(r) => Ok(r),
            Err(ref e) => {
                let _ = USIStdErrorWriter::write(&e.to_string());
                Err(ApplicationError::StartupError(String::from(
                    "An error occurred when loading the configuration file."
                )))
            }
        }
    }
}
fn main() {
    match run() {
        Ok(()) => (),
        Err(ref e) =>  {
            let _ = USIStdErrorWriter::write(&e.to_string());
        }
    };
}
fn run() -> Result<(),ApplicationError> {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.optopt("", "kifudir", "Directory of game data to be used of learning.", "path string.");
    opts.optopt("", "testdir", "Directory of test data to validate learning results.", "path string.");
    opts.optflag("", "yaneuraou", "YaneuraOu format teacher phase.");
    opts.optflag("", "hcpe", "hcpe format teacher phase.");
    opts.optflag("", "yaneuraou,hcpe", "YaneuraOu format teacher phase. and hcpe format test phase.");
    opts.optflag("", "hcpe,yaneuraou", "hcpe format teacher phase. and YaneuraOu format test phase.");
    opts.optopt("e", "maxepoch", "Number of epochs in batch learning.", "number of epoch");
    opts.optopt("", "eval", "Test only the evaluation of learned models. The argument is the path to the directory containing the teacher phase for testing", "path string.");
    opts.optopt("", "maelambda", "The coefficients of the win-loss terms used during training for MAE calculations", "lambda");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(ref e) => {
            return Err(ApplicationError::StartupError(e.to_string()));
        }
    };

    #[allow(unused_variables)]
    if let Some(kifudir) = matches.opt_str("kifudir") {
        #[cfg(feature = "cuda")]
        {
            let testdir = matches.opt_str("testdir").unwrap_or(kifudir.clone());

            let logger = Arc::new(Mutex::new(FileLogger::new(String::from("logs/log.txt"))?));
            let on_error_handler = Arc::new(Mutex::new(OnErrorHandler::new(logger)));

            let config = ConfigLoader::new("settings.toml")?.load()?;

            let maxepoch = matches.opt_str("maxepoch").unwrap_or(String::from("1")).parse::<usize>()?;

            let mut evalutor = TrainerCreator::create(String::from("data"),
                                                      String::from("nn.bin"),
                                                      &config,
                                                      &matches,
                                                      MemoryPoolAllocator::with_size(4 * 1024 * 1024 * 1024,DeviceAlloc::new())?)?;

            let r = if matches.opt_present("yaneuraou") || matches.opt_present("yaneuraou,hcpe") {
                if let Err(e) = Learnener::new().learning_from_yaneuraou_bin(kifudir,
                                                                             &mut evalutor,
                                                                             on_error_handler.clone(),
                                                                             config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                                                             config.learn_batch_size.unwrap_or(LEAN_BATCH_SIZE),
                                                                             config.lambda.unwrap_or(0.1),
                                                                             config.verbose.unwrap_or(false),
                                                                             config.save_batch_count.unwrap_or(20),
                                                                             maxepoch,
                                                                             config.batches_per_epoch) {
                    Err(e)
                } else {
                    let _ = evalutor;

                    let evalutor = Arc::new(EvalutorCreator::create(String::from("data"),"nn.bin",&config)?);

                    if matches.opt_present("yaneuraou") {
                        evalutor.eval_test(testdir,"bin",40,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                                           matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_sfens(packed)
                                           })
                    } else {
                        evalutor.eval_test(testdir,"hcpe",38,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                                           matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_hcpe(packed)
                                           })
                    }
                }
            } else if matches.opt_present("hcpe") || matches.opt_present("hcpe,yaneuraou") {
                if let Err(e) = Learnener::new().learning_from_hcpe(kifudir,
                                                                    &mut evalutor,
                                                                    on_error_handler.clone(),
                                                                    config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                                                    config.learn_batch_size.unwrap_or(LEAN_BATCH_SIZE),
                                                                    config.lambda.unwrap_or(0.1),
                                                                    config.verbose.unwrap_or(false),
                                                                    config.save_batch_count.unwrap_or(20),
                                                                    maxepoch,
                                                                    config.batches_per_epoch) {
                    Err(e)
                } else {
                    let _ = evalutor;

                    let evalutor = Arc::new(EvalutorCreator::create(String::from("data"),"nn.bin",&config)?);

                    if matches.opt_present("hcpe,yaneuraou") {
                        evalutor.eval_test(testdir,"bin",40,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                                           matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_sfens(packed)
                                           })
                    } else {
                        evalutor.eval_test(testdir,"hcpe",38,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                                           matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_hcpe(packed)
                                           })
                    }
                }
            } else {
                Err(ApplicationError::InvalidSettingError(String::from("learning mode is not specified.")))
            };

            if let Err(ref e) = r {
                let _ = on_error_handler.lock().map(|h| h.call(e));
            }

            r
        }

        #[cfg(not(feature = "cuda"))]
        {
            unimplemented!()
        }
    } else if let Some(testdir) = matches.opt_str("eval") {
        let config = ConfigLoader::new("settings.toml")?.load()?;

        let evalutor = Arc::new(EvalutorCreator::create(String::from("data"),"nn.bin",&config)?);

        if matches.opt_present("yaneuraou") {
            evalutor.eval_test(testdir,"bin",40,
               config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
               config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                               matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
               |evalutor, packed| {
                   evalutor.test_by_packed_sfens(packed)
               })?;
        } else {
            evalutor.eval_test(testdir,"hcpe",38,
               config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
               config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
                               matches.opt_str("maelambda").unwrap_or(String::from("0.5")).parse()?,
       |evalutor, packed| {
                   evalutor.test_by_packed_hcpe(packed)
               })?;
        }

        Ok(())
    } else {
        let config = ConfigLoader::new("settings.toml")?.load()?;

        let agent = UsiAgent::new(Tiger::new(move | model_name | EvalutorCreator::create(String::from("data"),model_name.clone(),&config)));

        let r = agent.start_default(|on_error_handler,e| {
            match on_error_handler {
                Some(ref h) => {
                    let _ = h.lock().map(|h| h.call(e));
                },
                None => (),
            }
        });
        r.map_err(|_| ApplicationError::AgentRunningError(String::from(
            "An error occurred while running USIAgent. See log for details..."
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, AddAssign};
    use std::hint::black_box;
    use std::sync::mpsc;
    use std::time::Instant;
    use nncombinator::arr::Arr;
    use nncombinator::layer::{ContinueForward, ForwardAll, OutputTensorScalar, PartialForward, PreTrain, PreTrainBase};
    use rand::RngCore;
    use rayon::ThreadPoolBuilder;
    use usiagent::math::Prng;
    use usiagent::movepick::{MovePicker, RandomPicker};
    use usiagent::position::Position;
    use usiagent::protocol::{ParsePosition, PositionParser};
    use usiagent::rule::{EvasionsAll, LegalMove, NonEvasionsAll, Rule, State};
    use usiagent::shogi::Teban;
    use simdoperator::backend::common::SignFloat;
    use crate::features::{HalfKP, HalfKPDiff};
    use crate::nn::{Evalutor, FEATURES_NUM};

    use super::*;
    #[derive(Default,Debug,PartialEq,Eq)]
    struct PerftResult {
        pub nodes:usize,
        pub captures:usize,
        pub promotions:usize,
        pub checks:usize,
        pub mates:usize
    }

    impl Add for PerftResult {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            PerftResult {
                nodes: self.nodes + rhs.nodes,
                captures: self.captures + rhs.captures,
                promotions: self.promotions + rhs.promotions,
                checks: self.checks + rhs.checks,
                mates: self.mates + rhs.mates
            }
        }
    }
    impl AddAssign for PerftResult {
        fn add_assign(&mut self, rhs: Self) {
            *self = PerftResult {
                nodes: self.nodes + rhs.nodes,
                captures: self.captures + rhs.captures,
                promotions: self.promotions + rhs.promotions,
                checks: self.checks + rhs.checks,
                mates: self.mates + rhs.mates
            }
        }
    }
    impl From<String> for PerftResult {
        fn from(value: String) -> Self {
            let fields: Vec<_> = value.split(',').map(|f| f.parse().unwrap()).collect();

            PerftResult {
                nodes: fields[0],
                captures: fields[1],
                promotions: fields[2],
                checks: fields[3],
                mates: fields[4]
            }
        }
    }

    struct GameState<'a> {
        pub self_partial_output: &'a Arr<f32,{256*2}>,
        pub opponent_partial_output: &'a Arr<f32,{256*2}>,
    }
    trait PerftSolverForPosition<M,const N: usize>
        where for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                         PreTrainBase + PreTrain + Send + Sync + OutputTensorScalar<Scalar=f32> +
                         PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                         ContinueForward + 'static,
                      <M as PreTrainBase>::OutStack: Send + Sync + 'static {
        fn perft<'a>(&self,teban:Teban,pos:&mut Position<N>,m:Option<LegalMove>,depth:usize,
                 evalutor: &Arc<Evalutor<M>>, gs:&GameState<'a>) -> PerftResult;
    }
    struct PerftSolverForPositionByEvasions;

    impl<M> PerftSolverForPosition<M,200> for PerftSolverForPositionByEvasions
        where for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32,1>> +
                         PreTrainBase + PreTrain + Send + Sync + OutputTensorScalar<Scalar=f32> +
                         PartialForward<DiffInput=HalfKPDiff<SignFloat<f32>>,PartialInput=Arr<f32,{256*2}>,PartialOutput=Arr<f32,{256*2}>> +
                         ContinueForward + 'static,
                      <M as PreTrainBase>::OutStack: Send + Sync + 'static {
        fn perft<'a>(&self,teban:Teban,pos:&mut Position<200>, m: Option<LegalMove>, depth: usize,
                 evalutor: &Arc<Evalutor<M>>, gs: &GameState<'a>) -> PerftResult {
            let mut rng = rand::thread_rng();
            let mut prng = Prng::new(rng.next_u64());

            let mut result = PerftResult::default();

            if depth == 0 {
                result.nodes += 1;

                black_box(evalutor.evalute(gs.self_partial_output)).unwrap();

                match m {
                    Some(LegalMove::To(m)) => {
                        if m.obtained().is_some() {
                            result.captures += 1;
                        }
                        if m.is_nari() {
                            result.promotions += 1;
                        }
                    },
                    _ => ()
                };

                if Rule::in_check(teban,pos.get_state()) {
                    result.checks += 1;

                    let mut buffer = RandomPicker::new(Prng::new(prng.rnd64()));

                    Rule::generate_moves::<EvasionsAll>(teban, pos.get_state(), pos.get_mc(), &mut buffer).unwrap();

                    if buffer.len() == 0 {
                        result.mates += 1;
                    }
                }
            } else {
                let mut buffer = RandomPicker::new(Prng::new(prng.rnd64()));

                if Rule::in_check(teban,pos.get_state()) {
                    Rule::generate_moves::<EvasionsAll>(teban, pos.get_state(), pos.get_mc(), &mut buffer).unwrap();
                } else {
                    Rule::generate_moves::<NonEvasionsAll>(teban, pos.get_state(), pos.get_mc(), &mut buffer).unwrap();
                }

                for m in buffer {
                    let use_diff = match m {
                        LegalMove::To(m) => m.src() != Rule::ou_square(teban,pos.get_state()) as u32,
                        _ => false
                    };

                    let (self_partial_output,opponent_partial_output) = if use_diff {
                        let self_partial_output = evalutor.prepare_evalute_by_diff(teban, teban,pos.get_state(),pos.get_mc(),m,gs.self_partial_output).unwrap();
                        let opponent_partial_output = evalutor.prepare_evalute_by_diff(teban, teban.opposite(),pos.get_state(),pos.get_mc(),m,gs.opponent_partial_output).unwrap();

                        pos.apply_move(teban,m).unwrap();

                        (self_partial_output,opponent_partial_output)
                    } else {
                        pos.apply_move(teban,m).unwrap();

                        let self_partial_output = evalutor.prepare_evalute(teban,pos.get_state(),pos.get_mc()).unwrap();
                        let opponent_partial_output = evalutor.prepare_evalute(teban.opposite(),pos.get_state(),pos.get_mc()).unwrap();

                        (self_partial_output,opponent_partial_output)
                    };

                    if !Rule::in_check(teban,pos.get_state()) {
                        let gs = GameState {
                            self_partial_output: &opponent_partial_output,
                            opponent_partial_output: &self_partial_output,
                        };

                        result += self.perft(teban.opposite(),pos,Some(m),depth - 1,evalutor,&gs);
                    }

                    pos.undo_move().unwrap();
                }
            }

            result
        }
    }

    const INITIAL_SFEN:&'static str = "sfen l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1";

    const MAX_THREADS: usize = 8;

    #[test]
    fn test_perft_and_evaluation() {
        let position_parser = PositionParser::new();

        let (teban, banmen, mc, _, _) = match position_parser.parse(&INITIAL_SFEN.split(" ").collect::<Vec<&str>>()) {
            Ok(position) => {
                position.extract()
            },
            _ => {
                panic!("invalid state.");
            }
        };

        let state = State::new(banmen);

        let evalutor = Arc::new(EvalutorCreator::create(String::from("data"),"nn.bin",&ConfigLoader::new("settings.toml").unwrap().load().unwrap()).unwrap());

        println!("test_perft_and_evaluation");

        let tb = ThreadPoolBuilder::new()
            .num_threads(MAX_THREADS).stack_size(1024 * 1024 * 200).build().unwrap();

        let (s,r) = mpsc::channel();

        let start = Instant::now();

        for _ in 0..MAX_THREADS {
            let evalutor = Arc::clone(&evalutor);
            let state = state.clone();
            let mc = mc.clone();
            let s = s.clone();

            tb.spawn(move || {
                let mut pos = Position::<200>::new(state, mc);

                let solver = PerftSolverForPositionByEvasions;

                let self_partial_output = evalutor.prepare_evalute(teban,pos.get_state(),pos.get_mc()).unwrap();
                let opponent_partial_output = evalutor.prepare_evalute(teban.opposite(),pos.get_state(),pos.get_mc()).unwrap();

                let gs = GameState {
                    self_partial_output: &self_partial_output,
                    opponent_partial_output: &opponent_partial_output
                };

                solver.perft(teban,&mut pos,None,3,&evalutor,&gs);

                let _ = s.send(());
            });
        }

        for _ in 0..MAX_THREADS {
            let _ = r.recv().unwrap();
        }

        let elapsed = start.elapsed();

        println!("processing time is {}.{:09} secs.", elapsed.as_secs(), elapsed.subsec_nanos());
    }
}