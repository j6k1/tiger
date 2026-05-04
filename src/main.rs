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
use std::sync::{Arc, Mutex};
use getopts::Options;
#[cfg(feature = "cuda")]
use nncombinator::cuda::allocator::{DeviceAlloc, MemoryPoolAllocator, MemoryPoolAllocatorInstantiation};
use usiagent::logger::FileLogger;
use usiagent::{OnErrorHandler, UsiAgent};
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
const LEAN_BATCH_SIZE:usize = 8192;
const EVAL_TEST_MAX_THREADS:usize = 16;
const EVAL_TEST_SAMPLES:usize = 10000;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    learn_sfen_read_size:Option<usize>,
    learn_batch_size:Option<usize>,
    batches_per_epoch:Option<usize>,
    lambda:Option<f32>,
    step_count:Option<usize>,
    gamma:Option<f32>,
    eta_min:Option<f32>,
    save_batch_count:Option<usize>,
    learning_rate:Option<f32>,
    eval_test_max_threads:Option<usize>,
    verbose:Option<bool>,
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

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(ref e) => {
            return Err(ApplicationError::StartupError(e.to_string()));
        }
    };

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
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_sfens(packed)
                                           })
                    } else {
                        evalutor.eval_test(testdir,"hcpe",38,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
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
                                           |evalutor, packed| {
                                               evalutor.test_by_packed_sfens(packed)
                                           })
                    } else {
                        evalutor.eval_test(testdir,"hcpe",38,
                                           config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                           config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
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
               |evalutor, packed| {
                   evalutor.test_by_packed_sfens(packed)
               })?;
        } else {
            evalutor.eval_test(testdir,"hcpe",38,
               config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
               config.eval_test_max_threads.unwrap_or(EVAL_TEST_MAX_THREADS),
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