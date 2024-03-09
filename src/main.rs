#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate rand;
extern crate rand_distr;
extern crate rand_xorshift;
extern crate statrs;
extern crate getopts;
extern crate toml;
extern crate rayon;

#[macro_use]
extern crate serde_derive;

extern crate usiagent;
extern crate nncombinator;
extern crate packedsfen;

extern crate core;

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};
use getopts::Options;
use usiagent::logger::FileLogger;
use usiagent::{OnErrorHandler, UsiAgent};
use usiagent::output::USIStdErrorWriter;
use crate::error::ApplicationError;
use crate::learning::Learnener;
use crate::nn::{EvalutorCreator, TrainerCreator};
use crate::player::Tiger;

pub mod kernel;
pub mod nn;
pub mod learning;
pub mod transposition_table;
pub mod player;
pub mod search;
pub mod error;

const LEAN_SFEN_READ_SIZE:usize = 1000 * 1000 * 10;
const LEAN_BATCH_SIZE:usize = 1000 * 100;

#[derive(Debug, Deserialize)]
pub struct Config {
    learn_sfen_read_size:Option<usize>,
    learn_batch_size:Option<usize>,
    save_batch_count:Option<usize>,
    learning_rate:Option<f32>
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
    opts.optopt("e", "maxepoch", "Number of epochs in batch learning.", "number of epoch");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(ref e) => {
            return Err(ApplicationError::StartupError(e.to_string()));
        }
    };

    if let Some(kifudir) = matches.opt_str("kifudir") {
        let testdir = matches.opt_str("testdir").unwrap_or(kifudir.clone());

        let logger = Arc::new(Mutex::new(FileLogger::new(String::from("logs/log.txt"))?));
        let on_error_handler = Arc::new(Mutex::new(OnErrorHandler::new(logger)));

        let config = ConfigLoader::new("settings.toml")?.load()?;

        let maxepoch = matches.opt_str("maxepoch").unwrap_or(String::from("1")).parse::<usize>()?;

        let r = if matches.opt_present("yaneuraou") {
            Learnener::new().learning_from_yaneuraou_bin(kifudir,
                                                         testdir,
                                                         TrainerCreator::create(String::from("data"),
                                                                                String::from("nn.bin"),
                                                                                config.learning_rate.unwrap_or(0.01))?,
                                                         on_error_handler.clone(),
                                                         config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                                         config.learn_batch_size.unwrap_or(LEAN_BATCH_SIZE),
                                                         config.save_batch_count.unwrap_or(1),
                                                         maxepoch)
        } else if matches.opt_present("hcpe") {
            Learnener::new().learning_from_hcpe(kifudir,
                                                testdir,
                                                TrainerCreator::create(String::from("data"),
                                                                       String::from("nn.bin"),
                                                                       config.learning_rate.unwrap_or(0.01))?,
                                                on_error_handler.clone(),
                                                config.learn_sfen_read_size.unwrap_or(LEAN_SFEN_READ_SIZE),
                                                config.learn_batch_size.unwrap_or(LEAN_BATCH_SIZE),
                                                config.save_batch_count.unwrap_or(1),
                                                maxepoch)
        } else {
            Err(ApplicationError::InvalidSettingError(String::from("learning mode is not specified.")))
        };

        if let Err(ref e) = r {
            let _ = on_error_handler.lock().map(|h| h.call(e));
        }

        r
    } else {
        let agent = UsiAgent::new(Tiger::new(| model_name | EvalutorCreator::create(String::from("data"),model_name.clone())));

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