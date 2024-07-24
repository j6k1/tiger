use std::cell::RefCell;
use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::{BufReader, Read, BufWriter};
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use usiagent::output::USIStdErrorWriter;
use usiagent::OnErrorHandler;
use usiagent::event::*;
use usiagent::logger::*;
use usiagent::input::*;

use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::device::DeviceGpu;
use nncombinator::layer::{BatchDataType, BatchForwardBase, BatchTrain, ForwardAll};
use nncombinator::lossfunction::CrossEntropy;
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence};
use shogi_dataloader::dataloader::{DataLoader, DataLoaderBuilder, UnifiedDataLoader};

use crate::error::ApplicationError;
use crate::features::HalfKP;
use crate::nn::{Trainer, FEATURES_NUM};

#[derive(Debug,Deserialize,Serialize)]
pub struct CheckPoint {
    filename:String,
    item:usize
}
pub struct CheckPointReader {
    reader:BufReader<File>
}
impl CheckPointReader {
    pub fn new<P: AsRef<Path>>(file:P) -> Result<CheckPointReader,ApplicationError> {
        if file.as_ref().exists() {
            Ok(CheckPointReader {
                reader: BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)
            })
        } else {
            Err(ApplicationError::StartupError(String::from(
                "指定されたチェックポイントファイルは存在しません。"
            )))
        }
    }
    pub fn read(&mut self) -> Result<CheckPoint,ApplicationError> {
        let mut buf = String::new();
        self.reader.read_to_string(&mut buf)?;
        match toml::from_str(buf.as_str()) {
            Ok(r) => Ok(r),
            Err(ref e) => {
                let _ = USIStdErrorWriter::write(&e.to_string());
                Err(ApplicationError::StartupError(String::from(
                    "チェックポイントファイルのロード時にエラーが発生しました。"
                )))
            }
        }
    }
}
pub struct CheckPointWriter<P: AsRef<Path>> {
    writer:BufWriter<File>,
    tmp:P,
    path:P
}
impl<'a,P: AsRef<Path>> CheckPointWriter<P> {
    pub fn new(tmp:P,file:P) -> Result<CheckPointWriter<P>,ApplicationError> {
        Ok(CheckPointWriter {
            writer: BufWriter::new(OpenOptions::new().write(true).create(true).open(&tmp)?),
            tmp:tmp,
            path:file
        })
    }
    pub fn save(&mut self,checkpoint:&CheckPoint) -> Result<(),ApplicationError> {
        let toml_str = toml::to_string(checkpoint)?;

        match write!(self.writer,"{}",toml_str) {
            Ok(()) => {
                self.writer.flush()?;
                fs::rename(&self.tmp,&self.path)?;
                Ok(())
            },
            Err(_) => {
                Err(ApplicationError::StartupError(String::from(
                    "チェックポイントファイルの保存時にエラーが発生しました。"
                )))
            }
        }
    }
}
pub struct Learnener<M>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
             BatchForwardBase<BatchInput=<HalfKP<FEATURES_NUM> as BatchDataType>::Type,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
             BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear> {
             nn:PhantomData<M>
}
impl<M> Learnener<M>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
             BatchForwardBase<BatchInput=<HalfKP<FEATURES_NUM> as BatchDataType>::Type,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
             BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear> + 'static,
             [(); FEATURES_NUM * 2]:,
             [(); FEATURES_NUM * 256]: {
    pub fn new() -> Learnener<M> {
        Learnener {
            nn: PhantomData::<M>
        }
    }

    fn start_read_stdinput_thread(&self, notify_run_test: Arc<AtomicBool>,
                                  notify_quit: Arc<AtomicBool>,
                                  on_error_handler: Arc<Mutex<OnErrorHandler<FileLogger>>>) {
        thread::spawn(move || {
            let mut input_reader = USIStdInputReader::new();

            loop {
                match input_reader.read() {
                    Ok(Some(line)) => {
                        match line.trim_end() {
                            "quit" => {
                                notify_run_test.store(false, Ordering::Release);
                                notify_quit.store(true,Ordering::Release);
                            },
                            "test" => {
                                notify_quit.store(true,Ordering::Release);
                            },
                            _ => (),
                        }
                    },
                    Ok(None) => {},
                    Err(ref e) => {
                        let _ = on_error_handler.lock().map(|h| h.call(e));

                        notify_quit.store(true,Ordering::Release);
                    }
                }
            }
        });
    }

    pub fn learning_from_yaneuraou_bin(&mut self, kifudir: String,
                                       testdir: String,
                                       evalutor: Trainer<M>,
                                       on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                       learn_sfen_read_size: usize,
                                       learn_batch_size: usize,
                                       save_batch_count: usize,
                                       maxepoch: usize) -> Result<(), ApplicationError> {
        self.learning_batch(kifudir,
                            testdir,
                            "bin",
                            40,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            save_batch_count,
                            maxepoch,
                            Trainer::<M>::make_packed_sfens_parser,
                            |evalutor, packed| {
                                evalutor.test_by_packed_sfens(packed)
                            })
    }

    pub fn learning_from_hcpe(&mut self, kifudir: String,
                              testdir: String,
                              evalutor: Trainer<M>,
                              on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                              learn_sfen_read_size: usize,
                              learn_batch_size: usize,
                              save_batch_count: usize,
                              maxepoch: usize
    ) -> Result<(), ApplicationError> {
        self.learning_batch(kifudir,
                            testdir,
                            "hcpe",
                            38,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            save_batch_count,
                            maxepoch,
                            Trainer::<M>::make_hcpe_parser,
                            |evalutor, packed| {
                                evalutor.test_by_packed_hcpe(packed)
                            })
    }

    pub fn learning_batch<'a, F, P>(&mut self, kifudir: String,
                                    testdir: String,
                                    ext: &str,
                                    item_size: usize,
                                    evalutor: Trainer<M>,
                                    on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                    learn_sfen_read_size: usize,
                                    learn_batch_size: usize,
                                    save_batch_count: usize,
                                    maxepoch: usize,
                                    sfen_parser_builder: fn() -> P,
                                    mut test_process: F
    ) -> Result<(), ApplicationError>
        where F: FnMut(&mut Trainer<M>, Vec<u8>) -> Result<(GameEndState, f32, Option<bool>), ApplicationError> + Send + 'static,
              P: FnMut(Vec<Vec<u8>>) -> Result<Option<(Vec<Arr<f32, 1>>, Vec<HalfKP<FEATURES_NUM>>)>, ApplicationError> + Send + 'static {
        let notify_quit_arc = Arc::new(AtomicBool::new(false));

        let mut evalutor = evalutor;

        print!("learning start... kifudir = {}\n", kifudir);

        let notify_run_test_arc = Arc::new(AtomicBool::new(true));
        let notify_run_test = notify_run_test_arc.clone();

        let notify_quit = notify_quit_arc.clone();

        let on_error_handler = on_error_handler_arc.clone();

        self.start_read_stdinput_thread(notify_run_test, notify_quit, on_error_handler);

        let notify_quit = notify_quit_arc.clone();

        let mut processed_count = 0;

        let mut pending_count = 0;

        let checkpoint_path = Path::new(&kifudir).join("checkpoint.toml");

        let checkpoint = if checkpoint_path.exists() {
            Some(CheckPointReader::new(&checkpoint_path)?.read()?)
        } else {
            None
        };

        let extend = RefCell::new(0);
        let mut resume = true;

        let lossf = CrossEntropy::<f32>::new();

        let mut current_filename = String::from("");
        let mut current_items = 0;

        'epochs: for _ in (0..).take_while(|&c| c < maxepoch + *extend.borrow()) {
            let mut dataloader_builder = DataLoaderBuilder::new(Path::new(&kifudir)
                .join("training"))
                .resume(resume)
                .shuffle(true)
                .ext(ext.to_string())
                .batch_size(learn_batch_size)
                .read_sfen_size(learn_sfen_read_size)
                .sfen_size(item_size);

            if let Some(ref checkpoint) = checkpoint {
                if resume {
                    dataloader_builder = dataloader_builder
                        .start_filename(Some(checkpoint.filename.clone()))
                        .processed_items(checkpoint.item)
                }
            }

            let mut dataloader = dataloader_builder.build(sfen_parser_builder())?;

            while let Some((filename,items,batch)) = dataloader.load()? {
                if notify_quit.load(Ordering::Acquire) {
                    break 'epochs;
                }

                current_items = items;

                if filename != current_filename {
                    print!("current_file = {}: items = {}\n", filename,items);

                    current_filename = filename;
                }
                let loss = evalutor.nn.batch_train(batch.0.into(), batch.1.into(), &lossf)?;

                println!("error_total: {}", loss);

                pending_count += 1;

                processed_count += learn_batch_size;

                self.save(&mut evalutor,
                          &checkpoint_path,
                          current_filename.as_str(),
                          current_items,
                          pending_count >= save_batch_count,
                          &mut pending_count)?;
            }

            self.save(&mut evalutor,
                      &checkpoint_path,
                      current_filename.as_str(),
                      current_items,
                      pending_count >= save_batch_count,
                      &mut pending_count)?;

            resume = false;
        }

        if notify_run_test_arc.load(Ordering::Acquire) {
            let dataloader_builder = DataLoaderBuilder::new(Path::new(&testdir)
                .join("tests"))
                .shuffle(true)
                .ext(ext.to_string())
                .batch_size(100)
                .read_sfen_size(learn_sfen_read_size)
                .sfen_size(item_size);

            let mut dataloader:UnifiedDataLoader<Vec<Vec<u8>>, ApplicationError> = dataloader_builder.build(| sfens | Ok(Some(sfens)))?;
            let mut successed = 0;
            let mut estimated_win = 0;
            let mut win = 0;
            let mut count = 0;
            let mut same_moves = 0;
            let mut compare_moves = 0;

            for packed in dataloader.load()?.ok_or(
                ApplicationError::InvalidStateError(String::from("Insufficient number of test data"))
            )?.2.into_iter() {
                let (s, score, same_move) = test_process(&mut evalutor, packed)?;

                match same_move {
                    Some(true) => {
                        compare_moves += 1;
                        same_moves += 1;
                    },
                    Some(false) => {
                        compare_moves += 1;
                    },
                    _ => ()
                }

                if score >= 0.5 {
                    estimated_win += 1;
                }

                let success = match s {
                    GameEndState::Draw => {
                        true
                    },
                    GameEndState::Win => {
                        win += 1;
                        score >= 0.5
                    },
                    _ => {
                        score < 0.5
                    }
                };

                match s {
                    GameEndState::Win => println!("結果　勝ち"),
                    GameEndState::Lose => println!("結果　負け"),
                    _ => println!("結果　引き分け")
                };

                if success {
                    successed += 1;
                    println!("勝率{} 正解!", score);
                } else {
                    println!("勝率{} 不正解...", score);
                }

                count += 1;
            }

            println!("勝ち {}% (勝ちと評価された局面の割合 {}%)", win as f32 / count as f32 * 100., estimated_win as f32 / count as f32 * 100.);
            println!("負け {}% (負けと評価された局面の割合 {}%)", (count - win) as f32 / count as f32 * 100.,
                     (count - estimated_win) as f32 / count as f32 * 100.);
            println!("正解率(勝敗) {}%", successed as f32 / count as f32 * 100.);
            println!("正解率(指し手の一致率) {}%", same_moves as f32 / compare_moves as f32 * 100.);
        }

        print!("{}局面を学習しました。\n", processed_count);

        Ok(())
    }

    fn save(&self, evalutor: &mut Trainer<M>,
            checkpoint_path: &PathBuf,
            current_filename: &str,
            current_item: usize,
            cond: bool,
            pending_count: &mut usize)
            -> Result<(), ApplicationError> {
        if cond {
            evalutor.save()?;

            let tmp_path = format!("{}.tmp", &checkpoint_path.as_path().to_string_lossy());
            let tmp_path = Path::new(&tmp_path);

            let mut checkpoint_writer = CheckPointWriter::new(tmp_path, &checkpoint_path.as_path())?;

            checkpoint_writer.save(&CheckPoint {
                filename: current_filename.to_string(),
                item: current_item
            })?;
            *pending_count = 0;
        }

        Ok(())
    }
}