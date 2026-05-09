use std::{thread};
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
use usiagent::logger::*;
use usiagent::input::*;

use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::layer::{BatchDataType, BatchForwardBase, BatchTrain, ForwardAll, PersistProgress, Step};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence};
use shogi_dataloader::dataloader::{DataLoader, DataLoaderBuilder};
use nncombinator::cuda::allocator::CudaAllocator;
use nncombinator::device::DeviceGpu;

use crate::error::ApplicationError;
use crate::features::HalfKP;
use crate::nn::{Trainer, FEATURES_NUM, LF};

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
const DEFAULT_EPOCH_SIZE:usize = 10000;

pub struct Learnener<M,A>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
             BatchForwardBase<BatchInput=<HalfKP<FEATURES_NUM> as BatchDataType>::Type,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
             BatchTrain<f32,DeviceGpu<f32,A>,LF> + Persistence<f32,BinFilePersistence,Linear> + Step,
          A: CudaAllocator {
          nn:PhantomData<M>,
          allocator:PhantomData<A>
}
impl<M,A> Learnener<M,A>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
             BatchForwardBase<BatchInput=<HalfKP<FEATURES_NUM> as BatchDataType>::Type,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
             BatchTrain<f32,DeviceGpu<f32,A>,LF> +
             Persistence<f32,BinFilePersistence,Linear> +
             PersistProgress<BinFilePersistence,Linear> +
             Step + 'static,
          A: CudaAllocator,
          [(); FEATURES_NUM * 2]:,
          [(); FEATURES_NUM * 256]: {
    pub fn new() -> Learnener<M,A> {
        Learnener {
            nn: PhantomData::<M>,
            allocator: PhantomData::<A>
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
                                       evalutor: &mut Trainer<M,A>,
                                       on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                       learn_sfen_read_size: usize,
                                       learn_batch_size: usize,
                                       lambda: f32,
                                       verbose: bool,
                                       save_batch_count: usize,
                                       maxepoch: usize,
                                       batches_per_epoch: Option<usize>) -> Result<(), ApplicationError> {
        self.learning_batch(kifudir,
                            "bin",
                            40,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            lambda,
                            verbose,
                            save_batch_count,
                            maxepoch,
                            batches_per_epoch,
                            Trainer::<M,A>::make_packed_sfens_parser)
    }

    pub fn learning_from_hcpe(&mut self, kifudir: String,
                              evalutor: &mut Trainer<M,A>,
                              on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                              learn_sfen_read_size: usize,
                              learn_batch_size: usize,
                              lambda: f32,
                              verbose: bool,
                              save_batch_count: usize,
                              maxepoch: usize,
                              batches_per_epoch: Option<usize>
    ) -> Result<(), ApplicationError> {
        self.learning_batch(kifudir,
                            "hcpe",
                            38,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            lambda,
                            verbose,
                            save_batch_count,
                            maxepoch,
                            batches_per_epoch,
                            Trainer::<M,A>::make_hcpe_parser)
    }

    pub fn learning_batch<P>(&mut self, kifudir: String,
                                    ext: &str,
                                    item_size: usize,
                                    evalutor: &mut Trainer<M,A>,
                                    on_error_handler_arc: Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                    learn_sfen_read_size: usize,
                                    learn_batch_size: usize,
                                    lambda: f32,
                                    verbose: bool,
                                    save_batch_count: usize,
                                    maxepoch: usize,
                                    batches_per_epoch: Option<usize>,
                                    sfen_parser_builder: fn(f32,bool) -> P
    ) -> Result<(), ApplicationError>
        where P: FnMut(Vec<Vec<u8>>) -> Result<Option<(Vec<Arr<f32, 1>>, Vec<HalfKP<FEATURES_NUM>>)>, ApplicationError> + Send + 'static {
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

        let mut resume = true;

        let lossf = LF::new();

        let mut current_filename = String::from("");
        let mut current_items = 0;

        let mut loss_logger = BufWriter::new(OpenOptions::new().append(true).create(true).open("logs/loss.log")?);

        let mut epoch_count = 0;

        let mut loss_total = 0.0f32;
        let mut processed_batch_count = 0usize;

        'epochs: while epoch_count < maxepoch && notify_quit.load(Ordering::Acquire) == false {
            let mut dataloader_builder = DataLoaderBuilder::new(Path::new(&kifudir)
                .join("training"))
                .resume(resume)
                .shuffle(true)
                .ext(ext.to_string())
                .batch_size(learn_batch_size)
                .read_sfen_size(learn_sfen_read_size)
                .sfen_size(item_size)
                .send_buffer_size(10);

            if let Some(ref checkpoint) = checkpoint {
                if resume {
                    dataloader_builder = dataloader_builder
                        .start_filename(Some(checkpoint.filename.clone()))
                        .processed_items(checkpoint.item)
                }
            }

            let mut dataloader = dataloader_builder.build(sfen_parser_builder(lambda,verbose))?;

            while let Some((filename,items,batch)) = dataloader.load()? {
                if notify_quit.load(Ordering::Acquire) {
                    break 'epochs;
                }

                current_items = items;

                if filename != current_filename {
                    print!("current_file = {}: items = {}\n", filename,items);

                    current_filename = filename;
                }
                let size = batch.0.len();

                let loss = evalutor.nn.batch_train(batch.0.into(), batch.1.into(), &lossf)?;

                loss_total += loss;
                processed_batch_count += 1;

                println!("loss: {}, error_total: {}", loss, loss_total / processed_batch_count as f32);

                loss_logger.write_all((loss_total / processed_batch_count as f32).to_string().as_bytes())?;
                loss_logger.write_all(b"\n")?;
                
                pending_count += 1;

                processed_count += size;

                evalutor.nn.frequently_step()?;

                if batches_per_epoch.unwrap_or(DEFAULT_EPOCH_SIZE) > 0 {
                    if processed_count >= batches_per_epoch.unwrap_or(DEFAULT_EPOCH_SIZE) * learn_batch_size * (epoch_count + 1) {
                        epoch_count += 1;
                        evalutor.nn.step()?;
                        println!("epoch: {}", epoch_count);

                        if epoch_count >= maxepoch {
                            break 'epochs;
                        }
                    }
                }

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

            if processed_count >= batches_per_epoch.unwrap_or(DEFAULT_EPOCH_SIZE) * learn_batch_size * (epoch_count + 1) {
                epoch_count += 1;
                evalutor.nn.step()?;
                println!("epoch: {}", epoch_count);
            }
        }

        print!("{}局面を学習しました。\n", processed_count);

        Ok(())
    }

    fn save(&self, evalutor: &mut Trainer<M,A>,
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