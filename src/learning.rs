use std::cell::RefCell;
use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::{BufReader, Read, BufWriter};
use std::fs::{DirEntry, File, OpenOptions};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use usiagent::output::USIStdErrorWriter;
use usiagent::OnErrorHandler;
use usiagent::event::*;
use usiagent::logger::*;
use usiagent::input::*;
use usiagent::error::*;

use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::device::DeviceGpu;
use nncombinator::layer::{BatchForwardBase, BatchTrain, ForwardAll};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence};

use crate::error::ApplicationError;
use crate::nn::{Trainer};

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
    where M: ForwardAll<Input=Arr<f32,2515>,Output=Arr<f32,1>> +
    BatchForwardBase<BatchInput=SerializedVec<f32,Arr<f32,2515>>,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
    BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear> {
    nn:PhantomData<M>}
impl<M> Learnener<M>
    where M: ForwardAll<Input=Arr<f32,2515>,Output=Arr<f32,1>> +
    BatchForwardBase<BatchInput=SerializedVec<f32,Arr<f32,2515>>,BatchOutput=SerializedVec<f32,Arr<f32,1>>> +
    BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear>{
    pub fn new() -> Learnener<M> {
        Learnener {
            nn:PhantomData::<M>
        }
    }

    fn create_event_dispatcher(&self, notify_quit:Arc<AtomicBool>,
                               on_error_handler:Arc<Mutex<OnErrorHandler<FileLogger>>>) -> USIEventDispatcher<SystemEventKind,
        SystemEvent,(),FileLogger,ApplicationError> {

        let mut system_event_dispatcher:USIEventDispatcher<SystemEventKind,
            SystemEvent,(),FileLogger,ApplicationError> = USIEventDispatcher::new(&on_error_handler);

        system_event_dispatcher.add_handler(SystemEventKind::Quit, move |_,e| {
            match e {
                &SystemEvent::Quit => {
                    notify_quit.store(true,Ordering::Release);
                    Ok(())
                },
                e => Err(EventHandlerError::InvalidState(e.event_kind())),
            }
        });

        system_event_dispatcher
    }

    fn start_read_stdinput_thread(&self,notify_run_test:Arc<AtomicBool>,
                                  system_event_queue:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>>,
                                  on_error_handler:Arc<Mutex<OnErrorHandler<FileLogger>>>) {
        thread::spawn(move || {
            let mut input_reader = USIStdInputReader::new();

            loop {
                match input_reader.read() {
                    Ok(Some(line)) => {
                        match line.trim_end() {
                            "quit" => {
                                match system_event_queue.lock() {
                                    Ok(mut system_event_queue) => {
                                        notify_run_test.store(false,Ordering::Release);
                                        system_event_queue.push(SystemEvent::Quit);
                                        return;
                                    },
                                    Err(ref e) => {
                                        let _ = on_error_handler.lock().map(|h| h.call(e));
                                        return;
                                    }
                                }
                            },
                            "test" => {
                                match system_event_queue.lock() {
                                    Ok(mut system_event_queue) => {
                                        system_event_queue.push(SystemEvent::Quit);
                                        return;
                                    },
                                    Err(ref e) => {
                                        let _ = on_error_handler.lock().map(|h| h.call(e));
                                        return;
                                    }
                                }
                            }
                            _ => (),
                        }
                    },
                    Ok(None) => {
                    },
                    Err(ref e) => {
                        let _ = on_error_handler.lock().map(|h| h.call(e));
                        match system_event_queue.lock() {
                            Ok(mut system_event_queue) => {
                                system_event_queue.push(SystemEvent::Quit);
                            },
                            Err(ref e) => {
                                let _ = on_error_handler.lock().map(|h| h.call(e));
                            }
                        }
                        return;
                    }
                }
            }
        });
    }


    pub fn learning_from_yaneuraou_bin(&mut self, kifudir:String,
                                       evalutor: Trainer<M>,
                                       on_error_handler_arc:Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                       learn_sfen_read_size:usize,
                                       learn_batch_size:usize,
                                       save_batch_count:usize,
                                       maxepoch:usize) -> Result<(),ApplicationError> {
        self.learning_batch(kifudir,
                            "bin",
                            40,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            save_batch_count,
                            maxepoch,
                            Self::learning_from_yaneuraou_bin_batch,
                            |evalutor,packed| {
                                evalutor.test_by_packed_sfens(packed)
                            })

    }

    pub fn learning_from_hcpe(&mut self, kifudir:String,
                              evalutor: Trainer<M>,
                              on_error_handler_arc:Arc<Mutex<OnErrorHandler<FileLogger>>>,
                              learn_sfen_read_size:usize,
                              learn_batch_size:usize,
                              save_batch_count:usize,
                              maxepoch:usize
    ) -> Result<(),ApplicationError> {
        self.learning_batch(kifudir,
                            "hcpe",
                            38,
                            evalutor,
                            on_error_handler_arc,
                            learn_sfen_read_size,
                            learn_batch_size,
                            save_batch_count,
                            maxepoch,
                            Self::learning_from_hcpe_batch,
                            |evalutor,packed| {
                                evalutor.test_by_packed_hcpe(packed)
                            })

    }

    pub fn learning_batch<'a,F>(&mut self,kifudir:String,
                                ext:&str,
                                item_size:usize,
                                evalutor: Trainer<M>,
                                on_error_handler_arc:Arc<Mutex<OnErrorHandler<FileLogger>>>,
                                learn_sfen_read_size:usize,
                                learn_batch_size:usize,
                                save_batch_count:usize,
                                maxepoch:usize,
                                learning_process:fn(
                                    &mut Trainer<M>,
                                    Vec<Vec<u8>>,
                                    &Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
                                ) -> Result<(),ApplicationError>,
                                mut test_process:F
    ) -> Result<(),ApplicationError>
        where F: FnMut(&mut Trainer<M>,Vec<u8>) -> Result<(GameEndState,f32),ApplicationError> {

        let system_event_queue_arc:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));
        let user_event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));

        let notify_quit_arc = Arc::new(AtomicBool::new(false));

        let on_error_handler = on_error_handler_arc.clone();

        let notify_quit = notify_quit_arc.clone();

        let mut system_event_dispatcher = self.create_event_dispatcher(notify_quit,on_error_handler);

        let mut evalutor = evalutor;

        print!("learning start... kifudir = {}\n", kifudir);

        let on_error_handler = on_error_handler_arc.clone();
        let system_event_queue = system_event_queue_arc.clone();

        let notify_run_test_arc = Arc::new(AtomicBool::new(true));
        let notify_run_test = notify_run_test_arc.clone();

        self.start_read_stdinput_thread(notify_run_test,system_event_queue,on_error_handler);

        let system_event_queue = system_event_queue_arc.clone();
        let notify_quit = notify_quit_arc.clone();

        let mut processed_count = 0;

        let mut record = Vec::with_capacity(item_size);

        let mut pending_count = 0;

        let checkpoint_path = Path::new(&kifudir).join("checkpoint.toml");

        let checkpoint = if checkpoint_path.exists() {
            Some(CheckPointReader::new(&checkpoint_path)?.read()?)
        } else {
            None
        };

        let mut current_filename = String::from("");

        let mut skip_files = checkpoint.is_some();
        let mut skip_items = checkpoint.is_some();

        let mut rng = rand::thread_rng();
        let mut rng = XorShiftRng::from_seed(rng.gen());

        let mut current_item = 0;

        let mut item_count = 0;

        let extend = RefCell::new(0);

        'epochs: for _ in (0..).take_while(|&c| c < maxepoch + *extend.borrow()) {
            let mut teachers = Vec::with_capacity(learn_sfen_read_size);

            let mut paths = fs::read_dir(Path::new(&kifudir)
                .join("training"))?.into_iter()
                .collect::<Vec<Result<DirEntry,_>>>();

            paths.sort_by(Self::cmp);

            for path in paths {
                let path = path?.path();

                current_filename = path.as_path().file_name().map(|s| {
                    s.to_string_lossy().to_string()
                }).unwrap_or(String::from(""));

                if let Some(ref checkpoint) = checkpoint {
                    if current_filename == checkpoint.filename {
                        skip_files = false;
                    }

                    if skip_files {
                        continue;
                    } else if current_filename != checkpoint.filename {
                        skip_items = false;
                    }
                }

                if !path.as_path().extension().map(|e| e == ext).unwrap_or(false) {
                    continue;
                }

                print!("{}\n", path.display());

                current_item = 0;

                for b in BufReader::new(File::open(path)?).bytes() {
                    let b = b?;

                    record.push(b);

                    if record.len() == item_size {
                        item_count += 1;
                        current_item += 1;

                        if let Some(ref checkpoint) = checkpoint {
                            if skip_items && current_item < checkpoint.item {
                                record.clear();
                                continue;
                            } else {
                                if skip_items && current_item == checkpoint.item {
                                    println!("Processing starts from {}th item of file {}", current_item, &current_filename);
                                    skip_items = false;
                                    record.clear();
                                    continue;
                                }
                            }
                        }
                        teachers.push(record);
                        record = Vec::with_capacity(item_size);
                    } else {
                        continue;
                    }

                    if teachers.len() == learn_sfen_read_size {
                        if !self.learning_loop(&mut evalutor,
                                               &checkpoint_path,
                                               &current_filename,
                                               current_item, save_batch_count,
                                               &notify_quit,
                                               teachers,
                                               learn_batch_size,
                                               learning_process,
                                               &mut system_event_dispatcher,
                                               &system_event_queue,
                                               &user_event_queue,
                                               &mut pending_count,
                                               &mut processed_count
                        )? {
                            break 'epochs;
                        }
                        teachers = Vec::with_capacity(learn_sfen_read_size);
                    }
                }

                if processed_count == 0 && item_count > 0 {
                    *extend.borrow_mut() += 1;
                }

                skip_files = false;
                skip_items = false;
            }

            if record.len() > 0 {
                return Err(ApplicationError::LearningError(String::from(
                    "The data size of the teacher phase is invalid."
                )));
            }

            if !notify_quit.load(Ordering::Acquire) && teachers.len() > 0 {
                if !self.learning_loop(&mut evalutor,
                                       &checkpoint_path,
                                       &current_filename,
                                       current_item, save_batch_count,
                                       &notify_quit,
                                       teachers,
                                       learn_batch_size,
                                       learning_process,
                                       &mut system_event_dispatcher,
                                       &system_event_queue,
                                       &user_event_queue,
                                       &mut pending_count,
                                       &mut processed_count
                )? {
                    break 'epochs;
                }
            }

            self.save(&mut evalutor,
                      &checkpoint_path,
                      &current_filename,
                      current_item,
                      pending_count > 0,
                      &mut pending_count)?;
        }

        if notify_run_test_arc.load(Ordering::Acquire) {
            let mut testdata = Vec::new();

            let mut paths = fs::read_dir(Path::new(&kifudir)
                .join("tests"))?.into_iter()
                .collect::<Vec<Result<DirEntry,_>>>();

            paths.sort_by(Self::cmp);

            'test_files: for path in paths {
                let path = path?.path();

                if !path.as_path().extension().map(|e| e == ext).unwrap_or(false) {
                    continue;
                }

                print!("{}\n", path.display());

                for b in BufReader::new(File::open(path)?).bytes() {
                    let b = b?;

                    record.push(b);

                    if record.len() == item_size {
                        testdata.push(record);
                        record = Vec::with_capacity(item_size);
                    } else {
                        continue;
                    }

                    if testdata.len() >= 10000 {
                        break 'test_files;
                    }
                }
            }

            testdata.shuffle(&mut rng);

            let mut successed = 0;
            let mut estimated_win = 0;
            let mut win = 0;
            let mut count = 0;

            for packed in testdata.into_iter().take(100) {
                let (s,score) = test_process(&mut evalutor,packed)?;

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
                    println!("勝率{} 正解!",score);
                } else {
                    println!("勝率{} 不正解...",score);
                }

                count += 1;
            }

            println!("勝ち {}% (勝ちと評価された局面の割合 {}%)",win as f32 / count as f32 * 100.,estimated_win as f32 / count as f32 * 100.);
            println!("負け {}% (負けと評価された局面の割合 {}%)",(count - win) as f32 / count as f32 * 100.,
                     (count - estimated_win) as f32 / count as f32 * 100.);
            println!("正解率 {}%",successed as f32 / count as f32 * 100.);
        }

        print!("{}局面を学習しました。\n", processed_count);

        Ok(())
    }

    fn learning_from_yaneuraou_bin_batch(evalutor:&mut Trainer<M>,
                                         batch:Vec<Vec<u8>>,
                                         user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
    ) -> Result<(),ApplicationError> {
        match evalutor.learning_by_packed_sfens(
            batch,
            &*user_event_queue) {
            Err(e) => {
                return Err(ApplicationError::LearningError(format!(
                    "An error occurred while learning the neural network. {}",e
                )));
            },
            Ok(ms) => {
                println!("error_total: {}",ms);
                Ok(())
            }
        }
    }

    fn learning_from_hcpe_batch(evalutor: &mut Trainer<M>,
                                batch:Vec<Vec<u8>>,
                                user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
    ) -> Result<(),ApplicationError> {
        match evalutor.learning_by_hcpe(
            batch,
            &*user_event_queue) {
            Err(e) => {
                return Err(ApplicationError::LearningError(format!(
                    "An error occurred while learning the neural network. {}",e
                )));
            },
            Ok(ms) => {
                println!("error_total: {}",ms);
                Ok(())
            }
        }
    }

    fn learning_loop(&self,evalutor:&mut Trainer<M>,
                     checkpoint_path:&PathBuf,
                     current_filename:&str,
                     current_item:usize,
                     save_batch_count:usize,
                     notify_quit:&Arc<AtomicBool>,
                     mut teachers:Vec<Vec<u8>>,
                     learn_batch_size:usize,
                     learning_process:fn(
                        &mut Trainer<M>,
                        Vec<Vec<u8>>,
                        &Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
                     ) -> Result<(),ApplicationError>,
                     system_event_dispatcher:&mut USIEventDispatcher<SystemEventKind,
                         SystemEvent,(),FileLogger,ApplicationError>,
                     system_event_queue:&Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>>,
                     user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                     pending_count:&mut usize,
                     processed_count:&mut usize) -> Result<bool,ApplicationError> {
        let mut rng = rand::thread_rng();
        teachers.shuffle(&mut rng);

        let mut batch = Vec::with_capacity(learn_batch_size);

        let it = teachers.into_iter();

        for sfen in it {
            batch.push(sfen);

            if batch.len() == learn_batch_size {
                learning_process(evalutor,
                                 batch,
                                 &user_event_queue)?;
                *pending_count += 1;

                batch = Vec::with_capacity(learn_batch_size);
                *processed_count += learn_batch_size;

                self.save(evalutor,
                          checkpoint_path,
                          current_filename,
                          current_item,
                          *pending_count >= save_batch_count,
                          pending_count)?;
            }

            system_event_dispatcher.dispatch_events(&(), &*system_event_queue)?;

            if notify_quit.load(Ordering::Acquire) {
                return Ok(false);
            }
        }

        let remaing = batch.len();

        if remaing > 0 {
            learning_process(evalutor,
                             batch,
                             &user_event_queue)?;
            *pending_count += 1;

            self.save(evalutor,
                      checkpoint_path,
                      current_filename,
                      current_item,
                      *pending_count >= save_batch_count,
                      pending_count)?;
            *processed_count += remaing;
        }

        system_event_dispatcher.dispatch_events(&(), &*system_event_queue)?;

        if notify_quit.load(Ordering::Acquire) {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn save(&self,evalutor: &mut Trainer<M>,
            checkpoint_path:&PathBuf,
            current_filename:&str,
            current_item:usize,
            cond:bool,
            pending_count:&mut usize)
            -> Result<(),ApplicationError> {
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

    fn cmp(a:&Result<DirEntry,std::io::Error>,b:&Result<DirEntry,std::io::Error>) -> core::cmp::Ordering {
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
    }
}