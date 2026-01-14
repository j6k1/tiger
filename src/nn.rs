use std::fmt::Debug;
use std::path::{Path};
use std::{fs, io};
use std::collections::HashSet;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::{mpsc, Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use libc::size_t;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, ThreadRng};
use rand_distr::{Normal};
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ClippedReLu, Sigmoid};
use nncombinator::arr::{Arr};
use nncombinator::cuda::{CudaMutPtr, CudaPtr, MemoryMoveTo, MemoryType, ReadMemory, WriteMemory};
use nncombinator::cuda::allocator::{CudaAllocator, DeviceAlloc, MemoryPoolAllocator, MemoryPoolAllocatorInstantiation};
use nncombinator::device::{Device, DeviceCpu, DeviceGpu};
use nncombinator::layer::{AddLayer, BatchDataType, BatchForwardBase, BatchForward, BatchSize, BatchTrain, ForwardAll, PreTrain, Step, TryAddLayer};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::lossfunction::{CrossEntropy, LossFunction};
use nncombinator::ope::{UnitValue};
use nncombinator::optimizer::{AdamWBuilder};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence, PersistenceType, SaveToFile};
use nncombinator::scheduler::StepLR;
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::traits::Reader;
use packedsfen::{hcpe, yaneuraou};
use packedsfen::hcpe::haffman_code::GameResult;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use rayon::prelude::{ParallelIterator, IntoParallelIterator};
use shogi_dataloader::dataloader::{DataLoader, DataLoaderBuilder, UnifiedDataLoader};
use usiagent::event::{GameEndState};
use usiagent::hash::KyokumenHash;
use usiagent::math::Prng;
use usiagent::movepick::{MovePicker, RandomPicker};
use usiagent::rule::{CaptureOrPawnPromotions, Evasions, LegalMove, NonEvasionsAll, Rule, SquareToPoint, State};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MOCHIGOMA_KINDS, MochigomaCollections, Teban, ObtainKind, MochigomaKind};
use crate::Config;
use crate::error::{ApplicationError};
use crate::features::HalfKP;
use crate::layer::feature_transform::FeatureTransformLayerBuilder;
use crate::transposition_table::ZobristHash;

const BANMEN_SIZE:usize = 81;

const FU_INDEX:usize = 0;
const KYOU_INDEX:usize = FU_INDEX + BANMEN_SIZE;
const KEI_INDEX:usize = KYOU_INDEX + BANMEN_SIZE;
const GIN_INDEX:usize = KEI_INDEX + BANMEN_SIZE;
const KIN_INDEX:usize = GIN_INDEX + BANMEN_SIZE;
const KAKU_INDEX:usize = KIN_INDEX + BANMEN_SIZE;
const HISHA_INDEX:usize = KAKU_INDEX + BANMEN_SIZE;
const NARIKAKU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
const NARIHISHA_INDEX:usize = NARIKAKU_INDEX + BANMEN_SIZE;
const OPPONENT_FU_INDEX:usize = NARIHISHA_INDEX + BANMEN_SIZE;
const OPPONENT_KYOU_INDEX:usize = OPPONENT_FU_INDEX + BANMEN_SIZE;
const OPPONENT_KEI_INDEX:usize = OPPONENT_KYOU_INDEX + BANMEN_SIZE;
const OPPONENT_GIN_INDEX:usize = OPPONENT_KEI_INDEX + BANMEN_SIZE;
const OPPONENT_KIN_INDEX:usize = OPPONENT_GIN_INDEX + BANMEN_SIZE;
const OPPONENT_KAKU_INDEX:usize = OPPONENT_KIN_INDEX + BANMEN_SIZE;
const OPPONENT_HISHA_INDEX:usize = OPPONENT_KAKU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKAKU_INDEX:usize = OPPONENT_HISHA_INDEX + BANMEN_SIZE;
const OPPONENT_NARIHISHA_INDEX:usize = OPPONENT_NARIKAKU_INDEX + BANMEN_SIZE;
const PIECE_END:usize = OPPONENT_NARIHISHA_INDEX + BANMEN_SIZE;
const MOCHIGOMA_FU_INDEX:usize = 0;
const MOCHIGOMA_KYOU_INDEX:usize = MOCHIGOMA_FU_INDEX + 19;
const MOCHIGOMA_KEI_INDEX:usize = MOCHIGOMA_KYOU_INDEX + 5;
const MOCHIGOMA_GIN_INDEX:usize = MOCHIGOMA_KEI_INDEX + 5;
const MOCHIGOMA_KIN_INDEX:usize = MOCHIGOMA_GIN_INDEX + 5;
const MOCHIGOMA_KAKU_INDEX:usize = MOCHIGOMA_KIN_INDEX + 5;
const MOCHIGOMA_HISHA_INDEX:usize = MOCHIGOMA_KAKU_INDEX + 3;
const OPPONENT_MOCHIGOMA_FU_INDEX:usize = MOCHIGOMA_HISHA_INDEX + 3;
const OPPONENT_MOCHIGOMA_KYOU_INDEX:usize = OPPONENT_MOCHIGOMA_FU_INDEX + 19;
const OPPONENT_MOCHIGOMA_KEI_INDEX:usize = OPPONENT_MOCHIGOMA_KYOU_INDEX + 5;
const OPPONENT_MOCHIGOMA_GIN_INDEX:usize = OPPONENT_MOCHIGOMA_KEI_INDEX + 5;
const OPPONENT_MOCHIGOMA_KIN_INDEX:usize = OPPONENT_MOCHIGOMA_GIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_KAKU_INDEX:usize = OPPONENT_MOCHIGOMA_KIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_HISHA_INDEX:usize = OPPONENT_MOCHIGOMA_KAKU_INDEX + 3;
const MOCHIGOMA_END:usize = PIECE_END + OPPONENT_MOCHIGOMA_HISHA_INDEX + 3;

pub const FEATURES_NUM:usize = MOCHIGOMA_END * BANMEN_SIZE;

pub const ACTIVE_INDICES:usize = 39;

const SELF_INDEX_MAP:[usize; 7] = [
    MOCHIGOMA_FU_INDEX,
    MOCHIGOMA_KYOU_INDEX,
    MOCHIGOMA_KEI_INDEX,
    MOCHIGOMA_GIN_INDEX,
    MOCHIGOMA_KIN_INDEX,
    MOCHIGOMA_KAKU_INDEX,
    MOCHIGOMA_HISHA_INDEX
];

const OPPONENT_INDEX_MAP:[usize; 7] = [
    OPPONENT_MOCHIGOMA_FU_INDEX,
    OPPONENT_MOCHIGOMA_KYOU_INDEX,
    OPPONENT_MOCHIGOMA_KEI_INDEX,
    OPPONENT_MOCHIGOMA_GIN_INDEX,
    OPPONENT_MOCHIGOMA_KIN_INDEX,
    OPPONENT_MOCHIGOMA_KAKU_INDEX,
    OPPONENT_MOCHIGOMA_HISHA_INDEX
];
pub trait BatchNeuralNetwork<U,D,P,PT,I,O,L>: ForwardAll<Input=I,Output=O> +
                                 BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
                                 BatchTrain<U,D,L> + Persistence<U,P,PT> + Step
                                 where U: UnitValue<U>,
                                       D: Device<U>,
                                       I: BatchDataType + Debug + Send + Sync,
                                       O: BatchDataType,
                                       PT: PersistenceType,
                                       L: LossFunction<U> {}
impl<T,U,D,P,PT,I,O,L> BatchNeuralNetwork<U,D,P,PT,I,O,L> for T
    where T: ForwardAll<Input=I,Output=O> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
             BatchTrain<U,D,L> + Persistence<U,P,PT> + Step,
             U: UnitValue<U>,
             D: Device<U>,
             I: BatchDataType + Debug + Send + Sync,
             O: BatchDataType,
             PT: PersistenceType,
             L: LossFunction<U>,
             <I as BatchDataType>::Type: Debug + BatchSize {}
pub struct EvalutorCreator {
}
impl EvalutorCreator {
    pub fn create(savedir: impl AsRef<Path> + 'static, nn_path: impl AsRef<Path> + 'static, config:&Config)
        -> Result<Evalutor<impl ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
                                PreTrain<f32, OutStack=impl Send + Sync + 'static> + Send + Sync + 'static>, ApplicationError> {
        let mut rnd = prelude::thread_rng();
        let mut rnd = XorShiftRng::from_seed(rnd.gen());

        let n1 = Normal::<f32>::new(0.0, 0.25 * 1.5 / (ACTIVE_INDICES as f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2. / (512f32 + 32f32)).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, (2. / (32f32 + 32f32)).sqrt()).unwrap();
        let n4 = Normal::<f32>::new(0.0, 2f32 / (32f32 + 1f32).sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let optimizer_builder_feature = AdamWBuilder::new(&device)
            .lr(config.learning_rate_for_input_layer.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle_large = AdamWBuilder::new(&device)
            .lr(config.learning_rate_middle_layer_large.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_out = AdamWBuilder::new(&device)
            .lr(config.learning_rate_for_output_layer.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let net: InputLayer<f32, HalfKP<FEATURES_NUM>, (), _> = InputLayer::new(&device);

        let mut nn = net.try_add_layer(|l| {
            FeatureTransformLayerBuilder::<FEATURES_NUM, 256>::new().build(l, &device,
                                                                           || n1.sample(&mut rnd),
                                                                           || 0.0,
                                                                           &optimizer_builder_feature)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<{256 * 2}, 32>::new().build(l, &device,
                                                             || n2.sample(&mut rnd),
                                                             || 0.0,
                                                             &optimizer_builder_middle_large
            )
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                     || n3.sample(&mut rnd),
                                                      || 0.0,
                                                      &optimizer_builder_middle)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                     || {
                                                         n4.sample(&mut rnd)
                                                     },|| (0.503f32 / 0.497f32).ln(), &optimizer_builder_out)
       })?.add_layer(|l| {
            ActivationLayer::new(l, Sigmoid::new(&device), &device)
        }).try_add_layer(|l| {
            LinearOutputLayer::new(l, &device)
        })?;

        let mut p = BinFilePersistence::new(savedir.as_ref().join(&nn_path))?;

        nn.load(&mut p)?;

        Ok(Evalutor {
            nn:nn,
            material_evalutor:crate::evalutor::material::Evalutor::new()
        })
    }
}
pub struct Evalutor<M>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
             PreTrain<f32> + Send + Sync + 'static,
             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    nn:M,
    material_evalutor:crate::evalutor::material::Evalutor
}
impl<M> Evalutor<M>
    where M: ForwardAll<Input=HalfKP<FEATURES_NUM>, Output=Arr<f32, 1>> +
             PreTrain<f32> + Send + Sync + 'static,
             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn evalute(&self, t:Teban, state:&State, mc:&MochigomaCollections) -> Result<i32,ApplicationError> {
        let input = HalfKP::new(InputCreator::make_input(t,state,mc),InputCreator::make_input(t.opposite(),state,mc));

        let r = self.nn.forward_all(input)?;

        Ok(((r[0] - 0.5) * 1200.) as i32)
    }

    pub fn evalute_material(&self,teban:Teban,state:&State,mc:&MochigomaCollections) -> i32 {
        self.material_evalutor.evalute(teban,state,mc)
    }
}
pub enum WorkerItem {
    Job(HalfKP<FEATURES_NUM>,Sender<Result<Arr<f32,1>,ApplicationError>>),
    Start(Arc<AtomicUsize>),
    Quit
}
pub struct ServiceWorker {
    sender: Sender<WorkerItem>,
    done_receiver:Arc<Mutex<Receiver<()>>>,
    error_receiver:Arc<Mutex<Receiver<Result<(),ApplicationError>>>>,
    working: Arc<AtomicBool>,
    threads:Arc<AtomicUsize>,
    busy_threads:Arc<AtomicUsize>
}
impl ServiceWorker {
    pub fn new(config: &Config) -> Result<Self,ApplicationError> {
        let (s,r) = mpsc::channel::<WorkerItem>();
        let (es,er) = mpsc::channel::<Result<(),ApplicationError>>();
        let (ds,dr) = mpsc::channel::<()>();

        let working = Arc::new(AtomicBool::new(true));

        {
            let config = config.clone();

            let working = Arc::clone(&working);

            let _ = std::thread::Builder::new()
                .stack_size(1024 * 1024 * 512)
                .name(String::from("evalution thread worker.")).spawn(move || {

                let allocator = match MemoryPoolAllocator::with_size(4 * 1024 * 1024 * 1024,DeviceAlloc::new()) {
                    Ok(allocator) => {
                        let _ = es.send(Ok(()));
                        allocator
                    },
                    Err(e) => {
                        let _ = es.send(Err(ApplicationError::from(e)));
                        return;
                    }
                };

                let trainer = match TrainerCreator::create(String::from("data"),
                                       String::from("nn.bin"),
                                       &config,
                                       allocator) {
                    Ok(trainer) => {
                        let _ = es.send(Ok(()));
                        trainer
                    },
                    Err(e) => {
                        let _ = es.send(Err(ApplicationError::from(e)));
                        return;
                    }
                };

                let mut batch = vec![];
                let mut sender = vec![];

                while working.load(Ordering::Acquire) {
                    match r.recv() {
                        Err(e) => {
                            let _ = es.send(Err(ApplicationError::from(e)));
                            return;
                        },
                        Ok(item) => {
                            match item {
                                WorkerItem::Quit => {
                                    if batch.len() > 0 {
                                        break;
                                    }
                                },
                                WorkerItem::Job(input,s) => {
                                    batch.push(input);
                                    sender.push(s);
                                    let _ = es.send(Ok(()));
                                },
                                WorkerItem::Start(busy_threads) => {
                                    if batch.len() > 0 {
                                        let r = match trainer.nn.batch_forward(batch.into()) {
                                            Ok(r) => {
                                                let _ = es.send(Ok(()));
                                                r
                                            },
                                            Err(e) => {
                                                let _ = es.send(Err(ApplicationError::from(e)));
                                                return;
                                            }
                                        };

                                        busy_threads.fetch_sub(r.len(),Ordering::SeqCst);

                                        for (r,s) in r.iter().zip(sender.iter()) {
                                            let _ = s.send(Ok(r.into()));
                                        }

                                        batch = vec![];
                                        sender = vec![];
                                    } else {
                                        let _ = es.send(Ok(()));
                                    }
                                }
                            }
                        },
                    };
                }

                let _ = trainer;

                let _ = ds.send(());
            });
        }

        er.recv()??;
        er.recv()??;

        Ok(ServiceWorker {
            sender:s,
            done_receiver:Arc::new(Mutex::new(dr)),
            error_receiver:Arc::new(Mutex::new(er)),
            working: Arc::clone(&working),
            threads: Arc::new(AtomicUsize::new(0)),
            busy_threads: Arc::new(AtomicUsize::new(0))
        })
    }

    pub fn start_thread(&self) {
        self.threads.fetch_add(1,Ordering::SeqCst);
    }

    pub fn leave_thread(&self) -> Result<(),ApplicationError> {
        self.threads.fetch_sub(1, Ordering::SeqCst);

        if self.busy_threads.load(Ordering::SeqCst) == self.threads.load(Ordering::SeqCst) {
            self.sender.send(WorkerItem::Start(Arc::clone(&self.busy_threads))).map_err(|_| {
                ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
            })?;

            match self.error_receiver.lock() {
                Ok(r) => {
                    r.recv().map_err(|_| {
                        ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
                    })??
                },
                Err(_) => {
                    return Err(ApplicationError::InvalidStateError(String::from("evalution thread worker is dead.")));
                }
            };
        }

        Ok(())
    }
    pub fn submit(&self, input:HalfKP<FEATURES_NUM>) -> Result<Arr<f32,1>,ApplicationError> {
        let (s,r) = mpsc::channel();

        self.busy_threads.fetch_add(1, Ordering::SeqCst);

        let immdiate = self.busy_threads.load(Ordering::SeqCst) == self.threads.load(Ordering::SeqCst);

        assert!(self.busy_threads.load(Ordering::SeqCst) <= self.threads.load(Ordering::SeqCst), "thread count is wrong.");

        self.sender.send(WorkerItem::Job(input,s)).map_err(|_| {
            ApplicationError::InvalidStateError(String::from("evalution thread worker task send failed."))
        })?;

        match self.error_receiver.lock() {
            Ok(r) => {
                r.recv().map_err(|_| {
                    ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
                })??
            },
            Err(_) => {
                return Err(ApplicationError::InvalidStateError(String::from("evalution thread worker is dead.")));
            }
        };

        if immdiate {
            self.sender.send(WorkerItem::Start(Arc::clone(&self.busy_threads))).map_err(|_| {
                ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
            })?;

            match self.error_receiver.lock() {
                Ok(r) => {
                    r.recv().map_err(|_| {
                        ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
                    })??
                },
                Err(_) => {
                    return Err(ApplicationError::InvalidStateError(String::from("evalution thread worker is dead.s")));
                }
            };
        }

        let r = r.recv().map_err(|_| ApplicationError::InvalidStateError(String::from("evalution thread worker is dead.")));

        r?
    }

    pub fn release(&self) -> Result<(),ApplicationError> {
        self.working.store(false,Ordering::Release);
        self.sender.send(WorkerItem::Quit).map_err(|_| {
            ApplicationError::InvalidStateError(String::from("evalution thread worker task send failed."))
        })?;

        match self.done_receiver.lock() {
            Ok(r) => {
                r.recv().map_err(|_| {
                    ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
                })?;
            },
            Err(_) => {
                return Err(ApplicationError::InvalidStateError(String::from("evalution thread worker is dead.")));
            }
        };

        Ok(())
    }
}
impl Drop for ServiceWorker {
    fn drop(&mut self) {
        self.working.store(false,Ordering::Release);
    }
}
pub struct EvalTester {
    worker:ServiceWorker,
    material_evalutor:crate::evalutor::material::Evalutor
}
impl EvalTester {
    pub fn new(config:&Config) -> Result<Self,ApplicationError> {
        Ok(EvalTester {
            worker:ServiceWorker::new(config)?,
            material_evalutor:crate::evalutor::material::Evalutor::new()
        })
    }

    pub fn evalute(&self, t:Teban, state:&State, mc:&MochigomaCollections) -> Result<i32,ApplicationError> {
        let input = HalfKP::new(InputCreator::make_input(t,state,mc),InputCreator::make_input(t.opposite(),state,mc));

        let r = self.worker.submit(input)?;

        Ok(((r[0] - 0.5) * 1200.) as i32)
    }

    fn process_result(&self,current_threads:&mut usize,
                      count:&mut usize,
                      same_moves:&mut usize,
                      successed:&mut usize,
                      estimated_win:&mut usize,
                      win:&mut usize,
                      sr:&Receiver<Result<Option<(GameEndState, f32, bool)>,ApplicationError>>,
    ) -> Result<(),ApplicationError> {
        while *current_threads > 0 {
            match sr.recv().map_err(|_| {
                ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
            })?? {
                None => {
                    self.worker.leave_thread()?;
                    *current_threads -= 1;

                    continue
                },
                Some((s, score, same_move)) => {
                    self.worker.leave_thread()?;

                    *current_threads -= 1;

                    if *count >= 350 {
                        continue;
                    }

                    if *count % 5 == 0 {
                        print!(".");
                        io::stdout().flush().unwrap();
                    }

                    if same_move {
                        *same_moves += 1;
                    }

                    if score >= 0.5 {
                        *estimated_win += 1;
                    }

                    let success = match s {
                        GameEndState::Draw => {
                            true
                        },
                        GameEndState::Win => {
                            *win += 1;
                            score >= 0.5
                        },
                        _ => {
                            score < 0.5
                        }
                    };

                    if success {
                        *successed += 1;
                    }

                    *count += 1;
                }
            }
        }

        Ok(())
    }
    pub fn eval_test<F>(self:Arc<Self>,
                        testdir: String,
                        ext: &str,
                        item_size: usize,
                        learn_sfen_read_size: usize,
                        test_process: F
    ) -> Result<(), ApplicationError>
    where F: Fn(&EvalTester, Vec<u8>) -> Result<Option<(GameEndState, f32, bool)>, ApplicationError> + Send + Sync + 'static, {
        let test_process = Arc::new(test_process);

        let dataloader_builder = DataLoaderBuilder::new(Path::new(&testdir)
            .join("tests"))
            .shuffle(true)
            .ext(ext.to_string())
            .batch_size(256)
            .read_sfen_size(learn_sfen_read_size)
            .sfen_size(item_size)
            .send_buffer_size(10);

        let mut dataloader:UnifiedDataLoader<Vec<Vec<u8>>, ApplicationError> = dataloader_builder.build(| sfens | Ok(Some(sfens)))?;
        let mut successed = 0usize;
        let mut estimated_win = 0usize;
        let mut win = 0usize;
        let mut count = 0usize;
        let mut same_moves = 0usize;

        let mut current_threads = 0usize;
        let (ss,sr) = mpsc::channel::<Result<Option<(GameEndState,f32,bool)>,ApplicationError>>();

        'outer: while let Some((_,_,batch)) = dataloader.load()? {
            for packed in batch.into_iter() {
                if current_threads >= 128 {
                    self.process_result(&mut current_threads,
                                        &mut count,
                                        &mut same_moves,
                                        &mut successed,
                                        &mut estimated_win,
                                        &mut win,
                                        &sr)?;
                    if count >= 350 {
                        break 'outer;
                    }
                } else {
                    let this = Arc::clone(&self);
                    let test_process = Arc::clone(&test_process);

                    self.worker.start_thread();

                    {
                        let ss = ss.clone();

                        std::thread::Builder::new()
                            .stack_size(1024 * 1024 * 512).spawn(move || {
                            let r = test_process(&this, packed);

                            let _ = ss.send(r);
                        })?;
                    }

                    current_threads += 1;
                }
            }

            self.process_result(&mut current_threads,
                                &mut count,
                                &mut same_moves,
                                &mut successed,
                                &mut estimated_win,
                                &mut win,
                                &sr)?;
            if count >= 350 {
                break 'outer;
            }
        }

        self.worker.release()?;

        println!("");
        println!("勝ち {}% (勝ちと評価された局面の割合 {}%)", win as f32 / count as f32 * 100., estimated_win as f32 / count as f32 * 100.);
        println!("負け {}% (負けと評価された局面の割合 {}%)", (count - win) as f32 / count as f32 * 100.,
                 (count - estimated_win) as f32 / count as f32 * 100.);
        println!("正解率(勝敗) {}%", successed as f32 / count as f32 * 100.);
        println!("正解率(指し手の一致率) {}%", same_moves as f32 / count as f32 * 100.);
        println!("{}件のテストサンプルを利用しました。",count);

        Ok(())
    }

    pub fn test_by_packed_sfens(&self,
                                packed_sfen:Vec<u8>)
                                -> Result<Option<(GameEndState,f32,bool)>,ApplicationError> {
        let mut packed_sfen_reader = PackedSfenReader::new();

        let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
            value: _,
            best_move,
            end_ply: _,
            game_result
        }) = packed_sfen_reader.read_sfen_with_extended(packed_sfen)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
            InputCreator::make_input(teban, &state, &mc),
            InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.worker.submit(input)?;

        let same = match best_move {
            yaneuraou::reader::BestMove::MoveTo(sx,sy,dx,dy,n) => {
                self.select_bestmove(teban, &state, mc)?.map(|m| {
                    match m {
                        LegalMove::To(m) => {
                            let (bsx, bsy) = m.src().square_to_point();
                            let (bdx, bdy) = m.dst().square_to_point();
                            let bn = m.is_nari();

                            if sx == bsx && sy == bsy && bdx == dx && bdy == dy && bn == n {
                                true
                            } else {
                                false
                            }
                        },
                        _ => false
                    }
                }).unwrap_or(false)
            },
            yaneuraou::reader::BestMove::MovePut(k,x,y) => {
                self.select_bestmove(teban, &state, mc)?.map(|m| {
                    match m {
                        LegalMove::Put(m) => {
                            let (bx,by) = m.dst().square_to_point();
                            let bk = m.kind();

                            if x == bx && y == by && bk == k {
                                true
                            } else {
                                false
                            }
                        },
                        _ => false
                    }
                }).unwrap_or(false)
            },
            _ => {
                return Ok(None);
            }
        };

        Ok(Some((game_result,r[0],same)))
    }

    pub fn test_by_packed_hcpe(&self,
                               hcpe:Vec<u8>)
                               -> Result<Option<(GameEndState,f32,bool)>,ApplicationError> {
        let mut hcpe_reader = HcpeReader::new();

        let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
            eval: _,
            best_move,
            game_result
        }) = hcpe_reader.read_sfen_with_extended(hcpe)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
            InputCreator::make_input(teban, &state, &mc),
            InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.worker.submit(input)?;

        let same = match best_move {
            hcpe::reader::BestMove::MoveTo(sx,sy,dx,dy,n) => {
                self.select_bestmove(teban, &state, mc)?.map(|m| {
                    match m {
                        LegalMove::To(m) => {
                            let (bsx, bsy) = m.src().square_to_point();
                            let (bdx, bdy) = m.dst().square_to_point();
                            let bn = m.is_nari();

                            if sx == bsx && sy == bsy && bdx == dx && bdy == dy && bn == n {
                                true
                            } else {
                                false
                            }
                        },
                        _ => false
                    }
                }).unwrap_or(false)
            },
            hcpe::reader::BestMove::MovePut(k,x,y) => {
                self.select_bestmove(teban, &state, mc)?.map(|m| {
                    match m {
                        LegalMove::Put(m) => {
                            let (bx,by) = m.dst().square_to_point();
                            let bk = m.kind();

                            if x == bx && y == by && bk == k {
                                true
                            } else {
                                false
                            }
                        },
                        _ => false
                    }
                }).unwrap_or(false)
            },
            _ => {
                return Ok(None)
            }
        };

        let s = match game_result {
            GameResult::SenteWin if teban == Teban::Sente => {
                GameEndState::Win
            },
            GameResult::SenteWin => {
                GameEndState::Lose
            },
            GameResult::GoteWin if teban == Teban::Gote => {
                GameEndState::Win
            },
            GameResult::GoteWin => {
                GameEndState::Lose
            },
            _ => GameEndState::Draw
        };

        Ok(Some((s,r[0],same)))
    }
    fn qsearch(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
               zh: &ZobristHash<u64>,
               hasher: &KyokumenHash<u64>,
               history:&mut HashSet<(Teban,u64,u64)>,
               mut alpha:i32,beta:i32,depth:usize,rng:&mut ThreadRng) -> Result<i32,ApplicationError> {
        let (mk,sk) = zh.keys();

        if history.contains(&(teban,mk,sk)) {
            let score = self.evalute(teban, state, mc)?;
            return Ok(score);
        }

        let mut picker = RandomPicker::new(Prng::new(rng.gen()));

        let in_check = Rule::in_check(teban.opposite(),state);

        if in_check {
            Rule::generate_moves::<Evasions>(teban,state,mc,&mut picker)?;
        } else {
            Rule::generate_moves_by_banmen::<CaptureOrPawnPromotions>(teban,state,&mut picker)?;
        }

        if in_check && picker.len() == 0 {
            return Ok(-30000);
        }

        let mvs = picker.filter(|m| m.obtained().is_some()).collect::<Vec<_>>();

        if mvs.len() == 0 {
            return Ok(self.evalute(teban,state,mc)?);
        }

        let (mk,sk) = zh.keys();

        history.insert((teban,mk,sk));

        let stand_pat = self.material_evalutor.evalute(teban,state,mc);

        if stand_pat >= beta {
            return Ok(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut bestscore = -30000;

        for m in mvs {
            if let Some(ObtainKind::Ou) = match m {
                LegalMove::To(m) => m.obtained(),
                _ => None
            } {
                history.remove(&(teban,mk,sk));

                if depth == 0 {
                    return Ok(30000);
                } else {
                    return Ok(beta);
                }
            }

            let o = match m {
                LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                _ => None
            };

            let zh = zh.updated(&hasher, teban, state.get_banmen(), mc, m.to_applied_move(), &o);

            let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

            let score = -self.qsearch(teban.opposite(),&next,&nmc,&zh,hasher,history,-beta,-alpha,depth+1,rng)?;

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

        history.remove(&(teban,mk,sk));

        Ok(bestscore)
    }

    pub fn select_bestmove(&self, teban:Teban, state:&State, mc:MochigomaCollections) -> Result<Option<LegalMove>,ApplicationError> {
        let hasher = KyokumenHash::new();

        let zh = match &mc {
            &MochigomaCollections::Empty => {
                ZobristHash::new(&hasher,teban,state.get_banmen(),&Mochigoma::new(),&Mochigoma::new())
            },
            &MochigomaCollections::Pair(ref ms,ref mg) if teban == Teban::Sente => {
                ZobristHash::new(&hasher,teban,state.get_banmen(),ms,mg)
            },
            &MochigomaCollections::Pair(ref mg,ref ms) => {
                ZobristHash::new(&hasher,teban,state.get_banmen(),ms,mg)
            }
        };

        let mut rnd = rand::thread_rng();
        let mut picker = RandomPicker::new(Prng::new(rnd.gen()));

        Rule::generate_moves::<NonEvasionsAll>(teban,state,&mc,&mut picker)?;

        let mut best_score = None;
        let mut best_move = None;

        for m in &mut picker {
            let next = Rule::apply_move_none_check(state, teban, &mc, m.to_applied_move());

            match next {
                (state, mc, _) => {
                    let s = -self.qsearch(
                        teban.opposite(),&state,&mc,&zh,&hasher,&mut HashSet::new(),-30000,30000,0,&mut rnd
                    )?;

                    match best_score {
                        None => {
                            best_score = Some(s);
                            best_move = Some(m);
                        },
                        Some(bs) if s > bs => {
                            best_score = Some(s);
                            best_move = Some(m);
                        },
                        _ => ()
                    }
                }
            }
        }

        Ok(best_move)
    }
}
pub type LF = CrossEntropy<f32>;

pub struct Trainer<M,A>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence<f32>,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,
          A:CudaAllocator {
    pub nn:M,
    a:PhantomData<A>,
    nn_path:String,
    nnsavedir:String
}
pub struct TrainerCreator {
}
impl TrainerCreator {
    pub fn create<A: CudaAllocator + MemoryType + 'static>(save_dir:String, nn_path:String, config:&Config, allocator:A)
        -> Result<Trainer<impl BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence<f32>,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,A>, ApplicationError>
        where A: CudaAllocator,
              for<'a> CudaPtr<f32,A>: ReadMemory<f32> +
                                      WriteMemory<f32> + MemoryMoveTo<f32,CudaMutPtr<'a,f32,A>>,
              CudaPtr<usize,A>: WriteMemory<usize>,
              CudaPtr<u8,A>: WriteMemory<u8> {

        //println!("FEATURES_NUM = {}",FEATURES_NUM);

        let mut rnd = prelude::thread_rng();
        let mut rnd = XorShiftRng::from_seed(rnd.gen());

        let n1 = Normal::<f32>::new(0.0, 0.25 * 1.5 / (ACTIVE_INDICES as f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2. / (512f32 + 32f32)).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, (2. / (32f32 + 32f32)).sqrt()).unwrap();
        let n4 = Normal::<f32>::new(0.0, 2f32 / (32f32 + 1f32).sqrt()).unwrap();

        let device = DeviceGpu::new(&allocator)?;

        let optimizer_builder_feature = AdamWBuilder::new(&device)
            .lr(config.learning_rate_for_input_layer.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle_large = AdamWBuilder::new(&device)
            .lr(config.learning_rate_middle_layer_large.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_out = AdamWBuilder::new(&device)
            .lr(config.learning_rate_for_output_layer.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let net: InputLayer<f32, HalfKP<FEATURES_NUM>, (), _> = InputLayer::new(&device);

        let mut nn = net.try_add_layer(|l| {
            FeatureTransformLayerBuilder::<FEATURES_NUM,256>::new().build(l,&device,
                                                                          || n1.sample(&mut rnd), || 0.0,
                                                                          &optimizer_builder_feature)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<{256 * 2}, 32>::new().build(l, &device,
                || n2.sample(&mut rnd),
                   || 0.0, &optimizer_builder_middle_large)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<32, 32>::new().build(l, &device,
                || n3.sample(&mut rnd),
                   || 0.0 ,&optimizer_builder_middle)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ClippedReLu::new(&device,1.0), &device)
        }).try_add_layer(|l| {
            LinearLayerBuilder::<32, 1>::new().build(l, &device,
            move || {
                n4.sample(&mut rnd)
            },|| (0.503f32 / 0.497f32).ln(), &optimizer_builder_out)
        })?.add_layer(|l| {
            ActivationLayer::new(l, Sigmoid::new(&device), &device)
        }).try_add_layer(|l| {
            LinearOutputLayer::new(l, &device)
        })?;

        {
            let save_dir = Path::new(&save_dir);

            let nn_path = Path::new(&nn_path);

            if save_dir.join(nn_path).exists() {
                let mut p = BinFilePersistence::new(save_dir
                    .join(nn_path)
                )?;

                nn.load(&mut p)?;
            }
        }

        Ok(Trainer {
            nn:nn,
            a:PhantomData::<A>,
            nn_path: nn_path,
            nnsavedir: save_dir
        })
    }
}
impl<M,A> Trainer<M,A>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence<f32>,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,
          A: CudaAllocator {
    fn sigmoid(x:f32) -> f32 {
        1. / (1. + (-0.0017928004128957029 * x).exp())
    }

    pub fn make_packed_sfens_parser<'a>(lambda:f32, verbose: bool)
        -> impl FnMut(Vec<Vec<u8>>)
                -> Result<Option<(Vec<Arr<f32,1>>,Vec<HalfKP<FEATURES_NUM>>)>,ApplicationError> + Send + 'static {
        move | packed_sfens | {
            let sfens_with_extended = packed_sfens.into_par_iter().map(|entry|  {
                let mut packed_sfen_reader = PackedSfenReader::new();

                let ((teban, banmen, mc), yaneuraou::haffman_code::ExtendFields {
                    value: score,
                    best_move: _,
                    end_ply: _,
                    game_result
                }) = match packed_sfen_reader.read_sfen_with_extended(entry) {
                    Ok(r) => r,
                    Err(e) => {
                        return Err(e)
                    }
                };

                Ok((teban, banmen, mc, game_result, score))
            }).collect::<Result<Vec<_>,_>>()?;

            let batch = sfens_with_extended.into_par_iter()
                .map(|(teban, banmen, mc, es, score)| {
                    let state = State::new(banmen);

                    let input = HalfKP::new(
                        InputCreator::make_input(teban, &state, &mc),
                        InputCreator::make_input(teban.opposite(), &state, &mc)
                    );

                    let mut t = Arr::<f32, 1>::new();

                    t[0] = {
                        let t = match es {
                            GameEndState::Win if teban == Teban::Sente => {
                                //sente_rate
                                1.
                            },
                            GameEndState::Win => {
                                1.
                            },
                            GameEndState::Lose if teban == Teban::Sente => {
                                0.
                            },
                            GameEndState::Lose => {
                                0.
                            },
                            _ => 0.5f32
                        };

                        let r = t * lambda + Self::sigmoid(score as f32) * (1. - lambda);

                        r.max(0.01).min(0.99)
                  };

                    (t, input)
                }).fold(|| (Vec::new(), Vec::new()), |mut acc, (t, i)| {
                    acc.0.push(t);
                    acc.1.push(i);

                    acc
                }).reduce(|| (Vec::new(),Vec::new()), | mut acc, (mut t,mut i)| {
                    acc.0.append(&mut t);
                    acc.1.append(&mut i);
                    acc
                });

            if verbose {
                let o = &batch.0;

                let len = o.len();

                let mean = o.iter().fold(0.0, |acc, x| acc + x[0]) / len as f32;
                let min = o.iter().fold(0.0 / 0.0, |acc, x| x[0].min(acc));
                let max = o.iter().fold(0.0 / 0.0, |acc, x| x[0].max(acc));
                let std = o.iter().map(|x| (x[0] - mean).powf(2.0)).sum::<f32>() / len as f32;

                println!("label mean: {:.9e}, min: {:.9e}, max: {:.9e}, std: {:.9e}", mean, min, max, std);
            }

            Ok(Some(batch))
        }
    }

    pub fn make_hcpe_parser<'a>(lambda:f32, verbose: bool)
        -> impl FnMut(Vec<Vec<u8>>) ->
                Result<Option<(Vec<Arr<f32,1>>,Vec<HalfKP<FEATURES_NUM>>)>,ApplicationError> + Send + 'static {
        move | hcpes | {
            let sfens_with_extended = hcpes.into_par_iter().map(|entry| {
                let mut hcpe_reader = HcpeReader::new();

                let ((teban, banmen, mc), hcpe::haffman_code::ExtendFields {
                    eval: score,
                    best_move: _,
                    game_result
                }) = match hcpe_reader.read_sfen_with_extended(entry) {
                    Ok(r) => r,
                    Err(e) => {
                        return Err(e);
                    }
                };

                Ok((teban, banmen, mc, game_result, score))
            }).collect::<Result<Vec<_>,_>>()?;

            let batch = sfens_with_extended.into_par_iter()
                .map(|(teban, banmen, mc, es, score)| {
                    let state = State::new(banmen);

                    let input = HalfKP::new(
                        InputCreator::make_input(teban, &state, &mc),
                        InputCreator::make_input(teban.opposite(), &state, &mc)
                    );

                    let (rate, es) = match (es, teban) {
                        (GameResult::Draw, _) => {
                            (1., GameEndState::Draw)
                        },
                        (GameResult::SenteWin, Teban::Sente) => {
                            (1., GameEndState::Win)
                        },
                        (GameResult::GoteWin, Teban::Gote) => {
                            (1., GameEndState::Win)
                        },
                        (GameResult::SenteWin, Teban::Gote) => {
                            (1., GameEndState::Lose)
                        },
                        (GameResult::GoteWin, Teban::Sente) => {
                            (1., GameEndState::Lose)
                        }
                    };

                    let mut t = Arr::<f32, 1>::new();

                    t[0] = {
                        let t = match es {
                            GameEndState::Win => {
                                rate
                            }
                            GameEndState::Lose => {
                                0.
                            },
                            _ => 0.5f32
                        };

                        let r = t * lambda + Self::sigmoid(score as f32) * (1. - lambda);

                        r.max(0.01).min(0.99)
                    };

                    (t, input)
                }).fold(|| (Vec::new(), Vec::new()), | mut acc, (t, i)| {
                    acc.0.push(t);
                    acc.1.push(i);
                    acc
                }).reduce(|| (Vec::new(),Vec::new()), | mut acc, (mut t,mut i)| {
                    acc.0.append(&mut t);
                    acc.1.append(&mut i);
                    acc
                });

            if verbose {
                let o = &batch.0;

                let len = o.len();

                let mean = o.iter().fold(0.0, |acc, x| acc + x[0]) / len as f32;
                let min = o.iter().fold(0.0 / 0.0, |acc, x| x[0].min(acc));
                let max = o.iter().fold(0.0 / 0.0, |acc, x| x[0].max(acc));
                let std = o.iter().map(|x| (x[0] - mean).powf(2.0)).sum::<f32>() / len as f32;

                println!("label mean: {:.9e}, min: {:.9e}, max: {:.9e}, std: {:.9e}", mean, min, max, std);
            }

            Ok(Some(batch))
        }
    }

    pub fn save(&mut self) -> Result<(),ApplicationError> {
        let tmp_nn_path = Path::new(&self.nnsavedir).join(&format!("{}.{}", &self.nn_path, "tmp"));

        let mut p = BinFilePersistence::new(tmp_nn_path.as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        self.nn.save(&mut p)?;

        p.save(&tmp_nn_path)?;

        fs::rename(Path::new(&tmp_nn_path),Path::new(&self.nnsavedir).join(&self.nn_path).as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        Ok(())
    }
}
pub struct InputCreator;

impl InputCreator {
    pub fn make_input(t:Teban,state:&State,mc:&MochigomaCollections) -> Vec<size_t> {
        let mut inputs = Vec::new();

        let ou_position = if t == Teban::Sente {
            Rule::ou_square(t,state)
        } else {
            80 -  Rule::ou_square(t,state)
        };

        match state.get_banmen() {
            &Banmen(ref kinds) => {
                for y in 0..9 {
                    for x in 0..9 {
                        let kind = kinds[y][x];

                        if kind != KomaKind::Blank {
                            let index = InputCreator::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

                            if index < MOCHIGOMA_END {
                                inputs.push((ou_position as usize * (MOCHIGOMA_END) + index) as size_t);
                            }
                        }
                    }
                }
            }
        }

        let ms = Mochigoma::new();
        let mg = Mochigoma::new();
        let (ms,mg) = match mc {
            &MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
            &MochigomaCollections::Empty => (&ms,&mg),
        };

        let (ms,mg) = match t {
            Teban::Sente => (ms,mg),
            Teban::Gote => (mg,ms),
        };

        let s = ou_position as usize * MOCHIGOMA_END + PIECE_END;

        for &k in &MOCHIGOMA_KINDS {
            for i in 0..ms.get(k) {
                inputs.push((s + SELF_INDEX_MAP[k as usize] + i) as size_t);
            }

            for i in 0..mg.get(k) {
                inputs.push((s + OPPONENT_INDEX_MAP[k as usize] + i) as size_t);
            }
        }
        inputs
    }

    #[inline]
    fn input_index_of_banmen(teban:Teban,kind:KomaKind,x:u32,y:u32) -> Result<usize,ApplicationError> {
        const SENTE_INDEX_MAP:[usize; 28] = [
            FU_INDEX,
            KYOU_INDEX,
            KEI_INDEX,
            GIN_INDEX,
            KIN_INDEX,
            KAKU_INDEX,
            HISHA_INDEX,
            MOCHIGOMA_END,
            KIN_INDEX,
            KIN_INDEX,
            KIN_INDEX,
            KIN_INDEX,
            NARIKAKU_INDEX,
            NARIHISHA_INDEX,
            OPPONENT_FU_INDEX,
            OPPONENT_KYOU_INDEX,
            OPPONENT_KEI_INDEX,
            OPPONENT_GIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KAKU_INDEX,
            OPPONENT_HISHA_INDEX,
            MOCHIGOMA_END,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_NARIKAKU_INDEX,
            OPPONENT_NARIHISHA_INDEX
        ];

        const GOTE_INDEX_MAP:[usize; 28] = [
            OPPONENT_FU_INDEX,
            OPPONENT_KYOU_INDEX,
            OPPONENT_KEI_INDEX,
            OPPONENT_GIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KAKU_INDEX,
            OPPONENT_HISHA_INDEX,
            MOCHIGOMA_END,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_NARIKAKU_INDEX,
            OPPONENT_NARIHISHA_INDEX,
            FU_INDEX,
            KYOU_INDEX,
            KEI_INDEX,
            GIN_INDEX,
            KIN_INDEX,
            KAKU_INDEX,
            HISHA_INDEX,
            MOCHIGOMA_END,
            KIN_INDEX,
            KIN_INDEX,
            KIN_INDEX,
            KIN_INDEX,
            NARIKAKU_INDEX,
            NARIHISHA_INDEX
        ];

        let index = match teban {
            Teban::Sente | Teban::Gote if kind == KomaKind::Blank => {
                return Err(ApplicationError::LogicError(
                    String::from(
                        "Calculation of index of difference input data of neural network failed. (KomaKind is 'Blank')"
                    )));
            },
            Teban::Sente => {
                SENTE_INDEX_MAP[kind as usize] + x as usize * 9 + y as usize
            },
            Teban::Gote => {
                let (x,y) = (8-x,8-y);

                GOTE_INDEX_MAP[kind as usize] + x as usize * 9 + y as usize
            }
        };

        Ok(index as usize)
    }
}