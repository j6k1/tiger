use std::fmt::Debug;
use std::path::{Path};
use std::{fs};
use std::marker::PhantomData;
use std::sync::{mpsc, Arc};
use std::sync::mpsc::{Receiver};
use getopts::{Matches};
use libc::{size_t};
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution};
use rand_distr::{Normal};
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ClippedReLu, Sigmoid};
use nncombinator::arr::{Arr};
use nncombinator::device::{Device, DeviceCpu};
use nncombinator::layer::{AddLayer, BatchDataType, BatchForwardBase, BatchSize, BatchTrain, ContinueForward, ForwardAll, PartialForward, PersistProgress, PreTrain, Step, TryAddLayer};
use nncombinator::layer::input::{DiffInputLayer, InputLayer};
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::lossfunction::{CrossEntropy, LossFunction};
use nncombinator::ope::{UnitValue};
use nncombinator::optimizer::{AdamWBuilder};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence, PersistenceType, SaveToFile};
use nncombinator::scheduler::{CosineAnnealingLR, LinearWarmupLR, Scheduler, StepLR};
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::traits::Reader;
use packedsfen::{hcpe, yaneuraou};
use packedsfen::hcpe::haffman_code::GameResult;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use rayon::prelude::{ParallelIterator, IntoParallelIterator};
use shogi_dataloader::dataloader::{DataLoader, DataLoaderBuilder, UnifiedDataLoader};
use usiagent::event::{GameEndState};
use usiagent::rule::{LegalMove, Rule, SquareToPoint, State};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MOCHIGOMA_KINDS, MochigomaCollections, Teban, MochigomaKind};
#[cfg(feature = "cuda")]
use nncombinator::device::{DeviceGpu};
#[cfg(feature = "cuda")]
use nncombinator::cuda::{CudaMutPtr, CudaPtr, MemoryMoveTo, MemoryType, ReadMemory, WriteMemory};
#[cfg(feature = "cuda")]
use nncombinator::cuda::allocator::{CudaAllocator};
use crate::{Config, EVAL_TEST_SAMPLES};
use crate::error::{ApplicationError};
use crate::features::{HalfKP, HalfKPDiff};
use crate::layer::diff_feature_transform::DiffFeatureTransformLayerBuilder;
use crate::layer::feature_transform::FeatureTransformLayerBuilder;
use crate::math::{Sign, SignFloat};

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

pub const FEATURES_OUTPUT:usize = 256;

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
pub fn sigmoid(x:f32) -> f32 {
    1. / (1. + (-0.0017928004128957029 * x).exp())
}
pub trait BatchNeuralNetwork<U,D,P,PT,I,O,L>: ForwardAll<Input=I,Output=O> +
                                 BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
                                 BatchTrain<U,D,L> + Persistence<U,P,PT> + PersistProgress<P,PT> + Step
                                 where U: UnitValue<U>,
                                       D: Device<U>,
                                       I: BatchDataType + Debug + Send + Sync,
                                       O: BatchDataType,
                                       PT: PersistenceType,
                                       L: LossFunction<U> {}
impl<T,U,D,P,PT,I,O,L> BatchNeuralNetwork<U,D,P,PT,I,O,L> for T
    where T: ForwardAll<Input=I,Output=O> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
             BatchTrain<U,D,L> + Persistence<U,P,PT> + PersistProgress<P,PT> + Step,
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
        -> Result<Evalutor<impl ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
                                PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                                ContinueForward<ConinueOutput=Arr<f32,1>> +
                                PreTrain<f32,OutStack=impl Send + Sync + 'static> + Send + Sync + 'static>, ApplicationError> {
        let mut rnd = prelude::thread_rng();
        let mut rnd = XorShiftRng::from_seed(rnd.gen());

        let n1 = Normal::<f32>::new(0.0, 0.25 * 1.5 / (ACTIVE_INDICES as f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2. / (512f32 + 32f32)).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, (2. / (32f32 + 32f32)).sqrt()).unwrap();
        let n4 = Normal::<f32>::new(0.0, 2f32 / (32f32 + 1f32).sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let optimizer_builder_feature = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle_large = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_middle = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let optimizer_builder_out = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(StepLR::new(config.step_count.unwrap_or(1),config.gamma.unwrap_or(0.5)))
            .weight_decay(1e-5);

        let net: DiffInputLayer<f32, HalfKP<FEATURES_NUM>, HalfKPDiff<f32,SignFloat<f32>,256>, Arr<f32,{256*2}>, (), _> = DiffInputLayer::new(&device);

        let mut nn = net.try_add_layer(|l| {
            DiffFeatureTransformLayerBuilder::<FEATURES_NUM,256>::new().build(l, &device,
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
    where for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32, 1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    nn:M,
    material_evalutor:crate::evalutor::material::Evalutor
}
impl<M> Evalutor<M>
    where for<'a> M: ForwardAll<Input=HalfKP<FEATURES_NUM>,Output=Arr<f32,1>> +
                     PreTrain<f32> + Send + Sync +
                     PartialForward<DiffInput=HalfKPDiff<f32,SignFloat<f32>,256>,PartialOutput=Arr<f32,{256*2}>,PartialOutputByDiff=Arr<f32,{256*2}>> +
                     ContinueForward<ConinueOutput=Arr<f32,1>> + 'static,
                  <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn prepare_evalute(&self, t:Teban, state:&State, mc:&MochigomaCollections) -> Result<Arr<f32,{256*2}>,ApplicationError> {
        let input = HalfKP::new(InputCreator::make_input(t,state,mc),InputCreator::make_input(t.opposite(),state,mc));

        let r = self.nn.partial_forward(input)?;

        Ok(r)
    }

    pub fn prepare_evalute_by_diff<'a>(&self, active_player: Teban, t:Teban, state:&State, mc:&MochigomaCollections,
                                       next:&State, nmc:&MochigomaCollections, m:LegalMove, partial_output:Arc<Arr<f32,{256*2}>>)
        -> Result<Arr<f32,{256*2}>,ApplicationError> {
        let ou_position = Rule::ou_square(active_player, state) as u32;

        match m {
            LegalMove::To(m) if m.src() == ou_position => {
                let input = HalfKP::new(InputCreator::make_input(t, next, nmc), InputCreator::make_input(t.opposite(), next, nmc));

                let r = self.nn.partial_forward(input)?;

                Ok(r)
            },
            m => {
                let input = HalfKPDiff::new(
                    InputCreator::make_diff_input(active_player, t, state, mc, m)?,
                    InputCreator::make_diff_input(active_player, t.opposite(), state, mc, m)?,
                    partial_output
                );

                let r = self.nn.partial_forward_by_diff(input)?;

                Ok(r)
            }
        }
    }
    pub fn evalute(&self, partial_output: &Arr<f32,{256*2}>) -> Result<i32,ApplicationError> {
        let r = self.nn.continue_forward(partial_output)?;

        Ok(((r[0] - 0.5) * 2230.) as i32)
    }
    pub fn evalute_material(&self,teban:Teban,state:&State,mc:&MochigomaCollections) -> i32 {
        self.material_evalutor.evalute(teban,state,mc)
    }

    fn process_result(&self,
                      current_threads:&mut usize,
                      count:&mut usize,
                      successed:&mut usize,
                      estimated_win:&mut usize,
                      win:&mut usize,
                      mae_acc:&mut f32,
                      lambda:f32,
                      sr:&Receiver<Result<Option<(GameEndState, f32, i16)>,ApplicationError>>,
    ) -> Result<(),ApplicationError> {
        while *current_threads > 0 {
            match sr.recv().map_err(|_| {
                ApplicationError::InvalidStateError(String::from("evalution thread worker is dead."))
            })?? {
                None => {
                    *current_threads -= 1;

                    continue
                },
                Some((s, score, eval)) => {
                    *current_threads -= 1;

                    if *count >= EVAL_TEST_SAMPLES {
                        continue;
                    }

                    if score >= 0.5 {
                        *estimated_win += 1;
                    }

                    let success = match s {
                        GameEndState::Draw => {
                            let t = lambda * 0.5 + sigmoid(eval as f32) * (1. - lambda);

                            *mae_acc += (t - score).abs();
                            true
                        },
                        GameEndState::Win => {
                            let t = lambda * 1.0 + sigmoid(eval as f32) * (1. - lambda);

                            *mae_acc += (t - score).abs();
                            *win += 1;
                            score >= 0.5
                        },
                        _ => {
                            let t = lambda * 0.0 + sigmoid(eval as f32) * (1. - lambda);

                            *mae_acc += (t - score).abs();

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
                        eval_test_max_threads: usize,
                        lambda: f32,
                        test_process: F
    ) -> Result<(), ApplicationError>
    where F: Fn(&Evalutor<M>, Vec<u8>) -> Result<Option<(GameEndState, f32, i16)>, ApplicationError> + Send + Sync + 'static, {
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
        let mut mae_acc = 0f32;

        let mut current_threads = 0usize;
        let (ss,sr) = mpsc::channel::<Result<Option<(GameEndState,f32,i16)>,ApplicationError>>();

        'outer: while let Some((_,_,batch)) = dataloader.load()? {
            for packed in batch.into_iter() {
                if current_threads >= eval_test_max_threads {
                    self.process_result(&mut current_threads,
                                        &mut count,
                                        &mut successed,
                                        &mut estimated_win,
                                        &mut win,
                                        &mut mae_acc,
                                        lambda,
                                        &sr)?;
                    if count >= EVAL_TEST_SAMPLES {
                        break 'outer;
                    }
                } else {
                    let this = Arc::clone(&self);
                    let test_process = Arc::clone(&test_process);

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
                                &mut successed,
                                &mut estimated_win,
                                &mut win,
                                &mut mae_acc,
                                lambda,
                                &sr)?;
            if count >= EVAL_TEST_SAMPLES {
                break 'outer;
            }
        }

        println!("勝ち {}% (勝ちと評価された局面の割合 {}%)", win as f32 / count as f32 * 100., estimated_win as f32 / count as f32 * 100.);
        println!("負け {}% (負けと評価された局面の割合 {}%)", (count - win) as f32 / count as f32 * 100.,
                 (count - estimated_win) as f32 / count as f32 * 100.);
        println!("正解率(勝敗) {}%", successed as f32 / count as f32 * 100.);
        println!("MAE {}", mae_acc / count as f32);
        println!("{}件のテストサンプルを利用しました。",count);

        Ok(())
    }

    pub fn test_by_packed_sfens(&self,
                                packed_sfen:Vec<u8>)
                                -> Result<Option<(GameEndState,f32,i16)>,ApplicationError> {
        let mut packed_sfen_reader = PackedSfenReader::new();

        let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
            value: score,
            best_move: _,
            end_ply: _,
            game_result
        }) = packed_sfen_reader.read_sfen_with_extended(packed_sfen)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
            InputCreator::make_input(teban, &state, &mc),
            InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.nn.forward_all(input)?;

        Ok(Some((game_result,r[0],score)))
    }

    pub fn test_by_packed_hcpe(&self,
                               hcpe:Vec<u8>)
                               -> Result<Option<(GameEndState,f32,i16)>,ApplicationError> {
        let mut hcpe_reader = HcpeReader::new();

        let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
            eval: score,
            best_move: _,
            game_result
        }) = hcpe_reader.read_sfen_with_extended(hcpe)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
            InputCreator::make_input(teban, &state, &mc),
            InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.nn.forward_all(input)?;

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

        Ok(Some((s,r[0],score)))
    }
}
pub type LF = CrossEntropy<f32>;

#[cfg(feature = "cuda")]
pub struct Trainer<M,A>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,
          A:CudaAllocator {
    pub nn:M,
    a:PhantomData<A>,
    nn_path:String,
    nnsavedir:String
}
pub struct TrainerCreator {
}
#[cfg(feature = "cuda")]
impl TrainerCreator {
    pub fn create<A: CudaAllocator + MemoryType + 'static>(
        save_dir:String, nn_path:String, config:&Config, options:&Matches, allocator:A
    ) -> Result<Trainer<impl BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,A>, ApplicationError>
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

        let total_steps = if let Some(epoch) = options.opt_str("maxepoch") {
            epoch.parse::<usize>()? - config.warmup_epochs.unwrap_or(0)
        } else {
            1
        };

        let optimizer_builder_feature = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(LinearWarmupLR::new(
                config.warmup_steps.unwrap_or(1000),
                config.learning_rate.unwrap_or(0.001),
                config.start_factor.unwrap_or(0.1)
            ).seq(config.warmup_epochs.unwrap_or(0),
                  CosineAnnealingLR::new(config.learning_rate.unwrap_or(0.001),
                                         total_steps,config.eta_min.unwrap_or(0.00001)))
            ).weight_decay(1e-5);

        let optimizer_builder_middle_large = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(LinearWarmupLR::new(
                config.warmup_steps.unwrap_or(1000),
                config.learning_rate.unwrap_or(0.001),
                config.start_factor.unwrap_or(0.1)
            ).seq(config.warmup_epochs.unwrap_or(0),
                  CosineAnnealingLR::new(config.learning_rate.unwrap_or(0.001),
                                         total_steps,config.eta_min.unwrap_or(0.00001)))
            ).weight_decay(1e-5);

        let optimizer_builder_middle = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(LinearWarmupLR::new(
                config.warmup_steps.unwrap_or(1000),
                config.learning_rate.unwrap_or(0.001),
                config.start_factor.unwrap_or(0.1)
            ).seq(config.warmup_epochs.unwrap_or(0),
                  CosineAnnealingLR::new(config.learning_rate.unwrap_or(0.001),
                                         total_steps,config.eta_min.unwrap_or(0.00001)))
            ).weight_decay(1e-5);

        let optimizer_builder_out = AdamWBuilder::new(&device)
            .lr(config.learning_rate.unwrap_or(0.001))
            .scheduler(LinearWarmupLR::new(
                config.warmup_steps.unwrap_or(1000),
                config.learning_rate.unwrap_or(0.001),
                config.start_factor.unwrap_or(0.1)
            ).seq(config.warmup_epochs.unwrap_or(0),
                  CosineAnnealingLR::new(config.learning_rate.unwrap_or(0.001),
                                         total_steps,config.eta_min.unwrap_or(0.00001)))
            ).weight_decay(1e-5);

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

        {
            let save_dir = Path::new(&save_dir);

            let progress_path = Path::new("progress.bin");

            if save_dir.join(progress_path).exists() {
                let mut p = BinFilePersistence::new(save_dir
                    .join(progress_path)
                )?;

                nn.load_progress(&mut p)?;
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
#[cfg(feature = "cuda")]
impl<M,A> Trainer<M,A>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32,A>,BinFilePersistence,Linear,HalfKP<FEATURES_NUM>,Arr<f32,1>,LF>,
          A: CudaAllocator {
    fn sigmoid(x:f32) -> f32 {
        sigmoid(x)
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

        let tmp_progress_path = Path::new(&self.nnsavedir).join("progress.bin.tmp");

        let mut p = BinFilePersistence::new(tmp_progress_path.as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("学習状態の保存処理時にエラーが発生しました。")
        ))?)?;

        self.nn.save_progress(&mut p)?;

        p.save(&tmp_progress_path)?;

        fs::rename(Path::new(&tmp_progress_path),Path::new(&self.nnsavedir).join("progress.bin").as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("学習状態の保存処理時のパスの処理時にエラーが発生しました。")
        ))?)?;

        Ok(())
    }
}
pub struct InputCreator;

impl InputCreator {
    #[inline]
    pub fn make_input(t:Teban,state:&State,mc:&MochigomaCollections) -> Vec<size_t> {
        let mut inputs = Vec::new();

        let p = Rule::ou_square(t,state);

        assert_ne!(p,-1);
        
        let ou_position = if t == Teban::Sente {
            p
        } else {
            80 - p
        };

        match state.get_banmen() {
            &Banmen(ref kinds) => {
                for y in 0..9 {
                    for x in 0..9 {
                        let kind = kinds[y][x];

                        if kind != KomaKind::Blank {
                            let index = InputCreator::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

                            if index < MOCHIGOMA_END {
                                inputs.push((ou_position as usize * MOCHIGOMA_END + index) as size_t);
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
    pub fn make_diff_input(active_player: Teban, t:Teban,state:&State,mc:&MochigomaCollections,m:LegalMove) -> Result<Vec<(size_t,SignFloat<f32>)>,ApplicationError> {
        let mut inputs = Vec::new();

        let p = Rule::ou_square(t,state);

        match m {
            LegalMove::To(m) if m.src() == p as u32 => {
                Err(ApplicationError::UnsupportedOperationError(String::from(
                    "Calculating the difference in moves when the active player's king moves is not currently supported."
                )))
            },
            LegalMove::To(m) => {
                let ou_position = if t == Teban::Sente {
                    p
                } else {
                    80 - p
                };

                let banmen = state.get_banmen().0;

                let (sx,sy) = m.src().square_to_point();

                let kind = banmen[sy as usize][sx as usize];

                if kind != KomaKind::Blank {
                    let index = InputCreator::input_index_of_banmen(t, kind, sx as u32, sy as u32).unwrap();

                    if index < MOCHIGOMA_END {
                        inputs.push(((ou_position as usize * MOCHIGOMA_END + index) as size_t, SignFloat::minus()));
                    }
                }

                let (dx,dy) = m.dst().square_to_point();

                let kind = if m.is_nari() {
                    kind.to_nari()
                } else {
                    kind
                };

                if kind != KomaKind::Blank {
                    let index = InputCreator::input_index_of_banmen(t, kind, dx as u32, dy as u32).unwrap();

                    if index < MOCHIGOMA_END {
                        inputs.push(((ou_position as usize * MOCHIGOMA_END + index) as size_t, SignFloat::plus()));
                    }
                }

                let kind = banmen[dy as usize][dx as usize];

                if kind != KomaKind::Blank {
                    let index = InputCreator::input_index_of_banmen(t, kind, dx as u32, dy as u32).unwrap();

                    if index < MOCHIGOMA_END {
                        inputs.push(((ou_position as usize * MOCHIGOMA_END + index) as size_t, SignFloat::minus()));
                    }
                }

                if let Some(o) = m.obtained() {
                    let ms = Mochigoma::new();
                    let mg = Mochigoma::new();
                    let (ms,mg) = match mc {
                        &MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
                        &MochigomaCollections::Empty => (&ms,&mg),
                    };

                    let mc = match active_player {
                        Teban::Sente => ms,
                        Teban::Gote => mg,
                    };

                    if let Ok(k) = MochigomaKind::try_from(o) {
                        let c = mc.get(k);

                        let s = ou_position as usize * MOCHIGOMA_END + PIECE_END;

                        if t == active_player {
                            inputs.push((s + SELF_INDEX_MAP[k as usize] + c, SignFloat::plus()));
                        } else {
                            inputs.push((s + OPPONENT_INDEX_MAP[k as usize] + c, SignFloat::plus()));
                        }
                    }
                }

                Ok(inputs)
            },
            LegalMove::Put(m) => {
                let ou_position = if t == Teban::Sente {
                    p
                } else {
                    80 - p
                };

                let kind = m.kind();

                match KomaKind::try_from((active_player,kind)) {
                    Ok(k) => {
                        let (dx,dy) = m.dst().square_to_point();

                        let index = InputCreator::input_index_of_banmen(t, k, dx as u32, dy as u32).unwrap();

                        if index < MOCHIGOMA_END {
                            inputs.push(((ou_position as usize * MOCHIGOMA_END + index) as size_t, SignFloat::plus()));
                        }
                    },
                    _ => ()
                }

                let ms = Mochigoma::new();
                let mg = Mochigoma::new();

                let (ms,mg) = match mc {
                    &MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
                    &MochigomaCollections::Empty => (&ms,&mg),
                };

                let mc = match active_player {
                    Teban::Sente => ms,
                    Teban::Gote => mg,
                };

                let s = ou_position as usize * MOCHIGOMA_END + PIECE_END;

                let c = mc.get(kind);

                if c > 0 {
                    if t == active_player {
                        inputs.push((s + SELF_INDEX_MAP[kind as usize] + c - 1, SignFloat::minus()));
                    } else {
                        inputs.push((s + OPPONENT_INDEX_MAP[kind as usize] + c - 1, SignFloat::minus()));
                    }
                }

                Ok(inputs)
            }
        }
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

        Ok(index)
    }
}