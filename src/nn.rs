use std::cell::RefCell;
use std::ops::DerefMut;
use std::path::{Path};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::fs;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid};
use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::device::{Device, DeviceCpu, DeviceGpu};
use nncombinator::layer::{AddLayer, AddLayerTrain, BatchForwardBase, BatchTrain, ForwardAll, PreTrain, TryAddLayer};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::batchnormalization::BatchNormalizationLayerBuilder;
use nncombinator::lossfunction::{CrossEntropy};
use nncombinator::ope::UnitValue;
use nncombinator::optimizer::{SGD};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence, PersistenceType, SaveToFile};
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::traits::Reader;
use packedsfen::{hcpe, yaneuraou};
use packedsfen::hcpe::haffman_code::GameResult;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use usiagent::event::{EventQueue, GameEndState, UserEvent, UserEventKind};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MOCHIGOMA_KINDS, MochigomaCollections, Teban};
use crate::error::{ApplicationError};

const BANMEN_SIZE:usize = 81;

const OU_INDEX:usize = 0;
const FU_INDEX:usize = OU_INDEX + BANMEN_SIZE;
const KYOU_INDEX:usize = FU_INDEX + BANMEN_SIZE;
const KEI_INDEX:usize = KYOU_INDEX + BANMEN_SIZE;
const GIN_INDEX:usize = KEI_INDEX + BANMEN_SIZE;
const KIN_INDEX:usize = GIN_INDEX + BANMEN_SIZE;
const KAKU_INDEX:usize = KIN_INDEX + BANMEN_SIZE;
const HISHA_INDEX:usize = KAKU_INDEX + BANMEN_SIZE;
const NARIFU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
const NARIKYOU_INDEX:usize = NARIFU_INDEX + BANMEN_SIZE;
const NARIKEI_INDEX:usize = NARIKYOU_INDEX + BANMEN_SIZE;
const NARIGIN_INDEX:usize = NARIKEI_INDEX + BANMEN_SIZE;
const NARIKAKU_INDEX:usize = NARIGIN_INDEX + BANMEN_SIZE;
const NARIHISHA_INDEX:usize = NARIKAKU_INDEX + BANMEN_SIZE;
const OPPONENT_FU_INDEX:usize = NARIHISHA_INDEX + BANMEN_SIZE;
const OPPONENT_KYOU_INDEX:usize = OPPONENT_FU_INDEX + BANMEN_SIZE;
const OPPONENT_KEI_INDEX:usize = OPPONENT_KYOU_INDEX + BANMEN_SIZE;
const OPPONENT_GIN_INDEX:usize = OPPONENT_KEI_INDEX + BANMEN_SIZE;
const OPPONENT_KIN_INDEX:usize = OPPONENT_GIN_INDEX + BANMEN_SIZE;
const OPPONENT_KAKU_INDEX:usize = OPPONENT_KIN_INDEX + BANMEN_SIZE;
const OPPONENT_HISHA_INDEX:usize = OPPONENT_KAKU_INDEX + BANMEN_SIZE;
const OPPONENT_OU_INDEX:usize = OPPONENT_HISHA_INDEX + BANMEN_SIZE;
const OPPONENT_NARIFU_INDEX:usize = OPPONENT_OU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKYOU_INDEX:usize = OPPONENT_NARIFU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKEI_INDEX:usize = OPPONENT_NARIKYOU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIGIN_INDEX:usize = OPPONENT_NARIKEI_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKAKU_INDEX:usize = OPPONENT_NARIGIN_INDEX + BANMEN_SIZE;
const OPPONENT_NARIHISHA_INDEX:usize = OPPONENT_NARIKAKU_INDEX + BANMEN_SIZE;

const MOCHIGOMA_FU_INDEX:usize = OPPONENT_NARIHISHA_INDEX + BANMEN_SIZE;
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

pub trait BatchNeuralNetwork<U,D,P,PT,I,O>: ForwardAll<Input=I,Output=O> +
                                 BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,O>> +
                                 BatchTrain<U,D> + Persistence<U,P,PT>
                                 where U: UnitValue<U>,
                                       D: Device<U>,
                                       PT: PersistenceType {}
impl<T,U,D,P,PT,I,O> BatchNeuralNetwork<U,D,P,PT,I,O> for T
    where T: ForwardAll<Input=I,Output=O> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,O>> +
             BatchTrain<U,D> + Persistence<U,P,PT>,
             U: UnitValue<U>,
             D: Device<U>,
             PT: PersistenceType {}
pub struct EvalutorCreator;
impl EvalutorCreator {
    pub fn create<P: AsRef<Path>>(savedir: P, nn_path: P)
        -> Result<Evalutor<impl ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                PreTrain<f32, OutStack=impl Send + Sync + 'static> + Send + Sync + 'static>, ApplicationError> {
        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32 / 2515f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32 / 256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32 / 32f32.sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let net: InputLayer<f32, Arr<f32, 2515>, _> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nn = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2515, 256>::new().build(l, &device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256, 32>::new().build(l, &device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32, 1>::new().build(l, &device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l, Sigmoid::new(&device), &device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l, &device)
        });

        if savedir.as_ref().join(&nn_path).exists() {
            let mut p = BinFilePersistence::new(savedir.as_ref().join(&nn_path))?;

            nn.load(&mut p)?;
        }

        Ok(Evalutor {
            nn:nn
        })
    }
}
pub struct Evalutor<M> where M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                                PreTrain<f32>,
                                <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    nn:M
}
impl<M> Evalutor<M> where M: ForwardAll<Input=Arr<f32, 2515>, Output=Arr<f32, 1>> +
                             PreTrain<f32> + Send + Sync + 'static,
                             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn evalute(&self, t:Teban, b:&Banmen, mc:&MochigomaCollections) -> Result<i32,ApplicationError> {
        let input = InputCreator::make_input(t,b,mc);

        let r = self.nn.forward_all(input)?;

        Ok(((r[0] - 0.5) * (1 << 20) as f32) as i32)
    }
}
pub struct Trainer<M>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2515>,Arr<f32,1>> {

    nn:M,
    optimizer:SGD<f32>,
    nn_path:String,
    nnsavedir:String,
    packed_sfen_reader:PackedSfenReader,
    hcpe_reader:HcpeReader
}
pub struct TrainerCreator {
}
impl TrainerCreator {
    pub fn create(save_dir:String, nn_path:String)
        -> Result<Trainer<impl BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2515>,Arr<f32,1>>>,ApplicationError> {

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32/2515f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

        let memory_pool = Arc::new(Mutex::new(MemoryPool::new(Alloctype::Device)?));

        let device = DeviceGpu::new(&memory_pool)?;

        let net:InputLayer<f32,Arr<f32,2515>,_> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nn = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2515,256>::new().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256,32>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32,1>::new().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,Sigmoid::new(&device),&device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l,&device)
        });

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
            optimizer:SGD::new(0.005),
            nn_path: nn_path,
            nnsavedir: save_dir,
            packed_sfen_reader:PackedSfenReader::new(),
            hcpe_reader:HcpeReader::new()
        })
    }
}
impl<M> Trainer<M> where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2515>,Arr<f32,1>> {
    fn sigmoid(&self,x:i16) -> f32 {
        1. / (1. + (-0.00173873964459554 * x as f32).exp())
    }

    pub fn learning_by_packed_sfens<'a>(&mut self,
                                        packed_sfens:Vec<Vec<u8>>,
                                        _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
                                        -> Result<f32,ApplicationError> {

        let lossf = CrossEntropy::new();

        let mut sfens_with_extended = Vec::with_capacity(packed_sfens.len());

        for entry in packed_sfens.into_iter() {
            let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
                value: score,
                best_move: _,
                end_ply: _,
                game_result
            }) = self.packed_sfen_reader.read_sfen_with_extended(entry)?;

            sfens_with_extended.push((teban,banmen,mc,game_result,score));
        }

        /*
        let (sente_win_count,gote_win_count) = sfens_with_extended.iter()
            .map(|(teban,_,_,es,_)| {
                let (s,g) = match (es,teban) {
                    (&GameEndState::Draw,_) => {
                        (0,0)
                    },
                    (&GameEndState::Win,&Teban::Sente) | (&GameEndState::Lose,&Teban::Gote) => {
                        (1,0)
                    },
                    _ => {
                        (0,1)
                    }
                };

                (s,g)
            }).fold((0,0), |acc,(s,g)| {
                (acc.0 + s, acc.1 + g)
            });

        let (sente_rate,gote_rate) = if sente_win_count >= gote_win_count {
            (gote_win_count as f32 / sente_win_count as f32,1.)
        } else {
            (1.,sente_win_count as f32 / gote_win_count as f32)
        };
        */

        let batch = sfens_with_extended.iter()
            .map(|(teban,banmen,mc,es, score)| {
                let teban = *teban;

                let input = InputCreator::make_input(teban, banmen, mc);

                let mut t = Arr::<f32,1>::new();

                t[0] = {
                    let t = match es {
                        GameEndState::Win if teban == Teban::Sente => {
                            0.99
                            //sente_rate
                        },
                        GameEndState::Win => {
                            0.99
                            //gote_rate
                        },
                        GameEndState::Lose if teban == Teban::Sente => {
                            0.01
                            //-gote_rate
                        },
                        GameEndState::Lose => {
                            0.01
                            //-sente_rate
                        },
                        _ => 0.5f32
                    };

                    t
                    //t * 0.667 + self.sigmoid(*score) * 0.333
                };

                (t,input)
        }).fold((Vec::new(),Vec::new()),  | mut acc, (t,i) | {
            acc.0.push(t);
            acc.1.push(i);

            acc
        });

        let m = self.nn.batch_train(batch.0.into(),batch.1.into(),&mut self.optimizer,&lossf)?;

        Ok(m)
    }

    pub fn test_by_packed_sfens(&mut self,
                                packed_sfen:Vec<u8>)
                                -> Result<(GameEndState,f32),ApplicationError> {
        let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
            value: _,
            best_move: _,
            end_ply: _,
            game_result
        }) = self.packed_sfen_reader.read_sfen_with_extended(packed_sfen)?;

        let input = InputCreator::make_input(teban, &banmen, &mc);

        let r = self.nn.forward_all(input.clone())?;

        Ok((game_result,r[0]))
    }

    pub fn learning_by_hcpe<'a>(&mut self,
                                hcpes:Vec<Vec<u8>>,
                                _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
                                -> Result<f32,ApplicationError> {

        let lossf = CrossEntropy::new();

        let mut sfens_with_extended = Vec::with_capacity(hcpes.len());

        for entry in hcpes.into_iter() {
            let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
                eval: score,
                best_move: _,
                game_result
            }) = self.hcpe_reader.read_sfen_with_extended(entry)?;

            sfens_with_extended.push((teban, banmen, mc, game_result, score));
        }

        /*
        let (sente_win_count,gote_win_count) = sfens_with_extended.iter().map(|(_,_,_,es,_)| {
            match es {
                GameResult::Draw => (0,0),
                GameResult::SenteWin => (1,0),
                _ => (0,1)
            }
        }).fold((0,0), |acc,(s,g)| {
            (acc.0 + s, acc.1 + g)
        });

        let (sente_rate,gote_rate) = if sente_win_count >= gote_win_count {
            (gote_win_count as f32 / sente_win_count as f32,1.)
        } else {
            (1.,sente_win_count as f32 / gote_win_count as f32)
        };
        */
        let batch = sfens_with_extended.iter()
            .map(|(teban,banmen,mc,es,score)| {
                let teban = *teban;

                let input = InputCreator::make_input(teban, banmen, mc);

                /*
                let (rate,es) = match (es,teban) {
                    (GameResult::Draw,_) => {
                        (1.,GameEndState::Draw)
                    },
                    (GameResult::SenteWin,Teban::Sente) => {
                        (sente_rate,GameEndState::Win)
                    },
                    (GameResult::GoteWin,Teban::Gote) => {
                        (gote_rate,GameEndState::Win)
                    },
                    (GameResult::SenteWin,Teban::Gote) => {
                        (sente_rate,GameEndState::Lose)
                    },
                    (GameResult::GoteWin,Teban::Sente) => {
                        (gote_rate,GameEndState::Lose)
                    }
                };
                */
                let es = match (es,teban) {
                    (GameResult::Draw,_) => {
                        GameEndState::Draw
                    },
                    (GameResult::SenteWin,Teban::Sente) => {
                        GameEndState::Win
                    },
                    (GameResult::GoteWin,Teban::Gote) => {
                        GameEndState::Win
                    },
                    (GameResult::SenteWin,Teban::Gote) => {
                        GameEndState::Lose
                    },
                    (GameResult::GoteWin,Teban::Sente) => {
                        GameEndState::Lose
                    }
                };
                let mut t = Arr::<f32,1>::new();

                t[0] = {
                    let t = match es {
                        GameEndState::Win => {
                            0.99
                            //rate
                        }
                        GameEndState::Lose => {
                            //-rate
                            0.01
                        },
                        _ => 0.5f32
                    };

                    t
                    //t * 0.667 + self.sigmoid(*score) * 0.333
                };

                (t,input)
            }).fold((Vec::new(),Vec::new()), | mut acc, (t,i) | {
                acc.0.push(t);
                acc.1.push(i);
                acc
            });

        let m = self.nn.batch_train(batch.0.into(),batch.1.into(),&mut self.optimizer,&lossf)?;

        Ok(m)
    }

    pub fn test_by_packed_hcpe(&mut self,
                               hcpe:Vec<u8>)
                               -> Result<(GameEndState,f32),ApplicationError> {
        let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
            eval: _,
            best_move: _,
            game_result
        }) = self.hcpe_reader.read_sfen_with_extended(hcpe)?;

        let input = InputCreator::make_input(teban, &banmen, &mc);

        let r = self.nn.forward_all(input.clone())?;

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

        Ok((s,r[0]))
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
    pub fn make_input(t:Teban,b:&Banmen,mc:&MochigomaCollections) -> Arr<f32,2515> {
        let mut inputs = Arr::new();

        match b {
            &Banmen(ref kinds) => {
                for y in 0..9 {
                    for x in 0..9 {
                        let kind = kinds[y][x];

                        if kind != KomaKind::Blank {
                            let index = InputCreator::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

                            inputs[index] = 1f32;
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

        for &k in &MOCHIGOMA_KINDS {
            let c = ms.get(k);

            for i in 0..c {
                let offset = SELF_INDEX_MAP[k as usize];

                let offset = offset as usize;

                inputs[offset + i as usize] = 1f32;
            }

            let c = mg.get(k);

            for i in 0..c {
                let offset = OPPONENT_INDEX_MAP[k as usize];

                let offset = offset as usize;

                inputs[offset + i as usize] = 1f32;
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
            OU_INDEX,
            NARIFU_INDEX,
            NARIKYOU_INDEX,
            NARIKEI_INDEX,
            NARIGIN_INDEX,
            NARIKAKU_INDEX,
            NARIHISHA_INDEX,
            OPPONENT_FU_INDEX,
            OPPONENT_KYOU_INDEX,
            OPPONENT_KEI_INDEX,
            OPPONENT_GIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KAKU_INDEX,
            OPPONENT_HISHA_INDEX,
            OPPONENT_OU_INDEX,
            OPPONENT_NARIFU_INDEX,
            OPPONENT_NARIKYOU_INDEX,
            OPPONENT_NARIKEI_INDEX,
            OPPONENT_NARIGIN_INDEX,
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
            OPPONENT_OU_INDEX,
            OPPONENT_NARIFU_INDEX,
            OPPONENT_NARIKYOU_INDEX,
            OPPONENT_NARIKEI_INDEX,
            OPPONENT_NARIGIN_INDEX,
            OPPONENT_NARIKAKU_INDEX,
            OPPONENT_NARIHISHA_INDEX,
            FU_INDEX,
            KYOU_INDEX,
            KEI_INDEX,
            GIN_INDEX,
            KIN_INDEX,
            KAKU_INDEX,
            HISHA_INDEX,
            OU_INDEX,
            NARIFU_INDEX,
            NARIKYOU_INDEX,
            NARIKEI_INDEX,
            NARIGIN_INDEX,
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
                SENTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
            },
            Teban::Gote => {
                let (x,y) = (8-x,8-y);

                GOTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
            }
        };

        Ok(index as usize)
    }
}