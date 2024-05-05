use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{DerefMut};
use std::path::{Path};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::fs;
use std::marker::PhantomData;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution};
use rand_distr::{Normal};
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid};
use nncombinator::arr::{Arr, Arr2, SerializedVec};
use nncombinator::{Cons, Stack};
use nncombinator::cuda::{CudaTensor1dPtr, CudaTensor2dPtr, Memory};
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use nncombinator::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use nncombinator::layer::{AddLayer, AddLayerTrain, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, BatchTrain, Forward, ForwardAll, Loss, PreTrain, TryAddLayer, UpdateWeight};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::lossfunction::{CrossEntropy, LossFunction};
use nncombinator::mem::AsRawSlice;
use nncombinator::ope::UnitValue;
use nncombinator::optimizer::{MomentumSGDBuilder, Optimizer, OptimizerBuilder};
use nncombinator::persistence::{BinFilePersistence, Linear, LinearPersistence, Persistence, PersistenceType, SaveToFile};
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::traits::Reader;
use packedsfen::{hcpe, yaneuraou};
use packedsfen::hcpe::haffman_code::GameResult;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use usiagent::event::{EventQueue, GameEndState, UserEvent, UserEventKind};
use usiagent::math::Prng;
use usiagent::movepick::RandomPicker;
use usiagent::rule::{LegalMove, NonEvasionsAll, Rule, SquareToPoint, State};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MOCHIGOMA_KINDS, MochigomaCollections, Teban};
use crate::device::DeviceFeatureTransform;
use crate::error::{ApplicationError};
use crate::features::HalfKP;

const BANMEN_SIZE:usize = 81;

const FU_INDEX:usize = 0;
const KYOU_INDEX:usize = FU_INDEX + BANMEN_SIZE;
const KEI_INDEX:usize = KYOU_INDEX + BANMEN_SIZE;
const GIN_INDEX:usize = KEI_INDEX + BANMEN_SIZE;
const KIN_INDEX:usize = GIN_INDEX + BANMEN_SIZE;
const KAKU_INDEX:usize = KIN_INDEX + BANMEN_SIZE;
const HISHA_INDEX:usize = KAKU_INDEX + BANMEN_SIZE;
//const NARIFU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
//const NARIKYOU_INDEX:usize = NARIFU_INDEX + BANMEN_SIZE;
//const NARIKEI_INDEX:usize = NARIKYOU_INDEX + BANMEN_SIZE;
//const NARIGIN_INDEX:usize = NARIKEI_INDEX + BANMEN_SIZE;
//const NARIKAKU_INDEX:usize = NARIGIN_INDEX + BANMEN_SIZE;
const NARIKAKU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
const NARIHISHA_INDEX:usize = NARIKAKU_INDEX + BANMEN_SIZE;
const OPPONENT_FU_INDEX:usize = NARIHISHA_INDEX + BANMEN_SIZE;
const OPPONENT_KYOU_INDEX:usize = OPPONENT_FU_INDEX + BANMEN_SIZE;
const OPPONENT_KEI_INDEX:usize = OPPONENT_KYOU_INDEX + BANMEN_SIZE;
const OPPONENT_GIN_INDEX:usize = OPPONENT_KEI_INDEX + BANMEN_SIZE;
const OPPONENT_KIN_INDEX:usize = OPPONENT_GIN_INDEX + BANMEN_SIZE;
const OPPONENT_KAKU_INDEX:usize = OPPONENT_KIN_INDEX + BANMEN_SIZE;
const OPPONENT_HISHA_INDEX:usize = OPPONENT_KAKU_INDEX + BANMEN_SIZE;
//const OPPONENT_NARIFU_INDEX:usize = OPPONENT_HISHA_INDEX + BANMEN_SIZE;
//const OPPONENT_NARIKYOU_INDEX:usize = OPPONENT_NARIFU_INDEX + BANMEN_SIZE;
//const OPPONENT_NARIKEI_INDEX:usize = OPPONENT_NARIKYOU_INDEX + BANMEN_SIZE;
//const OPPONENT_NARIGIN_INDEX:usize = OPPONENT_NARIKEI_INDEX + BANMEN_SIZE;
//const OPPONENT_NARIKAKU_INDEX:usize = OPPONENT_NARIGIN_INDEX + BANMEN_SIZE;
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
pub struct FeatureTransformLayer<U,P,I,C,B,D,OP,const NI:usize,const NO:usize>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<U,NI>>,
          D: Device<U> + 'static,
          OP: Optimizer<U,D> {
    parent:P,
    device:D,
    units:C,
    bias:B,
    unit_optimizer:OP,
    bias_optimizer:OP,
    i:PhantomData<I>
}
impl<U,P,I,OP,const NI:usize,const NO:usize> FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             PreTrain<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    #[inline]
    pub fn new<OB: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent:P,device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b:&OB)
        -> Result<FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>, LayerInstantiationError> {

        let mut units = Vec::with_capacity(NI*NO);
        let mut bias = Vec::with_capacity(NO);

        units.resize_with(NI*NO,ui);
        bias.resize_with(NO,bi);

        Ok(FeatureTransformLayer {
            parent: parent,
            device: device.clone(),
            units: units.try_into()?,
            bias: bias.try_into()?,
            unit_optimizer: b.build(NI*NO)?,
            bias_optimizer: b.build(NO)?,
            i:PhantomData::<I>
        })
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<U,NI>> +
          PreTrain<U>,
          DeviceGpu<U>: Device<U> + 'static,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>> {
    #[inline]
    pub fn new<UI,BI,OB: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent:P,device:&DeviceGpu<U>,ui: UI, bi: BI, b:&OB)
        -> Result<FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>, LayerInstantiationError>
        where U: UnitValue<U>,
              UI: FnMut() -> U,
              BI: FnMut() -> U {
        let memory_pool = device.get_memory_pool();

        Ok(FeatureTransformLayer {
            parent: parent,
            device: device.clone(),
            units: CudaTensor2dPtr::with_initializer(memory_pool,ui)?,
            bias: CudaTensor1dPtr::with_initializer(memory_pool,bi)?,
            unit_optimizer: b.build(NI*NO)?,
            bias_optimizer: b.build(NO)?,
            i:PhantomData::<I>
        })
    }
}
impl<T,U,P,I,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> + 'static {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + PreTrain<U> + Loss<U>,
          DeviceGpu<U>: Device<U>,
          U: UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>> + 'static {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<U,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),NO)?;
        self.units.memcpy(units.as_raw_slice().as_ptr(),NI*NO)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = Arr::<U,NO>::try_from(self.bias.read_to_vec()?)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        let units = Arr2::<U,NI,NO>::try_from(self.units.read_to_vec()?)?;

        for u in units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> Forward<HalfKP<U,NI>,Result<Arr<U,{NO * 2}>,EvaluateError>>
    for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             PreTrain<U> + Loss<U>,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,D> + 'static {
    #[inline]
    fn forward(&self, input:&HalfKP<U,NI>) -> Result<Arr<U,{NO * 2}>,EvaluateError> {
        self.device.forward_feature_transform(&self.bias,&self.units,input.into())
    }
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> ForwardAll for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             PreTrain<U> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type Input = I;
    type Output = Arr<U,{NO*2}>;

    #[inline]
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> PreTrain<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U> +
             ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + PreTrain<U> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Arr<U,{NO * 2}>>;

    #[inline]
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r.into()))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,B,P,D,I,OP,const NI:usize,const NO:usize> Backward<U,&Arr<U,{NO * 2}>,Result<HalfKP<U,NI>,TrainingError>>
    for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    fn backward(&mut self, input: &Arr<U,{NO * 2}>) -> Result<HalfKP<U,NI>,TrainingError> {
        self.device.backward_feature_transform(&self.units,input.into())
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BackwardAll<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U> +
             ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> + 'static,
          U: UnitValue<U>,
          DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type LossInput = Arr<U,{NO*2}>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    #[inline]
    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_feature_transform_weight_gradient(o.into(),(&loss).into())
        })?;

        let bg = self.device.backward_feature_transform_bias_gradient((&loss).into())?;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BackwardAll<U>
for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: PreTrain<U> +
             ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceGpu<U>> + 'static,
          DeviceGpu<U>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          [(); NO * 2]: {
    type LossInput = Arr<U,{NO*2}>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    #[inline]
    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
                                        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_feature_transform_weight_gradient(o.into(),(&loss).into())
        })?;

        let bg = self.device.backward_feature_transform_bias_gradient((&loss).into())?;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> UpdateWeight<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + UpdateWeight<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    #[inline]
    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> UpdateWeight<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + UpdateWeight<U> + 'static,
          DeviceGpu<U>: Device<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceGpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>)>;

    #[inline]
    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> Loss<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {}
impl<U,P,I,OP,const NI:usize,const NO:usize> Loss<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> +
             BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> + 'static,
          DeviceGpu<U>: Device<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceGpu<U>> + 'static,
          DeviceGpu<U>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          [(); NO * 2]: {}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchForwardBase for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]: {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = SerializedVec<U,Arr<U,{NO*2}>>;
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchForward for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]: {
    #[inline]
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        self.device.batch_forward_feature_transform(&self.bias,&self.units,(&input).try_into()?)
    }
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchPreTrainBase<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]: {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack, SerializedVec<U,Arr<U,{NO*2}>>>;
}
impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchPreTrain<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]:,
          Self: PreTrain<U> {
    #[inline]
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_forward_feature_transform(&self.bias,&self.units,input.try_into()?)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BatchBackward<U>
    for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
             DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type BatchLossInput = SerializedVec<U,Arr<U,{NO*2}>>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    #[inline]
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_feature_transform(&self.units, (&loss).try_into()?)?;

        let g = s.map(|o| {
            (&loss).try_into().map(|loss| self.device.batch_backward_feature_transform_weight_gradient(o.try_into()?, loss))
        })??;

        let bg = self.device.batch_feature_transform_bias_gradient((&loss).try_into()?)?;

        let (
            s,loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BatchBackward<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          DeviceGpu<U>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          [(); NO * 2]: {
    type BatchLossInput = SerializedVec<U,Arr<U,{NO*2}>>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    #[inline]
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_feature_transform(&self.units, (&loss).try_into()?)?;

        let g = s.map(|o| {
            (&loss).try_into().map(|loss| self.device.batch_backward_feature_transform_weight_gradient(o.try_into()?, loss))
        })??;

        let bg = self.device.batch_feature_transform_bias_gradient((&loss).try_into()?)?;

        let (
            s,loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BatchLoss<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
}
impl<U,P,I,OP,const NI:usize,const NO:usize> BatchLoss<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,HalfKP<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,HalfKP<U,NI>>> + 'static,
          DeviceGpu<U>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          [(); NO * 2]: {
}
/// Trait for LinearLayer instance creation
pub trait FeatureTransformLayerInstantiation<U,P,I,C,BC,D,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          D: Device<U> {
    /// Create an instance of LinedarLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    fn instantiation<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                         -> Result<FeatureTransformLayer<U,P,I,C,BC,D,OP,NI,NO>,LayerInstantiationError>;
}
impl<U,P,I,OP,const NI:usize,const NO:usize> FeatureTransformLayerInstantiation<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                                    -> Result<FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>,LayerInstantiationError> {
        FeatureTransformLayer::<_,_,_,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,_,NI,NO>::new(parent,device,ui,bi,b)
    }
}
pub struct FeatureTransformLayerBuilder<const NI:usize,const NO:usize> {
}
impl<const NI:usize,const NO:usize> FeatureTransformLayerBuilder<NI,NO> {
    /// Create an instance of FeatureTransformLayerBuilder
    pub fn new() -> FeatureTransformLayerBuilder<NI,NO> {
        FeatureTransformLayerBuilder {
        }
    }
}
impl<U,P,I,OP,const NI:usize,const NO:usize> FeatureTransformLayerInstantiation<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn instantiation<B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent: P, device:&DeviceGpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                                    -> Result<FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,OP,NI,NO>,LayerInstantiationError> {
        FeatureTransformLayer::<_,_,_,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,DeviceGpu<U>,_,NI,NO>::new(parent,device,ui,bi,b)
    }
}
impl<const NI:usize,const NO:usize> FeatureTransformLayerBuilder<NI,NO> {
    /// Create an instance of FeatureTransformLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,B,P,D,I,OP,OB>(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b:&OB)
        -> Result<FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=HalfKP<U,NI>> + BackwardAll<U,LossInput=HalfKP<U,NI>> +
                 PreTrain<U> + Loss<U>,
              U: Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync + BatchDataType,
              D: Device<U> + 'static,
              OP: Optimizer<U,D> + 'static,
              OB: OptimizerBuilder<U,D,Output=OP>,
              <I as BatchDataType>::Type: Debug + BatchSize,
              FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>: FeatureTransformLayerInstantiation<U,P,I,C,B,D,OP,NI,NO> {
        Ok(FeatureTransformLayer::<U,P,I,C,B,D,OP,NI,NO>::instantiation(parent,device,ui,bi,b)?)
    }
}
pub trait BatchNeuralNetwork<U,D,P,PT,I,O>: ForwardAll<Input=I,Output=O> +
                                 BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
                                 BatchTrain<U,D> + Persistence<U,P,PT>
                                 where U: UnitValue<U>,
                                       D: Device<U>,
                                       I: BatchDataType,
                                       O: BatchDataType,
                                       PT: PersistenceType {}
impl<T,U,D,P,PT,I,O> BatchNeuralNetwork<U,D,P,PT,I,O> for T
    where T: ForwardAll<Input=I,Output=O> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<O as BatchDataType>::Type> +
             BatchTrain<U,D> + Persistence<U,P,PT>,
             U: UnitValue<U>,
             D: Device<U>,
             I: BatchDataType,
             O: BatchDataType,
             PT: PersistenceType,
             <I as BatchDataType>::Type: Debug + BatchSize {}
pub struct EvalutorCreator {
}
impl EvalutorCreator {
    pub fn create(savedir: impl AsRef<Path> + 'static, nn_path: impl AsRef<Path> + 'static)
        -> Result<Evalutor<impl ForwardAll<Input=HalfKP<f32,FEATURES_NUM>, Output=Arr<f32, 1>> +
                                PreTrain<f32, OutStack=impl Send + Sync + 'static> + Send + Sync + 'static>, ApplicationError> {
        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32 / FEATURES_NUM as f32).sqrt()).unwrap();

        let n2 = Normal::<f32>::new(0.0, (2f32 / 512f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32 / 32f32.sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let optimizer_builder = MomentumSGDBuilder::with_params(&device,0.01,0.9,0.0001);

        let net: InputLayer<f32, HalfKP<f32,FEATURES_NUM>, HalfKP<f32,FEATURES_NUM>> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nn = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            FeatureTransformLayerBuilder::<FEATURES_NUM,256>::new().build(l,&device,
                                                                          move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                                          &optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<{256 * 2}, 32>::new().build(l, &device,
                                                             move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                     move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                        move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
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
pub struct Evalutor<M>
    where M: ForwardAll<Input=HalfKP<f32,FEATURES_NUM>, Output=Arr<f32, 1>> +
             PreTrain<f32> + Send + Sync + 'static,
             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    nn:M
}
impl<M> Evalutor<M>
    where M: ForwardAll<Input=HalfKP<f32,FEATURES_NUM>, Output=Arr<f32, 1>> +
             PreTrain<f32> + Send + Sync + 'static,
             <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    pub fn evalute(&self, t:Teban, state:&State, mc:&MochigomaCollections) -> Result<i32,ApplicationError> {
        let input = HalfKP::new(InputCreator::make_input(t,state,mc),InputCreator::make_input(t.opposite(),state,mc));

        let r = self.nn.forward_all(input)?;

        Ok(((r[0] - 0.5) * (1 << 20) as f32) as i32)
    }
}
pub struct Trainer<M>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,HalfKP<f32,FEATURES_NUM>,Arr<f32,1>> {
    nn:M,
    nn_path:String,
    nnsavedir:String,
    packed_sfen_reader:PackedSfenReader,
    hcpe_reader:HcpeReader
}
pub struct TrainerCreator {
}
impl TrainerCreator {
    pub fn create(save_dir:String, nn_path:String, learning_rate:f32)
        -> Result<Trainer<impl BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,HalfKP<f32,FEATURES_NUM>,Arr<f32,1>>>, ApplicationError> {

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let memory_pool = Arc::new(Mutex::new(MemoryPool::with_size(1024 * 1024 * 1024 * 4,Alloctype::Device)?));

        let n1 = Normal::<f32>::new(0.0, (2f32 / FEATURES_NUM as f32).sqrt()).unwrap();

        let n2 = Normal::<f32>::new(0.0, (2f32 / 512f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32 / 32f32.sqrt()).unwrap();

        let device = DeviceGpu::new(&memory_pool)?;

        let optimizer_builder = MomentumSGDBuilder::with_params(&device,learning_rate,0.9,0.0001);

        let net: InputLayer<f32, HalfKP<f32,FEATURES_NUM>, HalfKP<f32,FEATURES_NUM>> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nn = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            FeatureTransformLayerBuilder::<FEATURES_NUM,256>::new().build(l,&device,
                                                                          move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                                          &optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<{256 * 2}, 32>::new().build(l, &device,
                move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32, 32>::new().build(l, &device,
                move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, ReLu::new(&device), &device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                        move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,&optimizer_builder)
        })?.add_layer(|l| {
            ActivationLayer::new(l, Sigmoid::new(&device), &device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l, &device)
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
            nn_path: nn_path,
            nnsavedir: save_dir,
            packed_sfen_reader:PackedSfenReader::new(),
            hcpe_reader:HcpeReader::new()
        })
    }
}
impl<M> Trainer<M>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,HalfKP<f32,FEATURES_NUM>,Arr<f32,1>> {
    fn sigmoid(&self,x:i16) -> f32 {
        1. / (1. + (-0.00173873964459554 * x as f32).exp())
    }

    pub fn select_bestmove(&self, teban:Teban, state:&State, mc:MochigomaCollections) -> Result<Option<LegalMove>,ApplicationError> {
        let mut rnd = rand::thread_rng();
        let mut picker = RandomPicker::new(Prng::new(rnd.gen()));

        Rule::legal_moves_all_by_strategy::<NonEvasionsAll>(teban,state,&mc,&mut picker)?;

        let mut best_score = None;
        let mut best_move = None;

        for m in &mut picker {
            let next = Rule::apply_move_none_check(state, teban, &mc, m.to_applied_move());

            match next {
                (state, mc, _) => {
                    let input = HalfKP::new(
                                    InputCreator::make_input(teban.opposite(),&state,&mc),
                                    InputCreator::make_input(teban,&state,&mc)
                    );

                    let r = self.nn.forward_all(input)?;
                    let r = r[0].clone() - 0.5;

                    match best_score {
                        None => {
                            best_score = Some(-r);
                            best_move = Some(m);
                        },
                        Some(s) if -r > s => {
                            best_score = Some(-r);
                            best_move = Some(m);
                        },
                        _ => ()
                    }
                }
            }
        }

        Ok(best_move)
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

        let batch = sfens_with_extended.into_iter()
            .map(|(teban,banmen,mc,es, score)| {
                let state = State::new(banmen);

                let input = HalfKP::new(
                    InputCreator::make_input(teban, &state, &mc),
                    InputCreator::make_input(teban.opposite(),&state,&mc)
                );

                let mut t = Arr::<f32,1>::new();

                t[0] = {
                    let t = match es {
                        GameEndState::Win if teban == Teban::Sente => {
                            sente_rate
                        },
                        GameEndState::Win => {
                            gote_rate
                        },
                        GameEndState::Lose if teban == Teban::Sente => {
                            0.5 - 0.5 * gote_rate
                        },
                        GameEndState::Lose => {
                            0.5 - 0.5 * sente_rate
                        },
                        _ => 0.5f32
                    };

                    t * 0.667 + self.sigmoid(score) * 0.333
                };

                (t,input)
        }).fold((Vec::new(),Vec::new()),  | mut acc, (t,i) | {
            acc.0.push(t);
            acc.1.push(i);

            acc
        });

        let m = self.nn.batch_train(batch.0.into(),batch.1.into(),&lossf)?;

        Ok(m)
    }

    pub fn test_by_packed_sfens(&mut self,
                                packed_sfen:Vec<u8>)
                                -> Result<(GameEndState,f32,Option<bool>),ApplicationError> {
        let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
            value: _,
            best_move,
            end_ply: _,
            game_result
        }) = self.packed_sfen_reader.read_sfen_with_extended(packed_sfen)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
                        InputCreator::make_input(teban, &state, &mc),
                        InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.nn.forward_all(input)?;

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
                }).or(Some(false))
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
                }).or(Some(false))
            },
            _ => None
        };

        Ok((game_result,r[0],same))
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

        let batch = sfens_with_extended.into_iter()
            .map(|(teban,banmen,mc,es,score)| {
                let state = State::new(banmen);

                let input = HalfKP::new(
                            InputCreator::make_input(teban, &state, &mc),
                            InputCreator::make_input(teban.opposite(),&state,&mc)
                );

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

                let mut t = Arr::<f32,1>::new();

                t[0] = {
                    let t = match es {
                        GameEndState::Win => {
                            rate
                        }
                        GameEndState::Lose => {
                            0.5 - 0.5 * rate
                        },
                        _ => 0.5f32
                    };

                    t * 0.667 + self.sigmoid(score) * 0.333
                };

                (t,input)
            }).fold((Vec::new(),Vec::new()), | mut acc, (t,i) | {
                acc.0.push(t);
                acc.1.push(i);
                acc
            });

        let m = self.nn.batch_train(batch.0.into(),batch.1.into(),&lossf)?;

        Ok(m)
    }

    pub fn test_by_packed_hcpe(&mut self,
                               hcpe:Vec<u8>)
                               -> Result<(GameEndState,f32,Option<bool>),ApplicationError> {
        let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
            eval: _,
            best_move,
            game_result
        }) = self.hcpe_reader.read_sfen_with_extended(hcpe)?;

        let state = State::new(banmen);

        let input = HalfKP::new(
            InputCreator::make_input(teban, &state, &mc),
            InputCreator::make_input(teban.opposite(),&state,&mc)
        );

        let r = self.nn.forward_all(input)?;

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
                }).or(Some(false))
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
                }).or(Some(false))
            },
            _ => None
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

        Ok((s,r[0],same))
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
    pub fn make_input(t:Teban,state:&State,mc:&MochigomaCollections) -> Arr<f32,FEATURES_NUM> {
        let mut inputs = Arr::new();

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
                                 inputs[ou_position as usize * (MOCHIGOMA_END) + index] = 1f32;
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
            inputs[s + SELF_INDEX_MAP[k as usize] + ms.get(k) as usize] = 1f32;
            inputs[s + OPPONENT_INDEX_MAP[k as usize] + mg.get(k) as usize] = 1f32;
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
//            OU_INDEX,
//            NARIFU_INDEX,
//            NARIKYOU_INDEX,
//            NARIKEI_INDEX,
//            NARIGIN_INDEX,
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
//            OPPONENT_OU_INDEX,
//            OPPONENT_NARIFU_INDEX,
//            OPPONENT_NARIKYOU_INDEX,
//            OPPONENT_NARIKEI_INDEX,
//            OPPONENT_NARIGIN_INDEX,
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
//            OPPONENT_OU_INDEX,
//            OPPONENT_NARIFU_INDEX,
//            OPPONENT_NARIKYOU_INDEX,
//            OPPONENT_NARIKEI_INDEX,
//            OPPONENT_NARIGIN_INDEX,
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
//            OU_INDEX,
//            NARIFU_INDEX,
//            NARIKYOU_INDEX,
//            NARIKEI_INDEX,
//            NARIGIN_INDEX,
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