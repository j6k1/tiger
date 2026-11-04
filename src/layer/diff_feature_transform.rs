//! Feature Transformation Layer Implementation
use std::fmt::{Debug};
use std::marker::PhantomData;

use nncombinator::arr::{Arr, Arr2};
use nncombinator::{Cons, Stack};
use nncombinator::device::{Device, DeviceSimd};
use nncombinator::error::{EvaluateError, LayerInstantiationError, TrainingError, ModelLoadError, PersistenceError};
use nncombinator::layer::{BackwardAll, BackwardBase, BatchDataType, BatchSize, Bridge, BridgeRepr, ContinueForward, Forward, ForwardAll, ForwardDiff, OnStep, OutputTensorScalar, PartialForward, PreTrain, PreTrainBase, UpdateWeight};
use nncombinator::layer::{BridgeBase};
use nncombinator::mapper::{IdentityMapper};
use nncombinator::optimizer::{Optimizer, OptimizerBuilder};
use nncombinator::persistence::{Linear, LinearPersistence, Persistence};
use nncombinator::ope::UnitValue;
use simdoperator::backend::common::Backend;
use crate::device::{DeviceDiffFeatureTransform, DeviceFeatureTransformEvaluation, DeviceFeatureTransformEvaluationScaling};
use crate::features::HalfKP;

pub struct DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static,
          OP: Optimizer<U,D> {
    parent:P,
    device:D,
    units:C,
    bias:B,
    unit_optimizer:OP,
    bias_optimizer:OP,
    i:PhantomData<I>,
    di:PhantomData<DI>,
    u:PhantomData<U>
}

impl<BE,U,P,I,DI,OP,const NI:usize,const NO:usize> DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>> {
    #[inline]
    pub fn new<OB: OptimizerBuilder<U,DeviceSimd<BE>,Output=OP>>(parent:P,device:&DeviceSimd<BE>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b:&OB)
        -> Result<DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>, LayerInstantiationError> {

        let mut units = Vec::with_capacity(NI*NO);
        let mut bias = Vec::with_capacity(NO);

        units.resize_with(NI*NO,ui);
        bias.resize_with(NO,bi);

        Ok(DiffFeatureTransformLayer {
            parent: parent,
            device: device.clone(),
            units: units.try_into()?,
            bias: bias.try_into()?,
            unit_optimizer: b.build(NI*NO)?,
            bias_optimizer: b.build(NO)?,
            i:PhantomData::<I>,
            di:PhantomData::<DI>,
            u:PhantomData::<U>
        })
    }
}
impl<T,BE,U,P,I,DI,OP,const NI:usize,const NO:usize> Persistence<T,Linear> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + Persistence<T,Linear> +
             BackwardBase<LossInputScalar=(), LossInput=()> + PreTrain,
          U: UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>> + 'static {
    fn load(&mut self, persistence: &mut T) -> Result<(),ModelLoadError> {
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
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> Forward<HalfKP<NI>,Result<<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,EvaluateError>>
    for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO*2]: {
    #[inline]
    fn forward(&self, input:&HalfKP<NI>) -> Result<<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,EvaluateError> {
        let r = self.device.forward_feature_transform(&self.bias,&self.units,input.into())?;

        Ok(self.device.forward_feature_transform_scaling(&r)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ForwardAll for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type Input = I;
    type Output = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;

    #[inline]
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> PartialForward for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PartialForward<DiffInput=DI> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          for<'a> D: Device<U> +
                     DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
                     DeviceDiffFeatureTransform<U,DI,C,B,NI,NO> + 'static,
          [(); NO * 2]: {
    type PartialInput = <D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output;
    type PartialOutput = <D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output;
    type DiffInput = DI;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        let r = self.parent.forward_all(input)?;

        Ok(self.device.forward_feature_transform(&self.bias,&self.units,(&r).into())?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_inpput: &Self::PartialInput) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.device.forward_diff_feature_transform(&self.bias,&self.units,&input,partial_inpput)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ForwardDiff for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PartialForward<DiffInput=DI> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          for<'a> D: Device<U> +
                     DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
                     DeviceDiffFeatureTransform<U,DI,C,B,NI,NO> + 'static,
          Self: ForwardAll<Input=<P as ForwardAll>::Output,Output=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output> + PreTrain,
          [(); NO * 2]: {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.forward_diff_feature_transform(&self.bias,&self.units,&input,partial_input)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ContinueForward for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PartialForward<DiffInput=DI> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain,
      C: 'static,
      B: 'static,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      I: Debug + Send + Sync,
      DI: Debug,
      OP: Optimizer<U,D>,
      for<'a> D: Device<U> +
                 DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
                 DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
                 DeviceDiffFeatureTransform<U,DI,C,B,NI,NO> + 'static,
      Self: PartialForward<PartialInput=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output,PartialOutput=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output>,
      <D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output: Clone,
      Self: ForwardAll<Input=<P as ForwardAll>::Output,Output=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output>,
      [(); NO * 2]: {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(input.clone())
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> PreTrainBase for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type PreOutput = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
    type OutStack = Cons<<P as PreTrainBase>::OutStack,<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput>;
}

impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> PreTrain for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r.into()))?;

        Ok(Cons(r,u))
    }
}
impl<BE,U,P,I,DI,OP,const NI:usize,const NO:usize> BackwardBase for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + BackwardBase<LossInputScalar=(), LossInput=()> + 'static,
          U: UnitValue<U>,
          DeviceSimd<BE>: Device<U> +
                     DeviceFeatureTransformEvaluation<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type LossInputScalar = U;
    type LossInput = <DeviceSimd<BE> as DeviceFeatureTransformEvaluationScaling<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::ScaledOutput;
}
impl<BE,U,P,I,DI,OP,const NI:usize,const NO:usize> BackwardAll<U> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          P: PreTrainBase<PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardBase<LossInputScalar=(), LossInput=()> + BackwardAll<()> + PreTrain + 'static,
          U: UnitValue<U>,
          DeviceSimd<BE>: Device<U> +
                     DeviceFeatureTransformEvaluation<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type LossOutput = ();

    #[inline]
    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<((),<Self as UpdateWeight>::GradientStack), TrainingError> {
        unimplemented!()
    }
}
impl<BE,U,P,I,DI,OP,const NI:usize,const NO:usize> UpdateWeight for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + UpdateWeight + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceSimd<BE>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceSimd<BE>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    #[inline]
    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        unimplemented!()
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> OnStep for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + OnStep,
          C: 'static,
          B: 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          D: Device<U> + 'static,
          OP: Optimizer<U,D> {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;

        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_frequently_step(step,frequently_step)?;
        self.bias_optimizer.on_frequently_step(step,frequently_step)?;

        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> OutputTensorScalar for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static,
          OP: Optimizer<U,D> {
    type Scalar = U;
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> BridgeBase
    for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug + 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>,
          C: 'static,
          B: 'static,
          OP: Optimizer<U,D> + 'static,
          <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput: Debug + BatchDataType + 'static,
          <<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          [(); NO * 2]: {
    type UseDevice = D;
    type RealScale = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
    type RealOutput = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
    type SourceInput = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> Bridge
    for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrainBase<PreOutput=HalfKP<NI>> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>,
          OP: Optimizer<U,D> + 'static,
          <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput: Debug + BatchDataType + 'static,
          <<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          [(); NO * 2]: {
    type RealMapper<'a> = IdentityMapper<'a,<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,Self::UseDevice>
        where Self: 'a;

    fn as_real<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::RealMapper<'a>, EvaluateError>
        where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> BridgeRepr
    for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug + 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>,
          C: 'static,
          B: 'static,
          OP: Optimizer<U,D> + 'static,
          <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput: Debug + BatchDataType + 'static,
          <<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          [(); NO * 2]: {
    type RepresentationOutput = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
    type ReprMapper<'a> = IdentityMapper<'a,<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,Self::UseDevice>
        where Self: 'a;

    fn as_repr<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::ReprMapper<'a>, EvaluateError>
        where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
pub trait DiffFeatureTransformLayerInstantiation<U,P,I,DI,C,BC,D,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,D>,
          D: Device<U>,
          [(); NO * 2]: {
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
                                                         -> Result<DiffFeatureTransformLayer<U,P,I,DI,C,BC,D,OP,NI,NO>,LayerInstantiationError>;
}

impl<BE,U,P,I,DI,OP,const NI:usize,const NO:usize> DiffFeatureTransformLayerInstantiation<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>
    where BE: Backend + Clone,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardBase<LossInputScalar=(), LossInput=()> +
             PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceSimd<BE>>,
          [(); NO * 2]:  {
    fn instantiation<B: OptimizerBuilder<U,DeviceSimd<BE>,Output=OP>>(parent: P, device:&DeviceSimd<BE>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                                    -> Result<DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,OP,NI,NO>,LayerInstantiationError> {
        DiffFeatureTransformLayer::<_,_,_,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceSimd<BE>,_,NI,NO>::new(parent, device, ui, bi, b)
    }
}

pub struct DiffFeatureTransformLayerBuilder<const NI:usize,const NO:usize> {
}

impl<const NI:usize,const NO:usize> DiffFeatureTransformLayerBuilder<NI,NO> {
    /// Create an instance of DiffFeatureTransformLayerBuilder
    pub fn new() -> DiffFeatureTransformLayerBuilder<NI,NO> {
        DiffFeatureTransformLayerBuilder {
        }
    }
}
impl<const NI:usize,const NO:usize> DiffFeatureTransformLayerBuilder<NI,NO> {
    /// Create an instance of DiffFeatureTransformLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,B,P,D,I,DI,OP,OB>(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b:&OB)
        -> Result<DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=HalfKP<NI>> + PartialForward<DiffInput=DI> +
                 BackwardBase<LossInputScalar=(), LossInput=()> +
                 PreTrainBase<PreOutput=HalfKP<NI>> + PreTrain,
              U: Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync + BatchDataType,
              DI: Debug,
              D: Device<U> + 'static,
              OP: Optimizer<U,D> + 'static,
              OB: OptimizerBuilder<U,D,Output=OP>,
              <I as BatchDataType>::Type: Debug + BatchSize,
              DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>: DiffFeatureTransformLayerInstantiation<U,P,I,DI,C,B,D,OP,NI,NO>,
              [(); NO * 2]: {
        Ok(DiffFeatureTransformLayer::<U,P,I,DI,C,B,D,OP,NI,NO>::instantiation(parent, device, ui, bi, b)?)
    }
}
