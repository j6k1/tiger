//! Feature Transformation Layer Implementation
use std::fmt::{Debug};
use std::marker::PhantomData;

use nncombinator::arr::{Arr, Arr2};
use nncombinator::{Cons, Stack};
use nncombinator::device::{Device, DeviceBatchAveraging, DeviceCpu};
use nncombinator::error::{EvaluateError, LayerInstantiationError, TrainingError, ModelLoadError, PersistenceError};
use nncombinator::layer::{BackwardAll, BatchDataType, BatchSize, ContinueForward, Forward, ForwardAll, ForwardDiff, Loss, OnStep, PartialForward, PreTrain, UpdateWeight};
use nncombinator::lossfunction::LossFunction;
use nncombinator::optimizer::{Optimizer, OptimizerBuilder};
use nncombinator::persistence::{Linear, LinearPersistence, Persistence};
use nncombinator::ope::UnitValue;
use nncombinator::device::linear::DeviceLinear;
use nncombinator::layer::linear::LinearLayer;
use crate::device::{DeviceDiffFeatureTransform, DeviceFeatureTransform};
use crate::features::HalfKP;
use crate::layer::feature_transform::FeatureTransformLayer;

pub struct DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
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

impl<U,P,I,DI,OP,const NI:usize,const NO:usize> DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> {
    #[inline]
    pub fn new<OB: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent:P,device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b:&OB)
        -> Result<DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>, LayerInstantiationError> {

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
impl<T,U,P,I,DI,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> + 'static {
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
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> Forward<HalfKP<NI>,Result<<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output,EvaluateError>>
    for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO*2]: {
    #[inline]
    fn forward(&self, input:&HalfKP<NI>) -> Result<<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output,EvaluateError> {
        self.device.forward_feature_transform(&self.bias,&self.units,input.into())
    }
}

impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ForwardAll for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type Input = I;
    type Output = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;

    #[inline]
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> PartialForward for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + PartialForward<DiffOutput=HalfKP<NI>,PartialOutput=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          for<'a> D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> +
                     DeviceDiffFeatureTransform<'a,U,C,B,NI,NO,DiffInput=DI> + 'static,
          Self: ForwardAll<Input=I>,
          Self: PreTrain<U>,
          [(); NO * 2]: {
    type PartialOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;
    type PartialOutputByDiff = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;
    type DiffInput = DI;
    type DiffOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        let input = self.parent.partial_forward(input)?;

        Ok(self.forward(&input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput) -> Result<Self::PartialOutputByDiff, EvaluateError> {
        Ok(self.device.forward_diff_feature_transform(&self.bias,&self.units,&input)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ForwardDiff for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + PartialForward<DiffOutput=HalfKP<NI>> + ForwardDiff +
             BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,D>,
          for<'a> D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> +
                     DeviceDiffFeatureTransform<'a,U,C,B,NI,NO,DiffInput=DI> + 'static,
          Self: ForwardAll<Input=I> + PreTrain<U> +
                PartialForward<DiffInput=DI,DiffOutput=<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output>,
          [(); NO * 2]: {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput, EvaluateError> {
        Ok(self.device.forward_diff_feature_transform(&self.bias,&self.units,&input)?)
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> ContinueForward for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + PartialForward<DiffOutput=HalfKP<NI>> +
             ContinueForward<ConinueOutput=<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output> +
             BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
      C: 'static,
      B: 'static,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      I: Debug + Send + Sync,
      DI: Debug,
      OP: Optimizer<U,D>,
      for<'a> D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> +
                 DeviceDiffFeatureTransform<'a,U,C,B,NI,NO,DiffInput=DI> + 'static,
      Self: PartialForward<PartialOutput=<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output>,
      <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output: Clone,
      [(); NO * 2]: {
    type ConinueOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;
    fn continue_forward(&self, input: &Self::PartialOutput) -> Result<Self::ConinueOutput, EvaluateError> {
        Ok(input.clone())
    }
}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> PreTrain<U> for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,D> + 'static,
          [(); NO * 2]: {
    type PreOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output>;

    #[inline]
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r.into()))?;

        Ok(Cons(r,u))
    }
}

impl<U,P,I,DI,OP,const NI:usize,const NO:usize> BackwardAll<U> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          U: UnitValue<U>,
          DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type LossInput = <DeviceCpu<U> as DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::Output;
    type LossOutput = ();

    #[inline]
    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.backward_feature_transform_weight_gradient(o.into(),&loss)
        })?;

        let bg = self.device.backward_feature_transform_bias_gradient(&loss)?;

        let (_,s) = self.parent.backward_all((), s, lossf)?;

        Ok(((),Cons(s,(g,bg))))
    }
}
impl<U,P,I,DI,OP,const NI:usize,const NO:usize> UpdateWeight<U> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + UpdateWeight<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    #[inline]
    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        let bg = self.device.batch_averaging(bg,batch_size * 2)?;
        let g = self.device.batch_averaging(g,batch_size * 2)?;

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s,batch_size)?)
    }
}
impl<U,P,I,DI,OP,const NI:usize,const NO:usize> Loss<U> for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {}
impl<U,P,I,DI,C,B,D,OP,const NI:usize,const NO:usize> OnStep for DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + OnStep,
          D: Device<U> + 'static,
          OP: Optimizer<U,D> {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;

        Ok(self.parent.on_step(step)?)
    }
}
pub trait DiffFeatureTransformLayerInstantiation<U,P,I,DI,C,BC,D,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
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
                                                         -> Result<DiffFeatureTransformLayer<U,P,I,DI,C,BC,D,OP,NI,NO>,LayerInstantiationError>;
}

impl<U,P,I,DI,OP,const NI:usize,const NO:usize> DiffFeatureTransformLayerInstantiation<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    for DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                                    -> Result<DiffFeatureTransformLayer<U,P,I,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>,LayerInstantiationError> {
        DiffFeatureTransformLayer::<_,_,_,DI,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,_,NI,NO>::new(parent, device, ui, bi, b)
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
                 BackwardAll<U,LossInput=()> +
                 PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
              U: Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync + BatchDataType,
              DI: Debug,
              D: Device<U> + 'static,
              OP: Optimizer<U,D> + 'static,
              OB: OptimizerBuilder<U,D,Output=OP>,
              <I as BatchDataType>::Type: Debug + BatchSize,
              DiffFeatureTransformLayer<U,P,I,DI,C,B,D,OP,NI,NO>: DiffFeatureTransformLayerInstantiation<U,P,I,DI,C,B,D,OP,NI,NO> {
        Ok(DiffFeatureTransformLayer::<U,P,I,DI,C,B,D,OP,NI,NO>::instantiation(parent, device, ui, bi, b)?)
    }
}
