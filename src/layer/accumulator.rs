//! Accumulator Layer Implementation (simplified: removed generic parameters OP, C, B)

use std::fmt::Debug;
use std::str::FromStr;
use std::marker::PhantomData;

use nncombinator::{Cons, Stack};
use nncombinator::device::{Device};
use nncombinator::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use nncombinator::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, ContinueForward, DiffInput, Forward, ForwardAll, ForwardDiff, Loss, OnStep, PartialForward, PreTrain, UpdateWeight};
use nncombinator::lossfunction::LossFunction;
use nncombinator::ope::UnitValue;
use nncombinator::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};
use crate::device::DeviceAccumulator;

/// Trait for AccumulatorLayer instance creation
pub trait AccumulatorLayerInstantiation<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + Clone,
          I: Debug + Send + Sync,
          PI: Debug {
    /// Create and return an instance.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    fn instantiation(parent:P,device:&D) -> Result<AccumulatorLayer<U,P,D,I,PI,N>,LayerInstantiationError>;
}

/// Accumulator Layer Implementation
pub struct AccumulatorLayer<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug {
    parent:P,
    device:D,
    u: PhantomData<U>
}

// Persistence just delegates to parent
impl<U,P,D,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),PersistenceError> {
        self.parent.save(persistence)
    }
}

impl<T,U,P,D,I,PI,const N:usize> Persistence<U,T,Linear> for AccumulatorLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> { self.parent.load(persistence) }
    fn save(&mut self, persistence: &mut T) -> Result<(),PersistenceError> { self.parent.save(persistence) }
}

impl<U,P,D,I,PI,const N:usize> Forward<PI,Result<PI,EvaluateError>> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_accumulator(input)
    }
}

impl<U,P,D,I,PI,const N:usize> ForwardAll for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output,EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}

impl<U,P,D,I,PI,const N:usize> PreTrain<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack,EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}

impl<U,P,D,I,PI,const N:usize> Backward<U,PI,Result<PI,TrainingError>> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn backward(&mut self, input: PI) -> Result<PI,TrainingError> {
        self.device.backward_accumulator(&input)
    }
}

impl<U,P,D,I,PI,const N:usize> BackwardAll<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack),TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss= self.backward(loss)?;

        let (s,next_loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(next_loss, s, lossf)?;

        Ok((l,s))
    }
}

impl<U,P,D,I,PI,const N:usize> UpdateWeight<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          D: Device<U>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type GradientStack = <P as UpdateWeight<U>>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(),TrainingError> {
        self.parent.update_weight(stack)
    }
}

impl<U,P,D,I,PI,const N:usize> PartialForward for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> +
            BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type PartialOutputByDiff = <P as PartialForward>::PartialOutputByDiff;
    type DiffInput = <P as PartialForward>::DiffInput;
    type DiffOutput = PI;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput,EvaluateError> {
        self.parent.partial_forward(input)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput) -> Result<Self::PartialOutputByDiff,EvaluateError> {
        self.parent.partial_forward_by_diff(input)
    }
}

impl<U,P,D,I,PI,const N:usize> ForwardDiff for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> + ForwardDiff +
            BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput,EvaluateError> {
        let input = self.parent.forward_diff(input)?;
        self.forward(&input)
    }
}

impl<U,P,D,I,PI,const N:usize> ContinueForward for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> + ContinueForward<ConinueOutput=PI> +
            BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
    type ConinueOutput = Self::Output;
    fn continue_forward(&self, input: &Self::PartialOutput) -> Result<Self::ConinueOutput,EvaluateError> {
        let input = self.parent.continue_forward(input)?;
        self.forward(&input)
    }
}

impl<U,P,D,I,PI,const N:usize> Loss<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {}

impl<U,P,D,I,PI,const N:usize> BatchForwardBase for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}

impl<U,P,D,I,PI,const N:usize> BatchForward for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        let input = self.parent.batch_forward(input)?;
        self.device.batch_forward_accumulator(&input)
    }
}

impl<U,P,D,I,PI,const N:usize> BatchPreTrainBase<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U,PreOutput=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchPreOutput>;
}

impl<U,P,D,I,PI,const N:usize> BatchPreTrain<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceAccumulator<U,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack,TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_accumulator(&input))?;

        Ok(Cons(r,u))
    }
}

impl<U,P,D,I,PI,const N:usize> BatchBackward<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceAccumulator<U,PI,N> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack),TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_accumulator(&loss)?;

        let (s,next_loss) = self.parent.batch_loss(next_loss,lossf,s)?;

        let (l,s) = self.parent.batch_backward(next_loss, s, lossf)?;

        Ok((l,s))
    }
}

impl<U,P,D,I,PI,const N:usize> BatchLoss<U> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceAccumulator<U,PI,N> {}

// OnStep implementation delegates to parent
impl<U,P,D,I,PI,const N:usize> OnStep for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> + OnStep,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug {
    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> { self.parent.on_step(step) }
}

// Generic instantiation for any device implementing Clone
impl<U,P,D,I,PI,const N:usize> AccumulatorLayerInstantiation<U,P,D,I,PI,N> for AccumulatorLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + Clone,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType {
    fn instantiation(parent: P, device: &D) -> Result<AccumulatorLayer<U,P,D,I,PI,N>,LayerInstantiationError> {
        Ok(AccumulatorLayer { parent, device: device.clone(), u: PhantomData::<U> })
    }
}

/// AccumulatorLayer builder
pub struct AccumulatorLayerBuilder<const N:usize> {}

impl<const N:usize> AccumulatorLayerBuilder<N> {
    pub fn new() -> AccumulatorLayerBuilder<N> { AccumulatorLayerBuilder {} }

    /// Create an instance of AccumulatorLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,P,D,I,PI>(&self,parent:P,device:&D)
        -> Result<AccumulatorLayer<U,P,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U> + Clone,
              I: Debug + Send + Sync + BatchDataType,
              PI: Debug + BatchDataType,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              AccumulatorLayer<U,P,D,I,PI,N>: AccumulatorLayerInstantiation<U,P,D,I,PI,N> {
        AccumulatorLayer::instantiation(parent, device)
    }
}
