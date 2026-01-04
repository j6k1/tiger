//! Feature Transformation Layer Implementation
use std::fmt::{Debug};
use std::marker::PhantomData;

use nncombinator::arr::{Arr, Arr2};
use nncombinator::{Cons, Stack};
use nncombinator::cuda::{CudaPtr, CudaTensor1dPtr, CudaTensor2dPtr, ReadMemory, WriteMemory};
use nncombinator::cuda::allocator::CudaAllocator;
use nncombinator::device::{Device, DeviceAllocator, DeviceBatchAveraging, DeviceCpu, DeviceGpu};
use nncombinator::error::{EvaluateError, LayerInstantiationError, TrainingError, ConfigReadError, PersistenceError};
use nncombinator::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, Forward, ForwardAll, Loss, OnStep, PreTrain, UpdateWeight};
use nncombinator::lossfunction::LossFunction;
use nncombinator::optimizer::{Optimizer, OptimizerBuilder};
use nncombinator::mem::AsRawSlice;
use nncombinator::persistence::{Linear, LinearPersistence, Persistence};
use nncombinator::ope::UnitValue;

use crate::device::DeviceFeatureTransform;
use crate::features::HalfKP;

pub struct FeatureTransformLayer<U,P,I,C,B,D,OP,const NI:usize,const NO:usize>
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
    u:PhantomData<U>
}

impl<U,P,I,OP,const NI:usize,const NO:usize> FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> +
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
            i:PhantomData::<I>,
            u:PhantomData::<U>
        })
    }
}

impl<U,P,I,A,OP,const NI:usize,const NO:usize> FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where U: UnitValue<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U>,
             DeviceGpu<U,A>: Device<U> + 'static,
          I: Debug + Send + Sync,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>>,
          CudaPtr<U,A>: WriteMemory<U> {
    #[inline]
    pub fn new<UI,BI,OB: OptimizerBuilder<U,DeviceGpu<U,A>,Output=OP>>(parent:P,device:&DeviceGpu<U,A>,ui: UI, bi: BI, b:&OB)
        -> Result<FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>, LayerInstantiationError>
        where U: UnitValue<U>,
              UI: FnMut() -> U,
              BI: FnMut() -> U {
        let get_allocator = device.get_allocator();

        Ok(FeatureTransformLayer {
            parent: parent,
            device: device.clone(),
            units: CudaTensor2dPtr::with_initializer(get_allocator,ui)?,
            bias: CudaTensor1dPtr::with_initializer(get_allocator,bi)?,
            unit_optimizer: b.build(NI*NO)?,
            bias_optimizer: b.build(NO)?,
            i:PhantomData::<I>,
            u:PhantomData::<U>
        })
    }
}

impl<T,U,P,I,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U>,
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

impl<T,U,P,I,A,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U>,
          DeviceGpu<U,A>: Device<U>,
          U: UnitValue<U>,
          I: Debug + Send + Sync,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
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

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> Forward<HalfKP<NI>,Result<<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output,EvaluateError>>
    for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,D> + 'static,
          [(); NO*2]: {
    #[inline]
    fn forward(&self, input:&HalfKP<NI>) -> Result<<D as DeviceFeatureTransform<U,C,B,NI,NO>>::Output,EvaluateError> {
        self.device.forward_feature_transform(&self.bias,&self.units,input.into())
    }
}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> ForwardAll for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
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

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> PreTrain<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
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

impl<U,P,I,OP,const NI:usize,const NO:usize> BackwardAll<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          U: UnitValue<U>,
          DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          I: Debug + Send + Sync + 'static,
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

impl<U,P,I,A,OP,const NI:usize,const NO:usize> BackwardAll<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> +
             ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: WriteMemory<U> + ReadMemory<U>,
          CudaTensor2dPtr<U,A,NI,NO>: WriteMemory<U> + ReadMemory<U>,
          DeviceGpu<U,A>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO> +
                          DeviceBatchAveraging<CudaTensor2dPtr<U,A,NI,NO>,U> +
                          DeviceBatchAveraging<CudaTensor1dPtr<U,A,NO>,U> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor1dPtr<U,A,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor2dPtr<U,A,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,A,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,A,NI,NO>>,
          [(); NO * 2]: {
    type LossInput = <DeviceGpu<U,A> as DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    #[inline]
    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
                                        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.backward_feature_transform_weight_gradient(o.into(),&loss)
        })?;

        let bg = self.device.backward_feature_transform_bias_gradient(&loss)?;

        let (s,loss) = self.parent.loss((),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}

impl<U,P,I,OP,const NI:usize,const NO:usize> UpdateWeight<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + UpdateWeight<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
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

impl<U,P,I,A,OP,const NI:usize,const NO:usize> UpdateWeight<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + UpdateWeight<U> + 'static,
          DeviceGpu<U,A>: Device<U> +
                          DeviceBatchAveraging<CudaTensor2dPtr<U,A,NI,NO>,U> +
                          DeviceBatchAveraging<CudaTensor1dPtr<U,A,NO>,U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: ReadMemory<U>,
          CudaTensor2dPtr<U,A,NI,NO>: ReadMemory<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor1dPtr<U,A,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor2dPtr<U,A,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,A,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,A,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>)>;

    #[inline]
    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        let bg = self.device.batch_averaging(bg,batch_size * 2)?;
        let g = self.device.batch_averaging(g,batch_size * 2)?;

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s, batch_size)?)
    }
}

impl<U,P,I,OP,const NI:usize,const NO:usize> Loss<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {}

impl<U,P,I,A,OP,const NI:usize,const NO:usize> Loss<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> +
             BackwardAll<U,LossInput=()> + Loss<U> + 'static,
          DeviceGpu<U,A>: Device<U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: ReadMemory<U>,
          CudaTensor2dPtr<U,A,NI,NO>: ReadMemory<U>,
          DeviceGpu<U,A>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO> +
                          DeviceBatchAveraging<CudaTensor2dPtr<U,A,NI,NO>,U> +
                          DeviceBatchAveraging<CudaTensor1dPtr<U,A,NO>,U> + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor1dPtr<U,A,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor2dPtr<U,A,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,A,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,A,NI,NO>>,
          [(); NO * 2]: {}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchForwardBase for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]: {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::BatchOutput;
}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchForward for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
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

        self.device.batch_forward_feature_transform(&self.bias,&self.units,(&input).into())
    }
}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchPreTrainBase<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> + DeviceFeatureTransform<U,C,B,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,D> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          [(); NO * 2]: {
    type BatchPreOutput = <D as DeviceFeatureTransform<U,C,B,NI,NO>>::BatchOutput;
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack, <D as DeviceFeatureTransform<U,C,B,NI,NO>>::BatchOutput>;
}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> BatchPreTrain<U> for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
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
            self.device.batch_forward_feature_transform(&self.bias,&self.units,input.into())
        })?;

        Ok(Cons(r,u))
    }
}

impl<U,P,I,OP,const NI:usize,const NO:usize> BatchBackward<U>
    for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
             DeviceCpu<U>: Device<U> + DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {
    type BatchLossInput = <DeviceCpu<U> as DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::BatchOutput;
    type BatchLossOutput = ();

    #[inline]
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.batch_backward_feature_transform_weight_gradient(o.into(), &loss)
        })?;

        let bg = self.device.batch_feature_transform_bias_gradient(&loss)?;

        let (_,s) = self.parent.batch_backward((), s, lossf)?;

        Ok(((),Cons(s,(g,bg))))
    }
}

impl<U,P,I,A,OP,const NI:usize,const NO:usize> BatchBackward<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> +
             BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
          DeviceGpu<U,A>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO> +
                          DeviceBatchAveraging<CudaTensor2dPtr<U,A,NI,NO>,U> +
                          DeviceBatchAveraging<CudaTensor1dPtr<U,A,NO>,U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: ReadMemory<U>,
          CudaTensor2dPtr<U,A,NI,NO>: ReadMemory<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor1dPtr<U,A,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor2dPtr<U,A,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,A,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,A,NI,NO>>,
          [(); NO * 2]: {
    type BatchLossInput = <DeviceGpu<U,A> as DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO>>::BatchOutput;
    type BatchLossOutput = ();

    #[inline]
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.batch_backward_feature_transform_weight_gradient(o.into(), &loss)
        })?;

        let bg = self.device.batch_feature_transform_bias_gradient(&loss)?;

        let (_,s) = self.parent.batch_backward((), s, lossf)?;

        Ok(((),Cons(s,(g,bg))))
    }
}

impl<U,P,I,OP,const NI:usize,const NO:usize> BatchLoss<U> for FeatureTransformLayer<U,P,I,Arr2<U,NI,NO>,Arr<U,NO>,DeviceCpu<U>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          [(); NO * 2]: {}

impl<U,P,I,A,OP,const NI:usize,const NO:usize> BatchLoss<U>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: PreTrain<U,PreOutput=HalfKP<NI>> + ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<HalfKP<NI> as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=()> + 'static,
          DeviceGpu<U,A>: Device<U> + DeviceFeatureTransform<U,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,NI,NO> +
                          DeviceBatchAveraging<CudaTensor2dPtr<U,A,NI,NO>,U> +
                          DeviceBatchAveraging<CudaTensor1dPtr<U,A,NO>,U> + 'static,
          U: UnitValue<U>,
          I: Debug + Send + Sync + 'static + BatchDataType,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>> + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: ReadMemory<U>,
          CudaTensor2dPtr<U,A,NI,NO>: ReadMemory<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor1dPtr<U,A,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalType: From<&'a CudaTensor2dPtr<U,A,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,A,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U,A>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,A,NI,NO>>,
          [(); NO * 2]: {}

impl<U,P,I,C,B,D,OP,const NI:usize,const NO:usize> OnStep for FeatureTransformLayer<U,P,I,C,B,D,OP,NI,NO>
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

pub trait FeatureTransformLayerInstantiation<U,P,I,C,BC,D,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
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
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
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

impl<U,P,I,A,OP,const NI:usize,const NO:usize> FeatureTransformLayerInstantiation<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    for FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          A: CudaAllocator,
          OP: Optimizer<U,DeviceGpu<U,A>>,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    fn instantiation<B: OptimizerBuilder<U,DeviceGpu<U,A>,Output=OP>>(parent: P, device:&DeviceGpu<U,A>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                                                                    -> Result<FeatureTransformLayer<U,P,I,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,OP,NI,NO>,LayerInstantiationError> {
        FeatureTransformLayer::<_,_,_,CudaTensor2dPtr<U,A,NI,NO>,CudaTensor1dPtr<U,A,NO>,DeviceGpu<U,A>,_,NI,NO>::new(parent,device,ui,bi,b)
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
        where P: ForwardAll<Input=I,Output=HalfKP<NI>> + BackwardAll<U,LossInput=()> +
                 PreTrain<U,PreOutput=HalfKP<NI>> + Loss<U>,
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
