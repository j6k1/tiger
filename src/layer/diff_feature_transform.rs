//! Feature Transformation Layer Implementation
use std::fmt::{Debug};
use std::marker::PhantomData;

use nncombinator::arr::{Arr, Arr2};
use nncombinator::device::{Device, DeviceSimd};
use nncombinator::device::bridge::DeviceBridge;
use nncombinator::device::scale::DeviceScale;
use nncombinator::error::{EvaluateError, LayerInstantiationError, ModelLoadError, PersistenceError, TrainingError};
use nncombinator::layer::{BatchDataType, BatchSize, Bridge, BridgeBase, ContinueForward, Forward, ForwardAll, ForwardDiff, InputMax, InputScale, OnStep, OutputTensorScalar, PartialForward, Quantization};
use nncombinator::mapper::ScalingMapper;
use nncombinator::persistence::{Linear, LinearPersistence, Persistence};
use simdoperator::backend::common::Backend;
use crate::device::{DeviceDiffFeatureTransform, DeviceFeatureTransformEvaluation, DeviceFeatureTransformEvaluationScaling, DeviceQuantizedFeatureTransform};
use crate::features::HalfKP;

pub struct DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static {
    parent:P,
    device:D,
    units:C,
    bias:B,
    scale:f32,
    shift:usize,
    input_scale:f32,
    input_max:f32,
    i:PhantomData<I>,
    di:PhantomData<DI>,
    u:PhantomData<U>,
    w:PhantomData<W>
}

impl<BE,U,W,P,I,DI,const NI:usize,const NO:usize> DiffFeatureTransformLayer<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>
    where BE: Backend + Clone,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug {
    #[inline]
    pub fn new(parent:P,device:&DeviceSimd<BE>)
        -> Result<DiffFeatureTransformLayer<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>, LayerInstantiationError> {

        let mut units = Vec::<W>::with_capacity(NI*NO);
        let mut bias = Vec::<W>::with_capacity(NO);

        units.resize_with(NI*NO,Default::default);
        bias.resize_with(NO,Default::default);

        Ok(DiffFeatureTransformLayer {
            parent: parent,
            device: device.clone(),
            units: units.try_into()?,
            bias: bias.try_into()?,
            scale: 1.,
            shift: 1,
            input_scale: 1.,
            input_max: 1.,
            i:PhantomData::<I>,
            di:PhantomData::<DI>,
            u:PhantomData::<U>,
            w:PhantomData::<W>
        })
    }
}
impl<T,BE,U,W,P,I,DI,const NI:usize,const NO:usize> Persistence<T,Linear> for DiffFeatureTransformLayer<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>
    where BE: Backend + Clone,
          T: LinearPersistence<W>,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + Persistence<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug {
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
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> Forward<HalfKP<NI>,Result<<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,EvaluateError>>
    for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          [(); NO*2]: {
    #[inline]
    fn forward(&self, input:&HalfKP<NI>) -> Result<<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,EvaluateError> {
        let r = self.device.forward_feature_transform(&self.bias,&self.units,input.into())?;

        Ok(self.device.forward_feature_transform_scaling(self.shift,&r)?)
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> ForwardAll for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          [(); NO * 2]: {
    type Input = I;
    type Output = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;

    #[inline]
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> PartialForward for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
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
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> ForwardDiff for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          C: 'static,
          B: 'static,
          U: Default + Clone + Copy + Send + Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug,
          for<'a> D: Device<U> +
                     DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
                     DeviceDiffFeatureTransform<U,DI,C,B,NI,NO> +
                     DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> + 'static,
          Self: ForwardAll<Input=<P as ForwardAll>::Output,Output=<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput>,
          [(); NO * 2]: {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let r = self.device.forward_diff_feature_transform(&self.bias,&self.units,&input,partial_input)?;

        Ok(self.device.forward_feature_transform_scaling(self.shift,&r)?)
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> ContinueForward for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> +
             PartialForward<DiffInput=DI>,
      W: Default + Clone + Copy + Debug + Send + Sync + 'static,
      C: 'static,
      B: 'static,
      U: Default + Clone + Copy + Send + Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      DI: Debug,
      for<'a> D: Device<U> +
                 DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
                 DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
                 DeviceDiffFeatureTransform<U,DI,C,B,NI,NO> + 'static,
      Self: PartialForward<PartialInput=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output,PartialOutput=<D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output>,
      <D as DeviceFeatureTransformEvaluation<U,C,B,NI,NO>>::Output: Clone,
      Self: ForwardAll<Input=<P as ForwardAll>::Output,Output=<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput>,
      [(); NO * 2]: {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.forward_feature_transform_scaling(self.shift,input)?)
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> OutputTensorScalar for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static {
    type Scalar = U;
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> InputMax for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static {
    type Scalar = f32;
    fn input_max(&self) -> Self::Scalar {
        self.input_max
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> InputScale for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static {
    fn input_scale(&self) -> f32 {
        self.input_scale
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> OnStep for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>> + OnStep,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> + 'static {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.parent.on_step(step)
    }

    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.parent.on_frequently_step(step, frequently_step)
    }
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> BridgeBase
    for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          U: Default + Clone + Copy + Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug + 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
             DeviceQuantizedFeatureTransform<C,B,NI,NO> +
             DeviceBridge<U,f32,<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,D::RealOutput> +
             DeviceScale<f32,D::RealOutput,NO>,
          C: 'static,
          B: 'static,
          <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput: Debug + BatchDataType + 'static,
          <<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceQuantizedFeatureTransform<C,B,NI,NO>>::RealOutput: Debug + BatchDataType + 'static,
          <<D as DeviceQuantizedFeatureTransform<C,B,NI,NO>>::RealOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          [(); NO * 2]: {
    type UseDevice = D;
    type RealScale = D::RealOutput;
    type RealOutput = D::RealOutput;
    type SourceInput = <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput;
}
impl<U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> Bridge
    for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          U: Default + Clone + Copy + Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          DI: Debug + 'static,
          C: 'static,
          B: 'static,
          D: Device<U> +
             DeviceFeatureTransformEvaluation<U,C,B,NI,NO> +
             DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO> +
             DeviceQuantizedFeatureTransform<C,B,NI,NO> +
             DeviceBridge<U,f32,<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput,D::RealOutput> +
             DeviceScale<f32,D::RealOutput,NO>,
          <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput: Debug + BatchDataType + 'static,
          <<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceQuantizedFeatureTransform<C,B,NI,NO>>::RealOutput: Debug + BatchDataType + 'static,
          <<D as DeviceQuantizedFeatureTransform<C,B,NI,NO>>::RealOutput as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Output=<D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput>,
          [(); NO * 2]: {
    type RealMapper<'a> = ScalingMapper<'a,f32,D::ScaledOutput,D::RealOutput,D,NO>
       where Self: 'a;
    fn as_real<'a>(&'a self, input: &'a <D as DeviceFeatureTransformEvaluationScaling<U,C,B,NI,NO>>::ScaledOutput) -> Result<Self::RealMapper<'a>, EvaluateError>
        where Self: 'a {
        let o = self.device.bridge_forward(input)?;

        Ok(ScalingMapper::new(&self.device,input,&o,self.scale)?)
    }
}
impl<T,U,W,P,I,DI,C,B,D,const NI:usize,const NO:usize> Quantization<T,Linear> for DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>
    where T: LinearPersistence<f32>,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          I: Debug + Send + Sync,
          DI: Debug,
          D: Device<U> +
             DeviceQuantizedFeatureTransform<C,B,NI,NO> + 'static {
    fn quantization(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        let mut bias = Arr::<f32,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<f32,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        let (scale,shift,input_scale,input_max,units, bias) = self.device.quantization(&units, &bias)?;

        self.scale = scale;
        self.shift = shift;
        self.input_scale = input_scale;
        self.input_max = input_max;
        self.units = units;
        self.bias = bias;

        Ok(())
    }
}
pub trait DiffFeatureTransformLayerInstantiation<U,W,P,I,DI,C,BC,D,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          U: Default + Clone + Copy + Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
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
    fn instantiation(parent:P,device:&D) -> Result<DiffFeatureTransformLayer<U,W,P,I,DI,C,BC,D,NI,NO>,LayerInstantiationError>;
}

impl<BE,U,W,P,I,DI,const NI:usize,const NO:usize> DiffFeatureTransformLayerInstantiation<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>
    for DiffFeatureTransformLayer<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>
    where BE: Backend + Clone,
          P: ForwardAll<Input=I,Output=HalfKP<NI>>,
          U: Default + Clone + Copy + Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          [(); NO * 2]:  {
    fn instantiation(parent: P, device:&DeviceSimd<BE>) -> Result<DiffFeatureTransformLayer<U,W,P,I,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>,LayerInstantiationError> {
        DiffFeatureTransformLayer::<_,_,_,_,DI,Arr2<W,NI,NO>,Arr<W,NO>,DeviceSimd<BE>,NI,NO>::new(parent, device)
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
    pub fn build<U,W,C,B,P,D,I,DI>(&self,parent: P, device:&D)
        -> Result<DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=HalfKP<NI>>,
              U: Clone + Copy + Default + Clone + Copy + Debug + Send + Sync + 'static,
              W: Default + Clone + Copy + Debug + Send + Sync + 'static,
              I: Debug + Send + Sync + BatchDataType,
              DI: Debug,
              D: Device<U> + 'static,
              <I as BatchDataType>::Type: Debug + BatchSize,
              DiffFeatureTransformLayer<U,W,P,I,DI,C,B,D,NI,NO>: DiffFeatureTransformLayerInstantiation<U,W,P,I,DI,C,B,D,NI,NO>,
              [(); NO * 2]: {
        Ok(DiffFeatureTransformLayer::<U,W,P,I,DI,C,B,D,NI,NO>::instantiation(parent, device)?)
    }
}
