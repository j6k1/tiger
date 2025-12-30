use std::fmt::Debug;

use libc::{size_t};
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

use nncombinator::arr::SerializedVec;
use nncombinator::cuda::kernel::device::{BackwardLinear, BackwardLinearArgs, BackwardLinearBatch, BackwardLinearBatchArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use nncombinator::mem::{AsRawSlice};
use nncombinator::arr::{Arr, Arr2};
use nncombinator::cuda::{CudaConstPtr, CudaTensor1dPtr, CudaTensor2dPtr, CudaVec, CudaVecView, Kernel, WriteMemory, MemoryMoveTo, CudaPtr, CudaTensor1dPtrView, AsCudaPtr, AsCudaMutPtr, CudaMutPtr, ReadMemory};
use nncombinator::cuda::allocator::CudaAllocator;
use nncombinator::device::{DeviceAllocator, DeviceCpu, DeviceGpu};
use nncombinator::error::{EvaluateError, TrainingError, TypeConvertError};
use nncombinator::layer::{BatchDataType, BatchSize};
use nncombinator::ope::UnitValue;

use crate::features::{HalfKP, HalfKPListView, HalfKPView};
use crate::kernel::{TransformFeaturesForward, TransformFeaturesForwardArgs, TransformFeaturesForwardBatch, TransformFeaturesForwardBatchArgs, TransformFeaturesGradient, TransformFeaturesGradientArgs, TransformFeaturesGradientBatch, TransformFeaturesGradientBatchArgs, TransformFeaturesInputToBits, TransformFeaturesInputToBitsArgs};

pub trait DeviceFeatureTransform<U,T,B,const NI: usize,const NO: usize>
    where U: UnitValue<U>, [(); NO*2]: {
    type Output: BatchDataType + Debug + 'static;
    type BatchOutput: BatchSize + Debug + 'static;
    fn forward_feature_transform<'a>(&self,bias:&B,units:&T,input:HalfKPView<'a,NI>) -> Result<Self::Output,EvaluateError>;
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,NI>,loss:&'a Self::Output) -> Result<T,TrainingError>;
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:&'a Self::Output) -> Result<B,TrainingError>;
    fn batch_forward_feature_transform<'a>(&self,bias:&B,units:&T,input:HalfKPListView<'a,NI>) -> Result<Self::BatchOutput,TrainingError>;
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPListView<'a,NI>,loss:&'a Self::BatchOutput) -> Result<T,TrainingError>;
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:&'a Self::BatchOutput) -> Result<B,TrainingError>;
}
impl<U,const NI: usize,const NO:usize> DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U> 
    where U: UnitValue<U>, [(); NO*2]: {

    type Output = Arr<U,{NO*2}>;
    type BatchOutput = SerializedVec<U,Arr<U,{NO*2}>>;
    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:HalfKPView<'a,NI>) 
        -> Result<Arr<U,{NO*2}>,EvaluateError> {
        let mut r = Vec::with_capacity(NO*2);

        r.extend_from_slice(&bias);
        r.extend_from_slice(&bias);

        for input in input.iter() {
            for &i in input.iter() {
                units.iter().nth(i).map(|it| {
                    for (&w,r) in  it.iter().zip(r.iter_mut().take(NO)) {
                        *r += w;
                    }
                });
            }

            for &i in input.iter() {
                units.iter().nth(i).map(|it| {
                    for (&w,r) in  it.iter().zip(r.iter_mut().skip(NO)) {
                        *r += w;
                    }
                });
            }
        }

        Ok(r.try_into()?)
    }

    #[inline]
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,NI>,loss:&'a Arr<U,{NO*2}>) -> Result<Arr2<U,NI,NO>,TrainingError> {
        let mut acc = Arr2::<U,NI,NO>::new();

        let d = U::from_f64(2.).unwrap();

        let loss = loss.iter().map(|&l| l / d).collect::<Vec<U>>();
        let (sl,ol) = loss.split_at(NO);

        let sl = <&[U;NO]>::try_from(sl)?;
        let ol = <&[U;NO]>::try_from(ol)?;

        let input = input.to_vec::<U>();

        let (si,oi) = input.split_at(NI);

        for (input,loss) in [si,oi].into_iter()
                                                                .zip([sl,ol].into_iter()){
            for (&input,mut acc) in input.iter().zip(acc.iter_mut()) {
                for (&loss,acc) in loss.iter().zip(acc.iter_mut()) {
                    *acc += input * loss;
                }
            }
        }

        Ok(acc)
    }

    #[inline]
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:&'a Arr<U,{NO*2}>) -> Result<Arr<U,NO>,TrainingError> {
        let mut acc = Arr::<U,NO>::new();

        {
            let d = U::from_f64(2.).unwrap();

            let (sl,ol) = loss.as_raw_slice().split_at(NO);

            for (acc,s) in acc.iter_mut().zip(sl.iter()) {
                *acc += *s / d;
            }

            for (acc,o) in acc.iter_mut().zip(ol.iter()) {
                *acc += *o / d;
            }        
        }

        Ok(acc)
    }

    #[inline]
    fn batch_forward_feature_transform<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:HalfKPListView<'a,NI>)
         -> Result<SerializedVec<U,Arr<U,{NO*2}>>,TrainingError> {
        
        Ok(<&'a Vec<HalfKP<NI>>>::from(input).par_iter().map(|input| {
            self.forward_feature_transform(bias, units, input.into())
        }).collect::<Result<Vec<Arr<U,{NO*2}>>,EvaluateError>>()?.into())
    }

    #[inline]
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPListView<'a,NI>,loss:&'a SerializedVec<U,Arr<U,{NO*2}>>)
        -> Result<Arr2<U,NI,NO>,TrainingError> {

        <&'a Vec<HalfKP<NI>>>::from(input).par_iter().zip(loss.par_iter()).map(|(i,l)| {
            let mut acc = Arr2::<U,NI,NO>::new();

            let d = U::from_f64(2.).unwrap();

            let loss = l.iter().map(|&l| l / d).collect::<Vec<U>>();
            let (sl,ol) = loss.split_at(NO);

            let sl = <&[U;NO]>::try_from(sl)?;
            let ol = <&[U;NO]>::try_from(ol)?;

            let input = i.to_vec::<U>();

            let (si,oi) = input.split_at(NI);

            for (input,loss) in [si,oi].into_iter()
                .zip([sl,ol].into_iter()){
                for (&input,mut acc) in input.iter().zip(acc.iter_mut()) {
                    for (&loss,acc) in loss.iter().zip(acc.iter_mut()) {
                        *acc += input * loss;
                    }
                }
            }

            Ok(acc)
        }).reduce(|| Ok(Arr2::new()), | acc, g | {
            acc.and_then(| mut acc | g.and_then(|g| {
                for (mut acc, g) in acc.iter_mut().zip(g.iter()) {
                    for (acc, &g) in acc.iter_mut().zip(g.iter()) {
                        *acc += g;
                    }
                }

                Ok(acc)
            }))
        })
    }

    #[inline]
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:&'a SerializedVec<U,Arr<U,{NO*2}>>) -> Result<Arr<U,NO>,TrainingError> {
        let g = loss.par_iter().fold(|| Arr::<U,{NO*2}>::new(), | mut acc, loss | {
            for (acc,&loss) in acc.iter_mut().zip(loss.iter()) {
                *acc += loss;
            }

            acc
        }).reduce(|| Arr::new(), | mut acc, g | {
            for (acc,&g) in acc.iter_mut().zip(g.iter()) {
                *acc += g;
            }

            acc
        });

        let mut acc = Arr::<U,NO>::new();

        {
            let d = U::from_f64(2.).unwrap();

            let (sl,ol) = g.as_raw_slice().split_at(NO);

            for (acc,s) in acc.iter_mut().zip(sl.iter()) {
                *acc += *s / d;
            }

            for (acc,o) in acc.iter_mut().zip(ol.iter()) {
                *acc += *o / d;
            }        
        }

        Ok(acc)
    }
}
impl<A,const NI: usize,const NO:usize> DeviceFeatureTransform<f32,CudaTensor2dPtr<f32,A,NI,NO>,CudaTensor1dPtr<f32,A,NO>,NI,NO> for DeviceGpu<f32,A>
    where A: CudaAllocator + 'static,
          CudaPtr<usize,A>: WriteMemory<usize>,
          CudaPtr<u8,A>: WriteMemory<u8>,
          CudaVec<f32,CudaTensor1dPtr<f32,A,NO>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>: ReadMemory<f32>,
          CudaTensor2dPtr<f32,A,NI,NO>: ReadMemory<f32>,
          for<'a> CudaPtr<f32,A>: WriteMemory<f32> + MemoryMoveTo<f32,CudaMutPtr<'a,f32,A>>,
          for<'a> CudaTensor1dPtrView<'a,f32,{NO*2}>: From<&'a CudaTensor1dPtr<f32,A,{NO*2}>>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,NI>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A>,Error=TypeConvertError>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,NO>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,A,NO>,A>,Error=TypeConvertError>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,{NO*2}>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>,Error=TypeConvertError>,
          for<'a> TransformFeaturesForward::<'a,f32,A,NI,NO>: Kernel<Args=TransformFeaturesForwardArgs<'a,f32,A,NI,NO>>,
          for<'a> TransformFeaturesGradient::<'a,f32,A,NI,NO>: Kernel<Args=TransformFeaturesGradientArgs<'a,f32,A,NI,NO>>,
          for<'a> BackwardLinear::<'a,f32,A,NI,NO>: Kernel<Args=BackwardLinearArgs<'a,f32,A,NI,NO>>,
          for<'a> LinearGradientBatch::<'a,f32,A,NI,NO>: Kernel<Args=LinearGradientBatchArgs<'a,f32,A,NI,NO>>,
          for<'a> TransformFeaturesForwardBatch::<'a,f32,A,NI,NO>: Kernel<Args=TransformFeaturesForwardBatchArgs<'a,f32,A,NI,NO>>,
          for<'a> TransformFeaturesGradientBatch::<'a,f32,A,NI,NO>: Kernel<Args=TransformFeaturesGradientBatchArgs<'a,f32,A,NI,NO>>,
          for<'a> BackwardLinearBatch::<'a,f32,A,NI,NO>: Kernel<Args=BackwardLinearBatchArgs<'a,f32,A,NI,NO>>,
          for<'a> ReduceLinearBatch::<'a,f32,A,NO>: Kernel<Args=ReduceLinearBatchArgs<'a,f32,A,NO>>, [(); NO*2]: {
    type Output = CudaTensor1dPtr<f32,A,{NO*2}>;
    type BatchOutput = CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>;
    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,A,NO>,units:&CudaTensor2dPtr<f32,A,NI,NO>,input:HalfKPView<'a,NI>)
        -> Result<CudaTensor1dPtr<f32,A,{NO*2}>,EvaluateError> {
        let (indexes,boundaries):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let mut indexes_ptr = CudaPtr::new(indexes.len(),self.get_allocator())?;
        let mut boundaries_ptr = CudaPtr::new(3,self.get_allocator())?;

        indexes_ptr.memcpy(indexes.as_ptr(),indexes.len())?;
        boundaries_ptr.memcpy(boundaries.as_ptr(),3)?;

        let output = CudaTensor1dPtr::<f32,A,{NO*2}>::new(self.get_allocator())?;

        let mut args = TransformFeaturesForwardArgs::new(
                                                   indexes_ptr,
                                                   boundaries_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output);

        let mut kernel = TransformFeaturesForward::<f32,A,NI,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    #[inline]
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,NI>,loss:&'a CudaTensor1dPtr<f32,A,{NO*2}>)
        -> Result<CudaTensor2dPtr<f32,A,NI,NO>,TrainingError> {
        let (indexes,_):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let loss = CudaTensor1dPtrView::<f32,{NO*2}>::from(loss);

        let input = indexes.iter().fold(vec![0u8; (NI + 7) / 8], | mut acc, &i | {
            let chunk_index = i / 8;
            let bit_index = i - chunk_index * 8;

            acc[chunk_index] |= 1 << bit_index;
            acc
        });

        let mut input_ptr = CudaPtr::new((NI + 7) / 8 * 2,self.get_allocator())?;
        let output = CudaTensor2dPtr::<f32,A,NI,NO>::new(self.get_allocator())?;

        input_ptr.memcpy(input.as_ptr(),input.len())?;

        let mut args = TransformFeaturesGradientArgs::new(
            CudaConstPtr::new(&loss),
            input_ptr,
            output);

        let mut kernel = TransformFeaturesGradient::<f32,A,NI,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    #[inline]
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:&'a CudaTensor1dPtr<f32,A,{NO*2}>) -> Result<CudaTensor1dPtr<f32,A,NO>,TrainingError> {
        let mut loss_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(2,self.get_allocator())?;
        loss.as_cuda_ptr().memcpy_to(&mut loss_ptr.as_cuda_mut_ptr(),NO*2)?;

        let output_ptr = CudaTensor1dPtr::<f32,A,NO>::new(self.get_allocator())?;

        let loss_ptr = (&loss_ptr).try_into()?;

        let mut args = ReduceLinearBatchArgs::new(&loss_ptr,output_ptr,NO,2);

        let mut kernel = ReduceLinearBatch::<f32,A,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }


    #[inline]
    fn batch_forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,A,NO>,units:&CudaTensor2dPtr<f32,A,NI,NO>,
                                            input:HalfKPListView<'a,NI>)
         -> Result<CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>,TrainingError> {
        let (indexes,boundaries):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let len = input.size();

        let mut indexes_ptr = CudaPtr::new(indexes.len(),self.get_allocator())?;
        let mut boundaries_ptr = CudaPtr::new(boundaries.len(),self.get_allocator())?;

        indexes_ptr.memcpy(indexes.as_ptr(),indexes.len())?;
        boundaries_ptr.memcpy(boundaries.as_ptr(), boundaries.len())?;

        let output = CudaVec::<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>::new(len,self.get_allocator())?;

        let mut args = TransformFeaturesForwardBatchArgs::new(
            indexes_ptr,
            boundaries_ptr,
            CudaConstPtr::new(units),
            CudaConstPtr::new(bias),
            output,
            len);

        let mut kernel = TransformFeaturesForwardBatch::<f32,A,NI,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    #[inline]
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPListView<'a,NI>,loss:&'a CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>)
        -> Result<CudaTensor2dPtr<f32,A,NI,NO>,TrainingError> {
        let len = input.size();

        let (indexes,boundaries):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let mut indexes_ptr = CudaPtr::new(indexes.len(),self.get_allocator())?;
        let mut boundaries_ptr = CudaPtr::new(boundaries.len(),self.get_allocator())?;

        indexes_ptr.memcpy(indexes.as_ptr(),indexes.len())?;
        boundaries_ptr.memcpy(boundaries.as_ptr(), boundaries.len())?;

        let bits = CudaPtr::<u8,A>::with_initializer((NI + 7) / 8 * len * 2, self.get_allocator(), || 0)?;

        let mut args = TransformFeaturesInputToBitsArgs::<A,NI>::new(indexes_ptr,boundaries_ptr,bits,len);

        let mut kernel = TransformFeaturesInputToBits::<A,NI>::new();

        kernel.launch(&mut args)?;

        let input_ptr = args.bits;
        let loss = CudaVecView::<f32,CudaTensor1dPtrView<f32,{NO*2}>>::try_from(loss)?;

        let output = CudaTensor2dPtr::<f32,A,NI,NO>::new(self.get_allocator())?;

        let mut args = TransformFeaturesGradientBatchArgs::new(
            CudaConstPtr::new(&loss),
            input_ptr,
            output,
            len);

        let mut kernel = TransformFeaturesGradientBatch::<f32,A,NI,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    #[inline]
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:&'a CudaVec<f32,CudaTensor1dPtr<f32,A,{NO*2}>,A>) -> Result<CudaTensor1dPtr<f32,A,NO>,TrainingError> {
        let len = loss.size();

        let mut loss_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(len * 2,self.get_allocator())?;
        loss.memcpy_to(&mut loss_ptr,len * 2 * NO)?;

        let output_ptr = CudaTensor1dPtr::<f32,A,NO>::new(self.get_allocator())?;

        let loss_ptr = (&loss_ptr).try_into()?;

        let mut args = ReduceLinearBatchArgs::new(&loss_ptr,output_ptr,NO,2 * len);

        let mut kernel = ReduceLinearBatch::<f32,A,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}