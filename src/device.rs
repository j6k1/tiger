use std::fmt::Debug;
use std::mem;

use libc::{size_t};
use libc::c_uint;
use cuda_runtime_sys::{dim3};
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

use nncombinator::arr::SerializedVec;
use nncombinator::cuda::kernel::device::{BackwardLinear, BackwardLinearArgs, BackwardLinearBatch, BackwardLinearBatchArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use nncombinator::mem::{AsRawSlice};
use nncombinator::arr::{Arr, Arr2};
use nncombinator::cuda::{CudaConstPtr, CudaMemoryPoolPtr, CudaTensor1dPtr, CudaTensor2dPtr, CudaVec, CudaVecView, Kernel, WriteMemory, MemoryMoveTo, CudaPtr, AsPtr, AsMutPtr};
use nncombinator::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool};
use nncombinator::error::{EvaluateError, TrainingError, TypeConvertError};
use nncombinator::layer::{BatchDataType, BatchSize};
use nncombinator::ope::UnitValue;
use rcublas_sys::{cublasOperation_t, cublasSgemm_v2, cublasStatus_t};

use crate::features::{HalfKP, HalfKPListView, HalfKPView};
use crate::kernel::{TransformFeaturesForward, TransformFeaturesForwardArgs, TransformFeaturesForwardBatch, TransformFeaturesForwardBatchArgs};

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
                   for (r,&w) in r.iter_mut().zip(it.iter()) {
                       *r = w;
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
impl<const NI: usize,const NO:usize> DeviceFeatureTransform<f32,CudaTensor2dPtr<f32,NI,NO>,CudaTensor1dPtr<f32,NO>,NI,NO> for DeviceGpu<f32>
    where for<'a> CudaVecView<'a,f32,CudaTensor1dPtr<f32,NI>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,NI>>,Error=TypeConvertError>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtr<f32,NO>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,NO>>,Error=TypeConvertError>,
          for<'a> TransformFeaturesForward::<'a,f32,NI,NO>: Kernel<Args=TransformFeaturesForwardArgs<'a,f32,NI,NO>>,
          for<'a> BackwardLinear::<'a,f32,NI,NO>: Kernel<Args=BackwardLinearArgs<'a,f32,NI,NO>>,
          for<'a> LinearGradientBatch::<'a,f32,NI,NO>: Kernel<Args=LinearGradientBatchArgs<'a,f32,NI,NO>>,
          for<'a> TransformFeaturesForwardBatch::<'a,f32,NI,NO>: Kernel<Args=TransformFeaturesForwardBatchArgs<'a,f32,NI,NO>>,
          for<'a> BackwardLinearBatch::<'a,f32,NI,NO>: Kernel<Args=BackwardLinearBatchArgs<'a,f32,NI,NO>>,
          for<'a> ReduceLinearBatch::<'a,f32,NO>: Kernel<Args=ReduceLinearBatchArgs<'a,f32,NO>>, [(); NO*2]: {
    type Output = CudaTensor1dPtr<f32,{NO*2}>;
    type BatchOutput = CudaVec<f32,CudaTensor1dPtr<f32,{NO*2}>>;
    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,input:HalfKPView<'a,NI>)
        -> Result<CudaTensor1dPtr<f32,{NO*2}>,EvaluateError> {
        let (indexes,boundaries):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let mut indexes_ptr = CudaMemoryPoolPtr::new(indexes.len(),self.get_memory_pool())?;
        let mut boundaries_ptr = CudaMemoryPoolPtr::new(3,self.get_memory_pool())?;

        indexes_ptr.memcpy(indexes.as_ptr(),indexes.len())?;
        boundaries_ptr.memcpy(boundaries.as_ptr(),3)?;

        let output = CudaTensor1dPtr::<f32,{NO*2}>::new(self.get_memory_pool())?;

        let mut args = TransformFeaturesForwardArgs::new(
                                                   indexes_ptr,
                                                   boundaries_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output);

        let mut kernel = TransformFeaturesForward::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint * 2, y: 1, z: 1 },
                      dim3 { x: 32, y: 1, z: 1 },&mut args,0)?;

        Ok(args.output)
    }

    #[inline]
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,NI>,loss:&'a CudaTensor1dPtr<f32,{NO*2}>)
        -> Result<CudaTensor2dPtr<f32,NI,NO>,TrainingError> {
        
        let mut input_ptr = CudaVec::<f32,CudaTensor1dPtr::<f32,NI>>::new(2,self.get_memory_pool())?;
        let mut output_ptr = CudaTensor2dPtr::<f32,NI,NO>::with_initializer(self.get_memory_pool(),Default::default)?;

        let input = input.to_vec();

        input_ptr.memcpy(input.as_ptr(),NI * 2)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas().id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_T,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           2 as ::libc::c_int,
                           alpha.as_ptr(),
                           loss.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr)
            },
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(TrainingError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(TrainingError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(TrainingError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(TrainingError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:&'a CudaTensor1dPtr<f32,{NO*2}>) -> Result<CudaTensor1dPtr<f32,NO>,TrainingError> {
        let mut loss_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,NO>>::new(2,self.get_memory_pool())?;
        loss.memcpy_to(&mut loss_ptr,NO*2)?;

        let output_ptr = CudaTensor1dPtr::<f32,NO>::new(self.get_memory_pool())?;

        let loss_ptr = (&loss_ptr).try_into()?;

        let mut args = ReduceLinearBatchArgs::new(&loss_ptr,output_ptr,NO,2);

        let mut kernel = ReduceLinearBatch::<f32,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<f32>())?;

        Ok(args.output)
    }


    #[inline]
    fn batch_forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,
                                            input:HalfKPListView<'a,NI>)
         -> Result<CudaVec<f32,CudaTensor1dPtr<f32,{NO*2}>>,TrainingError> {
        let (indexes,boundaries):(Vec<size_t>,Vec<size_t>) = (&input).into();

        let len = input.size();

        let mut indexes_ptr = CudaMemoryPoolPtr::new(indexes.len(),self.get_memory_pool())?;
        let mut boundaries_ptr = CudaMemoryPoolPtr::new(boundaries.len(),self.get_memory_pool())?;

        indexes_ptr.memcpy(indexes.as_ptr(),indexes.len())?;
        boundaries_ptr.memcpy(boundaries.as_ptr(), boundaries.len())?;

        let output = CudaVec::<f32,CudaTensor1dPtr<f32,{NO*2}>>::new(len,self.get_memory_pool())?;

        let mut args = TransformFeaturesForwardBatchArgs::new(
            indexes_ptr,
            boundaries_ptr,
            CudaConstPtr::new(units),
            CudaConstPtr::new(bias),
            output,
            len);

        let mut kernel = TransformFeaturesForwardBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: (NO * 2 * len) as c_uint, y: 1, z: 1 },
                      dim3 { x: 32, y: 1, z: 1 },&mut args,0)?;

        Ok(args.output)
    }

    #[inline]
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPListView<'a,NI>,loss:&'a CudaVec<f32,CudaTensor1dPtr<f32,{NO*2}>>)
        -> Result<CudaTensor2dPtr<f32,NI,NO>,TrainingError> {
        let len = input.size();

        let input = <Box<[f32]>>::from(&input);

        let mut input_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,NI>>::new(len * 2,self.get_memory_pool())?;
        let mut output_ptr = CudaTensor2dPtr::<f32,NI,NO>::with_initializer(self.get_memory_pool(),Default::default)?;

        input_ptr.memcpy(input.as_ptr(),NI * 2 * len)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas().id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_T,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           (2 * len) as ::libc::c_int,
                           alpha.as_ptr(),
                           loss.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr)
            },
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(TrainingError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(TrainingError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(TrainingError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(TrainingError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:&'a CudaVec<f32,CudaTensor1dPtr<f32,{NO*2}>>) -> Result<CudaTensor1dPtr<f32,NO>,TrainingError> {
        let len = loss.size();

        let mut loss_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,NO>>::new(len * 2,self.get_memory_pool())?;
        loss.memcpy_to(&mut loss_ptr,len * 2 * NO)?;

        let output_ptr = CudaTensor1dPtr::<f32,NO>::new(self.get_memory_pool())?;

        let loss_ptr = (&loss_ptr).try_into()?;

        let mut args = ReduceLinearBatchArgs::new(&loss_ptr,output_ptr,NO,2 * len);

        let mut kernel = ReduceLinearBatch::<f32,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<f32>())?;

        Ok(args.output)
    }
}