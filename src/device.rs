use std::mem;

use libc::c_int;
use libc::c_uint;
use cuda_runtime_sys::dim3;
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use rcublas_sys::cublasSscal_v2;
use rcublas_sys::{cublasStatus_t};

use nncombinator::arr::SerializedVec;
use nncombinator::arr::SerializedVecView;
use nncombinator::cuda::kernel::device::{BackwardLinearBatch, BackwardLinearBatchArgs, ForwardLinearBatch, ForwardLinearBatchArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use nncombinator::mem::{AsRawMutSlice, AsRawSlice};
use nncombinator::arr::{Arr, Arr2, ArrView};
use nncombinator::cuda::{AsMutPtr, AsPtr, CudaConstPtr, CudaMemoryPoolPtr, CudaPtr, CudaTensor1dPtr, CudaTensor2dPtr, ffi, Kernel, Memory};
use nncombinator::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool};
use nncombinator::error::{EvaluateError, TrainingError};
use nncombinator::ope::UnitValue;

use crate::features::{HalfKP, HalfKPView};

pub trait DeviceFeatureTransform<U,T,B,const NI: usize,const NO: usize> where U: UnitValue<U> {
    fn forward_feature_transform<'a>(&self,bias:&B,units:&T,input:HalfKPView<'a,U,NI>) -> Result<Arr<U,{NO*2}>,EvaluateError>;
    fn backward_feature_transform<'a>(&self,units:&T,input:ArrView<'a,U,{NO*2}>) -> Result<HalfKP<U,NI>,TrainingError>;
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,U,NI>,loss:ArrView<'a,U,{NO*2}>) -> Result<T,TrainingError>;
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:ArrView<'a,U,{NO*2}>) -> Result<B,TrainingError>;
    fn batch_forward_feature_transform<'a>(&self,bias:&B,units:&T,input:SerializedVecView<'a,U,HalfKP<U,NI>>) -> Result<SerializedVec<U,Arr<U,{NO*2}>>,TrainingError>;
    fn batch_backward_feature_transform<'a>(&self,units:&T,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>) -> Result<SerializedVec<U,HalfKP<U,NI>>,TrainingError>;
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:SerializedVecView<'a,U,HalfKP<U,NI>>,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>) -> Result<T,TrainingError>;
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>) -> Result<B,TrainingError>;
}
impl<U,const NI: usize,const NO:usize> DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U> 
    where U: UnitValue<U> {

    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:HalfKPView<'a,U,NI>) 
    -> Result<Arr<U,{NO*2}>,EvaluateError> {
        let mut r = Vec::with_capacity(NO*2);

        r.extend_from_slice(&bias);
        r.extend_from_slice(&bias);

        for (input,offset) in input.iter().zip([0,NO]) {
            let it = input.iter().enumerate().filter(|(_,&v)| {
                v > U::default()
            }).map(|(i,_)| i);
            
            for i in it {
                for j in 0..NO {
                    r[j + offset] += units[(i,j)];
                }
            }
        }
        
        Ok(r.try_into()?)
    }

    #[inline]
    fn backward_feature_transform<'a>(&self,units:&Arr2<U,NI,NO>,input:ArrView<'a,U,{NO*2}>) -> Result<HalfKP<U,NI>,TrainingError> {
        let mut sr = Arr::<U,NI>::new();
        let mut or = Arr::<U,NI>::new();

        let(s,o) = input.as_raw_slice().split_at(NO);

        let s = <&[U;NO]>::try_from(s)?;

        for (i,w) in (0..NI).zip(units.iter()) {
            let w = <&[U;NO]>::try_from(w.as_raw_slice())?;
            let mut r = U::default();

            for j in 0..NO {
                r +=  s[j] * w[j];
            }

            sr[i] = r;
        }

        let o = <&[U;NO]>::try_from(o)?;

        for (i,w) in (0..NI).zip(units.iter()) {
            let w = <&[U;NO]>::try_from(w.as_raw_slice())?;
            let mut r = U::default();

            for j in 0..NO {
                r +=  o[j] * w[j];
            }

            or[i] = r;
        }

        Ok(HalfKP::new(sr, or))
    }

    #[inline]
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,U,NI>,loss:ArrView<'a,U,{NO*2}>) -> Result<Arr2<U,NI,NO>,TrainingError> {
        let mut acc = Arr2::<U,NI,NO>::new();

        let d = U::from_f64(2.).unwrap();

        let loss = loss.iter().map(|&l| l / d).collect::<Vec<U>>();
        let (sl,ol) = loss.split_at(NO);

        let sl = <&[U;NO]>::try_from(sl)?;
        let ol = <&[U;NO]>::try_from(ol)?;

        for ((input,l),mut acc) in input.iter().zip([sl,ol]).zip(acc.iter_mut()) {
            let acc = <&mut [U;NO]>::try_from(acc.as_raw_mut_slice())?;

            for &input in input.iter() {
                for i in 0..NO {
                    acc[i] += input * l[i];
                }
            }
        }

        Ok(acc)
    }

    #[inline]
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:ArrView<'a,U,{NO*2}>) -> Result<Arr<U,NO>,TrainingError> {
        let mut acc = Arr::<U,NO>::new();

        {
            let d = U::from_f64(2.).unwrap();

            let acc = <&mut [U;NO]>::try_from(acc.as_raw_mut_slice())?;

            let (sl,ol) = loss.as_raw_slice().split_at(NO);

            for i in 0..NO {
                acc[i] += sl[i] / d;
            }

            for i in 0..NO {
                acc[i] += ol[i] / d;
            }        
        }

        Ok(acc)
    }

    #[inline]
    fn batch_forward_feature_transform<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:SerializedVecView<'a,U,HalfKP<U,NI>>)
         -> Result<SerializedVec<U,Arr<U,{NO*2}>>,TrainingError> {
        
        Ok(input.par_iter().map(|input| {
            self.forward_feature_transform(bias, units, input)
        }).collect::<Result<Vec<Arr<U,{NO*2}>>,EvaluateError>>()?.into())
    }

    #[inline]
    fn batch_backward_feature_transform<'a>(&self,units:&Arr2<U,NI,NO>,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>) 
         -> Result<SerializedVec<U,HalfKP<U,NI>>,TrainingError> {
        
        Ok(loss.par_iter().map(|loss| {
            self.backward_feature_transform(units, loss)
        }).collect::<Result<Vec<HalfKP<U,NI>>,TrainingError>>()?.into())
    }

    #[inline]
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:SerializedVecView<'a,U,HalfKP<U,NI>>,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>)
        -> Result<Arr2<U,NI,NO>,TrainingError> {
        
        input.par_iter().zip(loss.par_iter()).map(|(i,l)| {
            self.backward_feature_transform_weight_gradient(i, l)
        }).reduce(|| Ok(Arr2::new()), | acc, g | {
            acc.and_then(|mut acc| g.and_then(|g| {
                for (mut acc,g) in acc.iter_mut().zip(g.iter()) {
                    for (acc,&g) in acc.iter_mut().zip(g.iter()) {
                        *acc += g;
                    }
                }

                Ok(acc)
            }))
        })
    }

    #[inline]
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:SerializedVecView<'a,U,Arr<U,{NO*2}>>) -> Result<Arr<U,NO>,TrainingError> {
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

            let acc = <&mut [U;NO]>::try_from(acc.as_raw_mut_slice())?;

            let (sl,ol) = g.as_raw_slice().split_at(NO);

            for i in 0..NO {
                acc[i] += sl[i] / d;
            }

            for i in 0..NO {
                acc[i] += ol[i] / d;
            }        
        }

        Ok(acc)
    }
}
impl<const NI: usize,const NO:usize> DeviceFeatureTransform<f32,CudaTensor2dPtr<f32,NI,NO>,CudaTensor1dPtr<f32,NO>,NI,NO> for DeviceGpu<f32>
    where for<'a> ForwardLinearBatch::<'a,f32,NI,NO>: Kernel<Args=ForwardLinearBatchArgs<'a,f32,NI,NO>>,
          for<'a> BackwardLinearBatch::<'a,f32,NI,NO>: Kernel<Args=BackwardLinearBatchArgs<'a,f32,NI,NO>>,
          LinearGradientBatch::<f32,NI,NO>: Kernel<Args=LinearGradientBatchArgs<f32,NI,NO>>,
          ReduceLinearBatch::<f32,NO>: Kernel<Args=ReduceLinearBatchArgs<f32,NO>> {
        
    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,input:HalfKPView<'a,f32,NI>)
        -> Result<Arr<f32,{NO*2}>,EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * 2,self.get_memory_pool())?;
        let output = CudaMemoryPoolPtr::new(NO * 2,self.get_memory_pool())?;

        input_ptr.memcpy(input.as_ptr(),NI * 2)?;

        let mut args = ForwardLinearBatchArgs::new(input_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output,
                                                   2);

        let mut kernel = ForwardLinearBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: (NI as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * 2 * mem::size_of::<f32>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn backward_feature_transform<'a>(&self,units:&CudaTensor2dPtr<f32,NI,NO>,input:ArrView<'a,f32,{NO*2}>) -> Result<HalfKP<f32,NI>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO * 2,self.get_memory_pool())?;
        let output = CudaMemoryPoolPtr::new(NI * 2,self.get_memory_pool())?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO * 2)?;

        let mut args = BackwardLinearBatchArgs::new(input_ptr,
                                                    CudaConstPtr::new(units),
                                                    output,
                                                    2);

        let mut kernel = BackwardLinearBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: NI as c_uint, y: 1, z: (NO as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn backward_feature_transform_weight_gradient<'a>(&self,input:HalfKPView<'a,f32,NI>,loss:ArrView<'a,f32,{NO*2}>)
        -> Result<CudaTensor2dPtr<f32,NI,NO>,TrainingError> {
        
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * 2,self.get_memory_pool())?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * 2,self.get_memory_pool())?;
        let output_ptr = CudaTensor2dPtr::<f32,NI,NO>::new(self.get_memory_pool())?;

        input_ptr.memcpy(input.as_ptr(),NI * 2)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * 2)?;

        let m = CudaPtr::try_from(0.5)?;

        match unsafe {
            cublasSscal_v2(*self.cublas().id_c(),
                (NO * 2) as c_int,
                m.as_ptr(),
                loss_ptr.as_mut_ptr(),
                1 as c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
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

        ffi::device_synchronize()?;

        let mut args = LinearGradientBatchArgs::new(
            loss_ptr,
            input_ptr,
            output_ptr,
            2
        );

        let mut kernel = LinearGradientBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output)
    }

    #[inline]
    fn backward_feature_transform_bias_gradient<'a>(&self,loss:ArrView<'a,f32,{NO*2}>) -> Result<CudaTensor1dPtr<f32,NO>,TrainingError> {
        let mut loss_ptr = CudaPtr::new(NO * 2).unwrap();
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * 2).unwrap();

        let m = CudaPtr::try_from(0.5)?;

        match unsafe {
            cublasSscal_v2(*self.cublas().id_c(),
                (NO * 2) as c_int,
                m.as_ptr(),
                loss_ptr.as_mut_ptr(),
                1 as c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
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

        ffi::device_synchronize()?;

        let output_ptr = CudaTensor1dPtr::<f32,NO>::new(self.get_memory_pool()).unwrap();

        let mut args = ReduceLinearBatchArgs::new(loss_ptr,output_ptr,NO,2);

        let mut kernel = ReduceLinearBatch::<f32,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output)
    }


    #[inline]
    fn batch_forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,
                                            input:SerializedVecView<'a,f32,HalfKP<f32,NI>>)
         -> Result<SerializedVec<f32,Arr<f32,{NO*2}>>,TrainingError> {
        let len = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NI * 2 * len,self.get_memory_pool())?;
        let output = CudaMemoryPoolPtr::new(NO * 2 * len,self.get_memory_pool())?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * 2 * len)?;

        let mut args = ForwardLinearBatchArgs::new(input_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output,
                                                   2 * len);

        let mut kernel = ForwardLinearBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: (NO * len) as c_uint, y: 1, z: (NI as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * 2 * mem::size_of::<f32>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn batch_backward_feature_transform<'a>(&self,units:&CudaTensor2dPtr<f32,NI,NO>,loss:SerializedVecView<'a,f32,Arr<f32,{NO*2}>>)
        -> Result<SerializedVec<f32,HalfKP<f32,NI>>,TrainingError> {
        let len = loss.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NO * 2 * len,self.get_memory_pool())?;
        let output = CudaMemoryPoolPtr::new(NI * 2 * len,self.get_memory_pool())?;

        input_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * 2 * len)?;

        let mut args = BackwardLinearBatchArgs::new(input_ptr,
                                                    CudaConstPtr::new(units),
                                                    output,
                                                    2 * len);

        let mut kernel = BackwardLinearBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * len) as c_uint, y: 1, z: (NO as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output.read_to_vec()?.into_boxed_slice().try_into()?)
    }       

    #[inline]
    fn batch_backward_feature_transform_weight_gradient<'a>(&self,input:SerializedVecView<'a,f32,HalfKP<f32,NI>>,loss:SerializedVecView<'a,f32,Arr<f32,{NO*2}>>)
        -> Result<CudaTensor2dPtr<f32,NI,NO>,TrainingError> {
        let len = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NI * 2 * len,self.get_memory_pool())?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * 2 * len,self.get_memory_pool())?;
        let output_ptr = CudaTensor2dPtr::<f32,NI,NO>::new(self.get_memory_pool())?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * 2 * len)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * 2 * len)?;

        let m = CudaPtr::try_from(0.5)?;

        match unsafe {
            cublasSscal_v2(*self.cublas().id_c(),
                           (NO * 2 * len) as c_int,
                           m.as_ptr(),
                           loss_ptr.as_mut_ptr(),
                           1 as c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
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

        ffi::device_synchronize()?;

        let mut args = LinearGradientBatchArgs::new(
            loss_ptr,
            input_ptr,
            output_ptr,
            len * 2
        );

        let mut kernel = LinearGradientBatch::<f32,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: (len as c_uint * 2 + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output)
    }

    #[inline]
    fn batch_feature_transform_bias_gradient<'a>(&self,loss:SerializedVecView<'a,f32,Arr<f32,{NO*2}>>) -> Result<CudaTensor1dPtr<f32,NO>,TrainingError> {
        let len = loss.len();

        let mut loss_ptr = CudaPtr::new(NO * 2 * len).unwrap();
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * 2 * len).unwrap();

        let m = CudaPtr::try_from(0.5)?;

        match unsafe {
            cublasSscal_v2(*self.cublas().id_c(),
                           (NO * 2 * len) as c_int,
                           m.as_ptr(),
                           loss_ptr.as_mut_ptr(),
                           1 as c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
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

        ffi::device_synchronize()?;

        let output_ptr = CudaTensor1dPtr::<f32,NO>::new(self.get_memory_pool()).unwrap();

        let mut args = ReduceLinearBatchArgs::new(loss_ptr,output_ptr,NO,2 * len);

        let mut kernel = ReduceLinearBatch::<f32,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: (len as c_uint * 2) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>())?;

        Ok(args.output)
    }
}