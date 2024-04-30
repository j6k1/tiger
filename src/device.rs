use std::iter;

use nncombinator::mem::AsRawSlice;
use rcublas_sys::{cublasOperation_t, cublasSgemm_v2, cublasStatus_t};

use nncombinator::arr::{Arr, Arr2, ArrView};
use nncombinator::cuda::{AsMutPtr, AsPtr, CudaMemoryPoolPtr, CudaPtr, CudaTensor1dPtr, CudaTensor2dPtr, Memory, MemoryMoveTo};
use nncombinator::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool};
use nncombinator::error::{EvaluateError, TrainingError};
use nncombinator::ope::UnitValue;

use crate::features::{HalfKP, HalfKPView};

pub trait DeviceFeatureTransform<U,T,B,const NI: usize,const NO: usize> where U: UnitValue<U> {
    fn forward_feature_transform<'a>(&self,bias:&B,units:&T,input:HalfKPView<'a,U,NI>) -> Result<Arr<U,{NO*2}>,EvaluateError>;
    fn backward_feature_transform<'a>(&self,units:&T,input:ArrView<'a,U,{NO*2}>) -> Result<HalfKP<U,NI>,TrainingError>;
}
impl<U,const NI: usize,const NO:usize> DeviceFeatureTransform<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U> 
    where U: UnitValue<U> {

    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:HalfKPView<'a,U,NI>) 
    -> Result<Arr<U,{NO*2}>,EvaluateError> {
        let mut r = Arr::<U,{NO*2}>::new();

        for input in input.iter() {
            for i in 0..NO {
                let w = units.iter().map(|w| w[i]).collect::<Vec<U>>();
                let w = <&[U;NI]>::try_from(w.as_slice())?;

                for index in 0..NI {
                    r[i] += input[index] * w[index];
                }
            }
        }

        Ok(r)
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
                r +=  s[i] * w[i];
            }

            sr[i] = r;
        }

        let o = <&[U;NO]>::try_from(o)?;

        for (i,w) in (0..NI).zip(units.iter()) {
            let w = <&[U;NO]>::try_from(w.as_raw_slice())?;
            let mut r = U::default();

            for j in 0..NO {
                r +=  o[i] * w[i];
            }

            or[i] = r;
        }

        Ok(HalfKP::new(sr, or))
    }
}
impl<const NI: usize,const NO:usize> DeviceFeatureTransform<f32,CudaTensor2dPtr<f32,NI,NO>,CudaTensor1dPtr<f32,NO>,NI,NO> for DeviceGpu<f32> {
    #[inline]
    fn forward_feature_transform<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,input:HalfKPView<'a,f32,NI>) 
    -> Result<Arr<f32,{NO*2}>,EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * 2 ,self.get_memory_pool())?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO * 2,self.get_memory_pool())?;

        let bias = iter::repeat(bias.read_to_vec()?.into_boxed_slice().iter().cloned().collect::<Vec<f32>>())
                                    .take(2).collect::<Vec<Vec<f32>>>()
                                    .into_iter().flatten().collect::<Vec<f32>>();

        input_ptr.memcpy(input.as_ptr(),NI * 2)?;
        output_ptr.memcpy(bias.as_slice().as_ptr(),NO * 2)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas().id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           2 as libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr.read_to_vec()?.try_into()?)
            },
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(EvaluateError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(EvaluateError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(EvaluateError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(EvaluateError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn backward_feature_transform<'a>(&self,units:&CudaTensor2dPtr<f32,NI,NO>,input:ArrView<'a,f32,{NO*2}>) -> Result<HalfKP<f32,NI>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO*2,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI*2,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO*2)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas().id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as ::libc::c_int,
                           2 as libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.read_to_vec()?.try_into()?),
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
}