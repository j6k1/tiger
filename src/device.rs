use libc::{c_uint};
use cuda_runtime_sys::dim3;
use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::cuda::{CudaPtr, DataTypeInfo, Kernel, Memory};
use nncombinator::device::{DeviceCpu, DeviceGpu};
use nncombinator::error::EvaluateError;
use nncombinator::mem::AsRawSlice;
use nncombinator::ope::UnitValue;
use crate::kernel::device::{FeaturesBatchCombine, FeaturesBatchCombineArgs, LossInputTransformToFeatures, LossInputTransformToFeaturesArgs};

pub trait DeviceFeatureTransform<U,const FEATURES_NUM: usize,const NO: usize> where U: UnitValue<U> {
    fn features_batch_combine<'a>(&self,self_output:&'a SerializedVec<U,Arr<U,NO>>,opponent_output:&'a SerializedVec<U,Arr<U,NO>>)
        -> Result<SerializedVec<U,Arr<U,{NO*2}>>,EvaluateError>;
    fn loss_input_transform_to_features<'a>(&self,combined_input:&'a SerializedVec<U,Arr<U,{NO*2}>>)
        -> Result<(SerializedVec<U,Arr<U,NO>>,SerializedVec<U,Arr<U,NO>>),EvaluateError>;
}
impl<U,const FEATURES_NUM:usize,const NO:usize> DeviceFeatureTransform<U,FEATURES_NUM,NO> for DeviceCpu<U> where U: UnitValue<U> {
    fn features_batch_combine<'a>(&self, self_output: &'a SerializedVec<U, Arr<U, NO>>, opponent_output: &'a SerializedVec<U, Arr<U, NO>>)
        -> Result<SerializedVec<U, Arr<U, { NO * 2 }>>, EvaluateError> {
        let mut combined:Vec<Arr<U,{NO*2}>> = Vec::with_capacity(self_output.len());

        for (s,o) in self_output.iter().zip(opponent_output.iter()) {
            let mut p = Vec::with_capacity(NO*2);

            p.extend_from_slice(s.as_raw_slice());
            p.extend_from_slice(o.as_raw_slice());

            combined.push(p.try_into()?);
        }

        Ok(combined.into())
    }

    fn loss_input_transform_to_features<'a>(&self, combined_input: &'a SerializedVec<U, Arr<U, { NO * 2 }>>) -> Result<(SerializedVec<U, Arr<U, NO>>, SerializedVec<U, Arr<U, NO>>), EvaluateError> {
        let len = combined_input.len();

        let mut sl:Vec<Arr<U,NO>> = Vec::with_capacity(len);
        let mut ol:Vec<Arr<U,NO>> = Vec::with_capacity(len);

        for loss in combined_input.iter() {
            let (s,o) = loss.as_raw_slice().split_at(NO);

            sl.push(s.to_vec().try_into()?);
            ol.push(o.to_vec().try_into()?);
        }

        Ok((sl.into(),ol.into()))
    }
}
impl<U,const FEATURES_NUM:usize,const NO:usize> DeviceFeatureTransform<U,FEATURES_NUM,NO> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          FeaturesBatchCombine<U>: Kernel<Args=FeaturesBatchCombineArgs<U>>,
          LossInputTransformToFeatures<U>: Kernel<Args=LossInputTransformToFeaturesArgs<U>> {
    fn features_batch_combine<'a>(&self, self_output: &'a SerializedVec<U, Arr<U, NO>>, opponent_output: &'a SerializedVec<U, Arr<U, NO>>)
        -> Result<SerializedVec<U, Arr<U, { NO * 2 }>>, EvaluateError> {
        let len = self_output.len();

        let combined_output: CudaPtr<U> = CudaPtr::new(NO * 2 * len)?;
        let mut self_output_ptr: CudaPtr<U> = CudaPtr::new(NO * len)?;
        let mut opponent_output_ptr: CudaPtr<U> = CudaPtr::new(NO * len)?;

        self_output_ptr.memcpy(self_output.as_raw_slice().as_ptr(), NO * len)?;
        opponent_output_ptr.memcpy(opponent_output.as_raw_slice().as_ptr(), NO * len)?;

        let mut args = FeaturesBatchCombineArgs::new(
            self_output_ptr,
            opponent_output_ptr,
            combined_output,
            NO,
            len);

        let mut kernel = FeaturesBatchCombine::<U>::new();

        kernel.launch(dim3 { x: (NO as c_uint * 2 + 32 - 1) / 32,
            y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.combined_output.read_to_vec()?.try_into()?)
    }

    fn loss_input_transform_to_features<'a>(&self, combined_input: &'a SerializedVec<U, Arr<U, { NO * 2 }>>)
        -> Result<(SerializedVec<U, Arr<U, NO>>, SerializedVec<U, Arr<U, NO>>), EvaluateError> {
        let len = combined_input.len();

        let mut combined_input_ptr: CudaPtr<U> = CudaPtr::new(NO * 2 * len)?;
        let self_input: CudaPtr<U> = CudaPtr::new(NO * len)?;
        let opponent_input: CudaPtr<U> = CudaPtr::new(NO * len)?;

        combined_input_ptr.memcpy(combined_input.as_raw_slice().as_ptr(), NO * 2 * len)?;

        let mut args = LossInputTransformToFeaturesArgs::new(
            self_input,
            opponent_input,
            combined_input_ptr,
            NO,
            len);

        let mut kernel = LossInputTransformToFeatures::<U>::new();

        kernel.launch(dim3 { x: (NO as c_uint * 2 + 32 - 1) / 32,
            y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok((args.self_input.read_to_vec()?.try_into()?,args.opponent_input.read_to_vec()?.try_into()?))
    }
}