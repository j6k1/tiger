use std::marker::PhantomData;
use nncombinator::cuda::{AsMutKernelPtr, CudaPtr, DataTypeInfo, Kernel, KernelArgs};
use libc::{size_t, c_void};

extern "C" {
    fn features_batch_combine_float(self_output: *const f32, opponent_output: *const f32, combined_output: *mut f32, nlen: size_t, batch_size: size_t) -> c_void;
    fn features_batch_combine_double(self_output: *const f64, opponent_output: *const f64, combined_output: *mut f64, nlen: size_t, batch_size: size_t) -> c_void;
    fn loss_input_transform_to_features_float(self_input: *mut f32, opponent_input: *mut f32, combined_input: *const f32, nlen: size_t, batch_size: size_t) -> c_void;
    fn loss_input_transform_to_features_double(self_input: *mut f64, opponent_input: *mut f64, combined_input: *const f64, nlen: size_t, batch_size: size_t) -> c_void;
}
/// Defines the arguments passed to the cuda kernel function
/// that converts the output of the FeatureTransformLayer into input to the underlying LinearLayer.
pub struct FeaturesBatchCombineArgs<T> where T: DataTypeInfo {
    self_output: CudaPtr<T>,
    opponent_output: CudaPtr<T>,
    /// output
    pub combined_output: CudaPtr<T>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing
/// the list of arguments to be passed to the Cuda kernel that transforms the output of the FeatureTransformLayer.
impl<T> FeaturesBatchCombineArgs<T> where T: DataTypeInfo {
    /// Create a FeaturesBatchCombineArgs instance
    /// # Arguments
    /// * `self_output` - active side output
    /// * `opponent_output` - non-active side output
    /// * `combined_output` - combined_output
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch_count
    pub fn new(self_output:CudaPtr<T>,opponent_output:CudaPtr<T>,combined_output:CudaPtr<T>,out_len:usize,batch_len:usize)
        -> FeaturesBatchCombineArgs<T> {
        FeaturesBatchCombineArgs {
            self_output: self_output,
            opponent_output: opponent_output,
            combined_output: combined_output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for FeaturesBatchCombineArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.self_output,
            &mut self.opponent_output,
            &mut self.combined_output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
/// Implementation of transforms the output of the FeatureTransformLayers
pub struct FeaturesBatchCombine<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> FeaturesBatchCombine<T> where T: DataTypeInfo {
    /// Create a FeaturesBatchCombine instance
    pub fn new() -> FeaturesBatchCombine<T> {
        FeaturesBatchCombine {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for FeaturesBatchCombine<f32> {
    const FUNC_PTR: *const c_void = features_batch_combine_float as *const c_void;
    type Args = FeaturesBatchCombineArgs<f32>;
}
impl Kernel for FeaturesBatchCombine<f64> {
    const FUNC_PTR: *const c_void = features_batch_combine_double as *const c_void;
    type Args = FeaturesBatchCombineArgs<f64>;
}
/// Define the arguments passed to the cuda kernel function
/// Convert the output of the lower LinearLayer loss calculation to the loss of the inner layer of the FeatureTransformLayer.
pub struct LossInputTransformToFeaturesArgs<T> where T: DataTypeInfo {
    self_input: CudaPtr<T>,
    opponent_input: CudaPtr<T>,
    /// output
    pub combined_input: CudaPtr<T>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing a list of arguments to be passed to the Cuda kernel
/// that transforms the input loss from the lower layers of the FeatureTransformLayer
impl<T> LossInputTransformToFeaturesArgs<T> where T: DataTypeInfo {
    /// Create a LossInputTransformToFeaturesArgs instance
    /// # Arguments
    /// * `self_input` - active side loss output
    /// * `opponent_input` - non-active side loss output
    /// * `combined_input` - combined_input
    /// * `out_len` - Number of scalar values in forward output
    /// * `batch_len` - batch_count
    pub fn new(self_input:CudaPtr<T>,opponent_input:CudaPtr<T>,combined_input:CudaPtr<T>,out_len:usize,batch_len:usize)
        -> LossInputTransformToFeaturesArgs<T> {
        LossInputTransformToFeaturesArgs {
            self_input: self_input,
            opponent_input: opponent_input,
            combined_input: combined_input,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LossInputTransformToFeaturesArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.self_input,
            &mut self.opponent_input,
            &mut self.combined_input,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
/// Implementation of loss of the LinearLayer to the loss of the FeatureTransformLayer.
pub struct LossInputTransformToFeatures<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> LossInputTransformToFeatures<T> where T: DataTypeInfo {
    /// Create a LossInputTransformToFeatures instance
    pub fn new() -> LossInputTransformToFeatures<T> {
        LossInputTransformToFeatures {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LossInputTransformToFeatures<f32> {
    const FUNC_PTR: *const c_void = loss_input_transform_to_features_float as *const c_void;
    type Args =LossInputTransformToFeaturesArgs<f32>;
}
impl Kernel for LossInputTransformToFeatures<f64> {
    const FUNC_PTR: *const c_void = loss_input_transform_to_features_double as *const c_void;
    type Args = LossInputTransformToFeaturesArgs<f64>;
}
