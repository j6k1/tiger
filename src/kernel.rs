use std::ffi::c_void;
use std::fmt::Debug;
use std::marker::PhantomData;
use libc::size_t;
use nncombinator::cuda::{AsKernelPtr, CudaConstPtr, CudaMemoryPoolPtr, CudaTensor1dPtr, CudaTensor2dPtr, CudaVec, DataTypeInfo, Kernel, KernelArgs};
use nncombinator::ope::UnitValue;

extern "C" {
    fn forward_transform_features_batch_float(indexes: *const size_t, boundaries: *const size_t,
                                        units: *const f32, bias: *const f32, output: *mut f32,
                                        output_len: size_t, batch_size: size_t) -> c_void;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of Forward propagation of linear layers specialized for processing HalfKP.
pub struct TransformFeaturesForwardArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    indexes: CudaMemoryPoolPtr<size_t>,
    boundaries: CudaMemoryPoolPtr<size_t>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
    pub output: CudaTensor1dPtr<T,{NO*2}>,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer specialized for processing HalfKP.
impl<'a,T,const NI:usize,const NO:usize> TransformFeaturesForwardArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatchArgs instance
    /// # Arguments
    /// * `indexes` - Indexes at which the input resides
    /// * `boundaries` - Index Boundaries
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output
    /// * `batch_len` - batch_count
    pub fn new(indexes:CudaMemoryPoolPtr<size_t>,
               boundaries:CudaMemoryPoolPtr<size_t>,
               units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
               bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
               output:CudaTensor1dPtr<T,{NO*2}>) -> TransformFeaturesForwardArgs<'a,T,NI,NO> {
        TransformFeaturesForwardArgs {
            indexes: indexes,
            boundaries: boundaries,
            units: units,
            bias: bias,
            output: output,
            output_len: NO,
            batch_size: 2
        }
    }
}
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesForwardArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.indexes,
            &mut self.boundaries,
            &mut self.units,
            &mut self.bias,
            &mut self.output,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of forward propagation calculations for linear layers
pub struct TransformFeaturesForward<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> TransformFeaturesForward<'a,T,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatch instance
    pub fn new() -> TransformFeaturesForward<'a,T,NI,NO> {
        TransformFeaturesForward {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for TransformFeaturesForward<'a,f32,NI,NO> where [(); NO*2]: {
    const FUNC_PTR: *const c_void = forward_transform_features_batch_float as *const c_void;
    type Args = TransformFeaturesForwardArgs<'a,f32,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of Forward propagation of linear layers specialized for processing HalfKP.
pub struct TransformFeaturesForwardBatchArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    indexes: CudaMemoryPoolPtr<size_t>,
    boundaries: CudaMemoryPoolPtr<size_t>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,{NO*2}>>,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer specialized for processing HalfKP.
impl<'a,T,const NI:usize,const NO:usize> TransformFeaturesForwardBatchArgs<'a, T, NI, NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatchArgs instance
    /// # Arguments
    /// * `indexes` - Indexes at which the input resides
    /// * `boundaries` - Index Boundaries
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output
    /// * `batch_len` - batch_count
    pub fn new(indexes:CudaMemoryPoolPtr<size_t>,
               boundaries:CudaMemoryPoolPtr<size_t>,
               units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
               bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
               output:CudaVec<T,CudaTensor1dPtr<T,{NO*2}>>, batch_size: usize) -> TransformFeaturesForwardBatchArgs<'a, T, NI, NO> {
        TransformFeaturesForwardBatchArgs {
            indexes: indexes,
            boundaries: boundaries,
            units: units,
            bias: bias,
            output: output,
            output_len: NO,
            batch_size: batch_size * 2
        }
    }
}
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesForwardBatchArgs<'a, T, NI, NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.indexes,
            &mut self.boundaries,
            &mut self.units,
            &mut self.bias,
            &mut self.output,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of forward propagation calculations for linear layers
pub struct TransformFeaturesForwardBatch<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> TransformFeaturesForwardBatch<'a, T, NI, NO, >
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatch instance
    pub fn new() -> TransformFeaturesForwardBatch<'a, T, NI, NO> {
        TransformFeaturesForwardBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for TransformFeaturesForwardBatch<'a, f32, NI, NO> where [(); NO*2]: {
    const FUNC_PTR: *const c_void = forward_transform_features_batch_float as *const c_void;
    type Args = TransformFeaturesForwardBatchArgs<'a, f32, NI, NO>;
}
