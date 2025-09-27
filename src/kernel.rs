use std::ffi::c_void;
use std::fmt::Debug;
use std::marker::PhantomData;
use libc::size_t;
use nncombinator::cuda::{AsKernelPtr, CudaConstPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs};
use nncombinator::cuda::allocator::CudaAllocator;
use nncombinator::ope::UnitValue;

extern "C" {
    fn forward_transform_features_batch_float(indexes: *const size_t, boundaries: *const size_t,
                                        units: *const f32, bias: *const f32, output: *mut f32,
                                        output_len: size_t, batch_size: size_t) -> c_void;
    fn transform_features_gradient_batch_float(loss: *const f32,indexes: *const size_t,
                                               boundaries: *const size_t,
                                               output: *mut f32,
                                               input_len: size_t,
                                               output_len: size_t,
                                               batch_size: size_t) -> c_void;
    fn transform_features_input_to_bits(indexes: *const size_t,
                                        boundaries: *const size_t,
                                        bits: *mut u8,
                                        input_len: size_t,
                                        batch_size: size_t) -> c_void;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of Forward propagation of linear layers specialized for processing HalfKP.
pub struct TransformFeaturesForwardArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    indexes: CudaPtr<size_t,A>,
    boundaries: CudaPtr<size_t,A>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
    pub output: CudaTensor1dPtr<T,A,{NO*2}>,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer specialized for processing HalfKP.
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesForwardArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatchArgs instance
    /// # Arguments
    /// * `indexes` - Indexes at which the input resides
    /// * `boundaries` - Index Boundaries
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output
    /// * `batch_len` - batch_count
    pub fn new(indexes:CudaPtr<size_t,A>,
               boundaries:CudaPtr<size_t,A>,
               units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
               bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
               output:CudaTensor1dPtr<T,A,{NO*2}>) -> TransformFeaturesForwardArgs<'a,T,A,NI,NO> {
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
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesForwardArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
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
pub struct TransformFeaturesForward<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    t:PhantomData<T>,
    a:PhantomData<A>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesForward<'a,T,A,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatch instance
    pub fn new() -> TransformFeaturesForward<'a,T,A,NI,NO> {
        TransformFeaturesForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for TransformFeaturesForward<'a,f32,A,NI,NO>
    where A: CudaAllocator + 'a,
          [(); NO*2]: {
    const FUNC_PTR: *const c_void = forward_transform_features_batch_float as *const c_void;
    type Args = TransformFeaturesForwardArgs<'a,f32,A,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of Forward propagation of linear layers specialized for processing HalfKP.
pub struct TransformFeaturesForwardBatchArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    indexes: CudaPtr<size_t,A>,
    boundaries: CudaPtr<size_t,A>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,{NO*2}>,A>,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer specialized for processing HalfKP.
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesForwardBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatchArgs instance
    /// # Arguments
    /// * `indexes` - Indexes at which the input resides
    /// * `boundaries` - Index Boundaries
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output
    /// * `batch_len` - batch_count
    pub fn new(indexes:CudaPtr<size_t,A>,
               boundaries:CudaPtr<size_t,A>,
               units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
               bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
               output:CudaVec<T,CudaTensor1dPtr<T,A,{NO*2}>,A>, batch_size: usize) -> TransformFeaturesForwardBatchArgs<'a,T,A,NI,NO> {
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
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesForwardBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
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
pub struct TransformFeaturesForwardBatch<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    t:PhantomData<T>,
    a:PhantomData<A>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesForwardBatch<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatch instance
    pub fn new() -> TransformFeaturesForwardBatch<'a,T,A,NI,NO> {
        TransformFeaturesForwardBatch {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for TransformFeaturesForwardBatch<'a,f32,A,NI,NO>
    where A: CudaAllocator + 'a,
          [(); NO*2]: {
    const FUNC_PTR: *const c_void = forward_transform_features_batch_float as *const c_void;
    type Args = TransformFeaturesForwardBatchArgs<'a,f32,A,NI,NO>;
}
/// Define the list passed to the CUDA kernel function as an argument
/// for calculating the weight update amount for the linear layer specialized for HalfKP processing.
pub struct TransformFeaturesGradientArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,{NO*2}>>,
    input: CudaPtr<u8,A>,
    pub output: CudaTensor2dPtr<T,A,NI,NO>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// During the calculation of weight updates for the linear layer specialized
/// for HalfKP processing, create an instance of the object representing the argument list.
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesGradientArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesForwardBatchArgs instance
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - input (bits)
    /// * `output` - output
    /// * `input_len` - input size
    /// * `output_len` - output size
    /// * `batch_len` - batch_count
    pub fn new(loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,{NO*2}>>,
               input: CudaPtr<u8,A>,
               output:CudaTensor2dPtr<T,A,NI,NO>) -> TransformFeaturesGradientArgs<'a,T,A,NI,NO> {
        TransformFeaturesGradientArgs {
            loss: loss,
            input: input,
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: 2
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesGradientArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.input,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of Weight Update Calculation in Linear Layers
pub struct TransformFeaturesGradient<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>, [(); NO*2]: {
    t:PhantomData<T>,
    a:PhantomData<A>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesGradient<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesGradientBatch instance
    pub fn new() -> TransformFeaturesGradient<'a,T,A,NI,NO> {
        TransformFeaturesGradient {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for TransformFeaturesGradient<'a,f32,A,NI,NO>
    where A: CudaAllocator + 'a,
          [(); NO*2]: {
    const FUNC_PTR: *const c_void = transform_features_gradient_batch_float as *const c_void;
    type Args = TransformFeaturesGradientArgs<'a, f32, A, NI, NO>;
}
/// Define the list passed to the CUDA kernel function as an argument
/// for calculating the weight update amount of the linear layer specialized for HalfKP processing.
pub struct TransformFeaturesGradientBatchArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,{NO*2}>>>,
    input: CudaPtr<u8,A>,
    pub output: CudaTensor2dPtr<T,A,NI,NO>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// During the calculation of weight updates for the linear layer specialized
/// for HalfKP processing, create an instance of the object representing the argument list.
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesGradientBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesGradientBatchArgs instance
    /// * `loss` - loss
    /// * `input` - input (bits)
    /// * `output` - output
    /// * `input_len` - input size
    /// * `output_len` - output size
    /// * `batch_len` - batch_count
    pub fn new(loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,{NO*2}>>>,
               input: CudaPtr<u8,A>,
               output:CudaTensor2dPtr<T,A,NI,NO>,
               batch_size: usize) -> TransformFeaturesGradientBatchArgs<'a,T,A,NI,NO> {
        TransformFeaturesGradientBatchArgs {
            loss: loss,
            input: input,
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: batch_size * 2
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for TransformFeaturesGradientBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.input,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of Weight Update Calculation in Linear Layers Specialized for HalfKP
pub struct TransformFeaturesGradientBatch<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    t:PhantomData<T>,
    a:PhantomData<A>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> TransformFeaturesGradientBatch<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a,
          [(); NO*2]: {
    /// Create a TransformFeaturesGradientBatch instance
    pub fn new() -> TransformFeaturesGradientBatch<'a,T,A,NI,NO> {
        TransformFeaturesGradientBatch {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for TransformFeaturesGradientBatch<'a,f32,A,NI,NO>
    where A: CudaAllocator + 'a,
          [(); NO*2]: {
    const FUNC_PTR: *const c_void = transform_features_gradient_batch_float as *const c_void;
    type Args = TransformFeaturesGradientBatchArgs<'a,f32,A,NI,NO>;
}
/// Expand sparse inputs into bit vector format inputs.
pub struct TransformFeaturesInputToBitsArgs<A,const NI:usize>
    where A: CudaAllocator {
    indexes: CudaPtr<size_t,A>,
    boundaries: CudaPtr<size_t,A>,
    pub bits: CudaPtr<u8,A>,
    input_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list passed to a CUDA kernel that expands sparse inputs into bit vectors.
impl<A,const NI:usize> TransformFeaturesInputToBitsArgs<A,NI>
    where A: CudaAllocator {
    /// Create a TransformFeaturesInputToBitsArgs instance
    /// # Arguments
    /// * `indexes` - Indexes at which the input resides
    /// * `boundaries` - Index Boundaries
    /// * `bits` - Input bits
    /// * `input_len` - Input size
    /// * `batch_len` - batch_count
    pub fn new(indexes:CudaPtr<size_t,A>,
               boundaries:CudaPtr<size_t,A>,
               bits:CudaPtr<u8,A>,
               batch_size: usize) -> TransformFeaturesInputToBitsArgs<A,NI> {
        TransformFeaturesInputToBitsArgs {
            indexes: indexes,
            boundaries: boundaries,
            bits: bits,
            input_len: NI,
            batch_size: batch_size
        }
    }
}
impl<A,const NI:usize> KernelArgs for TransformFeaturesInputToBitsArgs<A,NI>
    where A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.indexes,
            &mut self.boundaries,
            &mut self.bits,
            &mut self.input_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of sparse input conversion to bit vectors
pub struct TransformFeaturesInputToBits<A,const NI:usize> {
    a:PhantomData<A>,
    ni:PhantomData<[();NI]>
}
impl<A,const NI:usize> TransformFeaturesInputToBits<A,NI>
    where A: CudaAllocator {
    /// Create a TransformFeaturesInputToBits instance
    pub fn new() -> TransformFeaturesInputToBits<A,NI> {
        TransformFeaturesInputToBits {
            a: PhantomData::<A>,
            ni:PhantomData::<[();NI]>
        }
    }
}
impl<A,const NI:usize> Kernel for TransformFeaturesInputToBits<A,NI>
    where A: CudaAllocator {
    const FUNC_PTR: *const c_void = transform_features_input_to_bits as *const c_void;
    type Args = TransformFeaturesInputToBitsArgs<A,NI>;
}
