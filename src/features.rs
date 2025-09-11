use std::fmt::Debug;
use libc::size_t;
use nncombinator::cuda::allocator::CudaAllocator;
use nncombinator::cuda::ToCuda;
use nncombinator::device::DeviceGpu;
use nncombinator::error::TypeConvertError;
use nncombinator::layer::{BatchDataType, BatchSize};
use nncombinator::ope::UnitValue;
use rand_distr::num_traits::FromPrimitive;

/// InputFeatures Implementaion
#[derive(Debug)]
pub struct HalfKP<const N:usize> {
    s:Vec<usize>,
    o:Vec<usize>
}
impl<const N:usize> BatchDataType for HalfKP<N> {
    type Type = HalfKPList<N>;
}
impl<U,A,const N:usize> ToCuda<U,A> for HalfKP<N>
    where U: UnitValue<U>,
          A: CudaAllocator {
    type Output = Self;

    fn to_cuda(self, _: &DeviceGpu<U,A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self)
    }
}
impl<const N:usize> HalfKP<N> {
    /// Create an instance of HalfKP
    pub fn new(s:Vec<usize>,o:Vec<usize>) -> HalfKP<N> {
        HalfKP {
            s:s,
            o:o
        }
    }
    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> HalfKPIter<'a,N> {
        HalfKPIter{ s: &self.s, o: &self.o, index: 0 }
    }

    pub fn to_vec<T>(&self) -> Box<[T]>
        where T: Debug + Clone + Default + Send + Sync + FromPrimitive {
        let mut arr = vec![T::default();N*2].into_boxed_slice();

        for &i in self.s.iter() {
            arr[i] = T::from_f64(1.).unwrap();
        }

        for &i in self.o.iter() {
            arr[N + i] = T::from_f64(1.).unwrap();
        }

        arr
    }
}
impl<const N:usize> Clone for HalfKP<N> {
    fn clone(&self) -> Self {
        HalfKP {
            s:self.s.clone(),
            o:self.o.clone()
        }
    }
}
/// Implementation of an immutable iterator for HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPIter<'a,const N:usize> {
    s: &'a Vec<usize>,
    o: &'a Vec<usize>,
    index: usize
}
impl<'a,const N:usize> Iterator for HalfKPIter<'a,N> {
    type Item = &'a Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == 0 {
            self.index += 1;

            Some(self.s)
        } else if self.index == 1 {
            self.index += 1;

            Some(self.o)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;

        if self.index == 0 {
            self.index += 1;

            Some(self.s)
        } else if self.index == 1 {
            self.index += 1;

            Some(self.o)
        } else {
            None
        }
    }
}
/// Implementation of an immutable view of a HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPView<'a,const N:usize> {
    s: &'a Vec<usize>,
    o: &'a Vec<usize>
}
impl<'a,const N:usize> HalfKPView<'a,N> {
    /// Obtaining a immutable iterator
    pub fn iter(&'a self) -> HalfKPIter<'a,N> {
        HalfKPIter { s: self.s, o: self.o, index: 0 }
    }

    pub fn to_vec<T>(&'a self) -> Box<[T]>
        where T: Debug + Clone + Default + Send + Sync + FromPrimitive {
        let mut arr = vec![T::default();N*2].into_boxed_slice();

        for &i in self.s.iter() {
            arr[i] = T::from_f64(1.).unwrap();
        }

        for &i in self.o.iter() {
            arr[N + i] = T::from_f64(1.).unwrap();
        }

        arr
    }
}
impl<'a,const N:usize> Clone for HalfKPView<'a,N> {
    fn clone(&self) -> Self {
        HalfKPView{ s: self.s, o: self.o }
    }
}
impl<'a,const N:usize> From<&'a HalfKP<N>> for HalfKPView<'a,N> {
    fn from(value: &'a HalfKP<N>) -> Self {
        HalfKPView {
            s: &value.s,
            o: &value.o
        }
    }
}
impl<'a,const N:usize> From<&'a HalfKPView<'a,N>> for (Vec<size_t>,Vec<size_t>) {
    fn from(value: &'a HalfKPView<'a, N>) -> Self {
        let mut indexes = Vec::new();
        let mut boundaries = vec![0];

        let mut b = 0;

        indexes.extend_from_slice(value.s);
        b += value.s.len();
        boundaries.push(b);
        indexes.extend_from_slice(value.o);
        b += value.o.len();
        boundaries.push(b);

        (indexes,boundaries)
    }
}
#[derive(Debug,Clone)]
pub struct HalfKPList<const N: usize> {
    items: Vec<HalfKP<N>>
}
impl<const N: usize> HalfKPList<N> {
    pub fn new(items:Vec<HalfKP<N>>) -> HalfKPList<N> {
        HalfKPList {
            items
        }
    }
}
impl<const N: usize> From<Vec<HalfKP<N>>> for HalfKPList<N> {
    fn from(value: Vec<HalfKP<N>>) -> Self {
        HalfKPList {
            items: value
        }
    }
}
impl<const N: usize> BatchSize for HalfKPList<N> {
    fn size(&self) -> usize {
        self.items.len()
    }
}
impl<U,A,const N:usize> ToCuda<U,A> for HalfKPList<N>
    where U: UnitValue<U>,
          A: CudaAllocator {
   type Output = Self;

    fn to_cuda(self, _: &DeviceGpu<U,A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self)
    }
}
#[derive(Debug,Clone)]
pub struct HalfKPListView<'a,const N: usize> {
    items: &'a Vec<HalfKP<N>>
}
impl<'a,const N: usize> HalfKPListView<'a,N> {
    pub fn new(items:&'a Vec<HalfKP<N>>) -> HalfKPListView<'a,N> {
        HalfKPListView {
            items
        }
    }
}
impl<'a,const N: usize> BatchSize for HalfKPListView<'a,N> {
    fn size(&self) -> usize {
        self.items.len()
    }
}
impl<'a,const N:usize> From<&'a HalfKPList<N>> for HalfKPListView<'a,N> {
    fn from(value: &'a HalfKPList<N>) -> Self {
        HalfKPListView {
            items: &value.items
        }
    }
}
impl<'a,const N:usize> From<&'a HalfKPListView<'a,N>> for (Vec<size_t>,Vec<size_t>) {
    fn from(value: &'a HalfKPListView<'a,N>) -> Self {
        let mut indexes = Vec::new();
        let mut boundaries = vec![0];

        let mut b = 0;

        for HalfKP { s, o } in value.items.iter() {
            indexes.extend_from_slice(s);
            b += s.len();
            boundaries.push(b);
            indexes.extend_from_slice(o);
            b += o.len();
            boundaries.push(b);
        }

        (indexes,boundaries)
    }
}
impl<'a,const N:usize> From<HalfKPListView<'a,N>> for &'a Vec<HalfKP<N>> {
    fn from(value: HalfKPListView<'a, N>) -> Self {
        value.items
    }
}
impl<'a,T,const N:usize> From<&HalfKPListView<'a,N>> for Box<[T]>
    where T: Default + Clone + Copy + Send + Sync + FromPrimitive {
    fn from(value: &HalfKPListView<'a, N>) -> Self {
        value.items.iter().map(| item | {
            let mut arr = vec![T::default();N*2];

            for &i in item.s.iter() {
                arr[i] = T::from_f64(1.).unwrap();
            }

            for &i in item.o.iter() {
                arr[N + i] = T::from_f64(1.).unwrap();
            }

            arr
        }).fold(Vec::with_capacity(N * 2 * value.items.len()), | mut acc, i | {
            acc.extend_from_slice(&i);
            acc
        }).into_boxed_slice()
    }
}
