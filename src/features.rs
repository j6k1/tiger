use std::ffi::c_void;
use nncombinator::arr::{Arr, ArrView, ArrViewMut, AsView, AsViewMut, MakeView, MakeViewMut, SerializedVec, SliceSize};
use nncombinator::cuda::{AsPtr, AsVoidPtr};
use nncombinator::error::SizeMismatchError;
use nncombinator::layer::BatchDataType;
use nncombinator::mem::AsRawSlice;

/// InputFeatures Implementaion
#[derive(Debug)]
pub struct HalfKP<T,const N:usize> where T: Default + Clone + Send {
    arr:Box<[T]>
}
impl<T,const N:usize> BatchDataType for HalfKP<T,N> where T: Default + Clone + Send {
    type Type = SerializedVec<T,HalfKP<T,N>>;
}
impl<T,const N:usize> HalfKP<T,N> where T: Default + Clone + Send {
    /// Create an instance of HalfKP
    pub fn new(s:Arr<T,N>,o:Arr<T,N>) -> HalfKP<T,N> {
        let mut arr = Vec::with_capacity(N * 2);
        
        arr.extend_from_slice(&s);
        arr.extend_from_slice(&o);

        HalfKP {
            arr:arr.into_boxed_slice()
        }
    }
    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> HalfKPIter<'a,T,N> {
        HalfKPIter{ arr: &*self.arr }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> HalfKPIterMut<'a,T,N> {
        HalfKPIterMut{ arr: &mut *self.arr }
    }
}
impl<T,const N:usize> Clone for HalfKP<T,N> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        HalfKP {
            arr:self.arr.clone(),
        }
    }
}
impl<T,const N:usize> TryFrom<Box<[T]>> for HalfKP<T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(arr: Box<[T]>) -> Result<Self, Self::Error> {
        if arr.len() != N * 2 {
            Err(SizeMismatchError(arr.len(),N * 2))
        } else {
            Ok(HalfKP {
                arr:arr
            })
        }
    }
}
impl<T,const N:usize> TryFrom<Vec<T>> for HalfKP<T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        if v.len() != N * 2 {
            Err(SizeMismatchError(v.len(),N * 2))
        } else {
            let arr = v.into_boxed_slice();

            Ok(HalfKP {
                arr:arr
            })
        }
    }
}
impl<T,const N:usize> Default for HalfKP<T,N> where T: Default + Clone + Send {
    fn default() -> HalfKP<T,N> {
        HalfKP {
            arr: vec![T::default();N*2].into_boxed_slice()
        }
    }
}
impl<T,const N:usize> AsRawSlice<T> for HalfKP<T,N> where T: Default + Clone + Send + Sync {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<T,const N:usize> AsPtr<T> for HalfKP<T,N> where T: Default + Clone + Send {
    fn as_ptr(&self) -> *const T {
        self.arr.as_ptr()
    }
}
impl<T,const N:usize> AsVoidPtr for HalfKP<T,N> where T: Default + Clone + Send {
    fn as_void_ptr(&self) -> *const c_void {
        self.arr.as_ptr() as *const c_void
    }
}
/// Implementation of an immutable iterator for HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPIter<'a,T,const N:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const N:usize> HalfKPIter<'a,T,N> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for HalfKPIter<'a,T,N> where T: Default + Clone + Send {
    type Item = ArrView<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrView. The sizes do not match."))
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else if n == 0 {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrView. The sizes do not match."))
        } else {
            let (_,r) = slice.split_at(self.element_size() * n);
            let (l,r) = r.split_at(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrView. The sizes do not match."))
        }
    }
}
/// Implementation of an mutable iterator for HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPIterMut<'a,T,const N:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const N:usize> HalfKPIterMut<'a,T,N> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for HalfKPIterMut<'a,T,N> where T: Default + Clone + Send {
    type Item = ArrViewMut<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrViewMut. The sizes do not match."))
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else if n == 0 {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrViewMut. The sizes do not match."))
        } else {
            let (_,r) = slice.split_at_mut(self.element_size() * n);
            let (l,r) = r.split_at_mut(self.element_size());

            self.arr = r;

            Some(l.try_into().expect("An error occurred in the conversion from Slice to ArrViewMut. The sizes do not match."))
        }
    }
}
/// Implementation of an immutable view of a HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPView<'a,T,const N:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const N:usize> HalfKPView<'a,T,N> where T: Default + Clone + Send {
    /// Obtaining a immutable iterator
    pub fn iter(&'a self) -> HalfKPIter<'a,T,N> {
        HalfKPIter { arr: &*self.arr }
    }
}
impl<'a,T,const N:usize> Clone for HalfKPView<'a,T,N> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        HalfKPView{ arr: self.arr }
    }
}
impl<'a,T,const N:usize> AsPtr<T> for HalfKPView<'a,T,N> where T: Default + Clone + Send {
    fn as_ptr(&self) -> *const T {
        self.arr.as_ptr()
    }
}
impl<'a,T,const N:usize> AsVoidPtr for HalfKPView<'a,T,N> where T: Default + Clone + Send {
    fn as_void_ptr(&self) -> *const c_void {
        self.arr.as_ptr() as *const c_void
    }
}
impl<'a,T,const N:usize> From<&'a HalfKP<T,N>> for HalfKPView<'a,T,N> where T: Default + Clone + Send {
    fn from(value: &'a HalfKP<T,N>) -> Self {
        HalfKPView {
            arr:&value.arr
        }
    }
}
/// Implementation of an mutable view of a HalfKP
#[derive(Debug,Eq,PartialEq)]
pub struct HalfKPViewMut<'a,T,const N:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const N:usize> HalfKPViewMut<'a,T,N> where T: Default + Clone + Send {
    /// Obtaining a immutable iterator
    pub fn iter(&'a self) -> HalfKPIter<'a,T,N> {
        HalfKPIter { arr: &*self.arr }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut(&'a mut self) -> HalfKPIterMut<'a,T,N> {
        HalfKPIterMut { arr: &mut *self.arr }
    }
}
impl<T,const N:usize> SliceSize for HalfKP<T,N>
    where T: Default + Clone + Send + Sync {
    const SIZE: usize = N * 2;
}
impl<'a,T,const N:usize> AsView<'a> for HalfKP<T,N>
    where T: Default + Clone + Send + Sync + 'a {
    type ViewType = HalfKPView<'a,T,N>;

    fn as_view(&'a self) -> Self::ViewType {
        HalfKPView {
            arr: &*self.arr
        }
    }
}
impl<'a,T,const N:usize> MakeView<'a,T> for HalfKP<T,N>
    where T: Default + Clone + Send + Sync + 'a {

    fn make_view(arr: &'a [T]) -> Result<Self::ViewType,SizeMismatchError> {
        if arr.len() != HalfKP::<T,N>::slice_size() {
            Err(SizeMismatchError(HalfKP::<T,N>::slice_size(),arr.len()))
        } else {
            Ok(HalfKPView {
                arr: arr
            })
        }
    }
}
impl<'a,T,const N:usize> AsViewMut<'a> for HalfKP<T,N>
    where T: Default + Clone + Send + Sync + 'a {
    type ViewType = HalfKPViewMut<'a,T,N>;

    fn as_view(&'a mut self) -> Self::ViewType {
        HalfKPViewMut {
            arr:&mut *self.arr
        }
    }
}
impl<'a,T,const N:usize> MakeViewMut<'a,T> for HalfKP<T,N>
    where T: Default + Clone + Send + Sync + 'a {

    fn make_view_mut(arr: &'a mut [T]) -> Result<Self::ViewType,SizeMismatchError> {
        if arr.len() != HalfKP::<T,N>::slice_size() {
            Err(SizeMismatchError(HalfKP::<T,N>::slice_size(),arr.len()))
        } else {
            Ok(HalfKPViewMut {
                arr: arr
            })
        }
    }
}
