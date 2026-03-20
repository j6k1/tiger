use std::fmt::Debug;
use std::mem;
use std::ops::{BitXor, Mul};
use std::simd::{Simd};
use crate::device::LANES_F32;
use crate::device::LANES_F64;

pub trait Sign: Debug + Clone + Copy {
    fn plus() -> Self;
    fn minus() -> Self;
}
pub trait Bits {
    type Int: Clone + Copy + Debug;

    fn to_bits(&self) -> Self::Int;
    fn from_bits(bits: Self::Int) -> Self;
}
impl Bits for f32 {
    type Int = u32;
    fn to_bits(&self) -> Self::Int {
        f32::to_bits(*self)
    }
    fn from_bits(bits: Self::Int) -> Self {
        f32::from_bits(bits)
    }
}
impl Bits for f64 {
    type Int = u64;

    fn to_bits(&self) -> Self::Int {
        f64::to_bits(*self)
    }
    fn from_bits(bits: Self::Int) -> Self {
        f64::from_bits(bits)
    }
}
#[derive(Debug,Copy,Clone)]
pub struct SignFloat<T>
    where T: Bits + Clone + Copy + Debug,
          <T as Bits>::Int: Clone + Copy + Debug {
    mask: <T as Bits>::Int,
}
impl Sign for SignFloat<f32> {
    fn plus() -> Self {
        SignFloat {
            mask: 0x00000000
        }
    }

    fn minus() -> Self {
        SignFloat {
            mask: 0x80000000
        }
    }
}
impl Sign for SignFloat<f64> {
    fn plus() -> Self {
        SignFloat {
            mask: 0x00000000_00000000
        }
    }

    fn minus() -> Self {
        SignFloat {
            mask: 0x80000000_00000000
        }
    }
}
impl Bits for Simd<f32,LANES_F32> {
    type Int = Simd<u32,LANES_F32>;
    fn to_bits(&self) -> Self::Int {
        unsafe { mem::transmute::<Simd<f32,LANES_F32>,Simd<u32,LANES_F32>>(*self) }
    }
    fn from_bits(bits: Self::Int) -> Self {
        unsafe { mem::transmute::<Simd<u32,LANES_F32>,Simd<f32,LANES_F32>>(bits) }
    }
}
impl Bits for Simd<f64,LANES_F64> {
    type Int = Simd<u64,LANES_F64>;
    fn to_bits(&self) -> Self::Int {
        unsafe { mem::transmute::<Simd<f64,LANES_F64>,Simd<u64,LANES_F64>>(*self) }
    }
    fn from_bits(bits: Self::Int) -> Self {
        unsafe { mem::transmute::<Simd<u64,LANES_F64>,Simd<f64,LANES_F64>>(bits) }
    }
}
impl Sign for SignFloat<Simd<f32,LANES_F32>> {
    fn plus() -> Self {
        SignFloat {
            mask: Simd::splat(0x00000000)
        }
    }

    fn minus() -> Self {
        SignFloat {
            mask: Simd::splat(0x80000000)
        }
    }
}
impl<T> Mul<T> for SignFloat<T>
    where T: Bits + Clone + Copy + Debug,
          <T as Bits>::Int: Clone + Copy + Debug +
                            BitXor<<T as Bits>::Int,Output=<T as Bits>::Int> {
    type Output = T;
    fn mul(self, rhs: T) -> Self::Output {
        T::from_bits(rhs.to_bits() ^ self.mask)
    }
}
impl Mul<Simd<f32,LANES_F32>> for SignFloat<f32>
    where  {
    type Output = Simd<f32,LANES_F32>;
    fn mul(self, rhs: Simd<f32,LANES_F32>) -> Self::Output {
        Simd::from_bits(rhs.to_bits() ^ Simd::splat(self.mask))
    }
}
impl Mul<Simd<f64,LANES_F64>> for SignFloat<f64>
where  {
    type Output = Simd<f64,LANES_F64>;
    fn mul(self, rhs: Simd<f64,LANES_F64>) -> Self::Output {
        Simd::from_bits(rhs.to_bits() ^ Simd::splat(self.mask))
    }
}
impl Mul<SignFloat<f32>> for f32 {
    type Output = f32;
    fn mul(self, rhs: SignFloat<f32>) -> Self::Output {
        rhs * self
    }
}
impl Mul<SignFloat<f64>> for f64 {
    type Output = f64;
    fn mul(self, rhs: SignFloat<f64>) -> Self::Output {
        rhs * self
    }
}
impl Mul<SignFloat<f32>> for Simd<f32,LANES_F32> {
    type Output = Simd<f32,LANES_F32>;
    fn mul(self, rhs: SignFloat<f32>) -> Self::Output {
        rhs * self
    }
}
impl Mul<SignFloat<f64>> for Simd<f64,LANES_F64> {
    type Output = Simd<f64,LANES_F64>;
    fn mul(self, rhs: SignFloat<f64>) -> Self::Output {
        rhs * self
    }
}
