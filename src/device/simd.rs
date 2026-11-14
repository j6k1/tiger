use std::ops::{AddAssign, Shr};
use nncombinator::arr::{Arr, Arr2};
use nncombinator::device::{DeviceSimd};
use nncombinator::error::{EvaluateError, SpecializationError, TypeConvertError};
use nncombinator::mem::{AsRawSlice};
use simdoperator::backend::common::{Backend, SignInt};
use simdoperator::traits::{Demote, LinearCombination};
use simdoperator::{AccVector, Matrix, Vector, VectorMut, VectorView};
use simdoperator::error::InstantiationError;
use crate::device::{DeviceDiffFeatureTransform, DeviceFeatureTransformEvaluation, DeviceFeatureTransformEvaluationScaling, DeviceQuantizedFeatureTransform};
use crate::features::{HalfKPDiff, HalfKPView};

/// Scale applied to weights and bias when they are quantized
pub const SCALE:f32 = 512.;
/// Shift width used to scale down the accumulated value
pub const SHIFT:usize = 8;

impl<BE,const NI: usize, const NO: usize> DeviceQuantizedFeatureTransform<Arr2<i16, NI, NO>, Arr<i16, NO>, NI, NO> for DeviceSimd<BE>
    where BE: Backend,
          [(); NO*2]: {
    type RealOutput = Arr<f32,{NO*2}>;
    fn quantization(&self, units: &Arr2<f32, NI, NO>, bias: &Arr<f32, NO>) -> Result<(Arr2<i16, NI, NO>, Arr<i16, NO>),SpecializationError> {
        let mut qunits = Arr2::<i16, NI, NO>::new();

        for (row, mut qrow) in units.iter().zip(qunits.iter_mut()) {
            for (&w, qw) in row.iter().zip(qrow.iter_mut()) {
                *qw = (w * SCALE) as i16;
            }
        }

        let qbias = bias.iter().map(|&b| (b * SCALE) as i16).collect::<Vec<i16>>().try_into()?;

       Ok((qunits, qbias))
    }
}
impl<BE,const NI: usize, const NO: usize> DeviceFeatureTransformEvaluation<i8,Arr2<i16, NI, NO>, Arr<i16, NO>, NI, NO> for DeviceSimd<BE>
    where BE: Backend,
          for<'a,'b> VectorMut<'a,i16,NO,BE>: AddAssign<Vector<'b,i16,NO,BE>>,
          [(); NO*2]: {
    type Output = Arr<i16,{NO*2}>;

    fn forward_feature_transform<'a>(&self, bias: &Arr<i16, NO>, units: &Arr2<i16, NI, NO>, input: HalfKPView<'a, NI>) -> Result<Self::Output, EvaluateError> {
        let units = Matrix::<i16,NI,NO,BE>::try_from(units.as_raw_slice())?;

        let bias = bias.as_raw_slice();

        let mut result = [0i16;NO*2];

        result[..NO].copy_from_slice(bias);
        result[NO..].copy_from_slice(bias);

        for (rs,indexes) in result.chunks_exact_mut(NO).zip(input.iter()) {
            let mut r = VectorMut::try_from(rs)?;

            for &i in indexes.iter() {
                r += units.row(i);
            }
        }

        Ok(Arr::try_from(result)?)
    }
}
impl<BE,const NI: usize, const NO: usize> DeviceFeatureTransformEvaluationScaling<i8,Arr2<i16,NI,NO>,Arr<i16,NO>,NI,NO> for DeviceSimd<BE>
    where BE: Backend,
          AccVector<i16,{NO*2},BE>: Demote<AccVector<i8,{NO*2},BE>>,
          Box<[i8;NO*2]>: From<AccVector<i8,{NO*2},BE>>,
          for<'a,'b> VectorMut<'a,i16,NO,BE>: AddAssign<Vector<'b,i16,NO,BE>>,
          for<'a> Vector<'a,i16,{NO*2},BE>: TryFrom<&'a [i16],Error=InstantiationError> +
                                            Shr<usize,Output=AccVector<i16,{NO*2},BE>>,
          [(); NO*2]: {
    type ScaledOutput = Arr<i8,{NO*2}>;

    fn forward_feature_transform_scaling<'a>(&self, input: &<Self as DeviceFeatureTransformEvaluation<i8,Arr2<i16,NI,NO>,Arr<i16,NO>,NI,NO>>::Output) -> Result<Self::ScaledOutput, EvaluateError> {
        let input = Vector::try_from(input)?;

        Ok(Arr::try_from(*<Box<[i8;NO*2]>>::from((input >> SHIFT).demotion()))?)
    }
}
impl<BE,const NI: usize,const NO: usize> DeviceDiffFeatureTransform<i8,HalfKPDiff<SignInt<i16>>,Arr2<i16,NI,NO>,Arr<i16,NO>,NI,NO> for DeviceSimd<BE>
    where BE: Backend,
          DeviceSimd<BE>: DeviceFeatureTransformEvaluation<i8,Arr2<i16,NI,NO>,Arr<i16,NO>,NI,NO,Output=Arr<i16,{NO*2}>>,
          Arr<i16,{NO*2}>: TryFrom<Vec<i16>,Error=TypeConvertError>,
          for<'a> VectorView<'a,i16,NO>: From<&'a Vector<'a,i16,NO,BE>>,
          for<'a,'b> VectorMut<'a,i16,NO,BE>: LinearCombination<VectorView<'b,i16,NO>,SignInt<i16>>,
          [(); NO*2]: {
    fn forward_diff_feature_transform(&self, _: &Arr<i16, NO>, units: &Arr2<i16, NI, NO>, input: &HalfKPDiff<SignInt<i16>>, partial_input: &Self::Output) -> Result<Self::Output, EvaluateError> {
        let mut r = partial_input.to_vec();

        let units = Matrix::<i16,NI,NO,BE>::try_from(units.as_raw_slice())?;

        let chunks = r.chunks_exact_mut(NO);
        for (chunk,indexes) in chunks.zip(input.iter()) {
            let mut lines = Vec::with_capacity(indexes.len());

            for (index,sign) in indexes.iter().copied() {
                lines.push((units.row(index).into(), sign));
            }

            let mut pr = VectorMut::try_from(chunk)?;

            pr.linear_combination(&lines);
        }

        Ok(r.try_into()?)
    }
}