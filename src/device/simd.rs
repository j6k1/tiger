use std::ops::{Add, AddAssign, Shr};
use nncombinator::arr::{Arr, Arr2};
use nncombinator::device::{DeviceSimd};
use nncombinator::error::{EvaluateError, SpecializationError, TypeConvertError};
use nncombinator::mem::{AsRawSlice};
use rayon::prelude::{IntoParallelIterator,ParallelIterator};
use simdoperator::backend::common::{Backend, SignInt};
use simdoperator::traits::{Demote, LinearCombination};
use simdoperator::{AccVector, Matrix, Vector, VectorMut, VectorView};
use simdoperator::error::InstantiationError;
use usiagent::shogi::{KomaKind, MochigomaKind, Teban};
use crate::device::{DeviceDiffFeatureTransform, DeviceFeatureTransformEvaluation, DeviceFeatureTransformEvaluationScaling, DeviceQuantizedFeatureTransform};
use crate::features::{HalfKPDiff, HalfKPView};
use crate::nn::{InputCreator, MOCHIGOMA_END, OPPONENT_INDEX_MAP, PIECE_END, SELF_INDEX_MAP};

// Scale applied to weights and bias when they are quantized
pub(crate) const SCALE:f32 = 512.;

impl<BE,const NI: usize, const NO: usize> DeviceQuantizedFeatureTransform<Arr2<i16, NI, NO>, Arr<i16, NO>, NI, NO> for DeviceSimd<BE>
    where BE: Backend,
          [(); NO*2]: {
    type RealOutput = Arr<f32,{NO*2}>;
    fn quantization(&self, units: &Arr2<f32, NI, NO>, bias: &Arr<f32, NO>)
        -> Result<(f32,usize,f32,f32,Arr2<i16, NI, NO>, Arr<i16, NO>),SpecializationError> {
        let mut qunits = Arr2::<i16, NI, NO>::new();

        for (row, mut qrow) in units.iter().zip(qunits.iter_mut()) {
            for (&w, qw) in row.iter().zip(qrow.iter_mut()) {
                *qw = (w * SCALE + 0.5) as i16;
            }
        }

        let qbias = bias.iter().map(|&b| {
            (b * SCALE + 0.5) as i16
        }).collect::<Vec<i16>>();

        fn insert(acc: &mut [f32;18],max_size:usize,mut current_size:usize,w:f32) -> usize {
            if current_size == 0 {
                acc[current_size] = w;
                current_size += 1;
                return current_size;
            }

            if current_size < max_size {
                acc[current_size] = w;
            } else if w > acc[current_size] {
                acc[current_size] = w;
            } else {
                return current_size;
            }

            for i in (1..=current_size).rev() {
                if acc[i] > acc[i-1] {
                    let tmp = acc[i-1];
                    acc[i] = acc[i-1];
                    acc[i-1] = tmp;
                } else {
                    break;
                }
            }

            current_size
        }

        let mut raw_next_input_max = 0.0;

        for i in 0..NO {
            let ws = (0..81).into_par_iter().map(|oi| {
                let mut sum = 0.0;

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SFu, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GFu, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Fu as usize];

                for p in 0..18 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Fu as usize];

                for p in 0..18 {
                    cur = insert(&mut acc,18,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(18).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKyou, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKyou, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Kyou as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Kyou as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(4).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKei, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKei, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Kei as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Kei as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(4).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SGin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GGin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Gin as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Gin as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(4).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKin, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Kin as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Kin as usize];

                for p in 0..4 {
                    cur = insert(&mut acc,4,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(4).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKaku, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SKakuN, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKaku, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GKakuN, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Kaku as usize];

                for p in 0..2 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Kaku as usize];

                for p in 0..2 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(2).fold(0.0, |acc, &w| acc + w);

                let mut acc = [0.0; 18];

                let mut cur = 0;

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SHisha, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::SHishaN, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GHisha, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + InputCreator::input_index_of_banmen(Teban::Sente, KomaKind::GHishaN, 0, 0).unwrap();

                for p in 0..81 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + SELF_INDEX_MAP[MochigomaKind::Hisha as usize];

                for p in 0..2 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                let s = MOCHIGOMA_END * oi + PIECE_END + OPPONENT_INDEX_MAP[MochigomaKind::Hisha as usize];

                for p in 0..2 {
                    cur = insert(&mut acc,2,cur,qunits[(s+p,i)] as f32);
                }

                sum += acc.iter().take(2).fold(0.0, |acc, &w| acc + w);

                sum
            }).fold(|| 0.0/0.0, |acc, w| f32::max(acc, w)).reduce(|| 0.0/0.0, |acc, w| f32::max(acc, w));

            if ws > raw_next_input_max {
                raw_next_input_max = ws;
            }
        }

        let shift = (raw_next_input_max / i8::MAX as f32).max(1.0).log2().ceil() as usize;

        let next_input_max = raw_next_input_max / (1 << shift) as f32;
        let scale = SCALE / (1 << shift) as f32;

        Ok((1. / scale,shift,scale,next_input_max,qunits,qbias.try_into()?))
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
    where BE: Backend + Clone,
          AccVector<i16,{NO*2},BE>: Demote<AccVector<i8,{NO*2},BE>>,
          Box<[i8;NO*2]>: From<AccVector<i8,{NO*2},BE>>,
          for<'a,'b> VectorMut<'a,i16,NO,BE>: AddAssign<Vector<'b,i16,NO,BE>>,
          for<'a> Vector<'a,i16,{NO*2},BE>: TryFrom<&'a [i16],Error=InstantiationError> +
                                            Add<Vector<'a,i16,{NO*2},BE>,Output=AccVector<i16,{NO*2},BE>> +
                                            Shr<usize,Output=AccVector<i16,{NO*2},BE>>,
          [(); NO*2]: {
    type ScaledOutput = Arr<i8,{NO*2}>;

    fn forward_feature_transform_scaling<'a>(&self, shift: usize, input: &<Self as DeviceFeatureTransformEvaluation<i8,Arr2<i16,NI,NO>,Arr<i16,NO>,NI,NO>>::Output) -> Result<Self::ScaledOutput, EvaluateError> {
        let pad = [(1 << (shift - 1));NO*2];

        let pad = Vector::try_from(&pad)?;
        let input = Vector::try_from(input)?;

        Ok(Arr::try_from(*<Box<[i8;NO*2]>>::from(((input + pad) >> shift).demotion()))?)
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