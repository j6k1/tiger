use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::error::EvaluateError;

pub trait DeviceFeatureTransform<U,T,const FEATURES_NUM: usize,const NO: usize> where U: Default + Clone + Send {
    fn features_batch_combine<'a>(&self,self_output:&'a SerializedVec<U,Arr<U,NO>>,opponent_output:&'a SerializedVec<U,Arr<U,NO>>)
        -> Result<SerializedVec<U,Arr<U,{NO*2}>>,EvaluateError>;
    fn loss_input_transform_to_features(&self,combined_input:&SerializedVec<U,Arr<U,{NO*2}>>)
        -> Result<(SerializedVec<U,Arr<U,NO>>,SerializedVec<U,Arr<U,NO>>),EvaluateError>;
}