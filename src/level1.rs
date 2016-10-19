use ndarray::*;
use mat::*;

pub fn dot_sr<LT, RT, F, S1, S2, S3, AF, MF>(left: &CompressedSparseVector<LT, S1, S2>,
                                             right: &ArrayBase<S3, Ix>,
                                             add: AF,
                                             mult: MF,
                                             zero: F)
                                             -> F
    where S1: DataClone<Elem = LT>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = RT>,
          AF: Fn(&F, &F) -> F,
          MF: Fn(&LT, &RT) -> F
{
    assert!(left.len() == right.len());
    let mut sum = zero;
    for (i, v) in left.iter() {
        sum = add(&sum, &mult(v, &right[i]));
    }

    sum
}

pub fn dot_sr_<LT, RT, F, S1, S2, S3, AF, MF>(left: &ArrayBase<S3, Ix>,
                                              right: &CompressedSparseVector<LT, S1, S2>,
                                              add: AF,
                                              mult: MF,
                                              zero: F)
                                              -> F
    where S1: DataClone<Elem = LT>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = RT>,
          AF: Fn(&F, &F) -> F,
          MF: Fn(&LT, &RT) -> F
{
    dot_sr(right, left, add, mult, zero)
}
