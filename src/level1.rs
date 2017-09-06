use ndarray::*;
use mat::*;
use std::cmp::*;

/// Dot product of sparse vector and dense vector on a semiring
pub fn dot_sd_sr<LT, RT, F, I, S1, S2, S3, AF, MF>(
    left: &CompressedSparseVector<LT, I, S1, S2>,
    right: &ArrayBase<S3, Ix1>,
    add: AF,
    mult: MF,
    zero: F,
) -> F
where
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = RT>,
    I: Index,
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
{
    assert!(left.len() == right.len());
    let mut sum = zero;
    unsafe {
        for i in 0..left.nnz() {
            sum = add(
                &sum,
                &mult(
                    left.values.uget(i),
                    &right.uget(left.indices.uget(i).to_usize()),
                ),
            );
        }
    }

    sum
}

/// Dot product of sparse vector and sparse vector on a semiring
pub fn dot_ss_sr<LT, RT, F, I, S1, S2, S3, S4, AF, MF>(
    left: &CompressedSparseVector<LT, I, S1, S2>,
    right: &CompressedSparseVector<RT, I, S3, S4>,
    add: AF,
    mult: MF,
    zero: F,
) -> F
where
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = RT>,
    S4: DataClone<Elem = I>,
    I: Index,
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
{
    let mut sum = zero;

    let mut left_iter = left.indexed_iter();
    let mut right_iter = right.indexed_iter();
    let mut cur_left = left_iter.next();
    let mut cur_right = right_iter.next();
    loop {
        match (cur_left, cur_right) {
            (Some((li, lv)), Some((ri, rv))) => {
                if li < ri {
                    cur_left = left_iter.next();
                    continue;
                }
                if li > ri {
                    cur_right = right_iter.next();
                    continue;
                }

                sum = add(&sum, &mult(lv, rv));
                cur_right = right_iter.next();
                cur_left = left_iter.next();
            }
            _ => break,
        }
    }

    sum
}

pub fn dot_ss_sr_fast<LT, RT, F, I, S1, S2, S3, S4, AF, MF>(
    left: &CompressedSparseVector<LT, I, S1, S2>,
    right: &CompressedSparseVector<RT, I, S3, S4>,
    add: AF,
    mult: MF,
    zero: F,
) -> F
where
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = RT>,
    S4: DataClone<Elem = I>,
    I: Index,
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
{
    let mut left_ind = 0;
    let mut right_ind = 0;

    let mut res = zero;

    while left_ind < left.nnz() && right_ind < right.nnz() {
        let li = &left.indices[left_ind];
        let ri = &right.indices[right_ind];
        match li.cmp(&ri) {
            Ordering::Less => left_ind = left_ind + 1,
            Ordering::Greater => right_ind = right_ind + 1,
            Ordering::Equal => {
                res = add(
                    &res,
                    &mult(&left.values[left_ind], &right.values[right_ind]),
                );
            }
        }
    }

    res
}

pub fn dot_fast<I, S1, S2, S3>(
    left: &CompressedSparseVector<f64, I, S1, S2>,
    right: &ArrayBase<S3, Ix1>,
) -> f64
where
    S1: DataClone<Elem = f64>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = f64>,
    I: Index,
{
    assert!(left.len() == right.len());
    // let mut sum = 0.0;
    // unsafe {
    //     for (i, v) in left.iter() {
    //         sum = sum + v * right.uget(i);
    //     }
    // }
    // sum

    unsafe {
        left.indices
            .iter()
            .zip(left.values.iter())
            .fold(0.0, |acc, (i, v)| acc + v * right.uget(i.to_usize()))
    }
}

/// Dot product on semiring using iterators
pub fn dot_iter_sr<LT, RT, F, I, S1, S2, S3, AF, MF>(
    left: &CompressedSparseVector<LT, I, S1, S2>,
    right: &ArrayBase<S3, Ix1>,
    add: AF,
    mult: MF,
    zero: F,
) -> F
where
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = RT>,
    I: Index,
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
{
    let mut sum = zero;

    let mut left_iter = left.indexed_iter();
    let mut right_iter = right.indexed_iter();
    let mut cur_left = left_iter.next();
    let mut cur_right = right_iter.next();
    loop {
        match (cur_left, cur_right) {
            (Some((li, lv)), Some((ri, rv))) => {
                if li < ri {
                    cur_left = left_iter.next();
                    continue;
                }
                if li > ri {
                    cur_right = right_iter.nth(li - ri);
                    continue;
                }

                sum = add(&sum, &mult(lv, rv));
                cur_right = right_iter.next();
                cur_left = left_iter.next();
            }
            _ => break,
        }
    }

    sum
}
