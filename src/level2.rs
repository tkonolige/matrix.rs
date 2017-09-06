use ndarray::*;
use mat::*;

/// Multiply sparse matrix `mat` by dense vector `x` using the semiring formed
/// with `add`, `mult`,
/// and `zero`.
pub fn spmv_sr<LT, RT, F, I, S, AF, MF, Order, S1, S2, S3>(
    mat: &CompressedSparseMatrix<LT, I, Order, S1, S2, S3>,
    x: &ArrayBase<S, Ix1>,
    add: AF,
    mult: MF,
    zero: F,
) -> Array<F, Ix1>
where
    AF: Fn(&F, &F) -> F, // TODO: don't need references?
    MF: Fn(&LT, &RT) -> F,
    S: DataClone<Elem = RT>,
    Order: Storage,
    LT: Clone, // TODO: need clone?
    F: Clone,
    I: Index,
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
{
    let mut y = Array::from_elem(mat.rows(), zero.clone());
    spmv_sr_mut(mat, x, &mut y, add, mult);
    y
}

/// Compute b = A * x + b with preallocated b
pub fn spmv_sr_mut<LT, RT, F, I, S, SS, AF, MF, Order, S1, S2, S3>(
    mat: &CompressedSparseMatrix<LT, I, Order, S1, S2, S3>,
    x: &ArrayBase<S, Ix1>,
    b: &mut ArrayBase<SS, Ix1>,
    add: AF,
    mult: MF,
) where
    AF: Fn(&F, &F) -> F, // TODO: don't need references?
    MF: Fn(&LT, &RT) -> F,
    S: DataClone<Elem = RT>,
    SS: DataMut<Elem = F>,
    Order: Storage,
    LT: Clone, // TODO: need clone?
    F: Clone,
    I: Index,
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
{
    unsafe {
        let mut inner_idx: usize = 0;
        for outer_idx in 1..mat.outer_ptr().len() {
            let inner_end = mat.outer_ptr().uget(outer_idx).to_usize();
            assert!(inner_end <= mat.inner_indices().len());
            while inner_idx < inner_end {
                assert!(inner_idx < mat.inner_indices().len());
                let (i, j) = Order::reorder_ix((
                    outer_idx - 1,
                    mat.inner_indices().uget(inner_idx).to_usize(),
                ));
                *(b.uget_mut(i)) = add(&b.uget(i), &mult(mat.values().uget(inner_idx), x.uget(j)));
                inner_idx += 1;
            }
        }
    }
}
