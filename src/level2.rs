use ndarray::*;
use mat::*;

/// Multiply sparse matrix `mat` by dense vector `x` using the semiring formed with `add`, `mult`,
/// and `zero`.
pub fn spmv_sr<LT, RT, F, S, AF, MF, Order, S1, S2, S3>(mat: &CompressedSparseMatrix<LT,
                                                                                     Order,
                                                                                     S1,
                                                                                     S2,
                                                                                     S3>,
                                                        x: &ArrayBase<S, Ix>,
                                                        add: AF,
                                                        mult: MF,
                                                        zero: F)
                                                        -> Array<F, Ix>
    where AF: Fn(&F, &F) -> F, // TODO: don't need references?
          MF: Fn(&LT, &RT) -> F,
          S: DataClone<Elem = RT>,
          Order: Storage,
          LT: Clone, // TODO: need clone?
          F: Clone,
          S1: DataClone<Elem = LT>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>
{
    let mut y = Array::<F, Ix>::from_elem(mat.size().0, zero);
    for (i, j, v) in mat.iter() {
        y[i] = add(&y[i], &mult(v, &x[j]));
    }

    y
}
