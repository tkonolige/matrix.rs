use mat::*;
use rand::*;

/// Generate a random `CompressedSparseMatrix` by randomly generating nonzero
/// values and locations.
/// The number of nonzeros in the resulting matrix can be less that `nnz`
pub fn rand_matrix<T: Clone, S: Storage, I: Index>(
    n: usize,
    m: usize,
    nnz: usize,
    val: T,
) -> OwnedSparseMatrix<T, S, I> {
    let mut xs = Vec::<I>::new();
    let mut ys = Vec::<I>::new();
    let mut zs = Vec::<T>::new();
    for _ in 0..nnz {
        let x = thread_rng().gen_range(0, n);
        xs.push(I::from_usize(x));
        let y = thread_rng().gen_range(0, m);
        ys.push(I::from_usize(y));
        zs.push(val.clone());
    }

    OwnedSparseMatrix::from_vecs(xs, ys, zs, n, m, |x, _| x)
}
