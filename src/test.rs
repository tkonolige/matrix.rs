// Test suite for sparse matrix ops
// TODO: use quickcheck

use ndarray::*;
use mat::*;
use level1::*;
use level2::*;
use level3::*;

#[test]
fn transpose_transpose_csr() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSRMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    assert_eq!(mat, mat.transpose().transpose().to_owned());
}

#[test]
fn transpose_transpose_csc() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSCMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    assert_eq!(mat, mat.transpose().transpose().to_owned());
}

#[test]
fn spmv_sr_plus_times_csr() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSRMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    let x = arr1(&[10, 11, 12, 13]);

    let y = spmv_sr(&mat, &x, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(y, arr1(&[60, 139, 0]));
}

#[test]
fn spmv_sr_plus_times_csc() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSCMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    let x = arr1(&[10, 11, 12, 13]);

    let y = spmv_sr(&mat, &x, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(y, arr1(&[60, 139, 0]));
}

#[test]
fn spmdm_sr_plus_times_csr() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSRMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    let x = arr2(&[[10, 3], [11, 1], [12, 9], [13, 0]]);

    let y = spmdm_sr(&mat, &x, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(y, arr2(&[[60, 18], [139, 68], [0, 0]]));
}

#[test]
fn spmdm_sr_plus_times_csc() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSCMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    let x = arr2(&[[10, 3], [11, 1], [12, 9], [13, 0]]);

    let y = spmdm_sr(&mat, &x, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(y, arr2(&[[60, 18], [139, 68], [0, 0]]));
}

#[test]
fn dot_sr_plus_times() {
    let vec = CompressedSparseVector::new(5, arr1(&[0, 3, 4]), arr1(&[1, 2, -1]));
    let dvec = arr1(&[-2, 3, 3, 5, -4]);

    let res1 = dot_sr(&vec, &dvec, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(res1, 12);
    let res2 = dot_sr_(&dvec, &vec, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(res2, 12);
}
