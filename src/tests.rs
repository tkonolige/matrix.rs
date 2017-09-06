// Test suite for sparse matrix ops
// TODO: use quickcheck
// TODO: find way to create benchmarks for multiple sizes

use ndarray::*;
use mat::*;
use level1::*;
use level2::*;
use level3::*;
use util::*;

use test;

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
    let k = vec![2.0, 6.0, 3.0, 7.0];
    let mat = CSRMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    let x = arr1::<f64>(&[10.0, 11.0, 12.0, 13.0]);

    let y =
        spmv_sr::<f64, f64, f64, _, _, _, _, _, _, _, _>(&mat, &x, |x, y| x + y, |x, y| x * y, 0.0);
    assert_eq!(y, arr1::<f64>(&[60.0, 139.0, 0.0]));
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
    let vec = CompressedSparseVector::<isize>::new(5, arr1(&[0, 3, 4]), arr1(&[1, 2, -1]));
    let dvec = arr1(&[-2, 3, 3, 5, -4]);

    let res1 = dot_sd_sr(&vec, &dvec, |x, y| x + y, |x, y| x * y, 0);
    assert_eq!(res1, 12);
}

fn dot_setup() -> (Array<f64, Ix1>, CompressedSparseVector<f64, i32>) {
    let len = 100;
    let x = Array::from_elem(len, 1.0);
    let y = CompressedSparseVector::<f64, i32>::new(
        len,
        Array::from_iter((0..i32::from_usize(len)).step_by(10)),
        Array::from_elem(len / 10, 1.0),
    );
    (x, y)
}

#[bench]
fn bench_sparse_sparse_dot(b: &mut test::Bencher) {
    let x = CompressedSparseVector::<f64, i64>::rand(100000, 10000, 1.0);
    let y = CompressedSparseVector::<f64, i64>::rand(100000, 10000, 1.0);
    b.iter(|| dot_ss_sr(&x, &y, |x, y| x + y, |x, y| x * y, 0.0))
}

#[bench]
fn bench_sparse_sparse_dot_fast(b: &mut test::Bencher) {
    let x = CompressedSparseVector::<f64, i64>::rand(100000, 10000, 1.0);
    let y = CompressedSparseVector::<f64, i64>::rand(100000, 10000, 1.0);
    b.iter(|| dot_ss_sr_fast(&x, &y, |x, y| x + y, |x, y| x * y, 0.0))
}

#[bench]
fn bench_dot_iter(b: &mut test::Bencher) {
    let (x, y) = dot_setup();
    b.iter(|| dot_iter_sr(&y, &x, |x, y| x + y, |x, y| x * y, 0.0))
}

#[bench]
fn bench_dot(b: &mut test::Bencher) {
    let (x, y) = dot_setup();
    b.iter(|| dot_sd_sr(&y, &x, |x, y| x + y, |x, y| x * y, 0.0))
}

#[bench]
fn bench_dot_manual(b: &mut test::Bencher) {
    let (right, left) = dot_setup();
    b.iter(|| {
        let mut sum = 0.0;
        for i in 0..left.nnz().to_usize() {
            unsafe {
                let j = left.indices.uget(i);
                let x = left.values.uget(i);
                sum += x * right.uget(j.to_usize());
            }
        }
        sum
    })
}

#[bench]
fn bench_dot_iter_fold(b: &mut test::Bencher) {
    let (right, left) = dot_setup();
    b.iter(|| unsafe {
        left.indices
            .iter()
            .zip(left.values.iter())
            .fold(0.0, |acc, (i, v)| acc + v * right.uget(i.to_usize()))
    })
}

#[bench]
fn bench_dot_fast(b: &mut test::Bencher) {
    let (x, y) = dot_setup();
    b.iter(|| dot_fast(&y, &x))
}

#[link(name = "mkl_intel_lp64", kind = "dylib")]
#[link(name = "mkl_core", kind = "dylib")]
#[link(name = "mkl_sequential", kind = "dylib")]
extern "C" {
    pub fn cblas_ddoti(
        N: ::std::os::raw::c_int,
        X: *const f64,
        indx: *const ::std::os::raw::c_int,
        Y: *const f64,
    ) -> f64;
}

#[bench]
fn bench_dot_mkl(b: &mut test::Bencher) {
    let (x, y) = dot_setup();
    b.iter(|| unsafe {
        cblas_ddoti(
            y.indices.len() as i32,
            y.values.as_ptr(),
            y.indices.as_ptr(),
            x.as_ptr(),
        )
    })
}

fn spmv_setup<S: Storage, I: Index>() -> (OwnedSparseMatrix<f64, S, I>, Array<f64, Ix1>) {
    let x = Array::from_elem(1000, 1.0);
    let m = rand_matrix(1000, 1000, 2000, 1.0);
    (m, x)
}

#[bench]
fn bench_spmv_csr(b: &mut test::Bencher) {
    let (m, x) = spmv_setup::<RowMajor, i32>();
    let mut y = Array::from_elem(x.len(), 0.0);
    b.iter(|| spmv_sr_mut(&m, &x, &mut y, |x, y| x + y, |x, y| x * y))
}


#[link(name = "mkl_intel_lp64", kind = "dylib")]
#[link(name = "mkl_core", kind = "dylib")]
#[link(name = "mkl_sequential", kind = "dylib")]
extern "C" {
    pub fn mkl_cspblas_dcsrgemv(
        transa: *const ::std::os::raw::c_char,
        m: *const ::std::os::raw::c_int,
        a: *const f64,
        ia: *const ::std::os::raw::c_int,
        ja: *const ::std::os::raw::c_int,
        x: *const f64,
        y: *mut f64,
    );
// pub fn mkl_cspblas_dcscgemv(transa: *const ::std::os::raw::c_char,
//                             m: *const ::std::os::raw::c_int,
//                             a: *const f64,
//                             ia: *const ::std::os::raw::c_int,
//                             ja: *const ::std::os::raw::c_int,
//                             x: *const f64,
//                             y: *mut f64);
}

#[bench]
fn bench_spmv_csr_mkl(b: &mut test::Bencher) {
    let (m, x) = spmv_setup::<RowMajor, i32>();
    let mut y = Array::from_elem(x.len(), 0.0);
    b.iter(|| unsafe {
        mkl_cspblas_dcsrgemv(
            "n".as_ptr() as *const i8,
            &(m.rows() as i32) as *const i32,
            m.values().as_ptr(),
            m.outer_ptr().as_ptr(),
            m.inner_indices().as_ptr(),
            x.as_ptr(),
            y.as_mut_ptr(),
        )
    })
}

#[bench]
fn bench_spmv_csc(b: &mut test::Bencher) {
    let (m, x) = spmv_setup::<ColMajor, i32>();
    b.iter(|| spmv_sr(&m, &x, |x, y| x + y, |x, y| x * y, 0.0))
}

#[bench]
fn bench_spmv_csc_mkl(b: &mut test::Bencher) {
    let (m, x) = spmv_setup::<ColMajor, i32>();
    let mut y = Array::from_elem(x.len(), 0.0);
    b.iter(|| unsafe {
        // use csr gemv with A transposed
        mkl_cspblas_dcsrgemv(
            "T".as_ptr() as *const i8,
            &(m.rows() as i32) as *const i32,
            m.values().as_ptr(),
            m.outer_ptr().as_ptr(),
            m.inner_indices().as_ptr(),
            x.as_ptr(),
            y.as_mut_ptr(),
        )
    })
}

fn bench_spmm_csr_csc(dim: usize, nnz: usize, b: &mut test::Bencher) {
    let m1: CSRMatrix<_> = rand_matrix(dim, dim, nnz, 1.0);
    let m2: CSCMatrix<_> = rand_matrix(dim, dim, nnz, 1.0);
    b.iter(|| spmm_sr(&m1, &m2, &|x, y| x + y, &|x, y| x * y, 0.0))
}

#[bench]
fn spmm_sr_csr_csc_1000_2000(b: &mut test::Bencher) {
    bench_spmm_csr_csc(1000, 2000, b)
}

fn bench_spmm_csr_csr(dim: usize, nnz: usize, b: &mut test::Bencher) {
    let m1: CSRMatrix<_> = rand_matrix(dim, dim, nnz, 1.0);
    let m2: CSRMatrix<_> = rand_matrix(dim, dim, nnz, 1.0);
    b.iter(|| {
        spmm_sr_csr_csr(&m1, &m2, &|x, y| x + y, &|x, y| x * y, 0.0)
    })
}

#[bench]
fn spmm_sr_csr_csr_opt_1000_2000(b: &mut test::Bencher) {
    bench_spmm_csr_csr(1000, 2000, b)
}

#[bench]
fn spmm_sr_csr_csr_opt_5000_2000(b: &mut test::Bencher) {
    bench_spmm_csr_csr(5000, 2000, b)
}

#[test]
fn spmm_sr_csr_csc_test() {
    let a = CSRMatrix::<f64>::from_dense(
        &arr2(&[[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]),
        0.0,
    );
    let b = CSCMatrix::<f64>::from_dense(
        &arr2(&[[3.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 5.0]]),
        0.0,
    );
    let c = CSRMatrix::<f64>::from_dense(
        &arr2(&[[5.0, 2.0, 3.0], [0.0, 0.0, 5.0], [13.0, 4.0, 32.0]]),
        0.0,
    );
    let res: CSRMatrix<f64> = spmm_sr(&a, &b, &|x, y| x + y, &|x, y| x * y, 0.0);
    assert_eq!(res, c)
}

#[test]
fn spmm_sr_csr_csr_test() {
    let a = CSRMatrix::<f64>::from_dense(
        &arr2(&[[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]),
        0.0,
    );
    let b = CSRMatrix::<f64>::from_dense(
        &arr2(&[[3.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 5.0]]),
        0.0,
    );
    let c = CSRMatrix::<f64>::from_dense(
        &arr2(&[[5.0, 2.0, 3.0], [0.0, 0.0, 5.0], [13.0, 4.0, 32.0]]),
        0.0,
    );
    let res: CSRMatrix<f64> = spmm_sr_csr_csr(&a, &b, &|x, y| x + y, &|x, y| x * y, 0.0);
    assert_eq!(res, c)
}
