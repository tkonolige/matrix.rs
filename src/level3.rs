use ndarray::*;
use mat::*;
use bit_vec::BitVec;
use num_traits;
use std::cmp::*;
use std::fmt::Debug;

use level1::*;

/// Sparse matrix - dense matrix multiplication on a semiring
// TODO: is sparse-dense product sparse or dense?
pub fn spmdm_sr<LT, RT, I, F, S, AF, MF, Order, S1, S2, S3>(
    mat: &CompressedSparseMatrix<LT, I, Order, S1, S2, S3>,
    x: &ArrayBase<S, Ix2>,
    add: AF,
    mult: MF,
    zero: F,
) -> Array<F, Ix2>
where
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
    S: DataClone<Elem = RT>,
    Order: Storage,
    LT: Clone,
    RT: Clone,
    F: Clone,
    I: Index,
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
{
    let mut y = Array::<F, Ix2>::from_elem((mat.size().0, x.shape()[1]), zero);
    for (i, j, v) in mat.iter() {
        for r in 0..x.shape()[1] {
            y[(i, r)] = add(&y[(i, r)], &mult(v, &x[(j, r)]));
        }
    }

    y
}

/// Structure to accumulate the results of sparse operations
/// See http://gauss.cs.ucsb.edu/~aydin/GALLA-sparse.pdf
/// TODO: look at better accumulator
/// Invariant: if used[i] then values[i] is initialized
/// Invariant: if used[i] then i in indices
struct SparseAccumulator<V, I>
where
    I: Index,
{
    indices: Vec<I>,
    values: Vec<V>,
    used: BitVec,
}

impl<V, I> SparseAccumulator<V, I>
where
    V: Clone,
    I: Index,
{
    pub fn new(length: usize, zero: V) -> Self {
        SparseAccumulator {
            indices: Vec::new(),
            values: vec![zero; length],
            used: BitVec::from_elem(length, false),
        }
    }

    /// Set a value in the accumulator
    pub fn set<F>(&mut self, index: I, value: V, add: &F)
    where
        F: Fn(&V, &V) -> V,
    {
        if !self.used[index.to_usize()] {
            self.indices.push(index.clone());
            self.values[index.to_usize()] = value;
            self.used.set(index.to_usize(), true);
        } else {
            self.values[index.to_usize()] = add(&self.values[index.to_usize()], &value)
        }
    }

    /// Remove all values from the accumulator and put them into the passed
    /// vectors
    /// Returns the number of nonzero entries
    pub fn drain_into(&mut self, inds: &mut Vec<I>, values: &mut Vec<V>) -> usize
    {
        // sort indices so we get them out in the right order
        self.indices.sort(); // TODO: linear time sort?
        let nzcount = self.indices.len();
        for i in self.indices.iter() {
            self.used.set(i.to_usize(), false);
        }

        // append new values
        // TODO: no clone?
        values.extend(
            self.indices
                .iter()
                .map(|i| self.values[i.to_usize()].clone()),
        );
        inds.append(&mut self.indices);
        nzcount
    }
}


/// Sparse matrix - sparse matrix multiplication on a semiring
/// This version uses a dot product on every row and column
pub fn spmm_sr<LT, RT, I, F, AF, MF, S1, S2, S3, S4, S5, S6>(
    lhs: &CompressedSparseMatrix<LT, I, RowMajor, S1, S2, S3>,
    rhs: &CompressedSparseMatrix<RT, I, ColMajor, S4, S5, S6>,
    add: &AF,
    mult: &MF,
    zero: F,
) -> CompressedSparseMatrix<F, I, RowMajor, Vec<F>, Vec<I>, Vec<I>>
where
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
    LT: Clone,
    RT: Clone,
    F: Clone,
    I: Index,
    F: PartialEq<F>,
    F: num_traits::Zero,
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    S4: DataClone<Elem = RT>,
    S5: DataClone<Elem = I>,
    S6: DataClone<Elem = I>,
{
    assert_eq!(lhs.size().1, rhs.size().0);

    // Create result matrix
    let nnz_est = max(lhs.nnz(), rhs.nnz());
    let mut row_ptr = Vec::with_capacity(lhs.size().0);
    let mut col_ind = Vec::<I>::with_capacity(nnz_est);
    let mut values = Vec::with_capacity(nnz_est);

    for (_, row) in lhs.outer_iter() {
        // add end of previous row, start of new one
        row_ptr.push(I::from_usize(values.len()));
        for (ci, col) in rhs.outer_iter() {
            let r = dot_ss_sr(&row, &col, add, mult, zero.clone());
            if r != zero {
                col_ind.push(I::from_usize(ci));
                values.push(r);
            }
        }
    }

    // push end of final row
    row_ptr.push(I::from_usize(values.len()));

    CompressedSparseMatrix::from_parts(lhs.size().0, rhs.size().1, row_ptr, col_ind, values)
}

/// Sparse matrix - sparse matrix multiplication on a semiring
/// see http://gauss.cs.ucsb.edu/~aydin/GALLA-sparse.pdf figure 13.11
/// This version uses a sparse accumulator to compute rows in the resulting
/// matrix
pub fn spmm_sr_csr_csr<LT, RT, I, F, AF, MF, S1, S2, S3, S4, S5, S6>(
    lhs: &CompressedSparseMatrix<LT, I, RowMajor, S1, S2, S3>,
    rhs: &CompressedSparseMatrix<RT, I, RowMajor, S4, S5, S6>,
    add: &AF,
    mult: &MF,
    zero: F,
) -> CompressedSparseMatrix<F, I, RowMajor, Vec<F>, Vec<I>, Vec<I>>
where
    AF: Fn(&F, &F) -> F,
    MF: Fn(&LT, &RT) -> F,
    LT: Clone,
    RT: Clone,
    F: Clone,
    I: Index,
    I: Debug,
    F: PartialEq<F>,
    F: num_traits::Zero,
    S1: DataClone<Elem = LT>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    S4: DataClone<Elem = RT>,
    S5: DataClone<Elem = I>,
    S6: DataClone<Elem = I>,
{
    assert_eq!(lhs.size().1, rhs.size().0);

    let mut spa = SparseAccumulator::new(lhs.size().0, zero);
    let nnz_est = max(lhs.nnz(), rhs.nnz());
    let mut row_ptr = Vec::<I>::with_capacity(lhs.size().0);
    row_ptr.push(I::from_usize(0));
    let mut col_ind = Vec::<I>::with_capacity(nnz_est);
    let mut values = Vec::with_capacity(nnz_est);

    // for each row in left matrix
    for i in 0..lhs.size().0 {
        // for each entry in left matrix row
        for k in lhs.outer_ptr()[i].to_usize()..lhs.outer_ptr()[i + 1].to_usize() {
            let column_idx = lhs.inner_indices()[k].to_usize();
            // for each entry in right matrix row == column index of left entry
            for j in
                rhs.outer_ptr()[column_idx].to_usize()..rhs.outer_ptr()[column_idx + 1].to_usize()
            {
                let value = mult(&lhs.values()[k], &rhs.values()[j]);
                spa.set(rhs.inner_indices()[j].clone(), value, &add);
            }
        }
        let nzcnt = spa.drain_into(&mut col_ind, &mut values);
        let next_row = row_ptr[i].clone() + I::from_usize(nzcnt);
        row_ptr.push(next_row);
    }

    CompressedSparseMatrix::from_parts(lhs.size().0, rhs.size().1, row_ptr, col_ind, values)
}
