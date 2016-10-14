use std::ops::*;
use std::num::*;
use std::marker::PhantomData;

extern crate ndarray;
use ndarray::*;

#[derive(Debug, PartialEq, Clone)]
struct RowMajor;
#[derive(Debug, PartialEq, Clone)]
struct ColMajor;

trait Storage {
    type Output;
}

impl Storage for RowMajor {
    type Output = RowMajor;
}

impl Storage for ColMajor {
    type Output = ColMajor;
}

#[derive(Debug, PartialEq, Clone)]
pub struct CompressedSparseMatrix<T, Order: Storage> {
    values: Vec<T>, // TODO: should be able to use views here
    outside: Vec<Ix>,
    inside: Vec<Ix>,
    shape: (Ix, Ix),
    phantom: PhantomData<Order>,
}

impl<'a, T, Order: Storage> CompressedSparseMatrix<T, Order> {
    pub fn iter(&'a self) -> NNZIterator<'a, T, Order> {
        NNZIterator {
            mat: &self,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

trait CompressedMat<Order: Storage> {
    fn reorder_ix((Ix, Ix)) -> (Ix, Ix);
}

impl<T> CompressedMat<ColMajor> for CompressedSparseMatrix<T, ColMajor> {
    fn reorder_ix(ixs: (Ix, Ix)) -> (Ix, Ix) {
        (ixs.1, ixs.0)
    }
}

impl<T> CompressedMat<ColMajor> for CompressedSparseMatrix<T, RowMajor> {
    fn reorder_ix(ixs: (Ix, Ix)) -> (Ix, Ix) {
        ixs
    }
}

// iterator over all matrix entries in efficient matter
struct NNZIterator<'a, T: 'a, Order: 'a + Storage> {
    mat: &'a CompressedSparseMatrix<T, Order>,
    outer_idx: Ix,
    inner_idx: Ix,
}

impl<'a, T, Order: Storage> Iterator for NNZIterator<'a, T, Order> {
    type Item = (Ix, Ix, T);

    fn next(&mut self) -> Option<(Ix, Ix, T)> {
        if self.outer_idx == CompressedSparseMatrix::<T, Order>::reorder_ix(self.mat.shape).0 {
            return None;
        }

        let v = self.mat.values[self.inner_idx];
        let i = self.mat.inside[self.inner_idx];
        let o = self.outer_idx;
        if self.inner_idx == self.mat.outside[self.outer_idx] - 1 {
            self.outer_idx += 1;
            self.inner_idx = self.mat.outside[self.outer_idx];
        } else {
            self.inner_idx += 1;
        }

        let (i, j) = CompressedSparseMatrix::reorder_ix((o, i));
        Some((i, j, v))
    }
}

// Sparse matrix stored with either compressed columns or rows
#[derive(Debug, PartialEq, Clone)]
pub struct CompressedSparseVector<T, Ix> {
    values: Vec<T>,
    is: Vec<Ix>,
    shape: Ix,
}

pub fn multAA<T, AF, MF, Order: Storage>(A: CompressedSparseMatrix<T, Order>,
                                         B: CompressedSparseMatrix<T, Order>,
                                         add: AF,
                                         mult: MF)
                                         -> CompressedSparseMatrix<T, Order>
    where AF: Fn(T, T) -> T,
          MF: Fn(T, T) -> T
{
}

pub fn multAspV<T, AF, MF, Order: Storage>(A: CompressedSparseMatrix<T, Order>,
                                           B: CompressedSparseVector<T, Order>,
                                           add: AF,
                                           mult: MF)
                                           -> CompressedSparseVector<T, Order>
    where AF: Fn(T, T) -> T,
          MF: Fn(T, T) -> T
{
}

pub fn multAv<T: ndarray::Data, AF, MF, Order: Storage>(A: CompressedSparseMatrix<T, Order>,
                                                        x: ArrayBase<T, Order>,
                                                        add: AF,
                                                        mult: MF)
                                                        -> Array<T, Order>
    where AF: Fn(T, T) -> T,
          MF: Fn(T, T) -> T
{
    let y = Array::zeros(x.shape());
    for (i, j, v) in A.iter() {
        y[i] = v * x[j]
    }

    y
}
