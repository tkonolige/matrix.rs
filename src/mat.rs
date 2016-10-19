use std::ops::*;
use std::marker::PhantomData;
use std::cmp;

use ndarray::*;
use itertools::*;

/// Matrix stored in CSR format
#[derive(Debug, PartialEq, Clone)]
pub struct RowMajor;
/// Matrix stored in CSC format
#[derive(Debug, PartialEq, Clone)]
pub struct ColMajor;

/// Trait for controlling storage order of matrix
pub trait Storage {
    type Output;
    type Transpose: Storage;
    /// Converts a tuple (row, column) to (outer, inner). Ex. (1,2) in ColMajor (CSC) becomes (2,1)
    /// and RowMajor becomes (1,2)
    fn reorder_ix(ixs: (Ix, Ix)) -> (Ix, Ix);
}

impl Storage for RowMajor {
    type Output = RowMajor;
    type Transpose = ColMajor;
    fn reorder_ix(ixs: (Ix, Ix)) -> (Ix, Ix) {
        ixs
    }
}

impl Storage for ColMajor {
    type Output = ColMajor;
    type Transpose = RowMajor;
    fn reorder_ix(ixs: (Ix, Ix)) -> (Ix, Ix) {
        (ixs.1, ixs.0)
    }
}

/// Compressed sparse matrix in CSC or CSR format
/// Indices are of the form (row, column)
/// Storage is handled by `ndarray`
#[derive(Debug, Clone)]
pub struct CompressedSparseMatrix<T, Order, S1, S2, S3>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>,
          Order: Storage
{
    values: ArrayBase<S1, Ix>, // TODO: should be able to use views here
    outer_ptr: ArrayBase<S2, Ix>,
    inner_indices: ArrayBase<S3, Ix>,
    shape: (Ix, Ix), // (rows, columns)
    phantom: PhantomData<Order>,
}

/// Compressed Sparse Column matrix
pub type CSCMatrix<T> = CompressedSparseMatrix<T, ColMajor, Vec<T>, Vec<Ix>, Vec<Ix>>;
/// View into a `CSCMatrix`
pub type CSCView<T> = CompressedSparseMatrix<T, ColMajor, ViewRepr<T>, ViewRepr<Ix>, ViewRepr<Ix>>;
/// Compressed Sparse Row matrix
pub type CSRMatrix<T> = CompressedSparseMatrix<T, RowMajor, Vec<T>, Vec<Ix>, Vec<Ix>>;
/// View into a `CSRMatrix`
pub type CSRView<T> = CompressedSparseMatrix<T, RowMajor, ViewRepr<T>, ViewRepr<Ix>, ViewRepr<Ix>>;

impl<LOrder, ROrder, T, LS1, LS2, LS3, RS1, RS2, RS3>
PartialEq<CompressedSparseMatrix<T, ROrder, RS1, RS2, RS3>>
for CompressedSparseMatrix<T, LOrder, LS1, LS2, LS3>
    where LOrder: Storage,
          ROrder: Storage,
          LS1: DataClone<Elem=T>,
          LS2: DataClone<Elem=Ix>,
          LS3: DataClone<Elem=Ix>,
          RS1: DataClone<Elem=T>,
          RS2: DataClone<Elem=Ix>,
          RS3: DataClone<Elem=Ix>,
          T: PartialEq,
          T: Clone,
          T: Ord
{
    fn eq(&self, other: &CompressedSparseMatrix<T, ROrder, RS1, RS2, RS3>) -> bool {
        self.iter().sorted() == other.iter().sorted() &&
        self.shape == other.shape
    }
}

impl<LOrder, T, LS1, LS2, LS3> Eq for CompressedSparseMatrix<T, LOrder, LS1, LS2, LS3>
    where LOrder: Storage,
          LS1: DataClone<Elem = T>,
          LS2: DataClone<Elem = Ix>,
          LS3: DataClone<Elem = Ix>,
          T: PartialEq,
          T: Clone,
          T: Ord
{
}

impl<'a, T, Order, S1, S2, S3> CompressedSparseMatrix<T, Order, S1, S2, S3>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>,
          Order: Storage,
          T: Clone
{
    /// Iterator over the nonzero values of the matrix as `(row, column, value)`
    pub fn iter(&'a self) -> NNZIterator<'a, T, Order, S1, S2, S3> {
        NNZIterator {
            mat: self,
            outer_idx: 0,
            inner_idx: 0,
        }
    }

    /// Size of the matrix
    pub fn size(&'a self) -> (Ix, Ix) {
        self.shape
    }

    /// Number of nonzero elements
    pub fn nnz(&self) -> Ix {
        self.values.len()
    }

    /// Transpose view of the matrix
    pub fn transpose(&'a self)
                     -> CompressedSparseMatrix<T,
                                               Order::Transpose,
                                               ViewRepr<&'a T>,
                                               ViewRepr<&'a Ix>,
                                               ViewRepr<&'a Ix>> {
        CompressedSparseMatrix {
            outer_ptr: self.outer_ptr.view(),
            inner_indices: self.inner_indices.view(),
            values: self.values.view(),
            shape: (self.shape.1, self.shape.0),
            phantom: PhantomData,
        }
    }

    /// Create a unique copy of a matrix
    pub fn to_owned(&self) -> CompressedSparseMatrix<T, Order, Vec<T>, Vec<Ix>, Vec<Ix>> {
        CompressedSparseMatrix {
            outer_ptr: self.outer_ptr.to_owned(),
            inner_indices: self.inner_indices.to_owned(),
            values: self.values.to_owned(),
            shape: self.shape,
            phantom: PhantomData,
        }
    }

    /// Sparse identity matrix
    pub fn diagm(m: Ix, n: Ix, val: T) -> CompressedSparseMatrix<T, Order, Vec<T>, Vec<Ix>, Vec<Ix>>
    {
        let min_dim = cmp::min(m, n);
        let ks = Array::from_elem(min_dim, val);
        let is = Array::from_iter(0..min_dim);
        let js = is.clone();
        Self::from_ndarrays(is, js, ks, m, n, |x, _| x)
    }

    /// Construct a sparse matrix from `ndarray`'s. See `from_vecs` for details
    pub fn from_ndarrays<F>
        (i: Array<Ix, Ix>,
         j: Array<Ix, Ix>,
         k: Array<T, Ix>, // TODO: take ArrayBase here
         rows: Ix,
         cols: Ix,
         add_dups: F)
         -> CompressedSparseMatrix<T, Order, Vec<T>, Vec<Ix>, Vec<Ix>>
        where F: Fn(T, T) -> T,
              Order: Storage,
              T: Clone,
    {
        Self::from_vecs(i.into_raw_vec(),
                        j.into_raw_vec(),
                        k.into_raw_vec(),
                        rows,
                        cols,
                        add_dups)
    }

    /// Construct an `rows` by `cols` Sparse Matrix from `i`,`j`,`k` vectors. Each entry in `ijk`
    /// corresponds to a `(row,column,value)` tuple. `add_dups` controls how duplicates are handled.
    /// Zero values are not removed.
    pub fn from_vecs<F>(i: Vec<Ix>,
                        j: Vec<Ix>,
                        k: Vec<T>, // TODO: take ArrayBase here
                        rows: Ix,
                        cols: Ix,
                        add_dups: F)
                        -> CompressedSparseMatrix<T, Order, Vec<T>, Vec<Ix>, Vec<Ix>>
        where F: Fn(T, T) -> T,
              Order: Storage,
              T: Clone
    {
        // TODO: if we have lots of duplicates, we will over allocate
        let nnz_guess = j.len();

        let deduped =
            multizip((i, j, k)).map(|(x, y, v)| { // reorder ij so that major axis comes second
        let t = Order::reorder_ix((x, y));
        (t.0, t.1, v)
    })
    // sort by outer_ptr, then inner_indices indices
    .sorted_by(|x, y| Ord::cmp(&(x.0, x.1), &(y.0, y.1))).into_iter()
    .coalesce(|(x1, y1, v1), (x2, y2, v2)| if x1 == x2 && y1 == y2 { // merge duplicates
        Ok((x1, y1, add_dups(v1, v2)))
    } else {
        Err(((x1, y1, v1), (x2, y2, v2)))
    });

        let (outer_size, inner_size) = Order::reorder_ix((rows, cols));
        let mut outer = Vec::with_capacity(outer_size + 1);
        let mut inner = Vec::with_capacity(nnz_guess);
        let mut values = Vec::with_capacity(nnz_guess);
        outer.push(0);
        let mut cur_outer = 0;
        for (k, group) in deduped.group_by(|x| x.0).into_iter() {
            assert!(k >= 0 && k < outer_size);

            // empty major vectors will not appear in group by, we need to catch up if we are
            // behind
            while cur_outer < k {
                outer.push(values.len());
                cur_outer += 1;
            }

            for (_, j, k) in group {
                assert!(j >= 0 && j < inner_size);
                inner.push(j);
                values.push(k);
            }
            outer.push(values.len());
            cur_outer += 1;
        }
        while cur_outer < outer_size {
            outer.push(values.len());
            cur_outer += 1;
        }

        CompressedSparseMatrix {
            outer_ptr: Array::<Ix, Ix>::from_vec(outer),
            inner_indices: Array::<Ix, Ix>::from_vec(inner),
            values: Array::<T, Ix>::from_vec(values),
            shape: (rows, cols),
            phantom: PhantomData,
        }
    }
}

/// Iterator over all nonzero matrix entries in efficient matter
pub struct NNZIterator<'a, T: 'a, Order: 'a + Storage, S1: 'a, S2: 'a, S3: 'a>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>
{
    mat: &'a CompressedSparseMatrix<T, Order, S1, S2, S3>,
    outer_idx: Ix,
    inner_idx: Ix,
}

impl<'a, T, Order, S1, S2, S3> Iterator for NNZIterator<'a, T, Order, S1, S2, S3>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>,
          Order: Storage
{
    type Item = (Ix, Ix, &'a T);

    fn next(&mut self) -> Option<(Ix, Ix, &'a T)> {
        let m = Order::reorder_ix(self.mat.shape).0;
        if self.outer_idx == m {
            return None;
        }

        while self.inner_idx == self.mat.outer_ptr[self.outer_idx + 1] {
            self.outer_idx += 1;
            if self.outer_idx == m {
                return None;
            }
            self.inner_idx = self.mat.outer_ptr[self.outer_idx];
        }

        let v = &self.mat.values[self.inner_idx]; // TODO: avoid bounds checking
        let i = self.mat.inner_indices[self.inner_idx];
        let o = self.outer_idx;
        self.inner_idx += 1;

        let (i, j) = Order::reorder_ix((o, i));
        Some((i, j, v))
    }
}

/// Iterator over outer structure, elements are sparse vectors corresponding to rows or columns
pub struct OuterIterator<'a, T: 'a, Order: 'a + Storage, S1: 'a, S2: 'a, S3: 'a>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>
{
    mat: &'a CompressedSparseMatrix<T, Order, S1, S2, S3>,
    outer_idx: Ix,
}

impl<'a, T, Order: Storage, S1, S2, S3> Iterator for OuterIterator<'a, T, Order, S1, S2, S3>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>
{
    type Item = (Ix, CompressedSparseVector<T, ViewRepr<&'a T>, ViewRepr<&'a Ix>>);

    fn next(&mut self) -> Option<Self::Item> {
        let (m, n) = Order::reorder_ix(self.mat.shape); // m is outer size, n is inner size
        let slice = s![self.mat.outer_ptr[self.outer_idx] as Ixs ..
                       self.mat.outer_ptr[self.outer_idx + 1] as Ixs];
        if self.outer_idx < m {
            let spvec = CompressedSparseVector {
                values: self.mat.values.slice(slice),
                indices: self.mat.inner_indices.slice(slice),
                shape: n,
            };
            self.outer_idx += 1;
            Some((self.outer_idx - 1, spvec))
        } else {
            None
        }
    }
}

/// Sparse vector
#[derive(Debug, PartialEq, Clone)]
pub struct CompressedSparseVector<T, S1, S2>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>
{
    values: ArrayBase<S1, Ix>,
    indices: ArrayBase<S2, Ix>,
    shape: Ix,
}

impl<T, S1, S2> CompressedSparseVector<T, S1, S2>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>
{
    pub fn len(&self) -> Ix {
        self.shape
    }

    pub fn nnz(&self) -> Ix {
        self.values.len()
    }

    pub fn iter(&self) -> SparseVectorIterator<T, S1, S2> {
        SparseVectorIterator {
            vec: self,
            index: 0,
        }
    }

    pub fn new(n: Ix,
               is: ArrayBase<S2, Ix>,
               vs: ArrayBase<S1, Ix>)
               -> CompressedSparseVector<T, S1, S2> {
        CompressedSparseVector {
            shape: n,
            indices: is,
            values: vs,
        }
    }
}

pub struct SparseVectorIterator<'a, T, S1, S2>
    where S1: 'a + DataClone<Elem = T>,
          S2: 'a + DataClone<Elem = Ix>,
          T: 'a
{
    vec: &'a CompressedSparseVector<T, S1, S2>,
    index: Ix,
}

impl<'a, T, S1, S2> Iterator for SparseVectorIterator<'a, T, S1, S2>
    where S1: DataClone<Elem = T>,
          S2: DataClone<Elem = Ix>
{
    type Item = (Ix, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.indices.len() {
            let r = (self.vec.indices[self.index], &self.vec.values[self.index]);
            self.index += 1;
            Some(r)
        } else {
            None
        }
    }
}


#[test]
fn sparse_from_vecs_csc() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSCMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    assert_eq!(mat.outer_ptr, arr1(&[0, 1, 2, 3, 3]));
    assert_eq!(mat.inner_indices, arr1(&[0, 1, 1]));
    assert_eq!(mat.values, arr1(&[6, 5, 7]));
}

#[test]
fn sparse_from_vecs_csr() {
    let i = vec![1, 0, 1, 1];
    let j = vec![1, 0, 1, 2];
    let k = vec![2, 6, 3, 7];
    let mat = CSRMatrix::from_vecs(i, j, k, 3, 4, |x, y| x + y);
    assert_eq!(mat.outer_ptr, arr1(&[0, 1, 3, 3]));
    assert_eq!(mat.inner_indices, arr1(&[0, 1, 2]));
    assert_eq!(mat.values, arr1(&[6, 5, 7]));
}
