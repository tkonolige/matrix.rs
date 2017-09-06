use std::ops::*;
use std::marker::PhantomData;
use std::cmp;
use std::iter;
use num_integer::*;
use num_traits;
use std::iter::TrustedLen;
use rand::*;
use std::fmt::Display;

use ndarray::*;
use itertools::*;

/// Indices used within sparse structure. These indices must be smaller or
/// equal in size to
/// `usize`.
pub trait Index: Integer + Clone + iter::Step + Add + Ord + Display {
    /// Conversion from native indices. Native index size must be >= to Self
    /// size.
    fn from_usize(usize) -> Self;

    /// Conversion to native indices. Native index size must be >= to Self size.
    fn to_usize(&self) -> usize;
}

impl Index for usize {
    fn from_usize(x: usize) -> Self {
        x
    }
    fn to_usize(&self) -> usize {
        *self
    }
}

impl Index for u32 {
    fn from_usize(x: usize) -> Self {
        x as Self
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Index for u64 {
    fn from_usize(x: usize) -> Self {
        x as Self
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Index for isize {
    fn from_usize(x: usize) -> Self {
        x as Self
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Index for i32 {
    fn from_usize(x: usize) -> Self {
        x as Self
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Index for i64 {
    fn from_usize(x: usize) -> Self {
        x as Self
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

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
    /// Converts a tuple (row, column) to (outer, inner). Ex. (1,2) in ColMajor
    /// (CSC) becomes (2,1)
    /// and RowMajor becomes (1,2)
    fn reorder_ix<I>(ixs: (I, I)) -> (I, I);
}

impl Storage for RowMajor {
    type Output = RowMajor;
    type Transpose = ColMajor;
    fn reorder_ix<I>(ixs: (I, I)) -> (I, I) {
        ixs
    }
}

impl Storage for ColMajor {
    type Output = ColMajor;
    type Transpose = RowMajor;
    fn reorder_ix<I>(ixs: (I, I)) -> (I, I) {
        (ixs.1, ixs.0)
    }
}

// TODO: use numeric tower to guarantee indices are logical?
/// Compressed sparse matrix in CSC or CSR format
/// Indices are of the form (row, column)
/// Storage is handled by `ndarray`
#[derive(Debug, Clone)]
pub struct CompressedSparseMatrix<T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    Order: Storage,
    I: Index,
{
    values: ArrayBase<S1, Ix1>, // TODO: should be able to use views here
    outer_ptr: ArrayBase<S2, Ix1>,
    inner_indices: ArrayBase<S3, Ix1>,
    shape: (usize, usize), // (rows, columns)
    phantom: PhantomData<Order>,
}

pub type OwnedSparseMatrix<T, S, I = usize> = CompressedSparseMatrix<
    T,
    I,
    S,
    Vec<T>,
    Vec<I>,
    Vec<I>,
>;

/// Compressed Sparse Column matrix
pub type CSCMatrix<T, I = usize> = CompressedSparseMatrix<T, I, ColMajor, Vec<T>, Vec<I>, Vec<I>>;
/// View into a `CSCMatrix`
pub type CSCView<T, I = usize> = CompressedSparseMatrix<
    T,
    I,
    ColMajor,
    ViewRepr<T>,
    ViewRepr<I>,
    ViewRepr<I>,
>;
/// Compressed Sparse Row matrix
pub type CSRMatrix<T, I = usize> = CompressedSparseMatrix<T, I, RowMajor, Vec<T>, Vec<I>, Vec<I>>;
/// View into a `CSRMatrix`
pub type CSRView<T, I = usize> = CompressedSparseMatrix<
    T,
    I,
    RowMajor,
    ViewRepr<T>,
    ViewRepr<I>,
    ViewRepr<I>,
>;

// TODO: PartialEq for CSR and CSC
impl<
    Order,
    T,
    I,
    LS1,
    LS2,
    LS3,
    RS1,
    RS2,
    RS3,
> PartialEq<CompressedSparseMatrix<T, I, Order, RS1, RS2, RS3>>
    for CompressedSparseMatrix<T, I, Order, LS1, LS2, LS3>
where
    Order: Storage,
    LS1: DataClone<Elem = T>,
    LS2: DataClone<Elem = I>,
    LS3: DataClone<Elem = I>,
    RS1: DataClone<Elem = T>,
    RS2: DataClone<Elem = I>,
    RS3: DataClone<Elem = I>,
    T: PartialEq,
    T: Clone,
    I: Index,
    I: Clone,
{
    fn eq(&self, other: &CompressedSparseMatrix<T, I, Order, RS1, RS2, RS3>) -> bool {
        self.outer_ptr() == other.outer_ptr() && self.shape == other.shape &&
            self.inner_indices() == other.inner_indices() && self.values() == other.values()
    }
}

impl<'a, T, I, Order, S1, S2, S3> CompressedSparseMatrix<T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    Order: Storage,
    I: Index,
    I: Clone,
    T: Clone,
{
    /// Iterator over the nonzero values of the matrix as `(row, column, value)`
    pub fn iter(&'a self) -> NNZIterator<'a, T, I, Order, S1, S2, S3> {
        NNZIterator {
            mat: self,
            outer_idx: 0,
            inner_idx: 0,
        }
    }

    /// Iterator over the outermost dimension of this matrix. (Columns if
    /// `ColMajor`, Rows if
    /// `RowMajor`).
    pub fn outer_iter(&'a self) -> OuterIterator<'a, T, I, Order, S1, S2, S3> {
        OuterIterator {
            mat: self,
            outer_idx: 0,
        }
    }

    /// Size of the matrix
    pub fn size(&'a self) -> (usize, usize) {
        self.shape
    }

    /// Number of rows in this matrix
    pub fn rows(&'a self) -> usize {
        self.shape.0
    }

    /// Number of columns in this matrix
    pub fn cols(&'a self) -> usize {
        self.shape.1
    }

    /// Number of nonzero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Transpose view of the matrix
    pub fn transpose(
        &'a self,
    ) -> CompressedSparseMatrix<
        T,
        I,
        Order::Transpose,
        ViewRepr<&'a T>,
        ViewRepr<&'a I>,
        ViewRepr<&'a I>,
    > {
        CompressedSparseMatrix {
            outer_ptr: self.outer_ptr.view(),
            inner_indices: self.inner_indices.view(),
            values: self.values.view(),
            shape: (self.shape.1, self.shape.0),
            phantom: PhantomData,
        }
    }

    /// Create a unique copy of a matrix
    pub fn to_owned(&self) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>> {
        CompressedSparseMatrix {
            outer_ptr: self.outer_ptr.to_owned(),
            inner_indices: self.inner_indices.to_owned(),
            values: self.values.to_owned(),
            shape: self.shape,
            phantom: PhantomData,
        }
    }

    // TODO: Construct a sparse matrix from a triple of owned iterators

    /// View into the nonzero values of the matrix
    pub fn values(&self) -> ArrayView1<T> {
        self.values.view()
    }

    /// View into the outer pointer of the matrix
    pub fn outer_ptr(&'a self) -> ArrayView1<'a, I> {
        self.outer_ptr.view()
    }

    /// View into the inner indices of the matrix
    pub fn inner_indices(&self) -> ArrayView1<I> {
        self.inner_indices.view()
    }

    /// Validate internal invariants. Will panic if there is an error
    pub fn check_invariants(&self) {
        match self.validate_invariants() {
            Some(err) => panic!(err),
            None => (),
        }
    }

    /// Check if sparse matrix invariants hold
    pub fn validate_invariants(&self) -> Option<String> {
        // check that outer pointers are sorted
        if self.outer_ptr[0].to_usize() != 0 {
            return Some("Outer pointer must start with 0".to_owned());
        }
        let mut last = I::from_usize(0);
        for i in 1..self.outer_ptr.len() {
            let next = self.outer_ptr[i].clone();
            if next < last {
                return Some("Outer pointers are not sorted".to_owned());
            }
            if last.to_usize() > self.inner_indices.len() {
                return Some("Outer pointer points past end of inner storage".to_owned());
            }
            last = next;
        }

        // check that inner indices and nonzeros are the same length
        if self.inner_indices.len() != self.values.len() {
            return Some(
                "Inner indices and values are not the same length".to_owned(),
            );
        }

        // check that all inner indices are valid and in the correct order
        let inner_dim = Order::reorder_ix(self.shape).1;
        for (_, row) in self.outer_iter() {
            for i in 0..row.indices.len() {
                let v = row.indices[i].clone();
                if i > 0 && v <= row.indices[i-1] {
                    return Some("Inner indices are not sorted correctely".to_owned())
                }
                if v < I::from_usize(0) || v >= I::from_usize(inner_dim) {
                    return Some(format!(
                        "Invalid inner index {} should be less than {}",
                        v,
                        inner_dim
                    ));
                }
            }
        }

        None
    }

    // /// Construct a CompressedSparseMatrix from a `Vec<(T,I,I)>`
    // pub fn from_tuples(vec: Vec<(T,I,I)>) {
    //     let (x,y,z) = vec.into_iter().unzip3();
    //     Self::from_vecs(x,y,z)
    // }
}

// Methods that contruct an owned sparse matrix
impl<'a, T, I, Order> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>>
where
    Order: Storage,
    I: Index,
    I: Clone,
    T: Clone,
{
    /// Sparse identity matrix
    pub fn diagm(
        m: usize,
        n: usize,
        val: T,
    ) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>>
    where
        for<'b> &'b I: Add<&'b I, Output = I>,
    {
        let min_dim = cmp::min(m, n);
        let ks = Array::<T, Ix1>::from_elem::<Ix1>(Dim(min_dim.to_usize()), val);
        let is = Array::<I, Ix1>::from_iter(Range {
            start: I::zero(),
            end: I::from_usize(min_dim),
        });
        let js = is.clone();
        Self::from_ndarrays(is, js, ks, m, n, |x, _| x)
    }

    /// Construct a sparse matrix from a dense one
    /// Will remove zero elements
    pub fn from_dense(dense: &Array2<T>, zero: T) -> Self
    where
        T: PartialEq,
    {
        let mut is = Vec::<I>::new();
        let mut js = Vec::<I>::new();
        let mut ks = Vec::<T>::new();
        for ((x, y), v) in dense.indexed_iter() {
            if v != &zero {
                is.push(I::from_usize(x));
                js.push(I::from_usize(y));
                ks.push(v.clone());
            }
        }

        Self::from_vecs(is, js, ks, dense.shape()[0], dense.shape()[1], |x, _| x)
    }

    /// Construct a CSR or CSC matrix from raw information. The user is
    /// responsible for ensuring
    /// the correct CSR or CSC format.
    /// TODO: write down invariants
    pub unsafe fn from_parts_unsafe(
        rows: usize,
        cols: usize,
        outer_ptr: Vec<I>,
        inner_indices: Vec<I>,
        values: Vec<T>,
    ) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>> {
        CompressedSparseMatrix {
            outer_ptr: Array1::from_vec(outer_ptr),
            inner_indices: Array1::from_vec(inner_indices),
            values: Array1::from_vec(values),
            shape: (rows, cols),
            phantom: PhantomData,
        }
    }

    /// Construct a CSR or CSC matrix from raw information.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        outer_ptr: Vec<I>,
        inner_indices: Vec<I>,
        values: Vec<T>,
    ) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>> {
        let res = unsafe { Self::from_parts_unsafe(rows, cols, outer_ptr, inner_indices, values) };
        res.check_invariants();
        res
    }
    /// Empty sparse matrix of the given size
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: num_traits::Zero,
    {
        CompressedSparseMatrix {
            outer_ptr: Array::<I, Ix1>::zeros(Order::reorder_ix((rows, cols)).0),
            inner_indices: Array::<I, Ix1>::zeros(0),
            values: Array::<T, Ix1>::zeros(0),
            shape: (rows, cols),
            phantom: PhantomData,
        }
    }

    /// Construct an `rows` by `cols` Sparse Matrix from `i`,`j`,`k` vectors.
    /// Each entry in `ijk`
    /// corresponds to a `(row,column,value)` tuple. `add_dups` controls how
    /// duplicates are handled.
    /// Zero values are not removed.
    pub fn from_vecs<F>(
        i: Vec<I>,
        j: Vec<I>,
        k: Vec<T>, // TODO: take ArrayBase here
        rows: usize,
        cols: usize,
        add_dups: F,
    ) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>>
    where
        F: Fn(T, T) -> T,
        Order: Storage,
        T: Clone,
    {
        // TODO: if we have lots of duplicates, we will over allocate
        let nnz_guess = j.len();

        // resolve duplicates
        let deduped = multizip((i, j, k)).map(|(x, y, v)| {
                // reorder ij so that major axis comes second
                let t = Order::reorder_ix((x, y));
                (t.0, t.1, v)
            })

            // sort by outer_ptr, then inner_indices indices
            .sorted_by(|x, y| Ord::cmp(&(&x.0, &x.1), &(&y.0, &y.1))).into_iter()
                .coalesce(|(x1, y1, v1), (x2, y2, v2)| if x1 == x2 && y1 == y2 {
                    // merge duplicates
                    Ok((x1, y1, add_dups(v1, v2)))
                } else {
                    Err(((x1, y1, v1), (x2, y2, v2)))
                });

        let (outer_size, inner_size) = Order::reorder_ix((rows, cols));
        let mut outer = Vec::<I>::with_capacity(outer_size + 1);
        let mut inner = Vec::<I>::with_capacity(nnz_guess);
        let mut values = Vec::<T>::with_capacity(nnz_guess);

        // outer column/row starts at 0 index
        outer.push(I::zero());

        let mut cur_outer: usize = 0;
        for (k, group) in deduped.group_by(|x| x.0.clone()).into_iter() {
            assert!(k >= I::zero() && k.to_usize() < outer_size);

            // empty major vectors will not appear in group by, we need to catch up if we
            // are
            // behind
            while cur_outer < k.to_usize() {
                outer.push(I::from_usize(values.len()));
                cur_outer += 1;
            }

            // insert entries for inner column/row
            for (_, j, k) in group {
                assert!(j >= I::zero() && j.to_usize() < inner_size);
                inner.push(j);
                values.push(k);
            }

            // start of next inner column/row
            // outer.push(I::from_usize(values.len()));
            // assert!(outer[outer.len()-1].to_usize() < inner_size);
            // cur_outer += 1;
        }

        // fill outer columns/rows until we reach the desired matrix dimensions
        while cur_outer < outer_size {
            outer.push(I::from_usize(values.len()));
            cur_outer += 1;
        }

        let res = CompressedSparseMatrix {
            outer_ptr: Array::<I, Ix1>::from_vec(outer),
            inner_indices: Array::<I, Ix1>::from_vec(inner),
            values: Array::<T, Ix1>::from_vec(values),
            shape: (rows, cols),
            phantom: PhantomData,
        };

        res.check_invariants();

        res
    }

    /// Construct a sparse matrix from `ndarray`'s. See `from_vecs` for details
    pub fn from_ndarrays<F>(
        i: Array<I, Ix1>,
        j: Array<I, Ix1>,
        k: Array<T, Ix1>, // TODO: take ArrayBase here
        rows: usize,
        cols: usize,
        add_dups: F,
    ) -> CompressedSparseMatrix<T, I, Order, Vec<T>, Vec<I>, Vec<I>>
    where
        F: Fn(T, T) -> T,
        Order: Storage,
        T: Clone,
        I: Clone,
    {
        Self::from_vecs(
            i.into_raw_vec(),
            j.into_raw_vec(),
            k.into_raw_vec(),
            rows,
            cols,
            add_dups,
        )
    }
}

/// Iterator over all nonzero matrix entries in efficient matter
pub struct NNZIterator<'a, T: 'a, I: 'a, Order: 'a + Storage, S1: 'a, S2: 'a, S3: 'a>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    I: Index,
{
    mat: &'a CompressedSparseMatrix<T, I, Order, S1, S2, S3>,
    outer_idx: usize,
    inner_idx: usize,
}

impl<'a, T, I, Order, S1, S2, S3> Iterator for NNZIterator<'a, T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    Order: Storage,
    I: Index,
    T: Clone,
{
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let m = Order::reorder_ix(self.mat.shape).0;
            if m == self.outer_idx {
                return None;
            }

            while self.mat.outer_ptr.uget(self.outer_idx + 1).to_usize() == self.inner_idx {
                self.outer_idx += 1;
                if m == self.outer_idx {
                    return None;
                }
                self.inner_idx = self.mat.outer_ptr.uget(self.outer_idx).to_usize();
            }

            let v = &self.mat.values.uget(self.inner_idx); // TODO: avoid bounds checking
            let i: usize = self.mat.inner_indices.uget(self.inner_idx).to_usize();
            let o = self.outer_idx;
            self.inner_idx += 1;

            let (i, j) = Order::reorder_ix((o, i));
            Some((i, j, v))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = self.mat.nnz() - self.inner_idx;
        (s, Some(s))
    }

    fn count(self) -> usize {
        self.size_hint().0
    }
}

impl<'a, T, I, Order, S1, S2, S3> ExactSizeIterator for NNZIterator<'a, T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    Order: Storage,
    I: Index,
    T: Clone,
{
}

unsafe impl<'a, T, I, Order, S1, S2, S3> TrustedLen for NNZIterator<'a, T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    Order: Storage,
    I: Index,
    T: Clone,
{
}

/// Iterator over outer structure, elements are sparse vectors corresponding to
/// rows or columns
pub struct OuterIterator<'a, T: 'a, I: 'a, Order: 'a + Storage, S1: 'a, S2: 'a, S3: 'a>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    I: Index,
{
    mat: &'a CompressedSparseMatrix<T, I, Order, S1, S2, S3>,
    outer_idx: usize,
}

impl<'a, T, I, Order: Storage, S1, S2, S3> Iterator for OuterIterator<'a, T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    I: Index,
{
    type Item = (
        usize,
        CompressedSparseVector<T, I, ViewRepr<&'a T>, ViewRepr<&'a I>>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        let (m, n) = Order::reorder_ix(self.mat.shape); // m is outer size, n is inner size
        if m > self.outer_idx {
            // still in bounds
            let slice = s![
                self.mat.outer_ptr[self.outer_idx].to_usize() as Ixs..
                    self.mat.outer_ptr[self.outer_idx + 1].to_usize() as Ixs
            ];
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (m, _) = Order::reorder_ix(self.mat.shape);
        (0, Some(m - self.outer_idx))
    }
}

impl<'a, T, I, Order: Storage, S1, S2, S3> ExactSizeIterator
    for OuterIterator<'a, T, I, Order, S1, S2, S3>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    S3: DataClone<Elem = I>,
    I: Index,
{
}

/// Sparse vector
#[derive(Debug, PartialEq, Clone)]
pub struct CompressedSparseVector<T, I = usize, S1 = Vec<T>, S2 = Vec<I>>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    I: Index,
{
    pub values: ArrayBase<S1, Ix1>,
    pub indices: ArrayBase<S2, Ix1>,
    shape: usize,
}

impl<T, I, S1, S2> CompressedSparseVector<T, I, S1, S2>
where
    S1: DataClone<Elem = T>,
    S2: DataClone<Elem = I>,
    I: Index,
{
    /// Length of this sparse vector
    pub fn len(&self) -> usize {
        self.shape
    }

    /// Number of nonzero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterator over nonzero elements
    pub fn iter(&self) -> Iter<T, Ix1> {
        self.values.iter()
    }

    /// Iterator over nonzero elements and their indices
    pub fn indexed_iter(&self) -> SparseVectorIterator<T, I> {
        SparseVectorIterator {
            index: self.indices.iter(),
            value: self.values.iter(),
        }
    }

    pub fn new(
        n: usize,
        is: ArrayBase<S2, Ix1>,
        vs: ArrayBase<S1, Ix1>,
    ) -> CompressedSparseVector<T, I, S1, S2> {
        assert!(is.len() == vs.len());
        assert!(is[is.len() - 1].to_usize() < n);
        CompressedSparseVector {
            shape: n,
            indices: is,
            values: vs,
        }
    }
}

impl<T, I> CompressedSparseVector<T, I, Vec<T>, Vec<I>>
where
    I: Index,
    T: Clone,
{
    /// Construct a vector of length `n` with `nnz` nonzero values
    pub fn rand(n: usize, nnz: usize, val: T) -> Self {
        let mut inds = Vec::<I>::new();
        for _ in 0..nnz {
            inds.push(I::from_usize(thread_rng().gen_range(0, n)));
        }
        let values = Array1::from_elem(nnz, val);
        CompressedSparseVector::new(n, Array1::from_shape_vec(nnz, inds).unwrap(), values)
    }
}

pub struct SparseVectorIterator<'a, T, I>
where
    T: 'a,
    I: 'a + Index,
{
    index: Iter<'a, I, Ix1>,
    value: Iter<'a, T, Ix1>,
}

impl<'a, T, I> Iterator for SparseVectorIterator<'a, T, I>
where
    I: Index,
{
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.index.next(), self.value.next()) {
            (Some(i), Some(v)) => Some((i.to_usize(), v)),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }

    fn count(self) -> usize {
        self.index.count()
    }
}

impl<'a, T, I> ExactSizeIterator for SparseVectorIterator<'a, T, I>
where
    T: 'a,
    I: 'a + Index,
{
}

unsafe impl<'a, T, I> TrustedLen for SparseVectorIterator<'a, T, I>
where
    T: 'a,
    I: 'a + Index,
{
}

pub trait SparseMatrix<V, I: Index> {
    // TODO: use iterators? or ArrayBase?
    fn from_coords(usize, usize, Vec<(I, I, V)>) -> Self;

    // Slice the matrix A[i, j] = B where B has all indices in i and j
    // fn slice(Vec<I>, Vec<I>) -> Self;
}

impl<V: Clone, I: Index, Order: Storage> SparseMatrix<V, I> for OwnedSparseMatrix<V, Order, I> {
    fn from_coords(m: usize, n: usize, ijk: Vec<(I, I, V)>) -> Self {
        let i = ijk.iter().map(|x| x.0.clone()).collect();
        let j = ijk.iter().map(|x| x.1.clone()).collect();
        let k = ijk.iter().map(|x| x.2.clone()).collect();
        Self::from_vecs(i, j, k, m, n, |x, _| x) // TODO: resolve better?
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
