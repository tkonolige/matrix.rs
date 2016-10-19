use ndarray::*;
use mat::*;

pub fn spmdm_sr<LT, RT, F, S, AF, MF, Order, S1, S2, S3>(mat: &CompressedSparseMatrix<LT,
                                                                                      Order,
                                                                                      S1,
                                                                                      S2,
                                                                                      S3>,
                                                         x: &ArrayBase<S, (Ix, Ix)>,
                                                         add: AF,
                                                         mult: MF,
                                                         zero: F)
                                                         -> Array<F, (Ix, Ix)>
    where AF: Fn(&F, &F) -> F,
          MF: Fn(&LT, &RT) -> F,
          S: DataClone<Elem = RT>,
          Order: Storage,
          LT: Clone,
          RT: Clone,
          F: Clone,
          S1: DataClone<Elem = LT>,
          S2: DataClone<Elem = Ix>,
          S3: DataClone<Elem = Ix>
{
    let mut y = Array::<F, (Ix, Ix)>::from_elem((mat.size().0, x.shape()[1]), zero);
    for (i, j, v) in mat.iter() {
        for r in 0..x.shape()[1] {
            y[(i, r)] = add(&y[(i, r)], &mult(v, &x[(j, r)]));
        }
    }

    y
}
