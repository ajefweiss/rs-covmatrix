#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

use itertools::{Itertools, zip_eq};
use nalgebra::{
    Const, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn, Matrix,
    MatrixView, OMatrix, OVector, Owned, RealField, Scalar, U1, VectorView, allocator::Allocator,
};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Div, DivAssign, Mul, MulAssign, Sub},
};

/// A covariance matrix.
///
/// This type represents a statically or dynamically sized covariance matrix.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "T: Serialize, OMatrix<T, D, D>: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>"))]
pub struct CovMatrix<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    /// The lower triangular matrix from the Cholesky decomposition.
    ///
    /// This matrix is computed using a variant of the LDL algorithm to account for dimensions with zero variance.
    l: Option<OMatrix<T, D, D>>,

    /// The underlying covariance matrix.
    m: OMatrix<T, D, D>,

    /// The determinant & pseudo-determinant of the covariance matrix.
    ///
    /// This value is computed together with the Cholesky decomposition, and is thus only stored when the `l` field is also set. When the `l` field is not set, the determinant, or pseudo-determinant, values are calculated on the fly.
    d: Option<(T, T)>,

    /// The Moore–Penrose inverse of the underlying covariance matrix.
    i: OMatrix<T, D, D>,
}

impl<T, D> CovMatrix<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    /// Compute the Cholesky decomposition using a variant of the LDL algorithm and compute the determinant and pseudo-determinant values.
    ///
    /// This method uses the LDL algorithm to compute the lower triangular matrix of the Cholesky decomposition,
    /// further accounting for dimensions with zero variance. These dimensions are zero'd out.
    pub fn cholesky_ldl(&mut self) {
        let dim = self.m.shape_generic().0;

        let mut d = OVector::<T, D>::zeros_generic(dim, Const::<1>);
        let mut l = OMatrix::<T, D, D>::zeros_generic(dim, dim);

        for cdx in 0..dim.value() {
            let mut d_j = self.m[(cdx, cdx)].clone();

            if cdx > 0 {
                for k in 0..cdx {
                    d_j -= d[k].clone() * l[(cdx, k)].clone().powi(2);
                }
            }

            d[cdx] = d_j;

            for rdx in cdx..dim.value() {
                let mut l_ij = self.m[(rdx, cdx)].clone();

                for k in 0..cdx {
                    l_ij -= d[k].clone() * l[(cdx, k)].clone() * l[(rdx, k)].clone();
                }

                if matches!(
                    d[cdx]
                        .partial_cmp(&T::zero())
                        .expect("matrix contains NaN values"),
                    Ordering::Equal
                ) {
                    l[(rdx, cdx)] = T::zero();
                } else {
                    l[(rdx, cdx)] = l_ij / d[cdx].clone();
                }
            }
        }

        self.l = Some(
            l * OMatrix::from_diagonal(&OVector::from_iterator_generic(
                dim,
                Const::<1>,
                d.iter().map(|value| value.clone().sqrt()),
            )),
        );

        self.d = Some((
            d.fold(T::one(), |acc, next| acc * next),
            d.fold(
                T::one(),
                |acc, next| {
                    if next != T::zero() { acc * next } else { acc }
                },
            ),
        ));
    }

    /// Returns the determinant of the covariance matrix.
    pub fn determinant(&self) -> Option<T> {
        Some(self.d.as_ref()?.0.clone())
    }

    /// Create a [`CovMatrix`] from a matrix, with optional weights.
    pub fn from_vectors<RStride, CStride>(
        vectors: &MatrixView<T, D, Dyn, RStride, CStride>,
        opt_weights: Option<&[T]>,
        compute_ld: bool,
    ) -> Option<CovMatrix<T, D>>
    where
        T: Copy
            + RealField
            + for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        D: DimName + DimMin<D>,
        RStride: Dim,
        CStride: Dim,
        DimMinimum<D, D>: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<DimMinimum<D, D>, U1>>
            + Allocator<DimMinimum<D, D>, D>
            + Allocator<D, DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
        usize: AsPrimitive<T>,
    {
        let n_dim = D::USIZE;

        let mut matrix = OMatrix::<T, D, D>::from_iterator((0..(n_dim.pow(2))).map(|idx| {
            let jdx = idx / n_dim;
            let kdx = idx % n_dim;

            if jdx <= kdx {
                let x = vectors.row(jdx);
                let y = vectors.row(kdx);

                if !x.iter().all_equal() && !y.iter().all_equal() {
                    match opt_weights {
                        Some(w) => covariance_with_weights(x, y, w),
                        None => covariance(x, y),
                    }
                } else {
                    T::zero()
                }
            } else {
                T::zero()
            }
        }));

        // Fill up the other side of the matrix.
        matrix += matrix.transpose() - OMatrix::from_diagonal(&matrix.diagonal());

        CovMatrix::new(matrix, compute_ld)
    }

    /// Returns a reference to the lower triangular matrix from the Cholesky decomposition.
    pub fn l(&self) -> Option<&OMatrix<T, D, D>> {
        self.l.as_ref()
    }

    /// Compute the multivariate likelihood using an observation vector `x` and mean vector `mu`.
    pub fn likelihood(&self, x: &VectorView<T, D>, mu: &VectorView<T, D>) -> Option<T>
    where
        T: Copy,
        D: DimName + Dim,
        usize: AsPrimitive<T>,
    {
        let delta = OVector::<T, D>::from_iterator(x.iter().zip(mu).map(|(i, j)| *i - *j));

        let pd = self.d.as_ref()?.1;

        Some(
            -(pd.ln() + self.rank().as_() * T::two_pi()) / 2.as_()
                - self.mahalanobis_distance(&delta.as_view()) / 2.as_(),
        )
    }

    /// Returns the (squared) Mahalanobis distance of a vector from the origin.
    pub fn mahalanobis_distance<RStride, CStride>(
        &self,
        x: &VectorView<T, D, RStride, CStride>,
    ) -> T
    where
        RStride: Dim,
        CStride: Dim,
    {
        (&x.transpose() * &self.i * x)[(0, 0)].clone()
    }

    /// Returns a reference to the underlying covariance matrix.
    pub fn matrix(&self) -> &OMatrix<T, D, D> {
        &self.m
    }

    /// Create a new [`CovMatrix`] from a positive semi-definite matrix.
    pub fn new(matrix: OMatrix<T, D, D>, compute_ld: bool) -> Option<Self>
    where
        D: DimMin<D>,
        DimMinimum<D, D>: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<DimMinimum<D, D>, U1>>
            + Allocator<DimMinimum<D, D>, D>
            + Allocator<D, DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    {
        let inverse = {
            let mut pinv = matrix
                .clone_owned()
                .pseudo_inverse(T::default_epsilon())
                .expect("failed to compute pseudo inverse");

            // Zero out rows/columns with vanishing variance.
            let dim = matrix.shape_generic().0;

            matrix
                .diagonal()
                .iter()
                .enumerate()
                .for_each(|(idx, value)| {
                    if matches!(
                        value
                            .partial_cmp(&T::zero())
                            .expect("covariance matrix contains NaN values"),
                        Ordering::Equal
                    ) {
                        pinv.set_column(idx, &OVector::<T, D>::zeros_generic(dim, Const::<1>));
                        pinv.set_row(
                            idx,
                            &Matrix::<T, U1, D, Owned<T, U1, D>>::zeros_generic(Const::<1>, dim),
                        );
                    }
                });

            pinv
        };

        let mut new = Self {
            l: None,
            m: matrix,
            d: None,
            i: inverse,
        };

        if compute_ld {
            new.cholesky_ldl();
        }

        Some(new)
    }

    /// Returns the pseudo-determinant of the covariance matrix.
    pub fn pseudo_determinant(&self) -> Option<T> {
        Some(self.d.as_ref()?.1.clone())
    }

    /// Returns a reference to the pseudo-inverse of the covariance matrix.
    pub fn pseudo_inverse(&self) -> &OMatrix<T, D, D> {
        &self.i
    }

    /// Returns the rank of the covariance matrix.
    ///
    /// The rank is calculated by counting the non-zero entries along the diagonal of the covariance matrix.
    pub fn rank(&self) -> usize {
        self.m
            .diagonal()
            .fold(0, |acc, next| if next != T::zero() { acc + 1 } else { acc })
    }
}

impl<T, D> Div<T> for CovMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = CovMatrix<T, D>;

    fn div(self, rhs: T) -> Self::Output {
        let n_dim = self.m.ncols() as i32;

        Self::Output {
            l: self.l.map(|value| value / rhs.sqrt()),
            m: self.m / rhs,
            d: self
                .d
                .map(|(d, pd)| (d / rhs.powi(n_dim), pd / rhs.powi(n_dim))),

            i: self.i * rhs,
        }
    }
}

impl<T, D> DivAssign<T> for CovMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn div_assign(&mut self, rhs: T) {
        let n_dim = self.m.ncols() as i32;

        self.l = self.l.clone().map(|value| value / rhs.sqrt());
        self.m = &self.m / rhs;
        self.d = self
            .d
            .map(|(d, pd)| (d / rhs.powi(n_dim), pd / rhs.powi(n_dim)));
        self.i = self.i.clone() * rhs;
    }
}

impl<T, D> Mul<T> for CovMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = CovMatrix<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        let n_dim = self.m.ncols() as i32;

        Self::Output {
            l: self.l.map(|value| value * rhs.sqrt()),
            m: self.m * rhs,
            d: self
                .d
                .map(|(d, pd)| (d * rhs.powi(n_dim), pd * rhs.powi(n_dim))),

            i: self.i / rhs,
        }
    }
}

impl<T, D> MulAssign<T> for CovMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn mul_assign(&mut self, rhs: T) {
        let n_dim = self.m.ncols() as i32;

        self.l = self.l.clone().map(|value| value * rhs.sqrt());
        self.m = &self.m * rhs;
        self.d = self
            .d
            .map(|(d, pd)| (d * rhs.powi(n_dim), pd * rhs.powi(n_dim)));
        self.i = self.i.clone() / rhs;
    }
}

/// Computes the unbiased covariance over two slices.
///
/// The length of both iterators must be equal (panic).
pub fn covariance<'a, T, I>(x: I, y: I) -> T
where
    T: Copy + RealField + for<'x> Sub<&'x T, Output = T> + Sum + for<'x> Sum<&'x T>,
    I: IntoIterator<Item = &'a T>,
    <I as IntoIterator>::IntoIter: Clone,
    usize: AsPrimitive<T>,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    let mu_x = x_iter.clone().sum::<T>() / length.as_();
    let mu_y = x_iter.clone().sum::<T>() / length.as_();

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - val_x) * (mu_y - val_y))
        .sum::<T>()
        / (length - 1).as_()
}

/// Computes the unbiased covariance over two slices with weights.
///
/// The length of all three iterators must be equal (panic).
pub fn covariance_with_weights<'a, T, IV, IW>(x: IV, y: IV, w: IW) -> T
where
    T: Copy
        + RealField
        + for<'x> Mul<&'x T, Output = T>
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    IV: IntoIterator<Item = &'a T>,
    IW: IntoIterator<Item = &'a T>,
    <IV as IntoIterator>::IntoIter: Clone,
    <IW as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let w_iter = w.into_iter();

    let wsum = w_iter.clone().sum::<T>();
    let wsumsq = w_iter.clone().map(|val_w| val_w.powi(2)).sum::<T>();

    let wfac = wsum - wsumsq / wsum;

    let mu_x = zip_eq(x_iter.clone(), w_iter.clone())
        .map(|(val_x, val_w)| val_x * val_w)
        .sum::<T>()
        / wsum;

    let mu_y = zip_eq(y_iter.clone(), w_iter.clone())
        .map(|(val_y, val_w)| val_y * val_w)
        .sum::<T>()
        / wsum;

    zip_eq(x_iter, zip_eq(y_iter, w_iter))
        .map(|(val_x, (val_y, val_w))| (mu_x - val_x) * (mu_y - val_y) * val_w)
        .sum::<T>()
        / wfac
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Dyn, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_covariance() {
        assert!(ulps_eq!(
            covariance(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0]
            ),
            5.75
        ));

        assert!(ulps_eq!(
            covariance_with_weights(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0],
                &[1.0, 0.8, 1.1, 1.3, 0.9]
            ),
            19.523237
        ));
    }

    #[test]
    fn test_covmatrix() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = Uniform::new(-0.5, 0.5).unwrap();

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10,
            (0..30).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    (idx % 3) as f32 + rng.sample(uniform)
                }
            }),
        );

        let covmat =
            CovMatrix::<f32, U3>::from_vectors::<Dyn, U3>(&array.as_view(), None, true).unwrap();

        assert!(ulps_eq!(covmat.l.as_ref().unwrap()[(0, 0)], 0.40718567));
        assert!(ulps_eq!(covmat.l.as_ref().unwrap()[(2, 0)], 0.07841061));

        assert!(ulps_eq!(covmat.i[(0, 0)], 6.915894));
        assert!(ulps_eq!(covmat.i[(2, 0)], -4.5933948));

        assert!(ulps_eq!(covmat.d.as_ref().unwrap().1, 0.0069507807));
    }
}
