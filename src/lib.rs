#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

use derive_more::Deref;
use itertools::{Itertools, zip_eq};
use nalgebra::{
    Const, DVector, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn,
    Matrix, MatrixView, OMatrix, OVector, Owned, RealField, Scalar, U1, VectorView,
    allocator::Allocator,
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Div, DivAssign, Mul, MulAssign, Sub},
};

/// A statically or dynamically sized covariance matrix.
///
/// This type can be constructed directly from a positive semi-definite square matrix or from an ensemble of vectors.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
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
    /// This matrix is computed using the LDL algorithm to account for vanishing eigenvalues.
    cholesky_l: Option<OMatrix<T, D, D>>,

    /// The underlying covariance matrix.
    #[deref]
    matrix: OMatrix<T, D, D>,

    /// The determinant of the covariance matrix.
    ///
    /// This value is computed with the Cholesky decomposition.
    determinant: Option<T>,

    /// The pseudo-determinant of the covariance matrix.
    ///
    /// This value is computed with the Cholesky decomposition.
    pseudo_determinant: Option<T>,

    /// The Moore–Penrose inverse of the underlying covariance matrix.
    pseudo_inverse: OMatrix<T, D, D>,
}

impl<T, D> CovMatrix<T, D>
where
    T: RealField,
    D: Dim + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
{
    /// Compute the cholesky decomposition and the determinant values
    pub fn cholesky(&mut self) {
        let (cholesky_l, determinant, pseudo_determinant) = cholesky_ldl_determinants(&self.matrix)
            .expect("failed to perform the cholesky / ldl decomposition");

        self.cholesky_l = Some(cholesky_l);
        self.determinant = Some(determinant);
        self.pseudo_determinant = Some(pseudo_determinant);
    }

    /// Returns a reference to the lower triangular matrix from the Cholesky decomposition.
    pub fn cholesky_l(&self) -> Option<&OMatrix<T, D, D>> {
        self.cholesky_l.as_ref()
    }

    /// Generate a random sample vector using the covariance matrix.
    pub fn generate_sample(&self, rng: &mut impl Rng) -> OVector<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        let n_dim = self.matrix.shape_generic().0;

        let random_vector = OVector::<T, D>::from_iterator_generic(
            n_dim,
            Const::<1>,
            (0..self.matrix.nrows()).map(|_| rng.sample(StandardNormal)),
        );

        self.cholesky_l.as_ref().unwrap() * random_vector
    }

    /// Returns `true` if the covariance matrix is full rank.
    pub fn is_full_rank(&self) -> bool
    where
        D: DimMin<D, Output = D> + DimSub<U1>,
    {
        let determinant = match &self.determinant {
            Some(value) => value,
            None => &self.matrix.determinant(),
        };

        !(determinant == &T::zero())
    }

    /// Compute the multivariate likelihood using an observation `x` and mean `mu`.
    pub fn likelihood(&self, x: &VectorView<T, D>, mu: &VectorView<T, D>) -> Option<T>
    where
        T: Copy,
        D: DimName + DimMin<D, Output = D> + DimSub<U1>,
        usize: AsPrimitive<T>,
    {
        let delta = DVector::from(x.iter().zip(mu).map(|(i, j)| *i - *j).collect::<Vec<T>>());

        Some(
            -(self.pseudo_determinant?.ln() + self.rank(T::default_epsilon()).as_() * T::two_pi())
                / 2.as_()
                - self.mahalanobis(&delta.as_view()) / 2.as_(),
        )
    }

    /// Compute the (squared) Mahalanobis distance of a vector from the origin.
    pub fn mahalanobis(&self, x: &VectorView<T, D>) -> T {
        (&x.transpose() * &self.pseudo_inverse * x)[(0, 0)].clone()
    }

    /// Create a new [`CovMatrix`] from a positive semi-definite matrix.
    pub fn new(matrix: OMatrix<T, D, D>, cholesky: bool) -> Option<Self> {
        let (cholesky_l, determinant, pseudo_determinant) = match cholesky {
            true => {
                let (cholesky_l, determinant, pseudo_determinant) =
                    cholesky_ldl_determinants(&matrix)?;

                (
                    Some(cholesky_l),
                    Some(determinant),
                    Some(pseudo_determinant),
                )
            }
            false => (None, None, None),
        };

        let pseudo_inverse = {
            let mut pinv = matrix
                .clone_owned()
                .pseudo_inverse(T::default_epsilon())
                .expect("failed to construct pseudo inverse");

            // Zero out rows/columns with zero eigenvalues.
            let dim = matrix.shape_generic().0;

            matrix
                .diagonal()
                .iter()
                .enumerate()
                .for_each(|(idx, value)| {
                    if matches!(
                        value
                            .partial_cmp(&T::zero())
                            .expect("matrix contains NaN values"),
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

        Some(Self {
            cholesky_l,
            matrix,
            determinant,
            pseudo_determinant,
            pseudo_inverse,
        })
    }

    /// Returns the pseudo-determinant of the covariance matrix.
    pub fn pseudo_determinant(&self) -> Option<T> {
        self.pseudo_determinant.clone()
    }

    /// Returns a reference to the inverse of the covariance matrix.
    pub fn pseudo_inverse(&self) -> &OMatrix<T, D, D> {
        &self.pseudo_inverse
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
        let ndim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky_l: self.cholesky_l.map(|value| value / rhs.sqrt()),
            matrix: self.matrix / rhs,
            determinant: self.determinant.map(|value| value / rhs.powi(ndim)),
            pseudo_determinant: self.pseudo_determinant.map(|value| value / rhs.powi(ndim)),
            pseudo_inverse: self.pseudo_inverse * rhs,
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
        let ndim = self.matrix.ncols() as i32;

        self.cholesky_l = self.cholesky_l.clone().map(|value| value / rhs.sqrt());

        self.matrix = &self.matrix / rhs;

        self.determinant = self.determinant.map(|value| value / rhs.powi(ndim));

        self.pseudo_determinant = self.pseudo_determinant.map(|value| value / rhs.powi(ndim));

        self.pseudo_inverse = self.pseudo_inverse.clone() * rhs;
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
        let ndim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky_l: self.cholesky_l.map(|value| value * rhs.sqrt()),
            matrix: self.matrix * rhs,
            determinant: self.determinant.map(|value| value * rhs.powi(ndim)),
            pseudo_determinant: self.pseudo_determinant.map(|value| value * rhs.powi(ndim)),
            pseudo_inverse: self.pseudo_inverse / rhs,
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
        let ndim = self.matrix.ncols() as i32;

        self.cholesky_l = self.cholesky_l.clone().map(|value| value * rhs.sqrt());

        self.matrix = &self.matrix * rhs;

        self.determinant = self.determinant.map(|value| value * rhs.powi(ndim));

        self.pseudo_determinant = self.pseudo_determinant.map(|value| value * rhs.powi(ndim));

        self.pseudo_inverse = self.pseudo_inverse.clone() / rhs;
    }
}

/// Compute the Cholesky decomposition of positive semi-definite `matrix` using the LDL algorithm.
pub fn cholesky_ldl<T, D>(matrix: &OMatrix<T, D, D>) -> Option<OMatrix<T, D, D>>
where
    T: RealField + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    let dim = matrix.shape_generic().0;

    let mut diag = OVector::<T, D>::zeros_generic(dim, Const::<1>);
    let mut cholesky_l = OMatrix::<T, D, D>::zeros_generic(dim, dim);

    for cdx in 0..dim.value() {
        let mut d_j = matrix[(cdx, cdx)].clone();

        if cdx > 0 {
            for k in 0..cdx {
                d_j -= diag[k].clone() * cholesky_l[(cdx, k)].clone().powi(2);
            }
        }

        diag[cdx] = d_j;

        for rdx in cdx..dim.value() {
            let mut l_ij = matrix[(rdx, cdx)].clone();

            for k in 0..cdx {
                l_ij -=
                    diag[k].clone() * cholesky_l[(cdx, k)].clone() * cholesky_l[(rdx, k)].clone();
            }

            if matches!(diag[cdx].partial_cmp(&T::zero())?, Ordering::Equal) {
                cholesky_l[(rdx, cdx)] = T::zero();
            } else {
                cholesky_l[(rdx, cdx)] = l_ij / diag[cdx].clone();
            }
        }
    }

    Some(
        cholesky_l
            * OMatrix::from_diagonal(&OVector::from_iterator_generic(
                dim,
                Const::<1>,
                diag.iter().map(|value| value.clone().sqrt()),
            )),
    )
}

fn cholesky_ldl_determinants<T, D>(matrix: &OMatrix<T, D, D>) -> Option<(OMatrix<T, D, D>, T, T)>
where
    T: RealField,
    D: Dim + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
{
    let decomp = cholesky_ldl(matrix)?;

    let determinant = decomp
        .diagonal()
        .fold(T::one(), |acc, next| acc * next)
        .powi(2);

    let pseudo_determinant = decomp
        .diagonal()
        .fold(
            T::one(),
            |acc, next| {
                if next != T::zero() { acc * next } else { acc }
            },
        )
        .powi(2);

    Some((decomp, determinant, pseudo_determinant))
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

/// Create a [`CovarianceMatrix`] from an ensemble of vectors.
pub fn covmatrix_from_vectors<T, D>(
    vectors: &MatrixView<T, D, Dyn>,
    opt_weights: Option<&[T]>,
    cholesky: bool,
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
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    usize: AsPrimitive<T>,
{
    let n = D::USIZE;

    let mut matrix = OMatrix::<T, D, D>::from_iterator((0..(n.pow(2))).map(|idx| {
        let jdx = idx / n;
        let kdx = idx % n;

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

    CovMatrix::new(matrix, cholesky)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Dyn, Matrix3, U3, VecStorage};
    use rand::SeedableRng;
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
    fn test_covariance_matrix() {
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

        let array_view = &array.as_view();

        let covmat: CovMatrix<f32, U3> = covmatrix_from_vectors(array_view, None, true).unwrap();

        assert!(ulps_eq!(
            covmat.cholesky_l.as_ref().unwrap()[(0, 0)],
            0.40718567
        ));
        assert!(ulps_eq!(
            covmat.cholesky_l.as_ref().unwrap()[(2, 0)],
            0.07841061
        ));

        assert!(ulps_eq!(covmat.pseudo_inverse[(0, 0)], 6.915894));
        assert!(ulps_eq!(covmat.pseudo_inverse[(2, 0)], -4.5933948));

        assert!(ulps_eq!(covmat.pseudo_determinant.unwrap(), 0.0069507807));
    }

    #[test]
    fn test_ldl() {
        let m = Matrix3::new(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0);

        let results = cholesky_ldl_determinants(&m).unwrap();

        // Rebuild
        let p = results.0 * results.0.transpose();

        assert!(ulps_eq!(m, p));
        assert!(ulps_eq!(results.1, 4.0));
        assert!(ulps_eq!(results.2, 4.0));
    }

    #[test]
    fn test_ldl_degenerate() {
        let m = Matrix3::new(2.0, -1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0);

        let results = cholesky_ldl_determinants(&m).unwrap();

        // Rebuild
        let p = results.0 * results.0.transpose();

        assert!(ulps_eq!(m, p));
        assert!(ulps_eq!(results.1, 0.0));
        assert!(ulps_eq!(results.2, 3.0));
    }
}
