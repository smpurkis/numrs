use num_traits:: {One, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use rayon::iter::{
    IntoParallelRefIterator,
    ParallelIterator,
};
use std::{
    cmp::min,
    fmt::{Debug, Display},
    marker::Sync,
    ops::{Add, Div, Mul, Sub},
};

/// 1D Array
///
///
/// Uses a Vec internally.
/// Takes ownership of the data
///
/// # Example
/// ```
/// use array_nd::Array1D;
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let array: Array1D<f64> = Array1D::new(data);
/// assert_eq!(array.data, vec![1.0, 2.0, 3.0]);
/// ```
#[derive(Clone)]
pub struct Array1D<T: Copy> {
    pub data: Vec<T>,
    max: T,
    min: T,
    shape: (usize,),
    size: usize,
}

impl<T> Array1D<T>
where
    T: Copy
        + Zero
        + One
        + std::cmp::PartialOrd
        + Send
        + Sync
        + rand::distributions::uniform::SampleUniform,
    Standard: Distribution<T>,
{
    /// Creates a new 1D Array
    ///
    /// # Example
    /// ```
    /// use array_nd::Array1D;
    /// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let array: Array1D<f64> = Array1D::new(data);
    /// ```
    pub fn new(data: Vec<T>) -> Array1D<T> {
        let min = find_min(&data);
        let max = find_max(&data);
        Array1D {
            shape: (data.len() as usize,),
            size: data.len(),
            data,
            min,
            max,
        }
    }

    /// Sums the data inside 1D Array
    ///
    /// Uses a sequential sum when the Array size is small (less than 1 million)
    ///
    /// Uses a parallel sum when the Array size is large (greater than 1 million)
    ///
    /// # Example
    /// ```
    /// use array_nd::Array1D;
    /// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let array: Array1D<f64> = Array1D::new(data);
    /// assert_eq!(array.sum(), 6.0);
    /// ```
    pub fn sum(&self) -> T {
        if self.size > 1_000_000 {
            self.par_sum()
        } else {
            self.seq_sum()
        }
    }

    /// Seqential Sum used inside 1D Array
    pub fn seq_sum(&self) -> T {
        self.data.iter().fold(T::zero(), |sum, &val| sum + val)
    }

    /// Parallel Sum used inside 1D Array
    pub fn par_sum(&self) -> T {
        self.data
            .par_iter()
            .fold(|| T::zero(), |sum, &val| sum + val)
            .collect::<Vec<T>>()
            .iter()
            .fold(T::zero(), |sum, &val| sum + val)
    }

    /// Generates a random 1D Array
    ///
    /// # Example
    /// ```
    /// use array_nd::Array1D;
    /// let array: Array1D<f64> = Array1D::random(10);
    /// ```
    pub fn random(size: usize) -> Array1D<T> {
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..size).map(|_| rng.gen::<T>()).collect();
        Array1D::new(data)
    }

    /// Generates a random 1D Array with a range
    ///
    /// # Example
    /// ```
    /// use array_nd::Array1D;
    /// let array: Array1D<i64> = Array1D::random_range(10, 1, 10);
    /// ```
    pub fn random_range(size: usize, min: T, max: T) -> Array1D<T> {
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..size).map(|_| rng.gen_range(min..max)).collect();
        Array1D::new(data)
    }
}

/// Generates a 1D Array with consecutive numbers
///
/// # Example
/// ```
/// use array_nd::Array1D;
/// let array: Array1D<i64> = Array1D::arange(10);
/// ```
pub fn arange(size: i64) -> Array1D<i64> {
    let data: Vec<i64> = (0..size).collect();
    Array1D::new(data)
}

fn find_min<T: Copy + Zero + std::cmp::PartialOrd>(data: &[T]) -> T {
    data.iter()
        .reduce(|x, y| if x < y { x } else { y })
        .cloned()
        .unwrap()
}

fn find_max<T: Copy + Zero + std::cmp::PartialOrd>(data: &[T]) -> T {
    data.iter()
        .reduce(|x, y| if x > y { x } else { y })
        .cloned()
        .unwrap()
}

impl<T> Display for Array1D<T>
where
    T: Copy + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Array2D, min: {:?}, max: {:?}, data[..100] {:?}",
            self.min,
            self.max,
            &self.data[..100]
        )
    }
}

impl<T> Add<Array1D<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Add<Output = T> + std::cmp::PartialOrd,
{
    type Output = Array1D<T>;

    fn add(self, rhs: Self) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.shape.0);
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs.data).collect();
        let data = data.iter().map(|(i, j)| i.clone() + j.clone()).collect();
        let min = {
            if lhs.min < rhs.min {
                lhs.min
            } else {
                rhs.min
            }
        };
        let max = {
            if lhs.max < rhs.max {
                lhs.max
            } else {
                rhs.max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl<T> Sub<Array1D<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Sub<Output = T> + std::cmp::PartialOrd,
{
    type Output = Array1D<T>;

    fn sub(self, rhs: Self) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.shape.0);
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs.data).collect();
        let data = data.iter().map(|(i, j)| i.clone() - j.clone()).collect();
        let min = {
            if lhs.min < rhs.min {
                lhs.min
            } else {
                rhs.min
            }
        };
        let max = {
            if lhs.max < rhs.max {
                lhs.max
            } else {
                rhs.max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl<T> Mul<Array1D<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Mul<Output = T> + std::cmp::PartialOrd,
{
    type Output = Array1D<T>;

    fn mul(self, rhs: Self) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.shape.0);
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs.data).collect();
        let data = data.iter().map(|(i, j)| i.clone() * j.clone()).collect();
        let min = {
            if lhs.min < rhs.min {
                lhs.min
            } else {
                rhs.min
            }
        };
        let max = {
            if lhs.max < rhs.max {
                lhs.max
            } else {
                rhs.max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl<T> Div<Array1D<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Div<Output = T> + std::cmp::PartialOrd,
{
    type Output = Array1D<T>;

    fn div(self, rhs: Self) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.shape.0);
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs.data).collect();
        let data = data.iter().map(|(i, j)| i.clone() / j.clone()).collect();
        let min = {
            if lhs.min < rhs.min {
                lhs.min
            } else {
                rhs.min
            }
        };
        let max = {
            if lhs.max < rhs.max {
                lhs.max
            } else {
                rhs.max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl<T> Add<Vec<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Add<Output = T> + std::cmp::PartialOrd + Zero,
{
    type Output = Array1D<T>;

    fn add(self, rhs: Vec<T>) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.len());
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<T> = data.iter().map(|(i, j)| i.clone() + j.clone()).collect();
        let rhs_min = find_min(&data);
        let rhs_max = find_min(&data);
        let min = {
            if lhs.min < rhs_min {
                lhs.min
            } else {
                rhs_min
            }
        };
        let max = {
            if lhs.max < rhs_max {
                lhs.max
            } else {
                rhs_max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl<T> Sub<Vec<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Sub<Output = T> + std::cmp::PartialOrd + Zero,
{
    type Output = Array1D<T>;

    fn sub(self, rhs: Vec<T>) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.len());
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<T> = data.iter().map(|(i, j)| i.clone() - j.clone()).collect();
        let rhs_min = find_min(&data);
        let rhs_max = find_min(&data);
        let min = {
            if lhs.min < rhs_min {
                lhs.min
            } else {
                rhs_min
            }
        };
        let max = {
            if lhs.max < rhs_max {
                lhs.max
            } else {
                rhs_max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl<T> Mul<Vec<T>> for Array1D<T>
where
    T: Copy + Debug + std::cmp::PartialOrd + Zero + std::ops::Mul<Output = T>,
{
    type Output = Array1D<T>;

    fn mul(self, rhs: Vec<T>) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.len());
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<T> = data.iter().map(|(i, j)| i.clone() * j.clone()).collect();
        let rhs_min = find_min(&data);
        let rhs_max = find_min(&data);
        let min = {
            if lhs.min < rhs_min {
                lhs.min
            } else {
                rhs_min
            }
        };
        let max = {
            if lhs.max < rhs_max {
                lhs.max
            } else {
                rhs_max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl<T> Div<Vec<T>> for Array1D<T>
where
    T: Copy + Debug + std::ops::Div<Output = T> + std::cmp::PartialOrd + Zero,
{
    type Output = Array1D<T>;

    fn div(self, rhs: Vec<T>) -> Array1D<T> {
        let lhs = self;
        assert_eq!(lhs.shape.0, rhs.len());
        let data: Vec<(T, T)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<T> = data.iter().map(|(i, j)| i.clone() / j.clone()).collect();
        let rhs_min = find_min(&data);
        let rhs_max = find_min(&data);
        let min = {
            if lhs.min < rhs_min {
                lhs.min
            } else {
                rhs_min
            }
        };
        let max = {
            if lhs.max < rhs_max {
                lhs.max
            } else {
                rhs_max
            }
        };

        Array1D {
            shape: (lhs.shape.0,),
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl<T> Add<T> for Array1D<T>
where
    T: Copy + Debug + std::cmp::PartialOrd + Zero + std::ops::AddAssign,
{
    type Output = Array1D<T>;

    fn add(mut self, rhs: T) -> Array1D<T> {
        for el in self.data.iter_mut() {
            *el += rhs
        }
        self.min += rhs;
        self.max += rhs;
        self
    }
}

impl<T> Sub<T> for Array1D<T>
where
    T: Copy + Debug + std::cmp::PartialOrd + Zero + std::ops::SubAssign,
{
    type Output = Array1D<T>;

    fn sub(mut self, rhs: T) -> Array1D<T> {
        for el in self.data.iter_mut() {
            *el -= rhs
        }
        self.min -= rhs;
        self.max -= rhs;
        self
    }
}
impl<T> Mul<T> for Array1D<T>
where
    T: Copy + Debug + std::cmp::PartialOrd + Zero + std::ops::MulAssign,
{
    type Output = Array1D<T>;

    fn mul(mut self, rhs: T) -> Array1D<T> {
        for el in self.data.iter_mut() {
            *el *= rhs
        }
        self.min *= rhs;
        self.max *= rhs;
        self
    }
}

impl<T> Div<T> for Array1D<T>
where
    T: Copy + Debug + std::cmp::PartialOrd + Zero + std::ops::DivAssign,
{
    type Output = Array1D<T>;

    fn div(mut self, rhs: T) -> Array1D<T> {
        for el in self.data.iter_mut() {
            *el /= rhs
        }
        self.min /= rhs;
        self.max /= rhs;
        self
    }
}

impl<T> Debug for Array1D<T>
where
    T: Copy + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shape.0 > 100 {
            let print_limit = min(self.shape.0, 100);
            write!(
                f,
                "Array2D {:?}, min: {:?}, max: {:?}, data[..{:?}] {:?}...",
                self.shape,
                self.min,
                self.max,
                print_limit,
                &self.data[..print_limit]
            )
        } else {
            write!(
                f,
                "Array2D {:?}, min: {:?}, max: {:?}, data: {:?}",
                self.shape,
                self.min,
                self.max,
                &self.data[..self.shape.0]
            )
        }
    }
}

impl<T: std::marker::Copy + std::cmp::PartialEq> PartialEq<Array1D<T>> for Array1D<T> {
    fn eq(&self, other: &Array1D<T>) -> bool {
        self.data == other.data
    }

    fn ne(&self, other: &Array1D<T>) -> bool {
        self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use crate::numrs::Array1D;
    use crate::numrs;

    fn get_array_1d_integer() -> Array1D<i32> {
        Array1D::new(vec![1, 2, 3, 4, 5, 6, 7])
    }

    fn get_array_1d_float() -> Array1D<f64> {
        Array1D::new(vec![1., 2., 3., 4., 5., 6., 7.])
    }

    #[test]
    fn add_integer() {
        let data_addition_mult = get_array_1d_integer()
            + get_array_1d_integer()
            + get_array_1d_integer()
            + get_array_1d_integer()
            + get_array_1d_integer();
        let array1 = get_array_1d_integer();
        let array2 = get_array_1d_integer();
        let data_addition = array1.clone() + array2;

        assert_eq!(data_addition, Array1D::new(vec![2, 4, 6, 8, 10, 12, 14]));
        assert_eq!(
            array1.clone() + vec![1, 2, 3, 4, 5, 6, 7],
            Array1D::new(vec![2, 4, 6, 8, 10, 12, 14])
        );
        assert_eq!(array1 + 1, Array1D::new(vec![2, 3, 4, 5, 6, 7, 8]));

        let expected_array = Array1D::new(vec![5, 10, 15, 20, 25, 30, 35]);
        assert_eq!(data_addition_mult, expected_array);

        let incorrect_array: Array1D<i32> = Array1D::new(vec![2, 4, 6, 8, 10, 12, 14, 16]);
        assert_ne!(data_addition, incorrect_array);

        let incorrect_array = Array1D::new(vec![2, 4, 6, 8, 10, 12, 123]);
        assert_ne!(data_addition, incorrect_array)
    }

    #[test]
    fn add_float() {
        let data_addition_mult = get_array_1d_float()
            + get_array_1d_float()
            + get_array_1d_float()
            + get_array_1d_float()
            + get_array_1d_float();
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();
        let data_addition = array1.clone() + array2;

        assert_eq!(
            data_addition,
            Array1D::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(
            array1.clone() + vec![1., 2., 3., 4., 5., 6., 7.],
            Array1D::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(array1 + 1., Array1D::new(vec![2., 3., 4., 5., 6., 7., 8.]));

        let expected_array = Array1D::new(vec![5., 10., 15., 20., 25., 30., 35.]);
        assert_eq!(data_addition_mult, expected_array,);

        let incorrect_array = Array1D::new(vec![2., 4., 6., 8., 10., 12., 14., 16.]);
        assert_ne!(data_addition, incorrect_array);

        let incorrect_array = Array1D::new(vec![2., 4., 6., 8., 10., 12., 123.]);
        assert_ne!(data_addition, incorrect_array)
    }

    #[test]
    fn sub_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() - array2,
            Array1D::new(vec![0., 0., 0., 0., 0., 0., 0.])
        );
        assert_eq!(
            array1.clone() - vec![1., 2., 3., 4., 5., 6., 7.],
            Array1D::new(vec![0., 0., 0., 0., 0., 0., 0.])
        );
        assert_eq!(
            array1.clone() - 2.,
            Array1D::new(vec![-1., 0., 1., 2., 3., 4., 5.])
        );
    }

    #[test]
    fn mul_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() * array2,
            Array1D::new(vec![1., 4., 9., 16., 25., 36., 49.])
        );
        assert_eq!(
            array1.clone() * vec![1., 2., 3., 4., 5., 6., 7.],
            Array1D::new(vec![1., 4., 9., 16., 25., 36., 49.])
        );
        assert_eq!(
            array1.clone() * 2.,
            Array1D::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(array1.clone() * 1., array1);
    }

    #[test]
    fn div_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() / array2,
            Array1D::new(vec![1., 1., 1., 1., 1., 1., 1.])
        );
        assert_eq!(
            array1.clone() / vec![1., 2., 3., 4., 5., 6., 7.],
            Array1D::new(vec![1., 1., 1., 1., 1., 1., 1.])
        );
        assert_eq!(
            array1.clone() / 2.,
            Array1D::new(vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5])
        );
        assert_eq!(array1.clone() / 1., array1);
    }

    #[test]
    fn sum_float() {
        let array1 = get_array_1d_float();

        assert_eq!(array1.sum(), 28.);
        assert_eq!(array1.seq_sum(), 28.);
        assert_eq!(array1.par_sum(), 28.);
    }

    #[test]
    fn mixed_operation_float() {
        let array1 = get_array_1d_float();
        let array = array1.clone()
            + ((array1.clone() * 2.) - array1.clone()) / ((array1.clone() + array1.clone()) / 2.);

        assert_eq!(array, Array1D::new(vec![2., 3., 4., 5., 6., 7., 8.]));
    }

    #[test]
    fn random() {
        let array1: Array1D<f64> = Array1D::random(3);
        let array2: Array1D<i64> = Array1D::random_range(3, 1, 10);

        assert_eq!(array1.shape, (3,));
        assert!(array1.data[0] >= 0. && array1.data[0] < 1.);
        assert!(array2.data[0] >= 1 && array2.data[0] <= 10);
    }

    #[test]
    fn arange() {
        let array1 = numrs::arange(10);
        assert_eq!(array1, Array1D::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
    }
}
