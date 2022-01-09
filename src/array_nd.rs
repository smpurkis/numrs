use num_traits::Zero;
use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone)]
pub struct Array1D<T: Copy> {
    pub data: Vec<T>,
    shape: (usize,),
    min: T,
    max: T,
}

// constructor for 1d array
impl<T> Array1D<T>
where
    T: Copy + Zero + std::cmp::PartialOrd,
{
    pub fn new(data: Vec<T>) -> Array1D<T> {
        let min = find_min(&data);
        let max = find_max(&data);
        Array1D {
            shape: (data.len() as usize,),
            data,
            min,
            max,
        }
    }

    pub fn sum(&self) -> T {
        self.data.iter().fold(T::zero(), |sum, &val| sum + val)
    }
}

fn find_min<T: Copy + Zero + std::cmp::PartialOrd>(data: &[T]) -> T {
    data.iter()
        .reduce(|x, y| if x < y { x } else { y })
        .cloned()
        .unwrap()
}

fn find_max<T: Copy + Zero + std::cmp::PartialOrd>(data: &[T]) -> T {
    data.iter()
        .reduce(|x, y| if x < y { x } else { y })
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
            write!(
                f,
                "Array2D {:?}, min: {:?}, max: {:?}, data[..{:?}] {:?}",
                self.shape,
                self.min,
                self.max,
                self.shape.0,
                &self.data[..self.shape.0]
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
    use crate::array_nd::Array1D;

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
        let array1_sum = array1.sum();

        assert_eq!(array1_sum, 28.);
    }

    #[test]
    fn mixed_operation_float() {
        let array1 = get_array_1d_float();
        let array =
           array1.clone() + ((array1.clone() * 2.) - array1.clone()) / ((array1.clone() + array1.clone()) / 2.);

        assert_eq!(array, Array1D::new(vec![2., 3., 4., 5., 6., 7., 8.]));
    }
}
