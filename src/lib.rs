use num_traits::Zero;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    cmp::min,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
};
use wasm_bindgen::prelude::wasm_bindgen;


/// 1D Array
///
///
/// Uses a Vec internally.
/// Takes ownership of the data
///
/// # Example
/// ```
/// use numrs::ArrayND;
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let array: ArrayND = ArrayND::new(data);
/// ``` 
#[wasm_bindgen]
#[derive(Clone)]
pub struct ArrayND {
    data: Vec<f64>,
    pub max: f64,
    pub min: f64,
    shape: Vec<usize>,
    size: usize,
}

#[wasm_bindgen]
impl ArrayND {
    /// Creates a new 1D Array
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let array: ArrayND = ArrayND::new(data);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f64>) -> ArrayND {
        let min: f64 = find_min::<f64>(&data);
        let max: f64 = find_max::<f64>(&data);

        ArrayND {
            shape: vec![data.len()],
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
    /// use numrs::ArrayND;
    /// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let array: ArrayND = ArrayND::new(data);
    /// assert_eq!(array.sum(), 6.0);
    /// ```
    pub fn sum(&self) -> f64 {
        if self.size > 1_000_000 {
            self.par_sum()
        } else {
            self.seq_sum()
        }
    }

    /// Seqential Sum used inside 1D Array
    pub fn seq_sum(&self) -> f64 {
        self.data.iter().fold(0., |sum, &val| sum + val)
    }

    #[cfg(target_family = "wasm")]
    /// Seqential Sum used inside 1D Array
    pub fn par_sum(&self) -> f64 {
        self.seq_sum()
    }

    #[cfg(target_family = "unix")]
    /// Parallel Sum used inside 1D Array
    pub fn par_sum(&self) -> f64 {
        self.data
            .par_iter()
            .fold(|| 0., |sum, &val| sum + val)
            .collect::<Vec<f64>>()
            .iter()
            .fold(0., |sum, &val| sum + val)
    }

    /// Generates a random 1D Array
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::random(10);
    /// ```
    pub fn random(size: usize) -> ArrayND {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen::<f64>()).collect();
        ArrayND::new(data)
    }

    /// Generates a random 1D Array with a range
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::random_range(10, 1., 10.);
    /// ```
    pub fn random_range(size: usize, min: f64, max: f64) -> ArrayND {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen_range(min..max)).collect();
        ArrayND::new(data)
    }

    pub fn add(mut self, num: f64) -> ArrayND {
        self.data.iter_mut().for_each(|x| *x += num);
        self
    }

    pub fn to_string(&self) -> String {
        let mut string = String::new();
        for i in 0..self.size {
            string.push_str(&format!("{} ", self.data[i]));
        }
        string
    }

    pub fn arange(start: f64, stop: f64, step: f64) -> ArrayND {
        let mut data: Vec<f64> = Vec::new();
        let mut i = start;
        while i < stop {
            data.push(i);
            i += step;
        }
        ArrayND::new(data)
    }
}

#[wasm_bindgen]
pub fn asarray(data: Vec<f64>) -> ArrayND {
    ArrayND::new(data)
}

fn find_min<T: Copy + Zero + std::cmp::PartialOrd>(data: &[f64]) -> f64 {
    data.iter()
        .reduce(|x, y| if x < y { x } else { y })
        .cloned()
        .unwrap()
}

fn find_max<T: Copy + Zero + std::cmp::PartialOrd>(data: &[f64]) -> f64 {
    data.iter()
        .reduce(|x, y| if x > y { x } else { y })
        .cloned()
        .unwrap()
}

impl Display for ArrayND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArrayND, min: {:?}, max: {:?}, data[..100] {:?}",
            self.min,
            self.max,
            &self.data[..100]
        )
    }
}

impl Add<ArrayND> for ArrayND {
    type Output = ArrayND;

    fn add(self, rhs: Self) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.shape[0]);
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs.data).collect();
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

        ArrayND {
            shape: vec![lhs.size],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl Sub<ArrayND> for ArrayND {
    type Output = ArrayND;

    fn sub(self, rhs: Self) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.shape[0]);
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs.data).collect();
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl Mul<ArrayND> for ArrayND {
    type Output = ArrayND;

    fn mul(self, rhs: Self) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.shape[0]);
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs.data).collect();
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl Div<ArrayND> for ArrayND {
    type Output = ArrayND;

    fn div(self, rhs: Self) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.shape[0]);
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs.data).collect();
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl Add<Vec<f64>> for ArrayND {
    type Output = ArrayND;

    fn add(self, rhs: Vec<f64>) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.len());
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<f64> = data.iter().map(|(i, j)| i.clone() + j.clone()).collect();
        let rhs_min = find_min::<f64>(&data);
        let rhs_max = find_min::<f64>(&data);
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl Sub<Vec<f64>> for ArrayND {
    type Output = ArrayND;

    fn sub(self, rhs: Vec<f64>) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.len());
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<f64> = data.iter().map(|(i, j)| i.clone() - j.clone()).collect();
        let rhs_min = find_min::<f64>(&data);
        let rhs_max = find_min::<f64>(&data);
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl Mul<Vec<f64>> for ArrayND {
    type Output = ArrayND;

    fn mul(self, rhs: Vec<f64>) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.len());
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<f64> = data.iter().map(|(i, j)| i.clone() * j.clone()).collect();
        let rhs_min = find_min::<f64>(&data);
        let rhs_max = find_min::<f64>(&data);
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}
impl Div<Vec<f64>> for ArrayND {
    type Output = ArrayND;

    fn div(self, rhs: Vec<f64>) -> ArrayND {
        let lhs = self;
        assert_eq!(lhs.shape[0], rhs.len());
        let data: Vec<(f64, f64)> = lhs.data.into_iter().zip(rhs).collect();
        let data: Vec<f64> = data.iter().map(|(i, j)| i.clone() / j.clone()).collect();
        let rhs_min = find_min::<f64>(&data);
        let rhs_max = find_min::<f64>(&data);
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

        ArrayND {
            shape: vec![lhs.shape[0]],
            size: lhs.size,
            data,
            min,
            max,
        }
    }
}

impl Add<f64> for ArrayND {
    type Output = ArrayND;

    fn add(mut self, rhs: f64) -> ArrayND {
        for el in self.data.iter_mut() {
            *el += rhs
        }
        self.min += rhs;
        self.max += rhs;
        self
    }
}

impl Sub<f64> for ArrayND {
    type Output = ArrayND;

    fn sub(mut self, rhs: f64) -> ArrayND {
        for el in self.data.iter_mut() {
            *el -= rhs
        }
        self.min -= rhs;
        self.max -= rhs;
        self
    }
}
impl Mul<f64> for ArrayND {
    type Output = ArrayND;

    fn mul(mut self, rhs: f64) -> ArrayND {
        for el in self.data.iter_mut() {
            *el *= rhs
        }
        self.min *= rhs;
        self.max *= rhs;
        self
    }
}

impl Div<f64> for ArrayND {
    type Output = ArrayND;

    fn div(mut self, rhs: f64) -> ArrayND {
        for el in self.data.iter_mut() {
            *el /= rhs
        }
        self.min /= rhs;
        self.max /= rhs;
        self
    }
}

impl Debug for ArrayND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shape[0] > 100 {
            let print_limit = min(self.shape[0], 100);
            write!(
                f,
                "ArrayND {:?}, min: {:?}, max: {:?}, data[..{:?}] {:?}...",
                self.shape,
                self.min,
                self.max,
                print_limit,
                &self.data[..print_limit]
            )
        } else {
            write!(
                f,
                "ArrayND {:?}, min: {:?}, max: {:?}, data: {:?}",
                self.shape,
                self.min,
                self.max,
                &self.data[..self.shape[0]]
            )
        }
    }
}

impl PartialEq<ArrayND> for ArrayND {
    fn eq(&self, other: &ArrayND) -> bool {
        self.data == other.data
    }

    fn ne(&self, other: &ArrayND) -> bool {
        self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use crate::ArrayND;

    // fn get_array_1d_integer() -> ArrayND {
    //     ArrayND::new(vec![1, 2, 3, 4, 5, 6, 7])
    // }

    fn get_array_1d_float() -> ArrayND {
        ArrayND::new(vec![1., 2., 3., 4., 5., 6., 7.])
    }

    #[test]
    // fn add_integer() {
    //     let data_addition_mult = get_array_1d_integer()
    //         + get_array_1d_integer()
    //         + get_array_1d_integer()
    //         + get_array_1d_integer()
    //         + get_array_1d_integer();
    //     let array1 = get_array_1d_integer();
    //     let array2 = get_array_1d_integer();
    //     let data_addition = array1.clone() + array2;

    //     assert_eq!(data_addition, ArrayND::new(vec![2, 4, 6, 8, 10, 12, 14]));
    //     assert_eq!(
    //         array1.clone() + vec![1, 2, 3, 4, 5, 6, 7],
    //         ArrayND::new(vec![2, 4, 6, 8, 10, 12, 14])
    //     );
    //     assert_eq!(array1 + 1, ArrayND::new(vec![2, 3, 4, 5, 6, 7, 8]));

    //     let expected_array = ArrayND::new(vec![5, 10, 15, 20, 25, 30, 35]);
    //     assert_eq!(data_addition_mult, expected_array);

    //     let incorrect_array: ArrayND<i32> = ArrayND::new(vec![2, 4, 6, 8, 10, 12, 14, 16]);
    //     assert_ne!(data_addition, incorrect_array);

    //     let incorrect_array = ArrayND::new(vec![2, 4, 6, 8, 10, 12, 123]);
    //     assert_ne!(data_addition, incorrect_array)
    // }
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
            ArrayND::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(
            array1.clone() + vec![1., 2., 3., 4., 5., 6., 7.],
            ArrayND::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(array1 + 1., ArrayND::new(vec![2., 3., 4., 5., 6., 7., 8.]));

        let expected_array = ArrayND::new(vec![5., 10., 15., 20., 25., 30., 35.]);
        assert_eq!(data_addition_mult, expected_array,);

        let incorrect_array = ArrayND::new(vec![2., 4., 6., 8., 10., 12., 14., 16.]);
        assert_ne!(data_addition, incorrect_array);

        let incorrect_array = ArrayND::new(vec![2., 4., 6., 8., 10., 12., 123.]);
        assert_ne!(data_addition, incorrect_array)
    }

    #[test]
    fn sub_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() - array2,
            ArrayND::new(vec![0., 0., 0., 0., 0., 0., 0.])
        );
        assert_eq!(
            array1.clone() - vec![1., 2., 3., 4., 5., 6., 7.],
            ArrayND::new(vec![0., 0., 0., 0., 0., 0., 0.])
        );
        assert_eq!(
            array1.clone() - 2.,
            ArrayND::new(vec![-1., 0., 1., 2., 3., 4., 5.])
        );
    }

    #[test]
    fn mul_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() * array2,
            ArrayND::new(vec![1., 4., 9., 16., 25., 36., 49.])
        );
        assert_eq!(
            array1.clone() * vec![1., 2., 3., 4., 5., 6., 7.],
            ArrayND::new(vec![1., 4., 9., 16., 25., 36., 49.])
        );
        assert_eq!(
            array1.clone() * 2.,
            ArrayND::new(vec![2., 4., 6., 8., 10., 12., 14.])
        );
        assert_eq!(array1.clone() * 1., array1);
    }

    #[test]
    fn div_float() {
        let array1 = get_array_1d_float();
        let array2 = get_array_1d_float();

        assert_eq!(
            array1.clone() / array2,
            ArrayND::new(vec![1., 1., 1., 1., 1., 1., 1.])
        );
        assert_eq!(
            array1.clone() / vec![1., 2., 3., 4., 5., 6., 7.],
            ArrayND::new(vec![1., 1., 1., 1., 1., 1., 1.])
        );
        assert_eq!(
            array1.clone() / 2.,
            ArrayND::new(vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5])
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

        assert_eq!(array, ArrayND::new(vec![2., 3., 4., 5., 6., 7., 8.]));
    }

    #[test]
    fn random() {
        let array1: ArrayND = ArrayND::random(3);
        // let array2: ArrayND<i64> = ArrayND::random_range(3, 1., 10.);

        assert_eq!(array1.shape, vec![3]);
        assert!(array1.data[0] >= 0. && array1.data[0] < 1.);
        // assert!(array2.data[0] >= 1 && array2.data[0] <= 10);
    }

    // #[test]
    // fn arange_test() {
    //     let array1 = arange(10);
    //     let correct_array: ArrayND<i64> = ArrayND::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    //     assert_eq!(array1, correct_array);
    // }
}
