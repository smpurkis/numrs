use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
use std::{
    cmp::min,
    fmt::{Debug, Display},
    ops::{Index, Range, RangeFrom, RangeTo},
};

// constants
const PI: f64 = std::f64::consts::PI;
const TAU: f64 = 2.0 * PI;
const E: f64 = std::f64::consts::E;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1D Array
///
///
/// Uses a Vec internally.
/// Takes ownership of the data
///
/// # Example
/// ```
/// use numrs::{ArrayND,ArrayData, asarray};
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let array: ArrayND = asarray(ArrayData::OneD(data));
/// ```

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArrayData {
    OneD(Vec<f64>),
    TwoD(Vec<Vec<f64>>),
    ThreeD(Vec<Vec<Vec<f64>>>),
    FourD(Vec<Vec<Vec<Vec<f64>>>>),
}

impl Index<usize> for ArrayData {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Self::OneD(arg0) => &arg0[index],
            Self::TwoD(arg0) => &arg0[index][0],
            Self::ThreeD(arg0) => &arg0[index][0][0],
            Self::FourD(arg0) => &arg0[index][0][0][0],
        }
    }
}

impl Index<Range<usize>> for ArrayData {
    type Output = [f64];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        match self {
            Self::OneD(arg0) => &arg0[index],
            Self::TwoD(arg0) => &arg0[index][0],
            Self::ThreeD(arg0) => &arg0[index][0][0],
            Self::FourD(arg0) => &arg0[index][0][0][0],
        }
    }
}

impl Index<RangeTo<usize>> for ArrayData {
    type Output = [f64];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        match self {
            Self::OneD(arg0) => &arg0[index],
            Self::TwoD(arg0) => &arg0[index][0],
            Self::ThreeD(arg0) => &arg0[index][0][0],
            Self::FourD(arg0) => &arg0[index][0][0][0],
        }
    }
}

impl Index<RangeFrom<usize>> for ArrayData {
    type Output = [f64];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        match self {
            Self::OneD(arg0) => &arg0[index],
            Self::TwoD(arg0) => &arg0[index][0],
            Self::ThreeD(arg0) => &arg0[index][0][0],
            Self::FourD(arg0) => &arg0[index][0][0][0],
        }
    }
}

impl Iterator for ArrayData {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::OneD(arg0) => arg0.pop(),
            Self::TwoD(arg0) => arg0.pop().map(|arg1| arg1[0]),
            Self::ThreeD(arg0) => arg0.pop().map(|arg1| arg1[0][0]),
            Self::FourD(arg0) => arg0.pop().map(|arg1| arg1[0][0][0]),
        }
    }
}

impl ArrayData {
    fn iter(&self) -> ArrayData {
        match self {
            Self::OneD(arg0) => ArrayData::OneD(arg0.iter().cloned().collect()),
            Self::TwoD(arg0) => ArrayData::TwoD(
                arg0.iter()
                    .map(|arg1| arg1.iter().cloned().collect())
                    .collect(),
            ),
            Self::ThreeD(arg0) => ArrayData::ThreeD(
                arg0.iter()
                    .map(|arg1| {
                        arg1.iter()
                            .map(|arg2| arg2.iter().cloned().collect())
                            .collect()
                    })
                    .collect(),
            ),
            Self::FourD(arg0) => ArrayData::FourD(
                arg0.iter()
                    .map(|arg1| {
                        arg1.iter()
                            .map(|arg2| {
                                arg2.iter()
                                    .map(|arg3| arg3.iter().cloned().collect())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArrayND {
    data: ArrayData,
    pub max: f64,
    pub min: f64,
    shape: Vec<usize>,
    pub size: usize,
}

impl ArrayND {
    pub fn new(data: ArrayData) -> Self {
        match data {
            ArrayData::OneD(data) => {
                let size = data.len();
                let shape = vec![size];
                let array_data = ArrayData::OneD(data);
                let min: f64 = find_min(&array_data);
                let max: f64 = find_max(&array_data);

                ArrayND {
                    shape,
                    size,
                    data: array_data,
                    min,
                    max,
                }
            }
            ArrayData::TwoD(data) => {
                let shape = vec![data.len(), data[0].len()];
                let size = data.len() * data[0].len();
                let array_data = ArrayData::TwoD(data);
                let min: f64 = find_min(&array_data);
                let max: f64 = find_max(&array_data);

                ArrayND {
                    shape,
                    size,
                    data: array_data,
                    min,
                    max,
                }
            }
            ArrayData::ThreeD(data) => {
                let shape = vec![data.len(), data[0].len(), data[0][0].len()];
                let size = data.len() * data[0].len() * data[0][0].len();
                let array_data = ArrayData::ThreeD(data);
                let min: f64 = find_min(&array_data);
                let max: f64 = find_max(&array_data);

                ArrayND {
                    shape,
                    size,
                    data: array_data,
                    min,
                    max,
                }
            }
            ArrayData::FourD(data) => {
                let shape = vec![
                    data.len(),
                    data[0].len(),
                    data[0][0].len(),
                    data[0][0][0].len(),
                ];
                let size = data.len() * data[0].len() * data[0][0].len() * data[0][0][0].len();
                let array_data = ArrayData::FourD(data);
                let min: f64 = find_min(&array_data);
                let max: f64 = find_max(&array_data);

                ArrayND {
                    shape,
                    size,
                    data: array_data,
                    min,
                    max,
                }
            }
        }
    }

    fn _new(data: ArrayData) -> ArrayND {
        match data {
            ArrayData::OneD(arg0) => ArrayND::new1d(arg0),
            ArrayData::TwoD(arg0) => ArrayND::new2d(arg0),
            ArrayData::ThreeD(arg0) => ArrayND::new3d(arg0),
            ArrayData::FourD(arg0) => ArrayND::new4d(arg0),
        }
    }

    fn new1d(data: Vec<f64>) -> ArrayND {
        let size = data.len();
        let shape = vec![size];
        let array_data = ArrayData::OneD(data);
        let min: f64 = find_min(&array_data);
        let max: f64 = find_max(&array_data);

        ArrayND {
            shape,
            size,
            data: array_data,
            min,
            max,
        }
    }

    fn new2d(data: Vec<Vec<f64>>) -> ArrayND {
        let shape = vec![data.len(), data[0].len()];
        let size = data.len() * data[0].len();
        let array_data = ArrayData::TwoD(data);
        let min: f64 = find_min(&array_data);
        let max: f64 = find_max(&array_data);

        ArrayND {
            shape,
            size,
            data: array_data,
            min,
            max,
        }
    }

    fn new3d(data: Vec<Vec<Vec<f64>>>) -> ArrayND {
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];
        let size = data.len() * data[0].len() * data[0][0].len();
        let array_data = ArrayData::ThreeD(data);
        let min: f64 = find_min(&array_data);
        let max: f64 = find_max(&array_data);

        ArrayND {
            shape,
            size,
            data: array_data,
            min,
            max,
        }
    }

    fn new4d(data: Vec<Vec<Vec<Vec<f64>>>>) -> ArrayND {
        let shape = vec![
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
        ];
        let size = data.len() * data[0].len() * data[0][0].len() * data[0][0][0].len();
        let array_data = ArrayData::FourD(data);
        let min: f64 = find_min(&array_data);
        let max: f64 = find_max(&array_data);

        ArrayND {
            shape,
            size,
            data: array_data,
            min,
            max,
        }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn reshape(&mut self, shape: Vec<usize>) {
        let new_size: usize = shape.iter().product();
        if self.size != new_size {
            panic!("Cannot reshape array to new shape with different size");
        }
        match shape.len() {
            1 => {
                let flattened_data = &self.flatten();
                let mut new_data = vec![];
                for i in 0..shape[0] {
                    new_data.push(flattened_data[i]);
                }
                self.data = ArrayData::OneD(new_data);
            }
            2 => {
                let flattened_data = &self.flatten();
                let mut new_data = vec![];
                for i in 0..shape[0] {
                    let mut row = vec![];
                    for j in 0..shape[1] {
                        row.push(flattened_data[i * shape[1] + j]);
                    }
                    new_data.push(row);
                }
                self.data = ArrayData::TwoD(new_data);
            }
            3 => {
                let flattened_data = &self.flatten();
                let mut new_data = vec![];
                for i in 0..shape[0] {
                    let mut row = vec![];
                    for j in 0..shape[1] {
                        let mut col = vec![];
                        for k in 0..shape[2] {
                            col.push(flattened_data[i * shape[1] * shape[2] + j * shape[2] + k]);
                        }
                        row.push(col);
                    }
                    new_data.push(row);
                }
                self.data = ArrayData::ThreeD(new_data);
            }
            4 => {
                let flattened_data = &self.flatten();
                let mut new_data = vec![];
                for i in 0..shape[0] {
                    let mut row = vec![];
                    for j in 0..shape[1] {
                        let mut col = vec![];
                        for k in 0..shape[2] {
                            let mut layer = vec![];
                            for l in 0..shape[3] {
                                layer.push(
                                    flattened_data[i * shape[1] * shape[2] * shape[3]
                                        + j * shape[2] * shape[3]
                                        + k * shape[3]
                                        + l],
                                );
                            }
                            col.push(layer);
                        }
                        row.push(col);
                    }
                    new_data.push(row);
                }
                self.data = ArrayData::FourD(new_data);
            }
            _ => panic!("Cannot reshape array to new shape with more than 4 dimensions"),
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
    /// use numrs::{ArrayND,ArrayData, asarray};
    /// let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let array: ArrayND = asarray(ArrayData::OneD(data));
    /// assert_eq!(array.sum(), 6.0);
    /// ```
    pub fn sum(&self) -> f64 {
        if self.size > 1_000_000 {
            self.seq_sum()
        } else {
            self.seq_sum()
        }
    }

    /// Seqential Sum used inside 1D Array
    pub fn seq_sum(&self) -> f64 {
        match &self.data {
            ArrayData::OneD(arg0) => arg0.iter().sum(),
            ArrayData::TwoD(arg0) => {
                let mut total = 0.;
                for arg1 in arg0.iter() {
                    total += arg1.iter().sum::<f64>();
                }
                total
            }
            ArrayData::ThreeD(arg0) => {
                let mut total = 0.;
                for arg1 in arg0.iter() {
                    for arg2 in arg1.iter() {
                        total += arg2.iter().sum::<f64>();
                    }
                }
                total
            }
            ArrayData::FourD(arg0) => {
                let mut total = 0.;
                for arg1 in arg0.iter() {
                    for arg2 in arg1.iter() {
                        for arg3 in arg2.iter() {
                            total += arg3.iter().sum::<f64>();
                        }
                    }
                }
                total
            }
        }
    }

    // #[cfg(target_family = "wasm")]
    // /// Seqential Sum used inside 1D Array
    // pub fn par_sum(&self) -> f64 {
    //     self.seq_sum()
    // }

    // #[cfg(target_family = "unix")]
    // /// Parallel Sum used inside 1D Array
    // pub fn par_sum(&self) -> f64 {
    //     match &self.data {
    //         ArrayData::OneD(arg0) => arg0.par_iter().sum(),
    //         ArrayData::TwoD(arg0) => {
    //             arg0.par_iter().map(|arg1| arg1.sum()).sum()
    //         }
    //         ArrayData::ThreeD(arg0) => {
    //             let mut total = 0.;
    //             for arg1 in arg0.par_iter() {
    //                 for arg2 in arg1.par_iter() {
    //                     total += arg2.par_iter().sum::<f64>();
    //                 }
    //             }
    //             total
    //         }
    //         ArrayData::FourD(arg0) => {
    //             let mut total = 0.;
    //             for arg1 in arg0.par_iter() {
    //                 for arg2 in arg1.par_iter() {
    //                     for arg3 in arg2.par_iter() {
    //                         total += arg3.par_iter().sum::<f64>();
    //                     }
    //                 }
    //             }
    //             total
    //         }
    //     }
    // }

    /// Generates a random 1D Array
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::random(vec![10]);
    /// let array2d: ArrayND = ArrayND::random(vec![10, 10]);
    /// assert_eq!(array.size, 10);
    /// assert_eq!(array2d.size, 100);
    /// ```
    pub fn random(shape: Vec<usize>) -> ArrayND {
        generate(|| rand::random::<f64>(), shape)
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
        ArrayND::new1d(data)
    }

    /// Convert an array to a nice matrix string output
    ///
    /// # Example
    /// ```
    /// use numrs::{asarray,ArrayData,ArrayND};
    /// let array: ArrayND = asarray(ArrayData::OneD(vec![1.0, 2.0, 3.0]));
    /// let array_string: String = array.to_string();
    /// assert_eq!(array_string, "1 2 3 ");
    /// ```
    pub fn to_string(&self) -> String {
        let mut string = String::new();
        match &self.data {
            ArrayData::OneD(arg0) => {
                for arg1 in arg0.iter() {
                    string.push_str(&arg1.to_string());
                    string.push_str(" ");
                }
            }
            ArrayData::TwoD(arg0) => {
                for arg1 in arg0.iter() {
                    for arg2 in arg1.iter() {
                        string.push_str(&arg2.to_string());
                        string.push_str(" ");
                    }
                    string.push_str("\n");
                }
            }
            ArrayData::ThreeD(arg0) => {
                for arg1 in arg0.iter() {
                    for arg2 in arg1.iter() {
                        for arg3 in arg2.iter() {
                            string.push_str(&arg3.to_string());
                            string.push_str(" ");
                        }
                        string.push_str("\n");
                    }
                    string.push_str("\n");
                }
            }
            ArrayData::FourD(arg0) => {
                for arg1 in arg0.iter() {
                    for arg2 in arg1.iter() {
                        for arg3 in arg2.iter() {
                            for arg4 in arg3.iter() {
                                string.push_str(&arg4.to_string());
                                string.push_str(" ");
                            }
                            string.push_str("\n");
                        }
                        string.push_str("\n");
                    }
                    string.push_str("\n");
                }
            }
        }
        string
    }

    /// Flatten ndarray to 1d array
    ///
    /// # Example
    /// ```
    /// use numrs::{asarray,ArrayData,ArrayND};
    /// let array: ArrayND = asarray(ArrayData::TwoD(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]));
    /// let array_1d: Vec<f64> = array.flatten();
    /// assert_eq!(array_1d, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn flatten(&self) -> Vec<f64> {
        match &self.data {
            ArrayData::OneD(arg0) => arg0.clone(),
            ArrayData::TwoD(arg0) => arg0.iter().flatten().cloned().collect(),
            ArrayData::ThreeD(arg0) => arg0.iter().flatten().flatten().cloned().collect(),
            ArrayData::FourD(arg0) => arg0.iter().flatten().flatten().flatten().cloned().collect(),
        }
    }

    /// Generates a random 1D Array with a range
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::arange(0., 10., 1.);
    /// assert_eq!(array.flatten(), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    /// ```
    pub fn arange(start: f64, stop: f64, step: f64) -> ArrayND {
        let mut data: Vec<f64> = Vec::new();
        let mut i = start;
        while i < stop {
            data.push(i);
            i += step;
        }
        ArrayND::new1d(data)
    }

    /// Generates a array of specified shape filled with value
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::ones(vec![2, 2]);
    /// assert_eq!(array.flatten(), vec![1., 1., 1., 1.]);
    /// let array: ArrayND = ArrayND::fill(29., vec![3, 3]);
    /// assert_eq!(array.flatten(), vec![29., 29., 29., 29., 29., 29., 29., 29., 29.]);
    /// ```
    pub fn fill(value: f64, shape: Vec<usize>) -> ArrayND {
        generate(|| value, shape)
    }

    pub fn zeros(shape: Vec<usize>) -> ArrayND {
        ArrayND::fill(0., shape)
    }

    pub fn ones(shape: Vec<usize>) -> ArrayND {
        ArrayND::fill(1., shape)
    }

    fn apply_elementwise_with_other_array<F>(&self, other: ArrayND, mut func: F) -> ArrayND
    where
        F: FnMut(f64, f64) -> f64,
    {
        if self.shape != other.shape {
            panic!("ArrayND::apply_elementwise_with_other_array() requires both arrays to have the same shape");
        }
        let self_array = self.clone();
        let other_array = other.clone();

        let stuff = (self_array, other_array);
        match stuff {
                (ArrayND { data: ArrayData::OneD(arg0), .. }, ArrayND { data: ArrayData::OneD(arg1), .. }) => {
                    let mut data: Vec<f64> = Vec::new();
                    for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                        data.push(func(*arg0, *arg1));
                    }
                    ArrayND::new1d(data)
                }
                (ArrayND { data: ArrayData::TwoD(arg0), .. }, ArrayND { data: ArrayData::TwoD(arg1), .. }) => {
                    let mut data: Vec<Vec<f64>> = Vec::new();
                    for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                        let mut row: Vec<f64> = Vec::new();
                        for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                            row.push(func(*arg0, *arg1));
                        }
                        data.push(row);
                    }
                    ArrayND::new2d(data)
                }
                (ArrayND { data: ArrayData::ThreeD(arg0), .. }, ArrayND { data: ArrayData::ThreeD(arg1), .. }) => {
                    let mut data: Vec<Vec<Vec<f64>>> = Vec::new();
                    for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                        let mut second_array: Vec<Vec<f64>> = Vec::new();
                        for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                            let mut row: Vec<f64> = Vec::new();
                            for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                                row.push(func(*arg0, *arg1));
                            }
                            second_array.push(row);
                        }
                        data.push(second_array);
                    }
                    ArrayND::new3d(data)
                },
                (ArrayND { data: ArrayData::FourD(arg0), .. }, ArrayND { data: ArrayData::FourD(arg1), .. }) => {
                    let mut data: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
                    for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                        let mut third_array: Vec<Vec<Vec<f64>>> = Vec::new();
                        for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                            let mut second_array: Vec<Vec<f64>> = Vec::new();
                            for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                                let mut row: Vec<f64> = Vec::new();
                                for (arg0, arg1) in arg0.iter().zip(arg1.iter()) {
                                    row.push(func(*arg0, *arg1));
                                }
                                second_array.push(row);
                            }
                            third_array.push(second_array);
                        }
                        data.push(third_array);
                    }
                    ArrayND::new4d(data)
                },
                _ => panic!("ArrayND::apply_elementwise_with_other_array() only supports 1D, 2D, 3D, and 4D arrays"),
            }
    }

    fn apply_inplace<F>(self, mut func: F) -> ArrayND
    where
        F: FnMut(f64) -> f64,
    {
        match self.data {
            ArrayData::OneD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    *arg1 = func(*arg1);
                }
                ArrayND::new1d(arg0)
            }
            ArrayData::TwoD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    for arg2 in arg1.iter_mut() {
                        *arg2 = func(*arg2);
                    }
                }
                ArrayND::new2d(arg0)
            }
            ArrayData::ThreeD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    for arg2 in arg1.iter_mut() {
                        for arg3 in arg2.iter_mut() {
                            *arg3 = func(*arg3);
                        }
                    }
                }
                ArrayND::new3d(arg0)
            }
            ArrayData::FourD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    for arg2 in arg1.iter_mut() {
                        for arg3 in arg2.iter_mut() {
                            for arg4 in arg3.iter_mut() {
                                *arg4 = func(*arg4);
                            }
                        }
                    }
                }
                ArrayND::new4d(arg0)
            }
        }
    }

    fn apply_clone<F>(&self, mut func: F) -> ArrayND
    where
        F: FnMut(f64) -> f64,
    {
        let new_array = self.clone();
        match new_array.data {
            ArrayData::OneD(arg0) => {
                let mut data: Vec<f64> = Vec::new();
                for arg1 in arg0.iter() {
                    data.push(func(*arg1));
                }
                ArrayND::new1d(data)
            }
            ArrayData::TwoD(arg0) => {
                let mut data: Vec<Vec<f64>> = Vec::new();
                for arg1 in arg0.iter() {
                    let mut row = Vec::new();
                    for arg2 in arg1.iter() {
                        row.push(func(*arg2));
                    }
                    data.push(row);
                }
                ArrayND::new2d(data)
            }
            ArrayData::ThreeD(arg0) => {
                let mut data: Vec<Vec<Vec<f64>>> = Vec::new();
                for arg1 in arg0.iter() {
                    let mut row = Vec::new();
                    for arg2 in arg1.iter() {
                        let mut col = Vec::new();
                        for arg3 in arg2.iter() {
                            col.push(func(*arg3));
                        }
                        row.push(col);
                    }
                    data.push(row);
                }
                ArrayND::new3d(data)
            }
            ArrayData::FourD(arg0) => {
                let mut data: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
                for arg1 in arg0.iter() {
                    let mut row = Vec::new();
                    for arg2 in arg1.iter() {
                        let mut col = Vec::new();
                        for arg3 in arg2.iter() {
                            let mut layer = Vec::new();
                            for arg4 in arg3.iter() {
                                layer.push(func(*arg4));
                            }
                            col.push(layer);
                        }
                        row.push(col);
                    }
                    data.push(row);
                }
                ArrayND::new4d(data)
            }
        }
    }

    pub fn add(&self, num: f64) -> ArrayND {
        self.apply_clone(|x| x + num)
    }

    // pub fn add(&self, other: ArrayND) {
    //     self.apply_elementwise_with_other_array(other, |a, b| a + b);
    // }

    pub fn sub(&self, num: f64) -> ArrayND {
        self.apply_clone(|x| x - num)
    }

    pub fn mul(&self, num: f64) -> ArrayND {
        self.apply_clone(|x| x * num)
    }

    pub fn div(&self, num: f64) -> ArrayND {
        self.apply_clone(|x| x / num)
    }

    pub fn cos(&self) -> ArrayND {
        self.apply_clone(|x| x.cos())
    }

    pub fn sin(&self) -> ArrayND {
        self.apply_clone(|x| x.sin())
    }

    pub fn tan(&self) -> ArrayND {
        self.apply_clone(|x| x.tan())
    }

    pub fn acos(&self) -> ArrayND {
        self.apply_clone(|x| x.acos())
    }

    pub fn asin(&self) -> ArrayND {
        self.apply_clone(|x| x.asin())
    }

    pub fn atan(&self) -> ArrayND {
        self.apply_clone(|x| x.atan())
    }

    pub fn cosh(&self) -> ArrayND {
        self.apply_clone(|x| x.cosh())
    }

    pub fn sinh(&self) -> ArrayND {
        self.apply_clone(|x| x.sinh())
    }

    pub fn tanh(&self) -> ArrayND {
        self.apply_clone(|x| x.tanh())
    }

    pub fn exp(&self) -> ArrayND {
        self.apply_clone(|x| x.exp())
    }

    pub fn log(&self, base: f64) -> ArrayND {
        self.apply_clone(|x| x.log(base))
    }

    pub fn log2(&self) -> ArrayND {
        self.apply_clone(|x| x.log(2.0))
    }

    pub fn log10(&self) -> ArrayND {
        self.apply_clone(|x| x.log(10.0))
    }

    pub fn loge(&self) -> ArrayND {
        self.apply_clone(|x| x.log(E))
    }

    pub fn logpi(&self) -> ArrayND {
        self.apply_clone(|x| x.log(PI))
    }

    pub fn abs(&self) -> ArrayND {
        self.apply_clone(|x| x.abs())
    }

    pub fn sqrt(&self) -> ArrayND {
        self.apply_clone(|x| x.sqrt())
    }

    pub fn cbrt(&self) -> ArrayND {
        self.apply_clone(|x| x.cbrt())
    }

    pub fn pow(&self, exponent: f64) -> ArrayND {
        self.apply_clone(|x| x.powf(exponent))
    }

    pub fn sq(&self) -> ArrayND {
        self.apply_clone(|x| x.powi(2))
    }

    pub fn cube(&self) -> ArrayND {
        self.apply_clone(|x| x.powi(3))
    }

    pub fn is_finite(&self) -> ArrayND {
        self.apply_clone(|x| (x.is_finite() as i32) as f64)
    }

    pub fn is_infinite(&self) -> ArrayND {
        self.apply_clone(|x| (x.is_infinite() as i32) as f64)
    }

    pub fn is_nan(&self) -> ArrayND {
        self.apply_clone(|x| (x.is_nan() as i32) as f64)
    }

    pub fn clamp(&self, min: f64, max: f64) -> ArrayND {
        self.apply_clone(|x| x.clamp(min, max))
    }

    pub fn floor(&self) -> ArrayND {
        self.apply_clone(|x| x.floor())
    }

    pub fn ceil(&self) -> ArrayND {
        self.apply_clone(|x| x.ceil())
    }

    pub fn round(&self) -> ArrayND {
        self.apply_clone(|x| x.round())
    }

    pub fn trunc(&self) -> ArrayND {
        self.apply_clone(|x| x.trunc())
    }

    pub fn signum(&self) -> ArrayND {
        self.apply_clone(|x| x.signum())
    }

    pub fn greater(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x > value) as i32 as f64)
    }

    pub fn greater_equal(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x >= value) as i32 as f64)
    }

    pub fn less(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x < value) as i32 as f64)
    }

    pub fn less_equal(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x <= value) as i32 as f64)
    }

    pub fn equal(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x == value) as i32 as f64)
    }

    pub fn not_equal(&self, value: f64) -> ArrayND {
        self.apply_clone(|x| (x != value) as i32 as f64)
    }

    pub fn and(&self, other: ArrayND) -> ArrayND {
        self.apply_elementwise_with_other_array(other, |a, b| (a > 0.0 && b > 0.0) as i32 as f64)
    }

    pub fn or(&self, other: ArrayND) -> ArrayND {
        self.apply_elementwise_with_other_array(other, |a, b| (a > 0.0 || b > 0.0) as i32 as f64)
    }

    pub fn xor(&self, other: ArrayND) -> ArrayND {
        self.apply_elementwise_with_other_array(other, |a, b| ((a > 0.0) != (b > 0.0)) as i32 as f64)
    }

    pub fn dot(&self, other: ArrayND) -> ArrayND {
        self.apply_elementwise_with_other_array(other, |a, b| a * b)
    }

    pub fn cross(&self, other: ArrayND) -> ArrayND {
        self.apply_elementwise_with_other_array(other, |a, b| a * b)
    }

    fn is_zero(&self) -> ArrayND {
        self.apply_clone(|x| (x == 0.0) as i32 as f64)
    }

    fn is_one(&self) -> ArrayND {
        self.apply_clone(|x| (x == 1.0) as i32 as f64)
    }

    
}

pub fn asarray(data: ArrayData) -> ArrayND {
    match data {
        ArrayData::OneD(arg0) => ArrayND::new1d(arg0),
        ArrayData::TwoD(arg0) => ArrayND::new2d(arg0),
        ArrayData::ThreeD(arg0) => ArrayND::new3d(arg0),
        ArrayData::FourD(arg0) => ArrayND::new4d(arg0),
    }
}

fn generate<F>(mut func: F, shape: Vec<usize>) -> ArrayND
where
    F: FnMut() -> f64,
{
    // let mut rng = rand::thread_rng().gen::<f64>();
    match shape.len() {
        1 => {
            let mut data: Vec<f64> = vec![];
            for _ in 0..shape[0] {
                data.push(func());
            }
            ArrayND::new1d(data)
        }
        2 => {
            let mut data: Vec<Vec<f64>> = vec![];
            for _ in 0..shape[0] {
                let mut row: Vec<f64> = vec![];
                for _ in 0..shape[1] {
                    row.push(func());
                }
                data.push(row);
            }
            ArrayND::new2d(data)
        }
        3 => {
            let mut data: Vec<Vec<Vec<f64>>> = vec![];
            for _ in 0..shape[0] {
                let mut second_array: Vec<Vec<f64>> = vec![];
                for _ in 0..shape[1] {
                    let mut row: Vec<f64> = vec![];
                    for _ in 0..shape[2] {
                        row.push(func());
                    }
                    second_array.push(row);
                }
                data.push(second_array);
            }
            ArrayND::new3d(data)
        }
        4 => {
            let mut data: Vec<Vec<Vec<Vec<f64>>>> = vec![];
            for _ in 0..shape[0] {
                let mut third_array: Vec<Vec<Vec<f64>>> = vec![];
                for _ in 0..shape[1] {
                    let mut second_array: Vec<Vec<f64>> = vec![];
                    for _ in 0..shape[2] {
                        let mut row: Vec<f64> = vec![];
                        for _ in 0..shape[3] {
                            row.push(func());
                        }
                        second_array.push(row);
                    }
                    third_array.push(second_array);
                }
                data.push(third_array);
            }
            ArrayND::new4d(data)
        }
        _ => panic!("ArrayND::generate() only supports 1D, 2D, 3D, and 4D arrays"),
    }
}

// #[wasm_bindgen]
// pub fn asarray_all(data: &JsValue) -> ArrayND {
//     unsafe {
//         console::log_1(&"asarray_all 0".into());
//     }
//     let data: ArrayData = data.into_serde().unwrap();
//     unsafe {
//         console::log_1(&"asarray_all 1".into());
//     }
//     match data {
//         ArrayData::OneD(arg0) => ArrayND::new1d(arg0),
//         ArrayData::TwoD(arg0) => ArrayND::new2d(arg0),
//         ArrayData::ThreeD(arg0) => ArrayND::new3d(arg0),
//         ArrayData::FourD(arg0) => ArrayND::new4d(arg0),
//     }
// }

fn find_min(data: &ArrayData) -> f64 {
    match data {
        ArrayData::OneD(arg0) => {
            *(arg0
                .iter()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap())
        }
        ArrayData::TwoD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap())
        }
        ArrayData::ThreeD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .flatten()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap())
        }
        ArrayData::FourD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap())
        }
    }
}

fn find_max(data: &ArrayData) -> f64 {
    match data {
        ArrayData::OneD(arg0) => {
            *(arg0
                .iter()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap())
        }
        ArrayData::TwoD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap())
        }
        ArrayData::ThreeD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .flatten()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap())
        }
        ArrayData::FourD(arg0) => {
            *(arg0
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap())
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayNDJS {
    array: ArrayND,
}

#[wasm_bindgen]
impl ArrayNDJS {
    pub fn new(data: &JsValue) -> Self {
        match data.into_serde::<Vec<f64>>() {
            Ok(num) => {
                let array = ArrayND::new(ArrayData::OneD(num));
                ArrayNDJS { array }
            }
            Err(_) => {
                match data.into_serde::<Vec<Vec<f64>>>() {
                    Ok(num) => {
                        let array = ArrayND::new(ArrayData::TwoD(num));
                        ArrayNDJS { array }
                    }
                    Err(_) => {
                        match data.into_serde::<Vec<Vec<Vec<f64>>>>() {
                            Ok(num) => {
                                let array = ArrayND::new(ArrayData::ThreeD(num));
                                ArrayNDJS { array }
                            }
                            Err(_) => {
                                match data.into_serde::<Vec<Vec<Vec<Vec<f64>>>>>() {
                                    Ok(num) => {
                                        let array = ArrayND::new(ArrayData::FourD(num));
                                        ArrayNDJS { array }
                                    }
                                    Err(_) => panic!("Unable to parse data"),
                                }
                            }
                        }
                    }
                }
            },
        }
    }

    // #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.array.to_string()
    }

    pub fn random(shape: &JsValue) -> Self {
        let shape: Vec<usize> = shape.into_serde().unwrap_or(panic!("Invalid shape"));
        let array = ArrayND::random(shape);
        ArrayNDJS { array }
    }

    pub fn cos(&self) -> Self {
        let array = self.array.cos();
        ArrayNDJS { array }
    }

    pub fn sin(&self) -> Self {
        let array = self.array.sin();
        ArrayNDJS { array }
    }

    pub fn tan(&self) -> Self {
        let array = self.array.tan();
        ArrayNDJS { array }
    }

    pub fn cosh(&self) -> Self {
        let array = self.array.cosh();
        ArrayNDJS { array }
    }

    pub fn sinh(&self) -> Self {
        let array = self.array.sinh();
        ArrayNDJS { array }
    }

    pub fn tanh(&self) -> Self {
        let array = self.array.tanh();
        ArrayNDJS { array }
    }

    pub fn exp(&self) -> Self {
        let array = self.array.exp();
        ArrayNDJS { array }
    }

    pub fn log(&self, base: f64) -> Self {
        let array = self.array.log(base);
        ArrayNDJS { array }
    }
    pub fn log2(&self) -> Self {
        let array = self.array.log2();
        ArrayNDJS { array }
    }

    pub fn loge(&self) -> Self {
        let array = self.array.loge();
        ArrayNDJS { array }
    }

    pub fn logpi(&self) -> Self {
        let array = self.array.logpi();
        ArrayNDJS { array }
    }

    pub fn log10(&self) -> Self {
        let array = self.array.log10();
        ArrayNDJS { array }
    }

    pub fn sq(&self) -> Self {
        let array = self.array.sq();
        ArrayNDJS { array }
    }

    pub fn cube(&self) -> Self {
        let array = self.array.cube();
        ArrayNDJS { array }
    }

    pub fn sqrt(&self) -> Self {
        let array = self.array.sqrt();
        ArrayNDJS { array }
    }

    pub fn is_nan(&self) -> Self {
        let array = self.array.is_nan();
        ArrayNDJS { array }
    }

    pub fn is_infinite(&self) -> Self {
        let array = self.array.is_infinite();
        ArrayNDJS { array }
    }

    pub fn is_finite(&self) -> Self {
        let array = self.array.is_finite();
        ArrayNDJS { array }
    }

    pub fn is_zero(&self) -> Self {
        let array = self.array.is_zero();
        ArrayNDJS { array }
    }

    pub fn is_one(&self) -> Self {
        let array = self.array.is_one();
        ArrayNDJS { array }
    }

    pub fn cbrt(&self) -> Self {
        let array = self.array.cbrt();
        ArrayNDJS { array }
    }

    pub fn abs(&self) -> Self {
        let array = self.array.abs();
        ArrayNDJS { array }
    }

    pub fn floor(&self) -> Self {
        let array = self.array.floor();
        ArrayNDJS { array }
    }

    pub fn ceil(&self) -> Self {
        let array = self.array.ceil();
        ArrayNDJS { array }
    }

    pub fn round(&self) -> Self {
        let array = self.array.round();
        ArrayNDJS { array }
    }

    pub fn trunc(&self) -> Self {
        let array = self.array.trunc();
        ArrayNDJS { array }
    }

    pub fn pow(&self, power: f64) -> Self {
        let array = self.array.pow(power);
        ArrayNDJS { array }
    }

    pub fn min(&self) -> f64 {
        self.array.min
    }

    pub fn max(&self) -> f64 {
        self.array.max
    }

    pub fn greater(&self, value: f64) -> Self {
        let array = self.array.greater(value);
        ArrayNDJS { array }
    }

    pub fn greater_equal(&self, value: f64) -> Self {
        let array = self.array.greater_equal(value);
        ArrayNDJS { array }
    }

    pub fn less(&self, value: f64) -> Self {
        let array = self.array.less(value);
        ArrayNDJS { array }
    }

    pub fn less_equal(&self, value: f64) -> Self {
        let array = self.array.less_equal(value);
        ArrayNDJS { array }
    }

    pub fn equal(&self, value: f64) -> Self {
        let array = self.array.equal(value);
        ArrayNDJS { array }
    }

    pub fn not_equal(&self, value: f64) -> Self {
        let array = self.array.not_equal(value);
        ArrayNDJS { array }
    }

    pub fn add(&self, value: f64) -> Self {
        let array = self.array.add(value);
        ArrayNDJS { array }
    }

    pub fn sub(&self, value: f64) -> Self {
        let array = self.array.sub(value);
        ArrayNDJS { array }
    }

    pub fn mul(&self, value: f64) -> Self {
        let array = self.array.mul(value);
        ArrayNDJS { array }
    }

    pub fn div(&self, value: f64) -> Self {
        let array = self.array.div(value);
        ArrayNDJS { array }
    }

    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let array = self.array.clamp(min, max);
        ArrayNDJS { array }
    }

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

#[cfg(test)]
mod tests {
    use super::{asarray, ArrayData, ArrayND};

    fn get_array_1d_float() -> ArrayND {
        asarray(ArrayData::OneD(vec![1., 2., 3., 4., 5., 6., 7.]))
    }

    fn get_array_2d_float() -> ArrayND {
        ArrayND::new2d(vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]])
    }

    fn get_array_3d_float() -> ArrayND {
        ArrayND::new3d(vec![
            vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]],
            vec![
                vec![10., 11., 12.],
                vec![13., 14., 15.],
                vec![16., 17., 18.],
            ],
            vec![
                vec![19., 20., 21.],
                vec![22., 23., 24.],
                vec![25., 26., 27.],
            ],
        ])
    }

    #[test]
    fn sum_float() {
        let array1d = get_array_1d_float();
        assert_eq!(array1d.sum(), 28.);

        let array2d = get_array_2d_float();
        assert_eq!(array2d.sum(), 45.);

        let array3d = get_array_3d_float();
        assert_eq!(array3d.sum(), 378.);
    }
}
