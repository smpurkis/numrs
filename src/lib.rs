use num_traits::Zero;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    fmt::{Debug, Display},
    ops::{Add, Div, Index, Mul, Range, RangeFrom, RangeTo, Sub},
};
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
use web_sys::console;

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

#[wasm_bindgen]
#[derive(Clone)]
pub struct ArrayND {
    data: ArrayData,
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
            data: ArrayData::OneD(data),
            min,
            max,
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
        ArrayND::new(data)
    }

    fn new2d(data: Vec<Vec<f64>>) -> ArrayND {
        let min: f64 = 0.;
        let max: f64 = 0.;
        let shape = vec![data.len(), data[0].len()];

        ArrayND {
            shape,
            size: data.len(),
            data: ArrayData::TwoD(data),
            min,
            max,
        }
    }

    fn new3d(data: Vec<Vec<Vec<f64>>>) -> ArrayND {
        let min: f64 = 0.;
        let max: f64 = 0.;
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];

        ArrayND {
            shape,
            size: data.len(),
            data: ArrayData::ThreeD(data),
            min,
            max,
        }
    }

    fn new4d(data: Vec<Vec<Vec<Vec<f64>>>>) -> ArrayND {
        let min: f64 = 0.;
        let max: f64 = 0.;
        let shape = vec![
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
        ];

        ArrayND {
            shape,
            size: data.len(),
            data: ArrayData::FourD(data),
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

    #[cfg(target_family = "wasm")]
    /// Seqential Sum used inside 1D Array
    pub fn par_sum(&self) -> f64 {
        self.seq_sum()
    }

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

    pub fn add(self, num: f64) -> ArrayND {
        match self.data {
            ArrayData::OneD(mut arg0) => {
                arg0.iter_mut().for_each(|x| *x += num);
                ArrayND {
                    data: ArrayData::OneD(arg0),
                    ..self
                }
            }
            ArrayData::TwoD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    arg1.iter_mut().for_each(|x| *x += num);
                }
                ArrayND {
                    data: ArrayData::TwoD(arg0),
                    ..self
                }
            }
            ArrayData::ThreeD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    for arg2 in arg1.iter_mut() {
                        arg2.iter_mut().for_each(|x| *x += num);
                    }
                }
                ArrayND {
                    data: ArrayData::ThreeD(arg0),
                    ..self
                }
            }
            ArrayData::FourD(mut arg0) => {
                for arg1 in arg0.iter_mut() {
                    for arg2 in arg1.iter_mut() {
                        for arg3 in arg2.iter_mut() {
                            arg3.iter_mut().for_each(|x| *x += num);
                        }
                    }
                }
                ArrayND {
                    data: ArrayData::FourD(arg0),
                    ..self
                }
            }
        }
    }

    /// Convert an array to a nice matrix string output
    ///
    /// # Example
    /// ```
    /// use numrs::ArrayND;
    /// let array: ArrayND = ArrayND::new(vec![1.0, 2.0, 3.0]);
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
            ArrayData::ThreeD(arg0) => {
                arg0.iter().flatten().flatten().cloned().collect()
            }
            ArrayData::FourD(arg0) => {
                arg0.iter().flatten().flatten().flatten().cloned().collect()
            }
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
        ArrayND::new(data)
    }

    pub fn fill(value: f64, shape: Vec<usize>) -> ArrayND {
        let data: ArrayData = match shape.len() {
            1 => {
                ArrayData::OneD(vec![value; shape[0]])
            },
            2 => {
                ArrayData::TwoD(vec![vec![value; shape[1]]; shape[0]])
            },
            3 => {
                ArrayData::ThreeD(vec![vec![vec![value; shape[2]]; shape[1]]; shape[0]])
            },
            4 => {
                ArrayData::FourD(vec![vec![vec![vec![value; shape[3]]; shape[2]]; shape[1]]; shape[0]])
            }
            _ => {
                panic!("ArrayND::fill() only supports 1D, 2D, 3D, and 4D arrays")
            }
        };
        ArrayND::_new(data)
    }

    pub fn zeros(shape: Vec<usize>) -> ArrayND {
        ArrayND::fill(0., shape)
    }

    pub fn ones(shape: Vec<usize>) -> ArrayND {
        ArrayND::fill(1., shape)
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

#[wasm_bindgen]
pub fn asarray1d(data: Vec<f64>) -> ArrayND {
    ArrayND::new(data)
}

#[wasm_bindgen]
pub fn asarray2d(data: &JsValue) -> ArrayND {
    let data: Vec<Vec<f64>> = data.into_serde().unwrap();
    ArrayND::new2d(data)
}
#[wasm_bindgen]
pub fn asarray3d(data: &JsValue) -> ArrayND {
    let data: Vec<Vec<Vec<f64>>> = data.into_serde().unwrap();
    ArrayND::new3d(data)
}
#[wasm_bindgen]
pub fn asarray4d(data: &JsValue) -> ArrayND {
    let data: Vec<Vec<Vec<Vec<f64>>>> = data.into_serde().unwrap();
    ArrayND::new4d(data)
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
    use super::{asarray, ArrayND,ArrayData};


    fn get_array_1d_float() -> ArrayND {
        ArrayND::new(vec![1., 2., 3., 4., 5., 6., 7.])
    }

    fn get_array_2d_float() -> ArrayND {
        ArrayND::new2d(vec![
            vec![1., 2., 3.],
            vec![4., 5., 6.],
            vec![7., 8., 9.],
        ])
    }

    fn get_array_3d_float() -> ArrayND {
        ArrayND::new3d(vec![
            vec![
                vec![1., 2., 3.],
                vec![4., 5., 6.],
                vec![7., 8., 9.],
            ],
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
