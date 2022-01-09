mod array_nd;

use array_nd::Array1D;

fn main() {
    let data: Vec<f64> = vec![1., 2., 3., 4., 5., 328., 283.];
    let array: Array1D<f64> = Array1D::new(data);
    println!(
        "{:?}",
        array.clone() + array.clone() + array.clone() + array.clone() + array.clone()
    );
    println!("{:?}", array);
    println!("sum: {:?}", array.sum());
    println!("Hello, world!");
}
