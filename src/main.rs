mod array_nd;

use array_nd::Array1D;

fn main() {
    let data: Vec<i64> = (1..100000000).collect();
    let array: Array1D<i64> = Array1D::new(data);
    println!(
        "{:?}",
        &(array.clone() / array.clone()).data[1..100]
    );
    // println!("{:?}", array);
    println!("sum: {:?}", array.sum());
    println!("Hello, world!");
}
