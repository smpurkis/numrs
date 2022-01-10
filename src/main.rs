mod numrs;

use std::time::Instant;

use numrs::Array1D;
use rand::Rng;


fn main() {
    // let array: Array1D<f32> = Array1D::random_range(1000000000, 1., 10.);
    let array: Array1D<_> = numrs::arange(1000000000);
    println!("{:?}", array);
    let now = Instant::now();
    println!("sum seq: {:?}", array.seq_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum par: {:?}", array.par_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum combo: {:?}", array.sum());
    println!("Time Taken: {:?}", now.elapsed());
    println!("Hello, world!");
}
