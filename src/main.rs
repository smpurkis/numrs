mod lib;

use std::time::Instant;

use lib::Array1D;


fn main() {

    let array: Array1D = Array1D::random(100_000_000);
    // let array: Array1D<_> = numrs::arange(100_000_000);
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
