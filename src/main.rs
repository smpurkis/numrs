mod lib;

use std::time::Instant;

use lib::ArrayND;


fn main() {

    let array: ArrayND = ArrayND::random(100_000_000);
    // let array: ArrayND<_> = numrs::arange(100_000_000);
    println!("{:?}", array);
    let now = Instant::now();
    println!("sum seq: {:?}", array.seq_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum par: {:?}", array.seq_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum combo: {:?}", array.sum());
    println!("Time Taken: {:?}", now.elapsed());
    println!("Hello, world!");
}
