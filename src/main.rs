use std::time::Instant;

use numrs::ArrayND;

mod lib;



fn main() {


    let array: ArrayND = ArrayND::random(vec![1_000_000]);
    let array_clone = array.clone();
    println!("{:?}", array);
    println!("{:?}", array_clone.add(numrs::OperationData::Array(&array)));
    let now = Instant::now();
    println!("sum seq: {:?}", array.seq_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum par: {:?}", array.seq_sum());
    println!("Time Taken: {:?}", now.elapsed());
    let now = Instant::now();
    println!("sum combo: {:?}", array.sum());
    println!("Time Taken: {:?}", now.elapsed());

    let now = Instant::now();
    println!("cos sum combo: {:?}", array.cos().sin().atan().sum());
    println!("Time Taken: {:?}", now.elapsed());
    println!("Hello, world!");
}
