use std::env;
use rustblur::run;

fn main() {
    let args: Vec<String> = env::args().collect();
    let passes = &args[1].parse::<usize>().expect("Error parsing passes.");
    let offset = &args[2].parse::<f32>().expect("Error parsing offset value.");
    let file_path = &args[3];
    pollster::block_on(run(file_path, *passes, *offset));
}
