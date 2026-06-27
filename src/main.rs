use rustblur::run;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let passes = &args[1].parse::<usize>().expect("Error parsing passes.");
    let offset = &args[2].parse::<f32>().expect("Error parsing offset value.");
    let file_path = &args[3];
    let output_path = &args[4];

    pollster::block_on(run(file_path, *passes, *offset, output_path)).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        std::process::exit(1)
    })
}
