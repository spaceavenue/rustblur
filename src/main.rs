use std::env;
use std::path::Path;
mod blur;
use blur::{ImageBuffer, Pixel};
use image::{Rgba, RgbaImage};

fn write_err(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        write_err("Usage: <passes> <offset> <input_path> <output_path>");
    }

    let Ok(passes) = args[1].parse::<usize>() else {
        write_err("Passes must be a positive integer");
    };
    let Ok(offset) = args[2].parse::<f32>() else {
        write_err("Offset must be a valid float");
    };
    let input_path = &args[3];
    let output_path = &args[4];

    // load image
    let Ok(img) = image::open(Path::new(input_path)) else {
        write_err("Failed to open input image");
    };

    let img = img.to_rgba8();

    let (width, height) = img.dimensions();

    // Convert pixel arrays to native engine f32 workspace format
    let mut input_buffer = ImageBuffer::new(width as usize, height as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            input_buffer.data[(y as usize) * (width as usize) + (x as usize)] = Pixel {
                r: pixel[0] as f32 / 255.0,
                g: pixel[1] as f32 / 255.0,
                b: pixel[2] as f32 / 255.0,
                a: pixel[3] as f32 / 255.0,
            };
        }
    }
    let output_buffer = blur::run(input_buffer, passes, offset);

    // Convert back into structural u8 integers for file encoding
    let mut out_img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let src_color = output_buffer.data[(y as usize) * (width as usize) + (x as usize)];

            // Map ranges back onto safe bounds
            let r = (src_color.r * 255.0).clamp(0.0, 255.0) as u8;
            let g = (src_color.g * 255.0).clamp(0.0, 255.0) as u8;
            let b = (src_color.b * 255.0).clamp(0.0, 255.0) as u8;
            let a = (src_color.a * 255.0).clamp(0.0, 255.0) as u8;

            out_img.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let Ok(_) = out_img.save(Path::new(output_path)) else {
        write_err("Failed to write output image to disk");
    };
}
