mod blur;
mod image_io;
mod mip;
mod pipeline;
mod wgpu_ctx;

use crate::blur::BlurCtx;
use crate::pipeline::{BlurPipelines, BlurUniforms};
use crate::wgpu_ctx::WgpuCtx;

pub async fn run(
    file_path: &str,
    passes: usize,
    offset: f32,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // initialize wgpu state
    let wgpu_ctx = WgpuCtx::init().await?;

    let image = image_io::InputImage::load(file_path)?;

    // setup textures/mip chain. each texture is half the size of the last. first level (0) is the
    // image itself.
    let mip_chain = mip::MipChain::new(&wgpu_ctx, (image.width, image.height), passes);

    // load initial pixel data from image into texture level 0
    wgpu_ctx.queue.write_texture(
        mip_chain.texture.as_image_copy(),
        &image.bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image.width),
            rows_per_image: Some(image.height),
        },
        mip_chain.texture.size(),
    );

    // set up the blur context
    let blur_params = BlurUniforms::new(&wgpu_ctx, &mip_chain.levels, offset);
    let pipelines = BlurPipelines::new(&wgpu_ctx, mip_chain.texture.format());
    let mut encoder = wgpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let blur_ctx = BlurCtx::new(&wgpu_ctx, &pipelines, &blur_params);

    // execute the blur passes
    blur_ctx.execute(&mut encoder, passes, &mip_chain);

    // pad the output buffer since wgpu requres it
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_width_in_bytes = ((4 * image.width) + align - 1) & !(align - 1);

    // output buffer to hold our image
    let output_buffer = wgpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_width_in_bytes * image.height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });

    // copy data from level 0 texture to the buffer
    encoder.copy_texture_to_buffer(
        mip_chain.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_width_in_bytes),
                rows_per_image: Some(image.height),
            },
        },
        mip_chain.texture.size(),
    );

    wgpu_ctx.queue.submit(Some(encoder.finish()));

    image_io::write_output_image(
        &wgpu_ctx,
        &output_buffer,
        (image.width, image.height),
        padded_width_in_bytes,
        output_path,
    )?;

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <passes> <offset> <input-path> <output-path>",
            args[0]
        );
        std::process::exit(1)
    }

    let passes = &args[1].parse::<usize>().expect("Error parsing passes.");
    let offset = &args[2].parse::<f32>().expect("Error parsing offset value.");
    let file_path = &args[3];
    let output_path = &args[4];

    pollster::block_on(run(file_path, *passes, *offset, output_path)).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        std::process::exit(1)
    })
}
