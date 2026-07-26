use std::error::Error;
use std::fs::File;
use std::io::Cursor;

use image::ImageReader;
use memmap2::Mmap;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::wgpu_ctx::WgpuCtx;

pub struct InputImage {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}
impl InputImage {
    pub fn load(file_path: &str) -> Result<InputImage, Box<dyn Error>> {
        let in_mmap = unsafe { Mmap::map(&File::open(file_path)?)? };

        if in_mmap.starts_with(b"QOZ1") {
            let (header, bytes) = qoz::decode(&in_mmap)?;
            return Ok(InputImage {
                bytes,
                width: header.width,
                height: header.height,
            });
        }

        let img = ImageReader::new(Cursor::new(&in_mmap))
            .with_guessed_format()?
            .decode()?;
        let img_dims = (img.width(), img.height());

        Ok(InputImage {
            bytes: img.into_rgba8().into_raw(),
            width: img_dims.0,
            height: img_dims.1,
        })
    }
}

pub fn write_output_image(
    state: &WgpuCtx,
    output_buffer: &wgpu::Buffer,
    (width, height): (u32, u32),
    padded_width_in_bytes: u32,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    // writing output to image.png
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(
        wgpu::MapMode::Read,
        move |result: Result<(), wgpu::BufferAsyncError>| {
            let _ = tx.send(result);
        },
    );
    state.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;

    let out_image = {
        let data = buffer_slice.get_mapped_range();
        let mut out_image = vec![0u8; (width * height * 3) as usize];

        data.par_chunks(padded_width_in_bytes as usize)
            .zip(out_image.par_chunks_mut(3 * width as usize))
            .for_each(|(src_row, dst_row)| {
                dst_row
                    .chunks_exact_mut(3)
                    .zip(src_row[..4 * width as usize].chunks_exact(4))
                    .for_each(|(d, s)| {
                        d.copy_from_slice(&s[..3]);
                    });
            });
        out_image
    };

    image::RgbImage::from_raw(width, height, out_image)
        .ok_or("Buffer size mismatch")?
        .save(output_path)?;
    output_buffer.unmap();

    Ok(())
}
