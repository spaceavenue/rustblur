use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Cursor};

use image::{DynamicImage, RgbaImage};
use memmap2::Mmap;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use wgpu::{Buffer, BufferAsyncError, MapMode, PollType};

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
            // check if theres 3 channels. if there are, we have to convert to an RGBA buffer first
            // unfortunately. maybe something like `into_rgba8() from image.rs should also exist for
            // qoz?
            let header = qoz::read_header(&in_mmap)?;
            let bytes = if header.channels != 4 {
                let (_, bytes) = qoz::decode(&in_mmap)?;
                let mut bytes_new = vec![0u8; (header.width * header.height * 4) as usize];
                bytes
                    .par_chunks(3 * header.width as usize)
                    .zip(bytes_new.par_chunks_mut(4 * header.width as usize))
                    .for_each(|(src_row, dst_row)| {
                        dst_row
                            .chunks_exact_mut(4)
                            .zip(src_row.chunks_exact(3))
                            .for_each(|(d, s)| {
                                d[0] = s[0];
                                d[1] = s[1];
                                d[2] = s[2];
                                d[3] = 255u8;
                            });
                    });
                bytes_new
            } else {
                let (_, bytes) = qoz::decode(&in_mmap)?;
                bytes
            };
            return Ok(InputImage {
                bytes,
                width: header.width,
                height: header.height,
            });
        }
        let img = image::ImageReader::new(Cursor::new(&in_mmap))
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
    output_buffer: &Buffer,
    (width, height): (u32, u32),
    padded_width_in_bytes: u32,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    // writing output to image.png
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(
        MapMode::Read,
        move |result: Result<(), BufferAsyncError>| {
            let _ = tx.send(result);
        },
    );
    state.device.poll(PollType::wait_indefinitely())?;
    rx.recv()??;

    // our output buffer's stride is 4 * padded_width_in_bytes, because we padded it to comply with
    // the buffer alignment in wgpu. we need to extract rows that are of 4 * width to get the actual
    // image. if it's already 4*width (no padding was needed), we just get the entire buffer as is.
    let row_bytes = (width * 4) as usize;
    let out_image = if padded_width_in_bytes as usize == row_bytes {
        buffer_slice.get_mapped_range().to_vec()
    } else {
        let mut out_image = vec![0u8; row_bytes * height as usize];
        buffer_slice
            .get_mapped_range()
            .par_chunks(padded_width_in_bytes as usize)
            .zip(out_image.par_chunks_mut(row_bytes))
            .for_each(|(src_row, dst_row)| {
                dst_row.copy_from_slice(&src_row[..row_bytes]);
            });
        out_image
    };

    // try and save in inferred format from output_path.
    match image::ImageFormat::from_path(output_path) {
        Ok(fmt) => {
            DynamicImage::ImageRgba8(
                RgbaImage::from_raw(width, height, out_image)
                    .ok_or("Could not make Image Buffer")?,
            )
            .write_to(&mut BufWriter::new(File::create(output_path)?), fmt)?;
        }
        Err(_) => {
            std::fs::write(
                output_path,
                qoz::encode(&out_image, width, height, &qoz::EncodeOptions::default())?,
            )?;
        }
    }
    output_buffer.unmap();
    Ok(())
}
