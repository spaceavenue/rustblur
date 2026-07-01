extern crate alloc;
use alloc::vec::Vec;

use rayon::iter::ParallelIterator;
#[cfg(feature = "rayon")]
use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};

impl core::ops::Add for Pixel {
    type Output = Pixel;

    fn add(self, rhs: Self) -> Self::Output {
        Pixel {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
            a: self.a + rhs.a,
        }
    }
}
impl core::ops::Mul<f32> for Pixel {
    type Output = Pixel;

    fn mul(self, rhs: f32) -> Self::Output {
        Pixel {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
            a: self.a * rhs,
        }
    }
}
#[derive(Clone, Copy)]
pub struct Pixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}
impl Pixel {
    const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
}
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Pixel>,
}
impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        ImageBuffer {
            width,
            height,
            data: vec![Pixel::BLACK; width * height],
        }
    }
    fn get_pixel(&self, x: isize, y: isize) -> Pixel {
        let x = x.clamp(0, (self.width - 1) as isize) as usize;
        let y = y.clamp(0, (self.height - 1) as isize) as usize;

        // SAFETY: we literally just checked above
        unsafe {
            *self
                .data
                .get_unchecked(y as usize * self.width + x as usize)
        }
    }
    fn bilinear_sample(&self, x: f32, y: f32) -> Pixel {
        let x_shifted = x - 0.5;
        let y_shifted = y - 0.5;

        let x0f = x_shifted.floor();
        let y0f = y_shifted.floor();

        let x0 = x0f as isize;
        let y0 = y0f as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = x_shifted - x0f;
        let ty = y_shifted - y0f;

        let p00 = self.get_pixel(x0, y0);
        let p01 = self.get_pixel(x1, y0);
        let p10 = self.get_pixel(x0, y1);
        let p11 = self.get_pixel(x1, y1);

        let top = p00 * (1.0 - tx) + p01 * tx;
        let bottom = p10 * (1.0 - tx) + p11 * tx;
        let out = top * (1.0 - ty) + bottom * ty;

        out
    }
}

fn downsample(src: &ImageBuffer, offset: f32) -> ImageBuffer {
    let dst_w = src.width >> 1;
    let dst_h = src.height >> 1;
    let scale_x = src.width as f32 / dst_w as f32;
    let scale_y = src.height as f32 / dst_h as f32;

    let mut dst = ImageBuffer::new(dst_w, dst_h);
    #[cfg(feature = "rayon")]
    dst.data
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = (y as f32 + 0.5) * scale_y;
            for x in 0..dst_w {
                let src_x = (x as f32 + 0.5) * scale_x;

                let c = src.bilinear_sample(src_x, src_y) * 4.0
                    + src.bilinear_sample(src_x - 0.5 * offset, src_y + 0.5 * offset)
                    + src.bilinear_sample(src_x + 0.5 * offset, src_y - 0.5 * offset)
                    + src.bilinear_sample(src_x + 0.5 * offset, src_y + 0.5 * offset)
                    + src.bilinear_sample(src_x - 0.5 * offset, src_y - 0.5 * offset);

                row[x] = c * 0.125;
            }
        });

    #[cfg(not(feature = "rayon"))]
    for y in 0..dst_h {
        let src_y = (y as f32 + 0.5) * scale_y;
        for x in 0..dst_w {
            let src_x = (x as f32 + 0.5) * scale_x;

            let c = src.bilinear_sample(src_x, src_y) * 4.0
                + src.bilinear_sample(src_x - 0.5 * offset, src_y + 0.5 * offset)
                + src.bilinear_sample(src_x + 0.5 * offset, src_y - 0.5 * offset)
                + src.bilinear_sample(src_x + 0.5 * offset, src_y + 0.5 * offset)
                + src.bilinear_sample(src_x - 0.5 * offset, src_y - 0.5 * offset);

            dst.data[y * dst_w + x] = c * (1.0 as f32 / 8.0 as f32);
        }
    }
    dst
}
fn upsample(src: &ImageBuffer, offset: f32) -> ImageBuffer {
    let dst_w = src.width << 1;
    let dst_h = src.height << 1;
    let scale_x = src.width as f32 / dst_w as f32;
    let scale_y = src.height as f32 / dst_h as f32;

    let mut dst = ImageBuffer::new(dst_w, dst_h);

    #[cfg(feature = "rayon")]
    dst.data
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = (y as f32 + 0.5) * scale_y;
            for x in 0..dst_w {
                let src_x = (x as f32 + 0.5) * scale_x;
                let o = 0.5 * offset;

                let c = src.bilinear_sample(src_x - 2.0 * o, src_y)
                    + src.bilinear_sample(src_x + 2.0 * o, src_y)
                    + src.bilinear_sample(src_x, src_y + 2.0 * o)
                    + src.bilinear_sample(src_x, src_y - 2.0 * o)
                    + src.bilinear_sample(src_x - o, src_y + o) * 2.0
                    + src.bilinear_sample(src_x + o, src_y + o) * 2.0
                    + src.bilinear_sample(src_x + o, src_y - o) * 2.0
                    + src.bilinear_sample(src_x - o, src_y - o) * 2.0;

                row[x] = c * (1.0 as f32 / 12.0 as f32);
            }
        });
    #[cfg(not(feature = "rayon"))]
    for y in 0..dst_h {
        let src_y = (y as f32 + 0.5) * scale_y;
        for x in 0..dst_w {
            let src_x = (x as f32 + 0.5) * scale_x;
            let o = 0.5 * offset;

            let c = src.bilinear_sample(src_x - 2.0 * o, src_y)
                + src.bilinear_sample(src_x + 2.0 * o, src_y)
                + src.bilinear_sample(src_x, src_y + 2.0 * o)
                + src.bilinear_sample(src_x, src_y - 2.0 * o)
                + src.bilinear_sample(src_x - o, src_y + o) * 2.0
                + src.bilinear_sample(src_x + o, src_y + o) * 2.0
                + src.bilinear_sample(src_x + o, src_y - o) * 2.0
                + src.bilinear_sample(src_x - o, src_y - o) * 2.0;

            dst.data[y * dst_w + x] = c * (1.0 as f32 / 12.0 as f32);
        }
    }
    dst
}

pub fn run(src_image: ImageBuffer, passes: usize, offset: f32) -> ImageBuffer {
    if passes == 0 {
        return src_image;
    }
    let mut mip_chain: Vec<ImageBuffer> = Vec::with_capacity(passes + 1);
    let mut current = src_image;
    for _ in 0..passes {
        let next = downsample(&current, offset);
        mip_chain.push(current);
        current = next;
    }
    while mip_chain.pop().is_some() {
        current = upsample(&current, offset);
    }
    current
}
