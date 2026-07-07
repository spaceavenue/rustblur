use std::error::Error;
use std::fs::{File, OpenOptions};
use std::time::Instant;

use memmap2::{Mmap, MmapMut};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use wgpu::{DynamicOffset, PowerPreference};

struct WgpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl WgpuState {
    async fn init() -> Result<Self, Box<dyn Error>> {
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        Ok(Self { device, queue })
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    halfpixel: [f32; 2],
    offset: f32,
    _padding: u32, //to pad buffer to 16 bytes
}

struct InputImage {
    bytes: Vec<u8>,
    dims: (u32, u32),
}

fn process_images(file_path: &str) -> Result<InputImage, Box<dyn Error>> {
    let in_mmap = unsafe { Mmap::map(&File::open(file_path)?)? };
    let header = qoi::decode_header(&in_mmap)?;
    let image_bytes = qoi::Decoder::new(&in_mmap)?
        .with_channels(qoi::Channels::Rgba)
        .decode_to_vec()?;
    Ok(InputImage {
        bytes: image_bytes,
        dims: (header.width, header.height),
    })
}

fn setup_textures(
    state: &WgpuState,
    label: &str,
    (width, height): (u32, u32),
    passes: u32,
) -> wgpu::Texture {
    state.device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: passes + 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
        label: Some(label),
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    })
}

fn setup_render_pipeline(
    state: &WgpuState,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    texture_format: wgpu::TextureFormat,
    entry_point: &str,
) -> wgpu::RenderPipeline {
    let render_pipeline_layout =
        state
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(bind_group_layout)],
                immediate_size: 0,
            });

    state
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Downsample Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry_point),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
}

fn setup_blur_params(
    state: &WgpuState,
    passes: usize,
    image_dims: (u32, u32),
    offset: f32,
    alignment: usize,
) -> wgpu::Buffer {
    let mut blur_params = Vec::<u8>::new();
    for i in 0..passes {
        let params = BlurParams {
            halfpixel: [
                0.5 / (image_dims.0 >> (i + 1)).max(1) as f32,
                0.5 / (image_dims.1 >> (i + 1)).max(1) as f32,
            ],
            offset,
            _padding: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        blur_params.extend_from_slice(bytes);
        blur_params.resize(blur_params.len() + alignment - bytes.len(), 0);
    }
    let blur_params_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Parameter buffer"),
        size: blur_params.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    state
        .queue
        .write_buffer(&blur_params_buffer, 0, &blur_params);
    blur_params_buffer
}

#[allow(clippy::too_many_arguments)]
fn blur_pass(
    state: &WgpuState,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    src_view: &wgpu::TextureView,
    dst_view: &wgpu::TextureView,
    blur_params_buffer: &wgpu::Buffer,
    params_offset: DynamicOffset,
    dst_w: f32,
    dst_h: f32,
) {
    let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Texture Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: blur_params_buffer,
                    offset: 0, // base offset is 0, the dynamic offset is added to this later
                    size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<BlurParams>() as u64)
                            .unwrap(),
                    ),
                }),
            },
        ],
    });

    let render_pass_desc = wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        ..Default::default()
    };

    // start recording the render pass to the command encoder
    let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[params_offset]);
    render_pass.set_viewport(0., 0., dst_w, dst_h, 0., 0.);
    render_pass.draw(0..3, 0..1);
}

fn write_output(
    state: &WgpuState,
    output_buffer: &wgpu::Buffer,
    img_dims: (u32, u32),
    padded_width_in_bytes: u32,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let now = Instant::now();
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
    println!("map buffer: {:?}", now.elapsed());

    let now = Instant::now();
    let out_image = {
        let data = buffer_slice.get_mapped_range();
        let width = img_dims.0 as usize;
        let height = img_dims.1 as usize;
        // let mut out_image = vec![0u8; 3 * width * height];
        let mut out_image = Vec::<u8>::with_capacity(3 * width * height);
        data.par_chunks(padded_width_in_bytes as usize)
            .zip(out_image.par_chunks_mut(3 * width))
            .for_each(|(src_row, dst_row)| {
                dst_row
                    .chunks_exact_mut(3)
                    .zip(src_row[..4 * width].chunks_exact(4))
                    .for_each(|(d, s)| {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                    });
            });
        out_image
    };
    println!("{}", out_image.len());
    output_buffer.unmap();
    println!("process output image: {:?}", now.elapsed());

    let out_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)?;
    out_file.set_len(((img_dims.0 * img_dims.1 * 4) + 22) as u64)?;
    let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };

    let now = Instant::now();
    let encoder = qoi::Encoder::new(&out_image, img_dims.0, img_dims.1)?;
    encoder.encode_to_buf(&mut out_mmap)?;
    println!("encoder image: {:?}", now.elapsed());

    out_mmap.flush_async()?;
    Ok(())
}

pub async fn run(
    file_path: &str,
    passes: usize,
    offset: f32,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // initialize wgpu state
    let state = WgpuState::init().await?;

    let now = Instant::now();
    let image = process_images(file_path)?;
    println!("process input image: {:?}", now.elapsed());

    // setup textures/mip chain. each texture is half the size of the last. first level (0) is the
    // image itself.
    let textures = setup_textures(&state, "Blur mip chain", image.dims, passes as u32);

    let now = Instant::now();
    // load initial pixel data from image into texture level 0
    state.queue.write_texture(
        textures.as_image_copy(),
        &image.bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image.dims.0),
            rows_per_image: Some(image.dims.1),
        },
        textures.size(),
    );
    println!("image upload: {:?}", now.elapsed());

    // create all our texture views
    let mut texture_views = Vec::new();
    for i in 0..passes + 1 {
        let view = textures.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: i as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        texture_views.push(view);
    }

    // create texture sampler
    let texture_sampler = state.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let alignment = state.device.limits().min_uniform_buffer_offset_alignment as usize;
    let blur_params_buffer = setup_blur_params(&state, passes, image.dims, offset, alignment);

    let bind_group_layout =
        state
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Texture Bind Group Layout"),
            });

    let shader = state
        .device
        .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let pipeline_down = setup_render_pipeline(
        &state,
        &bind_group_layout,
        &shader,
        textures.format(),
        "fs_down",
    );
    let pipeline_up = setup_render_pipeline(
        &state,
        &bind_group_layout,
        &shader,
        textures.format(),
        "fs_up",
    );

    let now = Instant::now();
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    for i in 0..passes {
        // downsample loop
        // i -> src texture
        // (i + 1) -> dst texture
        //
        // destination texture dimensions, needed for setting viewport size and half-pixel
        // calculation
        let dst_w = (image.dims.0 >> (i + 1)).max(1) as f32;
        let dst_h = (image.dims.1 >> (i + 1)).max(1) as f32;

        blur_pass(
            &state,
            &mut encoder,
            &pipeline_down,
            &bind_group_layout,
            &texture_sampler,
            &texture_views[i],
            &texture_views[i + 1],
            &blur_params_buffer,
            (i * alignment) as DynamicOffset,
            dst_w,
            dst_h,
        );
    }

    for j in (0..passes).rev() {
        // upsample loop
        // (j + 1) -> src texture
        // j -> dst texture
        //
        // destination texture dimensions, needed for setting viewport size
        let dst_w = (image.dims.0 >> j).max(1) as f32;
        let dst_h = (image.dims.1 >> j).max(1) as f32;

        blur_pass(
            &state,
            &mut encoder,
            &pipeline_up,
            &bind_group_layout,
            &texture_sampler,
            &texture_views[j + 1],
            &texture_views[j],
            &blur_params_buffer,
            (j * alignment) as DynamicOffset,
            dst_w,
            dst_h,
        );
    }

    let padded_width_in_bytes = ((4 * image.dims.0) + 255) & !255;
    // output buffer to hold our image
    let output_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_width_in_bytes * image.dims.1) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });

    // copy data from level 0 texture to the buffer
    encoder.copy_texture_to_buffer(
        textures.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_width_in_bytes),
                rows_per_image: Some(image.dims.1),
            },
        },
        textures.size(),
    );

    state.queue.submit(Some(encoder.finish()));
    println!("up + down + texture copy: {:?}", now.elapsed());

    write_output(
        &state,
        &output_buffer,
        image.dims,
        padded_width_in_bytes,
        output_path,
    )?;

    Ok(())
}
