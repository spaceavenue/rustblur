use std::fs::File;
use std::io::Write;
use std::time::Instant;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use wgpu::PowerPreference;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    halfpixel: [f32; 2],
    offset: f32,
    _padding: u32, //to pad buffer to 16 bytes
}

pub fn setup_textures(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    passes: u32,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: width,
            height: height,
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
    });
    texture
}

pub fn setup_render_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    texture_format: wgpu::TextureFormat,
    entry_point: &str,
) -> wgpu::RenderPipeline {
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    });
    render_pipeline
}

fn blur_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    src_view: &wgpu::TextureView,
    dst_view: &wgpu::TextureView,
    params: BlurParams,
    dst_w: f32,
    dst_h: f32,
) {
    let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Parameter buffer"),
        size: std::mem::size_of::<BlurParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // write the data of blur params to uniform buffer, will be submitted to gpu at the end of
    // the pass
    queue.write_buffer(&blur_params_buffer, 0, bytemuck::cast_slice(&[params]));
    {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &bind_group_layout,
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
                    resource: blur_params_buffer.as_entire_binding(),
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

        // start recording the render pass to the command encoder will be submitted at the end
        // of the pass
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_viewport(0., 0., dst_w, dst_h, 0., 0.);
        render_pass.draw(0..3, 0..1);
    }
}

fn write_output(
    device: &wgpu::Device,
    output_buffer: &wgpu::Buffer,
    img_dims: (u32, u32),
    padded_width_in_bytes: u32,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // writing output to image.png
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(
        wgpu::MapMode::Read,
        move |result: Result<(), wgpu::BufferAsyncError>| {
            let _ = tx.send(result);
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let _ = rx.recv();
    let out_image = {
        let data = buffer_slice.get_mapped_range();
        let mut out_image = Vec::<u8>::with_capacity((4 * img_dims.0 * img_dims.1) as usize);
        data.chunks_exact((padded_width_in_bytes) as usize)
            .for_each(|chunk| out_image.extend_from_slice(&chunk[..(4 * img_dims.0) as usize]));
        out_image
    };

    output_buffer.unmap();

    let mut buffer = Vec::new();
    let image_encoder =
        PngEncoder::new_with_quality(&mut buffer, CompressionType::Fast, FilterType::Paeth);
    image_encoder.write_image(&out_image, img_dims.0, img_dims.1, ExtendedColorType::Rgba8)?;
    File::create(output_path)?.write_all(&buffer)?;

    Ok(())
}

pub async fn run(
    file_path: &str,
    passes: usize,
    offset: f32,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // get image data
    let mut image_bytes = image::open(file_path)?.into_rgba8();

    let now = Instant::now();
    // premultiply alpha, in parallel
    image_bytes.par_chunks_mut(4).for_each(|pixel| {
        let alpha = pixel[3] as f32 / 255.0;
        pixel[0] = (pixel[0] as f32 * alpha) as u8;
        pixel[1] = (pixel[1] as f32 * alpha) as u8;
        pixel[2] = (pixel[2] as f32 * alpha) as u8;
    });
    println!("alpha premult: {:?}", now.elapsed());

    // image dimenstions
    let image_width = image_bytes.width();
    let image_height = image_bytes.height();
    // for padding out the width to multiple of 256, for the output buffer
    let padded_width_in_bytes = (4 * image_width) + 255 & !255;

    // setup textures/mip chain. each texture is half the size of the last. first level (0) is the
    // image itself.
    let textures = setup_textures(
        &device,
        &format!("Blur mip chain"),
        (image_width, image_height),
        passes as u32,
    );
    let texture_size = textures.size();
    let texture_format = textures.format();

    let now = Instant::now();
    // loading initial pixel data from image into texture level 0
    queue.write_texture(
        textures.as_image_copy(),
        &image_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image_width),
            rows_per_image: Some(image_height),
        },
        texture_size,
    );
    println!("image upload: {:?}", now.elapsed());

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Texture Bind Group Layout"),
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let pipeline_down = setup_render_pipeline(
        &device,
        &bind_group_layout,
        &shader,
        texture_format,
        "fs_down",
    );
    let pipeline_up = setup_render_pipeline(
        &device,
        &bind_group_layout,
        &shader,
        texture_format,
        "fs_up",
    );

    let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let now = Instant::now();
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // downsample loop
    // here:
    // i -> src texture
    // (i + 1) -> dst texture
    for i in 0..passes {
        // destination texture dimensions, needed for setting viewport size and half-pixel
        // calculation
        let dst_w = (image_width >> (i + 1)).max(1) as f32;
        let dst_h = (image_height >> (i + 1)).max(1) as f32;

        let params = BlurParams {
            halfpixel: [0.5 / dst_w, 0.5 / dst_h],
            offset: offset,
            _padding: 0,
        };
        let src_view = &textures.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: i as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = &textures.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: (i + 1) as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        blur_pass(
            &device,
            &queue,
            &mut encoder,
            &pipeline_down,
            &bind_group_layout,
            &texture_sampler,
            src_view,
            dst_view,
            params,
            dst_w,
            dst_h,
        );
    }

    // upsample loop
    // here:
    // (j + 1) -> src texture
    // j -> dst texture
    for j in (0..passes).rev() {
        // source texture dimensions, needed for half-pixel calculations
        let src_w = (image_width >> (j + 1)) as f32;
        let src_h = (image_height >> (j + 1)) as f32;

        // destination texture dimensions, needed for setting viewport size
        let dst_w = (image_width >> j).max(1) as f32;
        let dst_h = (image_height >> j).max(1) as f32;

        let params = BlurParams {
            halfpixel: [0.5 / src_w, 0.5 / src_h],
            offset: offset,
            _padding: 0,
        };
        let src_view = &textures.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: (j + 1) as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = &textures.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: j as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        blur_pass(
            &device,
            &queue,
            &mut encoder,
            &pipeline_up,
            &bind_group_layout,
            &texture_sampler,
            src_view,
            dst_view,
            params,
            dst_w,
            dst_h,
        );
    }

    // output buffer to hold our image
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_width_in_bytes * image_height) as wgpu::BufferAddress,
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
                rows_per_image: Some(image_height),
            },
        },
        textures.size(),
    );

    queue.submit(Some(encoder.finish()));
    println!("up + down + texture copy: {:?}", now.elapsed());

    let now = Instant::now();
    write_output(
        &device,
        &output_buffer,
        (image_width, image_height),
        padded_width_in_bytes,
        output_path,
    )?;
    println!("map buffer + write output: {:?}", now.elapsed());

    Ok(())
}
