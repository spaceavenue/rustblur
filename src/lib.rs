use image;
use wgpu;

fn get_image_data(file_path: &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::open(file_path).expect("Failed to open file.").to_rgba8();
    image
}

pub fn setup_textures(
    device: &wgpu::Device, 
    label: &str,
    (width, height): (u32, u32)
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
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
    entry_point: &str
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

pub async fn run(file_path: &str, d_samples: usize, offset: f32) {
    
    // wgpu setup, keep everything default unless required
    let instance = wgpu::Instance::default();

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();

    let mut image_bytes = get_image_data(file_path);
    
    for pixel in image_bytes.pixels_mut() {
        let alpha = pixel[3] as f32 / 255.0;
        pixel[0] = (pixel[0] as f32 * alpha) as u8;
        pixel[1] = (pixel[1] as f32 * alpha) as u8;
        pixel[2] = (pixel[2] as f32 * alpha) as u8;
    }

    // image dimenstions
    let image_width = image_bytes.width();
    let image_height = image_bytes.height();
    
    let padded_width_in_bytes = (4 * image_width) + 255 & !255;
    
    let mut texture_array: Vec<wgpu::Texture> = Vec::new();
    texture_array.push(setup_textures(&device, "Image", (image_width, image_height)));

    for i in 0..d_samples {
        let c_width = (image_width >> (i + 1)).max(1);
        let c_height = (image_height >> (i + 1)).max(1);
        texture_array.push(setup_textures(&device, &format!("Level_{}", i + 1), (c_width, c_height)));
    }
    
    let texture_size = texture_array[0].size();
    let texture_format = texture_array[0].format();

    // loading initial pixel data from image into texture
    queue.write_texture(
        texture_array[0].as_image_copy(),
        &image_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image_width),
            rows_per_image: Some(image_height),
        },
        texture_size,
    );

    #[repr(C)]
    #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BlurParams {
        halfpixel: [f32; 2],
        offset: f32,
        _padding: u32 //to pad buffer to 16 bytes
    }

    let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Parameter buffer"),
        size: std::mem::size_of::<BlurParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None
            }
        ],
        label: Some("Texture Bind Group Layout"),
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let pipeline_down = setup_render_pipeline(&device, &bind_group_layout, &shader, texture_format, "fs_down");
    let pipeline_up = setup_render_pipeline(&device, &bind_group_layout, &shader, texture_format, "fs_up");
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    for i in 0..d_samples {

        // destination texture dimensions
        let w = (image_width >> (i + 1)).max(1) as f32;
        let h = (image_height >> (i + 1)).max(1) as f32;

        let blur_params = BlurParams {
            halfpixel: [0.5/w, 0.5/h],
            offset: offset,
            _padding: 0
        };

        queue.write_buffer(&blur_params_buffer, 0, bytemuck::cast_slice(&[blur_params]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_array[i].create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_params_buffer.as_entire_binding()
                }
            ],
        });

        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &texture_array[i + 1].create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })
            ],
            ..Default::default()
        };

        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        render_pass.set_pipeline(&pipeline_down);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_viewport(0., 0., w, h, 0., 0.);
        render_pass.draw(0..3, 0..1);
    }

    for j in (0..d_samples).rev() {
        let w = (image_width >> j).max(1) as f32;
        let h = (image_height >> j).max(1) as f32;

        let blur_params = BlurParams {
            //upsampling requires source, aka (j + 1)th, dimensions 
            halfpixel: [0.5/(image_width >> (j + 1)) as f32, 0.5/(image_height >> (j + 1)) as f32],
            offset: offset,
            _padding: 0
        };

        queue.write_buffer(&blur_params_buffer, 0, bytemuck::cast_slice(&[blur_params]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_array[j + 1].create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_params_buffer.as_entire_binding()
                }
            ],
        });

        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &texture_array[j].create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })
            ],
            ..Default::default()
        };

        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        render_pass.set_pipeline(&pipeline_up);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_viewport(0., 0., w, h, 0., 0.);
        render_pass.draw(0..3, 0..1);
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_width_in_bytes * image_height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });
    
    encoder.copy_texture_to_buffer(
        texture_array[0].as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_width_in_bytes),
                rows_per_image: Some(image_height)
            },
        },
        texture_array[0].size()
    );
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    {
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        use image::{ImageBuffer, Rgba};
        let buffer =
            ImageBuffer::<Rgba<u8>, _>::from_raw(padded_width_in_bytes/4, image_height, data).unwrap();
        buffer.save("image.png").unwrap();
    }
    output_buffer.unmap();
}