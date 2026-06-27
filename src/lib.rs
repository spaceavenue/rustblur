use image::{ImageBuffer, Rgba};
use wgpu;

pub fn setup_textures(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
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

pub async fn run(
    file_path: &str,
    passes: usize,
    offset: f32,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // wgpu setup, keep everything default unless required
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // get image data
    let mut image_bytes = image::open(file_path)?.to_rgba8();

    // alpha
    for pixel in image_bytes.pixels_mut() {
        let alpha = pixel[3] as f32 / 255.0;
        pixel[0] = (pixel[0] as f32 * alpha) as u8;
        pixel[1] = (pixel[1] as f32 * alpha) as u8;
        pixel[2] = (pixel[2] as f32 * alpha) as u8;
    }

    // image dimenstions
    let image_width = image_bytes.width();
    let image_height = image_bytes.height();

    // setup textures/mip chain, each texture half the size of the last. theres a way to do this
    // with actual mip chains but a vec of textures works too. first level (0) is the image itself.
    let mut texture_array: Vec<wgpu::Texture> = Vec::new();
    for i in 0..passes + 1 {
        let c_width = (image_width >> i).max(1);
        let c_height = (image_height >> i).max(1);
        texture_array.push(setup_textures(
            &device,
            &format!("Level_{}", i),
            (c_width, c_height),
        ));
    }

    let texture_size = texture_array[0].size();
    let texture_format = texture_array[0].format();

    // loading initial pixel data from image into texture level 0
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
        _padding: u32, //to pad buffer to 16 bytes
    }

    let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
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

    // downsample loop
    // here:
    // i -> src texture
    // (i + 1) -> dest texture
    for i in 0..passes {
        // destination texture dimensions, needed for setting viewport size and half-pixel
        // calculation
        let dest_w = (image_width >> (i + 1)).max(1) as f32;
        let dest_h = (image_height >> (i + 1)).max(1) as f32;

        let blur_params = BlurParams {
            halfpixel: [0.5 / dest_w, 0.5 / dest_h],
            offset: offset,
            _padding: 0,
        };

        // create the command encoder for this pass. will accept commands, turn them into a buffer,
        // and send to the gpu for exec at the end
        let mut down_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // write the data of blur params to uniform buffer, will be submitted to gpu at the end of
        // the pass
        queue.write_buffer(&blur_params_buffer, 0, bytemuck::cast_slice(&[blur_params]));
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &texture_array[i].create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
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
                    view: &texture_array[i + 1]
                        .create_view(&wgpu::TextureViewDescriptor::default()),
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
            let mut render_pass = down_encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&pipeline_down);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_viewport(0., 0., dest_w, dest_h, 0., 0.);
            render_pass.draw(0..3, 0..1);
        }

        // write the command buffer with our render pass to the queue and submit for exec. important
        // that we do it at the end of every pass and not at once, otherwise the blur_params data
        // will keep overwriting the buffer after each pass, resulting in only the last one being
        // read by the gpu
        queue.submit(Some(down_encoder.finish()));
    }

    // upsample loop
    // here:
    // (j + 1) -> src texture
    // j -> dest texture
    for j in (0..passes).rev() {
        // source texture dimensions, needed for half-pixel calculations
        let src_w = (image_width >> (j + 1)) as f32;
        let src_h = (image_height >> (j + 1)) as f32;

        // destination texture dimensions, needed for setting viewport size
        let dest_w = (image_width >> j).max(1) as f32;
        let dest_h = (image_height >> j).max(1) as f32;

        let blur_params = BlurParams {
            halfpixel: [0.5 / src_w, 0.5 / src_h],
            offset: offset,
            _padding: 0,
        };

        let mut up_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        queue.write_buffer(&blur_params_buffer, 0, bytemuck::cast_slice(&[blur_params]));
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &texture_array[j + 1]
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
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
                    view: &texture_array[j].create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            };

            let mut render_pass = up_encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&pipeline_up);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_viewport(0., 0., dest_w, dest_h, 0., 0.);
            render_pass.draw(0..3, 0..1);
        }
        queue.submit(Some(up_encoder.finish()));
    }

    // for padding out the width to multiple of 256, for the output buffer
    let padded_width_in_bytes = (4 * image_width) + 255 & !255;

    let mut out_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // output buffer to hold our image
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_width_in_bytes * image_height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });

    // copy data from level 0 texture to the buffer
    out_encoder.copy_texture_to_buffer(
        texture_array[0].as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_width_in_bytes),
                rows_per_image: Some(image_height),
            },
        },
        texture_array[0].size(),
    );
    queue.submit(Some(out_encoder.finish()));

    // writing output to image.png
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    {
        buffer_slice.map_async(
            wgpu::MapMode::Read,
            move |result: Result<(), wgpu::BufferAsyncError>| {
                let _ = tx.send(result);
            },
        );
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let _ = rx.recv();

        let data = buffer_slice.get_mapped_range();
        let mut unpadded_data = Vec::<u8>::with_capacity((4 * image_width) as usize);
        data.chunks_exact((padded_width_in_bytes) as usize)
            .for_each(|chunk| {
                unpadded_data.extend_from_slice(&chunk[..(4 * image_width) as usize])
            });
        let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(image_width, image_height, unpadded_data)
            .ok_or("Unable to create image buffer.")?;
        buffer.save_with_format(output_path, image::ImageFormat::Png)?;
    }
    output_buffer.unmap();

    Ok(())
}
