use wgpu::*;

use crate::mip::MipLevel;
use crate::wgpu_ctx::WgpuCtx;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    halfpixel: [f32; 2],
    offset: f32,
    _padding: u32, // to pad buffer to 16 bytes
}

pub struct BlurUniforms {
    pub buffer: Buffer,
    pub alignment: usize,
}
impl BlurUniforms {
    pub fn new(wgpu_ctx: &WgpuCtx, levels: &[MipLevel], offset: f32) -> Self {
        let alignment = wgpu_ctx.device.limits().min_uniform_buffer_offset_alignment as usize;
        let mut blur_params = Vec::new();
        for level in &levels[1..] {
            let params = BlurParams {
                halfpixel: [0.5 / level.width as f32, 0.5 / level.height as f32],
                offset,
                _padding: 0,
            };
            let bytes = bytemuck::bytes_of(&params);
            blur_params.extend_from_slice(bytes);
            blur_params.resize(blur_params.len() + alignment - bytes.len(), 0);
        }
        let buffer = wgpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Parameter buffer"),
            size: blur_params.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        wgpu_ctx.queue.write_buffer(&buffer, 0, &blur_params);
        Self { buffer, alignment }
    }

    pub fn offset_for_pass(&self, pass_index: usize) -> DynamicOffset {
        (pass_index * self.alignment) as DynamicOffset
    }
}

pub struct BlurPipelines {
    pub downsample: RenderPipeline,
    pub upsample: RenderPipeline,
    pub sampler: Sampler,
    pub bind_group_layout: BindGroupLayout,
}
impl BlurPipelines {
    pub fn new(wgpu_ctx: &WgpuCtx, texture_format: TextureFormat) -> Self {
        // create texture sampler
        let sampler = wgpu_ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout =
            wgpu_ctx
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

        let render_pipeline_layout =
            wgpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let shader = wgpu_ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let setup_render_pipeline = |entry_point: &str| -> wgpu::RenderPipeline {
            wgpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Downsample Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: PipelineCompilationOptions::default(),
                    },
                    fragment: Some(FragmentState {
                        module: &shader,
                        entry_point: Some(entry_point),
                        targets: &[Some(ColorTargetState {
                            format: texture_format,
                            blend: Some(BlendState::REPLACE),
                            write_mask: ColorWrites::ALL,
                        })],
                        compilation_options: PipelineCompilationOptions::default(),
                    }),
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        polygon_mode: PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                })
        };

        Self {
            downsample: setup_render_pipeline("fs_down"),
            upsample: setup_render_pipeline("fs_up"),
            sampler,
            bind_group_layout,
        }
    }
    pub fn create_bind_group(
        &self,
        wgpu_ctx: &WgpuCtx,
        src_view: &TextureView,
        blur_params: &BlurUniforms,
    ) -> BindGroup {
        wgpu_ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &blur_params.buffer,
                        offset: 0, // base offset is 0, the dynamic offset is added to this later
                        size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<BlurParams>() as u64)
                                .unwrap(),
                        ),
                    }),
                },
            ],
        })
    }
}
