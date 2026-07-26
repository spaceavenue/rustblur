use wgpu::{CommandEncoder, DynamicOffset, RenderPipeline};

use crate::mip::{MipChain, MipLevel};
use crate::pipeline::{BlurPipelines, BlurUniforms};
use crate::wgpu_ctx::WgpuCtx;

pub struct BlurCtx<'a> {
    pub wgpu_ctx: &'a WgpuCtx,
    pub pipelines: &'a BlurPipelines,
    pub blur_params: &'a BlurUniforms,
}
impl<'a> BlurCtx<'a> {
    pub fn new(
        wgpu_ctx: &'a WgpuCtx,
        pipelines: &'a BlurPipelines,
        blur_params: &'a BlurUniforms,
    ) -> Self {
        Self {
            wgpu_ctx,
            pipelines,
            blur_params,
        }
    }

    pub fn execute(&self, encoder: &mut CommandEncoder, passes: usize, mip_chain: &MipChain) {
        // downsample loop
        // i -> src texture
        // (i + 1) -> dst texture
        for i in 0..passes {
            self.blur_pass(
                encoder,
                &self.pipelines.downsample,
                &mip_chain.levels[i],
                &mip_chain.levels[i + 1],
                self.blur_params.offset_for_pass(i),
            );
        }

        // upsample loop
        // (j + 1) -> src texture
        // j -> dst texture
        for j in (0..passes).rev() {
            self.blur_pass(
                encoder,
                &self.pipelines.upsample,
                &mip_chain.levels[j + 1],
                &mip_chain.levels[j],
                self.blur_params.offset_for_pass(j + 1),
            );
        }
    }

    fn blur_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &RenderPipeline,
        src_level: &MipLevel,
        dst_level: &MipLevel,
        offset: DynamicOffset,
    ) {
        let bind_group =
            self.pipelines
                .create_bind_group(self.wgpu_ctx, &src_level.view, self.blur_params);
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_level.view,
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
        render_pass.set_bind_group(0, &bind_group, &[offset]);
        render_pass.set_viewport(
            0.,
            0.,
            dst_level.width as f32,
            dst_level.height as f32,
            0.,
            0.,
        );
        render_pass.draw(0..3, 0..1);
    }
}
