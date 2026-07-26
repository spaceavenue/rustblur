use wgpu::{
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use crate::wgpu_ctx::WgpuCtx;

// high level abstraction over one level of a mip chain. holds views and dimensions at that level.
pub struct MipLevel {
    pub view: TextureView,
    pub width: u32,
    pub height: u32,
}

// abstraction over the regular texture mip chain, holds views and dimensions for each level as
// MipLevel, which are used later.
pub struct MipChain {
    pub texture: Texture,
    pub levels: Vec<MipLevel>,
}
impl MipChain {
    pub fn new(state: &WgpuCtx, (width, height): (u32, u32), passes: usize) -> Self {
        let texture = state.device.create_texture(&TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: passes as u32 + 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING,
            label: Some("Blur mip chain"),
            view_formats: &[TextureFormat::Rgba8Unorm],
        });
        let levels = (0..=passes)
            .map(|i| {
                let view = texture.create_view(&TextureViewDescriptor {
                    base_mip_level: i as u32,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                MipLevel {
                    view,
                    width: (width >> i).max(1),
                    height: (height >> i).max(1),
                }
            })
            .collect();

        Self { texture, levels }
    }
}
