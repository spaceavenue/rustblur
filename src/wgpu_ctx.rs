use std::error::Error;

use wgpu::{Device, DeviceDescriptor, Instance, PowerPreference, Queue, RequestAdapterOptions};

pub struct WgpuCtx {
    pub device: Device,
    pub queue: Queue,
}
impl WgpuCtx {
    pub async fn init() -> Result<Self, Box<dyn Error>> {
        let adapter = Instance::default()
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter.request_device(&DeviceDescriptor::default()).await?;
        Ok(Self { device, queue })
    }
}
