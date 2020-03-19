use std::marker::PhantomData;

pub struct UniformBuffer<Content> {
    buffer: wgpu::Buffer,
    content: PhantomData<Content>,
}

impl<Content: Copy + 'static> UniformBuffer<Content> {
    pub fn new(device: &wgpu::Device) -> UniformBuffer<Content> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<Content>() as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        UniformBuffer {
            buffer,
            content: PhantomData,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn size(&self) -> u64 {
        std::mem::size_of::<Content>() as u64
    }

    pub fn update_content(&self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, content: Content) {
        let buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC).fill_from_slice(&[content]);
        encoder.copy_buffer_to_buffer(&buffer, 0, &self.buffer, 0, std::mem::size_of_val(&content) as u64);
    }
}
