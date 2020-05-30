use std::marker::PhantomData;

pub struct UniformBuffer<Content> {
    buffer: wgpu::Buffer,
    content: PhantomData<Content>,
}

impl<Content: Copy + 'static> UniformBuffer<Content> {
    fn name() -> &'static str {
        let type_name = std::any::type_name::<Content>();
        let pos = type_name.rfind(':').unwrap();
        &type_name[(pos + 1)..]
    }

    pub fn new(device: &wgpu::Device) -> UniformBuffer<Content> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("UniformBuffer: {}", Self::name())),
            size: std::mem::size_of::<Content>() as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        UniformBuffer {
            buffer,
            content: PhantomData,
        }
    }

    pub fn update_content(&self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, content: Content) {
        // TODO: This is a clear usecase for queue.write_buffer
        let size = std::mem::size_of_val(&content) as wgpu::BufferAddress;
        let mut mapped_buffer = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some(&format!("UniformBuffer Update: {}", Self::name())),
            size,
            usage: wgpu::BufferUsage::COPY_SRC,
        });
        unsafe {
            std::ptr::copy(
                (&content as *const Content) as *const u8,
                mapped_buffer.data().as_mut_ptr(),
                size as usize,
            );
        }
        encoder.copy_buffer_to_buffer(&mapped_buffer.finish(), 0, &self.buffer, 0, size);
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer(self.buffer.slice(..))
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct PaddedVector3 {
    vector: cgmath::Vector3<f32>,
    padding: f32,
}
impl From<cgmath::Vector3<f32>> for PaddedVector3 {
    fn from(vector: cgmath::Vector3<f32>) -> Self {
        PaddedVector3 { vector, padding: 0.0 }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct PaddedPoint3 {
    point: cgmath::Point3<f32>,
    padding: f32,
}
impl From<cgmath::Point3<f32>> for PaddedPoint3 {
    fn from(point: cgmath::Point3<f32>) -> Self {
        PaddedPoint3 { point, padding: 1.0 }
    }
}
