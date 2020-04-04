pub fn create_compute_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module,
            entry_point: super::shader::SHADER_ENTRY_POINT_NAME,
        },
    })
}
