use crate::hybrid_fluid::*;
use crate::wgpu_utils::pipelines::*;
use crate::{
    render_output::hdr_backbuffer::HdrBackbuffer,
    wgpu_utils::{
        self,
        binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder, BindGroupLayoutWithDesc},
        binding_glsl,
        shader::*,
    },
};
use std::{path::Path, rc::Rc};

struct ScreenDependentProperties {
    texture_fluid_thickness: wgpu::Texture,
    texture_fluid_depth: wgpu::Texture,
    bind_group_final_compose: wgpu::BindGroup,
    target_textures_resolution: wgpu::Extent3d,
}

struct ScreenIndependentProperties {
    pipeline_final_compose: ComputePipelineHandle,
    group_layout_final_compose: BindGroupLayoutWithDesc,
}

pub struct ScreenSpaceFluid {
    screen_independent: ScreenIndependentProperties,
    screen_dependent: ScreenDependentProperties,
}

impl ScreenSpaceFluid {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
        backbuffer: &HdrBackbuffer,
    ) -> ScreenSpaceFluid {
        let group_layout_final_compose = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image2d(HdrBackbuffer::FORMAT, false))
            .create(device, "BindGroupLayout: SSFluid, Final Compose");

        let pipeline_final_compose = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        &per_frame_bind_group_layout,
                        &fluid_renderer_group_layout,
                        &group_layout_final_compose.layout,
                    ],
                })),
                Path::new("screenspace_fluid/final_compose.comp"),
            ),
        );

        let screen_independent = ScreenIndependentProperties {
            group_layout_final_compose,
            pipeline_final_compose,
        };

        let screen_dependent = Self::create_screen_dependent_properties(&screen_independent, device, backbuffer);

        ScreenSpaceFluid {
            screen_dependent,
            screen_independent,
        }
    }

    fn create_screen_dependent_properties(
        screen_independent: &ScreenIndependentProperties,
        device: &wgpu::Device,
        backbuffer: &HdrBackbuffer,
    ) -> ScreenDependentProperties {
        let target_textures_resolution = wgpu::Extent3d {
            width: backbuffer.resolution().width,
            height: backbuffer.resolution().height,
            depth: 1,
        };
        let texture_fluid_thickness = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Fluid Thickness"),
            size: target_textures_resolution,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float, // R8UNorm might do as well?
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        });
        let texture_fluid_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Fluid Depth"),
            size: target_textures_resolution,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        });

        let bind_group_final_compose = BindGroupBuilder::new(&screen_independent.group_layout_final_compose)
            .texture(&backbuffer.texture_view())
            .create(device, "BindGroup: SSFluid, Final Compose");

        ScreenDependentProperties {
            texture_fluid_thickness,
            texture_fluid_depth,
            target_textures_resolution,
            bind_group_final_compose,
        }
    }

    pub fn on_window_resize(&mut self, device: &wgpu::Device, backbuffer: &HdrBackbuffer) {
        self.screen_dependent = Self::create_screen_dependent_properties(&self.screen_independent, device, backbuffer);
    }

    pub fn draw<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &'a PipelineManager,
        per_frame_bind_group: &wgpu::BindGroup,
        fluid: &HybridFluid,
    ) {
        const COMPUTE_LOCAL_SIZE_FINAL_COMPOSE: wgpu::Extent3d = wgpu::Extent3d {
            width: 32,
            height: 32,
            depth: 1,
        };
        let compose_work_group = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, COMPUTE_LOCAL_SIZE_FINAL_COMPOSE);

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_bind_group(0, &per_frame_bind_group, &[]);
            cpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
            cpass.set_bind_group(2, &self.screen_dependent.bind_group_final_compose, &[]);

            cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_final_compose));
            cpass.dispatch(compose_work_group.width, compose_work_group.height, compose_work_group.depth);
        }
    }
}
