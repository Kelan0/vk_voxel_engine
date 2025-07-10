use anyhow::{Result, anyhow};
use foldhash::HashMap;
use foldhash::HashSet;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::discard_rectangle::DiscardRectangleState;
use vulkano::pipeline::graphics::fragment_shading_rate::FragmentShadingRateState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::tessellation::TessellationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, PipelineCreateFlags, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::shader::spirv::ExecutionModel;

type VkGraphicsPipeline = vulkano::pipeline::graphics::GraphicsPipeline;

#[derive(Clone)]
pub struct PipelineLayoutBuilder {
    pub shader_modules: Vec<Arc<ShaderModule>>,
    pub entry_points: HashMap<u32, &'static str>,
}

impl PipelineLayoutBuilder {
    pub fn new(
        shader_modules: Vec<Arc<ShaderModule>>,
        entry_points: HashMap<u32, &'static str>,
    ) -> Self {
        PipelineLayoutBuilder {
            shader_modules,
            entry_points,
        }
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineBuilder {
    pub flags: PipelineCreateFlags,
    pub shader_modules: Vec<Arc<ShaderModule>>,
    pub entry_points: HashMap<u32, &'static str>,
    pub subpass_type: Option<PipelineSubpassType>,
    pub input_assembly_state: Option<InputAssemblyState>,
    pub tessellation_state: Option<TessellationState>,
    pub viewport_state: Option<ViewportState>,
    pub rasterization_state: Option<RasterizationState>,
    pub multisample_state: Option<MultisampleState>,
    pub depth_stencil_state: Option<DepthStencilState>,
    pub color_blend_state: Option<ColorBlendState>,
    pub dynamic_states: Vec<DynamicState>,
    pub base_pipeline: Option<Arc<VkGraphicsPipeline>>,
    pub discard_rectangle_state: Option<DiscardRectangleState>,
    pub fragment_shading_rate_state: Option<FragmentShadingRateState>,
}

impl GraphicsPipelineBuilder {
    pub fn new(
        subpass_type: PipelineSubpassType,
        shader_modules: Vec<Arc<ShaderModule>>,
        entry_points: HashMap<u32, &'static str>,
    ) -> Self {
        GraphicsPipelineBuilder {
            flags: PipelineCreateFlags::empty(),
            subpass_type: Some(subpass_type),
            shader_modules,
            entry_points,
            input_assembly_state: None,
            tessellation_state: None,
            viewport_state: None,
            rasterization_state: None,
            multisample_state: None,
            depth_stencil_state: None,
            color_blend_state: None,
            dynamic_states: vec![],
            base_pipeline: None,
            discard_rectangle_state: None,
            fragment_shading_rate_state: None,
        }
    }

    pub fn new_derive(
        graphics_pipeline: Arc<VkGraphicsPipeline>,
        shader_modules: Vec<Arc<ShaderModule>>,
        entry_points: HashMap<u32, &'static str>,
    ) -> Self {
        GraphicsPipelineBuilder {
            flags: graphics_pipeline.flags() | PipelineCreateFlags::DERIVATIVE,
            subpass_type: Some(graphics_pipeline.subpass().clone()),
            shader_modules,
            entry_points,
            input_assembly_state: graphics_pipeline.input_assembly_state().cloned(),
            tessellation_state: graphics_pipeline.tessellation_state().cloned(),
            viewport_state: graphics_pipeline.viewport_state().cloned(),
            rasterization_state: Some(graphics_pipeline.rasterization_state().clone()),
            multisample_state: graphics_pipeline.multisample_state().cloned(),
            depth_stencil_state: graphics_pipeline.depth_stencil_state().cloned(),
            color_blend_state: graphics_pipeline.color_blend_state().cloned(),
            dynamic_states: Vec::from_iter(graphics_pipeline.dynamic_state().iter().cloned()),
            base_pipeline: Some(graphics_pipeline.clone()),
            discard_rectangle_state: graphics_pipeline.discard_rectangle_state().cloned(),
            fragment_shading_rate_state: graphics_pipeline.fragment_shading_rate_state().cloned(),
        }
    }

    pub fn new_layout(
        shader_modules: Vec<Arc<ShaderModule>>,
        entry_points: HashMap<u32, &'static str>,
    ) -> Self {
        GraphicsPipelineBuilder {
            flags: PipelineCreateFlags::empty(),
            subpass_type: None,
            shader_modules,
            entry_points,
            input_assembly_state: None,
            tessellation_state: None,
            viewport_state: None,
            rasterization_state: None,
            multisample_state: None,
            depth_stencil_state: None,
            color_blend_state: None,
            dynamic_states: vec![],
            base_pipeline: None,
            discard_rectangle_state: None,
            fragment_shading_rate_state: None,
        }
    }

    pub fn set_flags(&mut self, flags: PipelineCreateFlags) -> &mut Self {
        self.flags = flags;
        self
    }

    pub fn add_flags(&mut self, flags: PipelineCreateFlags) -> &mut Self {
        self.flags |= flags;
        self
    }

    pub fn set_input_assembly_state(
        &mut self,
        input_assembly_state: InputAssemblyState,
    ) -> &mut Self {
        self.input_assembly_state = Some(input_assembly_state);
        self
    }

    pub fn set_tessellation_state(&mut self, tessellation_state: TessellationState) -> &mut Self {
        self.tessellation_state = Some(tessellation_state);
        self
    }

    pub fn set_viewport_state(&mut self, viewport_state: ViewportState) -> &mut Self {
        self.viewport_state = Some(viewport_state);
        self
    }

    pub fn set_rasterization_state(
        &mut self,
        rasterization_state: RasterizationState,
    ) -> &mut Self {
        self.rasterization_state = Some(rasterization_state);
        self
    }

    pub fn set_multisample_state(&mut self, multisample_state: MultisampleState) -> &mut Self {
        self.multisample_state = Some(multisample_state);
        self
    }

    pub fn set_depth_stencil_state(&mut self, depth_stencil_state: DepthStencilState) -> &mut Self {
        self.depth_stencil_state = Some(depth_stencil_state);
        self
    }

    pub fn set_color_blend_state(&mut self, color_blend_state: ColorBlendState) -> &mut Self {
        self.color_blend_state = Some(color_blend_state);
        self
    }

    pub fn set_dynamic_states(&mut self, dynamic_states: Vec<DynamicState>) -> &mut Self {
        self.dynamic_states = dynamic_states;
        self
    }

    pub fn set_base_pipeline(&mut self, base_pipeline: Arc<VkGraphicsPipeline>) -> &mut Self {
        self.base_pipeline = Some(base_pipeline);
        self
    }

    pub fn set_discard_rectangle_state(
        &mut self,
        discard_rectangle_state: DiscardRectangleState,
    ) -> &mut Self {
        self.discard_rectangle_state = Some(discard_rectangle_state);
        self
    }

    pub fn set_fragment_shading_rate_state(
        &mut self,
        fragment_shading_rate_state: FragmentShadingRateState,
    ) -> &mut Self {
        self.fragment_shading_rate_state = Some(fragment_shading_rate_state);
        self
    }

    pub fn collect_stages(&self) -> Result<SmallVec<[PipelineShaderStageCreateInfo; 5]>> {
        let mut stages: SmallVec<[PipelineShaderStageCreateInfo; 5]> = smallvec![];

        for (module_index, entry_point_name) in self.entry_points.iter() {
            let module = &self.shader_modules[*module_index as usize];
            let entry_point = module.entry_point(entry_point_name).ok_or_else(|| {
                anyhow!(
                    "Unable to find entry point \"{}\" for shader module at index {}: {:?}",
                    entry_point_name,
                    module_index,
                    module
                )
            })?;

            let stage = PipelineShaderStageCreateInfo::new(entry_point);
            stages.push(stage);
        }
        Ok(stages)
    }

    fn get_vertex_module_index(
        &self,
        stages: &SmallVec<[PipelineShaderStageCreateInfo; 5]>,
    ) -> Option<usize> {
        let mut vertex_module_index = None;

        for (idx, stage) in stages.iter().enumerate() {
            if stage.entry_point.info().execution_model == ExecutionModel::Vertex {
                vertex_module_index = Some(idx);
            }
        }

        vertex_module_index
    }

    pub fn build_pipeline<V: Vertex>(
        &self,
        device: Arc<Device>,
    ) -> Result<Arc<VkGraphicsPipeline>> {
        let stages = self.collect_stages()?;

        let create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())?;
        let pipeline_layout = PipelineLayout::new(device.clone(), create_info)?;

        let vertex_module_index = self.get_vertex_module_index(&stages);
        if vertex_module_index.is_none() {
            return Err(anyhow!("config.vertex_module_index was not a valid index"));
        }

        let vertex_module_index = vertex_module_index.unwrap();

        let vertex_stage: &PipelineShaderStageCreateInfo = &stages[vertex_module_index];
        let vertex_input_state = V::per_vertex().definition(&vertex_stage.entry_point)?;

        let dynamic_state = HashSet::from_iter(self.dynamic_states.clone());

        let mut flags = self.flags;

        let base_pipeline = self.base_pipeline.clone();

        if base_pipeline.is_some() {
            flags |= PipelineCreateFlags::DERIVATIVE;
        }

        let create_info = GraphicsPipelineCreateInfo {
            flags: self.flags,
            stages,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: self.input_assembly_state.clone(),
            tessellation_state: self.tessellation_state.clone(),
            viewport_state: self.viewport_state.clone(),
            rasterization_state: self.rasterization_state.clone(),
            multisample_state: self.multisample_state.clone(),
            depth_stencil_state: self.depth_stencil_state.clone(),
            color_blend_state: self.color_blend_state.clone(),
            dynamic_state,
            subpass: self.subpass_type.clone(),
            base_pipeline,
            discard_rectangle_state: self.discard_rectangle_state.clone(),
            fragment_shading_rate_state: self.fragment_shading_rate_state.clone(),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        let pipeline = VkGraphicsPipeline::new(device.clone(), None, create_info)?;
        Ok(pipeline)
    }
}
