use crate::application::window::SdlWindow;
use crate::core::event::EventBus;
use crate::core::renderer::command_buffer::CommandBuffer;
use crate::core::util::util::get_raw_bytes;
use crate::core::{AshCommandBuffer, CommandBufferImpl};
use crate::{log_error_and_anyhow, log_error_and_throw};
use anyhow::{anyhow, Result};
use ash::{khr, vk};
use log::{debug, error, info, warn};
use shaderc::{CompilationArtifact, CompileOptions, Compiler, ShaderKind};
use smallvec::smallvec;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::Read;
use std::slice;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceOwned, DeviceOwnedVulkanObject, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageType, ImageUsage, SampleCount};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryAllocator, MemoryTypeFilter};
use vulkano::memory::MemoryHeapFlags;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::query::{QueryControlFlags, QueryPipelineStatisticFlags, QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, SubpassDependency, SubpassDescription};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::{ColorSpace, CompositeAlpha, PresentFuture, PresentMode, Surface, SurfaceCapabilities, SurfaceInfo, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo};
use vulkano::sync::fence::{Fence, FenceCreateFlags, FenceCreateInfo};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::semaphore::{Semaphore, SemaphoreCreateInfo};
use vulkano::sync::{AccessFlags, GpuFuture, PipelineStage, PipelineStages, Sharing};
use vulkano::{sync, DeviceSize, VulkanLibrary, VulkanObject};

const ENABLE_VALIDATION_LAYERS: bool = true;

// pub type CommandBuffer<L> = AutoCommandBufferBuilder<L>;
// pub type PrimaryCommandBuffer = CommandBuffer<PrimaryAutoCommandBuffer>;
// pub type SecondaryCommandBuffer = CommandBuffer<SecondaryAutoCommandBuffer>;

pub type StandardMemoryAllocator = GenericMemoryAllocator<FreeListAllocator>;

type QueueMap = HashMap<&'static str, Arc<Queue>>;

// #[derive(Debug)]
pub struct GraphicsManager {
    event_bus: EventBus,
    instance: Arc<Instance>,
    ash_instance: ash::Instance,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    ash_device: ash::Device,
    swapchain_loader: khr::swapchain::Device,
    queue_details: QueueDetails,
    queues: QueueMap,
    command_pools: HashMap<u32, vk::CommandPool>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipeline_stats_query_pool: Arc<QueryPool>,
    timestamp_query_pool: Arc<QueryPool>,
    debug_messenger: Option<DebugUtilsMessenger>,
    shader_compiler: Arc<Compiler>,
    preferred_present_mode: PresentMode,
    present_mode: PresentMode,
    color_format: Format,
    color_space: ColorSpace,
    depth_format: Format,
    surface_capabilities: Option<SurfaceCapabilities>,
    update_swapchain_request: Option<UpdateSwapchainRequest>,
    direct_image_present_enabled: bool,
    swapchain_image_sampled: bool,
    swapchain_info: SwapchainInfo,
    render_pass: Option<Arc<RenderPass>>,
    max_concurrent_frames: usize,
    current_frame_index: usize,
    state: State,
    debug_pipeline_statistics_enabled: bool,
    debug_pipeline_statistics: Option<DebugPipelineStatistics>,
    current_timestamp_query_index: u32,
    timestamp_query_results: Vec<u64>,
    debug_time_blocked: f64,
}

#[derive(Default, Debug)]
struct UpdateSwapchainRequest {
    image_extent: [u32; 2],
}

// This type is annoying...
// type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<SwapchainAcquireFuture>>>;
// type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<FenceSignalFuture<SwapchainAcquireFuture>>>>;
// type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<SwapchainAcquireFuture, Box<dyn GpuFuture>>>>>;
// type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<FenceSignalFuture<SwapchainAcquireFuture>, Box<dyn GpuFuture>>>>>;
// type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<Box<dyn GpuFuture>>>>;
type InFlightFrameFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>;

pub struct SwapchainInfo {
    swapchain: Option<Arc<Swapchain>>,
    images: Vec<Arc<Image>>,
    image_views: Vec<(Arc<ImageView>, Arc<ImageView>)>,
    framebuffers: Vec<Arc<Framebuffer>>,
    // command_buffers: Vec<Arc<CommandBuffer>>,
    command_buffers: Vec<CommandBuffer>,
    acquire_future: Option<SwapchainAcquireFuture>,
    // in_flight_frames: Vec<Box<dyn GpuFuture>>,
    in_flight_frames: Vec<Option<Arc<InFlightFrameFuture>>>,
    // acquire_futures: Vec<Option<SwapchainAcquireFuture>>,
    current_image_idx: u32,
    prev_image_idx: u32,
    image_extent: [u32; 2],
    buffer_mode: SwapchainBufferMode,


    image_available_semaphores: Vec<Arc<Semaphore>>,
    render_finished_semaphores: Vec<Arc<Semaphore>>,
    frame_complete_fences: Vec<Arc<Fence>>,
}

impl SwapchainInfo {
    fn new(_device: &Arc<Device>) -> Self {
        SwapchainInfo{
            swapchain: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            framebuffers: Default::default(),
            command_buffers: Default::default(),
            acquire_future: Default::default(),
            // acquire_futures: Default::default(),
            in_flight_frames: Default::default(),
            current_image_idx: 0,
            prev_image_idx: 0,
            image_extent: Default::default(),
            buffer_mode: Default::default(),

            image_available_semaphores: Default::default(),
            render_finished_semaphores: Default::default(),
            frame_complete_fences: Default::default(),
        }
    }
}

#[derive(Default, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum SwapchainBufferMode {
    SingleBuffer,
    DoubleBuffer,
    #[default]
    TripleBuffer,
}




// TODO: this should be replaced with some kind of message/event system
// e.g. recreate_swapchain will signal a RecreateSwapchain event to all who care
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct State {
    first_frame: bool,
    swapchain_recreated: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for State {
    fn default() -> Self {
        State{
            first_frame: false,
            swapchain_recreated: false,
        }
    }
}

impl State {
    pub fn first_frame(&self) -> bool {
        self.first_frame
    }

    pub fn swapchain_recreated(&self) -> bool {
        self.swapchain_recreated
    }
}


#[derive(Clone, Copy, PartialEq)]
pub struct DebugPipelineStatistics {
    gpu_time: f64,
    input_assembly_primitives: u64,
    vertex_shader_invocations: u64,
    fragment_shader_invocations: u64,
    draw_commands: u64,
}

impl Debug for DebugPipelineStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU Time: {:.3} msec, Draw commands: {}, Input assembly primitives: {}, Vertex shader invocations: {}, Fragment shader invocations: {}", self.gpu_time, self.draw_commands, self.input_assembly_primitives, self.vertex_shader_invocations, self.fragment_shader_invocations)
    }
}



#[derive(Debug)]
pub struct QueueLayoutDefinition {
    name: &'static str,
    required_queues: QueueFlags,
    requires_present: bool,
    group: u32,
    priority: f32,
}

impl QueueLayoutDefinition {
    fn new(name: &'static str) -> Self {
        QueueLayoutDefinition{
            name,
            required_queues: QueueFlags::empty(),
            requires_present: false,
            group: 0,
            priority: 0.0,
        }
    }

    fn requires_queue(mut self, queue: QueueFlags) -> Self {
        self.required_queues |= queue;
        self
    }

    fn requires_present(mut self) -> Self {
        self.requires_present = true;
        self
    }

    fn group(mut self, group: u32) -> Self {
        self.group = group;
        self
    }

    fn priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }
}



#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
pub enum QueueId {
    GraphicsMain,
    ComputeMain,
    TransferMain,
    GraphicsTransferMain,
}

impl QueueId {
    pub fn name(&self) -> &'static str {
        match self {
            QueueId::GraphicsMain => "graphics_main",
            QueueId::ComputeMain => "compute_main",
            QueueId::TransferMain => "transfer_main",
            QueueId::GraphicsTransferMain => "graphics_transfer_main",
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum BeginFrameResult<E = anyhow::Error> {
    Begin(CommandBuffer),
    Skip,
    Err(E)
}

#[derive(Debug, Clone, Copy)]
pub struct RecreateSwapchainEvent {
    pub old_extent: [u32; 2],
    pub new_extent: [u32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct CleanupFrameResourcesEvent {
    pub frame_index: usize,
}



impl GraphicsManager {
    pub fn new(sdl_window: &SdlWindow) -> Result<Self> {
        info!("Initializing renderer");

        let event_bus = EventBus::new();

        info!("Initializing vulkan");
        let library = VulkanLibrary::new()
            .inspect_err(|_| error!("Error initializing VulkanLLibrary for renderer"))?;

        let instance = Self::create_instance(&library, sdl_window)
            .inspect_err(|_| error!("Error creating Vulkan instance"))?;

        let ash_instance = ash::Instance::from_parts_1_3(instance.handle(), instance.fns().v1_0.clone(), instance.fns().v1_1.clone(), instance.fns().v1_3.clone());

        let debug_messenger = if ENABLE_VALIDATION_LAYERS {
            Some(Self::create_debug_utils_messenger(&instance)?)
        } else {
            None
        };

        let surface = Self::create_surface(&instance, sdl_window)
            .inspect_err(|_| error!("Error creating surface for renderer"))?;

        let physical_device = Self::select_physical_device(&instance)
            .inspect_err(|_| error!("Error selecting physical device for renderer"))?;

        let required_queue_flags = QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER;
        let queue_details   = Self::select_queue_families(&physical_device, required_queue_flags, Some(surface.as_ref()))
            .inspect_err(|_| error!("Error selecting queue families for renderer"))?;

        let device_extensions = DeviceExtensions{
            khr_swapchain: true,
            ext_extended_dynamic_state: true,
            ext_extended_dynamic_state2: true,
            ext_extended_dynamic_state3: true,
            ext_line_rasterization: true,
            ..Default::default()
        };

        let device_features = DeviceFeatures{
            fill_mode_non_solid: true,
            shader_sampled_image_array_dynamic_indexing: true,
            shader_uniform_buffer_array_dynamic_indexing: true,
            shader_storage_image_array_dynamic_indexing: true,
            shader_storage_buffer_array_dynamic_indexing: true,
            wide_lines: true,
            pipeline_statistics_query: true,
            runtime_descriptor_array: true,
            // buffer_device_address: true,
            // smooth_lines: true,
            // bresenham_lines: true,
            ..Default::default()
        };

        let queue_layout = [
            QueueLayoutDefinition::new(QueueId::GraphicsMain.name()).requires_queue(QueueFlags::GRAPHICS).priority(1.0).requires_present(),
            QueueLayoutDefinition::new(QueueId::ComputeMain.name()).requires_queue(QueueFlags::COMPUTE).priority(0.5),
            QueueLayoutDefinition::new(QueueId::TransferMain.name()).requires_queue(QueueFlags::TRANSFER).priority(0.2),
            QueueLayoutDefinition::new(QueueId::GraphicsTransferMain.name()).requires_queue(QueueFlags::GRAPHICS | QueueFlags::TRANSFER).priority(0.5),
        ];
        let (device, queues) = Self::create_logical_device(&physical_device, device_extensions, device_features, &queue_details, &queue_layout)
            .inspect_err(|_| error!("Error creating logical device for renderer"))?;

        let ash_device = ash::Device::from_parts_1_3(device.handle(), device.fns().v1_0.clone(), device.fns().v1_1.clone(), device.fns().v1_2.clone(), device.fns().v1_3.clone());

        let swapchain_loader = khr::swapchain::Device::new(&ash_instance, &ash_device);

        let memory_allocator = Self::create_memory_allocator(&device)
            .inspect_err(|_| error!("Error creating memory allocator"))?;

        let command_pools = Self::create_command_pools(&ash_device, &queue_details)
            .inspect_err(|_| error!("Error creating command pools for renderer"))?;

        let command_buffer_allocator = Self::create_command_buffer_allocator(&device)
            .inspect_err(|_| error!("Error creating command buffer allocator"))?;

        let descriptor_set_allocator = Self::create_descriptor_set_allocator(&device)
            .inspect_err(|_| error!("Error creating command buffer allocator"))?;

        let pipeline_stats_query_pool = Self::create_query_pool_pipeline_statistics(&device)
            .inspect_err(|_| error!("Error creating query pool for PipelineStatistics"))?;

        let timestamp_query_pool = Self::create_query_pool_timestamp(&device, 256)
            .inspect_err(|_| error!("Error creating query pool for Timestamp"))?;

        info!("Creating shader compiler");
        let shader_compiler = Arc::new(Compiler::new()?);

        let direct_image_present_enabled = false;
        let swapchain_image_sampled = false;
        let (width, height) = sdl_window.size_in_pixels();

        // Initial "previous frame" - There is no previous frame on the first frame, so the GpuFuture is just now

        let swapchain_info = SwapchainInfo::new(&device);

        let mut renderer = GraphicsManager {
            event_bus,
            instance,
            ash_instance,
            surface,
            physical_device,
            device,
            ash_device,
            swapchain_loader,
            queue_details,
            queues,
            command_pools,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            pipeline_stats_query_pool,
            timestamp_query_pool,
            debug_messenger,
            shader_compiler,
            preferred_present_mode: PresentMode::Immediate,
            present_mode: PresentMode::Immediate,
            color_format: Format::UNDEFINED,
            color_space: ColorSpace::SrgbNonLinear,
            depth_format: Format::UNDEFINED,
            surface_capabilities: None,
            update_swapchain_request: None,
            direct_image_present_enabled,
            swapchain_image_sampled,
            swapchain_info,
            render_pass: None,
            max_concurrent_frames: 3,
            current_frame_index: 0,
            state: Default::default(),
            debug_pipeline_statistics_enabled: true,
            debug_pipeline_statistics: None,
            current_timestamp_query_index: 0,
            timestamp_query_results: vec![],
            debug_time_blocked: 0.0,
        };

        renderer.set_resolution(width, height)?;

        Ok(renderer)
    }

    pub fn init(&mut self) -> Result<()> {
        self.state.first_frame = true;

        // self.swapchain_info.in_flight_frames.resize_with(self.max_concurrent_frames, || {
        //     // sync::now(self.device.clone())
        //     None
        // });

        Ok(())
    }

    fn create_instance(library: &Arc<VulkanLibrary>, sdl_window: &SdlWindow) -> Result<Arc<Instance>> {
        info!("Creating Vulkan instance");

        let required_extensions = sdl_window.vulkan_instance_extensions()
            .inspect_err(|err| error!("Failed to retrieve required Vulkan instance extensions from SDL window: {err}"))?;

        let mut enabled_extensions = InstanceExtensions::from_iter(required_extensions.iter().map(|s| s.as_str()));

        info!("Required instance extensions for the supplied window: {enabled_extensions:?}", );

        if ENABLE_VALIDATION_LAYERS {
            warn!("Enabling Vulkan validation layers - This will probably heavily impact performance");
            enabled_extensions.ext_debug_report = true;
            enabled_extensions.ext_debug_utils = true;
        }

        let enabled_layers = Self::select_validation_layers(library)?;

        let create_info = InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions,
            enabled_layers,
            ..Default::default()
        };

        let instance = Instance::new(library.clone(), create_info)?;

        Ok(instance)
    }

    fn select_validation_layers(library: &Arc<VulkanLibrary>) -> Result<Vec<String>> {
        let mut required_validation_layers = vec![];

        if ENABLE_VALIDATION_LAYERS {
            required_validation_layers.push("VK_LAYER_KHRONOS_validation");
        }

        let mut available_layer_properties = library.layer_properties()
            .inspect_err(|err| error!("Unable to enumerate Vulkan instance layer properties: {err}"))?;

        let mut layer_names = vec![];

        for validation_layer_name in required_validation_layers {
            let found = available_layer_properties.any(|layer| validation_layer_name == layer.name());

            if !found {
                return Err(log_error_and_anyhow!("Required validation layer \"{validation_layer_name}\" was not found"));
            }

            layer_names.push(validation_layer_name.to_string());
        }

        Ok(layer_names)
    }

    fn create_surface(instance: &Arc<Instance>, sdl_window: &SdlWindow) -> Result<Arc<Surface>> {
        info!("Creating Vulkan SDL surface");

        let surface =
            unsafe { Surface::from_window_ref(instance.clone(), &sdl_window) }
            .inspect_err(|err| error!("Failed to construct Vulkan surface from provided SDL window handle - reason: {err}"))?;

        Ok(surface)
    }

    fn select_physical_device(instance: &Arc<Instance>) -> Result<Arc<PhysicalDevice>> {
        info!("Selecting physical device for rendering");

        let all_physical_devices: Vec<_> = instance.enumerate_physical_devices()
            .inspect_err(|err| error!("Failed to enumerate physical devices: {err}"))?
            .collect();

        if all_physical_devices.is_empty() {
            return Err(log_error_and_anyhow!("No physical devices available"));
        }

        for physical_device in &all_physical_devices {
            let device_name = &physical_device.properties().device_name;
            let suitable = Self::is_physical_device_suitable(physical_device);
            info!("Found physical device: \"{device_name}\"{}", if !suitable { " (Not suitable)" } else { "" });
        }

        let mut all_physical_devices: Vec<_> = all_physical_devices.into_iter()
            .filter(Self::is_physical_device_suitable)
            .collect();

        all_physical_devices.sort_by(Self::compare_physical_devices);

        let selected_physical_device = all_physical_devices.pop();

        let selected_physical_device = selected_physical_device.ok_or_else(|| {
            log_error_and_anyhow!("Failed to select a usable physical device")
        })?;

        info!("Selecting physical device: {}", selected_physical_device.properties().device_name);

        for family in selected_physical_device.queue_family_properties() {
            info!("Found a queue family with {:?} queue(s) for [{:?}]", family.queue_count, family.queue_flags);
        }
        Ok(selected_physical_device)
    }

    //noinspection DuplicatedCode
    // Return Greater if physical_device1 is better than physical_device2
    // Return Less if physical_device1 is worse than physical_device2
    // Return Equal otherwise
    fn compare_physical_devices(physical_device1: &Arc<PhysicalDevice>, physical_device2: &Arc<PhysicalDevice>) -> Ordering {
        let props1 = physical_device1.properties();
        let props2 = physical_device2.properties();

        if props1.device_type != props2.device_type {
            if props1.device_type == PhysicalDeviceType::DiscreteGpu {
                return Greater; // device1 is better than device2
            }
            if props2.device_type == PhysicalDeviceType::DiscreteGpu {
                return Less; // device2 is better than device1
            }
        }

        let features1 = physical_device1.supported_features();
        let features2 = physical_device2.supported_features();

        let num_features_1 = features1.into_iter().filter(|&(_feature_name, feature_enabled)|{ feature_enabled }).count();
        let num_features_2 = features2.into_iter().filter(|&(_feature_name, feature_enabled)|{ feature_enabled }).count();

        if num_features_1 > num_features_2 {
            return Greater; // device1 has more features than device2
        }
        if num_features_1 < num_features_2 {
            return Less; // device2 has more features than device1
        }

        let mem_props1 = physical_device1.memory_properties();
        let mem_props2 = physical_device2.memory_properties();

        let vram1 = mem_props1.memory_heaps.iter()
            .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<DeviceSize>();
        let vram2 = mem_props2.memory_heaps.iter()
            .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<DeviceSize>();

        if vram1 > vram2 {
            return Greater; // device1 has more memory than device2
        }
        if vram2 > vram1 {
            return Less; // device2 has more memory than device1
        }

        // TODO: compare more stuff

        Equal
    }

    fn is_physical_device_suitable(physical_device: &Arc<PhysicalDevice>) -> bool {
        let props = physical_device.properties();

        matches!(props.device_type, PhysicalDeviceType::DiscreteGpu | PhysicalDeviceType::IntegratedGpu | PhysicalDeviceType::VirtualGpu)
    }

    fn select_queue_families(physical_device: &Arc<PhysicalDevice>, required_queue_flags: QueueFlags, surface: Option<&Surface>) -> Result<QueueDetails> {

        let requires_graphics = required_queue_flags.contains(QueueFlags::GRAPHICS);
        let requires_compute = required_queue_flags.contains(QueueFlags::COMPUTE);
        let requires_transfer = required_queue_flags.contains(QueueFlags::TRANSFER);
        let requires_sparse_binding = required_queue_flags.contains(QueueFlags::SPARSE_BINDING);
        let requires_protected = required_queue_flags.contains(QueueFlags::PROTECTED);
        let requires_present = surface.is_some();

        let mut queue_details = QueueDetails::default();

        let queue_family_properties = physical_device.queue_family_properties();
        for (i, prop) in queue_family_properties.iter().enumerate() {
            let supports_graphics = prop.queue_flags.contains(QueueFlags::GRAPHICS);
            let supports_compute = prop.queue_flags.contains(QueueFlags::COMPUTE);
            let supports_transfer = prop.queue_flags.contains(QueueFlags::TRANSFER);
            let supports_sparse_binding = prop.queue_flags.contains(QueueFlags::SPARSE_BINDING);
            let supports_protected = prop.queue_flags.contains(QueueFlags::PROTECTED);
            let supports_present = if let Some(surface) = surface {
                physical_device.surface_support(i as u32, surface).unwrap_or(false)
            } else {
                false
            };

            if supports_graphics { queue_details.graphics_queue_family_index.get_or_insert(i as u32); }
            if supports_compute { queue_details.compute_queue_family_index.get_or_insert(i as u32); }
            if supports_transfer { queue_details.transfer_queue_family_index.get_or_insert(i as u32); }
            if supports_sparse_binding { queue_details.sparse_binding_queue_family_index.get_or_insert(i as u32); }
            if supports_protected { queue_details.protected_queue_family_index.get_or_insert(i as u32); }
            if supports_present { queue_details.present_queue_family_index.get_or_insert(i as u32); }
        }

        let mut missing_queue_types = QueueFlags::empty();
        if requires_graphics && queue_details.graphics_queue_family_index.is_none() { missing_queue_types |= QueueFlags::GRAPHICS; }
        if requires_compute && queue_details.compute_queue_family_index.is_none() { missing_queue_types |= QueueFlags::COMPUTE; }
        if requires_transfer && queue_details.transfer_queue_family_index.is_none() { missing_queue_types |= QueueFlags::TRANSFER; }
        if requires_sparse_binding && queue_details.sparse_binding_queue_family_index.is_none() { missing_queue_types |= QueueFlags::SPARSE_BINDING; }
        if requires_protected && queue_details.protected_queue_family_index.is_none() { missing_queue_types |= QueueFlags::PROTECTED; }
        let missing_present = requires_present && queue_details.present_queue_family_index.is_none();


        if !missing_queue_types.is_empty() {
            return Err(log_error_and_anyhow!("Failed to select queue families - Required support for queues: [{:?}], missing support for queues: [{:?}]", required_queue_flags, missing_queue_types));
        }

        if requires_present && missing_present {
            return Err(log_error_and_anyhow!("Failed to select queue families - Required support for PRESENT but no queues supported it"));
        }

        Ok(queue_details)
    }

    fn create_logical_device(physical_device: &Arc<PhysicalDevice>, enabled_extensions: DeviceExtensions, enabled_features: DeviceFeatures, queue_details: &QueueDetails, queue_layout: &[QueueLayoutDefinition])
        -> Result<(Arc<Device>, QueueMap)> {

        info!("Creating logical device for Vulkan");

        let mut queue_create_infos = vec![];
        let enabled_extensions = DeviceExtensions{ ..enabled_extensions };
        let enabled_features = DeviceFeatures { ..enabled_features };

        let queue_family_properties = physical_device.queue_family_properties();

        let unique_queue_indices = queue_details.unique_indices();

        // Convert the provided list of QueueLayoutDefinition into a HashMap
        let mut queue_layout: HashMap<&str, (&QueueLayoutDefinition, bool)> = queue_layout.iter().map(|e| (e.name, (e, false))).collect();

        let mut queue_map: HashMap<(u32, u32), &QueueLayoutDefinition> = HashMap::new();

        for queue_family_index in unique_queue_indices {
            let mut queue_priorities = vec![];

            if queue_layout.is_empty() {
                break;
            }
            let props = &queue_family_properties[queue_family_index as usize];

            let is_present = queue_details.present_queue_family_index == Some(queue_family_index);

            for (_id, (layout_def, is_removed)) in queue_layout.iter_mut() {

                if queue_priorities.len() >= props.queue_count as usize {
                    break;
                }
                if *is_removed {
                    continue;
                }

                if props.queue_flags.contains(layout_def.required_queues) && (!layout_def.requires_present || is_present) {
                    queue_map.insert((queue_family_index, queue_priorities.len() as u32), layout_def);
                    queue_priorities.push(layout_def.priority);
                    *is_removed = true;
                }
            }

            queue_layout.retain(|_, (_, is_removed)| !*is_removed );

            if queue_priorities.is_empty() {
                error!("Could not initialize the desired queue layout for queue family {queue_family_index}");
                continue;
            }

            let queue_create_info = QueueCreateInfo{
                queue_family_index,
                queues: queue_priorities.clone(),
                ..Default::default()
            };
            queue_create_infos.push(queue_create_info);
        }


        let create_info = DeviceCreateInfo{
            queue_create_infos,
            enabled_extensions,
            enabled_features,
            ..Default::default()
        };
        let (device, queues_iter) = Device::new(physical_device.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan device: {err}"))?;

        let mut queues = HashMap::new();
        for queue in queues_iter {
            let layout_def = queue_map.get(&(queue.queue_family_index(), queue.queue_index())).unwrap();
            queues.insert(layout_def.name, queue);
        }

        Ok((device, queues))
    }

    fn create_command_pools(device: &ash::Device, queue_details: &QueueDetails) -> Result<HashMap<u32, vk::CommandPool>> {

        let mut command_pools = HashMap::new();

        let queue_family_index = queue_details.graphics_queue_family_index.unwrap();

        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let command_pool = unsafe { device.create_command_pool(&create_info, None) }?;

        command_pools.insert(queue_family_index, command_pool);

        // for queue_family_index in queue_details.unique_indices() {
        //     let command_pool = CommandPoolConfiguration::new()
        //         .set_queue_family_index(queue_details.graphics_queue_family_index.unwrap())
        //         .set_transient(false)
        //         .set_reset_command_buffer(true)
        //         .build(device.clone())
        //         .inspect_err(|err| error!("Failed to create CommandPool: {err}"))?;
        //
        //     command_pools.insert(queue_family_index, command_pool);
        // }

        Ok(command_pools)
    }

    fn create_memory_allocator(device: &Arc<Device>) -> Result<Arc<StandardMemoryAllocator>> {
        info!("Creating memory allocator for Vulkan");
        // let create_info = GenericMemoryAllocatorCreateInfo{
        //     ..Default::default()
        // };

        let allocator = StandardMemoryAllocator::new_default(device.clone());
        Ok(Arc::new(allocator))
    }

    fn create_command_buffer_allocator(device: &Arc<Device>) -> Result<Arc<StandardCommandBufferAllocator>> {
        info!("Creating command buffer allocator for Vulkan");
        let create_info = StandardCommandBufferAllocatorCreateInfo{
            primary_buffer_count: 32,
            secondary_buffer_count: 8,
            ..Default::default()
        };

        let allocator = StandardCommandBufferAllocator::new(device.clone(), create_info);
        Ok(Arc::new(allocator))

        // CommandBufferAllocator::new(device.clone(), 32, 0)
    }

    fn create_descriptor_set_allocator(device: &Arc<Device>) -> Result<Arc<StandardDescriptorSetAllocator>> {
        info!("Creating descriptor set allocator for Vulkan");
        let create_info = StandardDescriptorSetAllocatorCreateInfo{
            set_count: 32,
            update_after_bind: false,
            ..Default::default()
        };

        let allocator = StandardDescriptorSetAllocator::new(device.clone(), create_info);
        Ok(Arc::new(allocator))

        // CommandBufferAllocator::new(device.clone(), 32, 0)
    }

    fn create_query_pool_pipeline_statistics(device: &Arc<Device>) -> Result<Arc<QueryPool>> {

        let create_info = QueryPoolCreateInfo{
            query_count: 8,
            pipeline_statistics: QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES | QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS | QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS,
            ..QueryPoolCreateInfo::query_type(QueryType::PipelineStatistics)
        };

        let query_pool = QueryPool::new(device.clone(), create_info)?;

        Ok(query_pool)
    }

    fn create_query_pool_timestamp(device: &Arc<Device>, query_count: u32) -> Result<Arc<QueryPool>> {

        let create_info = QueryPoolCreateInfo{
            query_count,
            ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        };

        let query_pool = QueryPool::new(device.clone(), create_info)?;

        Ok(query_pool)
    }

    fn create_debug_utils_messenger(instance: &Arc<Instance>) -> Result<DebugUtilsMessenger> {
        let mut create_info = DebugUtilsMessengerCreateInfo::user_callback(unsafe {
            DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                if message_severity.contains(DebugUtilsMessageSeverity::ERROR) {
                    error!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::WARNING) {
                    warn!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::INFO) {
                    info!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::VERBOSE) {
                    debug!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
            })
        });

        create_info.message_severity = DebugUtilsMessageSeverity::INFO | DebugUtilsMessageSeverity::WARNING | DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::VERBOSE;
        create_info.message_type = DebugUtilsMessageType::GENERAL | DebugUtilsMessageType::VALIDATION | DebugUtilsMessageType::PERFORMANCE;

        let messenger = DebugUtilsMessenger::new(instance.clone(), create_info)?;
        Ok(messenger)
    }

    fn create_render_pass(&mut self) -> Result<()> {
        info!("Creating render pass");

        let color_attachment = AttachmentDescription{
            format: self.color_format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::PresentSrc,
            ..Default::default()
        };

        let depth_attachment = AttachmentDescription{
            format: self.depth_format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..Default::default()
        };

        let subpass_description = SubpassDescription{
            color_attachments: vec![ Some(AttachmentReference{attachment: 0, layout: ImageLayout::ColorAttachmentOptimal, ..Default::default()}) ],
            depth_stencil_attachment: Some(AttachmentReference{attachment: 1, layout: ImageLayout::DepthStencilAttachmentOptimal, ..Default::default()}),
            ..Default::default()
        };

        let subpass_dependency_colour = SubpassDependency{
            src_subpass: None,
            dst_subpass: Some(0),
            src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let subpass_dependency_depth = SubpassDependency{
            src_subpass: None,
            dst_subpass: Some(0),
            src_stages: PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS,
            dst_stages: PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS,
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let create_info = RenderPassCreateInfo{
            attachments: vec![color_attachment, depth_attachment],
            subpasses: vec![subpass_description],
            dependencies: vec![subpass_dependency_colour, subpass_dependency_depth],
            ..Default::default()
        };

        let render_pass = RenderPass::new(self.device.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan RenderPass: {err}"))?;

        self.render_pass = Some(render_pass);

        Ok(())
    }

    fn get_surface_capabilities(physical_device: &PhysicalDevice, surface: &Surface, surface_info: SurfaceInfo) -> Result<SurfaceCapabilities> {

        let surface_capabilities = physical_device.surface_capabilities(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get surface capabilities for the Vulkan physical device: {err}"))?;

        Ok(surface_capabilities)
    }

    fn init_surface_details(&mut self, width: u32, height: u32) -> Result<bool> {
        info!("Reinitializing renderer surface for resolution [{width} x {height}]");

        let surface = self.surface.as_ref();
        let surface_info = SurfaceInfo::default();

        let surface_capabilities = Self::get_surface_capabilities(&self.physical_device, &self.surface, surface_info.clone())?;

        self.swapchain_info.image_extent[0] = u32::clamp(width, surface_capabilities.min_image_extent[0], surface_capabilities.max_image_extent[0]);
        self.swapchain_info.image_extent[1] = u32::clamp(height, surface_capabilities.min_image_extent[1], surface_capabilities.max_image_extent[1]);

        self.surface_capabilities = Some(surface_capabilities);

        let formats = self.physical_device.surface_formats(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get surface formats for the Vulkan physical device: {err}"))?;

        if formats.is_empty() {
            error!("Current Vulkan device supports no surface formats for this surface");
            return Ok(false);
        }

        let use_srgb = false;

        let mut selected_format_idx = usize::MAX;

        for (i, (format, color_space)) in formats.iter().enumerate() {
            if use_srgb {
                if *format == Format::B8G8R8A8_SRGB && *color_space == ColorSpace::SrgbNonLinear {
                    selected_format_idx = i;
                }
            } else if *format == Format::B8G8R8A8_UNORM && *color_space == ColorSpace::SrgbNonLinear {
                selected_format_idx = i;
            }
        }
        if selected_format_idx == usize::MAX {
            warn!("Preferred surface format and colour space was not found. Defaulting to first available option");
            selected_format_idx = 0;
        }

        (self.color_format, self.color_space) = formats[selected_format_idx];

        info!("Using output render colour format {:?} with colour space {:?}", self.color_format, self.color_space);

        let present_modes = self.physical_device.surface_present_modes(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get present modes for Vulkan physical device: {err}"))?;

        if present_modes.is_empty() {
            error!("Current Vulkan device supports no present modes for this surface");
            return Ok(false);
        }

        let present_mode;

        if let Some(val) = present_modes.iter().find(|val| **val == self.preferred_present_mode) {
            present_mode = val;
        } else {
            present_mode = &present_modes[0];
            warn!("Preferred surface present mode {:?} was not found. Defaulting to {:?}", self.preferred_present_mode, present_mode);
        }

        self.present_mode = *present_mode;
        self.depth_format = Format::D32_SFLOAT;

        Ok(true)
    }

    fn create_swapchain_image_views(&mut self) -> Result<()> {
        self.swapchain_info.image_views = vec![];

        for (i, image) in self.swapchain_info.images.iter().enumerate() {
            set_vulkan_debug_name(&image, Some(format!("GraphicsManager-Swapchain-ColourImage({i})").as_str()))?;

            let image_view_colour = ImageView::new_default(image.clone())?;
            set_vulkan_debug_name(&image_view_colour, Some(format!("GraphicsManager-Swapchain-ColourImageView({i})").as_str()))?;
            
            let depth_image_create_info = ImageCreateInfo{
                image_type: ImageType::Dim2d,
                format: self.depth_format,
                extent: [self.swapchain_info.image_extent[0], self.swapchain_info.image_extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            };
            
            let allocation_info = AllocationCreateInfo{
                ..Default::default()
            };
            
            let depth_image = Image::new(self.memory_allocator.clone(), depth_image_create_info, allocation_info)?;
            set_vulkan_debug_name(&depth_image, Some(format!("GraphicsManager-Swapchain-DepthImage({i})").as_str()))?;
            
            let depth_image_view_create_info = ImageViewCreateInfo{
                view_type: ImageViewType::Dim2d,
                format: self.depth_format,
                subresource_range: ImageSubresourceRange::from_parameters(self.depth_format, 1, 1),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            };
            
            let image_view_depth = ImageView::new(depth_image.clone(), depth_image_view_create_info)?;
            set_vulkan_debug_name(&image_view_depth, Some(format!("GraphicsManager-Swapchain-DepthImageView({i})").as_str()))?;

            self.swapchain_info.image_views.push((image_view_colour, image_view_depth));
        }

        Ok(())
    }

    fn create_swapchain_framebuffers(&mut self) -> Result<()> {
        let render_pass = self.render_pass();

        self.swapchain_info.framebuffers.clear();

        for (i, (colour_image_view, depth_image_view)) in self.swapchain_info.image_views.iter().enumerate() {

            let create_info = FramebufferCreateInfo{
                attachments: vec![colour_image_view.clone(), depth_image_view.clone()],
                extent: self.swapchain_info.image_extent,
                layers: 1,
                ..Default::default()
            };

            let framebuffer = Framebuffer::new(render_pass.clone(), create_info)?;
            set_vulkan_debug_name(&framebuffer, Some(format!("GraphicsManager-Swapchain-Framebuffer({i})",).as_str()))?;

            self.swapchain_info.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> Result<()> {
        info!("Recreating Vulkan swapchain...");

        let update_swapchain_request = self.update_swapchain_request.take().unwrap();

        self.flush_rendering()?;

        let old_extent = self.swapchain_info.image_extent;

        let success = self.init_surface_details(update_swapchain_request.image_extent[0], update_swapchain_request.image_extent[1])?;
        if !success {
            error!("Error when initializing surface details. Unable to recreate swapchain");
            return Ok(());
        }

        self.create_render_pass()
            .inspect_err(|_| error!("Failed to create RenderPass"))?;

        // self.swapchain_info.acquire_futures.clear();
        self.swapchain_info.command_buffers.clear();

        self.swapchain_info.acquire_future = None;
        self.swapchain_info.in_flight_frames.clear();

        self.swapchain_info.image_available_semaphores.clear();
        self.swapchain_info.render_finished_semaphores.clear();
        self.swapchain_info.frame_complete_fences.clear();

        // self.swapchain_info.in_flight_frames.resize_with(self.max_concurrent_frames, || {
        //     // sync::now(self.device.clone())
        //     None
        // });

        // self.swapchain_info.command_buffers.clear();
        // for i in 0..self.max_concurrent_frames {
        //     let cmd_pool = self.get_graphics_command_pool_mut();
        //     _ = cmd_pool.free_command_buffer_if_exists(&format!("swapchain_cmd{i}"))?;
        // }

        if self.swapchain_info.swapchain.is_some() {
            let swapchain = self.swapchain_info.swapchain.as_ref().unwrap();
            let ref_count = Arc::strong_count(swapchain);
            if ref_count > 1 {
                warn!("Swapchain has {} strong references, it may fail to be recreated. here goes...", ref_count - 1);
            }
        }

        self.swapchain_info.framebuffers.clear();
        self.swapchain_info.image_views.clear();
        self.swapchain_info.images.clear();
        self.swapchain_info.swapchain = None;

        let queue_details = &self.queue_details;
        let surface_capabilities = self.surface_capabilities.as_ref().unwrap();

        let mut image_count: u32 = match self.swapchain_info.buffer_mode {
            SwapchainBufferMode::SingleBuffer => 1,
            SwapchainBufferMode::DoubleBuffer => 2,
            SwapchainBufferMode::TripleBuffer => 3,
        };

        image_count = u32::max(image_count, surface_capabilities.min_image_count);
        if let Some(max_image_count) = surface_capabilities.max_image_count {
            image_count = u32::min(image_count, max_image_count);
        }

        info!("Initializing swapchain with {} buffers", image_count);

        let image_array_layers: u32 = 1;

        let mut image_usage = ImageUsage::COLOR_ATTACHMENT;
        if self.direct_image_present_enabled {
            image_usage |= ImageUsage::TRANSFER_DST;
        }
        if self.swapchain_image_sampled {
            image_usage |= ImageUsage::SAMPLED;
        }

        let image_sharing =
            if queue_details.graphics_queue_family_index != queue_details.present_queue_family_index {
                warn!("The GRAPHICS queue family is not the same as the PRESENT queue family - This may affect performance");
                let queues = smallvec![queue_details.graphics_queue_family_index.unwrap(), queue_details.present_queue_family_index.unwrap()];
                Sharing::Concurrent(queues)
            } else {
                Sharing::Exclusive
            };

        let create_info = SwapchainCreateInfo{
            min_image_count: image_count,
            image_format: self.color_format,
            image_color_space: self.color_space,
            image_extent: self.swapchain_info.image_extent,
            image_array_layers,
            image_usage,
            image_sharing,

            pre_transform: surface_capabilities.current_transform,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: self.present_mode,

            ..Default::default()
        };

        let (swapchain, images) = Swapchain::new(self.device.clone(), self.surface.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan swapchain: {err}"))?;

        set_vulkan_debug_name(&swapchain, Some("GraphicsManager-Swapchain"))?;

        self.swapchain_info.in_flight_frames.resize_with(images.len(), || {
            // sync::now(self.device.clone())
            None
        });

        let semaphore_create_info = SemaphoreCreateInfo::default();
        let fence_create_info = FenceCreateInfo{
            flags: FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        for _ in 0..self.max_concurrent_frames() {
            self.swapchain_info.command_buffers.push(CommandBuffer::new(self.ash_device.clone(), self.get_graphics_command_pool().clone(), vk::CommandBufferLevel::PRIMARY));
            self.swapchain_info.image_available_semaphores.push(Arc::new(Semaphore::new(self.device.clone(), semaphore_create_info.clone())?));
            self.swapchain_info.render_finished_semaphores.push(Arc::new(Semaphore::new(self.device.clone(), semaphore_create_info.clone())?));
            self.swapchain_info.frame_complete_fences.push(Arc::new(Fence::new(self.device.clone(), fence_create_info.clone())?));
        }


        self.swapchain_info.swapchain = Some(swapchain);
        self.swapchain_info.images = images;
        self.state.swapchain_recreated = true;

        // self.swapchain_info.acquire_futures.resize_with(self.swapchain_info.images.len(), || None);

        self.create_swapchain_image_views()
            .inspect_err(|err| error!("Failed to create ImageViews for the swapchain: {err}"))?;

        self.create_swapchain_framebuffers()
            .inspect_err(|err| error!("Failed to create Framebuffers for the swapchain: {err}"))?;

        // self.swapchain_info.command_buffers.reserve(self.max_concurrent_frames as usize);
        //
        // for i in 0..self.max_concurrent_frames {
        //     let alloc_config = CommandBufferAllocConfiguration::new(&format!("swapchain_cmd{i}"));
        //
        //     let cmd_buf = self.get_graphics_command_pool_mut().allocate_command_buffer(CommandBufferLevel::Primary, alloc_config)
        //         .inspect_err(|err| error!("Failed to allocate command buffer {i} for swapchain: {err}"))?;
        //
        //     self.swapchain_info.command_buffers.insert(i as usize, cmd_buf.clone());
        // }

        info!("Created swapchain for resolution [{} x {}]", self.resolution_width(), self.resolution_height());


        let new_extent = self.swapchain_info.image_extent;

        self.event_bus.emit(RecreateSwapchainEvent{
            old_extent,
            new_extent,
        });

        Ok(())
    }

    pub fn debug_print_ref_counts(&self) {

        let swapchain_ref_count = self.swapchain_info.swapchain.as_ref().map_or(0, |val| { Arc::strong_count(val) });
        debug!("Device has {} references - Swapchain has {} references", Arc::strong_count(&self.device), swapchain_ref_count);
    }

    fn cleanup_frame_resources(&mut self, frame_index: usize) {
        self.event_bus.emit(CleanupFrameResourcesEvent{
            frame_index,
        })
    }

    pub fn begin_frame(&mut self) -> BeginFrameResult {

        self.debug_time_blocked = 0.0;

        // Check if swapchain recreation was requested...
        if self.update_swapchain_request.is_some() {
            if let Err(err) = self.recreate_swapchain() {
                error!("Failed to recreate Swapchain");
                return BeginFrameResult::Err(err);
            }

            return BeginFrameResult::Skip;
        }

        let swapchain = self.swapchain().expect("Failed to get swapchain").handle();

        let current_frame_index = self.current_frame_index();
        let image_available_semaphore = self.swapchain_info.image_available_semaphores[current_frame_index].handle();
        let frame_fence = self.swapchain_info.frame_complete_fences[current_frame_index].clone();

        frame_fence.wait(None).expect("Failed to wait on GPU Fence future");
        unsafe { frame_fence.reset() }.expect("Failed to reset GPU Fence future");

        self.cleanup_frame_resources(current_frame_index);

        // ==== ACQUIRE THE NEXT IMAGE ====

        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(swapchain)
            .device_mask(1)
            .timeout(u64::MAX)
            .semaphore(image_available_semaphore)
            .fence(vk::Fence::null());

        let (image_index, is_suboptimal) = match unsafe { self.swapchain_loader.acquire_next_image2(&acquire_info) } {
            Ok(result) => result,
            Err(err) => {
                return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to acquire next image for begin_frame"))
            }
        };

        let a = Instant::now();

        // Wait for the current frame future to be finished before beginning the next frame
        if let Some(future) = &mut self.swapchain_info.in_flight_frames[image_index as usize] {
            future.cleanup_finished();
            future.wait(None).expect("Failed to wait on GPU Fence future");
        }

        let b = Instant::now();
        self.debug_time_blocked += b.duration_since(a).as_secs_f64();


        if is_suboptimal {
            // Swapchain will be recreated next time
            self.request_recreate_swapchain();
        }

        // Store the acquire_next_image index for later use in present_frame
        self.swapchain_info.prev_image_idx = self.swapchain_info.current_image_idx;
        self.swapchain_info.current_image_idx = image_index;

        // USE ASH COMMAND BUFFERS
        let mut cmd_buf = self.swapchain_info.command_buffers[current_frame_index].clone();
        // let mut cmd_buf = AshCommandBuffer::new(self.ash_device.clone(), self.get_graphics_command_pool().clone(), vk::CommandBufferLevel::PRIMARY);
        if let Err(err) = cmd_buf.begin(CommandBufferUsage::MultipleSubmit) {
            return BeginFrameResult::Err(log_error_and_throw!(err, "Failed to begin CommandBuffer for begin_frame()"))
        }

        // USE VULKANO COMMAND BUFFERS
        // // Get the command buffer from the pool for this frame
        // let allocator = self.command_buffer_allocator.clone();
        // let queue_family_index = self.queue_details.graphics_queue_family_index.unwrap();
        // let cmd_buf = match AutoCommandBufferBuilder::primary(allocator.clone(), queue_family_index, CommandBufferUsage::MultipleSubmit) {
        //     Ok(r) => r,
        //     Err(err) => return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to allocate command buffer for begin_frame"))
        // };
        // let mut cmd_buf = CommandBuffer::new(VulkanoCommandBufferType::Primary(cmd_buf));

        // Begin frame stats measurement
        if let Err(err) = self.begin_frame_stats_query(&mut cmd_buf) {
            return BeginFrameResult::Err(log_error_and_throw!(err, "Failed to begin PipelineStatistics query"))
        }

        BeginFrameResult::Begin(cmd_buf)
    }

    pub fn present_frame(&mut self, mut cmd_buf: CommandBuffer) -> Result<bool> {

        self.state = Default::default();

        // End frame stats measurement
        self.end_frame_stats_query(&mut cmd_buf)
            .inspect_err(|_| error!("Failed to enf PipelineStatistics query"))?;

        // Finalize the command buffer
        // let cmd_buf = cmd_buf.build_primary()?;
        cmd_buf.end()?;


        let swapchain = self.swapchain()?.handle();
        let image_index = self.swapchain_info.current_image_idx;

        let mut queue = self.queue(QueueId::GraphicsMain.name())
            .ok_or_else(|| anyhow!("Failed to get the GRAPHICS queue \"{}\"", QueueId::GraphicsMain.name()))?;

        // let present_info = SwapchainPresentInfo::swapchain_image_index(swapchain, image_index);

        let current_frame_index = self.current_frame_index();
        let image_available_semaphore = self.swapchain_info.image_available_semaphores[current_frame_index].handle();
        let render_finished_semaphore = self.swapchain_info.render_finished_semaphores[current_frame_index].handle();
        let frame_complete_fence = self.swapchain_info.frame_complete_fences[current_frame_index].handle();

        let wait_stages = [ vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT ];

        let submit_infos = [ vk::SubmitInfo::default()
            .wait_semaphores(slice::from_ref(&image_available_semaphore))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(slice::from_ref(cmd_buf.handle()))
            .signal_semaphores(slice::from_ref(&render_finished_semaphore)) ];

        unsafe { self.ash_device.queue_submit(queue.handle(), &submit_infos, frame_complete_fence) }?;

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&render_finished_semaphore))
            .swapchains(slice::from_ref(&swapchain))
            .image_indices(slice::from_ref(&image_index));

        unsafe { self.swapchain_loader.queue_present(queue.handle(), &present_info) }?;

        self.read_frame_stats_query_results(&cmd_buf)?;

        let current_frame_index = self.swapchain_info.current_image_idx as usize;

        // Increment the ring buffer for the next
        self.current_frame_index = (self.current_frame_index + 1) % self.max_concurrent_frames;
        // // info!("begin_frame for next frame {}, image index {} -> {}", self.current_frame_index, self.swapchain_info.current_image_idx, image_index);

        Ok(true)
    }


    // pub fn begin_frame(&mut self) -> BeginFrameResult {
    //
    //     self.debug_time_blocked = 0.0;
    //
    //     // Check if swapchain recreation was requested...
    //     if self.update_swapchain_request.is_some() {
    //         if let Err(err) = self.recreate_swapchain() {
    //             error!("Failed to recreate Swapchain");
    //             return BeginFrameResult::Err(err);
    //         }
    //
    //         return BeginFrameResult::Skip;
    //     }
    //
    //     let swapchain = self.swapchain().expect("Failed to get swapchain");
    //
    //     // ==== ACQUIRE THE NEXT IMAGE ====
    //
    //     let (image_index, is_suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
    //         Ok(r) => r,
    //         Err(Validated::Error(VulkanError::OutOfDate)) => {
    //             self.request_recreate_swapchain();
    //             return BeginFrameResult::Skip;
    //         }
    //         Err(err) => return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to acquire next image for begin_frame"))
    //     };
    //
    //     let a = Instant::now();
    //
    //     // Wait for the current frame future to be finished before beginning the next frame
    //     if let Some(future) = &mut self.swapchain_info.in_flight_frames[image_index as usize] {
    //         future.cleanup_finished();
    //         future.wait(None).expect("Failed to wait on GPU Fence future");
    //     }
    //
    //     let b = Instant::now();
    //     self.debug_time_blocked += b.duration_since(a).as_secs_f64();
    //
    //
    //     if is_suboptimal {
    //         // Swapchain will be recreated next time
    //         self.request_recreate_swapchain();
    //     }
    //
    //
    //     // Store the acquire_next_image future and index for later use in present_frame
    //     self.swapchain_info.prev_image_idx = self.swapchain_info.current_image_idx;
    //     self.swapchain_info.current_image_idx = image_index;
    //     self.swapchain_info.acquire_future = Some(acquire_future);
    //     // self.swapchain_info.acquire_futures[image_index as usize] = Some(acquire_future);
    //
    //
    //     // Get the command buffer from the pool for this frame
    //     let allocator = self.command_buffer_allocator.clone();
    //     let queue_family_index = self.queue_details.graphics_queue_family_index.unwrap();
    //
    //     let cmd_buf = match AutoCommandBufferBuilder::primary(allocator.clone(), queue_family_index, CommandBufferUsage::MultipleSubmit) {
    //         Ok(r) => r,
    //         Err(err) => return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to allocate command buffer for begin_frame"))
    //     };
    //
    //     let mut cmd_buf = CommandBuffer::new(VulkanoCommandBufferType::Primary(cmd_buf));
    //
    //     // Begin frame stats measurement
    //     if let Err(err) = self.begin_frame_stats_query(&mut cmd_buf) {
    //         return BeginFrameResult::Err(log_error_and_throw!(err, "Failed to begin PipelineStatistics query"))
    //     }
    //
    //     BeginFrameResult::Begin(cmd_buf)
    // }
    //
    // pub fn present_frame(&mut self, mut cmd_buf: CommandBuffer) -> Result<bool> {
    //
    //     self.state = Default::default();
    //
    //     // End frame stats measurement
    //     self.end_frame_stats_query(&mut cmd_buf)
    //         .inspect_err(|_| error!("Failed to enf PipelineStatistics query"))?;
    //
    //     // Finalize the command buffer
    //     // cmd_buf.get().build()?;
    //     let cmd_buf = cmd_buf.build_primary()?;
    //
    //     let swapchain = self.swapchain()?.clone();
    //     let image_index = self.swapchain_info.current_image_idx;
    //     let acquire_future = self.swapchain_info.acquire_future.take().unwrap();
    //
    //     let prev_frame_index = self.swapchain_info.prev_image_idx as usize;
    //     let prev_frame_end = match self.swapchain_info.in_flight_frames[prev_frame_index].clone() {
    //         None => self.sync_now(),
    //         Some(fence) => fence.boxed()
    //     };
    //
    //     let queue = self.queue(QueueId::GraphicsMain.name())
    //         .ok_or_else(|| anyhow!("Failed to get the GRAPHICS queue \"{}\"", QueueId::GraphicsMain.name()))?;
    //
    //     let present_info = SwapchainPresentInfo::swapchain_image_index(swapchain, image_index);
    //
    //     // We join on the previous frame end future for the current frame
    //     let future = prev_frame_end
    //         .join(acquire_future)
    //         .then_execute(queue.clone(), cmd_buf)?
    //         .then_swapchain_present(queue.clone(), present_info)
    //         .then_signal_fence_and_flush();
    //
    //     let future = match future {
    //         Ok(future) => {
    //             #[allow(clippy::arc_with_non_send_sync)]
    //             Some(Arc::new(future))
    //         }
    //         Err(Validated::Error(VulkanError::OutOfDate)) => {
    //             self.request_recreate_swapchain();
    //             None
    //         }
    //         Err(err) => {
    //             error!("Failed to flush future: {err:?}");
    //             None
    //         }
    //     };
    //
    //     self.read_frame_stats_query_results()?;
    //
    //     let current_frame_index = self.swapchain_info.current_image_idx as usize;
    //
    //     // Store the current present future in the ring buffer
    //     self.swapchain_info.in_flight_frames[current_frame_index] = future;
    //
    //     // // Increment the ring buffer for the next
    //     // self.current_frame_index = (self.current_frame_index + 1) % self.max_concurrent_frames;
    //     // // info!("begin_frame for next frame {}, image index {} -> {}", self.current_frame_index, self.swapchain_info.current_image_idx, image_index);
    //
    //     Ok(true)
    // }

    pub fn flush_rendering(&mut self) -> Result<()>{
        for frame_future in self.swapchain_info.in_flight_frames.iter_mut() {
            if let Some(future) = frame_future {
                future.cleanup_finished();
                future.wait(None)?;
            }

            *frame_future = None;

            // let mut prev_frame_end = Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>;
            // mem::swap(&mut prev_frame_end, frame_future);
            // prev_frame_end.cleanup_finished();
        }

        unsafe { self.device.wait_idle() }?;

        Ok(())
    }

    fn begin_frame_stats_query(&mut self, cmd_buf: &mut CommandBuffer) -> Result<()> {

        if !self.debug_pipeline_statistics_enabled {
            // Do nothing
            return Ok(())
        }

        // Pipeline statistics
        cmd_buf.reset_query_pool(self.pipeline_stats_query_pool.clone(), 0..self.pipeline_stats_query_pool.query_count())
            .inspect_err(|_| error!("Failed to reset query pool 0..{} for PipelineStatistics", self.pipeline_stats_query_pool.query_count()))?;

        cmd_buf.begin_query(self.pipeline_stats_query_pool.clone(), 0, QueryControlFlags::empty())
            .inspect_err(|_| error!("Failed to begin query 0 for PipelineStatistics"))?;


        // Timestamp
        self.current_timestamp_query_index = 0;
        cmd_buf.reset_query_pool(self.timestamp_query_pool.clone(), 0..self.timestamp_query_pool.query_count())
            .inspect_err(|_| error!("Failed to reset query pool 0..{} for Timestamp", self.timestamp_query_pool.query_count()))?;

        if self.write_timestamp(cmd_buf, PipelineStage::TopOfPipe).is_none() {
            error!("Failed to write timestamp query for TopOfPipe (begin_frame)");
        }

        // unsafe { cmd_buf.write_timestamp(self.timestamp_query_pool.clone(), 0, PipelineStage::TopOfPipe) }
        //     .inspect_err(|_| error!("Failed to write timestamp query 0 for TopOfPipe"))?;

        Ok(())

    }

    fn end_frame_stats_query(&mut self, cmd_buf: &mut CommandBuffer) -> Result<()> {

        if !self.debug_pipeline_statistics_enabled {
            // Do nothing
            return Ok(())
        }

        // Pipeline statistics
        cmd_buf.end_query(self.pipeline_stats_query_pool.clone(), 0)
            .inspect_err(|_| error!("Failed to end query 0 for PipelineStatistics"))?;


        // Timestamp
        if self.write_timestamp(cmd_buf, PipelineStage::BottomOfPipe).is_none() {
            error!("Failed to write timestamp query for BottomOfPipe (end_frame)");
        }

        // unsafe { cmd_buf.write_timestamp(self.timestamp_query_pool.clone(), 1, PipelineStage::BottomOfPipe) }
        //     .inspect_err(|_| error!("Failed to write timestamp query 0 for BottomOfPipe (end_frame)"))?;

        Ok(())
    }

    pub fn write_timestamp(&mut self, cmd_buf: &mut CommandBuffer, stage: PipelineStage) -> Option<u32> {

        let query_index = self.current_timestamp_query_index;
        // debug_assert!(query_index < self.timestamp_query_pool.query_count());
        if query_index >= self.get_max_timestamp_query_index() {
            return None;
        }

        if cmd_buf.write_timestamp(self.timestamp_query_pool.clone(), query_index, stage)
            .inspect_err(|err| error!("Failed to write timestamp query {} for BottomOfPipe: {:?}\n{:?}", query_index, err, err.source()))
            .is_err() {
            return None;
        }

        self.current_timestamp_query_index += 1;

        Some(query_index)
    }

    pub fn get_max_timestamp_query_index(&self) -> u32 {
        self.timestamp_query_pool.query_count() - 1
    }

    fn read_frame_stats_query_results(&mut self, cmd_buf: &CommandBuffer) -> Result<()> {
        if self.debug_pipeline_statistics_enabled {
            let mut results: [u64; 4] = [0; 4];
            self.pipeline_stats_query_pool.get_results(0..1, &mut results, QueryResultFlags::WAIT)?;

            self.timestamp_query_results.resize(self.timestamp_query_pool.query_count() as usize, 0u64);

            self.timestamp_query_pool.get_results(0..self.current_timestamp_query_index, &mut self.timestamp_query_results, QueryResultFlags::WAIT)?;

            let end_idx = (self.current_timestamp_query_index - 1) as usize;

            let pipeline_stats = DebugPipelineStatistics {
                gpu_time: (self.timestamp_query_results[end_idx] - self.timestamp_query_results[0]) as f64 / 1000000.0,
                input_assembly_primitives: results[0],
                vertex_shader_invocations: results[1],
                fragment_shader_invocations: results[2],
                draw_commands: cmd_buf.debug_draw_commands() as u64,
            };

            self.debug_pipeline_statistics = Some(pipeline_stats);

        } else {
            self.debug_pipeline_statistics = None;
        }

        Ok(())
    }

    pub fn debug_pipeline_statistics(&self) -> Option<DebugPipelineStatistics> {
        self.debug_pipeline_statistics
    }

    pub fn sync_now(&self) -> Box<dyn GpuFuture> {
        sync::now(self.device.clone()).boxed()
        // Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>
    }

    fn swapchain(&self) -> Result<&Arc<Swapchain>> {
        let swapchain = self.swapchain_info.swapchain.as_ref()
            .ok_or_else(|| anyhow!("Failed to get swapchain"))?;
        Ok(swapchain)
    }

    pub fn event_bus(&mut self) -> &mut EventBus {
        &mut self.event_bus
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn memory_allocator(&self) -> Arc<StandardMemoryAllocator> {
        self.memory_allocator.clone()
    }

    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }

    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }

    pub fn shader_compiler(&self) -> Arc<Compiler> {
        self.shader_compiler.clone()
    }

    pub fn compile_spirv_from_source(&self, source: &str, file_identifier: &str, entry_point_name: &str, kind: ShaderKind, options: Option<&CompileOptions>) -> Result<CompilationArtifact> {
        let artifact = self.shader_compiler
            .compile_into_spirv(source, kind, file_identifier, entry_point_name, options)?;

        Ok(artifact)
    }

    pub fn load_shader_module_from_source(&self, source: &str, file_identifier: &str, entry_point_name: &str, kind: ShaderKind, options: Option<&CompileOptions>) -> Result<Arc<ShaderModule>> {
        let compiled_code = self.compile_spirv_from_source(source, file_identifier, entry_point_name, kind, options)?;
        let shader_code = vulkano::shader::spirv::bytes_to_words(compiled_code.as_binary_u8())?;

        let device = self.device.clone();
        let shader_create_info = ShaderModuleCreateInfo::new(&shader_code);
        let shader_module = unsafe { ShaderModule::new(device.clone(), shader_create_info) }?;
        Ok(shader_module)
    }

    pub fn load_shader_module_from_file(&self, path: &str, entry_point_name: &str, kind: ShaderKind, options: Option<&CompileOptions>) -> Result<Arc<ShaderModule>> {
        let mut file = File::open(path)?;

        let mut source = String::new();
        file.read_to_string(&mut source)?;

        self.load_shader_module_from_source(source.as_str(), path, entry_point_name, kind, options)
    }

    pub fn create_staging_subbuffer<T: BufferContents>(allocator: Arc<dyn MemoryAllocator>, len: DeviceSize) -> Result<Subbuffer<[T]>> {

        let buffer_create_info = BufferCreateInfo{
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            // memory_type_filter: MemoryTypeFilter{
            //     required_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            //     preferred_flags: MemoryPropertyFlags::empty(),
            //     not_preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_CACHED,
            // },
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let buffer = Buffer::new_slice::<T>(allocator, buffer_create_info, allocation_info, len)?;
        set_vulkan_debug_name(buffer.buffer(), Some("GraphicsManager-StagingBuffer"))?;

        // debug!("create_staging_subbuffer for {len}x \"{}\" elements each {} bytes (buffer requires {} bytes) - buffer length is {} elements ({} bytes)", type_name::<T>(), size_of::<T>(), len * size_of::<T>() as DeviceSize, buffer.len(), buffer.len() * size_of::<T>() as DeviceSize);
        // debug!("Underlying buffer is {} ({} bytes capacity, {} bytes offset, {} bytes size)", buffer.buffer().handle() as u64, buffer.buffer().size(), buffer.offset(), buffer.size());
        Ok(buffer)
    }

    pub fn create_readback_subbuffer<T: BufferContents>(allocator: Arc<dyn MemoryAllocator>, len: DeviceSize) -> Result<Subbuffer<[T]>> {

        let buffer_create_info = BufferCreateInfo{
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            // memory_type_filter: MemoryTypeFilter{
            //     required_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            //     preferred_flags: MemoryPropertyFlags::empty(),
            //     not_preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_CACHED,
            // },
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        };

        let buffer = Buffer::new_slice::<T>(allocator, buffer_create_info, allocation_info, len)?;
        set_vulkan_debug_name(buffer.buffer(), Some("GraphicsManager-ReadbackBuffer"))?;
        Ok(buffer)
    }

    pub fn upload_buffer_data_iter<T, I>(buffer: &Subbuffer<[T]>, iter: I) -> Result<()>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator{

        let mut write = buffer.write()?;

        for (o, i) in write.iter_mut().zip(iter) {
            *o = i;
        }

        Ok(())
    }

    pub fn upload_buffer_data_bytes_iter<T, I>(buffer: &Subbuffer<[u8]>, iter: I) -> Result<()>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator{

        let mut write = buffer.write()?;

        let mut idx = 0;
        for i in iter {
            let src = get_raw_bytes(&i);
            let dst = &mut write[idx .. idx+src.len()];
            dst.clone_from_slice(src);
            idx += src.len();
        }

        Ok(())
    }

    pub fn upload_buffer_data_iter_ref<'a, T, I>(buffer: &Subbuffer<[T]>, iter: I) -> Result<()>
    where
        T: BufferContents + Clone + Copy,
        I: IntoIterator<Item = &'a T>,
        I::IntoIter: ExactSizeIterator{

        let mut write = buffer.write()?;

        for (o, &i) in write.iter_mut().zip(iter) {
            *o = i;
        }

        Ok(())
    }

    pub fn upload_buffer_data_sized<T>(buffer: &Subbuffer<[T]>, data: &[T]) -> Result<()>
    where
        T: BufferContents + Sized + Clone,
    {
        let mut write = buffer.write()?;

        write[0..data.len()].clone_from_slice(data);
        // data.clone_into(&mut &*write);

        Ok(())
    }

    pub fn upload_buffer_data_unsized<T>(buffer: &Subbuffer<T>, data: &T) -> Result<()>
    where
        T: BufferContents + ?Sized,
    {
        let write = buffer.write()?;
        data.clone_into(&mut &*write);

        Ok(())
    }
    
    pub fn begin_transfer_commands(&self) -> Result<CommandBuffer>{
        // // USE VULKANO COMMAND BUFFERS
        // let queue_family_index = self.queue_details.transfer_queue_family_index.unwrap();
        // let allocator = self.command_buffer_allocator();
        // let cmd_buf = AutoCommandBufferBuilder::primary(allocator, queue_family_index, CommandBufferUsage::OneTimeSubmit)?;
        // let cmd_buf = CommandBuffer::new(VulkanoCommandBufferType::Primary(cmd_buf));

        // USE ASH COMMAND BUFFERS
        let mut cmd_buf = CommandBuffer::new(self.ash_device.clone(), self.get_transfer_command_pool().clone(), vk::CommandBufferLevel::PRIMARY);
        cmd_buf.begin(CommandBufferUsage::OneTimeSubmit)?;

        Ok(cmd_buf)
    }

    // // USE VULKANO COMMAND BUFFERS
    // pub fn submit_transfer_commands(&self, cmd_buf: VulkanoCommandBuffer) -> Result<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>> {
    //     let queue = self.transfer_queue();
    //
    //     let cmd_buf = cmd_buf.build_primary()?;
    //     let fence = cmd_buf.execute(queue)?
    //         .then_signal_fence_and_flush()?;
    //
    //     Ok(fence)
    // }

    // USE ASH COMMAND BUFFERS
    pub fn submit_transfer_commands(&self, mut cmd_buf: AshCommandBuffer) -> Result<Fence> {
        let queue = self.transfer_queue();

        cmd_buf.end()?;
        let cmd_buf = cmd_buf.handle();

        let submit_infos = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&cmd_buf));

        let fence_create_info = FenceCreateInfo::default();
        let fence = Fence::new(self.device.clone(), fence_create_info)?;

        unsafe { self.ash_device.queue_submit(queue.handle(), slice::from_ref(&submit_infos), fence.handle()) }?;

        Ok(fence)
    }
    
    pub fn transition_image_layout<L>(_cmd_buf: CommandBuffer, _image: Arc<Image>) {
        // ImageLayout::ShaderReadOnlyOptimal;
        // let barrier = ImageMemoryBarrier{
        //     
        // };
    }


    pub fn render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.as_ref().unwrap().clone()
    }

    pub fn queue_details(&self) -> &QueueDetails {
        &self.queue_details
    }

    pub fn get_command_pool(&self, queue_family_index: u32) -> Option<&vk::CommandPool> {
        self.command_pools.get(&queue_family_index)
    }

    pub fn get_command_pool_mut(&mut self, queue_family_index: u32) -> Option<&mut vk::CommandPool> {
        self.command_pools.get_mut(&queue_family_index)
    }

    pub fn get_graphics_command_pool(&self) -> &vk::CommandPool {
        // We expect to always have a graphics command pool, these unwraps are fine.
        self.get_command_pool(self.queue_details.graphics_queue_family_index.unwrap()).unwrap()
    }

    pub fn get_graphics_command_pool_mut(&mut self) -> &mut vk::CommandPool {
        self.get_command_pool_mut(self.queue_details.graphics_queue_family_index.unwrap()).unwrap()
    }

    pub fn get_transfer_command_pool(&self) -> &vk::CommandPool {
        // We expect to always have a graphics command pool, these unwraps are fine.
        self.get_command_pool(self.queue_details.transfer_queue_family_index.unwrap()).unwrap()
    }

    pub fn get_transfer_command_pool_mut(&mut self) -> &mut vk::CommandPool {
        self.get_command_pool_mut(self.queue_details.transfer_queue_family_index.unwrap()).unwrap()
    }
    
    pub fn queue(&self, queue_id: &'static str) -> Option<&Arc<Queue>> {
        self.queues.get(queue_id)
    }
    
    pub fn graphics_queue(&self) -> Arc<Queue> {
        self.queue(QueueId::GraphicsMain.name()).unwrap().clone()
    }
    pub fn transfer_queue(&self) -> Arc<Queue> {
        self.queue(QueueId::TransferMain.name()).unwrap().clone()
    }

    pub fn request_recreate_swapchain(&mut self) {
        if self.update_swapchain_request.is_none() {
            self.update_swapchain_request = Some(UpdateSwapchainRequest{
                image_extent: self.swapchain_info.image_extent,
            });
        }
    }

    pub fn set_resolution(&mut self, width: u32, height: u32) -> Result<bool> {
        if width == self.swapchain_info.image_extent[0] && height == self.swapchain_info.image_extent[1] {
            return Ok(false); // width & height is unchanged
        }

        self.request_recreate_swapchain();
        let a = self.update_swapchain_request.as_mut().unwrap();
        a.image_extent[0] = width;
        a.image_extent[1] = height;

        Ok(true)
    }

    pub fn resolution(&self) -> [u32; 2] {
        self.swapchain_info.image_extent
    }

    pub fn get_viewport(&self) -> Viewport {
        const FLIP_Y: bool = true;

        let [width, height] = self.resolution();

        let (y, height) = if FLIP_Y {
            (height as f32, -(height as f32))
        } else {
            (0.0, height as f32)
        };

        Viewport{
            offset: [0.0, y],
            extent: [width as f32, height],
            depth_range: 0.0..=1.0,
        }
    }

    pub fn resolution_width(&self) -> u32 {
        self.swapchain_info.image_extent[0]
    }

    pub fn resolution_height(&self) -> u32 {
        self.swapchain_info.image_extent[1]
    }

    pub fn color_format(&self) -> Format {
        self.color_format
    }

    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    pub fn current_framebuffer(&self) -> Arc<Framebuffer> {
        self.swapchain_info.framebuffers.get(self.swapchain_info.current_image_idx as usize).unwrap().clone()
    }

    pub fn max_concurrent_frames(&self) -> usize {
        self.max_concurrent_frames
        // self.swapchain_info.images.len()
    }

    pub fn current_frame_index(&self) -> usize {
        self.current_frame_index
        // self.swapchain_info.current_image_idx as usize
    }

    // pub fn get_current_graphics_cmd_buffer(&self) -> &Arc<CommandBuffer> {
    //     let a = self.swapchain_info.command_buffers.get(self.current_frame_index as usize)
    //         .ok_or_else(|| anyhow!("Failed to get command buffer for frame {}", self.swapchain_info.current_frame_idx));
    //
    //     a.unwrap()
    // }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn debug_time_blocked(&self) -> f64 {
        self.debug_time_blocked
    }
}




#[derive(Default, Debug)]
pub struct QueueDetails {
    pub graphics_queue_family_index: Option<u32>,
    pub compute_queue_family_index: Option<u32>,
    pub transfer_queue_family_index: Option<u32>,
    pub sparse_binding_queue_family_index: Option<u32>,
    pub protected_queue_family_index: Option<u32>,
    pub present_queue_family_index: Option<u32>,
}

impl QueueDetails {
    pub fn indices(&self) -> [Option<u32>; 6] {
        [
            self.graphics_queue_family_index,
            self.compute_queue_family_index,
            self.transfer_queue_family_index,
            self.sparse_binding_queue_family_index,
            self.protected_queue_family_index,
            self.present_queue_family_index,
        ]
    }

    pub fn unique_indices(&self) -> HashSet<u32> {
        self.indices().into_iter().flatten().collect()
    }
}



pub fn set_vulkan_debug_name<T>(object: T, object_name: Option<&str>) -> Result<()>
where T: DeviceOwned + VulkanObject {
    if ENABLE_VALIDATION_LAYERS {
        object.set_debug_utils_object_name(object_name)?;
    }
    Ok(())
}