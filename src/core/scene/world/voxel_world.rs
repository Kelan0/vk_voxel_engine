
pub mod world {
    use crate::core::{AxisDirection, MeshDataConfig, RenderType, Scene, VoxelChunkRenderComponent, VoxelVertex};
    use anyhow::Result;
    use ash::vk::DeviceSize;
    use foldhash::{HashMap, HashSet, HashSetExt};
    use foldhash::HashMapExt;
    use glam::{DVec3, IVec3, U16Vec2, U8Vec4, UVec3, Vec3};
    use log::{debug, warn};
    use std::any::Any;
    use std::array;

    use crate::application::Ticker;
    use crate::core::scene::world::chunk_loader::ChunkLoader;
    use crate::core::{debug_mesh, util, AxisAlignedBoundingBox, BaseVertex, BoundingVolume, BoundingVolumeDebugDraw, CommandBuffer, DebugRenderContext, Engine, GraphicsManager, Mesh, MeshData, MeshPrimitiveType, RenderComponent, Transform, WorldGenerator};
    use crate::{function_name, profile_scope, profile_scope_fn};
    use std::sync::Arc;
    use std::time::Instant;
    use vulkano::buffer::Subbuffer;

    pub const CHUNK_SIZE_EXP: u32 = 5;
    pub const CHUNK_SIZE: u32 = 1 << CHUNK_SIZE_EXP;
    pub const CHUNK_BOUNDS: UVec3 = UVec3::new(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    pub const CHUNK_BLOCK_COUNT: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;


    // #[derive(Debug)]
    pub struct VoxelWorld {
        // chunks: Vec<VoxelChunk>,
        loaded_chunks: HashMap<IVec3, VoxelChunkEntity>,
        requested_chunks: HashMap<IVec3, ChunkRequest>,
        chunk_unload_queue: Vec<IVec3>,
        chunk_load_center_pos: IVec3,
        player_chunk_pos: IVec3,
        unloaded_block_edits: HashMap<IVec3, HashMap<UVec3, u32>>,
        chunk_load_radius: u32,
        max_async_chunks_per_frame: u32,
        world_generator: Arc<WorldGenerator>,
        chunk_loader: ChunkLoader,

        debug_info: DebugInfo,
    }

    // #[derive(Debug)]
    pub struct VoxelChunkData {
        // entity: Entity<'static>,
        chunk_pos: IVec3,
        blocks: [u32; CHUNK_BLOCK_COUNT],
        // block_flags: [u8; CHUNK_BLOCK_COUNT],
        dirty: bool,
        block_count: u32,
        updated_mesh_data: [Option<MeshData<VoxelVertex>>; 6],
        bounds: [Option<AxisAlignedBoundingBox>; 6],
        mesh: [Option<Arc<Mesh<VoxelVertex>>>; 6],
        unloaded_block_edits: HashMap<UVec3, u32>,
    }

    pub struct VoxelChunkEntity {
        entity_id: bevy_ecs::entity::Entity,
        // entity_dir_id: [bevy_ecs::entity::Entity; 6],
        chunk_data: Option<Box<VoxelChunkData>>,
    }

    #[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq)]
    enum ChunkRequest {
        LoadRequested,
        UnloadRequested,
        LoadPending,
        UnloadPending,
    }

    #[derive(Default, Clone, Copy, Debug)]
    struct DebugInfo {
        max_loaded_chunks_count: usize,
        max_requested_chunks_count: usize,
        max_generate_chunks_count: usize,
        max_mesh_chunks_count: usize,
        max_unload_chunks_count: usize,
        max_canceled_chunks_count: usize,
    }

    impl DebugInfo {
        pub fn reset(&mut self) {
            self.max_loaded_chunks_count = 0;
            self.max_requested_chunks_count = 0;
            self.max_generate_chunks_count = 0;
            self.max_mesh_chunks_count = 0;
            self.max_unload_chunks_count = 0;
            self.max_canceled_chunks_count = 0;
        }
    }



    impl VoxelWorld {
        pub fn new(world_generator: WorldGenerator) -> Self {

            let world_generator = Arc::new(world_generator);

            let mut chunk_loader = ChunkLoader::new(world_generator.clone());
            chunk_loader.set_thread_count(12);

            VoxelWorld {
                // chunks: vec![],
                loaded_chunks: HashMap::new(),
                requested_chunks: HashMap::new(),
                chunk_unload_queue: Vec::new(),
                chunk_load_center_pos: IVec3::ZERO,
                player_chunk_pos: IVec3::MAX,
                unloaded_block_edits: HashMap::new(),
                chunk_load_radius: 40,
                max_async_chunks_per_frame: 500,
                world_generator,
                chunk_loader,

                debug_info: Default::default()
            }
        }

        pub fn shutdown(&mut self, engine: &mut Engine) {
            // for (chunk_pos, chunk) in self.loaded_chunks.iter() {
            //     self.unload_chunk(engine, *chunk_pos);
            // }

            let mut chunk_positions = Vec::with_capacity(self.loaded_chunks.len());

            for (chunk_pos, _chunk) in self.loaded_chunks.iter() {
                chunk_positions.push(*chunk_pos);
            }

            for chunk_pos in chunk_positions {
                self.unload_chunk(engine, chunk_pos);
            }
        }

        pub fn draw_gui(&mut self, ticker: &mut Ticker, ctx: &egui::Context) {

            if ticker.time_since_last_dbg() >= ticker.debug_interval() {
                self.debug_info.reset();

                self.debug_info.max_loaded_chunks_count = usize::max(self.debug_info.max_loaded_chunks_count, self.loaded_chunks.len());
                self.debug_info.max_requested_chunks_count = usize::max(self.debug_info.max_requested_chunks_count, self.requested_chunks.len());
                self.debug_info.max_generate_chunks_count = usize::max(self.debug_info.max_generate_chunks_count, self.chunk_loader.chunk_generate_request_count());
                self.debug_info.max_mesh_chunks_count = usize::max(self.debug_info.max_mesh_chunks_count, self.chunk_loader.chunk_mesh_request_count());
                self.debug_info.max_unload_chunks_count = usize::max(self.debug_info.max_unload_chunks_count, self.chunk_unload_queue.len());
                self.debug_info.max_canceled_chunks_count = usize::max(self.debug_info.max_canceled_chunks_count, self.chunk_loader.canceled_chunks_count());
            }

            egui::Window::new("World Info")
                .anchor(egui::Align2::RIGHT_TOP, [30.0, 30.0])
                .show(ctx, |ui| {
                    let chunk_pos_str = format!("Player chunk pos: {}", self.player_chunk_pos);
                    let loaded_chunks_count_str = format!("Loaded chunks: {} ({})", self.loaded_chunks.len(), self.debug_info.max_loaded_chunks_count);
                    let requested_chunks_count_str = format!("Requested chunks: {} ({})", self.requested_chunks.len(), self.debug_info.max_requested_chunks_count);
                    let generate_chunks_count_str = format!("Generation queued: {} ({})", self.chunk_loader.chunk_generate_request_count(), self.debug_info.max_generate_chunks_count);
                    let mesh_chunks_count_str = format!("Meshing queued: {} ({})", self.chunk_loader.chunk_mesh_request_count(), self.debug_info.max_mesh_chunks_count);
                    let unload_chunks_count_str = format!("Unload queued: {} ({})", self.chunk_unload_queue.len(), self.debug_info.max_unload_chunks_count);
                    let canceled_chunks_count_str = format!("Canceled chunks: {} ({})", self.chunk_loader.canceled_chunks_count(), self.debug_info.max_canceled_chunks_count);
                    ui.label(chunk_pos_str);
                    ui.label(loaded_chunks_count_str);
                    ui.label(requested_chunks_count_str);
                    ui.label(generate_chunks_count_str);
                    ui.label(mesh_chunks_count_str);
                    ui.label(unload_chunks_count_str);
                    ui.label(canceled_chunks_count_str);
                });

        }

        pub fn update(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {

            profile_scope_fn!(&engine.frame_profiler);

            self.update_requested_chunks(engine)?;
            self.update_chunk_queues(engine)?;
            self.update_chunk_buffers(engine)?;

            self.chunk_loader.update();

            // WorldGenerator::test_all();

            // let mut count = 0;
            // let mut max_staging_size = 0;
            //
            // for (_chunk_pos, chunk) in &mut self.loaded_chunks {
            //     let chunk_data = chunk.chunk_data.as_ref().unwrap();
            //     let staging_size = chunk_data.get_staging_buffer_size();
            //     if staging_size > 0 {
            //         count += 1;
            //         max_staging_size = DeviceSize::max(max_staging_size, staging_size);
            //     }
            // }
            //
            // if max_staging_size > 0 {
            //     let t0 = Instant::now();
            //
            //     let allocator = engine.graphics.memory_allocator();
            //
            //     let mut res: Vec<Arc<dyn Any>> = vec![];
            //     let mut fences = vec![];
            //
            //     let mut stage_count = 0;
            //     let mut staging_offset = 0;
            //
            //     let mut subbuffer: Option<Subbuffer<[u8]>> = None;
            //     let mut staged_bytes = 0;
            //
            //     let mut iter = self.loaded_chunks.iter_mut();
            //     while let Some((_chunk_pos, chunk)) = iter.next() {
            //
            //         let chunk_data = chunk.chunk_data.as_ref().unwrap();
            //
            //         if chunk_data.get_staging_buffer_size() == 0 {
            //             continue;
            //         }
            //         let mut cmd_buf = engine.graphics.begin_transfer_commands()?;
            //
            //         if let Some(buf) = subbuffer.as_ref() {
            //             if buf.size() < chunk_data.get_staging_buffer_size() {
            //                 subbuffer = None;
            //             }
            //         }
            //
            //         if subbuffer.is_none() {
            //             let staging_buffer = GraphicsManager::create_staging_subbuffer(allocator.clone(), max_staging_size * 3)?;
            //             subbuffer = Some(staging_buffer.clone());
            //             res.push(staging_buffer.buffer().clone());
            //             stage_count += 1;
            //             staged_bytes += staging_buffer.size();
            //         }
            //
            //         staging_offset += chunk_data.get_staging_buffer_size();
            //
            //         chunk.update_buffers(&mut cmd_buf, &mut subbuffer, engine)?;
            //
            //         let fence = engine.graphics.submit_transfer_commands(cmd_buf)?;
            //         fences.push(fence);
            //     }
            //
            //     for fence in fences {
            //         fence.wait(None)?
            //     }
            //
            //     // let dur = t0.elapsed().as_secs_f64() * 1000.0;
            //     // debug!("Updated {count} chunks in {dur} msec - staging size: {staged_bytes} bytes in {stage_count} uploads");
            // }

            // if ticker.time_since_last_dbg() > ticker.debug_interval() {
            //     debug!("Player chunk pos: {}", self.player_chunk_pos);
            //     debug!("World has {} loaded chunks, {} requested chunks, {} generation queued, {} meshing queued, {} unloads queued", self.loaded_chunks.len(), self.requested_chunks.len(), self.chunk_loader.chunk_generate_request_count(), self.chunk_loader.chunk_mesh_request_count(), self.chunk_unload_queue.len())
            // }
            Ok(())
        }

        /// Loop over the list of chunk requests in the last frame, and insert entries into the appropriate load
        /// or unload queues. Change the request status to pending.
        /// Also drain the completed chunks from the ChunkLoader thread
        fn update_requested_chunks(&mut self, engine: &mut Engine) -> Result<()> {
            profile_scope_fn!(&engine.frame_profiler);

            let mut chunks_to_load = vec![];
            for (chunk_pos, request) in &mut self.requested_chunks {

                if *request == ChunkRequest::LoadRequested {
                    *request = ChunkRequest::LoadPending;
                    chunks_to_load.push(*chunk_pos);

                } else if *request == ChunkRequest::UnloadRequested {
                    *request = ChunkRequest::UnloadPending;
                    self.chunk_unload_queue.push(*chunk_pos);
                }
            }

            let t0 = Instant::now();
            self.chunk_loader.request_generate_chunks(chunks_to_load)?;
            let dur = t0.elapsed().as_secs_f64() * 1000.0;
            if dur > 2.0 {
                warn!("request_load_chunks blocked for {dur} msec");
            }

            let mut num_completed = 0;
            let mut num_canceled = 0;

            let t0 = Instant::now();


            self.chunk_loader.drain_canceled_chunks(usize::MAX, false, |chunk_pos| {
                self.requested_chunks.remove(&chunk_pos);
                num_canceled += 1;
            })?;

            self.chunk_loader.drain_completed_chunks(self.max_async_chunks_per_frame as usize, |chunk| {
                let chunk_pos = chunk.chunk_pos;
                self.loaded_chunks.insert(chunk_pos, VoxelChunkEntity::load(chunk, engine));
                self.requested_chunks.remove(&chunk_pos);
                num_completed += 1;
            })?;

            let dur = t0.elapsed().as_secs_f64() * 1000.0;
            if dur > 2.0 {
                let remaining_completed = self.chunk_loader.num_completed_chunks().unwrap();
                warn!("ChunkLoader drained {num_completed} of {remaining_completed} completed chunks, {num_canceled} canceled chunks - Took {dur} msec");
            }

            Ok(())
        }

        /// Update the queues of chunks to load & unload. Keep the queues sorted by distance around the player position
        /// The load queue is sorted so the nearest chunks are loaded first, and the unload queue is sorted in the
        /// opposite direction.
        /// We then submit a limited list of requests to the ChunkLoader thread to handle.
        fn update_chunk_queues(&mut self, engine: &mut Engine) -> Result<()> {
            profile_scope_fn!(&engine.frame_profiler);

            let center_pos = self.chunk_load_center_pos.as_dvec3();

            if self.chunk_load_center_pos != self.player_chunk_pos {
                self.chunk_load_center_pos = self.player_chunk_pos;

                self.chunk_loader.update_chunk_queues(self.chunk_load_center_pos, self.chunk_load_radius, Some(|chunk_pos| {
                    self.requested_chunks.remove(&chunk_pos);
                }))?;

                // Furthest chunks from the player end up at the back of the list (unloaded first)
                self.chunk_unload_queue.sort_by(|pos1, pos2| {
                    let dist1 = pos1.as_dvec3().distance_squared(center_pos);
                    let dist2 = pos2.as_dvec3().distance_squared(center_pos);
                    dist1.total_cmp(&dist2)
                })
            }

            for _ in 0..100 {
                if self.chunk_unload_queue.is_empty() {
                    break;
                }

                let chunk_pos = self.chunk_unload_queue.pop().unwrap();
                // debug!("Unloading chunk {chunk_pos}");
                if !self.is_chunk_load_requested(&chunk_pos) || chunk_pos.as_dvec3().distance_squared(center_pos) >= self.chunk_load_radius as f64 {

                    self.unload_chunk(engine, chunk_pos);
                }

                self.requested_chunks.remove(&chunk_pos);
            }

            Ok(())
        }

        pub fn update_chunk_buffers(&mut self, engine: &mut Engine) -> Result<()> {
            profile_scope_fn!(&engine.frame_profiler);

            for (chunk_pos, chunk) in &mut self.loaded_chunks {
                chunk.update_render_component(engine);
            }

            Ok(())
        }

        pub fn update_player_position(&mut self, player_pos: DVec3) {
            let chunk_pos = get_chunk_pos_for_world_pos(player_pos);

            if self.player_chunk_pos != chunk_pos {
                // debug!("Player moved chunk {} -> {}", self.player_chunk_pos, chunk_pos);
                self.player_chunk_pos = chunk_pos;

                self.load_chunks_in_radius(chunk_pos, self.chunk_load_radius as i32);
                self.unload_chunks_outside_radius(chunk_pos, self.chunk_load_radius as i32);
            }
        }

        fn load_chunks_in_radius(&mut self, center_chunk_pos: IVec3, radius: i32) {
            let r = radius as f32;
            let r2 = r * r;

            for x in -radius ..= radius {
                let fx = x as f32;// + 0.5;
                let fx2 = fx * fx;

                for y in -radius ..= radius {
                    let fy = y as f32;// + 0.5;
                    let fy2 = fy * fy;

                    for z in -radius ..= radius {
                        let fz = z as f32;// + 0.5;
                        let fz2 = fz * fz;

                        let d2 = fx2 + fy2 + fz2;

                        if d2 < r2 {
                            self.request_load_chunk(center_chunk_pos + IVec3::new(x, y, z));
                        }
                    }
                }
            }
        }

        fn unload_chunks_outside_radius(&mut self, center_chunk_pos: IVec3, radius: i32) {
            let center_pos = center_chunk_pos.as_vec3();
            let r = radius as f32;
            let r2 = r * r;

            let mut unload_chunk_positions = vec![];

            for (chunk_pos, _index) in &self.loaded_chunks {

                if self.is_chunk_unload_requested(chunk_pos) {
                    continue;
                }

                let pos = chunk_pos.as_vec3();

                let d2 = pos.distance_squared(center_pos);
                if d2 >= r2 {
                    unload_chunk_positions.push(*chunk_pos);
                }
            }

            // debug!("Unloading {} chunks", unload_chunk_positions.len());

            for chunk_pos in unload_chunk_positions {
                self.request_unload_chunk(chunk_pos);
            }
        }

        fn unload_chunk(&mut self, engine: &mut Engine, chunk_pos: IVec3) {
            if let Some(mut chunk) = self.loaded_chunks.remove(&chunk_pos) {
                // engine.scene.destroy_entity(chunk.entity_id);
                chunk.unload(engine);
                chunk.chunk_data = None;
            }
        }

        pub fn request_load_chunk(&mut self, chunk_pos: IVec3) {
            if !self.is_chunk_loaded(chunk_pos) && !self.is_chunk_load_pending(&chunk_pos) {
                self.requested_chunks.insert(chunk_pos, ChunkRequest::LoadRequested);
            }
        }

        pub fn request_unload_chunk(&mut self, chunk_pos: IVec3) {
            if self.is_chunk_loaded(chunk_pos) && !self.is_chunk_unload_pending(&chunk_pos) {
                self.requested_chunks.insert(chunk_pos, ChunkRequest::UnloadRequested);
            }
        }

        fn get_chunk_request(&self, chunk_pos: &IVec3) -> Option<&ChunkRequest> {
            self.requested_chunks.get(&chunk_pos)
        }

        fn is_chunk_load_requested(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::LoadRequested || *req == ChunkRequest::LoadPending)
        }

        fn is_chunk_load_pending(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::LoadPending)
        }

        fn is_chunk_unload_requested(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::UnloadRequested || *req == ChunkRequest::UnloadPending)
        }

        fn is_chunk_unload_pending(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::UnloadPending)
        }

        pub fn is_chunk_loaded(&self, chunk_pos: IVec3) -> bool {
            self.loaded_chunks.contains_key(&chunk_pos)
        }

        pub fn for_each_loaded_chunk<F>(&mut self, mut f: F)
        where F: FnMut(&IVec3, &VoxelChunkEntity) {
            self.loaded_chunks.iter().for_each(|(chunk_pos, chunk)| {
                f(chunk_pos, chunk);
            });
        }

        pub fn get_chunk(&self, chunk_pos: IVec3) -> Option<&VoxelChunkEntity> {
            // let index = self.loaded_chunks.get(&chunk_pos)?;
            // let chunk = self.chunks.get(*index)?;
            // Some(chunk)

            self.loaded_chunks.get(&chunk_pos)
        }

        pub fn get_chunk_mut(&mut self, chunk_pos: IVec3) -> Option<&mut VoxelChunkEntity> {
            // let index = self.loaded_chunks.get(&chunk_pos)?;
            // let chunk = self.chunks.get_mut(*index)?;
            // Some(chunk)

            self.loaded_chunks.get_mut(&chunk_pos)
        }

        pub fn get_block(&self, block_pos: IVec3) -> Option<u32> {
            let chunk_pos = get_chunk_pos_for_block_pos(block_pos);
            self.get_chunk(chunk_pos).map(|chunk| {
                chunk.get_block(get_local_block_pos_in_chunk(chunk_pos, block_pos))
            })
        }

        pub fn set_block(&mut self, block_pos: IVec3, block: u32) {
            let chunk_pos = get_chunk_pos_for_block_pos(block_pos);
            let loaded_chunk = self.get_chunk_mut(chunk_pos);

            if let Some(chunk) = loaded_chunk {
                let block_pos = get_local_block_pos_in_chunk(chunk_pos, block_pos);
                chunk.set_block(block_pos, block);
            } else {
                let chunk_edits = self.unloaded_block_edits.entry(chunk_pos).or_insert_with(|| HashMap::new());
                let block_pos = get_local_block_pos_in_chunk(chunk_pos, block_pos);
                chunk_edits.insert(block_pos, block);
            }
        }

        pub fn draw_debug(&self, ctx: &mut DebugRenderContext) -> Result<()> {
            for (_chunk_pos, chunk) in self.loaded_chunks.iter() {
                chunk.draw_debug_bounds(ctx)?;
            }

            if let Some(chunk) = self.loaded_chunks.get(&self.player_chunk_pos) {
                chunk.draw_debug_grid(ctx)?;
            }

            Ok(())
        }
    }



    impl VoxelChunkEntity {
        fn load(chunk_data: Box<VoxelChunkData>, engine: &mut Engine) -> Self {
            let chunk_pos = *chunk_data.chunk_pos();
            let pos = get_chunk_world_pos(chunk_pos);

            // let entity_dir_id: [_; 6] = array::from_fn(|i| {
            //     let mut entity = engine.scene.create_entity(format!("chunk({},{},{})/{}", chunk_pos.x, chunk_pos.y, chunk_pos.z, AxisDirection::from_index(i as u32).unwrap().name()).as_str());
            //     entity.add_component(*Transform::new().set_translation(pos.as_vec3()));
            //     entity.add_component(RenderComponent::<BaseVertex>::new(RenderType::Static, None));
            //     entity.id().id()
            // });

            let mut render_component = VoxelChunkRenderComponent::new(chunk_pos);

            engine.voxel_renderer.load_chunk(&mut render_component);

            let mut entity = engine.scene.create_entity(format!("chunk({},{},{})", chunk_pos.x, chunk_pos.y, chunk_pos.z).as_str());
            entity.add_component(*Transform::new().set_translation(pos.as_vec3()));
            // entity.add_component(RenderComponent::<BaseVertex>::new(RenderType::Static, None));
            entity.add_component(render_component);

            VoxelChunkEntity {
                entity_id: entity.id().id(),
                // entity_dir_id,
                chunk_data: Some(chunk_data)
            }
        }

        fn update_render_component(&mut self, engine: &mut Engine) {
            let chunk_data = self.chunk_data.as_mut().unwrap();

            if let Some(mut render_component) = engine.scene.ecs.get_mut::<VoxelChunkRenderComponent>(self.entity_id) {
                for i in 0..6 {
                    if let Some(mesh_data) = chunk_data.updated_mesh_data[i].take() {
                        render_component.updated_mesh_data[i] = Some(mesh_data);
                        render_component.update_mesh[i] = true;
                    }
                }
                render_component.bounds = chunk_data.bounds.clone();
            }
        }

        fn unload(&mut self, engine: &mut Engine) {
            // debug!("Unload chunk [{}, {}, {}]", self.chunk_pos().x, self.chunk_pos().y, self.chunk_pos().z);
            if let Some(mut render_component) = engine.scene.ecs.get_mut::<VoxelChunkRenderComponent>(self.entity_id) {
                engine.voxel_renderer.unload_chunk(render_component.as_mut());
            }
            engine.scene.destroy_entity(self.entity_id);

            // for i in 0..6 {
            //     engine.scene.destroy_entity(self.entity_dir_id[i]);
            // }
        }

        // fn update_mesh_data(&mut self) -> Result<()>{
        //     self.chunk_data.update_mesh_data()
        // }

        pub fn get_block(&self, pos: UVec3) -> u32 {
            self.chunk_data().get_block(pos)
        }

        pub fn set_block(&mut self, pos: UVec3, block: u32) {
            self.chunk_data_mut().set_block(pos, block)
        }

        pub fn draw_debug_bounds(&self, ctx: &mut DebugRenderContext) -> Result<()> {
            self.chunk_data().draw_debug_bounds(ctx)
        }

        pub fn draw_debug_grid(&self, ctx: &mut DebugRenderContext) -> Result<()> {
            self.chunk_data().draw_debug_grid(ctx)
        }

        pub fn chunk_data(&self) -> &Box<VoxelChunkData> {
            self.chunk_data.as_ref().unwrap()
        }

        pub fn chunk_data_mut(&mut self) -> &mut Box<VoxelChunkData> {
            self.chunk_data.as_mut().unwrap()
        }

        pub fn chunk_pos(&self) -> &IVec3 {
            self.chunk_data().chunk_pos()
        }
    }


    impl VoxelChunkData {
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_NEG_X: u8 = 0b00000001;
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_POS_X: u8 = 0b00000010;
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_NEG_Y: u8 = 0b00000100;
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_POS_Y: u8 = 0b00001000;
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_NEG_Z: u8 = 0b00010000;
        pub const BLOCK_FLAG_HAS_NEIGHBOUR_POS_Z: u8 = 0b00100000;
        pub const BLOCK_FLAGS_IS_ENCLOSED: u8 = Self::BLOCK_FLAG_HAS_NEIGHBOUR_NEG_X | Self::BLOCK_FLAG_HAS_NEIGHBOUR_POS_X | Self::BLOCK_FLAG_HAS_NEIGHBOUR_NEG_Y | Self::BLOCK_FLAG_HAS_NEIGHBOUR_POS_Y | Self::BLOCK_FLAG_HAS_NEIGHBOUR_NEG_Z | Self::BLOCK_FLAG_HAS_NEIGHBOUR_POS_Z;
        pub const BLOCK_FLAG_IS_TRANSPARENT: u8 = 0b01000000;


        pub fn new(chunk_pos: IVec3) -> Self {

            VoxelChunkData {
                chunk_pos,
                blocks: [0; CHUNK_BLOCK_COUNT],
                // block_flags: [0; CHUNK_BLOCK_COUNT],
                block_count: 0,
                dirty: true,
                updated_mesh_data: Default::default(), // [None; 6]
                bounds: Default::default(), // [None; 6]
                mesh: Default::default(), // [None; 6]
                unloaded_block_edits: HashMap::new(),
            }
        }

        pub fn chunk_pos(&self) -> &IVec3 {
            &self.chunk_pos
        }

        // pub fn mesh(&self, dir: AxisDirection) -> Option<&Arc<Mesh<VoxelVertex>>> {
        //     self.mesh[dir.index() as usize].as_ref()
        //     // self.mesh.as_ref()
        // }

        pub fn get_block(&self, pos: UVec3) -> u32 {
            let index = calc_index_for_coord(pos, CHUNK_BOUNDS) as usize;
            self.blocks[index]
        }

        pub fn set_block(&mut self, pos: UVec3, block: u32) {
            let index = calc_index_for_coord(pos, CHUNK_BOUNDS) as usize;
            let prev_block = self.blocks[index];
            self.blocks[index] = block;

            if prev_block != block {

                // let pos = pos.as_ivec3();
                // self.update_neighbour_flags(pos, index);

                if prev_block == 0 && block != 0 {
                    self.block_count += 1;

                } else if prev_block != 0 && block == 0 {
                    self.block_count -= 1;
                }

                self.dirty = true;
            }
        }

        pub fn set_blocks(&mut self, start_index: u32, end_index: u32, block: u32) {
            let is_solid = block != 0;
            for i in start_index .. end_index {
                let prev_block = &mut self.blocks[i as usize];
                if *prev_block != block {
                    *prev_block = block;

                    if is_solid {
                        self.block_count += 1;
                    } else {
                        self.block_count -= 1;
                    }
                }
            }
            self.dirty = true;
        }

        // pub fn update_mesh_data(&mut self) -> Result<()> {
        //     if self.dirty {
        //         self.dirty = false;
        //
        //         // let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_neg_x = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_pos_x = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_neg_y = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_pos_y = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_neg_z = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //         let mut mesh_data_pos_z = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
        //
        //         // let p1 = (CHUNK_SIZE as f32) * 0.5;
        //         // mesh_data.create_cuboid([p1, p1, p1], [p1, p1, p1]);
        //
        //         // let estimated_block_count = self.block_count / 2;
        //         // let block_vertex_count = 24;
        //         // let block_index_count = 36;
        //         // mesh_data.reserve_vertices(block_vertex_count * estimated_block_count as usize);
        //         // mesh_data.reserve_indices(block_index_count * estimated_block_count as usize);
        //
        //         fn add_face_neg_x(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x - 0, y - 0, z + 1).as_vec3(), Vec3::NEG_X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x - 0, y - 0, z - 0).as_vec3(), Vec3::NEG_X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x - 0, y + 1,  z - 0).as_vec3(), Vec3::NEG_X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x - 0, y + 1,  z + 1).as_vec3(), Vec3::NEG_X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //         fn add_face_pos_x(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x + 1, y - 0, z - 0).as_vec3(), Vec3::X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x + 1, y - 0, z + 1).as_vec3(), Vec3::X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x + 1, y + 1, z + 1).as_vec3(), Vec3::X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x + 1, y + 1, z - 0).as_vec3(), Vec3::X, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //         fn add_face_neg_y(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x - 0, y - 0, z - 0).as_vec3(), Vec3::NEG_Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x - 0, y - 0, z + 1).as_vec3(), Vec3::NEG_Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x + 1, y - 0, z + 1).as_vec3(), Vec3::NEG_Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x + 1, y - 0, z - 0).as_vec3(), Vec3::NEG_Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //         fn add_face_pos_y(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x + 1, y + 1, z + 1).as_vec3(), Vec3::Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x - 0, y + 1, z + 1).as_vec3(), Vec3::Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x - 0, y + 1, z - 0).as_vec3(), Vec3::Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x + 1, y + 1, z - 0).as_vec3(), Vec3::Y, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //         fn add_face_neg_z(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x + 1, y + 1, z - 0).as_vec3(), Vec3::NEG_Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x - 0, y + 1, z - 0).as_vec3(), Vec3::NEG_Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x - 0, y - 0, z - 0).as_vec3(), Vec3::NEG_Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x + 1, y - 0, z - 0).as_vec3(), Vec3::NEG_Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //         fn add_face_pos_z(mesh_data: &mut MeshData<BaseVertex>, x: i32, y: i32, z: i32) {
        //             let v00 = BaseVertex::new(IVec3::new(x - 0, y + 1, z + 1).as_vec3(), Vec3::Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v01 = BaseVertex::new(IVec3::new(x + 1, y + 1, z + 1).as_vec3(), Vec3::Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v11 = BaseVertex::new(IVec3::new(x + 1, y - 0, z + 1).as_vec3(), Vec3::Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             let v10 = BaseVertex::new(IVec3::new(x - 0, y - 0, z + 1).as_vec3(), Vec3::Z, U8Vec4::new(255, 255, 255, 255), U16Vec2::new(0, 0));
        //             mesh_data.create_quad(v00, v01, v11, v10);
        //         }
        //
        //         let dim_x = CHUNK_BOUNDS.x as i32;
        //         let dim_y = CHUNK_BOUNDS.y as i32;
        //         let dim_z = CHUNK_BOUNDS.z as i32;
        //
        //         let y_stride = dim_x;
        //         let z_stride = dim_x * dim_y;
        //
        //         for z in 0..dim_z {
        //             let iz = z;
        //             let z_index = z * z_stride;
        //             let z_index_z_n1 = (z - 1) * z_stride;
        //             let z_index_z_p1 = (z + 1) * z_stride;
        //
        //
        //             for y in 0..dim_y {
        //                 let iy = y;
        //                 let yz_index = (y * y_stride) + z_index;
        //                 let yz_index_y_n1 = ((y - 1) * y_stride) + z_index;;
        //                 let yz_index_y_p1 = ((y + 1) * y_stride) + z_index;
        //                 let yz_index_z_n1 = (y * y_stride) + z_index_z_n1;
        //                 let yz_index_z_p1 = (y * y_stride) + z_index_z_p1;
        //
        //                 for x in 0..dim_z {
        //                     let ix = x;
        //
        //                     let xyz_index = (x + yz_index) as usize;
        //
        //                     let block = self.blocks[xyz_index];
        //                     if block == 0 {
        //                         continue;
        //                     }
        //
        //                     let xyz_index_x_n1 = ((x - 1) + yz_index) as usize;
        //                     let xyz_index_x_p1 = ((x + 1) + yz_index) as usize;
        //                     let xyz_index_y_n1 = (x + yz_index_y_n1) as usize;
        //                     let xyz_index_y_p1 = (x + yz_index_y_p1) as usize;
        //                     let xyz_index_z_n1 = (x + yz_index_z_n1) as usize;
        //                     let xyz_index_z_p1 = (x + yz_index_z_p1) as usize;
        //
        //                     // let xyz_index_x_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(1, 0, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //                     // let xyz_index_x_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(-1, 0, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //                     // let xyz_index_y_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 1, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //                     // let xyz_index_y_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, -1, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //                     // let xyz_index_z_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 0, 1), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //                     // let xyz_index_z_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 0, -1), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
        //
        //                     let block_x_n1 = if x > 0 {
        //                         // assert_eq!(xyz_index_x_n1, calc_index_for_coord(UVec3::new((x - 1) as u32, y as u32, z as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_x_n1]
        //                     } else { 0 };
        //                     let block_x_p1 = if x < (dim_x - 1) {
        //                         // assert_eq!(xyz_index_x_p1, calc_index_for_coord(UVec3::new((x + 1) as u32, y as u32, z as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_x_p1]
        //                     } else { 0 };
        //                     let block_y_n1 = if y > 0 {
        //                         // assert_eq!(xyz_index_y_n1, calc_index_for_coord(UVec3::new(x as u32, (y - 1) as u32, z as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_y_n1]
        //                     } else { 0 };
        //                     let block_y_p1 = if y < (dim_y - 1) {
        //                         // assert_eq!(xyz_index_y_p1, calc_index_for_coord(UVec3::new(x as u32, (y + 1) as u32, z as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_y_p1]
        //                     } else { 0 };
        //                     let block_z_n1 = if z > 0 {
        //                         // assert_eq!(xyz_index_z_n1, calc_index_for_coord(UVec3::new(x as u32, y as u32, (z - 1) as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_z_n1]
        //                     } else { 0 };
        //                     let block_z_p1 = if z < (dim_z - 1) {
        //                         // assert_eq!(xyz_index_z_p1, calc_index_for_coord(UVec3::new(x as u32, y as u32, (z + 1) as u32), CHUNK_BOUNDS) as usize);
        //                         self.blocks[xyz_index_z_p1]
        //                     } else { 0 };
        //
        //                     let has_all_neighbours =
        //                         block_x_n1 != 0 && block_x_p1 != 0 &&
        //                             block_y_n1 != 0 && block_y_p1 != 0 &&
        //                             block_z_n1 != 0 && block_z_p1 != 0;
        //
        //                     if has_all_neighbours {
        //                         continue;
        //                     }
        //
        //                     if block_x_n1 == 0 {
        //                         add_face_neg_x(&mut mesh_data_neg_x, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::NegX, center, [half_size[1], half_size[2]], half_size[0]);
        //                     }
        //                     if block_x_p1 == 0 {
        //                         add_face_pos_x(&mut mesh_data_pos_x, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::PosX, center, [half_size[1], half_size[2]], half_size[0]);
        //                     }
        //                     if block_y_n1 == 0 {
        //                         add_face_neg_y(&mut mesh_data_neg_y, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::NegY, center, [half_size[0], half_size[2]], half_size[1]);
        //                     }
        //                     if block_y_p1 == 0 {
        //                         add_face_pos_y(&mut mesh_data_pos_y, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::PosY, center, [half_size[0], half_size[2]], half_size[1]);
        //                     }
        //                     if block_z_n1 == 0 {
        //                         add_face_neg_z(&mut mesh_data_neg_z, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::NegZ, center, [half_size[0], half_size[1]], half_size[2]);
        //                     }
        //                     if block_z_p1 == 0 {
        //                         add_face_pos_z(&mut mesh_data_pos_z, ix, iy, iz);
        //                         // mesh_data.create_box_face(AxisDirection::PosZ, center, [half_size[0], half_size[1]], half_size[2]);
        //                     }
        //                 }
        //             }
        //         }
        //
        //         // assert_eq!(self.block_count, test_block_count);
        //
        //         let chunk_world_pos = get_chunk_world_pos(self.chunk_pos);
        //
        //         for i in 0..6 {
        //             self.updated_mesh_data[i] = None;
        //             self.bounds[i] = None;
        //         }
        //         if !mesh_data_neg_x.vertices.is_empty() {
        //             self.bounds[AxisDirection::NegX.index() as usize] = Some(mesh_data_neg_x.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::NegX.index() as usize] = Some(mesh_data_neg_x)
        //         }
        //         if !mesh_data_pos_x.vertices.is_empty() {
        //             self.bounds[AxisDirection::PosX.index() as usize] = Some(mesh_data_pos_x.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::PosX.index() as usize] = Some(mesh_data_pos_x)
        //         }
        //         if !mesh_data_neg_y.vertices.is_empty() {
        //             self.bounds[AxisDirection::NegY.index() as usize] = Some(mesh_data_neg_y.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::NegY.index() as usize] = Some(mesh_data_neg_y)
        //         }
        //         if !mesh_data_pos_y.vertices.is_empty() {
        //             self.bounds[AxisDirection::PosY.index() as usize] = Some(mesh_data_pos_y.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::PosY.index() as usize] = Some(mesh_data_pos_y)
        //         }
        //         if !mesh_data_neg_z.vertices.is_empty() {
        //             self.bounds[AxisDirection::NegZ.index() as usize] = Some(mesh_data_neg_z.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::NegZ.index() as usize] = Some(mesh_data_neg_z)
        //         }
        //         if !mesh_data_pos_z.vertices.is_empty() {
        //             self.bounds[AxisDirection::PosZ.index() as usize] = Some(mesh_data_pos_z.calculate_aabb().translate(chunk_world_pos));
        //             self.updated_mesh_data[AxisDirection::PosZ.index() as usize] = Some(mesh_data_pos_z)
        //         }
        //
        //         // let has_vertices =
        //         //     !mesh_data_neg_x.vertices.is_empty() || !mesh_data_pos_x.vertices.is_empty() ||
        //         //     !mesh_data_neg_y.vertices.is_empty() || !mesh_data_pos_y.vertices.is_empty() ||
        //         //     !mesh_data_neg_z.vertices.is_empty() || !mesh_data_pos_z.vertices.is_empty();
        //         //
        //         // if has_vertices {
        //         //     self.updated_mesh_data = Some(mesh_data);
        //         // } else {
        //         //     self.updated_mesh_data = None;
        //         // }
        //     }
        //
        //     Ok(())
        // }

        pub fn update_mesh_data(&mut self) -> Result<()> {
            if self.dirty {
                self.dirty = false;

                // let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);
                let mut mesh_data_neg_x = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));
                let mut mesh_data_pos_x = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));
                let mut mesh_data_neg_y = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));
                let mut mesh_data_pos_y = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));
                let mut mesh_data_neg_z = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));
                let mut mesh_data_pos_z = MeshData::<VoxelVertex>::new(MeshDataConfig::new(MeshPrimitiveType::TriangleList));

                // let p1 = (CHUNK_SIZE as f32) * 0.5;
                // mesh_data.create_cuboid([p1, p1, p1], [p1, p1, p1]);

                // let estimated_block_count = self.block_count / 2;
                // let block_vertex_count = 24;
                // let block_index_count = 36;
                // mesh_data.reserve_vertices(block_vertex_count * estimated_block_count as usize);
                // mesh_data.reserve_indices(block_index_count * estimated_block_count as usize);

                // For each face, we repeat the same "vertex" 4 times. The vertex position is simply the voxel position within the chunk (0-31)
                // The face vertex coordinates are reconstructed in the vertex shader from the voxel position and face normal index

                fn add_face_neg_x(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::NegX);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }
                fn add_face_pos_x(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::PosX);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }
                fn add_face_neg_y(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::NegY);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }
                fn add_face_pos_y(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::PosY);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }
                fn add_face_neg_z(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::NegZ);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }
                fn add_face_pos_z(mesh_data: &mut MeshData<VoxelVertex>, x: u32, y: u32, z: u32) {
                    let v00 = VoxelVertex::from_unpacked(x, y, z, AxisDirection::PosZ);
                    mesh_data.create_quad(v00.clone(), v00.clone(), v00.clone(), v00);
                }

                let dim_x = CHUNK_BOUNDS.x as i32;
                let dim_y = CHUNK_BOUNDS.y as i32;
                let dim_z = CHUNK_BOUNDS.z as i32;

                let y_stride = dim_x;
                let z_stride = dim_x * dim_y;

                for z in 0..dim_z {
                    let iz = z as u32;
                    let z_index = z * z_stride;
                    let z_index_z_n1 = (z - 1) * z_stride;
                    let z_index_z_p1 = (z + 1) * z_stride;


                    for y in 0..dim_y {
                        let iy = y as u32;
                        let yz_index = (y * y_stride) + z_index;
                        let yz_index_y_n1 = ((y - 1) * y_stride) + z_index;;
                        let yz_index_y_p1 = ((y + 1) * y_stride) + z_index;
                        let yz_index_z_n1 = (y * y_stride) + z_index_z_n1;
                        let yz_index_z_p1 = (y * y_stride) + z_index_z_p1;

                        for x in 0..dim_z {
                            let ix = x as u32;

                            let xyz_index = (x + yz_index) as usize;

                            let block = self.blocks[xyz_index];
                            if block == 0 {
                                continue;
                            }

                            let xyz_index_x_n1 = ((x - 1) + yz_index) as usize;
                            let xyz_index_x_p1 = ((x + 1) + yz_index) as usize;
                            let xyz_index_y_n1 = (x + yz_index_y_n1) as usize;
                            let xyz_index_y_p1 = (x + yz_index_y_p1) as usize;
                            let xyz_index_z_n1 = (x + yz_index_z_n1) as usize;
                            let xyz_index_z_p1 = (x + yz_index_z_p1) as usize;

                            // let xyz_index_x_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(1, 0, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
                            // let xyz_index_x_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(-1, 0, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
                            // let xyz_index_y_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 1, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
                            // let xyz_index_y_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, -1, 0), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
                            // let xyz_index_z_p1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 0, 1), CHUNK_BOUNDS).map_or(0, |val| val) as usize;
                            // let xyz_index_z_n1 = calc_index_for_coord_offset(IVec3::new(x as i32, y as i32, z as i32), IVec3::new(0, 0, -1), CHUNK_BOUNDS).map_or(0, |val| val) as usize;

                            let block_x_n1 = if x > 0 {
                                // assert_eq!(xyz_index_x_n1, calc_index_for_coord(UVec3::new((x - 1) as u32, y as u32, z as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_x_n1]
                            } else { 0 };
                            let block_x_p1 = if x < (dim_x - 1) {
                                // assert_eq!(xyz_index_x_p1, calc_index_for_coord(UVec3::new((x + 1) as u32, y as u32, z as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_x_p1]
                            } else { 0 };
                            let block_y_n1 = if y > 0 {
                                // assert_eq!(xyz_index_y_n1, calc_index_for_coord(UVec3::new(x as u32, (y - 1) as u32, z as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_y_n1]
                            } else { 0 };
                            let block_y_p1 = if y < (dim_y - 1) {
                                // assert_eq!(xyz_index_y_p1, calc_index_for_coord(UVec3::new(x as u32, (y + 1) as u32, z as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_y_p1]
                            } else { 0 };
                            let block_z_n1 = if z > 0 {
                                // assert_eq!(xyz_index_z_n1, calc_index_for_coord(UVec3::new(x as u32, y as u32, (z - 1) as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_z_n1]
                            } else { 0 };
                            let block_z_p1 = if z < (dim_z - 1) {
                                // assert_eq!(xyz_index_z_p1, calc_index_for_coord(UVec3::new(x as u32, y as u32, (z + 1) as u32), CHUNK_BOUNDS) as usize);
                                self.blocks[xyz_index_z_p1]
                            } else { 0 };

                            let has_all_neighbours =
                                block_x_n1 != 0 && block_x_p1 != 0 &&
                                    block_y_n1 != 0 && block_y_p1 != 0 &&
                                    block_z_n1 != 0 && block_z_p1 != 0;

                            if has_all_neighbours {
                                continue;
                            }

                            if block_x_n1 == 0 {
                                add_face_neg_x(&mut mesh_data_neg_x, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::NegX, center, [half_size[1], half_size[2]], half_size[0]);
                            }
                            if block_x_p1 == 0 {
                                add_face_pos_x(&mut mesh_data_pos_x, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::PosX, center, [half_size[1], half_size[2]], half_size[0]);
                            }
                            if block_y_n1 == 0 {
                                add_face_neg_y(&mut mesh_data_neg_y, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::NegY, center, [half_size[0], half_size[2]], half_size[1]);
                            }
                            if block_y_p1 == 0 {
                                add_face_pos_y(&mut mesh_data_pos_y, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::PosY, center, [half_size[0], half_size[2]], half_size[1]);
                            }
                            if block_z_n1 == 0 {
                                add_face_neg_z(&mut mesh_data_neg_z, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::NegZ, center, [half_size[0], half_size[1]], half_size[2]);
                            }
                            if block_z_p1 == 0 {
                                add_face_pos_z(&mut mesh_data_pos_z, ix, iy, iz);
                                // mesh_data.create_box_face(AxisDirection::PosZ, center, [half_size[0], half_size[1]], half_size[2]);
                            }
                        }
                    }
                }

                // assert_eq!(self.block_count, test_block_count);

                let chunk_world_pos = get_chunk_world_pos(self.chunk_pos);

                for i in 0..6 {
                    self.updated_mesh_data[i] = None;
                    self.bounds[i] = None;
                }
                if !mesh_data_neg_x.vertices.is_empty() {
                    // self.bounds[AxisDirection::NegX.index() as usize] = Some(mesh_data_neg_x.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::NegX.index() as usize] = Some(mesh_data_neg_x)
                }
                if !mesh_data_pos_x.vertices.is_empty() {
                    // self.bounds[AxisDirection::PosX.index() as usize] = Some(mesh_data_pos_x.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::PosX.index() as usize] = Some(mesh_data_pos_x)
                }
                if !mesh_data_neg_y.vertices.is_empty() {
                    // self.bounds[AxisDirection::NegY.index() as usize] = Some(mesh_data_neg_y.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::NegY.index() as usize] = Some(mesh_data_neg_y)
                }
                if !mesh_data_pos_y.vertices.is_empty() {
                    // self.bounds[AxisDirection::PosY.index() as usize] = Some(mesh_data_pos_y.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::PosY.index() as usize] = Some(mesh_data_pos_y)
                }
                if !mesh_data_neg_z.vertices.is_empty() {
                    // self.bounds[AxisDirection::NegZ.index() as usize] = Some(mesh_data_neg_z.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::NegZ.index() as usize] = Some(mesh_data_neg_z)
                }
                if !mesh_data_pos_z.vertices.is_empty() {
                    // self.bounds[AxisDirection::PosZ.index() as usize] = Some(mesh_data_pos_z.calculate_aabb().translate(chunk_world_pos));
                    self.updated_mesh_data[AxisDirection::PosZ.index() as usize] = Some(mesh_data_pos_z)
                }

                // let has_vertices =
                //     !mesh_data_neg_x.vertices.is_empty() || !mesh_data_pos_x.vertices.is_empty() ||
                //     !mesh_data_neg_y.vertices.is_empty() || !mesh_data_pos_y.vertices.is_empty() ||
                //     !mesh_data_neg_z.vertices.is_empty() || !mesh_data_pos_z.vertices.is_empty();
                //
                // if has_vertices {
                //     self.updated_mesh_data = Some(mesh_data);
                // } else {
                //     self.updated_mesh_data = None;
                // }
            }

            Ok(())
        }


        pub fn create_gpu_mesh(&mut self, cmd_buf: &mut CommandBuffer, staging_buffer: &mut Option<Subbuffer<[u8]>>, engine: &mut Engine) -> Result<bool> {
            let staging_size = self.get_staging_buffer_size(); // Careful to call this before updated_mesh_data.take()

            if staging_size == 0 {
                return Ok(false)
            }

            let mut changed = false;

            for i in 0..6 {
                let mesh_data = self.updated_mesh_data[i].take();

                if let Some(mesh_data) = mesh_data {
                    let allocator = engine.graphics.memory_allocator();

                    let staging_size = mesh_data.get_required_staging_buffer_size();
                    let staging_buffer = util::chop_buffer_at(staging_buffer, staging_size).unwrap();

                    let mesh = Arc::new(mesh_data.build_mesh_staged(allocator, cmd_buf, &staging_buffer)?);
                    self.mesh[i] = Some(mesh);

                    changed = true;
                }
            }


            Ok(changed)
        }

        fn get_staging_buffer_size(&self) -> DeviceSize {
            let mut size = 0;
            for mesh_data in self.updated_mesh_data.iter() {
                if let Some(mesh_data) = mesh_data {
                    size += mesh_data.get_required_staging_buffer_size();
                }
            }
            size

            // if let Some(mesh_data) = self.updated_mesh_data.as_ref() {
            //     mesh_data.get_required_staging_buffer_size()
            // } else {
            //     0
            // }
        }

        fn load_edits(&mut self, block_edits: HashMap<UVec3, u32>) {
            if !block_edits.is_empty() {
                debug!("Loading {} block edits for chunk ({}, {}, {})", block_edits.len(), self.chunk_pos.x, self.chunk_pos.y, self.chunk_pos.z);
            }
            for (pos, block) in block_edits {
                self.set_block(pos, block);
            }
        }

        fn draw_debug_bounds(&self, ctx: &mut DebugRenderContext) -> Result<()> {
            self.get_bounds().draw_debug(ctx)
        }

        fn draw_debug_grid(&self, ctx: &mut DebugRenderContext) -> Result<()> {
            let bounds = self.get_bounds();

            let h = bounds.half_extent();
            let colour = U8Vec4::new(255, 0, 0, 100);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() + DVec3::X * h.x).as_vec3())
                .rotate_y(f32::to_radians(90.0))
                .scale(bounds.extent().as_vec3()), colour);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() - DVec3::X * h.x).as_vec3())
                .rotate_y(f32::to_radians(90.0))
                .scale(bounds.extent().as_vec3()), colour);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() + DVec3::Y * h.y).as_vec3())
                .rotate_x(f32::to_radians(90.0))
                .scale(bounds.extent().as_vec3()), colour);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() - DVec3::Y * h.y).as_vec3())
                .rotate_x(f32::to_radians(90.0))
                .scale(bounds.extent().as_vec3()), colour);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() + DVec3::Z * h.z).as_vec3())
                .scale(bounds.extent().as_vec3()), colour);

            ctx.add_mesh(debug_mesh::mesh_grid_32_lines(), *Transform::new()
                .translate((bounds.center() - DVec3::Z * h.z).as_vec3())
                .scale(bounds.extent().as_vec3()), colour);


            Ok(())
        }

        pub fn get_bounds(&self) -> AxisAlignedBoundingBox {
            let pos0 = get_chunk_world_pos(self.chunk_pos);
            let pos1 = get_chunk_world_pos(self.chunk_pos + IVec3::ONE);
            AxisAlignedBoundingBox::new(pos0, pos1)
        }

        pub fn block_count(&self) -> u32 {
            // self.block_count
            0
        }
    }

    impl Drop for VoxelChunkEntity {
        fn drop(&mut self) {
            debug_assert!(self.chunk_data.is_none(), "VoxelChunkEntity was not deallocated before being dropped")
            // debug!("Dropping chunk: {:?}", self.chunk_pos);
        }
    }



    // TODO: investigate space-filling curves such as hilbert curves for better cache locality
    // https://github.com/spectral3d/hilbert_hpp/blob/master/hilbert.hpp
    pub fn calc_index_for_coord(coord: UVec3, bounds: UVec3) -> u32 {
        let x = coord.x;
        let y = coord.y;
        let z = coord.z;
        let sx = bounds.x;
        let sy = bounds.y;

        x + (y * sx) + (z * sx * sy)
    }

    pub fn calc_index_for_coord_checked(coord: IVec3, bounds: UVec3) -> Option<u32> {
        if coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= bounds.x as i32 || coord.y >= bounds.y as i32 || coord.z >= bounds.z as i32 {
            return None;
        }
        Some(calc_index_for_coord(coord.as_uvec3(), bounds))
    }

    pub fn calc_index_for_coord_offset(coord: IVec3, offset: IVec3, bounds: UVec3) -> Option<u32> {
        let coord = coord + offset;
        if !check_coord_bounds(coord, bounds) {
            return None;
        }
        Some(calc_index_for_coord(coord.as_uvec3(), bounds))
    }

    pub fn check_coord_bounds(coord: IVec3, bounds: UVec3) -> bool {
        if coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= bounds.x as i32 || coord.y >= bounds.y as i32 || coord.z >= bounds.z as i32 {
            return false;
        }
        true
    }

    pub fn calc_coord_for_index(index: u32, bounds: UVec3) -> UVec3 {
        let sx = bounds.x;
        let sy = bounds.y;
        let x = index % sx;
        let y  = (index / sx) % sy;
        let z = index / (sx * sy);

        UVec3::new(x, y, z)
    }

    pub fn get_block_pos_for_world_pos(world_pos: DVec3) -> IVec3 {
        let x = f64::floor(world_pos.x) as i32;
        let y = f64::floor(world_pos.y) as i32;
        let z = f64::floor(world_pos.z) as i32;
        IVec3::new(x, y, z)
    }

    pub fn get_chunk_pos_for_world_pos(world_pos: DVec3) -> IVec3 {
        let x = f64::floor(world_pos.x / CHUNK_BOUNDS.x as f64) as i32;
        let y = f64::floor(world_pos.y / CHUNK_BOUNDS.y as f64) as i32;
        let z = f64::floor(world_pos.z / CHUNK_BOUNDS.z as f64) as i32;
        IVec3::new(x, y, z)
    }

    pub fn get_chunk_pos_for_block_pos(block_pos: IVec3) -> IVec3 {
        let x = div_floor_p2(block_pos.x, CHUNK_SIZE_EXP);
        let y = div_floor_p2(block_pos.y, CHUNK_SIZE_EXP);
        let z = div_floor_p2(block_pos.z, CHUNK_SIZE_EXP);
        IVec3::new(x, y, z)
    }

    pub fn get_local_block_pos_in_chunk(chunk_pos: IVec3, world_block_pos: IVec3) -> UVec3 {
        let pos = chunk_pos * CHUNK_BOUNDS.as_ivec3();
        let block_pos = (world_block_pos - pos).as_uvec3();
        block_pos
    }

    pub fn get_world_block_pos_for_chunk_block_pos(chunk_pos: IVec3, chunk_block_pos: UVec3) -> IVec3 {
        chunk_pos * CHUNK_BOUNDS.as_ivec3() + chunk_block_pos.as_ivec3()
    }

    pub fn get_chunk_world_pos(chunk_pos: IVec3) -> DVec3 {
        (chunk_pos * CHUNK_BOUNDS.as_ivec3()).as_dvec3()
    }

    pub fn get_chunk_center_world_pos(chunk_pos: IVec3) -> DVec3 {
        get_chunk_world_pos(chunk_pos) + CHUNK_BOUNDS.as_dvec3() * 0.5
    }

    pub fn distance_sq_chunk_center_to_world_pos(chunk_pos: IVec3, world_pos: DVec3) -> f64 {
        DVec3::distance_squared(get_chunk_center_world_pos(chunk_pos), world_pos)
    }

    pub fn distance_sq_between_chunks(chunk_pos1: IVec3, chunk_pos2: IVec3) -> f64 {
        DVec3::distance_squared(chunk_pos1.as_dvec3(), chunk_pos2.as_dvec3())
    }

    pub fn calculate_chunk_bounds(chunk_pos: IVec3) -> AxisAlignedBoundingBox {
        let pos0 = get_chunk_world_pos(chunk_pos);
        let pos1 = get_chunk_world_pos(chunk_pos + IVec3::ONE);
        AxisAlignedBoundingBox::new(pos0, pos1)
    }

    pub fn closest_point_on_chunk(world_pos: DVec3, chunk_pos: IVec3) -> DVec3 {
        calculate_chunk_bounds(chunk_pos).closest_point(world_pos)
    }

    /// same as div_floor, but more efficient for powers of two
    fn div_floor_p2(a: i32, b_exp: u32) -> i32 {
        a >> b_exp
    }

    /// Integer division that always rounds down towards negative infinity, instead of truncating towards zero.
    /// TODO: test if this is actually faster than converting to a float.
    #[inline(always)]
    fn div_floor(a: i32, b: i32) -> i32 {
        let q = a / b;
        let r = a % b;
        q - ((r != 0 && (r ^ b) < 0) as i32)
    }
}
