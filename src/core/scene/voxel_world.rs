
pub mod world {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use anyhow::Result;
    use ash::vk::DeviceSize;
    use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
    use glam::{IVec3, UVec3, Vec3};
    use log::debug;
    use vulkano::buffer::Subbuffer;
    use vulkano::memory::allocator::MemoryAllocator;
    use crate::core::{BaseVertex, CommandBuffer, Engine, Entity, GraphicsManager, Mesh, MeshData, MeshPrimitiveType, RenderComponent, RenderType, Transform};
    use crate::core::util::util::chop_buffer_at;

    pub const CHUNK_SIZE_EXP: u32 = 5;
    pub const CHUNK_SIZE: u32 = 1 << CHUNK_SIZE_EXP;
    pub const CHUNK_BOUNDS: UVec3 = UVec3::new(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    pub const CHUNK_BLOCK_COUNT: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;


    // #[derive(Debug)]
    pub struct VoxelWorld {
        // chunks: Vec<VoxelChunk>,
        loaded_chunks: HashMap<IVec3, VoxelChunk>,
        requested_chunks: HashMap<IVec3, ChunkRequest>,
        chunk_load_queue: Vec<IVec3>,
        chunk_unload_queue: Vec<IVec3>,
        chunk_load_center_pos: IVec3,
        player_chunk_pos: IVec3,
        unloaded_block_edits: HashMap<IVec3, HashMap<UVec3, u32>>,
        chunk_load_radius: u32,
    }

    // #[derive(Debug)]
    pub struct VoxelChunk {
        // entity: Entity<'static>,
        entity_id: bevy_ecs::entity::Entity,
        chunk_pos: IVec3,
        blocks: [u32; CHUNK_BLOCK_COUNT],
        dirty: bool,
        updated_mesh_data: Option<MeshData<BaseVertex>>,
        mesh: Option<Arc<Mesh<BaseVertex>>>
    }

    #[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq)]
    enum ChunkRequest {
        Load,
        Unload,
        LoadPending,
        UnloadPending,
    }

    impl VoxelWorld {
        pub fn new() -> Self {
            VoxelWorld {
                // chunks: vec![],
                loaded_chunks: HashMap::new(),
                requested_chunks: HashMap::new(),
                chunk_load_queue: Vec::new(),
                chunk_unload_queue: Vec::new(),
                chunk_load_center_pos: IVec3::ZERO,
                player_chunk_pos: IVec3::MAX,
                unloaded_block_edits: HashMap::new(),
                chunk_load_radius: 12,
            }
        }

        pub fn update(&mut self, engine: &mut Engine) -> Result<()> {
            self.update_requested_chunks();
            self.update_chunk_load_queue(engine);

            let mut staging_size = 0;

            for (_chunk_pos, chunk) in &mut self.loaded_chunks {
                // let chunk = &mut self.chunks[*index];
                chunk.update_mesh_data()?;
                staging_size += chunk.get_staging_buffer_size();
            }

            if staging_size > 0 {
                let mut cmd_buf = engine.graphics.begin_transfer_commands()?;

                let allocator = engine.graphics.memory_allocator();

                let mut staging_buffer = GraphicsManager::create_staging_subbuffer(allocator, staging_size)?;
                let mut subbuffer = Some(staging_buffer.clone());

                for (_chunk_pos, chunk) in &mut self.loaded_chunks {
                    chunk.update_buffers(&mut cmd_buf, &mut subbuffer, engine)?;
                }

                engine.graphics.submit_transfer_commands(cmd_buf)?
                    .wait(None)?;
            }
            Ok(())
        }

        fn update_requested_chunks(&mut self) {
            for (chunk_pos, request) in &mut self.requested_chunks {

                if *request == ChunkRequest::Load {
                    *request = ChunkRequest::LoadPending;
                    self.chunk_load_queue.push(*chunk_pos);

                } else if *request == ChunkRequest::Unload {
                    *request = ChunkRequest::UnloadPending;
                    self.chunk_unload_queue.push(*chunk_pos);
                }
            }
        }

        fn update_chunk_load_queue(&mut self, engine: &mut Engine) {
            if self.chunk_load_center_pos != self.player_chunk_pos {
                debug!("Player moved chunks: {} -> {}", self.chunk_load_center_pos, self.player_chunk_pos);
                self.chunk_load_center_pos = self.player_chunk_pos;

                let center_pos = self.chunk_load_center_pos.as_dvec3();

                // Closest chunks to player end up at the back of the list (loaded first)
                self.chunk_load_queue.sort_by(|pos1, pos2| {
                    let mut d1 = center_pos - pos1.as_dvec3();
                    let mut d2 = center_pos - pos2.as_dvec3();
                    d1.y *= 2.0; // Bias the loading order to prioritise horizontal distance, vertical difference is less important.
                    d2.y *= 2.0;
                    let dist1 = d1.length_squared();
                    let dist2 = d2.length_squared();
                    dist2.total_cmp(&dist1)
                });

                // Furthest chunks from the player end up at the back of the list (unloaded first)
                self.chunk_unload_queue.sort_by(|pos1, pos2| {
                    let dist1 = pos1.as_dvec3().distance_squared(center_pos);
                    let dist2 = pos2.as_dvec3().distance_squared(center_pos);
                    dist1.total_cmp(&dist2)
                })
            }

            if !self.chunk_load_queue.is_empty() {
                let chunk_pos = self.chunk_load_queue.pop().unwrap();

                let r = self.chunk_load_radius as f32;
                let r2 = r * r;

                if distance_sq_between_chunks(chunk_pos, self.chunk_load_center_pos) >= r2 {
                    // chunk_load_queue is sorted by distance. If we encounter a chunk too far away, all remaining chunks are also too far away.
                    self.chunk_load_queue.clear();
                }

                if !self.is_chunk_unload_requested(&chunk_pos) {
                    self.load_chunk(engine, chunk_pos);
                }

                self.requested_chunks.remove(&chunk_pos);
            }
            
            for _ in 0..100 {
                if self.chunk_unload_queue.is_empty() {
                    break;
                }
                
                let chunk_pos = self.chunk_unload_queue.pop().unwrap();
                debug!("Unloading chunk {chunk_pos}");
                if !self.is_chunk_load_requested(&chunk_pos) {
                    self.unload_chunk(engine, chunk_pos);
                }
                
                self.requested_chunks.remove(&chunk_pos);
            }
        }

        pub fn update_player_position(&mut self, player_pos: Vec3) {
            let chunk_pos = get_chunk_pos_for_world_pos(player_pos);

            if self.player_chunk_pos != chunk_pos {
                debug!("Player moved chunk {} -> {}", self.player_chunk_pos, chunk_pos);
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

            for (chunk_pos, index) in &self.loaded_chunks {

                if self.is_chunk_unload_requested(chunk_pos) {
                    continue;
                }

                let pos = chunk_pos.as_vec3();

                let d2 = pos.distance_squared(center_pos);
                if d2 >= r2 {
                    unload_chunk_positions.push(*chunk_pos);
                }
            }

            debug!("Unloading {} chunks", unload_chunk_positions.len());

            for chunk_pos in unload_chunk_positions {
                self.request_unload_chunk(chunk_pos);
            }
        }

        fn load_chunk(&mut self, engine: &mut Engine, chunk_pos: IVec3) {
            let mut chunk = VoxelChunk::new(chunk_pos, engine);

            // let index = self.chunks.len();
            // self.chunks.push(chunk);

            if let Some(chunk_edits) = self.unloaded_block_edits.remove(&chunk_pos) {
                chunk.load_edits(chunk_edits);
            }
            self.loaded_chunks.insert(chunk_pos, chunk);
        }

        fn unload_chunk(&mut self, engine: &mut Engine, chunk_pos: IVec3) {
            if let Some(chunk) = self.loaded_chunks.get_mut(&chunk_pos) {
                chunk.unload(engine);
            }
            self.loaded_chunks.remove(&chunk_pos);
        }

        pub fn request_load_chunk(&mut self, chunk_pos: IVec3) {
            if !self.is_chunk_loaded(chunk_pos) && !self.is_chunk_load_pending(&chunk_pos) {
                self.requested_chunks.insert(chunk_pos, ChunkRequest::Load);
            }
        }

        pub fn request_unload_chunk(&mut self, chunk_pos: IVec3) {
            if self.is_chunk_loaded(chunk_pos) && !self.is_chunk_unload_pending(&chunk_pos) {
                self.requested_chunks.insert(chunk_pos, ChunkRequest::Unload);
            }
        }

        fn get_chunk_request(&self, chunk_pos: &IVec3) -> Option<&ChunkRequest> {
            self.requested_chunks.get(&chunk_pos)
        }

        fn is_chunk_load_requested(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::Load || *req == ChunkRequest::LoadPending)
        }

        fn is_chunk_load_pending(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::LoadPending)
        }

        fn is_chunk_unload_requested(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::Unload || *req == ChunkRequest::UnloadPending)
        }

        fn is_chunk_unload_pending(&self, chunk_pos: &IVec3) -> bool {
            self.get_chunk_request(chunk_pos).map_or(false, |req| *req == ChunkRequest::UnloadPending)
        }

        pub fn is_chunk_loaded(&self, chunk_pos: IVec3) -> bool {
            self.loaded_chunks.contains_key(&chunk_pos)
        }

        pub fn get_chunk(&self, chunk_pos: IVec3) -> Option<&VoxelChunk> {
            // let index = self.loaded_chunks.get(&chunk_pos)?;
            // let chunk = self.chunks.get(*index)?;
            // Some(chunk)

            self.loaded_chunks.get(&chunk_pos)
        }

        pub fn get_chunk_mut(&mut self, chunk_pos: IVec3) -> Option<&mut VoxelChunk> {
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
    }

    impl VoxelChunk {
        fn new(chunk_pos: IVec3, engine: &mut Engine) -> Self {

            let pos = get_chunk_world_pos(chunk_pos);

            let mut entity = engine.scene.create_entity(format!("chunk({},{},{})", chunk_pos.x, chunk_pos.y, chunk_pos.z).as_str());
            entity.add_component(*Transform::new().set_translation(pos));
            entity.add_component(RenderComponent::<BaseVertex>::new(RenderType::Static, None));

            VoxelChunk {
                // entity,
                entity_id: entity.id().id(),
                chunk_pos,
                blocks: [0; CHUNK_BLOCK_COUNT],
                dirty: true,
                updated_mesh_data: None,
                mesh: None,
            }
        }

        pub fn pos(&self) -> &IVec3 {
            &self.chunk_pos
        }

        pub fn mesh(&self) -> Option<&Arc<Mesh<BaseVertex>>> {
            self.mesh.as_ref()
        }

        pub fn get_block(&self, pos: UVec3) -> u32 {
            let index = calc_index_for_coord(pos, CHUNK_BOUNDS) as usize;
            self.blocks[index]
        }

        pub fn set_block(&mut self, pos: UVec3, block: u32) {
            let index = calc_index_for_coord(pos, CHUNK_BOUNDS) as usize;
            let prev_block = self.blocks[index];
            self.blocks[index] = block;

            if prev_block != block {
                self.dirty = true;
            }
        }

        fn update_mesh_data(&mut self) -> Result<()> {
            if self.dirty {
                self.dirty = false;

                let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::TriangleList);

                let p1 = (CHUNK_SIZE as f32) * 0.5;
                mesh_data.create_cuboid([p1, p1, p1], [p1, p1, p1]);

                for x in 0..CHUNK_BOUNDS.x {
                    let cx = x as f32 + 0.5;

                    for y in 0..CHUNK_BOUNDS.y {
                        let cy = y as f32 + 0.5;

                        for z in 0..CHUNK_BOUNDS.z {
                            let cz = z as f32 + 0.5;

                            let block = self.get_block(UVec3::new(x, y, z));
                            if block != 0 {
                                mesh_data.create_cuboid([cx, cy, cz], [0.5, 0.5, 0.5]);
                            }
                        }
                    }
                }

                self.updated_mesh_data = Some(mesh_data);
            }

            Ok(())
        }

        fn update_buffers(&mut self, cmd_buf: &mut CommandBuffer, staging_buffer: &mut Option<Subbuffer<[u8]>>, engine: &mut Engine) -> Result<()> {
            let staging_size = self.get_staging_buffer_size();
            if let Some(mesh_data) = self.updated_mesh_data.take() {

                let allocator = engine.graphics.memory_allocator();

                let staging_buffer = chop_buffer_at(staging_buffer, staging_size).unwrap();

                let mesh = Arc::new(mesh_data.build_mesh_staged(allocator, cmd_buf, &staging_buffer)?);

                engine.scene.ecs.modify_component(self.entity_id, |render_component: &mut RenderComponent<BaseVertex>| {
                    render_component.mesh = Some(mesh.clone());
                })?;

                self.mesh = Some(mesh);
            }

            Ok(())
        }

        fn get_staging_buffer_size(&self) -> DeviceSize {
            if let Some(mesh_data) = self.updated_mesh_data.as_ref() {
                mesh_data.get_required_staging_buffer_size()
            } else {
                0
            }
        }

        fn unload(&mut self, engine: &mut Engine) {
            engine.scene.ecs.despawn(self.entity_id);
        }

        fn load_edits(&mut self, block_edits: HashMap<UVec3, u32>) {
            if !block_edits.is_empty() {
                debug!("Loading {} block edits for chunk ({}, {}, {})", block_edits.len(), self.chunk_pos.x, self.chunk_pos.y, self.chunk_pos.z);
            }
            for (pos, block) in block_edits {
                self.set_block(pos, block);
            }
        }
    }

    impl Drop for VoxelChunk {
        fn drop(&mut self) {
            debug!("Dropping chunk: {:?}", self.chunk_pos);
        }
    }



    pub fn calc_index_for_coord(coord: UVec3, bounds: UVec3) -> u32 {
        let x = coord.x;
        let y = coord.y;
        let z = coord.z;
        let sx = bounds.x;
        let sy = bounds.y;

        x + (y * sx) + (z * sx * sy)
    }

    pub fn calc_coord_for_index(index: u32, bounds: UVec3) -> UVec3 {
        let sx = bounds.x;
        let sy = bounds.y;
        let x = index % sx;
        let y  = (index / sx) % sy;
        let z = index / (sx * sy);

        UVec3::new(x, y, z)
    }

    pub fn get_block_pos_for_world_pos(world_pos: Vec3) -> IVec3 {
        let x = f32::floor(world_pos.x) as i32;
        let y = f32::floor(world_pos.y) as i32;
        let z = f32::floor(world_pos.z) as i32;
        IVec3::new(x, y, z)
    }

    pub fn get_chunk_pos_for_world_pos(world_pos: Vec3) -> IVec3 {
        let x = f32::floor(world_pos.x / CHUNK_BOUNDS.x as f32) as i32;
        let y = f32::floor(world_pos.y / CHUNK_BOUNDS.y as f32) as i32;
        let z = f32::floor(world_pos.z / CHUNK_BOUNDS.z as f32) as i32;
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

    pub fn get_chunk_world_pos(chunk_pos: IVec3) -> Vec3 {
        (chunk_pos * CHUNK_BOUNDS.as_ivec3()).as_vec3()
    }

    pub fn get_chunk_center_world_pos(chunk_pos: IVec3) -> Vec3 {
        get_chunk_world_pos(chunk_pos) + CHUNK_BOUNDS.as_vec3() * 0.5
    }

    pub fn distance_sq_chunk_center_to_world_pos(chunk_pos: IVec3, world_pos: Vec3) -> f32 {
        Vec3::distance_squared(get_chunk_center_world_pos(chunk_pos), world_pos)
    }

    pub fn distance_sq_between_chunks(chunk_pos1: IVec3, chunk_pos2: IVec3) -> f32 {
        Vec3::distance_squared(chunk_pos1.as_vec3(), chunk_pos2.as_vec3())
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
