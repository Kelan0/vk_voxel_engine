use std::sync::{Arc, Mutex, MutexGuard};
use std::{array, iter, mem, slice, thread};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use anyhow::anyhow;
use anyhow::Result;
use foldhash::{HashMap, HashMapExt};
use glam::{DVec3, IVec3};
use log::{debug, info};
use rayon::slice::ParallelSliceMut;
use crate::core::scene::world::{distance_sq_between_chunks, VoxelChunkData};
use crate::core::{AxisDirection, Engine, WorldGenerator};
use crate::core::world::ChunkStage;

type ThreadHandle = JoinHandle<()>;

pub struct ChunkManager {
    threads: Vec<ChunkLoaderThreadWrapper>,
    shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>,
    world_generator: Arc<WorldGenerator>,
    chunk_generate_request_count: usize,
    chunk_mesh_request_count: usize,
    canceled_chunks_count: usize,
}

struct ChunkLoaderSharedContext {
    chunk_tasks: HashMap<IVec3, ChunkLoadTask>,
    chunk_generate_queue: VecDeque<IVec3>,
    chunk_mesh_queue: VecDeque<IVec3>,
    chunk_complete_queue: Vec<ChunkTaskResult>,
    // generated_chunk_neighbours: HashMap<IVec3, [Option<NeighbourChunkData>; 6]>,
    chunk_canceled_queue: Vec<ChunkLoadTask>,
    new_completed_chunks: bool,
    new_canceled_chunks: bool,
    center_chunk_pos: IVec3,
    chunk_load_radius: f64,
    // engine: Arc<Engine>,
    queues_ready: bool,
}

enum NeighbourChunkData {
    NotEmpty(VoxelChunkData),
    Empty
}

struct ChunkLoaderThreadContext {
    stopped: Mutex<bool>,
    world_generator: Arc<WorldGenerator>
}

struct ChunkLoaderThreadWrapper {
    thread_handle: Option<ThreadHandle>,
    thread_ctx: Arc<ChunkLoaderThreadContext>,
}

// pub enum ChunkLoadResult {
//     Complete(Box<VoxelChunkData>),
//     Canceled(IVec3)
// }
//
// pub enum ChunkLoadRequest {
//     Generate(IVec3),
//     Mesh(Box<VoxelChunkData>),
// }
pub enum ChunkLoadTask {
    GenerateVoxels{
        chunk_pos: IVec3
    },
    GenerateMesh{
        chunk_data: VoxelChunkData,
        neighbours: [Option<VoxelChunkData>; 6],
    },
}

pub enum ChunkTaskResult {
    GenerateVoxels {
        chunk_data: VoxelChunkData,
    },
    GenerateMesh {
        chunk_data: VoxelChunkData,
    }
}

impl ChunkLoadTask {
    pub fn chunk_pos(&self) -> &IVec3 {
        match self {
            ChunkLoadTask::GenerateVoxels { chunk_pos } => chunk_pos,
            ChunkLoadTask::GenerateMesh { chunk_data, .. } => chunk_data.chunk_pos()
        }
    }
}

impl ChunkTaskResult {
    pub fn chunk_pos(&self) -> &IVec3 {
        match self {
            ChunkTaskResult::GenerateVoxels { chunk_data } => chunk_data.chunk_pos(),
            ChunkTaskResult::GenerateMesh { chunk_data } => chunk_data.chunk_pos()
        }
    }
}

// unsafe impl Send for ChunkLoaderThreadWrapper {}
// unsafe impl Sync for ChunkLoaderThreadWrapper {}
// unsafe impl Send for ChunkLoaderThreadContext {}
// unsafe impl Sync for ChunkLoaderThreadContext {}

impl ChunkManager {
    pub fn new(world_generator: Arc<WorldGenerator>) -> Self {
        let shared_ctx = ChunkLoaderSharedContext {
            chunk_tasks: HashMap::new(),
            chunk_generate_queue: VecDeque::new(),
            chunk_mesh_queue: VecDeque::new(),
            chunk_complete_queue: vec![],
            chunk_canceled_queue: vec![],
            new_completed_chunks: false,
            new_canceled_chunks: false,
            // generated_chunk_neighbours: HashMap::new(),
            center_chunk_pos: IVec3::ZERO,
            chunk_load_radius: 0.0,
            queues_ready: false,
        };

        ChunkManager {
            threads: vec![],
            shared_ctx: Arc::new(Mutex::new(shared_ctx)),
            world_generator,
            chunk_generate_request_count: 0,
            chunk_mesh_request_count: 0,
            canceled_chunks_count: 0,
        }
    }

    pub fn set_thread_count(&mut self, thread_count: usize) {
        info!("ChunkLoader - Initializing thread count: {}", thread_count);
        let _lock = self.shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

        let mut threads = mem::take(&mut self.threads);

        while threads.len() < thread_count {
            // We are increasing the number of threads
            let index = threads.len();
            let shared_ctx = self.shared_ctx.clone();
            let thread = Self::init_thread(index, shared_ctx, self.world_generator.clone());
            threads.push(thread);
        }

        while threads.len() > thread_count {
            // We are decreasing the number of threads
            // TODO: any de-initialization here?
            threads.pop();
        }

        mem::swap(&mut self.threads, &mut threads);
    }

    pub fn update(&mut self) {
        if let Ok(ctx) = self.shared_ctx.lock() {
            self.chunk_generate_request_count = ctx.chunk_generate_queue.len();
            self.chunk_mesh_request_count = ctx.chunk_mesh_queue.len();
            self.canceled_chunks_count = ctx.chunk_canceled_queue.len();
        }
    }

    pub fn request_generate_chunk(&mut self, chunk_pos: IVec3) -> Result<()> {
        self.request_generate_chunks(slice::from_ref(&chunk_pos))
    }

    pub fn request_generate_chunks(&mut self, chunk_positions: &[IVec3]) -> Result<()> {
        if chunk_positions.is_empty() {
            return Ok(())
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error requesting chunk load: could not lock mutex: {e}"))?;

        for &chunk_pos in chunk_positions {
            ctx.chunk_tasks.insert(chunk_pos, ChunkLoadTask::GenerateVoxels {
                chunk_pos
            });
            ctx.chunk_generate_queue.push_back(chunk_pos);
        }

        self.chunk_generate_request_count = ctx.chunk_generate_queue.len();
        self.chunk_mesh_request_count = ctx.chunk_mesh_queue.len();
        self.canceled_chunks_count = ctx.chunk_canceled_queue.len();
        Ok(())
    }

    pub fn request_mesh_chunk(&mut self, chunk: VoxelChunkData, neighbours: [Option<&VoxelChunkData>; 6]) -> Result<()> {
        self.request_mesh_chunks(slice::from_ref(&(chunk, neighbours)))
    }

    pub fn request_mesh_chunks(&mut self, chunks: &[(VoxelChunkData, [Option<&VoxelChunkData>; 6])]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(())
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error requesting chunk load: could not lock mutex: {e}"))?;

        for (chunk_data, neighbours) in chunks.iter() {
            let mut temp_neighbours: [Option<VoxelChunkData>; 6] = Default::default();
            for i in 0..6 {
                temp_neighbours[i] = neighbours[i].cloned();
            }
            let chunk_pos = *chunk_data.chunk_pos();
            ctx.chunk_tasks.insert(chunk_pos, ChunkLoadTask::GenerateMesh {
                chunk_data: chunk_data.clone(),
                neighbours: temp_neighbours,
            });
            ctx.chunk_mesh_queue.push_back(chunk_pos);
        }

        self.chunk_generate_request_count = ctx.chunk_generate_queue.len();
        self.chunk_mesh_request_count = ctx.chunk_mesh_queue.len();
        self.canceled_chunks_count = ctx.chunk_canceled_queue.len();

        Ok(())
    }

    pub fn num_completed_chunks(&self) -> Result<usize> {
        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;

        Ok(ctx.chunk_complete_queue.len())
    }

    pub fn drain_completed_chunks<F>(&mut self, mut count: usize, mut callback_fn: F) -> Result<()>
    where F: FnMut(ChunkTaskResult) {
        if count == 0 {
            return Ok(());
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;

        let center_pos = ctx.center_chunk_pos;

        if ctx.new_completed_chunks {
            ctx.new_completed_chunks = false;

            // Sorted so that the closest chunks are at the end of the list (pop() first)
            ctx.chunk_complete_queue.par_sort_unstable_by(|a, b| {
                let dist1 = (center_pos - a.chunk_pos()).as_i64vec3().length_squared();
                let dist2 = (center_pos - b.chunk_pos()).as_i64vec3().length_squared();
                dist2.cmp(&dist1)
            });
        }


        let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;
        let center_chunk_pos = ctx.center_chunk_pos;

        // Pop the closest chunks from the queue until 'count' completed ones are found, or there are none left.
        while let Some(result) = ctx.chunk_complete_queue.pop() {
            callback_fn(result);
            // out_chunks.push(result);

            count -= 1;
            if count == 0 {
                break;
            }
        }


        // Find the index where all elements after are within the load radius, and all elements before are outside the load radius
        let idx = ctx.chunk_complete_queue.partition_point(|chunk| {
            let dist_sq = distance_sq_between_chunks(*chunk.chunk_pos(), center_chunk_pos);
            dist_sq > r2
        });

        // De-allocate the chunks for elements outside the load radius
        for i in 0..idx {
            let chunk_pos = *ctx.chunk_complete_queue[i].chunk_pos();
            ctx.chunk_canceled_queue.push(ChunkLoadTask::GenerateVoxels {
                chunk_pos
            });
            ctx.new_canceled_chunks = true;
        }
        ctx.chunk_complete_queue.drain(0..idx);

        if idx > 0 {
            debug!("ChunkLoader::drain_completed_chunks() - {idx} loaded chunks from drain queue were out of range - !!!WASTED WORK :(!!!")
        }

        // let idx = ctx.chunk_complete_queue.len() - usize::min(count, ctx.chunk_complete_queue.len());
        // for chunk in ctx.chunk_complete_queue.drain(idx..) {
        //     if distance_sq_between_chunks(*chunk.chunk_pos(), center_chunk_pos) > r2 {
        //         clear_queue = true;
        //         break;
        //     }
        //     out_chunks.push(chunk);
        // }
        //
        //
        // if clear_queue {
        //     debug!("ChunkLoader::drain_completed_chunks() - Cleared {} chunks out of range", ctx.chunk_complete_queue.len());
        //     ctx.chunk_complete_queue.clear();
        // }
        Ok(())
    }

    pub fn drain_canceled_chunks<F>(&mut self, mut count: usize, sorted: bool, mut callback_fn: F) -> Result<()>
    where F: FnMut(IVec3) {
        if count == 0 {
            return Ok(());
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;

        if sorted {
            let center_pos = ctx.center_chunk_pos;

            if ctx.new_canceled_chunks {
                ctx.new_canceled_chunks = false;

                // Sorted so that the furthest positions are at the end of the list (pop() first)
                ctx.chunk_canceled_queue.par_sort_unstable_by(|pos1, pos2| {
                    let dist1 = (center_pos - pos1.chunk_pos()).as_i64vec3().length_squared();
                    let dist2 = (center_pos - pos2.chunk_pos()).as_i64vec3().length_squared();
                    dist1.cmp(&dist2)
                });
            }
        }

        // Pop the chunks from the queue until 'count' elements were removed, or there are none left.
        while let Some(chunk_pos) = ctx.chunk_canceled_queue.pop() {
            // out_chunk_positions.push(chunk_pos);
            callback_fn(chunk_pos.chunk_pos().clone());

            count -= 1;
            if count == 0 {
                break;
            }
        }

        Ok(())
    }

    // pub fn drain_completed_chunks(&mut self, mut count: usize, out_chunks: &mut Vec<ChunkLoadResult>) -> Result<()> {
    //     if count == 0 {
    //         return Ok(());
    //     }
    //
    //     let mut ctx = self.shared_ctx.lock()
    //         .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;
    //
    //     let center_pos = ctx.center_chunk_pos.as_dvec3();
    //
    //     // Sorted so that the closest chunks are at the end of the list (pop() first)
    //     ctx.chunk_complete_queue.sort_by(|chunk1, chunk2| {
    //         let dist1 = if let ChunkLoadResult::Complete(chunk) = chunk1 {
    //             (center_pos - chunk.chunk_pos().as_dvec3()).length_squared()
    //         } else {
    //             -1.0 // Canceled chunks always popped first
    //         };
    //
    //         let dist2 = if let ChunkLoadResult::Complete(chunk) = chunk2 {
    //             (center_pos - chunk.chunk_pos().as_dvec3()).length_squared()
    //         } else {
    //             -1.0
    //         };
    //         dist2.total_cmp(&dist1)
    //     });
    //
    //
    //     let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;
    //     let center_chunk_pos = ctx.center_chunk_pos;
    //
    //     // Pop the closest chunks from the queue until 'count' completed ones are found, or there are none left.
    //     while let Some(result) = ctx.chunk_complete_queue.pop() {
    //         if matches!(result, ChunkLoadResult::Complete(_)) {
    //             count -= 1;
    //         }
    //         out_chunks.push(result);
    //
    //         if count == 0 {
    //             break;
    //         }
    //     }
    //
    //
    //     // Find the index where all elements after are within the load radius, and all elements before are outside the load radius
    //     let idx = ctx.chunk_complete_queue.partition_point(|elem| {
    //         let distSq = if let ChunkLoadResult::Complete(chunk) = elem {
    //             distance_sq_between_chunks(*chunk.chunk_pos(), center_chunk_pos)
    //         } else {
    //             -1.0
    //         };
    //         distSq > r2
    //     });
    //
    //     // De-allocate the chunks for elements outside the load radius
    //     for i in 0..idx {
    //         let res = &mut ctx.chunk_complete_queue[i];
    //         if let ChunkLoadResult::Complete(chunk) = res {
    //             debug_assert!(distance_sq_between_chunks(*chunk.chunk_pos(), center_chunk_pos) > r2);
    //             *res = ChunkLoadResult::Canceled(*chunk.chunk_pos());
    //         }
    //     }
    //
    //     if idx > 0 {
    //         debug!("ChunkLoader::drain_completed_chunks() - {idx} loaded chunks from drain queue were out of range")
    //     }
    //
    //     // let idx = ctx.chunk_complete_queue.len() - usize::min(count, ctx.chunk_complete_queue.len());
    //     // for chunk in ctx.chunk_complete_queue.drain(idx..) {
    //     //     if distance_sq_between_chunks(*chunk.chunk_pos(), center_chunk_pos) > r2 {
    //     //         clear_queue = true;
    //     //         break;
    //     //     }
    //     //     out_chunks.push(chunk);
    //     // }
    //     //
    //     //
    //     // if clear_queue {
    //     //     debug!("ChunkLoader::drain_completed_chunks() - Cleared {} chunks out of range", ctx.chunk_complete_queue.len());
    //     //     ctx.chunk_complete_queue.clear();
    //     // }
    //     Ok(())
    // }

    fn init_thread(index: usize, shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>, world_generator: Arc<WorldGenerator>) -> ChunkLoaderThreadWrapper {
        let thread_ctx = Arc::new(ChunkLoaderThreadContext{
            stopped: Mutex::new(false),
            world_generator: world_generator.clone()
        });

        let thread_ctx_internal = thread_ctx.clone();

        let thread_handle = thread::Builder::new()
            .name(format!("ChunkLoader-Thread-{}", index))
            .spawn(|| {
                Self::exec(thread_ctx_internal, shared_ctx);
            })
            .expect("ChunkLoader - Failed to spawn ChunkLoader thread");

        ChunkLoaderThreadWrapper {
            thread_handle: Some(thread_handle),
            thread_ctx
        }
    }

    fn thread_name() -> String {
        let thread_name = thread::current().name().map_or_else(
            || format!("UnnamedThread-{:?}", thread::current().id()),
            |e| e.to_string());

        thread_name
    }

    fn exec(thread_ctx: Arc<ChunkLoaderThreadContext>, shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>) {
        let mut generate_count = 0;
        let mut mesh_count = 0;
        let mut start_time = Instant::now();

        let mut last_time = Instant::now();

        let mut generate_chunk_positions = vec![];
        let mut generate_mesh_chunks = vec![];

        let max_chunk_mesh_count = 10;
        let max_chunk_generate_count = 10;

        loop {
            {
                let stopped = thread_ctx.stopped.lock().unwrap();
                if *stopped {
                    info!("ChunkLoader thread stopping {}", Self::thread_name());
                    break;
                }
            }

            let dur = start_time.elapsed().as_secs_f64() * 1000.0;

            if dur >= 10.0 {
                thread::sleep(Duration::from_millis(4));
                start_time = Instant::now();
                continue;
            }

            if Self::get_next_chunk_to_mesh(&shared_ctx, max_chunk_mesh_count, &mut generate_mesh_chunks) > 0 {

                if mesh_count == 0 || generate_count == 0 {
                    start_time = Instant::now();
                }
                mesh_count += 1;

                while let Some((mut chunk, neighbours)) = generate_mesh_chunks.pop() {
                    // Build chunk mesh
                    Self::build_chunk_mesh(&mut chunk, neighbours);

                    let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");
                    ctx.chunk_complete_queue.push(ChunkTaskResult::GenerateMesh { chunk_data: VoxelChunkData::new_stage(chunk, ChunkStage::Ready) });
                    ctx.new_completed_chunks = true;
                }

            }

            if Self::get_next_chunk_generate_pos(&shared_ctx, max_chunk_generate_count, &mut generate_chunk_positions) > 0 {

                if mesh_count == 0 || generate_count == 0 {
                    start_time = Instant::now();
                }
                generate_count += 1;

                while let Some(chunk_pos) = generate_chunk_positions.pop() {
                    // Allocate chunk
                    let mut chunk = VoxelChunkData::new(chunk_pos);

                    // Generate chunk data
                    Self::generate_chunk_data(&mut chunk, &thread_ctx);

                    let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

                    // for i in 0..6 {
                    //     let neighbour_dir = AxisDirection::from_index(i).unwrap(); // point to neighbour from this chunk
                    //     let this_dir = neighbour_dir.opposite(); // point to this chunk from neighbour
                    //     let neighbours = ctx.generated_chunk_neighbours.entry(chunk_pos + neighbour_dir.ivec()).or_default();
                    //     neighbours[this_dir.index() as usize] = if chunk.block_count() != 0 {
                    //         Some(NeighbourChunkData::NotEmpty(chunk.clone()))
                    //     } else {
                    //         Some(NeighbourChunkData::Empty)
                    //     }
                    // }


                    ctx.chunk_complete_queue.push(ChunkTaskResult::GenerateVoxels { chunk_data: VoxelChunkData::new_stage(chunk, ChunkStage::NotMeshed) });
                    ctx.new_completed_chunks = true;


                    // if chunk.block_count() == 0 {
                    //     // This chunk has no blocks, no mesh needed, it is complete.
                    //     ctx.chunk_complete_queue.push(ChunkTaskResult::GenerateVoxels { chunk_data: chunk });
                    //     ctx.new_completed_chunks = true;
                    //
                    // } else {
                    //     // Blocks were generated for this chunk, add it to the mesh queue
                    //     ctx.chunk_tasks.insert(chunk_pos, ChunkLoadTask::GenerateMesh {
                    //         chunk_data: chunk
                    //     });
                    //     ctx.chunk_mesh_queue.push_back(chunk_pos);
                    // }
                }


            } else {
                if generate_count > 0 || mesh_count > 0 {
                    let thread_name = Self::thread_name();

                    let dur = start_time.elapsed().as_secs_f64() * 1000.0;

                    // info!("ChunkLoader thread {thread_name} Generated {generate_count} chunks, Meshed {mesh_count} chunks - Took {dur} msec");
                    generate_count = 0;
                    mesh_count = 0;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }

        info!("ChunkLoader - thread stopped");
    }

    fn get_next_chunk_generate_pos(shared_ctx: &Arc<Mutex<ChunkLoaderSharedContext>>, max_count: usize, out_positions: &mut Vec<IVec3>) -> usize {
        let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

        if !ctx.queues_ready {
            return 0;
        }

        // ctx.chunk_load_queue.pop_front()

        if ctx.chunk_generate_queue.is_empty() {
            return 0;
        }

        let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;

        let mut skip_count = 0;

        let mut count = 0;

        while let Some(chunk_pos) = ctx.chunk_generate_queue.pop_front() {
            if let Some(task) = ctx.chunk_tasks.remove(&chunk_pos) {

                if distance_sq_between_chunks(ctx.center_chunk_pos, chunk_pos) < r2 {
                    // Return the first position within range
                    out_positions.push(chunk_pos);
                    count += 1;

                    if count >= max_count {
                        break;
                    }

                } else {
                    // This position is out of range, so ensure a Canceled state is returned later.
                    ctx.chunk_canceled_queue.push(task);
                    ctx.new_canceled_chunks = true;
                    skip_count += 1;
                }
            }
        }
        if skip_count > 0 {
            debug!("ChunkLoader::get_next_chunk_generate_pos() - {skip_count} queued generate positions were out of range");
        }

        count
    }

    fn get_next_chunk_to_mesh(shared_ctx: &Arc<Mutex<ChunkLoaderSharedContext>>, max_count: usize, out_chunk_data: &mut Vec<(VoxelChunkData, [Option<VoxelChunkData>; 6])>) -> usize {
        let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

        if !ctx.queues_ready {
            return 0;
        }

        if ctx.chunk_mesh_queue.is_empty() {
            return 0;
        }

        let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;

        let mut skip_count = 0;

        let mut count = 0;

        while let Some(chunk_pos) = ctx.chunk_mesh_queue.pop_front() {
            if let Some(task) = ctx.chunk_tasks.remove(&chunk_pos) {

                if let ChunkLoadTask::GenerateMesh { chunk_data, neighbours } = task {

                    if distance_sq_between_chunks(ctx.center_chunk_pos, chunk_pos) < r2 {

                        // if !Self::has_required_neighbours_for_chunk_pos(chunk_pos, &ctx) {
                        //     // The position we popped did not have the required neighbours, push it back to the queue.
                        //     // TODO: this is not a good solution... think harder
                        //     ctx.chunk_tasks.insert(chunk_pos, ChunkLoadTask::GenerateMesh {
                        //         chunk_data,
                        //     });
                        //     ctx.chunk_mesh_queue.push_back(chunk_pos);
                        // } else {
                        //     // // Return the first position within range that has the required neighbours
                        //     // let neighbours = if let Some(neighbours) =ctx.generated_chunk_neighbours.remove(&chunk_pos) {
                        //     //     neighbours
                        //     // } else {
                        //     //     Default::default()
                        //     // };
                        //
                        //     out_chunk_data.push((chunk_data, Default::default()));
                        // }

                        out_chunk_data.push((chunk_data, neighbours));

                        count += 1;

                        if count >= max_count {
                            break;
                        }

                    } else {
                        // This position is out of range, so ensure a Canceled state is returned later.
                        ctx.chunk_canceled_queue.push(ChunkLoadTask::GenerateMesh {
                            chunk_data,
                            neighbours: Default::default()
                        });
                        // ctx.generated_chunk_neighbours.remove(&chunk_pos);
                        ctx.new_canceled_chunks = true;
                        skip_count += 1;
                    }
                }
            }
        }

        if skip_count > 0 {
            debug!("ChunkLoader::get_next_chunk_to_mesh() - {skip_count} queued mesh positions were out of range (WorldGen work was wasted)");
        }

        count
    }

    fn has_required_neighbours_for_chunk_pos(chunk_pos: IVec3, shared_ctx: &MutexGuard<ChunkLoaderSharedContext>) -> bool {
        // let r2 = shared_ctx.chunk_load_radius * shared_ctx.chunk_load_radius;
        //
        // let neighbours = shared_ctx.generated_chunk_neighbours.get(&chunk_pos);
        // if neighbours.is_none() {
        //     return false;
        // }
        // let neighbours = neighbours.unwrap();
        //
        // for i in 0..6 {
        //     let dir = AxisDirection::from_index(i).unwrap();
        //     let neighbour_pos = chunk_pos + dir.ivec();
        //     if distance_sq_between_chunks(shared_ctx.center_chunk_pos, neighbour_pos) > r2 {
        //         continue;
        //     }
        //
        //     if neighbours[i as usize].is_none() {
        //         return false;
        //     }
        // }

        true
    }

    pub fn generate_chunk_data(chunk: &mut VoxelChunkData, thread_ctx: &Arc<ChunkLoaderThreadContext>) {

        let world_generator = &thread_ctx.world_generator;

        world_generator.load_chunk(chunk).expect("ChunkLoader - Failed to load chunk");
    }

    pub fn build_chunk_mesh(chunk: &mut VoxelChunkData, neighbours: [Option<VoxelChunkData>; 6]) {
        chunk.update_mesh_data(neighbours).expect("ChunkLoader - Failed to build chunk mesh");
    }

    fn count_neighbours(neighbour_chunks: [Option<VoxelChunkData>; 6]) -> i32 {
        let mut count = 0;
        for neighbour in neighbour_chunks.iter() {
            if neighbour.is_some() {
                count += 1;
            }
        }
        count
    }

    fn chunk_load_sort_comparator(center_pos: &IVec3, pos1: &IVec3, pos2: &IVec3) -> Ordering {
        let mut d1 = (center_pos - pos1).as_i64vec3();
        let mut d2 = (center_pos - pos2).as_i64vec3();
        d1.y *= 2; // Bias the loading order to prioritise horizontal distance, vertical difference is less important.
        d2.y *= 2;
        let dist1 = d1.length_squared();
        let dist2 = d2.length_squared();
        dist1.cmp(&dist2)
    }

    /// Update the queues of chunks to load & unload. Keep the queues sorted by distance around the player position
    /// The load queue is sorted so the nearest chunks are loaded first, and the unload queue is sorted in the
    /// opposite direction.
    /// Can pass cancelled_fn as None::<fn(_)> if needed
    pub fn update_chunk_queues<F>(&mut self, center_chunk_pos: IVec3, chunk_load_radius: u32, mut canceled_callback_fn: Option<F>) -> Result<()>
    where F: FnMut(IVec3) {

        let center_pos = center_chunk_pos;

        let mut shared_ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Unable to update chunk queue - failed to lock mutex for shared context: {e}"))?;

        // info!("Neighbour chunks stored: {}", shared_ctx.generated_chunk_neighbours.len());
        
        shared_ctx.chunk_load_radius = chunk_load_radius as f64;

        if !shared_ctx.queues_ready || shared_ctx.center_chunk_pos != center_chunk_pos {
            shared_ctx.center_chunk_pos = center_chunk_pos;

            // Sorted so that the closest chunks are at the front of the queue (pop_front() first)
            shared_ctx.chunk_generate_queue.make_contiguous().par_sort_unstable_by(|pos1, pos2| {
                Self::chunk_load_sort_comparator(&center_pos, pos1, pos2)
            });
            shared_ctx.chunk_mesh_queue.make_contiguous().par_sort_unstable_by(|pos1, pos2| {
                Self::chunk_load_sort_comparator(&center_pos, pos1, pos2)
            });
        }

        let generate_queue_canceled_count = Self::cleanup_chunk_generate_queue(center_chunk_pos, &mut shared_ctx, |chunk_pos| {
            // shared_ctx.chunk_canceled_queue.push(chunk_pos);
            if let Some(callback) = &mut canceled_callback_fn {
                callback(chunk_pos);
            }
        });

        let mesh_queue_canceled_count = Self::cleanup_chunk_mesh_queue(center_chunk_pos, &mut shared_ctx, |chunk_pos| {
            // shared_ctx.chunk_canceled_queue.push(chunk_pos);
            if let Some(callback) = &mut canceled_callback_fn {
                callback(chunk_pos);
            }
        });

        if generate_queue_canceled_count > 0 || mesh_queue_canceled_count > 0 {
            debug!("=== ChunkLoader::update_chunk_queues() - Canceled {generate_queue_canceled_count} from generation queue, Canceled {mesh_queue_canceled_count} from meshing queue, {} remaining in generation queue, {} remaining in meshing queue",
                shared_ctx.chunk_generate_queue.len(),
                shared_ctx.chunk_mesh_queue.len());
        }

        shared_ctx.queues_ready = true;

        Ok(())
    }

    fn cleanup_chunk_generate_queue<F>(center_chunk_pos: IVec3, shared_ctx: &mut MutexGuard<ChunkLoaderSharedContext>, mut canceled_callback_fn: F) -> usize
    where F: FnMut(IVec3) {
        let r2 = shared_ctx.chunk_load_radius * shared_ctx.chunk_load_radius;
        let mut canceled_count = 0;

        // Find the index, after which, all chunks in the queue are beyond the load radius.
        let idx = shared_ctx.chunk_generate_queue.partition_point(|chunk_pos| {
            let dist_sq = distance_sq_between_chunks(*chunk_pos, center_chunk_pos);
            dist_sq < r2
        });

        // Cancel the chunks outside the load radius
        for i in idx..shared_ctx.chunk_generate_queue.len() {
            let chunk_pos = shared_ctx.chunk_generate_queue[i];
            shared_ctx.chunk_tasks.remove(&chunk_pos);
            canceled_callback_fn(chunk_pos);
            canceled_count += 1;
        }
        shared_ctx.chunk_generate_queue.truncate(idx);


        // In-place removal of duplicate entries.
        let mut write = 0;
        for read in 0..shared_ctx.chunk_generate_queue.len() {
            if write == 0 || shared_ctx.chunk_generate_queue[read] != shared_ctx.chunk_generate_queue[write - 1] {
                shared_ctx.chunk_generate_queue[write] = shared_ctx.chunk_generate_queue[read];
                write += 1;
            }
        }
        // let dup_count = shared_ctx.chunk_generate_queue.len() - write;
        // if canceled_count > 0 || dup_count > 0 {
        //     debug!("cleanup_chunk_generate_queue() - Canceled {canceled_count}, Removed {dup_count} duplicates");
        // }

        shared_ctx.chunk_generate_queue.truncate(write);

        canceled_count
    }

    fn cleanup_chunk_mesh_queue<F>(center_chunk_pos: IVec3, shared_ctx: &mut MutexGuard<ChunkLoaderSharedContext>, mut canceled_callback_fn: F) -> usize
    where F: FnMut(IVec3) {
        let r2 = shared_ctx.chunk_load_radius * shared_ctx.chunk_load_radius;
        let mut canceled_count = 0;

        // Find the index, after which, all chunks in the queue are beyond the load radius.
        let idx = shared_ctx.chunk_mesh_queue.partition_point(|chunk_pos| {
            let dist_sq = distance_sq_between_chunks(*chunk_pos, center_chunk_pos);
            dist_sq < r2
        });

        // Cancel the chunks outside the load radius
        for i in idx..shared_ctx.chunk_mesh_queue.len() {
            let chunk_pos = shared_ctx.chunk_mesh_queue[i];
            shared_ctx.chunk_tasks.remove(&chunk_pos);
            canceled_callback_fn(chunk_pos);
            canceled_count += 1;
        }
        shared_ctx.chunk_mesh_queue.truncate(idx);

        // In-place removal of duplicate entries.
        let mut write = 0;
        for read in 0..shared_ctx.chunk_mesh_queue.len() {
            if write == 0 || shared_ctx.chunk_mesh_queue[read] != shared_ctx.chunk_mesh_queue[write - 1] {
                shared_ctx.chunk_mesh_queue[write] = shared_ctx.chunk_mesh_queue[read];
                write += 1;
            }
        }

        canceled_count
    }

    pub fn chunk_generate_request_count(&self) -> usize {
        self.chunk_generate_request_count
    }

    pub fn chunk_mesh_request_count(&self) -> usize {
        self.chunk_mesh_request_count
    }

    pub fn canceled_chunks_count(&self) -> usize {
        self.canceled_chunks_count
    }

    pub fn stop_all(&mut self, wait: bool) {
        for thread in self.threads.iter_mut() {
            thread.stop(false);
        }
        if wait {
            for thread in self.threads.iter_mut() {
                if let Some(thread_handle) = thread.thread_handle.take() {
                    thread_handle.join().expect("Failed to join thread");
                }
            }
        }
    }
}



impl ChunkLoaderThreadWrapper {
    fn stop(&mut self, wait: bool) {
        {
            let mut stopped = self.thread_ctx.stopped.lock().expect("Failed to lock thread_ctx");
            *stopped = true;
        }

        if wait {
            if let Some(thread_handle) = self.thread_handle.take() {
                thread_handle.join().expect("Failed to join thread");
            }
        }
    }
}


impl Drop for ChunkManager {
    fn drop(&mut self) {
        info!("ChunkLoader - Dropping everything");

        let mut ctx = self.shared_ctx.lock()
            .expect("ChunkLoader - Error requesting chunk load, could not lock mutex");

        ctx.chunk_generate_queue.clear();
        ctx.chunk_mesh_queue.clear();
        ctx.chunk_canceled_queue.clear();
        ctx.chunk_complete_queue.clear();
    }
}


impl Drop for ChunkLoaderThreadWrapper {
    fn drop(&mut self) {
        info!("ChunkLoader - Dropping thread");
        let t0 = Instant::now();

        self.stop(true);

        let dur = t0.elapsed();
        info!("Took {} msec to wait for thread to finish", dur.as_secs_f64() * 1000.0);
    }
}