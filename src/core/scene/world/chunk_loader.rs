use std::sync::{Arc, Mutex};
use std::{array, iter, mem, slice, thread};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use anyhow::anyhow;
use anyhow::Result;
use glam::{DVec3, IVec3};
use log::{debug, info};
use crate::core::scene::world::{distance_sq_between_chunks, VoxelChunkData};
use crate::core::{Engine, WorldGenerator};

type ThreadHandle = JoinHandle<()>;

pub struct ChunkLoader {
    threads: Vec<ChunkLoaderThreadWrapper>,
    shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>,
    world_generator: Arc<WorldGenerator>,
    chunk_generate_request_count: usize,
    chunk_mesh_request_count: usize,
    canceled_chunks_count: usize,
}

struct ChunkLoaderSharedContext {
    chunk_generate_queue: VecDeque<ChunkLoadTask>,
    chunk_mesh_queue: VecDeque<ChunkLoadTask>,
    chunk_complete_queue: Vec<Box<VoxelChunkData>>,
    chunk_canceled_queue: Vec<ChunkLoadTask>,
    new_completed_chunks: bool,
    new_canceled_chunks: bool,
    center_chunk_pos: IVec3,
    chunk_load_radius: f64,
    // engine: Arc<Engine>,
    changed: bool,
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
        chunk_data: Box<VoxelChunkData>,
        neighbour_chunks: [Option<Box<VoxelChunkData>>; 6]
    },
}

impl ChunkLoadTask {
    pub fn chunk_pos(&self) -> &IVec3 {
        match self {
            ChunkLoadTask::GenerateVoxels { chunk_pos } => chunk_pos,
            ChunkLoadTask::GenerateMesh { chunk_data, .. } => chunk_data.chunk_pos()
        }
    }
}

// unsafe impl Send for ChunkLoaderThreadWrapper {}
// unsafe impl Sync for ChunkLoaderThreadWrapper {}
// unsafe impl Send for ChunkLoaderThreadContext {}
// unsafe impl Sync for ChunkLoaderThreadContext {}

impl ChunkLoader {
    pub fn new(world_generator: Arc<WorldGenerator>) -> Self {
        let shared_ctx = ChunkLoaderSharedContext {
            chunk_generate_queue: VecDeque::new(),
            chunk_mesh_queue: VecDeque::new(),
            chunk_complete_queue: vec![],
            chunk_canceled_queue: vec![],
            new_completed_chunks: false,
            new_canceled_chunks: false,
            center_chunk_pos: IVec3::ZERO,
            chunk_load_radius: 0.0,
            changed: false,
        };

        ChunkLoader{
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
        self.request_generate_chunks(vec![chunk_pos])
    }

    pub fn request_generate_chunks(&mut self, chunk_positions: Vec<IVec3>) -> Result<()> {
        if chunk_positions.is_empty() {
            return Ok(())
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error requesting chunk load: could not lock mutex: {e}"))?;

        for chunk_pos in chunk_positions {
            ctx.chunk_generate_queue.push_back(ChunkLoadTask::GenerateVoxels {
                chunk_pos
            });
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
    where F: FnMut(Box<VoxelChunkData>) {
        if count == 0 {
            return Ok(());
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;

        let center_pos = ctx.center_chunk_pos.as_dvec3();

        if ctx.new_completed_chunks {
            ctx.new_completed_chunks = false;

            // Sorted so that the closest chunks are at the end of the list (pop() first)
            ctx.chunk_complete_queue.sort_unstable_by(|chunk1, chunk2| {
                let dist1 = (center_pos - chunk1.chunk_pos().as_dvec3()).length_squared();
                let dist2 = (center_pos - chunk2.chunk_pos().as_dvec3()).length_squared();
                dist2.total_cmp(&dist1)
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
            let center_pos = ctx.center_chunk_pos.as_dvec3();

            if ctx.new_canceled_chunks {
                ctx.new_canceled_chunks = false;

                // Sorted so that the furthest positions are at the end of the list (pop() first)
                ctx.chunk_canceled_queue.sort_unstable_by(|pos1, pos2| {
                    let dist1 = (center_pos - pos1.chunk_pos().as_dvec3()).length_squared();
                    let dist2 = (center_pos - pos2.chunk_pos().as_dvec3()).length_squared();
                    dist1.total_cmp(&dist2)
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

    fn exec(thread_ctx: Arc<ChunkLoaderThreadContext>, shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>) {
        let mut generate_count = 0;
        let mut mesh_count = 0;
        let mut start_time = Instant::now();
        loop {
            {
                let stopped = thread_ctx.stopped.lock().unwrap();
                if *stopped {
                    break;
                }
            }

            let dur = start_time.elapsed().as_secs_f64() * 1000.0;

            if dur >= 10.0 {
                thread::sleep(Duration::from_millis(4));
                start_time = Instant::now();
                continue;
            }

            if let Some(mut chunk) = Self::get_next_chunk_to_mesh(&shared_ctx) {

                if mesh_count == 0 || generate_count == 0 {
                    start_time = Instant::now();
                }
                mesh_count += 1;

                // Build chunk mesh
                Self::build_chunk_mesh(&mut chunk);

                let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");
                ctx.chunk_complete_queue.push(chunk);
                ctx.new_completed_chunks = true;

            } else if let Some(chunk_pos) = Self::get_next_chunk_generate_pos(&shared_ctx) {

                if mesh_count == 0 || generate_count == 0 {
                    start_time = Instant::now();
                }
                generate_count += 1;

                // Allocate chunk
                let mut chunk = Box::new(VoxelChunkData::new(chunk_pos));

                // Generate chunk data
                Self::generate_chunk_data(&mut chunk, &thread_ctx);

                let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");
                ctx.chunk_mesh_queue.push_back(ChunkLoadTask::GenerateMesh {
                    chunk_data: chunk,
                    neighbour_chunks: array::from_fn(|_| None)
                });


            } else {
                if generate_count > 0 || mesh_count > 0 {
                    let thread_name = thread::current().name().map_or_else(
                        || format!("UnnamedThread-{:?}", thread::current().id()),
                        |e| e.to_string());

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

    fn get_next_chunk_generate_pos(shared_ctx: &Arc<Mutex<ChunkLoaderSharedContext>>) -> Option<IVec3> {
        let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

        // ctx.chunk_load_queue.pop_front()

        if ctx.chunk_generate_queue.is_empty() {
            return None;
        }

        ctx.changed = true;
        let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;

        let mut skip_count = 0;

        while let Some(task) = ctx.chunk_generate_queue.pop_front() {
            let chunk_pos = task.chunk_pos();

            if distance_sq_between_chunks(ctx.center_chunk_pos, *chunk_pos) < r2 {
                // Return the first position within range
                return Some(*chunk_pos);

            } else {
                // This position is out of range, so ensure a Canceled state is returned later.
                ctx.chunk_canceled_queue.push(task);
                ctx.new_canceled_chunks = true;
            }
            skip_count += 1;
        }
        if skip_count > 0 {
            debug!("ChunkLoader::get_next_chunk_generate_pos() - {skip_count} queued generate positions were out of range");
        }
        None // None were within range, return none.
    }

    fn get_next_chunk_to_mesh(shared_ctx: &Arc<Mutex<ChunkLoaderSharedContext>>) -> Option<Box<VoxelChunkData>> {
        let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");

        if ctx.chunk_mesh_queue.is_empty() {
            return None;
        }

        ctx.changed = true;
        let r2 = ctx.chunk_load_radius * ctx.chunk_load_radius;

        let mut skip_count = 0;

        while let Some(task) = ctx.chunk_mesh_queue.pop_front() {
            if let ChunkLoadTask::GenerateMesh { chunk_data, neighbour_chunks } = task {

                if distance_sq_between_chunks(ctx.center_chunk_pos, *chunk_data.chunk_pos()) < r2 {
                    // Return the first position within range
                    return Some(chunk_data);

                } else {
                    // This position is out of range, so ensure a Canceled state is returned later.
                    ctx.chunk_canceled_queue.push(ChunkLoadTask::GenerateMesh {
                        chunk_data,
                        neighbour_chunks
                    });
                    ctx.new_canceled_chunks = true;
                }
            }

            skip_count += 1;
        }
        if skip_count > 0 {
            debug!("ChunkLoader::get_next_chunk_to_mesh() - {skip_count} queued mesh positions were out of range (WorldGen work was wasted)");
        }

        None // None were within range, return none.
    }

    fn generate_chunk_data(chunk: &mut Box<VoxelChunkData>, thread_ctx: &Arc<ChunkLoaderThreadContext>) {

        let world_generator = &thread_ctx.world_generator;

        world_generator.load_chunk(chunk).expect("ChunkLoader - Failed to load chunk");
    }

    fn build_chunk_mesh(chunk: &mut Box<VoxelChunkData>) {
        chunk.update_mesh_data().expect("ChunkLoader - Failed to build chunk mesh");
    }

    fn chunk_load_sort_comparator(center_pos: &DVec3, pos1: DVec3, pos2: DVec3) -> Ordering {
        let mut d1 = center_pos - pos1;
        let mut d2 = center_pos - pos2;
        d1.y *= 2.0; // Bias the loading order to prioritise horizontal distance, vertical difference is less important.
        d2.y *= 2.0;
        let dist1 = d1.length_squared();
        let dist2 = d2.length_squared();
        dist1.total_cmp(&dist2)
    }

    /// Update the queues of chunks to load & unload. Keep the queues sorted by distance around the player position
    /// The load queue is sorted so the nearest chunks are loaded first, and the unload queue is sorted in the
    /// opposite direction.
    /// Can pass cancelled_fn as None::<fn(_)> if needed
    pub fn update_chunk_queues<F>(&mut self, center_chunk_pos: IVec3, chunk_load_radius: u32, mut canceled_callback_fn: Option<F>) -> Result<()>
    where F: FnMut(IVec3) {

        let center_pos = center_chunk_pos.as_dvec3();

        let mut shared_ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Unable to update chunk queue - failed to lock mutex for shared context: {e}"))?;

        shared_ctx.center_chunk_pos = center_chunk_pos;
        shared_ctx.chunk_load_radius = chunk_load_radius as f64;
        let r2 = shared_ctx.chunk_load_radius * shared_ctx.chunk_load_radius;

        // Sorted so that the closest chunks are at the front of the queue (pop_front() first)
        shared_ctx.chunk_generate_queue.make_contiguous().sort_by(|pos1, pos2| {
            Self::chunk_load_sort_comparator(&center_pos, pos1.chunk_pos().as_dvec3(), pos2.chunk_pos().as_dvec3())
        });
        shared_ctx.chunk_mesh_queue.make_contiguous().sort_by(|chunk1, chunk2| {
            Self::chunk_load_sort_comparator(&center_pos, chunk1.chunk_pos().as_dvec3(), chunk2.chunk_pos().as_dvec3())
        });

        let generate_queue_canceled_count = Self::cleanup_chunk_generate_queue(center_chunk_pos, shared_ctx.chunk_load_radius, &mut shared_ctx.chunk_generate_queue, |chunk_pos| {
            // shared_ctx.chunk_canceled_queue.push(chunk_pos);
            if let Some(callback) = &mut canceled_callback_fn {
                callback(chunk_pos);
            }
        });

        let mesh_queue_canceled_count = Self::cleanup_chunk_mesh_queue(center_chunk_pos, shared_ctx.chunk_load_radius, &mut shared_ctx.chunk_mesh_queue, |chunk_pos| {
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

        Ok(())
    }

    fn cleanup_chunk_generate_queue<F>(center_chunk_pos: IVec3, chunk_load_radius: f64, chunk_generate_queue: &mut VecDeque<ChunkLoadTask>, mut canceled_callback_fn: F) -> usize
    where F: FnMut(IVec3) {
        let r2 = chunk_load_radius * chunk_load_radius;
        let mut canceled_count = 0;

        for (index, task) in chunk_generate_queue.iter().enumerate() {
            let chunk_pos = task.chunk_pos();

            if distance_sq_between_chunks(*chunk_pos, center_chunk_pos) >= r2 {
                // The queue is sorted by distance. If we encounter a chunk too far away, all remaining chunks are also too far away.
                // Pop off all remaining items at the end of the queue
                while chunk_generate_queue.len() > index {
                    if let Some(task) = chunk_generate_queue.pop_back() {
                        canceled_callback_fn(*task.chunk_pos());
                    }
                    canceled_count += 1;
                }
                break;
            }
        }

        canceled_count
    }

    fn cleanup_chunk_mesh_queue<F>(center_chunk_pos: IVec3, chunk_load_radius: f64, chunk_mesh_queue: &mut VecDeque<ChunkLoadTask>, mut canceled_callback_fn: F) -> usize
    where F: FnMut(IVec3) {
        let r2 = chunk_load_radius * chunk_load_radius;
        let mut canceled_count = 0;

        for (index, task) in chunk_mesh_queue.iter().enumerate() {

            if distance_sq_between_chunks(*task.chunk_pos(), center_chunk_pos) >= r2 {
                // The queue is sorted by distance. If we encounter a chunk too far away, all remaining chunks are also too far away.
                // Pop off all remaining items at the end of the queue
                while chunk_mesh_queue.len() > index {
                    if let Some(chunk) = chunk_mesh_queue.pop_back() {
                        canceled_callback_fn(*chunk.chunk_pos());
                    }
                    canceled_count += 1;
                }
                break;
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
}

impl Drop for ChunkLoaderThreadWrapper {
    fn drop(&mut self) {
        info!("ChunkLoader - Dropping thread");
        let t0 = Instant::now();

        {
            let mut stopped = self.thread_ctx.stopped.lock().expect("Failed to lock thread_ctx");
            *stopped = true;
        }

        let thread_handle = self.thread_handle.take().unwrap();
        thread_handle.join().expect("Failed to join thread");

        let dur = t0.elapsed();
        info!("Took {} msec to wait for thread to finish", dur.as_secs_f64() * 1000.0);
    }
}