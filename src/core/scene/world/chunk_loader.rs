use std::sync::{Arc, Mutex};
use std::{iter, mem, slice, thread};
use std::collections::VecDeque;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use anyhow::anyhow;
use anyhow::Result;
use glam::IVec3;
use log::{debug, info};
use crate::core::scene::world::{distance_sq_between_chunks, VoxelChunkData};
use crate::core::{Engine, WorldGenerator};

type ThreadHandle = JoinHandle<()>;

pub struct ChunkLoader {
    threads: Vec<ChunkLoaderThreadWrapper>,
    shared_ctx: Arc<Mutex<ChunkLoaderSharedContext>>,
    world_generator: Arc<WorldGenerator>,
    chunk_load_request_count: usize,
}

struct ChunkLoaderSharedContext {
    chunk_load_queue: VecDeque<IVec3>,
    chunk_complete_queue: Vec<Box<VoxelChunkData>>,
    center_chunk_pos: IVec3,
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

// unsafe impl Send for ChunkLoaderThreadWrapper {}
// unsafe impl Sync for ChunkLoaderThreadWrapper {}
// unsafe impl Send for ChunkLoaderThreadContext {}
// unsafe impl Sync for ChunkLoaderThreadContext {}

impl ChunkLoader {
    pub fn new(world_generator: Arc<WorldGenerator>) -> Self {
        let shared_ctx = ChunkLoaderSharedContext {
            chunk_load_queue: VecDeque::new(),
            chunk_complete_queue: vec![],
            center_chunk_pos: IVec3::ZERO,
            changed: false,
        };

        ChunkLoader{
            threads: vec![],
            shared_ctx: Arc::new(Mutex::new(shared_ctx)),
            world_generator,
            chunk_load_request_count: 0,
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
            self.chunk_load_request_count = ctx.chunk_load_queue.len();
        }
    }

    pub fn request_load_chunk(&mut self, chunk_pos: IVec3) -> Result<()> {
        self.request_load_chunks(vec![chunk_pos])
    }

    pub fn request_load_chunks(&mut self, chunk_positions: Vec<IVec3>) -> Result<()> {
        if chunk_positions.is_empty() {
            return Ok(())
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error requesting chunk load: could not lock mutex: {e}"))?;

        for chunk_pos in chunk_positions {
            ctx.chunk_load_queue.push_back(chunk_pos);
        }

        self.chunk_load_request_count = ctx.chunk_load_queue.len();
        Ok(())
    }

    pub fn drain_completed_chunks(&mut self, count: usize, out_chunks: &mut Vec<Box<VoxelChunkData>>) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let mut ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Error retrieving completed chunks: could not lock mutex: {e}"))?;

        let center_pos = ctx.center_chunk_pos.as_dvec3();

        // Sorted so that the closest chunks are at the end of the list (pop() first)
        ctx.chunk_complete_queue.sort_by(|chunk1, chunk2| {
            let d1 = center_pos - chunk1.pos().as_dvec3();
            let d2 = center_pos - chunk2.pos().as_dvec3();
            let dist1 = d1.length_squared();
            let dist2 = d2.length_squared();
            dist2.total_cmp(&dist1)
        });

        let idx = ctx.chunk_complete_queue.len() - usize::min(count, ctx.chunk_complete_queue.len());

        for chunk in ctx.chunk_complete_queue.drain(idx..) {
            out_chunks.push(chunk);
        }
        Ok(())
    }

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
        let mut count = 0;
        let mut start_time = Instant::now();
        loop {
            {
                let stopped = thread_ctx.stopped.lock().unwrap();
                if *stopped {
                    break;
                }
            }

            if let Some(chunk_pos) = Self::get_next_chunk_load_pos(&shared_ctx) {

                if count == 0 {
                    start_time = Instant::now();
                }

                // Allocate chunk
                let mut chunk = Box::new(VoxelChunkData::new(chunk_pos));

                // Generate chunk data
                Self::generate_chunk_data(&mut chunk, &thread_ctx);

                // Build chunk mesh
                Self::build_chunk_mesh(&mut chunk);

                let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");
                ctx.chunk_complete_queue.push(chunk);
                count += 1;

            } else {
                if count > 0 {
                    let thread_name = thread::current().name().map_or_else(
                        || format!("UnnamedThread-{:?}", thread::current().id()),
                        |e| e.to_string());

                    let dur = start_time.elapsed().as_secs_f64() * 1000.0;

                    info!("ChunkLoader thread {thread_name} processed {count} chunks in {dur} msec");
                    count = 0;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }

        info!("ChunkLoader - thread stopped");
    }

    fn get_next_chunk_load_pos(shared_ctx: &Arc<Mutex<ChunkLoaderSharedContext>>) -> Option<IVec3> {
        let mut ctx = shared_ctx.lock().expect("ChunkLoader - Failed to lock shared context");
        let chunk_pos = ctx.chunk_load_queue.pop_front();
        ctx.changed = true;
        chunk_pos
    }

    fn generate_chunk_data(chunk: &mut Box<VoxelChunkData>, thread_ctx: &Arc<ChunkLoaderThreadContext>) {

        let world_generator = &thread_ctx.world_generator;

        world_generator.load_chunk(chunk).expect("ChunkLoader - Failed to load chunk");
    }

    fn build_chunk_mesh(chunk: &mut Box<VoxelChunkData>) {
        chunk.update_mesh_data().expect("ChunkLoader - Failed to build chunk mesh");
    }


    /// Update the queues of chunks to load & unload. Keep the queues sorted by distance around the player position
    /// The load queue is sorted so the nearest chunks are loaded first, and the unload queue is sorted in the
    /// opposite direction.
    pub fn update_chunk_queues(&mut self, center_chunk_pos: IVec3, chunk_load_radius: u32) -> Result<()> {
        let center_pos = center_chunk_pos.as_dvec3();

        let mut shared_ctx = self.shared_ctx.lock()
            .map_err(|e| anyhow!("ChunkLoader - Unable to update chunk queue - failed to lock mutex for shared context: {e}"))?;

        shared_ctx.center_chunk_pos = center_chunk_pos;

        // Sorted so that the closest chunks are at the front of the queue (pop_front() first)
        shared_ctx.chunk_load_queue.make_contiguous().sort_by(|pos1, pos2| {
            let mut d1 = center_pos - pos1.as_dvec3();
            let mut d2 = center_pos - pos2.as_dvec3();
            d1.y *= 2.0; // Bias the loading order to prioritise horizontal distance, vertical difference is less important.
            d2.y *= 2.0;
            let dist1 = d1.length_squared();
            let dist2 = d2.length_squared();
            dist1.total_cmp(&dist2)
        });

        for (index, chunk_pos) in shared_ctx.chunk_load_queue.iter().enumerate() {

            let r = chunk_load_radius as f64;
            let r2 = r * r;

            if distance_sq_between_chunks(*chunk_pos, center_chunk_pos) >= r2 {
                // chunk_load_queue is sorted by distance. If we encounter a chunk too far away, all remaining chunks are also too far away.
                // Pop off all remaining items at the end of the queue
                while shared_ctx.chunk_load_queue.len() > index {
                    shared_ctx.chunk_load_queue.pop_back();
                }
                break;
            }
        }

        Ok(())
    }

    pub fn chunk_load_request_count(&self) -> usize {
        self.chunk_load_request_count
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