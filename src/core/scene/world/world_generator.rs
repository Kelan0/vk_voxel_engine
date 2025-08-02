use crate::core::scene::world::noise::{sample_region_simplex_3d_simd, simplex_3d_simd};
use crate::core::scene::world::{calc_coord_for_index, get_chunk_world_pos, VoxelChunkData, CHUNK_BLOCK_COUNT, CHUNK_BOUNDS};
use anyhow::{anyhow, Result};
use glam::IVec3;
use log::{debug, info};
use simdeez::sse41::Sse41;
use simdeez::Simd;
use std::sync::Mutex;
use std::time::Instant;

pub struct WorldGenerator {
    terrain_generator: Box<dyn DynGenerator3D>
}

impl WorldGenerator {
    pub fn new(terrain_generator: Box<dyn DynGenerator3D>) -> WorldGenerator {
        Self {
            terrain_generator
        }
    }

    // pub fn test_all() {
    //
    //     let seed = 99;
    //     let dim = [256, 256, 256];
    //
    //     let has_avx2 = is_x86_feature_detected!("avx2");
    //     let has_sse41 = is_x86_feature_detected!("sse4.1");
    //     let has_sse2 = is_x86_feature_detected!("sse2");
    //
    //     debug!("SSE2: {}, SSE4.1: {}, AVX2: {}", has_sse2, has_sse41, has_avx2);
    //
    //     debug!("D:");
    //     let d = unsafe { Self::test::<Avx2, 3>(seed, dim) };
    //     debug!("A:");
    //     let a = unsafe { Self::test::<Scalar, 3>(seed, dim) };
    //     debug!("B:");
    //     let b = unsafe { Self::test::<Sse41, 3>(seed, dim) };
    //     debug!("C:");
    //     let c = unsafe { Self::test::<Sse2, 3>(seed, dim) };
    //
    //     info!("Max diff between A and B is {}", Self::test_diff(&a, &b));
    //     info!("Max diff between A and C is {}", Self::test_diff(&a, &c));
    //     info!("Max diff between A and D is {}", Self::test_diff(&a, &d));
    //
    //     // panic!();
    // }
    //
    // fn test_diff(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    //     let mut diff = 0.0;
    //     let len = a.len();
    //
    //     for i in 0..len {
    //         diff = f32::max(diff, f32::abs(a[i] - b[i]));
    //     }
    //
    //     diff
    // }
    // unsafe fn test<S: Simd, const D: usize>(seed: i32, dim: [usize; D]) -> Vec<f32> {
    //
    //
    //     let len = dim.iter().product();
    //     let mut result: Vec<f32> = vec![0.0; len];
    //
    //     assert_eq!(result.as_ptr() as usize % 32, 0);
    //
    //     let vector_width = S::VF32_WIDTH;
    //
    //     let mut x_arr = vec![0.0; vector_width];
    //     for i in (0..vector_width).rev() {
    //         x_arr[i] = i as f32;
    //     }
    //
    //     let mut iterations = 0;
    //     let mut idx = 0;
    //     let vdx = S::set1_ps(vector_width as f32);
    //     let v_1 = S::set1_ps(1.0);
    //
    //
    //     let t0 = Instant::now();
    //
    //     let mut fff = |t, x: &mut S::Vf32| {
    //         S::storeu_ps(result.get_unchecked_mut(idx), t);
    //         *x = S::add_ps(*x, vdx);
    //         idx += vector_width;
    //         iterations += 1;
    //     };
    //
    //     match D {
    //         1 => {
    //             let mut x = S::load_ps(&x_arr[0]);
    //             for _ in 0..dim[0] / vector_width {
    //                 let t = simplex::simplex_1d::<S>(x, seed);
    //                 fff(t, &mut x);
    //             }
    //         },
    //         2 => {
    //             let mut y = S::set1_ps(0.0);
    //             for _ in 0..dim[1] {
    //                 let mut x = S::load_ps(&x_arr[0]);
    //                 for _ in 0..dim[0] / vector_width {
    //                     let t = simplex::simplex_2d::<S>(x, y, seed);
    //                     fff(t, &mut x);
    //                 }
    //                 y = S::add_ps(y, v_1);
    //             }
    //         },
    //         3 => {
    //             let mut z = S::set1_ps(0.0);
    //             for _ in 0..dim[2] {
    //                 let mut y = S::set1_ps(0.0);
    //                 for _ in 0..dim[1] {
    //                     let mut x = S::loadu_ps(&x_arr[0]);
    //                     for _ in 0..dim[0] / vector_width {
    //                         let t = simplex::simplex_3d::<S>(x, y, z, seed);
    //                         fff(t, &mut x);
    //                     }
    //                     y = S::add_ps(y, v_1);
    //                 }
    //                 z = S::add_ps(z, v_1)
    //             }
    //         },
    //         _ => {}
    //     }
    //
    //
    //     let t1 = Instant::now();
    //
    //
    //     let mut min = f32::INFINITY;
    //     let mut max = f32::NEG_INFINITY;
    //
    //     for x in &result {
    //         min = f32::min(min, *x);
    //         max = f32::max(max, *x);
    //     }
    //
    //     let dur = t1.duration_since(t0).as_secs_f64() * 1000.0;
    //     debug!("TEST RESULT for {} ({} wide): range ({} -> {}), Took {} msec for {} samples in {} iterations",  type_name::<S>(), vector_width, min, max, dur, idx, iterations);
    //
    //     result
    // }

    pub fn load_chunk(&self, chunk: &mut Box<VoxelChunkData>) -> Result<()> {
        let chunk_pos = chunk.chunk_pos().clone();

        // let t0 = Instant::now();

        // let buf = self.sample_block_density_region(get_world_block_pos_for_chunk_block_pos(chunk_pos, UVec3::ZERO), get_world_block_pos_for_chunk_block_pos(chunk_pos, CHUNK_BOUNDS));

        let offset = get_chunk_world_pos(chunk_pos);

        let mut samples = vec![0.0; CHUNK_BLOCK_COUNT];
        let (min, max) = self.terrain_generator.sample_region(offset.as_vec3().to_array(), [0.013, 0.013, 0.013], CHUNK_BOUNDS.as_usizevec3().to_array(), &mut samples);

        // let (samples, min, max) = NoiseBuilder::fbm_3d_offset(offset.x, 32, offset.y, 32, offset.z, 32)
        //     .with_freq(0.013)
        //     .with_octaves(2)
        //     .with_gain(2.0)
        //     .with_seed(99)
        //     .with_lacunarity(0.5)
        //     .generate();

        // let t1 = Instant::now();

        let min_y = -32;
        let max_y = 32;
        for i in 0..CHUNK_BLOCK_COUNT {
            let chunk_block_pos = calc_coord_for_index(i as u32, CHUNK_BOUNDS);
            let world_block_pos = chunk_pos * CHUNK_BOUNDS.as_ivec3() + chunk_block_pos.as_ivec3();
            let density = samples[i];
            // let density = (density - min) / (max - min);

            let density_threshold = f32::clamp((world_block_pos.y - min_y) as f32 / (max_y - min_y) as f32, 0.0, 1.0);
            if density > density_threshold {
                chunk.set_block(chunk_block_pos, 1);
            }
        }

        // let t2 = Instant::now();
        // //
        // let dur1 = t1.duration_since(t0);
        // let dur2 = t2.duration_since(t1);
        // debug!("load_chunk Samples took {} msec ({} ns per sample), update took {} msec, range: {} to {}", dur1.as_secs_f64() * 1000.0, dur1.as_nanos() as f64 / CHUNK_BLOCK_COUNT as f64, dur2.as_secs_f64() * 1000.0, min, max);

        Ok(())
    }

    pub fn sample_block_density(&self, block_pos: IVec3) -> f32 {
        let x = block_pos.x as f32;
        let y = block_pos.y as f32;
        let z = block_pos.z as f32;
        self.terrain_generator.sample([x, y, z])
    }

    // pub fn sample_block_density_region(&self, pos_min: IVec3, pos_max: IVec3) -> NoiseBuffer<3> {
    //     let x0 = pos_min.x as f64;
    //     let y0 = pos_min.y as f64;
    //     let z0 = pos_min.z as f64;
    //     let sx = (pos_max.x - pos_min.x) as usize;
    //     let sy = (pos_max.y - pos_min.y) as usize;
    //     let sz = (pos_max.z - pos_min.z) as usize;
    //     self.terrain_generator.buffer([x0, y0, z0], [sx, sy, sz])
    // }

    pub fn terrain_generator(&self) -> &Box<dyn DynGenerator3D> {
        &self.terrain_generator
    }

    pub fn default_terrain_generator(seed: u64) -> Box<dyn DynGenerator3D> {

        // // load_chunk took 5.9897 msec (182.7911376953125 ns per sample)
        // let generator = Source::simplex(seed)
        //     .fbm(5, 0.013, 2.0, 0.5);
        //     // .abs()
        //     // .mul(2.0)
        //     // .lambda(|x| 1.0 - x.exp() / 2.8);
        //     // .displace_x(
        //     //     Source::worley(43)
        //     //         .scale([0.005; 3])
        //     //         .fbm(3, 1.0, 2.0, 0.5)
        //     //         .mul(5.0)
        //     // )
        //     // .blend(
        //     //     Source::worley(45)
        //     //         .scale([0.033; 3]),
        //     //     Source::perlin(45)
        //     //         .scale([0.033; 3])
        //     //         .add(0.3)
        //     // );

        let generator = CustomGenerator::new(seed);

        Box::new(generator)
    }
}


pub trait DynGenerator3D: Send + Sync {
    fn sample(&self, point: [f32; 3]) -> f32;

    fn sample_region(&self, offset: [f32; 3], scale: [f32; 3], dim: [usize; 3], buffer: &mut [f32]) -> (f32, f32);
}

// impl<T> DynGenerator3D for T
// where
//     T: Generator<3> + Clone + 'static,
// {
//     fn sample(&self, point: [f64; 3]) -> f64 {
//         self.sample(point)
//     }
//
//     fn buffer(&self, offset: [f64; 3], shape: [usize; 3]) -> NoiseBuffer<3> {
//         let r = self.clone().translate(offset);
//         NoiseBuffer::<3>::new(shape, &r)
//     }
// }




// struct CustomGenerator {
//     generator: Fbm<Value>
// }
//
// impl CustomGenerator {
//     pub fn new(seed: u64) -> Self {
//
//         let m = Fbm::<Value>::new(seed as u32)
//             .set_octaves(5)
//             .set_frequency(0.013)
//             .set_lacunarity(2.0)
//             .set_persistence(0.5);
//
//         CustomGenerator {
//             generator: m,
//         }
//     }
// }
//
// impl DynGenerator3D for CustomGenerator {
//     fn sample(&self, point: [f64; 3]) -> f64 {
//
//         self.generator.get(point)
//
//         // let mut expected = 0.0;
//         // let mut amp = 1.0;
//         // let mut freq = frequency;
//         // for _ in 0..octaves {
//         //     self.generator.set_sources()
//         //     expected += amp * self.generator.sample(point.map(|x| x * freq));
//         //     freq *= lacunarity;
//         //     amp *= persistence;
//         // }
//         // expected /= (0..octaves).fold(0.0, |acc, octave| acc + persistence.powi(octave as i32));
//     }
// }





struct CustomGenerator {
    seed: u64,
}

impl CustomGenerator {
    pub fn new(seed: u64) -> Self {
        CustomGenerator {
            seed
        }
    }
}

impl DynGenerator3D for CustomGenerator {
    fn sample(&self, point: [f32; 3]) -> f32 {
        let mut buffer = [0.0];

        self.sample_region(point, [1.0, 1.0, 1.0], [1, 1, 1], &mut buffer);

        buffer[0]
    }

    fn sample_region(&self, offset: [f32; 3], scale: [f32; 3], dim: [usize; 3], buffer: &mut [f32]) -> (f32, f32) {
        let seed = self.seed as i32;

        fn sample_fn<S: Simd>(x: S::Vf32, y: S::Vf32, z: S::Vf32) -> S::Vf32{
            // unsafe {
            //     let scale = S::set1_ps(32.69587493801679);
            //     let r = simplex::simplex_3d::<S>(x, y, z, 99);
            //     S::mul_ps(r, scale)
            // }

            unsafe { simplex_3d_simd::<S>(x, y, z, 99) }
        }

        let (min, max) = unsafe { sample_region_simplex_3d_simd::<Sse41>(offset, scale, dim, buffer, sample_fn::<Sse41>) };

        (min, max)
    }

}