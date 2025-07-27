use std::arch::x86_64::_mm_set_ps1;
use num::traits::real::Real;
use simdeez::Simd;
use simdeez::sse41::Sse41;
use smallvec::smallvec;

/// Skew factor for 2D simplex noise
const F2_32: f32 = 0.36602540378;
pub const F2_64: f64 = 0.36602540378;
/// Skew factor for 3D simplex noise
const F3_32: f32 = 1.0 / 3.0;
pub const F3_64: f64 = 1.0 / 3.0;
/// Skew factor for 4D simplex noise
const F4_32: f32 = 0.309016994;
pub const F4_64: f64 = 0.309016994;
/// Unskew factor for 2D simplex noise
const G2_32: f32 = 0.2113248654;
pub const G2_64: f64 = 0.2113248654;
const G22_32: f32 = G2_32 * 2.0;
pub const G22_64: f64 = G2_64 * 2.0;
/// Unskew factor for 3D simplex noise
const G3_32: f32 = 1.0 / 6.0;
pub const G3_64: f64 = 1.0 / 6.0;
const G33_32: f32 = 3.0 / 6.0 - 1.0;
pub const G33_64: f64 = 3.0 / 6.0 - 1.0;
/// Unskew factor for 4D simplex noise
const G4_32: f32 = 0.138196601;
pub const G4_64: f64 = 0.138196601;
const G24_32: f32 = 2.0 * G4_32;
pub const G24_64: f64 = 2.0 * G4_64;
const G34_32: f32 = 3.0 * G4_32;
pub const G34_64: f64 = 3.0 * G4_64;
const G44_32: f32 = 4.0 * G4_32;
pub const G44_64: f64 = 4.0 * G4_64;


pub const X_PRIME_32: i32 = 1619;
pub const X_PRIME_64: i64 = 1619;
pub const Y_PRIME_32: i32 = 31337;
pub const Y_PRIME_64: i64 = 31337;
pub const Z_PRIME_32: i32 = 6971;
pub const Z_PRIME_64: i64 = 6971;


static PERM: [i32; 512] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194,
    233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
    159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253,
    19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];


macro_rules! get_3d_noise {
    ($setting:expr) => {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::get_3d_noise($setting) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { sse41::get_3d_noise($setting) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { sse2::get_3d_noise($setting) }
        } else {
            unsafe { scalar::get_3d_noise($setting) }
        }
    };
}

pub unsafe fn sample_region_simplex_3d_simd<S: Simd>(offset: [f32; 3], scale: [f32; 3], dim: [usize; 3], buffer: &mut [f32], sample_fn: fn(x: S::Vf32, y: S::Vf32, z: S::Vf32) -> S::Vf32) -> (f32, f32) {

    let freq_x = S::set1_ps(scale[0]);
    let freq_y = S::set1_ps(scale[1]);
    let freq_z = S::set1_ps(scale[2]);
    let start_x = S::set1_ps(offset[0]);
    let start_y = S::set1_ps(offset[1]);
    let start_z = S::set1_ps(offset[2]);
    let width = dim[0];
    let height = dim[1];
    let depth = dim[2];

    let vector_width = S::VF32_WIDTH;
    let vector_width_s = S::set1_ps(vector_width as f32);
    let one_s = S::set1_ps(1.0);

    let mut min_s = S::set1_ps(f32::MAX);
    let mut max_s = S::set1_ps(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    assert!(buffer.len() >= width * height * depth);
    let mut idx = 0;
    let remainder = width % vector_width;
    let mut x_arr = vec![0.0; vector_width];
    for i in (0..vector_width).rev() {
        x_arr[i] = offset[0] + i as f32;
    }

    let mut temp_arr = vec![0.0; vector_width];

    let iter_x = dim[0] / vector_width;
    let iter_y = dim[1];
    let iter_z = dim[2];

    let mut z = start_z;
    for _ in 0..iter_z {

        let mut y = start_y;
        for _ in 0..iter_y {

            let mut x = S::loadu_ps(&x_arr[0]);
            for _ in 0..iter_x {

                let f = sample_fn(x * freq_x, y * freq_y, z * freq_z);
                max_s = S::max_ps(max_s, f);
                min_s = S::min_ps(min_s, f);
                S::storeu_ps(buffer.get_unchecked_mut(idx), f);

                idx += vector_width;
                x = S::add_ps(x, vector_width_s);
            }

            if remainder != 0 {
                let f = sample_fn(x * freq_x, y * freq_y, z * freq_z);
                S::storeu_ps(temp_arr.get_unchecked_mut(0), f);
                for j in 0..remainder {
                    buffer[idx] = temp_arr[j];

                    min = f32::min(min, temp_arr[j]);
                    max = f32::max(max, temp_arr[j]);
                    idx += 1;
                }
            }
            y = S::add_ps(y, one_s);
        }
        z = S::add_ps(z, one_s);
    }

    S::storeu_ps(temp_arr.get_unchecked_mut(0), min_s);
    // buffer.set_len(width * height * depth);
    for i in 0..vector_width {
        min = f32::min(min, temp_arr[i]);
    }
    S::storeu_ps(temp_arr.get_unchecked_mut(0), max_s);
    for i in 0..vector_width {
        max = f32::max(max, temp_arr[i]);
    }

    (min, max)
}

pub fn simplex_3d<S: Simd>(x: S::Vf32, y: S::Vf32, z: S::Vf32, seed: i32) -> S::Vf32 {
    unsafe { simplex_3d_simd::<S>(x, y, z, seed) }
}

/// Samples 2-dimensional simplex noise
///
/// Produces a value -1 ≤ n ≤ 1.
#[inline(always)]
pub fn simplex_2d_simd<S: Simd>(x: S::Vf32, y: S::Vf32, seed: i32) -> S::Vf32 {
    simplex_2d_deriv_simd::<S>(x, y, seed).0
}

/// Like `simplex_2d`, but also computes the derivative
#[inline(always)]
pub fn simplex_2d_deriv_simd<S: Simd>(x: S::Vf32, y: S::Vf32, seed: i32) -> (S::Vf32, [S::Vf32; 2]) {
    todo!()
    // // Skew to distort simplexes with side length sqrt(2)/sqrt(3) until they make up
    // // squares
    // let s = S::set1_ps(F2_32) * (x + y);
    // let ips = (x + s).floor();
    // let jps = (y + s).floor();
    //
    // // Integer coordinates for the base vertex of the triangle
    // let i = ips.cast_i32();
    // let j = jps.cast_i32();
    //
    // let t = (i + j).cast_f32() * S::set1_ps(G2_32);
    //
    // // Unskewed distances to the first point of the enclosing simplex
    // let x0 = x - (ips - t);
    // let y0 = y - (jps - t);
    //
    // let i1 = (x0.cmp_gte(y0)).bitcast_i32();
    //
    // let j1 = (y0.cmp_gt(x0)).bitcast_i32();
    //
    // // Distances to the second and third points of the enclosing simplex
    // let x1 = (x0 + i1.cast_f32()) + S::set1_ps(G2_32);
    // let y1 = (y0 + j1.cast_f32()) + S::set1_ps(G2_32);
    // let x2 = (x0 + S::set1_ps(-1.0)) + S::set1_ps(G22_32);
    // let y2 = (y0 + S::set1_ps(-1.0)) + S::set1_ps(G22_32);
    //
    // let ii = i & S::Vi32::set1(0xff);
    // let jj = j & S::Vi32::set1(0xff);
    //
    // let (gi0, gi1, gi2) = unsafe {
    //     assert_in_perm_range::<S>(ii);
    //     assert_in_perm_range::<S>(jj);
    //     assert_in_perm_range::<S>(ii - i1);
    //     assert_in_perm_range::<S>(jj - j1);
    //     assert_in_perm_range::<S>(ii + 1);
    //     assert_in_perm_range::<S>(jj + 1);
    //
    //     let gi0 = gather_32::<S>(&PERM, ii + gather_32::<S>(&PERM, jj));
    //     let gi1 = gather_32::<S>(&PERM, (ii - i1) + gather_32::<S>(&PERM, jj - j1));
    //     let gi2 = gather_32::<S>(
    //         &PERM,
    //         (ii - S::Vi32::set1(-1)) + gather_32::<S>(&PERM, jj - S::Vi32::set1(-1)),
    //     );
    //
    //     (gi0, gi1, gi2)
    // };
    //
    // // Weights associated with the gradients at each corner
    // // These FMA operations are equivalent to: let t = 0.5 - x*x - y*y
    // let mut t0 = S::Vf32::neg_mul_add(y0, y0, S::Vf32::neg_mul_add(x0, x0, S::set1_ps(0.5)));
    // let mut t1 = S::Vf32::neg_mul_add(y1, y1, S::Vf32::neg_mul_add(x1, x1, S::set1_ps(0.5)));
    // let mut t2 = S::Vf32::neg_mul_add(y2, y2, S::Vf32::neg_mul_add(x2, x2, S::set1_ps(0.5)));
    //
    // // Zero out negative weights
    // t0 &= t0.cmp_gte(S::Vf32::zeroes());
    // t1 &= t1.cmp_gte(S::Vf32::zeroes());
    // t2 &= t2.cmp_gte(S::Vf32::zeroes());
    //
    // let t20 = t0 * t0;
    // let t40 = t20 * t20;
    // let t21 = t1 * t1;
    // let t41 = t21 * t21;
    // let t22 = t2 * t2;
    // let t42 = t22 * t22;
    //
    // let [gx0, gy0] = grad2::<S>(seed, gi0);
    // let g0 = gx0 * x0 + gy0 * y0;
    // let n0 = t40 * g0;
    // let [gx1, gy1] = grad2::<S>(seed, gi1);
    // let g1 = gx1 * x1 + gy1 * y1;
    // let n1 = t41 * g1;
    // let [gx2, gy2] = grad2::<S>(seed, gi2);
    // let g2 = gx2 * x2 + gy2 * y2;
    // let n2 = t42 * g2;
    //
    // // Scaling factor found by numerical approximation
    // let scale = S::set1_ps(45.26450774985561631259);
    // let value = (n0 + (n1 + n2)) * scale;
    // let derivative = {
    //     let temp0 = t20 * t0 * g0;
    //     let mut dnoise_dx = temp0 * x0;
    //     let mut dnoise_dy = temp0 * y0;
    //     let temp1 = t21 * t1 * g1;
    //     dnoise_dx += temp1 * x1;
    //     dnoise_dy += temp1 * y1;
    //     let temp2 = t22 * t2 * g2;
    //     dnoise_dx += temp2 * x2;
    //     dnoise_dy += temp2 * y2;
    //     dnoise_dx *= S::set1_ps(-8.0);
    //     dnoise_dy *= S::set1_ps(-8.0);
    //     dnoise_dx += t40 * gx0 + t41 * gx1 + t42 * gx2;
    //     dnoise_dy += t40 * gy0 + t41 * gy1 + t42 * gy2;
    //     dnoise_dx *= scale;
    //     dnoise_dy *= scale;
    //     [dnoise_dx, dnoise_dy]
    // };
    // (value, derivative)
}


/// Samples 3-dimensional simplex noise
///
/// Produces a value -1 ≤ n ≤ 1.
#[inline(always)]
pub unsafe fn simplex_3d_simd<S: Simd>(x: S::Vf32, y: S::Vf32, z: S::Vf32, seed: i32) -> S::Vf32 {
    // vectorized constants
    let f3: S::Vf32 = S::set1_ps(F3_32);
    let x_prime = S::set1_epi32(X_PRIME_32);
    let y_prime = S::set1_epi32(Y_PRIME_32);
    let z_prime = S::set1_epi32(Z_PRIME_32);
    let g3 = S::set1_ps(G3_32);
    let g33 = S::set1_ps(G33_32);
    let zero = S::setzero_ps();
    let one = S::set1_ps(1.0);
    let zero_point_six = S::set1_ps(0.6);
    // Scaling factor found by numerical approximation
    let scale = S::set1_ps(32.69587493801679);


    let f = S::mul_ps(f3, S::add_ps(S::add_ps(x,y),z));
    let mut x0 = S::fast_floor_ps(S::add_ps(x,f));
    let mut y0 = S::fast_floor_ps(S::add_ps(y,f));
    let mut z0 = S::fast_floor_ps(S::add_ps(z,f));

    let i = S::mullo_epi32(S::cvtps_epi32(x0),x_prime);
    let j = S::mullo_epi32(S::cvtps_epi32(y0),y_prime);
    let k = S::mullo_epi32(S::cvtps_epi32(z0),z_prime);

    let g = S::mul_ps(g3, S::add_ps(S::add_ps(x0,y0),z0));
    x0 = S::sub_ps(x, S::sub_ps(x0,g));
    y0 = S::sub_ps(y, S::sub_ps(y0,g));
    z0 = S::sub_ps(z, S::sub_ps(z0,g));

    let x0_ge_y0 = S::cmpge_ps(x0,y0);
    let y0_ge_z0 = S::cmpge_ps(y0,z0);
    let x0_ge_z0 = S::cmpge_ps(x0,z0);

    let i1 = S::and_ps(x0_ge_y0, x0_ge_z0) as S::Vf32;
    let j1 = S::andnot_ps(x0_ge_y0, y0_ge_z0);
    let k1 = S::andnot_ps(x0_ge_z0, !y0_ge_z0);

    let i2 = S::or_ps(x0_ge_y0, x0_ge_z0) as S::Vf32;
    let j2 = S::or_ps((!x0_ge_y0), y0_ge_z0) as S::Vf32;
    let k2 = (!S::and_ps(x0_ge_z0, y0_ge_z0)) as S::Vf32;

    let x1 = S::add_ps( S::sub_ps(x0,S::and_ps(i1, one)), g3);
    let y1 = S::add_ps( S::sub_ps(y0,S::and_ps(j1, one)), g3);
    let z1 = S::add_ps( S::sub_ps(z0,S::and_ps(k1, one)), g3);

    let x2 = S::add_ps( S::sub_ps(x0,S::and_ps(i2, one)), f3);
    let y2 = S::add_ps( S::sub_ps(y0,S::and_ps(j2, one)), f3);
    let z2 = S::add_ps( S::sub_ps(z0,S::and_ps(k2, one)), f3);

    let x3 = S::add_ps(x0, g33);
    let y3 = S::add_ps(y0, g33);
    let z3 = S::add_ps(z0, g33);

    let mut t0 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six,S::mul_ps(x0,x0)),S::mul_ps(y0,y0)),S::mul_ps(z0,z0));
    let mut t1 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six,S::mul_ps(x1,x1)),S::mul_ps(y1,y1)),S::mul_ps(z1,z1));
    let mut t2 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six,S::mul_ps(x2,x2)),S::mul_ps(y2,y2)),S::mul_ps(z2,z2));
    let mut t3 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six,S::mul_ps(x3,x3)),S::mul_ps(y3,y3)),S::mul_ps(z3,z3));

    let n0 = S::cmpge_ps(t0, zero);
    let n1 = S::cmpge_ps(t1, zero);
    let n2 = S::cmpge_ps(t2, zero);
    let n3 = S::cmpge_ps(t3, zero);

    t0 = S::mul_ps(t0, t0);
    t1 = S::mul_ps(t1, t1);
    t2 = S::mul_ps(t2, t2);
    t3 = S::mul_ps(t3, t3);

    t0 = S::mul_ps(t0, t0);
    t1 = S::mul_ps(t1, t1);
    t2 = S::mul_ps(t2, t2);
    t3 = S::mul_ps(t3, t3);


    let v0 = S::mul_ps(t0, grad3d_dot_simd::<S>(seed,i,j,k,x0,y0,z0));

    let v1x = S::add_epi32(i,S::and_epi32(S::castps_epi32(i1), x_prime));
    let v1y = S::add_epi32(j,S::and_epi32(S::castps_epi32(j1), y_prime));
    let v1z = S::add_epi32(k,S::and_epi32(S::castps_epi32(k1), z_prime));
    let v1 = S::mul_ps(t1, grad3d_dot_simd::<S>(seed,v1x,v1y,v1z,x1,y1,z1));


    let v2x = S::add_epi32(i,S::and_epi32(S::castps_epi32(i2), x_prime));
    let v2y = S::add_epi32(j,S::and_epi32(S::castps_epi32(j2), y_prime));
    let v2z = S::add_epi32(k,S::and_epi32(S::castps_epi32(k2), z_prime));
    let v2 = S::mul_ps(t2, grad3d_dot_simd::<S>(seed,v2x,v2y,v2z,x2,y2,z2));

    let v3x = S::add_epi32(i,x_prime);
    let v3y = S::add_epi32(j,y_prime);
    let v3z = S::add_epi32(k,z_prime);
    let v3 = S::mul_ps(t3, grad3d_dot_simd::<S>(seed,v3x,v3y,v3z,x3,y3,z3));


    let p1 = S::add_ps(S::and_ps(n3, v3), S::and_ps(n2,v2));
    let p2 = S::add_ps(p1, S::and_ps(n1,v1));
    let result = S::add_ps(p2, S::and_ps(n0,v0));

    S::mul_ps(result, scale)
}

/// Like `simplex_3d`, but also computes the derivative
#[inline(always)]
pub unsafe fn simplex_3d_deriv_simd<S: Simd>(x: S::Vf32, y: S::Vf32, z: S::Vf32, seed: i32) -> (S::Vf32, [S::Vf32; 3]) {
    // vectorized constants
    let f3: S::Vf32 = S::set1_ps(F3_32);
    let x_prime = S::set1_epi32(X_PRIME_32);
    let y_prime = S::set1_epi32(Y_PRIME_32);
    let z_prime = S::set1_epi32(Z_PRIME_32);
    let g3 = S::set1_ps(G3_32);
    let g33 = S::set1_ps(G33_32);
    let zero = S::setzero_ps();
    let one = S::set1_ps(1.0);
    let zero_point_six = S::set1_ps(0.6);
    // Scaling factor found by numerical approximation
    let scale = S::set1_ps(32.69587493801679);

    // Find skewed simplex grid coordinates associated with the input coordinates
    let f = S::mul_ps(f3, S::add_ps(S::add_ps(x, y), z)); // f3 * ((x + y) + z)
    let mut x0 = S::fast_floor_ps(S::add_ps(x, f)); // (x + f).fast_floor()
    let mut y0 = S::fast_floor_ps(S::add_ps(y, f)); // (y + f).fast_floor()
    let mut z0 = S::fast_floor_ps(S::add_ps(z, f)); // (z + f).fast_floor()

    // Integer grid coordinates
    let i = S::mullo_epi32(S::cvtps_epi32(x0), x_prime); // x0.cast_i32() * X_PRIME_32
    let j = S::mullo_epi32(S::cvtps_epi32(y0), y_prime); // y0.cast_i32() * Y_PRIME_32
    let k = S::mullo_epi32(S::cvtps_epi32(z0), z_prime); // z0.cast_i32() * Z_PRIME_32

    // Compute distance from first simplex vertex to input coordinates
    let g = S::mul_ps(g3, S::add_ps(S::add_ps(x0, y0), z0)); // g3 * ((x0 + y0) + z0);
    x0 = S::sub_ps(x, S::sub_ps(x0, g)); // x - (x0 - g)
    y0 = S::sub_ps(y, S::sub_ps(y0, g)); // y - (y0 - g)
    z0 = S::sub_ps(z, S::sub_ps(z0, g)); // z - (z0 - g)

    let x0_ge_y0 = S::cmpge_ps(x0, y0); // x0.cmp_gte(y0);
    let y0_ge_z0 = S::cmpge_ps(y0, z0); // y0.cmp_gte(z0);
    let x0_ge_z0 = S::cmpge_ps(x0, z0); // x0.cmp_gte(z0);

    let i1 = S::and_ps(x0_ge_y0, x0_ge_z0); // x0_ge_y0 & x0_ge_z0;
    let j1 = S::andnot_ps(x0_ge_y0, y0_ge_z0); // y0_ge_z0.and_not(x0_ge_y0);
    let k1 = S::andnot_ps(x0_ge_z0, !y0_ge_z0); // (!y0_ge_z0).and_not(x0_ge_z0);

    let i2 = S::or_ps(x0_ge_y0, x0_ge_z0); // x0_ge_y0 | x0_ge_z0;
    let j2 = S::or_ps(!x0_ge_y0, y0_ge_z0); // (!x0_ge_y0) | y0_ge_z0;
    let k2 = !S::and_ps(x0_ge_z0, y0_ge_z0) as S::Vf32;// !(x0_ge_z0 & y0_ge_z0);

    // Compute distances from remaining simplex vertices to input coordinates
    let x1 = S::add_ps(S::sub_ps(x0, S::and_ps(i1, one)), g3); // x0 - (i1 & 1) + g3;
    let y1 = S::add_ps(S::sub_ps(y0, S::and_ps(j1, one)), g3); // y0 - (j1 & 1) + g3;
    let z1 = S::add_ps(S::sub_ps(z0, S::and_ps(k1, one)), g3); // z0 - (k1 & 1) + g3;

    let x2 = S::add_ps(S::sub_ps(x0, S::and_ps(i2, one)), f3); // x0 - (i2 & 1) + f3;
    let y2 = S::add_ps(S::sub_ps(y0, S::and_ps(j2, one)), f3); // y0 - (j2 & 1) + f3;
    let z2 = S::add_ps(S::sub_ps(z0, S::and_ps(k2, one)), f3); // z0 - (k2 & 1) + f3;

    let x3 = S::add_ps(x0, g33); // x0 + g33;
    let y3 = S::add_ps(y0, g33); // y0 + g33;
    let z3 = S::add_ps(z0, g33); // z0 + g33;

    // Compute base weight factors associated with each vertex, `0.6 - v . v` where v is the
    // distance to the vertex. Strictly the constant should be 0.5, but 0.6 is thought by Gustavson
    // to give visually better results at the cost of subtle discontinuities.
    let mut t0 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six, S::mul_ps(x0, x0)), S::mul_ps(y0, y0)), S::mul_ps(z0, z0)); // ((0.6 - (x0 * x0)) - (y0 * y0)) - (z0 * z0);
    let mut t1 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six, S::mul_ps(x1, x1)), S::mul_ps(y1, y1)), S::mul_ps(z1, z1)); // ((0.6 - (x1 * x1)) - (y1 * y1)) - (z1 * z1);
    let mut t2 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six, S::mul_ps(x2, x2)), S::mul_ps(y2, y2)), S::mul_ps(z2, z2)); // ((0.6 - (x2 * x2)) - (y2 * y2)) - (z2 * z2);
    let mut t3 = S::sub_ps(S::sub_ps(S::sub_ps(zero_point_six, S::mul_ps(x3, x3)), S::mul_ps(y3, y3)), S::mul_ps(z3, z3)); // ((0.6 - (x3 * x3)) - (y3 * y3)) - (z3 * z3);


    // Check negative weights
    let n0 = S::cmpge_ps(t0,S::setzero_ps());
    let n1 = S::cmpge_ps(t1,S::setzero_ps());
    let n2 = S::cmpge_ps(t2,S::setzero_ps());
    let n3 = S::cmpge_ps(t3,S::setzero_ps());

    // // Zero out negative weights
    // t0 = S::and_ps(t0, S::cmpge_ps(t0, zero)); // t0 = t0 & t0.cmp_gte(0);
    // t1 = S::and_ps(t1, S::cmpge_ps(t1, zero)); // t1 = t1 & t1.cmp_gte(0);
    // t2 = S::and_ps(t2, S::cmpge_ps(t2, zero)); // t2 = t2 & t2.cmp_gte(0);
    // t3 = S::and_ps(t3, S::cmpge_ps(t3, zero)); // t3 = t3 & t3.cmp_gte(0);

    // Square each weight
    let t20 = S::mul_ps(t0, t0); // t0 * t0;
    let t21 = S::mul_ps(t1, t1); // t1 * t1;
    let t22 = S::mul_ps(t2, t2); // t2 * t2;
    let t23 = S::mul_ps(t3, t3); // t3 * t3;

    // ...twice!
    let t40 = S::mul_ps(t20, t20); // t20 * t20;
    let t41 = S::mul_ps(t21, t21); // t21 * t21;
    let t42 = S::mul_ps(t22, t22); // t22 * t22;
    let t43 = S::mul_ps(t23, t23); // t23 * t23;

    // Compute contribution from each vertex
    let g0 = grad3d_dot_simd::<S>(seed, i, j, k, x0, y0, z0);
    let v0 = S::mul_ps(t40, g0); // t40 * g0;

    let v1x = S::add_epi32(i, S::and_epi32(S::castps_epi32(i1), x_prime)); // i + (i1.bitcast_i32() & x_prime);
    let v1y = S::add_epi32(j, S::and_epi32(S::castps_epi32(j1), y_prime)); // j + (j1.bitcast_i32() & y_prime);
    let v1z = S::add_epi32(k, S::and_epi32(S::castps_epi32(k1), z_prime)); // k + (k1.bitcast_i32() & z_prime);
    let g1 = grad3d_dot_simd::<S>(seed, v1x, v1y, v1z, x1, y1, z1);
    let v1 = S::mul_ps(t41, g1);

    let v2x = S::add_epi32(i, S::and_epi32(S::castps_epi32(i2), x_prime)); // i + (i2.bitcast_i32() & x_prime);
    let v2y = S::add_epi32(j, S::and_epi32(S::castps_epi32(j2), y_prime)); // j + (j2.bitcast_i32() & y_prime);
    let v2z = S::add_epi32(k, S::and_epi32(S::castps_epi32(k2), z_prime)); // k + (k2.bitcast_i32() & z_prime);
    let g2 = grad3d_dot_simd::<S>(seed, v2x, v2y, v2z, x2, y2, z2);
    let v2 = S::mul_ps(t42, g2);

    let v3x = S::add_epi32(i, x_prime); // i + x_prime
    let v3y = S::add_epi32(j, y_prime); // j + y_prime
    let v3z = S::add_epi32(k, z_prime); // k + z_prime

    let g3 = grad3d_dot_simd::<S>(seed, v3x, v3y, v3z, x3, y3, z3);
    let v3 = S::mul_ps(t43, g3); // t43 * g3

    let p1 = S::add_ps(S::and_ps(n3, v3), S::and_ps(n2, v2)); // v3 + v2
    let p2 = S::add_ps(p1, S::and_ps(n1, v1)); // p1 + v1
    let p3 = S::add_ps(p2, S::and_ps(n0, v0)); // p2 + v0

    let result = S::mul_ps(p3, scale); // (p2 + v0) * scale;

    let derivative = {
        let neg_zero_point_eight = S::set1_ps(-8.0);
        let temp0 =  S::mul_ps(S::mul_ps(t20, t0), g0); // t20 * t0 * g0;
        let mut dnoise_dx = S::mul_ps(temp0, x0); // temp0 * x0
        let mut dnoise_dy = S::mul_ps(temp0, y0); // temp0 * y0
        let mut dnoise_dz = S::mul_ps(temp0, z0); // temp0 * z0
        let temp1 = S::mul_ps(S::mul_ps(t21, t1), g1); // t21 * t1 * g1;
        dnoise_dx = S::add_ps(dnoise_dx, S::mul_ps(temp1, x1)); // dnoise_dx = dnoise_dx + temp1 * x1
        dnoise_dy = S::add_ps(dnoise_dy, S::mul_ps(temp1, y1)); // dnoise_dy = dnoise_dy + temp1 * y1
        dnoise_dz = S::add_ps(dnoise_dz, S::mul_ps(temp1, z1)); // dnoise_dz = dnoise_dz + temp1 * z1
        let temp2 = S::mul_ps(S::mul_ps(t22, t2), g2); // t22 * t2 * g2
        dnoise_dx = S::add_ps(dnoise_dx, S::mul_ps(temp2, x2)); // dnoise_dx = dnoise_dx + temp2 * x2
        dnoise_dy = S::add_ps(dnoise_dy, S::mul_ps(temp2, y2)); // dnoise_dy = dnoise_dy + temp2 * y2
        dnoise_dz = S::add_ps(dnoise_dz, S::mul_ps(temp2, z2)); // dnoise_dz = dnoise_dz + temp2 * z2
        let temp3 = S::mul_ps(S::mul_ps(t23, t3), g3); // t23 * t3 * g3
        dnoise_dx = S::add_ps(dnoise_dx, S::mul_ps(temp3, x3)); // dnoise_dx = dnoise_dx + temp3 * x3
        dnoise_dy = S::add_ps(dnoise_dy, S::mul_ps(temp3, y3)); // dnoise_dy = dnoise_dy + temp3 * y3
        dnoise_dz = S::add_ps(dnoise_dz, S::mul_ps(temp3, z3)); // dnoise_dz = dnoise_dz + temp3 * z3
        dnoise_dx = S::mul_ps(dnoise_dx, neg_zero_point_eight);
        dnoise_dy = S::mul_ps(dnoise_dy, neg_zero_point_eight);
        dnoise_dz = S::mul_ps(dnoise_dz, neg_zero_point_eight);
        let [gx0, gy0, gz0] = grad3d_simd::<S>(seed, i, j, k);
        let [gx1, gy1, gz1] = grad3d_simd::<S>(seed, v1x, v1y, v1z);
        let [gx2, gy2, gz2] = grad3d_simd::<S>(seed, v2x, v2y, v2z);
        let [gx3, gy3, gz3] = grad3d_simd::<S>(seed, v3x, v3y, v3z);
        dnoise_dx = S::add_ps(S::add_ps(S::add_ps(S::add_ps(dnoise_dx, S::mul_ps(t40, gx0)), S::mul_ps(t41, gx1)), S::mul_ps(t42, gx2)), S::mul_ps(t43, gx3)); // dnoise_dx = dnoise_dx + t40 * gx0 + t41 * gx1 + t42 * gx2 + t43 * gx3;
        dnoise_dy = S::add_ps(S::add_ps(S::add_ps(S::add_ps(dnoise_dy, S::mul_ps(t40, gy0)), S::mul_ps(t41, gy1)), S::mul_ps(t42, gy2)), S::mul_ps(t43, gy3)); // dnoise_dy = dnoise_dy + t40 * gy0 + t41 * gy1 + t42 * gy2 + t43 * gy3;
        dnoise_dz = S::add_ps(S::add_ps(S::add_ps(S::add_ps(dnoise_dz, S::mul_ps(t40, gz0)), S::mul_ps(t41, gz1)), S::mul_ps(t42, gz2)), S::mul_ps(t43, gz3)); // dnoise_dz = dnoise_dz + t40 * gz0 + t41 * gz1 + t42 * gz2 + t43 * gz3;
        // Scale into range
        dnoise_dx = S::mul_ps(dnoise_dx, scale);
        dnoise_dy = S::mul_ps(dnoise_dy, scale);
        dnoise_dz = S::mul_ps(dnoise_dz, scale);
        [dnoise_dx, dnoise_dy, dnoise_dz]
    };
    (result, derivative)
}


/// Generates a random gradient vector from the origin towards the midpoint of an edge of a
/// double-unit cube and computes its dot product with [x, y, z]
#[inline(always)]
pub unsafe fn grad3d_dot_simd<S: Simd>(seed: i32, i: S::Vi32, j: S::Vi32, k: S::Vi32, x: S::Vf32, y: S::Vf32, z: S::Vf32) -> S::Vf32 {
    let mut hash = S::xor_epi32(i,S::set1_epi32(seed));
    hash = S::xor_epi32(j, hash);
    hash = S::xor_epi32(k,hash);
    hash = S::mullo_epi32(S::mullo_epi32(S::mullo_epi32(hash,hash),S::set1_epi32(60493)),hash);
    hash =  S::xor_epi32(S::srai_epi32(hash,13),hash);
    let hasha13 = S::and_epi32(hash,S::set1_epi32(13));

    let l8 = S::castepi32_ps(S::cmplt_epi32(hasha13,S::set1_epi32(8)));
    let u = S::blendv_ps(y, x, l8);

    let l4 = S::castepi32_ps(S::cmplt_epi32(hasha13,S::set1_epi32(2)));
    let h12_or_14 = S::castepi32_ps(S::cmpeq_epi32(S::set1_epi32(12),hasha13));
    let v = S::blendv_ps(S::blendv_ps(z,x,h12_or_14),y,l4);

    let h1 = S::castepi32_ps(S::slli_epi32(hash,31));
    let h2 = S::castepi32_ps(S::slli_epi32(S::and_epi32(hash,S::set1_epi32(2)),30));
    S::add_ps(S::xor_ps(u, h1), S::xor_ps(v, h2))

    // let h = hash3d_simd::<S>(seed, i, j, k);
    // let u = S::blendv_ps(y, x, h.l8); // h.l8.blendv(y, x)
    // let v = S::blendv_ps(S::blendv_ps(z, x, h.h12_or_14), y, h.l4); // h.l4.blendv(h.h12_or_14.blendv(z, x), y);
    // let result = S::add_ps(S::xor_ps(u, h.h1), S::xor_ps(v, h.h2)); // (u ^ h.h1) + (v ^ h.h2)
    // debug_assert_eq!(
    //     result[0],
    //     {
    //         let [gx, gy, gz] = grad3d_simd::<S>(seed, i, j, k);
    //         gx * x + gy * y + gz * z
    //     }[0],
    //     "results match"
    // );
    // result
}

/// The gradient vector generated by `grad3d_dot`
///
/// This is a separate function because it's slower than `grad3d_dot` and only needed when computing
/// derivatives.
pub unsafe fn grad3d_simd<S: Simd>(seed: i32, i: S::Vi32, j: S::Vi32, k: S::Vi32) -> [S::Vf32; 3] {
    let h = hash3d_simd::<S>(seed, i, j, k);

    let one = S::set1_ps(1.0);
    let first = S::or_ps(one, h.h1); // one | h.h1
    let mut gx = S::and_ps(h.l8, first);// h.l8 & first;
    let mut gy = S::andnot_ps(h.l8, first); // first.and_not(h.l8);

    let second = S::or_ps(one, h.h2); // one | h.h2;
    gy = S::blendv_ps(gy, second, h.l4); // h.l4.blendv(gy, second);
    gx = S::blendv_ps(gx, second, S::andnot_ps(h.l4, h.h12_or_14)); // h.h12_or_14.and_not(h.l4).blendv(gx, second);
    let gz = S::andnot_ps(S::or_ps(h.h12_or_14, h.l4), second); // second.and_not(h.h12_or_14 | h.l4);
    debug_assert_eq!(
        gx[0].abs() + gy[0].abs() + gz[0].abs(),
        2.0,
        "exactly two axes are chosen"
    );
    [gx, gy, gz]
}


pub struct Hash3d<S: Simd> {
    // Masks guiding dimension selection
    pub l8: S::Vf32,
    pub l4: S::Vf32,
    pub h12_or_14: S::Vf32,

    // Signs for the selected dimensions
    pub h1: S::Vf32,
    pub h2: S::Vf32,
}

impl<S> Hash3d<S>
where
    S: Simd,
{
    pub fn new(l8: S::Vf32, l4: S::Vf32, h12_or_14: S::Vf32, h1: S::Vf32, h2: S::Vf32) -> Self {
        Self { l8, l4, h12_or_14, h1, h2 }
    }
}
/// Compute hash values used by `grad3d` and `grad3d_dot`

#[inline(always)]
pub unsafe fn hash3d_simd<S: Simd>(seed: i32, i: S::Vi32, j: S::Vi32, k: S::Vi32) -> Hash3d<S> {
    // It seems that this function is inspired by FastNoise-SIMD and Auburn/FastNoise2Simd
    // https://github.com/jackmott/FastNoise-SIMD/blob/31c4a74d649ef4bc93aaabe4bf94fa81e4c0eadc/FastNoise/FastNoise3d.cpp#L348-L353

    let mut hash;
    hash = S::xor_epi32(i, S::set1_epi32(seed)); // i ^ seed
    hash = S::xor_epi32(j, hash); // j ^ hash
    hash = S::xor_epi32(k, hash); // k ^ hash;
    hash = S::mullo_epi32(S::mullo_epi32(S::mullo_epi32(hash, hash), S::set1_epi32(60493)), hash); // ((hash * hash) * 60493) * hash;
    hash = S::xor_epi32(S::sra_epi32(hash, 13), hash); // (hash >> 13) ^ hash; // TODO: check srai instead of sra
    let hash_and_13 = S::and_epi32(hash, S::set1_epi32(13)); // hash & 13;
    Hash3d::new(
        S::castepi32_ps(S::cmplt_epi32(hash_and_13, S::set1_epi32(8))), // (hash13.cmp_lt(8)).bitcast_f32(),
        S::castepi32_ps(S::cmplt_epi32(hash_and_13, S::set1_epi32(2))), // (hash13.cmp_lt(2)).bitcast_f32(),
        S::castepi32_ps(S::cmpeq_epi32(hash_and_13, S::set1_epi32(12))), // (hash13.cmp_eq(12)).bitcast_f32(),
        S::castepi32_ps(S::sll_epi32(hash, 31)), // (hash << 31).bitcast_f32(),
        S::castepi32_ps(S::sll_epi32(S::and_epi32(hash, S::set1_epi32(2)), 30)), // ((hash & 2) << 30).bitcast_f32(),
    )
}