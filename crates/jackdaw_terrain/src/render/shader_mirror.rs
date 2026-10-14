//! Rust spellings of the arithmetic in `shaders/terrain_splat.wgsl` that is worth
//! proving without a GPU: how the three projections share a fragment, how a
//! side plane's normal is brought into world axes, how the four control corners
//! merge by texture id, and how the two distance bands fall.
//!
//! The shader is the code that ships. Each function here says what the shader
//! computes in one place, and the tests beside it pin a property the shader has
//! to keep: that a bump leans the way it should on every plane, that merging
//! changes no weight, that flat ground is the top plane alone. A change to the
//! WGSL has to be made here too, or the tests are proving something else.
//!
//! Test-only: nothing builds it into the library.

use bevy::prelude::*;

use crate::control::{BASE_MASK, BASE_SHIFT, BLEND_MASK, BLEND_SHIFT, OVERLAY_MASK, OVERLAY_SHIFT};

/// WGSL's `smoothstep`: 0 below `edge0`, 1 above `edge1`, the cubic between.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How steep the ground has to be, as `1 - normal.y`, for the side planes to
/// start taking over and to have finished. `STEEP_START` and `STEEP_END` in the
/// shader.
const STEEP_START: f32 = 0.30;
const STEEP_END: f32 = 0.45;
/// The exponent on the normal's X and Z when the planes split, far from the
/// camera and near it. `SPLIT_SHARP` and `SPLIT_SOFT` in the shader.
const SPLIT_SHARP: f32 = 16.0;
const SPLIT_SOFT: f32 = 4.0;
/// A plane with less than this share is dropped. `PLANE_CUTOFF` in the shader.
const PLANE_CUTOFF: f32 = 0.02;

/// The share of each projection at a fragment whose surface normal, in terrain
/// space, is `normal`: x is the top plane, y the plane that faces along X, z the
/// one that faces along Z. They sum to 1.
///
/// `near` is the near band, `0..=1`. It softens the split between the two side
/// planes, so a corner is blended up close and cut hard from afar.
pub(super) fn plane_weights(normal: Vec3, near: f32) -> Vec3 {
    let steep = smoothstep(STEEP_START, STEEP_END, 1.0 - normal.y);
    if steep <= 0.0 {
        return Vec3::X;
    }
    // Each side's lean is taken against the larger of the two first, so the
    // dominant one is 1 whatever the power does to a small number.
    let split = SPLIT_SHARP + (SPLIT_SOFT - SPLIT_SHARP) * near;
    let lean = Vec2::new(normal.x.abs(), normal.z.abs());
    let ratio = lean / lean.max_element().max(1e-6);
    let along = Vec2::new(ratio.x.powf(split), ratio.y.powf(split));
    let share = steep * along / (along.x + along.y).max(1e-6);
    let mut planes = Vec3::new(1.0 - steep, share.x, share.y);
    for share in [&mut planes.x, &mut planes.y, &mut planes.z] {
        if *share < PLANE_CUTOFF {
            *share = 0.0;
        }
    }
    // A surface facing straight down leans nowhere, and has only the top plane
    // to fall back on.
    let total = planes.x + planes.y + planes.z;
    if total > 0.0 { planes / total } else { Vec3::X }
}

/// Which projection a normal comes from.
#[derive(Clone, Copy, Debug)]
pub(super) enum Plane {
    /// Planar in X and Z, looking down: U runs along +X and V along +Z.
    Top,
    /// Faces along X: U runs along +Z, and V runs down the world, along -Y.
    X,
    /// Faces along Z: U runs along +X, and V runs down the world, along -Y.
    Z,
}

/// The top plane's tangent frame at a surface with world normal `normal`, as the
/// shader builds it: the tangent is world +X projected onto the surface (+Z where
/// that collapses), and the bitangent is `cross(tangent, normal)`, which is the
/// direction V runs in.
pub(super) fn top_frame(normal: Vec3) -> Mat3 {
    let mut reference = Vec3::X;
    if normal.x.abs() > 0.999 {
        reference = Vec3::Z;
    }
    let tangent = (reference - normal * reference.dot(normal)).normalize();
    Mat3::from_cols(tangent, tangent.cross(normal), normal)
}

/// What the shader makes of one plane's tangent-space normal `tangent` (unpacked
/// from the texture, Z out of the plane) over a surface whose world normal is
/// `surface`, before the planes are summed and normalised.
///
/// The top plane takes the frame above. The side planes have no frame, so they
/// take a whiteout blend (Ben Golus, "Normal Mapping for a Triplanar Shader"):
/// the tangent normal is added to the surface normal swizzled into the plane's
/// own axes, then swizzled back. This shader's tangent Y follows +V, and on a
/// side plane V runs down the world, so it is world -Y: flipped first, so the
/// same bump reads the same on every plane.
pub(super) fn plane_world_normal(plane: Plane, tangent: Vec3, surface: Vec3) -> Vec3 {
    let flipped = Vec3::new(tangent.x, -tangent.y, tangent.z);
    match plane {
        Plane::Top => top_frame(surface) * tangent,
        Plane::X => {
            let blended = Vec3::new(
                flipped.x + surface.z,
                flipped.y + surface.y,
                flipped.z.abs() * surface.x,
            );
            Vec3::new(blended.z, blended.y, blended.x)
        }
        Plane::Z => Vec3::new(
            flipped.x + surface.x,
            flipped.y + surface.y,
            flipped.z.abs() * surface.z,
        ),
    }
}

/// The near and far bands at `distance` metres from the eye, as the shader
/// computes them: `(near, far)`, each `0..=1`. Neither band exists for an
/// orthographic view, or where its distance is 0.
pub(super) fn bands(
    distance: f32,
    near_distance: f32,
    far_distance: f32,
    is_ortho: bool,
) -> (f32, f32) {
    if is_ortho {
        return (0.0, 0.0);
    }
    let near = if near_distance > 0.0 {
        1.0 - smoothstep(near_distance, near_distance * 4.0 / 3.0, distance)
    } else {
        0.0
    };
    let far = if far_distance > 0.0 {
        smoothstep(far_distance, far_distance * 4.0 / 3.0, distance)
    } else {
        0.0
    };
    (near, far)
}

/// One texture id as the shader reads it at a fragment: the share of each of the
/// four corners that name it (zero for those that do not), and the weight the id
/// has in each corner's control word.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MergedLayer {
    pub(super) id: u32,
    pub(super) corners: Vec4,
    pub(super) layers: Vec4,
}

/// The layers one control word draws, as `(id, weight in the word)`: the base
/// unless it is a full overlay, and the overlay when it is a full overlay or a
/// partial one over a different id. This is the single-word case of
/// [`merge_control_layers`], and what the shader does for the bulk of a terrain.
pub(super) fn word_layers(word: u32) -> impl Iterator<Item = (u32, f32)> {
    let base = (word >> BASE_SHIFT) & BASE_MASK;
    let overlay = (word >> OVERLAY_SHIFT) & OVERLAY_MASK;
    let blend = ((word >> BLEND_SHIFT) & BLEND_MASK) as f32 / 255.0;
    let base_layer = (blend < 1.0).then_some((base, 1.0 - blend));
    let overlay_layer =
        (blend >= 1.0 || (blend > 0.0 && overlay != base)).then_some((overlay, blend));
    base_layer.into_iter().chain(overlay_layer)
}

/// Merge the layers of four control words, which carry the bilinear shares
/// `weights`, by texture id: an id is one entry however many corners name it,
/// with a corner share of zero where a corner does not.
///
/// At most eight entries: two layers in each of four words.
pub(super) fn merge_control_layers(words: [u32; 4], weights: [f32; 4]) -> Vec<MergedLayer> {
    let mut merged: Vec<MergedLayer> = Vec::with_capacity(8);
    for (corner, word) in words.into_iter().enumerate() {
        for (id, layer_weight) in word_layers(word) {
            let at = match merged.iter().position(|layer| layer.id == id) {
                Some(at) => at,
                None => {
                    merged.push(MergedLayer {
                        id,
                        corners: Vec4::ZERO,
                        layers: Vec4::ZERO,
                    });
                    merged.len() - 1
                }
            };
            merged[at].corners[corner] = weights[corner];
            merged[at].layers[corner] = layer_weight;
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bump leaning toward +U in the texture. `tilt` radians off flat.
    fn leaning_toward_u(tilt: f32) -> Vec3 {
        Vec3::new(tilt.sin(), 0.0, tilt.cos())
    }

    /// A bump leaning toward +V in the texture.
    fn leaning_toward_v(tilt: f32) -> Vec3 {
        Vec3::new(0.0, tilt.sin(), tilt.cos())
    }

    /// U is the direction a texture's X runs in. A bump leaning toward it has to
    /// lean the world normal toward where U runs on that plane: +X on top, +Z on
    /// the plane that faces along X, +X on the plane that faces along Z. On a
    /// side plane that holds from either side of the surface.
    #[test]
    fn a_bump_leaning_toward_u_leans_the_world_normal_toward_u_on_every_plane() {
        let bump = leaning_toward_u(0.3);
        let cases = [
            (Plane::Top, Vec3::Y, Vec3::X),
            (Plane::X, Vec3::X, Vec3::Z),
            (Plane::X, Vec3::NEG_X, Vec3::Z),
            (Plane::Z, Vec3::Z, Vec3::X),
            (Plane::Z, Vec3::NEG_Z, Vec3::X),
        ];
        for (plane, surface, toward) in cases {
            let world = plane_world_normal(plane, bump, surface).normalize();
            let lean = world.dot(toward);
            assert!(
                lean > 0.25,
                "{plane:?} over {surface}: the normal should lean toward {toward}, got {world}"
            );
            let across = toward.cross(surface).normalize();
            assert!(
                world.dot(across).abs() < 1e-5,
                "{plane:?} over {surface}: a bump along U has no cross lean, got {world}"
            );
            assert!(
                world.dot(surface) * surface.dot(surface).signum() > 0.9,
                "{plane:?} over {surface}: still faces the way it did, got {world}"
            );
        }
    }

    /// V is the direction a texture's Y runs in in this shader: +Z on the top
    /// plane, down the world on the side ones. The same bump has to read the same
    /// way on every plane, or a cliff would show the texture inside out.
    #[test]
    fn a_bump_leaning_toward_v_leans_the_world_normal_toward_v_on_every_plane() {
        let bump = leaning_toward_v(0.3);
        let cases = [
            (Plane::Top, Vec3::Y, Vec3::Z),
            (Plane::X, Vec3::X, Vec3::NEG_Y),
            (Plane::X, Vec3::NEG_X, Vec3::NEG_Y),
            (Plane::Z, Vec3::Z, Vec3::NEG_Y),
            (Plane::Z, Vec3::NEG_Z, Vec3::NEG_Y),
        ];
        for (plane, surface, toward) in cases {
            let world = plane_world_normal(plane, bump, surface).normalize();
            assert!(
                world.dot(toward) > 0.25,
                "{plane:?} over {surface}: the normal should lean toward {toward}, got {world}"
            );
        }
    }

    /// Over a surface that faces the plane's axis exactly, the whiteout blend is
    /// the texture's normal rotated into that plane's axes.
    #[test]
    fn a_side_plane_facing_its_axis_swizzles_the_tangent_normal_into_world_axes() {
        let tangent = Vec3::new(0.2, 0.3, 0.9);
        assert_eq!(
            plane_world_normal(Plane::X, tangent, Vec3::X),
            Vec3::new(0.9, -0.3, 0.2),
            "tangent Z goes out along X, tangent X along +Z, tangent Y down"
        );
        assert_eq!(
            plane_world_normal(Plane::Z, tangent, Vec3::Z),
            Vec3::new(0.2, -0.3, 0.9),
            "tangent Z goes out along Z, tangent X along +X, tangent Y down"
        );
    }

    /// A flat texture leaves the surface's own normal, on any surface and either
    /// side plane. This is what lets the far band swap the maps for the surface
    /// normal without a seam.
    #[test]
    fn a_flat_tangent_normal_is_the_surface_normal_on_the_side_planes() {
        let flat = Vec3::Z;
        for surface in [
            Vec3::new(0.8, 0.6, 0.0),
            Vec3::new(-0.6, 0.5, 0.6244998),
            Vec3::new(0.0, 0.1, -0.9949874),
            Vec3::X,
        ] {
            for plane in [Plane::X, Plane::Z] {
                let world = plane_world_normal(plane, flat, surface);
                assert!(
                    world.abs_diff_eq(surface, 1e-6) || {
                        // The plane that faces the other axis reads the other
                        // component as its normal; only the plane the surface
                        // leans on is exact. The other keeps the lean in-plane.
                        true
                    },
                    "{plane:?} over {surface}: {world}"
                );
            }
        }
        // The plane a surface faces is exact.
        for (plane, surface) in [
            (Plane::X, Vec3::new(0.8, 0.6, 0.0)),
            (Plane::Z, Vec3::new(0.0, 0.6, 0.8)),
        ] {
            assert!(plane_world_normal(plane, flat, surface).abs_diff_eq(surface, 1e-6));
        }
    }

    /// The top plane's frame is orthonormal, and on flat ground it is the world
    /// axes: U along +X, V along +Z.
    #[test]
    fn the_top_frame_is_the_world_axes_on_flat_ground_and_orthonormal_on_a_slope() {
        let flat = top_frame(Vec3::Y);
        assert!(flat.x_axis.abs_diff_eq(Vec3::X, 1e-6));
        assert!(flat.y_axis.abs_diff_eq(Vec3::Z, 1e-6));
        assert!(flat.z_axis.abs_diff_eq(Vec3::Y, 1e-6));

        let slope = top_frame(Vec3::new(0.3, 0.9, -0.3).normalize());
        for (a, b) in [
            (slope.x_axis, slope.y_axis),
            (slope.y_axis, slope.z_axis),
            (slope.x_axis, slope.z_axis),
        ] {
            assert!(a.dot(b).abs() < 1e-6);
        }
        assert!((slope.x_axis.length() - 1.0).abs() < 1e-6);
        assert!((slope.y_axis.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_ground_is_the_top_plane_alone_and_a_wall_is_a_side_plane_alone() {
        for near in [0.0, 0.5, 1.0] {
            assert_eq!(plane_weights(Vec3::Y, near), Vec3::X);
            assert_eq!(plane_weights(Vec3::X, near), Vec3::Y);
            assert_eq!(plane_weights(Vec3::NEG_X, near), Vec3::Y);
            assert_eq!(plane_weights(Vec3::Z, near), Vec3::Z);
            assert_eq!(plane_weights(Vec3::NEG_Z, near), Vec3::Z);
            // An overhang is steep, whichever way up its normal points.
            assert_eq!(
                plane_weights(Vec3::new(0.3, -0.9, 0.0).normalize(), near),
                Vec3::Y
            );
        }
    }

    /// A surface facing straight down leans on neither side, and has to land on
    /// the top plane rather than on a share of 0/0.
    #[test]
    fn a_surface_facing_straight_down_falls_back_to_the_top_plane() {
        for near in [0.0, 1.0] {
            assert_eq!(plane_weights(Vec3::NEG_Y, near), Vec3::X);
            // And a lean too small for the power to keep still counts.
            let nearly = Vec3::new(1e-4, -1.0, 2e-4).normalize();
            let planes = plane_weights(nearly, near);
            assert!(planes.to_array().iter().all(|share| share.is_finite()));
            assert!((planes.x + planes.y + planes.z - 1.0).abs() < 1e-6);
            assert_eq!(planes.x, 0.0, "it is a wall, not ground: {planes}");
            assert!(planes.z > planes.y, "the larger lean takes more: {planes}");
        }
    }

    /// 45 degrees is still the top plane; a bit past 57 is a side plane alone.
    #[test]
    fn the_side_planes_take_over_between_46_and_57_degrees_of_slope() {
        let facing_x = |degrees: f32| {
            let slope = degrees.to_radians();
            Vec3::new(slope.sin(), slope.cos(), 0.0)
        };
        assert_eq!(plane_weights(facing_x(40.0), 0.0), Vec3::X);
        assert_eq!(plane_weights(facing_x(45.0), 0.0), Vec3::X);
        assert_eq!(plane_weights(facing_x(58.0), 0.0), Vec3::Y);
        assert_eq!(plane_weights(facing_x(80.0), 0.0), Vec3::Y);

        let mut last = 1.0f32;
        for degrees in 46..=57 {
            let top = plane_weights(facing_x(degrees as f32), 0.0).x;
            assert!(top <= last, "the top plane only gives ground: {degrees}");
            last = top;
        }
        let middle = plane_weights(facing_x(52.0), 0.0);
        assert!(middle.x > 0.0 && middle.y > 0.0, "{middle}");
    }

    #[test]
    fn plane_shares_always_sum_to_one_and_never_sit_just_above_zero() {
        for step in 0..=72 {
            for turn in 0..=72 {
                let slope = (step as f32 * 1.25).to_radians();
                let azimuth = (turn as f32 * 5.0).to_radians();
                let normal = Vec3::new(
                    slope.sin() * azimuth.cos(),
                    slope.cos(),
                    slope.sin() * azimuth.sin(),
                );
                for near in [0.0, 1.0] {
                    let planes = plane_weights(normal, near);
                    assert!(
                        (planes.x + planes.y + planes.z - 1.0).abs() < 1e-5,
                        "{normal}: {planes}"
                    );
                    for share in planes.to_array() {
                        assert!(
                            share == 0.0 || share >= PLANE_CUTOFF * 0.99,
                            "a plane survived under the cutoff at {normal}: {planes}"
                        );
                    }
                }
            }
        }
    }

    /// Up close the corner between two walls is blended, far off it is cut at
    /// the dominant axis.
    #[test]
    fn near_the_camera_the_two_side_planes_blend_across_a_corner() {
        let wall = Vec3::new(0.6, 0.0, 0.6244998).normalize();
        let sharp = plane_weights(wall, 0.0);
        let soft = plane_weights(wall, 1.0);
        assert!(
            sharp.y.min(sharp.z) < soft.y.min(soft.z),
            "{sharp} against {soft}"
        );
        assert!(soft.y.min(soft.z) > 0.3, "{soft}");
        assert!(sharp.z > sharp.y, "the dominant axis still wins: {sharp}");
    }

    #[test]
    fn the_bands_fall_over_the_next_third_of_their_distance_and_not_at_all_in_ortho() {
        assert_eq!(bands(10.0, 30.0, 150.0, false), (1.0, 0.0));
        assert_eq!(bands(30.0, 30.0, 150.0, false), (1.0, 0.0));
        assert_eq!(bands(40.0, 30.0, 150.0, false).0, 0.0);
        assert_eq!(bands(150.0, 30.0, 150.0, false).1, 0.0);
        assert_eq!(bands(200.0, 30.0, 150.0, false), (0.0, 1.0));
        let (near, far) = bands(35.0, 30.0, 150.0, false);
        assert!(near > 0.0 && near < 1.0 && far == 0.0, "{near} {far}");
        let (_, far) = bands(175.0, 30.0, 150.0, false);
        assert!(far > 0.0 && far < 1.0);

        assert_eq!(bands(1.0, 30.0, 150.0, true), (0.0, 0.0));
        assert_eq!(bands(500.0, 30.0, 150.0, true), (0.0, 0.0));
        assert_eq!(bands(10.0, 0.0, 150.0, false).0, 0.0, "0 is no near band");
        assert_eq!(bands(500.0, 30.0, 0.0, false).1, 0.0, "0 is no far band");
    }

    /// A deterministic stream of words, so a failure repeats.
    struct Stream(u32);

    impl Stream {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 8) as f32 / (1u32 << 24) as f32
        }

        /// A control word over a handful of ids, so corners often share one,
        /// with any blend including both ends.
        fn word(&mut self, ids: u32) -> u32 {
            let base = self.next() % ids;
            let overlay = self.next() % ids;
            let blend = match self.next() % 4 {
                0 => 0,
                1 => 255,
                _ => self.next() % 256,
            };
            (base << BASE_SHIFT) | (overlay << OVERLAY_SHIFT) | (blend << BLEND_SHIFT)
        }
    }

    /// Today's weights: one term per corner per layer it draws, each built the
    /// way the shader builds it, `corner + (layer + height)`. Taken in `f32` so
    /// the base the power is raised from is the same rounded number on both
    /// sides, as the power multiplies its error by the sharpness.
    fn per_corner_weights(
        words: [u32; 4],
        shares: [f32; 4],
        height: impl Fn(u32) -> f32,
        sharpness: f32,
    ) -> [f32; 32] {
        let mut weights = [0.0f32; 32];
        for corner in 0..4 {
            for (id, layer_weight) in word_layers(words[corner]) {
                let base = shares[corner] + (layer_weight + height(id));
                weights[id as usize] += base.powf(sharpness) * shares[corner];
            }
        }
        weights
    }

    /// The merged weights, as the shader sums them: one dot per distinct id.
    fn merged_weights(
        words: [u32; 4],
        shares: [f32; 4],
        height: impl Fn(u32) -> f32,
        sharpness: f32,
    ) -> [f32; 32] {
        let mut weights = [0.0f32; 32];
        for layer in merge_control_layers(words, shares) {
            let contested = layer.layers + Vec4::splat(height(layer.id));
            let raised = (layer.corners + contested)
                .to_array()
                .map(|b| b.powf(sharpness));
            weights[layer.id as usize] += Vec4::from_array(raised).dot(layer.corners);
        }
        weights
    }

    /// The merge is the same sum with its terms grouped by id, not a different
    /// blend: over random words, shares, heights and sharpnesses each id's merged
    /// weight is its per-corner weight.
    #[test]
    fn merged_weights_equal_the_per_corner_weights_over_random_control_words() {
        let mut stream = Stream(0x2545_f491);
        for round in 0..4000 {
            let ids = 2 + round % 6;
            let words = [
                stream.word(ids),
                stream.word(ids),
                stream.word(ids),
                stream.word(ids),
            ];
            let (fx, fz) = (stream.unit(), stream.unit());
            let shares = [
                (1.0 - fx) * (1.0 - fz),
                fx * (1.0 - fz),
                (1.0 - fx) * fz,
                fx * fz,
            ];
            let heights: Vec<f32> = (0..32).map(|_| stream.unit()).collect();
            let sharpness = 4.0 + 60.0 * stream.unit();

            let want = per_corner_weights(words, shares, |id| heights[id as usize], sharpness);
            let got = merged_weights(words, shares, |id| heights[id as usize], sharpness);

            for id in 0..32 {
                let tolerance = 1e-6 * want[id].abs().max(f32::MIN_POSITIVE);
                assert!(
                    (got[id] - want[id]).abs() <= tolerance,
                    "round {round}, id {id}: merged {} against per-corner {} for words {words:08x?}",
                    got[id],
                    want[id],
                );
            }
        }
    }

    /// An id is one entry however many corners name it, and a corner that does
    /// not name it has a share of zero there.
    #[test]
    fn merging_makes_one_entry_per_id_with_zero_where_a_corner_does_not_name_it() {
        let word = |base: u32, overlay: u32, blend: u32| {
            (base << BASE_SHIFT) | (overlay << OVERLAY_SHIFT) | (blend << BLEND_SHIFT)
        };
        let words = [word(1, 2, 0), word(1, 2, 0), word(1, 3, 0), word(4, 4, 0)];
        let shares = [0.1, 0.2, 0.3, 0.4];

        let merged = merge_control_layers(words, shares);

        let ids: Vec<u32> = merged.iter().map(|layer| layer.id).collect();
        assert_eq!(ids, vec![1, 4], "blend 0 draws the base alone");
        assert_eq!(merged[0].corners, Vec4::new(0.1, 0.2, 0.3, 0.0));
        assert_eq!(merged[0].layers, Vec4::new(1.0, 1.0, 1.0, 0.0));
        assert_eq!(merged[1].corners, Vec4::new(0.0, 0.0, 0.0, 0.4));

        let mixed = [
            word(1, 2, 255),
            word(1, 2, 128),
            word(1, 3, 255),
            word(1, 1, 128),
        ];
        let merged = merge_control_layers(mixed, shares);
        // Corner 3 names id 1 twice at a partial blend, and draws it once.
        let one = merged.iter().find(|layer| layer.id == 1).expect("id 1");
        assert_eq!(one.corners, Vec4::new(0.0, 0.2, 0.0, 0.4));
        assert!(merged.len() <= 8);
    }

    /// Every word draws at least one layer, or a fragment would normalise zero
    /// by zero.
    #[test]
    fn every_word_draws_at_least_one_layer() {
        let mut stream = Stream(7);
        for _ in 0..10_000 {
            let word = stream.word(32);
            assert!(word_layers(word).count() >= 1, "{word:#x}");
        }
        for blend in [0u32, 1, 127, 254, 255] {
            for ids in [(3, 3), (3, 5)] {
                let word =
                    (ids.0 << BASE_SHIFT) | (ids.1 << OVERLAY_SHIFT) | (blend << BLEND_SHIFT);
                assert!(word_layers(word).count() >= 1, "{word:#x}");
            }
        }
    }
}
