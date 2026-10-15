// Terrain splat shading: two texture ids per control texel, blended by comparing
// their height maps.
//
// The control map is `texture_2d<u32>` read with `textureLoad` only: a packed
// control word must never be filtered into a different packed word. Four texels
// are read per fragment and their decoded contributions are mixed with bilinear
// weights.
//
// A terrain with autoterrain on replaces the word of every cell no hand has
// claimed with one its own slope stands for. That slope comes from `slope_map`,
// which holds the ground's own slope per grid point, and never from the surface
// being drawn: the surface is a clipmap, so a slope taken from the mesh normal
// reads gentler with distance.
//
// Every texture read is `textureSampleGrad` with derivatives taken once in
// uniform control flow at the top of `fragment`: the per-layer branches below
// make implicit-derivative sampling illegal. The two plain `textureSample`s, the
// tint and the colour map, sit in that same uniform region.
//
// Two bands are set by the distance from the camera, and neither applies to an
// orthographic view (a shadow cascade has no eye to measure from):
//
//   near   within `near_distance`, fading out over the next third: parallax
//          relief on the dominant texture, a grain read at four times its
//          tiling, and a wider blend between the planes of a cliff.
//   far    past `far_distance`, fading in over the next third: the normal,
//          occlusion and roughness maps are not read at all, and the slot
//          averages stand in for them. Albedo and height stay in every band, as
//          the height blend needs them, and so does detiling.
//
// Steep ground is projected along the terrain's own X and Z axes as well as from
// above, so a cliff is not a texture smeared down a slope. The top plane is the
// planar projection the shader has always used, detiling and all; the two side
// planes are not detiled.
//
// Cost, per fragment: flat ground in the far band reads albedo and height once
// per texture id and tap, less than before. Flat ground in the mid band reads
// what it always did. A fragment on a cliff reads a second plane. The relief
// march is the only loop, and it runs only in the near band, on slots that have
// a height map of their own, when the relief depth is set.
//
// Four optional parts are compiled only for a material that uses them, so a
// terrain that has none of them pays for none of them. `TerrainSplatKey` in
// render/mod.rs keys the pipeline on what the material has and sets the defs:
//
//   TERRAIN_RELIEF      relief depth above 0: the relief march
//   TERRAIN_GRAIN       grain above 0: the grain read
//   TERRAIN_COLOUR_MAP  a colour map: its sample, and the mix of it with the
//                       texture's own colour
//   TERRAIN_NORMAL_MAP  a normal map: the surface is lit from it per pixel
//                       rather than from the mesh's normal, as far as each
//                       texel's alpha says the map still holds
//   TERRAIN_DOMINANT    either of the first two: the choice of the texture they
//                       are made from
//
// The bands, the side projections and the far band's savings are in every
// variant.

#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_bindings::mesh,
    mesh_functions::{get_local_from_world, get_world_from_local},
    mesh_view_bindings::view,
    pbr_functions,
    pbr_types,
}

// The remap of a 0..1 authoring dial onto a power exponent. 4 is a wide,
// soft cross-fade; 64 is a near-binary height cutout.
const SHARPNESS_MIN: f32 = 4.0;
const SHARPNESS_RANGE: f32 = 60.0;

const BASE_SHIFT: u32 = 0u;
const BASE_MASK: u32 = 0x1Fu;
const OVERLAY_SHIFT: u32 = 5u;
const OVERLAY_MASK: u32 = 0x1Fu;
const BLEND_SHIFT: u32 = 10u;
const BLEND_MASK: u32 = 0xFFu;
// A hand painted this cell, so its own word is what it draws.
const MANUAL_BIT: u32 = 0x40000u;

// Entries in `uv_scales`, which is four `vec4`s wide.
const MAX_LAYERS: u32 = 16u;

// What a slot that sets no reflectance shades at: a StandardMaterial's
// default, 4% at normal incidence.
const DEFAULT_REFLECTANCE: f32 = 0.5;

// Rec.709 luminance weights. The albedo array is sRGB, so what is sampled from
// it is already linear and these apply as they stand.
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// How steep the ground has to be, as `1 - normal.y`, for the side planes to
// start and to finish taking over from the top one. 0.30 is about 46 degrees
// of slope and 0.45 about 57.
//
// A deliberate change of look, not a fix: every terrain used to be projected
// from above all the way up a cliff, stretching its texture down the face, and
// ground steeper than STEEP_START now takes the side projection instead. Gentler
// ground, flat ground above all, is the top plane alone and draws as it did.
const STEEP_START: f32 = 0.30;
const STEEP_END: f32 = 0.45;
// How sharply the two side planes split, as the exponent on the normal's X and
// Z components. High picks the dominant axis; low blends the corner. Near the
// camera the corner is blended, far from it the split is hard and cheap.
const SPLIT_SHARP: f32 = 16.0;
const SPLIT_SOFT: f32 = 4.0;
// A plane with less than this share is not sampled at all.
const PLANE_CUTOFF: f32 = 0.02;

// The grain read is this many times finer than the texture's own tiling.
const GRAIN_TILING: f32 = 4.0;
// The most the grain may scale the albedo by. A texel many times brighter than
// its texture's average is a speck, not detail.
const GRAIN_RATIO_MAX: f32 = 4.0;
// The least mean luminance the grain divides by.
const GRAIN_MIN_LUMINANCE: f32 = 0.01;

// The smallest sum of height-blend weights worth dividing by. A power of 64 on
// a base under one is small, not zero, and the albedo is the weighted sum over
// it, so the floor has to sit below any weight the powers can leave.
const MIN_TOTAL_WEIGHT: f32 = 1e-30;

// The relief march, after `parallax_field_trace` in crates/parallax of the
// Hayjump monorepo: a step budget from the grazing angle, then a binary search
// between the last sample above the surface and the first below it.
const RELIEF_MIN_STEPS: f32 = 8.0;
const RELIEF_MAX_STEPS: f32 = 32.0;
const RELIEF_REFINE_STEPS: u32 = 5u;
// An open surface has no face to leave through, so a view that skims the ground
// would march far sideways. The ray is cut to the length that crosses the shell
// at this steepness (its vertical share), which bounds the shift to four shell
// depths.
const RELIEF_MIN_STEEPNESS: f32 = 0.25;
// A detiled texture is marched once per tile it is blended from, so each march
// gets this share of the step budget the near band would give a lone one.
const RELIEF_TILED_BUDGET: f32 = 0.5;

struct SplatUniform {
    // 16 per-id UV scales, packed four to a vec4 for uniform alignment.
    uv_scales: array<vec4<f32>, 4>,
    // 16 per-id detiling strengths, packed the same way. 0 is off.
    detile_strengths: array<vec4<f32>, 4>,
    // 16 per-id perceptual roughness values a slot sets of its own, packed
    // the same way: a scale on the slot's roughness map, or the roughness
    // where it has none. Negative where it sets none, which leaves the map
    // as it is, or the terrain's own roughness.
    slot_roughness: array<vec4<f32>, 4>,
    // 16 per-id reflectance values, packed and unset the same way.
    slot_reflectance: array<vec4<f32>, 4>,
    // Terrain XZ extent in world units, for turning UV0 back into a
    // terrain-local position. Local, not world, so a terrain that moves
    // carries its texturing with it.
    terrain_size: vec2<f32>,
    blend_sharpness: f32,
    perceptual_roughness: f32,
    control_resolution: u32,
    layer_count: u32,
    // Nonzero where this terrain textures the cells no hand has claimed
    // from their slope. 0 leaves every cell to its own control word.
    autoterrain_enabled: u32,
    autoterrain_base_slot: u32,
    autoterrain_slope_slot: u32,
    // The slope range in radians, converted from the authored degrees on
    // the way in so this side carries no conversion of its own.
    autoterrain_slope_start: f32,
    autoterrain_slope_end: f32,
    // How much of the tint texture reaches the finished albedo, 0..1.
    // The layer's own white is the identity, so a terrain that has never
    // been tinted draws the same at every strength.
    tint_strength: f32,
    // Bit i is set where texture id i has an occlusion map. An id without
    // one shades unoccluded instead of reading its filled array layer.
    occlusion_slots: u32,
    // Bit i is set where texture id i has a roughness map. An id without
    // one takes `perceptual_roughness`.
    roughness_slots: u32,
    // Per id: x = how much of the slot's own texture colour it keeps against
    // the colour map (1 keeps all of it), y = the mean luminance of its albedo
    // (linear, never under 0.01), z = the mean of its roughness map's green
    // (1 without a map), w = the mean of its occlusion map's red (1 without).
    slot_stats: array<vec4<f32>, 16>,
    // Metres from the camera within which the near band applies, and past which
    // the far band does. 0 turns the band off.
    near_distance: f32,
    far_distance: f32,
    // Depth of the parallax relief below the surface, in metres. 0 is off.
    relief_depth: f32,
    // How much of the textures' finer grain the near band lays over the colour
    // map, 0..1.
    grain: f32,
    // Bit i is set where texture id i has a height map of its own, not the
    // filled 0.5 layer a slot without one gets. Relief needs a real one.
    height_slots: u32,
    // Texels per grid cell of the colour map, 0 where the terrain has none.
    colour_map_scale: u32,
    // Texels along one edge of the colour map.
    colour_map_side: u32,
    // Texels per grid cell of the normal map, 0 where the terrain has none.
    normal_map_scale: u32,
    // Texels along one edge of the normal map.
    normal_map_side: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> splat: SplatUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var layer_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var height_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var control_map: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var slope_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var tint_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var tint_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var occlusion_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var roughness_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var colour_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var colour_map_sampler: sampler;
// The normal map has no sampler of its own: it is read with `colour_map_sampler`,
// which filters and clamps the same way. Metal allows 16 samplers a stage and
// the splat material, with the view's, is close to using them.
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var normal_map: texture_2d<f32>;

// Everything about the fragment that every texture id's accumulation shares,
// worked out once at the top of `fragment`.
//
// Private state rather than parameters: the accumulation runs once per id and
// the struct is read from several places, and nothing in it is an array, so
// nothing is copied per call.
struct Frame {
    // The fragment in terrain-local space, and how it moves across the screen.
    // The two side planes are projected from this.
    local: vec3<f32>,
    local_ddx: vec3<f32>,
    local_ddy: vec3<f32>,
    // The top plane's position, planar in X and Z, and how it moves. Taken from
    // UV0 rather than from `local`, which keeps it bit for bit what it was, and
    // shifted by the relief.
    top: vec2<f32>,
    top_ddx: vec2<f32>,
    top_ddy: vec2<f32>,
    // The share of each projection, summing to 1: x is the top plane, y the
    // plane that faces along X, z the plane that faces along Z.
    planes: vec3<f32>,
    // The distance bands, each 0..1. Neither is ever above 0 for an
    // orthographic view.
    near: f32,
    far: f32,
    // The surface's normal in terrain space, before any texture has bent it. The
    // side planes are the terrain's own axes, so their whiteout blend is done in
    // its space, and turned into world axes once at the end.
    local_normal: vec3<f32>,
#ifdef TERRAIN_COLOUR_MAP
    // The colour map under the fragment, linear.
    colour: vec3<f32>,
#endif
}

var<private> frame: Frame;

// The last id this material can address: one short of the bound set's
// layer count, and never past the UV-scale array. Every id is clamped
// through here exactly once, so the layer sampled and the UV scale
// applied can never disagree.
fn last_layer() -> u32 {
    return min(max(splat.layer_count, 1u), MAX_LAYERS) - 1u;
}

// `id` must already be clamped by `last_layer`.
fn uv_scale_for(id: u32) -> f32 {
    return splat.uv_scales[id / 4u][id % 4u];
}

// `id` must already be clamped by `last_layer`.
fn detile_for(id: u32) -> f32 {
    return splat.detile_strengths[id / 4u][id % 4u];
}

// A slot's own roughness, negative where it sets none. `id` must already be
// clamped by `last_layer`. Read straight from the uniform: handing the packed
// array to a function would copy it on every call.
fn slot_roughness_for(id: u32) -> f32 {
    return splat.slot_roughness[id / 4u][id % 4u];
}

// A slot's own reflectance, negative where it sets none. `id` must already be
// clamped by `last_layer`.
fn slot_reflectance_for(id: u32) -> f32 {
    return splat.slot_reflectance[id / 4u][id % 4u];
}

fn luminance(colour: vec3<f32>) -> f32 {
    return dot(colour, LUMA);
}

// Two uncorrelated values in 0..1 from a tile coordinate. Integer-valued
// input, so the hash only has to separate whole numbers.
fn tile_hash(tile: vec2<f32>) -> vec2<f32> {
    var h = fract(vec3<f32>(tile.x, tile.y, tile.x + tile.y) * vec3<f32>(0.1031, 0.1030, 0.0973));
    h += dot(h, h.yzx + 33.33);
    return fract((h.xx + h.yz) * h.zy);
}

// One tile's own rigid sample frame: the tile's rotation and shift are applied
// whole, and the four tiles around a fragment are blended after each has been
// sampled in its own frame.
struct TileTap {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
    // The rotation that produced `uv`, as a unit `(cos, sin)`. Anything
    // sampled through it that is itself a direction - a tangent-space
    // normal - has to turn back by the same angle.
    turn: vec2<f32>,
    weight: f32,
}

// The tiles one fragment reads, with the share each of them takes.
struct Sampling {
    taps: array<TileTap, 4>,
    count: u32,
}

// `v` turned by the unit `(cos, sin)` in `turn`.
fn turned_by(v: vec2<f32>, turn: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(turn.x * v.x - turn.y * v.y, turn.y * v.x + turn.x * v.y);
}

// Break a tiled texture out of its grid.
//
// Every tile of the repeat is turned and shifted by its own amount, drawn from a
// hash of its coordinate. The four nearest tiles are each sampled in their own
// frame and the sampled values blended with the bilinear weights of the position
// between their centres, so a tile's weight reaches zero where it is left and the
// blend is continuous across the plane.
//
// Sampling per tile, rather than blending the four tiles' coordinates and
// sampling once, is the whole point: interpolating between rigid transforms is
// not a rigid transform, and the blended coordinate field stretches and shears
// between tile centres. This is the standard technique (Inigo Quilez's texture
// repetition; Unity HDRP and Unreal call it procedural or stochastic tiling).
//
// Each tap carries its own derivatives, and a rigid frame makes them exact. The
// rotation pivots on the tile centres rather than the texture origin, so the
// displacement does not grow with how far the terrain runs.
//
// `strength` 0 returns a single tap holding exactly what came in.
fn detile(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, strength: f32) -> Sampling {
    var out: Sampling;
    out.count = 1u;
    out.taps[0].uv = uv;
    out.taps[0].ddx = ddx;
    out.taps[0].ddy = ddy;
    out.taps[0].turn = vec2<f32>(1.0, 0.0);
    out.taps[0].weight = 1.0;
    if strength <= 0.0 {
        return out;
    }

    // Half a tile back, so `base` names the four tile centres around the
    // sample rather than the corners of the tile it is inside.
    let p = uv - vec2<f32>(0.5);
    let base = floor(p);
    let f = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), fract(p));

    out.count = 4u;
    for (var j = 0u; j < 2u; j++) {
        for (var i = 0u; i < 2u; i++) {
            let corner = base + vec2<f32>(f32(i), f32(j));
            let weight = mix(1.0 - f.x, f.x, f32(i)) * mix(1.0 - f.y, f.y, f32(j));
            let hash = tile_hash(corner);
            let angle = (hash.x * 2.0 - 1.0) * strength * 3.14159265;
            let shift = (hash - vec2<f32>(0.5)) * strength;
            let pivot = corner + vec2<f32>(0.5);
            let turn = vec2<f32>(cos(angle), sin(angle));
            let index = j * 2u + i;
            out.taps[index].uv = pivot + turned_by(uv - pivot, turn) + shift;
            out.taps[index].ddx = turned_by(ddx, turn);
            out.taps[index].ddy = turned_by(ddy, turn);
            out.taps[index].turn = turn;
            out.taps[index].weight = weight;
        }
    }
    return out;
}

// The texel a grid coordinate's lower corner sits in. `control_resolution`
// texels cover grid points 0..resolution-1, and the upper corner of the
// bilinear tap is this + 1, so the base clamps one short of the end.
fn control_corner(g: vec2<f32>) -> vec2<i32> {
    let last = i32(max(splat.control_resolution, 2u) - 2u);
    return clamp(vec2<i32>(floor(g)), vec2<i32>(0), vec2<i32>(last));
}

fn load_control(coord: vec2<i32>) -> u32 {
    let last = i32(max(splat.control_resolution, 1u) - 1u);
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(last));
    return textureLoad(control_map, clamped, 0).r;
}

fn load_slope(coord: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(slope_map)) - vec2<i32>(1);
    let clamped = clamp(coord, vec2<i32>(0), max(last, vec2<i32>(0)));
    return textureLoad(slope_map, clamped, 0).r;
}

// The stored slope under a fragment, bilinear across the four grid points around
// it.
//
// Filtered rather than nearest so the band between the two textures moves
// smoothly across a cell, and taken on the slope rather than on the two control
// words it produces so the four corners agree wherever the ground is uniformly
// flat or steep, which keeps the single-word fast path below covering most of a
// terrain.
fn slope_at(corner: vec2<i32>, f: vec2<f32>) -> f32 {
    let s00 = load_slope(corner);
    let s10 = load_slope(corner + vec2<i32>(1, 0));
    let s01 = load_slope(corner + vec2<i32>(0, 1));
    let s11 = load_slope(corner + vec2<i32>(1, 1));
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

struct Accum {
    albedo: vec3<f32>,
    // The top plane's normals, tangent space, summed by weight.
    normal_top: vec3<f32>,
    // The side planes' normals, already in world space, summed by weight.
    normal_side: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    reflectance: f32,
    total: f32,
}

// One texture id's contribution, summed over the corners that named it.
//
// `corner_weights` holds the four bilinear corner shares, with zero in the
// corners that did not name this id, and `layer_weights` the weight this id has
// in each corner's word. The weight is
//
//     sum over corners of pow(corner_w + layer_w + height, sharpness) * corner_w
//
// which is the height-blend curve. `corner_w` rides inside the power so a near
// corner gets a height advantage, and multiplies outside it so a far corner
// cannot win on height alone. A corner that did not name the id has a
// `corner_w` of zero, so whatever its `layer_w` is changes nothing.
//
// Taking the four corners as vectors rather than one call per corner is what
// makes both paths in `fragment` exact: the caller merges corners that name the
// same id, so each id is sampled once, and the sum above is the same sum the
// separate per-corner calls made.
//
// The id is sampled on every plane that has a share of this fragment and the
// planes are blended by their shares, so albedo, height, occlusion and
// roughness are one value per id like they always were. The normals are the
// exception: the top plane keeps its tangent-space normal, which `fragment`
// turns by the one frame it has, and each side plane's normal is brought into
// the terrain's axes by a whiteout blend here, because a side plane has no
// frame.
//
// `single` is set when this is the only id the fragment reads. Its weight then
// cancels in the normalisation whatever its height is, so the height is not read
// and the weight is 1: exact, and one array a tap cheaper.
fn accumulate(
    accum: Accum,
    id: u32,
    corner_weights: vec4<f32>,
    layer_weights: vec4<f32>,
    single: bool,
) -> Accum {
    let layer_id = min(id, last_layer());
    let layer = i32(layer_id);
    let scale = uv_scale_for(layer_id);
    let stats = splat.slot_stats[layer_id];
    let has_occlusion = ((splat.occlusion_slots >> layer_id) & 1u) != 0u;
    let has_roughness = ((splat.roughness_slots >> layer_id) & 1u) != 0u;
    // Past the far band the maps below the albedo are not read, and the slot's
    // averages stand in for them.
    let detailed = frame.far < 1.0;

    var albedo = vec3<f32>(0.0);
    var height = 0.0;
    var normal_top = vec3<f32>(0.0);
    var normal_side = vec3<f32>(0.0);
    // The slot's average stands in for the map until the planes below have
    // something better to say, which is where the far band stays.
    var occlusion = stats.w;
    var roughness_sample = stats.z;
    if detailed {
        occlusion = 0.0;
        roughness_sample = 0.0;
    }
    for (var plane = 0u; plane < 3u; plane++) {
        let plane_weight = frame.planes[plane];
        if plane_weight <= 0.0 {
            continue;
        }
        // This plane's position and derivatives in repeats of the texture. The
        // side planes' V runs down the image while Y runs up the world.
        var uv = frame.top * scale;
        var ddx = frame.top_ddx * scale;
        var ddy = frame.top_ddy * scale;
        var strength = detile_for(layer_id);
        if plane == 1u {
            uv = vec2<f32>(frame.local.z, -frame.local.y) * scale;
            ddx = vec2<f32>(frame.local_ddx.z, -frame.local_ddx.y) * scale;
            ddy = vec2<f32>(frame.local_ddy.z, -frame.local_ddy.y) * scale;
            strength = 0.0;
        } else if plane == 2u {
            uv = vec2<f32>(frame.local.x, -frame.local.y) * scale;
            ddx = vec2<f32>(frame.local_ddx.x, -frame.local_ddx.y) * scale;
            ddy = vec2<f32>(frame.local_ddy.x, -frame.local_ddy.y) * scale;
            strength = 0.0;
        }
        let tiled = detile(uv, ddx, ddy, strength);

        // One sample per tile, blended after sampling: a rigid UV and its own
        // derivatives per tap, so no tap is stretched by its neighbours. A
        // detiled top plane takes four taps of up to five arrays, and every
        // other plane one tap; the far band reads two arrays a tap, not five.
        var albedo_sum = vec3<f32>(0.0);
        var height_sum = 0.0;
        var normal_sum = vec3<f32>(0.0);
        var occlusion_sum = 0.0;
        var roughness_sum = 0.0;
        var taken = 0.0;
        for (var t = 0u; t < tiled.count; t++) {
            let tap = tiled.taps[t];
            // A tile whose share has fallen to nothing is three samples that
            // change no pixel. The gradients are explicit, so this branch does
            // not have to be uniform.
            if tap.weight <= 1e-4 {
                continue;
            }
            albedo_sum += tap.weight
                * textureSampleGrad(albedo_array, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy).rgb;
            if !single {
                height_sum += tap.weight
                    * textureSampleGrad(height_array, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy).r;
            }
            taken += tap.weight;
            if !detailed {
                continue;
            }
            let packed =
                textureSampleGrad(normal_array, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy).rgb;
            let unpacked = packed * 2.0 - 1.0;
            // This tap's UV was turned by `tap.turn`, so its texture reads
            // rotated by the inverse across the surface and the tangent-space
            // normal stored in it has to turn back before the tiles are
            // blended. Green is already in the shader's convention:
            // `flip_normal_y` is applied on the way into the array.
            let back = turned_by(unpacked.xy, vec2<f32>(tap.turn.x, -tap.turn.y));
            normal_sum += tap.weight * vec3<f32>(back, unpacked.z);
            // A layer with no map of its own holds a filled array layer, so
            // it is not sampled: it contributes the identity instead,
            // unoccluded and at full roughness before the slot's own value.
            if has_occlusion {
                occlusion_sum += tap.weight * textureSampleGrad(
                    occlusion_array, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy).r;
            } else {
                occlusion_sum += tap.weight;
            }
            if has_roughness {
                roughness_sum += tap.weight * textureSampleGrad(
                    roughness_array, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy).g;
            } else {
                roughness_sum += tap.weight;
            }
        }
        let inv_taps = 1.0 / max(taken, 1e-8);
        albedo += plane_weight * albedo_sum * inv_taps;
        height += plane_weight * height_sum * inv_taps;
        if detailed {
            let tangent = normal_sum * inv_taps;
            if plane == 0u {
                normal_top += plane_weight * tangent;
            } else {
                // Whiteout blend (Ben Golus, "Normal Mapping for a Triplanar
                // Shader"): the plane's tangent-space normal, added to the
                // surface normal swizzled into the plane's own axes, then
                // swizzled back out. The planes are the terrain's own axes, so
                // all of this is in the terrain's space and a terrain turned
                // about Y keeps its cliffs right; `fragment` turns the sum into
                // world axes once. This shader's tangent Y follows +V, which on
                // a side plane is down the image, so it is terrain -Y; flipping
                // it first leaves everything below in terrain orientation, and
                // the same bump reads the same on every plane. `abs` keeps the
                // plane's back side from inverting.
                let flipped = vec3<f32>(tangent.x, -tangent.y, tangent.z);
                if plane == 1u {
                    // U runs along +Z; the surface's X is the plane's normal.
                    let blended = vec3<f32>(
                        flipped.xy + frame.local_normal.zy,
                        abs(flipped.z) * frame.local_normal.x,
                    );
                    normal_side += plane_weight * blended.zyx;
                } else {
                    // U runs along +X; the surface's Z is the plane's normal.
                    let blended = vec3<f32>(
                        flipped.xy + frame.local_normal.xy,
                        abs(flipped.z) * frame.local_normal.z,
                    );
                    normal_side += plane_weight * blended;
                }
            }
            occlusion += plane_weight * occlusion_sum * inv_taps;
            roughness_sample += plane_weight * roughness_sum * inv_taps;
        }
    }
    if detailed && frame.far > 0.0 {
        // The fade ring: both, mixed by how far into it the fragment is.
        occlusion = mix(occlusion, stats.w, frame.far);
        roughness_sample = mix(roughness_sample, stats.z, frame.far);
    }

#ifdef TERRAIN_COLOUR_MAP
    // With a colour map the colour is the map's and the texture supplies only
    // its light and dark, relative to its own average, as far as the slot
    // gives up its colour. A slot that keeps all of it draws as it always did.
    if splat.colour_map_scale > 0u && stats.x < 1.0 {
        let grainy = frame.colour * (luminance(albedo) / stats.y);
        albedo = mix(grainy, albedo, stats.x);
    }
#endif

    let own_roughness = slot_roughness_for(layer_id);
    var roughness_scale = 1.0;
    if own_roughness >= 0.0 {
        roughness_scale = own_roughness;
    } else if !has_roughness {
        roughness_scale = splat.perceptual_roughness;
    }
    let own_reflectance = slot_reflectance_for(layer_id);
    let reflectance = select(DEFAULT_REFLECTANCE, own_reflectance, own_reflectance >= 0.0);
    let roughness = min(roughness_sample * roughness_scale, 1.0);

    var weight = 1.0;
    if !single {
        let sharpness = SHARPNESS_MIN + SHARPNESS_RANGE * splat.blend_sharpness;
        let contested = layer_weights + vec4<f32>(height);
        weight = dot(pow(corner_weights + contested, vec4<f32>(sharpness)), corner_weights);
    }

    var out = accum;
    out.albedo += albedo * weight;
    out.normal_top += normal_top * weight;
    out.normal_side += normal_side * weight;
    out.occlusion += occlusion * weight;
    out.roughness += roughness * weight;
    out.reflectance += reflectance * weight;
    out.total += weight;
    return out;
}

// The control word's fields.
fn base_of(control: u32) -> u32 {
    return (control >> BASE_SHIFT) & BASE_MASK;
}

fn overlay_of(control: u32) -> u32 {
    return (control >> OVERLAY_SHIFT) & OVERLAY_MASK;
}

fn blend_of(control: u32) -> f32 {
    return f32((control >> BLEND_SHIFT) & BLEND_MASK) / 255.0;
}

// The control word a cell's own slope stands for.
//
// A word, not a colour: the two slots and the blend between them are what a hand
// would have painted for this slope, so the fragment carries on into the
// layers below and picks up the height blend, the detiling and the mip
// derivatives that painted ground gets.
//
// `smoothstep` across the authored range means flat ground is purely the base
// slot and anything past the far end purely the slope slot.
fn autoterrain_control(slope: f32) -> u32 {
    let t = smoothstep(splat.autoterrain_slope_start, splat.autoterrain_slope_end, slope);
    let blend = u32(clamp(t, 0.0, 1.0) * 255.0 + 0.5);
    return ((splat.autoterrain_base_slot & BASE_MASK) << BASE_SHIFT)
        | ((splat.autoterrain_slope_slot & OVERLAY_MASK) << OVERLAY_SHIFT)
        | ((blend & BLEND_MASK) << BLEND_SHIFT);
}

#ifdef TERRAIN_RELIEF
// The mip level an isotropic filter would read a height map at, from the
// derivatives of the coordinates it is read with: the longer of the two screen
// axes, in texels.
//
// The relief march reads the height map at this one level instead of through the
// anisotropic sampler. An explicit level turns anisotropy off, and the march is
// up to 38 fetches a fragment, each of which would otherwise cost the sixteen
// taps a grazing view asks of it.
fn relief_mip(ddx: vec2<f32>, ddy: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(height_array));
    let footprint = max(length(ddx * size), length(ddy * size));
    return max(log2(max(footprint, 1.0)), 0.0);
}

// How a ray going into the ground stands against the heightfield `travel`
// metres along it: how far below the surface the ray is, less how far below the
// top of the shell the heightfield is there. At or under the surface is >= 0.
//
// `uv` is where the ray enters, in the dominant tile's own frame, and
// `uv_per_metre` is how far the sample moves per metre of travel, in the same
// frame. `steepness` is the ray's share along the surface normal, so the depth
// it has reached is `travel * steepness`. `mip` is from `relief_mip` of the
// unshifted derivatives, as in any parallax lookup: the shift is small against
// the texture's own scale and does not change its filtering.
fn relief_gap(
    layer: i32,
    uv: vec2<f32>,
    mip: f32,
    uv_per_metre: vec2<f32>,
    steepness: f32,
    depth: f32,
    travel: f32,
) -> f32 {
    let height = textureSampleLevel(
        height_array, layer_sampler, uv + uv_per_metre * travel, layer, mip).r;
    return travel * steepness - (1.0 - height) * depth;
}

// The distance along the ray, in metres, at which it meets the heightfield.
//
// Ported from `parallax_field_trace` and `parallax_field_refine` in
// crates/parallax/src/shaders/parallax_field.wgsl of the Hayjump monorepo, for
// a surface with no sides to leave through. The step budget runs from
// `RELIEF_MIN_STEPS` head on to `RELIEF_MAX_STEPS` at a grazing angle, and
// `lod` thins it back toward the head-on count.
fn relief_travel(
    layer: i32,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
    uv_per_metre: vec2<f32>,
    steepness: f32,
    depth: f32,
    lod: f32,
) -> f32 {
    let mip = relief_mip(ddx, ddy);
    // Grazing rays cross many texels per unit of depth, so they get the larger
    // budget.
    let full = RELIEF_MAX_STEPS + (RELIEF_MIN_STEPS - RELIEF_MAX_STEPS) * steepness;
    let count = max(RELIEF_MIN_STEPS + (full - RELIEF_MIN_STEPS) * clamp(lod, 0.0, 1.0), 1.0);
    // One step is a fraction of the ray that crosses the whole shell.
    let reach = depth / max(steepness, RELIEF_MIN_STEEPNESS);
    let dt = reach / count;
    let steps = u32(ceil(count));

    // A ray that enters at or under the surface has nothing to march through.
    if relief_gap(layer, uv, mip, uv_per_metre, steepness, depth, 0.0) >= 0.0 {
        return 0.0;
    }
    var previous = 0.0;
    var travel = 0.0;
    for (var i = 0u; i < steps; i++) {
        travel += dt;
        if relief_gap(layer, uv, mip, uv_per_metre, steepness, depth, travel) >= 0.0 {
            // Binary search between the last sample above and this one below.
            var above = previous;
            var below = travel;
            for (var j = 0u; j < RELIEF_REFINE_STEPS; j++) {
                let middle = (above + below) * 0.5;
                if relief_gap(layer, uv, mip, uv_per_metre, steepness, depth, middle) >= 0.0 {
                    below = middle;
                } else {
                    above = middle;
                }
            }
            return (above + below) * 0.5;
        }
        previous = travel;
    }
    // Out of budget without crossing anything: stay at the last sample rather
    // than punch a hole in the surface.
    return previous;
}
#endif

#ifdef TERRAIN_NORMAL_MAP
// A normal read back shorter than this points nowhere: texels leaning opposite
// ways, filtered to the middle of a fold. The mix of the map's normal with the
// mesh's, below, is held to the same floor.
const NORMAL_MAP_MIN_LENGTH_SQ: f32 = 1e-4;

// The terrain's own normal under a fragment, in world space, read from the baked
// normal map. `geometric` is the mesh's normal, which stands in where the map
// says nothing.
//
// A texel's alpha is how far the map is to be believed there. 255 is a texel
// that was baked; 0 is ground that was sculpted after the bake, which the map no
// longer describes, so it is lit from the mesh. In between the two normals are
// mixed and normalized again, which is what the mips and the bilinear filter
// leave at the edge of such a region: it fades rather than cuts.
//
// The map is laid out like the colour map, over its own finer grid: texel
// `scale * i` sits on grid point `i`. UV0 puts grid point `i` at `i / (res - 1)`
// while a sampled texel `j` is centred at `(j + 0.5) / side`, so the coordinate
// is remapped onto texel centres. `render::colour_map_uv` spells the same
// mapping in Rust. The map is read with the colour map's sampler.
//
// A texel stores the terrain-space unit normal as (x, z, y), in linear bytes: a
// tangent-space normal of the flat terrain, U along X and V along Z, so the
// texel is turned into terrain axes by a swizzle and into world axes by the
// transform that moves normals, whatever the terrain's own rotation and scale.
// The mips were renormalized when the map was uploaded, so a filtered texel is
// close to unit length and the normalize below only removes what 8 bits and the
// filter leave.
//
// Called from uniform control flow, for the derivatives the sample takes.
fn baked_normal(
    uv: vec2<f32>,
    normal_to_world: mat3x3<f32>,
    geometric: vec3<f32>,
) -> vec3<f32> {
    let side = f32(max(splat.normal_map_side, 2u));
    let texel_uv = (uv * (side - 1.0) + 0.5) / side;
    let texel = textureSample(normal_map, colour_map_sampler, texel_uv);
    let stored = texel.rgb * 2.0 - 1.0;
    let local = vec3<f32>(stored.x, stored.z, stored.y);
    if dot(local, local) < NORMAL_MAP_MIN_LENGTH_SQ {
        return geometric;
    }
    // Both normals are in world space here, where the mix is taken. Opposite
    // normals at half weight cancel; the mesh's stands in for them.
    let mixed = mix(geometric, normalize(normal_to_world * local), texel.a);
    if dot(mixed, mixed) < NORMAL_MAP_MIN_LENGTH_SQ {
        return geometric;
    }
    return normalize(mixed);
}
#endif

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    // UV0 is 0..1 across the whole terrain, shared bit for bit by chunks
    // that meet on a grid line, which is what keeps the control lookup
    // seamless across a chunk join.
    let local_xz = (in.uv - vec2<f32>(0.5)) * splat.terrain_size;
    frame.top = local_xz;
    frame.top_ddx = dpdx(local_xz);
    frame.top_ddy = dpdy(local_xz);

    // The fragment in the terrain's own space, for the side planes. The chunks
    // are children of the terrain with no transform of their own, so this is
    // the same space for every chunk of one terrain.
    let local_from_world = get_local_from_world(in.instance_index);
    let local = (local_from_world * vec4<f32>(in.world_position.xyz, 1.0)).xyz;
    frame.local = local;
    frame.local_ddx = dpdx(local);
    frame.local_ddy = dpdy(local);

    // The distance bands. An orthographic view has no eye to measure from, and
    // a shadow cascade is one: it gets neither band, so it draws the same ground
    // wherever the camera is.
    let is_ortho = view.clip_from_view[3].w == 1.0;
    let eye_distance = distance(in.world_position.xyz, view.world_position);
    frame.near = 0.0;
    frame.far = 0.0;
    if !is_ortho {
        if splat.near_distance > 0.0 {
            frame.near = 1.0 - smoothstep(
                splat.near_distance, splat.near_distance * 4.0 / 3.0, eye_distance);
        }
        if splat.far_distance > 0.0 {
            frame.far = smoothstep(
                splat.far_distance, splat.far_distance * 4.0 / 3.0, eye_distance);
        }
    }

    // The tint is sampled, not loaded: it is a colour, so two of them interpolate
    // into a colour. Taken here, with the derivatives, because the per-layer
    // branches below are not uniform control flow.
    //
    // Grid point i sits at i/(res-1) in UV0, while texel i is sampled at its
    // centre, (i+0.5)/res. Without the remap the layer would be half a cell out of
    // register with the control map, in opposite directions at the two edges.
    // `render::tint_uv` spells the same mapping in Rust.
    let tint_res = f32(max(splat.control_resolution, 2u));
    let tint_uv = (in.uv * (tint_res - 1.0) + 0.5) / tint_res;
    let tint = textureSample(tint_map, tint_sampler, tint_uv).rgb;

#ifdef TERRAIN_COLOUR_MAP
    // The colour map is laid out the same way over its own, finer grid: texel
    // `scale * i` sits on grid point `i`. `render::colour_map_uv` spells the
    // same mapping in Rust. The branch is on a uniform, so the sample stays in
    // uniform control flow.
    frame.colour = vec3<f32>(1.0);
    if splat.colour_map_scale > 0u {
        let colour_res = f32(max(splat.colour_map_side, 2u));
        let colour_uv = (in.uv * (colour_res - 1.0) + 0.5) / colour_res;
        frame.colour = textureSample(colour_map, colour_map_sampler, colour_uv).rgb;
    }
#endif

    // The surface the mesh draws. Shadows are biased against it whatever the
    // normal map says (`pbr_input.world_normal` below).
    let world_normal = pbr_functions::prepare_world_normal(in.world_normal, false, is_front);
    let geometric = normalize(world_normal);
    // A normal goes to world space with the inverse transpose of the transform
    // that moves positions: `local_from_world` transposed.
    let normal_to_world = transpose(mat3x3<f32>(
        local_from_world[0].xyz, local_from_world[1].xyz, local_from_world[2].xyz));

    // The surface's normal, and the frame the top plane's normals are read in.
    // A terrain baked with a normal map takes it from there, per pixel, finer
    // than the mesh it is drawn on: the fine creases the mesh averages away, and
    // the same normal on both sides of the edge between two detail levels. The
    // choice is on a uniform, so the sample stays in uniform control flow. Every
    // use of `n` below, the planes' shares, the cliffs' whiteout, the far band
    // and the relief included, is of this normal.
#ifdef TERRAIN_NORMAL_MAP
    let n = select(
        geometric,
        baked_normal(in.uv, normal_to_world, geometric),
        splat.normal_map_scale > 0u,
    );
#else
    let n = geometric;
#endif
    // The mesher emits no tangents: UV0, and the texture coordinates the top plane
    // takes from it, run along the terrain's own X and Z, which a turned terrain
    // carries off the world's. So the tangent is the terrain's +X, in world
    // space, projected onto the surface normal. That projection collapses where
    // the surface stands vertical and faces along that axis, so the terrain's +Z
    // stands in there. Picking the axis the normal leans on least would swap the
    // frame by 90 degrees wherever its lean along each crosses, seaming flat
    // ground. For a terrain that is not turned these are the world's X and Z.
    let world_from_local = get_world_from_local(in.instance_index);
    let local_x = normalize(world_from_local[0].xyz);
    var reference = local_x;
    if abs(dot(n, local_x)) > 0.999 {
        reference = normalize(world_from_local[2].xyz);
    }
    let t = normalize(reference - n * dot(reference, n));
    let b = cross(t, n);
    let tbn = mat3x3<f32>(t, b, n);

    // How the three projections share this fragment. The ground's own up is the
    // terrain's Y, so the normal is taken into terrain space first: with
    // `world_from_local` transposed, where `normal_to_world` above is
    // `local_from_world` transposed.
    let to_local = transpose(mat3x3<f32>(
        world_from_local[0].xyz, world_from_local[1].xyz, world_from_local[2].xyz));
    let local_normal = normalize(to_local * n);
    frame.local_normal = local_normal;
    let steep = smoothstep(STEEP_START, STEEP_END, 1.0 - local_normal.y);
    frame.planes = vec3<f32>(1.0, 0.0, 0.0);
    if steep > 0.0 {
        // The sharper the split, the more the dominant side axis takes. Each
        // side's lean is taken against the larger of the two first, so the
        // dominant one is 1 whatever the power does to a small number.
        let split = mix(SPLIT_SHARP, SPLIT_SOFT, frame.near);
        let lean = abs(local_normal.xz);
        let along = pow(lean / max(max(lean.x, lean.y), 1e-6), vec2<f32>(split));
        var planes = vec3<f32>(1.0 - steep, steep * along / max(along.x + along.y, 1e-6));
        // A plane with next to no share is not worth its reads.
        planes = select(planes, vec3<f32>(0.0), planes < vec3<f32>(PLANE_CUTOFF));
        // The shares sum to 1, so one is always left; a surface facing straight
        // down leans nowhere, and has only the top plane to fall back on.
        let total = planes.x + planes.y + planes.z;
        if total > 0.0 {
            frame.planes = planes / total;
        }
    }

    let g = in.uv * f32(max(splat.control_resolution, 2u) - 1u);
    let corner = control_corner(g);
    let f = clamp(g - vec2<f32>(corner), vec2<f32>(0.0), vec2<f32>(1.0));

    var c00 = load_control(corner);
    var c10 = load_control(corner + vec2<i32>(1, 0));
    var c01 = load_control(corner + vec2<i32>(0, 1));
    var c11 = load_control(corner + vec2<i32>(1, 1));

    // A terrain with autoterrain off never reaches this, and a cell a
    // hand claimed keeps the word it was painted with. Everything
    // downstream reads these four words and cannot tell which of them
    // came from the control map.
    if splat.autoterrain_enabled != 0u {
        let unclaimed = autoterrain_control(slope_at(corner, f));
        c00 = select(unclaimed, c00, (c00 & MANUAL_BIT) != 0u);
        c10 = select(unclaimed, c10, (c10 & MANUAL_BIT) != 0u);
        c01 = select(unclaimed, c01, (c01 & MANUAL_BIT) != 0u);
        c11 = select(unclaimed, c11, (c11 & MANUAL_BIT) != 0u);
    }

    let inv = vec2<f32>(1.0) - f;
    let w = vec4<f32>(inv.x * inv.y, f.x * inv.y, inv.x * f.y, f.x * f.y);

    // The texture ids this fragment reads, each with the share of the four
    // corners that name it and the weight it has in each of those corners. At
    // most eight: two layers in each of four words. Kept in arrays local to this
    // function and never handed to another one, which would copy them on every
    // call.
    var entry_ids: array<u32, 8>;
    var entry_corners: array<vec4<f32>, 8>;
    var entry_layers: array<vec4<f32>, 8>;
    var entries = 0u;

    // A word names its base layer unless it is a full overlay, and its overlay
    // layer when it is a full overlay or a partial one over a different id. The
    // guards must between them cover every word, or a fragment reads nothing and
    // normalizes zero by zero. Blending a texture with itself is that texture,
    // so the same id twice at a partial blend reads once; at full blend the base
    // has already been skipped, which is the case that would otherwise fall
    // through both.
    if c00 == c10 && c00 == c01 && c00 == c11 {
        // Every unpainted fragment, and the interior of every uniformly painted
        // area: one control word, so there is nothing to merge. The corner
        // shares are all four of them.
        let blend = blend_of(c00);
        if blend < 1.0 {
            entry_ids[entries] = base_of(c00);
            entry_corners[entries] = w;
            entry_layers[entries] = vec4<f32>(1.0 - blend);
            entries += 1u;
        }
        if blend >= 1.0 || (blend > 0.0 && overlay_of(c00) != base_of(c00)) {
            entry_ids[entries] = overlay_of(c00);
            entry_corners[entries] = w;
            entry_layers[entries] = vec4<f32>(blend);
            entries += 1u;
        }
    } else {
        // Words that differ: merge the corners that name the same id, so an id
        // is read once however many corners it is in. A corner that does not
        // name it keeps a corner share of zero, which is what makes the sum in
        // `accumulate` the sum the separate per-corner calls made.
        var words = array<u32, 4>(c00, c10, c01, c11);
        for (var k = 0u; k < 4u; k++) {
            let word = words[k];
            let blend = blend_of(word);
            let corner_mask = vec4<bool>(k == 0u, k == 1u, k == 2u, k == 3u);
            for (var layer = 0u; layer < 2u; layer++) {
                var id = base_of(word);
                var layer_weight = 1.0 - blend;
                var named = blend < 1.0;
                if layer == 1u {
                    id = overlay_of(word);
                    layer_weight = blend;
                    named = blend >= 1.0 || (blend > 0.0 && id != base_of(word));
                }
                if !named {
                    continue;
                }
                var slot = entries;
                for (var e = 0u; e < entries; e++) {
                    if entry_ids[e] == id {
                        slot = e;
                        break;
                    }
                }
                if slot == entries {
                    entry_ids[slot] = id;
                    entry_corners[slot] = vec4<f32>(0.0);
                    entry_layers[slot] = vec4<f32>(0.0);
                    entries += 1u;
                }
                // Written through a mask rather than by component: a dynamic
                // index into a vector store is not available everywhere.
                entry_corners[slot] = select(entry_corners[slot], vec4<f32>(w[k]), corner_mask);
                entry_layers[slot] = select(entry_layers[slot], vec4<f32>(layer_weight), corner_mask);
            }
        }
    }

#ifdef TERRAIN_DOMINANT
    // The id with the largest share of the fragment, before the height blend
    // has a say. It is the one the relief and the grain are made from.
    var dominant = 0u;
    var dominant_share = -1.0;
    for (var e = 0u; e < entries; e++) {
        let share = dot(entry_corners[e], entry_layers[e]);
        if share > dominant_share {
            dominant_share = share;
            dominant = e;
        }
    }
    let dominant_id = min(entry_ids[dominant], last_layer());
    let dominant_layer = i32(dominant_id);
    let dominant_scale = uv_scale_for(dominant_id);
#endif

    // How many of those ids take any of the fragment at all. A corner share of
    // nothing on every corner, as on a grid line, is a set of reads that change
    // no pixel, so such an id is skipped below; and when one id is all that is
    // left, its height cannot change the result and is not read.
    var live = 0u;
    for (var e = 0u; e < entries; e++) {
        let corners = entry_corners[e];
        if corners.x + corners.y + corners.z + corners.w > 0.0 {
            live += 1u;
        }
    }
    let single = live == 1u;

#ifdef TERRAIN_RELIEF
    // Parallax relief: march a ray from the eye into a shell of `relief_depth`
    // under the top plane, against the dominant texture's height map, and shift
    // the top plane's position, for every id, to where it met it. Colour and
    // normals are read there afterwards; nothing here writes depth, so the
    // silhouette of the ground is the mesh's.
    //
    // Only where it can be seen and is worth it: near the camera, on a texture
    // that has a height map of its own, on ground the top plane mostly covers.
    let has_height = ((splat.height_slots >> dominant_id) & 1u) != 0u;
    if frame.near > 0.0 && splat.relief_depth > 0.0 && has_height && frame.planes.x > 0.5 {
        // The view in the top plane's frame. The ray goes the other way, into
        // the ground.
        let to_eye = normalize(view.world_position - in.world_position.xyz);
        let eye_in_frame = vec3<f32>(dot(to_eye, t), dot(to_eye, b), dot(to_eye, n));
        if eye_in_frame.z > 0.01 {
            let ray = -normalize(eye_in_frame);
            let steepness = abs(ray.z);
            // The shell fades out with the band, so relief does not pop away at
            // its far edge.
            let depth = splat.relief_depth * frame.near;
            // Metres of sideways travel in the terrain's X and Z, per metre of
            // travel along the ray. The plane is planar in the terrain's X and
            // Z, so what moves the sample is the ray's X and Z in the terrain's
            // space, not its length along the slope, and not the world's axes
            // where the terrain is turned.
            let along_ray = mat3x3<f32>(
                local_from_world[0].xyz, local_from_world[1].xyz, local_from_world[2].xyz)
                * (ray.x * t + ray.y * b);
            let xz_per_metre = along_ray.xz;
            // The dominant texture's own tile frames at this fragment. With
            // detiling off there is one tap and this is the plain tiling. With it
            // on, each tile the texture is blended from is marched in its own
            // frame and the distances blended by the same shares, so the shift is
            // as continuous as the blend is; the shares are what a hard choice of
            // one tile would jump across.
            let tiled = detile(
                frame.top * dominant_scale,
                frame.top_ddx * dominant_scale,
                frame.top_ddy * dominant_scale,
                detile_for(dominant_id),
            );
            var lod = frame.near;
            if tiled.count > 1u {
                lod *= RELIEF_TILED_BUDGET;
            }
            var travel = 0.0;
            var shares = 0.0;
            for (var i = 0u; i < tiled.count; i++) {
                let tap = tiled.taps[i];
                if tap.weight <= 1e-4 {
                    continue;
                }
                let uv_per_metre = turned_by(xz_per_metre * dominant_scale, tap.turn);
                travel += tap.weight * relief_travel(
                    dominant_layer, tap.uv, tap.ddx, tap.ddy, uv_per_metre, steepness, depth, lod);
                shares += tap.weight;
            }
            frame.top += xz_per_metre * (travel / max(shares, 1e-8));
        }
    }
#endif

    var accum: Accum;
    accum.albedo = vec3<f32>(0.0);
    accum.normal_top = vec3<f32>(0.0);
    accum.normal_side = vec3<f32>(0.0);
    accum.occlusion = 0.0;
    accum.roughness = 0.0;
    accum.reflectance = 0.0;
    accum.total = 0.0;
    for (var e = 0u; e < entries; e++) {
        let corners = entry_corners[e];
        if corners.x + corners.y + corners.z + corners.w <= 0.0 {
            continue;
        }
        accum = accumulate(accum, entry_ids[e], corners, entry_layers[e], single);
    }

    let inv_total = 1.0 / max(accum.total, MIN_TOTAL_WEIGHT);
    var albedo = accum.albedo * inv_total;

#ifdef TERRAIN_GRAIN
    // Grain: one more read of the dominant texture, at four times its tiling on
    // the plane that has most of the fragment, laid over the albedo as the
    // ratio of its light to the texture's average. It brings back the fine
    // detail a colour map is too coarse to carry, so it is only for ground a
    // colour map colours, and only as far as the texture has given its colour
    // up to the map: a texture that keeps all of its colour already carries its
    // own grain, and a second copy at four times the tiling would square the
    // contrast.
    let dominant_stats = splat.slot_stats[dominant_id];
    let grain = splat.grain * frame.near * (1.0 - dominant_stats.x);
    if splat.colour_map_scale > 0u && grain > 0.0 {
        var uv = frame.top * dominant_scale;
        var ddx = frame.top_ddx * dominant_scale;
        var ddy = frame.top_ddy * dominant_scale;
        if frame.planes.y > frame.planes.x && frame.planes.y >= frame.planes.z {
            uv = vec2<f32>(frame.local.z, -frame.local.y) * dominant_scale;
            ddx = vec2<f32>(frame.local_ddx.z, -frame.local_ddx.y) * dominant_scale;
            ddy = vec2<f32>(frame.local_ddy.z, -frame.local_ddy.y) * dominant_scale;
        } else if frame.planes.z > frame.planes.x {
            uv = vec2<f32>(frame.local.x, -frame.local.y) * dominant_scale;
            ddx = vec2<f32>(frame.local_ddx.x, -frame.local_ddx.y) * dominant_scale;
            ddy = vec2<f32>(frame.local_ddy.x, -frame.local_ddy.y) * dominant_scale;
        }
        let fine = textureSampleGrad(
            albedo_array,
            layer_sampler,
            uv * GRAIN_TILING,
            dominant_layer,
            ddx * GRAIN_TILING,
            ddy * GRAIN_TILING,
        ).rgb;
        let ratio = clamp(
            luminance(fine) / max(dominant_stats.y, GRAIN_MIN_LUMINANCE), 0.0, GRAIN_RATIO_MAX);
        albedo *= mix(1.0, ratio, grain);
    }
#endif

    albedo *= mix(vec3<f32>(1.0), tint, splat.tint_strength);
    let occlusion = accum.occlusion * inv_total;
    let roughness = accum.roughness * inv_total;
    let reflectance = accum.reflectance * inv_total;

    // The top plane's tangent-space normals take the frame above; the side
    // planes' are in the terrain's axes, and turn into world axes here, once,
    // with the length they had. Out in the far band no normal map was read and
    // the surface's own normal stands; in the fade ring the two are mixed.
    var mapped_normal = n;
    if frame.far < 1.0 {
        if frame.planes.x >= 1.0 {
            // The top plane alone, which is all of flat ground: the sum, the
            // frame and the normalisation in the order they have always been
            // taken, so flat ground comes out bit for bit as it did.
            mapped_normal = normalize(tbn * normalize(accum.normal_top * inv_total));
        } else {
            let side = normal_to_world * accum.normal_side;
            let side_length = length(accum.normal_side);
            let side_world = select(
                vec3<f32>(0.0),
                normalize(side) * side_length,
                dot(side, side) > 0.0,
            );
            mapped_normal = normalize((tbn * accum.normal_top + side_world) * inv_total);
        }
        if frame.far > 0.0 {
            mapped_normal = normalize(mix(mapped_normal, n, frame.far));
        }
    }

    var pbr_input = pbr_types::pbr_input_new();
    // Mesh flags carry the shadow-receiver and transmitted-shadow-receiver
    // bits. Leaving them zero renders terrain that no shadow ever falls on.
    pbr_input.flags = mesh[in.instance_index].flags;
    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = roughness;
    pbr_input.material.reflectance = vec3<f32>(reflectance);
    pbr_input.material.metallic = 0.0;
    pbr_input.diffuse_occlusion = vec3<f32>(occlusion);
    // `standard_material_new` leaves fog off; `StandardMaterial`'s Rust
    // default has it on, and terrain that ignores the scene's fog while
    // everything beside it obeys reads as a hole in the distance.
    pbr_input.material.flags |= pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    // The shadows are biased along the surface the mesh draws, so a crease in a
    // normal map cannot pull the bias off the real surface; the lighting reads
    // the shading normal, `N`.
    pbr_input.world_normal = geometric;
    pbr_input.N = mapped_normal;
    pbr_input.is_orthographic = is_ortho;
    pbr_input.V = pbr_functions::calculate_view(in.world_position, pbr_input.is_orthographic);

    var out: FragmentOutput;
    out.color = pbr_functions::apply_pbr_lighting(pbr_input);
    out.color = pbr_functions::main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
