//! The splat material: terrain shaded from its control map against a texture
//! set, with height-blended transitions between the two ids a control word
//! names. Behind the `render` feature, so the data model below it stays
//! buildable with no GPU crate in the graph.
//!
//! A terrain's material names become the three texture arrays the shader binds
//! in two steps: [`resolve_with`] maps each slot onto a [`TextureSetEntry`], and
//! [`splat_images`] stacks the entries' images into arrays. Finding the material
//! behind a name is the host's job, done through a closure passed to
//! [`resolve_with`].
//!
//! The blend is height-weighted rather than a cross-fade: a layer's weight is
//! `pow(corner_weight + layer_weight + height, sharpness)`, normalized across
//! every contributing layer. See `shaders/terrain_splat.wgsl`.
//!
//! Beyond that blend the material has a near band and a far band, set by
//! [`SurfaceSettings`]. Close to the camera the ground gains parallax relief on
//! the textures that have a height map, and a fine grain over a colour map; far
//! from it the normal, occlusion and roughness maps give way to each slot's
//! averages ([`SlotStats`]). Steep ground is projected from the sides as well as
//! from above, so a cliff is not a texture smeared down a slope. And a terrain
//! may carry a colour map baked outside the editor ([`ColourMap`]), which gives
//! the colour while the slot textures give their light and dark.

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, TextureDataOrder, TextureDimension, TextureFormat,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::shader::ShaderRef;
use path_slash::PathExt as _;

pub mod detail;
pub mod scatter;
#[cfg(test)]
mod shader_mirror;

pub use detail::{
    ATTRIBUTE_HEIGHT_FRACTION, BuiltDetailMesh, DETAIL_TILE_BUDGET, DETAIL_TILE_CELLS,
    DetailAssets, DetailBindings, DetailDirty, DetailInstanceBuffer, DetailKey, DetailLayerSource,
    DetailLooks, DetailMeshes, DetailPipeline, DetailPressers, DetailRenderPlugin, DetailSettings,
    DetailSource, DetailSystems, DetailTile, DetailViewer, MAX_DETAIL_PRESSERS,
    TerrainDetailSource, card_mesh, detail_instance_layout, white_image, wind_noise_image,
};
pub use scatter::{
    GROUND_COVER_CULL_DISTANCE, GROUND_COVER_HEIGHT, ScatterAssetPlugin, ScatterAssets,
    ScatterChunk, ScatterDirty, ScatterPrefab, ScatterPrefabs, ScatterPrimitive, ScatterRegion,
    ScatterRenderPlugin, ScatterRendered, ScatterSystems, TerrainScatter, palette_entry_bounds,
};

use crate::heightmap::Heightmap;
use crate::sidecar::{AutoTerrainSettings, ColourMap, SurfaceSettings, TerrainMaterialSlot};
use crate::splat::ControlTexels;
use crate::texture_set::{
    DEFAULT_PERCEPTUAL_ROUGHNESS, DEFAULT_TEXTURE_COLOUR, MAX_TEXTURES, TextureSet,
    TextureSetEntry, TextureSetError, check_layer_sizes,
};

const SHADER_PATH: &str = "embedded://jackdaw_terrain/render/shaders/terrain_splat.wgsl";

pub use crate::sidecar::DEFAULT_BLEND_SHARPNESS;

/// Tangent-space normal of a flat surface, as an RGBA8 texel.
const FLAT_NORMAL_TEXEL: [u8; 4] = [128, 128, 255, 255];
/// Height of a layer with no height map: the middle of the range.
///
/// The height blend compares `layer_weight + height` between the two ids a
/// control word names. The top of the range would let a material with no depth
/// map beat every material that has one at any paint weight, and the bottom
/// would lose them all; the middle leaves the paint deciding.
const FLAT_HEIGHT_TEXEL: [u8; 4] = [128, 128, 128, 255];
/// What fills an occlusion or roughness layer for a slot that has no such
/// map.
///
/// The shader's per-slot flag keeps it from reading these layers at all, so
/// the value only has to be something the array can hold; white is the
/// identity for occlusion either way.
const UNMAPPED_LAYER_TEXEL: [u8; 4] = [255, 255, 255, 255];
/// Albedo of a slot whose material has no base colour texture, or none
/// this project can resolve. Neutral grey rather than a missing layer, so
/// the slot keeps its texture id and the ids painted after it do not
/// shift.
const FALLBACK_ALBEDO_TEXEL: [u8; 4] = [128, 128, 128, 255];
/// Anisotropic filtering the layer arrays and the colour map are sampled with:
/// the most wgpu allows. Ground is mostly seen at a grazing angle, where
/// trilinear filtering smears a texture into mush.
const ANISOTROPY: u16 = 16;
/// The least a slot's mean albedo luminance is taken to be. The shader divides
/// a texel's luminance by it, and a near-black texture would otherwise divide
/// by next to nothing.
const MIN_MEAN_LUMINANCE: f32 = 0.01;

/// The loaded image handles for a [`TextureSet`], parallel to its entries.
///
/// Held per entry rather than pre-stacked because stacking needs the decoded
/// images. Every slot is `Option`: a material may be missing, or may not bind
/// that texture.
#[derive(Clone, Debug, Default)]
pub struct TextureSetImages {
    /// Albedo handle per entry, in id order; `None` where the material has
    /// no base colour texture or could not be resolved.
    pub albedo: Vec<Option<Handle<Image>>>,
    /// Normal handle per entry, `None` where the entry has none.
    pub normal: Vec<Option<Handle<Image>>>,
    /// Height handle per entry, `None` where the entry has none.
    pub height: Vec<Option<Handle<Image>>>,
    /// Occlusion handle per entry, `None` where the entry has none.
    pub occlusion: Vec<Option<Handle<Image>>>,
    /// Roughness handle per entry, `None` where the entry has none.
    pub roughness: Vec<Option<Handle<Image>>>,
}

impl TextureSetImages {
    /// Every image handle, in no particular order. What a change watcher
    /// compares an `AssetEvent` against.
    pub fn handles(&self) -> impl Iterator<Item = &Handle<Image>> {
        self.albedo
            .iter()
            .flatten()
            .chain(self.normal.iter().flatten())
            .chain(self.height.iter().flatten())
            .chain(self.occlusion.iter().flatten())
            .chain(self.roughness.iter().flatten())
    }
}

/// A terrain's material slots, resolved into what [`splat_images`] reads.
#[derive(Clone, Debug, Default)]
pub struct ResolvedSlots {
    /// One entry per slot, in id order.
    pub set: TextureSet,
    /// Image handles parallel to [`Self::set`].
    pub images: TextureSetImages,
    /// Names of slots whose material the host could not find, in slot
    /// order. Those slots still hold their ids and draw the fallback
    /// layer; this is what a panel or a log names them by.
    pub missing: Vec<String>,
}

/// Turn a terrain's material names into texture entries, with `lookup`
/// answering what material a name addresses.
///
/// Index-preserving: entry `i` is slot `i` is texture id `i`, whether the slot
/// resolved, went missing, or was vacated. The control map is addressed by these
/// indices, so compacting the list would renumber every id painted above the
/// dropped one.
///
/// The slot mapping is fixed: albedo is `base_color_texture`, normal is
/// `normal_map_texture`, height is `depth_map`, occlusion is
/// `occlusion_texture` and roughness is `metallic_roughness_texture`. A
/// material flagged `flip_normal_map_y` carries that flag through to the
/// array builder.
///
/// A slot naming an occlusion or roughness map of its own takes that image
/// instead of the material's, so a terrain can shade a shared material's
/// ground without editing what every other surface draws.
///
/// Tiling and detiling come from the slot, never from the material: one material
/// is shared across surfaces and tiles differently on each.
pub fn resolve_with<'m>(
    slots: &[TerrainMaterialSlot],
    lookup: impl Fn(&str) -> Option<&'m StandardMaterial>,
    assets: &AssetServer,
) -> ResolvedSlots {
    let path_of = |handle: &Option<Handle<Image>>| {
        handle
            .as_ref()
            .and_then(|handle| assets.get_path(handle.id()))
            .map(|path| path.path().to_slash_lossy().into_owned())
    };

    // A map the slot names of its own, loaded here so it reaches the array
    // builder the way a material's own texture does.
    let slot_map = |named: &str| -> Option<Handle<Image>> {
        (!named.is_empty()).then(|| assets.load(named.to_string()))
    };

    let mut resolved = ResolvedSlots::default();
    for slot in slots {
        // A vacated id draws the fallback and is not reported as missing.
        if slot.is_tombstone() {
            resolved.push(TextureSetEntry::vacant(), SlotHandles::default());
            continue;
        }
        let Some(material) = lookup(&slot.material) else {
            resolved.missing.push(slot.material.clone());
            resolved.push(
                TextureSetEntry {
                    detile: slot.detile,
                    perceptual_roughness: slot.perceptual_roughness,
                    reflectance: slot.reflectance,
                    ..TextureSetEntry::unresolved(&slot.material, slot.uv_scale)
                },
                SlotHandles::default(),
            );
            continue;
        };

        let occlusion = slot_map(&slot.occlusion).or_else(|| material.occlusion_texture.clone());
        let roughness =
            slot_map(&slot.roughness).or_else(|| material.metallic_roughness_texture.clone());
        resolved.push(
            TextureSetEntry {
                material: slot.material.clone(),
                albedo: path_of(&material.base_color_texture),
                normal: path_of(&material.normal_map_texture),
                flip_normal_y: material.flip_normal_map_y,
                height: path_of(&material.depth_map),
                occlusion: path_of(&occlusion),
                roughness: path_of(&roughness),
                uv_scale: slot.uv_scale,
                detile: slot.detile,
                perceptual_roughness: slot.perceptual_roughness,
                reflectance: slot.reflectance,
            },
            SlotHandles {
                albedo: material.base_color_texture.clone(),
                normal: material.normal_map_texture.clone(),
                height: material.depth_map.clone(),
                occlusion,
                roughness,
            },
        );
    }
    resolved
}

/// One slot's image handles, so the parallel lists in [`TextureSetImages`]
/// can only be appended to together.
#[derive(Default)]
struct SlotHandles {
    albedo: Option<Handle<Image>>,
    normal: Option<Handle<Image>>,
    height: Option<Handle<Image>>,
    occlusion: Option<Handle<Image>>,
    roughness: Option<Handle<Image>>,
}

impl ResolvedSlots {
    /// Append one slot's entry and its handles together, so the entries and
    /// the handle lists cannot fall out of step.
    fn push(&mut self, entry: TextureSetEntry, handles: SlotHandles) {
        self.set.entries.push(entry);
        self.images.albedo.push(handles.albedo);
        self.images.normal.push(handles.normal);
        self.images.height.push(handles.height);
        self.images.occlusion.push(handles.occlusion);
        self.images.roughness.push(handles.roughness);
    }
}

/// The per-slot texture arrays a splat material binds.
#[derive(Clone, Debug)]
pub struct SplatImages {
    pub albedo: Image,
    pub normal: Image,
    pub height: Image,
    pub occlusion: Image,
    pub roughness: Image,
    /// What was measured of each layer on the way in, one per texture id.
    pub stats: Vec<SlotStats>,
}

/// What the shader knows of a slot's textures without reading them: the
/// averages of its layers, taken from the smallest mip of each as the arrays
/// are stacked.
///
/// The far band draws from these in place of the maps they average, and the
/// colour map takes the albedo's light and dark relative to the first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotStats {
    /// Mean luminance of the albedo, Rec. 709 weights on linear values and never
    /// under 0.01.
    pub mean_luminance: f32,
    /// Mean of the roughness map's green channel. 1 for a slot with no
    /// roughness map, which is the sample the shader takes of such a slot: its
    /// own roughness, or the terrain's, scales it.
    pub mean_roughness: f32,
    /// Mean of the occlusion map's red channel. 1 without one.
    pub mean_occlusion: f32,
}

impl SlotStats {
    /// What an id past the end of the set carries. The shader clamps every id
    /// into the set, so nothing reads it; it only has to be finite.
    pub const NEUTRAL: Self = Self {
        mean_luminance: 1.0,
        mean_roughness: 1.0,
        mean_occlusion: 1.0,
    };
}

/// Why a texture set could not be stacked into arrays yet, or at all.
#[derive(Clone, Debug, PartialEq)]
pub enum SplatBuildError {
    /// At least one of the set's images has not finished loading. Not a
    /// failure: the caller tries again next frame.
    NotReady,
    /// The set cannot be used, and retrying will not help.
    Invalid(TextureSetError),
    /// An image decoded, but into a format neither bevy's conversion nor
    /// the narrowing below can turn into the array's.
    Unconvertible {
        from: TextureFormat,
        to: TextureFormat,
    },
}

impl core::fmt::Display for SplatBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotReady => write!(f, "texture set images are still loading"),
            Self::Invalid(reason) => write!(f, "{reason}"),
            Self::Unconvertible { from, to } => write!(
                f,
                "a texture decoded as {from:?}, which cannot be reinterpreted as the \
                 array's {to:?}"
            ),
        }
    }
}

impl core::error::Error for SplatBuildError {}

/// Stack a resolved texture set into its per-slot arrays.
///
/// Returns [`SplatBuildError::NotReady`] until every image the set names has
/// decoded, and [`SplatBuildError::Invalid`], naming the material and path, when
/// the layers disagree on size.
///
/// An entry with no map for one of the arrays gets a filled layer rather than a
/// missing one, so a slot never loses its texture id. When no entry has a given
/// map, that array is built 1x1: sampling a one-texel layer returns the constant
/// anywhere, so the shader needs no flag to tell the two cases apart.
///
/// A normal map flagged [`TextureSetEntry::flip_normal_y`] has its green channel
/// inverted here, once, on the way into the array.
///
/// The albedo, occlusion and roughness layers are averaged on the way, from the
/// last mip of each chain, into [`SplatImages::stats`].
pub fn splat_images(
    set: &TextureSet,
    handles: &TextureSetImages,
    images: &Assets<Image>,
) -> Result<SplatImages, SplatBuildError> {
    if let Err(reason) = set.validate() {
        return Err(SplatBuildError::Invalid(reason));
    }
    // The handle lists are parallel to the entries, and every array below
    // declares `entry_count.max(1)` layers. No handles at all would declare one
    // layer over an empty buffer, which is a texture wgpu cannot make.
    if handles.albedo.is_empty() {
        return Err(SplatBuildError::Invalid(TextureSetError::Empty));
    }

    let albedo_layers = decode_layers(&handles.albedo, set, images, |entry| {
        entry.albedo.as_deref().unwrap_or_default()
    })?;
    // The layer size is whatever the resolved albedos agree on. A set
    // whose materials all went missing has none to measure, and builds
    // one texel per layer.
    let (width, height) = match albedo_layers.sizes.first() {
        Some(_) => check_layer_sizes(&albedo_layers.sizes).map_err(SplatBuildError::Invalid)?,
        None => (1, 1),
    };

    let (albedo, albedo_means) = stack(
        &albedo_layers,
        handles.albedo.len(),
        (width, height),
        TextureFormat::Rgba8UnormSrgb,
        FALLBACK_ALBEDO_TEXEL,
        &[],
        MipFilter::Srgb,
    )?;

    let normal_flips: Vec<bool> = set.entries.iter().map(|e| e.flip_normal_y).collect();
    let (normal, _) = stack_optional(
        &handles.normal,
        set,
        images,
        (width, height),
        FLAT_NORMAL_TEXEL,
        &normal_flips,
        MipFilter::Normal,
        |entry| entry.normal.as_deref().unwrap_or_default(),
    )?;
    let (height_array, _) = stack_optional(
        &handles.height,
        set,
        images,
        (width, height),
        FLAT_HEIGHT_TEXEL,
        &[],
        MipFilter::Linear,
        |entry| entry.height.as_deref().unwrap_or_default(),
    )?;
    let (occlusion, occlusion_means) = stack_optional(
        &handles.occlusion,
        set,
        images,
        (width, height),
        UNMAPPED_LAYER_TEXEL,
        &[],
        MipFilter::Linear,
        |entry| entry.occlusion.as_deref().unwrap_or_default(),
    )?;
    let (roughness, roughness_means) = stack_optional(
        &handles.roughness,
        set,
        images,
        (width, height),
        UNMAPPED_LAYER_TEXEL,
        &[],
        MipFilter::Linear,
        |entry| entry.roughness.as_deref().unwrap_or_default(),
    )?;

    // A layer with no map is the fill, whose mean is the fill: white, so 1 for
    // the occlusion and for the roughness sample the shader takes of such a
    // slot.
    let stats = albedo_means
        .iter()
        .zip(&roughness_means)
        .zip(&occlusion_means)
        .map(|((albedo, roughness), occlusion)| SlotStats {
            mean_luminance: linear_luminance(*albedo).max(MIN_MEAN_LUMINANCE),
            mean_roughness: f32::from(roughness[1]) / 255.0,
            mean_occlusion: f32::from(occlusion[0]) / 255.0,
        })
        .collect();

    Ok(SplatImages {
        albedo,
        normal,
        height: height_array,
        occlusion,
        roughness,
        stats,
    })
}

/// Decoded images for one per-entry handle list, with the sizes of the
/// ones that resolved.
struct DecodedLayers<'a> {
    decoded: Vec<Option<&'a Image>>,
    sizes: Vec<(usize, &'a str, &'a str, (u32, u32))>,
}

fn decode_layers<'a>(
    handles: &[Option<Handle<Image>>],
    set: &'a TextureSet,
    images: &'a Assets<Image>,
    path_of: impl Fn(&TextureSetEntry) -> &str,
) -> Result<DecodedLayers<'a>, SplatBuildError> {
    let mut decoded = Vec::with_capacity(handles.len());
    let mut sizes = Vec::new();
    for (index, handle) in handles.iter().enumerate() {
        let Some(handle) = handle else {
            decoded.push(None);
            continue;
        };
        let image = images.get(handle).ok_or(SplatBuildError::NotReady)?;
        let size = image.size();
        if let Some(entry) = set.entries.get(index) {
            sizes.push((
                index,
                entry.material.as_str(),
                path_of(entry),
                (size.x, size.y),
            ));
        }
        decoded.push(Some(image));
    }
    Ok(DecodedLayers { decoded, sizes })
}

/// Build one array from a per-entry list of optional handles. A layer with no
/// handle is filled with `fill`; if no entry has one at all, the whole array
/// collapses to 1x1.
///
/// Returns each layer's smallest mip beside the array.
fn stack_optional(
    handles: &[Option<Handle<Image>>],
    set: &TextureSet,
    images: &Assets<Image>,
    size: (u32, u32),
    fill: [u8; 4],
    flip_green: &[bool],
    filter: MipFilter,
    path_of: impl Fn(&TextureSetEntry) -> &str,
) -> Result<(Image, Vec<[u8; 4]>), SplatBuildError> {
    if handles.iter().all(Option::is_none) {
        let layers = handles.len().max(1);
        return Ok((fill_array(1, 1, layers as u32, fill), vec![fill; layers]));
    }
    let layers = decode_layers(handles, set, images, path_of)?;
    stack(
        &layers,
        handles.len(),
        size,
        TextureFormat::Rgba8Unorm,
        fill,
        flip_green,
        filter,
    )
}

/// Lay decoded layers out as one array image, filling the entries that
/// resolved to nothing and refusing any that disagrees on size.
///
/// Returns each layer's smallest mip beside the array: one texel, which is the
/// layer's average as far as the mip filter can tell.
fn stack(
    layers: &DecodedLayers<'_>,
    entry_count: usize,
    (width, height): (u32, u32),
    format: TextureFormat,
    fill: [u8; 4],
    flip_green: &[bool],
    filter: MipFilter,
) -> Result<(Image, Vec<[u8; 4]>), SplatBuildError> {
    // Measured against the albedo size, not against each other: every
    // array in a set shares one layer size.
    for (entry, material, path, size) in layers.sizes.iter().copied() {
        if size != (width, height) {
            return Err(SplatBuildError::Invalid(TextureSetError::MismatchedSize {
                entry,
                material: material.to_string(),
                path: path.to_string(),
                found: size,
                expected: (width, height),
            }));
        }
    }

    let texel_count = (width as usize) * (height as usize);
    let mut data = Vec::with_capacity(chain_texels(width, height) * 4 * entry_count.max(1));
    let mut smallest = Vec::with_capacity(entry_count.max(1));
    for (index, image) in layers.decoded.iter().enumerate() {
        let base = match image {
            Some(image) => {
                let mut texels = converted(image, format)?;
                if flip_green.get(index).copied().unwrap_or(false) {
                    for texel in texels.chunks_exact_mut(4) {
                        texel[1] = 255 - texel[1];
                    }
                }
                texels
            }
            None => fill.iter().copied().cycle().take(texel_count * 4).collect(),
        };
        smallest.push(push_mip_chain(&mut data, base, width, height, filter));
    }
    let array = array_image(width, height, entry_count.max(1) as u32, format, data);
    Ok((array, smallest))
}

/// An image's texels in `format`, converting if it decoded as something
/// else. The arrays are 8 bit per channel, and a texture pack may hand the
/// loader a 16-bit file for any slot; converting here keeps such a pack
/// usable without a pre-converted copy.
fn converted(image: &Image, format: TextureFormat) -> Result<Vec<u8>, SplatBuildError> {
    if image.texture_descriptor.format == format {
        return image.data.clone().ok_or(SplatBuildError::NotReady);
    }
    // Image data is stripped for the render world, so a decoded image
    // with none is unavailable; one that will not convert is an error,
    // and reporting it as still loading would retry forever.
    let Some(data) = image.data.as_deref() else {
        return Err(SplatBuildError::NotReady);
    };
    let from = image.texture_descriptor.format;
    if let Some(texels) = image.convert(format).and_then(|converted| converted.data) {
        return Ok(texels);
    }
    narrowed_to_rgba8(data, from, format).ok_or(SplatBuildError::Unconvertible { from, to: format })
}

/// Narrow 16-bit-per-channel texels to RGBA8, keeping the high byte of each
/// channel.
///
/// `Image::convert` routes through `DynamicImage`, which has no path out of the
/// formats bevy decodes deep files into. A single channel replicates across RGB,
/// so a 16-bit height map stacks as grey rather than red.
fn narrowed_to_rgba8(data: &[u8], from: TextureFormat, to: TextureFormat) -> Option<Vec<u8>> {
    if !matches!(
        to,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return None;
    }
    match from {
        TextureFormat::Rgba16Unorm | TextureFormat::Rgba16Uint => Some(
            data.chunks_exact(8)
                .flat_map(|t| [t[1], t[3], t[5], t[7]])
                .collect(),
        ),
        TextureFormat::Rg16Unorm | TextureFormat::Rg16Uint => Some(
            data.chunks_exact(4)
                .flat_map(|t| [t[1], t[1], t[1], t[3]])
                .collect(),
        ),
        TextureFormat::R16Unorm | TextureFormat::R16Uint => Some(
            data.chunks_exact(2)
                .flat_map(|t| [t[1], t[1], t[1], 255])
                .collect(),
        ),
        _ => None,
    }
}

/// How a mip level averages the four texels it stands for. A texture array is
/// filtered as stored bytes, so what averaging is correct depends on what the
/// bytes mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MipFilter {
    /// Plain average of the stored bytes.
    Linear,
    /// Average in linear light, stored back non-linearly.
    Srgb,
    /// Average as tangent-space directions, renormalized.
    Normal,
}

/// Mip levels a `width` by `height` texture carries, down to 1x1:
/// `floor(log2(max(width, height))) + 1`, which is what wgpu's
/// `mip_level_size` computes.
fn mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

/// Texels one layer's whole mip chain holds, base level included.
fn chain_texels(width: u32, height: u32) -> usize {
    (0..mip_level_count(width, height))
        .map(|level| {
            let w = (width >> level).max(1) as usize;
            let h = (height >> level).max(1) as usize;
            w * h
        })
        .sum()
}

/// sRGB byte to linear light, tabulated: a full chain over a 16-layer
/// 2K set decodes hundreds of millions of texels, and `powf` on each is
/// seconds of hitch on the rebuild.
static SRGB_TO_LINEAR: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|byte| {
        let c = byte as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
});

fn srgb_to_linear(byte: u8) -> f32 {
    SRGB_TO_LINEAR[byte as usize]
}

fn linear_to_srgb(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let encoded = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5) as u8
}

fn averaged_byte(taps: [[u8; 4]; 4], channel: usize) -> u8 {
    let sum: u32 = taps.iter().map(|t| u32::from(t[channel])).sum();
    ((sum as f32) / 4.0 + 0.5) as u8
}

fn average_texels(taps: [[u8; 4]; 4], filter: MipFilter) -> [u8; 4] {
    let mut out = [0u8; 4];
    // Alpha is a coverage weight in every one of these arrays, never a
    // colour or a direction, so it averages linearly whatever the rest do.
    out[3] = averaged_byte(taps, 3);
    match filter {
        MipFilter::Linear => {
            for (channel, byte) in out.iter_mut().enumerate().take(3) {
                *byte = averaged_byte(taps, channel);
            }
        }
        MipFilter::Srgb => {
            for (channel, byte) in out.iter_mut().enumerate().take(3) {
                let sum: f32 = taps.iter().map(|t| srgb_to_linear(t[channel])).sum();
                *byte = linear_to_srgb(sum / 4.0);
            }
        }
        MipFilter::Normal => {
            let mut vector = [0f32; 3];
            for (channel, component) in vector.iter_mut().enumerate() {
                let sum: f32 = taps
                    .iter()
                    .map(|t| f32::from(t[channel]) / 255.0 * 2.0 - 1.0)
                    .sum();
                *component = sum / 4.0;
            }
            let length =
                (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
            // Four directions that cancel leave nothing to point at; a flat
            // normal is the only answer that still shades.
            if length < 1e-4 {
                vector = [0.0, 0.0, 1.0];
            } else {
                for component in &mut vector {
                    *component /= length;
                }
            }
            for (channel, byte) in out.iter_mut().enumerate().take(3) {
                *byte = ((vector[channel] * 0.5 + 0.5) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn texel_at(src: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y as usize) * (width as usize) + (x as usize)) * 4;
    [src[at], src[at + 1], src[at + 2], src[at + 3]]
}

/// One mip level below `src`, box filtered.
///
/// An odd side halves by flooring, which leaves a row or column the
/// footprint would run past; the second tap clamps back onto the last one
/// rather than reading into the next row.
fn downsample(src: &[u8], width: u32, height: u32, filter: MipFilter) -> Vec<u8> {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut out = Vec::with_capacity((next_width as usize) * (next_height as usize) * 4);
    for y in 0..next_height {
        let y0 = (y * 2).min(height - 1);
        let y1 = (y * 2 + 1).min(height - 1);
        for x in 0..next_width {
            let x0 = (x * 2).min(width - 1);
            let x1 = (x * 2 + 1).min(width - 1);
            let taps = [
                texel_at(src, width, x0, y0),
                texel_at(src, width, x1, y0),
                texel_at(src, width, x0, y1),
                texel_at(src, width, x1, y1),
            ];
            out.extend_from_slice(&average_texels(taps, filter));
        }
    }
    out
}

/// Append one layer's base level and every mip below it to `data`.
///
/// The arrays upload as [`TextureDataOrder::LayerMajor`], which reads a
/// layer's whole chain before the next layer's base, so a caller stacking
/// layers appends each chain whole.
///
/// Returns the chain's last texel: the one-texel mip, the layer's average.
fn push_mip_chain(
    data: &mut Vec<u8>,
    base: Vec<u8>,
    width: u32,
    height: u32,
    filter: MipFilter,
) -> [u8; 4] {
    let levels = mip_level_count(width, height);
    data.extend_from_slice(&base);
    let mut level = base;
    let (mut w, mut h) = (width, height);
    for _ in 1..levels {
        level = downsample(&level, w, h, filter);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        data.extend_from_slice(&level);
    }
    [level[0], level[1], level[2], level[3]]
}

/// Rec. 709 luminance of an sRGB-encoded texel, taken in linear light.
fn linear_luminance(texel: [u8; 4]) -> f32 {
    0.2126 * srgb_to_linear(texel[0])
        + 0.7152 * srgb_to_linear(texel[1])
        + 0.0722 * srgb_to_linear(texel[2])
}

fn fill_array(width: u32, height: u32, layers: u32, fill: [u8; 4]) -> Image {
    let texels = (width as usize) * (height as usize) * (layers as usize);
    let data: Vec<u8> = fill.iter().copied().cycle().take(texels * 4).collect();
    array_image(width, height, layers, TextureFormat::Rgba8Unorm, data)
}

fn array_image(
    width: u32,
    height: u32,
    layers: u32,
    format: TextureFormat,
    data: Vec<u8>,
) -> Image {
    let mut image = mip_chained_image(
        width,
        height,
        layers,
        format,
        data,
        ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            ..linear_anisotropic()
        },
    );
    // A one-layer array still has to present as an array to the shader,
    // which wgpu will not infer from `depth_or_array_layers == 1`.
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

/// A sampler that filters linearly within and between mip levels, at
/// [`ANISOTROPY`]. wgpu refuses anisotropy on a sampler with any other filter.
fn linear_anisotropic() -> ImageSamplerDescriptor {
    ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: ANISOTROPY,
        ..default()
    }
}

/// An image whose data already holds a whole mip chain per layer, sampled by
/// `sampler`.
fn mip_chained_image(
    width: u32,
    height: u32,
    layers: u32,
    format: TextureFormat,
    data: Vec<u8>,
    sampler: ImageSamplerDescriptor,
) -> Image {
    // Built uninitialized because `Image::new` checks the buffer against
    // one mip level, and these images carry a whole chain per layer.
    let mut image = Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    // The sampler filters between mip levels, so the levels have to exist:
    // without them every fragment reads the sharpest texels, and detiled UVs
    // that decorrelate neighbouring fragments turn mid-range ground into
    // noise. `push_mip_chain` laid the data out to match.
    image.texture_descriptor.mip_level_count = mip_level_count(width, height);
    image.data_order = TextureDataOrder::LayerMajor;
    image.data = Some(data);
    image.sampler = ImageSampler::Descriptor(sampler);
    image
}

/// The control map as an uploadable image: `R32Uint`, one texel per grid
/// point, never filtered.
pub fn control_image(texels: &ControlTexels) -> Image {
    control_image_from_bytes(&texels.to_bytes(), texels.resolution)
}

/// [`control_image`] from a texel buffer the caller already holds, so an editor
/// keeping the CPU-side mirror of the uploaded texture can patch the rows a
/// brush touched instead of rebuilding the buffer every frame of a stroke.
pub fn control_image_from_bytes(bytes: &[u8], resolution: u32) -> Image {
    let side = resolution.max(1);
    let mut data = bytes.to_vec();
    data.resize((side as usize) * (side as usize) * 4, 0);
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    // `textureLoad` ignores the sampler, but wgpu still validates one
    // against the format, and `R32Uint` is unfilterable.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    image
}

/// The colour layer as an uploadable image: `Rgba8Unorm`, one texel per grid
/// point, laid out exactly like the control map.
///
/// Not an sRGB texture, because the layer is a multiplier rather than a colour
/// to be lit: a byte the picker draws at half brightness has to arrive as a half
/// multiply.
///
/// Filtered rather than loaded, unlike the control map, so a coarse
/// hand-painted wash reads as a smooth one between the strokes. No mip chain:
/// the tint is already the low-frequency half of the ground's look.
///
/// A short buffer pads with white, which is the identity.
pub fn tint_image_from_bytes(bytes: &[u8], resolution: u32) -> Image {
    let side = resolution.max(1);
    let mut data = bytes.to_vec();
    data.resize((side as usize) * (side as usize) * 4, u8::MAX);
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Nearest,
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..default()
    });
    image
}

/// UV0 remapped onto the tint layer's texel centres.
///
/// The splat shader computes this inline; both spell the same mapping. UV0
/// places control grid point `i` at `i/(res-1)` while a sampled texel `i` is
/// centred at `(i+0.5)/res`, and the error reverses sign across the terrain, so
/// no constant bias corrects it.
#[must_use]
pub fn tint_uv(uv: f32, resolution: u32) -> f32 {
    let res = resolution.max(2) as f32;
    (uv * (res - 1.0) + 0.5) / res
}

/// The terrain's colour map as an uploadable image: `Rgba8UnormSrgb` with a
/// full mip chain, or one white texel for a terrain with none.
///
/// Filtered, clamped to the edge and anisotropic. The map is coarse against the
/// ground it colours, so it is magnified close up and wants its mips from far
/// off. The chain is averaged in linear light, as the albedo arrays' are.
///
/// A map that fails [`ColourMap::validate`] builds the white texel rather than
/// reading past the texels it has.
pub fn colour_map_image(map: Option<&ColourMap>) -> Image {
    let (side, base) = match map.filter(|map| map.validate().is_ok()) {
        Some(map) => (map.side, map.texels.iter().flatten().copied().collect()),
        None => (1, vec![u8::MAX; 4]),
    };
    let mut data = Vec::with_capacity(chain_texels(side, side) * 4);
    push_mip_chain(&mut data, base, side, side, MipFilter::Srgb);
    mip_chained_image(
        side,
        side,
        1,
        TextureFormat::Rgba8UnormSrgb,
        data,
        ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            ..linear_anisotropic()
        },
    )
}

/// UV0 remapped onto the colour map's texel centres.
///
/// The splat shader computes this inline, as it does [`tint_uv`], and for the
/// same reason: UV0 places grid point `i` at `i/(res-1)`, and the map's texel
/// `scale * i` lies on that point and is centred at `(scale*i + 0.5)/side`. For
/// a scale of 1 the side is the control resolution and this is [`tint_uv`] on
/// each axis.
#[must_use]
pub fn colour_map_uv(uv: Vec2, side: u32) -> Vec2 {
    let side = side.max(2) as f32;
    (uv * (side - 1.0) + Vec2::splat(0.5)) / side
}

/// A terrain's colour map as the material binds it: the uploaded image, and
/// what the shader needs to sample it.
#[derive(Clone, Debug)]
pub struct ColourMapBinding {
    /// [`colour_map_image`] of the map, once it is in `Assets<Image>`.
    pub image: Handle<Image>,
    /// Texels per grid cell, or 0 where the terrain has no colour map and the
    /// image is a single white texel.
    pub scale: u32,
    /// Texels along one edge of the image.
    pub side: u32,
}

impl ColourMapBinding {
    /// Add the image for `map`, or the white one for none, to `images`, and
    /// describe it. A map that fails [`ColourMap::validate`] is bound as none.
    pub fn upload(map: Option<&ColourMap>, images: &mut Assets<Image>) -> Self {
        let map = map.filter(|map| map.validate().is_ok());
        Self {
            image: images.add(colour_map_image(map)),
            scale: map.map_or(0, |map| u32::from(map.scale)),
            side: map.map_or(1, |map| map.side),
        }
    }
}

impl Default for ColourMapBinding {
    /// No colour map, and no image yet: what a material starts with before it
    /// is pointed at one.
    fn default() -> Self {
        Self {
            image: Handle::default(),
            scale: 0,
            side: 1,
        }
    }
}

/// [`tint_image_from_bytes`] from a dense colour layer.
///
/// A layer shorter than `resolution^2` pads with white, so an unpainted
/// terrain uploads no tint rather than failing.
pub fn tint_image(colors: &[[u8; 4]], resolution: u32) -> Image {
    let side = resolution.max(1);
    let cells = (side as usize) * (side as usize);
    let mut bytes = Vec::with_capacity(cells * 4);
    for texel in colors.iter().take(cells) {
        bytes.extend_from_slice(texel);
    }
    tint_image_from_bytes(&bytes, side)
}

/// A heightmap's slope field as an uploadable image: `R32Float`, one texel per
/// grid point, laid out exactly like the control map so a fragment reads both
/// with one grid coordinate.
///
/// The shader shades autoterrain from this rather than from the surface it is
/// drawing: the surface is a clipmap, so a slope read off the geometry changes
/// as a level hands over. These values are the same from every distance.
pub fn slope_image(map: &Heightmap) -> Image {
    let side = map.resolution.max(1);
    let slopes = map.slope_field();
    let mut data = Vec::with_capacity((side as usize) * (side as usize) * 4);
    for texel in &slopes {
        data.extend_from_slice(&texel.to_le_bytes());
    }
    data.resize((side as usize) * (side as usize) * 4, 0);
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // `textureLoad` ignores the sampler, but wgpu still validates one
    // against the format, and `R32Float` is unfilterable.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    image
}

/// Terrain shaded from a control map against a texture set.
///
/// One material per terrain: the control map and terrain size live here rather
/// than per chunk, so every chunk of a terrain shares one material and one
/// upload.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct TerrainSplatMaterial {
    /// Per-id UV scales, four to a `Vec4`. The shader unpacks by id.
    #[uniform(0)]
    pub uv_scales: [Vec4; MAX_TEXTURES / 4],
    /// Per-id detiling strengths, packed the same way. 0 leaves a layer
    /// sampled exactly where its UV scale puts it.
    #[uniform(0)]
    pub detile_strengths: [Vec4; MAX_TEXTURES / 4],
    /// Per-id perceptual roughness each slot sets of its own, packed the
    /// same way. Negative where a slot sets none: its map as it is, or
    /// [`Self::perceptual_roughness`] without one.
    #[uniform(0)]
    pub slot_roughness: [Vec4; MAX_TEXTURES / 4],
    /// Per-id reflectance each slot sets of its own, packed the same way.
    /// Negative where a slot sets none, which shades at
    /// [`crate::texture_set::DEFAULT_REFLECTANCE`].
    #[uniform(0)]
    pub slot_reflectance: [Vec4; MAX_TEXTURES / 4],
    /// Terrain XZ extent in world units.
    #[uniform(0)]
    pub terrain_size: Vec2,
    /// `0..1`, remapped by the shader onto a `4..64` power exponent. Low is
    /// a soft cross-fade, high a near-binary height cutout.
    #[uniform(0)]
    pub blend_sharpness: f32,
    #[uniform(0)]
    pub perceptual_roughness: f32,
    /// Grid points per terrain edge; the control map's side length.
    #[uniform(0)]
    pub control_resolution: u32,
    /// Texture ids the bound set defines. Ids past it clamp to the last.
    #[uniform(0)]
    pub layer_count: u32,
    /// Nonzero where this terrain textures the cells no hand has claimed
    /// from their slope. 0 leaves every cell to its own control word.
    #[uniform(0)]
    pub autoterrain_enabled: u32,
    /// Texture id flat ground draws where autoterrain is on.
    #[uniform(0)]
    pub autoterrain_base_slot: u32,
    /// Texture id steep ground draws where autoterrain is on.
    #[uniform(0)]
    pub autoterrain_slope_slot: u32,
    /// Slope at which the base texture starts giving way, in radians.
    /// Converted from the authored degrees here so the shader carries no
    /// conversion of its own.
    #[uniform(0)]
    pub autoterrain_slope_start: f32,
    /// Slope at which the slope texture has fully taken over, in radians.
    #[uniform(0)]
    pub autoterrain_slope_end: f32,
    /// How much of the tint texture reaches the finished albedo, `0..1`. 0 draws
    /// the textures untinted.
    #[uniform(0)]
    pub tint_strength: f32,
    /// Bit `i` is set where texture id `i` has an occlusion map. An id
    /// without one shades unoccluded.
    #[uniform(0)]
    pub occlusion_slots: u32,
    /// Bit `i` is set where texture id `i` has a roughness map. An id
    /// without one takes [`Self::perceptual_roughness`].
    #[uniform(0)]
    pub roughness_slots: u32,
    /// Per id: x is how much of the slot's own texture colour it keeps against
    /// the colour map, y the mean luminance of its albedo, z the mean of its
    /// roughness map (1 without one) and w the mean of its occlusion map (1
    /// without one). See [`SlotStats`]. The far band reads the means in place
    /// of the maps, and the colour map takes the albedo's light and dark
    /// relative to the first.
    #[uniform(0)]
    pub slot_stats: [Vec4; MAX_TEXTURES],
    /// Metres from the camera within which the near band applies. 0 turns it
    /// off. See [`SurfaceSettings::near_distance`].
    #[uniform(0)]
    pub near_distance: f32,
    /// Metres from the camera past which the far band applies. 0 turns it off.
    /// See [`SurfaceSettings::far_distance`].
    #[uniform(0)]
    pub far_distance: f32,
    /// Depth of the near band's parallax relief, in metres. 0 is off. See
    /// [`SurfaceSettings::relief_depth`].
    #[uniform(0)]
    pub relief_depth: f32,
    /// How much of the textures' grain the near band lays over the colour map,
    /// `0..1`. See [`SurfaceSettings::grain`].
    #[uniform(0)]
    pub grain: f32,
    /// Bit `i` is set where texture id `i` has a height map of its own, not
    /// the filled middle-grey layer a slot without one gets. The relief marches
    /// only through a real one.
    #[uniform(0)]
    pub height_slots: u32,
    /// Texels per grid cell of the colour map, or 0 where the terrain has none.
    #[uniform(0)]
    pub colour_map_scale: u32,
    /// Texels along one edge of the colour map.
    #[uniform(0)]
    pub colour_map_side: u32,
    #[texture(1, dimension = "2d_array")]
    #[sampler(2)]
    pub albedo: Handle<Image>,
    #[texture(3, dimension = "2d_array")]
    pub normal: Handle<Image>,
    #[texture(4, dimension = "2d_array")]
    pub height: Handle<Image>,
    #[texture(5, sample_type = "u_int")]
    pub control: Handle<Image>,
    /// Slope per grid point, in radians. Only autoterrain reads it.
    #[texture(6, sample_type = "float", filterable = false)]
    pub slope: Handle<Image>,
    /// The colour layer the finished albedo is multiplied by. Filtered,
    /// so it has a sampler of its own rather than sharing the layer
    /// arrays'.
    #[texture(7)]
    #[sampler(8)]
    pub tint: Handle<Image>,
    /// Ambient occlusion per slot, read from the red channel.
    #[texture(9, dimension = "2d_array")]
    pub occlusion: Handle<Image>,
    /// Roughness per slot, read from the green channel the way a
    /// metallic-roughness texture stores it.
    #[texture(10, dimension = "2d_array")]
    pub roughness: Handle<Image>,
    /// The terrain's colour map, or one white texel without one. Filtered and
    /// clamped, so it has a sampler of its own.
    #[texture(11)]
    #[sampler(12)]
    pub colour_map: Handle<Image>,
}

impl Material for TerrainSplatMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
    }
}

impl TerrainSplatMaterial {
    /// Build a material for a terrain from its set, its arrays, its
    /// control map and how it textures the cells no hand has claimed.
    ///
    /// `texture_colours` is each slot's [`TerrainMaterialSlot::texture_colour`],
    /// one per texture id; an id past its end keeps its own colour.
    /// `colour_map` is the terrain's, from [`ColourMapBinding::upload`].
    pub fn new(
        set: &TextureSet,
        texture_colours: &[f32],
        arrays: SplatArrayHandles,
        control: Handle<Image>,
        slope: Handle<Image>,
        tint: Handle<Image>,
        colour_map: ColourMapBinding,
        terrain_size: Vec2,
        control_resolution: u32,
        autoterrain: AutoTerrainSettings,
        surface: SurfaceSettings,
    ) -> Self {
        let scales = set.uv_scales();
        let mut uv_scales = [Vec4::ZERO; MAX_TEXTURES / 4];
        for (i, scale) in scales.iter().enumerate() {
            uv_scales[i / 4][i % 4] = *scale;
        }
        let mut detile_strengths = [Vec4::ZERO; MAX_TEXTURES / 4];
        for (i, strength) in set.detile_strengths().iter().enumerate() {
            detile_strengths[i / 4][i % 4] = *strength;
        }
        let packed = |values: [Option<f32>; MAX_TEXTURES]| {
            let mut out = [Vec4::splat(-1.0); MAX_TEXTURES / 4];
            for (i, value) in values.iter().enumerate() {
                if let Some(value) = value {
                    out[i / 4][i % 4] = *value;
                }
            }
            out
        };
        let mut material = Self {
            uv_scales,
            detile_strengths,
            slot_roughness: packed(set.perceptual_roughness()),
            slot_reflectance: packed(set.reflectances()),
            terrain_size,
            blend_sharpness: DEFAULT_BLEND_SHARPNESS,
            perceptual_roughness: DEFAULT_PERCEPTUAL_ROUGHNESS,
            control_resolution,
            layer_count: set.len().max(1) as u32,
            autoterrain_enabled: 0,
            autoterrain_base_slot: 0,
            autoterrain_slope_slot: 0,
            autoterrain_slope_start: 0.0,
            autoterrain_slope_end: 0.0,
            tint_strength: crate::sidecar::DEFAULT_TINT_STRENGTH,
            occlusion_slots: slot_mask(set, |entry| entry.occlusion.is_some()),
            roughness_slots: slot_mask(set, |entry| entry.roughness.is_some()),
            slot_stats: [Vec4::ZERO; MAX_TEXTURES],
            near_distance: crate::sidecar::DEFAULT_NEAR_DISTANCE,
            far_distance: crate::sidecar::DEFAULT_FAR_DISTANCE,
            relief_depth: crate::sidecar::DEFAULT_RELIEF_DEPTH,
            grain: crate::sidecar::DEFAULT_GRAIN,
            height_slots: slot_mask(set, |entry| entry.height.is_some()),
            colour_map_scale: colour_map.scale,
            colour_map_side: colour_map.side,
            albedo: arrays.albedo,
            normal: arrays.normal,
            height: arrays.height,
            control,
            slope,
            tint,
            occlusion: arrays.occlusion,
            roughness: arrays.roughness,
            colour_map: colour_map.image,
        };
        for (id, stats) in material.slot_stats.iter_mut().enumerate() {
            let measured = arrays.stats.get(id).copied().unwrap_or(SlotStats::NEUTRAL);
            *stats = Vec4::new(
                DEFAULT_TEXTURE_COLOUR,
                measured.mean_luminance,
                measured.mean_roughness,
                measured.mean_occlusion,
            );
        }
        material.set_texture_colours(texture_colours);
        material.set_autoterrain(autoterrain);
        material.set_surface(surface);
        material
    }

    /// Point each slot's texture colour at `colours`, one per texture id; an id
    /// past the end keeps its own colour. A uniform write, like
    /// [`Self::set_surface`], and sanitized on the way in for the same reason.
    pub fn set_texture_colours(&mut self, colours: &[f32]) {
        for (id, stats) in self.slot_stats.iter_mut().enumerate() {
            let wanted = colours.get(id).copied().unwrap_or(DEFAULT_TEXTURE_COLOUR);
            stats.x = crate::sidecar::sane_unit(wanted, DEFAULT_TEXTURE_COLOUR);
        }
    }

    /// Point the colour map half of the uniform and the binding at `colour_map`,
    /// from [`ColourMapBinding::upload`]. The old image is dropped with the
    /// handle, and the material is written, which is what makes the bind group
    /// pick the new one up.
    pub fn set_colour_map(&mut self, colour_map: ColourMapBinding) {
        self.colour_map_scale = colour_map.scale;
        self.colour_map_side = colour_map.side;
        self.colour_map = colour_map.image;
    }

    /// Point the autoterrain half of the uniform at `settings`.
    ///
    /// Separate from [`Self::new`] because changing these is a uniform write:
    /// the control map, the arrays and the mesh all stand.
    ///
    /// The settings are sanitized on the way in: they reach a `smoothstep` the
    /// fragment cannot guard.
    pub fn set_autoterrain(&mut self, settings: AutoTerrainSettings) {
        let settings = settings.sanitized();
        self.autoterrain_enabled = u32::from(settings.enabled);
        self.autoterrain_base_slot = settings.base_slot as u32;
        self.autoterrain_slope_slot = settings.slope_slot as u32;
        self.autoterrain_slope_start = settings.slope_start_deg.to_radians();
        self.autoterrain_slope_end = settings.slope_end_deg.to_radians();
    }

    /// Point the surface half of the uniform at `settings`.
    ///
    /// A uniform write, like [`Self::set_autoterrain`], and sanitized on the way
    /// in for the same reason: a NaN reaches the fragment's `pow` and its `mix`
    /// unguarded.
    pub fn set_surface(&mut self, settings: SurfaceSettings) {
        let settings = settings.sanitized();
        self.blend_sharpness = settings.blend_sharpness;
        self.tint_strength = settings.tint_strength;
        self.near_distance = settings.near_distance;
        self.far_distance = settings.far_distance;
        self.relief_depth = settings.relief_depth;
        self.grain = settings.grain;
    }
}

/// Which texture ids answer `has_map`, as the bit per id the shader tests.
///
/// Ids past [`MAX_TEXTURES`] cannot be addressed by a control word, so a set
/// longer than the id space contributes nothing past it.
fn slot_mask(set: &TextureSet, has_map: impl Fn(&TextureSetEntry) -> bool) -> u32 {
    let mut mask = 0u32;
    for (id, entry) in set.entries.iter().take(MAX_TEXTURES).enumerate() {
        if has_map(entry) {
            mask |= 1 << id;
        }
    }
    mask
}

/// The array images once they are in `Assets<Image>`, with what
/// [`splat_images`] measured of their layers on the way.
#[derive(Clone, Debug, Default)]
pub struct SplatArrayHandles {
    pub albedo: Handle<Image>,
    pub normal: Handle<Image>,
    pub height: Handle<Image>,
    pub occlusion: Handle<Image>,
    pub roughness: Handle<Image>,
    /// [`SplatImages::stats`]: one entry per texture id. Ids it does not reach
    /// carry [`SlotStats::NEUTRAL`].
    pub stats: Vec<SlotStats>,
}

/// Registers the texture-set asset, its loader, the splat material and its
/// shader.
pub struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/terrain_splat.wgsl");
        app.add_plugins(MaterialPlugin::<TerrainSplatMaterial>::default());
    }
}

/// Resolution: turning material names into entries, through a lookup
/// closure standing in for the host's name store.
#[cfg(test)]
mod resolve_tests {
    use std::collections::HashMap;

    use bevy::app::App;
    use bevy::asset::{AssetPlugin, AssetServer, Assets};
    use bevy::prelude::*;

    use super::{ResolvedSlots, resolve_with};
    use crate::sidecar::TerrainMaterialSlot;

    /// A host's name store, minus everything resolution does not use.
    #[derive(Resource, Default)]
    struct Saved(HashMap<String, Handle<StandardMaterial>>);

    fn resolve_app() -> App {
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AssetPlugin::default()));
        app.init_asset::<Image>();
        app.init_asset::<StandardMaterial>();
        app.init_resource::<Saved>();
        app
    }

    /// Save a material under `name` with the three slots a terrain reads.
    fn saved_material(app: &mut App, name: &str, flip_y: bool) {
        let server = app.world().resource::<AssetServer>().clone();
        let base = server.load::<Image>(format!("t/{name}_base.png"));
        let normal = server.load::<Image>(format!("t/{name}_normal.png"));
        let depth = server.load::<Image>(format!("t/{name}_height.png"));
        let handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color_texture: Some(base),
                normal_map_texture: Some(normal),
                depth_map: Some(depth),
                flip_normal_map_y: flip_y,
                ..default()
            });
        app.world_mut()
            .resource_mut::<Saved>()
            .0
            .insert(name.to_string(), handle);
    }

    /// Save a material under `name` carrying the two maps a slot shades
    /// its occlusion and roughness from.
    fn saved_shaded_material(app: &mut App, name: &str) {
        let server = app.world().resource::<AssetServer>().clone();
        let occlusion = server.load::<Image>(format!("t/{name}_ao.png"));
        let roughness = server.load::<Image>(format!("t/{name}_orm.png"));
        let handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                occlusion_texture: Some(occlusion),
                metallic_roughness_texture: Some(roughness),
                ..default()
            });
        app.world_mut()
            .resource_mut::<Saved>()
            .0
            .insert(name.to_string(), handle);
    }

    fn resolve_slots(app: &App, slots: &[TerrainMaterialSlot]) -> ResolvedSlots {
        let saved = app.world().resource::<Saved>();
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        resolve_with(
            slots,
            |name| saved.0.get(name).and_then(|handle| materials.get(handle)),
            app.world().resource::<AssetServer>(),
        )
    }

    /// Base colour is albedo, the normal map is the normal, and the depth
    /// map is the height.
    #[test]
    fn a_materials_three_slots_become_albedo_normal_and_height() {
        let mut app = resolve_app();
        saved_material(&mut app, "grass", false);

        let resolved = resolve_slots(
            &app,
            &[TerrainMaterialSlot {
                material: "grass".to_string(),
                uv_scale: 0.3,
                detile: 0.6,
                occlusion: String::new(),
                roughness: String::new(),
                perceptual_roughness: None,
                reflectance: None,
                texture_colour: 1.0,
            }],
        );

        assert!(resolved.missing.is_empty());
        let entry = &resolved.set.entries[0];
        assert_eq!(entry.material, "grass");
        assert_eq!(entry.albedo.as_deref(), Some("t/grass_base.png"));
        assert_eq!(entry.normal.as_deref(), Some("t/grass_normal.png"));
        assert_eq!(entry.height.as_deref(), Some("t/grass_height.png"));
        assert_eq!(entry.uv_scale, 0.3, "tiling comes from the slot");
        assert_eq!(entry.detile, 0.6, "so does detiling");
        assert!(!entry.flip_normal_y);
        assert!(resolved.images.albedo[0].is_some());
        assert!(resolved.images.normal[0].is_some());
        assert!(resolved.images.height[0].is_some());
        assert_eq!(resolved.set.validate(), Ok(()));
    }

    /// A material whose normal map has green pointing down carries that
    /// flag to the array builder, the only place that can flip it before
    /// the shader samples it.
    #[test]
    fn a_materials_normal_convention_flag_reaches_the_texture_entry() {
        let mut app = resolve_app();
        saved_material(&mut app, "rock", true);

        let resolved = resolve_slots(&app, &[TerrainMaterialSlot::new("rock")]);

        assert!(resolved.set.entries[0].flip_normal_y);
    }

    /// A terrain referencing a material with no file keeps the slot:
    /// dropping it would renumber every id painted after it. The name is
    /// reported so the host can name the one that went.
    #[test]
    fn a_missing_material_keeps_its_slot_and_is_named() {
        let mut app = resolve_app();
        saved_material(&mut app, "grass", false);

        let resolved = resolve_slots(
            &app,
            &[
                TerrainMaterialSlot::new("grass"),
                TerrainMaterialSlot::new("deleted"),
                TerrainMaterialSlot::new("grass2"),
            ],
        );

        assert_eq!(resolved.set.entries.len(), 3, "ids keep their positions");
        assert_eq!(resolved.set.entries[1].material, "deleted");
        assert_eq!(resolved.set.entries[1].albedo, None);
        assert_eq!(resolved.images.albedo[1], None);
        assert_eq!(resolved.missing, vec!["deleted", "grass2"]);
    }

    /// A vacated id keeps its place and draws the fallback, and is not
    /// reported as missing.
    #[test]
    fn a_vacated_slot_holds_its_id_without_being_called_missing() {
        let mut app = resolve_app();
        saved_material(&mut app, "grass", false);

        let resolved = resolve_slots(
            &app,
            &[
                TerrainMaterialSlot::new("grass"),
                TerrainMaterialSlot::tombstone(),
                TerrainMaterialSlot::new("grass"),
            ],
        );

        assert_eq!(resolved.set.entries.len(), 3);
        assert!(resolved.set.entries[1].is_vacant());
        assert_eq!(resolved.images.albedo[1], None);
        assert!(resolved.missing.is_empty());
    }

    /// A material with no textures at all is not an error: it resolves to
    /// a slot that draws the fallback, the same as a missing one, but is
    /// not reported as missing because its name still resolves.
    #[test]
    fn a_material_with_no_textures_resolves_without_being_called_missing() {
        let mut app = resolve_app();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        app.world_mut()
            .resource_mut::<Saved>()
            .0
            .insert("plain".to_string(), handle);

        let resolved = resolve_slots(&app, &[TerrainMaterialSlot::new("plain")]);

        assert!(resolved.missing.is_empty());
        assert_eq!(resolved.set.entries[0].albedo, None);
        assert_eq!(resolved.set.validate(), Ok(()));
    }

    /// The entries and the three handle lists are one row per slot, so
    /// the array builder can read them by index.
    #[test]
    fn every_slot_contributes_one_entry_and_one_handle_row() {
        let mut app = resolve_app();
        saved_material(&mut app, "grass", false);
        let slots = [
            TerrainMaterialSlot::new("grass"),
            TerrainMaterialSlot::tombstone(),
            TerrainMaterialSlot::new("gone"),
        ];

        let resolved = resolve_slots(&app, &slots);

        assert_eq!(resolved.set.entries.len(), slots.len());
        assert_eq!(resolved.images.albedo.len(), slots.len());
        assert_eq!(resolved.images.normal.len(), slots.len());
        assert_eq!(resolved.images.height.len(), slots.len());
        assert_eq!(resolved.images.occlusion.len(), slots.len());
        assert_eq!(resolved.images.roughness.len(), slots.len());
    }

    /// Occlusion is the material's occlusion map and roughness is its
    /// metallic-roughness map, so a ground texture set that ships both
    /// shades with them without anything new to author.
    #[test]
    fn a_slot_whose_material_has_an_occlusion_map_binds_it() {
        let mut app = resolve_app();
        saved_shaded_material(&mut app, "gravel");

        let resolved = resolve_slots(&app, &[TerrainMaterialSlot::new("gravel")]);

        let entry = &resolved.set.entries[0];
        assert_eq!(entry.occlusion.as_deref(), Some("t/gravel_ao.png"));
        assert_eq!(entry.roughness.as_deref(), Some("t/gravel_orm.png"));
        assert!(resolved.images.occlusion[0].is_some());
        assert!(resolved.images.roughness[0].is_some());
    }

    /// A material with neither map leaves the slot with none, so the
    /// shader shades it unoccluded at the terrain's own roughness.
    #[test]
    fn a_slot_whose_material_has_no_maps_binds_none() {
        let mut app = resolve_app();
        saved_material(&mut app, "grass", false);

        let resolved = resolve_slots(&app, &[TerrainMaterialSlot::new("grass")]);

        assert_eq!(resolved.set.entries[0].occlusion, None);
        assert_eq!(resolved.set.entries[0].roughness, None);
        assert_eq!(resolved.images.occlusion[0], None);
        assert_eq!(resolved.images.roughness[0], None);
    }

    /// A map named on the slot wins over the one its material carries: one
    /// material is shared across surfaces, and a terrain may shade its own
    /// ground without editing what the others draw.
    #[test]
    fn a_map_named_on_the_slot_wins_over_the_materials() {
        let mut app = resolve_app();
        saved_shaded_material(&mut app, "gravel");

        let resolved = resolve_slots(
            &app,
            &[TerrainMaterialSlot {
                occlusion: "ground/path_ao.png".to_string(),
                ..TerrainMaterialSlot::new("gravel")
            }],
        );

        let entry = &resolved.set.entries[0];
        assert_eq!(entry.occlusion.as_deref(), Some("ground/path_ao.png"));
        assert_eq!(
            entry.roughness.as_deref(),
            Some("t/gravel_orm.png"),
            "the slot named no roughness, so the material's still stands"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Control;
    use crate::texture_set::TextureSetEntry;

    fn solid(width: u32, height: u32, texel: [u8; 4], format: TextureFormat) -> Image {
        let data: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((width as usize) * (height as usize) * 4)
            .collect();
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    /// A solid image whose texel is however many bytes `texel` is, for the
    /// formats bevy decodes deep files into.
    fn solid_texel(width: u32, height: u32, texel: &[u8], format: TextureFormat) -> Image {
        let data: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((width as usize) * (height as usize) * texel.len())
            .collect();
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    /// A set of solid-colour entries, all loaded, one per requested size.
    fn loaded_set(sizes: &[(u32, u32)]) -> (Assets<Image>, TextureSet, TextureSetImages) {
        let mut images = Assets::<Image>::default();
        let mut albedo = Vec::new();
        let mut entries = Vec::new();
        for (i, (w, h)) in sizes.iter().enumerate() {
            albedo.push(Some(images.add(solid(
                *w,
                *h,
                [10, 20, 30, 255],
                TextureFormat::Rgba8UnormSrgb,
            ))));
            entries.push(TextureSetEntry::new(format!("m{i}"), format!("t{i}.png")));
        }
        let count = sizes.len();
        (
            images,
            TextureSet { entries },
            TextureSetImages {
                albedo,
                normal: vec![None; count],
                height: vec![None; count],
                occlusion: vec![None; count],
                roughness: vec![None; count],
            },
        )
    }

    #[test]
    fn a_uniform_set_stacks_into_one_layer_per_entry() {
        let (images, set, handles) = loaded_set(&[(4, 4), (4, 4), (4, 4)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.albedo.texture_descriptor.size.width, 4);
        assert_eq!(
            built.albedo.texture_descriptor.size.depth_or_array_layers,
            3
        );
        assert_eq!(
            built
                .albedo
                .texture_view_descriptor
                .as_ref()
                .unwrap()
                .dimension,
            Some(TextureViewDimension::D2Array)
        );
    }

    #[test]
    fn a_single_entry_set_still_presents_as_an_array() {
        let (images, set, handles) = loaded_set(&[(2, 2)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(
            built.albedo.texture_descriptor.size.depth_or_array_layers,
            1
        );
        assert_eq!(
            built
                .albedo
                .texture_view_descriptor
                .as_ref()
                .unwrap()
                .dimension,
            Some(TextureViewDimension::D2Array)
        );
    }

    /// The layer size in texels of one mip level of a `size` square.
    fn level_texels(size: u32, level: u32) -> usize {
        let side = (size >> level).max(1) as usize;
        side * side
    }

    /// One layer's mip `level` out of a stacked array, as texels.
    ///
    /// The arrays upload layer-major: a layer's whole chain, then the
    /// next layer's base. Walking it here is what proves the builder
    /// wrote the order `TextureDataOrder::LayerMajor` reads back.
    fn mip_level(image: &Image, layer: u32, level: u32) -> Vec<[u8; 4]> {
        let size = image.texture_descriptor.size.width;
        let levels = image.texture_descriptor.mip_level_count;
        let chain: usize = (0..levels).map(|l| level_texels(size, l) * 4).sum();
        let start =
            chain * (layer as usize) + (0..level).map(|l| level_texels(size, l) * 4).sum::<usize>();
        let end = start + level_texels(size, level) * 4;
        image.data.as_ref().expect("built arrays carry data")[start..end]
            .chunks_exact(4)
            .map(|t| [t[0], t[1], t[2], t[3]])
            .collect()
    }

    /// A 2x2-checkered layer of `a` and `b`, so mip 1 is one texel and
    /// has to be their average.
    fn checker(size: u32, a: [u8; 4], b: [u8; 4], format: TextureFormat) -> Image {
        let mut data = Vec::with_capacity((size as usize) * (size as usize) * 4);
        for y in 0..size {
            for x in 0..size {
                let texel = if (x + y) % 2 == 0 { a } else { b };
                data.extend_from_slice(&texel);
            }
        }
        Image::new(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    fn one_layer_set(
        albedo: Option<Image>,
        normal: Option<Image>,
    ) -> (Assets<Image>, TextureSet, TextureSetImages) {
        let mut images = Assets::<Image>::default();
        let albedo = albedo.map(|image| images.add(image));
        let normal = normal.map(|image| images.add(image));
        (
            images,
            TextureSet {
                entries: vec![TextureSetEntry::new("m0", "t0.png")],
            },
            TextureSetImages {
                albedo: vec![albedo],
                normal: vec![normal],
                height: vec![None],
                occlusion: vec![None; 1],
                roughness: vec![None; 1],
            },
        )
    }

    /// Without a chain the sampler's `mipmap_filter` has nothing to read
    /// and every fragment lands on the sharpest texels, which is what
    /// turns detiled mid-range ground into noise.
    #[test]
    fn a_built_array_carries_a_full_mip_chain() {
        let (images, set, handles) = loaded_set(&[(64, 64), (64, 64)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.albedo.texture_descriptor.mip_level_count, 7);
        let expected: usize = (0..7).map(|l| level_texels(64, l) * 4).sum::<usize>() * 2;
        assert_eq!(built.albedo.data.as_ref().unwrap().len(), expected);
    }

    /// Layer-major, exactly: the second layer's base level starts where the
    /// first layer's whole chain ends. Getting it wrong by one level's worth of
    /// bytes shifts every layer past the first.
    #[test]
    fn the_second_layer_starts_where_the_first_layers_chain_ends() {
        let red = [255u8, 0, 0, 255];
        let blue = [0u8, 0, 255, 255];
        let mut images = Assets::<Image>::default();
        let layers = [
            images.add(solid(8, 8, red, TextureFormat::Rgba8UnormSrgb)),
            images.add(solid(8, 8, blue, TextureFormat::Rgba8UnormSrgb)),
        ];
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::new("m0", "t0.png"),
                TextureSetEntry::new("m1", "t1.png"),
            ],
        };
        let handles = TextureSetImages {
            albedo: layers.iter().cloned().map(Some).collect(),
            normal: vec![None, None],
            height: vec![None, None],
            occlusion: vec![None; 2],
            roughness: vec![None; 2],
        };
        let built = splat_images(&set, &handles, &images).expect("stacks");
        let levels = built.albedo.texture_descriptor.mip_level_count;
        assert_eq!(levels, 4, "8x8 chains down to 1x1");

        let chain: usize = (0..levels).map(|l| level_texels(8, l) * 4).sum();
        assert_eq!(chain, chain_texels(8, 8) * 4);
        let data = built.albedo.data.as_ref().expect("built arrays carry data");
        assert_eq!(data.len(), chain * 2, "one whole chain per layer");
        assert_eq!(
            &data[chain..chain + 4],
            &blue,
            "layer 1 mip 0 sits at the sum of layer 0's chain"
        );

        // Every level of a solid layer is that colour: the filter averages
        // equals. A wrong stride shows up as one layer's colour appearing
        // in the other's chain.
        for level in 0..levels {
            for texel in mip_level(&built.albedo, 0, level) {
                assert_eq!(texel, red, "layer 0, level {level}");
            }
            for texel in mip_level(&built.albedo, 1, level) {
                assert_eq!(texel, blue, "layer 1, level {level}");
            }
        }
    }

    /// A non-power-of-two side floors at every level, the way wgpu's
    /// `mip_level_size` does, and stops at 1x1 rather than running past it.
    #[test]
    fn a_non_power_of_two_layer_still_gets_a_chain_that_ends_at_one_texel() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(48, 48), 6);
        assert_eq!(mip_level_count(64, 3), 7);
        let (images, set, handles) = loaded_set(&[(48, 48)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.albedo.texture_descriptor.mip_level_count, 6);
        let expected: usize = (0..6)
            .map(|l| {
                let side = (48u32 >> l).max(1) as usize;
                side * side * 4
            })
            .sum();
        assert_eq!(built.albedo.data.as_ref().unwrap().len(), expected);
    }

    /// Colour averages in linear light. A black-and-white checker whose
    /// mip read mid grey as `(0 + 255) / 2 = 128` would darken every
    /// distant slope by the whole sRGB gamma.
    #[test]
    fn an_albedo_mip_averages_a_checker_in_linear_light() {
        let (images, set, handles) = one_layer_set(
            Some(checker(
                4,
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                TextureFormat::Rgba8UnormSrgb,
            )),
            None,
        );
        let built = splat_images(&set, &handles, &images).expect("stacks");
        let mip = mip_level(&built.albedo, 0, 1);
        assert_eq!(mip.len(), 4);
        // Half linear light encodes to sRGB 188, not to the byte midpoint.
        for texel in mip {
            assert_eq!(texel[0], 188, "expected mid grey in linear light");
            assert_eq!(texel[3], 255);
        }
    }

    /// A normal is a direction: averaging two slopes that lean opposite
    /// ways shortens the vector, and a short normal read as a unit one
    /// flattens the lighting instead of smoothing it.
    #[test]
    fn a_normal_mip_stays_unit_length() {
        let (images, set, handles) = one_layer_set(
            Some(solid(
                4,
                4,
                [10, 20, 30, 255],
                TextureFormat::Rgba8UnormSrgb,
            )),
            Some(checker(
                4,
                [255, 128, 128, 255],
                [0, 128, 128, 255],
                TextureFormat::Rgba8Unorm,
            )),
        );
        let built = splat_images(&set, &handles, &images).expect("stacks");
        for level in 0..built.normal.texture_descriptor.mip_level_count {
            for texel in mip_level(&built.normal, 0, level) {
                let vector: Vec<f32> = texel[..3]
                    .iter()
                    .map(|c| f32::from(*c) / 255.0 * 2.0 - 1.0)
                    .collect();
                let length =
                    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
                assert!(
                    (length - 1.0).abs() < 0.02,
                    "mip {level} holds a normal of length {length}"
                );
            }
        }
    }

    #[test]
    fn a_mismatched_albedo_names_the_entry_and_both_sizes() {
        let (images, set, handles) = loaded_set(&[(4, 4), (8, 4)]);
        let err = splat_images(&set, &handles, &images).expect_err("mismatched sizes");
        let SplatBuildError::Invalid(reason) = err else {
            panic!("expected an invalid-set error, got {err:?}");
        };
        let message = reason.to_string();
        assert!(message.contains("t1.png"), "{message}");
        assert!(message.contains("8x4"), "{message}");
        assert!(message.contains("4x4"), "{message}");
    }

    /// `splat_images` is public, so handles that do not line up with the
    /// entries are refused rather than turned into an undersized texture.
    #[test]
    fn handles_with_no_layers_at_all_are_refused_rather_than_built() {
        let images = Assets::<Image>::default();
        let set = TextureSet {
            entries: vec![TextureSetEntry::new("grass", "a.png")],
        };
        let err = splat_images(&set, &TextureSetImages::default(), &images)
            .expect_err("handles that name no layers are refused");
        assert_eq!(err, SplatBuildError::Invalid(TextureSetError::Empty));
    }

    #[test]
    fn an_image_that_has_not_decoded_yet_reports_not_ready_rather_than_failing() {
        let (_, set, handles) = loaded_set(&[(4, 4)]);
        let empty = Assets::<Image>::default();
        assert!(matches!(
            splat_images(&set, &handles, &empty),
            Err(SplatBuildError::NotReady)
        ));
    }

    #[test]
    fn a_set_with_no_normal_maps_gets_a_one_texel_flat_array() {
        let (images, set, handles) = loaded_set(&[(64, 64), (64, 64)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.normal.texture_descriptor.size.width, 1);
        assert_eq!(
            built.normal.texture_descriptor.size.depth_or_array_layers,
            2
        );
        assert_eq!(
            built.normal.data.as_ref().unwrap()[..4],
            FLAT_NORMAL_TEXEL,
            "a set with no normal maps must shade flat, not black"
        );
        assert_eq!(
            built.height.data.as_ref().unwrap()[..4],
            FLAT_HEIGHT_TEXEL,
            "a set with no height maps must blend evenly, not lose every band"
        );
    }

    /// The fill for a layer with no height map has to sit in the middle of the
    /// range, not at either end.
    ///
    /// The shader blends two ids by `layer_weight + height`, so a fill at either
    /// end lets one material's missing file decide the whole terrain's look.
    #[test]
    fn a_layer_with_no_height_map_is_filled_mid_range() {
        assert_eq!(
            FLAT_HEIGHT_TEXEL,
            [128, 128, 128, 255],
            "a heightless layer must neither win nor lose every band"
        );
        let (images, set, handles) = loaded_set(&[(4, 4), (4, 4)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        for texel in mip_level(&built.height, 1, 0) {
            assert_eq!(texel, FLAT_HEIGHT_TEXEL);
        }
    }

    /// Optionality is per entry: one entry with a height map and one
    /// without must both end up in a full-size array, the second filled.
    #[test]
    fn an_entry_without_a_height_map_gets_a_filled_layer_beside_one_that_has_it() {
        let mut images = Assets::<Image>::default();
        let a = images.add(solid(4, 4, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let b = images.add(solid(4, 4, [4, 5, 6, 255], TextureFormat::Rgba8UnormSrgb));
        let h = images.add(solid(4, 4, [7, 7, 7, 255], TextureFormat::Rgba8Unorm));
        let set = TextureSet {
            entries: vec![
                TextureSetEntry {
                    height: Some("a_h.png".into()),
                    ..TextureSetEntry::new("grass", "a.png")
                },
                TextureSetEntry::new("rock", "b.png"),
            ],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(a), Some(b)],
            normal: vec![None, None],
            height: vec![Some(h), None],
            occlusion: vec![None; 2],
            roughness: vec![None; 2],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.height.texture_descriptor.size.width, 4);
        assert_eq!(
            built.height.texture_descriptor.size.depth_or_array_layers,
            2
        );
        assert_eq!(
            mip_level(&built.height, 0, 0)[0][0],
            7,
            "entry 0 keeps its own height map"
        );
        assert_eq!(
            mip_level(&built.height, 1, 0)[0],
            FLAT_HEIGHT_TEXEL,
            "entry 1 is filled, not left black"
        );
    }

    #[test]
    fn a_height_map_of_the_wrong_size_names_itself_not_the_albedo() {
        let mut images = Assets::<Image>::default();
        let a = images.add(solid(4, 4, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let h = images.add(solid(8, 8, [7, 7, 7, 255], TextureFormat::Rgba8Unorm));
        let set = TextureSet {
            entries: vec![TextureSetEntry {
                height: Some("tall_h.png".into()),
                ..TextureSetEntry::new("grass", "a.png")
            }],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(a)],
            normal: vec![None],
            height: vec![Some(h)],
            occlusion: vec![None; 1],
            roughness: vec![None; 1],
        };
        let err = splat_images(&set, &handles, &images).expect_err("mismatched height map");
        let SplatBuildError::Invalid(reason) = err else {
            panic!("expected an invalid-set error, got {err:?}");
        };
        assert!(reason.to_string().contains("tall_h.png"), "{reason}");
        assert!(reason.to_string().contains("grass"), "{reason}");
    }

    /// A slot whose material has no file behind it must still occupy its
    /// own layer: dropping it would shift every id painted after it.
    #[test]
    fn a_slot_with_no_albedo_gets_a_fallback_layer_and_keeps_its_id() {
        let mut images = Assets::<Image>::default();
        let a = images.add(solid(4, 4, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let c = images.add(solid(4, 4, [9, 9, 9, 255], TextureFormat::Rgba8UnormSrgb));
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::new("grass", "a.png"),
                TextureSetEntry::unresolved("gone", 0.1),
                TextureSetEntry::new("rock", "c.png"),
            ],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(a), None, Some(c)],
            normal: vec![None, None, None],
            height: vec![None, None, None],
            occlusion: vec![None; 3],
            roughness: vec![None; 3],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(
            built.albedo.texture_descriptor.size.depth_or_array_layers, 3,
            "the missing slot must keep its own layer",
        );
        assert_eq!(
            mip_level(&built.albedo, 0, 0)[0][0],
            1,
            "slot 0 still draws its own texture"
        );
        assert_eq!(
            mip_level(&built.albedo, 1, 0)[0],
            FALLBACK_ALBEDO_TEXEL,
            "the missing slot draws the fallback rather than nothing",
        );
        assert_eq!(
            mip_level(&built.albedo, 2, 0)[0][0],
            9,
            "slot 2 kept its id"
        );
    }

    /// Every material missing leaves nothing to measure, so the arrays
    /// collapse to one texel rather than failing to build at all, and the
    /// terrain still renders and stays paintable.
    #[test]
    fn a_set_whose_materials_all_went_missing_still_builds() {
        let images = Assets::<Image>::default();
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::unresolved("gone", 0.1),
                TextureSetEntry::unresolved("also_gone", 0.1),
            ],
        };
        let handles = TextureSetImages {
            albedo: vec![None, None],
            normal: vec![None, None],
            height: vec![None, None],
            occlusion: vec![None; 2],
            roughness: vec![None; 2],
        };
        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.albedo.texture_descriptor.size.width, 1);
        assert_eq!(
            built.albedo.texture_descriptor.size.depth_or_array_layers,
            2
        );
        assert_eq!(
            built.albedo.data.as_ref().unwrap()[..4],
            FALLBACK_ALBEDO_TEXEL
        );
    }

    /// A normal map authored with green pointing down is flipped once
    /// here, on the way into the array, so the shader never has to know
    /// which convention a material used.
    #[test]
    fn a_material_flagged_for_dx_normals_has_its_green_channel_flipped() {
        let mut images = Assets::<Image>::default();
        let a = images.add(solid(2, 2, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let b = images.add(solid(2, 2, [4, 5, 6, 255], TextureFormat::Rgba8UnormSrgb));
        let dx = images.add(solid(2, 2, [128, 40, 255, 255], TextureFormat::Rgba8Unorm));
        let gl = images.add(solid(2, 2, [128, 40, 255, 255], TextureFormat::Rgba8Unorm));
        let set = TextureSet {
            entries: vec![
                TextureSetEntry {
                    normal: Some("dx_n.png".into()),
                    flip_normal_y: true,
                    ..TextureSetEntry::new("dx_rock", "a.png")
                },
                TextureSetEntry {
                    normal: Some("gl_n.png".into()),
                    ..TextureSetEntry::new("gl_rock", "b.png")
                },
            ],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(a), Some(b)],
            normal: vec![Some(dx), Some(gl)],
            height: vec![None, None],
            occlusion: vec![None; 2],
            roughness: vec![None; 2],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        let flagged = mip_level(&built.normal, 0, 0)[0];
        assert_eq!(
            flagged[1],
            255 - 40,
            "the flagged layer's green is inverted"
        );
        assert_eq!(flagged[0], 128, "red is untouched");
        assert_eq!(flagged[2], 255, "blue is untouched");
        assert_eq!(
            mip_level(&built.normal, 1, 0)[0][1],
            40,
            "an unflagged layer in the same set keeps its green",
        );
    }

    /// A downloaded pack ships 16-bit maps. They must stack from the
    /// originals, narrowed here, rather than needing a converted copy.
    #[test]
    fn a_sixteen_bit_set_stacks_from_the_high_byte_of_every_channel() {
        let mut images = Assets::<Image>::default();
        let albedo = images.add(solid_texel(
            2,
            2,
            &[0x00, 0x11, 0x00, 0x22, 0x00, 0x33, 0xff, 0xff],
            TextureFormat::Rgba16Unorm,
        ));
        let normal = images.add(solid_texel(
            2,
            2,
            &[0x00, 0x80, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff],
            TextureFormat::Rgba16Unorm,
        ));
        let height = images.add(solid_texel(2, 2, &[0x00, 0x99], TextureFormat::R16Uint));
        let set = TextureSet {
            entries: vec![TextureSetEntry {
                normal: Some("n.png".into()),
                height: Some("h.png".into()),
                ..TextureSetEntry::new("rock", "a.png")
            }],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(albedo)],
            normal: vec![Some(normal)],
            height: vec![Some(height)],
            occlusion: vec![None; 1],
            roughness: vec![None; 1],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(
            built.albedo.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(built.albedo.data.as_ref().unwrap()[..4], [17, 34, 51, 255]);
        assert_eq!(
            built.normal.data.as_ref().unwrap()[..4],
            [128, 64, 255, 255]
        );
        assert_eq!(
            built.height.data.as_ref().unwrap()[..4],
            [153, 153, 153, 255],
            "a single-channel map greys across RGB rather than stacking as red",
        );
    }

    /// Narrowing happens on the way in, so the flip still lands on the
    /// green channel of a 16-bit normal map.
    #[test]
    fn a_sixteen_bit_normal_map_flagged_for_dx_still_flips_its_green() {
        let mut images = Assets::<Image>::default();
        let albedo = images.add(solid(2, 2, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let normal = images.add(solid_texel(
            2,
            2,
            &[0x00, 0x80, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff],
            TextureFormat::Rgba16Unorm,
        ));
        let set = TextureSet {
            entries: vec![TextureSetEntry {
                normal: Some("n.png".into()),
                flip_normal_y: true,
                ..TextureSetEntry::new("rock", "a.png")
            }],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(albedo)],
            normal: vec![Some(normal)],
            height: vec![None],
            occlusion: vec![None; 1],
            roughness: vec![None; 1],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.normal.data.as_ref().unwrap()[1], 255 - 64);
    }

    /// Grayscale-plus-alpha lands on `Rg16Uint`: the grey channel spreads
    /// across RGB and the second channel stays the alpha it was.
    #[test]
    fn a_sixteen_bit_grayscale_alpha_height_map_keeps_its_alpha() {
        let mut images = Assets::<Image>::default();
        let albedo = images.add(solid(2, 2, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let height = images.add(solid_texel(
            2,
            2,
            &[0x00, 0x40, 0x00, 0x80],
            TextureFormat::Rg16Uint,
        ));
        let set = TextureSet {
            entries: vec![TextureSetEntry {
                height: Some("h.png".into()),
                ..TextureSetEntry::new("rock", "a.png")
            }],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(albedo)],
            normal: vec![None],
            height: vec![Some(height)],
            occlusion: vec![None; 1],
            roughness: vec![None; 1],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");
        assert_eq!(built.height.data.as_ref().unwrap()[..4], [64, 64, 64, 128]);
    }

    /// A format with no narrowing path still fails, and the message names
    /// the two formats rather than blaming a colour space it cannot know.
    #[test]
    fn a_format_with_no_narrowing_path_names_both_formats() {
        let mut images = Assets::<Image>::default();
        let albedo = images.add(solid(2, 2, [1, 2, 3, 255], TextureFormat::Rgba8UnormSrgb));
        let normal = images.add(solid_texel(2, 2, &[0u8; 16], TextureFormat::Rgba32Float));
        let set = TextureSet {
            entries: vec![TextureSetEntry {
                normal: Some("n.exr".into()),
                ..TextureSetEntry::new("rock", "a.png")
            }],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(albedo)],
            normal: vec![Some(normal)],
            height: vec![None],
            occlusion: vec![None; 1],
            roughness: vec![None; 1],
        };

        let err = splat_images(&set, &handles, &images).expect_err("no narrowing path");
        assert_eq!(
            err,
            SplatBuildError::Unconvertible {
                from: TextureFormat::Rgba32Float,
                to: TextureFormat::Rgba8Unorm,
            }
        );
        let message = err.to_string();
        assert!(message.contains("Rgba32Float"), "{message}");
        assert!(message.contains("Rgba8Unorm"), "{message}");
        assert!(!message.contains("sRGB"), "{message}");
    }

    #[test]
    fn an_unusable_set_reports_invalid_rather_than_not_ready() {
        let (images, mut set, handles) = loaded_set(&[(4, 4)]);
        set.entries[0].uv_scale = 0.0;
        assert!(matches!(
            splat_images(&set, &handles, &images),
            Err(SplatBuildError::Invalid(TextureSetError::BadUvScale { .. }))
        ));
    }

    #[test]
    fn the_control_image_is_unfiltered_r32uint_at_the_terrain_resolution() {
        let control = vec![Control::default().with_base_id(3); 16];
        let texels = ControlTexels::from_control(&control, 4);
        let image = control_image(&texels);
        assert_eq!(image.texture_descriptor.format, TextureFormat::R32Uint);
        assert_eq!(image.texture_descriptor.size.width, 4);
        assert_eq!(image.texture_descriptor.size.height, 4);
        assert_eq!(image.data.as_ref().unwrap().len(), 4 * 4 * 4);
        assert_eq!(&image.data.as_ref().unwrap()[0..4], &3u32.to_le_bytes());
    }

    #[test]
    fn a_zero_resolution_terrain_still_produces_a_bindable_control_image() {
        let image = control_image(&ControlTexels::from_control(&[], 0));
        assert_eq!(image.texture_descriptor.size.width, 1);
        assert_eq!(image.data.as_ref().unwrap().len(), 4);
    }

    /// The control-word layout the shader unpacks by hand rather than
    /// through an accessor. A drift in the manual bit would silently
    /// shade painted cells as unclaimed ground; a drift in a shift or a
    /// mask would hand every cell the wrong texture id.
    #[test]
    fn the_shaders_control_layout_matches_the_control_words() {
        use crate::control::{
            BASE_MASK, BASE_SHIFT, BLEND_MASK, BLEND_SHIFT, MANUAL_BIT, OVERLAY_MASK, OVERLAY_SHIFT,
        };

        let pinned = |line: String| {
            assert!(
                SHADER_SOURCE.contains(&line),
                "the shader drifted from control.rs: expected `{line}`"
            );
        };
        pinned(format!("const BASE_SHIFT: u32 = {BASE_SHIFT}u;"));
        pinned(format!("const BASE_MASK: u32 = 0x{BASE_MASK:X}u;"));
        pinned(format!("const OVERLAY_SHIFT: u32 = {OVERLAY_SHIFT}u;"));
        pinned(format!("const OVERLAY_MASK: u32 = 0x{OVERLAY_MASK:X}u;"));
        pinned(format!("const BLEND_SHIFT: u32 = {BLEND_SHIFT}u;"));
        pinned(format!("const BLEND_MASK: u32 = 0x{BLEND_MASK:X}u;"));
        pinned(format!("const MANUAL_BIT: u32 = 0x{MANUAL_BIT:X}u;"));
    }

    /// A material over `set` with nothing behind its handles, for the tests that
    /// read the uniform.
    fn material_over(set: &TextureSet, texture_colours: &[f32]) -> TerrainSplatMaterial {
        TerrainSplatMaterial::new(
            set,
            texture_colours,
            SplatArrayHandles::default(),
            Handle::default(),
            Handle::default(),
            Handle::default(),
            ColourMapBinding::default(),
            Vec2::splat(100.0),
            256,
            AutoTerrainSettings::default(),
            SurfaceSettings::default(),
        )
    }

    fn test_material(autoterrain: AutoTerrainSettings) -> TerrainSplatMaterial {
        let mut material = material_over(
            &TextureSet {
                entries: vec![TextureSetEntry::new("grass", "a.png")],
            },
            &[],
        );
        material.set_autoterrain(autoterrain);
        material
    }

    /// Off is the default, and off means the uniform says nothing the
    /// shader's guard can act on.
    #[test]
    fn a_terrain_that_never_asked_for_autoterrain_uploads_it_disabled() {
        let material = test_material(AutoTerrainSettings::default());
        assert_eq!(material.autoterrain_enabled, 0);
    }

    /// The angles are authored in degrees and sampled in radians, so the
    /// conversion happens once, here.
    #[test]
    fn autoterrain_slots_and_angles_reach_the_uniform_in_radians() {
        let material = test_material(AutoTerrainSettings {
            enabled: true,
            base_slot: 1,
            slope_slot: 3,
            slope_start_deg: 30.0,
            slope_end_deg: 60.0,
        });

        assert_eq!(material.autoterrain_enabled, 1);
        assert_eq!(material.autoterrain_base_slot, 1);
        assert_eq!(material.autoterrain_slope_slot, 3);
        assert!((material.autoterrain_slope_start - core::f32::consts::FRAC_PI_6).abs() < 1e-6);
        assert!((material.autoterrain_slope_end - core::f32::consts::FRAC_PI_3).abs() < 1e-6);
    }

    /// Settings that never went through a sidecar decode still cannot
    /// reach the shader's `smoothstep` as a NaN or a backwards range.
    #[test]
    fn unsanitized_autoterrain_settings_are_pulled_back_before_they_are_uploaded() {
        let material = test_material(AutoTerrainSettings {
            enabled: true,
            base_slot: 0,
            slope_slot: 1,
            slope_start_deg: 80.0,
            slope_end_deg: f32::NAN,
        });

        assert!(material.autoterrain_slope_start.is_finite());
        assert!(material.autoterrain_slope_end.is_finite());
        assert!(
            material.autoterrain_slope_start <= material.autoterrain_slope_end,
            "the range must reach the shader the right way round",
        );
    }

    /// Changing the settings is a uniform write: nothing else about the
    /// material moves, so an editor moving a slider re-uploads one
    /// buffer rather than rebuilding the terrain.
    #[test]
    fn setting_autoterrain_again_touches_nothing_but_its_own_fields() {
        let mut material = test_material(AutoTerrainSettings::default());
        let before = material.clone();

        material.set_autoterrain(AutoTerrainSettings {
            enabled: true,
            base_slot: 2,
            slope_slot: 4,
            slope_start_deg: 10.0,
            slope_end_deg: 20.0,
        });

        assert_eq!(material.autoterrain_enabled, 1);
        assert_eq!(material.uv_scales, before.uv_scales);
        assert_eq!(material.detile_strengths, before.detile_strengths);
        assert_eq!(material.control_resolution, before.control_resolution);
        assert_eq!(material.layer_count, before.layer_count);
        assert_eq!(material.terrain_size, before.terrain_size);
        assert_eq!(material.control, before.control);
    }

    #[test]
    fn uv_scales_reach_the_uniform_in_id_order() {
        let set = TextureSet {
            entries: vec![
                TextureSetEntry {
                    uv_scale: 0.25,
                    ..TextureSetEntry::new("grass", "a.png")
                },
                TextureSetEntry {
                    uv_scale: 4.0,
                    ..TextureSetEntry::new("rock", "b.png")
                },
            ],
        };
        let material = material_over(&set, &[]);
        assert_eq!(material.uv_scales[0].x, 0.25);
        assert_eq!(material.uv_scales[0].y, 4.0);
        assert_eq!(material.layer_count, 2);
        assert_eq!(material.control_resolution, 256);
        assert_eq!(material.terrain_size, Vec2::splat(100.0));
    }

    /// A slot's own roughness and reflectance reach the uniform at its id,
    /// and every id that sets none, including those past the set, reads
    /// as unset so it shades as it always has.
    #[test]
    fn slot_roughness_and_reflectance_reach_the_uniform_unset_where_not_given() {
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::new("grass", "a.png"),
                TextureSetEntry {
                    perceptual_roughness: Some(0.4),
                    reflectance: Some(0.2),
                    ..TextureSetEntry::new("wet_rock", "b.png")
                },
            ],
        };
        let material = material_over(&set, &[]);
        assert_eq!(material.slot_roughness[0].y, 0.4);
        assert_eq!(material.slot_reflectance[0].y, 0.2);
        assert!(material.slot_roughness[0].x < 0.0);
        assert!(material.slot_reflectance[0].x < 0.0);
        assert!(material.slot_roughness[3].w < 0.0);
        assert_eq!(material.perceptual_roughness, DEFAULT_PERCEPTUAL_ROUGHNESS);
    }

    /// The shader reads a slot's maps only where the uniform's bit for
    /// that id says it has them, so an id with no map shades unoccluded at
    /// the terrain's own roughness rather than sampling a filled layer.
    #[test]
    fn only_the_slots_with_maps_are_flagged_in_the_uniform() {
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::new("grass", "a.png"),
                TextureSetEntry {
                    occlusion: Some("b_ao.png".to_string()),
                    ..TextureSetEntry::new("rock", "b.png")
                },
                TextureSetEntry {
                    roughness: Some("c_orm.png".to_string()),
                    ..TextureSetEntry::new("sand", "c.png")
                },
            ],
        };
        let material = material_over(&set, &[]);

        assert_eq!(material.occlusion_slots, 0b010);
        assert_eq!(material.roughness_slots, 0b100);
    }

    /// The shader source, for the binding checks below.
    ///
    /// It cannot be compiled here: it is naga-oil input, not WGSL. The checks
    /// are textual, and catch binding numbers drifting between the
    /// `AsBindGroup` derive and the shader that reads them.
    const SHADER_SOURCE: &str = include_str!("shaders/terrain_splat.wgsl");

    #[test]
    fn every_material_binding_is_declared_in_the_shader_at_its_derive_index() {
        for (binding, declaration) in [
            (0, "var<uniform> splat: SplatUniform"),
            (1, "var albedo_array: texture_2d_array<f32>"),
            (2, "var layer_sampler: sampler"),
            (3, "var normal_array: texture_2d_array<f32>"),
            (4, "var height_array: texture_2d_array<f32>"),
            (5, "var control_map: texture_2d<u32>"),
            (7, "var tint_map: texture_2d<f32>"),
            (9, "var occlusion_array: texture_2d_array<f32>"),
            (10, "var roughness_array: texture_2d_array<f32>"),
            (11, "var colour_map: texture_2d<f32>"),
            (12, "var colour_map_sampler: sampler"),
        ] {
            let expected =
                format!("@group(#{{MATERIAL_BIND_GROUP}}) @binding({binding}) {declaration};");
            assert!(
                SHADER_SOURCE.contains(&expected),
                "the AsBindGroup derive binds {binding} as `{declaration}`, \
                 but the shader does not declare it that way"
            );
        }
    }

    /// The uniform is one buffer built from the `#[uniform(0)]` fields in
    /// declaration order, so the WGSL struct has to list them in the same
    /// order or every field reads its neighbour's bytes.
    #[test]
    fn the_uniform_struct_matches_the_materials_field_order() {
        let start = SHADER_SOURCE
            .find("struct SplatUniform {")
            .expect("SplatUniform is declared");
        let body = &SHADER_SOURCE[start..];
        let end = body.find('}').expect("SplatUniform is closed");
        let body = &body[..end];

        let mut at = 0usize;
        for field in [
            "uv_scales: array<vec4<f32>, 4>",
            "detile_strengths: array<vec4<f32>, 4>",
            "slot_roughness: array<vec4<f32>, 4>",
            "slot_reflectance: array<vec4<f32>, 4>",
            "terrain_size: vec2<f32>",
            "blend_sharpness: f32",
            "perceptual_roughness: f32",
            "control_resolution: u32",
            "layer_count: u32",
            "autoterrain_enabled: u32",
            "autoterrain_base_slot: u32",
            "autoterrain_slope_slot: u32",
            "autoterrain_slope_start: f32",
            "autoterrain_slope_end: f32",
            "tint_strength: f32",
            "occlusion_slots: u32",
            "roughness_slots: u32",
            "slot_stats: array<vec4<f32>, 16>",
            "near_distance: f32",
            "far_distance: f32",
            "relief_depth: f32",
            "grain: f32",
            "height_slots: u32",
            "colour_map_scale: u32",
            "colour_map_side: u32",
        ] {
            let found = body[at..]
                .find(field)
                .unwrap_or_else(|| panic!("`{field}` is missing or out of order in SplatUniform"));
            at += found + field.len();
        }
    }

    /// Detiling turns the UV, so the tangent-space normal stored in the texture
    /// has to turn back with it. With one rotation per tile, the turn has to be
    /// applied per tap, before the taps are blended.
    #[test]
    fn the_shader_turns_each_sampled_normal_with_its_own_tile_rotation() {
        assert!(
            SHADER_SOURCE.contains("turn: vec2<f32>,"),
            "`TileTap` must carry the rotation it applied to the UV"
        );
        let start = SHADER_SOURCE
            .find("fn accumulate(")
            .expect("accumulate is declared");
        let body = &SHADER_SOURCE[start..];
        let packed = body
            .find("normal_array")
            .expect("accumulate samples the normals");
        let turned = body
            .find("turned_by(unpacked.xy, vec2<f32>(tap.turn.x, -tap.turn.y))")
            .expect("accumulate must turn each tap's normal back by that tap's rotation");
        assert!(
            turned > packed,
            "the rotation must be applied after the sample, not before"
        );
    }

    /// Detiling samples once per tile and blends the samples, rather than
    /// blending the tiles' coordinates and sampling once.
    ///
    /// Interpolating between four rigid transforms is not a rigid transform: the
    /// blended coordinate field stretches and shears between tile centres, and a
    /// texture read through it comes back marbled.
    #[test]
    fn detiling_samples_once_per_tile_and_blends_the_samples() {
        assert!(
            SHADER_SOURCE.contains("taps: array<TileTap, 4>"),
            "detile must hand back one frame per tile, not one blended UV"
        );
        let start = SHADER_SOURCE
            .find("fn accumulate(")
            .expect("accumulate is declared");
        let body = &SHADER_SOURCE[start..];
        let loop_at = body
            .find("for (var t = 0u; t < tiled.count; t++)")
            .expect("accumulate must walk the taps");
        for array in ["albedo_array", "height_array", "normal_array"] {
            let sampled = body
                .find(&format!(
                    "textureSampleGrad({array}, layer_sampler, tap.uv, layer, tap.ddx, tap.ddy)"
                ))
                .unwrap_or_else(|| panic!("{array} must be sampled at a tap's own rigid UV"));
            assert!(
                sampled > loop_at,
                "{array} must be sampled inside the per-tile loop"
            );
        }
        assert!(
            !body.contains("tiled.uv"),
            "no sample may read a blended coordinate"
        );
    }

    /// Detiling off is one sample, not four of the same one.
    #[test]
    fn detiling_off_leaves_a_single_tap() {
        let start = SHADER_SOURCE
            .find("fn detile(")
            .expect("detile is declared");
        let body = &SHADER_SOURCE[start..];
        let single = body
            .find("out.count = 1u;")
            .expect("detile starts at one tap");
        let guard = body
            .find("if strength <= 0.0 {")
            .expect("detile short-circuits at strength 0");
        let four = body
            .find("out.count = 4u;")
            .expect("detile opens to four taps");
        assert!(
            single < guard && guard < four,
            "strength 0 must return before the four tiles are built"
        );
    }

    /// Tint strength 0 has to leave the albedo exactly as the layers accumulated
    /// it. Two halves: the uniform carries the 0, and the fragment mixes from
    /// white, so `mix(white, tint, 0)` is white and multiplying by white is the
    /// identity.
    #[test]
    fn a_tint_strength_of_zero_leaves_the_albedo_alone() {
        let mut material = material_over(&TextureSet::default(), &[]);
        assert_eq!(material.tint_strength, 1.0, "the default is the full layer");

        material.set_surface(SurfaceSettings {
            blend_sharpness: 0.25,
            tint_strength: 0.0,
            ..SurfaceSettings::default()
        });
        assert_eq!(material.tint_strength, 0.0);
        assert_eq!(material.blend_sharpness, 0.25);

        assert!(
            SHADER_SOURCE.contains("mix(vec3<f32>(1.0), tint, splat.tint_strength)"),
            "the fragment must mix the tint up from white, so strength 0 is the identity"
        );
    }

    /// A cell painted at control grid point `i` has to shade from tint texel `i`
    /// at both ends of the terrain and in the middle. Sampling raw UV0 misses by
    /// half a cell, in opposite directions at the two ends.
    #[test]
    fn a_tint_texel_is_sampled_at_the_grid_point_it_was_painted_at() {
        const RESOLUTION: u32 = 64;
        let res = f32::from(RESOLUTION as u16);
        // Which texel a linear sampler reads the centre of at `v`.
        let texel = |v: f32| (v * res - 0.5).round() as i64;
        let mut raw_misses = 0;
        for i in 0..RESOLUTION {
            let uv = i as f32 / (res - 1.0);
            assert_eq!(
                texel(tint_uv(uv, RESOLUTION)),
                i64::from(i),
                "grid point {i} must land on its own texel"
            );
            if texel(uv) != i64::from(i) {
                raw_misses += 1;
            }
        }
        assert!(
            raw_misses > 0,
            "raw UV0 must be the wrong convention, or the remap is pointless"
        );

        assert!(
            SHADER_SOURCE.contains("(in.uv * (tint_res - 1.0) + 0.5) / tint_res"),
            "the fragment must sample the tint at texel centres, as `tint_uv` does"
        );
    }

    /// A NaN reaches the fragment's `pow` and its `mix` unguarded, so the
    /// uniform write clamps rather than trusting its caller.
    #[test]
    fn the_surface_uniform_is_sanitized_on_the_way_in() {
        let mut material = material_over(&TextureSet::default(), &[]);
        material.set_surface(SurfaceSettings {
            blend_sharpness: f32::NAN,
            tint_strength: 9.0,
            near_distance: f32::NAN,
            far_distance: 1e9,
            relief_depth: -1.0,
            grain: f32::INFINITY,
        });
        assert_eq!(material.blend_sharpness, DEFAULT_BLEND_SHARPNESS);
        assert_eq!(material.tint_strength, 1.0);
        assert_eq!(
            material.near_distance,
            crate::sidecar::DEFAULT_NEAR_DISTANCE
        );
        assert_eq!(material.far_distance, crate::sidecar::MAX_FAR_DISTANCE);
        assert_eq!(material.relief_depth, 0.0);
        assert_eq!(material.grain, crate::sidecar::DEFAULT_GRAIN);
    }

    /// `MAX_TEXTURES` sizes the UV-scale array on both sides.
    #[test]
    fn the_shaders_layer_ceiling_matches_the_rust_one() {
        assert_eq!(MAX_TEXTURES, 16);
        assert!(
            SHADER_SOURCE.contains("const MAX_LAYERS: u32 = 16u;"),
            "the shader's layer ceiling drifted from MAX_TEXTURES"
        );
        assert!(
            SHADER_SOURCE.contains("uv_scales: array<vec4<f32>, 4>"),
            "the UV-scale array must hold MAX_TEXTURES / 4 vec4s"
        );
        assert!(
            SHADER_SOURCE.contains("slot_stats: array<vec4<f32>, 16>"),
            "the slot statistics must hold one vec4 per texture id"
        );
    }

    fn sampler_of(image: &Image) -> &ImageSamplerDescriptor {
        match &image.sampler {
            ImageSampler::Descriptor(descriptor) => descriptor,
            ImageSampler::Default => panic!("the builder always names its sampler"),
        }
    }

    /// Ground is mostly seen at a grazing angle. wgpu refuses anisotropy on a
    /// sampler with any filter that is not linear, so the filters are pinned
    /// beside it.
    #[test]
    fn every_layer_array_is_sampled_anisotropically_with_linear_filters() {
        let (images, set, handles) = loaded_set(&[(4, 4), (4, 4)]);
        let built = splat_images(&set, &handles, &images).expect("stacks");
        for array in [
            &built.albedo,
            &built.normal,
            &built.height,
            &built.occlusion,
            &built.roughness,
        ] {
            let sampler = sampler_of(array);
            assert_eq!(sampler.anisotropy_clamp, 16);
            assert_eq!(sampler.mag_filter, ImageFilterMode::Linear);
            assert_eq!(sampler.min_filter, ImageFilterMode::Linear);
            assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
            assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
        }
    }

    /// Rec. 709 luminance of an sRGB texel, spelled out with the transfer
    /// function and not the builder's table.
    fn luminance_of(srgb: [u8; 3]) -> f32 {
        let linear = |byte: u8| ((f32::from(byte) / 255.0 + 0.055) / 1.055).powf(2.4);
        0.2126 * linear(srgb[0]) + 0.7152 * linear(srgb[1]) + 0.0722 * linear(srgb[2])
    }

    /// The statistics come from each layer's smallest mip: the albedo's mean
    /// luminance in linear light, and the roughness and occlusion means as the
    /// channels the shader reads.
    #[test]
    fn slot_statistics_are_the_means_of_the_layers_the_shader_would_read() {
        let mut images = Assets::<Image>::default();
        let sunlit = [200u8, 180, 90];
        let albedo = [
            images.add(solid(
                8,
                8,
                [sunlit[0], sunlit[1], sunlit[2], 255],
                TextureFormat::Rgba8UnormSrgb,
            )),
            images.add(solid(
                8,
                8,
                [30, 60, 90, 255],
                TextureFormat::Rgba8UnormSrgb,
            )),
        ];
        let roughness = images.add(solid(8, 8, [0, 51, 0, 255], TextureFormat::Rgba8Unorm));
        let occlusion = images.add(solid(8, 8, [102, 0, 0, 255], TextureFormat::Rgba8Unorm));
        let set = TextureSet {
            entries: vec![
                TextureSetEntry {
                    occlusion: Some("o.png".into()),
                    roughness: Some("r.png".into()),
                    ..TextureSetEntry::new("sunlit", "a.png")
                },
                TextureSetEntry::new("bare", "b.png"),
            ],
        };
        let handles = TextureSetImages {
            albedo: albedo.iter().cloned().map(Some).collect(),
            normal: vec![None, None],
            height: vec![None, None],
            occlusion: vec![Some(occlusion), None],
            roughness: vec![Some(roughness), None],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");

        assert_eq!(built.stats.len(), 2);
        let [sunlit_stats, bare] = [built.stats[0], built.stats[1]];
        assert!((sunlit_stats.mean_luminance - luminance_of(sunlit)).abs() < 2e-3);
        assert!((sunlit_stats.mean_roughness - 0.2).abs() < 1e-6);
        assert!((sunlit_stats.mean_occlusion - 0.4).abs() < 1e-6);
        assert!((bare.mean_luminance - luminance_of([30, 60, 90])).abs() < 2e-3);
        assert_eq!(
            bare.mean_roughness, 1.0,
            "a slot with no roughness map is sampled as 1, and scaled by its own roughness after"
        );
        assert_eq!(bare.mean_occlusion, 1.0, "no occlusion map is unoccluded");
    }

    /// The shader divides by the luminance, so a black texture must not.
    #[test]
    fn a_black_albedo_has_a_mean_luminance_floor() {
        let mut images = Assets::<Image>::default();
        let albedo = images.add(solid(4, 4, [0, 0, 0, 255], TextureFormat::Rgba8UnormSrgb));
        let set = TextureSet {
            entries: vec![TextureSetEntry::new("coal", "a.png")],
        };
        let handles = TextureSetImages {
            albedo: vec![Some(albedo)],
            normal: vec![None],
            height: vec![None],
            occlusion: vec![None],
            roughness: vec![None],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");

        assert_eq!(built.stats[0].mean_luminance, 0.01);
        assert_eq!(built.stats[0].mean_roughness, 1.0);
        assert_eq!(built.stats[0].mean_occlusion, 1.0);
    }

    /// A slot whose material has no albedo draws the grey fallback, and its mean
    /// is that grey's luminance.
    #[test]
    fn a_slot_with_no_albedo_averages_as_the_fallback_grey() {
        let images = Assets::<Image>::default();
        let set = TextureSet {
            entries: vec![TextureSetEntry::unresolved("gone", 0.1)],
        };
        let handles = TextureSetImages {
            albedo: vec![None],
            normal: vec![None],
            height: vec![None],
            occlusion: vec![None],
            roughness: vec![None],
        };

        let built = splat_images(&set, &handles, &images).expect("stacks");

        assert!((built.stats[0].mean_luminance - luminance_of([128, 128, 128])).abs() < 2e-3);
    }

    /// The stats ride to the uniform beside the slot's texture colour, ids past
    /// the set carry the neutral stats, and the dials and masks land where the
    /// shader reads them.
    #[test]
    fn the_uniform_carries_the_stats_the_dials_and_the_height_mask() {
        let set = TextureSet {
            entries: vec![
                TextureSetEntry::new("grass", "a.png"),
                TextureSetEntry {
                    height: Some("b_h.png".to_string()),
                    ..TextureSetEntry::new("rock", "b.png")
                },
            ],
        };
        let arrays = SplatArrayHandles {
            stats: vec![
                SlotStats {
                    mean_luminance: 0.25,
                    mean_roughness: 0.5,
                    mean_occlusion: 0.75,
                },
                SlotStats {
                    mean_luminance: 0.125,
                    mean_roughness: 1.0,
                    mean_occlusion: 1.0,
                },
            ],
            ..SplatArrayHandles::default()
        };
        let material = TerrainSplatMaterial::new(
            &set,
            &[0.0, 0.5],
            arrays,
            Handle::default(),
            Handle::default(),
            Handle::default(),
            ColourMapBinding {
                image: Handle::default(),
                scale: 2,
                side: 129,
            },
            Vec2::splat(100.0),
            65,
            AutoTerrainSettings::default(),
            SurfaceSettings {
                near_distance: 45.0,
                far_distance: 300.0,
                relief_depth: 0.125,
                grain: 0.5,
                ..SurfaceSettings::default()
            },
        );

        assert_eq!(material.slot_stats[0], Vec4::new(0.0, 0.25, 0.5, 0.75));
        assert_eq!(material.slot_stats[1], Vec4::new(0.5, 0.125, 1.0, 1.0));
        assert_eq!(
            material.slot_stats[2],
            Vec4::new(DEFAULT_TEXTURE_COLOUR, 1.0, 1.0, 1.0),
            "an id past the set keeps its colour and carries neutral means"
        );
        assert_eq!(material.near_distance, 45.0);
        assert_eq!(material.far_distance, 300.0);
        assert_eq!(material.relief_depth, 0.125);
        assert_eq!(material.grain, 0.5);
        assert_eq!(
            material.height_slots, 0b10,
            "only the slot with a height map of its own is marked"
        );
        assert_eq!(material.colour_map_scale, 2);
        assert_eq!(material.colour_map_side, 129);
    }

    /// With nothing said the dials are what existing terrains render at.
    #[test]
    fn a_default_material_has_the_default_bands_and_no_colour_map() {
        let material = material_over(&TextureSet::default(), &[]);
        assert_eq!(material.near_distance, 30.0);
        assert_eq!(material.far_distance, 150.0);
        assert_eq!(material.relief_depth, 0.0);
        assert_eq!(material.grain, 0.0);
        assert_eq!(material.colour_map_scale, 0);
        assert_eq!(material.height_slots, 0);
        assert!(material.slot_stats.iter().all(|stats| stats.x == 1.0));
    }

    /// A texture colour is a uniform write: it clamps, touches only its own
    /// component, and an id with none to read goes back to keeping its own colour.
    #[test]
    fn texture_colours_are_written_alone_and_clamped() {
        let set = TextureSet {
            entries: vec![TextureSetEntry::new("grass", "a.png")],
        };
        let mut material = material_over(&set, &[]);
        material.slot_stats[1] = Vec4::new(0.9, 0.2, 0.3, 0.4);
        let before = material.slot_stats;

        material.set_texture_colours(&[0.25, f32::NAN, 7.0, -1.0]);

        assert_eq!(material.slot_stats[0].x, 0.25);
        assert_eq!(material.slot_stats[1].x, 1.0, "NaN keeps the slot's colour");
        assert_eq!(material.slot_stats[2].x, 1.0);
        assert_eq!(material.slot_stats[3].x, 0.0);
        assert_eq!(material.slot_stats[4].x, 1.0, "past the list, the default");
        for id in 0..MAX_TEXTURES {
            assert_eq!(
                material.slot_stats[id].to_array()[1..],
                before[id].to_array()[1..],
                "id {id}: only the texture colour moved"
            );
        }
    }

    /// The colour map goes up with a whole mip chain, sRGB, clamped at its edge.
    #[test]
    fn the_colour_map_image_is_srgb_with_a_full_mip_chain_and_a_clamped_sampler() {
        let texels: Vec<[u8; 4]> = (0..25u8).map(|i| [i, i * 2, i * 3, 255]).collect();
        let map = ColourMap::new(3, 2, texels.clone()).expect("5 by 5");

        let image = colour_map_image(Some(&map));

        assert_eq!(
            image.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(image.texture_descriptor.size.width, 5);
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 1);
        assert_eq!(
            image.texture_descriptor.mip_level_count, 3,
            "5, 2 and 1 texels"
        );
        let data = image.data.as_ref().expect("built images carry data");
        assert_eq!(data.len(), (25 + 4 + 1) * 4);
        assert_eq!(&data[..4], &texels[0]);
        assert_eq!(&data[24 * 4..25 * 4], &texels[24]);
        assert!(
            image.texture_view_descriptor.is_none(),
            "a plain 2D texture, not an array"
        );
        let sampler = sampler_of(&image);
        assert_eq!(sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(sampler.address_mode_v, ImageAddressMode::ClampToEdge);
        assert_eq!(sampler.mag_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.min_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.anisotropy_clamp, 16);
    }

    /// The mips are averaged in linear light like the albedo arrays', so a
    /// black and white checker reads mid grey in linear light, not 128.
    #[test]
    fn the_colour_map_mips_average_in_linear_light() {
        let mut texels = Vec::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                texels.push(if (x + y) % 2 == 0 {
                    [0, 0, 0, 255]
                } else {
                    [255, 255, 255, 255]
                });
            }
        }
        let map = ColourMap::new(2, 1, texels).expect("2 by 2");

        let image = colour_map_image(Some(&map));

        let data = image.data.as_ref().expect("built images carry data");
        assert_eq!(&data[16..20], &[188, 188, 188, 255]);
    }

    /// No colour map is one white texel, and so is one the format could not
    /// have stored.
    #[test]
    fn no_colour_map_binds_a_single_white_texel() {
        let mut images = Assets::<Image>::default();
        let none = ColourMapBinding::upload(None, &mut images);
        assert_eq!((none.scale, none.side), (0, 1));
        let image = images.get(&none.image).expect("uploaded");
        assert_eq!(image.texture_descriptor.size.width, 1);
        assert_eq!(image.data.as_deref(), Some(&[255u8; 4][..]));

        let unfit = ColourMap {
            scale: 9,
            side: 2,
            texels: vec![[1; 4]; 4],
        };
        let refused = ColourMapBinding::upload(Some(&unfit), &mut images);
        assert_eq!(
            (refused.scale, refused.side),
            (0, 1),
            "a map the format cannot hold is bound as none"
        );
    }

    #[test]
    fn a_colour_map_binding_describes_the_map_it_uploaded() {
        let mut images = Assets::<Image>::default();
        let map = ColourMap::new(3, 4, vec![[9, 8, 7, 255]; 81]).expect("9 by 9");

        let binding = ColourMapBinding::upload(Some(&map), &mut images);

        assert_eq!((binding.scale, binding.side), (4, 9));
        assert_eq!(
            images
                .get(&binding.image)
                .expect("uploaded")
                .texture_descriptor
                .size
                .width,
            9
        );
        let mut material = material_over(&TextureSet::default(), &[]);
        material.set_colour_map(binding.clone());
        assert_eq!(material.colour_map_scale, 4);
        assert_eq!(material.colour_map_side, 9);
        assert_eq!(material.colour_map, binding.image);
    }

    /// At a scale of 1 the colour map is laid out like the tint, texel for grid
    /// point, so the two mappings must agree exactly. At higher scales texel
    /// `scale * i` is centred on grid point `i`.
    #[test]
    fn the_colour_map_is_sampled_at_texel_centres_as_the_tint_is() {
        for resolution in [2u32, 3, 64, 257] {
            for u in [0.0f32, 0.1, 0.5, 0.999, 1.0] {
                let uv = Vec2::new(u, 1.0 - u);
                let mapped = colour_map_uv(uv, resolution);
                assert_eq!(mapped.x, tint_uv(uv.x, resolution), "x at {resolution}");
                assert_eq!(mapped.y, tint_uv(uv.y, resolution), "y at {resolution}");
            }
        }

        let (resolution, scale) = (33u32, 4u8);
        let side = crate::sidecar::side_for(resolution, scale);
        let res = resolution as f32;
        let texel_under =
            |uv: f32| (colour_map_uv(Vec2::splat(uv), side).x * side as f32 - 0.5).round();
        for i in 0..resolution {
            let uv = i as f32 / (res - 1.0);
            assert_eq!(
                texel_under(uv),
                f32::from(scale) * i as f32,
                "grid point {i}"
            );
        }

        assert!(
            SHADER_SOURCE.contains("(in.uv * (colour_res - 1.0) + 0.5) / colour_res"),
            "the fragment must sample the colour map at texel centres, as `colour_map_uv` does"
        );
    }

    /// The body of a function in the shader, from its signature to the next
    /// function.
    fn shader_function(name: &str) -> &'static str {
        let start = SHADER_SOURCE
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("the shader declares {name}"));
        let rest = &SHADER_SOURCE[start..];
        let end = rest[1..]
            .find("\nfn ")
            .or_else(|| rest[1..].find("\n@fragment"))
            .map_or(rest.len(), |at| at + 1);
        &rest[..end]
    }

    /// Naga's Metal backend copies an array handed to a function by value on
    /// every call, which cost another renderer ~75 ms a frame. Nothing here takes
    /// one: the packed uniform arrays are read in place, and the eight texture
    /// entries a fragment merges live in `fragment` and are never passed on.
    #[test]
    fn no_shader_function_takes_an_array_by_value() {
        let mut checked = 0;
        for (at, _) in SHADER_SOURCE.match_indices("\nfn ") {
            let signature = &SHADER_SOURCE[at..];
            let open = signature
                .find('(')
                .expect("a function opens its parameters");
            let close = signature
                .find(") ")
                .expect("a function closes its parameters");
            let parameters = &signature[open..close];
            assert!(
                !parameters.contains("array<"),
                "an array parameter in{}",
                &signature[..close + 1]
            );
            checked += 1;
        }
        assert!(checked > 10, "the scan found {checked} functions");
        assert!(SHADER_SOURCE.contains("var entry_ids: array<u32, 8>;"));
    }

    /// A normal map, an occlusion map and a roughness map are read behind
    /// `detailed` and nowhere else, which is what makes the far band cost less
    /// than the mid one.
    #[test]
    fn the_far_band_reads_no_normal_occlusion_or_roughness_map() {
        let body = shader_function("accumulate");
        let detailed = body
            .find("if !detailed {")
            .expect("the tap loop skips the rest in the far band");
        for array in ["normal_array", "occlusion_array", "roughness_array"] {
            let read = body
                .find(&format!("{array}, layer_sampler"))
                .unwrap_or_else(|| panic!("{array} is read"));
            assert!(
                read > detailed,
                "{array} must be read after the far band has been skipped"
            );
        }
        for array in ["albedo_array", "height_array"] {
            let read = body
                .find(&format!("{array}, layer_sampler"))
                .unwrap_or_else(|| panic!("{array} is read"));
            assert!(
                read < detailed,
                "{array} is read in every band, before the skip"
            );
        }
    }

    /// Derivatives, the tint and the colour map are taken before any branch on a
    /// per-fragment value: the only code ahead of them that branches does so on a
    /// uniform. Naga's uniformity analysis checks the rest, below.
    #[test]
    fn derivatives_and_implicit_samples_come_before_the_first_per_fragment_branch() {
        let body = shader_function("fragment");
        let first_branch = [body.find("if frame."), body.find("for (")]
            .into_iter()
            .flatten()
            .min()
            .expect("the fragment branches");
        for call in [
            "dpdx(local_xz)",
            "dpdy(local_xz)",
            "dpdx(local)",
            "dpdy(local)",
            "textureSample(tint_map",
            "textureSample(colour_map",
        ] {
            let at = body
                .find(call)
                .unwrap_or_else(|| panic!("the fragment takes {call}"));
            assert!(
                at < first_branch,
                "{call} comes after a per-fragment branch"
            );
        }
        assert!(
            body.contains("view.clip_from_view[3].w == 1.0"),
            "an orthographic view gets neither band"
        );
    }

    /// The march is up to 38 height fetches a fragment, so it reads at one
    /// explicit level worked out once. An explicit level turns the sampler's
    /// anisotropy off, where a gradient read would pay sixteen taps at a
    /// grazing angle for each.
    #[test]
    fn the_relief_march_reads_one_explicit_level_worked_out_once() {
        let gap = shader_function("relief_gap");
        assert!(gap.contains("textureSampleLevel("), "{gap}");
        assert!(!gap.contains("textureSampleGrad("), "{gap}");
        let travel = shader_function("relief_travel");
        let mip = travel
            .find("relief_mip(")
            .expect("the march works its level out");
        let first_loop = travel.find("for (").expect("the march loops");
        assert!(
            mip < first_loop,
            "the level is worked out ahead of the loops"
        );
        assert_eq!(travel.matches("relief_mip(").count(), 1);
    }

    /// With one tile there is one march, as before. With several, each tile the
    /// texture is blended from is marched in its own frame and the distances are
    /// blended by the tiles' shares: the tile with the most of the fragment
    /// changes across a contour, and a shift taken from it alone would jump.
    #[test]
    fn the_relief_shift_blends_the_tiles_it_marches_rather_than_picking_one() {
        let body = shader_function("fragment");
        assert!(body.contains("travel += tap.weight * relief_travel("));
        assert!(body.contains("travel / max(shares, 1e-8)"));
        assert!(
            !body.contains("tiled.taps[i].weight > tap.weight"),
            "no hard choice of a tile"
        );
        assert!(SHADER_SOURCE.contains("const RELIEF_TILED_BUDGET: f32 = 0.5;"));
    }

    /// The height of the only id a fragment reads cancels in the normalisation,
    /// so it is neither read nor weighed. The weight is the same 1, exactly, and
    /// a partial layer weight raised to a high power cannot drive the total under
    /// the floor and darken the ground.
    #[test]
    fn a_lone_id_neither_reads_nor_weighs_its_height() {
        let body = shader_function("accumulate");
        let skip = body
            .find("if !single {")
            .expect("the height is behind `single`");
        let read = body
            .find("height_array, layer_sampler")
            .expect("the height is read");
        let weight = body
            .find("pow(corner_weights")
            .expect("the weight is taken");
        assert!(skip < read, "the height read is behind the skip");
        assert!(
            body[..weight]
                .rfind("if !single {")
                .is_some_and(|at| at > read),
            "so is the weight"
        );
        assert!(shader_function("fragment").contains("let single = live == 1u;"));

        // The floor sits far under what the smallest base a layer can raise
        // leaves behind at the sharpest blend.
        assert!(SHADER_SOURCE.contains("const MIN_TOTAL_WEIGHT: f32 = 1e-30;"));
        let smallest_weight = 0.5f32.powf(SHARPNESS_CEILING) * 0.01;
        assert!(smallest_weight > 1e-30, "{smallest_weight}");
    }

    /// The sharpest exponent the blend dial maps to.
    const SHARPNESS_CEILING: f32 = 64.0;

    /// A colour map is coarse, and the grain is what brings the textures' detail
    /// back over it, so it is for ground a colour map colours, in the measure the
    /// slot has given its colour up, and its ratio is bounded: a texture that
    /// keeps its colour is not grained twice, and a speck is not a highlight.
    #[test]
    fn grain_is_for_colour_mapped_ground_scaled_by_what_the_slot_gave_up_and_bounded() {
        let body = shader_function("fragment");
        assert!(body.contains("(1.0 - dominant_stats.x)"));
        assert!(body.contains("splat.colour_map_scale > 0u && grain > 0.0"));
        assert!(body.contains("max(dominant_stats.y, GRAIN_MIN_LUMINANCE)"));
        assert!(body.contains("0.0, GRAIN_RATIO_MAX"));
        assert!(SHADER_SOURCE.contains("const GRAIN_RATIO_MAX: f32 = 4.0;"));
        assert!(SHADER_SOURCE.contains("const GRAIN_MIN_LUMINANCE: f32 = 0.01;"));
    }

    /// The side planes are the terrain's own axes, so a terrain turned about Y
    /// needs their whiteout blend in its space and not in the world's. The sum is
    /// turned into world axes once, with the inverse transpose of the transform
    /// that moves positions.
    #[test]
    fn the_cliff_whiteout_is_done_in_terrain_space_and_turned_to_world_once() {
        let accumulate = shader_function("accumulate");
        assert!(accumulate.contains("frame.local_normal"));
        assert!(
            !accumulate.contains("frame.normal"),
            "the world-space normal is not what the planes blend against"
        );
        let fragment = shader_function("fragment");
        assert!(fragment.contains("normal_to_world * accum.normal_side"));
        assert!(fragment.contains("local_from_world[0].xyz"));
        // The relief's sideways shift is in the terrain's axes too.
        assert!(fragment.contains("* (ray.x * t + ray.y * b)"));
    }

    /// What of Bevy the splat shader leans on, as stubs: the vertex output, the
    /// mesh and view data, and the lighting entry points. Everything else the
    /// test composes is the production source.
    const BEVY_STUBS: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) instance_index: u32,
}
struct FragmentOutput {
    @location(0) color: vec4<f32>,
}
struct MeshStub {
    flags: u32,
}
struct ViewStub {
    clip_from_view: mat4x4<f32>,
    world_position: vec3<f32>,
}
var<private> mesh: array<MeshStub, 1>;
var<private> view: ViewStub;
fn get_local_from_world(instance_index: u32) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn get_world_from_local(instance_index: u32) -> mat4x4<f32> {
    return get_local_from_world(instance_index);
}
struct StandardMaterialStub {
    base_color: vec4<f32>,
    perceptual_roughness: f32,
    reflectance: vec3<f32>,
    metallic: f32,
    flags: u32,
}
struct PbrInput {
    flags: u32,
    material: StandardMaterialStub,
    diffuse_occlusion: vec3<f32>,
    frag_coord: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    N: vec3<f32>,
    is_orthographic: bool,
    V: vec3<f32>,
}
const STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT: u32 = 1u;
fn pbr_input_new() -> PbrInput {
    var input: PbrInput;
    return input;
}
fn prepare_world_normal(world_normal: vec3<f32>, double_sided: bool, is_front: bool) -> vec3<f32> {
    return world_normal;
}
fn calculate_view(world_position: vec4<f32>, is_orthographic: bool) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 1.0);
}
fn apply_pbr_lighting(input: PbrInput) -> vec4<f32> {
    return input.material.base_color * vec4<f32>(input.N, input.V.z)
        * input.diffuse_occlusion.x * input.material.perceptual_roughness
        * input.material.reflectance.x;
}
fn main_pass_post_lighting_processing(input: PbrInput, color: vec4<f32>) -> vec4<f32> {
    return color;
}
"#;

    /// Compose the production splat shader and have naga validate it, with no
    /// GPU: the types, the bindings, and the uniformity analysis that refuses an
    /// implicit-derivative read (`textureSample`, `dpdx`) in divergent control
    /// flow. A shader error otherwise surfaces only when the first frame builds
    /// the pipeline.
    ///
    /// Only Bevy's own pieces are stubbed, and the shader's imports are cut,
    /// the way the Hayjump monorepo validates its prepass. The shader cache
    /// validates what it composes only with debug assertions on.
    #[cfg(debug_assertions)]
    #[test]
    fn the_splat_shader_composes_and_validates_against_stubs_of_bevys_pieces() {
        use bevy::render::render_resource::{DownlevelFlags, WgpuFeatures};
        use bevy::shader::{Shader, ShaderCache, ShaderCacheError, ShaderDefVal};

        let import = SHADER_SOURCE
            .find("#import")
            .expect("the shader imports from bevy_pbr");
        let import_end = import
            + SHADER_SOURCE[import..]
                .find("\n}\n")
                .expect("the import block closes")
            + 3;
        let source = format!(
            "{BEVY_STUBS}\n{}{}",
            &SHADER_SOURCE[..import],
            &SHADER_SOURCE[import_end..]
        )
        .replace("pbr_functions::", "")
        .replace("pbr_types::", "");

        let mut assets = Assets::<Shader>::default();
        let shader = Shader::from_wgsl(source, "terrain-splat-test.wgsl");
        let handle = assets.add(shader.clone());
        let mut cache = ShaderCache::new(
            (),
            WgpuFeatures::empty(),
            DownlevelFlags::all(),
            |_, _, _| Ok(()),
        );
        cache.set_shader(handle.id(), shader);

        let definitions = [ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 2)];
        if let Err(error) = cache.get(0, handle.id(), &definitions) {
            let reason = match &error {
                ShaderCacheError::ProcessShaderError(composed) => {
                    composed.emit_to_string(&cache.composer)
                }
                other => format!("{other:?}"),
            };
            panic!("the splat shader does not compose:\n{reason}");
        }
    }
}
