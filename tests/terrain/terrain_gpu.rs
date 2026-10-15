//! The splat shader on a real GPU, headless.
//!
//! Everything else that checks the shader composes it against stubs of Bevy or
//! mirrors its arithmetic in Rust. These render it: the real Bevy imports, the
//! real bind group layout, and the Metal, Vulkan or DX12 translation of the
//! WGSL, none of which a stub can vouch for. The editor's other integration
//! tests never open a device (`util::headless_app` asks for no backend), so this
//! one is `#[ignore]`d and run on purpose:
//!
//! ```text
//! cargo test -p jackdaw --test terrain terrain_gpu -- --ignored --nocapture
//! ```
//!
//! Nothing opens a window. A camera renders to an image and a screenshot reads
//! it back; the assertions are about what is in it. Set `TERRAIN_GPU_SHOTS` to a
//! folder and every shot is also written there as a PNG for a human to look at.
//!
//! The scenes between them turn on every optional part of the shader that the
//! material's pipeline key switches (the plain ground none, the colour map, the
//! grain and the relief one or more), so each of those variants is built on the
//! device too.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::app::SubApps;
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{
    Extent3d, PollType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use jackdaw_terrain::render::{
    ColourMapBinding, SplatArrayHandles, TerrainRenderPlugin, TerrainSplatMaterial,
    TextureSetImages, control_image, slope_image, splat_images, tint_image,
};
use jackdaw_terrain::sidecar::{AutoTerrainSettings, ColourMap, SurfaceSettings};
use jackdaw_terrain::splat::ControlTexels;
use jackdaw_terrain::texture_set::{TextureSet, TextureSetEntry};
use jackdaw_terrain::{Control, Heightmap};

/// Pixels along each side of the render target.
const SIDE: u32 = 128;
/// Grid points per edge of the control map every scene paints with.
const RESOLUTION: u32 = 5;
/// Metres along each edge of the plate and the wall.
const EXTENT: f32 = 4.0;
/// Texels along each edge of the test textures, and of one square of a pattern.
const TEXTURE_SIDE: u32 = 64;
const SQUARE: u32 = 16;
/// Frames to let a scene settle before it is photographed.
const SETTLE_FRAMES: u32 = 8;

/// The middle block of the render target, clear of any edge of the subject.
const MIDDLE: ((u32, u32), (u32, u32)) = ((40, 40), (88, 88));

/// A render target and an app that owns its update loop and has no window, after
/// Bevy's `externally_driven_headless_renderer` example.
struct Gpu {
    apps: SubApps,
    target: Handle<Image>,
    camera: Entity,
}

impl Gpu {
    fn new() -> Self {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    // Every pipeline is ready for the first frame it is asked
                    // for, so a shot is never of a half-built one.
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<bevy::audio::AudioPlugin>()
                .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>(),
        )
        .add_plugins(TerrainRenderPlugin);
        // Never `run`, so finish and clean up here.
        app.finish();
        app.cleanup();
        let mut apps = std::mem::take(app.sub_apps_mut());

        let mut target = Image::new_uninit(
            Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        target.texture_descriptor.usage |=
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
        let target = apps
            .main
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(target);
        let camera = apps
            .main
            .world_mut()
            .spawn((
                Camera3d::default(),
                RenderTarget::Image(target.clone().into()),
                bevy::core_pipeline::tonemapping::Tonemapping::None,
                // About one unit of light at the default exposure, so a texture
                // reads back as roughly its own colour.
                AmbientLight {
                    brightness: 1000.0,
                    ..default()
                },
                Transform::IDENTITY,
            ))
            .id();
        Self {
            apps,
            target,
            camera,
        }
    }

    fn world(&mut self) -> &mut World {
        self.apps.main.world_mut()
    }

    /// One update of every world, then the device drained, so what was submitted
    /// has finished before anything is read.
    fn update(&mut self) {
        self.apps.update();
        self.apps
            .main
            .world()
            .resource::<RenderDevice>()
            .wgpu_device()
            .poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the device drains");
    }

    /// Build `spec`, look at it once, and take it down again.
    fn shoot(&mut self, name: &str, spec: &Spec) -> Shot {
        let staged = self.stage(spec);
        let shot = self.photograph(name);
        self.strike(staged);
        shot
    }

    /// Put `spec` in front of the camera and let it reach the GPU.
    fn stage(&mut self, spec: &Spec) -> Staged {
        let camera = self.camera;
        let material = spec.material(self.world());
        let mesh = self
            .world()
            .resource_mut::<Assets<Mesh>>()
            .add(spec.mesh.clone());
        let subject = self
            .world()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                spec.placement,
            ))
            .id();
        let light = spec.light.map(|shining| {
            self.world()
                .spawn((
                    DirectionalLight {
                        illuminance: 12_000.0,
                        shadow_maps_enabled: false,
                        ..default()
                    },
                    Transform::default().looking_to(shining, Vec3::Y),
                ))
                .id()
        });
        *self.world().get_mut::<Transform>(camera).expect("camera") = spec.eye;

        // Let the material and its textures reach the GPU first.
        self.settle();
        Staged {
            subject,
            light,
            material,
        }
    }

    /// Frames for what was changed to reach the GPU.
    fn settle(&mut self) {
        for _ in 0..SETTLE_FRAMES {
            self.update();
        }
    }

    /// Take down what `stage` put up.
    fn strike(&mut self, staged: Staged) {
        self.world().entity_mut(staged.subject).despawn();
        if let Some(light) = staged.light {
            self.world().entity_mut(light).despawn();
        }
        // Another frame to retire the entities before the next scene.
        self.update();
    }

    /// A picture of what is staged, written out under `name` if asked for.
    fn photograph(&mut self, name: &str) -> Shot {
        let captured: Arc<Mutex<Option<Image>>> = Arc::default();
        let slot = captured.clone();
        let target = self.target.clone();
        let screenshot = self
            .world()
            .spawn(Screenshot::image(target))
            .observe(move |event: On<ScreenshotCaptured>| {
                *slot.lock().expect("the slot") = Some(event.image.clone());
            })
            .id();
        let mut image = None;
        for _ in 0..30 {
            self.update();
            image = captured.lock().expect("the slot").take();
            if image.is_some() {
                break;
            }
        }
        let image = image.expect("the screenshot never arrived");
        if let Ok(screenshot) = self.world().get_entity_mut(screenshot) {
            screenshot.despawn();
        }

        if let Some(folder) = std::env::var_os("TERRAIN_GPU_SHOTS").map(PathBuf::from) {
            std::fs::create_dir_all(&folder).expect("the evidence folder");
            image
                .clone()
                .try_into_dynamic()
                .expect("a picture")
                .to_rgb8()
                .save(folder.join(format!("{name}.png")))
                .expect("the picture is written");
        }
        Shot::of(&image)
    }
}

/// A scene put in front of the camera.
struct Staged {
    subject: Entity,
    light: Option<Entity>,
    material: Handle<TerrainSplatMaterial>,
}

/// What came back from the render target.
struct Shot {
    /// Linear RGB per pixel, row-major, top row first.
    pixels: Vec<Vec3>,
}

impl Shot {
    fn of(image: &Image) -> Self {
        let data = image.data.as_ref().expect("a captured image has data");
        assert_eq!(data.len(), (SIDE * SIDE * 4) as usize, "a render target");
        let pixels = data
            .chunks_exact(4)
            .map(|texel| {
                let linear = LinearRgba::from(Srgba::rgb_u8(texel[0], texel[1], texel[2]));
                Vec3::new(linear.red, linear.green, linear.blue)
            })
            .collect();
        Self { pixels }
    }

    fn at(&self, x: u32, y: u32) -> Vec3 {
        self.pixels[(y * SIDE + x) as usize]
    }

    /// The mean colour of a block of pixels.
    fn mean(&self, (x0, y0): (u32, u32), (x1, y1): (u32, u32)) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += self.at(x, y);
            }
        }
        sum / ((x1 - x0) * (y1 - y0)) as f32
    }

    /// How much the luminance varies along the lines of a block: the standard
    /// deviation along each row (`along_rows`) or column, averaged over the
    /// lines. Stripes show in one direction and not the other.
    fn variation(&self, (x0, y0): (u32, u32), (x1, y1): (u32, u32), along_rows: bool) -> f32 {
        let lines: Vec<Vec<f32>> = if along_rows {
            (y0..y1)
                .map(|y| (x0..x1).map(|x| luminance(self.at(x, y))).collect())
                .collect()
        } else {
            (x0..x1)
                .map(|x| (y0..y1).map(|y| luminance(self.at(x, y))).collect())
                .collect()
        };
        let spread: f32 = lines
            .iter()
            .map(|line| {
                let mean = line.iter().sum::<f32>() / line.len() as f32;
                (line.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / line.len() as f32).sqrt()
            })
            .sum();
        spread / lines.len() as f32
    }

    /// The standard deviation of luminance over a whole block.
    fn spread(&self, (x0, y0): (u32, u32), (x1, y1): (u32, u32)) -> f32 {
        let values: Vec<f32> = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .map(|(x, y)| luminance(self.at(x, y)))
            .collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
    }

    /// The mean absolute difference between two shots, per channel.
    fn difference(&self, other: &Shot) -> f32 {
        self.pixels
            .iter()
            .zip(&other.pixels)
            .map(|(a, b)| (*a - *b).abs().element_sum() / 3.0)
            .sum::<f32>()
            / self.pixels.len() as f32
    }

    fn finite_and_lit(&self) -> bool {
        self.pixels.iter().all(|p| p.is_finite())
            && self.mean((0, 0), (SIDE, SIDE)).max_element() > 0.01
    }
}

fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// What the test albedo and height textures are made of.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// A flat grey.
    Flat,
    /// Bands that vary along the texture's V.
    Stripes,
    /// A checkerboard.
    Checker,
}

impl Pattern {
    /// Whether texel `(x, y)` is in the bright half.
    fn bright(self, x: u32, y: u32) -> bool {
        match self {
            Self::Flat => true,
            Self::Stripes => (y / SQUARE).is_multiple_of(2),
            Self::Checker => ((x / SQUARE) + (y / SQUARE)).is_multiple_of(2),
        }
    }
}

/// One thing to look at, and how its ground is shaded.
#[derive(Clone)]
struct Spec {
    mesh: Mesh,
    placement: Transform,
    eye: Transform,
    pattern: Pattern,
    /// A checkerboard of normals leaning either way along the texture's U.
    bumps: bool,
    /// A height map in the shape of the pattern.
    heights: bool,
    /// A second, flat blue texture painted over the left half of the control
    /// map, so the shader meets cells whose four corners name different ids.
    split: bool,
    uv_scale: f32,
    detile: f32,
    texture_colour: f32,
    colour_map: Option<[u8; 3]>,
    surface: SurfaceSettings,
    /// The direction a directional light shines in, if there is one.
    light: Option<Vec3>,
}

impl Spec {
    fn new(mesh: Mesh, placement: Transform, eye: Transform) -> Self {
        Self {
            mesh,
            placement,
            eye,
            pattern: Pattern::Flat,
            bumps: false,
            heights: false,
            split: false,
            uv_scale: 0.5,
            detile: 0.0,
            texture_colour: 1.0,
            colour_map: None,
            surface: SurfaceSettings::default(),
            light: None,
        }
    }

    /// The splat material this spec shades with, built the way a host builds it.
    fn material(&self, world: &mut World) -> Handle<TerrainSplatMaterial> {
        world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
            let mut materials = world.resource_mut::<Assets<TerrainSplatMaterial>>();
            let pattern = self.pattern;
            let albedo = texture(TextureFormat::Rgba8UnormSrgb, |x, y| match pattern {
                Pattern::Flat => [128, 128, 128, 255],
                _ if pattern.bright(x, y) => [230, 230, 230, 255],
                _ => [40, 40, 40, 255],
            });
            // Normals leaning along U one way and the other, 45 degrees off flat.
            let normal = texture(TextureFormat::Rgba8Unorm, |x, y| {
                if ((x / SQUARE) + (y / SQUARE)).is_multiple_of(2) {
                    [217, 128, 218, 255]
                } else {
                    [38, 128, 218, 255]
                }
            });
            // High where the pattern is bright, so relief moves the lookups
            // across an edge the albedo shows.
            let height = texture(TextureFormat::Rgba8Unorm, |x, y| {
                if pattern.bright(x, y) {
                    [240, 240, 240, 255]
                } else {
                    [20, 20, 20, 255]
                }
            });
            let mut set = TextureSet {
                entries: vec![TextureSetEntry {
                    normal: self.bumps.then(|| "n.png".to_string()),
                    height: self.heights.then(|| "h.png".to_string()),
                    uv_scale: self.uv_scale,
                    detile: self.detile,
                    ..TextureSetEntry::new("ground", "a.png")
                }],
            };
            let mut handles = TextureSetImages {
                albedo: vec![Some(images.add(albedo))],
                normal: vec![self.bumps.then(|| images.add(normal))],
                height: vec![self.heights.then(|| images.add(height))],
                occlusion: vec![None],
                roughness: vec![None],
            };
            let mut colours = vec![self.texture_colour];
            if self.split {
                set.entries.push(TextureSetEntry {
                    uv_scale: self.uv_scale,
                    ..TextureSetEntry::new("second", "b.png")
                });
                handles.albedo.push(Some(
                    images.add(texture(TextureFormat::Rgba8UnormSrgb, |_, _| {
                        [60, 120, 200, 255]
                    })),
                ));
                handles.normal.push(None);
                handles.height.push(None);
                handles.occlusion.push(None);
                handles.roughness.push(None);
                colours.push(1.0);
            }
            let built = splat_images(&set, &handles, &images).expect("the set stacks");
            let arrays = SplatArrayHandles {
                albedo: images.add(built.albedo),
                normal: images.add(built.normal),
                height: images.add(built.height),
                occlusion: images.add(built.occlusion),
                roughness: images.add(built.roughness),
                stats: built.stats,
            };

            // Row-major, x fastest, like the texture the shader loads from.
            let mut painted = vec![Control::default(); (RESOLUTION * RESOLUTION) as usize];
            if self.split {
                for (index, word) in painted.iter_mut().enumerate() {
                    if (index as u32 % RESOLUTION) < RESOLUTION / 2 {
                        *word = Control::default().with_base_id(1);
                    }
                }
            }
            let control = images.add(control_image(&ControlTexels::from_control(
                &painted, RESOLUTION,
            )));
            let slope = images.add(slope_image(&Heightmap::new(
                RESOLUTION,
                Vec2::splat(EXTENT),
                1.0,
            )));
            let tint = images.add(tint_image(&[], RESOLUTION));
            let map = self.colour_map.map(|[r, g, b]| {
                ColourMap::new(
                    RESOLUTION,
                    1,
                    vec![[r, g, b, 255]; (RESOLUTION * RESOLUTION) as usize],
                )
                .expect("a map the format holds")
            });
            let colour_map = ColourMapBinding::upload(map.as_ref(), &mut images);
            materials.add(TerrainSplatMaterial::new(
                &set,
                &colours,
                arrays,
                control,
                slope,
                tint,
                colour_map,
                Vec2::splat(EXTENT),
                RESOLUTION,
                AutoTerrainSettings::default(),
                self.surface,
            ))
        })
    }
}

/// A square image whose texel at `(x, y)` is `texel(x, y)`.
fn texture(format: TextureFormat, texel: impl Fn(u32, u32) -> [u8; 4]) -> Image {
    let mut data = Vec::with_capacity((TEXTURE_SIDE * TEXTURE_SIDE * 4) as usize);
    for y in 0..TEXTURE_SIDE {
        for x in 0..TEXTURE_SIDE {
            data.extend_from_slice(&texel(x, y));
        }
    }
    Image::new(
        Extent3d {
            width: TEXTURE_SIDE,
            height: TEXTURE_SIDE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        // The set is stacked from these in the main world.
        RenderAssetUsages::default(),
    )
}

/// A quad facing `normal`, its winding made to match whichever way the corners
/// were given, with the UV0 each corner is given.
fn quad(corners: [Vec3; 4], normal: Vec3, uvs: [[f32; 2]; 4]) -> Mesh {
    let facing = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
    let order: [u32; 6] = if facing.dot(normal) < 0.0 {
        [0, 2, 1, 0, 3, 2]
    } else {
        [0, 1, 2, 0, 2, 3]
    };
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        corners.iter().map(Vec3::to_array).collect::<Vec<_>>(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal.to_array(); 4]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs.to_vec());
    mesh.insert_indices(Indices::U32(order.to_vec()));
    mesh
}

/// Ground, facing up, with the UV0 the terrain mesher writes: U along X and V
/// along Z, over the whole plate.
fn plate() -> Mesh {
    let h = EXTENT / 2.0;
    quad(
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        Vec3::Y,
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    )
}

/// A cliff face standing in the XY plane at z = 0 and facing +Z, with the UV0 the
/// terrain mesher would write for it: a function of X and Z alone, so constant
/// up the face. Projected from above, a texture is smeared into vertical streaks
/// there.
fn wall() -> Mesh {
    let h = EXTENT / 2.0;
    quad(
        [
            Vec3::new(-h, 0.0, 0.0),
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(h, EXTENT, 0.0),
            Vec3::new(-h, EXTENT, 0.0),
        ],
        Vec3::Z,
        [[0.0, 0.5], [1.0, 0.5], [1.0, 0.5], [0.0, 0.5]],
    )
}

/// A camera at `eye` looking at `at`.
fn looking(eye: Vec3, at: Vec3, up: Vec3) -> Transform {
    Transform::from_translation(eye).looking_at(at, up)
}

/// Straight down on the plate, with +X to the right.
fn overhead() -> Transform {
    looking(Vec3::new(0.0, 4.0, 0.01), Vec3::ZERO, Vec3::NEG_Z)
}

fn facing_the_wall() -> Transform {
    looking(Vec3::new(0.0, 2.0, 6.0), Vec3::new(0.0, 2.0, 0.0), Vec3::Y)
}

#[test]
#[ignore = "needs a GPU adapter: run with --ignored on a machine that has one"]
fn the_splat_shader_renders_on_a_gpu() {
    let mut gpu = Gpu::new();
    let ground = Spec::new(plate(), Transform::IDENTITY, overhead());

    // Flat ground, a flat grey: lit, finite, and grey.
    let flat = gpu.shoot("flat_ground", &ground);
    assert!(flat.finite_and_lit(), "flat ground renders lit");
    let grey = flat.mean(MIDDLE.0, MIDDLE.1);
    assert!(
        grey.max_element() < 0.98 && grey.min_element() > 0.02,
        "flat grey ground is neither black nor blown out: {grey}"
    );
    assert!(
        (grey.x - grey.z).abs() < 0.03 * grey.x,
        "a grey texture lights grey: {grey}"
    );

    // A colour map colours the ground as far as the slot has given up its colour.
    let red = gpu.shoot(
        "colour_map_taken",
        &Spec {
            colour_map: Some([255, 0, 0]),
            texture_colour: 0.0,
            ..ground.clone()
        },
    );
    let tinted = red.mean(MIDDLE.0, MIDDLE.1);
    assert!(
        tinted.x > 0.05 && tinted.x > 4.0 * tinted.y.max(tinted.z),
        "a red colour map under a slot that keeps none of its colour is red: {tinted}"
    );
    let kept = gpu.shoot(
        "colour_map_ignored",
        &Spec {
            colour_map: Some([255, 0, 0]),
            texture_colour: 1.0,
            ..ground.clone()
        },
    );
    let kept = kept.mean(MIDDLE.0, MIDDLE.1);
    assert!(
        (kept.x - kept.y).abs() < 0.05 * kept.x,
        "a slot that keeps its colour ignores the map: {kept}"
    );

    // Grain lays the texture's finer squares back over a colour map, as far as
    // the slot has given its colour up: a variant of the shader that only a
    // terrain with grain compiles, and a picture that differs from the same
    // terrain without it.
    let mapped = Spec {
        pattern: Pattern::Checker,
        colour_map: Some([200, 60, 60]),
        texture_colour: 0.0,
        ..ground.clone()
    };
    let plain = gpu.shoot("colour_map_no_grain", &mapped);
    let grained = gpu.shoot(
        "colour_map_grain",
        &Spec {
            surface: SurfaceSettings {
                grain: 1.0,
                ..SurfaceSettings::default()
            },
            ..mapped
        },
    );
    assert!(plain.finite_and_lit() && grained.finite_and_lit());
    let mean = grained.mean(MIDDLE.0, MIDDLE.1);
    assert!(
        mean.x > 2.0 * mean.z,
        "the grained ground is still the colour map's red: {mean}"
    );
    assert!(
        plain.difference(&grained) > 0.01,
        "grain changed nothing: {}",
        plain.difference(&grained)
    );

    // Cells whose corners name different ids take the merged path: blue on one
    // side, grey on the other, and a blend between that is neither black nor a
    // flash.
    let seam = gpu.shoot(
        "two_ids",
        &Spec {
            split: true,
            ..ground.clone()
        },
    );
    assert!(seam.finite_and_lit(), "merged ids render lit");
    let blue_side = seam.mean((2, 40), (20, 88));
    let grey_side = seam.mean((72, 40), (120, 88));
    assert!(
        blue_side.z > 1.5 * blue_side.x,
        "the left of the plate is the blue texture: {blue_side}"
    );
    assert!(
        (grey_side.x - grey_side.z).abs() < 0.03 * grey_side.x,
        "the right of the plate is the grey texture: {grey_side}"
    );
    let between = (luminance(blue_side).min(luminance(grey_side)) * 0.5)
        ..(luminance(blue_side).max(luminance(grey_side)) * 1.5);
    for x in 24..68 {
        let blended = luminance(seam.at(x, 64));
        assert!(
            between.contains(&blended),
            "column {x} of the blend is {blended}, outside {between:?}"
        );
    }

    // A cliff shows the texture's bands across its face, level and the whole way
    // up it. Projected from above, a face whose UV0 is constant up it would be
    // one row of the texture smeared into vertical streaks, with no variation
    // down it at all.
    let cliff = Spec {
        pattern: Pattern::Stripes,
        ..Spec::new(wall(), Transform::IDENTITY, facing_the_wall())
    };
    let face = gpu.shoot("cliff_stripes", &cliff);
    assert!(face.finite_and_lit(), "a cliff renders lit");
    let (from, to) = ((40, 30), (88, 98));
    let down = face.variation(from, to, false);
    let across = face.variation(from, to, true);
    assert!(
        down > 0.1 && across < 0.25 * down,
        "bands cross a cliff face: variation up it {down}, along it {across}"
    );

    // A terrain turned about Y, seen by a camera and lit by a light turned with
    // it, is the same picture: the planes, their coordinates and their normals
    // are the terrain's own. Checks and bumps vary along both of a plane's axes,
    // so a coordinate or a normal taken from the world instead would show.
    let lit_cliff = Spec {
        pattern: Pattern::Checker,
        bumps: true,
        light: Some(Vec3::new(-0.5, -0.4, -0.77).normalize()),
        ..Spec::new(wall(), Transform::IDENTITY, facing_the_wall())
    };
    let upright = gpu.shoot("cliff_bumps", &lit_cliff);
    assert!(
        upright.spread(MIDDLE.0, MIDDLE.1) > 0.05,
        "the bumps show on the cliff, or the comparison below proves nothing"
    );
    let turn = Quat::from_rotation_y(50.0_f32.to_radians());
    let turned = gpu.shoot(
        "cliff_bumps_turned",
        &Spec {
            placement: Transform::from_rotation(turn),
            eye: looking(
                turn * Vec3::new(0.0, 2.0, 6.0),
                turn * Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            ),
            light: lit_cliff.light.map(|shining| turn * shining),
            ..lit_cliff.clone()
        },
    );
    assert!(
        upright.difference(&turned) < 0.02,
        "a turned cliff looks the same from a camera turned with it: {}",
        upright.difference(&turned)
    );

    // The far band flattens a normal map: close, a light across tilted tiles
    // lights them unevenly, and past the far distance the plate is lit evenly.
    let bumpy = Spec {
        bumps: true,
        light: Some(Vec3::new(-1.0, -0.35, 0.0).normalize()),
        ..ground.clone()
    };
    let with_maps = gpu.shoot(
        "bumps_near",
        &Spec {
            surface: SurfaceSettings {
                far_distance: 0.0,
                ..SurfaceSettings::default()
            },
            ..bumpy.clone()
        },
    );
    let without = gpu.shoot(
        "bumps_far",
        &Spec {
            surface: SurfaceSettings {
                far_distance: 1.0,
                ..SurfaceSettings::default()
            },
            ..bumpy
        },
    );
    let (uneven, even) = (
        with_maps.spread(MIDDLE.0, MIDDLE.1),
        without.spread(MIDDLE.0, MIDDLE.1),
    );
    assert!(
        uneven > 0.02 && even < 0.3 * uneven,
        "the far band lights tilted tiles alike: spread {uneven} close, {even} far"
    );

    // Relief moves where the textures are read, and leaves the ground lit, with
    // and without detiling.
    let rough = Spec {
        pattern: Pattern::Stripes,
        heights: true,
        eye: looking(Vec3::new(0.0, 1.2, 2.5), Vec3::ZERO, Vec3::Y),
        ..ground
    };
    let level = gpu.shoot("relief_off", &rough);
    for detile in [0.0, 0.6] {
        let raised = gpu.shoot(
            &format!("relief_on_detile_{detile}"),
            &Spec {
                detile,
                surface: SurfaceSettings {
                    relief_depth: 0.3,
                    near_distance: 100.0,
                    ..SurfaceSettings::default()
                },
                ..rough.clone()
            },
        );
        assert!(
            raised.finite_and_lit(),
            "relief renders lit at detile {detile}"
        );
        if detile == 0.0 {
            assert!(
                level.difference(&raised) > 0.005,
                "relief moved nothing: {}",
                level.difference(&raised)
            );
        }
    }

    // The editor changes a dial on a material that is already on screen. The
    // pipeline is keyed on whether the relief is on, so the dial crossing zero
    // has to switch the variant under a live material, and back again.
    let staged = gpu.stage(&rough);
    let before = gpu.photograph("live_relief_off");
    let set_relief = |gpu: &mut Gpu, relief_depth: f32| {
        gpu.world()
            .resource_mut::<Assets<TerrainSplatMaterial>>()
            .get_mut(&staged.material)
            .expect("the material is still there")
            .set_surface(SurfaceSettings {
                relief_depth,
                near_distance: 100.0,
                ..SurfaceSettings::default()
            });
        gpu.settle();
    };
    set_relief(&mut gpu, 0.3);
    let live = gpu.photograph("live_relief_on");
    assert!(
        before.difference(&live) > 0.005,
        "a live material ignored the relief dial: {}",
        before.difference(&live)
    );
    set_relief(&mut gpu, 0.0);
    let back = gpu.photograph("live_relief_off_again");
    assert!(
        back.difference(&before) < 0.002,
        "the relief dial back at 0 draws what it drew before: {}",
        back.difference(&before)
    );
    gpu.strike(staged);
}
