"""Generate slide photographs from existing Blender classroom environments.

The module is import-safe outside Blender.  ``slide_dataset`` remains
responsible for discovery, deck splitting, sampling, and slide metadata;
everything in this module adapts an already-built environment at render time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
 
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from slide_dataset import (
    DEFAULT_SLIDE_SOURCE_DIR,
    discover_slides,
    sample_slides,
    split_by_deck,
)


@dataclass
class EnvironmentConfig:
    pass

@dataclass
class UniversityLectureHallConfig(EnvironmentConfig):
    # no change in y direction
    # Screen box: center (0, -5.67535, 2.60699), dims X=9.35636m, Z=3.20332m
    # Max offsets are half-dims so the screen stays within the box.
    background_max_x: float = 4.67818
    background_max_z: float = 1.60166
    # Z offset from slide center to screen center (screen center is higher Z)
    screen_slide_z_offset: float = 0.14
    camera_max_y: float = 8.0
    camera_max_x: float = 13.5
    slide_points_positive_y: bool = True

@dataclass
class CameraConfig:
    width: int = 1920
    height: int = 1080
    camera_min_focal_length: float = 24.0
    camera_max_focal_length: float = 50.0

@dataclass
class GeneratorConfig:
    slide_source_dir: Path = DEFAULT_SLIDE_SOURCE_DIR
    output_dir: Path = Path("dataset")
    environment_dir: Path | None = None
    environment_file: Path | None = None
    samples: int = 100
    width: int = 1920
    height: int = 1080
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    camera_min_focal_length: float = 24.0
    camera_max_focal_length: float = 50.0
    render_samples: int = 32
    render_engine: str = "BLENDER_EEVEE_NEXT"
    simplify_render: bool = True
    texture_limit: int = 2048
    proj_screen_name = "ProjectionScreen"
    camera_name = "DatasetCamera"
    debug: bool = False

    def validate(self) -> None:
        if self.samples < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("samples must be non-negative and resolution must be positive")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if self.environment_dir and self.environment_file:
            raise ValueError("provide either --environment-dir or --environment-file, not both")
        if self.camera_min_focal_length <= 0 or self.camera_max_focal_length < self.camera_min_focal_length:
            raise ValueError("camera focal-length limits are invalid")
        if self.render_engine not in {"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"}:
            raise ValueError("render-engine must be BLENDER_EEVEE_NEXT, BLENDER_EEVEE, or CYCLES")
        if self.texture_limit <= 0:
            raise ValueError("texture-limit must be positive")

def _require_blender() -> tuple[Any, Any, Any]:
    """Imports bpy, mathutils, and world_to_camera_view, raising a clear error outside Blender's Python
    interpreter. Isolates Blender dependencies to a single point so the module stays import-safe elsewhere."""
    try:
        import bpy
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Matrix, Vector, Euler
    except ImportError as exc:
        raise RuntimeError("Run scene generation with Blender's Python interpreter") from exc
    return bpy, Vector, Euler, (Matrix, world_to_camera_view)


def _debug(config: GeneratorConfig, *args: Any) -> None:
    """Print debug information only when ``config.debug`` is enabled."""
    if config.debug:
        print("[debug]", *args, flush=True)


def _world_bounds(obj: Any, Vector: Any) -> tuple[Any, Any]:
    """Computes one Blender object's axis-aligned bounding box in world space by transforming its
    local bound_box corners with matrix_world. Returns (min, max) vectors."""
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(point[i] for point in points) for i in range(3))),
        Vector((max(point[i] for point in points) for i in range(3))),
    )


def _bounds_center(objects: Iterable[Any], Vector: Any) -> tuple[Any, Any]:
    """Computes the center and size of the combined bounding box of all mesh objects. Used to find the
    room center and dimensions for clamping camera and human positions inside the environment."""
    bounds = [_world_bounds(obj, Vector) for obj in objects if obj.type == "MESH" and obj.bound_box]
    if not bounds:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    minimum = Vector((min(pair[0][i] for pair in bounds) for i in range(3)))
    maximum = Vector((max(pair[1][i] for pair in bounds) for i in range(3)))
    return (minimum + maximum) / 2, maximum - minimum


def discover_environments(config: GeneratorConfig, bpy: Any) -> list[Path]:
    """Resolves the .blend environment(s) to render: an explicit --environment-file, all .blend files
    under --environment-dir, or the currently opened Blender file as a fallback. Validates existence and
    returns a sorted list of blend paths."""

    if config.environment_file:
        path = config.environment_file.expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".blend":
            raise FileNotFoundError(f"Environment file does not exist: {path}")
        return [path]
    if config.environment_dir:
        directory = config.environment_dir.expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"Environment directory does not exist: {directory}")
        paths = sorted(path.resolve() for path in directory.rglob("*.blend"))
        if not paths:
            raise FileNotFoundError(f"No .blend environments found in {directory}")
        return paths
    current = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    if current and current.is_file():
        return [current]
    raise ValueError("Provide --environment-dir or --environment-file, or run with a .blend file")

def place_slide_on_image(
    bpy: Any,
    image_path: Path,
    generator_config: GeneratorConfig,
) -> tuple[Any, float]:

    projection_screen_name = generator_config.proj_screen_name

    projection_plane = bpy.data.objects.get(projection_screen_name)
    material = projection_plane.active_material

    image_node = next(
        node for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE"
    )
    image = bpy.data.images.load(
        str(image_path),
        check_existing=True
    )

    image_node.image = image
    image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Slide image must have positive dimensions")

    return projection_plane, image_width / image_height


def randomize_camera(bpy: Any, Vector: Any, Euler: Any, camera_config: CameraConfig, projection_screen, camera_name: str, env_config):

    camera = bpy.data.objects.get(camera_name)
    # Use scene camera as fallback
    if camera is None:
        camera = bpy.context.scene.camera

    camera_max_x = env_config.camera_max_x
    camera_max_y = env_config.camera_max_y
    screen_points_pos_y = env_config.slide_points_positive_y

    randomize_camera_pos(Vector, camera, camera_max_x, camera_max_y)
    randomize_camera_specs(camera, camera_config)
    corners, center = adjust_camera_angle(bpy, Vector, projection_screen, camera, screen_points_pos_y)
    randomize_camera_rotation(Vector, Euler, camera, corners, center)

    return corners

    
def randomize_camera_pos(Vector, camera, camera_max_x, camera_max_y):
    default_camera_pos = camera.location
    rand_offset_x = None
    rand_offset_y = None
    while rand_offset_x is None or abs(rand_offset_x) > camera_max_x:
        rand_offset_x = random.gauss(sigma=camera_max_x/2)
    while rand_offset_y is None or abs(rand_offset_y) > camera_max_y:
        rand_offset_y = random.gauss(sigma=camera_max_y/2)

    camera.location = default_camera_pos + Vector((rand_offset_x, rand_offset_y, 0))
    

def randomize_camera_specs(camera, camera_config: CameraConfig):
    camera_min_focal_len = camera_config.camera_min_focal_length
    camera_max_focal_len = camera_config.camera_max_focal_length
    mu = camera_min_focal_len + ((camera_max_focal_len - camera_min_focal_len) / 2)
    sigma = (camera_max_focal_len - camera_min_focal_len) / 4
    camera_focal_len = None

    while camera_focal_len is None or abs(camera_focal_len) > camera_max_focal_len or abs(camera_focal_len) < camera_min_focal_len:
        camera_focal_len = random.gauss(mu, sigma)

    camera.data.lens = camera_focal_len

    
def adjust_camera_angle(bpy, Vector, projection_screen, camera, screen_points_pos_y):
    camera_pos = camera.location

    bpy.context.view_layer.update()
    corners = compute_corners(Vector, projection_screen, screen_points_pos_y)
    center = sum(corners, Vector()) / len(corners)

    direction = center - camera_pos

    # Aligns rotation as quaternion s.t. camera view points in direction of center of projection_screen
    rotation = direction.to_track_quat("-Z", "Y").to_euler()
    camera.rotation_euler = rotation

    return corners, center
    

def compute_corners(Vector, projection_screen, screen_points_pos_y):
    # Use actual mesh vertices instead of bound_box corners
    corners = [
        projection_screen.matrix_world @ Vector(vertex.co)
        for vertex in projection_screen.data.vertices
    ]

    # Get either max or min Y coords; max if screen points in positive y direction, min when in negative y direction
    # Note: Works only if aligned with XZ; no rotation applied to screen
    if screen_points_pos_y:
        y = max(corner.y for corner in corners)
    else:
        y = min(corner.y for corner in corners)

    # Select only those vertices with the same y component
    selected_corners = [
        corner for corner in corners
        if abs(corner.y - y) < 1e-6
    ]

    return selected_corners


def randomize_camera_rotation(
    Vector,
    Euler,
    camera,
    corners,
    center,
    alpha=0.05,
    roll_factor=0.3
):
    # Distance from camera to screen
    distance = (center - camera.location).length

    # Calculate screen diagonal (maximum distance between corners)
    diagonal = max(
        (a - b).length
        for a in corners
        for b in corners
    )

    # Calculate angular screen size
    screen_angle = math.atan(diagonal / (2 * distance))

    # Calculate vertical field of view
    focal_length = camera.data.lens
    sensor_height = camera.data.sensor_height

    fov = 2 * math.atan(sensor_height / (2 * focal_length))

    # Determine angular standard deviation
    sigma = alpha * min(screen_angle, fov / 2)

    # Sample Gaussian noise
    noise_x = random.gauss(0, sigma)
    noise_y = random.gauss(0, sigma)
    noise_z = random.gauss(0, sigma * roll_factor)

    # Convert noise into a quaternion
    noise_rotation = Euler(
        (noise_x, noise_y, noise_z),
        'XYZ'
    ).to_quaternion()

    # Apply noise to the ideal rotation
    ideal_rotation = camera.rotation_euler.to_quaternion()

    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = ideal_rotation @ noise_rotation

def alter_environment(
        bpy: Any,
        Vector,
        Euler, 
        camera_config: CameraConfig,
        env_config: EnvironmentConfig,
        generator_config: GeneratorConfig,
        image_path: str,
        ):

    projection_plane, image_aspect = place_slide_on_image(
        bpy, image_path, generator_config
    )
    randomize_slide_pos(
        Vector, projection_plane, env_config, generator_config, image_aspect
    )
    corners = randomize_camera(
        bpy,
        Vector,
        Euler,
        camera_config,
        projection_plane,
        generator_config.camera_name,
        env_config,
    )

    return corners


def reset_environment(projection_plane, camera, projection_plane_pos, camera_pos, projection_plane_scale):
    projection_plane.location = projection_plane_pos
    projection_plane.scale = projection_plane_scale
    camera.location = camera_pos


def randomize_slide_pos(
    Vector,
    slide_obj,
    env_config,
    generator_config: GeneratorConfig,
    image_aspect: float,
):

    base_position = slide_obj.location.copy()
    base_dimensions = slide_obj.dimensions.copy()
    if image_aspect <= 0:
        raise ValueError("Slide image aspect ratio must be positive")

    max_offset_x = env_config.background_max_x
    max_offset_z = env_config.background_max_z
    z_offset = env_config.screen_slide_z_offset

    base_width = base_dimensions.x
    base_height = base_dimensions.y
    if base_width <= 0 or base_height <= 0:
        raise ValueError("Projection screen must have positive dimensions")
    min_dim_x = max(base_width * 0.5, base_height * 0.5 * image_aspect)
    available_half_height = max_offset_z - abs(z_offset)
    max_dim_x = min(2 * max_offset_x, 2 * available_half_height * image_aspect)
    if max_offset_x <= 0 or max_offset_z <= 0 or min_dim_x > max_dim_x:
        raise ValueError("Slide cannot fit within the configured screen bounds")

    mode_dim_x = min(max(base_width, min_dim_x), max_dim_x)
    new_dim_x = (
        random.triangular(min_dim_x, max_dim_x, mode_dim_x)
        if min_dim_x < max_dim_x
        else min_dim_x
    )
    new_dim_z = new_dim_x / image_aspect
    max_center_offset_x = max_offset_x - new_dim_x / 2
    max_center_offset_z = available_half_height - new_dim_z / 2
    offset_x = random.uniform(-max_center_offset_x, max_center_offset_x)
    offset_z = random.uniform(-max_center_offset_z, max_center_offset_z)

    _debug(generator_config, "Selected slide config without rejection sampling")
    new_scale = slide_obj.scale.copy()
    new_scale.x *= new_dim_x / base_width
    new_scale.y *= new_dim_z / base_height

    slide_obj.location = base_position + Vector((offset_x, 0, offset_z + z_offset))
    slide_obj.scale = new_scale


def take_image(bpy, config: GeneratorConfig, corners, camera, world_to_camera_view):

    scene = bpy.context.scene
    bpy.context.view_layer.update()

    pixel_coords = []
    visibility = []

    for corner in corners:
        co_2d = world_to_camera_view(scene, camera, corner)

        x = co_2d.x * config.width
        y = (1- co_2d.y) * config.height

        visibility.append(
            1
            if co_2d.z >= 0
            and 0 <= co_2d.x <= 1
            and 0 <= co_2d.y <= 1
            else 0
        )

        pixel_coords.append((round(x), round(y)))

    return pixel_coords, visibility
    

def _degrade(path: Path, rng: random.Random, probability: float) -> None:
    """With probability `probability`, applies mild degradation to the rendered PNG: optional Gaussian blur,
    contrast/brightness jitter, and quality-82-98 re-save. Silently no-ops if Pillow is missing or the draw
    is skipped."""
    if rng.random() >= probability:
        return
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        return
    with Image.open(path) as image:
        image = image.convert("RGB")
        if rng.random() < 0.6:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
        image.save(path, quality=rng.randint(82, 98))


def _set_render_settings(scene: Any, config: GeneratorConfig) -> None:
    """Configures the Blender scene for rendering: selects an available engine, sets PNG RGB output at the
    configured resolution, sample counts, and simplification (subdivision, particles, AO, texture limit, EEVEE
    NEXT TAA samples). Call once per environment load."""
    available = {item.identifier for item in scene.bl_rna.properties["render"].fixed_type.properties["engine"].enum_items}
    for engine in (config.render_engine, "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", scene.render.engine):
        if engine in available:
            scene.render.engine = engine
            break
    scene.render.resolution_x = config.width
    scene.render.resolution_y = config.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    if hasattr(scene, "cycles"):
        scene.cycles.samples = config.render_samples
    if config.simplify_render:
        scene.render.use_simplify = True
        scene.render.simplify_subdivision = 0
        scene.render.simplify_child_particles = 0
        if hasattr(scene.render, "simplify_ao"):
            scene.render.simplify_ao = True
        if hasattr(scene.render, "texture_limit"):
            scene.render.texture_limit = config.texture_limit
    if scene.render.engine == "BLENDER_EEVEE_NEXT":
        eevee = getattr(scene, "eevee", None)
        if eevee and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = min(config.render_samples, 16)


def _environment_assignments(
    selected: list[Any],
    environments: list[Path],
    seed: int,
) -> dict[Path, list[tuple[int, Any, int]]]:
    """Deterministically assigns each sampled slide to an environment using a seeded RNG (one per-sample
    seed), returning environment -> [(sample_index, slide, sample_seed)]. Ensures reproducible environment
    rotation across samples."""
    assignments = {environment: [] for environment in environments}
    rng = random.Random(seed)
    for sample_index, slide in enumerate(selected, start=1):
        sample_seed = rng.randrange(2**32)
        environment = environments[random.Random(sample_seed).randrange(len(environments))]
        assignments[environment].append((sample_index, slide, sample_seed))
    return assignments


def create_placeholder_slides(directory: Path, count: int = 12, seed: int = 12345) -> list[Path]:
    """Generates count simple lecture-slide placeholder PNGs (title bar, bullet lines) when the configured
    slide source is empty, so the pipeline can run end-to-end without real slides. Requires Pillow."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create placeholder slides") from exc
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[Path] = []
    for index in range(count):
        image = Image.new("RGB", (1600, 900), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        accent = tuple(rng.randint(35, 210) for _ in range(3))
        draw.rectangle((0, 0, 1600, 115), fill=accent)
        draw.text((70, 38), f"Lecture {index + 1}", fill="white")
        for row in range(5):
            y = 220 + row * 95
            draw.rectangle((155, y + 3, 1100 - row * 45, y + 20), fill=(55, 65, 80))
        path = directory / f"placeholder-{index + 1:03d}.png"
        image.save(path)
        paths.append(path)
    return paths


def generate_dataset(config: GeneratorConfig) -> None:
    """Generate images and minimal slide annotations."""
    config.validate()
    bpy, Vector, Euler, (Matrix, world_to_camera_view) = _require_blender()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _debug(
        config,
        "generate_dataset started",
        "output_dir=", config.output_dir,
        "slide_source_dir=", config.slide_source_dir,
        "samples=", config.samples,
        "seed=", config.seed,
        "render_engine=", config.render_engine,
        "render_samples=", config.render_samples,
        "debug=", config.debug,
    )
    source_root = config.slide_source_dir.expanduser()
    slides = discover_slides(source_root, cache_path=config.output_dir / ".slide_discovery.json")
    _debug(config, "discovered", len(slides), "slides from", source_root)
    if not slides:
        source_root = config.output_dir / "_placeholder_slides"
        create_placeholder_slides(source_root / "part0" / "group0" / "placeholder_deck", seed=config.seed)
        slides = discover_slides(source_root)
        _debug(config, "generated placeholders, discovered", len(slides), "slides")
    if not slides:
        raise RuntimeError("No readable slides were found or generated")
    environments = discover_environments(config, bpy)
    _debug(config, "found", len(environments), "environments")
    splits = split_by_deck(slides, train=config.train_ratio, validation=config.val_ratio, test=config.test_ratio, seed=config.seed)
    _debug(
        config,
        "splits:",
        {name: len(records) for name, records in splits.items()},
    )
    selected = sample_slides(slides, config.samples, seed=config.seed)
    _debug(config, "selected", len(selected), "slides for rendering")
    split_by_slide = {slide.slide_id: split for split, records in splits.items() for slide in records}
    annotations: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    assignments = _environment_assignments(selected, environments, config.seed)
    _debug(config, "environment assignments:", {str(e): len(s) for e, s in assignments.items() if s})
    camera_config = CameraConfig(
        width=config.width,
        height=config.height,
        camera_min_focal_length=config.camera_min_focal_length,
        camera_max_focal_length=config.camera_max_focal_length,
    )
    env_config = UniversityLectureHallConfig()

    for environment, environment_samples in assignments.items():
        if not environment_samples:
            continue
        print(
            f"Loading environment {environment.name} for {len(environment_samples)} sample(s).",
            flush=True,
        )
        bpy.ops.wm.open_mainfile(filepath=str(environment), load_ui=False)
        _set_render_settings(bpy.context.scene, config)

        camera = bpy.data.objects.get(config.camera_name)
        projection_screen = bpy.data.objects.get(config.proj_screen_name)
        if camera is None:
            camera = bpy.context.scene.camera
        if camera is None:
            raise ValueError("No camera is available")
        if projection_screen is None:
            raise ValueError(f"Projection screen {config.proj_screen_name!r} was not found")
        camera_base_position = camera.location.copy()
        projection_screen_base_position = projection_screen.location.copy()
        projection_screen_base_scale = projection_screen.scale.copy()
        _debug(
            config,
            "environment",
            environment.name,
            "camera=",
            config.camera_name,
            "screen=",
            config.proj_screen_name,
            "base_camera_pos=",
            tuple(round(c, 4) for c in camera_base_position),
            "base_screen_pos=",
            tuple(round(c, 4) for c in projection_screen_base_position),
        )

        for sample_index, slide, _sample_seed in environment_samples:
            _debug(
                config,
                "sample",
                sample_index,
                "slide=",
                slide.slide_id,
                "source=",
                slide.source_path,
                "split=",
                split_by_slide[slide.slide_id],
            )
            reset_environment(projection_screen, camera, projection_screen_base_position, camera_base_position, projection_screen_base_scale)
            corners = alter_environment(bpy,
                              Vector,
                              Euler,
                              camera_config,
                              env_config,
                              config,
                              source_root / slide.source_path)
            pixel_coords, visibility = take_image(bpy,
                       config,
                       corners,
                       camera,
                       world_to_camera_view)

            split = split_by_slide[slide.slide_id]
            render_path = (
                config.output_dir
                / "images"
                / split
                / f"sample_{sample_index:06d}.png"
            ).resolve()
            render_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(render_path)
            render_started = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            render_duration = time.perf_counter() - render_started
            _debug(
                config,
                "rendered sample",
                sample_index,
                "in",
                f"{render_duration:.2f}s",
                "to",
                render_path,
            )

            original_image_path = (source_root / slide.source_path).resolve()
            annotations[split].append({
                "image_path": str(render_path),
                "original_image_path": str(original_image_path),
                "pixel_coordinates": pixel_coords,
                "visibility": visibility,
            })

    _debug(config, "annotation counts:", {name: len(v) for name, v in annotations.items()})
    for split, values in annotations.items():
        path = config.output_dir / "annotations" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        _debug(config, "wrote annotations", path, "(", len(values), "entries)")
    _debug(config, "generate_dataset complete")


def _parse_args(argv: Iterable[str]) -> GeneratorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-source-dir", type=Path, default=DEFAULT_SLIDE_SOURCE_DIR)
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--camera-min-focal-length", type=float, default=24.0)
    parser.add_argument("--camera-max-focal-length", type=float, default=50.0)
    parser.add_argument("--render-samples", type=int, default=32)
    parser.add_argument(
        "--render-engine",
        choices=("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"),
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--texture-limit", type=int, default=2048)
    parser.add_argument("--no-render-simplify", action="store_true")
    parser.add_argument("--debug", action="store_true", help="print debug progress information")
    args = parser.parse_args(list(argv))
    return GeneratorConfig(
        slide_source_dir=args.slide_source_dir,
        environment_dir=args.environment_dir,
        environment_file=args.environment_file,
        output_dir=args.output_dir,
        samples=args.samples,
        width=args.width,
        height=args.height,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        camera_min_focal_length=args.camera_min_focal_length,
        camera_max_focal_length=args.camera_max_focal_length,
        render_samples=args.render_samples,
        render_engine=args.render_engine,
        texture_limit=args.texture_limit,
        simplify_render=not args.no_render_simplify,
        debug=args.debug,
    )


if __name__ == "__main__":
    """CLI entry point: strips arguments before '--' (Blender convention) and runs generate_dataset with
    the parsed config."""
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    generate_dataset(_parse_args(arguments))
