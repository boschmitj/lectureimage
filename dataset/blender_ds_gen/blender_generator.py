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
from dataclasses import asdict, dataclass, field
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


DEFAULT_HUMAN_MODEL = Path(
    "/home/josh/Documents/Datasets/slide_dataset/models/Human+body/Human/humanbody.blend"
)


@dataclass
class GeneratorConfig:
    slide_source_dir: Path = DEFAULT_SLIDE_SOURCE_DIR
    output_dir: Path = Path("dataset")
    environment_dir: Path | None = None
    environment_file: Path | None = None
    human_model: Path = DEFAULT_HUMAN_MODEL
    whiteboard_name: str | None = None
    samples: int = 100
    width: int = 1920
    height: int = 1080
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    camera_min_focal_length: float = 24.0
    camera_max_focal_length: float = 50.0
    camera_min_distance: float = 4.0
    camera_max_distance: float = 10.0
    camera_min_height: float = 1.2
    camera_max_height: float = 2.0
    camera_max_attempts: int = 80 # this is a problem -> TODO: No attempts should be needed
    human_min_distance: float = 1.0 # TODO: Check if the human is in front of the slide or not
    human_max_distance: float = 4.0
    render_samples: int = 32
    render_engine: str = "BLENDER_EEVEE_NEXT"
    simplify_render: bool = True
    texture_limit: int = 2048
    enable_degradation: bool = True
    degradation_probability: float = 0.35

    def validate(self) -> None:
        if self.samples < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("samples must be non-negative and resolution must be positive")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if self.environment_dir and self.environment_file:
            raise ValueError("provide either --environment-dir or --environment-file, not both")
        if self.camera_min_focal_length <= 0 or self.camera_max_focal_length < self.camera_min_focal_length:
            raise ValueError("camera focal-length limits are invalid")
        if self.camera_min_distance <= 0 or self.camera_max_distance < self.camera_min_distance:
            raise ValueError("camera distance limits are invalid")
        if self.camera_min_height < 0 or self.camera_max_height < self.camera_min_height:
            raise ValueError("camera height limits are invalid")
        if self.human_min_distance < 1.0 or self.human_max_distance < self.human_min_distance:
            raise ValueError("human distance limits are invalid")
        if self.camera_max_attempts <= 0:
            raise ValueError("camera-max-attempts must be positive")
        if self.render_engine not in {"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"}:
            raise ValueError("render-engine must be BLENDER_EEVEE_NEXT, BLENDER_EEVEE, or CYCLES")
        if self.texture_limit <= 0:
            raise ValueError("texture-limit must be positive")


@dataclass
class Surface:
    center: Any
    normal: Any
    u: Any
    v: Any
    width: float
    height: float
    object: Any


@dataclass
class RuntimeState:
    config: GeneratorConfig
    rng: random.Random
    source_root: Path
    environment: Path
    whiteboard: Any = None
    surface: Surface | None = None
    slide: Any = None
    slide_corners: list[Any] = field(default_factory=list)
    human_objects: list[Any] = field(default_factory=list)
    human_metadata: dict[str, Any] = field(default_factory=dict)
    camera_metadata: dict[str, Any] = field(default_factory=dict)
    material_metadata: dict[str, Any] = field(default_factory=dict)


def _require_blender() -> tuple[Any, Any, Any]:
    try:
        import bpy
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Matrix, Vector
    except ImportError as exc:
        raise RuntimeError("Run scene generation with Blender's Python interpreter") from exc
    return bpy, Vector, (Matrix, world_to_camera_view)


def _vec(value: Any) -> list[float]:
    return [round(float(component), 6) for component in value]


def _world_bounds(obj: Any, Vector: Any) -> tuple[Any, Any]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(point[i] for point in points) for i in range(3))),
        Vector((max(point[i] for point in points) for i in range(3))),
    )


def _bounds_center(objects: Iterable[Any], Vector: Any) -> tuple[Any, Any]:
    bounds = [_world_bounds(obj, Vector) for obj in objects if obj.type == "MESH" and obj.bound_box]
    if not bounds:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    minimum = Vector((min(pair[0][i] for pair in bounds) for i in range(3)))
    maximum = Vector((max(pair[1][i] for pair in bounds) for i in range(3)))
    return (minimum + maximum) / 2, maximum - minimum


def discover_environments(config: GeneratorConfig, bpy: Any) -> list[Path]:
    """Return explicit environment files, or the currently opened file."""

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


def _find_whiteboard(bpy: Any, Vector: Any, requested: str | None) -> Any:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if requested:
        found = bpy.data.objects.get(requested)
        if not found or found.type != "MESH":
            raise ValueError(f"Whiteboard mesh not found: {requested}")
        return found
    named = [
        obj for obj in meshes
        if any(token in obj.name.lower() for token in ("whiteboard", "blackboard", "projection", "screen"))
    ]
    if named:
        return max(named, key=lambda obj: obj.dimensions.x * obj.dimensions.y * obj.dimensions.z)
    candidates: list[tuple[float, Any]] = []
    for obj in meshes:
        if max(float(value) for value in obj.dimensions) > 30.0:
            continue
        for polygon in obj.data.polygons:
            normal = (obj.matrix_world.to_3x3() @ polygon.normal).normalized()
            area = polygon.area * abs(obj.matrix_world.to_3x3().determinant())
            if abs(normal.z) < 0.35 and area > 0.5:
                candidates.append((area, obj))
    if not candidates:
        raise ValueError("Could not detect a vertical planar whiteboard; use --whiteboard-name")
    return max(candidates, key=lambda item: item[0])[1]


def _environment_surface_reference(environment: Path, requested: str | None) -> tuple[str | None, float | None]:
    if requested:
        return requested, None
    environment_id = str(environment).lower()
    if "university-lecture-hall" in environment_id or environment.stem.lower() == "lecutre_hall":
        return "screen", 3.0
    if "seminar-room" in environment_id or environment.stem.lower() == "classroomblender":
        return "whiteboard.001", None
    if "74-classroom" in environment_id:
        return "blackBoard", None
    return None, None


def _surface_for_object(obj: Any, Vector, room_center: Any) -> Surface:
    transform = obj.matrix_world.to_3x3()
    best: tuple[float, Any] | None = None
    for polygon in obj.data.polygons:
        normal = (transform @ polygon.normal).normalized()
        area = polygon.area * abs(transform.determinant())
        if abs(normal.z) < 0.35 and (best is None or area > best[0]):
            best = (area, polygon)
    if best is None:
        raise ValueError(f"Whiteboard {obj.name!r} has no vertical planar surface")
    polygon = best[1]
    vertices = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
    center = sum(vertices, Vector()) / len(vertices)
    normal = (transform @ polygon.normal).normalized()
    if (room_center - center).dot(normal) < 0:
        normal.negate()
    edge = max(
        (vertices[index] - vertices[(index + 1) % len(vertices)] for index in range(len(vertices))),
        key=lambda vector: vector.length,
    )
    u = edge.normalized()
    v = normal.cross(u).normalized()
    width = max((point - center).dot(u) for point in vertices) - min((point - center).dot(u) for point in vertices)
    height = max((point - center).dot(v) for point in vertices) - min((point - center).dot(v) for point in vertices)
    return Surface(center, normal, u, v, float(width), float(height), obj)


def _slide_material(bpy: Any, image_path: Path) -> Any:
    image = bpy.data.images.load(str(image_path), check_existing=True)
    material = bpy.data.materials.get("GeneratedSlideMaterial") or bpy.data.materials.new("GeneratedSlideMaterial")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "SlideImage"
    texture.image = image
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def place_slide_on_whiteboard(
    bpy: Any,
    Vector: Any,
    surface: Surface,
    image_path: Path,
    rng: random.Random,
    max_width: float | None = None,
) -> tuple[Any, list[Any]]:
    """Create one image plane on the detected board face and return its corners."""

    image = bpy.data.images.load(str(image_path), check_existing=True)
    image_width, image_height = image.size
    aspect = image_width / image_height if image_height else 16 / 9
    margin = 0.96
    available_width = surface.width * margin
    if max_width is not None:
        available_width = min(available_width, max_width)
    width = min(available_width, surface.height * margin * aspect)
    height = width / aspect
    if height > surface.height * margin:
        height = surface.height * margin
        width = height * aspect
    center = surface.center + surface.normal * 0.006
    corners = [
        center - surface.u * width / 2 + surface.v * height / 2,
        center + surface.u * width / 2 + surface.v * height / 2,
        center + surface.u * width / 2 - surface.v * height / 2,
        center - surface.u * width / 2 - surface.v * height / 2,
    ]
    mesh = bpy.data.meshes.new("GeneratedSlideMesh")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()
    slide = bpy.data.objects.new("GeneratedSlide", mesh)
    bpy.context.scene.collection.objects.link(slide)
    mesh.materials.append(_slide_material(bpy, image_path))
    return slide, corners


def _append_human(bpy: Any, Vector: Any, path: Path) -> list[Any]:
    path = path.expanduser()
    if not path.is_file():
        alternatives = (
            path.parent / "Human" / "humanbody.blend",
            path.parent / path.stem / path.name,
        )
        resolved = next((candidate for candidate in alternatives if candidate.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"Human model does not exist: {path}")
        path = resolved
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.objects = [name for name in source.objects if name not in {"Camera", "Light"}]
    imported: list[Any] = []
    for obj in target.objects:
        if obj is None:
            continue
        bpy.context.scene.collection.objects.link(obj)
        obj["generated_human"] = True
        imported.append(obj)
    if not imported:
        imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not imported:
        raise ValueError(f"No mesh objects found in human model: {path}")
    return imported


def _place_human(
    bpy: Any, Vector: Any, Matrix: Any, state: RuntimeState, room_center: Any, room_size: Any
) -> None:
    objects = state.human_objects
    center, _ = _bounds_center(objects, Vector)
    _, human_size = _bounds_center(objects, Vector)
    if human_size.z > room_size.z * 0.85:
        scale = (room_size.z * 0.85) / human_size.z
        scale_matrix = Matrix.Diagonal((scale, scale, scale, 1.0))
        pivot = Matrix.Translation(center) @ scale_matrix @ Matrix.Translation(-center)
        for obj in objects:
            obj.matrix_world = pivot @ obj.matrix_world
        center, _ = _bounds_center(objects, Vector)
    room_min = room_center - room_size / 2
    room_max = room_center + room_size / 2
    original_matrices = [obj.matrix_world.copy() for obj in objects]
    for _ in range(64):
        angle = math.radians(state.rng.uniform(-25, 25))
        rotation = Matrix.Rotation(angle, 4, state.surface.normal)
        lateral = state.rng.uniform(-state.surface.width * 0.48, state.surface.width * 0.48)
        distance = state.rng.uniform(state.config.human_min_distance, state.config.human_max_distance)
        target = state.surface.center + state.surface.normal * distance + state.surface.u * lateral
        target.z = room_min.z + human_size.z / 2 + 0.05
        pivot = Matrix.Translation(target) @ rotation @ Matrix.Translation(-center)
        for obj, original_matrix in zip(objects, original_matrices):
            obj.matrix_world = pivot @ original_matrix
        signed_min = min(
            (point - state.surface.center).dot(state.surface.normal)
            for point in _bbox_points(objects, Vector)
        )
        if signed_min < state.config.human_min_distance:
            correction = state.surface.normal * (state.config.human_min_distance - signed_min)
            for obj in objects:
                obj.location += correction
        human_min, human_max = _combined_bounds(objects, Vector)
        if not any(
            human_min[i] < room_min[i] + 0.01 or human_max[i] > room_max[i] - 0.01
            for i in range(3)
        ):
            minimum_distance = min(
                (point - state.surface.center).dot(state.surface.normal)
                for point in _bbox_points(objects, Vector)
            )
            state.human_metadata = {
                "location": _vec(sum((obj.matrix_world.translation for obj in objects), Vector()) / len(objects)),
                "rotation_z": angle,
                "distance_to_whiteboard": round(minimum_distance, 6),
            }
            return
    raise ValueError("Unable to place the human model inside the environment bounds")


def _bbox_points(objects: Iterable[Any], Vector: Any) -> list[Any]:
    points: list[Any] = []
    for obj in objects:
        if obj.type == "MESH" and obj.bound_box:
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    return points


def _combined_bounds(objects: Iterable[Any], Vector: Any) -> tuple[Any, Any]:
    points = _bbox_points(objects, Vector)
    if not points:
        return Vector(), Vector()
    return (
        Vector((min(point[i] for point in points) for i in range(3))),
        Vector((max(point[i] for point in points) for i in range(3))),
    )


def _randomize_materials(bpy: Any, rng: random.Random, excluded: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for material in bpy.data.materials:
        if material.name in excluded or not material.use_nodes:
            continue
        principled = material.node_tree.nodes.get("Principled BSDF")
        if not principled:
            continue
        roughness = principled.inputs.get("Roughness")
        if roughness:
            original = float(roughness.default_value)
            value = max(0.05, min(1.0, original * rng.uniform(0.9, 1.1)))
            roughness.default_value = value
            metadata[material.name] = {"roughness": value}
        base = principled.inputs.get("Base Color")
        if base:
            color = list(base.default_value)
            factor = rng.uniform(0.97, 1.03)
            color[:3] = [max(0.0, min(1.0, channel * factor)) for channel in color[:3]]
            base.default_value = color
            metadata.setdefault(material.name, {})["base_color"] = color
    return metadata


def _configure_camera(bpy: Any, Vector: Any, state: RuntimeState, room_center: Any, room_size: Any) -> Any:
    from mathutils import Euler

    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if not cameras:
        camera_data = bpy.data.cameras.new(f"{state.environment.stem}_GeneratedCamera")
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        bpy.context.scene.collection.objects.link(camera)
        camera["generated_camera"] = True
        camera_created = True
    else:
        camera = cameras[0]
        camera_created = False
    scene = bpy.context.scene
    scene.camera = camera
    room_min = room_center - room_size / 2
    room_max = room_center + room_size / 2
    for _ in range(state.config.camera_max_attempts):
        distance = state.rng.uniform(state.config.camera_min_distance, state.config.camera_max_distance)
        lateral = state.rng.uniform(-3.0, 3.0)
        height = state.rng.uniform(state.config.camera_min_height, state.config.camera_max_height)
        position = state.surface.center + state.surface.normal * distance + state.surface.u * lateral
        position.z = room_min.z + height
        if any(position[i] < room_min[i] + 0.1 or position[i] > room_max[i] - 0.1 for i in range(3)):
            continue
        if (position - state.surface.center).dot(state.surface.normal) <= 0:
            continue
        target = (
            state.surface.center
            + state.surface.u * state.rng.uniform(-0.1, 0.1) * state.surface.width
            + state.surface.v * state.rng.uniform(-0.1, 0.1) * state.surface.height
        )
        camera.location = position
        base_rotation = (target - position).to_track_quat("-Z", "Y")
        offset = Euler(
            (
                math.radians(state.rng.uniform(-10.0, 10.0)),
                math.radians(state.rng.uniform(-20.0, 20.0)),
                math.radians(state.rng.uniform(-3.0, 3.0)),
            ),
            "XYZ",
        ).to_quaternion()
        camera.rotation_mode = "XYZ"
        camera.rotation_euler = (base_rotation @ offset).to_euler()
        camera.data.lens = state.rng.uniform(
            state.config.camera_min_focal_length, state.config.camera_max_focal_length
        )
        if _camera_geometry_valid(bpy, camera, state.surface, state.slide_corners, room_min, room_max):
            state.camera_metadata = {
                "created": camera_created,
                "horizontal_offset": lateral,
                "distance_from_whiteboard": distance,
                "height_above_floor": height,
                "yaw_degrees": math.degrees(offset.to_euler("XYZ").y),
                "pitch_degrees": math.degrees(offset.to_euler("XYZ").x),
                "roll_degrees": math.degrees(offset.to_euler("XYZ").z),
                "focal_length": float(camera.data.lens),
            }
            return camera
    raise RuntimeError("Unable to find a valid camera with all slide corners inside the image")


def _camera_geometry_valid(
    bpy: Any,
    camera: Any,
    surface: Surface,
    corners: list[Any],
    room_min: Any,
    room_max: Any,
) -> bool:
    position = camera.matrix_world.translation
    if any(position[i] <= room_min[i] + 0.1 or position[i] >= room_max[i] - 0.1 for i in range(3)):
        return False
    if (position - surface.center).dot(surface.normal) <= 0:
        return False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    aim = sum(corners, corners[0].copy() * 0) / len(corners)
    ray = aim - position
    if ray.length == 0:
        return False
    hit, hit_location, _, _, _, _ = bpy.context.scene.ray_cast(
        depsgraph, position, ray.normalized(), distance=ray.length
    )
    if hit and (hit_location - position).length < 0.1:
        return False
    from bpy_extras.object_utils import world_to_camera_view

    for corner in corners:
        projected = world_to_camera_view(bpy.context.scene, camera, corner)
        if not (
            0 <= projected.x < 1
            and 0 <= projected.y < 1
            and projected.z >= 0
        ):
            return False
    return True


def _projected_points(scene: Any, camera: Any, corners: list[Any], width: int, height: int, project: Any) -> list[tuple[float, float, float]]:
    return [
        (float(normalized.x * width), float((1 - normalized.y) * height), float(normalized.z))
        for normalized in (project(scene, camera, corner) for corner in corners)
    ]


def _projected_area_ratio(bpy: Any, camera: Any, corners: list[Any], width: int, height: int) -> float:
    from bpy_extras.object_utils import world_to_camera_view
    points = _projected_points(bpy.context.scene, camera, corners, width, height, world_to_camera_view)
    area = abs(sum(points[index][0] * points[(index + 1) % 4][1] - points[(index + 1) % 4][0] * points[index][1] for index in range(4))) / 2
    return area / (width * height)


def _project_annotations(bpy: Any, camera: Any, corners: list[Any], width: int, height: int) -> dict[str, Any]:
    from bpy_extras.object_utils import world_to_camera_view
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    points = _projected_points(bpy.context.scene, camera, corners, width, height, world_to_camera_view)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    slide_center = sum(corners, corners[0].copy() * 0) / len(corners)
    visibility: dict[str, bool] = {}
    pixels: dict[str, list[float]] = {}
    for name, point, corner in zip(names, points, corners):
        pixels[name] = [round(point[0], 3), round(point[1], 3)]
        # A ray aimed exactly at a mesh vertex can miss the triangle due to
        # edge precision. Aim a tiny distance inside the slide instead.
        target = corner.lerp(slide_center, 0.01)
        direction = target - camera.location
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(
            depsgraph, camera.location, direction.normalized(), distance=direction.length + 0.01
        )
        visibility[name] = (
            0 <= point[0] < width and 0 <= point[1] < height and point[2] >= 0
            and (not hit or hit_object.name == "GeneratedSlide")
        )
    return {
        "corners": pixels,
        "corner_visibility": visibility,
        "fully_in_frame": all(0 <= point[0] < width and 0 <= point[1] < height and point[2] >= 0 for point in points),
        "slide_area_ratio": _projected_area_ratio(bpy, camera, corners, width, height),
    }


def _missing_texture_paths(bpy: Any) -> list[str]:
    missing: set[str] = set()
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath and not Path(bpy.path.abspath(image.filepath)).is_file():
            missing.add(image.filepath)
    return sorted(missing)


def _degrade(path: Path, rng: random.Random, probability: float) -> None:
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


def _remove_generated_sample_objects(bpy: Any) -> None:
    generated = [
        obj for obj in bpy.data.objects
        if obj.name == "GeneratedSlide" or obj.get("generated_human", False)
    ]
    for obj in generated:
        bpy.data.objects.remove(obj, do_unlink=True)


def _environment_assignments(
    selected: list[Any],
    environments: list[Path],
    seed: int,
) -> dict[Path, list[tuple[int, Any, int]]]:
    assignments = {environment: [] for environment in environments}
    rng = random.Random(seed)
    for sample_index, slide in enumerate(selected, start=1):
        sample_seed = rng.randrange(2**32)
        environment = environments[random.Random(sample_seed).randrange(len(environments))]
        assignments[environment].append((sample_index, slide, sample_seed))
    return assignments


def create_placeholder_slides(directory: Path, count: int = 12, seed: int = 12345) -> list[Path]:
    """Create fallback images only when the configured slide source is empty."""

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
    config.validate()
    bpy, Vector, (Matrix, _) = _require_blender()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = config.slide_source_dir.expanduser()
    slides = discover_slides(source_root, cache_path=config.output_dir / ".slide_discovery.json")
    if not slides:
        source_root = config.output_dir / "_placeholder_slides"
        create_placeholder_slides(source_root / "part0" / "group0" / "placeholder_deck", seed=config.seed)
        slides = discover_slides(source_root)
    if not slides:
        raise RuntimeError("No readable slides were found or generated")
    environments = discover_environments(config, bpy)
    splits = split_by_deck(slides, train=config.train_ratio, validation=config.val_ratio, test=config.test_ratio, seed=config.seed)
    selected = sample_slides(slides, config.samples, seed=config.seed)
    split_by_slide = {slide.slide_id: split for split, records in splits.items() for slide in records}
    annotations: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    assignments = _environment_assignments(selected, environments, config.seed)

    for environment, environment_samples in assignments.items():
        if not environment_samples:
            continue
        print(
            f"Loading environment {environment.name} for {len(environment_samples)} sample(s).",
            flush=True,
        )
        bpy.ops.wm.open_mainfile(filepath=str(environment), load_ui=False)
        _set_render_settings(bpy.context.scene, config)
        room_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        room_center, room_size = _bounds_center(room_objects, Vector)
        reference_name, max_slide_width = _environment_surface_reference(environment, config.whiteboard_name)
        whiteboard = _find_whiteboard(bpy, Vector, reference_name)
        surface = _surface_for_object(whiteboard, Vector, room_center)
        missing_textures = _missing_texture_paths(bpy)

        for sample_index, slide, sample_seed in environment_samples:
            _remove_generated_sample_objects(bpy)
            state = RuntimeState(config, random.Random(sample_seed), source_root, environment)
            state.whiteboard = whiteboard
            state.surface = surface
            state.slide, state.slide_corners = place_slide_on_whiteboard(
                bpy,
                Vector,
                state.surface,
                source_root / slide.source_path,
                state.rng,
                max_width=max_slide_width,
            )
            state.human_objects = _append_human(bpy, Vector, config.human_model)
            _place_human(bpy, Vector, Matrix, state, room_center, room_size)
            state.material_metadata = _randomize_materials(
                bpy, state.rng, {state.slide.data.materials[0].name}
            )
            camera = _configure_camera(bpy, Vector, state, room_center, room_size)
            split = split_by_slide[slide.slide_id]
            render_path = config.output_dir / "images" / split / f"sample_{sample_index:06d}.png"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(render_path)
            annotation = {
                "image": render_path.name,
                "width": config.width,
                "height": config.height,
                "environment": environment.stem,
                "environment_path": str(environment),
                "whiteboard": state.whiteboard.name,
                "slide_world_corners": [_vec(corner) for corner in state.slide_corners],
                "slide_surface_dimensions": {
                    "width": max(
                        (state.slide_corners[1] - state.slide_corners[0]).length,
                        (state.slide_corners[2] - state.slide_corners[3]).length,
                    ),
                    "height": max(
                        (state.slide_corners[0] - state.slide_corners[3]).length,
                        (state.slide_corners[1] - state.slide_corners[2]).length,
                    ),
                },
                "seed": sample_seed,
                **_project_annotations(bpy, camera, state.slide_corners, config.width, config.height),
                **slide.to_dict(),
                "camera": {
                    "location": _vec(camera.location),
                    "rotation_euler": _vec(camera.rotation_euler),
                    "focal_length": float(camera.data.lens),
                    **state.camera_metadata,
                },
                "human": state.human_metadata,
                "material_randomization": state.material_metadata,
                "missing_textures": missing_textures,
                "render": {
                    "engine": bpy.context.scene.render.engine,
                    "simplify_render": config.simplify_render,
                    "texture_limit": config.texture_limit,
                    "render_samples": config.render_samples,
                },
            }
            bpy.ops.render.render(write_still=True)
            if config.enable_degradation:
                _degrade(render_path, state.rng, config.degradation_probability)
            annotations[split].append(annotation)

    for split, values in annotations.items():
        path = config.output_dir / "annotations" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    metadata = asdict(config)
    metadata["environments"] = [str(path) for path in environments]
    metadata["environment_count"] = len(environments)
    path = config.output_dir / "metadata" / "generation_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _parse_args(argv: Iterable[str]) -> GeneratorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-source-dir", type=Path, default=DEFAULT_SLIDE_SOURCE_DIR)
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--human-model", type=Path, default=DEFAULT_HUMAN_MODEL)
    parser.add_argument("--whiteboard-name")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-max-attempts", type=int, default=80)
    parser.add_argument("--human-min-distance", type=float, default=1.0)
    parser.add_argument("--human-max-distance", type=float, default=4.0)
    parser.add_argument("--render-samples", type=int, default=32)
    parser.add_argument(
        "--render-engine",
        choices=("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"),
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--texture-limit", type=int, default=2048)
    parser.add_argument("--no-render-simplify", action="store_true")
    parser.add_argument("--no-degradation", action="store_true")
    args = parser.parse_args(list(argv))
    return GeneratorConfig(
        slide_source_dir=args.slide_source_dir,
        environment_dir=args.environment_dir,
        environment_file=args.environment_file,
        human_model=args.human_model,
        whiteboard_name=args.whiteboard_name,
        output_dir=args.output_dir,
        samples=args.samples,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_max_attempts=args.camera_max_attempts,
        human_min_distance=args.human_min_distance,
        human_max_distance=args.human_max_distance,
        render_samples=args.render_samples,
        render_engine=args.render_engine,
        texture_limit=args.texture_limit,
        simplify_render=not args.no_render_simplify,
        enable_degradation=not args.no_degradation,
    )


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    generate_dataset(_parse_args(arguments))
