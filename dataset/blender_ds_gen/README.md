# Blender slide dataset utilities

`slide_dataset.py` is independent of Blender scene construction. It discovers
validated slide images, preserves source metadata, splits by slide deck, and
provides deterministic sampling.

```python
from slide_dataset import SLIDE_SOURCE_DIR, discover_slides, sample_slides, split_by_deck

slides = discover_slides(SLIDE_SOURCE_DIR)
splits = split_by_deck(slides, seed=123)
selected = sample_slides(splits["train"], 1000, seed=123)

for slide in selected:
    # Pass slide.source_path (relative to SLIDE_SOURCE_DIR) to Blender.
    annotation_metadata = slide.to_dict()
```

Only readable `.jpg`, `.jpeg`, and `.png` files at
`{partition}/{group_id}/{slide_deck_id}/...` are included. Decks are assigned
as a unit to `train`, `val`, or `test`, preventing slides from one presentation
from crossing split boundaries.

Install the image validator with:

```bash
python -m pip install -r requirements.txt
```

## Blender generation

The generator consumes finished classroom environments; it does not construct
rooms, furniture, lights, or cameras. It reuses the camera and lights already
stored in each environment and creates only one generated slide plane and one
appended human model per sample.

Samples are assigned to environments before rendering. Environments are then
processed sequentially: each `.blend` file is opened once, all assigned samples
are rendered, generated slide/human objects are removed between samples, and
only then is the next environment opened. No environments are processed in
parallel.

Run the complete generator in Blender 5.2.2:

```bash
blender --background classroom_environment.blend --python blender_generator.py -- \
  --slide-source-dir /home/josh/Documents/Datasets/slide_dataset/Raw \
  --environment-dir /home/josh/Documents/Datasets/slide_dataset/environments \
  --human-model /home/josh/Documents/Datasets/slide_dataset/models/Human+body/Human.humanbody.blend \
  --output-dir dataset --samples 100 --seed 12345
```

Use `--environment-file` instead of `--environment-dir` to pin a single
environment. If neither is supplied, the `.blend` file opened by Blender is
used. The human-model path has a compatibility fallback for the commonly
distributed nested path `Human+body/Human/humanbody.blend`.

```bash
blender --background --python blender_generator.py -- \
  --output-dir preview \
  --environment-file /path/to/classroom.blend \
  --samples 1
```

The generator detects a mesh named like `whiteboard`, `blackboard`, `screen`, or
`projection`, identifies its largest vertical planar face, and places the
slide directly in front of that face while preserving its aspect ratio. Use
`--whiteboard-name` when automatic detection is ambiguous. It randomizes the
existing node-based materials, appends the human model, and samples the
existing camera. If an environment has no camera, one camera is created for
that environment; no additional lights are created. Camera candidates use a
4-10 m whiteboard distance, +/-3 m horizontal offset, 1.2-2.0 m floor-relative
height, +/-20° yaw, +/-10° pitch, +/-3° roll, and a 24-50 mm focal length. A
candidate is accepted only when the camera is inside the room, on the
presentation side, geometrically clear, and all four slide corners are inside
the image. Slide area and classroom context are not used as rejection
criteria.
For faster camera-like images, the default render profile uses Eevee Next, one-to-sixteen Eevee render samples, render
simplification with subdivision and child particles disabled, and a 2048-pixel
texture limit. Select `--render-engine CYCLES` or `--no-render-simplify` only
when higher fidelity is required.
Annotations include the four geometric slide corners, independent corner
visibility, environment, camera and human metadata, missing texture paths, and
the original slide metadata. If the source directory has no valid images,
placeholder slides are generated automatically.

The first run validates every source image and may take a few minutes for a
large dataset. Progress is printed during discovery and rendering. A
`.slide_discovery.json` cache is written inside the output directory so later
runs skip image decoding when the source file list and timestamps are
unchanged.

The Blender module is intentionally import-safe outside Blender; only
`generate_dataset` requires `bpy`.
