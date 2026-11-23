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

The generator adapts an already-built classroom environment at render time; it
does not construct rooms, furniture, lights, or cameras, and it creates no new
objects. Each environment must contain a camera (an object named
`DatasetCamera`, falling back to the scene camera) and a mesh object named
`ProjectionScreen` whose active node-based material uses a `TEX_IMAGE`
texture node. If either is missing, generation fails with a clear error.

For every sample the generator loads the slide PNG into the screen's existing
texture node, jitters the screen's position around its base location, randomizes
the camera (Gaussian position offset around its base location, a Gaussian focal
length between 24 and 50 mm, aim at the screen center, plus small rotation
jitter), renders one frame, then resets the screen and camera to their base
positions before the next sample. No acceptance/rejection checks are performed
on camera placement or slide framing.

Samples are assigned to environments before rendering. Environments are then
processed sequentially: each `.blend` file is opened once, all assigned samples
are rendered, the screen and camera are reset between samples, and only then is
the next environment opened. No environments are processed in parallel.

Run the generator in Blender:

```bash
blender --background classroom_environment.blend --python blender_generator.py -- \
  --slide-source-dir /home/josh/Documents/Datasets/slide_dataset/Raw \
  --environment-dir /home/josh/Documents/Datasets/slide_dataset/environments \
  --output-dir dataset --samples 100 --seed 12345
```

Use `--environment-file` instead of `--environment-dir` to pin a single
environment. If neither is supplied, the `.blend` file opened by Blender is
used.

```bash
blender --background --python blender_generator.py -- \
  --output-dir preview \
  --environment-file /path/to/classroom.blend \
  --samples 1
```

Available options:

- `--slide-source-dir` — directory of validated slides (default: the packaged
  `DEFAULT_SLIDE_SOURCE_DIR`)
- `--environment-dir` — directory of `.blend` environments (mutually exclusive
  with `--environment-file`)
- `--environment-file` — single `.blend` environment to render
- `--output-dir` — output directory (default: `dataset`)
- `--samples` — number of samples to render (default: 100)
- `--width` / `--height` — render resolution (default: 1920×1080)
- `--seed` — random seed (default: 42)
- `--train-ratio` / `--val-ratio` / `--test-ratio` — deck split ratios
  (default: 0.8 / 0.1 / 0.1)
- `--camera-min-focal-length` / `--camera-max-focal-length` — focal length
  bounds in mm (default: 24 / 50)
- `--render-samples` — render samples (default: 32)
- `--render-engine` — `BLENDER_EEVEE_NEXT`, `BLENDER_EEVEE`, or `CYCLES`
  (default: `BLENDER_EEVEE_NEXT`)
- `--texture-limit` — texture pixel limit (default: 2048)
- `--no-render-simplify` — disable render simplification
- `--debug` — print debug progress information

The default render profile uses Eevee Next, 32 render samples (TAA render
samples capped at 16 for Eevee Next), render simplification with subdivision and
child particles disabled, ambient occlusion enabled, and a 2048-pixel texture
limit. Select `--render-engine CYCLES` or `--no-render-simplify` only when
higher fidelity is required.

Annotations include the rendered image path, the original slide path, the four
projected slide-corner pixel coordinates, and per-corner visibility. Rendered
images go to `output_dir/images/{split}/` and annotations to
`output_dir/annotations/{split}.json` for `train`, `val`, and `test`. Samples
are drawn from the full slide pool and each image is filed under the split
assigned to its slide deck. If the source directory has no valid images,
placeholder slides are generated automatically so the pipeline can run
end-to-end.

The first run validates every source image and may take a few minutes for a
large dataset. Progress is printed during discovery and rendering. A
`.slide_discovery.json` cache is written inside the output directory so later
runs skip image decoding when the source file list and timestamps are unchanged.

The Blender module is intentionally import-safe outside Blender; only
`generate_dataset` requires `bpy`.
