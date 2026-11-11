"""Discover, split, and sample lecture slides for Blender dataset generation.

This module intentionally has no Blender dependency.  Blender scene construction
can consume :class:`SlideRecord` values and remain independent of filesystem
discovery and sampling policy.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_SLIDE_SOURCE_DIR = Path("/home/josh/Documents/Datasets/slide_dataset/Raw")
# Public configuration name used by Blender entry points.
SLIDE_SOURCE_DIR = DEFAULT_SLIDE_SOURCE_DIR
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class SlideRecord:
    """Validated slide and the dataset metadata derived from its path."""

    slide_id: str
    dataset_partition: str
    group_id: str
    slide_deck_id: str
    slide_filename: str
    source_path: str

    @property
    def deck_key(self) -> tuple[str, str, str]:
        return (self.dataset_partition, self.group_id, self.slide_deck_id)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SlideDiscoveryError(RuntimeError):
    """Raised when slide discovery cannot validate its image dependency.""" 


def _file_signature(root: Path, paths: Sequence[Path]) -> list[list[str | int]]:
    return [
        [
            str(path.relative_to(root)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ]
        for path in paths
    ]


def _image_is_readable(path: Path) -> bool:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise SlideDiscoveryError(
            "Pillow is required to validate slide images; install it with "
            "`python -m pip install Pillow`."
        ) from exc

    try:
        with Image.open(path) as image:
            image.verify()
        # verify() does not decode all pixel data, so reopen and load it.
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def discover_slides(
    source_dir: str | Path = DEFAULT_SLIDE_SOURCE_DIR,
    *,
    cache_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[SlideRecord]:
    """Recursively discover readable slides below ``source_dir``.

    The first three relative path components are interpreted as partition,
    group ID, and deck ID.  Deeper paths are accepted, with the final
    component treated as the slide filename.  Files that do not have the
    expected dataset depth or are not readable images are skipped.
    """

    root = Path(source_dir).expanduser()
    if not root.is_dir():
        return []

    cache = Path(cache_path) if cache_path is not None else None
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if cache is not None and cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("root") == str(root.resolve()) and cached.get("files") == _file_signature(root, candidates):
                return [SlideRecord(**item) for item in cached["slides"]]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    records: list[SlideRecord] = []
    total = len(candidates)
    for index, path in enumerate(candidates, start=1):
        relative = path.relative_to(root)
        if len(relative.parts) < 4 or not _image_is_readable(path):
            if progress is not None:
                progress(index, total)
            continue

        partition, group_id, deck_id = relative.parts[:3]
        filename = relative.name
        source_path = relative.as_posix()
        records.append(
            SlideRecord(
                slide_id="_".join((partition, group_id, deck_id, path.stem)),
                dataset_partition=partition,
                group_id=group_id,
                slide_deck_id=deck_id,
                slide_filename=filename,
                source_path=source_path,
            )
        )
        if progress is not None:
            progress(index, total)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = json.dumps({
            "root": str(root.resolve()),
            "files": _file_signature(root, candidates),
            "slides": [record.to_dict() for record in records],
        })
        temporary_cache = cache.with_name(f".{cache.name}.tmp")
        temporary_cache.write_text(cache_payload, encoding="utf-8")
        temporary_cache.replace(cache)
    return records


def split_by_deck(
    slides: Sequence[SlideRecord],
    *,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    seed: int | None = None,
) -> dict[str, list[SlideRecord]]:
    """Split slides while keeping every deck in exactly one split."""

    ratios = (train, validation, test)
    if any(r < 0 for r in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")

    decks: dict[tuple[str, str, str], list[SlideRecord]] = {}
    for slide in slides:
        decks.setdefault(slide.deck_key, []).append(slide)

    deck_keys = list(decks)
    random.Random(seed).shuffle(deck_keys)
    counts = _split_counts(len(deck_keys), ratios)
    boundaries = (counts[0], counts[0] + counts[1])
    groups = (
        ("train", deck_keys[: boundaries[0]]),
        ("val", deck_keys[boundaries[0] : boundaries[1]]),
        ("test", deck_keys[boundaries[1] :]),
    )
    return {
        name: [slide for key in keys for slide in decks[key]]
        for name, keys in groups
    }


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True):
        if sum(counts) == total:
            break
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def sample_slides(
    slides: Sequence[SlideRecord],
    count: int,
    *,
    seed: int | None = None,
    avoid_repetition: bool = True,
) -> list[SlideRecord]:
    """Sample slides, cycling through a shuffled pool before repeating.

    When the available pool is at least ``count`` this is equivalent to a
    sample without replacement.  For larger requests, each additional cycle is
    independently shuffled, minimizing unnecessary adjacent repetition.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if not slides or count == 0:
        return []
    if not avoid_repetition:
        return random.Random(seed).choices(list(slides), k=count)

    rng = random.Random(seed)
    pool = list(slides)
    result: list[SlideRecord] = []
    previous: SlideRecord | None = None
    while len(result) < count:
        rng.shuffle(pool)
        if previous is not None and len(pool) > 1 and pool[0] == previous:
            pool[0], pool[1] = pool[1], pool[0]
        take = min(count - len(result), len(pool))
        result.extend(pool[:take])
        previous = result[-1]
    return result


def metadata_for_sampling(slide: SlideRecord) -> Mapping[str, str]:
    """Return annotation-ready source metadata for a selected slide."""

    return slide.to_dict()
