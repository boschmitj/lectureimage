import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from slide_dataset import discover_slides, sample_slides, split_by_deck


class SlideDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for deck, filenames in (("61095", ("slide-001.jpg", "slide-002.png")), ("61096", ("slide-001.jpeg",))):
            directory = self.root / "part19" / "6" / deck
            directory.mkdir(parents=True)
            for filename in filenames:
                Image.new("RGB", (8, 8), "white").save(directory / filename)
        invalid = self.root / "part19" / "6" / "61096" / "broken.jpg"
        invalid.write_text("not an image")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_discovers_valid_nested_images_and_metadata(self):
        slides = discover_slides(self.root)
        self.assertEqual(len(slides), 3)
        self.assertEqual(slides[0].to_dict(), {
            "slide_id": "part19_6_61095_slide-001",
            "dataset_partition": "part19",
            "group_id": "6",
            "slide_deck_id": "61095",
            "slide_filename": "slide-001.jpg",
            "source_path": "part19/6/61095/slide-001.jpg",
        })

    def test_sampling_without_repetition_until_pool_is_exhausted(self):
        slides = discover_slides(self.root)
        sampled = sample_slides(slides, 3, seed=4)
        self.assertEqual(len({slide.slide_id for slide in sampled}), 3)

    def test_discovery_cache_skips_image_validation_on_second_run(self):
        cache = self.root / "discovery.json"
        first = discover_slides(self.root, cache_path=cache)
        with patch("slide_dataset._image_is_readable", side_effect=AssertionError("cache was not used")):
            second = discover_slides(self.root, cache_path=cache)
        self.assertEqual(first, second)

    def test_split_keeps_decks_together(self):
        slides = discover_slides(self.root)
        splits = split_by_deck(slides, train=0.5, validation=0.5, test=0, seed=1)
        locations = {
            slide.deck_key: split
            for split, values in splits.items()
            for slide in values
        }
        self.assertEqual(len(locations), 2)
        self.assertEqual(
            {locations[slide.deck_key] for slide in slides},
            {locations[slides[0].deck_key], locations[slides[-1].deck_key]},
        )


if __name__ == "__main__":
    unittest.main()
