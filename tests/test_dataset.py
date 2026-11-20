"""ImageNet layout tools on a tiny synthetic tree."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from dataset import prepare_imagenet as tool


def make_tree(root: Path, classes, train=4, val=2) -> None:
    for split, count in (("train", train), ("val", val)):
        for name in classes:
            folder = root / split / name
            folder.mkdir(parents=True)
            for index in range(count):
                (folder / f"{name}_{index}.JPEG").write_bytes(b"jpeg")


class DatasetToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_subset_is_deterministic_and_links_originals(self):
        make_tree(self.root / "full", ["n01440764", "n01443537"])
        args = argparse.Namespace(source=str(self.root / "full"), output=str(self.root / "small"),
                                  train_per_class=2, val_per_class=1, seed=2022)
        self.quiet(tool.subset, args)
        first = sorted(p.relative_to(self.root / "small").as_posix() for p in (self.root / "small").rglob("*.JPEG"))
        self.assertEqual(len(first), 2 * 2 + 2 * 1)
        self.assertTrue(all((self.root / "small" / p).is_symlink() for p in first))
        args.output = str(self.root / "again")
        self.quiet(tool.subset, args)
        again = sorted(p.relative_to(self.root / "again").as_posix() for p in (self.root / "again").rglob("*.JPEG"))
        self.assertEqual(first, again)

    def test_validation_images_are_sorted_into_class_folders(self):
        flat = self.root / "val_flat"
        flat.mkdir()
        for index in (1, 2, 3):
            (flat / f"ILSVRC2012_val_{index:08d}.JPEG").write_bytes(b"jpeg")
        labels = self.root / "LOC_val_solution.csv"
        labels.write_text("ImageId,PredictionString\n"
                          "ILSVRC2012_val_00000001,n01751748 1 2 3 4\n"
                          "ILSVRC2012_val_00000002,n01440764 1 2 3 4 n01440764 5 6 7 8\n"
                          "ILSVRC2012_val_00000003,n01751748 1 2 3 4\n")
        args = argparse.Namespace(val_dir=str(flat), labels=str(labels), output=str(self.root / "val"), move=False)
        self.quiet(tool.val_folders, args)
        self.assertEqual(sorted(p.name for p in (self.root / "val/n01751748").iterdir()),
                         ["ILSVRC2012_val_00000001.JPEG", "ILSVRC2012_val_00000003.JPEG"])
        self.assertEqual(len(list((self.root / "val/n01440764").iterdir())), 1)
        (flat / "ILSVRC2012_val_00000004.JPEG").write_bytes(b"jpeg")
        with self.assertRaisesRegex(ValueError, "no label"):
            self.quiet(tool.val_folders, args)

    def test_check_rejects_an_incomplete_layout(self):
        make_tree(self.root / "data", ["00000", "00001"])
        self.assertFalse(self.quiet(tool.check_root, self.root / "data"))

    def test_hugging_face_names_are_parsed(self):
        match = tool.HF_NAME.match("n01440764_10026_n01440764.JPEG")
        self.assertEqual((match.group("stem"), match.group("wnid")), ("n01440764_10026", "n01440764"))
        match = tool.HF_NAME.match("ILSVRC2012_val_00000293_n01440764.JPEG")
        self.assertEqual((match.group("stem"), match.group("wnid")), ("ILSVRC2012_val_00000293", "n01440764"))


if __name__ == "__main__":
    unittest.main()
