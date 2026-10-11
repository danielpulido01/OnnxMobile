"""Integrity and recovery checks for the standalone dataset distribution tool."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("share_dataset", Path(__file__).parents[1] / "tools/share_dataset.py")
sharing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sharing)


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "fixture"
        self.data.mkdir()
        for name in sharing.REQUIRED:
            (self.data / name).write_bytes(b"small synthetic fixture: " + name.encode())
        sharing.write_json(self.data / "preprocessing.json", {"classes": ["yes"], "image_shape": [64, 101, 1]})
        sharing.write_json(self.data / "COMPLETED.json", {
            "counts": {"train": 1, "validation": 1, "test": 1},
            "manifest_sha256": sharing.digest(self.data / "manifest.csv")})
        original = self.root / "original"
        original.mkdir()
        (original / "LICENSE").write_text("Synthetic license fixture", encoding="utf-8")
        (original / "README.md").write_text("Synthetic attribution fixture", encoding="utf-8")
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.manifest = sharing.pack(self.data, self.root / "bundle", original)

    def test_exact_round_trip_and_existing_data_preserved(self):
        sharing.verify_or_unpack(self.manifest)
        destination = self.root / "restored"
        sharing.verify_or_unpack(self.manifest, destination)
        for name in sharing.REQUIRED:
            self.assertEqual((self.data / name).read_bytes(), (destination / "fixture" / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            sharing.verify_or_unpack(self.manifest, destination)

    def test_incomplete_download_rejected_before_extraction(self):
        manifest = sharing.read_json(self.manifest)
        part = self.manifest.parent / manifest["parts"][0]["name"]
        with part.open("r+b") as stream:
            stream.truncate(part.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "Incomplete or modified"):
            sharing.verify_or_unpack(self.manifest, self.root / "restored")
        self.assertFalse((self.root / "restored").exists())

    def test_unsafe_member_rejected(self):
        manifest = sharing.read_json(self.manifest)
        manifest["files"][0]["path"] = "fixture/../../outside.npy"
        sharing.write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            sharing.verify_or_unpack(self.manifest, self.root / "restored")

    def test_split_archive_reassembled_and_verified(self):
        manifest = sharing.read_json(self.manifest)
        archive = self.manifest.parent / manifest["parts"][0]["name"]
        content = archive.read_bytes()
        midpoint = len(content) // 2
        manifest["parts"] = []
        for index, chunk in enumerate((content[:midpoint], content[midpoint:]), start=1):
            part = archive.with_name(archive.name + f".part{index:03d}")
            part.write_bytes(chunk)
            manifest["parts"].append({"name": part.name, "size": len(chunk), "sha256": sharing.digest(part)})
        sharing.write_json(self.manifest, manifest)
        sharing.verify_or_unpack(self.manifest, self.root / "restored")
        self.assertEqual((self.root / "restored/fixture/X_test.npy").read_bytes(),
                         (self.data / "X_test.npy").read_bytes())


if __name__ == "__main__":
    unittest.main()
