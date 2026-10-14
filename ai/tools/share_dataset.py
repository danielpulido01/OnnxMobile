"""Pack, verify and unpack a lossless processed dataset for GitHub Releases.

Uses only the Python standard library. Upload this script alongside the bundle
so teammates can extract it before installing the ML dependencies.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

CHUNK = 1024 * 1024
RELEASE_LIMIT = 2 * 1024**3
REQUIRED = [f"{prefix}_{split}.npy" for split in ("train", "validation", "test") for prefix in ("X", "y")]
REQUIRED += ["manifest.csv", "preprocessing.json", "COMPLETED.json"]
AUGMENTED_REQUIRED = ["X_train.npy", "y_train.npy", "manifest.csv", "augmentation.json",
                      "preview.png", "COMPLETED.json"]
OPTIONAL = ["audio_audit.csv", "inventory.csv", "split_counts.csv", "unknown_word_counts.csv",
            "requirements-lock.txt", "summary.ipynb", "summary.md"]  # summary.md: exportaciones anteriores.


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pack(data, output, original):
    data, output, original = data.resolve(), output.resolve(), original.resolve()
    augmented = (data / "augmentation.json").is_file()
    required = AUGMENTED_REQUIRED if augmented else REQUIRED
    for name in required:
        if not (data / name).is_file():
            raise ValueError(f"Missing required file: {name}")
    completion = read_json(data / "COMPLETED.json")
    if augmented:
        expected_hashes = completion["hashes"]
        for name in AUGMENTED_REQUIRED[:-1]:
            if digest(data / name) != expected_hashes.get(name):
                raise ValueError(f"Augmented file does not match COMPLETED.json: {name}")
    elif digest(data / "manifest.csv") != completion["manifest_sha256"]:
        raise ValueError("Manifest does not match COMPLETED.json")
    metadata = read_json(data / ("augmentation.json" if augmented else "preprocessing.json"))
    if augmented and (metadata["format"] != "onnxmobile-augmented-train-v1"
                      or metadata["counts"] != completion["counts"]):
        raise ValueError("Incompatible augmented dataset metadata")
    for name in ("LICENSE", "README.md"):
        if not (original / name).is_file():
            raise ValueError(f"Missing original attribution file: {original / name}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"speech-commands-{data.name}.zip"
    manifest_path = output / f"speech-commands-{data.name}.bundle.json"
    if archive.exists() or manifest_path.exists():
        raise ValueError("Bundle already exists; use another output folder to preserve it")
    files = {name: data / name for name in required + OPTIONAL if (data / name).is_file()}
    files.update({"LICENSE.original": original / "LICENSE", "README.original.md": original / "README.md"})
    attribution = output / "ATTRIBUTION.md"
    attribution.write_text(
        "# Speech Commands v0.02: processed spectrograms\n\n"
        "Original corpus: Pete Warden / Google, Speech Commands v0.02 (2018).\n"
        "Paper: https://arxiv.org/abs/1804.03209\n"
        "Original download: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz\n"
        "Mirror: https://www.kaggle.com/datasets/yashdogra/speech-commands (version 1).\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "Derived by the OnnxMobile team: invalid/silent/duplicate audio removed, official speaker-disjoint "
        "partitions retained, ten commands plus unknown, mono 16 kHz one-second audio converted to "
        "64x101 single-channel float32 log-Mel spectrograms normalized to [0,1].\n" +
        ("Training contains the originals and fixed SpecAugment copies; policy, seed and source identity "
         "are recorded in augmentation.json. Validation/test come from the separate base dataset.\n"
         if augmented else "This is a processed derivative, not the original WAV corpus.\n") +
        "The original license and README are included unchanged.\n" +
        f"Exact processing parameters and provenance are in {'augmentation.json' if augmented else 'preprocessing.json'}; "
        "examples are indexed in manifest.csv.\n",
        encoding="utf-8")
    files[attribution.name] = attribution
    records = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for name, source in files.items():
            print(f"Packing {name} ({source.stat().st_size:,} bytes)", flush=True)
            relative = f"{data.name}/{name}"
            # Hash the exact bytes written into the archive, in one streaming pass.
            checksum, size = hashlib.sha256(), 0
            with source.open("rb") as incoming, bundle.open(relative, "w", force_zip64=True) as outgoing:
                for block in iter(lambda: incoming.read(CHUNK), b""):
                    checksum.update(block)
                    size += len(block)
                    outgoing.write(block)
            records.append({"path": relative, "size": size, "sha256": checksum.hexdigest()})
            if augmented and name in expected_hashes and checksum.hexdigest() != expected_hashes[name]:
                raise ValueError(f"Augmented file changed while packing: {name}")
    parts = []
    if archive.stat().st_size < RELEASE_LIMIT:
        parts.append(archive)
    else:
        # GitHub requires each asset to be under 2 GiB; use 1 GiB parts if necessary.
        with archive.open("rb") as incoming:
            number = 1
            while incoming.tell() < archive.stat().st_size:
                part = archive.with_name(f"{archive.name}.part{number:03d}")
                with part.open("xb") as outgoing:
                    remaining = 1024**3
                    while remaining:
                        block = incoming.read(min(CHUNK, remaining))
                        if not block:
                            break
                        outgoing.write(block)
                        remaining -= len(block)
                parts.append(part)
                number += 1
    manifest = {"format": ("onnxmobile-augmented-bundle-v1" if augmented else "onnxmobile-dataset-bundle-v1"), "dataset": data.name,
                "classes": metadata["classes"], "counts": completion["counts"],
                "layout": "NHWC", "image_shape": metadata["image_shape"],
                "archive_sha256": digest(archive), "archive_size": archive.stat().st_size,
                "parts": [{"name": p.name, "size": p.stat().st_size, "sha256": digest(p)} for p in parts],
                "files": records, "images_included": False, "raw_audio_included": False}
    write_json(manifest_path, manifest)
    print(f"Bundle manifest: {manifest_path}\nArchive bytes: {archive.stat().st_size:,}", flush=True)
    return manifest_path


def load_bundle(path):
    path = path.resolve()
    manifest = read_json(path)
    if manifest["format"] not in ("onnxmobile-dataset-bundle-v1", "onnxmobile-augmented-bundle-v1"):
        raise ValueError("Unsupported bundle format")
    dataset = manifest["dataset"]
    if not dataset or PurePosixPath(dataset).name != dataset or "\\" in dataset or dataset in (".", ".."):
        raise ValueError("Invalid dataset folder")
    if not manifest["parts"]:
        raise ValueError("Bundle has no parts")
    for part in manifest["parts"]:
        name = part["name"]
        if PurePosixPath(name).name != name or "\\" in name or name in (".", ".."):
            raise ValueError("Invalid part name")
        source = path.parent / name
        if source.stat().st_size != part["size"] or digest(source) != part["sha256"]:
            raise ValueError(f"Incomplete or modified download: {name}")
    for record in manifest["files"]:
        member = PurePosixPath(record["path"])
        if (len(member.parts) != 2 or member.parts[0] != dataset or member.name in (".", "..")
                or member.is_absolute() or "\\" in record["path"] or ":" in record["path"]):
            raise ValueError("Unsafe archive member")
    required = AUGMENTED_REQUIRED if manifest["format"] == "onnxmobile-augmented-bundle-v1" else REQUIRED
    required_paths = {f"{dataset}/{name}" for name in required}
    if not required_paths.issubset({record["path"] for record in manifest["files"]}):
        raise ValueError("Bundle is missing training files")
    return manifest


@contextmanager
def open_archive(path, manifest):
    if len(manifest["parts"]) == 1:
        yield path.parent / manifest["parts"][0]["name"]
    else:
        with tempfile.TemporaryDirectory(prefix="dataset-join-", dir=path.parent) as temporary:
            archive = Path(temporary) / "dataset.zip"
            archive.resolve().relative_to(path.parent.resolve())
            with archive.open("xb") as output:
                for part in manifest["parts"]:
                    with (path.parent / part["name"]).open("rb") as incoming:
                        shutil.copyfileobj(incoming, output, CHUNK)
            yield archive


def verify_or_unpack(manifest_path, output=None):
    manifest_path = manifest_path.resolve()
    manifest = load_bundle(manifest_path)
    output = output.resolve() if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        target = output / manifest["dataset"]
        target.resolve().relative_to(output)
        if target.exists():
            raise ValueError(f"Destination exists; it will not be overwritten: {target}")
    with open_archive(manifest_path, manifest) as archive:
        if archive.stat().st_size != manifest["archive_size"] or digest(archive) != manifest["archive_sha256"]:
            raise ValueError("Archive checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="dataset-extract-", dir=output or manifest_path.parent) as temporary:
            staging = Path(temporary)
            staging.resolve().relative_to(output or manifest_path.parent)
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                expected = {r["path"] for r in manifest["files"]}
                if len(names) != len(expected) or set(names) != expected:
                    raise ValueError("Archive contents do not match bundle manifest")
                for record in manifest["files"]:
                    checksum, size = hashlib.sha256(), 0
                    destination = staging / record["path"]
                    if output:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        stream = destination.open("xb")
                    else:
                        stream = None
                    try:
                        with bundle.open(record["path"]) as incoming:
                            for block in iter(lambda: incoming.read(CHUNK), b""):
                                checksum.update(block)
                                size += len(block)
                                if stream:
                                    stream.write(block)
                    finally:
                        if stream:
                            stream.close()
                    if size != record["size"] or checksum.hexdigest() != record["sha256"]:
                        raise ValueError(f"File checksum mismatch: {record['path']}")
                    print(f"Verified {record['path']}", flush=True)
            if output:
                source = staging / manifest["dataset"]
                source.resolve().relative_to(output)
                target.resolve().relative_to(output)
                source.rename(target)
                print(f"Ready for training: {target}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("pack")
    create.add_argument("--data", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--original", type=Path, required=True, help="Folder containing original LICENSE and README.md")
    for name in ("verify", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "unpack":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.data, args.output, args.original)
    else:
        verify_or_unpack(args.manifest, args.output if args.command == "unpack" else None)


if __name__ == "__main__":
    main()
