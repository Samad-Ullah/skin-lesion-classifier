"""Download HAM10000 from Harvard Dataverse and verify the files.

Run with:  uv run python -m skinlesion.data.download
"""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://dataverse.harvard.edu/api/access/datafile"
USER_AGENT = "skinlesion-downloader/0.1"

# filename -> (Dataverse file id, expected md5 or None)
FILES: dict[str, tuple[int, str | None]] = {
    "HAM10000_images_part_1.zip": (3172585, "4639bfa73ab251610530a97c898e6e46"),
    "HAM10000_images_part_2.zip": (3172584, "da43d6cc50f6613013be07e8986b384b"),
    "HAM10000_metadata.csv": (4338392, None),  # served as original CSV
}


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """MD5 of a file, read in 1 MB chunks so big files don't fill RAM."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download url to dest, writing to a .part file first."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.rename(dest)  # only a complete file gets the final name


def fetch(name: str, file_id: int, md5: str | None, raw_dir: Path) -> Path:
    """Download one file if missing, then verify its checksum."""
    dest = raw_dir / name
    if not dest.exists():
        url = f"{BASE_URL}/{file_id}"
        if name.endswith(".csv"):
            url += "?format=original"
        download(url, dest)
    if md5 is not None and md5sum(dest) != md5:
        dest.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}; file deleted, run again.")
    return dest


def main(raw_dir: Path = Path("data/raw")) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir = raw_dir / "images"
    image_dir.mkdir(exist_ok=True)

    for name, (file_id, md5) in FILES.items():
        path = fetch(name, file_id, md5, raw_dir)
        if path.suffix == ".zip":
            print(f"Extracting {name} ...")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(image_dir)
            path.unlink()  # save 1.4 GB of disk per zip

    n_images = len(list(image_dir.glob("*.jpg")))
    print(f"Done: {n_images} images in {image_dir}")


if __name__ == "__main__":
    main()
