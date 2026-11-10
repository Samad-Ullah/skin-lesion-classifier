import hashlib

from skinlesion.data.download import md5sum


def test_md5sum_matches_hashlib(tmp_path):
    f = tmp_path / "sample.bin"
    data = b"skin lesion" * 100_000  # ~1.1 MB, more than one chunk
    f.write_bytes(data)
    assert md5sum(f) == hashlib.md5(data).hexdigest()
