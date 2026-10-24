"""The fetch cache is tied to the source, so re-pointing an episode folder re-downloads."""

from shorts.fetch import _cached


def test_cached_only_for_the_same_source(tmp_path):
    dest = tmp_path / "source.mp4"
    done = tmp_path / "fetch.json"
    assert not _cached(dest, done, "https://x/a.mp4")  # nothing downloaded yet
    dest.write_bytes(b"video")
    done.write_text('{"source": "https://x/a.mp4", "size": 5}')
    assert _cached(dest, done, "https://x/a.mp4")
    assert not _cached(dest, done, "https://x/other.mp4")  # same folder, different source
    done.write_text("not json")
    assert not _cached(dest, done, "https://x/a.mp4")
