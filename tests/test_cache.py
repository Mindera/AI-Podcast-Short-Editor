from shorts.cache import atomic_write_text, is_fresh, make_key, mark_done, supersede


def test_cache_roundtrip(tmp_path):
    out = tmp_path / "o.txt"
    atomic_write_text(out, "x")
    k = make_key("a", 1)
    assert not is_fresh(tmp_path, "s", k, [out])
    mark_done(tmp_path, "s", k)
    assert is_fresh(tmp_path, "s", k, [out])
    assert not is_fresh(tmp_path, "s", make_key("a", 2), [out])


def test_supersede_moves_not_deletes(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_text("old")
    dest = supersede(f, tmp_path)
    assert not f.exists() and dest.read_text() == "old"
