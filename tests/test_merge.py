import random

from shorts.reframe import Run, merge_short_runs, snap_to_frames


def _runs(labels, fps=10):
    return [Run(i / fps, (i + 1) / fps, lab) for i, lab in enumerate(labels)]


def test_merge_terminates_on_random_label_sequences():
    rng = random.Random(1234)
    for _ in range(3000):
        n = rng.randint(1, 300)
        k = rng.randint(1, 5)
        labels = [rng.randrange(k) for _ in range(n)]
        min_len = rng.choice([0.05, 0.3, 0.6, 1.5, 100.0])
        out = merge_short_runs(_runs(labels), min_len)
        total = n / 10
        assert out, "never empty"
        assert abs(out[0].start) < 1e-9 and abs(out[-1].end - total) < 1e-9
        for a, b in zip(out, out[1:]):
            assert abs(a.end - b.start) < 1e-9, "contiguous"
            assert a.label != b.label, "coalesced"
        if len(out) > 1:
            assert all(r.dur >= min_len - 1e-9 for r in out)


def test_merge_absorbs_flicker():
    labels = ["a"] * 30 + ["b"] * 2 + ["a"] * 30
    out = merge_short_runs(_runs(labels), 0.6)
    assert [r.label for r in out] == ["a"]


def test_snap_to_frames_counts_add_up():
    segs = [{"start": 0.0, "end": 1.23}, {"start": 1.23, "end": 4.57}, {"start": 4.57, "end": 9.99}]
    out = snap_to_frames(segs, 25.0, 9.99)
    assert sum(s["frames"] for s in out) == round(9.99 * 25)
    assert out[0]["start"] == 0.0
