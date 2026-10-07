from shorts.snap import find_phrase, normalize_turns, snap_bounds, clean_hashtags
from shorts.pick import Boundary
from shorts.words import Word, apply_corrections, capitalize_first, drop_fillers, merge_tokens


def W(s, e, t):
    return Word(s, e, t)


def test_merge_continuation_tokens():
    raw = [{"word": " Kot", "start": 0, "end": 0.2}, {"word": "lin", "start": 0.2, "end": 0.4},
           {"word": " is", "start": 0.5, "end": 0.6}, {"word": " great", "start": 0.6, "end": 0.9},
           {"word": ".", "start": 0.9, "end": 0.9}]
    ws = merge_tokens(raw)
    assert [w.text for w in ws] == ["Kotlin", "is", "great."]
    assert ws[0].end == 0.4


def test_corrections_multiword_and_terms():
    ws = [W(0, .2, "at"), W(.2, .4, "my"), W(.4, .6, "nearer,"), W(.6, .8, "we"), W(.8, 1, "use"),
          W(1, 1.2, "jira"), W(1.2, 1.4, "and"), W(1.4, 1.6, "kotlin"), W(1.6, 1.8, "multiplatform.")]
    out = apply_corrections(ws, {"my nearer": "Mindera"}, ["Jira", "Kotlin Multiplatform"])
    assert [w.text for w in out] == ["at", "Mindera,", "we", "use", "Jira", "and", "Kotlin", "Multiplatform."]
    assert out[1].start == .2 and out[1].end == .6
    assert out[6].start == 1.4 and abs(out[7].end - 1.8) < 1e-6 and out[6].end == out[7].start


def test_fillers_and_capitalize():
    ws = [W(0, 1, "um,"), W(1, 2, "so"), W(2, 3, "yeah")]
    out = capitalize_first(drop_fillers(ws, ["um"]))
    assert [w.text for w in out] == ["So", "yeah"]


WORDS = [W(i, i + 0.8, t) for i, t in enumerate(
    "so what I learned is that teams ship faster when they own it . next question please".split())]


def test_find_phrase_and_snap_padding():
    b = Boundary(start_phrase="what I learned is", start=1.5, end_phrase="when they own it", end=11)
    s, e, i, j = snap_bounds(WORDS, b, 0.15)
    assert WORDS[i].text == "what" and WORDS[j].text == "it"
    # gap to neighbours is 0.2 s → pad limited to the midpoint (0.1 s)
    assert abs(s - (WORDS[i].start - 0.1)) < 1e-6
    assert abs(e - (WORDS[j].end + 0.1)) < 1e-6


def test_find_phrase_tolerates_typos():
    i = find_phrase(WORDS, "teams shipp faster", 6, at_end=False)
    assert WORDS[i].text == "teams"


def test_normalize_turns_fills_gaps():
    t = normalize_turns([[10, 12, "host"], [13, 20, "guest"]], 9.5, 21)
    assert t == [[9.5, 13, "host"], [13, 21, "guest"]]


def test_hashtags():
    assert clean_hashtags(["mindera", "#AI", "Dev Ops"], ["#Mindera"]) == ["#Mindera", "#AI", "#DevOps"]
