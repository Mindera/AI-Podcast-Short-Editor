from shorts import episode
from shorts.config import Config
from shorts.episode import Episode, default_slug, short_slug


def test_short_slug_keeps_distinctive_words():
    assert short_slug("Podcast EP12 - Jane Doe on AI (FINAL v3) 1080p.mp4") == "podcast-ep12-jane-doe-on-ai"
    assert short_slug("Conversa com João Silva.MOV") == "conversa-com-joao-silva"


def test_short_slug_cuts_at_word_boundary():
    s = short_slug("the quick brown fox jumps over the lazy dog and keeps running far away.mp4")
    assert len(s) <= 40 and not s.endswith("-") and s.startswith("the-quick-brown")


def test_short_slug_falls_back_when_only_noise():
    assert short_slug("final_v2.mp4") == "episode"


def test_default_slug_sources(monkeypatch):
    monkeypatch.setattr(episode, "drive_title", lambda did: "Mindera Talks #7 – Kotlin Multiplatform.mp4")
    assert default_slug("https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view") == "mindera-talks-7-kotlin-multiplatform"
    monkeypatch.setattr(episode, "drive_title", lambda did: None)
    assert default_slug("https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view") == "drive-1AbCdEfGhI"
    assert default_slug("/videos/My Episode 3 export.mp4") == "my-episode-3"


def test_create_reuses_folder_for_same_source_and_avoids_clashes(tmp_path):
    cfg = Config(out=str(tmp_path))
    a = tmp_path / "src1" / "Episode 1.mp4"
    b = tmp_path / "src2" / "Episode 1.mp4"
    ep1 = Episode.create(str(a), cfg)
    assert ep1.dir == tmp_path / "episode-1"
    assert Episode.create(str(a), cfg).dir == ep1.dir
    assert Episode.create(str(b), cfg).dir == tmp_path / "episode-1-2"
