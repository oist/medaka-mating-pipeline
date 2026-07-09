from __future__ import annotations

"""Tests for pipeline-stage detection and HTML catalogue rendering."""

from pathlib import Path

from scripts.common.pipeline_status import (
    STAGE_ORDER,
    detect_detectron2,
    detect_pipeline_status,
    detect_reconstruction_3d,
    detect_sleap_infer,
)
from scripts.common.render_catalog_html import render_catalog_html

VIEWS = ("top_cam", "side1_cam", "side2_cam")
EXP = "20251225_K_1_1_FCMC"


def _make_sleap(root: Path, exp: str, *, merged=VIEWS, rounds=("round4_260520",)) -> Path:
    exp_dir = root / exp
    exp_dir.mkdir(parents=True)
    for view in merged:
        (exp_dir / f"{view}.slp").write_text("x")
        (exp_dir / f"{view}.frames_0-100.slp").write_text("x")
    for r in rounds:
        (exp_dir / r).mkdir()
    (exp_dir / "logs").mkdir(exist_ok=True)
    return root


def _make_d2(root: Path, exp: str, views=VIEWS) -> Path:
    exp_dir = root / exp
    exp_dir.mkdir(parents=True)
    for view in views:
        (exp_dir / f"{view}.masks_0-100.h5").write_text("x")
        (exp_dir / f"{view}.masks_100-200.h5").write_text("x")
    return root


def test_detect_sleap_infer_full(tmp_path: Path):
    root = _make_sleap(tmp_path / "sleap", EXP)
    s = detect_sleap_infer(root, EXP)
    assert s.present and s.merged_all
    assert s.n_merged_views == 3
    assert s.rounds == ["round4_260520"]
    assert "logs" not in s.rounds
    assert all(s.n_chunks[v] == 1 for v in VIEWS)


def test_detect_sleap_infer_missing(tmp_path: Path):
    s = detect_sleap_infer(tmp_path / "sleap", EXP)
    assert not s.present and not s.merged_all and s.n_merged_views == 0


def test_detect_sleap_infer_partial(tmp_path: Path):
    root = _make_sleap(tmp_path / "sleap", EXP, merged=("top_cam",))
    s = detect_sleap_infer(root, EXP)
    assert s.present and not s.merged_all and s.n_merged_views == 1


def test_detect_sleap_infer_merged_in_round_subdir(tmp_path: Path):
    # merged .slp lives only inside a round subdir (round-tagged merge)
    exp_dir = tmp_path / "sleap" / EXP / "round4_260520"
    exp_dir.mkdir(parents=True)
    for view in VIEWS:
        (exp_dir / f"{view}.slp").write_text("x")
    s = detect_sleap_infer(tmp_path / "sleap", EXP)
    assert s.present and s.merged_all and s.n_merged_views == 3
    assert s.rounds == ["round4_260520"]


def test_detect_sleap_infer_empty_dir_not_present(tmp_path: Path):
    # a stray/empty output folder must not advance the funnel
    (tmp_path / "sleap" / EXP).mkdir(parents=True)
    s = detect_sleap_infer(tmp_path / "sleap", EXP)
    assert not s.present and s.n_merged_views == 0
    st = detect_pipeline_status(EXP, recorded=True, sleap_output_dir=tmp_path / "sleap")
    assert st.furthest_stage == "recorded"


def test_detect_detectron2(tmp_path: Path):
    root = _make_d2(tmp_path / "d2", EXP)
    d = detect_detectron2(root, EXP)
    assert d.present and d.n_views == 3
    assert d.n_mask_chunks["top_cam"] == 2
    assert not detect_detectron2(tmp_path / "d2", "nonexistent").present


def test_detect_reconstruction_root_and_subdir(tmp_path: Path):
    root = tmp_path / "recon"
    root.mkdir()
    # root-level 3d csv only
    (root / f"{EXP}_3d.csv").write_text("x")
    r = detect_reconstruction_3d(root, EXP)
    assert r.has_3d_csv and not r.has_identified_csv

    # subdir with identified csv
    sub = root / EXP / "round4_260520"
    sub.mkdir(parents=True)
    (sub / "round4_260520_3d_identified.csv").write_text("x")
    r2 = detect_reconstruction_3d(root, EXP)
    assert r2.has_identified_csv
    assert "round4_260520" in r2.configs


def test_furthest_stage_ordering(tmp_path: Path):
    sleap = _make_sleap(tmp_path / "sleap", EXP)
    d2 = _make_d2(tmp_path / "d2", EXP)
    recon = tmp_path / "recon"
    (recon).mkdir()
    (recon / f"{EXP}_3d_identified.csv").write_text("x")

    st = detect_pipeline_status(
        EXP,
        recorded=True,
        sleap_output_dir=sleap,
        detectron2_output_dir=d2,
        reconstruction_output_dir=recon,
    )
    assert st.furthest_stage == "identified_3d"

    # sleap only -> sleap_inferred
    st2 = detect_pipeline_status(EXP, sleap_output_dir=sleap)
    assert st2.furthest_stage == "sleap_inferred"

    # nothing -> recorded
    st3 = detect_pipeline_status(EXP, recorded=True)
    assert st3.furthest_stage == "recorded"
    assert STAGE_ORDER.index("identified_3d") == len(STAGE_ORDER) - 1


def test_render_html_escapes_and_writes(tmp_path: Path):
    manifests = [
        {
            "folder_name": "20251225_K_1_1_FCMC",
            "experiment": {"start_date": "2025-12-25", "group": "K", "trial": 1, "day": 1},
            "views": {"top_cam": {"total_frames": 348600}},
            "pipeline": detect_pipeline_status("20251225_K_1_1_FCMC", recorded=True).to_dict(),
        },
        {
            # adversarial: markup metacharacters must be escaped, not injected
            "folder_name": "evil<script>alert(1)</script>",
            "experiment": {"start_date": "2026-01-01", "group": "X&Y", "trial": 2, "day": 3},
            "views": {},
            "pipeline": detect_pipeline_status("evil", recorded=False).to_dict(),
        },
    ]
    out = tmp_path / "catalog.html"
    render_catalog_html(manifests, out)
    text = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "evil&lt;script&gt;" in text
    assert "X&amp;Y" in text
    assert text.count("<tr data-stage=") == 2
