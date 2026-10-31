"""Guards that keep k calibration a priori and smoke runs separate from real results."""

import pytest

from escape_eval import run_eval


def test_golden_source_requires_smoke(tmp_path):
    with pytest.raises(SystemExit, match="smoke"):
        run_eval.main(["--root", str(tmp_path), "--run-id", "x", "--source", "golden", "--domains", "py"])


def test_k_override_requires_smoke(tmp_path):
    with pytest.raises(SystemExit, match="smoke"):
        run_eval.main(["--root", str(tmp_path), "--run-id", "x", "--source", "corpus", "--domains", "py",
                       "--k-override", "py=1"])


def test_k_grid_only_for_prose_unless_smoke(tmp_path):
    with pytest.raises(SystemExit, match="k-grid"):
        run_eval.main(["--root", str(tmp_path), "--run-id", "x", "--source", "corpus", "--domains", "py", "prose",
                       "--k-grid", "0", "1"])


def test_scoring_requires_calibration_file(tmp_path):
    with pytest.raises(SystemExit, match="calibrate_k"):
        run_eval.main(["--root", str(tmp_path), "--run-id", "x", "--source", "corpus", "--domains", "py"])
