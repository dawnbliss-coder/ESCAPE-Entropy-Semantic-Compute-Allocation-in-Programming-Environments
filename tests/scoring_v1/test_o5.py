import json
import pytest
from escape_scoring.contract import Dataset
from escape_scoring.fixtures import create_fixture
from escape_scoring.o5 import region_table, regress


def test_region_join_and_overlap_deduplication(tmp_path):
    create_fixture(tmp_path / "data")
    data = Dataset(tmp_path / "data")
    s = data.load(data.manifest[0]["sample_id"])
    s.regions.append({**s.regions[-1], "region_id": "overlapping", "start_byte": 8, "end_byte": 12})
    rows = region_table([s])
    assert len(rows) == 2
    assert rows[0]["is_memory_unsafe"] == 1
    assert rows[1]["is_memory_unsafe"] == 0
    assert rows[0]["identifier_byte_proportion"] == 8 / rows[0]["region_length"]
    h = [r["entropy"] for r in s.patcher["entropy_records"]]
    assert rows[0]["mean_identifier_entropy_nats"] == pytest.approx(sum(h[6:14])/8)
    a, b = rows[0]["start_byte"], rows[0]["end_byte"]
    expected = sum(a <= r["byte_offset"] < b and r["termination_reason"] != "eof" for r in s.patcher["boundaries"])
    assert rows[0]["all_boundary_density"] == expected/(b-a)
    assert sum(r["inverse_bpp_compute_density"] * r["region_length"] for r in rows) == pytest.approx(len(s.patcher["patches"]))
    result, matrix = regress(rows)
    assert "unavailable" in result["inference"]
    assert not result["identified"]
    assert len(matrix) == 2
    assert "is_memory_unsafe" in result["coefficients"]
    # np.linalg.lstsq returns a NumPy rank: identified must be a JSON bool.
    assert json.loads(json.dumps(result, allow_nan=False))["identified"] is False


def test_o5_cli_exports_json_and_design(tmp_path):
    from escape_scoring.__main__ import main
    create_fixture(tmp_path / "data")
    main(["o5", "--data", str(tmp_path / "data"), "--output", str(tmp_path / "o5")])
    result = json.loads((tmp_path / "o5" / "o5.json").read_text())
    assert result["synthetic"] is True
    assert (tmp_path / "o5" / "o5_design.csv").exists()
    assert (tmp_path / "o5" / "o5_regions.csv").exists()
