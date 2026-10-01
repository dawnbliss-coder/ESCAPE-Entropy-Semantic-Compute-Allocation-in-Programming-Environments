import hashlib
import pytest

from escape_scoring.contract import Dataset, validate_sample, validate_record
from escape_scoring.fixtures import create_fixture


@pytest.fixture
def dataset(tmp_path):
    create_fixture(tmp_path / "data")
    return Dataset(tmp_path / "data")


def test_complete_fixture_validates(dataset):
    for row in dataset.manifest:
        validate_sample(dataset.load(row["sample_id"]))


@pytest.mark.parametrize("mutation", ["length", "gap", "negative", "float", "nan", "label", "entropy_offset", "hash", "boundary"])
def test_invalid_inputs_rejected(dataset, mutation):
    s = dataset.load(dataset.manifest[0]["sample_id"])
    if mutation == "length": s.manifest["byte_length"] -= 1
    if mutation == "gap": s.patcher["patches"][0]["start_byte"] = 1
    if mutation == "negative": s.nodes[0]["start_byte"] = -1
    if mutation == "float": s.nodes[0]["start_byte"] = 1.2
    if mutation == "nan": s.patcher["entropy_records"][0]["entropy"] = float("nan")
    if mutation == "label": s.patcher["patches"][0]["termination_reason"] = "eof"
    if mutation == "entropy_offset": s.patcher["entropy_records"][0]["byte_offset"] = 1
    if mutation == "hash": s.manifest["text_sha256"] = "0" * 64
    if mutation == "boundary": s.patcher["boundaries"][0]["byte_offset"] += 1
    with pytest.raises(ValueError):
        validate_sample(s)


def test_unicode_node_edges_but_byte_model_can_split_codepoint(dataset):
    s = dataset.load(dataset.manifest[0]["sample_id"])
    s.text = b"\xc3\xa9" + s.text[2:]
    s.manifest["text_sha256"] = hashlib.sha256(s.text).hexdigest()
    s.nodes = [{**s.nodes[0], "start_byte": 1, "end_byte": 2}]
    with pytest.raises(ValueError, match="UTF-8"):
        validate_sample(s)
    s.nodes[0]["start_byte"] = 0
    # Entropy is present at continuation byte 1 and remains valid.
    validate_sample(s)
    s.text = b"\xff" + s.text[1:]
    with pytest.raises(UnicodeDecodeError):
        validate_sample(s)


def test_schema_accepts_future_region_kind(dataset):
    s = dataset.load(dataset.manifest[0]["sample_id"])
    r = {**s.regions[0], "region_kind": "future_custom_kind"}
    validate_record("region_annotations", r)
