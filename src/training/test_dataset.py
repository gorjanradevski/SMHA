import pytest
from training.dataset import Dataset


@pytest.fixture
def json_path():
    return "data/testing_assets/json_file_test.json"


@pytest.fixture
def json_file():
    return {
        "images": [
            {"id": 1, "file_name": "file1"},
            {"id": 2, "file_name": "file2"},
            {"id": 3, "file_name": "file3"},
        ],
        "annotations": [
            {"image_id": 1, "caption": "first caption"},
            {"image_id": 2, "caption": "second caption"},
            {"image_id": 3, "caption": "third caption"},
            {"image_id": 1, "caption": "fourth caption"},
            {"image_id": 2, "caption": "fifth caption"},
        ],
    }


@pytest.fixture
def images_path():
    return "images/"


@pytest.fixture
def id_to_filename_true():
    return {1: "images/file1", 2: "images/file2", 3: "images/file3"}


@pytest.fixture
def caption():
    return ".A man +-<      gOeS to BuY>!!!++-= BEER!?@#$%^&"


@pytest.fixture
def caption_true():
    return ["a", "man", "goes", "to", "buy", "beer"]


@pytest.fixture
def id_to_captions_true():
    return {
        1: [["first", "caption"], ["fourth", "caption"]],
        2: [["second", "caption"], ["fifth", "caption"]],
        3: [["third", "caption"]],
    }


def test_read_json(json_path):
    json_file = Dataset.read_json(json_path)
    assert "images" in json_file
    assert "annotations" in json_file
    assert type(json_file["images"][0]["id"]) == int
    assert type(json_file["annotations"][0]["image_id"]) == int


def test_parse_image_paths(json_file, images_path, id_to_filename_true):
    id_to_filename = Dataset.parse_image_paths(json_file, images_path)
    assert id_to_filename == id_to_filename_true


def test_preprocess_caption(caption, caption_true):
    caption_filtered = Dataset.preprocess_caption(caption)
    assert caption_filtered == caption_true


def test_parse_captions(json_file, id_to_captions_true):
    id_to_captions = Dataset.parse_captions(json_file)
    assert id_to_captions == id_to_captions_true


def test_set_up_class_vars(id_to_captions_true):
    Dataset.set_up_class_vars(id_to_captions_true.values())
    assert len(Dataset.word2index) == 8
    assert len(Dataset.index2word) == 8
    assert sum(Dataset.index2word.keys()) == sum(range(8))
