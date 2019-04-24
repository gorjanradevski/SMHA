import pytest
import numpy as np
from training.datasets import BaseCocoDataset, TrainCocoDataset, preprocess_caption


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
    return ".A man +-<      gOeS to BuY>!!!++-= BEER!?@#$%^& BUT BEER or BEER'S"


@pytest.fixture
def caption_true():
    return ["a", "man", "goes", "to", "buy", "beer", "but", "beer", "or", "beer's"]


@pytest.fixture
def id_to_captions_true():
    return {
        1: [["first", "caption"], ["fourth", "caption"]],
        2: [["second", "caption"], ["fifth", "caption"]],
        3: [["third", "caption"]],
    }


@pytest.fixture
def min_unk_sub():
    return 2


@pytest.fixture
def train():
    return True


@pytest.fixture
def images_paths_true():
    return ["images/file1", "images/file2", "images/file3"]


@pytest.fixture
def id_to_captions_get_image_paths_and_corresponding_captions():
    return {
        1: [
            ["first", "caption"],
            ["fourth", "caption"],
            ["fourth", "caption"],
            ["fourth", "caption"],
            ["fourth", "caption"],
        ],
        2: [
            ["second", "caption"],
            ["fifth", "caption"],
            ["fifth", "caption"],
            ["fifth", "caption"],
            ["fifth", "caption"],
        ],
        3: [
            ["third", "caption"],
            ["third", "caption"],
            ["third", "caption"],
            ["third", "caption"],
            ["third", "caption"],
        ],
    }


def test_read_json(json_path):
    json_file = BaseCocoDataset.read_json(json_path)
    assert "images" in json_file
    assert "annotations" in json_file
    assert type(json_file["images"][0]["id"]) == int
    assert type(json_file["annotations"][0]["image_id"]) == int


def test_parse_image_paths(json_file, images_path, id_to_filename_true):
    id_to_filename = BaseCocoDataset.parse_image_paths(json_file, images_path)
    assert id_to_filename == id_to_filename_true


def test_preprocess_caption(caption, caption_true):
    caption_filtered = preprocess_caption(caption)
    assert caption_filtered == caption_true


def test_parse_captions(json_file, id_to_captions_true):
    id_to_captions = BaseCocoDataset.parse_captions(json_file)
    assert id_to_captions == id_to_captions_true


def test_set_up_class_vars(id_to_captions_true, min_unk_sub):
    BaseCocoDataset.set_up_class_vars(id_to_captions_true.values(), 0)
    assert len(BaseCocoDataset.word2index) == 8
    assert len(BaseCocoDataset.index2word) == 8
    assert sum(BaseCocoDataset.index2word.keys()) == sum(range(8))


def test_dataset_object_creation(images_path, json_path, min_unk_sub):
    dataset = TrainCocoDataset(images_path, json_path, min_unk_sub)
    assert len(dataset.id_to_filename.keys()) == 3
    assert len(dataset.id_to_captions.keys()) == 3


def test_get_image_paths_and_corresponding_captions(
    id_to_filename_true,
    id_to_captions_get_image_paths_and_corresponding_captions,
    min_unk_sub,
):
    BaseCocoDataset.set_up_class_vars(
        id_to_captions_get_image_paths_and_corresponding_captions.values(), min_unk_sub
    )
    image_paths, captions, lengths = TrainCocoDataset.get_data_wrapper(
        id_to_filename_true, id_to_captions_get_image_paths_and_corresponding_captions
    )
    assert len(image_paths) == 15
    assert len(captions) == 15
    assert len(lengths) == 15
    for caption in captions:
        assert len(caption) == 2
    for length in lengths:
        assert np.squeeze(length) == 2
