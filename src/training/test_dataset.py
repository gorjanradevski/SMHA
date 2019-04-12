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


def test_dataset_object_creation(images_path, json_path, min_unk_sub, train):
    dataset = Dataset(images_path, json_path, min_unk_sub, train)
    assert len(dataset.id_to_filename.keys()) == 3
    assert len(dataset.id_to_captions.keys()) == 3


def test_get_image_paths_and_corresponding_captions(
    id_to_filename_true,
    id_to_captions_get_image_paths_and_corresponding_captions,
    min_unk_sub,
):
    Dataset.set_up_class_vars(
        id_to_captions_get_image_paths_and_corresponding_captions.values()
    )
    image_paths, captions = Dataset.get_image_paths_and_corresponding_captions_wrapper(
        id_to_filename_true,
        id_to_captions_get_image_paths_and_corresponding_captions,
        min_unk_sub,
    )
    assert len(image_paths) == 15
    assert len(captions) == 15
    for caption in captions:
        assert len(caption) == 2
