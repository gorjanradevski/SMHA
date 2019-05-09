import pytest
import numpy as np
from training.datasets import (
    BaseCocoDataset,
    TrainCocoDataset,
    preprocess_caption,
    FlickrDataset,
)


@pytest.fixture
def coco_json_path():
    return "data/testing_assets/coco_json_file_test.json"


@pytest.fixture
def coco_json_file():
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
def coco_images_path():
    return "images/"


@pytest.fixture
def coco_id_to_filename_true():
    return {1: "images/file1", 2: "images/file2", 3: "images/file3"}


@pytest.fixture
def caption():
    return ".A man +-<      gOeS to BuY>!!!++-= BEER!?@#$%^& BUT BEER or BEER'S"


@pytest.fixture
def caption_true():
    return ["a", "man", "goes", "to", "buy", "beer", "but", "beer", "or", "beer's"]


@pytest.fixture
def coco_id_to_captions_true():
    return {
        1: [["first", "caption"], ["fourth", "caption"]],
        2: [["second", "caption"], ["fifth", "caption"]],
        3: [["third", "caption"]],
    }


@pytest.fixture
def min_unk_sub():
    return 2


@pytest.fixture
def coco_id_to_captions_get_image_paths_and_corresponding_captions():
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


@pytest.fixture
def flickr_texts_path():
    return "data/testing_assets/flickr_tokens.txt"


@pytest.fixture
def flickr_images_path():
    return "data/testing_assets/flickr_images/"


@pytest.fixture
def flickr_train_path():
    return "data/testing_assets/flickr_train.txt"


@pytest.fixture
def flickr_val_path():
    return "data/testing_assets/flickr_val.txt"


def test_coco_read_json(coco_json_path):
    json_file = BaseCocoDataset.read_json(coco_json_path)
    assert "images" in json_file
    assert "annotations" in json_file
    assert type(json_file["images"][0]["id"]) == int
    assert type(json_file["annotations"][0]["image_id"]) == int


def test_coco_parse_image_paths(
    coco_json_file, coco_images_path, coco_id_to_filename_true
):
    id_to_filename = BaseCocoDataset.parse_image_paths(coco_json_file, coco_images_path)
    assert id_to_filename == coco_id_to_filename_true


def test_preprocess_caption(caption, caption_true):
    caption_filtered = preprocess_caption(caption)
    assert caption_filtered == caption_true


def test_coco_parse_captions(coco_json_file, coco_id_to_captions_true):
    id_to_captions = BaseCocoDataset.parse_captions(coco_json_file)
    assert id_to_captions == coco_id_to_captions_true


def test_coco_set_up_class_vars(coco_id_to_captions_true, min_unk_sub):
    BaseCocoDataset.set_up_class_vars(coco_id_to_captions_true.values(), 0)
    assert len(BaseCocoDataset.word2index) == 8
    assert len(BaseCocoDataset.index2word) == 8
    assert sum(BaseCocoDataset.index2word.keys()) == sum(range(8))


def test_coco_dataset_object_creation(coco_images_path, coco_json_path, min_unk_sub):
    dataset = TrainCocoDataset(coco_images_path, coco_json_path, min_unk_sub)
    assert len(dataset.id_to_filename.keys()) == 3
    assert len(dataset.id_to_captions.keys()) == 3


def test_coco_get_image_paths_and_corresponding_captions(
    coco_id_to_filename_true,
    coco_id_to_captions_get_image_paths_and_corresponding_captions,
    min_unk_sub,
):
    BaseCocoDataset.set_up_class_vars(
        coco_id_to_captions_get_image_paths_and_corresponding_captions.values(),
        min_unk_sub,
    )
    image_paths, captions, lengths = TrainCocoDataset.get_data_wrapper(
        coco_id_to_filename_true,
        coco_id_to_captions_get_image_paths_and_corresponding_captions,
    )
    assert len(image_paths) == 15
    assert len(captions) == 15
    assert len(lengths) == 15
    for caption in captions:
        assert len(caption) == 2
    for length in lengths:
        assert np.squeeze(length) == 2


def test_flickr_parse_captions_filenames(flickr_texts_path):
    img_path_caption = FlickrDataset.parse_captions_filenames(flickr_texts_path)
    unique_img_paths = set()
    for img_path in img_path_caption.keys():
        assert len(img_path_caption[img_path]) == 5
        unique_img_paths.add(img_path)
    assert len(unique_img_paths) == 5


def test_flickr_get_data(
    flickr_images_path,
    flickr_texts_path,
    min_unk_sub,
    flickr_train_path,
    flickr_val_path,
):
    flickr_dataset = FlickrDataset(flickr_images_path, flickr_texts_path, min_unk_sub)
    train_images, train_captions, train_lengths = flickr_dataset.get_data(
        flickr_train_path
    )
    assert set(train_images) == {
        "data/testing_assets/flickr_images/1000268201_693b08cb0e.jpg",
        "data/testing_assets/flickr_images/1001773457_577c3a7d70.jpg",
        "data/testing_assets/flickr_images/1002674143_1b742ab4b8.jpg",
    }
    true_train_lengths = [17, 7, 8, 9, 12, 9, 14, 18, 12, 8, 19, 12, 20, 13, 9]
    for caption, length, true_len in zip(
        train_captions, train_lengths, true_train_lengths
    ):
        assert len(caption) == np.squeeze(length)
        assert np.squeeze(length) == true_len

    val_images, val_captions, val_lengths = flickr_dataset.get_data(flickr_val_path)
    true_val_lengths = [12, 14, 17, 11, 11, 9, 8, 11, 11, 12]
    for caption, length, true_len in zip(val_captions, val_lengths, true_val_lengths):
        assert len(caption) == np.squeeze(length)
        assert np.squeeze(length) == true_len
