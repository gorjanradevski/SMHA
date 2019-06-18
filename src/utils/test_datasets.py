import pytest
from utils.datasets import (
    BaseCocoDataset,
    TrainCocoDataset,
    preprocess_caption,
    FlickrDataset,
    PascalSentencesDataset,
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
    return ".A man +-<      gOeS to BuY>!!!++-= BEER!?@#$%^& BUT BEER or BEER'S   "


@pytest.fixture
def caption_true():
    return "a man goes to buy beer but beer or beer's"


@pytest.fixture
def coco_id_to_captions_true():
    return {
        1: ["first caption", "fourth caption"],
        2: ["second caption", "fifth caption"],
        3: ["third caption"],
    }


@pytest.fixture
def coco_id_to_captions_get_image_paths_and_corresponding_captions():
    return {
        1: [
            "first caption",
            "fourth caption",
            "fourth caption",
            "fourth caption",
            "fourth caption",
        ],
        2: [
            "second caption",
            "fifth caption",
            "fifth caption",
            "fifth caption",
            "fifth caption",
        ],
        3: [
            "third caption",
            "third caption",
            "third caption",
            "third caption",
            "third caption",
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


@pytest.fixture
def pascal_images_path():
    return "data/testing_assets/pascal_images_texts/images"


@pytest.fixture
def pascal_texts_path():
    return "data/testing_assets/pascal_images_texts/texts"


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
    print(caption_filtered)
    assert caption_filtered == caption_true


def test_coco_parse_captions(coco_json_file, coco_id_to_captions_true):
    id_to_captions = BaseCocoDataset.parse_captions(coco_json_file)
    assert id_to_captions == coco_id_to_captions_true


def test_coco_dataset_object_creation(coco_images_path, coco_json_path):
    dataset = TrainCocoDataset(coco_images_path, coco_json_path)
    assert len(dataset.id_to_filename.keys()) == 3
    assert len(dataset.id_to_captions.keys()) == 3


def test_coco_get_image_paths_and_corresponding_captions(
    coco_id_to_filename_true,
    coco_id_to_captions_get_image_paths_and_corresponding_captions,
):
    image_paths, captions = TrainCocoDataset.get_data_wrapper(
        coco_id_to_filename_true,
        coco_id_to_captions_get_image_paths_and_corresponding_captions,
    )
    assert len(image_paths) == 15
    assert len(captions) == 15


def test_flickr_parse_captions_filenames(flickr_texts_path):
    img_path_caption = FlickrDataset.parse_captions_filenames(flickr_texts_path)
    unique_img_paths = set()
    for img_path in img_path_caption.keys():
        assert len(img_path_caption[img_path]) == 5
        unique_img_paths.add(img_path)
    assert len(unique_img_paths) == 5


def test_pascal_parse_captions_filenames(pascal_images_path, pascal_texts_path):
    category_image_path_captions = PascalSentencesDataset.parse_captions_filenames(
        pascal_texts_path, pascal_images_path
    )
    count_cat = 0
    count_files = 0
    count_sentences = 0
    for category in category_image_path_captions:
        count_cat += 1
        for file in category_image_path_captions[category]:
            count_files += 1
            for sentence in category_image_path_captions[category][file]:
                count_sentences += 1

    assert count_cat == 3
    assert count_files == 9
    assert count_sentences == 45
