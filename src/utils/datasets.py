import json
import re
import os
import logging
from abc import ABC
from typing import Dict, Any, List, Tuple

from utils.constants import pascal_train_size, pascal_val_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_caption(caption: str) -> str:
    """Basic method used around all classes

    Performs pre-processing of the caption in the following way:

    1. Converts the whole caption to lower case.
    2. Removes all characters which are not letters.

    Args:
        caption: A list of words contained in the caption.

    Returns:

    """
    caption = caption.lower()
    caption = re.sub("[^a-z' ]+", "", caption)
    caption = re.sub("\s+", " ", caption).strip()  # NOQA
    caption = caption.strip()

    return caption


class BaseCocoDataset(ABC):

    # Adapted for working with the Microsoft COCO dataset.
    def __init__(self, images_path: str, json_path: str):
        """Creates a dataset object.

        Args:
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
        """
        json_file = self.read_json(json_path)
        self.id_to_filename = self.parse_image_paths(json_file, images_path)
        self.id_to_captions = self.parse_captions(json_file)
        logger.info("Object variables set...")

    @staticmethod
    def parse_image_paths(
        json_file: Dict[str, Any], images_path: str
    ) -> Dict[int, str]:
        """Parses the images metadata from the json file.

        Args:
            json_file: A dict representing the loaded json file.
            images_path: A path where the images are.

        Returns:
            A dict that contains the image id and the image filename.

        """
        id_to_filename = {}
        for image_data in json_file["images"]:
            id_to_filename[image_data["id"]] = os.path.join(
                images_path, image_data["file_name"]
            )

        return id_to_filename

    @staticmethod
    def parse_captions(json_file: Dict[str, Any]) -> Dict[int, List[str]]:
        """Parses the captions metadata from the json file.

        Args:
            json_file: A dict representing the loaded json file.

        Returns:
            A dict that contains the image id and a list with the image captions.

        """
        id_to_captions: Dict[int, List[str]] = {}
        for captions_data in json_file["annotations"]:
            if captions_data["image_id"] not in id_to_captions.keys():
                id_to_captions[captions_data["image_id"]] = []
            id_to_captions[captions_data["image_id"]].append(
                preprocess_caption(captions_data["caption"])
            )

        return id_to_captions

    @staticmethod
    def read_json(json_path: str) -> Dict[str, Any]:
        """Reads json file given a path.

        Args:
            json_path: Path where the json file is.

        Returns:
            A dictionary representing the json file.

        """
        with open(json_path) as file:
            json_file = json.load(file)

        return json_file

    @staticmethod
    def get_data_wrapper(
        id_to_filename: Dict[int, str], id_to_captions: Dict[int, List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Returns the image paths and captions.

        Because in the dataset there are 5 captions for each image, what the method does
        is create:

        - A list of image paths where each image path is repeated 5 times.
        - A list of lists of word tokens where the number of inner lists is equal to the
        number of image paths.

        Args:
            id_to_filename: Pair id to image filename dict.
            id_to_captions: Pair id to captions dict.

        Returns:
            The image paths, the captions and the lengths of the captions.

        """
        assert len(id_to_filename.keys()) == len(id_to_captions.keys())
        image_paths = []
        captions = []
        for pair_id in id_to_filename.keys():
            for i in range(5):
                image_paths.append(id_to_filename[pair_id])
                captions.append(id_to_captions[pair_id][i])

        assert len(image_paths) == len(captions)

        return image_paths, captions

    def get_data(self):
        image_paths, captions = self.get_data_wrapper(
            self.id_to_filename, self.id_to_captions
        )

        return image_paths, captions


class TrainCocoDataset(BaseCocoDataset):
    # Adapted for working with the Microsoft COCO dataset.

    def __init__(self, images_path: str, json_path: str):
        """Creates a dataset object.

        Args:
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
        """
        super().__init__(images_path, json_path)
        logger.info("Class variables set...")


class ValCocoDataset(BaseCocoDataset):
    # Adapted for working with the Microsoft COCO dataset.

    def __init__(self, images_path: str, json_path: str, val_size: int = None):
        """Creates a dataset object.

        Args:
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
            val_size: The size of the validation set.
        """
        super().__init__(images_path, json_path)
        self.val_size = val_size


class FlickrDataset:
    # Adapted for working with the Flickr8k and Flickr30k dataset.

    def __init__(self, images_path: str, texts_path: str):
        self.img_path_caption = self.parse_captions_filenames(texts_path)
        self.images_path = images_path
        logger.info("Object variables set...")

    @staticmethod
    def parse_captions_filenames(texts_path: str) -> Dict[str, List[str]]:
        """Creates a dictionary that holds:

        Key: The full path to the image.
        Value: A list of lists where each token in the inner list is a word. The number
        of sublists is 5.

        Args:
            texts_path: Path where the text doc with the descriptions is.

        Returns:
            A dictionary that represents what is explained above.

        """
        img_path_caption: Dict[str, List[str]] = {}
        with open(texts_path, "r") as file:
            for line in file:
                line_parts = line.split("\t")
                image_tag = line_parts[0].partition("#")[0]
                caption = line_parts[1]
                if image_tag not in img_path_caption:
                    img_path_caption[image_tag] = []
                img_path_caption[image_tag].append(preprocess_caption(caption))

        return img_path_caption

    @staticmethod
    def get_data_wrapper(
        imgs_file_path: str,
        img_path_caption: Dict[str, List[str]],
        images_dir_path: str,
    ):
        """Returns the image paths, the captions and the lengths of the captions.

        Args:
            imgs_file_path: A path to a file where all the images belonging to the
            validation part of the dataset are listed.
            img_path_caption: Image name to list of captions dict.
            images_dir_path: A path where all the images are located.

        Returns:
            Image paths, captions and lengths.

        """
        image_paths = []
        captions = []
        with open(imgs_file_path, "r") as file:
            for image_name in file:
                # Remove the newline character at the end
                image_name = image_name[:-1]
                # If there is no specified codec in the name of the image append jpg
                if not image_name.endswith(".jpg"):
                    image_name += ".jpg"
                for i in range(5):
                    image_paths.append(os.path.join(images_dir_path, image_name))
                    captions.append(img_path_caption[image_name][i])

        assert len(image_paths) == len(captions)

        return image_paths, captions

    def get_data(self, images_file_path: str):
        image_paths, captions = self.get_data_wrapper(
            images_file_path, self.img_path_caption, self.images_path
        )

        return image_paths, captions


class PascalSentencesDataset:
    # Adapted for working with the Pascal sentences dataset.

    def __init__(self, images_path, texts_path):
        self.category_image_path_captions = self.parse_captions_filenames(
            texts_path, images_path
        )

    @staticmethod
    def parse_captions_filenames(
        texts_path: str, images_path: str
    ) -> Dict[str, Dict[str, List[str]]]:
        """Creates a dictionary of dictionaries where:

        1. The keys of the first dict are the different categories of data.
        2. The keys of the second dict are the image paths for the corresponding
        category.
        3. The values of the of second dict are a list of list where each list holds the
        5 different captions for the image path, and each sublist holds the indexed
        words of the caption.

        Args:
            texts_path: Path where the image captions are.
            images_path: Path where the images are.

        Returns:
            A dictionary as explained above.

        """
        category_image_path_captions: Dict[str, Dict[str, List[str]]] = dict(dict())
        for category in os.listdir(texts_path):
            file_path = os.path.join(texts_path, category)
            if os.path.isdir(file_path):
                if category not in category_image_path_captions:
                    category_image_path_captions[category] = {}
                for txt_file in os.listdir(file_path):
                    if txt_file.endswith(".txt"):
                        image_path = os.path.join(
                            images_path, category, txt_file[:-3] + "jpg"
                        )
                        if image_path not in category_image_path_captions[category]:
                            category_image_path_captions[category][image_path] = []
                        txt_file_path = os.path.join(file_path, txt_file)
                        with open(txt_file_path, "r") as f:
                            for caption in f:
                                category_image_path_captions[category][
                                    image_path
                                ].append(preprocess_caption(caption))

        return category_image_path_captions

    @staticmethod
    def get_data_wrapper(category_image_path_captions, data_type: str):
        """Returns the image paths, the captions and the captions lengths.

        Args:
            category_image_path_captions: A really compex dict :(
            data_type: The type of the data that is returned (Train, val or test).

        Returns:
            The image paths, the captions and the captions lengths.

        """
        image_paths = []
        captions = []
        train_size = pascal_train_size * 50
        val_size = pascal_val_size * 50
        for category in category_image_path_captions.keys():
            for v, image_path in enumerate(
                category_image_path_captions[category].keys()
            ):
                for caption in category_image_path_captions[category][image_path]:
                    if data_type == "train":
                        if v < train_size:
                            image_paths.append(image_path)
                            captions.append(caption)
                    elif data_type == "val":
                        if train_size + val_size > v >= train_size:
                            image_paths.append(image_path)
                            captions.append(caption)
                    elif data_type == "test":
                        if v >= train_size + val_size:
                            image_paths.append(image_path)
                            captions.append(caption)
                    else:
                        raise ValueError("Wrong data type!")

        return image_paths, captions

    def get_train_data(self):
        img_paths, cap = self.get_data_wrapper(
            self.category_image_path_captions, "train"
        )

        return img_paths, cap

    def get_val_data(self):
        img_paths, cap = self.get_data_wrapper(self.category_image_path_captions, "val")

        return img_paths, cap

    def get_test_data(self):
        img_paths, cap = self.get_data_wrapper(
            self.category_image_path_captions, "test"
        )

        return img_paths, cap
