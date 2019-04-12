
import json
import re
import os
import logging
from typing import Dict, Any, List, Tuple, ValuesView

from utils.constants import PAD_ID, UNK_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class CocoDataset:
    # Adapted for working with the Microsoft COCO dataset.
    word_freq: Dict[str, int] = {}
    word2index: Dict[str, int] = {}
    index2word: Dict[int, str] = {}

    def __init__(self, images_path: str, json_path: str, min_unk_sub: int, train: bool):
        """Creates a dataset object.

        Args:
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
            train: Whether the dataset is training dataset.
        """
        json_file = self.read_json(json_path)
        self.min_unk_sub = min_unk_sub
        logger.info("Object variables set...")
        self.id_to_filename = self.parse_image_paths(json_file, images_path)
        self.id_to_captions = self.parse_captions(json_file)
        logger.info("Dictionaries created...")
        if train:
            self.set_up_class_vars(self.id_to_captions.values())
            logger.info("Class variables set...")

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

    @classmethod
    def parse_captions(cls, json_file: Dict[str, Any]) -> Dict[int, List[List[str]]]:
        """Parses the captions metadata from the json file.

        Args:
            json_file: A dict representing the loaded json file.

        Returns:
            A dict that contains the image id and a list with the image captions.

        """
        id_to_captions: Dict[int, List[List[str]]] = {}
        for captions_data in json_file["annotations"]:
            if captions_data["image_id"] not in id_to_captions.keys():
                id_to_captions[captions_data["image_id"]] = []
            id_to_captions[captions_data["image_id"]].append(
                cls.preprocess_caption(captions_data["caption"])
            )

        return id_to_captions

    @staticmethod
    def preprocess_caption(caption: str) -> List[str]:
        """Performs pre-processing of the caption in the following way:

        1. Converts the whole caption to lower case.
        2. Removes all characters which are not letters.

        Args:
            caption: A list of words contained in the caption.

        Returns:

        """
        caption = caption.lower()
        caption = re.sub("[^a-z ]+", "", caption)
        caption_words = caption.split()

        return caption_words

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

    @classmethod
    def set_up_class_vars(cls, captions: ValuesView[List[List[str]]]) -> None:
        """Sets up the class variables word_freq, word2index and index2word.

        1. Computes the word frequencies and sets the class variable with the value.
        The class variable is a dictionary where the key is the word and the value is
        how many times that word occurs in the dataset.
        2. Creates a dict where each word is mapped to an index.
        3. Creates a dict where each index is mapped to a word.

        Args:
            captions: A list of lists of captions.

        Returns:
            None

        """
        index = 2
        cls.word2index = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for captions_list in captions:
            for caption in captions_list:
                for word in caption:
                    if word not in cls.word_freq.keys():
                        cls.word_freq[word] = 0
                    cls.word_freq[word] += 1
                    if word not in cls.word2index.keys():
                        cls.word2index[word] = index
                        index += 1

        cls.index2word = dict(zip(cls.word2index.values(), cls.word2index.keys()))

    @staticmethod
    def get_img_paths_captions_lengths_wrapper(
        id_to_filename, id_to_captions, min_unk_sub
    ):
        """Returns the image paths and captions.

        Because in the dataset there are 5 captions for each image, what the method does
        is create:

        - A list of image paths where each image path is repeated 5 times.
        - A list of lists of word tokens where the number of inner lists is equal to the
        number of image paths.

        Args:
            id_to_filename: Pair id to image filename dict.
            id_to_captions: Pair id to captions dict.
            min_unk_sub: Minimum numbed of times a word has to appear in order to be
            left in the dataset.

        Returns:
            The image paths, the captions and the lengths of the captions.

        """
        assert len(id_to_filename.keys()) == len(id_to_captions.keys())
        image_paths = []
        captions = []
        lengths = []
        for pair_id in id_to_filename.keys():
            for i in range(5):
                image_paths.append(id_to_filename[pair_id])
                indexed_caption = [
                    CocoDataset.word2index[word]
                    if CocoDataset.word_freq[word] > min_unk_sub
                    else 0
                    for word in id_to_captions[pair_id][i]
                ]
                captions.append(indexed_caption)
                lengths.append(len(indexed_caption))

        assert len(image_paths) == len(captions)

        return image_paths, captions, lengths

    def get_img_paths_captions_lengths(
        self
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        image_paths, captions, lengths = self.get_img_paths_captions_lengths_wrapper(
            self.id_to_filename, self.id_to_captions, self.min_unk_sub
        )

        return image_paths, captions, lengths