# Adapted for working with the Microsoft COCO dataset.

import json
import re
import os
import logging
from typing import Dict, Any, List, Tuple, ValuesView

from utils.constants import PAD_ID, UNK_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Dataset:

    word_freq: Dict[str, int] = {}
    word2index: Dict[str, int] = {}
    index2word: Dict[int, str] = {}

    def __init__(self, train: bool, images_path: str, json_path: str, min_unk_sub: int):
        """Creates a dataset object.

        Args:
            train: Whether the dataset is training dataset.
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
        """
        self.images_path = images_path
        self.json_path = json_path
        self.json_file = self.read_json(json_path)
        self.min_unk_sub = min_unk_sub
        logger.info("Object variables set")
        self.id_to_filename = self.parse_image_paths(self.json_file)
        self.id_to_captions = self.parse_captions(self.json_file)
        logger.info("Dictionaries created")
        if train:
            self.set_up_class_vars(self.id_to_captions.values())
            logger.info("Class variables set")

    def parse_image_paths(self, json_file: Dict[str, Any]) -> Dict[int, str]:
        """Parses the images metadata from the json file.

        Args:
            json_file: A dict representing the loaded json file.

        Returns:
            A dict that contains the image id and the image filename.

        """
        id_to_filename = {}
        for image_data in json_file["images"]:
            id_to_filename[image_data["id"]] = os.path.join(
                self.images_path, image_data["file_name"]
            )

        return id_to_filename

    def parse_captions(self, json_file: Dict[str, Any]) -> Dict[int, List[str]]:
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
                self.preprocess_caption(captions_data["caption"])
            )

        return id_to_captions

    @staticmethod
    def preprocess_caption(caption: str) -> str:
        """Performs pre-processing of the caption in the following way:

        1. Converts the whole caption to lower case.
        2. Removes all characters which are not letters.

        Args:
            caption: A caption for an image

        Returns:

        """
        caption = caption.lower()
        caption = re.sub("[^a-z]+", "", caption)

        return caption

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
    def set_up_class_vars(cls, captions: ValuesView[List[str]]) -> None:
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
        word2index = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        word_freq: Dict[str, int] = {}
        for captions_list in captions:
            for caption in captions_list:
                words = caption.split(" ")
                for word in words:
                    if word not in word_freq.keys():
                        word_freq[word] = 0
                    word_freq[word] += 1
                    if word not in word2index.keys():
                        word2index[word] = index
                        index += 1

        cls.word_freq = word_freq
        cls.index2word = dict(zip(word2index.values(), word2index.keys()))
        cls.word2index = word2index

    def get_image_paths_and_corresponding_captions(
        self
    ) -> Tuple[List[str], List[List[int]]]:
        # Prototype version (Get only the first caption)
        assert len(self.id_to_filename.keys()) == len(self.id_to_captions.keys())
        image_paths = []
        captions = []
        for pair_id in self.id_to_filename.keys():
            image_paths.append(self.id_to_filename[pair_id])
            indexed_caption = [
                Dataset.word2index[word]
                if Dataset.word_freq[word] > self.min_unk_sub
                else 0
                for word in self.id_to_captions[pair_id][0].split()
            ]
            captions.append(indexed_caption)

        return image_paths, captions
