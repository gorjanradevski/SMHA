import json
import re
import os
import logging
from abc import ABC
from typing import Dict, Any, List, Tuple, ValuesView

from utils.constants import PAD_ID, UNK_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_caption(caption: str) -> List[str]:
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
    caption_words = caption.split()

    return caption_words


def get_vocab_size(caller_class) -> int:
    """Returns the size of the vocabulary.

    Args:
        caller_class: The class which is calling the method.

    Returns:
        The size of the vocabulary.

    """
    return len(caller_class.word2index)


class BaseCocoDataset(ABC):

    # Adapted for working with the Microsoft COCO dataset.
    word_freq: Dict[str, int] = {}
    word2index: Dict[str, int] = {}
    index2word: Dict[int, str] = {}

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
    def parse_captions(json_file: Dict[str, Any]) -> Dict[int, List[List[str]]]:
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

    @classmethod
    def set_up_class_vars(
        cls, captions: ValuesView[List[List[str]]], min_unk_sub: int
    ) -> None:
        """Sets up the class variables word_freq, word2index and index2word.

        1. Computes the word frequencies and sets the class variable with the value.
        The class variable is a dictionary where the key is the word and the value is
        how many times that word occurs in the dataset.
        2. Creates a dict where each word is mapped to an index.
        3. Creates a dict where each index is mapped to a word.

        Args:
            captions: A list of lists of captions.
            min_unk_sub: The minimum frequency a word has to have in order to be left
            in the vocabulary.

        Returns:
            None

        """
        index = 2
        BaseCocoDataset.word2index = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for captions_list in captions:
            for caption in captions_list:
                for word in caption:
                    if word not in BaseCocoDataset.word_freq.keys():
                        BaseCocoDataset.word_freq[word] = 0
                    BaseCocoDataset.word_freq[word] += 1
        for captions_list in captions:
            for caption in captions_list:
                for word in caption:
                    if (
                        word not in BaseCocoDataset.word2index.keys()
                        and BaseCocoDataset.word_freq[word] >= min_unk_sub
                    ):
                        BaseCocoDataset.word2index[word] = index
                        index += 1

        BaseCocoDataset.index2word = dict(
            zip(BaseCocoDataset.word2index.values(), BaseCocoDataset.word2index.keys())
        )

    @staticmethod
    def get_data_wrapper(
        id_to_filename: Dict[int, str], id_to_captions: Dict[int, List[List[str]]]
    ) -> Tuple[List[str], List[List[int]], List[List[int]]]:
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
        lengths = []
        for pair_id in id_to_filename.keys():
            for i in range(5):
                image_paths.append(id_to_filename[pair_id])
                # Must wrap with a list so that the rank will be the same as the
                # captions
                indexed_caption = [
                    BaseCocoDataset.word2index[word]
                    if word in BaseCocoDataset.word2index.keys()
                    else 1
                    for word in id_to_captions[pair_id][i]
                ]
                captions.append(indexed_caption)
                # Must wrap with a list so that the rank will be the same as the
                # captions
                lengths.append([len(indexed_caption)])

        assert len(image_paths) == len(captions)
        assert len(image_paths) == len(lengths)

        return image_paths, captions, lengths

    def get_data(self):
        image_paths, captions, lengths = self.get_data_wrapper(
            self.id_to_filename, self.id_to_captions
        )

        return image_paths, captions, lengths


class TrainCocoDataset(BaseCocoDataset):
    # Adapted for working with the Microsoft COCO dataset.

    def __init__(self, images_path: str, json_path: str, min_unk_sub: int):
        """Creates a dataset object.

        Args:
            images_path: Path where the images are located.
            json_path: Path to the json file where the mappings are indicated as well
            as the captions.
        """
        super().__init__(images_path, json_path)
        self.set_up_class_vars(self.id_to_captions.values(), min_unk_sub)
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
    word_freq: Dict[str, int] = {}
    word2index: Dict[str, int] = {}
    index2word: Dict[int, str] = {}

    def __init__(self, images_path: str, texts_path: str, min_unk_sub: int):
        self.img_path_caption = self.parse_captions_filenames(texts_path)
        self.images_path = images_path
        logger.info("Object variables set...")
        self.set_up_class_vars(self.img_path_caption.values(), min_unk_sub)
        logger.info("Class variables set...")

    @staticmethod
    def parse_captions_filenames(texts_path: str) -> Dict[str, List[List[str]]]:
        """Creates a dictionary that holds:

        Key: The full path to the image.
        Value: A list of lists where each token in the inner list is a word. The number
        of sublists is 5.

        Args:
            texts_path: Path where the text doc with the descriptions is.

        Returns:
            A dictionary that represents what is explained above.

        """
        img_path_caption: Dict[str, List[List[str]]] = {}
        with open(texts_path, "r") as file:
            for line in file:
                line_parts = line.split("\t")
                image_tag = line_parts[0].partition("#")[0]
                caption = line_parts[1]
                if image_tag not in img_path_caption:
                    img_path_caption[image_tag] = []
                img_path_caption[image_tag].append(preprocess_caption(caption))

        return img_path_caption

    @classmethod
    def set_up_class_vars(
        cls, captions: ValuesView[List[List[str]]], min_unk_sub: int
    ) -> None:
        """Sets up the class variables word_freq, word2index and index2word.

        1. Computes the word frequencies and sets the class variable with the values.
        The class variable is a dictionary where the key is the word and the value is
        how many times that word occurs in the dataset.
        2. Creates a dict where each word is mapped to an index.
        3. Creates a dict where each index is mapped to a word.

        Args:
            captions: A list of lists of captions.
            min_unk_sub: The minimum frequency a word has to have in order to be left
            in the vocabulary.

        Returns:
            None

        """
        for captions_list in captions:
            for caption in captions_list:
                for word in caption:
                    if word not in FlickrDataset.word_freq.keys():
                        FlickrDataset.word_freq[word] = 0
                    FlickrDataset.word_freq[word] += 1
        index = 2
        FlickrDataset.word2index = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for captions_list in captions:
            for caption in captions_list:
                for word in caption:
                    if (
                        word not in FlickrDataset.word2index.keys()
                        and FlickrDataset.word_freq[word] >= min_unk_sub
                    ):
                        FlickrDataset.word2index[word] = index
                        index += 1

        FlickrDataset.index2word = dict(
            zip(FlickrDataset.word2index.values(), FlickrDataset.word2index.keys())
        )

    @staticmethod
    def get_data_wrapper(
        imgs_file_path: str,
        img_path_caption: Dict[str, List[List[str]]],
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
        lengths = []
        with open(imgs_file_path, "r") as file:
            for image_name in file:
                # Remove the newline character at the end
                image_name = image_name[:-1]
                # If there is no specified codec in the name of the image append jpg
                if not image_name.endswith(".jpg"):
                    image_name += ".jpg"
                for i in range(5):
                    image_paths.append(os.path.join(images_dir_path, image_name))
                    # Must wrap with a list so that the rank will be the same as the
                    # captions
                    indexed_caption = [
                        FlickrDataset.word2index[word]
                        if word in FlickrDataset.word2index.keys()
                        else 1
                        for word in img_path_caption[image_name][i]
                    ]
                    captions.append(indexed_caption)
                    # Must wrap with a list so that the rank will be the same as the
                    # captions
                    lengths.append([len(indexed_caption)])

        assert len(image_paths) == len(captions)
        assert len(image_paths) == len(lengths)

        return image_paths, captions, lengths

    def get_data(self, images_file_path: str):
        image_paths, captions, lengths = self.get_data_wrapper(
            images_file_path, self.img_path_caption, self.images_path
        )

        return image_paths, captions, lengths


class PascalSentencesDataset:

    # Adapted for working with the Pascal sentences dataset.
    word_freq: Dict[str, int] = {}
    word2index: Dict[str, int] = {}
    index2word: Dict[int, str] = {}

    def __init__(self, images_path, texts_path, min_unk_sub):
        self.category_image_path_captions = self.parse_captions_filenames(
            texts_path, images_path
        )
        self.set_up_class_vars(self.category_image_path_captions, min_unk_sub)

    @staticmethod
    def parse_captions_filenames(
        texts_path: str, images_path: str
    ) -> Dict[str, Dict[str, List[List[str]]]]:
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
        category_image_path_captions: Dict[str, Dict[str, List[List[str]]]] = dict(
            dict()
        )
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

    @classmethod
    def set_up_class_vars(
        cls,
        category_image_path_captions: Dict[str, Dict[str, List[List[str]]]],
        min_unk_sub: int,
    ) -> None:
        """Sets up the class variables word_freq, word2index and index2word.

        1. Computes the word frequencies and sets the class variable with the values.
        The class variable is a dictionary where the key is the word and the value is
        how many times that word occurs in the dataset.
        2. Creates a dict where each word is mapped to an index.
        3. Creates a dict where each index is mapped to a word.

        Args:
            category_image_path_captions: A really really complex dictionary :(
            min_unk_sub: The minimum frequency a word has to have in order to be left
            in the vocabulary.

        Returns:
            None

        """
        for category in category_image_path_captions.keys():
            for file in category_image_path_captions[category].keys():
                for caption in category_image_path_captions[category][file]:
                    for word in caption:
                        if word not in PascalSentencesDataset.word_freq.keys():
                            PascalSentencesDataset.word_freq[word] = 0
                        PascalSentencesDataset.word_freq[word] += 1
        index = 2
        PascalSentencesDataset.word2index = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for category in category_image_path_captions.keys():
            for file in category_image_path_captions[category].keys():
                for caption in category_image_path_captions[category][file]:
                    for word in caption:
                        if (
                            word not in PascalSentencesDataset.word2index.keys()
                            and PascalSentencesDataset.word_freq[word] >= min_unk_sub
                        ):
                            PascalSentencesDataset.word2index[word] = index
                            index += 1

        PascalSentencesDataset.index2word = dict(
            zip(
                PascalSentencesDataset.word2index.values(),
                PascalSentencesDataset.word2index.keys(),
            )
        )

    @staticmethod
    def get_data_wrapper(
        category_image_path_captions, data_size: float, data_type: str
    ):
        """Returns the image paths, the captions and the captions lengths.

        Args:
            category_image_path_captions: A really compex dict :(
            data_size: The size of the data part.
            data_type: The type of the data that is returned (Train, val or test).

        Returns:
            The image paths, the captions and the captions lengths.

        """
        image_paths = []
        captions = []
        lengths = []
        data_size = data_size * 50
        for category in category_image_path_captions.keys():
            for v, image_path in enumerate(
                category_image_path_captions[category].keys()
            ):
                for caption in category_image_path_captions[category][image_path]:
                    indexed_caption = [
                        PascalSentencesDataset.word2index[word]
                        if word in PascalSentencesDataset.word2index.keys()
                        else 1
                        for word in caption
                    ]
                    if data_type == "train":
                        if v < data_size:
                            image_paths.append(image_path)
                            captions.append(indexed_caption)
                            lengths.append([len(caption)])
                    elif data_type == "val":
                        if v >= data_size:
                            image_paths.append(image_path)
                            captions.append(indexed_caption)
                            lengths.append([len(caption)])

        return image_paths, captions, lengths

    def get_train_data(self, size):
        img_paths, cap, lengths = self.get_data_wrapper(
            self.category_image_path_captions, size, "train"
        )

        return img_paths, cap, lengths

    def get_val_data(self, size):
        img_paths, cap, lengths = self.get_data_wrapper(
            self.category_image_path_captions, size, "val"
        )

        return img_paths, cap, lengths
