import tensorflow as tf
from typing import List, Tuple, Generator
import logging
from abc import ABC, abstractmethod

from utils.constants import WIDTH, HEIGHT, NUM_CHANNELS, VGG_MEAN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    def __init__(self, batch_size: int, prefetch_size: int):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

    @staticmethod
    def parse_data(
        image_path: str, caption: List[str]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Adapted: https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
        image_string = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=NUM_CHANNELS)
        image = tf.cast(image, tf.float32)
        smallest_side = 256.0  # Max for VGG19
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)

        scale = tf.cond(
            tf.greater(height, width),
            lambda: smallest_side / width,
            lambda: smallest_side / height,
        )
        new_height = tf.cast(height * scale, tf.float32)
        new_width = tf.cast(width * scale, tf.float32)
        image = tf.image.resize_images(image, [new_height, new_width])

        caption_words = tf.string_split([caption]).values
        caption_len = tf.shape(caption_words)[0]

        return image, caption_words, caption_len

    @staticmethod
    def parse_data_train(
        image: tf.Tensor, caption: tf.Tensor, caption_len: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image = tf.random_crop(image, [WIDTH, HEIGHT, NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means

        return image, caption, caption_len

    @staticmethod
    def parse_data_val_test(
        image: tf.Tensor, caption: tf.Tensor, caption_len: tf.Tensor
    ):
        image = tf.image.resize_image_with_crop_or_pad(image, WIDTH, HEIGHT)
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means

        return image, caption, caption_len

    @abstractmethod
    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        pass


class TrainValLoader(BaseLoader):
    def __init__(
        self,
        train_image_paths: List[str],
        train_captions: List[str],
        val_image_paths: List[str],
        val_captions: List[str],
        batch_size: int,
        prefetch_size: int,
    ):
        super().__init__(batch_size, prefetch_size)
        # Build multi_hop_attention dataset
        self.train_image_paths = train_image_paths
        self.train_captions = train_captions
        self.train_dataset = tf.data.Dataset.from_generator(
            generator=self.train_data_generator,
            output_types=(tf.string, tf.string),
            output_shapes=(None, None),
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=len(self.train_image_paths), reshuffle_each_iteration=True
        )
        self.train_dataset = self.train_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.map(
            self.parse_data_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.padded_batch(
            self.batch_size, padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [])
        )
        self.train_dataset = self.train_dataset.prefetch(self.prefetch_size)
        logger.info("Training dataset created...")

        # Build validation dataset
        self.val_image_paths = val_image_paths
        self.val_captions = val_captions
        self.val_dataset = tf.data.Dataset.from_generator(
            generator=self.val_data_generator,
            output_types=(tf.string, tf.string),
            output_shapes=(None, None),
        )
        self.val_dataset = self.val_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.val_dataset = self.val_dataset.map(
            self.parse_data_val_test, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.val_dataset = self.val_dataset.padded_batch(
            self.batch_size, padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [])
        )
        self.val_dataset = self.val_dataset.prefetch(self.prefetch_size)
        logger.info("Validation dataset created...")

        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes
        )

        # Initialize with required datasets
        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init = self.iterator.make_initializer(self.val_dataset)

        logger.info("Iterator created...")

    def train_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption in zip(self.train_image_paths, self.train_captions):
            yield image_path, caption

    def val_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption in zip(self.val_image_paths, self.val_captions):
            yield image_path, caption

    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        images, captions, captions_lengths = self.iterator.get_next()

        return images, captions, captions_lengths


class InferenceLoader(BaseLoader):
    def __init__(
        self,
        test_image_paths: List[str],
        test_captions: List[str],
        batch_size: int,
        prefetch_size: int,
    ):
        super().__init__(batch_size, prefetch_size)
        self.test_image_paths = test_image_paths
        self.test_captions = test_captions

        self.test_dataset = tf.data.Dataset.from_generator(
            generator=self.test_data_generator,
            output_types=(tf.string, tf.string),
            output_shapes=(None, None),
        )
        self.test_dataset = self.test_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.test_dataset = self.test_dataset.map(
            self.parse_data_val_test, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.test_dataset = self.test_dataset.padded_batch(
            self.batch_size, padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [])
        )
        self.test_dataset = self.test_dataset.prefetch(self.prefetch_size)
        logger.info("Test dataset created...")

        self.iterator = self.test_dataset.make_one_shot_iterator()
        logger.info("Iterator created...")

    def test_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption in zip(self.test_image_paths, self.test_captions):
            yield image_path, caption

    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        images, captions, captions_lengths = self.iterator.get_next()

        return images, captions, captions_lengths
