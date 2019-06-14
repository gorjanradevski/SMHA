import tensorflow as tf
from typing import List, Tuple, Generator
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    def __init__(self, batch_size: int, prefetch_size: int):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

    @staticmethod
    def parse_data(
        image_path: str, caption: tf.Tensor, caption_len: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image_string = tf.read_file(image_path)

        return image_string, caption, caption_len

    @abstractmethod
    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        pass


class TrainValLoader(BaseLoader):
    def __init__(
        self,
        train_image_paths: List[str],
        train_captions: List[List[int]],
        train_captions_lengths: List[List[int]],
        val_image_paths: List[str],
        val_captions: List[List[int]],
        val_captions_lengths: List[List[int]],
        batch_size: int,
        prefetch_size: int,
    ):
        super().__init__(batch_size, prefetch_size)
        # Build multi_hop_attention dataset
        self.train_image_paths = train_image_paths
        self.train_captions = train_captions
        self.train_captions_lengths = train_captions_lengths
        self.train_dataset = tf.data.Dataset.from_generator(
            generator=self.train_data_generator,
            output_types=(tf.string, tf.int32, tf.int32),
            output_shapes=(None, None, None),
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=len(self.train_image_paths), reshuffle_each_iteration=True
        )
        self.train_dataset = self.train_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.padded_batch(
            self.batch_size, padded_shapes=([], [None], [None])
        )
        self.train_dataset = self.train_dataset.prefetch(self.prefetch_size)
        logger.info("Training dataset created...")

        # Build validation dataset
        self.val_image_paths = val_image_paths
        self.val_captions = val_captions
        self.val_captions_lengths = val_captions_lengths
        self.val_dataset = tf.data.Dataset.from_generator(
            generator=self.val_data_generator,
            output_types=(tf.string, tf.int32, tf.int32),
            output_shapes=(None, None, None),
        )
        self.val_dataset = self.val_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.val_dataset = self.val_dataset.padded_batch(
            self.batch_size, padded_shapes=([], [None], [None])
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
        for image_path, caption, caption_len in zip(
            self.train_image_paths, self.train_captions, self.train_captions_lengths
        ):
            yield image_path, caption, caption_len

    def val_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption, caption_len in zip(
            self.val_image_paths, self.val_captions, self.val_captions_lengths
        ):
            yield image_path, caption, caption_len

    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        images, captions, captions_lengths = self.iterator.get_next()
        # Squeeze the redundant dimension of captions_lengths because they were added
        # just so the padded_batch function will work
        captions_lengths = tf.squeeze(captions_lengths, axis=1)

        return images, captions, captions_lengths


class InferenceLoader(BaseLoader):
    def __init__(
        self,
        test_image_paths: List[str],
        test_captions: List[List[int]],
        test_captions_lengths: List[List[int]],
        batch_size: int,
        prefetch_size: int,
    ):
        super().__init__(batch_size, prefetch_size)
        self.test_image_paths = test_image_paths
        self.test_captions = test_captions
        self.test_captions_lengths = test_captions_lengths

        self.test_dataset = tf.data.Dataset.from_generator(
            generator=self.test_data_generator,
            output_types=(tf.string, tf.int32, tf.int32),
            output_shapes=(None, None, None),
        )
        self.test_dataset = self.test_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.test_dataset = self.test_dataset.padded_batch(
            self.batch_size, padded_shapes=([], [None], [None])
        )
        self.test_dataset = self.test_dataset.prefetch(self.prefetch_size)
        logger.info("Test dataset created...")

        self.iterator = self.test_dataset.make_one_shot_iterator()
        logger.info("Iterator created...")

    def test_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption, caption_len in zip(
            self.test_image_paths, self.test_captions, self.test_captions_lengths
        ):
            yield image_path, caption, caption_len

    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        images, captions, captions_lengths = self.iterator.get_next()
        # Squeeze the redundant dimension of captions_lengths because they were added
        # just so the padded_batch function will work
        captions_lengths = tf.squeeze(captions_lengths, axis=1)

        return images, captions, captions_lengths
