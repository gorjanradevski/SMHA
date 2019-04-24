import tensorflow as tf
from typing import List, Tuple, Generator
import logging

from utils.constants import WIDTH, HEIGHT, NUM_CHANNELS, VGG_MEAN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainValLoader:
    def __init__(
        self,
        train_image_paths: List[str],
        train_captions: List[List[int]],
        train_captions_lengths: List[List[int]],
        train_labels: List[List[int]],
        val_image_paths: List[str],
        val_captions: List[List[int]],
        val_captions_lengths: List[List[int]],
        val_labels: List[List[int]],
        batch_size: int,
        prefetch_size: int,
    ):
        # Build training dataset
        self.train_image_paths = train_image_paths
        self.train_captions = train_captions
        self.train_captions_lengths = train_captions_lengths
        self.train_labels = train_labels
        self.train_dataset = tf.data.Dataset.from_generator(
            generator=self.train_data_generator,
            output_types=(tf.string, tf.int32, tf.int32, tf.int32),
            output_shapes=(None, None, None, None),
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=len(self.train_image_paths)
        )
        self.train_dataset = self.train_dataset.map(
            self.parse_data_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.padded_batch(
            batch_size,
            padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [None], [None]),
        )
        self.train_dataset = self.train_dataset.prefetch(prefetch_size)
        logger.info("Training dataset created...")

        # Build validation dataset
        self.val_image_paths = val_image_paths
        self.val_captions = val_captions
        self.val_captions_lengths = val_captions_lengths
        self.val_labels = val_labels
        self.val_dataset = tf.data.Dataset.from_generator(
            generator=self.val_data_generator,
            output_types=(tf.string, tf.int32, tf.int32, tf.int32),
            output_shapes=(None, None, None, None),
        )
        self.val_dataset = self.val_dataset.map(
            self.parse_data_val, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.val_dataset = self.val_dataset.padded_batch(
            batch_size,
            padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [None], [None]),
        )
        self.val_dataset = self.val_dataset.prefetch(prefetch_size)
        logger.info("Validation dataset created...")

        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes
        )

        # Initialize with required datasets
        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init = self.iterator.make_initializer(self.val_dataset)

        logger.info("Iterator created...")

    @staticmethod
    def parse_data_train(
        image_path: str, caption: tf.Tensor, caption_len: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Parse image
        image_string = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=NUM_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, WIDTH, HEIGHT)
        image = tf.image.random_flip_left_right(image)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means

        return image, caption, caption_len, label

    @staticmethod
    def parse_data_val(
        image_path: str, caption: tf.Tensor, caption_len: tf.Tensor, label: tf.Tensor
    ):
        image_string = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=NUM_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, WIDTH, HEIGHT)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means

        return image, caption, caption_len, label

    def train_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption, caption_len, label in zip(
            self.train_image_paths,
            self.train_captions,
            self.train_captions_lengths,
            self.train_labels,
        ):
            yield image_path, caption, caption_len, label

    def val_data_generator(self) -> Generator[tf.Tensor, None, None]:
        for image_path, caption, caption_len, label in zip(
            self.val_image_paths,
            self.val_captions,
            self.val_captions_lengths,
            self.val_labels,
        ):
            yield image_path, caption, caption_len, label

    def get_next(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        images, captions, captions_lengths, labels = self.iterator.get_next()
        # Squeeze the redundant dimension of captions_lengths and labels because they
        # were added just so the padded_batch function will work
        captions_lengths = tf.squeeze(captions_lengths, axis=1)
        labels = tf.squeeze(labels, axis=1)

        return images, captions, captions_lengths, labels
