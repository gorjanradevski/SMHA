import tensorflow as tf
from typing import List, Tuple, Generator

from utils.constants import WIDTH, HEIGHT, BATCH_SIZE, NUM_CHANNELS


class TrainValLoader:
    def __init__(
        self,
        train_image_paths: List[str],
        train_captions: List[List[int]],
        train_captions_lengths: List[int],
        val_image_paths: List[str],
        val_captions: List[List[int]],
        val_captions_lengths: List[int],
    ):
        # Build training dataset
        self.train_image_paths = train_image_paths
        self.train_captions = train_captions
        self.train_captions_lengths = train_captions_lengths
        self.train_dataset = tf.data.Dataset.from_generator(
            generator=self.train_data_generator,
            output_types=(tf.string, tf.int32, tf.int32),
            output_shapes=(None, None, None),
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=len(self.train_image_paths)
        )
        self.train_dataset = self.train_dataset.map(self.parse_data)
        self.train_dataset = self.train_dataset.padded_batch(
            BATCH_SIZE, padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [None])
        )
        self.train_dataset = self.train_dataset.prefetch(1)

        # Build validation dataset
        self.val_image_paths = val_image_paths
        self.val_captions = val_captions
        self.val_captions_lengths = val_captions_lengths
        self.val_dataset = tf.data.Dataset.from_generator(
            generator=self.val_data_generator,
            output_types=(tf.string, tf.int32, tf.int32),
            output_shapes=(None, None, None),
        )
        self.val_dataset = self.val_dataset(self.parse_data)
        self.val_dataset = self.val_dataset.padded_batch(
            BATCH_SIZE, padded_shapes=([WIDTH, HEIGHT, NUM_CHANNELS], [None], [None])
        )

        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes
        )

        # Initialize with required Datasets
        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init = self.iterator.make_initializer(self.val_dataset)

    @staticmethod
    def parse_data(
        image_path, caption, caption_len
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Parse image
        image_string = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=NUM_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [WIDTH, HEIGHT])

        return image, caption, caption_len

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

        return images, captions, captions_lengths
