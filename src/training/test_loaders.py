import pytest
import tensorflow as tf
import numpy as np

from training.loaders import CocoTrainValLoader


@pytest.fixture
def train_image_paths():
    return [
        "data/testing_assets/images_train/COCO_val2014_000000000042.jpg",
        "data/testing_assets/images_train/COCO_val2014_000000000073.jpg",
        "data/testing_assets/images_train/COCO_val2014_000000000074.jpg",
        "data/testing_assets/images_train/COCO_val2014_000000000133.jpg",
        "data/testing_assets/images_train/COCO_val2014_000000000136.jpg",
    ]


@pytest.fixture
def train_captions():
    return [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12], [13, 14, 15]]


@pytest.fixture
def train_captions_lengths():
    return [[4], [2], [5], [1], [3]]


@pytest.fixture
def train_labels():
    return [[1], [2], [3], [1], [2]]


@pytest.fixture
def val_image_paths():
    return [
        "data/testing_assets/images_val/COCO_val2014_000000000042.jpg",
        "data/testing_assets/images_val/COCO_val2014_000000000073.jpg",
        "data/testing_assets/images_val/COCO_val2014_000000000074.jpg",
    ]


@pytest.fixture
def val_captions():
    return [[1, 2], [3, 4, 5], [6]]


@pytest.fixture
def val_captions_lengths():
    return [[2], [3], [1]]


@pytest.fixture
def val_labels():
    return [[1], [2], [3]]


@pytest.fixture
def epochs():
    return 3


@pytest.fixture
def batch_size():
    return 2


def test_loader(
    train_image_paths,
    train_captions,
    train_captions_lengths,
    train_labels,
    val_image_paths,
    val_captions,
    val_captions_lengths,
    val_labels,
    epochs,
    batch_size,
):
    tf.reset_default_graph()
    loader = CocoTrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        train_labels,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        val_labels,
        batch_size,
    )
    images, captions, captions_lengths, labels = loader.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            sess.run(loader.train_init)
            try:
                while True:
                    images_batch, captions_batch, captions_lengths_batch, labels_batch = sess.run(
                        [images, captions, captions_lengths, labels]
                    )
                    _, width, height, _ = images_batch.shape
                    assert width == 224
                    assert height == 224
                    num_tokens_batch = captions_batch.shape[1]
                    for caption in captions_batch:
                        assert caption.shape[0] == num_tokens_batch
                    for caption, length in zip(captions_batch, captions_lengths_batch):
                        assert np.count_nonzero(caption) == length
                    for label in labels_batch:
                        pass
            except tf.errors.OutOfRangeError:
                pass
            sess.run(loader.val_init)
            try:
                while True:
                    images_batch, captions_batch, captions_lengths_batch, labels_batch = sess.run(
                        [images, captions, captions_lengths, labels]
                    )
                    _, width, height, _ = images_batch.shape
                    assert width == 224
                    assert height == 224
                    num_tokens_batch = captions_batch.shape[1]
                    for caption in captions_batch:
                        assert caption.shape[0] == num_tokens_batch
                    for caption, length in zip(captions_batch, captions_lengths_batch):
                        assert np.count_nonzero(caption) == length
                    for label in labels_batch:
                        pass
            except tf.errors.OutOfRangeError:
                pass
