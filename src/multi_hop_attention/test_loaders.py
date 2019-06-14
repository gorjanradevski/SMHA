import pytest
import tensorflow as tf
import numpy as np

from multi_hop_attention.loaders import TrainValLoader, InferenceLoader


@pytest.fixture
def train_image_paths():
    return [
        "data/testing_assets/coco_images_train/COCO_val2014_000000000042.jpg",
        "data/testing_assets/coco_images_train/COCO_val2014_000000000073.jpg",
        "data/testing_assets/coco_images_train/COCO_val2014_000000000074.jpg",
        "data/testing_assets/coco_images_train/COCO_val2014_000000000133.jpg",
        "data/testing_assets/coco_images_train/COCO_val2014_000000000136.jpg",
    ]


@pytest.fixture
def train_captions():
    return [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12], [13, 14, 15]]


@pytest.fixture
def train_captions_lengths():
    return [[4], [2], [5], [1], [3]]


@pytest.fixture
def val_image_paths():
    return [
        "data/testing_assets/coco_images_val/COCO_val2014_000000000042.jpg",
        "data/testing_assets/coco_images_val/COCO_val2014_000000000073.jpg",
        "data/testing_assets/coco_images_val/COCO_val2014_000000000074.jpg",
    ]


@pytest.fixture
def val_captions():
    return [[1, 2], [3, 4, 5], [6]]


@pytest.fixture
def val_captions_lengths():
    return [[2], [3], [1]]


@pytest.fixture
def epochs():
    return 3


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def prefetch_size():
    return 2


def test_train_val_loader_shapes(
    train_image_paths,
    train_captions,
    train_captions_lengths,
    val_image_paths,
    val_captions,
    val_captions_lengths,
    epochs,
    batch_size,
    prefetch_size,
):
    tf.reset_default_graph()
    loader = TrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            sess.run(loader.train_init)
            try:
                while True:
                    images_batch, captions_batch, captions_lengths_batch = sess.run(
                        [images, captions, captions_lengths]
                    )
                    _, width, height, _ = images_batch.shape
                    assert width == 224
                    assert height == 224
                    num_tokens_batch = captions_batch.shape[1]
                    for caption in captions_batch:
                        assert caption.shape[0] == num_tokens_batch
                    for caption, length in zip(captions_batch, captions_lengths_batch):
                        assert np.count_nonzero(caption) == length
            except tf.errors.OutOfRangeError:
                pass
            sess.run(loader.val_init)
            try:
                while True:
                    images_batch, captions_batch, captions_lengths_batch = sess.run(
                        [images, captions, captions_lengths]
                    )
                    _, width, height, _ = images_batch.shape
                    assert width == 224
                    assert height == 224
                    num_tokens_batch = captions_batch.shape[1]
                    for caption in captions_batch:
                        assert caption.shape[0] == num_tokens_batch
                    for caption, length in zip(captions_batch, captions_lengths_batch):
                        assert np.count_nonzero(caption) == length
            except tf.errors.OutOfRangeError:
                pass


def test_train_val_loader_batch_size_invariance_val(
    train_image_paths,
    train_captions,
    train_captions_lengths,
    val_image_paths,
    val_captions,
    val_captions_lengths,
    epochs,
    prefetch_size,
):

    tf.reset_default_graph()
    loader_5 = TrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        5,
        prefetch_size,
    )
    images, captions, captions_lengths = loader_5.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(loader_5.val_init)
        images_val_5, captions_val_5, captions_val_lengths_5 = sess.run(
            [images, captions, captions_lengths]
        )

    tf.reset_default_graph()
    loader_3 = TrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        3,
        prefetch_size,
    )
    images, captions, captions_lengths = loader_3.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(loader_3.val_init)
        images_val_3, captions_val_3, captions_val_lengths_3 = sess.run(
            [images, captions, captions_lengths]
        )

    np.testing.assert_equal(images_val_5[:3], images_val_3)
    np.testing.assert_equal(captions_val_lengths_5[:3], captions_val_lengths_3)
    max_len_3 = max(len(l) for l in captions_val_3)
    np.testing.assert_equal(captions_val_5[:3, :max_len_3], captions_val_3)


def test_train_val_loader_all_elements_val_taken(
    train_image_paths,
    train_captions,
    train_captions_lengths,
    val_image_paths,
    val_captions,
    val_captions_lengths,
):
    for batch_size in range(1, len(val_captions)):
        tf.reset_default_graph()
        loader = TrainValLoader(
            train_image_paths,
            train_captions,
            train_captions_lengths,
            val_image_paths,
            val_captions,
            val_captions_lengths,
            batch_size,
            1,
        )
        images, captions, captions_lengths = loader.get_next()
        index = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(loader.val_init)
            try:
                while True:
                    batch_caps, batch_lengths = sess.run([captions, captions_lengths])
                    for caption, length, val_cap in zip(
                        batch_caps,
                        batch_lengths,
                        val_captions[index : index + batch_size],
                    ):
                        np.testing.assert_array_equal(caption[:length], val_cap)
                    index += len(batch_caps)
            except tf.errors.OutOfRangeError:
                pass


def test_inference_loader(
    val_image_paths, val_captions, val_captions_lengths, batch_size, prefetch_size
):
    # Using the validation fixtures since it wouldn't make any difference
    tf.reset_default_graph()
    loader = InferenceLoader(
        val_image_paths, val_captions, val_captions_lengths, batch_size, prefetch_size
    )
    images, captions, captions_lengths = loader.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                images_batch, captions_batch, captions_lengths_batch = sess.run(
                    [images, captions, captions_lengths]
                )
                _, width, height, _ = images_batch.shape
                assert width == 224
                assert height == 224
                num_tokens_batch = captions_batch.shape[1]
                for caption in captions_batch:
                    assert caption.shape[0] == num_tokens_batch
                for caption, length in zip(captions_batch, captions_lengths_batch):
                    assert np.count_nonzero(caption) == length
        except tf.errors.OutOfRangeError:
            pass
