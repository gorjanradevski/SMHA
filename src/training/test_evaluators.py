import numpy as np
import pytest
from training.evaluators import Evaluator


@pytest.fixture
def num_samples():
    return 15


@pytest.fixture
def num_features():
    return 6


@pytest.fixture
def losses():
    return [[0.2, 0.3, 0.5], [0.1, 0.5, 0.2], [0.1, 1.0, 0.5]]


@pytest.fixture
def embedded_captions_p1():
    np.random.seed(42)
    return np.random.rand(15, 6)[:5, :]


@pytest.fixture
def embedded_captions_p2():
    np.random.seed(42)
    return np.random.rand(15, 6)[5:12, :]


@pytest.fixture
def embedded_captions_p3():
    np.random.seed(42)
    return np.random.rand(15, 6)[12:, :]


@pytest.fixture
def embedded_captions():
    np.random.seed(42)
    return np.random.rand(15, 6)


@pytest.fixture
def embedded_images_p1():
    np.random.seed(40)
    return np.random.rand(15, 6)[:5, :]


@pytest.fixture
def embedded_images_p2():
    np.random.seed(40)
    return np.random.rand(15, 6)[5:12, :]


@pytest.fixture
def embedded_images_p3():
    np.random.seed(40)
    return np.random.rand(15, 6)[12:, :]


@pytest.fixture
def embedded_images():
    np.random.seed(40)
    return np.random.rand(15, 6)


def test_loss_computation(num_samples, num_features, losses):
    evaluator = Evaluator(num_samples, num_features)
    for epoch in losses:
        evaluator.reset_all_vars()
        for loss in epoch:
            evaluator.update_metrics(loss)
        if evaluator.is_best_loss():
            evaluator.update_best_loss()

    assert evaluator.best_loss == 0.8


def test_embedded_update(
    embedded_images_p1,
    embedded_images_p2,
    embedded_images_p3,
    embedded_captions_p1,
    embedded_captions_p2,
    embedded_captions_p3,
    embedded_captions,
    embedded_images,
    num_samples,
    num_features,
):
    evaluator = Evaluator(num_samples, num_features)
    evaluator.update_embeddings(embedded_images_p1, embedded_captions_p1)
    evaluator.update_embeddings(embedded_images_p2, embedded_captions_p2)
    evaluator.update_embeddings(embedded_images_p3, embedded_captions_p3)
    np.testing.assert_equal(embedded_images, evaluator.embedded_images)
    np.testing.assert_equal(embedded_captions, evaluator.embedded_captions)


def test_recall_at_k():
    # TODO: Good test about recall at K
    pass
