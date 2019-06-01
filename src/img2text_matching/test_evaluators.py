import numpy as np
import pytest
from img2text_matching.evaluators import Evaluator


@pytest.fixture
def num_samples():
    return 50


@pytest.fixture
def num_features():
    return 6


@pytest.fixture
def losses():
    return [[0.2, 0.3, 0.5], [0.1, 0.5, 0.2], [0.1, 1.0, 0.5]]


@pytest.fixture
def embedded_captions():
    np.random.seed(42)
    return np.random.rand(50, 6)


@pytest.fixture
def embedded_images():
    np.random.seed(40)
    return np.random.rand(50, 6)


def test_loss_computation(num_samples, num_features, losses):
    evaluator = Evaluator(num_samples, num_features)
    for epoch in losses:
        evaluator.reset_all_vars()
        for loss in epoch:
            evaluator.update_metrics(loss)
        if evaluator.is_best_loss():
            evaluator.update_best_loss()

    assert evaluator.best_loss == 0.8


def test_embedded_update(embedded_captions, embedded_images, num_samples, num_features):
    evaluator = Evaluator(num_samples, num_features)
    evaluator.update_embeddings(embedded_images[:9], embedded_captions[:9])
    evaluator.update_embeddings(embedded_images[9:13], embedded_captions[9:13])
    evaluator.update_embeddings(embedded_images[13:27], embedded_captions[13:27])
    evaluator.update_embeddings(embedded_images[27:39], embedded_captions[27:39])
    evaluator.update_embeddings(embedded_images[39:43], embedded_captions[39:43])
    evaluator.update_embeddings(embedded_images[43:49], embedded_captions[43:49])
    evaluator.update_embeddings(embedded_images[49:], embedded_captions[49:])
    np.testing.assert_equal(embedded_images, evaluator.embedded_images)
    np.testing.assert_equal(embedded_captions, evaluator.embedded_captions)


def test_recall_at_k():
    # TODO: Good test about recall at K
    pass
