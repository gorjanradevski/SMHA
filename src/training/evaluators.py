import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, num_samples: int = 0, num_features: int = 0):
        self.loss = 0.0
        self.best_loss = sys.maxsize
        self.recall_at_k = -1.0
        self.best_recall_at_k = -1.0
        self.index_update = 0
        self.num_samples = num_samples
        self.num_features = num_features
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_captions = np.zeros((self.num_samples, self.num_features))

    def reset_all_vars(self) -> None:
        self.loss = 0
        self.recall_at_k = -1.0
        self.index_update = 0
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_captions = np.zeros((self.num_samples, self.num_features))

    def update_metrics(self, loss: float) -> None:
        self.loss += loss

    def update_embeddings(
        self, embedded_images: np.ndarray, embedded_captions: np.ndarray
    ) -> None:
        num_samples = embedded_images.shape[0]
        self.embedded_images[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_images
        self.embedded_captions[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_captions
        self.index_update += num_samples

    def is_best_loss(self) -> bool:
        if self.loss < self.best_loss:
            return True
        return False

    def is_best_recall_at_k(self) -> bool:
        if self.recall_at_k > self.best_recall_at_k:
            return True
        return False

    def update_best_loss(self):
        self.best_loss = self.loss

    def update_best_recall_at_k(self):
        self.best_recall_at_k = self.recall_at_k

    def image2text_recall_at_k(self, k: int) -> None:
        """Computes the recall at K and updates the object variable.

        Args:
            k: Recall at K (this is K).

        Returns:


        """
        num_images = self.embedded_images.shape[0] // 5
        feature_size = self.embedded_images.shape[1]
        ranks = np.zeros(num_images)
        for index in range(num_images):
            # Query image
            query_image = self.embedded_images[5 * index].reshape(1, feature_size)
            # Similarities
            similarities = np.dot(query_image, self.embedded_captions.T).flatten()
            indices = np.argsort(similarities)[::-1]
            # Score
            rank = sys.maxsize
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(indices == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        self.recall_at_k = len(np.where(ranks < k)[0]) / len(ranks)
