import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.loss = 0
        self.best_loss = sys.maxsize

    def reset_metrics(self):
        self.loss = 0

    def update_metrics(self, loss):
        self.loss += loss

    def is_best_loss(self):
        if self.loss < self.best_loss:
            return True
        return False

    def update_best_loss(self):
        self.best_loss = self.loss
