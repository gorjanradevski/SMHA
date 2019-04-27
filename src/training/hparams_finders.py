import tensorflow as tf
import numpy as np
import random
import string
from abc import ABC, abstractmethod
from hyperopt import fmin, tpe, space_eval, hp
from datetime import datetime
from ruamel.yaml import YAML
from typing import Dict, Any
import logging
from tqdm import tqdm

from training.datasets import Flickr8kDataset, get_vocab_size
from training.models import Text2ImageMatchingModel
from training.loaders import TrainValLoader
from training.evaluators import Evaluator

logging.getLogger("training.datasets").setLevel(logging.ERROR)
logging.getLogger("training.models").setLevel(logging.ERROR)
logging.getLogger("training.loaders").setLevel(logging.ERROR)
logging.getLogger("training.evaluators").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHparamsFinder(ABC):

    # Abstract class from which all finders must inherit
    def __init__(
        self,
        batch_size: int,
        prefetch_size: int,
        imagenet_checkpoint_path: str,
        epochs: int,
        recall_at: int,
    ):
        """Defines the search space and the general attributes.

        Args:
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            imagenet_checkpoint_path: The checkpoint to the pretrained imagenet weights.
            epochs: The number of epochs per experiment.
            recall_at: Recall at K (this is K) evaluation metric.
        """
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.imagenet_checkpoint_path = imagenet_checkpoint_path
        self.epochs = epochs
        self.recall_at = recall_at
        # Default seed value
        self.seed = 42
        # Generate experiment name
        self.name = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
        # Define the search space
        self.search_space = {
            "min_unk_sub": hp.choice("min_unk_sub", range(1, 10)),
            "embed_size": hp.choice("embed_size", range(50, 300)),
            "layers": hp.choice("layers", range(1, 3)),
            "rnn_hidden_size": hp.choice("rnn_hidden_size", range(64, 256)),
            "cell": hp.choice("cell", ["lstm", "gru"]),
            "keep_prob": hp.uniform("keep_prob", 0.4, 0.9),
            "wd": hp.loguniform("wd", np.log(0.000_001), np.log(0.1)),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.00001), np.log(0.3)
            ),
            "opt": hp.choice("opt", ["adam", "sgd", "adadelta"]),
            "margin": hp.uniform("margin", 0.001, 5),
            "attn_size": hp.choice("attn_size", range(5, 50)),
            "gradient_clip_val": hp.choice("gradient_clip_val", range(1, 10)),
        }

    @abstractmethod
    def objective(self, args: Dict[str, Any]) -> float:
        """The funciton to be minimized. In this case recall at K on the validation
        set.

        Args:
            args: The dictionary with arguments.

        Returns:
            The negative recall at K score.

        """
        pass

    def find_best(self, max_evals: int, dump_hparams_path: str) -> None:
        best_hparams = space_eval(
            self.search_space,
            fmin(
                self.objective,
                self.search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                show_progressbar=False,
            ),
        )
        best_hparams["name"] = self.name
        best_hparams["seed"] = self.seed
        with open(dump_hparams_path + self.name + ".yaml", "w") as yaml_file:
            YAML().dump(best_hparams, yaml_file)


class Flickr8kHparamsFinder(BaseHparamsFinder):
    def __init__(
        self,
        images_path: str,
        texts_path: str,
        train_imgs_file_path: str,
        val_imgs_file_path: str,
        batch_size: int,
        prefetch_size: int,
        imagenet_checkpoint_path: str,
        epochs: int,
        recall_at: int,
    ):
        """

        Args:
            images_path: The path to all Flickr8k images.
            texts_path: The path to the captions.
            train_imgs_file_path: File path to train images.
            val_imgs_file_path: File path to val images.
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            imagenet_checkpoint_path: The checkpoint to the pretrained imagenet weights.
            epochs: The number of epochs per experiment.
            recall_at: Recall at K (this is K) evaluation metric.
        """
        super().__init__(
            batch_size, prefetch_size, imagenet_checkpoint_path, epochs, recall_at
        )
        self.images_path = images_path
        self.texts_path = texts_path
        self.train_imgs_file_path = train_imgs_file_path
        self.val_imgs_file_path = val_imgs_file_path

    def objective(self, args: Dict[str, Any]):
        logger.info(f"Trying out hyperparameters: {args}")

        min_unk_sub = args["min_unk_sub"]
        rnn_hidden_size = args["rnn_hidden_size"]
        margin = args["margin"]
        embed_size = args["embed_size"]
        cell = args["cell"]
        layers = args["layers"]
        attn_size = args["attn_size"]
        opt = args["opt"]
        learning_rate = args["learning_rate"]
        gradient_clip_val = args["gradient_clip_val"]

        dataset = Flickr8kDataset(self.images_path, self.texts_path, min_unk_sub)
        train_image_paths, train_captions, train_captions_lengths = dataset.get_data(
            self.train_imgs_file_path
        )
        # Getting the vocabulary size of the train dataset
        val_image_paths, val_captions, val_captions_lengths = dataset.get_data(
            self.val_imgs_file_path
        )
        # The number of features at the output will be: rnn_hidden_size * 2
        evaluator_val = Evaluator(len(val_image_paths), rnn_hidden_size * 2)

        # Resetting the default graph and setting the random seed
        self.seed = datetime.now().microsecond
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        loader = TrainValLoader(
            train_image_paths,
            train_captions,
            train_captions_lengths,
            val_image_paths,
            val_captions,
            val_captions_lengths,
            self.batch_size,
            self.prefetch_size,
        )
        images, captions, captions_lengths = loader.get_next()
        model = Text2ImageMatchingModel(
            self.seed,
            images,
            captions,
            captions_lengths,
            margin,
            rnn_hidden_size,
            get_vocab_size(Flickr8kDataset),
            embed_size,
            cell,
            layers,
            attn_size,
            opt,
            learning_rate,
            gradient_clip_val,
        )

        with tf.Session() as sess:
            # Initialize model
            model.init(sess, self.imagenet_checkpoint_path, imagenet_checkpoint=True)
            for e in range(self.epochs):
                # Reset the evaluator
                evaluator_val.reset_all_vars()

                # Initialize iterator with training data
                sess.run(loader.train_init)
                try:
                    with tqdm(total=len(train_image_paths)) as pbar:
                        while True:
                            _, loss, lengths = sess.run(
                                [model.optimize, model.loss, model.captions_len]
                            )
                            pbar.update(len(lengths))
                except tf.errors.OutOfRangeError:
                    pass

                # Initialize iterator with validation data
                sess.run(loader.val_init)
                try:
                    while True:
                        loss, embedded_images, embedded_captions = sess.run(
                            [model.loss, model.attended_images, model.attended_captions]
                        )
                        evaluator_val.update_metrics(loss)
                        evaluator_val.update_embeddings(
                            embedded_images, embedded_captions
                        )
                except tf.errors.OutOfRangeError:
                    pass

                # Update recall at K
                evaluator_val.image2text_recall_at_k(self.recall_at)

                if evaluator_val.is_best_recall_at_k():
                    evaluator_val.update_best_recall_at_k()

        return -evaluator_val.best_recall_at_k
