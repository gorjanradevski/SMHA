import tensorflow as tf
from tensorflow.contrib.training import HParams
import numpy as np
import random
import string
from abc import ABC, abstractmethod
from hyperopt import fmin, tpe, space_eval, hp, Trials
from datetime import datetime
from ruamel.yaml import YAML
from typing import Dict, Any
import logging
import pickle
import sys

from img2text_matching.datasets import (
    FlickrDataset,
    PascalSentencesDataset,
    get_vocab_size,
)
from img2text_matching.models import Text2ImageMatchingModel
from img2text_matching.loaders import TrainValLoader
from img2text_matching.evaluators import Evaluator

logging.getLogger("img2text_matching.datasets").setLevel(logging.ERROR)
logging.getLogger("img2text_matching.models").setLevel(logging.ERROR)
logging.getLogger("img2text_matching.loaders").setLevel(logging.ERROR)
logging.getLogger("img2text_matching.evaluators").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YParams(HParams):
    def __init__(self, hparams_path: str):
        super().__init__()
        with open(hparams_path) as fp:
            for k, v in YAML().load(fp).items():
                self.add_hparam(k, v)


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
            recall_at: The recall at K.
        """
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.imagenet_checkpoint_path = imagenet_checkpoint_path
        self.epochs = epochs
        self.recall_at = recall_at
        # Set seed value for all experiments in the current iteration
        self.seed = datetime.now().microsecond
        # Generate experiment name
        self.name = "".join(random.choice(string.ascii_uppercase) for _ in range(5))
        # Define the search space
        self.search_space = {
            "min_unk_sub": hp.choice("min_unk_sub", range(3, 10)),
            "layers": hp.choice("layers", range(1, 3)),
            "rnn_hidden_size": hp.choice("rnn_hidden_size", [128, 256, 512]),
            "keep_prob": hp.choice("keep_prob", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "weight_decay": hp.loguniform("wd", np.log(0.00001), np.log(0.01)),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.00001), np.log(0.01)
            ),
            "margin": hp.choice("margin", range(1, 100, 5)),
            "attn_size": hp.choice("attn_size", range(20, 100, 5)),
            "attn_heads": hp.choice("attn_heads", range(20, 50, 5)),
            "frob_norm_pen": hp.loguniform("frob_norm_pen", np.log(1.0), np.log(5.0)),
            "gradient_clip_val": hp.choice("gradient_clip_val", range(1, 10)),
            "batch_hard": hp.choice("batch_hard", [True, False]),
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

    def find_best(self, num_iters: int, hparams_path: str, trials_path: str) -> None:
        for _ in range(num_iters):
            try:
                trials = pickle.load(open(trials_path, "rb"))
                last_best = trials.best_trial["result"]["loss"]
            except FileNotFoundError:
                trials = Trials()
                last_best = sys.maxsize
            logger.info(f"Last best from previous iteration was: {last_best}")
            best_hparams = space_eval(
                self.search_space,
                fmin(
                    self.objective,
                    self.search_space,
                    algo=tpe.suggest,
                    max_evals=len(trials.trials) + 1,
                    show_progressbar=False,
                    trials=trials,
                ),
            )
            # Dump Trials object always
            with open(trials_path, "wb") as trials_file:
                pickle.dump(trials, trials_file)

            # Dump hparams only if better result was achieved
            if trials.best_trial["result"]["loss"] < last_best:
                best_hparams["name"] = self.name
                best_hparams["seed"] = self.seed

                with open(hparams_path, "w") as yaml_file:
                    YAML().dump(best_hparams, yaml_file)


class FlickrHparamsFinder(BaseHparamsFinder):
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
        """Creates a finder that will find the best hyperparameters for the Flickr
        datasets.

        Args:
            images_path: The path to all Flickr8k images.
            texts_path: The path to the captions.
            train_imgs_file_path: File path to train images.
            val_imgs_file_path: File path to val images.
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            imagenet_checkpoint_path: The checkpoint to the pretrained imagenet weights.
            epochs: The number of epochs per experiment.
            recall_at: The recall at K.
        """
        super().__init__(
            batch_size, prefetch_size, imagenet_checkpoint_path, epochs, recall_at
        )
        self.images_path = images_path
        self.texts_path = texts_path
        self.train_imgs_file_path = train_imgs_file_path
        self.val_imgs_file_path = val_imgs_file_path

    def objective(self, args: Dict[str, Any]):
        min_unk_sub = args["min_unk_sub"]
        rnn_hidden_size = args["rnn_hidden_size"]
        margin = args["margin"]
        layers = args["layers"]
        attn_size = args["attn_size"]
        attn_heads = args["attn_heads"]
        frob_norm_pen = args["frob_norm_pen"]
        learning_rate = args["learning_rate"]
        gradient_clip_val = args["gradient_clip_val"]
        keep_prob = args["keep_prob"]
        weight_decay = args["weight_decay"]
        batch_hard = args["batch_hard"]

        dataset = FlickrDataset(self.images_path, self.texts_path, min_unk_sub)
        train_image_paths, train_captions, train_captions_lengths = dataset.get_data(
            self.train_imgs_file_path
        )
        val_image_paths, val_captions, val_captions_lengths = dataset.get_data(
            self.val_imgs_file_path
        )
        # The number of features at the output will be: rnn_hidden_size * attn_heads
        evaluator_val = Evaluator(len(val_image_paths), rnn_hidden_size * attn_heads)

        # Resetting the default graph and setting the random seed
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
            images,
            captions,
            captions_lengths,
            margin,
            rnn_hidden_size,
            get_vocab_size(FlickrDataset),
            layers,
            attn_size,
            attn_heads,
            learning_rate,
            gradient_clip_val,
            batch_hard,
        )

        with tf.Session() as sess:
            # Initialize model
            model.init(sess, self.imagenet_checkpoint_path, imagenet_checkpoint=True)
            for e in range(self.epochs):
                # Reset the evaluator
                evaluator_val.reset_all_vars()

                # Initialize iterator with img2text_matching data
                sess.run(loader.train_init)
                try:
                    while True:
                        _, loss = sess.run(
                            [model.optimize, model.loss],
                            feed_dict={
                                model.keep_prob: keep_prob,
                                model.weight_decay: weight_decay,
                                model.frob_norm_pen: frob_norm_pen,
                            },
                        )
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

                if evaluator_val.is_best_image2text_recall_at_k(self.recall_at):
                    evaluator_val.update_best_image2text_recall_at_k()

        logger.info(
            f"Current best image to text recall at {self.recall_at} is: "
            f"{evaluator_val.best_image2text_recall_at_k}"
        )

        return -evaluator_val.best_image2text_recall_at_k


class PascalHparamsFinder(BaseHparamsFinder):
    def __init__(
        self,
        images_path: str,
        texts_path: str,
        batch_size: int,
        prefetch_size: int,
        imagenet_checkpoint_path: str,
        epochs: int,
        recall_at: int,
    ):
        """Creates a finder that will find the best hyperparameters for the Pascal
        sentences datasets.

        Args:
            images_path: The path to all Pascal sentences images.
            texts_path: The path to the captions.
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            imagenet_checkpoint_path: The checkpoint to the pretrained imagenet weights.
            epochs: The number of epochs per experiment.
            recall_at: The recall at K.
        """

        super().__init__(
            batch_size, prefetch_size, imagenet_checkpoint_path, epochs, recall_at
        )
        self.images_path = images_path
        self.texts_path = texts_path

    def objective(self, args: Dict[str, Any]):
        min_unk_sub = args["min_unk_sub"]
        rnn_hidden_size = args["rnn_hidden_size"]
        margin = args["margin"]
        layers = args["layers"]
        attn_size = args["attn_size"]
        attn_heads = args["attn_heads"]
        frob_norm_pen = args["frob_norm_pen"]
        learning_rate = args["learning_rate"]
        gradient_clip_val = args["gradient_clip_val"]
        keep_prob = args["keep_prob"]
        weight_decay = args["weight_decay"]
        batch_hard = args["batch_hard"]

        dataset = PascalSentencesDataset(self.images_path, self.texts_path, min_unk_sub)
        train_image_paths, train_captions, train_captions_lengths = (
            dataset.get_train_data()
        )
        # Getting the vocabulary size of the train dataset
        val_image_paths, val_captions, val_captions_lengths = dataset.get_val_data()
        # The number of features at the output will be: rnn_hidden_size * attn_heads
        evaluator_val = Evaluator(len(val_image_paths), rnn_hidden_size * attn_heads)

        # Resetting the default graph and setting the random seed
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
            images,
            captions,
            captions_lengths,
            margin,
            rnn_hidden_size,
            get_vocab_size(PascalSentencesDataset),
            layers,
            attn_size,
            attn_heads,
            learning_rate,
            gradient_clip_val,
            batch_hard,
        )

        with tf.Session() as sess:
            # Initialize model
            model.init(sess, self.imagenet_checkpoint_path, imagenet_checkpoint=True)
            for e in range(self.epochs):
                # Reset the evaluator
                evaluator_val.reset_all_vars()

                # Initialize iterator with img2text_matching data
                sess.run(loader.train_init)
                try:
                    while True:
                        _, loss = sess.run(
                            [model.optimize, model.loss],
                            feed_dict={
                                model.keep_prob: keep_prob,
                                model.weight_decay: weight_decay,
                                model.frob_norm_pen: frob_norm_pen,
                            },
                        )
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

                if evaluator_val.is_best_image2text_recall_at_k(self.recall_at):
                    evaluator_val.update_best_image2text_recall_at_k()

        logger.info(
            f"Current best image to text recall at {self.recall_at} is: "
            f"{evaluator_val.best_image2text_recall_at_k}"
        )

        return -evaluator_val.best_image2text_recall_at_k
