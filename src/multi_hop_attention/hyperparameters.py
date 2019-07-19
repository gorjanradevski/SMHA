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
import absl.logging

from utils.datasets import FlickrDataset, PascalSentencesDataset
from multi_hop_attention.models import MultiHopAttentionModel
from multi_hop_attention.loaders import TrainValLoader
from utils.evaluators import Evaluator

logging.getLogger("utils.datasets").setLevel(logging.ERROR)
logging.getLogger("multi_hop_attention.models").setLevel(logging.ERROR)
logging.getLogger("multi_hop_attention.loaders").setLevel(logging.ERROR)
logging.getLogger("utils.evaluators").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


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
        epochs: int,
        recall_at: int,
        decay_rate_epochs: int,
    ):
        """Defines the search space and the general attributes.

        Args:
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            epochs: The number of epochs per experiment.
            recall_at: The recall at K.
            decay_rate_epochs: When to decay the learning rate.
        """
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.epochs = epochs
        self.decay_rate_epochs = decay_rate_epochs
        self.recall_at = recall_at
        self.last_best = sys.maxsize
        # Set seed value for all experiments in the current iteration
        self.seed = datetime.now().microsecond
        # Generate experiment name
        self.name = "".join(random.choice(string.ascii_uppercase) for _ in range(5))
        # Define the search space
        self.search_space = {
            "joint_space": hp.choice("joint_space", [256, 512, 1024]),
            "num_layers": hp.choice("num_layers", [1, 2, 3]),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.00001), np.log(0.01)
            ),
            "margin": hp.choice("margin", [0.02, 0.2, 0.4, 1.0, 3.0, 5.0]),
            "attn_size": hp.choice("attn_size", [64, 128, 256]),
            "attn_hops": hp.choice("attn_hops", [5, 10, 20, 30]),
            "frob_norm_pen": hp.loguniform("frob_norm_pen", np.log(0.5), np.log(3.0)),
            "gradient_clip_val": hp.choice("gradient_clip_val", [1, 3, 5, 7]),
            "keep_prob": hp.choice("keep_prob", [0.7, 0.8, 0.9, 1.0]),
            "weight_decay": hp.loguniform(
                "weight_decay", np.log(0.00001), np.log(0.01)
            ),
            "k": hp.choice("k", range(10, 101, 10)),
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
        """Searches for the best hyperparameters, where after each random sampling the
        trials object is saved with the updated history.

        Args:
            num_iters: For many times to do random sampling.
            hparams_path: Where to dump the hparams.
            trials_path: Where to dump the trials object.

        Returns:
            None

        """
        for _ in range(num_iters):
            try:
                trials = pickle.load(open(trials_path, "rb"))
                self.last_best = trials.best_trial["result"]["loss"]
            except FileNotFoundError:
                trials = Trials()
            logger.info(
                f"Last best from previous iteration was: {self.last_best} on "
                f"[{datetime.now().replace(second=0, microsecond=0)}]"
            )
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
            if trials.best_trial["result"]["loss"] < self.last_best:
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
        epochs: int,
        decay_rate_epochs: int,
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
            epochs: The number of epochs per experiment.
            recall_at: The recall at K.
            decay_rate_epochs: When to decay the learning rate.
        """
        super().__init__(
            batch_size, prefetch_size, epochs, recall_at, decay_rate_epochs
        )
        self.images_path = images_path
        self.texts_path = texts_path
        self.train_imgs_file_path = train_imgs_file_path
        self.val_imgs_file_path = val_imgs_file_path

    def objective(self, args: Dict[str, Any]):
        joint_space = args["joint_space"]
        num_layers = args["num_layers"]
        attn_size = args["attn_size"]
        attn_hops = args["attn_hops"]
        frob_norm_pen = args["frob_norm_pen"]
        learning_rate = args["learning_rate"]
        gradient_clip_val = args["gradient_clip_val"]
        margin = args["margin"]
        keep_prob = args["keep_prob"]
        weight_decay = args["weight_decay"]
        k = args["k"]

        dataset = FlickrDataset(self.images_path, self.texts_path)
        train_image_paths, train_captions = dataset.get_data(self.train_imgs_file_path)
        val_image_paths, val_captions = dataset.get_data(self.val_imgs_file_path)
        evaluator_val = Evaluator(len(val_image_paths), joint_space * attn_hops)

        # Resetting the default graph and setting the random seed
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        loader = TrainValLoader(
            train_image_paths,
            train_captions,
            val_image_paths,
            val_captions,
            self.batch_size,
            self.prefetch_size,
        )
        images, captions, captions_lengths = loader.get_next()

        decay_steps = self.decay_rate_epochs * len(train_image_paths) / self.batch_size
        model = MultiHopAttentionModel(
            images,
            captions,
            captions_lengths,
            margin,
            joint_space,
            num_layers,
            attn_size,
            attn_hops,
            k,
            learning_rate,
            gradient_clip_val,
            decay_steps,
        )

        with tf.Session() as sess:
            # Initialize model
            model.init(sess)
            for e in range(self.epochs):
                # Reset the evaluator
                evaluator_val.reset_all_vars()

                # Initialize iterator with train data
                sess.run(loader.train_init)
                try:
                    while True:
                        _, loss = sess.run(
                            [model.optimize, model.loss],
                            feed_dict={
                                model.frob_norm_pen: frob_norm_pen,
                                model.keep_prob: keep_prob,
                                model.weight_decay: weight_decay,
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

                if e >= self.epochs // 2:
                    # last_best is negative value, so taking - last_best
                    if evaluator_val.best_image2text_recall_at_k < -self.last_best / 3:
                        logger.info("Terminating early!")
                        break

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
        epochs: int,
        decay_rate_epochs: int,
        recall_at: int,
    ):
        """Creates a finder that will find the best hyperparameters for the Pascal
        sentences datasets.

        Args:
            images_path: The path to all Pascal sentences images.
            texts_path: The path to the captions.
            batch_size: The batch size that will be used to conduct the experiments.
            prefetch_size: The prefetching size when running on GPU.
            epochs: The number of epochs per experiment.
            recall_at: The recall at K.
            decay_rate_epochs: When to decay the learning rate.

        """
        super().__init__(
            batch_size, prefetch_size, epochs, recall_at, decay_rate_epochs
        )
        self.images_path = images_path
        self.texts_path = texts_path

    def objective(self, args: Dict[str, Any]):
        joint_space = args["joint_space"]
        num_layers = args["num_layers"]
        attn_size = args["attn_size"]
        attn_hops = args["attn_hops"]
        frob_norm_pen = args["frob_norm_pen"]
        learning_rate = args["learning_rate"]
        gradient_clip_val = args["gradient_clip_val"]
        margin = args["margin"]
        keep_prob = args["keep_prob"]
        weight_decay = args["weight_decay"]
        k = args["k"]

        dataset = PascalSentencesDataset(self.images_path, self.texts_path)
        train_image_paths, train_captions = dataset.get_train_data()
        # Getting the vocabulary size of the train dataset
        val_image_paths, val_captions = dataset.get_val_data()
        evaluator_val = Evaluator(len(val_image_paths), joint_space * attn_hops)

        # Resetting the default graph and setting the random seed
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        loader = TrainValLoader(
            train_image_paths,
            train_captions,
            val_image_paths,
            val_captions,
            self.batch_size,
            self.prefetch_size,
        )
        images, captions, captions_lengths = loader.get_next()

        decay_steps = self.decay_rate_epochs * len(train_image_paths) / self.batch_size
        model = MultiHopAttentionModel(
            images,
            captions,
            captions_lengths,
            margin,
            joint_space,
            num_layers,
            attn_size,
            attn_hops,
            k,
            learning_rate,
            gradient_clip_val,
            decay_steps,
        )
        with tf.Session() as sess:
            # Initialize model
            model.init(sess)
            for e in range(self.epochs):
                # Reset the evaluator
                evaluator_val.reset_all_vars()

                # Initialize iterator with train data
                sess.run(loader.train_init)
                try:
                    while True:
                        _, loss = sess.run(
                            [model.optimize, model.loss],
                            feed_dict={
                                model.frob_norm_pen: frob_norm_pen,
                                model.keep_prob: keep_prob,
                                model.weight_decay: weight_decay,
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

                if e >= self.epochs // 2:
                    # last_best is negative value, so taking - last_best
                    if evaluator_val.best_image2text_recall_at_k < -self.last_best / 3:
                        logger.info("Terminating early!")
                        break

        logger.info(
            f"Current best image to text recall at {self.recall_at} is: "
            f"{evaluator_val.best_image2text_recall_at_k}"
        )

        return -evaluator_val.best_image2text_recall_at_k
