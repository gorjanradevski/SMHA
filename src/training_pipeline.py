import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os

from training.datasets import CocoDataset
from training.hyperparameters import YParams
from training.loaders import CocoTrainValLoader
from training.models import Text2ImageMatchingModel
from training.evaluators import Evaluator

from utils.constants import BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

# TODO: Experiment name generation method


def train(
    hparams_path: str,
    train_images_path: str,
    train_json_path: str,
    val_images_path: str,
    val_json_path: str,
    epochs: int,
) -> None:
    """Starts a training session.

    Args:
        hparams_path: The path to the hyperparameters yaml file.
        train_images_path: The path to the training images.
        train_json_path: The path to the training annotations.
        val_images_path: The path to the validation images.
        val_json_path: The path to the validation annotations.
        epochs: The number of epochs to train the model.

    Returns:
        None

    """
    hparams = YParams(hparams_path)
    train_dataset = CocoDataset(
        train_images_path, train_json_path, hparams.min_unk_sub, train=True
    )
    train_image_paths, train_captions, train_captions_lengths, train_labels = (
        train_dataset.get_img_paths_captions_lengths()
    )
    # Getting the vocabulary size of the train dataset
    vocab_size = CocoDataset.get_vocab_size()

    logger.info("Train dataset created...")
    val_dataset = CocoDataset(
        val_images_path, val_json_path, hparams.min_unk_sub, train=False
    )
    val_image_paths, val_captions, val_captions_lengths, val_labels = (
        val_dataset.get_img_paths_captions_lengths()
    )
    logger.info("Validation dataset created...")

    evaluator_train = Evaluator()
    evaluator_val = Evaluator()

    logger.info("Evaluators created...")

    # Resetting the default graph and setting the random seed
    tf.reset_default_graph()
    tf.set_random_seed(hparams.seed)

    loader = CocoTrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        train_labels,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        val_labels,
        BATCH_SIZE,
    )
    images, captions, captions_lengths, labels = loader.get_next()
    logger.info("Loader created...")

    model = Text2ImageMatchingModel(
        hparams.seed,
        images,
        captions,
        captions_lengths,
        labels,
        hparams.margin,
        hparams.rnn_hidden_size,
        vocab_size,
        hparams.embed_size,
        hparams.cell,
        hparams.layers,
        hparams.attn_size1,
        hparams.attn_size2,
        hparams.opt,
        hparams.learning_rate,
        hparams.gradient_clip_val,
    )
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:

        # Initialize all variables in the graph
        model.init(sess)

        for e in range(epochs):

            # Reset evaluators
            evaluator_train.reset_metrics()
            evaluator_val.reset_metrics()

            # Initialize iterator with training data
            sess.run(loader.train_init)
            try:
                with tqdm(total=len(train_labels)) as pbar:
                    while True:
                        _, loss, labels = sess.run(
                            [model.optimize, model.loss, model.labels]
                        )
                        evaluator_train.update_metrics(loss)
                        pbar.update(len(labels))
                        pbar.set_postfix({"Batch loss": loss})
            except tf.errors.OutOfRangeError:
                pass

            # Initialize iterator with validation data
            sess.run(loader.val_init)
            try:
                while True:
                    loss = sess.run(model.loss)
                    evaluator_val.update_metrics(loss)
            except tf.errors.OutOfRangeError:
                pass

            if evaluator_val.is_best_loss():
                logger.info(f"Loss on epoch {e}: {evaluator_val.best_loss}")
                evaluator_val.update_best_loss()


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.hparams_path,
        args.train_images_path,
        args.train_json_path,
        args.val_images_path,
        args.val_json_path,
        args.epochs,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams_path",
        type=str,
        default="hyperparameters/default_hparams.yaml",
        help="Path to an hyperparameters yaml file.",
    )
    parser.add_argument(
        "--train_images_path",
        type=str,
        default="data/train2014/",
        help="Path where the train images are.",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/annotations/captions_train2014.json",
        help="Path where the train json file with the captions and image ids.",
    )
    parser.add_argument(
        "--val_images_path",
        type=str,
        default="data/val2014/",
        help="Path where the validation images are.",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/annotations/captions_val2014.json",
        help="Path where the val json file with the captions and image ids.",
    )
    parser.add_argument(
        "--log_model_path",
        type=str,
        default="logs/",
        help="Path where the val json file with the captions and image ids.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/",
        help="Path where the val json file with the captions and image ids.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
