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
    batch_size: int,
    checkpoint_path: str,
    imagenet_checkpoint: bool,
    save_model_path: str,
    log_model_path: str,
    recall_at: int,
) -> None:
    """Starts a training session.

    Args:
        hparams_path: The path to the hyperparameters yaml file.
        train_images_path: The path to the training images.
        train_json_path: The path to the training annotations.
        val_images_path: The path to the validation images.
        val_json_path: The path to the validation annotations.
        epochs: The number of epochs to train the model.
        batch_size: The batch size to be used.
        checkpoint_path: Path to a valid model checkpoint.
        imagenet_checkpoint: Whether the checkpoint points to an imagenet model.
        save_model_path: Where to save the model.
        log_model_path: Where to log the summaries.
        recall_at: Validate with recall at (input).

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
    # The number of features at the output will be: rnn_hidden_size * attn_size2
    evaluator_val = Evaluator(
        len(val_image_paths), hparams.rnn_hidden_size * hparams.attn_size2
    )

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
        batch_size,
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
        log_model_path,
    )
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:

        # Initializers
        model.init(sess, checkpoint_path, imagenet_checkpoint)
        model.add_summary_graph(sess)

        for e in range(epochs):
            # Reset evaluators
            evaluator_train.reset_all_vars()
            evaluator_val.reset_all_vars()

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

            # Write training summaries
            train_loss_summary = sess.run(
                model.train_loss_summary,
                feed_dict={model.train_loss_ph: evaluator_train.loss},
            )
            model.add_summary(sess, train_loss_summary)

            # Initialize iterator with validation data
            sess.run(loader.val_init)
            try:
                with tqdm(total=len(val_labels)) as pbar:
                    while True:
                        loss, labels, embedded_images, embedded_captions = sess.run(
                            [
                                model.loss,
                                model.labels,
                                model.attended_images,
                                model.attended_captions,
                            ]
                        )
                        evaluator_val.update_metrics(loss)
                        evaluator_val.update_embeddings(
                            embedded_images, embedded_captions
                        )
                        pbar.update(len(labels))
            except tf.errors.OutOfRangeError:
                pass

            # Write validation summaries
            val_loss_summary, val_recall_at_k = sess.run(
                [model.val_loss_summary, model.val_recall_at_k_summary],
                feed_dict={
                    model.val_loss_ph: evaluator_val.loss,
                    model.val_recall_at_k_ph: evaluator_val.best_recall_at_k,
                },
            )
            model.add_summary(sess, val_loss_summary)
            model.add_summary(sess, val_recall_at_k)

            if evaluator_val.is_best_recall_at_k():
                evaluator_val.update_best_recall_at_k()
                logger.info("=============================")
                logger.info(
                    f"Found new best on epoch {e+1} with recall at {recall_at}: "
                    f"{evaluator_val.best_recall_at_k}! Saving model..."
                )
                logger.info("=============================")
                model.save_model(sess, save_model_path)


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
        args.batch_size,
        args.checkpoint_path,
        args.imagenet_checkpoint,
        args.save_model_path,
        args.log_model_path,
        args.recall_at,
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
        "--checkpoint_path",
        type=str,
        default="models/image_encoders/vgg_16.ckpt",
        help="Path to a model checkpoint.",
    )
    parser.add_argument(
        "--imagenet_checkpoint",
        action="store_true",
        help="If the checkpoint is an imagenet checkpoint.",
    )
    parser.add_argument(
        "--log_model_path",
        type=str,
        default="logs/tryout",
        help="Where to log the summaries.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/tryout",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--recall_at", type=int, default=5, help="Validate with recall at K (input)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
