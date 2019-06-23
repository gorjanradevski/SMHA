import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os
import absl.logging

from utils.datasets import TrainCocoDataset, ValCocoDataset
from multi_hop_attention.hyperparameters import YParams
from multi_hop_attention.loaders import TrainValLoader
from multi_hop_attention.models import MultiHopAttentionModel
from utils.evaluators import Evaluator
from utils.constants import decay_rate_epochs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


def train(
    hparams_path: str,
    train_images_path: str,
    train_json_path: str,
    val_images_path: str,
    val_json_path: str,
    epochs: int,
    recall_at: int,
    batch_size: int,
    prefetch_size: int,
    checkpoint_path: str,
    save_model_path: str,
    log_model_path: str,
    learning_rate: float = None,
    frob_norm_pen: float = None,
    attn_heads: int = None,
    gor_pen: float = None,
    weight_decay: float = None,
) -> None:
    """Starts a training session.

    Args:
        hparams_path: The path to the hyperparameters yaml file.
        train_images_path: The path to the multi_hop_attention images.
        train_json_path: The path to the multi_hop_attention annotations.
        val_images_path: The path to the validation images.
        val_json_path: The path to the validation annotations.
        epochs: The number of epochs to train the model excluding the vgg.
        recall_at: The recall at K.
        batch_size: The batch size to be used.
        prefetch_size: The size of the prefetch on gpu.
        checkpoint_path: Path to a valid model checkpoint.
        save_model_path: Where to save the model.
        log_model_path: Where to log the summaries.
        learning_rate: If provided update the one in hparams.
        frob_norm_pen: If provided update the one in hparams.
        attn_heads: If provided update the one in hparams.
        gor_pen: If provided update the one in hparams.
        weight_decay: If provided update the one in hparams.

    Returns:
        None

    """
    hparams = YParams(hparams_path)
    # If learning rate is provided update the hparams learning rate
    if learning_rate is not None:
        hparams.set_hparam("learning_rate", learning_rate)
    # If frob_norm_pen is provided update the hparams frob_norm_pen
    if frob_norm_pen is not None:
        hparams.set_hparam("frob_norm_pen", frob_norm_pen)
    # If attn_heads is provided update the hparams attn_heads
    if attn_heads is not None:
        hparams.set_hparam("attn_heads", attn_heads)
    if gor_pen is not None:
        hparams.set_hparam("gor_pen", gor_pen)
    if weight_decay is not None:
        hparams.set_hparam("weight_decay", weight_decay)
    train_dataset = TrainCocoDataset(train_images_path, train_json_path)
    train_image_paths, train_captions = train_dataset.get_data()
    logger.info("Train dataset created...")
    val_dataset = ValCocoDataset(val_images_path, val_json_path)
    val_image_paths, val_captions = val_dataset.get_data()
    logger.info("Validation dataset created...")

    evaluator_train = Evaluator()
    # The number of features at the output will be: rnn_hidden_size * attn_heads
    evaluator_val = Evaluator(
        len(val_image_paths), hparams.rnn_hidden_size * hparams.attn_heads
    )

    logger.info("Evaluators created...")

    # Resetting the default graph and setting the random seed
    tf.reset_default_graph()
    tf.set_random_seed(hparams.seed)

    loader = TrainValLoader(
        train_image_paths,
        train_captions,
        val_image_paths,
        val_captions,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")

    decay_steps = decay_rate_epochs * len(train_image_paths) / batch_size
    model = MultiHopAttentionModel(
        images,
        captions,
        captions_lengths,
        hparams.margin,
        hparams.joint_space,
        hparams.num_layers,
        hparams.attn_size,
        hparams.attn_heads,
        hparams.learning_rate,
        hparams.gradient_clip_val,
        decay_steps,
        log_model_path,
        hparams.name,
    )
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:

        # Initializers
        model.init(sess, checkpoint_path)
        model.add_summary_graph(sess)

        for e in range(epochs):
            # Reset evaluators
            evaluator_train.reset_all_vars()
            evaluator_val.reset_all_vars()

            # Initialize iterator with train data
            sess.run(loader.train_init)
            try:
                with tqdm(total=len(train_image_paths)) as pbar:
                    while True:
                        _, loss, lengths = sess.run(
                            [model.optimize, model.loss, model.captions_len],
                            feed_dict={
                                model.frob_norm_pen: hparams.frob_norm_pen,
                                model.gor_pen: hparams.gor_pen,
                                model.keep_prob: hparams.keep_prob,
                                model.weight_decay: hparams.weight_decay,
                            },
                        )
                        evaluator_train.update_metrics(loss)
                        pbar.update(len(lengths))
                        pbar.set_postfix({"Batch loss": loss})
            except tf.errors.OutOfRangeError:
                pass

            # Initialize iterator with validation data
            sess.run(loader.val_init)
            try:
                with tqdm(total=len(val_image_paths)) as pbar:
                    while True:
                        loss, lengths, embedded_images, embedded_captions = sess.run(
                            [
                                model.loss,
                                model.captions_len,
                                model.attended_images,
                                model.attended_captions,
                            ]
                        )
                        evaluator_val.update_metrics(loss)
                        evaluator_val.update_embeddings(
                            embedded_images, embedded_captions
                        )
                        pbar.update(len(lengths))
            except tf.errors.OutOfRangeError:
                pass

            if evaluator_val.is_best_image2text_recall_at_k(recall_at):
                evaluator_val.update_best_image2text_recall_at_k()
                logger.info("=============================")
                logger.info(
                    f"Found new best on epoch {e+1} with recall at {recall_at}: "
                    f"{evaluator_val.best_image2text_recall_at_k}! Saving model..."
                )
                logger.info("=============================")
                model.save_model(sess, save_model_path)

            # Write multi_hop_attention summaries
            train_loss_summary = sess.run(
                model.train_loss_summary,
                feed_dict={model.train_loss_ph: evaluator_train.loss},
            )
            model.add_summary(sess, train_loss_summary)

            # Write validation summaries
            val_loss_summary, val_recall_at_k = sess.run(
                [model.val_loss_summary, model.val_recall_at_k_summary],
                feed_dict={
                    model.val_loss_ph: evaluator_val.loss,
                    model.val_recall_at_k_ph: evaluator_val.cur_image2text_recall_at_k,
                },
            )
            model.add_summary(sess, val_loss_summary)
            model.add_summary(sess, val_recall_at_k)


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
        args.recall_at,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.save_model_path,
        args.log_model_path,
        args.learning_rate,
        args.frob_norm_pen,
        args.attn_heads,
        args.gor_pen,
        args.weight_decay,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        description="Performs multi_hop_attention on the Microsoft COCO dataset."
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        default="hyperparameters/default_hparams.yaml",
        help="Path to an hyperparameters yaml file.",
    )
    parser.add_argument(
        "--train_images_path",
        type=str,
        default="data/MicrosoftCoco_dataset/train2014/",
        help="Path where the train images are.",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/MicrosoftCoco_dataset/annotations/captions_train2014.json",
        help="Path where the train json file with the captions and image ids.",
    )
    parser.add_argument(
        "--val_images_path",
        type=str,
        default="data/MicrosoftCoco_dataset/val2014/",
        help="Path where the validation images are.",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/MicrosoftCoco_dataset/annotations/captions_val2014.json",
        help="Path where the val json file with the captions and image ids.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to a model checkpoint."
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
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--recall_at", type=int, default=10, help="Validate on recall at K."
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="This will override the" "hparams learning rate.",
    )
    parser.add_argument(
        "--frob_norm_pen",
        type=float,
        default=None,
        help="This will override the hparams frob norm penalization rate.",
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=None,
        help="This will override the hparams attention heads.",
    )
    parser.add_argument(
        "--gor_pen",
        type=float,
        default=None,
        help="This will override the hparams gor penalization rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="This will override the hparams weight_decay penalization rate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
