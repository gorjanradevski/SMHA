import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os
import absl.logging

from utils.datasets import FlickrDataset, get_vocab_size

from vse_plusplus.loaders import TrainValLoader
from vse_plusplus.model import VsePlusPlus
from utils.evaluators import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


def train(
    images_path: str,
    texts_path: str,
    train_imgs_file_path: str,
    val_imgs_file_path: str,
    epochs: int,
    recall_at: int,
    batch_size: int,
    prefetch_size: int,
    checkpoint_path: str,
    save_model_path: str,
) -> None:
    """Starts a training session with the Flickr8k dataset.

    Args:
        images_path: A path where all the images are located.
        texts_path: Path where the text doc with the descriptions is.
        train_imgs_file_path: Path to a file with the train image names.
        val_imgs_file_path: Path to a file with the val image names.
        epochs: The number of epochs to train the model excluding the vgg.
        recall_at: Validate on recall at K.
        batch_size: The batch size to be used.
        prefetch_size: How many batches to keep on GPU ready for processing.
        checkpoint_path: Path to a valid model checkpoint.
        save_model_path: Where to save the model.

    Returns:
        None

    """
    dataset = FlickrDataset(images_path, texts_path, min_unk_sub=3)
    train_image_paths, train_captions, train_captions_lengths = dataset.get_data(
        train_imgs_file_path
    )
    val_image_paths, val_captions, val_captions_lengths = dataset.get_data(
        val_imgs_file_path
    )
    logger.info("Train dataset created...")
    logger.info("Validation dataset created...")

    evaluator_train = Evaluator()
    evaluator_val = Evaluator(len(val_image_paths), 1024)

    logger.info("Evaluators created...")

    # Resetting the default graph
    tf.reset_default_graph()

    loader = TrainValLoader(
        train_image_paths,
        train_captions,
        train_captions_lengths,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")
    model = VsePlusPlus(
        images,
        captions,
        captions_lengths,
        get_vocab_size(FlickrDataset),
        decay_after=15,
        training_size=len(train_image_paths),
        batch_size=batch_size,
    )
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:
        model.init(sess, checkpoint_path)

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
                            [model.optimizer_op, model.loss, model.captions_len],
                            feed_dict={model.is_training: True},
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
                        loss, lengths, encoded_images, encoded_captions = sess.run(
                            [
                                model.loss,
                                model.captions_len,
                                model.encoded_images,
                                model.encoded_captions,
                            ]
                        )
                        evaluator_val.update_metrics(loss)
                        evaluator_val.update_embeddings(
                            encoded_images, encoded_captions
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


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.images_path,
        args.texts_path,
        args.train_imgs_file_path,
        args.val_imgs_file_path,
        args.epochs,
        args.recall_at,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.save_model_path,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        description="Performs multi_hop_attention on the Flickr8k and Flicrk30k dataset."
        "Defaults to the Flickr8k dataset."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_Dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--train_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.trainImages.txt",
        help="Path to the file where the train images names are included.",
    )
    parser.add_argument(
        "--val_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the validation images names are included.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to a model checkpoint."
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
        default=15,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--recall_at", type=int, default=10, help="Validate on recall at K."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
