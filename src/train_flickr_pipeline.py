import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os

from img2text_matching.datasets import FlickrDataset, get_vocab_size
from img2text_matching.hyperparameters import YParams
from img2text_matching.loaders import TrainValLoader
from img2text_matching.models import Text2ImageMatchingModel
from img2text_matching.evaluators import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def train(
    hparams_path: str,
    images_path: str,
    texts_path: str,
    train_imgs_file_path: str,
    val_imgs_file_path: str,
    warm_up_epochs: int,
    train_epochs: int,
    recall_at: int,
    batch_size: int,
    prefetch_size: int,
    checkpoint_path: str,
    imagenet_checkpoint: bool,
    save_model_path: str,
    log_model_path: str,
    learning_rate: float = None,
) -> None:
    """Starts a training session with the Flickr8k dataset.

    Args:
        hparams_path: The path to the hyperparameters yaml file.
        images_path: A path where all the images are located.
        texts_path: Path where the text doc with the descriptions is.
        train_imgs_file_path: Path to a file with the train image names.
        val_imgs_file_path: Path to a file with the val image names.
        warm_up_epochs: The number of epochs to train the model exluding the vgg.
        train_epochs: The number of epochs to train the full model.
        recall_at: Validate on recall at K.
        batch_size: The batch size to be used.
        prefetch_size: How many batches to keep on GPU ready for processing.
        checkpoint_path: Path to a valid model checkpoint.
        imagenet_checkpoint: Whether the checkpoint points to an imagenet model.
        save_model_path: Where to save the model.
        log_model_path: Where to log the summaries.
        learning_rate: An optional learning rate. If provided update the one in
        hparams.

    Returns:
        None

    """
    hparams = YParams(hparams_path)
    # If learning rate is provided update the hparams learning rate
    if learning_rate is not None:
        hparams.set_hparam("learning_rate", learning_rate)
    dataset = FlickrDataset(images_path, texts_path, hparams.min_unk_sub)
    train_image_paths, train_captions, train_captions_lengths = dataset.get_data(
        train_imgs_file_path
    )
    val_image_paths, val_captions, val_captions_lengths = dataset.get_data(
        val_imgs_file_path
    )
    logger.info("Train dataset created...")
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
        train_captions_lengths,
        val_image_paths,
        val_captions,
        val_captions_lengths,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")

    model = Text2ImageMatchingModel(
        images,
        captions,
        captions_lengths,
        hparams.margin,
        hparams.rnn_hidden_size,
        get_vocab_size(FlickrDataset),
        hparams.layers,
        hparams.attn_size,
        hparams.attn_heads,
        hparams.learning_rate,
        hparams.gradient_clip_val,
        hparams.batch_hard,
        log_model_path,
        hparams.name,
    )
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:

        # Initializers
        model.init(sess, checkpoint_path, imagenet_checkpoint)
        model.add_summary_graph(sess)

        for e in range(warm_up_epochs + train_epochs):
            # Reset evaluators
            evaluator_train.reset_all_vars()
            evaluator_val.reset_all_vars()

            # Initialize iterator with train data
            sess.run(loader.train_init)
            try:
                with tqdm(total=len(train_image_paths)) as pbar:
                    while True:
                        _, loss, lengths = sess.run(
                            [
                                model.optimize_full
                                if e > warm_up_epochs
                                else model.optimize_no_vgg,
                                model.loss,
                                model.captions_len,
                            ],
                            feed_dict={
                                model.keep_prob: hparams.keep_prob,
                                model.weight_decay: hparams.weight_decay,
                                model.frob_norm_pen: hparams.frob_norm_pen,
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

            # Write img2text_matching summaries
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
        args.images_path,
        args.texts_path,
        args.train_imgs_file_path,
        args.val_imgs_file_path,
        args.warm_up_epochs,
        args.train_epochs,
        args.recall_at,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.imagenet_checkpoint,
        args.save_model_path,
        args.log_model_path,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        description="Performs img2text_matching on the Flickr8k and Flicrk30k dataset."
        "Defaults to the Flickr8k dataset."
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        default="hyperparameters/default_hparams.yaml",
        help="Path to a hyperparameters yaml file.",
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
        "--checkpoint_path",
        type=str,
        default="models/image_encoders/vgg_19.ckpt",
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
        "--warm_up_epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=15,
        help="The number of epochs to train the full model.",
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="This will override the" "hparams learning rate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
