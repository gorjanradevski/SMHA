import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os

from training.datasets import Flickr8kDataset, get_vocab_size
from training.hyperparameters import YParams
from training.loaders import TestLoader
from training.models import Text2ImageMatchingModel
from training.evaluators import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


inference_for_recall_at = [1, 5, 10]


def inference(
    hparams_path: str,
    images_path: str,
    texts_path: str,
    test_imgs_file_path,
    batch_size: int,
    prefetch_size: int,
    checkpoint_path: str,
) -> None:
    """Starts a training session with the Flickr8k dataset.

    Args:
        hparams_path: The path to the hyperparameters yaml file.
        images_path: A path where all the images are located.
        texts_path: Path where the text doc with the descriptions is.
        test_imgs_file_path: Path to a file with the test image names.
        batch_size: The batch size to be used.
        checkpoint_path: Path to a valid model checkpoint.

    Returns:
        None

    """
    hparams = YParams(hparams_path)
    dataset = Flickr8kDataset(images_path, texts_path, hparams.min_unk_sub)
    # Getting the vocabulary size of the train dataset
    test_image_paths, test_captions, test_captions_lengths = dataset.get_data(
        test_imgs_file_path
    )
    logger.info("Test dataset created...")
    # The number of features at the output will be: rnn_hidden_size * 2
    evaluator_test = Evaluator(len(test_image_paths), hparams.rnn_hidden_size * 2)

    logger.info("Test evaluator created...")

    # Resetting the default graph and setting the random seed
    tf.reset_default_graph()

    loader = TestLoader(
        test_image_paths,
        test_captions,
        test_captions_lengths,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")

    model = Text2ImageMatchingModel(
        hparams.seed,
        images,
        captions,
        captions_lengths,
        hparams.margin,
        hparams.rnn_hidden_size,
        get_vocab_size(Flickr8kDataset),
        hparams.embed_size,
        hparams.cell,
        hparams.layers,
        hparams.attn_size,
        hparams.opt,
        hparams.learning_rate,
        hparams.gradient_clip_val,
    )
    logger.info("Model created...")
    logger.info("Inference is starting...")

    with tf.Session() as sess:

        # Initializers
        model.init(sess, checkpoint_path, imagenet_checkpoint=False)
        try:
            with tqdm(total=len(test_image_paths)) as pbar:
                while True:
                    loss, lengths, embedded_images, embedded_captions = sess.run(
                        [
                            model.loss,
                            model.captions_len,
                            model.attended_images,
                            model.attended_captions,
                        ]
                    )
                    evaluator_test.update_metrics(loss)
                    evaluator_test.update_embeddings(embedded_images, embedded_captions)
                    pbar.update(len(lengths))
        except tf.errors.OutOfRangeError:
            pass

    for recall_at in inference_for_recall_at:
        evaluator_test.image2text_recall_at_k(recall_at)
        logger.info(f"The recall at {recall_at} is: {evaluator_test.recall_at_k}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.hparams_path,
        args.images_path,
        args.texts_path,
        args.test_imgs_file_path,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
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
        "--images_path",
        type=str,
        default="data/Flickr8k_dataset/Flicker8k_Dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--test_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the test images names are included.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/image_encoders/vgg_16.ckpt",
        help="Path to a model checkpoint.",
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