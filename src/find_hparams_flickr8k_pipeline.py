import argparse
import os
import tensorflow as tf
from training.hparams_finders import Flickr8kHparamsFinder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def optimize(
    images_path: str,
    texts_path: str,
    train_imgs_file_path: str,
    val_imgs_file_path: str,
    batch_size: int,
    prefetch_size: int,
    imagenet_checkpoint_path: str,
    epochs: int,
    recall_at: int,
    max_evals: int,
    dump_hparams_path: str,
) -> None:
    """Searches for the best hyperparameters based on the validation recall at K score
    and dumps them as a yaml file.

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
        max_evals: How many times to do random sampling.
        dump_hparams_path: Where to dump the hparams.

    Returns:
        None

    """
    hparams_finder = Flickr8kHparamsFinder(
        images_path,
        texts_path,
        train_imgs_file_path,
        val_imgs_file_path,
        batch_size,
        prefetch_size,
        imagenet_checkpoint_path,
        epochs,
        recall_at,
    )
    hparams_finder.find_best(max_evals, dump_hparams_path)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    optimize(
        args.images_path,
        args.texts_path,
        args.train_imgs_file_path,
        args.val_imgs_file_path,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.epochs,
        args.recall_at,
        args.max_evals,
        args.dump_hparams_path,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser()

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
        "--train_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.trainImages.txt",
        help="Path to the file where the train images names are included.",
    )
    parser.add_argument(
        "--val_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the train images names are included.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/image_encoders/vgg_16.ckpt",
        help="Path to a model checkpoint.",
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
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )
    parser.add_argument(
        "--recall_at", type=int, default=5, help="Validate with recall at K (input)."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="How many times to sample and evaluate in order to find the best hparams.",
    )
    parser.add_argument(
        "--dump_hparams_path",
        type=str,
        default="hyperparameters/",
        help="Where to dump the found hparams.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
