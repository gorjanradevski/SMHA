import argparse
import os
import tensorflow as tf
from multi_hop_attention.hyperparameters import PascalHparamsFinder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def optimize(
    images_path: str,
    texts_path: str,
    batch_size: int,
    prefetch_size: int,
    imagenet_checkpoint_path: str,
    epochs: int,
    recall_at: int,
    num_iters: int,
    hparams_path: str,
    trials_path: str,
) -> None:
    """Searches for the best hyperparameters based on the validation recall at K score
    and dumps them as a yaml file.

    Args:
        images_path: The path to the images.
        texts_path: The path to the captions.
        batch_size: The batch size that will be used to conduct the experiments.
        prefetch_size: The prefetching size when running on GPU.
        imagenet_checkpoint_path: The checkpoint to the pretrained imagenet weights.
        epochs: The number of epochs per experiment.
        recall_at: Validate on recall at K.
        num_iters: How many times to do random sampling.
        hparams_path: Where to dump the hparams.
        trials_path: Read/write the trials object.

    Returns:
        None

    """
    hparams_finder = PascalHparamsFinder(
        images_path,
        texts_path,
        batch_size,
        prefetch_size,
        imagenet_checkpoint_path,
        epochs,
        recall_at,
    )
    hparams_finder.find_best(num_iters, hparams_path, trials_path)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    optimize(
        args.images_path,
        args.texts_path,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.epochs,
        args.recall_at,
        args.num_iters,
        args.hparams_path,
        args.trials_path,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        description="Searches for the best model parameters on the Pascal sentences and"
        "datasets."
    )

    parser.add_argument(
        "--images_path",
        type=str,
        default="data/Pascal_sentences_dataset/dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Pascal_sentences_dataset/sentence",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/image_encoders/vgg_19.ckpt",
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
        "--recall_at", type=int, default=10, help="Find best hparams on recall at K."
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="How many times to sample and evaluate in order to find the best hparams.",
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        default="hyperparameters/experiment.yaml",
        help="Where to dump the found hparams.",
    )
    parser.add_argument(
        "--trials_path",
        type=str,
        default="trials/experiment.pkl",
        help="From where to read or where to dump the trials object.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
