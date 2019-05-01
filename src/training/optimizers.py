import tensorflow as tf


def optimizer_factory(optimizer_type: str, learning_rate: float) -> tf.train.Optimizer:
    """Returns a valid optimizer.

    Args:
        optimizer_type: The type of the optimizer.
        learning_rate: The learning rate of the optimizer.

    Returns:
        An optimizer.

    """
    if optimizer_type == "adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_type == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_type == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise ValueError("Optimizer type not recognized")
