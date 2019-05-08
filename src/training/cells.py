import tensorflow as tf


def cell_factory(
    cell_type: str, rnn_hidden_size: int, num_layers: int, keep_prob: float
) -> tf.nn.rnn_cell:
    """Returns a valid multi layer cell to be used in the rnn.

    Args:
        cell_type: The type of the cell.
        rnn_hidden_size: The size of the weight matrix in the cell.
        num_layers: The number of layers.
        keep_prob: The dropout probability.

    Returns:
        A cell.

    """
    if cell_type == "lstm":
        return tf.nn.rnn_cell.MultiRNNCell(
            [
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(rnn_hidden_size), output_keep_prob=keep_prob
                )
                for _ in range(num_layers)
            ]
        )
    elif cell_type == "gru":
        return tf.nn.rnn_cell.MultiRNNCell(
            [
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.GRUCell(rnn_hidden_size), output_keep_prob=keep_prob
                )
                for _ in range(num_layers)
            ]
        )
    else:
        raise ValueError("Cell type not recognized")
