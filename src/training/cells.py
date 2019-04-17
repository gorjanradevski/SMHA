import tensorflow as tf


def cell_factory(
    seed: int, cell_type: str, num_units: int, num_layers: int, keep_prob: float
) -> tf.nn.rnn_cell:
    if cell_type == "lstm":
        return tf.nn.rnn_cell.MultiRNNCell(
            [
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units),
                    output_keep_prob=keep_prob,
                    seed=seed,
                )
                for _ in range(num_layers)
            ]
        )
    elif cell_type == "gru":
        return tf.nn.rnn_cell.MultiRNNCell(
            [
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.GRUCell(num_units),
                    output_keep_prob=keep_prob,
                    seed=seed,
                )
                for _ in range(num_layers)
            ]
        )
