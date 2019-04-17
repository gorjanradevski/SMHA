import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope

from training.cells import cell_factory


class Text2ImageMatchingModel:
    def __init__(self):
        pass

    @staticmethod
    def image_encoder_graph(images: tf.Tensor, rnn_hidden_size: int) -> tf.Tensor:
        """Extract higher level features from the image using a conv net pretrained on
        Image net.

        As per: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/
        slim/python/slim/nets/vgg.py

        Args:
            images: The input images.
            rnn_hidden_size: The hidden size of its text counterpart.

        Returns:
            The encoded image.

        """
        with variable_scope.variable_scope("vgg_16", "vgg_16", [images]) as sc:
            end_points_collection = sc.original_name_scope + "_end_points"
            with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=end_points_collection,
            ):
                net = layers_lib.repeat(
                    images, 2, layers.conv2d, 64, [3, 3], scope="conv1", trainable=False
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool1")
                net = layers_lib.repeat(
                    net, 2, layers.conv2d, 128, [3, 3], scope="conv2", trainable=False
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool2")
                net = layers_lib.repeat(
                    net, 3, layers.conv2d, 256, [3, 3], scope="conv3", trainable=False
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool3")
                net = layers_lib.repeat(
                    net, 3, layers.conv2d, 512, [3, 3], scope="conv4", trainable=False
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool4")
                net = layers_lib.repeat(
                    net, 3, layers.conv2d, 512, [3, 3], scope="conv5", trainable=False
                )
                image_feature_extractor = layers_lib.max_pool2d(
                    net, [2, 2], scope="pool5"
                )
                project_layer = tf.layers.dense(
                    image_feature_extractor, 2 * rnn_hidden_size
                )
                return tf.reshape(
                    project_layer,
                    [
                        -1,
                        project_layer.shape[1] * project_layer.shape[2],
                        2 * rnn_hidden_size,
                    ],
                )

    @staticmethod
    def text_encoder_graph(
        seed: int,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        vocab_size: int,
        embedding_size: int,
        cell_type: str,
        num_units: int,
        num_layers: int,
        keep_prob: float,
    ):
        """Encodes the text it gets as input using a bidirectional rnn.

        Args:
            seed: The random seed.
            captions: The inputs.
            captions_len: The length of the inputs.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            cell_type: The cell type.
            num_units: The size of the weight matrix in the cell.
            num_layers: The number of layers of the rnn.
            keep_prob: The dropout probability (1.0 means keep everything)

        Returns:
            The encoded the text.

        """
        embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            dtype=tf.float32,
            trainable=True,
        )
        inputs = tf.nn.embedding_lookup(embeddings, captions)
        cell_fw = cell_factory(seed, cell_type, num_units, num_layers, keep_prob)
        cell_bw = cell_factory(seed, cell_type, num_units, num_layers, keep_prob)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, inputs, sequence_length=captions_len, dtype=tf.float32
        )
        return tf.concat([output_fw, output_bw], axis=2)

    @staticmethod
    def join_attention_graph(
        seed: int,
        attn_size1: int,
        attn_size2: int,
        encoded_input: tf.Tensor,
        reuse=False,
    ):
        """Applies the same attention on the encoded image and the encoded text.

        As per: https://arxiv.org/pdf/1703.03130.pdf

        The "A structured self-attentative sentence embedding" paper goes through
        the attention mechanism applied here.

        Args:
            seed: The random seed to initialize the weights.
            attn_size1: The size of the first projection.
            attn_size2: The size of the second projection.
            encoded_input: The encoded input, can be both the image and the text.
            reuse: Whether to reuse the variables during the second creation.

        Returns:
            Attended output.

        """
        project = tf.layers.dense(
            encoded_input,
            attn_size1,
            activation=tf.nn.tanh,
            kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
            bias_initializer=tf.zeros_initializer(),
            reuse=reuse,
        )
        alphas = tf.layers.dense(
            project,
            attn_size2,
            activation=tf.nn.softmax,
            kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
            bias_initializer=tf.zeros_initializer(),
            reuse=reuse,
        )
        return tf.matmul(tf.transpose(encoded_input, [0, 2, 1]), alphas)
