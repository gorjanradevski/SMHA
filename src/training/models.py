import tensorflow as tf
import logging
from typing import Tuple

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope

from training.cells import cell_factory
from training.optimizers import optimizer_factory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2ImageMatchingModel:
    def __init__(
        self,
        seed: int,
        images: tf.Tensor,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        margin: float,
        rnn_hidden_size: int,
        vocab_size: int,
        embedding_size: int,
        cell_type: str,
        num_layers: int,
        attn_size: int,
        attn_hops: int,
        frob_norm_pen: float,
        optimizer_type: str,
        learning_rate: float,
        clip_value: int,
        log_dir: str = "",
        name: str = "",
    ):
        # Name of the model
        self.name = name
        # Get images, captions, lengths and labels
        self.images = images
        self.captions = captions
        self.captions_len = captions_len
        # Create summary writers
        if log_dir != "":
            self.file_writer = tf.summary.FileWriter(log_dir + self.name)
            self.train_loss_ph, self.train_loss_summary = self.create_summary(
                "train_loss"
            )
            self.val_loss_ph, self.val_loss_summary = self.create_summary("val_loss")
            self.val_recall_at_k_ph, self.val_recall_at_k_summary = self.create_summary(
                "val_recall_at_k"
            )
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # Create dropout and weight decay placeholder
        self.keep_prob = tf.placeholder_with_default(1.0, None, name="keep_prob")
        self.weight_decay = tf.placeholder_with_default(0.0, None, name="weight_decay")
        # Build model
        self.image_encoded = self.image_encoder_graph(self.images, rnn_hidden_size)
        logger.info("Image encoder graph created...")
        self.text_encoded = self.text_encoder_graph(
            seed,
            self.captions,
            self.captions_len,
            vocab_size,
            embedding_size,
            cell_type,
            rnn_hidden_size,
            num_layers,
            self.keep_prob,
        )
        logger.info("Text encoder graph created...")
        self.attended_images, self.image_alphas = self.join_attention_graph(
            attn_size, attn_hops, self.image_encoded
        )
        # Reusing the same variables that were used for the images
        self.attended_captions, self.text_alphas = self.join_attention_graph(
            attn_size, attn_hops, self.text_encoded
        )
        logger.info("Attention graph created...")
        self.loss = self.compute_loss(margin, attn_hops, frob_norm_pen)
        self.optimize = self.apply_gradients_op(
            self.loss, optimizer_type, learning_rate, clip_value
        )
        # Imagenet graph loader and graph saver/loader
        self.image_encoder_loader = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16")
        )
        self.saver_loader = tf.train.Saver()
        logger.info("Graph creation finished...")

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

        flatten = tf.reshape(net, (-1, net.shape[3]))
        project_layer = tf.layers.dense(flatten, 2 * rnn_hidden_size)
        reshaped = tf.reshape(
            project_layer, (-1, net.shape[1] * net.shape[2], 2 * rnn_hidden_size)
        )

        return reshaped

    @staticmethod
    def text_encoder_graph(
        seed: int,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        vocab_size: int,
        embedding_size: int,
        cell_type: str,
        rnn_hidden_size: int,
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
            rnn_hidden_size: The size of the weight matrix in the cell.
            num_layers: The number of layers of the rnn.
            keep_prob: The dropout probability (1.0 means keep everything)

        Returns:
            The encoded the text.

        """
        with tf.variable_scope("text_encoder"):
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                dtype=tf.float32,
                trainable=True,
                name="embeddings",
            )
            inputs = tf.nn.embedding_lookup(embeddings, captions)
            cell_fw = cell_factory(
                seed, cell_type, rnn_hidden_size, num_layers, keep_prob
            )
            cell_bw = cell_factory(
                seed, cell_type, rnn_hidden_size, num_layers, keep_prob
            )
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=captions_len, dtype=tf.float32
            )
        return tf.concat([output_fw, output_bw], axis=2)

    @staticmethod
    def join_attention_graph(attn_size: int, attn_hops: int, encoded_input: tf.Tensor):
        """Applies the same attention on the encoded image and the encoded text.

        As per: https://arxiv.org/pdf/1703.03130.pdf

        The "A structured self-attentative sentence embedding" paper goes through
        the attention mechanism applied here.

        Args:
            attn_size: The size of the attention.
            attn_hops: How many hops of attention to apply.
            encoded_input: The encoded input, can be both the image and the text.

        Returns:
            Attended output.

        """
        with tf.variable_scope(name_or_scope="joint_attention", reuse=tf.AUTO_REUSE):
            # Shape parameters
            time_steps = tf.shape(encoded_input)[1]
            hidden_size = encoded_input.get_shape()[2].value

            # Trainable parameters
            w_omega = tf.get_variable(
                name="w_omega",
                initializer=tf.random_normal([hidden_size, attn_size], stddev=0.1),
            )
            b_omega = tf.get_variable(
                name="b_omega", initializer=tf.random_normal([attn_size], stddev=0.1)
            )
            u_omega = tf.get_variable(
                name="u_omega",
                initializer=tf.random_normal([attn_size, attn_hops], stddev=0.1),
            )

            # Apply attention
            # [B * T, H]
            encoded_input_reshaped = tf.reshape(encoded_input, [-1, hidden_size])
            # [B * T, A_size]
            v = tf.tanh(tf.matmul(encoded_input_reshaped, w_omega) + b_omega)
            # [B * T, A_hops]
            vu = tf.matmul(v, u_omega)
            # [B, T, A_hops]
            vu = tf.reshape(vu, [-1, time_steps, attn_hops])
            # [B, A_hops, T]
            vu_transposed = tf.transpose(vu, [0, 2, 1])
            # [B, A_hops, T]
            alphas = tf.nn.softmax(vu_transposed, name="alphas", axis=2)
            # [B, A_hops, H]
            output = tf.matmul(alphas, encoded_input)
            # [B, A_hops * H]
            output = tf.layers.flatten(output)

            return output, alphas

    @staticmethod
    def compute_frob_norm(attention_weights: tf.Tensor, attn_hops: int):
        """Computes the Frobenius norm of the attention weights tensor.

        Args:
            attention_weights: The attention weights.
            attn_hops: The number of attention hops

        Returns:
            The Frobenius norm of the attention weights tensor.

        """
        attn_w_dot_product = tf.matmul(
            attention_weights, tf.transpose(attention_weights, [0, 2, 1])
        )
        identity_matrix = tf.reshape(
            tf.tile(tf.eye(attn_hops), [tf.shape(attention_weights)[0], 1]),
            [-1, attn_hops, attn_hops],
        )
        return tf.reduce_mean(
            tf.square(
                tf.norm(attn_w_dot_product - identity_matrix, axis=[-2, -1], ord="fro")
            )
        )

    def compute_loss(
        self, margin: float, attn_hops: int, frob_norm_pen: float
    ) -> tf.Tensor:
        """Computes the triplet loss.

        Args:
            margin: The contrastive margin.
            attn_hops: The number of attention hops.
            frob_norm_pen: The weight assigned to the Frob norm.

        Returns:
            The contrastive loss using the batch-all strategy.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(
                self.attended_images, self.attended_captions, transpose_b=True
            )
            diagonal = tf.diag_part(scores)

            # compare every diagonal score to scores in its column
            # (i.e, all contrastive images for each sentence)
            cost_s = tf.maximum(0.0, margin - diagonal + scores)
            # compare every diagonal score to scores in its row
            # (i.e, all contrastive sentences for each image)
            cost_im = tf.maximum(0.0, margin - tf.reshape(diagonal, [-1, 1]) + scores)

            # clear diagonals
            cost_s = tf.linalg.set_diag(cost_s, tf.zeros(tf.shape(cost_s)[0]))
            cost_im = tf.linalg.set_diag(cost_im, tf.zeros(tf.shape(cost_im)[0]))

            loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)

            l2 = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if "bias" not in v.name
                    ]
                )
                * self.weight_decay
            )

            pen_image_alphas = (
                self.compute_frob_norm(self.image_alphas, attn_hops) * frob_norm_pen
            )
            pen_text_alphas = (
                self.compute_frob_norm(self.text_alphas, attn_hops) * frob_norm_pen
            )

            return loss + l2 + pen_image_alphas + pen_text_alphas

    def apply_gradients_op(
        self,
        loss: tf.Tensor,
        optimizer_type: str,
        learning_rate: float,
        clip_value: int,
    ) -> tf.Operation:
        """Applies the gradients on the variables.

        Args:
            loss: The computed loss.
            optimizer_type: The type of the optmizer.
            learning_rate: The optimizer learning rate.
            clip_value: The clipping value.

        Returns:
            An operation node to be executed in order to apply the computed gradients.

        """
        with tf.variable_scope(name_or_scope="optimizer"):
            optimizer = optimizer_factory(optimizer_type, learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

            return optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step
            )

    def init(
        self, sess: tf.Session, checkpoint_path: str, imagenet_checkpoint: bool
    ) -> None:
        """Initializes all variables in the graph.

        Args:
            sess: The active session.
            checkpoint_path: Path to a valid checkpoint.
            imagenet_checkpoint: Whether the checkpoint points to a model pretrained on
            imagenet or a full model.

        Returns:
            None

        """
        sess.run(tf.global_variables_initializer())
        if imagenet_checkpoint:
            self.image_encoder_loader.restore(sess, checkpoint_path)
        else:
            self.saver_loader.restore(sess, checkpoint_path)

    def add_summary_graph(self, sess: tf.Session) -> None:
        """Adds the graph to tensorboard.

        Args:
            sess: The active session.

        Returns:
            None

        """
        self.file_writer.add_graph(sess.graph)

    @staticmethod
    def create_summary(name: str) -> Tuple[tf.placeholder, tf.summary.scalar]:
        """Creates summary placeholder and node.

        Args:
            name: The name of the summary.

        Returns:
            The summary placeholder and it's node counterpart.

        """
        input_ph = tf.placeholder(tf.float32, shape=None, name=name + "_pl")
        summary = tf.summary.scalar(name, input_ph)

        return input_ph, summary

    def add_summary(self, sess: tf.Session, value: float) -> None:
        """Writes the summary to tensorboard.

        Args:
            sess: The active session.
            value: The value to write.

        Returns:
            None

        """
        self.file_writer.add_summary(
            value, tf.train.global_step(sess, self.global_step)
        )

    def save_model(self, sess: tf.Session, save_path: str) -> None:
        """Dumps the model definition.

        Args:
            sess: The active session.
            save_path: Where to save the model.

        Returns:

        """
        self.saver_loader.save(sess, save_path + self.name)
