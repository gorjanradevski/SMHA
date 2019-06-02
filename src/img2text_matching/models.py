import tensorflow as tf
import logging
from typing import Tuple
from tensorflow.contrib import slim

from utils.constants import embedding_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2ImageMatchingModel:
    def __init__(
        self,
        images: tf.Tensor,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        margin: float,
        rnn_hidden_size: int,
        vocab_size: int,
        num_layers: int,
        attn_size: int,
        attn_heads: int,
        learning_rate: float,
        clip_value: int,
        batch_hard: bool,
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
        self.frob_norm_pen = tf.placeholder_with_default(
            0.0, None, name="frob_norm_pen"
        )
        # Build model
        self.image_encoded = self.image_encoder_graph(self.images, rnn_hidden_size)
        logger.info("Image encoder graph created...")
        self.text_encoded = self.text_encoder_graph(
            self.captions,
            self.captions_len,
            vocab_size,
            rnn_hidden_size,
            num_layers,
            self.keep_prob,
        )
        logger.info("Text encoder graph created...")
        self.attended_images, self.image_alphas = self.join_attention_graph(
            attn_size, attn_heads, self.image_encoded, "siamese_attention"
        )
        # Reusing the same variables that were used for the images
        self.attended_captions, self.text_alphas = self.join_attention_graph(
            attn_size, attn_heads, self.text_encoded, "siamese_attention"
        )
        logger.info("Attention graph created...")
        self.loss = self.compute_loss(margin, attn_heads, batch_hard)
        self.optimize = self.apply_gradients_op(self.loss, learning_rate, clip_value)
        # ImageNet graph loader and graph saver/loader
        self.image_encoder_loader = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19")
        )
        self.saver_loader = tf.train.Saver()
        logger.info("Graph creation finished...")

    @staticmethod
    def image_encoder_graph(images: tf.Tensor, rnn_hidden_size: int) -> tf.Tensor:
        """Extract higher level features from the image using a conv net pretrained on
        Image net.

        https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py

        Args:
            images: The input images.
            rnn_hidden_size: The hidden size of its text counterpart.

        Returns:
            The encoded image.

        """
        with tf.variable_scope("vgg_19", "vgg_19", [images]) as sc:
            end_points_collection = sc.original_name_scope + "_end_points"
            # Collect outputs for conv2d and max_pool2d.
            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d],
                outputs_collections=end_points_collection,
            ):
                net = slim.repeat(
                    images, 2, slim.conv2d, 64, [3, 3], scope="conv1", trainable=False
                )
                net = slim.max_pool2d(net, [2, 2], scope="pool1")
                net = slim.repeat(
                    net, 2, slim.conv2d, 128, [3, 3], scope="conv2", trainable=False
                )
                net = slim.max_pool2d(net, [2, 2], scope="pool2")
                net = slim.repeat(
                    net, 4, slim.conv2d, 256, [3, 3], scope="conv3", trainable=False
                )
                net = slim.max_pool2d(net, [2, 2], scope="pool3")
                net = slim.repeat(
                    net, 4, slim.conv2d, 512, [3, 3], scope="conv4", trainable=False
                )
                net = slim.max_pool2d(net, [2, 2], scope="pool4")
                net = slim.repeat(
                    net, 4, slim.conv2d, 512, [3, 3], scope="conv5", trainable=False
                )
        with tf.variable_scope("image_encoder"):
            flatten = tf.reshape(net, (-1, net.shape[3]))
            project_layer = tf.layers.dense(
                flatten,
                rnn_hidden_size,
                kernel_initializer=tf.glorot_uniform_initializer(),
                activation=tf.nn.relu,
            )

            return tf.reshape(
                project_layer, (-1, net.shape[1] * net.shape[2], rnn_hidden_size)
            )

    @staticmethod
    def text_encoder_graph(
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        vocab_size: int,
        rnn_hidden_size: int,
        num_layers: int,
        keep_prob: float,
    ):
        """Encodes the text it gets as input using a bidirectional rnn.

        Args:
            captions: The inputs.
            captions_len: The length of the inputs.
            vocab_size: The size of the vocabulary.
            rnn_hidden_size: The size of the weight matrix in the cell.
            num_layers: The number of layers of the rnn.
            keep_prob: The dropout probability (1.0 means keep everything)

        Returns:
            The encoded text.

        """
        with tf.variable_scope(name_or_scope="text_encoder"):
            # As per: https://arxiv.org/pdf/1711.09160.pdf
            embeddings = tf.get_variable(
                name="embeddings",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
            )
            inputs = tf.nn.embedding_lookup(embeddings, captions)
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.GRUCell(rnn_hidden_size),
                        output_keep_prob=keep_prob,
                    )
                    for _ in range(num_layers)
                ]
            )
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.GRUCell(rnn_hidden_size),
                        output_keep_prob=keep_prob,
                    )
                    for _ in range(num_layers)
                ]
            )
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=captions_len, dtype=tf.float32
            )

            return tf.add(output_fw, output_bw) / 2

    @staticmethod
    def join_attention_graph(
        attn_size: int, attn_heads: int, encoded_input: tf.Tensor, scope: str
    ):
        """Applies attention on the encoded image and the encoded text.

        As per: https://arxiv.org/pdf/1703.03130.pdf

        The "A structured self-attentative sentence embedding" paper goes through
        the attention mechanism applied here.

        Args:
            attn_size: The size of the attention.
            attn_heads: How many hops of attention to apply.
            encoded_input: The encoded input, can be both the image and the text.
            scope: The scope of the graph block.

        Returns:
            Attended output.

        """
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Shape parameters
            time_steps = tf.shape(encoded_input)[1]
            hidden_size = encoded_input.get_shape()[2].value

            # As per: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            # Trainable parameters
            w_omega = tf.get_variable(
                name="w_omega",
                shape=[hidden_size, attn_size],
                initializer=tf.glorot_uniform_initializer(),
            )
            b_omega = tf.get_variable(
                name="b_omega", shape=[attn_size], initializer=tf.zeros_initializer()
            )
            u_omega = tf.get_variable(
                name="u_omega",
                shape=[attn_size, attn_heads],
                initializer=tf.glorot_uniform_initializer(),
            )

            # Apply attention
            # [B * T, H]
            encoded_input_reshaped = tf.reshape(encoded_input, [-1, hidden_size])
            # [B * T, A_size]
            v = tf.tanh(tf.matmul(encoded_input_reshaped, w_omega) + b_omega)
            # [B * T, A_heads]
            vu = tf.matmul(v, u_omega)
            # [B, T, A_heads]
            vu = tf.reshape(vu, [-1, time_steps, attn_heads])
            # [B, A_heads, T]
            vu_transposed = tf.transpose(vu, [0, 2, 1])
            # [B, A_heads, T]
            alphas = tf.nn.softmax(vu_transposed, name="alphas", axis=2)
            # [B, A_heads, H]
            output = tf.matmul(alphas, encoded_input)
            # [B, A_heads * H]
            output = tf.layers.flatten(output)

            return output, alphas

    @staticmethod
    def compute_frob_norm(attention_weights: tf.Tensor, attn_heads: int):
        """Computes the Frobenius norm of the attention weights tensor.

        Args:
            attention_weights: The attention weights.
            attn_heads: The number of attention hops.

        Returns:
            The Frobenius norm of the attention weights tensor.

        """
        attn_w_dot_product = tf.matmul(
            attention_weights, tf.transpose(attention_weights, [0, 2, 1])
        )
        identity_matrix = tf.reshape(
            tf.tile(tf.eye(attn_heads), [tf.shape(attention_weights)[0], 1]),
            [-1, attn_heads, attn_heads],
        )
        return tf.reduce_mean(
            tf.square(
                tf.norm(attn_w_dot_product - identity_matrix, axis=[-2, -1], ord="fro")
            )
        )

    def compute_loss(
        self, margin: float, attn_heads: int, batch_hard: bool
    ) -> tf.Tensor:
        """Computes the triplet loss based on:

        https://arxiv.org/pdf/1707.05612.pdf

        1. Computes the triplet loss between the image and text embeddings.
        2. Computes the Frob norm of the of the AA^T - I (image embeddings).
        3. Computes the Frob norm of the of the AA^T - I (text embeddings).
        4. Computes the L2 norm of the trainable weight matrices.
        5. Adds all together to compute the loss.

        Args:
            margin: The contrastive margin.
            attn_heads: The number of attention hops.
            batch_hard: Whether to train only on hard negatives.

        Returns:
            The triplet loss using the batch-hard strategy.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(
                self.attended_images, self.attended_captions, transpose_b=True
            )
            diagonal = tf.diag_part(scores)

            # Compare every diagonal score to scores in its column
            # All contrastive images for each sentence
            cost_s = tf.maximum(0.0, margin - tf.reshape(diagonal, [-1, 1]) + scores)
            # Compare every diagonal score to scores in its row
            # All contrastive sentences for each image
            cost_im = tf.maximum(0.0, margin - diagonal + scores)

            # Clear diagonals
            cost_s = tf.linalg.set_diag(cost_s, tf.zeros(tf.shape(cost_s)[0]))
            cost_im = tf.linalg.set_diag(cost_im, tf.zeros(tf.shape(cost_im)[0]))

            if batch_hard:
                # For each positive pair (i,s) pick the hardest contrastive image
                cost_s = tf.reduce_max(cost_s, axis=1)
                # For each positive pair (i,s) pick the hardest contrastive sentence
                cost_im = tf.reduce_max(cost_im, axis=0)

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
                self.compute_frob_norm(self.image_alphas, attn_heads)
                * self.frob_norm_pen
            )
            pen_text_alphas = (
                self.compute_frob_norm(self.text_alphas, attn_heads)
                * self.frob_norm_pen
            )

            return loss + l2 + pen_image_alphas + pen_text_alphas

    def apply_gradients_op(
        self, loss: tf.Tensor, learning_rate: float, clip_value: int
    ) -> tf.Operation:
        """Applies the gradients on the variables.

        Args:
            loss: The computed loss.
            learning_rate: The optimizer learning rate.
            clip_value: The clipping value.

        Returns:
            An operation node to be executed in order to apply the computed gradients.

        """
        with tf.variable_scope(name_or_scope="optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
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
