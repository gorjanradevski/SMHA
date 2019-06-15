import tensorflow as tf
import logging
from typing import Tuple

from utils.constants import embedding_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHopAttentionModel:
    def __init__(
        self,
        images: tf.Tensor,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        margin: int,
        rnn_hidden_size: int,
        vocab_size: int,
        num_layers: int,
        attn_size: int,
        attn_heads: int,
        learning_rate: float = 0.0,
        clip_value: int = 0,
        decay_steps: float = 0.0,
        batch_hard: bool = False,
        use_gor: bool = False,
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
        self.attended_images, self.image_alphas = self.attention_graph(
            attn_size, attn_heads, self.image_encoded, "siamese_attention"
        )
        # Reusing the same variables that were used for the images
        self.attended_captions, self.text_alphas = self.attention_graph(
            attn_size, attn_heads, self.text_encoded, "siamese_attention"
        )
        logger.info("Attention graph created...")
        self.loss = self.compute_loss(
            margin, batch_hard, use_gor, attn_heads, rnn_hidden_size
        )

        self.optimize = self.apply_gradients_op(
            self.loss, learning_rate, clip_value, decay_steps
        )
        self.saver_loader = tf.train.Saver()
        logger.info("Graph creation finished...")

    @staticmethod
    def image_encoder_graph(images: tf.Tensor, rnn_hidden_size: int) -> tf.Tensor:
        """Extract higher level features from the image using a conv vgg19 pretrained on
        Image net.

        Args:
            images: The input images.
            rnn_hidden_size: The hidden size of its text counterpart.

        Returns:
            The encoded image.

        """
        with tf.variable_scope("image_encoder"):
            vgg19 = tf.keras.applications.VGG19(
                include_top=False, weights="imagenet", pooling=None
            )
            for layer in vgg19.layers:
                layer.trainable = False
            output = vgg19(images)
            flatten = tf.reshape(output, (-1, output.shape[3]))
            # As per: https://arxiv.org/abs/1502.01852
            project_layer = tf.layers.dense(
                flatten,
                rnn_hidden_size,
                kernel_initializer=tf.glorot_uniform_initializer(),
            )

            return tf.reshape(
                project_layer, (-1, output.shape[1] * output.shape[2], rnn_hidden_size)
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
            # As per: https://arxiv.org/abs/1711.09160
            embeddings = tf.get_variable(
                name="embeddings",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
            )
            inputs = tf.nn.embedding_lookup(embeddings, captions)
            # As per: https://arxiv.org/abs/1406.1078
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

            return tf.add(output_fw, output_bw)

    @staticmethod
    def attention_graph(
        attn_size: int, attn_heads: int, encoded_input: tf.Tensor, scope: str
    ):
        """Applies attention on the encoded image and the encoded text.

        As per: https://arxiv.org/abs/1703.03130

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

            # As per: http://proceedings.mlr.press/v9/glorot10a.html
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
            # [B, A_heads * H] normalized output
            output = tf.math.l2_normalize(output, axis=1)

            return output, alphas

    @staticmethod
    def compute_frob_norm(attention_weights: tf.Tensor, attn_heads: int) -> tf.Tensor:
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

    @staticmethod
    def triplet_loss(scores: tf.Tensor, margin, batch_hard):
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

        return tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)

    @staticmethod
    def gor(scores: tf.Tensor, joint_space: int):
        """Computes the global orthogonal regularization term as per:

        https://arxiv.org/abs/1708.06320

        Args:
            scores: The per batch similarity scores.
            joint_space: The size of the joint space.

        Returns:
            The global orthogonal regularization term.

        """
        scores_diag = tf.linalg.set_diag(scores, tf.zeros(tf.shape(scores)[0]))
        non_zero = tf.cast(tf.count_nonzero(scores_diag), tf.float32)
        m1 = tf.reduce_sum(scores_diag) / non_zero
        m2 = tf.reduce_sum(tf.pow(scores_diag, 2)) / non_zero
        d = 1 / joint_space

        return tf.pow(m1, 2) + tf.maximum(0.0, m2 - d)

    def compute_loss(
        self,
        margin: float,
        batch_hard: bool,
        use_gor: bool,
        attn_heads: int,
        rnn_hidden_size: int,
    ) -> tf.Tensor:
        """Computes the final loss of the model.

        1. Computes the Triplet loss: https://arxiv.org/abs/1707.05612
        2. Computes the Frob norm of the of the AA^T - I (image embeddings).
        3. Computes the Frob norm of the of the AA^T - I (text embeddings).
        4. Adds all together to compute the loss.

        Args:
            margin: The contrastive margin.
            attn_heads: The number of attention heads.
            batch_hard: Whether to train only on the hard negatives.
            use_gor: Whether to compute the global orthogonal regularization.
            rnn_hidden_size: The size of the cell of the rnn.

        Returns:
            The final loss to be optimized.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(
                self.attended_images, self.attended_captions, transpose_b=True
            )
            triplet_loss = self.triplet_loss(scores, margin, batch_hard)
            gor = 0.0
            if use_gor:
                gor = self.gor(scores, attn_heads * rnn_hidden_size)

            pen_image_alphas = (
                self.compute_frob_norm(self.image_alphas, attn_heads)
                * self.frob_norm_pen
            )
            pen_text_alphas = (
                self.compute_frob_norm(self.text_alphas, attn_heads)
                * self.frob_norm_pen
            )

            return triplet_loss + gor + pen_image_alphas + pen_text_alphas

    def apply_gradients_op(
        self, loss: tf.Tensor, learning_rate: float, clip_value: int, decay_steps: float
    ) -> tf.Operation:
        """Applies the gradients on the variables.

        Args:
            loss: The computed loss.
            learning_rate: The optimizer learning rate.
            clip_value: The clipping value.
            decay_steps: Decay the learning rate every decay_steps.

        Returns:
            An operation node to be executed in order to apply the computed gradients.

        """
        with tf.variable_scope(name_or_scope="optimizer"):
            learning_rate = tf.train.exponential_decay(
                learning_rate,
                self.global_step,
                decay_steps,
                0.5,
                staircase=False,
                name="lr_decay",
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

            return optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step
            )

    def init(self, sess: tf.Session, checkpoint_path: str = None) -> None:
        """Initializes all variables in the graph.

        Args:
            sess: The active session.
            checkpoint_path: Path to a valid checkpoint.

        Returns:
            None

        """
        sess.run(tf.global_variables_initializer())
        if checkpoint_path is not None:
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