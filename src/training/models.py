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
        labels: tf.Tensor,
        margin: float,
        rnn_hidden_size: int,
        vocab_size: int,
        embedding_size: int,
        cell_type: str,
        num_layers: int,
        attn_size1: int,
        attn_size2: int,
        optimizer_type: str,
        learning_rate: float,
        clip_value: int,
        log_dir: str = None,
    ):
        # Get images, captions, lengths and labels
        self.images = images
        self.captions = captions
        self.captions_len = captions_len
        self.labels = labels
        # Create summary writers and global step
        self.file_writer = tf.summary.FileWriter(log_dir)
        self.train_loss_ph, self.train_loss_summary = self.create_summary("train_loss")
        self.val_loss_ph, self.val_loss_summary = self.create_summary("val_loss")
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
        self.attended_image = self.join_attention_graph(
            seed, attn_size1, attn_size2, self.image_encoded, reuse=False
        )
        self.attended_text = self.join_attention_graph(
            seed, attn_size1, attn_size2, self.text_encoded, reuse=True
        )
        logger.info("Attention graph created...")
        self.margin = margin
        self.loss = self.compute_loss(
            self.attended_image, self.attended_text, self.labels, self.margin
        )
        self.optimize = self.apply_gradients_op(
            self.loss, optimizer_type, learning_rate, clip_value
        )
        # Imagenet graph loader and graph saver/loader
        self.image_encoder_loader = self.create_image_encoder_loader()
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
                image_feature_extractor = layers_lib.max_pool2d(
                    net, [2, 2], scope="pool5"
                )

        project_layer = tf.layers.dense(
            image_feature_extractor, 2 * rnn_hidden_size, name="project_image"
        )
        return tf.cast(
            tf.reshape(
                project_layer,
                [
                    -1,
                    project_layer.shape[1] * project_layer.shape[2],
                    2 * rnn_hidden_size,
                ],
            ),
            tf.float32,
        )

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
        with tf.variable_scope("joint_attention"):
            project = tf.layers.dense(
                encoded_input,
                attn_size1,
                activation=tf.nn.tanh,
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                bias_initializer=tf.zeros_initializer(),
                reuse=reuse,
                name="Wa1",
            )
            alphas = tf.layers.dense(
                project,
                attn_size2,
                activation=tf.nn.softmax,
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                bias_initializer=tf.zeros_initializer(),
                reuse=reuse,
                name="Wa2",
            )
        return tf.matmul(tf.transpose(encoded_input, [0, 2, 1]), alphas)

    @staticmethod
    def compute_loss(
        attended_images: tf.Tensor,
        attended_texts: tf.Tensor,
        labels: tf.Tensor,
        margin: float,
    ) -> tf.Tensor:
        """Computes the triplet loss.

        Adapted from: https://omoindrot.github.io/triplet-loss

        Args:
            attended_images: The embedded images.
            attended_texts: The embedded sentences.
            labels: The labels.
            margin: The contrastive margin.

        Returns:
            The triplet loss.

        """
        embeeding_images = tf.reshape(
            attended_images,
            [-1, tf.shape(attended_images)[1] * tf.shape(attended_images)[2]],
        )
        embeeding_texts = tf.reshape(
            attended_texts,
            [-1, tf.shape(attended_texts)[1] * tf.shape(attended_texts)[2]],
        )
        embeddings = tf.concat([embeeding_images, embeeding_texts], 0)
        # Copy the labels 2 times since the embeddings are stacked on top of each other
        labels = tf.tile(labels, [2])
        transposed_embeddings = tf.transpose(embeddings, [1, 0])
        dot_product = tf.matmul(embeddings, transposed_embeddings)
        square_norm = tf.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            tf.expand_dims(square_norm, 0)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 1)
        )

        # Because of computation errors, some distances might be negative so we put
        # everything >= 0.0
        pairwise_dist = tf.maximum(distances, 0.0)

        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j,
        # negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # ------- Triplet mask start ----------
        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(
            tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k
        )

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        mask = tf.to_float(mask)

        # ----------- Triplet mask end --------------

        triplet_loss_masked = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss_masked = tf.maximum(triplet_loss_masked, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss_masked, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss_masked_summed = tf.reduce_sum(triplet_loss_masked) / (
            num_positive_triplets + 1e-16
        )

        return triplet_loss_masked_summed

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
        optimizer = optimizer_factory(optimizer_type, learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

        return optimizer.apply_gradients(
            zip(gradients, variables), global_step=self.global_step
        )

    @staticmethod
    def create_image_encoder_loader():
        """Creates a loader that can be used to load the image encoder.

        Returns:
            A use-case specific loader.

        """
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=["vgg_16"]
        )

        return tf.train.Saver(variables_to_restore)

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
        self.saver_loader.save(sess, save_path)
