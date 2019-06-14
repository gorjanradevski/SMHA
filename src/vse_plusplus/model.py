import tensorflow as tf
import tensorflow_hub as hub

# Tensorflow hub necessary import
# noinspection PyUnresolvedReferences
from tensorflow.contrib import image  # NOQA


class VsePlusPlus:
    def __init__(
        self,
        images: tf.Tensor,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        vocab_size: int,
        decay_after: int,
        training_size: int,
        batch_size: int,
    ):
        self.images = images
        self.captions = captions
        self.captions_len = captions_len
        self.is_training = tf.placeholder_with_default(False, None, name="is_training")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.encoded_images = self.image_encoder(images, self.is_training)
        self.encoded_captions = self.text_encoder(
            self.captions, self.captions_len, vocab_size
        )
        self.loss = self.compute_loss()
        self.optimizer_op = self.optimize(decay_after, training_size, batch_size)
        self.saver_loader = tf.train.Saver()

    @staticmethod
    def image_encoder(images: tf.Tensor, is_training: tf.Tensor):
        with tf.variable_scope(name_or_scope="image_encoder"):
            augmentation_module = hub.Module(
                "https://tfhub.dev/google/image_augmentation/nas_imagenet/1"
            )

            embedding_module = hub.Module(
                "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3"
            )
            image_size = hub.get_expected_image_size(embedding_module)
            features = augmentation_module(
                {
                    "encoded_images": images,
                    "image_size": image_size,
                    "augmentation": is_training,
                }
            )
            output = embedding_module(features)
            linear = tf.layers.dense(
                output,
                1024,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=None,
                name="image_linear",
            )

        return tf.math.l2_normalize(linear, axis=1)

    @staticmethod
    def text_encoder(captions: tf.Tensor, captions_len: tf.Tensor, vocab_size: int):
        with tf.variable_scope(name_or_scope="text_encoder"):
            embeddings = tf.get_variable(
                name="embeddings",
                shape=[vocab_size, 300],
                initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
            )
            inputs = tf.nn.embedding_lookup(embeddings, captions)
            cell = tf.nn.rnn_cell.GRUCell(1024)
            output, _ = tf.nn.dynamic_rnn(
                cell=cell, inputs=inputs, sequence_length=captions_len, dtype=tf.float32
            )
            # Select last relevant state
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (captions_len - 1)
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)

            linear = tf.layers.dense(
                relevant,
                1024,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=None,
                name="text_linear",
            )

            return tf.math.l2_normalize(linear, axis=1)

    def compute_loss(self) -> tf.Tensor:
        """Computes the final loss of the model.

        - Computes the Triplet loss: https://arxiv.org/abs/1707.05612

        Returns:
            The final loss to be optimized.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(
                self.encoded_images, self.encoded_captions, transpose_b=True
            )
            diagonal = tf.diag_part(scores)

            # Compare every diagonal score to scores in its column
            # All contrastive images for each sentence
            cost_s = tf.maximum(0.0, 0.2 - tf.reshape(diagonal, [-1, 1]) + scores)
            # Compare every diagonal score to scores in its row
            # All contrastive sentences for each image
            cost_im = tf.maximum(0.0, 0.2 - diagonal + scores)

            # Clear diagonals
            cost_s = tf.linalg.set_diag(cost_s, tf.zeros(tf.shape(cost_s)[0]))
            cost_im = tf.linalg.set_diag(cost_im, tf.zeros(tf.shape(cost_im)[0]))

            # For each positive pair (i,s) pick the hardest contrastive image
            cost_s = tf.reduce_max(cost_s, axis=1)
            # For each positive pair (i,s) pick the hardest contrastive sentence
            cost_im = tf.reduce_max(cost_im, axis=0)

            return tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)

    def optimize(self, decay_after: int, training_size: int, batch_size: int):
        """Creates an optimizer op.

        Returns:
            An optimizer op.

        """
        epochs_decay = decay_after * training_size / batch_size
        with tf.variable_scope(name_or_scope="optimizer"):
            learning_rate = tf.train.exponential_decay(
                0.0002,
                self.global_step,
                epochs_decay,
                0.1,
                staircase=False,
                name="decay",
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 2.0)

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
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        if checkpoint_path is not None:
            self.saver_loader.restore(sess, checkpoint_path)

    def save_model(self, sess: tf.Session, save_path: str) -> None:
        """Dumps the model definition.

        Args:
            sess: The active session.
            save_path: Where to save the model.

        Returns:
            None

        """
        self.saver_loader.save(sess, save_path)
