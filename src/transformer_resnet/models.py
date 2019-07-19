import tensorflow as tf
import logging
from typing import Tuple
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerResnet:
    def __init__(
        self,
        images: tf.Tensor,
        captions: tf.Tensor,
        margin: float,
        joint_space: int,
        learning_rate: float = 0.0,
        clip_value: float = 0.0,
        decay_steps: float = 0.0,
        k: int = 100,
        log_dir: str = "",
        name: str = "",
    ):
        # Name of the model
        self.name = name
        # Get images, captions, lengths and labels
        self.images = images
        self.captions = captions
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
        self.weight_decay = tf.placeholder_with_default(0.0, None, name="weight_decay")
        # Build model
        self.image_encoded = self.image_encoder_graph(self.images, joint_space)
        logger.info("Image encoder graph created...")
        self.text_encoded = self.text_encoder_graph(self.captions, joint_space)
        logger.info("Text encoder graph created...")
        self.loss = self.compute_loss(margin, k)
        self.optimize = self.apply_gradients_op(
            self.loss, learning_rate, clip_value, decay_steps
        )
        self.saver_loader = tf.train.Saver()
        logger.info("Graph creation finished...")

    @staticmethod
    def image_encoder_graph(images: tf.Tensor, joint_space: int) -> tf.Tensor:
        """Extract higher level features from the image using a resnet152 pretrained on
        ImageNet.

        Args:
            images: The input images.
            joint_space: The space where the encoded images and text are going to be
            projected to.

        Returns:
            The encoded image.

        """
        with tf.variable_scope("image_encoder"):
            resnet = hub.Module(
                "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3"
            )
            embeddings = resnet(images)
            linear = tf.layers.dense(
                embeddings,
                joint_space,
                kernel_initializer=tf.glorot_uniform_initializer(),
            )

            return tf.math.l2_normalize(linear, axis=1)

    @staticmethod
    def text_encoder_graph(captions: tf.Tensor, joint_space: int):
        """Encodes the text it gets as input using a bidirectional rnn.

        Args:
            captions: The inputs.
            joint_space: The space where the encoded images and text are going to be
            projected to.

        Returns:
            The encoded text.

        """
        with tf.variable_scope(name_or_scope="text_encoder"):
            transformer = hub.Module(
                "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            )
            embeddings = transformer(captions)
            linear = tf.layers.dense(
                embeddings,
                joint_space,
                kernel_initializer=tf.glorot_uniform_initializer(),
            )

            return tf.math.l2_normalize(linear, axis=1)

    def compute_loss(self, margin: float, k: int) -> tf.Tensor:
        """Computes the final loss of the model.

        1. Computes the Triplet loss: https://arxiv.org/abs/1707.05612
        2. Computes the L2 loss.
        3. Adds all together to compute the loss.

        Args:
            margin: The contrastive margin.
            k: The k% of hard negatives to train on.

        Returns:
            The final loss to be optimized.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(self.image_encoded, self.text_encoded, transpose_b=True)
            diagonal = tf.diag_part(scores)
            # Compare every diagonal score to scores in its column
            # All contrastive images for each sentence
            # noinspection PyTypeChecker
            cost_s = tf.maximum(0.0, margin - tf.reshape(diagonal, [-1, 1]) + scores)
            # Compare every diagonal score to scores in its row
            # All contrastive sentences for each image
            # noinspection PyTypeChecker
            cost_im = tf.maximum(0.0, margin - diagonal + scores)

            # Clear diagonals
            cost_s = tf.linalg.set_diag(cost_s, tf.zeros(tf.shape(cost_s)[0]))
            cost_im = tf.linalg.set_diag(cost_im, tf.zeros(tf.shape(cost_im)[0]))

            batch_size = tf.shape(scores)[0]
            # Convert k% to integer
            k = tf.cast(k * batch_size // 100, tf.int32)
            # Convert k% to integer
            k = tf.cast(k * batch_size // 100, tf.int32)
            # For each positive pair (i,s) pick the hardest contrastive image
            cost_s, _ = tf.math.top_k(cost_s, k=k)
            # For each positive pair (i,s) pick the hardest contrastive sentence
            cost_im, _ = tf.math.top_k(tf.transpose(cost_im), k=k)

            matching_loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)

            l2_loss = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if "bias" not in v.name
                    ]
                )
                * self.weight_decay
            )

            return matching_loss + l2_loss

    def apply_gradients_op(
        self,
        loss: tf.Tensor,
        learning_rate: float,
        clip_value: float,
        decay_steps: float,
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
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
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
