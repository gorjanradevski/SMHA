import tensorflow as tf
import numpy as np
import pytest


from multi_hop_attention.models import MultiHopAttentionModel


@pytest.fixture
def joint_space():
    return 50


@pytest.fixture
def input_images_image_encoder():
    np.random.seed(42)
    return np.random.rand(50, 224, 224, 3).astype(np.float32)


@pytest.fixture
def input_images():
    np.random.seed(42)
    return np.random.rand(3, 224, 224, 3).astype(np.float32)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def captions():
    return [
        ["goes", "to", "the", "shop", "where"],
        ["nobody", "really", "", "", ""],
        ["will", "see", "now", "what", ""],
    ]


@pytest.fixture
def captions_len():
    return [5, 2, 4]


@pytest.fixture
def margin():
    return 2


@pytest.fixture
def attn_size():
    return 10


@pytest.fixture
def encoded_input():
    np.random.seed(42)
    return np.random.rand(5, 10, 100).astype(np.float32)


@pytest.fixture
def learning_rate():
    return 0.0


@pytest.fixture
def clip_value():
    return 0.0


@pytest.fixture
def attn_heads():
    return 5


@pytest.fixture
def frob_norm_pen():
    return 1


@pytest.fixture
def decay_epochs():
    return 2


@pytest.fixture
def num_layers():
    return 1


@pytest.fixture
def keep_prob():
    return 0.5


def test_image_encoder(input_images_image_encoder, joint_space):
    tf.reset_default_graph()
    image_inputs_layer = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    image_encoded = MultiHopAttentionModel.image_encoder_graph(
        image_inputs_layer, joint_space
    )
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        output_shape = sess.run(
            image_encoded, feed_dict={image_inputs_layer: input_images_image_encoder}
        ).shape
    assert output_shape[0] == 50
    assert output_shape[2] == joint_space


def test_image_encoder_batch_size_invariance(input_images_image_encoder, joint_space):
    tf.reset_default_graph()
    image_inputs_layer = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    image_encoded = MultiHopAttentionModel.image_encoder_graph(
        image_inputs_layer, joint_space
    )
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        output_5 = sess.run(
            image_encoded,
            feed_dict={image_inputs_layer: input_images_image_encoder[:5]},
        )
        output_50 = sess.run(
            image_encoded, feed_dict={image_inputs_layer: input_images_image_encoder}
        )

        np.testing.assert_almost_equal(output_50[:5], output_5, decimal=3)


def test_text_encoder(captions, captions_len, joint_space, num_layers, keep_prob):
    tf.reset_default_graph()
    text_encoded = MultiHopAttentionModel.text_encoder_graph(
        captions, captions_len, joint_space, num_layers, keep_prob
    )
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        outputs = sess.run(text_encoded).shape
    assert outputs[0] == 3
    assert outputs[1] == 5
    assert outputs[2] == joint_space


def test_joint_attention(attn_size, attn_heads, encoded_input):
    tf.reset_default_graph()
    encoded_input_shape = encoded_input.shape
    input_layer = tf.placeholder(dtype=tf.float32, shape=encoded_input_shape)
    attention = MultiHopAttentionModel.attention_graph(
        attn_size, attn_heads, input_layer, "siamese_attention"
    )
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        attended_input, alphas = sess.run(
            attention, feed_dict={input_layer: encoded_input}
        )
        assert attended_input.shape[0] == 5
        assert attended_input.shape[1] == attn_heads * encoded_input_shape[2]
        assert alphas.shape[0] == encoded_input_shape[0]
        assert alphas.shape[1] == attn_heads
        assert alphas.shape[2] == encoded_input_shape[1]


def test_attended_image_text_shape(
    input_images,
    captions,
    captions_len,
    margin,
    joint_space,
    num_layers,
    attn_size,
    attn_heads,
    learning_rate,
    clip_value,
):
    tf.reset_default_graph()
    model = MultiHopAttentionModel(
        input_images,
        captions,
        captions_len,
        margin,
        joint_space,
        num_layers,
        attn_size,
        attn_heads,
    )
    assert model.attended_images.shape[0] == model.attended_captions.shape[0]
    assert model.attended_images.shape[1] == model.attended_captions.shape[1]
