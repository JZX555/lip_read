# encoding=utf8
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np


def backend_config():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    # # Don't pre-allocate memory; allocate as-needed
    # config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.99

    return config


def get_learning_rate(learning_rate, hidden_size, step,
                      learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = float(learning_rate_warmup_steps)
        # try:
        #     step = tf.to_float(tf.train.get_or_create_global_step())
        # except Exception:
        #     step = 1
        step = max(1.0, step)
        # step = max(1.0, 10)

        learning_rate *= (hidden_size**-0.5)
        # Apply linear warmup
        learning_rate *= min(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= (1 / np.sqrt(max(step, warmup_steps)))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate

        return learning_rate


def pad_tensors_to_same_length(x, y, pad_id=0):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(
        x, [[0, 0], [0, max_length - x_length], [0, 0]],
        constant_values=pad_id)
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]], constant_values=pad_id)
    return x, y


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'CPU'])


def bleu_summarise(bleu):
    summarise = []
    summarise.append(tf.summary.scalar("evaluation_bleu", bleu))

    return tf.summary.merge(summarise, name="bleu")


def train_summarise(loss=None, total=None, lr=None):
    """
        plot loss and accuracy for analyseing
    """
    summarise = []
    if loss is not None:
        summarise.append(tf.summary.scalar("batch_loss", loss))
        # tf.summary.histogram("loss", loss)
    if total is not None:
        summarise.append(tf.summary.scalar("model_total_loss", total))
        # tf.summary.histogram("total_loss", total)
        # tf.summary.histogram("bleu", bleu)
    if lr is not None:
        summarise.append(tf.summary.scalar("learning_rate", lr))
        # tf.summary.histogram("lr", lr)
    return tf.summary.merge(summarise, name='train_summaris')


def cross_entropy_loss_function(true, pred, mask_id=0):
    import pdb
    pdb.set_trace()
    mask = 1 - tf.cast(tf.equal(true, mask_id), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred, labels=true) * mask
    return tf.reduce_mean(loss)


def onehot_loss_function(true,
                         pred,
                         mask_id=0,
                         smoothing=0.1,
                         vocab_size=24000):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """

    # mask = 1 - tf.cast(tf.equal(true, mask_id), tf.float32)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=pred, labels=true) * mask
    # return tf.reduce_mean(loss)

    logits, labels = pad_tensors_to_same_length(pred, true)
    # Calculate smoothing cross entropy
    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    normalizing_constant = -(confidence * tf.log(confidence) + tf.to_float(
        vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
    xentropy -= normalizing_constant

    weights = tf.to_float(tf.not_equal(labels, mask_id))
    xentropy *= weights
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    return loss


def optimizer(lr=0.1):
    # optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.keras.optimizers.Adam(lr)
    return optimizer


# def compute_gradients(tape, loss, variables, clipping=5):
#     gradients, _ = tf.clip_by_global_norm(
#         tape.gradient(loss, variables), clipping)
#     return gradients
#
#
# def apply_gradients(optimizer, gradients, variables, global_step):
#     return optimizer.apply_gradients(
#         zip(gradients, variables), global_step=global_step)
