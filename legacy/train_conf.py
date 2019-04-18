# encoding=utf8
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def bleu(bleu):
    tf.summary.scalar("bleu", bleu)
    tf.summary.histogram("bleu", bleu)
    return tf.summary.merge_all()


def train_summarise(loss, total=None):
    """
        plot loss and accuracy for analyseing
    """
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("loss", loss)
    if total is not None:
        tf.summary.scalar("total_loss", total)
        tf.summary.histogram("total_loss", total)
    return tf.summary.merge_all()


def evaluation_summarise(bleu):
    tf.summary.scalar("bleu", bleu)
    tf.summary.histogram("bleu", bleu)
    return tf.summary.merge_all()


def loss_function(true, pred, mask_id=1):
    """Short summary.
    A little trick here:

    Using a mask, which filters '<EOS>' mark in true, it can give a more precise loss value.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """
    mask = 1 - tf.cast(tf.equal(true, mask_id), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred, labels=true) * mask
    return tf.reduce_mean(loss)


def optimizer(lr=0.1):
    optimizer = tf.train.AdamOptimizer(lr)
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
