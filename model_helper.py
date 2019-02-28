# encoding=utf8
import tensorflow as tf
import time


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


def dropout_mask_helper(ones, rate, training=None, count=1):
    def dropped_inputs():
        return tf.keras.backend.dropout(ones, rate)

    if count > 1:
        return [
            tf.keras.backend.in_train_phase(
                dropped_inputs, ones, training=training) for _ in range(count)
        ]
    return tf.keras.backend.in_train_phase(
        dropped_inputs, ones, training=training)


def embedding_helper(X, vocabulary_size, embedding_size):
    """Short summary.

    Args:
        X (type): Description of parameter `X`.
        vocabulary_size (type): Description of parameter `vocabulary_size`.
        embedding_size (type): Description of parameter `embedding_size`.
        name (type): Description of parameter `name`.

    Returns:
        embedded_X

    """
    embedding_matrix = tf.keras.layers.Embedding(vocabulary_size,
                                                 embedding_size)
    embedded_X = embedding_matrix(X)
    return embedded_X


def lstm_cell_helper(unit_num, dropout=1):
    """Short summary.

    Args:
        unit_num (type): Description of parameter `unit_num`.
        name (type): Description of parameter `name`.

    Returns:
        basice lstm cell
    """
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(
            unit_num, return_sequences=True, return_state=True)
    else:
        return tf.keras.layers.LSTM(
            unit_num,
            return_sequences=True,
            return_state=True,
            dropout=dropout)


def lstm_helper(X,
                unit_num,
                batch_size,
                bidirection=True,
                initial_state=None,
                merge_mode='concat'):
    """Short summary.
    Wrap lstm.
    Args:
        X (type): Description of parameter `X`.
        unit_num (type): Description of parameter `unit_num`.
        bidirection (type): Description of parameter `bidirection`.
        name (type): Description of parameter `name`.

    Returns:
        raw tf.keras.layer output
    """
    if initial_state is None:
        initial_state = tf.zeros([batch_size, unit_num])
    cell = lstm_cell_helper(unit_num)
    if bidirection:
        bi_cell = tf.keras.layers.Bidirectional(
            layer=cell, dtype=tf.float32, merge_mode='concat')
        output = bi_cell(X)
    else:
        output = cell(X)
    return output


def time_major_helper(X):
    """Short summary.
    convert [batch, time, vector] to [time, batch, vector] for
    truncated propogation.
    Args:
        X (type): Description of parameter `X`.
        axis (type):

    Returns:
        tensorf
    """
    return tf.transpose(X, [1, 0, 2])


def batch_major_helper(X):
    """Short summary.
    convert [time, batch, vector] to [batch, time, vector] for
    truncated propogation.
    Args:
        X (type): Description of parameter `X`.
        axis (type):

    Returns:
        tensorf
    """
    return tf.transpose(X, [1, 0, 2])


def beam_search_helper(decoder_cell, encoder_state, beam_width,
                       embedding_decoder, sos_id, eos_id, batch_size):
    # Replicate encoder infos beam_width times
    decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    start_tokens = tf.constant(value=sos_id, shape=[batch_size])
    # Define a beam-search decoder
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_decoder,
        start_tokens=start_tokens,
        end_token=eos_id,
        initial_state=decoder_initial_state,
        beam_width=beam_width)

    # Dynamic decoding
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    return outputs


def dense_helper(X, out_dimention, activation='relu'):
    """Short summary.
    simplely full connection wrapper
    Args:
        out_dimention (type): Description of parameter `out_dimention`.
        activation='relu'
    Returns:
        type: Description of returned object.

    """
    w = tf.keras.layers.Dense(out_dimention, activation=activation)
    return w(X)


def softmax_helper(X, class_num, dense=False):
    """Short summary.
    simplely softmax wrapper.

    Args:
        X (type): Description of parameter `X`.
        class_num (type): Description of parameter `class_num`.
        fc: whether it needs fc layer before softmax
    Returns:
        type: Description of returned object.

    """
    if dense is not True:
        X = dense_helper(X, class_num)
    softmax = tf.keras.layers.Softmax()
    return softmax(X)


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


if __name__ == '__main__':
    """
        test exmaple
    """
    tf.enable_eager_execution()
    import numpy as np
    src_3d = tf.constant(
        np.random.randint(1, 5), shape=[3, 5, 3], dtype=tf.float32)

    src_zero = tf.constant(0, shape=[3, 5, 3], dtype=tf.float32)
    src_2d = tf.constant(
        np.random.randint(1, 5), shape=[5, 5], dtype=tf.float32)
    tgt_3d = tf.constant(
        np.random.randint(5, 10), shape=[3, 6, 3], dtype=tf.float32)
    e_h = embedding_helper(src_2d, 10, 10)
    l_h = lstm_helper(src_3d, 10, 3)
    zero_t = dense_helper(src_zero, 6)
