#encoding=utf8
import tensorflow as tf
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.keras.callbacks import Callback
import hyper_and_conf.conf_metrics as conf_metrics
import hyper_and_conf.conf_fn as conf_fn
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
import time
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

# from tensorflow.python.ops import summary_ops_v2

# import os


class Onehot_CrossEntropy(Loss):
    def __init__(self, vocab_size, mask_id=0, smoothing=0.1):

        super(Onehot_CrossEntropy, self).__init__(name="Onehot_CrossEntropy")
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.smoothing = smoothing

    def call(self, true, pred):
        loss = conf_fn.onehot_loss_function(
            true=true,
            pred=pred,
            mask_id=self.mask_id,
            smoothing=self.smoothing,
            vocab_size=self.vocab_size)
        return loss


class Padded_Accuracy(Mean):
    def __init__(self, pad_id=0):
        super(Padded_Accuracy, self).__init__(
            name="padded_accuracy", dtype=tf.float32)
        self.pad_id = pad_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        score = conf_metrics.padded_accuracy_score(y_true, y_pred)
        return super(Padded_Accuracy, self).update_state(
            score, sample_weight=sample_weight)


class Padded_Accuracy_topk(Mean):
    def __init__(self, k=5, pad_id=0):
        super(Padded_Accuracy_topk, self).__init__(
            name="padded_accuracy_topk", dtype=tf.float32)
        self.k = k
        self.pad_id = pad_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        score = conf_metrics.padded_accuracy_score_topk(y_true, y_pred)
        return super(Padded_Accuracy_topk, self).update_state(
            score, sample_weight=sample_weight)


class Approx_BLEU_Metrics(Mean):
    def __init__(self, eos_id=1):
        super(Approx_BLEU_Metrics, self).__init__(
            name="BLEU", dtype=tf.float32)
        self.eos_id = eos_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        y_sample = y_pred.get_shape().as_list()[0]
        y_length = y_pred.get_shape().as_list()[1]
        if y_length is not None:
            ctc_length = tf.constant(y_length, shape=[y_sample])
            y_pred, y_score = tf.keras.backend.ctc_decode(
                y_pred, ctc_length, greedy=False, beam_width=4)
        else:
            y_pred = tf.to_int32(tf.argmax(y_pred, axis=-1))

        bleu, _ = conf_metrics.bleu_score(y_pred, y_true, self.eos_id)
        bleu = bleu * 100.0
        return super(Approx_BLEU_Metrics, self).update_state(
            bleu, sample_weight=sample_weight)


class Learning_Rate_Reporter(Metric):
    def __init__(self):
        super(Learning_Rate_Reporter, self).__init__(
            name='lr', dtype=tf.float32)
        self.lr_reporter = self.add_weight(
            'lr', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        state_ops.assign(self.lr_reporter,
                         tf.keras.backend.get_value(self.model.optimizer.lr))

    def result(self):
        return self.lr_reporter


class Dynamic_LearningRate(Callback):
    def __init__(self,
                 init_lr,
                 num_units,
                 learning_rate_warmup_steps,
                 verbose=0):
        super(Dynamic_LearningRate, self).__init__()
        self.init_lr = init_lr
        self.num_units = num_units
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.verbose = verbose
        self.sess = tf.keras.backend.get_session()
        self._total_batches_seen = 0
        self.current_lr = 0

    def on_train_begin(self, logs=None):
        self.current_lr = conf_fn.get_learning_rate(
            self.init_lr, self.num_units, self._total_batches_seen,
            self.learning_rate_warmup_steps)
        lr = float(self.current_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nStart  learning ' 'rate from %s.' % (lr))

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            self.current_lr = conf_fn.get_learning_rate(
                self.init_lr, self.num_units, self._total_batches_seen,
                self.learning_rate_warmup_steps)
        except Exception:  # Support for old API for backward compatibility
            self.current_lr = self.init_lr
        lr = float(self.current_lr)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            tf.logging.info('\nEpoch %05d: Changing  learning '
                            'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        # path = os.path.join("model_summary", "train")
        # writer = summary_ops_v2.create_file_writer(path)
        # with summary_ops_v2.always_record_summaries():
        #     with writer.as_default():
        #         summary_ops_v2.scalar(
        #             "lr", self.current_lr, step=self._total_batches_seen)
        self._total_batches_seen += 1
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    # def _log_lr(logs, prefix,step):


class BatchTiming(Callback):
    """
    It measure robust stats for timing of batches and epochs.
    Useful for measuring the training process.

    For each epoch it prints median batch time and total epoch time.
    After training it prints overall median batch time and median epoch time.

    Usage: model.fit(X_train, Y_train, callbacks=[BatchTiming()])

    All times are in seconds.

    More info: https://keras.io/callbacks/
    """

    def on_train_begin(self, logs={}):
        self.all_batch_times = []
        self.all_epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_batch_times = []

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.epoch_batch_times.append(elapsed_time)
        self.all_batch_times.append(elapsed_time)

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = np.sum(self.epoch_batch_times)
        self.all_epoch_times.append(epoch_time)
        median_batch_time = np.median(self.epoch_batch_times)
        print('Epoch timing - batch (median): %0.5f, epoch: %0.5f (sec)' % \
            (median_batch_time, epoch_time))

    def on_train_end(self, logs={}):
        median_batch_time = np.median(self.all_batch_times)
        median_epoch_time = np.median(self.all_epoch_times)
        print('Overall - batch (median): %0.5f, epoch (median): %0.5f (sec)' % \
            (median_batch_time, median_epoch_time))


class SamplesPerSec(Callback):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.all_samples_per_sec = []

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()
        # self.batch_size = logs['size']

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        samples_per_sec = self.batch_size / elapsed_time
        self.all_samples_per_sec.append(samples_per_sec)

    def on_epoch_end(self, epoch, logs={}):
        self.print_results()

    def print_results(self):
        print('Samples/sec: %0.2f' % np.median(self.all_samples_per_sec))


"""
Enables CUDA profiling (for usage in nvprof) just for a few batches.

The reasons are:

- profiling outputs are big (easily 100s MB - GBs) and repeating
- without a proper stop the outputs sometimes fail to save

Since initially the TensorFlow runtime may take time to optimize the graph we
skip a few epochs and then enable profiling for a few batches within the next
epoch.

It requires the `cudaprofile` package.
"""


class CudaProfile(Callback):
    def __init__(self, warmup_epochs=0, batches_to_profile=None):
        self.warmup_epochs = warmup_epochs
        self.batches_to_profile = batches_to_profile
        self.enabled = False

    def start(self):
        # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
        # the return value will unconditionally be 0. This check is just in case it changes in
        # the future.
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    def stop(self):
        ret = _cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)

    def set_params(self, params):
        self.params = params

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == self.warmup_epochs:
            self.start()
            self.enabled = True

    def on_batch_end(self, batch, logs={}):
        if self.enabled and batch >= self.batches_to_profile:
            self.stop()


class StagingAreaCallback(Callback):
    """
    staging_area_callback = StagingAreaCallback(x_train, y_train, batch_size)

    image = Input(tensor=staging_area_callback.input_tensor)
    x = Dense(512, activation='relu')(image)
    digit = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=image, outputs=digit)

    model.compile(optimizer='sgd', loss='categorical_crossentropy',
        target_tensors=[staging_area_callback.target_tensor],
        fetches=staging_area_callback.extra_ops)

    model.fit(steps_per_epoch=steps_per_epoch, epochs=2,
        callbacks=[staging_area_callback])

    Full example: https://gist.github.com/bzamecnik/b520e2b1e199b193b715477929e39b22
    """
    """
        Fork original one
    """

    def __init__(self, x, y, batch_size, prefetch_count=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count

        features_shape_src = (None, ) + x[0].shape[1:]
        features_shape_tgt = (None, ) + x[1].shape[1:]

        labels_shape = (None, ) + y.shape[1:]

        with tf.device('/cpu:0'):
            # for feeding inputs to the the StagingArea
            # Let's try to decouple feeding data to StagingArea.put()
            # from the training batch session.run()
            # https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data

            features_batch_next_value_src = tf.placeholder(
                dtype=x[0].dtype, shape=features_shape_src)
            features_batch_next_value_tgt = tf.placeholder(
                dtype=x[1].dtype, shape=features_shape_tgt)
            self.features_batch_next_value = (features_batch_next_value_src,
                                              features_batch_next_value_tgt)

            # - prevent the variable to be used as a model parameter: trainable=False, collections=[]
            # - allow dynamic variable shape (for the last batch): validate_shape=False
            features_batch_next = tf.Variable(
                self.features_batch_next_value,
                trainable=False,
                collections=[],
                validate_shape=False)
            self.labels_batch_next_value = tf.placeholder(
                dtype=y.dtype, shape=labels_shape)
            labels_batch_next = tf.Variable(
                self.labels_batch_next_value,
                trainable=False,
                collections=[],
                validate_shape=False)
        self.assign_next_batch = tf.group(features_batch_next.initializer,
                                          labels_batch_next.initializer)

        # will be used for prefetching to GPU
        area = tf.contrib.staging.StagingArea(
            dtypes=[x.dtype, y.dtype],
            shapes=[[features_shape_src, features_shape_tgt], labels_shape])

        self.area_put = area.put(
            [features_batch_next.value(),
             labels_batch_next.value()])
        area_get_features, area_get_labels = area.get()
        self.area_size = area.size()
        self.area_clear = area.clear()

        self.input_tensor = area_get_features
        self.target_tensor = area_get_labels
        self.extra_ops = [self.area_put]

    def set_params(self, params):
        super().set_params(params)
        self.steps_per_epoch = self.params['steps']

    def _slice_batch(self, i):
        start = i * self.batch_size
        end = start + self.batch_size
        return (self.x[start:end], self.y[start:end])

    def _assign_batch(self, session, data):
        x_batch, y_batch = data
        session.run(
            self.assign_next_batch,
            feed_dict={
                self.features_batch_next_value: x_batch,
                self.labels_batch_next_value: y_batch
            })

    def on_epoch_begin(self, epoch, logs=None):
        sess = tf.keras.backen.get_session()
        for i in range(self.prefetch_count):
            self._assign_batch(sess, self._slice_batch(i))
            sess.run(self.area_put)

    def on_batch_begin(self, batch, logs=None):
        sess = tf.keras.backen.get_session()
        # Slice for `prefetch_count` last batches is empty.
        # It serves as a dummy value which is put into StagingArea
        # but never read.
        data = self._slice_batch(batch + self.prefetch_count)
        self._assign_batch(sess, data)

    def on_epoch_end(self, epoch, logs=None):
        sess = tf.keras.backen.get_session()
        sess.run(self.area_clear)


#
#
# class StagingAreaCallbackFeedDict(Callback):
#     """
#     It allows to prefetch input batches to GPU using TensorFlow StagingArea,
#     making a simple asynchronous pipeline.
#
#     The classic mechanism of copying input data to GPU in Keras with TensorFlow
#     is `feed_dict`: a numpy array is synchronously copied from Python to TF memory
#     and then using a host-to-device memcpy to GPU memory. The computation,
#     however has to wait, which is wasteful.
#
#     This class makes the HtoD memcpy asynchronous using a GPU-resident queue
#     of size two (implemented by StaginArea). The mechanism is as follows:
#
#     - at the beginning of an epoch one batch is `put()` into the queue
#     - during each training step another is is `put()` into the queue and in
#       parallel the batch already present at the GPU is `get()` from the queue
#       at provide as tesnor input to the Keras model (this runs within a single
#       `tf.Session.run()`)
#
#     The input numpy arrays (features and targets) are provided via this
#     callback and sliced into batches inside it. The last batch might be of
#     smaller size without any problem (the StagingArea supports variable-sized
#     batches and allows to enforce constant data sample shape). In the last
#     batch zero-length slice is still put into the queue to keep the get+put
#     operation uniform across all batches.
#
#     We feed input data to StagingArea via `feed_dict` as an additional input
#     besides Keras inputs. Note that the `feed_dict` dictionary is passed as a
#     reference and its values are updated inside the callback. It is still
#     synchronous. A better, though more complicated way would be to use TF queues
#     (depracated) or Dataset API.
#
#     It seems to help on GPUs with low host-device bandwidth, such as desktop
#     machines with many GPUs sharing a limited number of PCIe channels.
#
#     In order to provide extra put() operation to `fetches`, we depend on a fork
#     of Keras (https://github.com/bzamecnik/keras/tree/tf-function-session-run-args).
#     A pull request to upstream will be made soon.
#
#     Example usage:
#
#     ```
#     staging_area_callback = StagingAreaCallback(x_train, y_train, batch_size)
#
#     image = Input(tensor=staging_area_callback.input_tensor)
#     x = Dense(512, activation='relu')(image)
#     digit = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=image, outputs=digit)
#
#     model.compile(optimizer='sgd', loss='categorical_crossentropy',
#         target_tensors=[staging_area_callback.target_tensor],
#         feed_dict=staging_area_callback.feed_dict,
#         fetches=staging_area_callback.extra_ops)
#
#     model.fit(steps_per_epoch=steps_per_epoch, epochs=2,
#         callbacks=[staging_area_callback])
#     ```
#
#     Full example: https://gist.github.com/bzamecnik/b520e2b1e199b193b715477929e39b22
#     """
#
#     def __init__(self, x, y, batch_size, prefetch_count=1):
#         self.x = x
#         self.y = y
#         self.batch_size = batch_size
#         self.prefetch_count = prefetch_count
#
#         features_shape = (None, ) + x.shape[1:]
#         labels_shape = (None, ) + y.shape[1:]
#
#         # inputs for feeding inputs to the the StagingArea
#         self.features_batch_next = tf.placeholder(
#             dtype=x.dtype, shape=features_shape)
#         self.labels_batch_next = tf.placeholder(
#             dtype=y.dtype, shape=labels_shape)
#         # We'll assign self.features_batch_next, self.labels_batch_next before
#         # each StagingArea.put() - feed_dict is passed by reference and updated
#         # from outside.
#         self.feed_dict = {}
#
#         # will be used for prefetching to GPU
#         area = tf.contrib.staging.StagingArea(
#             dtypes=[x.dtype, y.dtype], shapes=[features_shape, labels_shape])
#
#         self.area_put = area.put(
#             [self.features_batch_next, self.labels_batch_next])
#         area_get_features, area_get_labels = area.get()
#         self.area_size = area.size()
#         self.area_clear = area.clear()
#
#         self.input_tensor = area_get_features
#         self.target_tensor = area_get_labels
#         self.extra_ops = [self.area_put]
#
#     def set_params(self, params):
#         super().set_params(params)
#         self.steps_per_epoch = self.params['steps']
#
#     def _slice_batch(self, i):
#         start = i * self.batch_size
#         end = start + self.batch_size
#         return (self.x[start:end], self.y[start:end])
#
#     def _update_feed_dict(self, data):
#         x_batch, y_batch = data
#         self.feed_dict[self.features_batch_next] = x_batch
#         self.feed_dict[self.labels_batch_next] = y_batch
#
#     def on_epoch_begin(self, epoch, logs=None):
#         sess = tf.keras.backen.get_session()
#         # initially fill the StagingArea
#         for i in range(self.prefetch_count):
#             self._update_feed_dict(self._slice_batch(i))
#             sess.run(feed_dict=self.feed_dict, fetches=[self.area_put])
#
#     def on_batch_begin(self, batch, logs=None):
#         sess = tf.keras.backen.get_session()
#         # Slice for `prefetch_count` last batches is empty.
#         # It serves as a dummy value which is put into StagingArea
#         # but never read.
#         self._update_feed_dict(self._slice_batch(batch + self.prefetch_count))
#
#     def on_epoch_end(self, epoch, logs=None):
#         sess = tf.keras.backen.get_session()
#         sess.run(self.area_clear)
#
