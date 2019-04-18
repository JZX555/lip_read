# encoding=utf8
import sys

# sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/home/vivalavida/workspace/alpha/nmt')
sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/nmt')

import tensorflow as tf
import numpy as np
import time
import os
from core_model import BabelTowerFactory as BTF
from tensorflow.contrib.eager.python import tfe
import model_helper
#######eager module#######
print("eager mode")
tfe.enable_eager_execution()
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
LEARNING_RATE = 0.1
KEEP_PROB = 0.4

GLOBAL_STEP = tf.Variable(initial_value=0, trainable=False)
NUMBER_EPOCH = 1
CLIPPING = 5
# ~~~~~~~~~~~~~~ basic LSTM configure~~~~~~~~~~~~~~~~~
# TIME_STEP = 8  # number of hidden state, it also means the lenght of input and output sentence
NUMBER_UNITS = 16  # dimension of each hidden state
# ~~~~~~~~~~~~~ embedding configure ~~~~~~~~~~~~~~~~~
EMBEDDING_SIZE = NUMBER_UNITS
NUMBER_LAYER = 1


class Trainer():
    def __init__(self,
                 mode,
                 src_data_path,
                 tgt_data_path,
                 epoch_number=10,
                 batch_size=64,
                 lr=0.01,
                 keep_prob=0.4,
                 clipping=5,
                 data_shuffle=20000,
                 eager=True,
                 gpu=0):
        self.src_data_path = src_data_path
        self.tgt_data_path = tgt_data_path
        self.mode = mode
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.gpu = gpu
        # think twice before change it
        self.eager = eager
        if int(tfe.num_gpus()) > 0:
            self.gpu = tfe.num_gpus()
        # train timer
        # self.th = model_helper.TimeHistory()
        self._train_setup()

    def _prepocess(self, epochs=1):
        if self.mode == 'mini':
            self.model, self.sentenceHelper = BTF(
                self.src_data_path, self.tgt_data_path, BATCH_SIZE,
                eager=True).mini_model()
        if self.mode == 'small':
            self.model, self.sentenceHelper = BTF(
                self.src_data_path, self.tgt_data_path, BATCH_SIZE,
                eager=True).samll_model()
        if self.mode == 'large':
            self.model, self.sentenceHelper = BTF(
                self.src_data_path, self.tgt_data_path, BATCH_SIZE,
                eager=True).large_model()
        self.batch_dataset = self.sentenceHelper.prepare_data()
        self.src_vocabulary, self.tgt_vocabulary, self.src_ids2word, self.tgt_ids2word = self.sentenceHelper.prepare_vocabulary(
        )
        if self.gpu > 0:
            self.batch_dataset = self.batch_dataset.shuffle(
                self.data_shuffle).repeat(epochs)

    def _loss_function(self, pred, true):
        """Short summary.
        A little trick here:

        Using a mask, which filters '<EOS>' mark in true, it can give a more precise loss value.
        Args:
            pred (type): Description of parameter `pred`.
            true (type): Description of parameter `true`.

        Returns:
            type: Description of returned object.

        """
        if self.eager:
            mask = 1 - np.equal(true, self.sentenceHelper.EOS_ID)
        else:
            mask = 1 - np.equal(true.eval(), self.sentenceHelper.EOS_ID)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred, labels=true) * mask
        return tf.reduce_mean(loss)

    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)

    def _compute_gradients(self, tape, loss, variables):
        gradients, _ = tf.clip_by_global_norm(
            tape.gradient(loss, variables), self.clipping)
        return gradients

    def _apply_gradients(self, gradients, variables, global_step):
        self.optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)

    def _eager_boost(self):
        """Short summary.
        1) the forward computation, 2) the backward computation for the gradients,
        and 3) the application of gradients to variables
        Args:


        Returns:
            type: Description of returned object.

        """
        self.model.call = tfe.defun(self.model.call)
        # self.model.compute_gradients = tfe.defun(self.model.compute_gradients)
        self._apply_gradients = tfe.defun(self._apply_gradients)

    def _summarize(self, loss):
        """
            return:
                summarize_op
        """
        tf.summary.scalar("loss", tf.reduce_mean(loss))
        return tf.summary.merge_all()

    def _train_setup(self):
        self._prepocess()
        self._optimizer()
        self._eager_boost()

    def _train_one_epoch(self, epoch, writer, gpu_num=0):
        start = time.time()
        self.model.re_initialize_final_state()
        total_loss = 0
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            for (batch, ((src_input, tgt_input, src_length, tgt_length),
                         tgt_output)) in enumerate(self.batch_dataset):
                X = (src_input, tgt_input, src_length, tgt_length)
                with tf.GradientTape() as tape:
                    pred = self.model(X)
                    true = tgt_output
                    self.loss = self._loss_function(pred, true)
                    self.train_variables = self.model.variables
                    gradients, _ = tf.clip_by_global_norm(
                        tape.gradient(self.loss, self.train_variables),
                        self.clipping)
                    self.optimizer.apply_gradients(
                        zip(gradients, self.train_variables),
                        tf.train.get_or_create_global_step())
                    total_loss += (self.loss / int(tgt_output.shape[1]))
                    tf.contrib.summary.scalar("loss", total_loss)
                    print(gpu_num)
                if writer is not None:
                    tf.contrib.summary.scalar("loss", total_loss)
                if batch % 2 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        1, batch,
                        self.loss.numpy() / int(tgt_output.shape[1])))
                if batch % 5 == 0:
                    break
            print('Time taken for {} epoch {} sec\n'.format(
                epoch,
                time.time() - start))

    def train(self):
        try:
            os.makedirs(SYS_PATH + "/checkpoints_full")
            os.makedirs(SYS_PATH + "/checkpoints_full/nmt")
        except OSError:
            pass
        checkpoint_dir = SYS_PATH + '/checkpoints_full/nmt"'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)
        writer = tf.contrib.summary.create_file_writer(
            sys.path[1] + '/graphs_full/nmt', flush_millis=10000)
        if checkpoint:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        if self.gpu > 0:
            for index, gpu in enumerate(np.arange(0, self.gpu)):
                with tf.device("/device:GPU:" + str(gpu)):
                    for epoch in range(0, self.epoch_number):
                        self._train_one_epoch(epoch, writer, gpu)
                        checkpoint.save(file_prefix=checkpoint_prefix)
        else:
            # pass
            # sess = tf.Session()
            # tf.keras.backend.set_session(sess)
            # self.dataset = self.train_dataset.make_one_shot_iterator()
            # with sess.as_default():
            #     # sess.run(self.dataset.initializer)
            #     sess.run(tf.global_variables_initializer())
            #     # while True:
            #     src_input, tgt, src_length, tgt_length = self.dataset.get_next(
            #     )
            #     tgt_input, tgt_output = data_sentence_helper.tgt_formater(
            #         tgt)
            #     X = (src_input, tgt_input, src_length, tgt_length)
            #     pred = self.model(X)
            #     sess.run(pred.eval())

            for epoch in range(0, self.epoch_number):
                self._train_one_epoch(epoch, writer)
                checkpoint.save(file_prefix=checkpoint_prefix)
                break


######################################################################################
"""
    test exmaple
"""
BATCH_SIZE = 32
train_model = Trainer(
    'mini',
    DATA_PATH + "/europarl-v7.fr-en.fr",
    DATA_PATH + "/europarl-v7.fr-en.en",
    epoch_number=1,
    batch_size=BATCH_SIZE)
train_model.train()
