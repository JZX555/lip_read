# encoding=utf8
import sys

# sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/home/vivalavida/workspace/alpha/nmt')
sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/nmt')

import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import time
import os
from core_seq2seq_model import TheOldManAndSea_factory as F
from tensorflow.contrib.eager.python import tfe
import train_conf
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
    def _prepocess(self, epochs=1):
        if self.mode == 'test':
            self.model, self.sentenceHelper = F(self.src_data_path,
                                                self.tgt_data_path,
                                                BATCH_SIZE).test_model()
        if self.mode == 'small':
            self.model, self.sentenceHelper = F(self.src_data_path,
                                                self.tgt_data_path,
                                                BATCH_SIZE).samll_model()
        if self.mode == 'large':
            self.model, self.sentenceHelper = F(self.src_data_path,
                                                self.tgt_data_path,
                                                BATCH_SIZE).large_model()
        self.batch_dataset, _, _ = self.sentenceHelper.prepare_data()
        self.src_vocabulary, self.tgt_vocabulary, self.src_ids2word, self.tgt_ids2word = self.sentenceHelper.prepare_vocabulary(
        )

    def _evaluation(self, inputs, tgt_output):
        (src_input, tgt_input, src_length, tgt_length) = inputs
        pred = self.model.beam_search(inputs, self.sentenceHelper.SOS_ID,
                                      self.sentenceHelper.EOS_ID)
        import pdb;pdb.set_trace()
        pred = tf.reshape(pred, [self.batch_size, -1]).numpy().tolist()
        tgt_output = tf.reshape(tgt_output,
                                [self.batch_size, 1, -1]).numpy().tolist()
        bleu = 0
        for batch in range(self.batch_size):
            bleu += sentence_bleu(tgt_output[batch], pred[batch])
        return bleu / batch

    def _train_one_epoch(self, epoch, writer, gpu_num=0):
        start = time.time()
        total_loss = 0
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            for (batch, ((src_input, tgt_input, src_length, tgt_length),
                         tgt_output)) in enumerate(self.batch_dataset):
                X = (src_input, tgt_input, src_length, tgt_length)
                # src_time_step, tgt_time_step)
                with tf.GradientTape() as tape:
                    pred, _, _ = self.model(X)
                    self.model.summary()
                    true = tgt_output
                    self.loss = train_conf.loss_function(
                        true, pred, mask_id=self.sentenceHelper.EOS_ID)
                    self.train_variables = self.model.variables
                    gradients, _ = tf.clip_by_global_norm(
                        tape.gradient(self.loss, self.train_variables),
                        self.clipping)
                    self.optimizer.apply_gradients(
                        zip(gradients, self.train_variables),
                        tf.train.get_or_create_global_step())
                    bleu = self._evaluation(X, tgt_output)
                    total_loss += (self.loss / int(tgt_output.shape[1]))
                    # tf.contrib.summary.scalar("loss", total_loss)
                if writer is not None:
                    tf.contrib.summary.scalar("loss", total_loss)
                if batch % 2 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        1, batch,
                        self.loss.numpy() / (batch + 1)))
                if batch % 8 == 7:
                    break
            print('Time taken for {} epoch {} sec\n'.format(
                epoch,
                time.time() - start))

    def train(self):
        self._prepocess()
        self.optimizer = train_conf.optimizer()
        try:
            os.makedirs(SYS_PATH + "/checkpoints")
            os.makedirs(SYS_PATH + "/checkpoints/nmt_mini")
        except OSError:
            pass
        checkpoint_dir = SYS_PATH + '/checkpoints/nmt_mini"'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)
        writer = tf.contrib.summary.create_file_writer(
            sys.path[1] + '/graphs/nmt_mini', flush_millis=10000)
        if checkpoint:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        for epoch in range(0, self.epoch_number):
            self._train_one_epoch(epoch, writer)
            checkpoint.save(file_prefix=checkpoint_prefix)
            break


######################################################################################
"""
    test exmaple
"""
BATCH_SIZE = 16
train_model = Trainer(
    'test',
    DATA_PATH + "/europarl-v7.fr-en.fr",
    DATA_PATH + "/europarl-v7.fr-en.en",
    epoch_number=1,
    batch_size=BATCH_SIZE)
train_model.train()
