# encoder=utf.8
import sys

sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng')
sys.path.insert(1, '/home/vivalavida/workspace/alpha/nmt')
TRAIN_MODE = 'large'
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/nmt')
# TRAIN_MODE = 'small'
import tensorflow as tf
import numpy as np
# from core_seq2seq_model import BabelTowerFactory as BTF
from core_seq2seq_model import TheOldManAndSea_factory as F
import time
import os
import train_conf
import model_helper
from nltk.translate.bleu_score import sentence_bleu
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
BATCH_SIZE = 32
MAX_INFERENCE_LENGTH = 50


class Graph():
    def __init__(self,
                 mode,
                 src_data_path,
                 tgt_data_path,
                 epoch_number=10,
                 batch_size=32,
                 lr=0.001,
                 keep_prob=0.4,
                 clipping=5,
                 inference_length=50,
                 data_shuffle=20000):
        self.mode = mode
        self.src_data_path = src_data_path
        self.tgt_data_path = tgt_data_path
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.clipping = clipping
        self.inference_length = inference_length
        self.data_shuffle = data_shuffle
        self.gpus = train_conf.get_available_gpus()

    def _prepocess_dataset(self, epochs):
        if self.mode == 'test':
            self.model, self.sentenceHelper = F(
                self.src_data_path, self.tgt_data_path, 4).test_model()
        if self.mode == 'small':
            self.model, self.sentenceHelper = F(
                self.src_data_path,
                self.tgt_data_path,
                BATCH_SIZE,
                shuffle=self.data_shuffle).small_model(
                    batch_size=self.batch_size, unit=128)
        if self.mode == 'large':
            self.model, self.sentenceHelper = F(
                self.src_data_path, self.tgt_data_path, 64).large_model(
                    batch_size=64, unit=512)
        self.src_vocabulary, self.tgt_vocabulary, self.src_ids2word, self.tgt_ids2word = self.sentenceHelper.prepare_vocabulary(
        )
        if self.gpus > 0:
            self.batch_train, self.batch_val, self.batch_test = self.sentenceHelper.prepare_data(
            )
            self.batch_train = self.batch_train.prefetch(self.gpus)
        else:
            self.batch_train, self.batch_val, self.bathc_test = self.sentenceHelper.prepare_data(
            )

    def _tower_fusion_grad(self):
        """
            core part
        """

        def average_gradients(tower_grads):
            average_grads = []
            for grad in zip(*tower_grads):
                grads = []
                for g in grad:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                average_grads.append(grad)
            return average_grads

        loss = []
        grad = []
        # pred = 0
        if self.gpus > 0:
            for index, gpu in enumerate(np.arange(0, self.gpus)):
                with tf.device("/device:GPU:" + str(gpu)):
                    p, _, _ = self.model(
                        self.X, dropout=self.keep_prob, train=True)
                    tower_loss = train_conf.loss_function(
                        self.Y, p, self.sentenceHelper.EOS_ID)
                    del p
                    tower_grad, _ = zip(*self.optimizer.compute_gradients(
                        tower_loss, self.model.variables))
                    # tower_grad = self.optimizer.compute_gradients(
                    #     tower_loss, tf.trainable_variables())
                    # pred += p
                    with tf.device("/device:CPU:0"):
                        grad.append(tower_grad)
                        loss.append(tower_loss)
                    del tower_grad, tower_loss
            # pred = pred / self.gpus
            with tf.device("/device:CPU:0"):
                loss = tf.reduce_mean(loss, axis=0)
                grad = average_gradients(grad)
            # train_op = self.optimizer.apply_gradients(
            #     zip(grad, tf.trainable_variables()), self.global_step)
        else:
            pred, _, _ = self.model(self.X, dropout=self.keep_prob, train=True)
            loss = train_conf.loss_function(self.Y, pred,
                                            self.sentenceHelper.EOS_ID)
            grad, _ = zip(
                *self.optimizer.compute_gradients(loss, self.model.variables))
        with tf.device("/device:CPU:0"):
            try:
                grad, _ = tf.clip_by_global_norm(grad, self.clipping)
            except Exception:
                grad, _ = tf.clip_by_global_norm(
                    tf.zeros_like(grad), self.clipping)
            train_op = self.optimizer.apply_gradients(
                zip(grad, self.model.variables), self.global_step)
        return loss, train_op

    def _epoch_train(self,
                     sess,
                     loss,
                     train_op,
                     summarise_op,
                     saver,
                     writer,
                     epoch=0):
        sess.run(self.train_iterator)
        batch_num = 0.0
        total = 0.0
        try:
            while True:
                start_time = time.time()
                train_loss, _ = sess.run([loss, train_op])
                step = self.global_step.eval()
                total += train_loss
                if self.gpus > 0:
                    batch_num += self.gpus
                else:
                    batch_num += 1
                train_summarise = sess.run(
                    summarise_op,
                    feed_dict={self.total_loss: total / batch_num})
                writer.add_summary(train_summarise, global_step=step)
                print('Training_step {0}'.format(step))
                print('Average loss at epoch {0}:{1}'.format(
                    epoch, total / batch_num))
                print('Training cost: {0} seconds'.format(time.time() -
                                                          start_time))
                if step % 100 == 0:
                    saver.save(sess, "checkpoints/nmt_" + str(self.mode), step)
        except tf.errors.OutOfRangeError:
            pass
        self.model.save_weights(SYS_PATH)

    def _epoch_evaluation(self, sess, pred, bleu_summarise_op, writer,
                          epoch=0):
        sess.run(self.val_iterator)
        try:
            while True:
                bleu = 0
                pred = sess.run(pred.predicted_ids)
                tgt_output = tf.reshape(
                    self.Y, [self.batch_size, 1, -1]).eval().tolist()
                for batch in range(self.batch_size):
                    i = tgt_output[batch].index(self.sentenceHelper.SOS_ID)
                    bleu += sentence_bleu(tgt_output[batch][:i],
                                          pred[batch][0]) * 100
                bleu_summarise = sess.run(
                    bleu_summarise_op, feed_dict={self.bleu: bleu})
                step = self.global_step.eval()
                writer.add_summary(bleu_summarise, global_step=step)

        except tf.errors.OutOfRangeError:
            print("epoch {} evaluation finished".format(str(epoch)))

    def graph(self):
        try:
            os.makedirs('checkpoints')
            os.makedirs('checkpoints/nmt_' + str(self.mode))
        except OSError:
            pass
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        with tf.Session(config=config) as sess:
            tf.keras.backend.set_session(sess)
            self._prepocess_dataset(32)
            self.total_loss = tf.placeholder(tf.float32, shape=None)
            self.bleu = tf.placeholder(tf.float32, shape=None)
            self.optimizer = train_conf.optimizer(lr=self.lr)
            self.global_step = tf.train.get_or_create_global_step()
            iterator = tf.data.Iterator.from_structure(
                output_types=self.batch_train.output_types,
                output_shapes=self.batch_train.output_shapes)
            self.val_iterator = iterator.make_initializer(self.batch_val)
            self.train_iterator = iterator.make_initializer(self.batch_train)
            self.X, self.Y = iterator.get_next()
            loss, train_op = self._tower_fusion_grad()
            pred = self.model.beam_search(self.X, self.sentenceHelper.SOS_ID,
                                          self.sentenceHelper.EOS_ID)
            writer = tf.summary.FileWriter(
                SYS_PATH + '/graphs/nmt_' + str(self.mode), sess.graph)
            self.model.summary()
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(
                'checkpoints/nmt_' + str(self.mode) + '/checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(tf.global_variables_initializer())
            train_summarise_op = train_conf.train_summarise(
                loss, self.total_loss)
            bleu_summarise_op = train_conf.bleu(self.bleu)
            for epoch in range(1, self.epoch_number):
                print('Starting epoch {0}'.format(epoch))
                self._epoch_train(sess, loss, train_op, train_summarise_op,
                                  saver, writer, epoch)
                print('starting evaluation')
                self._epoch_evaluation(
                    sess, pred, bleu_summarise_op, writer, epoch=epoch)
            print('graph sucessed')


graph = Graph(
    TRAIN_MODE,
    DATA_PATH + "/europarl-v7.fr-en.fr",
    DATA_PATH + "/europarl-v7.fr-en.en",
    epoch_number=20)
graph.graph()
