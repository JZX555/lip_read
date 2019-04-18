# encoding=utf8
import sys
# sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/home/vivalavida/workspace/alpha/transformer_nmt')
sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
# import model_helper
import time
import os
from tensorflow.contrib.eager.python import tfe
import train_conf
print("eager mode")
tfe.enable_eager_execution()

# DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

# LEARNING_RATE = 0.1
# KEEP_PROB = 0.4
#
# GLOBAL_STEP = tf.Variable(initial_value=0, trainable=False)
# NUMBER_EPOCH = 1
# CLIPPING = 5
# NUMBER_UNITS = 16  # dimension of each hidden state
# EMBEDDING_SIZE = NUMBER_UNITS
# NUMBER_LAYER = 1


class Trainer():
    def __init__(self, model, dataset_manager, hyperParam, vocab_size, gpu=0):
        self.model = model
        self.dataset_manager = dataset_manager
        self.hyperParam = hyperParam
        self.epoch_number = hyperParam.epoch_num
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.clipping = hyperParam.clipping
        self.batch_size = hyperParam.batch_size
        self.GLOBAL_STEP = tf.Variable(initial_value=0, trainable=False)
        if int(tfe.num_gpus()) > 0:
            self.gpu = tfe.num_gpus()

    def _prepocess(self):
        if self.gpu > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.prepare_data(
            )
            self.train_dataset = self.train_dataset.prefetch(self.gpu * 2)
            self.val_dataset = self.val_dataset.prefetch(self.gpu * 2)
            self.test_dataset = self.test_dataset.prefetch(self.gpu * 2)
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.prepare_data()
            self.train_dataset = self.train_dataset.prefetch(2)
            self.val_dataset = self.val_dataset.prefetch(2)
            self.test_dataset = self.test_dataset.prefetch(2)

    def _evaluation(self, inputs, tgt_output):
        pred = self.model(inputs, train=False)
        pred = tf.reshape(pred, [self.batch_size, -1]).numpy().tolist()
        import pdb; pdb.set_trace()
        test = self.dataset_manager.decode(pred)
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
            for (batch, (src, tgt)) in enumerate(self.train_dataset):
                X = (src, tgt)
                # src_time_step, tgt_time_step)
                with tf.GradientTape() as tape:
                    pred = self.model(X)
                    attention_weights = self.model.get_attention()
                    # self.model.summary()
                    true = tgt
                    self.loss = train_conf.onehot_loss_function(
                        true,
                        pred,
                        mask_id=self.hyperParam.PAD_ID,
                        vocab_size=self.vocab_size)
                    self.train_variables = self.model.variables
                    gradients, _ = tf.clip_by_global_norm(
                        tape.gradient(self.loss, self.train_variables),
                        self.clipping)

                    self.optimizer.apply_gradients(
                        zip(gradients, self.train_variables),
                        tf.train.get_or_create_global_step())
                    # bleu = self._evaluation(src, tgt)
                    total_loss += (self.loss / int(tgt.shape[1]))
                    # tf.contrib.summary.scalar("loss", total_loss)
                if writer is not None:
                    tf.contrib.summary.scalar("loss", total_loss)
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    1, batch,
                    self.loss.numpy()))
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


######################################################################################

import hyperParam
import core_Transformer_model
import core_dataset_generator
# import test_trainer
# import graph_trainer
# set_up
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
# src_data_path = DATA_PATH + "/vivalavida.txt"
# tgt_data_path = DATA_PATH + "/vivalavida.txt"
src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"
hp = hyperParam.HyperParam('test')
data_manager = core_dataset_generator.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    max_length=hp.max_sequence_length)
# dataset_train_val_test = data_manager.prepare_data()
gpu = train_conf.get_available_gpus()
vocabulary_size = 24000
model = core_Transformer_model.Daedalus(
    max_seq_len=hp.max_sequence_length,
    vocabulary_size=vocabulary_size,
    embedding_size=hp.embedding_size,
    batch_size=hp.batch_size / gpu if gpu > 0 else 1,
    num_units=hp.num_units,
    num_heads=hp.num_heads,
    num_encoder_layers=hp.num_encoder_layers,
    num_decoder_layers=hp.num_decoder_layers,
    dropout=hp.dropout,
    eos_id=hp.EOS_ID,
    pad_id=hp.PAD_ID)
# test = test_trainer.Trainer(model=model, dataset=dataset_train_val_test)
# test.train()
test_train = Trainer(
    model=model,
    hyperParam=hp,
    vocab_size=vocabulary_size,
    dataset_manager=data_manager)

test_train.train()
