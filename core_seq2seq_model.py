# encoding=utf8
"""

    artist: Barid Ai


    This is a classic nmt model, using seq2seq and attention, based on
    "Bahdanau, D., Cho, K., & Bengio, Y. (n.d.). NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE.
    Retrieved from http://cs224d.stanford.edu/papers/nmt.pdf"
"""

import tensorflow as tf
import model_helper
from data_sentence_helper import SentenceHelper

# from tensorflow.contrib import autograph


class RNNcoder(tf.keras.Model):
    """Wrap a new RNN model
        !!!ATTENTION!!!
        Data structure must be in [batch, time, embedding], the batch major
        style,  this model will automaticly change it to time major style
        for compututional efficiency.
    """

    def __init__(self,
                 vocabulary_size,
                 batch_size,
                 unit_num,
                 embedding_size,
                 max_sequence=200,
                 time_major=True,
                 backforward=False,
                 embedding_matrix=True,
                 eager=True):
        super(RNNcoder, self).__init__()
        self.batch_size = batch_size
        self.unit_num = unit_num
        self.embedding_size = embedding_size
        self.max_sequence = max_sequence
        self.bw = backforward
        self.time_major = True
        self.vocabulary_size = vocabulary_size
        self.embedding_matrix = embedding_matrix
        self.eager = eager
        self.embedding_helper = tf.keras.layers.Embedding(
            self.vocabulary_size, self.embedding_size)
        self.max_length = 100
        self.layer_initializer()

    def layer_initializer(self):
        # initialize layer
        self.W_ = tf.keras.layers.Dense(self.unit_num)
        self.U_ = tf.keras.layers.Dense(self.unit_num)
        self.C_ = tf.keras.layers.Dense(self.unit_num)
        self.h_bar_ = tf.keras.layers.Dense(self.unit_num)
        self.energy_U_ = tf.keras.layers.Dense(self.unit_num)
        self.energy_W_ = tf.keras.layers.Dense(self.unit_num)
        self.engery_ = tf.keras.layers.Dense(1)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.unit_num))

    def _sigmid_w_u_c(self, embedding_x, hidden, attention):
        W = self.W_(embedding_x)
        U = self.U_(hidden)
        C = self.C_(attention)
        sigmid_w_u_c = tf.keras.backend.sigmoid(W + U + C)
        return sigmid_w_u_c

    def _gated_combination(self, embedding_x, hidden, attention=0):
        """Short summary.
        Args:
            X (type): Description of parameter `X`.
            hidden_s (type): Description of parameter `hidden_s`.
            hidden_h (type): Description of parameter `hidden_h`.
            attention (type): Description of parameter `attention`.

        Returns:
            type: Description of returned object.

        """
        if attention == 0:
            attention = self.initialize_hidden_state()
        r = self._sigmid_w_u_c(embedding_x, hidden, attention)
        z = self._sigmid_w_u_c(embedding_x, hidden, attention)
        h_bar = tf.keras.backend.tanh(embedding_x + self.h_bar_(r * hidden))

        H = (1 - z) * hidden + z * h_bar
        return H

    def _attention(self, time_step, hidden_base, hidden_context):
        """Short summary.
        Implementing the method from mentioned paper,
        it follows stand attention pipe line:
            energy -> softmax -> context
        Args:
             (type): Description of parameter ``.

        Returns:
            type: Description of returned object.

        """
        attention_context = []
        hidden_context = model_helper.time_major_helper(hidden_context)
        energy_U = self.energy_U_(hidden_context)
        energy_W = self.energy_W_(hidden_base)
        energy = self.engery_(tf.keras.backend.tanh((energy_U + energy_W)))
        # axis = 0 because it needs weights with respect to time.
        attention_score = tf.keras.layers.Softmax(axis=0)(energy)
        attention_score = tf.transpose(energy, [1, 2, 0])
        hidden_context = tf.transpose(hidden_context, [1, 0, 2])
        # attention_context = tf.keras.backend.dot(
        #     attention_score * hidden_context)
        for b in range(0, self.batch_size):
            temp = tf.keras.backend.dot(attention_score[b], hidden_context[b])
            attention_context.append(tf.reshape(temp, [-1]))
        attention_context = tf.convert_to_tensor(attention_context)
        return attention_context, attention_score

    def call(self, inputs):
        # embedded = model_helper.embedding_helper(
        #     X, self.vocabulary_size, self.embedding_size)
        (X, sequence_length, hidden, attention_hidden) = inputs
        if self.embedding_matrix:
            embedded = self.embedding_helper(X)
            embedding_x = model_helper.time_major_helper(embedded)
        else:
            embedding_x = model_helper.time_major_helper(X)
        time_step = X.shape[1].value
        cell = hidden
        cell_hidden = []
        if self.bw:
            embedding_x = tf.reverse_sequence(
                embedding_x, sequence_length, seq_axis=0, batch_axis=1)
        for t in range(0, time_step):
            if time_step > self.max_sequence:
                break
            attention_context = 0
            if attention_hidden != 0:
                attention_context, attention_score = self._attention(
                    time_step, hidden, attention_hidden)
            cell = self._gated_combination(embedding_x[t], cell,
                                           attention_context)
            cell_hidden.append(cell)
        cell_hidden = tf.convert_to_tensor(cell_hidden)
        cell_hidden = tf.transpose(cell_hidden, (1, 0, 2))
        return (cell_hidden, cell)


class BabelTower(tf.keras.Model):
    """Short summary.
    the main model
    Args:


    Attributes:
        tower_of_babel (type): Description of parameter `tower_of_babel`.

    """

    def __init__(self,
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 batch_size,
                 unit_num,
                 embedding_size,
                 backforward=True,
                 embedding_matrix=True,
                 eager=True):
        super(BabelTower, self).__init__()
        self.batch_size = batch_size
        self.unit_num = unit_num
        self.embedding_size = embedding_size
        self.bw = backforward
        self.time_major = True
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.embedding_matrix = embedding_matrix
        self.eager = eager
        self.layer_initializer()

    def layer_initializer(self):
        self.fw_encoder = RNNcoder(
            self.src_vocabulary_size,
            self.batch_size,
            self.unit_num,
            self.embedding_size,
            backforward=False,
            eager=self.eager)
        self.fw_final = self.fw_encoder.initialize_hidden_state()
        if self.bw:
            self.bw_encoder = RNNcoder(
                self.src_vocabulary_size,
                self.batch_size,
                self.unit_num,
                self.embedding_size,
                backforward=True,
                eager=self.eager)
            self.decoder_unit_num = self.unit_num * 2
            self.decoder_embedding_size = self.embedding_size * 2
            self.bw_final = self.fw_encoder.initialize_hidden_state()
        else:
            self.decoder_unit_num = self.unit_num
            self.decoder_embedding_size = self.embedding_size

        self.decoder = RNNcoder(
            self.tgt_vocabulary_size,
            self.batch_size,
            self.decoder_unit_num,
            self.decoder_embedding_size,
            eager=self.eager)
        self.project = tf.keras.layers.Dense(self.tgt_vocabulary_size)
        self.logit = tf.keras.layers.Softmax(-1)

    # @autograph.convert()
    def call(self, inputs):
        (src_input, tgt_input, src_length, tgt_length) = inputs
        fw_inputs = (src_input, src_length, self.fw_final, 0)
        fw_encoder_hidden, fw_final = self.fw_encoder(fw_inputs)
        if self.bw:
            bw_inputs = (src_input, src_length, self.bw_final, 0)
            bw_encoder_hidden, bw_final = self.bw_encoder(bw_inputs)
            self.encoder_hidden = tf.concat(
                (fw_encoder_hidden, bw_encoder_hidden), -1)
            self.encoder_final = tf.concat((fw_final, bw_final), -1)
        else:
            self.encoder_final = fw_final
            self.encoder_hidden = fw_encoder_hidden
        self.decoder_final = self.encoder_final
        decoder_inputs = (tgt_input, tgt_length, self.decoder_final,
                          self.encoder_hidden)
        self.decoder_hidden, self.decoder_final = self.decoder(decoder_inputs)
        projection = self.project(self.decoder_hidden)
        self.logits = self.logit(projection)
        return self.logits

    def get_state(self):
        """Short summary.

        Args:


        Returns:
            (self.encoder_final, self.encoder_hidden),
            (self.decoder_final, self.decoder_hidden)
        """
        return (self.encoder_final, self.encoder_hidden), (self.decoder_final,
                                                           self.decoder_hidden)

    def re_initialize_final_state(self):
        """Short summary.
        re initialize final state to initial RNN cell state
        Args:


        Returns:
            type: Description of returned object.

        """
        self.fw_final = self.fw_encoder.initialize_hidden_state()
        if self.bw:
            self.bw_final = self.fw_encoder.initialize_hidden_state()


class BabelTowerFactory():
    def __init__(self,
                 src_data_path,
                 tgt_data_path,
                 batch_size=64,
                 unit_num=128,
                 embedding_size=128,
                 backforward=True,
                 embedding_matrix=True,
                 eager=True):
        self.batch_size = batch_size
        self.unit_num = unit_num
        self.embedding_size = embedding_size
        self.bw = backforward
        self.time_major = True
        self.src_data_path = src_data_path
        self.tgt_data_path = tgt_data_path

        self.embedding_matrix = embedding_matrix
        self.eager = eager
        print("build model")

    def _get_data(self, source_data_path, tgt_data_path, batch_size):
        sentenceHelper = SentenceHelper(
            source_data_path, tgt_data_path, batch_size=batch_size)
        # dataset, src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2word = data_sentence_helper.prepare_data(
        #     source_data_path, tgt_data_path, batch_size=batch_size)
        # self.src_vocabulary_size = len(src_vocabulary)
        # self.tgt_vocabulary_size = len(tgt_vocabulary)
        # return dataset, (src_vocabulary, src_ids2word), (tgt_vocabulary,
        #                                                  tgt_ids2word)
        src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2word = sentenceHelper.prepare_vocabulary()
        self.src_vocabulary_size = len(src_vocabulary)
        self.tgt_vocabulary_size = len(tgt_vocabulary)
        return sentenceHelper

    def mini_model(self, batch=4):
        """Short summary.
            A very mine model is used to test model structure.
        Args:
            batch (type): Description of parameter `batch`.

        Returns:
            model dataset, (src_vocabulary,src_ids2word), (tgt_vocabulary,tgt_ids2word)

        """
        sentenceHelper = self._get_data(self.src_data_path, self.tgt_data_path,
                                        batch)
        return BabelTower(
            src_vocabulary_size=self.src_vocabulary_size,
            tgt_vocabulary_size=self.tgt_vocabulary_size,
            batch_size=batch,
            unit_num=4,
            embedding_size=4,
            eager=self.eager), sentenceHelper

    def small_model(self, batch=16, unit_num=16, embedding_size=16):
        """Short summary.
            A very small model is used to test model training.
        Args:
            batch (type): Description of parameter `batch`.

        Returns:
            model dataset, (src_vocabulary,src_ids2word), (tgt_vocabulary,tgt_ids2word)

        """
        sentenceHelper = self._get_data(self.src_data_path, self.tgt_data_path,
                                        batch)
        return BabelTower(
            src_vocabulary_size=self.src_vocabulary_size,
            tgt_vocabulary_size=self.tgt_vocabulary_size,
            batch_size=batch,
            unit_num=unit_num,
            embedding_size=embedding_size,
            eager=self.eager), sentenceHelper

    def large_model(self, batch_size=32, unit_num=128, embedding_size=128):
        """Short summary.
            A full model is used to train final model.
        Args:
            batch (type): Description of parameter `batch`.

        Returns:
            model dataset, (src_vocabulary,src_ids2word), (tgt_vocabulary,tgt_ids2word)

        """
        sentenceHelper = self._get_data(self.src_data_path, self.tgt_data_path,
                                        batch_size)
        return BabelTower(
            src_vocabulary_size=self.src_vocabulary_size,
            tgt_vocabulary_size=self.tgt_vocabulary_size,
            batch_size=batch_size,
            unit_num=unit_num,
            embedding_size=embedding_size,
            eager=self.eager), sentenceHelper
