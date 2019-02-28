# encoder= utf-8
"""
    Artist:
        Barid

    Seek perfect in imperfection.
"""
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import RNN
from data_sentence_helper import SentenceHelper
import model_helper
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import core as core_layer


class AlignAttention(tf.keras.layers.Layer):
    """
        batch major
        context = [b, t, e]
    """

    def __init__(self, unit_num, context=None, name=None, **kwargs):
        super(AlignAttention, self).__init__(name=name, **kwargs)
        self.units = unit_num
        self.context = context

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        # self.batch_size = input_shape[0]
        # context_dim = self.context.shape[1]
        # self.input_kernel = self.add_weight(
        #     name="attention_score_W", shape=[input_dim])
        # self.context_kernel = self.add_weight(
        #     name="attention_score_U", shape=[input_dim])
        # self.score_kernel = self.add_weight(
        #     name="attention_score_V", shape=[input_dim])
        # self.attention_score = tf.keras.layers.Softmax()
        #
        # self.context = model_helper.time_major_helper(self.context)
        # self.context = tf.reshape(self.context,
        #                           [-1, self.batch_size * input_dim])
        self.W = tf.keras.layers.Dense(input_dim)
        self.U = tf.keras.layers.Dense(input_dim)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """
            Implementing the method from mentioned paper,
            it follows stand attention pipe line:
                energy -> softmax -> context

            batch major
            inputs = [b, v]
        """
        if self.context is not None:
            hidden_with_time_axis = tf.expand_dims(inputs, 1)
            score = self.V(
                tf.nn.tanh(
                    self.W(self.context) + self.U(hidden_with_time_axis)))
            # attention_score = self.attention_score(score, axis=0)
            self.attention_weights = tf.nn.softmax(score, axis=1)
            self.context_vector = self.attention_weights * self.context
            self.context_vector = tf.reduce_sum(self.context_vector, axis=1)
            return self.context_vector, self.attention_weights

    @property
    def get_attention_weight(self):
        return self.attention_weights

    def set_attenton_context(self, context):
        self.context = context


class AlignCell(tf.keras.layers.Layer):
    """
        The new verison of rnn cell is implemented.
        And I have changed the old core_seq2seq_model to
        core_seq2seq_model_deprecated,


        Acknowledgement:
        Bahdanau, D., Cho, K., & Bengio, Y. (n.d.).
        NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE.
        Retrieved from http://cs224d.stanford.edu/papers/nmt.pdf
    """

    def __init__(
            self,
            # vocabulary_size,
            # batch_size,
            units,
            # embedding_size,
            attention=False,
            bias=True,
            backforward=False,
            embedding_size=256,
            projection=0,
            # embedding_matrix=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            dropout=1,
            name=None,
            **kwargs):
        super(AlignCell, self).__init__(name=name, **kwargs)
        self.units = units
        # self.state_size = units
        # self.output_size = units
        # self.embedding_size = embedding_size
        self.use_bias = bias
        self.attention = attention
        self.time_major = True
        self.dropout = min(1., max(0., dropout))
        # self.vocabulary_size = vocabulary_size
        # self.embedding_matrix = embedding_matrix
        # self.embedding_helper = tf.keras.layers.Embedding(
        #     self.vocabulary_size, self.embedding_size)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(
            recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.embedding_size = embedding_size
        self.projection = projection
        if self.projection != 0:
            self.dense_output = tf.keras.layers.Dense(
                int(self.projection), name='output_dense')
        else:
            self.dense_output = tf.keras.layers.Dense(
                int(self.units), name='output_dense')
        if self.embedding_size != self.units:
            self.dense_input = tf.keras.layers.Dense(
                int(self.units), name='input_dense')
        self.built = True
        if self.attention:
            self.attention_wrapper = AlignAttention(
                self.units, 0, name='attention')
        # used to control weights size
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.embedding_size != self.units:
            input_dim = self.units
        self.batch_size = input_shape[0]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer=self.kernel_initializer)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)
        # if self.attention_context is not None:
        #     self.attention = True
        # else:
        #     self.attention = False
        if self.attention:
            self.attention_kernel = self.add_weight(
                shape=(input_dim, self.units * 3),
                name='attention_kernel',
                initializer=self.kernel_initializer)
        if self.use_bias:
            bias_shape = (self.units * 3, )
            self.bias = self.add_weight(
                shape=bias_shape,
                name='bias',
                initializer=self.bias_initializer)
            self.input_bias, self.recurrent_bias = self.bias, None
        else:
            self.bias = None

    def call(self, inputs, states):
        if self.embedding_size != self.units:
            inputs = self.dense_input(inputs)
        if isinstance(states, list) or isinstance(states, tuple):
            h_tm1 = states[0]
            inference = False
        else:
            inference = True
            h_tm1 = states
        # import pdb; pdb.set_trace()
        if 0 < self.dropout < 1:
            self.dropout_mask = model_helper.dropout_mask_helper(
                array_ops.ones_like(inputs), self.dropout, training=True)
            h_tm1 = h_tm1 * self.dropout_mask
            inputs = inputs * self.dropout_mask
        x_z = tf.keras.backend.dot(inputs, self.kernel[:, :self.units])
        x_r = tf.keras.backend.dot(inputs,
                                   self.kernel[:, self.units:self.units * 2])
        x_h = tf.keras.backend.dot(inputs, self.kernel[:, self.units * 2:])

        if self.use_bias:
            x_z = tf.keras.backend.bias_add(x_z, self.input_bias[:self.units])
            x_r = tf.keras.backend.bias_add(
                x_r, self.input_bias[self.units:self.units * 2])
            x_h = tf.keras.backend.bias_add(x_h,
                                            self.input_bias[self.units * 2:])
        if self.attention:
            context_vector, attention_weights = self.attention_wrapper(h_tm1)
            if 0 < self.dropout < 1:
                context_vector = context_vector * self.dropout_mask
            c_z = tf.keras.backend.dot(context_vector,
                                       self.attention_kernel[:, :self.units])
            c_r = tf.keras.backend.dot(
                context_vector,
                self.attention_kernel[:, self.units:self.units * 2])
            c_h = tf.keras.backend.dot(
                context_vector, self.attention_kernel[:, self.units * 2:])
        else:
            c_z = 0
            c_r = 0
            c_h = 0
        recurrent_z = tf.keras.backend.dot(
            h_tm1, self.recurrent_kernel[:, :self.units])
        recurrent_r = tf.keras.backend.dot(
            h_tm1, self.recurrent_kernel[:, self.units:self.units * 2])
        z = tf.nn.sigmoid(x_z + recurrent_z + c_z)
        r = tf.nn.sigmoid(x_r + recurrent_r + c_r)
        recurrent_h = tf.keras.backend.dot(
            r * h_tm1, self.recurrent_kernel[:, self.units * 2:])
        h_bar = tf.nn.tanh(x_h + recurrent_h + c_h)

        h = z * h_tm1 + (1 - z) * h_bar
        if self.projection is not self.units:
            if inference:
                return self.dense_output(h), h

            return self.dense_output(h), [h]

        else:
            if inference:
                return h, h

            return h, [h]

    @property
    def get_attention_weight(self):
        if self.attention:
            return self.attention_wrapper.get_attenton_weights
        else:
            return 0

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def set_attention_context(self, context):
        self.attention = True
        self.attention_context = context
        self.attention_wrapper.set_attenton_context(context)

    def set_dropout(self, pro):
        self.dropout = pro

    def get_initial_state(self):
        self.build()
        return tf.zeros((self.batch_sz, self.units))


class Align(RNN):
    def __init__(self,
                 units,
                 attention=False,
                 return_sequences=True,
                 return_state=True,
                 activation="tanh",
                 embedding_size=256,
                 projection=0,
                 name=None):
        # State_size is defined as same as units in cells and which means, we dont
        # use projection for cell state
        self.state_size = units
        self.output_size = units
        cell = AlignCell(
            units,
            attention,
            embedding_size=embedding_size,
            projection=projection)
        super(Align, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            name=name)

    def call(self, inputs, initial_state=None):
        """
            batch major
        """
        return super(Align, self).call(inputs, initial_state=initial_state)

    def set_attention(self, context):
        self.cell.set_attention_context(context)

    def set_dropout(self, pro):
        self.cell.set_dropout(pro)


class TheOldManAndSea(tf.keras.Model):
    def __init__(self,
                 units,
                 batch_size,
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 embedding_size=256,
                 activation=None,
                 inference_length=50,
                 bidirectional=True):

        super(TheOldManAndSea, self).__init__()
        self.units = units
        self.batch_size = batch_size
        self.decoder_units = units
        self.bidirectional = bidirectional
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.embedding_size = embedding_size
        self.activation = activation
        self.inference_length = inference_length
        self.fw_encoder = Align(
            self.units, activation=self.activation,
            name='fw_encoder')  # No activation used
        # self.fw_encoder.set_dropout(self.dropout)
        if self.bidirectional:
            self.bw_encoder = Align(
                self.units, activation=self.activation, name='bw_encoder')
            self.decoder_units = units * 2
            # self.bw_encoder.set_dropout(self.dropout)
        self.encoder_zero_state = tf.zeros((self.batch_size, self.units))
        self.decoder = Align(
            self.decoder_units,
            attention=True,
            activation=self.activation,
            embedding_size=self.embedding_size,
            projection=self.embedding_size,
            name='decoder')
        # self.decoder.set_dropout(self.dropout)

    def feed_encoder(self, src_input, src_length):
        src_input = self.src_embedding(src_input)
        # src_input = self.src_dense(src_input)
        self.fw_encoder_hidden, self.fw_encoder_state = self.fw_encoder(
            src_input, initial_state=self.encoder_zero_state)

        if self.bidirectional:
            reversed_src_input = tf.reverse_sequence(
                src_input, src_length, seq_axis=1, batch_axis=0)
            self.bw_encoder_hidden, self.bw_encoder_state = self.bw_encoder(
                reversed_src_input, initial_state=self.encoder_zero_state)
            encoder_hidden = tf.concat(
                (self.fw_encoder_hidden, self.bw_encoder_hidden), -1)
            encoder_state = tf.concat(
                (self.fw_encoder_state, self.bw_encoder_state), -1)
        else:
            self.encoder_hidden = self.fw_encoder_hidden
        self.encoder_hidden = tf.keras.layers.BatchNormalization()(
            encoder_hidden)
        self.encoder_state = tf.keras.layers.BatchNormalization()(
            encoder_state)
        self.encoder_hidden = tf.nn.tanh(encoder_hidden)
        self.encoder_state = tf.nn.tanh(encoder_state)
        return self.encoder_hidden, self.encoder_state

    def feed_decoder(self, tgt_input, tgt_length, encoder_hidden,
                     encoder_state):
        tgt_input = self.tgt_embedding(tgt_input)
        # tgt_input = self.tgt_dense(tgt_input)
        self.decoder.set_attention(encoder_hidden)
        decoder_hidden, decoder_state = self.decoder(
            tgt_input, initial_state=encoder_state)
        decoder_hidden = tf.keras.layers.BatchNormalization()(decoder_hidden)
        decoder_state = tf.keras.layers.BatchNormalization()(decoder_state)
        self.decoder_hidden = tf.nn.tanh(decoder_hidden)
        self.decoder_state = tf.nn.tanh(decoder_state)
        projection = self.projection(decoder_hidden)
        self.logit = tf.keras.layers.Softmax()(projection)
        return self.logit, self.decoder_hidden, self.decoder_state

    def build(self, input_shape):
        self.src_embedding = tf.keras.layers.Embedding(
            self.src_vocabulary_size,
            self.embedding_size,
            name='src_embedding')
        self.tgt_embedding = tf.keras.layers.Embedding(
            self.tgt_vocabulary_size,
            self.embedding_size,
            name='tgt_embedding')
        # self.src_dense = tf.keras.layers.Dense(self.units, name='src_dense')
        # self.tgt_dense = tf.keras.layers.Dense(
        #     self.decoder_units, name='tgt_dense')
        # self.hidden_dense = tf.keras.layers.Dense(self.embedding_size)
        # self.projection = tf.keras.layers.Dense(self.tgt_vocabulary_size, )
        self.projection = core_layer.Dense(self.tgt_vocabulary_size, name='fc')

    def call(self, inputs, train=None, dropout=1):
        src_input, tgt_input, src_length, tgt_length = inputs
        if dropout < 1:
            self.fw_encoder.set_dropout(dropout)
            if self.bidirectional:
                self.bw_encoder.set_dropout(dropout)
        encoder_hidden, encoder_state = self.feed_encoder(
            src_input, src_length)
        decoder_hidden = self.feed_decoder(tgt_input, tgt_length,
                                           encoder_hidden, encoder_state)
        logits, decoder_hidden, decoder_state = self.feed_decoder(
            tgt_input, tgt_length, encoder_hidden, encoder_state)
        return logits, decoder_hidden, decoder_state

    def beam_search(self, inputs, sos_id, eos_id, beam_width=5):
        src_input, tgt_input, src_length, tgt_length = inputs
        encoder_hidden, encoder_state = self.feed_encoder(
            src_input, src_length)
        start_tokens = tf.constant(
            value=sos_id, shape=[self.batch_size], dtype=tf.int32)
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=2)
        encoder_hidden = tf.contrib.seq2seq.tile_batch(
            encoder_hidden, multiplier=2)
        self.decoder.set_attention(encoder_hidden)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder.cell,
            embedding=self.tgt_embedding,
            start_tokens=start_tokens,
            end_token=eos_id,
            initial_state=decoder_initial_state,
            output_layer=self.projection,
            beam_width=2)

        # Dynamic decoding
        pred, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.inference_length)
        return pred


class TheOldManAndSea_factory():
    def __init__(
            self,
            src_data_path,
            tgt_data_path,
            batch_size=64,
            shuffle=100,
            unit_num=128,
            embedding_size=128,
            bidirectional=True,
    ):
        self.batch_size = batch_size
        self.unit_num = unit_num
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.time_major = True
        self.src_data_path = src_data_path
        self.tgt_data_path = tgt_data_path
        self._get_data
        print("build model")

    def _get_data(self, batch_size):
        sentenceHelper = SentenceHelper(
            self.src_data_path,
            self.tgt_data_path,
            batch_size=batch_size,
            shuffle=100000)
        src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2word = sentenceHelper.prepare_vocabulary(
        )
        self.src_vocabulary_size = len(src_vocabulary)
        self.tgt_vocabulary_size = len(tgt_vocabulary)
        return sentenceHelper

    def test_model(self, batch_size=16, unit=16):
        sentenceHelper = self._get_data(batch_size)
        test_model = TheOldManAndSea(unit, batch_size,
                                     self.src_vocabulary_size,
                                     self.tgt_vocabulary_size)
        return test_model, sentenceHelper

    def small_model(self, batch_size=64, unit=64):
        sentenceHelper = self._get_data(batch_size)
        small_model = TheOldManAndSea(unit, batch_size,
                                      self.src_vocabulary_size,
                                      self.tgt_vocabulary_size)
        return small_model, sentenceHelper

    def large_model(self, batch_size=128, unit=128):
        sentenceHelper = self._get_data(batch_size)
        large_model = TheOldManAndSea(unit, batch_size,
                                      self.src_vocabulary_size,
                                      self.tgt_vocabulary_size)
        return large_model, sentenceHelper


# class AlignAndTranslate(tf.keras.Model):
#     """
#         inputs should be time major
#     """
#
#     def __init__(self, units, batch_size, bidirectional=True):
#         super(AlignAndTranslate, self).__init__()
#         self.units = units
#         self.batch_size = batch_size
#         self.decoder_units = units
#         self.bi = bidirectional
#         self.fw_encoder = Align(self.units, name='fw_encoder')
#         if self.bi:
#             self.bw_encoder = Align(self.units, name='bw_encoder')
#             self.decoder_units = units * 2
#         self.decoder = Align(
#             self.decoder_units, attention_context=0, name='decoder')
#         self.encoder_zero_state = tf.zeros((self.batch_size, self.units))
#
#     def encoder(self, src_input, src_length):
#         self.fw_encoder_hidden, self.fw_encoder_state = self.fw_encoder(
#             src_input, initial_state=self.encoder_zero_state)
#         if self.bi:
#             reversed_src_input = tf.reverse_sequence(
#                 src_input, src_length, seq_axis=1, batch_axis=0)
#             self.bw_encoder_hidden, self.bw_encoder_state = self.bw_encoder(
#                 reversed_src_input, initial_state=self.encoder_zero_state)
#             self.encoder_hidden = tf.concat(
#                 (self.fw_encoder_hidden, self.bw_encoder_hidden), -1)
#             self.encoder_state = tf.concat(
#                 (self.fw_encoder_state, self.bw_encoder_state), -1)
#         else:
#             self.encoder_hidden = self.fw_encoder_hidden
#         return self.encoder_hidden, self.encoder_state
#
#     def decoder(self, tgt_input, encoder_hidden, encoder_state):
#         self.decoder.set_attention(encoder_hidden)
#         self.decoder_hidden, self.decoder_state = self.decoder(
#             tgt_input, initial_state=encoder_state)
#         return self.decoder_hidden, self.decoder_state
#
#     def call(self, inputs):
#         (src_input, tgt_input, src_length, tgt_length) = inputs
#         # attention_context = model_helper.batch_major_helper(self.encoder_hidden)
#         encoder_hidden, encoder_state = self.encoder(src_input, src_length)
#         decoder_hidden, decoder_state = self.decoder(tgt_input, encoder_hidden,
#                                                      encoder_state)
#         return decoder_hidden, decoder_state
#
