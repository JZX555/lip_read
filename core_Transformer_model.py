import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import core as core_layer
import numpy as np


class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
    """

    def __init__(self,
                 epsilon=1e-9,
                 gamma_initializer="ones",
                 beta_initializer="zeros"):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gamma_kernel = self.add_weight(
            shape=(input_dim),
            name="gamma",
            initializer=self.gamma_initializer)
        self.beta_kernel = self.add_weight(
            shape=(input_dim), name="beta", initializer=self.beta_initializer)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (variance**2 + self.epsilon)
        output = self.gamma_kernel * normalized + self.beta_kernel
        return output


class Mask(tf.keras.layers.Layer):
    """
        We have two mask:
            1.Padding_Mask: each attention need to use it to padding the time major
            with -INF when their length are different;
            2.Sequence_Mask: we use this in Mask_Multi_Head_Attention to hide the
            future information;
            args:
                scaled: [b, t_q, t_k]
    """

    def __init__(self):
        super(Mask, self).__init__()
        self.padding_num = -2**32 + 1

    def call(self, inputs, type):
        scaled, Q, K = inputs
        self.type = type
        if self.type in ("k", "key", "keys"):
            mask = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, tf.shape(Q)[1], 1])

            paddings = tf.ones_like(scaled) * self.padding_num
            output = tf.where(tf.equal(mask, 0), paddings, scaled)

        elif self.type in ("q", "query", "queries"):
            mask = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, [1, 1, tf.shape(K)[1]])

            output = scaled * mask

        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(scaled[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals).to_dense()
            masks = tf.tile(
                tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * self.padding_num
            output = tf.where(tf.equal(masks, 0), paddings, scaled)

        else:
            print("need right type in mask!")

        return output


class Scaled_Dot_Product_Attention(tf.keras.layers.Layer):
    """
        The Scaled_Dot_Product_Attention is the basic of Multi_Head_Attention;
        It composed of the following layers:
            Attention: MatMul(Q, K)->scale->Mask->softmax
            output: Attention * V
        args:
        Q: [b, t, d_q]
        K: [b, t, d_k]
        V: [b, t, d_v]
        tips: d_q = d_k = d_v
    """

    def __init__(self, causality=False, dropout=1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.causality = causality
        self.dropout = min(1., max(0., dropout))
        self.Mask = Mask()

    def call(self, inputs):
        Q, K, V = inputs
        d_k = Q.get_shape().as_list()[-1]

        mat = tf.keras.backend.dot(Q, tf.transpose(K, [0, 2, 1]))
        scale = mat / (d_k**0.5)

        # the two mask opt maybe put together:
        # 1. mask(key)->mask(query)->softmax
        # 2. mask(key)->softmax->mask(query)
        # we use the second choice;
        # tips: key must use before query
        mask_input = (scale, Q, K)
        mask = self.Mask(mask_input, "key")

        if (self.causality):
            mask = self.Mask((mask, None, None), "future")

        softmax = tf.keras.layers.Softmax()(mask)
        self.attention = softmax

        mask_input = (softmax, Q, K)
        softmax = self.Mask(mask_input, "query")

        if 0 < self.dropout < 1:
            dropout_mask = tf.keras.backend.dropout(
                tf.ones_like(softmax), self.dropout)
            softmax = softmax * dropout_mask

        output = tf.keras.backend.dot(softmax, V)

        return output

    def get_attention(self):
        return self.attention


class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads=8,
                 num_units=512,
                 causality=False,
                 dropout=1,
                 ln=None,
                 query_initializer='glorot_uniform',
                 key_initializer='glorot_uniform',
                 value_initializer='glorot_uniform',
                 liner_initializer='glorot_uniform'):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.num_units = num_units
        self.causality = causality
        self.dropout = min(1., max(0., dropout))
        self.ln = ln

        self.query_initializer = tf.keras.initializers.get(query_initializer)
        self.key_initializer = tf.keras.initializers.get(key_initializer)
        self.value_initializer = tf.keras.initializers.get(value_initializer)
        self.liner_initializer = tf.keras.initializers.get(liner_initializer)

        self.attention = Scaled_Dot_Product_Attention(self.causality,
                                                      self.dropout)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.query_kernel = self.add_weight(
            shape=(input_dim, self.num_units),
            name="query_kernel",
            initializer=self.query_initializer)

        self.key_kernel = self.add_weight(
            shape=(input_dim, self.num_units),
            name="key_kernel",
            initializer=self.key_initializer)

        self.value_kernel = self.add_weight(
            shape=(input_dim, self.num_units),
            name="value_kernel",
            initializer=self.value_initializer)

        self.liner_kernel = self.add_weight(
            shape=(self.num_units, input_dim),
            name="liner_kernel",
            initializer=self.liner_initializer)

    def call(self, inputs):
        Q, K, V = inputs
        Q = tf.keras.backend.dot(Q, self.query_kernel)
        K = tf.keras.backend.dot(K, self.key_kernel)
        V = tf.keras.backend.dot(V, self.value_kernel)

        split_Q = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)
        split_K = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)
        split_V = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=0)

        attention_input = (split_Q, split_K, split_V)

        context = self.attention(attention_input)
        context = tf.concat(tf.split(context, self.num_heads, axis=0), axis=-1)

        liner = tf.keras.backend.dot(context, self.liner_kernel)

        # ResNet
        output = liner + Q
        output = self.ln(output)

        return output


class Feed_Forward_Network(tf.keras.layers.Layer):
    """
        FFN
    """

    def __init__(self,
                 num_units=2048,
                 bias=True,
                 ln=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super(Feed_Forward_Network, self).__init__()
        self.num_units = num_units
        self.bias = bias
        self.ln = ln

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.liner_kernel = self.add_weight(
            shape=(input_dim, self.num_units),
            name="liner_kernel",
            initializer=self.kernel_initializer)
        self.FFN_kernel = self.add_weight(
            shape=(self.num_units, input_dim),
            name="FFN_kernel",
            initializer=self.kernel_initializer)

        if (self.bias):
            self.liner_bias = self.add_weight(
                shape=self.num_units,
                name="liner_bias",
                initializer=self.bias_initializer)
            self.FFN_bias = self.add_weight(
                shape=input_dim,
                name="FFN_bias",
                initializer=self.bias_initializer)

    def call(self, inputs):
        liner = tf.keras.backend.dot(inputs, self.liner_kernel)
        if (self.bias):
            liner = liner + self.liner_bias

        FFN = tf.keras.backend.dot(liner, self.FFN_kernel)
        if (self.bias):
            FFN = FFN + self.FFN_bias

        output = self.ln(FFN + inputs)

        return output


class Positional_Encoding(tf.keras.layers.Layer):
    """
        Add the position information to encoding, which:
            PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        args:
            pos: the position of the word
            i: the word embedding dimention index
            d_model: the word embedding dimention
    """

    def __init__(self, max_seq_len):
        super(Positional_Encoding, self).__init__()
        self.max_seq_len = max_seq_len

    def call(self, inputs):
        dim = inputs.get_shape().as_list()[-1]
        b, t = tf.shape(inputs)[0], tf.shape(inputs)[1]

        position_idx = tf.tile(tf.expand_dims(tf.range(t), 0), [b, 1])

        # position_enc: [max_seq_len, dim]
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / dim) for i in range(dim)]
             for pos in range(self.max_seq_len)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        output = tf.nn.embedding_lookup(position_enc, position_idx)

        return tf.to_float(output)


class Transformer_cell(tf.keras.layers.Layer):
    """
        wrap transformer to seq2seq cell style
    """

    def __init__(self,
                 max_seq_len,
                 vocabulary_size,
                 num_units=512,
                 num_heads=6,
                 num_layers=6,
                 dropout=1,
                 masked_attention=False,
                 name=None):
        super(Transformer_cell, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.masked_attention = masked_attention

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.ln = LayerNorm()
        self.att = []
        self.ffn = []
        for i in range(self.num_layers):
            self.att.append(
                Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    ln=self.ln))
            self.ffn.append(
                Feed_Forward_Network(num_units=4 * self.num_units, ln=self.ln))
        if self.masked_attention:
            for i in range(self.num_layers):
                self.mask.append(
                    Multi_Head_Attention(
                        num_heads=self.num_heads,
                        num_units=self.num_units,
                        causality=True,
                        dropout=self.dropout,
                        ln=self.ln))

    def call(self, inputs, K_V=None):
        if self.masked_attention and K_V is None:
            assert ('Using maksed_attention, please give K_V')
        src_input = inputs * self.num_units**0.5

        positional_input = Positional_Encoding(self.max_seq_len)(src_input)

        inputs = src_input + positional_input

        for i in range(self.num_encoder_layers):
            if self.masked_attention:
                multi_att = self.mask[i]((inputs, inputs, inputs))
                multi_att = self.att[i]((multi_att, K_V, K_V))
            else:
                multi_att = self.att[i]((inputs, inputs, inputs))
            outputs = self.ffn[i](multi_att)
            inputs = outputs

        return outputs, K_V

    def inference(self):
        self.dropout = 1

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units


class Daedalus(tf.keras.Model):
    """
        Transformer
    """

    def __init__(self,
                 max_seq_len,
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 embeded_size=512,
                 batch_size=64,
                 num_units=512,
                 num_heads=6,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.4,
                 sos_id=1,
                 eos_id=2):
        super(Daedalus, self).__init__()
        self.max_seq_len = max_seq_len
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.embeded_size = embeded_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.projection = core_layer.Dense(self.tgt_vocabulary_size, name='fc')

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # input_dim = input_shape[-1]
        self.src_embedding = tf.keras.layers.Embedding(
            self.src_vocabulary_size,
            self.embedding_size,
            name='src_embedding')
        self.Encoder = Transformer_cell(
            max_seq_len=self.max_seq_len,
            vocabulary_size=self.src_vocabulary_size,
            num_heads=self.num_heads,
            num_units=self.num_units,
            num_layers=self.num_encoder_layers,
            dropout=self.dropout,
            masked_attention=False,
            name='encoder')
        self.tgt_embedding = tf.keras.layers.Embedding(
            self.tgt_vocabulary_size,
            self.embedding_size,
            name='tgt_embedding')
        self.Decoder = Transformer_cell(
            max_seq_len=self.max_seq_len,
            vocabulary_size=self.tgy_vocabulary_size,
            num_heads=self.num_heads,
            num_units=self.num_units,
            num_layers=self.num_Decoder_layers,
            dropout=self.dropout,
            masked_attention=True,
            name='decoder')
        self.projection = core_layer.Dense(self.tgt_vocabulary_size, name='fc')

    def call(self, inputs, train=True):
        if train:
            return self.train_model(inputs)
        else:
            return self.beam_search(inputs)

    def train_model(self, inputs):

        src_input, tgt_input, _, _ = inputs
        embedding_src_input = self.src_embedding(src_input)
        enc = self.Encoder(embedding_src_input)

        embedding_tgt_input = self.tgt_embedding(tgt_input)
        self.outputs, _ = self.Decoder(embedding_tgt_input, enc)

        projection = self.projection(self.outputs)
        logits = tf.keras.layers.Softmax()(projection)
        return logits

    def get_raw_outputs(self):
        try:
            return self.outputs
        except Exception:
            assert ('no Decoder Raw Output at all')

    def beam_search(self, inputs, beam_width=5):
        self.dropout_manager(1)
        src_input, tgt_input, src_length, tgt_length = inputs
        enc = self.Encoder(src_input)
        start_tokens = tf.constant(
            value=self.sos_id, shape=[self.batch_size], dtype=tf.int32)
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            enc, multiplier=beam_width)
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.Decoder,
            embedding=self.tgt_embedding,
            start_tokens=start_tokens,
            end_token=self.eos_id,
            initial_state=decoder_initial_state,
            output_layer=self.projection,
            beam_width=beam_width)

        # Dynamic decoding
        self.pred, state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.inference_length)
        final_resualt = tf.contrib.seq2seq.FinalBeamSearchDecoderOutput(
            self.pred, state)
        return final_resualt

    def get_raw_pred(self):
        try:
            return self.pred
        except Exception:
            assert ('no Decoder Raw pred at all')
