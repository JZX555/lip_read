import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.layers import core as core_layer
# import numpy as np
import beam_search


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, pad_id, name="embedding"):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(EmbeddingSharedWeights, self).__init__(name="embedding")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.shared_weights = self.add_variable(
            shape=[self.vocab_size, self.hidden_size],
            name="shared_weights",
            initializer=tf.random_normal_initializer(0., self.hidden_size
                                                     **-0.5))

    def call(self, inputs):
        """Get token embeddings of x.
    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
        mask = tf.to_float(tf.not_equal(inputs, self.pad_id))
        embeddings = tf.gather(self.shared_weights, inputs)
        embeddings *= tf.expand_dims(mask, -1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.hidden_size**0.5

        return embeddings

    def linear(self, inputs):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(inputs, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


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
        self.gamma_kernel = self.add_variable(
            shape=(input_dim),
            name="gamma",
            initializer=self.gamma_initializer)
        self.beta_kernel = self.add_variable(
            shape=(input_dim), name="beta", initializer=self.beta_initializer)
        self.built = True

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) * tf.rsqrt(variance + self.epsilon)
        output = self.gamma_kernel * normalized + self.beta_kernel
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

    def __init__(self, num_units, min_timescale=1.0, max_timescale=1.0e4):
        super(Positional_Encoding, self).__init__()
        # self.max_seq_len = max_seq_len
        self.num_units = num_units
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def call(self, inputs):
        position = tf.to_float(tf.range(inputs))
        num_timescales = self.num_units // 2
        log_timescale_increment = (
            tf.math.log(float(self.max_timescale) / float(self.min_timescale))
            / (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal


class Feed_Forward_Network(tf.keras.layers.Layer):
    """
        FFN
    """

    def __init__(self,
                 num_units=2048,
                 bias=True,
                 ln=None,
                 name=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super(Feed_Forward_Network, self).__init__(name=name)
        self.num_units = num_units
        self.bias = bias
        self.ln = LayerNorm()

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.linear_kernel = self.add_variable(
            shape=(input_dim, self.num_units),
            name="linear_kernel",
            initializer=self.kernel_initializer)
        self.FFN_kernel = self.add_variable(
            shape=(self.num_units, input_dim),
            name="FFN_kernel",
            initializer=self.kernel_initializer)

        if (self.bias):
            self.linear_bias = self.add_variable(
                shape=self.num_units,
                name="linear_bias",
                initializer=self.bias_initializer)
            self.FFN_bias = self.add_variable(
                shape=input_dim,
                name="FFN_bias",
                initializer=self.bias_initializer)
        self.built = True

    def call(self, inputs):
        inp = inputs
        inputs = self.ln(inputs)
        liner = tf.keras.backend.dot(inputs, self.linear_kernel)
        if (self.bias):
            liner = liner + self.linear_bias

        FFN = tf.keras.backend.dot(liner, self.FFN_kernel)
        if (self.bias):
            FFN = FFN + self.FFN_bias

        output = inp + FFN

        return output


class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads=8,
                 num_units=512,
                 dropout=1,
                 ln=None,
                 name=None,
                 masked_attention=False,
                 query_initializer='glorot_uniform',
                 key_initializer='glorot_uniform',
                 value_initializer='glorot_uniform',
                 liner_initializer='glorot_uniform'):
        super(Multi_Head_Attention, self).__init__(name=name)
        self.num_heads = num_heads
        self.num_units = num_units
        self.dropout = min(1., max(0., dropout))
        self.ln = LayerNorm()
        self.masked_attention = masked_attention
        self.query_initializer = tf.keras.initializers.get(query_initializer)
        self.key_initializer = tf.keras.initializers.get(key_initializer)
        self.value_initializer = tf.keras.initializers.get(value_initializer)
        self.liner_initializer = tf.keras.initializers.get(liner_initializer)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.query_kernel = self.add_variable(
            shape=(input_dim, self.num_units),
            name="query_kernel",
            initializer=self.query_initializer)

        self.key_kernel = self.add_variable(
            shape=(input_dim, self.num_units),
            name="key_kernel",
            initializer=self.key_initializer)

        self.value_kernel = self.add_variable(
            shape=(input_dim, self.num_units),
            name="value_kernel",
            initializer=self.value_initializer)

        self.liner_kernel = self.add_variable(
            shape=(input_dim, self.num_units),
            name="liner_kernel",
            initializer=self.liner_initializer)
        self.built = True

    def call(self, inputs, K_V=None, bias=0, cache=None):
        Q = inputs
        if K_V is None:
            K_V = (Q, Q)
        Q = self.ln(Q)
        K, V = K_V
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            K = tf.concat([cache["K"], K], axis=1)
            V = tf.concat([cache["V"], V], axis=1)
            # Update cache
            cache["K"] = K
            cache["V"] = V
        with tf.name_scope('Q_K_V_linear_projection'):
            Q = tf.keras.backend.dot(Q, self.query_kernel)
            K = tf.keras.backend.dot(K, self.key_kernel)
            V = tf.keras.backend.dot(V, self.value_kernel)
        attention_output = self.scale_dot_product_attention(Q, K, V, bias)
        with tf.name_scope('attention_linear_projection'):
            liner = tf.keras.backend.dot(attention_output, self.liner_kernel)

        # ResNet
        # output = self.ln(liner)
        output = liner + inputs

        return output

    def scale_dot_product_attention(self, Q, K, V, bias):
        with tf.name_scope('scale_dot_product'):
            Q = self.split_heads(Q)
            K = self.split_heads(K)
            V = self.split_heads(V)
            d_Q = tf.cast(tf.shape(Q)[-1], tf.float32)
            # mat = tf.keras.backend.batch_dot(Q, tf.transpose(K, [0, 2, 1])) # this one is ok as well
            mat = tf.matmul(Q, K, transpose_b=True)
            scale = mat / (tf.math.sqrt(d_Q))
            scale += bias
            softmax = tf.keras.layers.Softmax(name="attention_weights")(scale)
            self.attention = softmax
            if 0 < self.dropout < 1:
                dropout_mask = tf.keras.backend.dropout(
                    tf.ones_like(softmax), self.dropout)
                softmax = softmax * dropout_mask

            # output = tf.keras.backend.dot(softmax, V)
            output = tf.matmul(softmax, V)
            attention_output = self.combine_heads(output)
            return attention_output

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.num_units // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(
                x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.num_units])

    def get_attention(self):
        return self.attention


class Daedalus(tf.keras.Model):
    """
        Transformer
    """

    def __init__(self,
                 max_seq_len,
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 embedding_size=512,
                 batch_size=64,
                 num_units=512,
                 num_heads=6,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.4,
                 sos_id=1,
                 eos_id=2,
                 pad_id=2):
        super(Daedalus, self).__init__(name='transformer')
        self.max_seq_len = max_seq_len
        self.vocabulary_size = max(src_vocabulary_size, tgt_vocabulary_size)
        # self.src_vocabulary_size = src_vocabulary_size
        # self.tgt_vocabulary_size = tgt_vocabulary_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.possition_encoding = Positional_Encoding(self.num_units)
        self.en_att = []
        self.en_ffn = []
        self.de_att = []
        self.de_ffn = []
        self.de_mask_att = []
        self.negtive_infinit = -1e32
        for i in range(self.num_encoder_layers):
            self.en_att.append(
                Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="enc_multi_att_%d" % i))
            self.en_ffn.append(
                Feed_Forward_Network(
                    num_units=4 * self.num_units, name="enc_ffn_%d" % i))
        for i in range(self.num_decoder_layers):
            self.de_att.append(
                Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="de_multi_att_%d" % i))
            self.de_ffn.append(
                Feed_Forward_Network(
                    num_units=4 * self.num_units, name="dec_ffn_%d" % i))
            self.de_mask_att.append(
                Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=True,
                    name="masked_multi_att_%d" % i))
        self.shared_embedding = EmbeddingSharedWeights(
            self.vocabulary_size, self.num_units, self.pad_id)
        # self.projection = core_layer.Dense(self.tgt_vocabulary_size, name='fc')
        if self.embedding_size != self.num_units:
            self.src_dense = tf.keras.layers.Dense(
                self.num_units, name='src_embedding_dense')
            self.tgt_dense = tf.keras.layers.Dense(
                self.embedding_size, name='tgt_embedding_dense')

    # @tf_utils.shape_type_conversion
    # def build(self, input_shape):
    #     # input_dim = input_shape[-1]
    def Encoder(self, inputs, K_V=None, padding_matrix=None, length=None):
        with tf.name_scope("encoder"):
            if length is None:
                length = tf.shape(inputs)[1]
            if K_V is None:
                assert ('Using maksed_attention, please give K_V')
            # src_input = tf.multiply(tf.cast(inputs, tf.float32), self.num_units**0.5)
            src_input = inputs * self.num_units**0.5
            if padding_matrix is not None:
                padding_mask_bias = self.padding_bias(padding_matrix)
            else:
                padding_mask_bias = 0
            positional_input = self.possition_encoding(length)

            inputs = src_input + positional_input

            outputs = inputs
            for i in range(self.num_encoder_layers):
                with tf.name_scope('layer_%d' % i):
                    multi_att = self.en_att[i](outputs, (outputs, outputs),
                                               padding_mask_bias)
                    outputs = self.en_ffn[i](multi_att)

            return outputs

    def Decoder(self,
                inputs,
                enc=None,
                self_mask_bias=None,
                padding_matrix=None,
                length=None,
                cache=None):
        with tf.name_scope('decoder'):
            if length is None:
                length = tf.shape(inputs)[1]
            if enc is None:
                assert ('Using maksed_attention, please give enc')
            # src_input = tf.multiply(tf.cast(inputs, tf.float32), self.num_units**0.5)
            src_input = inputs * self.num_units**0.5
            if self_mask_bias is None:
                self_mask_bias = self.masked_self_attention_bias(length)
            if padding_matrix is not None:
                padding_mask_bias = self.padding_bias(padding_matrix)
            else:
                padding_mask_bias = 0
            positional_input = self.possition_encoding(length)

            inputs = src_input + positional_input

            outputs = inputs
            K_V = inputs
            for i in range(self.num_encoder_layers):
                if cache is not None:
                    # Combine cached keys and values with new keys and values.
                    K_V = tf.concat((cache[str(i)], outputs), axis=1)
                    # Update cache
                    cache[str(i)] = K_V
                with tf.name_scope('layer_%d' % i):
                    outputs = self.de_mask_att[i](outputs, (K_V, K_V),
                                                  self_mask_bias)
                    multi_att = self.de_att[i](outputs, (enc, enc),
                                               padding_mask_bias)
                    outputs = self.de_ffn[i](multi_att)

            return outputs

    def call(self, inputs, train=True):
        if train:
            return self.train_model(inputs)
        else:
            return self.inference_model(inputs)

    def train_model(self, inputs):
        src_input, tgt_input, _, _ = inputs
        src_padding = tf.to_float(tf.equal(src_input, self.pad_id))
        embedding_src_input = self.shared_embedding(src_input)
        embedding_tgt_input = self.shared_embedding(tgt_input)
        if self.embedding_size != self.num_units:
            embedding_src_input = self.src_dense(embedding_src_input)
            embedding_tgt_input = self.tgt_dense(embedding_tgt_input)

        enc = self.Encoder(embedding_src_input, padding_matrix=src_padding)
        dec = self.Decoder(
            embedding_tgt_input, enc, padding_matrix=src_padding)
        # projection = self.projection(self.outputs)
        # logits = tf.keras.layers.Softmax()(projection)
        logits = self.shared_embedding.linear(dec)
        return logits

    def inference_model(self, inputs):
        src_input, tgt_input, _, _ = inputs
        src_padding = tf.to_float(tf.equal(src_input, self.pad_id))
        embedding_src_input = self.shared_embedding(src_input)
        if self.embedding_size != self.num_units:
            embedding_src_input = self.src_dense(embedding_src_input)
        enc = self.Encoder(embedding_src_input, padding_matrix=src_padding)
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        cache = dict()
        # cache = {
        #     "layer_%d" % layer: {
        #         "K": tf.zeros([self.batch_size, 0, self.num_units]),
        #         "V": tf.zeros([self.batch_size, 0, self.num_units]),
        #     }
        #     for layer in range(self.num_decoder_layers)
        # }
        cache['enc'] = enc
        cache['src_padding'] = src_padding
        for i in range(self.num_decoder_layers):
            cache[str(i)] = tf.zeros([self.batch_size, 0, self.num_units])
        # cache['K'] = tf.zeros([self.batch_size, 0, self.num_units])
        # cache['V'] = tf.zeros([self.batch_size, 0, self.num_units])
        logits_body = self.symbols_to_logits_fn(self.max_seq_len)
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=logits_body,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary_size,
            beam_size=4,
            alpha=0.6,
            max_decode_length=self.max_seq_len,
            eos_id=self.eos_id)
        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        # top_scores = scores[:, 0]
        return top_decoded_ids

    def masked_self_attention_bias(self, length):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = self.negtive_infinit * (1.0 - valid_locs)
        return decoder_bias

    def padding_bias(self, padding):
        attention_bias = padding * self.negtive_infinit
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias

    def symbols_to_logits_fn(self, max_seq_len):
        inference_possition = self.possition_encoding(max_seq_len)
        masked_attention_bias = self.masked_self_attention_bias(max_seq_len)

        def body(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = self.shared_embedding(decoder_input)
            self_mask_bias = masked_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_input += inference_possition[i:i + 1]
            if self.embedding_size != self.num_units:
                decoder_input = self.src_dense(decoder_input)
            # Preprocess decoder input by getting embeddings and adding timing signal.
            outputs = self.Decoder(
                decoder_input,
                cache['enc'],
                padding_matrix=cache['src_padding'],
                self_mask_bias=self_mask_bias,
                cache=cache)
            # projection = self.projection(outputs)
            logits = self.shared_embedding.linear(outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return body
