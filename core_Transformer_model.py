# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer
# from tensorflow.python.layers import core as core_layer
# import numpy as np
from hyper_and_conf import hyper_beam_search as beam_search


class Prometheus(tf.keras.Model):
    """
        Transformer
    """

    def __init__(self,
                 max_seq_len,
                 vocabulary_size,
                 embedding_size=512,
                 batch_size=64,
                 num_units=512,
                 num_heads=6,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.4,
                 eos_id=1,
                 pad_id=0):
        super(Prometheus, self).__init__(name='transformer')
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        # self.vocabulary_size = 32000
        # self.src_vocabulary_size = src_vocabulary_size
        # self.tgt_vocabulary_size = tgt_vocabulary_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.possition_encoding = hyper_layer.Positional_Encoding(
            self.num_units)
        self.en_att = []
        self.en_ffn = []
        self.de_att = []
        self.de_ffn = []
        self.de_mask_att = []
        self.negtive_infinit = -1e32
        self.norm = hyper_layer.LayerNorm()
        for i in range(self.num_encoder_layers):
            self.en_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="enc_multi_att_%d" % i))
            self.en_ffn.append(
                hyper_layer.Feed_Forward_Network(
                    num_units=4 * self.num_units,
                    dropout=self.dropout,
                    name="enc_ffn_%d" % i))
        for i in range(self.num_decoder_layers):
            self.de_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="de_multi_att_%d" % i))
            self.de_ffn.append(
                hyper_layer.Feed_Forward_Network(
                    num_units=4 * self.num_units,
                    dropout=self.dropout,
                    name="dec_ffn_%d" % i))
            self.de_mask_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=True,
                    name="masked_multi_att_%d" % i))
        self.shared_embedding = hyper_layer.EmbeddingSharedWeights(
            self.vocabulary_size, self.embedding_size, self.pad_id)
        if self.embedding_size != self.num_units:
            self.src_dense = tf.keras.layers.Dense(
                self.num_units, name='src_embedding_dense')
            self.tgt_dense = tf.keras.layers.Dense(
                self.embedding_size, name='tgt_embedding_dense')

    def build(self, input_shape):
        self.build = True

    def Encoder(self, inputs, padding_matrix=None, length=None,
                training=False):
        with tf.name_scope("encoder"):
            if training is not False:
                dropout_mask_inputs = tf.keras.backend.dropout(
                    tf.ones_like(inputs), self.dropout)
                inputs = inputs * dropout_mask_inputs

            if length is None:
                length = tf.shape(inputs)[1]
            # src_input = tf.multiply(tf.cast(inputs, tf.float32), self.num_units**0.5)
            src_input = inputs * self.num_units**0.5
            if padding_matrix is not None:
                padding_mask_bias = self.padding_bias(padding_matrix)
            else:
                padding_mask_bias = 0
            positional_input = self.possition_encoding(length)

            inputs = src_input + positional_input

            outputs = self.norm(inputs)
            i = 0
            # for i in range(self.num_encoder_layers):
            with tf.name_scope('layer_%d' % i):
                multi_att = self.en_att[i](
                    outputs, (outputs, outputs),
                    padding_mask_bias,
                    training=training)
                multi_att = self.norm(multi_att + outputs)
                outputs = self.en_ffn[i](multi_att, training=training)
                outputs = self.norm(outputs + multi_att)

            return outputs

    def Decoder(self,
                inputs,
                enc=None,
                self_mask_bias=None,
                padding_matrix=None,
                length=None,
                cache=None,
                training=False):
        with tf.name_scope('decoder'):
            if enc is None:
                assert ('Using maksed_attention, please give enc')
            if training is not False:
                dropout_mask_inputs = tf.keras.backend.dropout(
                    tf.ones_like(inputs), self.dropout)
                inputs = inputs * dropout_mask_inputs
                dropout_mask_enc = tf.keras.backend.dropout(
                    tf.ones_like(enc), self.dropout)
                enc = enc * dropout_mask_enc
            if length is None:
                length = tf.shape(inputs)[1]
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
            outputs = self.norm(inputs)
            K_V = self.norm(inputs)
            i = 0
            # for i in range(self.num_decoder_layers):
            # if cache is not None:
            #     # Combine cached keys and values with new keys and values.
            #     K_V = tf.concat((cache[str(i)], outputs), axis=1)
            #     # Update cache
            #     cache[str(i)] = K_V
            with tf.name_scope('layer_%d' % i):
                de_outputs = self.de_mask_att[i](
                    outputs, (K_V, K_V),
                    self_mask_bias,
                    cache=cache,
                    training=training)
                de_outputs = self.norm(outputs + de_outputs)
                multi_att = self.de_att[i](
                    de_outputs, (enc, enc),
                    padding_mask_bias,
                    training=training)
                multi_att = self.norm(multi_att + de_outputs)
                outputs = self.de_ffn[i](multi_att, training=training)
                outputs = self.norm(outputs + multi_att)
            return outputs

    def call(self, inputs, training=False):
        # src, tgt = tf.split(inputs, 2, 0)
        # inputs = (src, tgt)
        if training:
            return self.train_model(inputs)
        else:
            return self.inference_model(inputs, training=False)

    def train_model(self, inputs, training=True):
        # src_input, tgt = tf.split(inputs, 2)
        src_input, tgt = inputs
        src_input = tf.cast(src_input, tf.int64)
        tgt = tf.cast(tgt, tf.int64)
        src_padding = tf.to_float(tf.equal(src_input, self.pad_id))
        embedding_src_input = self.shared_embedding(src_input)
        embedding_tgt_input = self.shared_embedding(tgt)
        embedding_tgt_input = tf.pad(embedding_tgt_input,
                                     [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        if self.embedding_size != self.num_units:
            embedding_src_input = self.src_dense(embedding_src_input)
            embedding_tgt_input = self.tgt_dense(embedding_tgt_input)

        enc = self.Encoder(
            embedding_src_input, padding_matrix=src_padding, training=training)
        dec = self.Decoder(
            embedding_tgt_input,
            enc,
            padding_matrix=src_padding,
            training=training)
        # projection = self.projection(self.outputs)
        # logits = tf.keras.layers.Softmax()(projection)
        logits = self.shared_embedding.linear(dec)
        return logits

    def inference_model(self, inputs, training):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            src_input, _ = inputs
        else:
            src_input = inputs

        initial_size = tf.shape(inputs)[0]
        src_padding = tf.to_float(tf.equal(src_input, self.pad_id))
        src_input = tf.cast(src_input, tf.int32)
        embedding_src_input = self.shared_embedding(src_input)
        if self.embedding_size != self.num_units:
            embedding_src_input = self.src_dense(embedding_src_input)
        enc = self.Encoder(
            embedding_src_input, padding_matrix=src_padding, training=False)
        # initial_ids = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.int32)
        initial_ids = tf.zeros([initial_size], dtype=tf.int32)

        cache = dict()
        cache['enc'] = enc
        cache['src_padding'] = src_padding
        for i in range(self.num_decoder_layers):
            cache[str(i)] = tf.zeros([initial_size, 0, self.num_units])
            # cache[str(i)] = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.float32)
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
        # self.attention = cache['attention']
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
                cache=cache,
                train=False)
            # projection = self.projection(outputs)
            logits = self.shared_embedding.linear(outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return body

    def get_attention(self):
        self.attention = self.de_att[self.num_decoder_layers -
                                     1].get_attention()
        return self.attention
