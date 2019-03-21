import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
import Module
import core_VGG_model as VGG
import numpy as np

class Prometheus(tf.keras.Model):
    def __init__(self, 
                 tgt_vocabulary_size,
                 batch_size = 64,
                 embedding_size = 512,
                 num_units = 512,
                 num_heads = 8,
                 dropout = 1,
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 sos_id=1,
                 eos_id=2,
                 pad_id=2):
        super(Prometheus, self).__init__()
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.negtive_infinit = -1e32

        self.initial_layer()

    def initial_layer(self):
        self.en_att = []
        self.en_fnn = []
        self.de_att = []
        self.de_fnn = []
        self.de_mask_att = []

        self.VGG = VGG.CNNcoder(batch_size = self.batch_size,
                                embed_size = self.embed_size,
                                eager = True)

        for i in range(self.num_encoder_layers):
            self.en_att.append(
                Module.Multi_Head_Attention(
                    num_heads = self.num_heads,
                    num_units = self.num_units,
                    dropout = self.dropout,
                    masked_attention = False,
                    name = "enc_multi_att_%d" % i))
            self.en_ffn.append(
                Module.Feed_Forward_Network(
                    num_units = 4 * self.num_units, name = "enc_ffn_%d" % i))
        for i in range(self.num_decoder_layers):
            self.de_att.append(
                Module.Multi_Head_Attention(
                    num_heads = self.num_heads,
                    num_units = self.num_units,
                    dropout = self.dropout,
                    masked_attention = False,
                    name = "de_multi_att_%d" % i))
            self.de_ffn.append(
                Module.Feed_Forward_Network(
                    num_units = 4 * self.num_units, name = "dec_ffn_%d" % i))
            self.de_mask_att.append(
                Module.Multi_Head_Attention(
                    num_heads = self.num_heads,
                    num_units = self.num_units,
                    dropout = self.dropout,
                    masked_attention = True,
                    name = "masked_multi_att_%d" % i))

        self.shared_embedding = Module.EmbeddingSharedWeights(
            self.tgt_vocabulary_size, self.num_units, self.pad_id)

        if self.embedding_size != self.num_units:
            self.tgt_dense = tf.keras.layers.Dense(
                self.embedding_size, name='tgt_embedding_dense')

    def Encoder(self, 
                inputs, 
                padding_matrix = None, 
                length = None):
        with tf.name_scope("encoder"):
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

            outputs = inputs
            for i in range(self.num_encoder_layers):
                with tf.name_scope('layer_%d' % i):
                    multi_att = self.en_att[i](outputs, (outputs, outputs),
                                               padding_mask_bias)
                    outputs = self.en_ffn[i](multi_att)

            return outputs

    def Decoder(self,
                inputs,
                enc = None,
                self_mask_bias = None,
                padding_matrix = None,
                length = None,
                cache = None):
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
    
    def call(self, inputs, train):
        if train:
            return self.train_model(inputs)
        else:
            pass
    
    def train_model(self, inputs):
        src_input, tgt_input, _, _ = inputs
        src_padding = tf.equal(tf.sign(tf.reduce_sum(tf.abs(src_input), axis = -1)), 0)

        embedding_tgt_input = self.shared_embedding(tgt_input)
        if self.embedding_size != self.num_units:
            embedding_tgt_input = self.tgt_dense(embedding_tgt_input)

        enc = self.Encoder(src_input, padding_matrix = src_padding)
        dec = self.Decoder(
            embedding_tgt_input, enc, padding_matrix = src_padding)
        # projection = self.projection(self.outputs)
        # logits = tf.keras.layers.Softmax()(projection)
        logits = self.shared_embedding.linear(dec)
        return logits
    
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