import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib

import core_seq2seq_model as seq2seq
import core_VGG_model as VGG

class Daedalus(tf.keras.Model):
    def __init__(self, 
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 batch_size = 64,
                 embed_size = 512,
                 num_units = 512,
                 backforward = False,
                 eager = False):
        super(Daedalus, self).__init__()
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.num_units = num_units
        self.bf = backforward
        self.eager = eager

        self.layer_initializer()

    def layer_initializer(self):
        self.VGG = VGG.VGGTower(batch_size = self.batch_size,
                                embed_size = self.embed_size,
                                num_units = self.num_units,
                                backforward = self.bf,
                                eager = self.eager)
        
        if self.bf:
            self.final_units = 2 * self.num_units

        else:
            self.final_units = self.num_units    
        self.final_embed_size = self.final_units 

        self.decoder1 = seq2seq.RNNcoder(self.tgt_vocabulary_size,
                            self.batch_size,
                            self.final_units,
                            self.final_embed_size,
                            backforward = False,
                            embedding_matrix = True,
                            eager=True) 
        self.decoder2 = seq2seq.RNNcoder(self.tgt_vocabulary_size,
                            self.batch_size,
                            self.final_units,
                            self.final_embed_size,
                            backforward = False,
                            embedding_matrix = False,
                            eager=True) 
        self.decoder3 = seq2seq.RNNcoder(self.tgt_vocabulary_size,
                            self.batch_size,
                            self.final_units,
                            self.final_embed_size,
                            backforward = False,
                            embedding_matrix = False,
                            eager=True) 
        self.project = tf.keras.layers.Dense(self.tgt_vocabulary_size)
        self.logit = tf.keras.layers.Softmax(axis = -1)

    def call(self, inputs):
        (src_input, tgt_input, src_length, tgt_length) = inputs

        VGG_input = (src_input, src_length)
        self.VGG_out, self.VGG_hidden = self.VGG(VGG_input)

        decoder1_input = (tgt_input, tgt_length, self.VGG_hidden[0], 0)
        self.decoder1_hidden, self.decoder1_final = self.decoder1(decoder1_input)

        decoder2_input = (self.decoder1_hidden, tgt_length, self.VGG_hidden[1], 0)
        self.decoder2_hidden, self.decoder2_final = self.decoder2(decoder2_input)   

        decoder3_input = (self.decoder2_hidden, tgt_length, self.VGG_hidden[2], self.VGG_out)
        self.decoder3_hidden, self.decoder3_final = self.decoder3(decoder3_input)    

        projection = self.project(self.decoder3_hidden)
        self.logits = self.logit(projection)
        return self.logits

    def get_VGG(self):
            return self.VGG, self.VGG_out, self.VGG_hidden
    
    def get_states(self):
            return self.decoder3_hidden, self.decoder3_final

class DaedalusFactory(tf.keras.Model):
    def __init__(self,
                 src_vocabulary_size,
                 tgt_vocabulary_size,
                 batch_size = 64,
                 embed_size = 512,
                 num_units = 512,
                 backforward = False,
                 eager = False):
        super(DaedalusFactory, self).__init__()
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.num_units = num_units
        self.bf = backforward
        self.eager = eager

    def mini_model(self):
        pass

    def small_model(self):
        pass

    def full_model(self):
        pass