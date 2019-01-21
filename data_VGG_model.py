import tensorflow as tf
import numpy as np
import data_image_helper

import tensorflow.contrib as tf_contrib

#tf_contrib.eager.enable_eager_execution()


img = tf.zeros(dtype = tf.float32, shape = (44, 10, 109, 109, 5))

class CNNcoder(tf.keras.Model):
    def __init__(self, 
                 time_step,
                 embed_size = 512):
        super(CNNcoder, self).__init__()
        self.time_step = time_step
        self.embed_size = embed_size

        self.cnn1 = tf.keras.layers.Conv2D(filters = 96, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.cnn2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.cnn3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.cnn4 = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.cnn5 = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.cnn6 = tf.keras.layers.Conv2D(filters = self.embed_size, kernel_size = 6)

        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))

        self.fc6 = tf.keras.layers.Dense(units = self.embed_size)

    def build_Graph(self, img):
        c_o1 = self.cnn1(img)
        p_o1 = self.pool1(c_o1)
        c_o2 = self.cnn2(p_o1)
        p_o2 = self.pool2(c_o2)
        c_o3 = self.cnn3(p_o2)
        c_o4 = self.cnn4(c_o3)
        c_o5 = self.cnn5(c_o4)
        p_o5 = self.pool5(c_o5)
        c_o6 = self.cnn6(p_o5)
        f_o = self.fc6(c_o6)
        
        return f_o

    def call(self, inputs):
        imgs = inputs
        embeded =[]

        for i in range(self.time_step):
            embeded.append(self.build_Graph(imgs[:, i, :, :, :]))
            
        embeded = tf.transpose(embeded, perm = [1, 0, 2, 3, 4])
        embeded = tf.reshape(embeded, shape = [-1, self.time_step, self.embed_size])
        return embeded

class LSTMcoder(tf.keras.Model):
    def __init__(self, 
                 batch_sz, 
                 num_units, 
                 time_step,
                 backforward = False):
        super(LSTMcoder, self).__init__()
        self.batch_sz = batch_sz
        self.num_units = num_units
        self.time_step = time_step
        self.bf = backforward
        self.lstm1 = tf.keras.layers.LSTM(units = self.num_units, 
                                          return_sequences = True, 
                                          return_state = True, 
                                          recurrent_initializer = 'glorot_uniform')
        self.lstm2 = tf.keras.layers.LSTM(units = self.num_units, 
                                          return_sequences = True, 
                                          return_state = True, 
                                          recurrent_initializer = 'glorot_uniform')
        self.lstm3 = tf.keras.layers.LSTM(units = self.num_units, 
                                          return_sequences = True, 
                                          return_state = True, 
                                          recurrent_initializer = 'glorot_uniform')
        
    def initial_hidden_state(self):
        return tf.zeros((self.batch_sz, self.num_units)), tf.zeros((self.batch_sz, self.num_units))
        
    def build_Graph(self, x, seq_lengths):
        if(not self.bf):
            x = tf.reverse_sequence(x, seq_lengths, seq_axis = 1, batch_axis = 0)


        out1, h1, state1 = self.lstm1(x,  initial_state = self.initial_hidden_state())
        out2, h2, state2 = self.lstm2(out1, initial_state = self.initial_hidden_state())
        out3, h3, state3 = self.lstm3(out2, initial_state = self.initial_hidden_state())

        if(self.bf):
            out1 = tf.reverse_sequence(out1, seq_lengths, seq_axis = 1, batch_axis = 0)
            out2 = tf.reverse_sequence(out2, seq_lengths, seq_axis = 1, batch_axis = 0)
            out3 = tf.reverse_sequence(out3, seq_lengths, seq_axis = 1, batch_axis = 0)

        return out1, out2, out3, h1, h2, h3, state1, state2, state3
    
    def call(self, inputs):
        (X, seq_lengths) = inputs

        # O1 = []
        # O2 = []
        # O3 = []
        # S1 = None
        # S2 = None
        # S3 = None
#         for i in range(self.time_step):
#             out1, out2, out3, state1, state2, state3 = self.build_Graph(X)
#             O1.append(out1)
#             O2.append(out2)
#             O3.append(out3)
#             S1 = state1
#             S2 = state2
#             S3 = state3
                
#         O1 = tf.transpose(O1, perm = [1, 0, 2])
#         O2 = tf.transpose(O2, perm = [1, 0, 2])
#         O3 = tf.transpose(O3, perm = [1, 0, 2])

        O1, O2, O3, H1, H2, H3, S1, S2, S3 = self.build_Graph(X, seq_lengths)
        
        return O1, O2, O3, H1, H2, H3, S1, S2, S3

class VGGTower(tf.keras.Model):
    def __init__(self, 
                 batch_size = 64,
                 time_step = 10,
                 embed_size = 512,
                 num_units = 512,
                 backforward = False):
        super(VGGTower, self).__init__()
        self.batch_size = batch_size
        self.time_step = time_step
        self.embed_size = embed_size
        self.num_units = num_units
        self.bf = backforward

        self.embeded = None
        self.final_out = None
        self.final_hidden = None
        self.final_state = None

        self.encoder = CNNcoder(time_step = self.time_step,
                                embed_size = self.embed_size)
        self.fw_decoder = LSTMcoder(batch_sz = self.batch_size,
                                    num_units = self.num_units,
                                    time_step = self.time_step,
                                    backforward = False)

        if(self.bf):
            self.bw_decoder = LSTMcoder(batch_sz = self.batch_size,
                                        num_units = self.num_units,
                                        time_step = self.time_step,
                                        backforward = True)
            self.output_size = self.num_units * 2

        else:
            self.output_size = self.num_units

        
    
    def call(self, inputs):
        """
        return:
            final_out: [batch_size, time_step, embed_size]
            final_hidden: [batch_size, embed_size]
        """
        (images) = inputs
        self.embeded = self.encoder(images)
        fw_o1, fw_o2, fw_o3, fw_h1, fw_h2, fw_h3, fw_s1, fw_s2, fw_s3 = self.fw_decoder((self.embeded, 
                                                                                        [self.time_step] * self.batch_size))

        if(self.bf):
            bw_o1, bw_o2, bw_o3, bw_h1, bw_h2, bw_h3, bw_s1, bw_s2, bw_s3 = self.bw_decoder((self.embeded, 
                                                                                            [self.time_step] * self.batch_size))
            self.final_out = [tf.concat((fw_o1, bw_o1), -1),
                              tf.concat((fw_o2, bw_o2), -1),
                              tf.concat((fw_o3, bw_o3), -1)]

            self.final_hidden = [tf.concat((fw_h1, bw_h1), -1),
                                 tf.concat((fw_h2, bw_h2), -1),
                                 tf.concat((fw_h3, bw_h3), -1)] 

            self.final_state = [tf.concat((fw_s1, bw_s1), -1),
                                tf.concat((fw_s2, bw_s2), -1),
                                tf.concat((fw_s3, bw_s3), -1)]
        else:
            self.final_out = [fw_o1, fw_o2, fw_o3]
            self.final_hidden = [fw_h1, fw_h2, fw_h3]
            self.final_state = [fw_s1, fw_s2, fw_s3]

        return self.final_out, self.final_hidden

    def get_state(self):
        return self.final_state

    def get_embeded(self):
        return self.embeded

    def get_output_size(self):
        return self.output_size


class VGGTowerFactory(tf.keras.Model):
    def __init__(self,
                 detector_path,
                 data_path,
                 embed_size = 512,
                 batch_size = 64,
                 num_units = 512, 
                 time_step = 10,
                 backforward = False):
        super(VGGTowerFactory, self).__init__()
        self.detector_path = detector_path
        self.data_path = data_path
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.time_step = time_step
        self.bf = backforward

        print("build VGG model")

    def get_data(self, detector_path, data_path, batch_size, time_step):
        reader = data_image_helper.data_image_helper(detector_path)
        dataset, images = reader.prepare_data(path = data_path, batch_size = batch_size, time_step = time_step)
        return dataset, images

    def mini_model(self, batch_size = 4):
        """short summary: 
            A mini model is used to train final model.
        Args:
            batch_size: the size take from the dataset.
        Returns:
            model, dataset, images
        """
        dataset, images = self.get_data(detector_path = self.detector_path,
                                        data_path = self.data_path,
                                        batch_size = self.batch_size,
                                        time_step = self.time_step) 
        return VGGTower(batch_size = batch_size,
                        time_step = self.time_step,
                        embed_size = 4,
                        num_units = 4,
                        backforward = self.bf), dataset, images

    def small_model(self, batch_size = 16, embed_size = 16, num_units = 16):
        """short summary: 
            A small_modle model is used to train final model.
        Args:
            batch_size: the size take from the dataset.
            embed_size: the size picture embed to.
            num_units: the units number of the lstm cell.
        Returns:
            model, dataset, images
        """
        dataset, images = self.get_data(detector_path = self.detector_path,
                                        data_path = self.data_path,
                                        batch_size = self.batch_size,
                                        time_step = self.time_step) 
        return VGGTower(batch_size = batch_size,
                        time_step = self.time_step,
                        embed_size = embed_size,
                        num_units = num_units,
                        backforward = self.bf), dataset, images
    
    def full_model(self):
        """short summary: 
            A full model is used to train final model.
        Args:
            Don't need.
        Returns:
            model, dataset, images
        """
        dataset, images = self.get_data(detector_path = self.detector_path,
                                        data_path = self.data_path,
                                        batch_size = self.batch_size,
                                        time_step = self.time_step) 
        return VGGTower(batch_size = self.batch_size,
                        time_step = self.time_step,
                        embed_size = self.embed_size,
                        num_units = self.num_units,
                        backforward = self.bf), dataset, images

v = VGGTower(batch_size = 44, time_step = 10, num_units = 512, backforward = True)

o, h= v(img)
s = v.get_state()
print(o[2].shape)
print(h[2].shape)
print(s[2].shape)