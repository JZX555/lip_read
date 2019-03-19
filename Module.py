import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
import numpy as np

class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
    """
    def __init__(self,
                 epsilon = 1e-9,
                 gamma_initializer = "ones",
                 beta_initializer = "zeros"):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gamma_kernel = self.add_weight(
            shape = (input_dim),
            name = "gamma",
            initializer = self.gamma_initializer)
        self.beta_kernel = self.add_weight(
            shape = (input_dim),
            name = "beta",
            initializer = self.beta_initializer)
    
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (variance ** 2 + self.epsilon)
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
        self.padding_num = - 2 ** 32 + 1

    def call(self, inputs, type):
        scaled, Q, K = inputs
        self.type = type
        if self.type in ("k", "key", "keys"):
            mask = tf.sign(tf.reduce_sum(tf.abs(K), axis = -1))
            mask = tf.expand_dims(mask, axis = 1)
            mask = tf.tile(mask, [1, tf.shape(Q)[1], 1])

            paddings = tf.ones_like(scaled) * self.padding_num
            output = tf.where(tf.equal(mask, 0), paddings, scaled)

        elif self.type in ("q", "query", "queries"):
            mask = tf.sign(tf.reduce_sum(tf.abs(Q), axis = -1))
            mask = tf.expand_dims(mask, axis = -1)
            mask = tf.tile(mask, [1, 1, tf.shape(K)[1]])

            output = scaled * mask
        
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(scaled[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])

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
    def __init__(self,
                 causality = False,
                 dropout = 1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.causality = causality
        self.dropout = min(1., max(0., dropout))
        self.Mask = Mask()

    def call(self, inputs):
        Q, K, V = inputs
        d_k = Q.get_shape().as_list()[-1]

        mat = tf.keras.backend.dot(Q, tf.transpose(K, [0, 2, 1]))
        scale = mat / (d_k ** 0.5)

        # the two mask opt maybe put together:
        # 1. mask(key)->mask(query)->softmax
        # 2. mask(key)->softmax->mask(query)
        # we use the second choice;
        # tips: key must use before query
        mask_input = (scale, Q, K)
        mask = self.Mask(mask_input, "key")

        if(self.causality):
            mask = self.Mask((mask, None, None), "future")
        
        softmax = tf.keras.layers.Softmax()(mask)
        self.attention = softmax

        mask_input = (softmax, Q, K)
        softmax = self.Mask(mask_input, "query")

        if 0 < self.dropout < 1:
            dropout_mask = tf.keras.backend.dropout(tf.ones_like(softmax), self.dropout)
            softmax = softmax * dropout_mask

        output = tf.keras.backend.dot(softmax, V)

        return output

    def get_attention(self):
        return self.attention

class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads = 8,
                 num_units = 512,
                 causality = False,
                 dropout = 1,
                 ln = None,
                 query_initializer = 'glorot_uniform',
                 key_initializer = 'glorot_uniform',
                 value_initializer = 'glorot_uniform',
                 liner_initializer = 'glorot_uniform'):
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

        self.attention = Scaled_Dot_Product_Attention(self.causality, self.dropout)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.query_kernel = self.add_weight(
            shape = (input_dim, self.num_units),
            name = "query_kernel",
            initializer = self.query_initializer)

        self.key_kernel = self.add_weight(
            shape = (input_dim, self.num_units),
            name = "key_kernel",
            initializer = self.key_initializer)

        self.value_kernel = self.add_weight(
            shape = (input_dim, self.num_units),
            name = "value_kernel",
            initializer = self.value_initializer)

        self.liner_kernel = self.add_weight(
            shape = (self.num_units, input_dim),
            name = "liner_kernel",
            initializer = self.liner_initializer)

    def call(self, inputs):
        Q, K, V = inputs
        Q = tf.keras.backend.dot(Q, self.query_kernel)
        K = tf.keras.backend.dot(K, self.key_kernel)
        V = tf.keras.backend.dot(V, self.value_kernel)

        split_Q = tf.concat(tf.split(Q, self.num_heads, axis = -1), axis = 0)
        split_K = tf.concat(tf.split(K, self.num_heads, axis = -1), axis = 0)
        split_V = tf.concat(tf.split(V, self.num_heads, axis = -1), axis = 0)

        attention_input = (split_Q, split_K, split_V)

        context = self.attention(attention_input)
        context = tf.concat(tf.split(context, self.num_heads, axis = 0), axis = -1)

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
                 num_units = 2048,
                 bias = True,
                 ln = None,
                 kernel_initializer = "glorot_uniform",
                 bias_initializer = "zeros"):
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
            shape = (input_dim, self.num_units),
            name = "liner_kernel",
            initializer = self.kernel_initializer)
        self.FFN_kernel = self.add_weight(
            shape = (self.num_units, input_dim),
            name = "FFN_kernel",
            initializer = self.kernel_initializer)

        if(self.bias):
            self.liner_bias = self.add_weight(     
                shape = self.num_units,
                name = "liner_bias",
                initializer = self.bias_initializer)
            self.FFN_bias = self.add_weight(     
                shape = input_dim,
                name = "FFN_bias",
                initializer = self.bias_initializer)                

    def call(self, inputs):
        liner = tf.keras.backend.dot(inputs, self.liner_kernel)
        if(self.bias):
            liner = liner + self.liner_bias
        
        FFN = tf.keras.backend.dot(liner, self.FFN_kernel)
        if(self.bias):
            FFN = FFN + self.FFN_bias
        
        output = self.ln(FFN + inputs)

        return output
        