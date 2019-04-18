# encoder=utf8
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


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
            initializer=tf.random_normal_initializer(0.,
                                                     self.hidden_size**-0.5))
        # self.build = True

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
        mode:
            add: ln(x) + x
            norm: ln(x)
    """

    def __init__(self,
                 epsilon=1e-6,
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

    def call(self, inputs, training=False):
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

    def call(self, inputs, training=False):
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
                 name=None,
                 dropout=1,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super(Feed_Forward_Network, self).__init__(name=name)
        self.num_units = num_units
        self.bias = bias
        self.dropout = min(1., max(0., dropout))

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

    def call(self, inputs, padding=None, training=None):
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        if padding is not None:
            pad_mask = tf.reshape(padding, [-1])

            nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

            # Reshape x to [batch_size*length, hidden_size] to remove padding
            inputs = tf.reshape(inputs, [-1, self.num_units])
            inputs = tf.gather_nd(inputs, indices=nonpad_ids)

            # Reshape x from 2 dimensions to 3 dimensions.
            inputs.set_shape([None, self.num_units])
            inputs = tf.expand_dims(inputs, axis=0)
        linear = tf.keras.backend.dot(inputs, self.linear_kernel)
        linear = tf.keras.layers.Activation("relu")(linear)
        if (self.bias):
            linear = linear + self.linear_bias

        if training is not False:
            dropout_mask_inputs = tf.keras.backend.dropout(
                tf.ones_like(inputs), self.dropout)
            inputs = inputs * dropout_mask_inputs
        FFN = tf.keras.backend.dot(linear, self.FFN_kernel)
        if (self.bias):
            FFN = FFN + self.FFN_bias
        if training is not False:
            dropout_mask_inputs = tf.keras.backend.dropout(
                tf.ones_like(inputs), self.dropout)
            FFN = FFN * dropout_mask_inputs
        output = FFN
        if padding is not None:
            output = tf.squeeze(output, axis=0)
            output = tf.scatter_nd(
                indices=nonpad_ids,
                updates=output,
                shape=[batch_size * length, self.num_units])
            output = tf.reshape(output, [batch_size, length, self.num_units])

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

    def call(self, inputs, K_V=None, bias=0, cache=None, training=False):
        Q = inputs
        if K_V is None:
            K_V = (Q, Q)
        K, V = K_V
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            K = tf.concat([cache["K"], K], axis=1)
            V = tf.concat([cache["V"], V], axis=1)
            # Update cache
            cache["K"] = K
            cache["V"] = V
        # Q = self.ln(Q)
        # K = self.ln(K)
        # V = self.ln(V)
        with tf.name_scope('Q_K_V_linear_projection'):
            Q = tf.keras.backend.dot(Q, self.query_kernel)
            K = tf.keras.backend.dot(K, self.key_kernel)
            V = tf.keras.backend.dot(V, self.value_kernel)
        attention_output = self.scale_dot_product_attention(
            Q, K, V, bias, training=training)
        with tf.name_scope('attention_linear_projection'):
            liner = tf.keras.backend.dot(attention_output, self.liner_kernel)

        if training is not False:
            dropout_mask = tf.keras.backend.dropout(
                tf.ones_like(liner), self.dropout)
            liner = liner * dropout_mask
        # ResNet
        output = liner
        return output

    def scale_dot_product_attention(self, Q, K, V, bias, training=False):
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
            self.softmax_attention = softmax
            # if training is not False:
            #     dropout_mask = tf.keras.backend.dropout(
            #         tf.ones_like(softmax), self.dropout)
            #     softmax = softmax * dropout_mask

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
        return self.softmax_attention
