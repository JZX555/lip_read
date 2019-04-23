# encoding=utf8
import tensorflow as tf
import core_Transformer_model
from hyper_and_conf import hyper_layer
# tf.enable_eager_execution()
ENGLISH_BYTE_VOCAB = 12000


class Daedalus(tf.keras.Model):
    def __init__(self, hp):
        super(Daedalus, self).__init__(name='lip_reading')
        self.hp = hp
        # self.shared_embedding = hyper_layer.EmbeddingSharedWeights(
        #     hp.vocabulary_size, hp.embedding_size, hp.PAD_ID)
        # if self.embedding_size != self.num_units:
        #     self.src_dense = tf.keras.layers.Dense(
        #         self.num_units, name='src_embedding_dense')
        #     self.tgt_dense = tf.keras.layers.Dense(
        #         self.embedding_size, name='tgt_embedding_dense')
        # self.vgg16 = self.get_vgg()
        self.transformer = self.get_transofomer(hp)
        self.word_embedding = hyper_layer.EmbeddingSharedWeights(
            vocab_size=ENGLISH_BYTE_VOCAB,
            hidden_size=hp.num_units,
            pad_id=hp.PAD_ID,
            name='word_embedding')
        self.kernel_initializer = tf.keras.initializers.get("glorot_uniform")
        self.bias_initializer = tf.keras.initializers.get("zeros")
        self.Q = self.add_weight(
            name='Q',
            shape=[25088 + self.hp.num_units, self.hp.num_units],
            initializer=self.kernel_initializer)
        # self.K = self.add_weight(
        #     name='K',
        #     shape=[25088, self.hp.num_units],
        #     initializer=self.kernel_initializer)
        # self.V = self.add_weight(
        #     name='V',
        #     shape=[12000, self.hp.num_units],
        #     initializer=self.kernel_initializer)
        # self.Q = tf.keras.layers.Dense(self.hp.num_units, name='fusion')
        # self.K = tf.keras.layers.Dense(self.hp.num_units, name='K')
        # self.V = tf.keras.layers.Dense(self.hp.num_units, name='V')

    def build(self, input_shape):
        self.build = True

    def get_vgg(self):
        if tf.gfile.Exists('pre_train/vgg16_pre_all'):
            vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
        else:
            vgg16 = tf.keras.applications.vgg16.VGG16(
                include_top=True, weights='imagenet')
        return vgg16

    def get_transofomer(self, hp):
        transformer = core_Transformer_model.Prometheus(
            max_seq_len=hp.max_sequence_length,
            vocabulary_size=hp.vocabulary_size,
            embedding_size=hp.embedding_size,
            batch_size=hp.batch_size,
            num_units=hp.num_units,
            num_heads=hp.num_heads,
            num_encoder_layers=hp.num_encoder_layers,
            num_decoder_layers=hp.num_decoder_layers,
            dropout=hp.dropout,
            eos_id=hp.EOS_ID,
            pad_id=hp.PAD_ID)
        return transformer

    def call(self, inputs, training=False):
        if training is False:
            pass
        else:
            logits = self.train_model(inputs, training=True)
            return logits
        # img_input = tf.keras.layers.Input(
        #     shape=[None, 25088], dtype=tf.float32)
        # tgt_input = tf.keras.layers.Input(
        #     shape=[None], dtype=tf.int64, name='tgt_input')

    def train_model(self, inputs, training=True):
        with tf.name_scope("VGG_features"):
            img_input, tgt_input = inputs
        # batch_size = self.hp.batch_size
        # Q_input_length = img_input.get_shape().as_list()[1]
        with tf.name_scope("fusion_layer"):
            img_input_padding = tf.to_float(
                tf.equal(img_input, self.hp.PAD_ID))[:, :, 0]

            mask_id = self.hp.MASK_ID
            mask_words = tf.zeros_like(
                img_input[:, :, 0], dtype=tf.int32) + mask_id
            # mask_words = tf.constant(mask_id, shape=[batch_size, Q_input_length])
            mask_embedding = self.word_embedding(mask_words)
            # fusion = tf.keras.layers.concatenate((img_input, mask_embedding), axis=0)
            fusion = tf.keras.layers.concatenate([img_input, mask_embedding],
                                                 axis=-1)

            # Q = self.Q(fusion)
            # K = self.K(img_input)
            # V = self.V(mask_embedding)
            Q = tf.keras.backend.dot(fusion, self.Q)
            K = Q
            V = mask_embedding
            # K = V

            # transformer_src = tf.keras.layers.Input(
            #     shape=[None], dtype=tf.int64, name='transformer_src_input')
            # transformer_tgt = tf.keras.layers.Input(
            #     shape=[None], dtype=tf.int64, name='transformer_output_input')
        with tf.name_scope("transformer"):
            encoder_out = self.transformer.Encoder((Q, K, V),
                                                   img_input_padding,
                                                   training=True)
            embedding_tgt_input = self.word_embedding(tgt_input)
            embedding_tgt_input = tf.pad(embedding_tgt_input,
                                         [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            decoder_out = self.transformer.Decoder(
                embedding_tgt_input,
                encoder_out,
                padding_matrix=img_input_padding,
                training=True)
        logits = self.word_embedding.linear(decoder_out)
        # model = tf.keras.Model([img_input, tgt_input], logits)
        return logits


# if '__name__' == '__main__':
# from hyper_and_conf import hyper_param
# hp = hyper_param.HyperParam(mode='test')
# model = main_model(hp)

# vgg16 = get_vgg()  # step = tf.shape(vgg16_input)
# # step = vgg16_input.get_shape().as_list()[1]
# output = []
# vgg16_flatten = vgg16.get_layer('flatten')
# vgg16_output = vgg16_flatten.output
# vgg16.input
# model = tf.keras.Model(vgg16.input, vgg16_output)
# test = tf.constant(0.0, shape=[2, 224, 224, 3])
# model(test)
