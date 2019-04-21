# encoding=utf8
import tensorflow as tf
import core_Transformer_model
from hyper_and_conf import hyper_layer
# tf.enable_eager_execution()
ENGLISH_BYTE_VOCAB = 12000


def get_vgg():
    if tf.gfile.Exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet')
    return vgg16


def get_transofomer(hp):
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


def main_model(hp):
    # vgg16_input = tf.keras.layers.Input(shape=[None, 224, 224, 3])
    # vgg16 = get_vgg()
    #
    # vgg16_flatten = vgg16.get_layer('fc1')
    # vgg16_output = vgg16_flatten.output
    # vgg16_padding = tf.to_float(tf.equal(vgg16_output, hp.PAD_ID))
    #
    # input_length = tf.shape(vgg16_output)[1]
    # batch_siz = tf.shape(vgg16_output)[0]

    img_input = tf.keras.layers.Input(shape=[None, 25088], dtype=tf.float32)
    tgt_input = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='tgt_input')
    batch_size = img_input.get_shape().as_list()[0]
    Q_input_length = img_input.get_shape().as_list()[1]
    img_input_padding = tf.squeeze(
        tf.to_float(tf.equal(img_input, hp.PAD_ID))[:, :, 0])

    mask_id = hp.MASK_ID
    word_embedding = hyper_layer.EmbeddingSharedWeights(
        vocab_size=ENGLISH_BYTE_VOCAB,
        hidden_size=hp.num_units,
        pad_id=hp.PAD_ID,
        name='word_embedding')
    mask_words = tf.zeros_like(img_input[:, :, 0],dtype=tf.int32) +  mask_id
    # mask_words = tf.constant(mask_id, shape=[batch_size, Q_input_length])
    mask_embedding = word_embedding(mask_words)
    V = tf.concat((img_input, mask_embedding), -1)
    V = tf.keras.layers.Dense(hp.num_units, name='fusion_V')(V)
    K = tf.keras.layers.Dense(hp.num_units, name='K')(img_input)
    Q = tf.keras.layers.Dense(hp.num_units, name='Q')(mask_embedding)

    # transformer_src = tf.keras.layers.Input(
    #     shape=[None], dtype=tf.int64, name='transformer_src_input')
    # transformer_tgt = tf.keras.layers.Input(
    #     shape=[None], dtype=tf.int64, name='transformer_output_input')
    transformer = get_transofomer(hp)
    encoder_out = transformer.Encoder((Q, K, V),
                                      img_input_padding,
                                      training=True)
    embedding_tgt_input = word_embedding(tgt_input)
    embedding_tgt_input = tf.pad(embedding_tgt_input,
                                 [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

    decoder_out = transformer.Decoder(
        embedding_tgt_input,
        encoder_out,
        padding_matrix=img_input_padding,
        training=True)
    logits = word_embedding.linear(decoder_out)
    model = tf.keras.Model([img_input, tgt_input], logits)
    return model


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
