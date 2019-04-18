# encoding=utf8
import tensorflow as tf
import core_Transformer_model


def get_vgg():
    if tf.gfile.Exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet')
    return vgg16


def get_transofomer(hp):
    transformer = core_Transformer_model.Daedalus(
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

    vgg16 = get_vgg()
    vgg16_flatten = vgg16.get_layer('flatten')
    vgg16_output = vgg16_flatten.output
    transformer_src = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='transformer_src_input')
    transformer_tgt = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='transformer_output_input')
    transformer = get_transofomer(hp)
