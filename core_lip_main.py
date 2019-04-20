# encoding=utf8
import tensorflow as tf
import core_Transformer_model
from hyper_and_conf import hyper_layer

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
    import pdb; pdb.set_trace()
    vgg16 = get_vgg()
    vgg16_flatten = vgg16.get_layer('flatten')
    vgg16_output = vgg16_flatten.output


    word_embedding = hyper_layer.EmbeddingSharedWeights(
        vocab_size=ENGLISH_BYTE_VOCAB,
        hidden_size=hp.num_units,
        pad_id=hp.PAD_ID,
        name='word_embedding')

    transformer_src = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='transformer_src_input')
    transformer_tgt = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='transformer_output_input')
    transformer = get_transofomer(hp)


# if '__name__' == '__main__':
from hyper_and_conf import hyper_param
hp = hyper_param.HyperParam(mode='test')
model = main_model(hp)
