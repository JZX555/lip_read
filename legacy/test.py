#encoding=utf8

import tensorflow as tf
import core_Transformer_model
import sys
import hyperParam
import train_conf
import core_dataset_generator
import visualization
# sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng/')
# sys.path.insert(1, '/home/vivalavida/workspace/alpha/transformer_nmt/')
# TRAIN_MODE = 'large'

sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
TRAIN_MODE = 'test'

DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"
tf.enable_eager_execution()
hp = hyperParam.HyperParam(TRAIN_MODE)
gpu = train_conf.get_available_gpus()
vocabulary_size = 24000
data_manager = core_dataset_generator.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    PAD_ID=hp.PAD_ID,
    EOS_ID=hp.EOS_ID,
    # shuffle=hp.data_shuffle,
    shuffle=hp.data_shuffle,
    max_length=hp.max_sequence_length)
model = core_Transformer_model.Daedalus(
    max_seq_len=hp.max_sequence_length,
    vocabulary_size=vocabulary_size,
    embedding_size=hp.embedding_size,
    batch_size=hp.batch_size / (gpu if gpu > 0 else 1),
    num_units=hp.num_units,
    num_heads=hp.num_heads,
    num_encoder_layers=hp.num_encoder_layers,
    num_decoder_layers=hp.num_decoder_layers,
    dropout=hp.dropout,
    eos_id=hp.EOS_ID,
    pad_id=hp.PAD_ID)
model.load_weights(SYS_PATH + '/model/model_weights')
test = tf.convert_to_tensor([data_manager.encode("I like cat")])
re = model(test, train=False)
att = model.get_attention().numpy()
