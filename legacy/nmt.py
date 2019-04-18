# encoding=utf-8

import sys
sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng/')
sys.path.insert(1, '/home/vivalavida/workspace/alpha/transformer_nmt/')
TRAIN_MODE = 'large'
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# TRAIN_MODE = 'test'
import hyperParam
import core_Transformer_model
import core_dataset_generator
import graph_trainer
import train_conf
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"
hp = hyperParam.HyperParam(TRAIN_MODE)
data_manager = core_dataset_generator.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    PAD_ID=hp.PAD_ID,
    EOS_ID=hp.EOS_ID,
    # shuffle=hp.data_shuffle,
    shuffle=hp.data_shuffle,
    max_length=hp.max_sequence_length)
# dataset_train_val_test = data_manager.prepare_data()
# src_vocabulary, src_ids2word, tgt_vocabulary, tgt_ids2word = data_manager.prepare_vocabulary(
# )
gpu = train_conf.get_available_gpus()

vocabulary_size = 24000
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
# test = test_trainer.Trainer(model=model, dataset=dataset_train_val_test)
# test.train()
large_train = graph_trainer.Graph(
    vocab_size=vocabulary_size,
    sys_path=SYS_PATH,
    hyperParam=hp)
large_train.graph(model, data_manager)
