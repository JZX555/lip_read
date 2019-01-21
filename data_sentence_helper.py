# encoding=utf8

import re
import tensorflow as tf
import json
import numpy as np
"""
This is a basic helper to format sentences into dataset.
In this implementation, I try to use raw python function as much as possible,
because I just do not want to use some functions from tf.contrib which is not LST,
considering tf 2.0 is coming.
Following this concept, I use tf.py_func in this implementation aloneside dataset.map.
The vocabulary will be stored in the hard driver, I strongly recommend this because
of tow following reasons, respecting tradition nlp and easy to maintain, however if the
vocabulary is changed, I suppose all the model should be re-train.
Args:
    source_data_path (type): Description of parameter `source_data_path`.
    target_data_path (type): Description of parameter `target_data_path`.
    num_sample (type): Description of parameter `num_sample`.
    batch_size (type): Description of parameter `batch_size`.
    split_token (type): Description of parameter `split_token`.

Attributes:
    UNK (type): Description of parameter `UNK`.
    SOS (type): Description of parameter `SOS`.
    EOS (type): Description of parameter `EOS`.
    UNK_ID (type): Description of parameter `UNK_ID`.
    SOS_ID (type): Description of parameter `SOS_ID`.
    EOS_ID (type): Description of parameter `EOS_ID`.
    source_data_path
    target_data_path
    num_sample
    batch_size
    split_token

"""

# def __init__(self,
#              source_data_path,
#              target_data_path,
#              batch_size=32,
#              num_sample=-1,
#              split_token='\n'):
#     """Short summary.
#
#     Args:
#         source_data_path (type): Description of parameter `source_data_path`.
#         target_data_path (type): Description of parameter `target_data_path`.
#         num_sample (type): Description of parameter `num_sample`.
#         batch_size (type): Description of parameter `batch_size`.
#         split_token (type): Description of parameter `split_token`.
#
#     Returns:
#         type: Description of returned object.
#
#     """
#     self.source_data_path = source_data_path
#     self.target_data_path = target_data_path
#     self.num_sample = num_sample
#     self.batch_size = batch_size
#     self.split_token = split_token
#     self.UNK = "<unk>"
#     self.SOS = "<sossss>"
#     self.EOS = "<eossss>"
#     self.UNK_ID = 0
#     self.SOS_ID = 1
#     self.EOS_ID = 2
SPLIT_TOKEN = '\n'
UNK = "<unk>"
SOS = "<sossss>"
EOS = "<eossss>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def preprocess_sentence(npa):
    """Short summary.
    This is the first step of sentence help, actually I found this on stack overflow,
    but I cannot find the reference any more.
    Args:
        npa (type): Description of parameter `npa`.

    Returns:
        type: Description of returned object.
    """
    npa = np.array([npa.decode("utf-8")])
    for i in range(0, len(npa)):
        w = npa[i].lower().strip()
        w = re.sub("([?.!,¿])", r" \1 ", w)
        w = re.sub('[" "]+', SPLIT_TOKEN, w)
        w = re.sub("[^a-zA-Z?.!,¿]+", SPLIT_TOKEN, w)
        npa[i] = w
    return npa


def lookup(npa, vocabulary):
    """Short summary.

    This is a folk of tf lookup_table, but it is not necessary to be initialized,
    which is an convinient in DEBUG.
    Args:
        npa (type): Description of parameter `npa`.
        vocabulary (type): Description of parameter `vocabulary`.

    Returns:
        type: Description of returned object.

    """

    npa = np.array(npa)
    vocabulary = json.loads(vocabulary)
    for i in range(0, len(npa)):
        try:
            npa[i] = vocabulary[npa[i].decode("utf-8")]
        except Exception:
            npa[i] = UNK_ID
    return npa.astype('int32')


def create_dataset(data_path):
    """Short summary.
        This is a little trick.
        TF supports naive python functions underlying tf.py_func.

    """
    # word2idx, idx2word = self.word_index(data_path)
    vocabulary, idx2word = word_index(data_path)
    json_vocabulary = json.dumps(vocabulary)
    dataset = tf.data.TextLineDataset(data_path)
    dataset = dataset.map(
        lambda string: tf.py_func(preprocess_sentence, [string], string.dtype))
    dataset = dataset.map(
        lambda string: tf.string_split(string, SPLIT_TOKEN).values)
    dataset = dataset.map(
        lambda string: tf.py_func(lookup, [string, json_vocabulary], tf.int32))
    return dataset, vocabulary, idx2word


def _build_vocabulary(vocabulary_file):
    """Short summary.

    tf.gfile.GFile is recommended here.
    This function try to initialize vocabulary from loacl, otherwise it will build a
    brand new vocabulary based on corpus, followed by storing process.
    Args:
        vocabulary_file (type): Description of parameter `vocabulary_file`.

    Returns:
        type: Description of returned object.

    """
    if tf.gfile.Exists(vocabulary_file + "_vocabulary.txt"):
        with tf.gfile.GFile(vocabulary_file + "_vocabulary.txt", "r") as f:
            word_to_id_table = f.read().split(SPLIT_TOKEN)
    else:
        with tf.gfile.GFile(vocabulary_file, "r") as f:
            raw = set()
            for k, v in enumerate(f):
                for s in preprocess_sentence(v):
                    temp = s.strip().split(SPLIT_TOKEN)
                    raw.update(temp)
            raw_data = sorted(raw)
        with tf.gfile.GFile(vocabulary_file + "_vocabulary.txt", "w") as f:
            f.write("%s\n" % UNK)
            f.write("%s\n" % SOS)
            f.write("%s\n" % EOS)
            for w in raw_data:
                f.write("%s\n" % w)
        with tf.gfile.GFile(vocabulary_file + "_vocabulary.txt", "r") as f:
            word_to_id_table = f.read().split(SPLIT_TOKEN)
    return word_to_id_table


def word_index(vocabulary_file):
    """Short summary.
    get vocabulary dict.
    Args:
        vocabulary_file (type): Description of parameter `vocabulary_file`.

    Returns:
        batched_dataset, src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2words

    """
    voc = _build_vocabulary(vocabulary_file)
    vocabulary = dict()
    idx2word = dict()
    for index, word in enumerate(voc):
        vocabulary[word] = index

    for word, index in vocabulary.items():
        idx2word[index] = word
    return vocabulary, idx2word


def prepare_data(source_data_path, target_data_path, batch_size):
    # src_dataset, src_word2idx, src_idx2word = self.create_dataset(
    #     self.source_data_path)
    src_dataset, src_vocabulary, src_ids2word = create_dataset(
        source_data_path)
    # tgt_dataset, tgt_word2idx, tgt_idx2word = self.create_dataset(
    #     self.target_data_path)
    tgt_dataset, tgt_vocabulary, tgt_ids2word = create_dataset(
        target_data_path)
    source_target_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    source_target_dataset = source_target_dataset.map(
        lambda src, tgt:
            (src, tf.concat(([SOS_ID], tgt), 0), tf.concat((tgt, [EOS_ID]), 0)))
    # source_target_dataset = source_target_dataset.map(
    #     lambda src, tgt: (src, tf.concat((tgt, [EOS_ID]), 0)))
    source_target_dataset = source_target_dataset.map(
        lambda src, tgt_in, tgt_out:
            ((src, tgt_in, tf.size(src), tf.size(tgt_in)), tgt_out))
    # source_target_dataset = source_target_dataset.map(
    #     lambda src, tgt: (src, tgt, tf.size(src), tf.size(tgt)))
    # batched_dataset = source_target_dataset.padded_batch(
    #     batch_size,
    #     padded_shapes=((
    #         tf.TensorShape([None]),  # source vectors of unknown size
    #         tf.TensorShape([None]),  # target vectors of unknown size
    #         tf.TensorShape([None]),  # target vectors of unknown size
    #         tf.TensorShape([]),
    #         tf.TensorShape([]),  # size(source)
    #     )),
    #     padding_values=((
    #         EOS_ID,  # source vectors padded on the right with src_eos_id
    #         EOS_ID,  # target vectors padded on the right with tgt_eos_id
    #         EOS_ID,
    #         0,
    #         0)))  # size(target) -- unused
    batched_dataset = source_target_dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (
                tf.TensorShape([None]),  # source vectors of unknown size
                tf.TensorShape([None]),  # target vectors of unknown size
                tf.TensorShape([]),
                tf.TensorShape([]),  # size(source)
            ),
            tf.TensorShape([None])),
        padding_values=(
            (
                EOS_ID,  # source vectors padded on the right with src_eos_id
                EOS_ID,  # target vectors padded on the right with tgt_eos_id
                0,
                0),
            EOS_ID))  # size(target) -- unused
    return batched_dataset, src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2word


def tgt_formater(tgt):
    """Short summary.
    format tgt to tgt_input[SOS_ID, tgt] and tgt_output[tgt, EOS_ID]
    Args:
        tgt (type): Description of parameter `tgt`.

    Returns:
        type: Description of returned object.

    """
    input_padding = tf.constant([[0, 0], [1, 0]])
    output_padding = tf.constant([[0, 0], [0, 1]])
    tgt_input = tf.pad(
        tgt, input_padding, mode='CONSTANT', constant_values=SOS_ID)
    tgt_output = tf.pad(
        tgt, output_padding, mode='CONSTANT', constant_values=EOS_ID)
    return tgt_input, tgt_output
