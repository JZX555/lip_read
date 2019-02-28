# encoding=utf8

import re
import tensorflow as tf
import json
import numpy as np


class SentenceHelper():
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

    def __init__(self,
                 source_data_path,
                 target_data_path,
                 batch_size=32,
                 shuffle=100,
                 num_sample=-1,
                 max_length=50,
                 split_token='\n'):
        """Short summary.

        Args:
            source_data_path (type): Description of parameter `source_data_path`.
            target_data_path (type): Description of parameter `target_data_path`.
            num_sample (type): Description of parameter `num_sample`.
            batch_size (type): Description of parameter `batch_size`.
            split_token (type): Description of parameter `split_token`.

        Returns:
            type: Description of returned object.

        """
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.split_token = split_token
        self.UNK = "<unk>"
        self.SOS = "<sossss>"
        self.EOS = "<eossss>"
        self.UNK_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.shuffle = shuffle
        self.max_length = max_length

    # SPLIT_TOKEN = '\n'
    # UNK = "<unk>"
    # SOS = "<sossss>"
    # EOS = "<eossss>"
    # UNK_ID = 0
    # SOS_ID = 1
    # EOS_ID = 2

    def preprocess_sentence(self, npa):
        """Short summary.
        This is the first step of sentence help, actually I found this on stack overflow,
        but I cannot find the reference any more.
        Args:
            npa (type): Description of parameter `npa`.

        Returns:
            type: Description of returned object.
        """
        if isinstance(npa, str):
            npa = np.array([npa])
        else:
            npa = np.array([npa.decode("utf-8")])
            # npa = u' '.join(npa).encode('utf-8').strip()
            # import pdb; pdb.set_trace()
            # npa = np.array([npa])
        for i in range(0, len(npa)):
            w = npa[i].lower().strip()
            w = re.sub("([?.!,¿])", r" \1 ", w)
            w = re.sub('[" "]+', self.split_token, w)
            w = re.sub("[^a-zA-Z?.!,¿]+", self.split_token, w)
            npa[i] = w
        return npa

    def lookup(self, npa, vocabulary):
        """Short summary.

        This is a folk of tf lookup_table, but it is not necessary to be initialized,
        which is an convinient in DEBUG.
        Args:
            npa (type): Description of parameter `npa`.
            vocabulary (type): Description of parameter `vocabulary`.

        Returns:
            type: Description of returned object.

        """
        try:
            npa = np.array(npa)
            vocabulary = json.loads(vocabulary)
            for i in range(0, len(npa)):
                try:
                    npa[i] = vocabulary[npa[i].decode("utf-8")]
                except Exception:
                    npa[i] = self.UNK_ID
        except Exception:
            print(npa)
        return npa.astype('int32')

    def create_dataset(self, data_path):
        """Short summary.
            This is a little trick.
            TF supports naive python functions underlying tf.py_func.

        """
        # word2idx, idx2word = self.word_index(data_path)
        vocabulary, idx2word = self.word_index(data_path)
        json_vocabulary = json.dumps(vocabulary)
        with tf.gfile.GFile(data_path, "r") as f:
            self.data_counter = len(f.readlines())
            f.close()
        dataset = tf.data.TextLineDataset(data_path)
        dataset = dataset.map(
            lambda string: tf.py_func(lambda string: string.lower(), [string], tf.string, stateful=False)
        )
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(tf.strings.strip(string), "([?.!,¿])", r" \1 ")
        )
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(string, '[" "]+', self.split_token)
        )
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(string, "[^a-zA-Z?.!,¿]+", self.split_token)
        )
        dataset = dataset.map(
            lambda string: tf.string_split([string], self.split_token).values)
        dataset = dataset.map(
            lambda string: tf.py_func(self.lookup, [string, json_vocabulary], tf.int32)
        )
        return dataset, vocabulary, idx2word

    def _build_vocabulary(self, vocabulary_file):
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
                word_to_id_table = f.read().split(self.split_token)
        else:
            with tf.gfile.GFile(vocabulary_file, "r") as f:
                raw = set()
                for k, v in enumerate(f):
                    for s in self.preprocess_sentence(v):
                        temp = s.strip().split(self.split_token)
                        raw.update(temp)
                raw_data = sorted(raw)
            with tf.gfile.GFile(vocabulary_file + "_vocabulary.txt", "w") as f:
                f.write("%s\n" % self.UNK)
                f.write("%s\n" % self.SOS)
                f.write("%s\n" % self.EOS)
                for w in raw_data:
                    f.write("%s\n" % w)
            with tf.gfile.GFile(vocabulary_file + "_vocabulary.txt", "r") as f:
                word_to_id_table = f.read().split(self.split_token)
        return word_to_id_table

    def word_index(self, vocabulary_file):
        """Short summary.
        get vocabulary dict.
        Args:
            vocabulary_file (type): Description of parameter `vocabulary_file`.

        Returns:
            batched_dataset, src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2words

        """
        voc = self._build_vocabulary(vocabulary_file)
        vocabulary = dict()
        idx2word = dict()
        for index, word in enumerate(voc):
            vocabulary[word] = index

        for word, index in vocabulary.items():
            idx2word[index] = word
        return vocabulary, idx2word

    def post_process(self, validation=0.15, test=0.15):
        # src_dataset, src_word2idx, src_idx2word = self.create_dataset(
        #     self.source_data_path)
        src_dataset, src_vocabulary, src_ids2word = self.create_dataset(
            self.source_data_path)
        # tgt_dataset, tgt_word2idx, tgt_idx2word = self.create_dataset(
        #     self.target_data_path)
        tgt_dataset, tgt_vocabulary, tgt_ids2word = self.create_dataset(
            self.target_data_path)
        source_target_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        source_target_dataset = source_target_dataset.map(
            lambda src, tgt: (src[:self.max_length], tgt[:self.max_length]))
        source_target_dataset = source_target_dataset.map(
            lambda src, tgt:
                (src, tf.concat(([self.SOS_ID], tgt), 0), tf.concat((tgt, [self.EOS_ID]), 0)))
        # source_target_dataset = source_target_dataset.map(
        #     lambda src, tgt: (src, tf.concat((tgt, [EOS_ID]), 0)))
        # source_target_dataset = source_target_dataset.map(
        #     lambda src, tgt_in, tgt_out:
        #         ((src, tgt_in, tf.size(src), tf.size(tgt_in)), tgt_out))
        source_target_dataset = source_target_dataset.map(
            lambda src, tgt_in, tgt_out:
                ((src, tgt_in, tf.size(src), tf.size(tgt_in)), tgt_out))
        source_target_dataset.shuffle(1000000)
        print('Total data {0}'.format(self.data_counter))
        val_size = int(validation * self.data_counter)
        test_size = int(test * self.data_counter)
        # train_size = self.data_counter - val_size - test_size
        # source_target_dataset = source_target_dataset.shuffle(self.shuffle)
        val_dataset = source_target_dataset.take(val_size)
        test_dataset = source_target_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
        train_dataset = test_dataset.skip(test_size)
        # test_dataset = source_target_dataset.skip(train_size)
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
        return train_dataset, val_dataset, test_dataset

    def padding(self, dataset):
        batched_dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                (
                    tf.TensorShape([None]),  # source vectors of unknown size
                    tf.TensorShape([None]),  # target vectors of unknown size
                    tf.TensorShape([]),
                    tf.TensorShape([])
                    # size(source)
                ),
                tf.TensorShape([None])),
            padding_values=(
                (
                    self.
                    EOS_ID,  # source vectors padded on the right with src_eos_id
                    self.
                    EOS_ID,  # target vectors padded on the right with tgt_eos_id
                    0,
                    0),
                self.EOS_ID),
            drop_remainder=True)  # size(target) -- unused
        return batched_dataset

    def prepare_data(self):
        train_dataset, val_dataset, test_dataset = self.post_process()
        train_dataset = self.padding(train_dataset)
        val_dataset = self.padding(val_dataset)
        test_dataset = self.padding(test_dataset)

        # 1-tf.cast(tf.equal(src, self.EOS_ID), tf.int32)
        # batched_dataset = batched_dataset.app
        return train_dataset, val_dataset, test_dataset

    def prepare_vocabulary(self):
        src_vocabulary, src_ids2word = self.word_index(self.source_data_path)
        tgt_vocabulary, tgt_ids2word = self.word_index(self.target_data_path)
        return src_vocabulary, src_ids2word, tgt_vocabulary, tgt_ids2word

    def tgt_formater(self, tgt):
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
            tgt, input_padding, mode='CONSTANT', constant_values=self.SOS_ID)
        tgt_output = tf.pad(
            tgt, output_padding, mode='CONSTANT', constant_values=self.EOS_ID)
        return tgt_input, tgt_output


DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
sentenceHelper = SentenceHelper(
    DATA_PATH + "/europarl-v7.fr-en.fr",
    DATA_PATH + "/europarl-v7.fr-en.en",
    batch_size=16,
    shuffle=100000)
