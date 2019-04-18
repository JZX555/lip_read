# encoding=utf8

import re
import tensorflow as tf
import collections
import train_conf


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
                 UNK_ID=0,
                 SOS_ID=1,
                 EOS_ID=2,
                 PAD_ID=3,
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
        self.PAD = "<paddddd>"
        self.UNK_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.PAD_ID = 3
        self.shuffle = shuffle
        self.max_length = max_length
        if train_conf.get_available_gpus() > 0:
            self.cpus = 8
        else:
            self.cpus = 2

    def cross_validation(self, data_path, validation=0.15, test=0.15):
        train_path = data_path + "_train"
        test_path = data_path + "_test"
        val_path = data_path + "_val"
        raw_data = []
        with tf.gfile.GFile(data_path, "r") as f:
            raw_data = f.readlines()
            self.data_counter = len(raw_data)
            val_size = int(validation * self.data_counter)
            test_size = int(test * self.data_counter)
            self.train_size = self.data_counter - val_size - test_size
            f.close()

            def writer(path, data):
                if tf.gfile.Exists(path) is not True:
                    with tf.gfile.GFile(path, "w") as f:
                        for w in data:
                            if len(w) >= 0:
                                f.write("%s" % w)
                        f.close()
                print("File exsits: {}".format(path))

            print('Total data {0}'.format(self.data_counter))
            print(('Train {0}, Validation {1}, Test {2}'.format(
                self.train_size, val_size, test_size)))
            writer(train_path, raw_data[:self.train_size])
            # writer(val_path, raw_data[:128])
            writer(val_path,
                   raw_data[self.train_size:self.train_size + val_size])
            writer(test_path, raw_data[self.train_size + val_size:])
        return train_path, test_path, val_path

    def preprocess_sentence(self, text, table=None):
        """Short summary.
        This is the first step of sentence help, actually I found this on stack overflow,
        but I cannot find the reference any more.
        Args:
            npa (type): Description of parameter `npa`.

        Returns:
            type: Description of returned object.
        """
        w = text.lower().strip()
        w = re.sub("([?.!,¿])", r" \1 ", w)
        w = re.sub('[" "]+', self.split_token, w)
        w = re.sub("[^a-zA-Z?.!,¿]+", self.split_token, w).strip()
        w = w.split(self.split_token)
        return w

    def create_dataset(self, data_path, vocabulary):
        """Short summary.
            This is a little trick.
            TF supports naive python functions underlying tf.py_func.

        """

        def lookup(npa):
            try:
                npa = npa.numpy()
                for i in range(0, len(npa)):
                    try:
                        npa[i] = vocabulary[npa[i].decode("utf-8")]
                    except Exception:
                        print(npa[i])
                        npa[i] = self.UNK_ID
            except Exception:
                print(npa)
            return npa.astype('int32')

        dataset = tf.data.TextLineDataset(data_path)
        dataset = dataset.map(
            lambda string: tf.py_function(lambda string: string.numpy().lower(), [string], tf.string),num_parallel_calls=self.cpus
        )
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(tf.strings.strip(string), "([?.!,¿])", r" \1 "), num_parallel_calls=self.cpus
        )
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(string, '[" "]+', self.split_token),
            num_parallel_calls=self.cpus)
        dataset = dataset.map(
            lambda string: tf.strings.regex_replace(string, "[^a-zA-Z?.!,¿]+", self.split_token),
            num_parallel_calls=self.cpus)
        dataset = dataset.map(
            lambda string: tf.string_split([string], self.split_token).values,
            num_parallel_calls=self.cpus)
        dataset = dataset.map(
            lambda string: tf.py_function(lookup, [string], tf.int32),
            num_parallel_calls=self.cpus)
        return dataset

    def build_vocabulary(self, vocabulary_file):
        vocabulary_name = vocabulary_file + "_vocabulary.txt"
        if tf.gfile.Exists(vocabulary_name) is not True:
            with tf.gfile.GFile(vocabulary_file, "r") as f:
                data = self.preprocess_sentence(f.read())
                counter = collections.Counter(data)
                count_pairs = sorted(
                    counter.items(), key=lambda x: (-x[1], x[0]))

                raw_data, _ = list(zip(*count_pairs))
            with tf.gfile.GFile(vocabulary_name, "w") as f:
                f.write("%s\n" % self.UNK)
                f.write("%s\n" % self.SOS)
                f.write("%s\n" % self.EOS)
                f.write("%s\n" % self.PAD)
                for w in raw_data:
                    f.write("%s\n" % w)
        with tf.gfile.GFile(vocabulary_name, "r") as f:
            vocabulary = dict()
            idx2word = dict()
            for index, word in enumerate(f):
                vocabulary[word.rstrip()] = index
            for word, index in vocabulary.items():
                idx2word[index] = word
        return vocabulary, idx2word

    def post_process(self, validation=0.15, test=0.15):
        src_train_path, src_test_path, src_val_path = self.cross_validation(
            self.source_data_path)
        src_vocabulary, src_idx2word = self.build_vocabulary(
            self.source_data_path)
        src_train_dataset = self.create_dataset(src_train_path, src_vocabulary)
        src_val_dataset = self.create_dataset(src_val_path, src_vocabulary)
        src_test_dataset = self.create_dataset(src_test_path, src_vocabulary)

        tgt_train_path, tgt_test_path, tgt_val_path = self.cross_validation(
            self.target_data_path)
        tgt_vocabulary, tgt_idx2word = self.build_vocabulary(
            self.target_data_path)
        tgt_train_dataset = self.create_dataset(tgt_train_path, tgt_vocabulary)
        tgt_val_dataset = self.create_dataset(tgt_val_path, tgt_vocabulary)
        tgt_test_dataset = self.create_dataset(tgt_test_path, tgt_vocabulary)

        def body(src_dataset, tgt_dataset):
            source_target_dataset = tf.data.Dataset.zip((src_dataset,
                                                         tgt_dataset))
            source_target_dataset = source_target_dataset.shuffle(self.shuffle)
            source_target_dataset = source_target_dataset.map(
                lambda src, tgt: (src[:self.max_length], tgt[:self.max_length]),
                num_parallel_calls=self.cpus)
            source_target_dataset = source_target_dataset.map(
                lambda src, tgt:
                    (src, tf.concat(([self.SOS_ID], tgt), 0), tf.concat((tgt, [self.EOS_ID]), 0)),num_parallel_calls=self.cpus)
            source_target_dataset = source_target_dataset.map(
                lambda src, tgt_in, tgt_out:
                    ((src, tgt_in, tf.size(src), tf.size(tgt_in)), tgt_out),num_parallel_calls=self.cpus)
            return source_target_dataset

        train_dataset = body(src_train_dataset, tgt_train_dataset)
        val_dataset = body(src_val_dataset, tgt_val_dataset)
        test_dataset = body(src_test_dataset, tgt_test_dataset)
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
                    PAD_ID,  # source vectors padded on the right with src_eos_id
                    self.
                    PAD_ID,  # target vectors padded on the right with tgt_eos_id
                    0,
                    0),
                self.PAD_ID),
            drop_remainder=True)  # size(target) -- unused
        return batched_dataset

    def prepare_data(self):
        train_dataset, val_dataset, test_dataset = self.post_process()
        train_dataset = self.padding(train_dataset)
        val_dataset = self.padding(val_dataset)
        test_dataset = self.padding(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def prepare_vocabulary(self):
        src_vocabulary, src_ids2word = self.build_vocabulary(
            self.source_data_path)
        tgt_vocabulary, tgt_ids2word = self.build_vocabulary(
            self.target_data_path)
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


# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
# sentenceHelper = SentenceHelper(
#     DATA_PATH + "/europarl-v7.fr-en.fr",
#     DATA_PATH + "/europarl-v7.fr-en.en",
#     batch_size=16,
#     shuffle=100000)
# a, b, c = sentenceHelper.prepare_data()
# d = a.make_one_shot_iterator()
# d.get_next()
# sess = tf.Session()
# d = sess.run(d.get_next())
# e = d[0]
