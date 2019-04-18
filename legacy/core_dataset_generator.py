# encoding=utf8
from hyper_and_conf import conf_fn as train_conf
from data import data_setentceToByte_helper
import numpy as np
import tensorflow as tf


class DatasetManager():
    def __init__(self,
                 source_data_path,
                 target_data_path,
                 batch_size=32,
                 shuffle=100,
                 num_sample=-1,
                 max_length=50,
                 EOS_ID=1,
                 PAD_ID=0,
                 byte_token='@@',
                 word_token=' ',
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
        self.byte_token = byte_token
        self.split_token = split_token
        self.word_token = word_token
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.shuffle = shuffle
        self.max_length = max_length
        self.byter = data_setentceToByte_helper.Subtokenizer(
            [self.source_data_path, self.target_data_path],
            PAD_ID=self.PAD_ID,
            EOS_ID=self.EOS_ID)
        if train_conf.get_available_gpus() > 0:
            self.cpus = 12
        else:
            self.cpus = 2
        self.cross_validation(self.source_data_path, self.target_data_path)

    def corpus_length_checker(self, path, data):
        short_20 = 0
        median_50 = 0
        long_100 = 0
        super_long = 0
        for k, v in enumerate(data):
            v = v.split(self.word_token)
            v_len = len(v)
            if v_len <= 20:
                short_20 += 1
            if v_len > 20 and v_len <= 50:
                median_50 += 1
            if v_len > 50 and v_len <= 100:
                long_100 += 1
            if v_len > 100:
                super_long += 1
        print("Statistics for %s" % path)
        print("short: %d" % short_20)
        print("median: %d" % median_50)
        print("long: %d" % long_100)
        print("super long: %d" % super_long)

    def cross_validation(self, src_path, tgt_path, validation=0.05, test=0.05):
        print("Cross validation process")
        self.train_path_byte = "./data/train_data_BYTE_LEVEL"
        self.test_path_byte = "./data/test_data_BYTE_LEVEL"
        self.val_path_byte = "./data/val_data_BYTE_LEVEL"
        train_path_word = "./data/train_data_WORD_LEVEL"
        test_path_word = "./data/test_data_WORD_LEVEL"
        val_path_word = "./data/val_data_WORD_LEVEL"
        with tf.gfile.GFile(src_path, "r") as f_src:
            src_raw_data = f_src.readlines()
            self.corpus_length_checker(src_path, src_raw_data)
            with tf.gfile.GFile(tgt_path, "r") as f_tgt:
                tgt_raw_data = f_tgt.readlines()
                self.corpus_length_checker(tgt_path, tgt_raw_data)
                raw_data = list(zip(src_raw_data, tgt_raw_data))
                f_src.close()
            self.data_counter = len(raw_data)
            # val_size = int(validation * self.data_counter)
            # test_size = int(test * self.data_counter)
            self.val_size = 32000
            self.test_size = 32000
            self.train_size = self.data_counter - self.val_size - self.test_size
            f_src.close()

            def parser(string):
                string = self.byter.encode(string, add_eos=True)
                string = self.word_token.join([str(s) for s in string])
                return string

            def writer(path, data, byte=False):
                if tf.gfile.Exists(path) is not True:
                    with tf.gfile.GFile(path, "w") as f:
                        for w in data:
                            if len(w) >= 0:
                                if byte:
                                    f.write(
                                        parser(w[0].rstrip()) +
                                        self.byte_token +
                                        parser(w[1].rstrip()) + '\n')
                                else:
                                    f.write(w[0].rstrip() + self.byte_token +
                                            w[1])
                        f.close()
                print("File exsits: {}".format(path))

            print('Total data {0}'.format(self.data_counter))
            print(('Train {0}, Validation {1}, Test {2}'.format(
                self.train_size, self.val_size, self.test_size)))
            writer(train_path_word, raw_data[:self.train_size])
            # writer(val_path, raw_data[:128])
            writer(val_path_word,
                   raw_data[self.train_size:self.train_size + self.val_size])
            writer(test_path_word, raw_data[self.train_size + self.val_size:])
            writer(self.train_path_byte, raw_data[:self.train_size], byte=True)
            # writer(val_path, raw_data[:128])
            writer(
                self.val_path_byte,
                raw_data[self.train_size:self.train_size + self.val_size],
                byte=True)
            writer(
                self.test_path_byte,
                raw_data[self.train_size + self.val_size:],
                byte=True)
            del raw_data, tgt_raw_data, src_raw_data
        return self.train_path_byte, self.test_path_byte, self.val_path_byte

    def encode(self, string):
        return self.byter.encode(string)

    # def decode(self, string):
    #     def body(string):
    #         string = string.numpy().tolist()
    #         re = []
    #         for s in string:
    #             s = bleu_metrics.token_trim(s,self.EOS_ID)
    #             s = self.byter.decode(s).decode("utf8")
    #             re.append(s)
    #         return re
    #     string = tf.py_function(body, [string], tf.string)
    #     return string
    def decode(self, string):
        return self.byter.decode(string)

    def parser_fn(self, string):
        """
            All data should be regarded as a numpy array
        """
        string = string.numpy().decode('utf8').strip()
        # string = string.split(self.byte_token)
        # res = (np.array(
        #     self.byter.encode(string[0], add_eos=True),
        #     dtype=np.int32)[:self.max_length],
        # np.array(
        #     self.byter.encode(string[1], add_eos=True),
        #     dtype=np.int32)[:self.max_length])
        res = np.array(
            self.byter.encode(string, add_eos=True),
            dtype=np.int32)[:self.max_length]
        return res

    def create_dataset(self, data_path):
        with tf.device("/cpu:0"):
            dataset = tf.data.TextLineDataset(data_path)
            # with tf.device("/device:CPU:0"):
            dataset = dataset.map(
                lambda string: tf.string_split([string], self.byte_token).values,
                num_parallel_calls=self.cpus)
            dataset = dataset.map(
                lambda string: (tf.string_split([string[0]], self.word_token
                                                ).values[:self.max_length],
                                tf.string_split([string[1]], self.word_token).
                                values[:self.max_length]),
                num_parallel_calls=self.cpus)
            dataset = dataset.map(
                lambda src, tgt: (tf.strings.to_number(src, tf.int32),
                                  tf.strings.to_number(tgt, tf.int32)),
                num_parallel_calls=self.cpus)
            # dataset = dataset.shuffle(self.shuffle)
            # dataset = dataset.map(
            #     lambda string: (tf.py_function(self.parser_fn, [string[0]], tf.int32), tf.py_function(self.parser_fn, [string[1]], tf.int32)),
            #     num_parallel_calls=self.cpus)
            return dataset

    def post_process(self, validation=0.05, test=0.05):
        train_path, test_path, val_path = self.cross_validation(
            self.source_data_path, self.target_data_path)
        train_dataset = self.create_dataset(train_path)
        val_dataset = self.create_dataset(val_path)
        test_dataset = self.create_dataset(test_path)
        return train_dataset, val_dataset, test_dataset

    def padding(self, dataset):
        batched_dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # source vectors of unknown size
                tf.TensorShape([None]),  # target vectors of unknown size
            ),
            padding_values=(
                self.
                PAD_ID,  # source vectors padded on the right with src_eos_id
                self.
                PAD_ID,  # target vectors padded on the right with tgt_eos_id
            ),
            drop_remainder=True)  # size(target) -- unused
        return batched_dataset

    def prepare_data(self):
        train_dataset, val_dataset, test_dataset = self.post_process()
        train_dataset = self.padding(train_dataset)
        val_dataset = self.padding(val_dataset)
        test_dataset = self.padding(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def get_raw_train_dataset(self):
        with tf.device('/cpu:0'):
            return self.create_dataset(self.train_path_byte)

    def get_raw_val_dataset(self):
        with tf.device('/cpu:0'):
            return self.create_dataset(self.val_path_byte)

    def get_raw_test_dataset(self):
        with tf.device("/cpu:0"):
            return self.create_dataset(self.test_path_byte)

    def get_train_size(self):
        return self.train_size

    def get_val_size(self):
        return self.val_size

    def get_test_size(self):
        return self.test_size


# tf.enable_eager_execution()
# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
# sentenceHelper = DatasetManager(
#     DATA_PATH + "/europarl-v7.fr-en.fr",
#     DATA_PATH + "/europarl-v7.fr-en.en",
#     batch_size=16,
#     shuffle=100)
# # a, b, c = sentenceHelper.prepare_data()
# a, b, c = sentenceHelper.post_process()
#
# for i, e in enumerate(a):
#     print(e[0])
#     print(i)
#     sentenceHelper.byter.decode(e[0].numpy())
#     break
# d = a.make_one_shot_iterator()
# d = d.get_next()
