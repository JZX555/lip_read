# encoding=utf8
import tensorflow as tf
import train_conf
import data_setentceToByte_helper
import numpy as np


class DatasetManager():
    def __init__(self,
                 source_data_path,
                 target_data_path,
                 batch_size=32,
                 shuffle=100,
                 num_sample=-1,
                 max_length=50,
                 SOS_ID=1,
                 EOS_ID=2,
                 PAD_ID=0,
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
        self.split_token = split_token
        self.word_token = word_token
        self.SOS_ID = SOS_ID
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

    def cross_validation(self, data_path, validation=0.15, test=0.15):
        train_path = data_path + "_train"
        test_path = data_path + "_test"
        val_path = data_path + "_val"
        raw_data = []
        with tf.gfile.GFile(data_path, "r") as f:
            raw_data = f.readlines()
            self.data_counter = len(raw_data)
            # val_size = int(validation * self.data_counter)
            val_size = 3200
            # test_size = int(test * self.data_counter)
            test_size = 3200
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

    def encode(self, string):
        return self.byter.encode(string)

    def decode(self, string):
        return self.byter.decode(string)

    def parser_fn(self, string):
        """
            All data should be regarded as a numpy array
        """
        string = string.numpy().decode('utf8').strip()
        return np.array(
            self.byter.encode(string, add_eos=True),
            dtype=np.int32)[:self.max_length]

    def create_dataset(self, data_path):
        dataset = tf.data.TextLineDataset(data_path)
        dataset = dataset.map(
            lambda string: tf.py_function(self.parser_fn, [string], tf.int32),
            num_parallel_calls=self.cpus)
        return dataset

    def post_process(self, validation=0.15, test=0.15):
        src_train_path, src_test_path, src_val_path = self.cross_validation(
            self.source_data_path)
        src_train_dataset = self.create_dataset(src_train_path)
        src_val_dataset = self.create_dataset(src_val_path)
        src_test_dataset = self.create_dataset(src_test_path)

        tgt_train_path, tgt_test_path, tgt_val_path = self.cross_validation(
            self.target_data_path)
        tgt_train_dataset = self.create_dataset(tgt_train_path)
        tgt_val_dataset = self.create_dataset(tgt_val_path)
        tgt_test_dataset = self.create_dataset(tgt_test_path)

        def body(src_dataset, tgt_dataset):
            source_target_dataset = tf.data.Dataset.zip((src_dataset,
                                                         tgt_dataset))
            source_target_dataset = source_target_dataset.shuffle(self.shuffle)
            return source_target_dataset

        train_dataset = body(src_train_dataset, tgt_train_dataset)
        val_dataset = body(src_val_dataset, tgt_val_dataset)
        test_dataset = body(src_test_dataset, tgt_test_dataset)
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


# tf.enable_eager_execution()
# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
# sentenceHelper = DatasetManager(
#     DATA_PATH + "/europarl-v7.fr-en.fr",
#     DATA_PATH + "/europarl-v7.fr-en.en",
#     batch_size=16,
#     shuffle=100)
# a, b, c = sentenceHelper.prepare_data()
#
# for i, (e,b) in enumerate(a):
#     e = b[1]
#     print(i)
#     break
# sentenceHelper.byter.decode(e.numpy())
# d = a.make_one_shot_iterator()
# d = d.get_next()
