# encoding=utf8
from hyper_and_conf import conf_fn as train_conf
from data import data_setentceToByte_helper
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
                 cross_val=[0.89, 0.1, 0.01],
                 byte_token='@@',
                 word_token=' ',
                 split_token='\n',
                 tfrecord_path=None):
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
        self.cross_val = cross_val
        assert isinstance(self.cross_val, list) is True
        self.byter = data_setentceToByte_helper.Subtokenizer(
            self.source_data_path + self.target_data_path,
            PAD_ID=self.PAD_ID,
            EOS_ID=self.EOS_ID)
        if train_conf.get_available_gpus() > 0:
            self.cpus = 12 * train_conf.get_available_gpus()
        else:
            self.cpus = 4
        self.tfrecord_path = tfrecord_path

    def corpus_length_checker(self, data=None, re=False):
        self.short_20 = 0
        self.median_50 = 0
        self.long_100 = 0
        self.super_long = 0
        for k, v in enumerate(data):
            v = v.split(self.word_token)
            v_len = len(v)
            if v_len <= 20:
                self.short_20 += 1
            if v_len > 20 and v_len <= 50:
                self.median_50 += 1
            if v_len > 50 and v_len <= 100:
                self.long_100 += 1
            if v_len > 100:
                self.super_long += 1
        if re:
            print("short: %d" % self.short_20)
            print("median: %d" % self.median_50)
            print("long: %d" % self.long_100)
            print("super long: %d" % self.super_long)

    def encode(self, string, add_eos=True):
        return self.byter.encode(string, add_eos=True)

    def decode(self, string):
        return self.byter.decode(string)

    def one_file_encoder(self, file_path, num=None):
        with tf.gfile.GFile(file_path, "r") as f:
            raw_data = f.readlines()
            re = []
            if num is None:
                for d in raw_data:
                    re.append(self.encode(d))
            else:
                text = raw_data[num].split(":")[1].lower().rstrip().strip()

                re = self.encode(text)
            f.close()
        return re

    def one_file_decoder(self, file_path, line_num=None):
        with tf.gfile.GFile(file_path, "r") as f:
            raw_data = f.readlines()
            re = []
            if line_num is None:
                for d in raw_data:
                    re.append(self.decode(d))
                f.close()
            else:
                re.append(self.decode(raw_data[line_num]))
        return re

    def create_dataset(self, data_path):
        def _parse_example(serialized_example):
            """Return inputs and targets Tensors from a serialized tf.Example."""
            data_fields = {
                "text": tf.VarLenFeature(tf.int64),
                "img": tf.VarLenFeature(tf.float32)
            }
            # import pdb;pdb.set_trace()
            parsed = tf.parse_single_example(serialized_example, data_fields)
            img = tf.sparse_tensor_to_dense(parsed["img"])
            text = tf.sparse_tensor_to_dense(parsed["text"])
            return img, text

        def _filter_max_length(example, max_length=256):
            return tf.logical_and(
                tf.size(example[0]) <= max_length,
                tf.size(example[1]) <= max_length)

        # with tf.device("/cpu:0"):
        dataset = tf.data.TFRecordDataset(
            data_path, compression_type='GZIP', buffer_size=200)
        dataset = dataset.map(_parse_example, num_parallel_calls=self.cpus)
        dataset = dataset.map(
            lambda img, text: (tf.reshape(img, [-1, 25088]), text),
            num_parallel_calls=self.cpus)
        # dataset = dataset.filter(lambda x, y: _filter_max_length((
        #     x, y), self.max_length))
        # dataset = dataset.apply(tf.data.experimental.ignore_errors())
        return dataset

    def get_raw_train_dataset(self):
        files = tf.data.Dataset.list_files(
            self.tfrecord_path + "/train_TFRecord*", shuffle=200)
        with tf.device('/cpu:0'):
            return self.create_dataset(files)

    # def get_raw_val_dataset(self):
    #     with tf.device('/cpu:0'):
    #         return self.create_dataset(self.val_tfr)
    #
    # def get_raw_test_dataset(self):
    #     with tf.device("/cpu:0"):
    #         return self.create_dataset(self.test_tfr)
    #
    # def get_train_size(self):
    #     return self.train_size
    #
    # def get_val_size(self):
    #     return self.val_size
    #
    # def get_test_size(self):
    #     return self.test_size


# tf.enable_eager_execution()
# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
# sentenceHelper = DatasetManager([DATA_PATH + "/europarl-v7.fr-en.en"],
#                                 [DATA_PATH + "/europarl-v7.fr-en.en"],
#                                 batch_size=16,
#                                 shuffle=100)
# dataset = sentenceHelper.get_raw_train_dataset()
# for i in range(5):
#     import pdb; pdb.set_trace()
#     d = dataset.make_one_shot_iterator()
#     d = d.get_next()
# # # # # a, b, c = sentenceHelper.prepare_data()
# # # # a, b, c = sentenceHelper.post_process()
# # for i, e in enumerate(a):
# #     print(e[0])
# #     print(i)
# #     sentenceHelper.byter.decode(e[0].numpy())
# #     break
#
#
# def dataset_prepross_fn(src, tgt):
#     return (src, tgt), tgt
#
#
# dataset = dataset.map(dataset_prepross_fn, num_parallel_calls=12)
# dataset = dataset.padded_batch(
#     1,
#     padded_shapes=(
#         (
#             tf.TensorShape([None]),  # source vectors of unknown size
#             tf.TensorShape([None]),  # target vectors of unknown size
#         ),
#         tf.TensorShape([None])),
#     drop_remainder=True)
