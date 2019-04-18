# encoding=utf8
import tensorflow as tf
import data_image_helper
import data_file_helper as fh
import core_data_SRCandTGT
import six
import os
tf.enable_eager_execution()
# import visualization
cwd = os.getcwd()
CORPUS_PATH = cwd + '/corpus/europarl-v7.fr-en.en'
print(CORPUS_PATH)
ROOT_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data'
WORD_ROOT_PATH = '/home/vivalavida/massive_data/lip_reading_data/word_level_lrw'
TFRecord_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data_TFRecord'
WORD_TFRecord_PATH = '/home/vivalavida/massive_data/lip_reading_TFRecord/word'
image_parser = data_image_helper.data_image_helper(detector='./cascades/')
text_parser = core_data_SRCandTGT.DatasetManager([CORPUS_PATH], [CORPUS_PATH],
                                                 tf_recoder=False)


def word_reader(path):
    video, _, word = fh.read_file(path)
    # raw_data = []
    print(video)
    for k, v in enumerate(video):
        # visualization.percent(k, len(video))
        v_data, _ = image_parser.prepare_data(
            paths=[v], batch_size=1, raw=True)
        # v_data = v_data.tolist()
        w = text_parser.encode(word[k])
        # raw_data.append((v_data, w))
        yield (v_data, w)


def tfrecord_generater(record_dir, raw_data):
    num_train = 0
    num_test = 0
    prefix_train = record_dir + "/train_TFRecord_"
    prefix_test = record_dir + "/test_TFRecord_"

    def all_exist(filepaths):
        """Returns true if all files in the list exist."""
        for fname in filepaths:
            if not tf.gfile.Exists(fname):
                return False
        return True

    def txt_line_iterator(path):
        with tf.gfile.Open(path) as f:
            for line in f:
                yield line.strip()

    def dict_to_example(dictionary):
        """Converts a dictionary of string->int to a tf.Example."""
        features = {}
        for k, v in six.iteritems(dictionary):
            if k == 'img':
                features[k] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=v))
            else:
                features[k] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=v))
        return tf.train.Example(features=tf.train.Features(feature=features))

    old_shard = -1
    for index, data in enumerate(raw_data):
        shard = int(index / 60)
        if old_shard == shard:
            pass
        else:
            train_writers = tf.python_io.TFRecordWriter(prefix_train +
                                                        str(shard))
            # test_writers = tf.python_io.TFRecordWriter(prefix_test +
            #                                            str(shard))
        # for k, v in enumerate(data):
        example = dict_to_example({
            "img": tf.reshape(data[0][0].tolist(), [-1]),
            "text": data[1],
            "org_shape": data[0][0].shape
        })
        train_writers.write(example.SerializeToString())
        # if index % 70 > 0:
        #     test_writers.write(example.SerializeToString())
        #     num_test += 1
        # else:
        #     train_writers.write(example.SerializeToString())
        #     num_train += 1
        checker = int((index + 1) / 60)
        num_train += 1
        print("Train samples are : {}".format(num_train))
        if checker > shard:
            print(
                "TFRecord {} is completed.".format(prefix_train + str(shard)))
            # print("Test samples are : {}".format(num_test))
            train_writers.close()
            # test_writers.close()
        old_shard = shard
    # train_tfr = tfrecorder(train_path, prefix_train)
    # print("Train TFRecord generated {}".format(self.train_tfr))
    # test_tfr = tfrecorder(test_path, prefix_test)
    # print("Test TFRecord generated {}".format(self.test_tfr))
    # print("All TFRecord generated")


raw_data = word_reader(WORD_ROOT_PATH)
tfrecord_generater(WORD_TFRecord_PATH, raw_data)
