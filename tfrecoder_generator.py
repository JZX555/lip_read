# encoding=utf8
import tensorflow as tf
tf.enable_eager_execution()
import data_image_helper
import data_file_helper as fh
import core_data_SRCandTGT
import six
import os
import time
import visualization
import core_lip_main
from multiprocessing import Process, Pool
from threading import Thread
# import visualization
cwd = os.getcwd()
CORPUS_PATH = cwd + '/corpus/europarl-v7.fr-en.en'
print(CORPUS_PATH)
# ROOT_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data'
# TFRecord_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data_TFRecord'
ROOT_PATH = '/home/vivalavida/massive_data/lip_reading_data/word_level_lrw'
TFRecord_PATH = '/home/vivalavida/massive_data/lip_reading_TFRecord/tfrecodr_word'
image_parser = data_image_helper.data_image_helper(detector='./cascades/')
text_parser = core_data_SRCandTGT.DatasetManager(
    [CORPUS_PATH],
    [CORPUS_PATH],
)

BUFFER_SIZE = 200


def get_vgg():
    with tf.device("/cpu:0"):
        if tf.gfile.Exists('pre_train/vgg16_pre_all'):
            vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
        else:
            vgg16 = tf.keras.applications.vgg16.VGG16(
                include_top=True, weights='imagenet')
        return vgg16


vgg16 = get_vgg()
vgg16_flatten = vgg16.get_layer('flatten')
vgg16_output = vgg16_flatten.output
vgg16.input
model = tf.keras.Model(vgg16.input, vgg16_output)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def word_reader(path):
    video, _, word = fh.read_file(path)
    total = len(video)
    # raw_data = []
    tf.logging.info("Total train samples:{}".format(len(video)))
    for k, v in enumerate(video):
        v_data = image_parser.get_raw_dataset(paths=v)
        v_data = tf.reshape(model(v_data), [-1])
        w = text_parser.encode(word[k])
        # tf.logging.info("Train sample:{}".format(k))
        visualization.percent(k, total)
        yield (v_data, w)


def tfrecord_generater(record_dir, raw_data, index):

    with tf.device("/cpu:0"):
        num_train = 0
        # num_test = 0
        prefix_train = record_dir + "/train_TFRecord_"

        # prefix_test = record_dir + "/test_TFRecord_"

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

        def dict_to_example(img, txt):
            """Converts a dictionary of string->int to a tf.Example."""
            features = {}
            features['img'] = _float_feature(img)
            features['text'] = _int64_feature(txt)
            return tf.train.Example(
                features=tf.train.Features(feature=features))

        checker = -1
        shard = 0
        options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)
        for k, v in enumerate(raw_data):
            v_data = image_parser.get_raw_dataset(path=v[0])
            # import pdb; pdb.set_trace()
            if len(v_data.shape) == 4:
                v_data = tf.reshape(model(v_data), [-1])
                w = text_parser.encode(v[1])
                if checker == shard:
                    pass
                else:
                    shard = k // BUFFER_SIZE
                    train_writers = tf.python_io.TFRecordWriter(
                        prefix_train + str(index * 10000 + shard),
                        options=options)
                example = dict_to_example(
                    v_data.numpy().tolist(),
                    w,
                )
                train_writers.write(example.SerializeToString())
                checker = int((k + 1) / BUFFER_SIZE)
                num_train += 1
                if num_train % BUFFER_SIZE == 0:
                    tf.logging.info("Train samples are : {}".format(num_train))
                if checker > shard:
                    print("TFRecord {} is completed.".format(prefix_train +
                                                             str(shard)))
                    # print("Test samples are : {}".format(num_test))
                    train_writers.close()

            visualization.percent(k, len(raw_data))


# raw_data = word_reader(ROOT_PATH)
# if __name__ == '__main__':
P = 8
t = time.time()
video, _, word = fh.read_file(ROOT_PATH)
worker = len(video) // P
raw_data = list(zip(video, word))
tfrecord_generater(TFRecord_PATH, raw_data, 1)
# p = Pool(P)
# i = 1
# r = p.map_async(tfrecord_generater,(TFRecord_PATH, raw_data[i * worker:(i + 1) * worker], i))
# r.wait()
# p.close()
# p.join()
# processes = []
# coord = tf.train.Coordinator()
# with tf.device("/cpu:0"):
#     for i in range(P):
#         t = Process(
#             target=tfrecord_generater,
#             args=(TFRecord_PATH, raw_data[i * worker:(i + 1) * worker], i))
#         t.start()
#         t.join()
    # processes.append(t)
# coord.join(processes)
# # for one_process in processes:
# #     one_process.join()
# print("Done!")
# p.start()
# p.join()
# files = tf.data.Dataset.list_files(TFRecord_PATH + "/train_TFRecord_*")
# dataset = tf.data.TFRecordDataset(
#     filenames=files, compression_type='GZIP', buffer_size=BUFFER_SIZE)
#
#
# # test = dataset.make_one_shot_iterator()
# #
# # img,text = test.get_next()
# def _parse_function(example_proto):
#     # Parse the input tf.Example proto using the dictionary above.
#     feature_description = {
#         'text': tf.VarLenFeature(tf.int64),
#         'img': tf.VarLenFeature(tf.float32),
#     }
#     parsed = tf.parse_single_example(example_proto, feature_description)
#     img = tf.sparse_tensor_to_dense(parsed["img"])
#     text = tf.sparse_tensor_to_dense(parsed["text"])
#     return img, text
#
#
# dataset = dataset.map(_parse_function)
#
# for d in dataset:
#     # print(d['img'].values)
#     # print(tf.reshape(d['img'].values, [-1,224,224,3]))
#     print(d[0])
#     break
