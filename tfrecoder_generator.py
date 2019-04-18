# encoding=utf8
import tensorflow as tf
import data_image_helper
import data_file_helper as fh
import core_data_SRCandTGT
import six
import os
tf.enable_eager_execution()
cwd = os.getcwd()
CORPUS_PATH = cwd + '/corpus/europarl-v7.fr-en.en'
print(CORPUS_PATH)
ROOT_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data'
image_parser = data_image_helper.data_image_helper(detector='./cascades/')
text_parser = core_data_SRCandTGT.DatasetManager([CORPUS_PATH], [CORPUS_PATH],tf_recoder=False)


def data_reader(path):
    video, txt = fh.read_file(path)
    raw_data = []
    try:
        for k, v in enumerate(video):
            v_data, _ = image_parser.prepare_data(
                paths=v, batch_size=1, raw=True)
            t_data = text_parser.one_file_decoder(txt[k])
            raw_data.append((v_data, t_data))
    except Exception:
        pass
    return raw_data


def tfrecord_generater(self):
    with tf.device("/gpu:0"):
        prefix_train = "./data/train_TFRecord_"
        prefix_val = "./data/val_TFRecord_"
        prefix_test = "./data/test_TFRecord_"
        if len(self.cross_val) == 2:
            # train_por = self.cross_val[0]
            val_por = 0
            test_por = self.test_val[1]
        else:
            # train_por = self.cross_val[0]
            val_por = self.cross_val[1]
            test_por = self.cross_val[2]

        train_path, test_path, val_path = self.cross_validation(
            self.source_data_path,
            self.target_data_path,
            validation=val_por,
            test=test_por)

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
                features[k] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=v))
            return tf.train.Example(
                features=tf.train.Features(feature=features))

        def tfrecorder(path_list, prefix):
            if len(path_list) > 0:
                tfr_path = [prefix + str(k) for k, _ in enumerate(path_list)]
            if all_exist(tfr_path):
                print("TFRecord already exists!")
            else:
                writers = [
                    tf.python_io.TFRecordWriter(fname) for fname in tfr_path
                ]
                for i, _ in enumerate(writers):
                    for k, v in enumerate(txt_line_iterator(path_list[i])):
                        src, tgt = v.split(self.byte_token)
                        example = dict_to_example({
                            "src":
                            self.byter.encode(src, add_eos=True),
                            "tgt":
                            self.byter.encode(tgt, add_eos=True)
                        })
                        writers[i].write(example.SerializeToString())

                for writer in writers:
                    writer.close()
            return tfr_path

    self.train_tfr = tfrecorder(train_path, prefix_train)
    print("Train TFRecord generated {}".format(self.train_tfr))
    self.val_tfr = tfrecorder(val_path, prefix_val)
    print("Val TFRecord generated {}".format(self.val_tfr))
    self.test_tfr = tfrecorder(test_path, prefix_test)
    print("Test TFRecord generated {}".format(self.test_tfr))
    print("All TFRecord generated")


data = data_reader()
