# encoding=utf-8
import sys
from hyper_and_conf import hyper_param as hyperParam
from hyper_and_conf import hyper_train
import core_lip_main
import core_data_SRCandTGT
from tensorflow.python.client import device_lib
import tensorflow as tf

DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
src_data_path = [DATA_PATH + "/corpus/europarl-v7.fr-en.en"]
tgt_data_path = [DATA_PATH + "/corpus/europarl-v7.fr-en.en"]
TFRECORD = '/home/vivalavida/massive_data/lip_reading_TFRecord/tfrecord_word'
# TFRECORD = '/Users/barid/Documents/workspace/batch_data/lip_data_TFRecord'

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'CPU'])


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


gpu = get_available_gpus()
TRAIN_MODE = 'large' if gpu > 0 else 'test'
hp = hyperParam.HyperParam(TRAIN_MODE, gpu=get_available_gpus())
PAD_ID = tf.cast(hp.PAD_ID, tf.int64)
with tf.device("/cpu:0"):
    daedalus = core_lip_main.Daedalus(hp)
    # if tf.gfile.Exists('pre_train/vgg16_pre_all'):
    #     vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    # else:
    #     vgg16 = tf.keras.applications.vgg16.VGG16(
    #         include_top=True, weights='imagenet')
    data_manager = core_data_SRCandTGT.DatasetManager(
        src_data_path,
        tgt_data_path,
        batch_size=hp.batch_size,
        PAD_ID=hp.PAD_ID,
        EOS_ID=hp.EOS_ID,
        # shuffle=hp.data_shuffle,
        shuffle=hp.data_shuffle,
        max_length=hp.max_sequence_length,
        tfrecord_path=TFRECORD
        )

# train_dataset, val_dataset, test_dataset = data_manager.prepare_data()


def get_hp():
    return hp


def backend_config():
    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    # # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.999
    config.allow_soft_placement = True

    return config


def input_fn(flag="TRAIN"):
    with tf.device('/cpu:0'):
        if flag == "VAL":
            dataset = data_manager.get_raw_val_dataset()
        if flag == "TEST":
            dataset = data_manager.get_raw_test_dataset()
        if flag == "TRAIN":
            dataset = data_manager.get_raw_train_dataset()
        else:
            assert ("data error")
        # repeat once in case tf.keras.fit out range error
        # dataset = dataset.apply(
        #     tf.data.experimental.shuffle_and_repeat(hp.data_shuffle, 1))
        return dataset


def pad_sample(dataset, seq2seq=False):
    if seq2seq:
        dataset = dataset.map(
            dataset_prepross_fn,
            num_parallel_calls=12 * get_available_gpus()
            if get_available_gpus() > 0 else 1)
        dataset = dataset.padded_batch(
            hp.batch_size,
            padded_shapes=(
                (
                    tf.TensorShape([None,
                                    None]),  # source vectors of unknown size
                    tf.TensorShape([None]),  # target vectors of unknown size
                ),
                tf.TensorShape([None])),
            padding_values=(
                (
                    tf.cast(
                        hp.PAD_ID, tf.float32
                    ),  # source vectors padded on the right with src_eos_id
                    PAD_ID
                    # target vectors padded on the right with tgt_eos_id
                ),
                PAD_ID),
            drop_remainder=True)

    else:

        dataset = dataset.padded_batch(
            hp.batch_size,
            padded_shapes=(
                tf.TensorShape([None, None]),  # source vectors of unknown size
                tf.TensorShape([None]),  # target vectors of unknown size
            ),
            padding_values=(
                tf.cast(
                    hp.PAD_ID, tf.float64
                ),  # source vectors padded on the right with src_eos_id
                PAD_ID
                # target vectors padded on the right with tgt_eos_id
            ),
            drop_remainder=True)
    if gpu > 0:
        for i in range(gpu):
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device("/GPU:" + str(i)))
    else:
        dataset = dataset.prefetch(gpu * 8)

    return dataset


def get_train_step():
    return data_manager.get_train_size() // hp.batch_size


def get_val_step():
    return data_manager.get_val_size() // hp.batch_size


def get_test_step():
    return data_manager.get_test_size() // hp.batch_size


def dataset_prepross_fn(src, tgt):
    return (src, tgt), tgt


def train_input(debug=False):
    with tf.device('/cpu:0'):
        dataset = input_fn('TRAIN')
        dataset = pad_sample(dataset, seq2seq=True)
        if debug:
            return dataset
        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()

    return x, y


def val_input():
    with tf.device('/cpu:0'):
        dataset = input_fn("VAL")
        dataset = pad_sample(dataset, seq2seq=True)
        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()

    return x, y


def model_structure(training=False):
    with tf.device('/cpu:0'):
        img_input = tf.keras.layers.Input(
            shape=[None, 25088], dtype=tf.float32)
        tgt_input = tf.keras.layers.Input(
            shape=[None], dtype=tf.int64, name='tgt_input')
        output = daedalus((img_input, tgt_input), training=training)
        model = tf.keras.models.Model(
            inputs=(img_input, tgt_input), outputs=output)
    # if multi_gpu and gpu > 0:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=gpu)
    return model


def train_model():
    return model_structure(True)


def test_model():
    return model_structure(False)


def get_metrics():
    # evaluation metrics
    # bleu = hyper_train.Approx_BLEU_Metrics(eos_id=hp.EOS_ID)
    accuracy = hyper_train.Padded_Accuracy(hp.PAD_ID)
    accuracy_topk = hyper_train.Padded_Accuracy_topk(k=10, pad_id=hp.PAD_ID)
    return [accuracy, accuracy_topk]


def get_optimizer():
    return tf.keras.optimizers.Adam()


def get_loss():
    return hyper_train.Onehot_CrossEntropy(hp.vocabulary_size)


def get_callbacks():
    LRschedule = hyper_train.Dynamic_LearningRate(hp.lr, hp.num_units,
                                                  hp.learning_warmup)
    TFboard = tf.keras.callbacks.TensorBoard(
        log_dir=hp.model_summary_dir,
        # histogram_freq=10,
        write_images=True,
        update_freq=10)

    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(hp.model_checkpoint_dir +
                                                      '/cp.ckpt')
    # BatchTime = hyper_train.BatchTiming()
    # SamplesPerSec = hyper_train.SamplesPerSec(hp.batch_size)
    # if get_available_gpus() > 0:
    #     CudaProfile = hyper_train.CudaProfile()
    #
    #     return [
    #         LRschedule, TFboard, TFchechpoint, BatchTime, SamplesPerSec,
    #         CudaProfile
    #     ]
    # else:
    return [LRschedule, TFboard, TFchechpoint]


def make_parallel(model, gpu_count, ps_device=None, training=True):
    if gpu_count <= 1:
        return model

    if ps_device is None:
        ps_device = '/gpu:0'

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = tf.keras.layers.Lambda(
                        get_slice,
                        output_shape=input_shape,
                        arguments={
                            'idx': i,
                            'parts': gpu_count
                        })(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on parameter server
    with tf.device(ps_device):
        merged = []
        for outputs in outputs_all:
            merged.append(tf.keras.layers.concatenate(outputs, axis=0))

        return tf.keras.Model(inputs=model.inputs, outputs=merged)


# model = train_model()
# model.summary()
