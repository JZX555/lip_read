import tensorflow as tf
import tensorflow.contrib.eager as eager
import core_lip_main
from hyper_and_conf import hyper_param
import core_data_SRCandTGT
import numpy as np
import data_image_helper

eager.enable_eager_execution()

CORPUS_PATH = '../europarl-v7.fr-en.en'

def get_vgg():
    # with tf.device("/cpu:0"):
    if tf.gfile.Exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet')
    return vgg16

class text_helper():
    def __init__(self, 
                 corpus_path = CORPUS_PATH,
                 mode = 'test'):
        self.hp = hyper_param.HyperParam(mode = mode)

        self.text_parser = core_data_SRCandTGT.DatasetManager(
            [corpus_path],
            [corpus_path],
        )

        self.helper = data_image_helper.data_image_helper(detector='./cascades/')
        self.vgg16 = get_vgg()
        self.vgg16_flatten = self.vgg16.get_layer('flatten')
        self.vgg16_output = self.vgg16_flatten.output
        self.vgg16.input
        self.vgg_model = tf.keras.Model(self.vgg16.input, self.vgg16_output)

        self.daedalus = core_lip_main.Daedalus(self.hp)

    def get_text(self, path):
        video = self.helper.get_raw_dataset(path)

        out = self.vgg_model(video)
        out = tf.expand_dims(out, 0)

        ids = self.daedalus(out)
        word = self.text_parser.decode(ids[0])
        
        return word
        
if __name__ == '__main__':
    test = '~/massive_data/lip_reading_data/word_level_lrw/ABOUT/test/ABOUT_00003.mp4'
    model = text_helper()
    word = model.get_text(test)
    print(word)