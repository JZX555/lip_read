import tensorflow as tf
import tensorflow.contrib as tf_contrib
import data_image_helper
import data_sentence_helper
import core_lipread_model as lipread

# tf_contrib.eager.enable_eager_execution()

BATCH_SIZE = 1
sentence = data_sentence_helper.SentenceHelper('D:/lip_data/ABOUT/train/ABOUT_00003.txt', 
                                               'D:/lip_data/ABOUT/train/ABOUT_00003.txt',
                                               batch_size = BATCH_SIZE)
tgt_dataset = sentence.prepare_data()

img_reader = data_image_helper.data_image_helper('')
src_dataset, img = img_reader.prepare_data(paths = ['D:/lip_data/ABOUT/train/ABOUT_00003.mp4'], batch_size = BATCH_SIZE)

dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))


src_vocabulary, tgt_vocabulary, src_ids2word, tgt_ids2word = sentence.prepare_vocabulary()
src_vocabulary_size = len(src_vocabulary)
tgt_vocabulary_size = len(tgt_vocabulary)
daedalus = lipread.Daedalus(src_vocabulary_size = src_vocabulary_size,
                        tgt_vocabulary_size = tgt_vocabulary_size,
                        batch_size = BATCH_SIZE,
                        embed_size = 10,
                        num_units = 10,
                        backforward = True,
                        eager = True)

for (i, ((src_input, src_length), ((_, tgt_in, __, tgt_length), tgt_out))) in enumerate(dataset):
    print(src_input.shape)
    print(tgt_length)
    inputs = (src_input, tgt_in, src_length, tgt_length)
    logits = daedalus(inputs)
    h, f = daedalus.get_states()
    v, v_o, v_h = daedalus.get_VGG()
    print(logits)
    print(h.shape)
    print(f.shape)
    print(v_o.shape)
    print(v_h[0].shape)
    print(tf.argmax(logits, -1))