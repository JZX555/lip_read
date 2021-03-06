# encode=utf8

# author barid
import tensorflow as tf
from hyper_and_conf import conf_fn
tf.enable_eager_execution()
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd + '/corpus')
sys.path.insert(1, cwd)
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# device = ["/device:CPU:0", "/device:GPU:0", "/device:GPU:1"]
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
# src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
# tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"

import core_model_initializer as init

# def main():
config = init.backend_config()
tf.keras.backend.set_session(tf.Session(config=config))
gpu = init.get_available_gpus()
# set session config
metrics = init.get_metrics()
with tf.device("/cpu:0"):
    train_x, train_y = init.train_input()
    X_Y = init.train_input(True)
    # val_x, val_y = init.val_input()
    # train_model = init.train_model()
    train_model = init.daedalus
    # with strategy.scope():
    hp = init.get_hp()
    # dataset
    # step
    train_step = 500
    # val_step = init.get_val_step()
    # get train model

    # optimizer
    optimizer = init.get_optimizer()

    # loss function
    # loss = init.get_loss()
    # evaluation metrics

    # callbacks

    callbacks = init.get_callbacks()
    # import pdb; pdb.set_trace()
    # test = train_model(train_x,training=True)
    # if gpu > 0:
    #     # train_model = tf.keras.utils.multi_gpu_model(
    #     #     train_model, gpu, cpu_merge=False)
    #     train_model = init.make_parallel(train_model, gpu, '/gpu:1')
    #     # staging_area_callback = hyper_train.StagingAreaCallback(
    #     #     train_x, train_y, hp.batch_size)
    #     # callbacks.append(staging_area_callback)
    #     # train_model.compile(
    #     #     optimizer=optimizer,
    #     #     loss=loss,
    #     #     metrics=metrics,
    #     #     target_tensors=[staging_area_callback.target_tensor],
    #     #     fetches=staging_area_callback.extra_ops)
    #     train_model.compile(
    #         optimizer=optimizer,
    #         loss=loss,
    #         metrics=metrics,
    #         target_tensors=train_y)
    # else:
    #     train_model.compile(
    #         optimizer=optimizer,
    #         loss=loss,
    #         metrics=metrics,
    #         target_tensors=train_y)
    # train_model.summary()
    optimizer = tf.train.AdamOptimizer()
    with tf.GradientTape() as tape:
        for index, (x,y) in enumerate(X_Y):
            logits = train_model(x, training=True)
            import pdb;pdb.set_trace()
            loss_v = conf_fn.onehot_loss_function(y, logits, vocab_size=12000)
            variables = train_model.trainable_variables
            grads = tape.gradient(loss_v, variables)
            optimizer.apply_gradients(zip(grads, variables),
                            global_step=tf.train.get_or_create_global_step())
