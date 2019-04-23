# encoding=utf-8
# author barid
import tensorflow as tf
# tf.enable_eager_execution()
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


def main():
    config = init.backend_config()
    tf.keras.backend.set_session(tf.Session(config=config))
    gpu = init.get_available_gpus()
    # set session config
    metrics = init.get_metrics()
    with tf.device("/cpu:0"):
        train_x, train_y = init.train_input()
        # val_x, val_y = init.val_input()
        # train_model = init.daedalus
        train_model = init.train_model()
        # with strategy.scope():
        hp = init.get_hp()
        # dataset
        # step
        train_step = 5000
        # val_step = init.get_val_step()
        # get train model

        # optimizer
        optimizer = init.get_optimizer()

        # loss function
        loss = init.get_loss()
        # evaluation metrics

        # callbacks

        callbacks = init.get_callbacks()
        # import pdb; pdb.set_trace()
        # test = train_model(train_x,training=True)
        if gpu > 0:
            # train_model = tf.keras.utils.multi_gpu_model(
            #     train_model, gpu, cpu_merge=False)
            train_model = init.make_parallel(train_model, gpu, '/gpu:1')
            # staging_area_callback = hyper_train.StagingAreaCallback(
            #     train_x, train_y, hp.batch_size)
            # callbacks.append(staging_area_callback)
            # train_model.compile(
            #     optimizer=optimizer,
            #     loss=loss,
            #     metrics=metrics,
            #     target_tensors=[staging_area_callback.target_tensor],
            #     fetches=staging_area_callback.extra_ops)
            train_model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                target_tensors=train_y)
        else:
            train_model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                target_tensors=tf.placeholder(
                    dtype=tf.int64, shape=[None, None]))
        # train_model.summary()
        # main
        train_model.fit(
            x=train_x,
            y=train_y,
            epochs=hp.epoch_num,
            steps_per_epoch=train_step,
            verbose=1,
            # validation_data=(val_x, val_y),
            # validation_steps=val_step,
            callbacks=callbacks,
            max_queue_size=8 * (gpu if gpu > 0 else 1),
            use_multiprocessing=True,
            workers=0)

        train_model.save_weights("model_weights")


if __name__ == '__main__':
    main()
