# encoder=utf.8
import sys
sys.path.insert(0, '/home/vivalavida/batch_data/corpus_fr2eng/')
sys.path.insert(1, '/home/vivalavida/workspace/alpha/transformer_nmt/')
TRAIN_MODE = 'large'
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# TRAIN_MODE = 'test'
import tensorflow as tf
import numpy as np
import time
import os
import train_conf
import model_helper
import bleu_metrics


class Graph():
    def __init__(self,
                 hyperParam,
                 sys_path,
                 vocab_size=24000,
                 keep_prob=0.1,
                 gpu=0):
        self.hp = hyperParam
        self.num_units = self.hp.num_units
        self.sys_path = sys_path
        self.epoch_number = self.hp.epoch_num
        self.lr = self.lr_init = self.hp.lr
        self.learning_warmup = self.hp.learning_warmup
        self.vocab_size = vocab_size
        self.clipping = self.hp.clipping
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.gpu = train_conf.get_available_gpus()
        # if self.gpu > 0:
        #     self.batch_size = int(self.hp.batch_size / self.gpu)
        # else:
        self.batch_size = self.hp.batch_size

    def _check_point_creater(self, model):
        try:
            os.makedirs('training_checkpoints')
        except OSError:
            pass
        checkpoint_dir = './training_checkpoints'

        checkpoint = tf.train.Checkpoint(model=model)
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        ckpt = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=10)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        return ckpt, status

    def _prepocess_dataset(self, data_manager):
        try:
            train_dataset, val_dataset, test_dataset = data_manager.prepare_data(
            )
            if self.gpu > 0:
                # self.train_dataset = self.train_dataset.prefetch(self.gpu * 24)
                val_dataset = val_dataset.prefetch(self.gpu * 4)
                test_dataset = test_dataset.prefetch(self.gpu * 4)
                for i in range(self.gpu):
                    train_dataset = train_dataset.apply(
                        tf.data.experimental.prefetch_to_device(
                            "/device:GPU:" + str(i)))
            else:
                train_dataset = train_dataset.prefetch(2)
                val_dataset = val_dataset.prefetch(2)
                test_dataset = test_dataset.prefetch(2)
            return train_dataset, val_dataset, test_dataset
        except Exception:
            print("Data Manager is not a proper one.")

    def _input_fn(self, dataset):
        iterator = tf.data.Iterator.from_structure(
            output_types=dataset.output_types,
            output_shapes=dataset.output_shapes)
        input_iterater = iterator.make_initializer(dataset)
        X, Y = iterator.get_next()
        return X, Y, input_iterater

    def _tower_fusion_grad(self, model, X, Y, optimizer):
        """
            core part
        """

        def average_gradients(tower_grads):
            average_grads = []
            for grad in zip(*tower_grads):
                grads = []
                for g in grad:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                average_grads.append(grad)
            return average_grads

        loss = []
        grad = []
        if self.gpu > 0:
            mini_batch_x = model_helper.mini_batch(X, self.gpu)
            mini_batch_y = model_helper.mini_batch(Y, self.gpu)
            for index, gpu in enumerate(np.arange(0, self.gpu)):
                with tf.device("/device:GPU:" + str(gpu)):
                    batch_X = mini_batch_x[index]
                    batch_Y = mini_batch_y[index]
                    # self.model.dropout_manager(self.keep_prob)
                    p = model((batch_X, batch_Y), train=True)
                    tower_loss = train_conf.onehot_loss_function(
                        batch_Y, p, self.hp.PAD_ID, vocab_size=self.vocab_size)
                    del p
                    tower_grad, _ = zip(*optimizer.compute_gradients(
                        tower_loss, model.variables))
                    grad.append(tower_grad)
                    loss.append(tower_loss)
                    del tower_grad, tower_loss
            loss = tf.reduce_mean(loss, axis=0)
            grad = average_gradients(grad)
        else:
            # self.model.dropout_manager(self.keep_prob)
            pred = model((X, Y), train=True)
            loss = train_conf.onehot_loss_function(
                Y, pred, self.hp.PAD_ID, vocab_size=self.vocab_size)
            grad, _ = zip(*optimizer.compute_gradients(loss, model.variables))
        grad, _ = tf.clip_by_global_norm(grad, self.clipping)
        # train_op = self.optimizer.apply_gradients(
        #     zip(grad, self.model.variables), self.global_step)
        train_op = optimizer.apply_gradients(
            zip(grad, model.variables), self.global_step)
        return loss, train_op

    def _epoch_train(self,
                     model,
                     sess,
                     loss,
                     dataset_iterator,
                     train_op,
                     summarise_op,
                     ckpt,
                     writer,
                     data_manager=None,
                     epoch=0):
        sess.run(dataset_iterator)
        batch_num = 0.0
        total_loss = 0.0
        try:
            while True:
                start_time = time.time()
                batch_loss, _ = sess.run([loss, train_op])
                step = self.global_step.eval()
                total_loss += batch_loss
                batch_num += 1
                total = total_loss / batch_num
                train_summarise = sess.run(
                    summarise_op,
                    feed_dict={
                        self.summary_bleu: 0,
                        self.summary_learning_rate: self.lr,
                        self.summary_total_loss: total,
                        self.summary_batch_loss: batch_loss,
                    })
                writer.add_summary(train_summarise, global_step=step)
                print('Training_step %d' % step)
                print("Batch loss %s" % batch_loss)
                print('Average loss at epoch %d:%s' % (epoch, total))
                print('Training cost: %s seconds' % (time.time() - start_time))
                if step % 5000 == 0:
                    ckpt.save()
                try:
                    if step % 100 == 0:
                        data_manager.train_percentage(
                            batch_num * self.batch_size)
                except Exception:
                    pass
        except tf.errors.OutOfRangeError:
            print("epoch %d training finished" % epoch)
            print("epoch %d: trained %d batches" % (epoch, batch_num))
            pass
        model.save_weights(self.sys_path + '/model/model_weights')

    def _tower_fusion_pred(self, model, X):
        pred = []
        if self.gpu > 0:
            mini_batch_x = model_helper.mini_batch(X, self.gpu)
            # mini_batch_y = model_helper.mini_batch(self.Y, self.gpu)
            for index, gpu in enumerate(np.arange(0, self.gpu)):
                with tf.device("/device:GPU:" + str(gpu)):
                    X = mini_batch_x[index]
                    p = model(X, train=False)
                    pred.append(p)
        else:
            p = model(X, train=False)
            pred.append(p)
        return pred

    def _epoch_evaluation(self,
                          sess,
                          Y,
                          pred,
                          val_iterator,
                          summarise_op,
                          writer,
                          data_manager,
                          epoch=0):
        sess.run(val_iterator)
        batch_size = 0
        self.num_eval = 0
        batch_bleu = []
        visual_true = []
        visual_pred = []
        # bleu_summarise_op = train_conf.train_summarise(bleu=self.summary_bleu)
        try:
            while True:
                self.num_eval += 1
                pred_eval = sess.run(pred)
                if self.gpu > 0:
                    eval_y = model_helper.mini_batch(Y, self.gpu)
                    batch_size += 1 * self.gpu
                else:
                    eval_y = [Y]
                    batch_size += 1
                with tf.device("/device:CPU:0"):
                    for i, p in enumerate(pred_eval):
                        y = eval_y[i].eval().tolist()
                        p = p.tolist()
                        for k, v in enumerate(p):
                            y[k] = data_manager.decode(
                                bleu_metrics.token_trim(y[k], self.hp.EOS_ID))
                            p[k] = data_manager.decode(
                                bleu_metrics.token_trim(p[k], self.hp.EOS_ID))
                        score, _ = bleu_metrics.bleu_score(
                            p, y, eos_id=self.hp.EOS_ID)
                        # score, _ = bleu_metrics.bleu_score(pred_eval[i], Y[i])
                        batch_bleu.append(score * 100.0)
                self.final_bleu = tf.reduce_mean(batch_bleu).eval()
                print("BLEU in  epoch %d at step %d is %s" %
                      (epoch, self.num_eval, self.final_bleu))
                summarise = sess.run(
                    summarise_op,
                    feed_dict={
                        self.summary_bleu: self.final_bleu,
                        self.summary_learning_rate: self.lr,
                        self.summary_total_loss: 0.0,
                        self.summary_batch_loss: 0.0,
                    })
                writer.add_summary(summarise, global_step=self.num_eval)
                r = np.random.randint(0, int(self.batch_size / 2 - 1))
                visual_true.append(
                    bleu_metrics.token_trim(Y[0][r].eval().tolist(),
                                            self.hp.EOS_ID))
                visual_pred.append(
                    bleu_metrics.token_trim(pred_eval[0][r], self.hp.EOS_ID))
        except tf.errors.OutOfRangeError:
            if self.final_bleu > self.best_bleu:
                self.best_bleu = self.final_bleu
                self.best_epoch = epoch
            print("The best BLEU score is %f in epoch %d" % (self.best_bleu,
                                                             epoch))

        eval_visual = []
        for k, v in enumerate(visual_pred):
            true_words = data_manager.decode(visual_true[k])
            pred_words = data_manager.decode(visual_pred[k])
            eval_visual.append(
                tf.summary.text('true_pred',
                                tf.convert_to_tensor([true_words,
                                                      pred_words])))
        # print(visual_attention)
        eval_visual = tf.summary.merge(eval_visual)
        eval_visual = sess.run(eval_visual)
        writer.add_summary(eval_visual, global_step=self.num_eval)

        print("epoch %d evaluation finished" % epoch)
        print("epoch %d: evaluated %d" % (epoch, batch_size))

    def graph(self, model, data_manager):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        with tf.Session(config=config) as sess:
            tf.keras.backend.set_session(sess)
            ########################################################
            self.summary_total_loss = tf.placeholder(
                tf.float32, shape=None, name="summary_total_loss")
            self.summary_batch_loss = tf.placeholder(
                tf.float32, shape=None, name="summary_batch_loss")
            self.summary_learning_rate = tf.placeholder(
                tf.float32, shape=None, name="summary_learning_rate")
            self.summary_bleu = tf.placeholder(
                tf.float32, shape=None, name="summary_bleu")
            self.global_step = tf.train.get_or_create_global_step()
            self.lr = train_conf.get_learning_rate(
                self.lr_init,
                self.num_units,
                learning_rate_warmup_steps=self.hp.learning_warmup)
            self.num_eval = 0
            self.best_bleu = 0
            self.best_epoch = 0
            ###########################################################
            ckpt, status = self._check_point_creater(model)
            train_dataset, val_dataset, test_dataset = self._prepocess_dataset(
                data_manager=data_manager)
            X_train, Y_train, train_iterator = self._input_fn(train_dataset)
            X_val, Y_val, val_iterator = self._input_fn(val_dataset)
            optimizer = train_conf.optimizer(lr=self.lr)
            loss, train_op = self._tower_fusion_grad(model, X_train, Y_train,
                                                     optimizer)
            model.summary()
            pred = self._tower_fusion_pred(model, X_val)

            writer = tf.summary.FileWriter(self.sys_path + '/graphs/nmt',
                                           sess.graph)
            train_summarise_op = train_conf.train_summarise(
                self.summary_batch_loss, self.summary_total_loss,
                self.summary_learning_rate)
            bleu_summarise_op = train_conf.bleu_summarise(self.summary_bleu)

            sess.run(tf.global_variables_initializer())
            status.initialize_or_restore(sess)
            for epoch in range(1, self.epoch_number):
                print('Starting epoch %d' % epoch)
                self._epoch_train(model, sess, loss, train_iterator, train_op,
                                  train_summarise_op, ckpt, writer,
                                  data_manager, epoch)
                print('starting evaluation')
                self._epoch_evaluation(sess, Y_val, pred, val_iterator,
                                       bleu_summarise_op, writer, data_manager,
                                       epoch)
            print('graph sucessed')


# ###########################################
import hyperParam
import core_Transformer_model
import core_dataset_generator
# import test_trainer
# import graph_trainer
# set_up
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"
hp = hyperParam.HyperParam(TRAIN_MODE)
data_manager = core_dataset_generator.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    PAD_ID=hp.PAD_ID,
    EOS_ID=hp.EOS_ID,
    # shuffle=hp.data_shuffle,
    shuffle=hp.data_shuffle,
    max_length=hp.max_sequence_length)
# dataset_train_val_test = data_manager.prepare_data()
# src_vocabulary, src_ids2word, tgt_vocabulary, tgt_ids2word = data_manager.prepare_vocabulary(
# )
gpu = train_conf.get_available_gpus()

vocabulary_size = 24000
model = core_Transformer_model.Daedalus(
    max_seq_len=hp.max_sequence_length,
    vocabulary_size=vocabulary_size,
    embedding_size=hp.embedding_size,
    batch_size=hp.batch_size / (gpu if gpu > 0 else 1),
    num_units=hp.num_units,
    num_heads=hp.num_heads,
    num_encoder_layers=hp.num_encoder_layers,
    num_decoder_layers=hp.num_decoder_layers,
    dropout=hp.dropout,
    eos_id=hp.EOS_ID,
    pad_id=hp.PAD_ID)
# test = test_trainer.Trainer(model=model, dataset=dataset_train_val_test)
# test.train()
large_train = Graph(
    vocab_size=vocabulary_size,
    sys_path=SYS_PATH,
    hyperParam=hp)
large_train.graph(model, data_manager)
