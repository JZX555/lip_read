# encoding=utf-8
import os


class HyperParam:
    def __init__(self,
                 mode,
                 gpu=0,
                 vocab=12000,
                 UNK_ID=0,
                 SOS_ID=0,
                 EOS_ID=1,
                 PAD_ID=0,
                 MASK_ID=2):
        self.gpu = gpu
        self.UNK_ID = UNK_ID
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.MASK_ID = MASK_ID
        self.model_summary_dir = "model_summary"
        self.model_weights_dir = "model_weights"
        self.model_checkpoint_dir = "model_checkpoint"
        try:
            os.makedirs(self.model_weights_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_checkpoint_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_summary_dir)
        except OSError:
            pass

        self.vocabulary_size = vocab

        if mode == 'test':
            self.test()
        if mode == 'small':
            self.small()
        if mode == 'large':
            self.large()

    def test(self,
             embedding_size=16,
             batch_size=8,
             epoch_num=5,
             num_units=16,
             num_heads=2,
             num_encoder_layers=2,
             num_decoder_layers=2,
             max_sequence_length=150,
             epoch=1,
             lr=2,
             clipping=5,
             inference_length=150,
             data_shuffle=100000,
             learning_warmup=1000,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup

    def small(self,
              embedding_size=16,
              batch_size=16,
              epoch_num=10,
              num_units=16,
              num_heads=2,
              num_encoder_layers=2,
              num_decoder_layers=2,
              max_sequence_length=150,
              epoch=1,
              lr=0.001,
              clipping=5,
              inference_length=150,
              data_shuffle=100,
              dropout=0.4,
              learning_warmup=10000):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup

    def large(self,
              embedding_size=64 * 16,
              batch_size=23,
              epoch_num=30,
              num_units=64 * 16,
              num_heads=16,
              num_encoder_layers=6,
              num_decoder_layers=6,
              max_sequence_length=50,
              epoch=20,
              lr=2,
              clipping=5,
              inference_length=50,
              data_shuffle=2000000,
              dropout=0.1,
              learning_warmup=3200):

        self.embedding_size = embedding_size
        self.batch_size = batch_size * self.gpu
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup
