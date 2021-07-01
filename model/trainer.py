import argparse
import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
from typing import List
import numpy as np
import logging
import pytorch_lightning as pl
from copy import deepcopy
import torch.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torch.optim import lr_scheduler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartTokenizer
from ..model import dataloader_lightning as Data_L
from ..model import utils 
from ..model import cnn_classifier as cnn_c

# parser 선언 및 checkpoint_path를 argument에 추가
parser = argparse.ArgumentParser(descroption = "Trest's Emotion classifier")
parser.add_argument('--checkpoint_path', type = str, help = 'checkpoint path')

# logger 선언
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# wandb를 사용해 학습을 시각화 할 예정 
wandb_logger = WandbLogger()

# argument들을 받아오는 class
class ArgsBase():
    @staticmethod
    def add_level_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents = [parent_parser], add_help = False
        )
        parser.add_argument('--train_file',
                            type = str,
                            default = 'dataset/ChatbotData_shuf_train.csv',
                            help = 'train_file')

        parser.add_argument('--val_file',
                            type = str,
                            default = 'dataset/ChatbotData_shuf_valid.csv',
                            help = 'val_file')

        parser.add_argument('--test_file',
                            type = str,
                            default = 'dataset/ChatbotData_shuf_test.csv',
                            help = 'test_file')

        parser.add_argument('--max_vocab_size',
                            type = int,
                            default = 32000,
                            help = "vocabulary's size")

        parser.add_argument('--vocab_path',
                            type = str,
                            default = 'made_tokenizers/vocab.json',
                            help = 'vocab_path')

        parser.add_argument('--merges_file',
                            type = str,
                            default = 'made_tokenizers/merges.txt',
                            help = 'merges_file')
        
        parser.add_argument('--batch_size',
                            type = int,
                            default = 32,
                            help = 'batch size')

        parser.add_argument('--embedding_size',
                            type = int,
                            default = 512,
                            help='embedding vector size')

        parser.add_argument('--use_cuda',
                            type = bool,
                            default=True,
                            help = 'decide using cuda')

        parser.add_argument('--max_seq_len',
                            type = int,
                            default = 128,
                            help = 'max_seq_len')

        parser.add_argument('--num_workers',
                            type = int,
                            default = 5,
                            help = 'num_workers')

        return parser
    
# 학습 base class
class Base(Data_L):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams
        # 학습 모델 선언
        self.cnn_c = cnn_c(self.hparams)

    # 추가적인 argument들을 불어온다
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser = [parent_parser], add_help = False)

        parser.add_argument('--batch-size', type = int, default = 16, help = 'batch size for trainng')

        parser.add_argument('--lr', type = float, default = 5e-5, help = 'The initial learning rate')

        parser.add_argument('--warmup_ratio', type = float, default = 0.1, help = 'warmup ratio')

        return parser

    # forward 함수 -> model 함수에 input을 보내 결과를 받아온다(softmax 결과)
    def forward(self, input):
        model_logit = self.cnn_c(input)

        return model_logit

    # logits(model의 결과)와 label의 차이를 구하는 loss function을 선언
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    # training step을 선언
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    # validation step을 선언
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)

        loss = self.cross_entropy_loss(logits, y)
        logs = {'val_loss' : loss}

        return logs

    # test step을 선언
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)

        loss = self.cross_entropy_loss(logits, y)
        logs = {'test_loss' : loss}

        return logs
    
    # optimizer을 선언
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        # decay되지 않을 parameter들의 list 선언
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # no_decay되는 parameter들은 weight_decay를 0으로, 나머지 parameter들은 0.01로 선언
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer로 AdamW를 사용
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams.lr, correct_bias=False)
        
        # num_workers 계산
        num_workers = [self.hparams.gpus if self.hparams.gpus is not None else 1] * (self.hparams.num_nodes if self.hparams.num_node is not None else 1)
        # train.dataloader의 dataset의 길이를 반환
        # 구조가 애매...
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers))
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_train_steps = num_train_steps)
        lr_scheduler = {'schedular' : schedular, 'monitor' : 'loss', 'interval' : 'step', 'frequency' : 1}

        return [optimizer], [lr_scheduler]

def main(hparams):
    model = Base(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath = os.path.join('Emotion_Classifier/model_checkpoint', '{epoch:d}'),
        verbose=True,
        save_last=True,
        save_top_k= 3,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor = 'val_acc',
        patience = 3,
        verbose = True,
        mode = 'max'
    )

    trainer_args = {
        'callbacks' : [checkpoint_callback, early_stopping],
        'gpus' : -1,
        'logger' : wandb_logger,
        'max_epoch' : 10
    }

    if hparams.checkpoint:
        # hparams에 checkpoint 추가
        trainer_args['resume_from_checkpoint'] = os.path.join('Emotion_Classifier/model_checkpoint', hparams.checkpoint)

    trainer = Trainer(**trainer_args)

    dataloader = Data_L.DataModule(hparams)
    model = Base(hparams)

    if hparams.checkpoint_path is not None:
        trainer.fit(model, dataloader)
    else:
        trainer.test()

    wandb.finish()

if __name__ == 'main':
    parser = ArgsBase.add_level_specific_args(parser)
    parser = Base.add_model_specific_args(parser)
    args = parser.parse_args()

    # logger부분 학습하기
    logging.info(args)

    main(args)