import argparse
import os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
from typing import List
import numpy as np
import torch
import logging
import pytorch_lightning as pl
from copy import deepcopy
import torch.functional as F

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torch.optim import lr_scheduler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import Emotion_Classifier.model.dataloader_lightning as Data_L
import Emotion_Classifier.model.utils as utils
import Emotion_Classifier.model.cnn_classifier as cnn_c


parser = argparse.ArgumentParser(descroption = "Trest's Emotion classifier")
parser.add_argument('--checkpoint_path', type = str, help = 'checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
wandb_logger = WandbLogger()
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

        parser.add_argument('--train_file',
                            type = str,
                            default = './',
                            help = 'train_data file link')

        parser.add_argument('--val_file',
                            type = str,
                            default = './',
                            help = 'val_data file link')

        parser.add_argument('--test_file',
                            type = str,
                            default = './',
                            help = 'test_data file link')

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

        parser.add_argument('--checkpoint',
                            type = str,
                            default = None,
                            help = 'pretrained model checkpoint')

        return parser
    
class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams
        self.cnn_c = cnn_c(self.hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser = [parent_parser], add_help = False)

        parser.add_argument('--batch-size', type = int, default = 16, help = 'batch size for trainng')

        parser.add_argument('--lr', type = float, default = 5e-5, help = 'The initial learning rate')

        parser.add_argument('--warmup_ratio', type = float, default = 0.1, help = 'warmup ratio')

        return parser

    def forward(self, input):
        model_logit = self.cnn_c(input)

        return model_logit

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)

        loss = self.cross_entropy_loss(logits, y)
        logs = {'val_loss' : loss}

        return logs

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)

        loss = self.cross_entropy_loss(logits, y)
        logs = {'test_loss' : loss}

        return logs

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams.lr, correct_bias=False)
        
        num_workers = [self.hparams.gpus if self.hparams.gpus is not None else 1] * (self.hparams.num_nodes if self.hparams.num_node is not None else 1)
        data_len = len(self.train.dataloader().dataset)
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

    if hparams.checkpoint is not None:
        trainer.fit(model, dataloader)
    else:
        trainer.test()

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser = ArgsBase.add_level_specific_args(parser)
    parser = Base.add_model_specific_args(parser)
    args = parser.parse_args()

    logging.info(args)

    main(args)