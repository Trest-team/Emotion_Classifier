import argparse
import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
from typing import List
import numpy as np
import logging
from copy import deepcopy
import torch.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torch.optim import lr_scheduler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartTokenizer
import Emotion_Classifier.model.dataloader_lightning as Data_L
import Emotion_Classifier.model.utils as utils
import Emotion_Classifier.model.cnn_classifier as cnn_c

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

        return parser

# input data를 학습에 쓸 수 있게 처리해주는 부분(masking,...)
class ChatDataSet(Dataset):
    def __init__(self, file_path, vocab_path, merge_file, max_seq_len = 128, bos_token = '<bos>', eos_token = '<eos>'):
    # def __init__(self, hparams):
        super().__init__()

        self.dataset = pd.read_csv(file_path)
        self.file_path = file_path
        self.vocab_path = vocab_path
        self.merges_file = merge_file
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.tokenizer = BartTokenizer(
            vocab_file = self.vocab_path,
            merges_file = self.merges_file,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
        )

    def __len__(self):
        return len(self.dataset)

    def make_input_id_mask(self, tokens, index):
        # token들을 id로 변환
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        # id가 있는 부분들은 1로 처리해준다
        mask = [1] * len(input_id)

        # max_seq_len이 될 때까지 mask에는 pad_token_id인 0을 붙여주고 input_id에는 pad_token_id를 붙여준다
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                mask += [0]
        # input_id의 길이가 max_seq_len보다 길 경우, max_seq_len에 맞춰 eos_token_id를 붙이고 input_id를 자른다
        else:
            input_id = input_id[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            mask = mask[: self.max_seq_len]
        return input_id, mask

    def __getitem__(self, index):
        # 원하는 index의 data를 받아온다
        record = self.dataset.iloc[index]
        # data(record)를 q와 l로 나눈다
        q, l = record['Q'], record['label']

        # q를 tokenizer에 넣어 token화 시키고 앞에는 bos, 끝에는 eos token을 붙여 학습에 사용할 수 있는 데이터로 만든다.
        q_tokens = [self.bos_token] + self.tokenizer.tokenize(q) + [self.eos_token]
        # l 또한 token화
        l_tokens = self.tokenizer.tokenize(l)

        input_id, masking = self.make_input_id_mask(q_tokens, index)
        label = self.tokenizer.convert_tokens_to_ids((l_tokens))

        return {
            'input_ids' : np.array(input_id, dtype = np.int_),
            'mask' : np.array(masking, dtype = np.float_),
            'labels' : np.array(label, dtype = np.int_)
        }

class DataModule(pl.LightningDataModule):
    #def __init__(self, train_file, val_file, test_file, vocab_path, merge_file, batch_size = 64, max_seq_len = 128, num_workers = 5):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.batch_size = hparams.batch_size
        self.max_seq_len = hparams.max_seq_len

        self.train_file_path = hparams.train_file
        self.val_file_path = hparams.val_file
        self.test_file_path = hparams.test_file

        self.vocab_path = hparams.vocab_path
        self.merges_file = hparams.merge_file

        self.num_workers = hparams.num_workers

    def prepare_data(self):
        pass

    # train_file, val_file, test_file들을 학습에 사용 가능한 형태로 수정
    def setup(self, stage = None):
        self.train_file = ChatDataSet(self.train_file_path, self.vocab_path, self.merges_file, self.max_seq_len)
        self.val_file = ChatDataSet(self.val_file_path, self.vocab_path, self.merges_file, self.max_seq_len)
        self.test_file = ChatDataSet(self.test_file_path, self.vocab_path, self.merges_file, self.max_seq_len)
    
    # self.train_file을 사용하는 dataloader 선언
    def train_dataloader(self):
        emotion_train = DataLoader(dataset = self.train_file, batch_size = self.batch_size)
        return emotion_train
    
    # self.val_file을 사용하는 dataloader 선언
    def val_dataloader(self):
        emotion_val = DataLoader(dataset = self.val_file, batch_size = self.batch_size)
        return emotion_val

    # self.test_file을 사용하는 dataloader 선언
    def test_dataloader(self):
        emotion_test = DataLoader(dataset = self.test_file, batch_size = self.batch_size)
        return emotion_test

class Base(pl.LightningModule):
    # 추가적인 argument들을 불러온다
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser = [parent_parser], add_help = False)

        parser.add_argument('--batch-size', type = int, default = 16, help = 'batch size for trainng')

        parser.add_argument('--lr', type = float, default = 5e-5, help = 'The initial learning rate')

        parser.add_argument('--warmup_ratio', type = float, default = 0.1, help = 'warmup ratio')

        return parser

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

# 학습 base class
class ChatClassification(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(ChatClassification, self).__init__()
        self.hparams = hparams
        # 학습 모델 선언
        self.cnn_c = cnn_c(self.hparams)

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

def main(hparams):
    model = ChatClassification(hparams)

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

    dataloader = DataModule(hparams)

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