import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import functional as F
from transformers import BartTokenizer


class ChatDataSet(Dataset):
    # def __init__(self, file_path, vocab_path, merge_file, max_seq_len = 128, bos_token = '<bos>', eos_token = '<eos>'):
    def __init__(self, hparams):
        super().__init__()

        self.dataset = pd.read_csv(hparams.file_path)
        self.file_path = hparams.file_path
        self.vocab_path = hparams.vocab_path
        self.merges_file = hparams.merge_file
        self.max_seq_len = hparams.max_seq_len
        self.bos_token = hparams.bos_token
        self.eos_token = hparams.eos_token

        self.tokenizer = BartTokenizer(
            vocab_file = self.vocab_path,
            merges_file = self.merges_file,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
        )

    def __len__(self):
        return len(self.dataset)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(input_id)

        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                mask += [0]

        else:
            input_id = input_id[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            mask = mask[: self.max_seq_len]
        return input_id, mask

    def __getitem__(self, index):
        record = self.dataset.iloc[index]
        q, l = record['Q'], record['label']

        q_tokens = [self.bos_token] + self.tokenizer.tokenize(q) + [self.eos_token]
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

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            emotion_train = ChatDataSet()
        if stage == 'test' or stage is None:
            pass
    
    def train_dataloader(self):
        emotion_train = DataLoader(dataset = self.train_file_path, batch_size = self.batch_size)
        return emotion_train
    
    def val_dataloader(self):
        emotion_val = DataLoader(dataset = self.val_file_path, batch_size = self.batch_size)
        return emotion_val

    def test_dataloader(self):
        emotion_test = DataLoader(dataset = self.test_file_path, batch_size = self.batch_size)
        return emotion_test