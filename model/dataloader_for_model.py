import torch
from torchtext import data
import pandas as pd
from konlpy.tag import Komoran 

komoran = Komoran()

class dataloader(object):
    def __init__(self, path, batch_size = 64, test_ratio = .2, max_vocab = 99999, min_freq = 8, device = -1, use_eos = False, shuffle = True):
        super(dataloader, self).__init__()

        self.label = data.Field(
            sequential = False,
            use_vocab = True,
            unk_token = False
        )
        self.text = data.Field(
            use_vocab = True,
            batch_first = True,
            include_lengths = False,
            tokenize = komoran
        )

        train, test = data.TabularDataset(
            path = path, # your data path
            format = 'csv',
            fields = [
                ('text' , self.text),
                ('label', self.label)
            ],
        ).split(split_ratio = (1-test_ratio))

        self.train_loader, self.test_loader = data.BucketIterator.splits(
            (train, test),
            batch_size = batch_size,
            device = device,
            shuffle = shuffle,
            sort_key = lambda x:len(x.text),
            sort_within_batch = True,
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, min_freq = min_freq, max_vocab = max_vocab)
