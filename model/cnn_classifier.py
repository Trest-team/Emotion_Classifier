import torch
import torch.nn as nn
# import model.dataloader_for_model 
import model.dataloader_lightning as DL
from torch.nn import functional as F

class cnn_model(nn.Module):
    def __init__(self, hparams):
        super(cnn_model(), self).__init__()
        # self.data_loader = DL(batch_size = 32)
        # vocab_size = len(self.data_loader.text) # 이건 어떻게 가져와야 하는지 모르겠음

        embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_size)
        linear_1 = nn.Linear(hparams.embedding_size, 256)
        conv1_1 = nn.Conv1d(256, 128, kernel = 2)
        pool1_1 = nn.MaxPool1d(2, stride = 1)

        self.conv_operate = nn.Sequential(
            embedding,
            linear_1,
            nn.Dropout(0.2),
            conv1_1,
            nn.ReLU(inplace=True),
            pool1_1
        )

        fc_1 = nn.Linear(128, 64) 
        fc_2 = nn.Linear(64, 32)
        fc_3 = nn.Linear(32, 7)

        self.fc_operate = nn.Sequential(
            fc_1,
            nn.Dropout(0.2),
            nn.ReLU(),
            fc_2,
            nn.Dropout(0.4),
            nn.ReLU(),
            fc_3
        )

        if hparams.use_cuda:
            self.conv_operate = self.conv_operate.cuda()
            self.fc_operate = self.fc_operate.cuda()

    def forward(self, input):
        conv_result = self.conv_operate(input)
        fc_result = self.fc_operate(conv_result)

        return F.softmax(fc_result)

