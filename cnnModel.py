import torch
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np
from data.dataset import Freesound
from data import dataset
from config import Config
from torch.utils.data import DataLoader
import librosa
class get_2d_model(nn.Module):
    def __init__(self,  config=Config()):
        torch.nn.Module.__init__(self)
        self.pre =  nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4,10), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=(4,10), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=(4,10), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=(4,10), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,config.num_classes),
            nn.Softmax(1)
        )
    def __initialize(self):
        torch.nn.init.kaiming_normal_(self.fc[0].weight.data)
        pass
    def forward(self, x):
        N = x.size(0)
        x = self.pre(x)
        #print(x.shape)
        x = x.view(N,-1)
        #print(x.shape)
        return self.fc(x)

if __name__ == '__main__':
    tsfm = transforms.Compose([
        #lambda x: x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6), # rescale to -1 to 1
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
        lambda x: torch.Tensor(x),
        lambda x: x.unsqueeze(0)
    ])
    config = Config()

    # todo: multiprocessing, padding data
    train_dataset = Freesound(transform=tsfm, mode="train",config=config)
    trainloader = DataLoader(
        train_dataset,
        batch_size=2,
        #shuffle=True,
        num_workers=0
    )


    net = get_2d_model()
    _train_loader, _valid_loader = \
        dataset.get_train_validation_data_loader(config,validation_size=0.3,shuffle=True)

    for index, (data, label, _) in enumerate(_train_loader):

        print('input_shape',data.shape)
        print(data[0,0,:,0])
        y = net(data)
        print('output_shape',y.shape)
        #print(y)
        _, prediction = torch.max(y.data, 1)
        print(prediction)
        if index == 0:
            break