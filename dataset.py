import sys, os
import torch
import librosa
import numpy as np
import pandas as pd
from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
SAMPLE_RATE = 44100
from config import Config


class Freesound(Dataset):
    def __init__(self, transform=None, mode="train", config = Config(),set_fname=None,set_labelid=None):
        self._config = config
        if transform is None:
            transform = transforms.Compose([
                lambda x: x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6), # rescale to -1 to 1
                lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
                lambda x: torch.Tensor(x),
                #lambda x: x.unsqueeze(0) # add the first channel 1x40x173
            ])
        # setting directories for data
        data_root = "dataset"

        self.mode = mode
        self.input_length = config.audio_length
        if self.mode is "train":
            self.set_fname = list(set_fname)
            self.set_labelid = list(set_labelid)
            self.data_dir = os.path.join(data_root, "audio_train")
            #self.csv_file = pd.read_csv(os.path.join(data_root, "train.csv"))
        elif self.mode is "test":
            self.data_dir = os.path.join(data_root, "audio_test")
            self.csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

        # dict for mapping class names into indices. can be obtained by
        # {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
        self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}

        self.transform = transform

    def __len__(self):
        if self.mode is "train":
            return len(self.set_fname)
        else:
            return self.csv_file.shape[0]

    def trim_silence(self, audio, threshold=0.005, frame_length=512):
        '''Removes silence at the beginning and end of a sample.'''
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rmse(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio

    def __getitem__(self, idx):
        if self.mode is "train":
            filename = self.set_fname[idx]
        else :
            filename = self.csv_file["fname"][idx]
        #filename = self.csv_file["fname"][idx]
        data, _ = librosa.load(os.path.join(self.data_dir, filename), sr=SAMPLE_RATE, mono=True)
        #print(max(data))
        #print('data_shape',data.shape)
        # if len > 2s, then trim silence the top and end
        if len(data) > self.input_length:
            data = self.trim_silence(data)
            #print('trim_shape',data.shape)

        # Random offset / Padding  , get 2s from one audio
        if len(data) > self.input_length:
            max_offset = len(data) - self.input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(self.input_length+offset)]
        else:
            if self.input_length > len(data):
                max_offset = self.input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, self.input_length - len(data) - offset), "constant")

        if self.transform is not None:
            data = self.transform(data)

        data=data[:,0:187]
        if self.mode is "train":
            label = self.set_labelid[idx]
            #label = self.classes[self.csv_file["label"][idx]]
            #verified = int(self.csv_file["manually_verified"][idx])
            return data, label #verified

        elif self.mode is "test":
            return data
'''
    def get_weight(self,validation_size=0,shuffle=True):
        num_train = self.__len__()
        indices = list(range(num_train))
        split = int(np.floor(validation_size*num_train))
        if shuffle:
            np.random.seed(self._config.random_seed)
            np.random.shuffle(indices)
        train_idx ,valid_idx = indices[split:], indices[:split]
        w_train = np.zeros(num_train)
        w_train[train_idx]=1
        w_val = np.zeros(num_train)
        w_val[valid_idx]=1
        for idx in range(num_train):
            verified = int(self.csv_file["manually_verified"][idx])
            if w_train[idx]==1 and verified ==1: w_train[idx]=2 # 确认过的训练样本
        return w_train, w_val

def get_train_validation_data_loader(config = Config(),validation_size = 0.3,
                                     shuffle=True):
    #read = np.load('mean_std.npz')
    mean = 10#read['mean']
    std = 1#read['std']
    tsfm = transforms.Compose([
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: (x - mean) / std,
        lambda x: torch.Tensor(x),
        lambda x: x.unsqueeze(0)
    ])
    train_dataset = Freesound(transform=tsfm, mode='train',config=config)
    valid_dataset = Freesound(transform=tsfm, mode='train',config=config)

    num_train = len(train_dataset)

    # 对确认过的样本采样几率提升一倍，仅对训练集处理
    w_train ,w_val = train_dataset.get_weight(validation_size=validation_size,shuffle=shuffle)

    # 随机权重采样，可以重复选取
    train_sampler = WeightedRandomSampler(w_train,num_train,replacement=True)
    valid_sampler = WeightedRandomSampler(w_val,num_train,replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, sampler=valid_sampler, num_workers=config.num_workers,
        drop_last=True
    )
    return train_loader, valid_loader
'''
def get_test_data_loader(config = Config()):
    read = np.load('mean_std.npz')
    mean = read['mean']
    std = read['std']
    mean = np.concatenate((mean,mean),1)[:,0:190]
    std = np.concatenate((std,std),1)[:,0:190]
    tsfm = transforms.Compose([
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: (x - mean) / std,
        lambda x: torch.Tensor(x),
        #lambda x: x.unsqueeze(0)
    ])
    test_dataset = Freesound(transform=tsfm, mode='test',config=config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, num_workers=config.num_workers,drop_last=False
    )
    return test_loader

def DataGenerator(config = Config(), train_set_fname=None,train_set_labelid=None,
                  valid_set_fname=None, valid_set_labelid=None):
    mean = 10#read['mean']
    std = 1#read['std']
    tsfm = transforms.Compose([
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: (x - mean) / std,
        lambda x: torch.Tensor(x),
        #lambda x: x.unsqueeze(0)
    ])
    train_dataset = Freesound(transform=tsfm, mode='train',config=config,set_fname=train_set_fname,set_labelid=train_set_labelid)
    valid_dataset = Freesound(transform=tsfm, mode='train',config=config,set_fname=valid_set_fname,set_labelid=valid_set_labelid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                               drop_last=True)
    return train_loader, valid_loader




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    '''
    tsfm = transforms.Compose([
        lambda x: x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6), # rescale to -1 to 1
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: Tensor(x)
    ])
    config = Config()

    # todo: multiprocessing
    train_dataset = Freesound(transform=tsfm, mode="train",config=config)
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)
    '''
    trainloader, _ = get_train_validation_data_loader(Config(),validation_size=0.3,shuffle=True)
    for index, (data, label,_) in enumerate(trainloader):
        print(label.numpy())
        print(data.squeeze().shape)
        #plt.imshow(data.squeeze().numpy()[ :, :50],cmap='hot', interpolation='nearest')
        #plt.show()

        if index == 0:
            break

