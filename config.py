import numpy as np
class Config(object):
    def __init__(self,
                 sampling_rate=44100, audio_duration=2.2, num_classes=41,
                 use_mfcc=True, n_folds=10, learning_rate=0.001,
                 max_epochs=100, n_mfcc=40,weight_decay=1e-4,batch_size=32,
                 path_save='models/',random_seed=9,num_workers=2):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.num_classes = num_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        some_else = 'notrim'
        self.path_save = path_save+ 'lr' + str(learning_rate) +'_'+'randomseed'+str(random_seed)+'_' +some_else+'model.pth'
        self.audio_length = int(self.sampling_rate * self.audio_duration)
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)