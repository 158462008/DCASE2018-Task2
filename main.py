import torch
import numpy as np
from config import Config
from data import dataset
from models.cnnModel import get_2d_model
import pandas as pd
import sys, os
from sklearn.model_selection import StratifiedKFold
import librosa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Manager(object):
    """
        Manager class to train cnn

    """
    def __init__(self, config=Config(), pre_model=None):
        print('Prepare the network and data')

        self._pre_model=pre_model
        self._config=config
        self._net = get_2d_model().to(device)
        self._criterion = torch.nn.CrossEntropyLoss().to(device)
        self._optim = torch.optim.Adam(self._net.parameters(), lr=self._config.learning_rate)

        if self._pre_model is not None:
            self._net.load_state_dict(torch.load(self._pre_model,map_location=lambda storage, loc: storage))
        # get train data, valid data, test data
        self._train_loader, self._valid_loader = \
            dataset.get_train_validation_data_loader(self._config,validation_size=0.3,shuffle=True)

        self._test_loader = dataset.get_test_data_loader(self._config)
        print('Complete.')


    def SKF_train(self):
        #########
        train = pd.read_csv("./dataset/train.csv")
        test = pd.read_csv("./dataset/sample_submission.csv")
        LABELS = list(self.train.label.unique())
        label_idx = {label: i for i, label in enumerate(LABELS)}
        #train.set_index("fname", inplace=True)
        #test.set_index("fname", inplace=True)
        train["label_idx"] = train.label.apply(lambda x: label_idx[x])
        skf = StratifiedKFold(n_splits=self._config.n_folds)
        ##########
        for i, (train_split, valid_split) in enumerate(skf.split(train.fname, train.label_idx)):
            net = 1
            #for i, (train_split, val_split) in enumerate(skf):
            train_set = train.iloc[train_split]
            valid_set = train.iloc[valid_split]
            train_loader, valid_loader = dataset.DataGenerator(Config(),train_set.fname,train_set.label_idx,
                                                       valid_set.fname,valid_set.label_idx)
            print('Training...',i)
            best_acc = 0
            best_epoch = None
            print('Epoch\tTrain loss\tTrain acc\tValid acc')
            for t in range(self._config.max_epochs):
                epoch_loss = []
                num_total = 0
                num_correct = 0
                for i, (X, y) in enumerate(train_loader):

                    #print(y.shape) # torch.Size([N])
                    #print(X.shape)  # torch.Size([N, 40, 187])
                    X = torch.autograd.Variable(X, requires_grad=True).to(device)
                    y = torch.autograd.Variable(y).to(device)
                    self._optim.zero_grad()
                    score = net(X)
                    loss = self._criterion(score, y)
                    epoch_loss.append(loss.data.item())
                    # Prediction
                    _, prediction = torch.max(score.data, 1)
                    num_total += y.size(0)
                    num_correct += torch.sum(prediction == y.data ).float()
                    # Backward
                    loss.backward()
                self._optim.step()







    def train(self):
        print('Training...')
        best_acc = 0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tValid acc')
        for t in range(self._config.max_epochs):
            epoch_loss = []
            num_total = 0
            num_correct = 0
            #for index, (data, label, _) in enumerate(_train_loader)
            for i, (X,y,_) in enumerate(self._train_loader):
                # Data
                X = torch.autograd.Variable(X, requires_grad=True).to(device)
                y = torch.autograd.Variable(y).to(device)
                # Clear current gradients
                self._optim.zero_grad()
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.data.item())
                # Prediction
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data ).float()
                # Backward
                loss.backward()
                self._optim.step()

            train_acc = 100.0 * num_correct / num_total
            valid_acc = self._accuarcy(self._net, self._valid_loader)
            #self._scheduler.step(valid_acc)
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = t + 1
                print('*', end='')
                torch.save(self._net.state_dict(),self._config.path_save)
            print("%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%" %
                  (t+1, sum(epoch_loss)/len(epoch_loss), train_acc, valid_acc))
        print("Best at epoch %d, test accuaray %f" % (best_epoch, best_acc))


    def train_mfcc_wavenet(self):
        pass
    def _accuarcy(self, model, dataloader):
        model.eval()
        num_correct = 0
        num_total = 0
        for i, (X,y,_) in enumerate(dataloader):
            # Data
            X = torch.autograd.Variable(X).to(device)
            y = torch.autograd.Variable(y).to(device)
            score = model(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data ).float()
        model.train()
        return 100 * num_correct / num_total

    def test(self):
        self._net.eval()
        predicted_labels =[]

        for i, (X) in enumerate(self._test_loader):
            X = torch.autograd.Variable(X, requires_grad=False).to(device)
            score = self._net(X)
            predictions = score.detach().numpy()
            classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}
            num_to_class = dict((v,k) for k,v in classes.items())
            top_3 = np.argsort(-predictions, axis=1)[:,:3]
            str_top = []
            for t3 in  top_3:
                str_top.append([''.join(list(num_to_class[x])) for x in t3])
            #print(str_top)
            predicted_labels += [' '.join(list(x)) for x in str_top] # ['s s s']
            #print(predicted_labels)
        data_root = "./dataset"
        csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))
        csv_file['label'] = predicted_labels
        csv_file[['fname','label']].to_csv('submission.csv', index=False)
    def help(self):
        print('help')
    def test_multi_model(self,model_list):
        model = get_2d_model().to(device)
        pred_list =[]
        ii=0
        for model_path in model_list:
            ii+=1
            model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage))
            model.eval()
            predicted = None
            for i, (X) in enumerate(self._test_loader):
                X = torch.autograd.Variable(X, requires_grad=False).to(device)
                score = model(X)
                predictions = score.detach().numpy()
                if predicted is None: predicted = predictions
                else: predicted = np.concatenate((predicted,predictions))
            pred_list.append(predicted)
            print('model %d is complete' % ii)

        prediction = np.ones_like(pred_list[0])
        for pred in pred_list:
            prediction = prediction*pred
        prediction = prediction**(1./len(pred_list))
        # Make a submission file
        classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}
        num_to_class = dict((v,k) for k,v in classes.items())
        top_3 = np.argsort(-prediction, axis=1)[:,:3]
        str_top = []
        for t3 in  top_3:
            str_top.append([''.join(list(num_to_class[x])) for x in t3])
        predicted_labels = [' '.join(list(x)) for x in str_top] # ['s s s']
        data_root = "./dataset"
        csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))
        csv_file['label'] = predicted_labels
        csv_file[['fname','label']].to_csv('submission_seed8.csv', index=False)


def  main():
    config = Config()
    model = []
    #pre_model = './models/63.58_lr0.001_model.pth'
    #model.append('./models/63.58_lr0.001_model.pth')
    ####model.append('./models/60.1_lr0.001_randomseed5_model.pth')
    #model.append('./models/lr0.001_randomseed6_model.pth')
    #model.append('./models/lr0.001_randomseed7_model.pth')
    model.append('./models/lr0.001_randomseed8_model.pth')

    manager = Manager(config)
    #manager.train()
    manager.test_multi_model(model)
    #manager.test()

if __name__=='__main__':
    main()