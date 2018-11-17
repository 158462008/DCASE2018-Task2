import torch
import numpy as np
from config import Config
from data import raw_data
from models.wavenet_model import WaveNetModel
import pandas as pd
import sys, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import librosa
from sklearn.model_selection import StratifiedKFold
class Manager(object):
    """
        Manager class to train cnn

    """
    def __init__(self, config=Config()):
        print('Prepare the network and data')
        #self._pre_model=pre_model
        self._config=config
        dtype = torch.FloatTensor # data type
        self._net = torch.nn.DataParallel(WaveNetModel(layers=5,
                                 blocks=3,
                                 in_channels=40,
                                 dilation_channels=32,
                                 residual_channels=32,
                                 skip_channels=512,
                                 end_channels=512,
                                 output_length=1,
                                 classes=41,
                                 kernel_size=3,
                                 dtype=dtype,
                                 bias=True)).to(device)
        self._criterion = torch.nn.CrossEntropyLoss().to(device)
        self._optim = torch.optim.Adam(self._net.parameters(), lr=self._config.learning_rate)

        #if self._pre_model is not None:
        #    self._net.load_state_dict(torch.load(self._pre_model,map_location=lambda storage, loc: storage))
        # get train data, valid data, test data
        #self._train_loader, self._valid_loader = \
        #    dataset.get_train_validation_data_loader(self._config,validation_size=0.3,shuffle=True)

        self._test_loader = dataset.get_test_data_loader(self._config)
        print('Complete.')

    def SKF_train(self):
        train = pd.read_csv("./dataset/train.csv")
        test = pd.read_csv("./dataset/sample_submission.csv")
        LABELS = list(train.label.unique())
        label_idx = {label: i for i, label in enumerate(LABELS)}
        train["label_idx"] = train.label.apply(lambda x: label_idx[x])
        skf = StratifiedKFold(n_splits=self._config.n_folds)
        for i, (train_split, valid_split) in enumerate(skf.split(train.fname, train.label_idx)):
            dtype = torch.FloatTensor # data type
            net = WaveNetModel(layers=5,
                                                     blocks=3,
                                                     in_channels=40,
                                                     dilation_channels=32,
                                                     residual_channels=32,
                                                     skip_channels=512,
                                                     end_channels=512,
                                                     output_length=1,
                                                     classes=41,
                                                     kernel_size=3,
                                                     dtype=dtype,
                                                     bias=True).to(device)
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
                for ii, (X, y) in enumerate(train_loader):
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

                train_acc = 100.0 * num_correct / num_total
                valid_acc = self._accuarcy(net, valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = t + 1
                    print('*', end='')
                    model_path='models/model_%d.pth' % (i)
                    torch.save(self._net.state_dict(),model_path)
                print("%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%" %
                      (t+1, sum(epoch_loss)/len(epoch_loss), train_acc, valid_acc))
            print("Best at epoch %d, test accuaray %f" % (best_epoch, best_acc))
    def train(self):
        net = torch.nn.DataParallel(WaveNetModel(layers=10,
                                                 blocks=3,
                                                 in_channels=256,
                                                 dilation_channels=32,
                                                 residual_channels=32,
                                                 skip_channels=512,
                                                 end_channels=512,
                                                 output_length=5126,
                                                 classes=256,
                                                 kernel_size=3,
                                                 bias=True)).to(device)
        train_loader, valid_loader = raw_data.get_train_validation_data_loader(batch_size=128,random_seed=0,validation_size=0.3,
                                                                 shuffle=True,item_length=net.module.receptive_field+net.module.output_length-1,
                                                                 target_length=net.module.output_length,
                                                                 classes=256, mode="train")
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optim = torch.optim.Adam(net.parameters(), lr=self._config.learning_rate)
        print('Training...')
        best_acc = 0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tValid acc')
        for t in range(self._config.max_epochs):
            epoch_loss = []
            num_total = 0
            num_correct = 0
            #for index, (data, label, _) in enumerate(_train_loader)
            for i, (X,target,label) in enumerate(train_loader):
                # Data
                X = torch.autograd.Variable(X, requires_grad=True).to(device)
                y = torch.autograd.Variable(y.view(-1)).to(device)
                label = torch.autograd.Variable(label).to(device)
                # Clear current gradients

                out_w, out_c = net(X) # [batch_size,41]
                loss_raw = criterion(out_w, y)
                loss_classify = criterion(out_c, label)
                loss = loss_raw*1e-4 + loss_classify
                optim.zero_grad()
                epoch_loss.append(loss.data.item())
                # Prediction
                _, prediction = torch.max(out_c.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == label.data ).float()
                # Backward
                loss.backward()
                optim.step()
                if i%10 == 0:
                    print(loss_raw, loss_classify)

            train_acc = 100.0 * num_correct / num_total
            valid_acc = self._accuarcy(net, valid_loader)
            #self._scheduler.step(valid_acc)
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = t + 1
                print('*', end='')
                torch.save(net.state_dict(),self._config.path_save)
            print("%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%" %
                  (t+1, sum(epoch_loss)/len(epoch_loss), train_acc, valid_acc))
        print("Best at epoch %d, test accuaray %f" % (best_epoch, best_acc))


    def _accuarcy(self, model, dataloader):
        model.eval()
        num_correct = 0
        num_total = 0
        for i, (X,_,label) in enumerate(dataloader):
            # Data
            X = torch.autograd.Variable(X).to(device)
            y = torch.autograd.Variable(label).to(device)
            _, score = model(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data ).float()
        model.train()
        return 100 * num_correct / num_total

    def test(self, model_path):
        net = torch.nn.DataParallel(WaveNetModel(layers=10,
                                                 blocks=3,
                                                 in_channels=256,
                                                 dilation_channels=32,
                                                 residual_channels=32,
                                                 skip_channels=256,
                                                 end_channels=256,
                                                 output_length=10246, #5126,2054,10246
                                                 classes=256,
                                                 kernel_size=3,
                                                 bias=True)).to(device)
        net.load_state_dict(torch.load(model_path))
        net.eval()
        test_loader = raw_data.get_test_loader(
            batch_size = self._config.batch_size,
            item_length = net.module.receptive_field+net.module.output_length-1,
            target_length = net.module.output_length,classes=256
        )
        predicted = None
        label_list =None
        for i, (X, label) in enumerate(test_loader): # label: [N]
            X = torch.autograd.Variable(X, requires_grad=False).to(device)
            _, score = net(X) # [N*41]
            predictions = score.cpu().detach().numpy()
            if predicted is None:
                predicted = predictions
                label_list = label
            else:
                predicted = np.concatenate((predicted, predictions))
                label_list = np.concatenate((label_list, label))
        num_sum = np.zeros(max(label_list)+1)
        prediction_wave = np.zeros([max(label_list)+1, 41])
        for ii in range(len(label_list)):
            prediction_wave[label_list[ii]]+=predicted[ii]
            num_sum[label_list[ii]]+=1
        prediction_wave /= num_sum.reshape(-1,1)
        np.save(model_path+'.npy',prediction_wave )





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
        dtype = torch.FloatTensor # data type
        model = WaveNetModel(layers=5,
                             blocks=3,
                             in_channels=40,
                             dilation_channels=32,
                             residual_channels=32,
                             skip_channels=512,
                             end_channels=512,
                             output_length=1,
                             classes=41,
                             kernel_size=3,
                             dtype=dtype,
                             bias=True).to(device)
        pred_list =[]
        ii=0
        for model_path in model_list:
            ii+=1
            model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage))
            model.eval()
            predicted = None
            for i, (X) in enumerate(self._test_loader):
                X = torch.autograd.Variable(X.squeeze(), requires_grad=False).to(device)
                softmax = torch.nn.Softmax(1)
                score = softmax(model(X))
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
    for i in range(10):
        model.append('models/model_%d.pth'%(i))

    #pre_model = './models/63.58_lr0.001_model.pth'
    #model.append('./models/63.58_lr0.001_model.pth')
    ####model.append('./models/60.1_lr0.001_randomseed5_model.pth')
    #model.append('./models/lr0.001_randomseed6_model.pth')
    #model.append('./models/lr0.001_randomseed7_model.pth')
    #model.append('./models/lr0.001_randomseed8_model.pth')

    manager = Manager(config)
    manager.train()
    manager.test_multi_model(model)
    #manager.test()

if __name__=='__main__':
    main()