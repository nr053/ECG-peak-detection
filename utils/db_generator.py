import numpy as np
import torch.utils.data as data
import pandas as pd

feature_shape = 2048

class Test_Loader(data.Dataset):
    def __init__(self, set_dict):
        self.list_feature = set_dict['feature']
        self.list_target = set_dict['target']
        self.iteration = len(self.list_feature)
        # print('Test ... No. of Records :', self.iteration)

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        feature = self.list_feature[index]
        target = self.list_target[index]

        padding = -feature.shape[0]%feature_shape # cropping
        if padding != 0:
            feature = np.pad(feature, ((0, padding), (0, 0)))
            target = np.pad(target, (0, padding))

        X = np.swapaxes(feature, 0, 1)
        y = target[np.newaxis, :]

        return X, y

class Train_Loader(data.Dataset):
    def __init__(self, set_dict):
        self.list_feature = set_dict['feature']
        self.list_target = set_dict['target']
        self.iteration = len(self.list_feature)
        # print('Test ... No. of Records :', self.iteration)

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        feature = self.list_feature.iloc[index]
        target = self.list_target.iloc[index]

        padding = -feature.shape[0]%feature_shape # cropping
        if padding != 0:
            feature = np.pad(feature, ((0, padding), (0, 0)))
            target = np.pad(target, (0, padding))

        X = np.swapaxes(feature, 0, 1)
        y = target[np.newaxis, :]

        return X, y



def Test_Generator(set_dict, shuffle):
    data_loader = Test_Loader(set_dict)
    data_generator = data.DataLoader(data_loader, batch_size=1)
    return data_generator

def Train_Generator(set_dict):
    #shuffle the data
    set_dict = pd.DataFrame.from_dict(set_dict).sample(frac=1, random_state=1)
    
    
    train_size = int(len(set_dict)*0.8)
    test_size = int((len(set_dict) - train_size)/2) #size of validation and test set

    train_dict = set_dict.iloc[:train_size].copy()
    validation_dict = set_dict.iloc[train_size:train_size+test_size].copy()
    test_dict = set_dict.iloc[train_size+test_size:].copy()

    data_loader_train = Train_Loader(train_dict)
    data_loader_test = Train_Loader(validation_dict)
    
    data_generator_train = data.DataLoader(data_loader_train, batch_size=32, shuffle=True)
    data_generator_test = data.DataLoader(data_loader_test, batch_size=32, shuffle=True)
    
    return data_generator_train, data_generator_test, set_dict
