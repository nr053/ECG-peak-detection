import torch
from utils.sep_conv import Sep_conv_detector
from utils.sep_conv import DiceLoss
from utils.db_loader import DB_loading
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator
from utils.db_generator import Validation_Generator
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

path_to_data = cfg['path_to_data']

class Train():
    def __init__(self, data_aug:bool, use_swt:bool):
        #config
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        #tag for saving plots and models
        self.tag = ""
        if data_aug:
            self.tag = self.tag + "_w_data_aug"
        if not use_swt:
            self.tag = self.tag + "_no_swt"

    def append_dicts(self, dict1, dict_list:[]):
        for dict in dict_list:    
            for key in dict:
                for item in dict[key]:
                    dict1[key].append(item)
        return dict1


    def create_data(self, data_aug:bool, use_swt:bool):
        #train data
        print("")
        print("Loading train data")
        print("")
        data_train = DB_loading()
        train_vafs = VAF_loading(path_to_data + "/10K_VAF_subset/VAF_subset/train/")
        dict_incart = data_train.create_set("INCART", train=True, use_swt=use_swt)
        dict_mit = data_train.create_set("MIT_BIH", train=True, use_swt=use_swt)
        dict_qt = data_train.create_set("QTDB", train=True, use_swt=use_swt)
        dict_vafs_train = train_vafs.create_set(train=True, use_swt=use_swt)
        self.dict_train = self.append_dicts(dict_incart,  [dict_mit, dict_qt, dict_vafs_train])
        self.dataloader_train = Train_Generator(self.dict_train, data_aug=data_aug)
        #test data
        print("")
        print("Loading test data")
        print("")
        data_test = DB_loading()
        #test_vafs = VAF_loading(path_to_data + "/10K_VAF_subset/VAF_subset/validation/")
        self.dict_test = data_test.create_set("MIT_BIH_ST", train=True, use_swt=use_swt)
        self.dataloader_test = Validation_Generator(self.dict_test)


    def training_loop(self):

        #config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Sep_conv_detector(n_channel=self.cfg['n_channel'], atrous_rate=self.cfg['atrous_rate']).to(device)
        #training parameters
        epochs = self.cfg['n_epochs']
        loss_fn = DiceLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.cfg['learning_rate'])

        #training loop
        train_loss_epoch = []
        test_loss_epoch = []
        epoch_number = 1
        best_epoch = 1

        for epoch in tqdm(range(epochs)):
            #training section
            train_loss_batch = []
            test_loss_batch = []
            model.train()
            print("Training Loop")
            loop = enumerate(tqdm(self.dataloader_train))
            for i, (feature, target) in loop:
                prediction = model(feature.to(device).float())

                #loss
                loss = loss_fn(prediction.cuda(), target.cuda())
                train_loss_batch.append(loss.item())

                model.train()

                #backpropogation
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            
            train_loss_epoch.append(mean(train_loss_batch))

            #evaluation section
            print("Validation Loop")
            model.eval()
            with torch.no_grad():
                loop = enumerate(tqdm(self.dataloader_test))
                for i, (feature, target) in loop:
                    prediction = model(feature.to(device).float())

                    #loss
                    loss = loss_fn(prediction.cuda(), target.cuda())
                    test_loss_batch.append(loss.item())
            test_loss_epoch.append(mean(test_loss_batch))

            #plot loss every 100 epochs
            if epoch_number % 10 == 0:
                plt.figure()
                plt.title(f"Loss at epoch: {epoch_number}")
                plt.plot(train_loss_epoch, label="train")
                plt.plot(test_loss_epoch, label="validation")
                plt.legend()
                plt.savefig("figures/training/loss_plot" + str(epoch_number) + self.tag + ".png")
                plt.close()
            epoch_number+=1

            #save model and epoch number at minimum loss
            if test_loss_epoch[-1] <= min(test_loss_epoch):
                torch.save(model.state_dict(), "/home/rose/Cortrium/ECG-peak-detection/model/self_trained_model" + self.tag + ".pt")
                best_epoch = epoch_number



        plt.figure()
        plt.title(f"Loss plot final. Best epoch: {best_epoch}")
        plt.plot(train_loss_epoch, label="train")
        plt.plot(test_loss_epoch, label="validation")
        plt.legend()
        plt.savefig("figures/training/loss_plot_final" + self.tag + ".png")
        plt.close()

    def visualise(self, idx):
        ecg = self.dict_train["ecg"]
        feature = self.dict_train["feature"]
        target = self.dict_train["target"]
        label = np.zeros(len(ecg[idx]))
        for i in self.dict_train["label"][idx]: 
            label[i] = 1
        mask_array = self.dict_train["mask_array"]


        fig = make_subplots(rows=5, cols=1, shared_xaxes=True)

        fig.add_trace(
            go.Scatter(y=ecg[idx], name=f"ECG"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,0], name="feature 1"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,1], name="feature 2"),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(y=target[idx], name="target"),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(y=label, name="label"),
            row=5, col=1
        )

        fig.update_layout(title_text=f"training sample {idx}")
        fig.write_image(f"figures/training/{idx}.png")



parser = argparse.ArgumentParser(
                    prog="training.py",
                    description="train a deep learning model to detect R-peak positions"
)
parser.add_argument("-aug", "--data_augmentation", help="use data augmentation in training process", action='store_true')
parser.add_argument("-no_swt", "--no_standard_wavelet_transform", help="don't use standard wavelet transform as feature 1", action='store_false')
args = parser.parse_args()

print(f"Using swt: {args.no_standard_wavelet_transform}")
print(f"Using data augmentation: {args.data_augmentation}")
train_class = Train(data_aug=args.data_augmentation, use_swt=args.no_standard_wavelet_transform)
print(f"TAG: {train_class.tag}")
print("Creating data")
train_class.create_data(data_aug=args.data_augmentation, use_swt=args.no_standard_wavelet_transform)
print("Starting training loop...")
train_class.training_loop()