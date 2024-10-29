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


class Train():
    def __init__(self, data_aug:bool, use_swt:bool):
        #config
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.tag = ""
        if data_aug:
            self.tag = self.tag + "_w_data_aug"
        if not use_swt:
            self.tag = self.tag + "_no_swt"

    def plot(idx):

        feature = set_dict['feature'].iloc[idx].transpose()
        target = set_dict['target'].iloc[idx]

        fig, axs = plt.subplots(3)
        axs[0].plot(feature[0])
        axs[1].plot(feature[1])
        axs[2].plot(target)
        plt.savefig(f"foo{idx}.png")

    def append_dicts(self, dict1, dict_list:[]):
        for dict in dict_list:    
            for key in dict:
                for item in dict[key]:
                    dict1[key].append(item)
        return dict1


    def create_data(self, data_aug:bool):
        #train data
        print("")
        print("Loading train data")
        print("")
        data_train = DB_loading()
        dict_incart = data_train.create_set("INCART", train=True)
        dict_mit = data_train.create_set("MIT_BIH", train=True)
        dict_qt = data_train.create_set("QTDB", train=True)
        dict_train = self.append_dicts(dict_incart,  [dict_mit, dict_qt])
        self.dataloader_train = Train_Generator(dict_train, data_aug=data_aug)
        #test data
        print("")
        print("Loading test data")
        print("")
        data_test = DB_loading()
        test_dict = data_test.create_set("MIT_BIH_ST", train=True)
        self.dataloader_test = Validation_Generator(test_dict)


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

                #visualise before back prop
                # model.eval()
                # if epoch_number == 1 and i == 0:
                #     fig, axs = plt.subplots(3)
                #     fig.title("Training")
                #     axs[0].plot(prediction[0,-1].cpu().detach())
                #     axs[0].set_title("Prediction")
                #     axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
                #     axs[1].set_title("sigmoid")
                #     axs[2].plot(target[0,-1].cpu().detach())
                #     axs[2].set_title("Target")
                #     plt.savefig("foo_untrained.png")
                #     plt.close()
                model.train()

                #backpropogation
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            
            train_loss_epoch.append(mean(train_loss_batch))

            # #plot an example from the training set to show progression throughout training
            # fig, axs = plt.subplots(3)
            # axs[0].plot(prediction[0,-1].cpu().detach())
            # axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
            # axs[2].plot(target[0,-1].cpu().detach())
            # plt.savefig("foo_training.png")
            # plt.close()

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

            # #plot an example from the training set to show progression throughout training
            # fig, axs = plt.subplots(3)
            # axs[0].plot(prediction[0,-1].cpu().detach())
            # axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
            # axs[2].plot(target[0,-1].cpu().detach())
            # plt.savefig("foo_validation.png")
            # plt.close()

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
print("Creating data")
train_class.create_data(data_aug=args.data_augmentation)
print("Starting training loop...")
train_class.training_loop()