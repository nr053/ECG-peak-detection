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

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

n_channel = cfg['n_channel']
atrous_rate = cfg['atrous_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#tools
def plot(idx):

    feature = set_dict['feature'].iloc[idx].transpose()
    target = set_dict['target'].iloc[idx]

    fig, axs = plt.subplots(3)
    axs[0].plot(feature[0])
    axs[1].plot(feature[1])
    axs[2].plot(target)
    plt.savefig(f"foo{idx}.png")

def append_dicts(dict1, dict_list:[]):
    for dict in dict_list:    
        for key in dict:
            for item in dict[key]:
                dict1[key].append(item)
    return dict1

#model
model = Sep_conv_detector(n_channel=n_channel, atrous_rate=atrous_rate).to(device)

#training parameters
learning_rate = cfg['learning_rate']
epochs = cfg['n_epochs']
loss_fn = DiceLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train data
print("")
print("Loading train data")
print("")
data_train = DB_loading()
dict_incart = data_train.create_set("INCART", train=True)
dict_mit = data_train.create_set("MIT_BIH", train=True)
dict_qt = data_train.create_set("QTDB", train=True)
dict_train = append_dicts(dict_incart,  [dict_mit, dict_qt])
dataloader_train = Train_Generator(dict_train)
#test data
print("")
print("Loading test data")
print("")
data_test = DB_loading()
test_dict = data_test.create_set("MIT_BIH_ST", train=True)
dataloader_test = Validation_Generator(test_dict)

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
    loop = enumerate(tqdm(dataloader_train))
    for i, (feature, target) in loop:
        prediction = model(feature.to(device).float())

        #loss
        loss = loss_fn(prediction.cuda(), target.cuda())
        train_loss_batch.append(loss.item())

        #visualise before back prop
        model.eval()
        if epoch_number == 1 and i == 0:
            fig, axs = plt.subplots(3)
            axs[0].plot(prediction[0,-1].cpu().detach())
            axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
            axs[2].plot(target[0,-1].cpu().detach())
            plt.savefig("foo_untrained.png")
            plt.close()
        model.train()

        #backpropogation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    train_loss_epoch.append(mean(train_loss_batch))

    #plot an example from the training set to show progression throughout training
    fig, axs = plt.subplots(3)
    axs[0].plot(prediction[0,-1].cpu().detach())
    axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
    axs[2].plot(target[0,-1].cpu().detach())
    plt.savefig("foo_training.png")
    plt.close()

    #evaluation section
    print("Validation Loop")
    model.eval()
    with torch.no_grad():
        loop = enumerate(tqdm(dataloader_test))
        for i, (feature, target) in loop:
            prediction = model(feature.to(device).float())

            #loss
            loss = loss_fn(prediction.cuda(), target.cuda())
            test_loss_batch.append(loss.item())
    test_loss_epoch.append(mean(test_loss_batch))

    #plot an example from the training set to show progression throughout training
    fig, axs = plt.subplots(3)
    axs[0].plot(prediction[0,-1].cpu().detach())
    axs[1].plot(torch.sigmoid(prediction[0,-1].cpu().detach()))
    axs[2].plot(target[0,-1].cpu().detach())
    plt.savefig("foo_validation.png")
    plt.close()

    #plot loss every 100 epochs
    if epoch_number % 10 == 0:
        plt.figure()
        plt.title(f"Loss at epoch: {epoch_number}")
        plt.plot(train_loss_epoch, label="train")
        plt.plot(test_loss_epoch, label="validation")
        plt.legend()
        plt.savefig("figures/training/loss_plot" + str(epoch_number) + ".png")
    epoch_number+=1

    #save model and epoch number at minimum loss
    if test_loss_epoch[-1] <= min(test_loss_epoch):
        torch.save(model.state_dict(), "/home/rose/Cortrium/ECG-peak-detection/model/self_trained_model.pt")
        best_epoch = epoch_number



plt.figure()
plt.title(f"Loss plot final. Best epoch: {best_epoch}")
plt.plot(train_loss_epoch, label="train")
plt.plot(test_loss_epoch, label="validation")
plt.legend()
plt.savefig("figures/training/loss_plot_final.png")



