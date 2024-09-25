import torch
from utils.sep_conv import Sep_conv_detector
from utils.sep_conv import DiceLoss
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator
from utils.db_generator import Test_Generator
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt

n_channel = 2
atrous_rate = [1,3,6,9]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model
model = Sep_conv_detector(n_channel=n_channel, atrous_rate=atrous_rate).to(device)
model.load_state_dict(torch.load("/home/rose/Cortrium/ECG-peak-detection/model/trained_model.pt"))

#data
data = VAF_loading('/home/rose/Cortrium/10K_VAF_subset/VAF_subset/')
dataloader_train, dataloader_test, set_dict = Train_Generator(data.create_set(train=True))



def plot(idx):

    feature = set_dict['feature'].iloc[idx].transpose()
    target = set_dict['target'].iloc[idx]

    fig, axs = plt.subplots(3)
    axs[0].plot(feature[0])
    axs[1].plot(feature[1])
    axs[2].plot(target)
    plt.savefig(f"foo{idx}.png")






#training parameters
learning_rate = 10e-3
loss_fn = DiceLoss()
epochs = 1000
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    #for i, (feature, target) in enumerate(dataloader_train):
        prediction = model(feature.to(device).float())

        #loss
        loss = loss_fn(prediction.cuda(), target.cuda())
        train_loss_batch.append(loss.item())

        #backpropogation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    train_loss_epoch.append(mean(train_loss_batch))

    #evaluation section
    print("Validation Loop")
    with torch.no_grad():
        loop = enumerate(tqdm(dataloader_test))
        for i, (feature, target) in loop:
            prediction = model(feature.to(device).float())

            #loss
            loss = loss_fn(prediction.cuda(), target.cuda())
            test_loss_batch.append(loss.item())

    test_loss_epoch.append(mean(test_loss_batch))


    #plot loss every 100 epochs
    if epoch_number % 100 == 0:
        plt.figure()
        plt.title(f"Loss at epoch: {epoch_number}")
        plt.plot(train_loss_epoch, label="train")
        plt.plot(test_loss_epoch, label="validation")
        plt.legend()
        plt.savefig("figures/training/loss_plot" + str(epoch_number) + ".png")
    epoch_number+=1

    #save model and epoch number at minimum loss
    if test_loss_epoch[-1] <= min(test_loss_epoch):
        torch.save(model.state_dict(), "/home/rose/Cortrium/ECG-peak-detection/model/trained_model_VAF.pt")
        best_epoch = epoch_number

plt.figure()
plt.title(f"Loss plot final. Best epoch: {best_epoch}")
plt.plot(train_loss_epoch, label="train")
plt.plot(test_loss_epoch, label="validation")
plt.legend()
plt.savefig("figures/training/loss_plot_final.png")



