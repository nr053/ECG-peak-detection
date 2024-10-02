import torch
from utils.sep_conv import Sep_conv_detector
from utils.sep_conv import DiceLoss
from utils.db_loader import DB_loading
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator
from utils.db_generator import Test_Generator
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
import yaml

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

n_channel = cfg['n_channel']
atrous_rate = cfg['atrous_rate']
repo_path = cfg['path_to_repository']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#tools
def plot(idx):
    fig, axs = plt.subplots(5)
    axs[0].plot(features[idx][-1,0,:])
    axs[1].plot(features[idx][-1,1,:])
    axs[2].plot(targets[idx][-1,0])
    axs[3].plot(predictions[idx][-1,0])
    axs[4].plot(torch.sigmoid(predictions[idx][-1,0]))
    plt.savefig(f"foo{idx}.png")


#model
model = Sep_conv_detector(n_channel=n_channel, atrous_rate=atrous_rate).to(device)
model.load_state_dict(torch.load(repo_path + '/model/self_trained_model_sigmoid.pt', weights_only=True))

#test data
print("")
print("Loading test data")
print("")
data_test = DB_loading()
test_dict = data_test.create_set("MIT_BIH_ST", train=True)
#test_dict = data_test.create_set("MIT_BIH", train=True)
#test_dict = data_test.create_set("INCART", train=True)
#test_dict = data_test.create_set("QTDB", train=True)
dataloader_test = Test_Generator(test_dict)


#evaluation section
print("Validation Loop")
model.eval()
predictions = []
features = []
targets = []
with torch.no_grad():
    loop = enumerate(tqdm(dataloader_test))
    for i, (feature, target) in loop:
        prediction = model(feature.to(device).float())
        predictions.append(prediction.cpu())
        features.append(feature)
        targets.append(target)


