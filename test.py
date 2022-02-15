
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import yaml
from custom_net import CustomNet
from datetime import datetime
print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")
import os 
config = None
with open('config.yml') as f:
    config = yaml.safe_load(f)

yml_data=yaml.dump(config)



directory =f"Test{datetime.now().strftime('%H%M_%m%d%Y')}.h5"
parent_dir =r'D:\ai intro\Pytorch\Clasificare_py_torch\Experiment1501_02112022.h5'
path = os.path.join(parent_dir, directory)
os.mkdir(path)

f= open(f"{path}\\yaml_config.txt","w+")
f.write(yml_data)
f.close()




test_bs = config["train"]["bs"]
# incarcam ponderile modelul antrenat

transforms = T.Compose([ 
        T.Resize((64,64)),
        T.ToTensor(), # converts a PIL.Image or numpy array into torch.Tensor
       
        # T.Normalize((0.1307,), (0.3081,)), # Normalize the dataset with mean and std specified
               ])

test_ds = dset.ImageFolder(config['net']['dir']+'/test',transform=transforms) 
test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=test_bs)


network = torch.load(r"D:\\ai intro\\Pytorch\\Clasificare_py_torch\\my_model.pth")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device ", device)

print(type(test_ds.targets), type(test_ds.targets[0]))
test_labels = test_ds.targets
predictions = []

network.eval()
for data in test_loader:
    ins, tgs = data
    ins = ins.to(device)

    current_predict = network(ins)
    current_predict = nn.Softmax(dim=1)(current_predict)
    current_predict = current_predict.argmax(dim=1)

    if 'cuda' in device.type:
        current_predict = current_predict.cpu().numpy()
    else:
        current_predict = current_predict.numpy()
    predictions = np.concatenate((predictions, current_predict))
print(type(predictions))
acc = np.sum(predictions == test_labels)/len(predictions)
print(f'Test accuracy is {acc*100}')
