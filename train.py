
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
print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")

config = None
with open('config.yml') as f:
    config = yaml.safe_load(f)



n_epochs = config["train"]["n_epochs"]
train_bs = config["train"]["bs"]
test_bs = config["train"]["bs"]


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

transforms = T.Compose([ 
        T.Resize((64,64)),
        T.ToTensor(), # converts a PIL.Image or numpy array into torch.Tensor
       
        # T.Normalize((0.1307,), (0.3081,)), # Normalize the dataset with mean and std specified
               ])

train_ds = dset.ImageFolder(config['net']['dir']+'/train',transform=transforms)



train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=train_bs)

print("Nr de imagini in setul de antrenare", len(train_ds))
print("Nr de imagini in setul de test", len(test_ds))

print("Dim primei imagini din Dataset", train_ds[0][0])
print("Etichete pt prima imagine", train_ds[0][1])

n_classes = len(np.unique(train_ds.targets))
print(np.unique(train_ds.targets))


print (n_classes)

network = CustomNet(3, config['net']['n1'], config['net']['n2'], config['net']['n3'], n_classes)
print(network)

# Specificarea functiei loss
criterion = nn.CrossEntropyLoss()

# definirea optimizatorului
opt = torch.optim.Adam(network.parameters(), lr=config["train"]["lr"])

n_epochs = config["train"]["n_epochs"]

total_acc = []
total_loss = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device ", device)

network.train()
network.to(device)
criterion.to(device)

for ep in range(n_epochs):
    predictions = []
    targets = []
    
    loss_epoch = 0
    for data in train_loader:
        ins, tgs = data
        ins = ins.to(device)
        tgs = tgs.to(device)
        # redimensionam tensor-ul input
        #print(ins.shape)
        #print(tgs.shape)
        
        # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
        opt.zero_grad()

        # se face forward propagation -> se calculeaza predictia
        output = network(ins)

        # se calculeaza eroarea/loss-ul
        loss = criterion(output, tgs)

        # se face backpropagation -> se calculeaza gradientii
        loss.backward()

        # se actualizează weights-urile
        opt.step()

        loss_epoch = loss_epoch + loss.item()

        with torch.no_grad():
            network.eval()
            current_predict = network(ins)

            # deoarece reteaua nu include un strat de softmax, predictia finala (cifra) trebuie calculata manual
            current_predict = nn.Softmax(dim=1)(current_predict)
            current_predict = current_predict.argmax(dim=1)

            if 'cuda' in device.type:
                current_predict = current_predict.cpu().numpy()
                current_target = tgs.cpu().numpy()
            else:
                current_predict = current_predict.numpy()
                current_target = tgs.numpy()

            # print(current_predict.shape)
            predictions = np.concatenate((predictions, current_predict), axis=0)
            targets = np.concatenate((targets, current_target))
    
    total_loss.append(loss_epoch/train_bs)
    
    # print(predictions.shape)
    # print(len(targets))
    # Calculam acuratetea
    acc = np.sum(predictions==targets)/len(predictions)
    total_acc.append(acc)
    print(f'Epoch {ep}: error {loss_epoch/train_bs} accuracy {acc*100}')

    # salvam ponderile modelului dupa fiecare epoca
    torch.save(network, 'my_model.pt')