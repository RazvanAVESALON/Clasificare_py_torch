
from json.tool import main
from msilib.schema import _Validation_records
from pathlib import Path
from winreg import ExpandEnvironmentStrings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import torchmetrics
import yaml
from custom_net import CustomNet
import torch
import torch.nn as nn
import torch.optim as optim       
from datetime import datetime
import os 
import random

from torchvision.datasets import ImageFolder
from tqdm import tqdm


def plot_acc_loss(result,path):
    acc = result['acc']['train']
    loss = result['loss']['train']
    val_acc = result['acc']['valid']
    val_loss = result['loss']['valid']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.savefig(f"{path}\\Curbe de învățare")

def set_parameter_requires_grad(model, freeze):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def train(network, train_loader, valid_loader, criterion, opt, epochs, thresh=0.5, weights_dir='weights', save_every_ep=10):
   
    total_loss = {'train': [], 'valid': []}
    total_acc = {'train': [], 'valid': []}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device {device} ...")

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    metric = torchmetrics.Accuracy()

    network.to(device)
    criterion.to(device)

    for ep in range(epochs):
        

        print(f"[INFO] Epoch {ep}/{epochs - 1}")
        
        print("-" * 20)        
        for phase in ['train', 'valid']:
            running_loss = 0.0

            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()   # Set model to evaluate mode

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs = data
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    #print (ins.size(),tgs.size())
                    # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
                    opt.zero_grad() 

                    with torch.set_grad_enabled(phase == 'train'):
                        # se face forward propagation -> se calculeaza predictia
                        output = network(ins)
                        #print(tgs.size())
                        #print(output.size())
                     
                        # second_output = Variable(torch.argmax(output,1).float(),requires_grad=True).cuda()
                        # output[:, 1, :, :] => 8 x 1 x 128 x 1288
                        # tgs => 8 x 1 x 128 x 128
                        # tgs.squeeze() => 8 x 128 x 128
                        
                        # se calculeaza eroarea/loss-ul
                        loss = criterion(output, tgs.squeeze())
                        
                        # deoarece reteaua nu include un strat de softmax, predictia finala trebuie calculata manual
                        current_predict = F.softmax(output, dim=1)[:, 1].float()
                        current_predict[current_predict >= thresh] = 1.0
                        current_predict[current_predict < thresh] = 0.0

                        if 'cuda' in device.type:
                            current_predict = current_predict.cpu()
                            current_target = tgs.cpu().type(torch.int).squeeze()
                        else:
                            current_predict = current_predict
                            current_target = tgs.type(torch.int).squeeze()

                        # print(current_predict.shape, current_target.shape)
                        # print(current_predict.dtype, current_target.dtype)
                        acc = metric(current_predict, current_target)
                        # print(f"\tAcc on batch {i}: {acc}")

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()
                    
                    running_loss += loss.item() * ins.size(0)
                    # print(running_loss, loss.item())

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        if ep % save_every_ep == 0:
                            torch.save(network, f"{weights_dir}\\my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")
                        
                    #     model_path = f"{weights_dir}\\model_epoch{ep}.pth"
                    #     torch.save({'epoch': ep,
                    #                 'model_state_dict': network.state_dict(),
                    #                 'optimizer_state_dict': opt.state_dict(),
                    #                 'loss': total_loss,
                    #                 }, model_path) 
                        
                     
                    pbar.update(ins.shape[0])

 
                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(running_loss/len(loaders[phase].dataset))
                
                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                acc = metric.compute()
                total_acc[phase].append(acc)
            
                postfix = f'error {total_loss[phase][-1]:.4f} accuracy {acc*100:.2f}%'
                pbar.set_postfix_str(postfix)
                        
                # Resetam pt a acumula valorile dintr-o noua epoca
                metric.reset()
                         
    return {'loss': total_loss, 'acc': total_acc}

def main():
    

  directory =f"Experiment_dataset_mare{datetime.now().strftime('%m%d%Y_%H%M')}"
  parent_dir = os.getcwd()
  path = os.path.join(parent_dir, directory)
  os.mkdir(path)
  dir="Weights"
  path=os.path.join(path, dir)
  os.mkdir(path)
  

  print(f"pyTorch version {torch.__version__}")
  print(f"torchvision version {torchvision.__version__}")
  print(f"CUDA available {torch.cuda.is_available()}")

  config = None
  with open('config.yml') as f:
     config = yaml.safe_load(f)

 
  
  

  random_seed = 1
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  transforms_valid=T.Compose([
        T.Resize((config['net']['img'])),
        T.ToTensor(),
     ]) 
  transforms =  T.Compose([
        T.Resize((config['net']['img'])),
        T.RandomRotation(degrees=random.randint(0,360)),
        T.RandomInvert(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
       
               ])

  train_ds = dset.ImageFolder(config['dataset']['ds_path']+'/train',transform=transforms)
  valid_ds = dset.ImageFolder(config['dataset']['ds_path']+'/valid',transform=transforms_valid)


  train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=config["train"]["bs"])
  valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=config["train"]["bs"])
   
  print("Nr de imagini in setul de antrenare", len(train_ds))

  print("Dim primei imagini din Dataset", train_ds[0][0])
  print("Etichete pt prima imagine", train_ds[0][1])

  n_classes = len(np.unique(train_ds.targets))
  print(np.unique(train_ds.targets))


  print (n_classes)

  #network = CustomNet(3, config['net']['n1'], config['net']['n2'], config['net']['n3'], n_classes)
 # print(network)
  model = torchvision.models.vgg16(pretrained=True)
  set_parameter_requires_grad(model, freeze=True)
  num_ftrs = model.classifier[6].in_features
  model.classifier[6] = nn.Linear(num_ftrs, n_classes) 
  print(model)
  # Specificarea functiei loss
  criterion = nn.CrossEntropyLoss()

  # definirea optimizatorului
  if config['train']['opt'] == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
  elif config['train']['opt'] == 'SGD':
    opt = torch.optim.SGD(model.parameters(), lr=config['train']['lr'])
    
  
  history = train(model, train_loader, valid_loader, criterion, opt, epochs=config['train']['n_epochs'], weights_dir=path)
  plot_acc_loss(history,path)
  
  
 

if __name__ == "__main__":
    main()
