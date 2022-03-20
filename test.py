 
from pathlib import Path
from tkinter.filedialog import SaveFileDialog, test
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
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix, accuracy_score,recall_score,f1_score,precision_score,plot_confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve
print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"CUDA available {torch.cuda.is_available()}\n")
import os 

def convert_prob(probs):
    converted_probs = []
    for i in range(len(probs)):
        if(probs[i] > 0.5):
            converted_probs.append(1)
        else:
            converted_probs.append(0)   
    return converted_probs    

def train(network,test_loader,test_ds):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Device ", device)

  print(type(test_ds.targets), type(test_ds.targets[0]))
  test_labels = test_ds.targets
  predictions = None

  for data in test_loader:
     ins, tgs = data
     ins = ins.to(device)

     current_predict = network(ins)
     current_predict = nn.Softmax(dim=1)(current_predict)
     # current_predict = current_predict.argmax(dim=1)

     if 'cuda' in device.type:
         current_predict = current_predict.detach().cpu().numpy()
     else:
         current_predict = current_predict.numpy()
    
     if predictions is None:
         predictions = current_predict # TEST BS x 2
     else:
         predictions = np.concatenate((predictions, current_predict), axis=0)
         
  converted_preds=convert_prob(predictions[:,1])
  acc = np.sum(np.array(converted_preds) == np.array(test_labels))/len(converted_preds)
  print(f'Test accuracy is {acc*100}')
  return converted_preds,test_labels,predictions 

def main():
  config = None
  with open('config.yml') as f:
      config = yaml.safe_load(f)

  directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
  parent_dir =r'D:\ai intro\Pytorch\Clasificare_py_torch\Experiment_dataset_mare03172022_1203' #Path(config[])
  path = os.path.join(parent_dir, directory)
  os.mkdir(path)

  save_config = f"{path}\\config.yaml"
  with open(save_config, 'w') as fp:
      yaml.dump(config, fp)
 
  test_bs = config["train"]["bs"]
  # incarcam ponderile modelul antrenat

  transforms = T.Compose([ 
          T.Resize((config['net']['img'])),
          T.ToTensor(), # converts a PIL.Image or numpy array into torch.Tensor
       
          # T.Normalize((0.1307,), (0.3081,)), # Normalize the dataset with mean and std specified
                 ])

  test_ds = dset.ImageFolder(config['dataset']['ds_path']+'/test',transform=transforms) 
  test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=test_bs)
  network = torch.load(config['test']['exp_path'])
  converted_preds, test_labels , predictions =train(network,test_loader,test_ds)
  
  fig=plt.figure()
  cm=confusion_matrix(test_labels, converted_preds)
  ConfusionMatrixDisplay.from_predictions(test_labels, converted_preds)
  plt.savefig(F"{path}\\Confusion_matrix")
  plt.figure()




  acc=accuracy_score(test_labels, converted_preds)
  preci=precision_score(test_labels, converted_preds)
  reca=recall_score(test_labels, converted_preds)
  F1=f1_score(test_labels, converted_preds)

    
  fpr1, tpr1, thresholds = roc_curve(test_labels, predictions[:, 1], pos_label=1)
  plt.plot(fpr1,tpr1, marker='.',color='C4')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  # plt.legend()
  plt.savefig(f"{path}\\ROC_Curve")
  plt.figure()
          
  auc=roc_auc_score(test_labels,predictions[:,1])


  precision, recall, thresholds = precision_recall_curve(test_labels, predictions[:,1])
  plt.plot(precision,recall, marker='.',color='C0')
  plt.xlabel('Precision')
  plt.ylabel('Recall')
  # plt.legend()
  plt.savefig(f"{path}\\Precision_Recall_Curve")

  with open(f"{path}\\metrics.txt","w+") as fp:
     fp.write("\n")
     fp.write("acc:")
     fp.write(acc.astype('str'))
     fp.write("\n")
     fp.write("PPV:")
     fp.write(preci.astype('str'))
     fp.write("\n")
     fp.write("Recall:")
     fp.write(reca.astype('str'))
     fp.write("\n")
     fp.write("F1:")
     fp.write(F1.astype('str'))
     fp.write("\n")
     fp.write("Auc:")
     fp.write(auc.astype('str'))
   
if __name__=="__main__":
    main()    




