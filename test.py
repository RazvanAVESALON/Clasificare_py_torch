
from pathlib import Path
from tkinter.filedialog import SaveFileDialog
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
print(f"CUDA available {torch.cuda.is_available()}")
import os 
config = None
with open('config.yml') as f:
    config = yaml.safe_load(f)


yml_data=yaml.dump(config)



directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
parent_dir =r"D:\ai intro\Pytorch\Clasificare_py_torch\Experiment1442_02162022"
path = os.path.join(parent_dir, directory)
os.mkdir(path)

f= open(f"{path}\\yaml_config.txt","w+")
f.write(yml_data)


# with open('config.yml','w') as f: 
#     yaml.dump(config,path)


test_bs = config["train"]["bs"]
# incarcam ponderile modelul antrenat

transforms = T.Compose([ 
        T.Resize((64,64)),
        T.ToTensor(), # converts a PIL.Image or numpy array into torch.Tensor
       
        # T.Normalize((0.1307,), (0.3081,)), # Normalize the dataset with mean and std specified
               ])

test_ds = dset.ImageFolder(config['net']['dir']+'/test',transform=transforms) 
test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=test_bs)


network = torch.load(r"D:\ai intro\Pytorch\Clasificare_py_torch\Experiment1442_02162022\Weights\my_model02162022_1448.pt")



# checkpoint = torch.load(r"D:\ai intro\Pytorch\Clasificare_py_torch\Experiment1335_02162022\Weights\model_epoch99.pth")

# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


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
fig=plt.figure()

cm=confusion_matrix(predictions,test_labels)
ConfusionMatrixDisplay.from_predictions(test_labels, predictions)
plt.savefig(F"{path}\\Confusion_matrix")
plt.figure()


acc=accuracy_score(predictions,test_labels)
preci=precision_score(predictions,test_labels)
reca=recall_score(predictions,test_labels)
F1=f1_score(predictions,test_labels)
f.write("\n")
f.write("acc:")
f.write(acc.astype('str'))
f.write("\n")
f.write("PPV:")
f.write(preci.astype('str'))
f.write("\n")
f.write("FPR:")
f.write(reca.astype('str'))
f.write("\n")
f.write("F1:")
f.write(F1.astype('str'))

fpr1, tpr1, thresholds = roc_curve(test_labels, predictions, pos_label=1)
plt.plot(fpr1,tpr1, marker='.', label='19',color='C4')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(f"{path}\\ROC_Curve")
plt.figure()
          
auc=roc_auc_score(test_labels,predictions)
f.write("\n")
f.write("Auc:")
f.write(auc.astype('str'))

precision, recall, thresholds = precision_recall_curve(test_labels, predictions)
plt.plot(precision,recall, marker='.', label='15',color='C0')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend()
plt.savefig(f"{path}\\Precision_Recall_Curve")





