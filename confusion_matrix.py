# %%
import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from dabs.src.systems import viewmaker_original
from tqdm import tqdm
import os 
from torchvision.utils import save_image
import torch.nn.functional as F

import torchattacks as ta

# %%
conf_yaml = '/workspace/dabs/conf/traffic.yaml'
conf_dataset_yaml = '/workspace/dabs/conf/dataset/traffic_sign_small.yaml'
conf_model_yaml = '/workspace/dabs/conf/model/traffic_model.yaml'
ckpt = '/workspace/dabs/exp/models/traffic_budget_budget=0.005/model.ckpt'
systemClass = viewmaker_original.TrafficViewMaker
batch_size = 32

# %%
print('loading config...')
config = OmegaConf.load(conf_yaml)
config.dataset = OmegaConf.load(conf_dataset_yaml)
config.model = OmegaConf.load(conf_model_yaml)
config.dataset.batch_size = batch_size
config.debug = True
pl.seed_everything(config.trainer.seed)

print('loading VM...')
if 'model.ckpt' in ckpt:
    system = torch.load(ckpt)
else:
    system = systemClass(config)
    system.setup('')
    system.load_state_dict(torch.load(ckpt)['state_dict'],strict=False)

system.cuda()
system.eval()


print('loading loader...')
part = 'val'
if part == 'train':
    loader = system.train_dataloader()
else :
    loader = system.val_dataloader()

# %%
class_names = [str(i) for i in range(43)] # map between label index and class name
   
from types import MethodType
def forward_original(self, x):
    # transform the input
    x = self.stn(x)

    # Perform forward pass
    x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
    x = self.conv_drop(x)
    x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
    x = self.conv_drop(x)
    x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
    x = self.conv_drop(x)
    x = x.view(-1, 250*2*2)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

system.model.traffic_model.forward = MethodType(forward_original, system.model.traffic_model)
threat_model  = system.model.traffic_model
# state_dict = torch.load('/workspace/gtsrb_pytorch/model_no_aug.pth')
# threat_model.load_state_dict(state_dict)
# threat_model.eval()

# from torchvision import datasets, transforms
# from gtsrb_pytorch.data import *
# loader = val_loader = torch.utils.data.DataLoader(datasets.ImageFolder('/workspace/gtsrb_pytorch/data/val_images',transform=data_transforms), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
# MEAN,STD = np.array([0.3337, 0.3064, 0.3171],dtype=np.float32), np.array([ 0.2672, 0.2564, 0.2629],dtype=np.float32)
class_names = [str(i) for i in range(43)] # map between label index and class name

# %%

df_dict = {
    "label" : [],
    "pred" : [],
    "adv_pred": []
}

for index , img , labels in tqdm(loader):
    with torch.no_grad():
        img = img.cuda()
        views1, _ = system.view(img, True)
        pred = threat_model(system.normalize(img)).argmax(1, keepdim = True)
        adv_pred = threat_model(views1).argmax(1, keepdim = True)

        df_dict['label'].append(labels.unsqueeze(-1).cpu())
        df_dict['pred'].append(pred.cpu())
        df_dict['adv_pred'].append(adv_pred.cpu())


df_dict['label'] = torch.vstack(df_dict['label']).squeeze().numpy()
df_dict['pred'] = torch.vstack(df_dict['pred']).squeeze().numpy()
df_dict['adv_pred'] = torch.vstack(df_dict['adv_pred']).squeeze().numpy()

df = pd.DataFrame.from_dict(df_dict)



