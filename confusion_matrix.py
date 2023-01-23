
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
import numpy as np
import torchattacks as ta
from torchvision import models, transforms
import pandas as pd
from torch import nn
from adversarial_examples_pytorch.adv_gan import target_models
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from types import MethodType
from adversarial_examples_pytorch.adv_gan.generators import Generator_MNIST as Generator
from MNIST.example import Model_C, MnistNet
from gtsrb_pytorch.model import Net as TrafficNET



def load_attack_fun(system_ckpt,attack_type,attack_dict):

    system = torch.load(system_ckpt)
    system.cuda()
    system.eval()

    if attack_type=='vm':
        def vm_attack(img):
            view,unnormalized_view = system.view(img,True)
            return unnormalized_view

        attack_fun = vm_attack

    elif attack_type=='advgan':
        # ### advGAN inference
        thresh=attack_dict['thresh']
        # load corresponding generator
        G = Generator()
        checkpoint_name_G = '%s_untargeted.pth.tar'%(attack_dict['threat_model_name'])
        checkpoint_path_G = os.path.join('/workspace/adversarial_examples_pytorch/adv_gan/saved', 'generators', 'bound_%.1f'%(thresh), checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval().cuda()

        def advgan_attack(x):
            pert = G(x).data.clamp(min=-thresh, max=thresh)
            x_adv = x + pert
            x_adv = x_adv.clamp(min=0, max=1)
            # views1 = system.normalize(x_adv)
            return x_adv
        
        attack_fun = advgan_attack



    elif attack_type=='fgsm':
        atk = ta.FGSM(attack_dict['threat_model'], eps=attack_dict['thresh'])
        # atk = ta.PGD(threat_model, eps=0.035, alpha=0.035/8, steps=10)
        # atk.set_normalization_used(system.dataset.dataset.MEAN,system.dataset.dataset.STD)

        def fgsm_attack(img):
            # return atk(system.normalize(img),labels)
            return atk(img,labels)

        attack_fun = fgsm_attack

    
    return attack_fun , system


def load_model(dataset,model_type,model_path):
    if dataset == 'traffic':
        threat_model  = TrafficNET()
        threat_model.forward = threat_model.forward_original
        # MEAN,STD = np.array([0.3337, 0.3064, 0.3171],dtype=np.float32), np.array([ 0.2672, 0.2564, 0.2629],dtype=np.float32)

    elif dataset == 'birds':
        model = models.resnet18(pretrained = True)
        IN_FEATURES = model.fc.in_features 
        OUTPUT_DIM = 200

        fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
        model.fc = fc
        threat_model  = model
        # MEAN = [0.483, 0.491, 0.424]
        # STD  = [0.228, 0.224, 0.259]

    elif dataset == 'mnist':

        if model_type == 'Model_C':
            threat_model = getattr(target_models, model_type)(1, 10)
        
        elif model_type == 'Model_C_80%':
            threat_model = getattr(target_models, model_type.replace("_80%",""))(1, 8)
        
        elif model_type == 'resnet18':
            threat_model = models.resnet18(pretrained=False)
            threat_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
            num_ftrs = threat_model.fc.in_features
            threat_model.fc = nn.Linear(num_ftrs, 10)
        
        # MEAN = [0] #[0.1307]
        # STD  = [1] #[0.3081]


    threat_model.load_state_dict(torch.load(model_path))
    return threat_model.eval().cuda()

# run_config={
#     'dataset' : 'mnist',
#     'system_ckpt':'/workspace/dabs/exp/models/mnist_sweep_2023-01-21_06:32:36/model.ckpt',
#     'threat_model_name': 'Model_C',
#     # 'threat_model_path': '/workspace/MNIST/model_c_not_normalized.pt',
#     'threat_model_path': '/workspace/MNIST/model_c_not_normalized_defended.pt',
#     # 'attack':'vm',
#     # 'attack':'advgan',
#     'attack':'fgsm',
#     'attack_config':{
#         'num_views': 3,
#         'threat_model_name': 'Model_C',
#         'thresh':0.3,
#         'threat_model_path': '/workspace/MNIST/model_c_not_normalized.pt',
#     }
# }

# run_config={
#     'dataset' : 'traffic',
#     'system_ckpt':'/workspace/dabs/exp/models/traffic_budget_budget=0.005/model.ckpt',
#     'threat_model_name': 'stn-cnn',
#     # 'threat_model_path': '/workspace/gtsrb_pytorch/model/model_40.pth',
#     'threat_model_path': '/workspace/gtsrb_pytorch/model_defended.pth',
#     # 'attack':'vm',
#     # 'attack':'advgan',
#     'attack':'fgsm',
#     'attack_config':{
#         'num_views': 3,
#         'threat_model_name': 'stn-cnn',
#         'thresh':0.005,
#         'threat_model_path': '/workspace/gtsrb_pytorch/model/model_40.pth',
#     }
# }

run_config={
    'dataset' : 'birds',
    'system_ckpt':'/workspace/dabs/exp/models/birds_dyn_sweep_budget=0.025/model.ckpt',
    'threat_model_name': 'resnet18',
    # 'threat_model_path': '/workspace/cubirds/resnet18-224.pt',
    'threat_model_path': '/workspace/cubirds/resnet18-adv.pt',
    # 'attack':'vm',
    # 'attack':'advgan',
    'attack':'fgsm',
    'attack_config':{
        'num_views': 3,
        'threat_model_name': 'resnet18',
        'thresh':0.025,
        'threat_model_path': '/workspace/cubirds/resnet18-224.pt',
    }
    
}



threat_model = load_model(run_config['dataset'],run_config['threat_model_name'],run_config['threat_model_path'])

if run_config['attack'] == 'fgsm':
    run_config['attack_config']['threat_model'] = load_model(run_config['dataset'],run_config['attack_config']['threat_model_name'],run_config['attack_config']['threat_model_path'])

attack_fun, system = load_attack_fun(run_config['system_ckpt'],run_config['attack'],run_config['attack_config']) # gets unnormalize image, return unnormalize adversarial image


pl.seed_everything(123654)


df_dict = {
    "label" : [],
    "pred" : [],
    "adv_pred": []
}

loaders = [system.train_dataloader(), system.val_dataloader()]

for loader in loaders:
    for index , img , labels in tqdm(loader):
        # with torch.no_grad():
        img = img.cuda()
        for i in range(run_config['attack_config']['num_views']):
            adv_img = attack_fun(img) # unnormalized in , unnormalized out
            pred = threat_model(system.normalize(img)).argmax(1, keepdim = True)
            adv_pred = threat_model(system.normalize(adv_img)).argmax(1, keepdim = True)

            df_dict['label'].append(labels.unsqueeze(-1).cpu())
            df_dict['pred'].append(pred.cpu())
            df_dict['adv_pred'].append(adv_pred.cpu())

      

df_dict['label'] = torch.vstack(df_dict['label']).squeeze().numpy()
df_dict['pred'] = torch.vstack(df_dict['pred']).squeeze().numpy()
df_dict['adv_pred'] = torch.vstack(df_dict['adv_pred']).squeeze().numpy()

df = pd.DataFrame.from_dict(df_dict)

consistency_acc = (df['pred'] == df['adv_pred']).mean()
adv_acc = (df['label'] == df['adv_pred']).mean()
acc = (df['pred'] == df['label']).mean()

print(f'Original Acc={100*acc:.2f}% , Adv Acc={100*adv_acc:.2f}% (ASR={100*(1-adv_acc):.2f}) , Consistency Acc={100*consistency_acc:.2f}(ASR={100*(1-consistency_acc):.2f})')
print(run_config)
