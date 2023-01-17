import warnings
warnings.filterwarnings("ignore")
import copy
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from dabs.src.systems import viewmaker_original
from tqdm import tqdm
import os 
from torchvision.utils import save_image
import torch.nn.functional as F

import torchattacks as ta
import pandas as pd
dataset ='traffic'
part = 'val'
num_views = 1
attack = 'FGSM'




if dataset == 'traffic':
    conf_yaml = '/workspace/dabs/conf/traffic.yaml'
    conf_dataset_yaml = '/workspace/dabs/conf/dataset/traffic_sign_small.yaml'
    conf_model_yaml = '/workspace/dabs/conf/model/traffic_model.yaml'
    ckpt = '/workspace/dabs/exp/models/traffic_budget_budget=0.005/model.ckpt'
    systemClass = viewmaker_original.TrafficViewMaker
    batch_size = 32
    
    root = '/workspace/dabs/data/adv_data/traffic_sign/FGSM/'+part

    label_counters ={}

    def save_func(original,views_similarities,label, class_names):
        path = os.path.join(root, class_names[label])
        if not os.path.exists(path):
            os.makedirs(path)

        if label in label_counters:
            label_counters[label] += 1
        else:
            label_counters[label] = 0

        save_image(original,f'{path}/{label_counters[label]:05d}_original.jpg')

        for j in range(len(views_similarities)):
            unnormalized_view = views_similarities[j]
            save_image(unnormalized_view,f'{path}/{label_counters[label]:05d}_view_{j+1}.jpg')

elif dataset == 'lfw':
    conf_yaml = '/workspace/dabs/conf/ceva.yaml'
    conf_dataset_yaml = '/workspace/dabs/conf/dataset/lfw_112.yaml'
    conf_model_yaml = '/workspace/dabs/conf/model/ceva_model.yaml'
    ckpt = '/workspace/dabs/exp/models/traffic_budget_budget=0.005/model.ckpt'
    systemClass = viewmaker_original.CevaViewmakerSystem
    batch_size = 32
    
    root = '/workspace/dabs/data/adv_data/lfw/date/experiment_name/'+part

    label_counters ={}

    def save_func(original,views_similarities,label, class_names):
        path = os.path.join(root, class_names[label])
        if not os.path.exists(path):
            os.makedirs(path)

        if label in label_counters:
            label_counters[label] += 1
        else:
            label_counters[label] = 0

        save_image(original,f'{path}/{label_counters[label]:04d}_original.jpg')

        for j in range(len(views_similarities)):
            unnormalized_view, similarity = views_similarities[j]
            save_image(unnormalized_view,f'{path}/{label_counters[label]:04d}_view_{j+1}_sim_{similarity:.3f}.jpg')

elif dataset == 'birds':
    conf_yaml = '/workspace/dabs/conf/birds.yaml'
    conf_dataset_yaml = '/workspace/dabs/conf/dataset/cu_birds_small.yaml'
    conf_model_yaml = '/workspace/dabs/conf/model/birds_model.yaml'
    ckpt = '/workspace/dabs/exp/models/birds_dyn_sweep_budget=0.025/model.ckpt'
    systemClass = viewmaker_original.BirdsViewMaker
    batch_size = 12
    
    root = '/workspace/dabs/data/adv_data/cu_birds/FGSM/'+part

    label_counters ={}

    def save_func(original,views_similarities,label, class_names):
        path = os.path.join(root, class_names[label])
        if not os.path.exists(path):
            os.makedirs(path)

        if label in label_counters:
            label_counters[label] += 1
        else:
            label_counters[label] = 0

        save_image(original,f'{path}/{label_counters[label]:04d}_original.jpg')

        for j in range(len(views_similarities)):
            unnormalized_view = views_similarities[j]
            save_image(unnormalized_view,f'{path}/{label_counters[label]:04d}_view_{j+1}.jpg')


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
if part == 'train':
    loader = system.train_dataloader()
else :
    loader = system.val_dataloader()

if dataset =='birds':
    with open('/workspace/dabs/data/natural_images/cu_birds/CUB_200_2011/classes.txt', 'r') as f:
        image_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
    class_names = {(int(a[0])-1):a[1] for a in image_info}


    from types import MethodType
    def new_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x

    threat_model  = copy.deepcopy(system.model.birds_model)
    threat_model.forward = MethodType(new_forward, threat_model)

elif dataset =='traffic':
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
        output = F.log_softmax(x, dim=1)
        return output.data.max(1, keepdim=True)[1]

    threat_model  = copy.deepcopy(system.model.traffic_model)
    threat_model.forward = MethodType(forward_original, threat_model)

else:
    class_names = list(loader.dataset.class_to_index.keys()) # map between label index and class name


if attack == 'FGSM':
    atk = ta.FGSM(threat_model, eps=0.005) #, alpha=2/255, steps=4)
elif attack == 'PGD':
    atk = ta.PGD(threat_model, eps=0.025, alpha=0.025/8, steps=10)

atk.set_normalization_used(mean=loader.dataset.dataset.MEAN, std=loader.dataset.dataset.STD)
    


def calc_views(img,orig_embeds):
    views1, unnormalized_view1 = system.view(img.unsqueeze(0), True)
    unnormalized_view1 = torch.clamp(unnormalized_view1, 0, 1.0).squeeze()
    views_embeds = system.model.forward([system.normalize(unnormalized_view1)]).squeeze()
    similarities = (orig_embeds * views_embeds).sum()/(orig_embeds.norm()*views_embeds.norm())
    return unnormalized_view1 , similarities
    

results = []


img_cntr = 0
for index , img , labels in tqdm(loader):
    img = img.cuda()
    orig_embeds = system.model.forward([system.normalize(img)])
    for i in range(len(img)):
        img_cntr+=1
        views_similarities = [ calc_views(img[i],orig_embeds[i]) for jj in range(num_views)]
        class_i = labels[i].item()
        
        # original = img[i].cpu()
        views_similarities = [ v for v,s in views_similarities]
        
        view_num = 0
        for view in views_similarities:
            view_num+=1
            a = threat_model(system.normalize(img[0].unsqueeze(0)))
            b = threat_model(system.normalize(view.unsqueeze(0)))
            res = {
                'img_cntr': img_cntr,
                'view_num': view_num,
                'class': class_i,
                'original_success':  torch.squeeze(a == class_i).item(),
                'consistensy': torch.squeeze(a == b).item(),
                'view_success': torch.squeeze(b==class_i).item() 
            }

            results.append(res)
        
        # save_func(original,views_similarities,class_i,class_names)
            

df = pd.DataFrame(results)
