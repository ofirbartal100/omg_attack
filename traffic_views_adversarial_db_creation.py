import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from dabs.src.systems import viewmaker_original
from tqdm import tqdm
import os 
from torchvision.utils import save_image


# config = OmegaConf.load('/workspace/dabs/conf/ceva.yaml')
# config.dataset = OmegaConf.load('/workspace/dabs/conf/dataset/lfw112.yaml')
# config.model = OmegaConf.load('/workspace/dabs/conf/model/jit_model.yaml')
# system = viewmaker_original.CevaViewmakerSystem(config)

dataset ='birds'
part = 'train'
num_views = 4

if dataset == 'traffic':
    conf_yaml = '/workspace/dabs/conf/traffic.yaml'
    conf_dataset_yaml = '/workspace/dabs/conf/dataset/traffic_sign_small.yaml'
    conf_model_yaml = '/workspace/dabs/conf/model/traffic_model.yaml'
    ckpt = '/workspace/dabs/exp/models/traffic_budget_budget=0.005/model.ckpt'
    systemClass = viewmaker_original.TrafficViewMaker
    batch_size = 32
    
    root = '/workspace/dabs/data/adv_data/traffic_sign/07_01_2023/traffic_budget_budget=0.005/'+part

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
            unnormalized_view, similarity = views_similarities[j]
            save_image(unnormalized_view,f'{path}/{label_counters[label]:05d}_view_{j+1}_sim_{similarity:.3f}.jpg')

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
    batch_size = 24
    
    root = '/workspace/dabs/data/adv_data/cu_birds/10_01_2023/birds_dyn_sweep_budget=0.025/'+part

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
else:
    class_names = list(loader.dataset.class_to_index.keys()) # map between label index and class name


def calc_views(img,orig_embeds):
    views1, unnormalized_view1 = system.view(img.unsqueeze(0), True)
    unnormalized_view1 = torch.clamp(unnormalized_view1, 0, 1.0).squeeze()
    views_embeds = system.model.forward([system.normalize(unnormalized_view1)]).squeeze()
    similarities = (orig_embeds * views_embeds).sum()/(orig_embeds.norm()*views_embeds.norm())
    return unnormalized_view1 , similarities
    

for index , img , labels in tqdm(loader):
    img = img.cuda()
    orig_embeds = system.model.forward([system.normalize(img)])
    for i in range(len(img)):
        views_similarities = [ calc_views(img[i],orig_embeds[i]) for jj in range(num_views)]
        class_i = labels[i].item()
        
        original = img[i].cpu()
        views_similarities = [ (v.cpu(),s.cpu()) for v,s in views_similarities]
        save_func(original,views_similarities,class_i,class_names)
            
