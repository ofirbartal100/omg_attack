import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from dabs.src.systems import viewmaker_original
from tqdm import tqdm
import os 
from torchvision.utils import save_image


# config = OmegaConf.load('/workspace/dabs/conf/traffic.yaml')
config = OmegaConf.load('/workspace/dabs/conf/ceva.yaml')
config.debug = True
# config.dataset = OmegaConf.load('/workspace/dabs/conf/dataset/traffic_sign_small.yaml')
config.dataset = OmegaConf.load('/workspace/dabs/conf/dataset/lfw112.yaml')
# config.model = OmegaConf.load('/workspace/dabs/conf/model/traffic_model.yaml')
config.model = OmegaConf.load('/workspace/dabs/conf/model/jit_model.yaml')

config.dataset.batch_size = 32

pl.seed_everything(config.trainer.seed)

print('loading VM...')
# system = viewmaker_original.TrafficViewMaker(config)
system = viewmaker_original.CevaViewmakerSystem(config)
system.setup('')
# system.load_state_dict(torch.load('/workspace/dabs/exp/models/traffic_gan/presentation.ckpt')['state_dict'],strict=False)
system.load_state_dict(torch.load('/workspace/dabs/exp/models/lfw_for_dataset_generation_b=0.015/epoch=80-step=90000.ckpt')['state_dict'],strict=False)

system.eval()
print('loading loader...')
# loader = system.val_dataloader()
loader = system.train_dataloader()
i=0
correct_src = 0
correct_views = 0
total_imgs = 0
num_views = 5
# root = '/workspace/dabs/data/natural_images/ceva_lfw_gen/15_11_2022/val'
root = '/workspace/dabs/data/natural_images/ceva_lfw_gen/15_11_2022/train'
label_counters ={}

def calc_views(img,orig_embeds):
    views1, unnormalized_view1 = system.view(img, True)
    unnormalized_view1 = torch.clamp(unnormalized_view1, 0, 1.0)
    views_embeds = system.model.forward([system.normalize(unnormalized_view1)])
    similarities = (orig_embeds * views_embeds).sum(1)/(orig_embeds.norm(dim=1)*views_embeds.norm(dim=1))
    return unnormalized_view1 , similarities
    


for index , img , labels in tqdm(loader):
    # correct_src += (system.model.traffic_model.forward_original(system.normalize(img)).max(1, keepdim=True)[1].flatten() == labels).sum()
    orig_embeds = system.model.forward([system.normalize(img)])
    views_similarities = [ calc_views(img,orig_embeds) for jj in range(num_views)]
    for i in range(len(img)):
        class_i = labels[i].item()
        path = os.path.join(root, f'{class_i}')
        if not os.path.exists(path):
            os.makedirs(path)

        if class_i in label_counters:
            label_counters[class_i] += 1
        else:
            label_counters[class_i] = 1
        save_image(img[i],f'{path}/{class_i}_{label_counters[class_i]:04d}_original.jpg')

        for jj in range(num_views):
            similarity = views_similarities[jj][1][i].item()
            view = views_similarities[jj][0][i]
            save_image(view,f'{path}/{class_i}_{label_counters[class_i]:04d}_view_{jj}_{similarity:.3f}.jpg')
            


# print((correct_src+0.0)/total_imgs ,(correct_views+0.0)/total_imgs ,total_imgs )



