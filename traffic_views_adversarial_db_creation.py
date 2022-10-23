import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from dabs.src.systems import viewmaker_original
from tqdm import tqdm
import os 
from torchvision.utils import save_image


config = OmegaConf.load('/workspace/dabs/conf/traffic.yaml')
config.debug = True
config.dataset = OmegaConf.load('/workspace/dabs/conf/dataset/traffic_sign_small.yaml')
config.model = OmegaConf.load('/workspace/dabs/conf/model/traffic_model.yaml')

config.dataset.batch_size = 64

pl.seed_everything(config.trainer.seed)

print('loading VM...')
system = viewmaker_original.TrafficViewMaker(config)
system.setup('')
system.load_state_dict(torch.load('/workspace/dabs/exp/models/traffic_gan/presentation.ckpt')['state_dict'],strict=False)

system.eval()
print('loading loader...')
loader = system.val_dataloader()
i=0
correct_src = 0
correct_views = 0
total_imgs = 0
root = '/workspace/dabs/data/natural_images/traffic_sign'
label_counters ={}
for index , img , labels in tqdm(loader):
    correct_src += (system.model.traffic_model.forward_original(system.normalize(img)).max(1, keepdim=True)[1].flatten() == labels).sum()
    views1, unnormalized_view1 = system.view(img, True)
    unnormalized_view1 = torch.clamp(unnormalized_view1, 0, 1.0)
    correct_views += (system.model.traffic_model.forward_original(system.normalize(unnormalized_view1)).max(1, keepdim=True)[1].flatten() == labels).sum()
    total_imgs += img.shape[0]

    for i in range(len(img)):
        class_i = labels[i].item()
        if class_i in label_counters:
            label_counters[class_i] += 1
        else:
            label_counters[class_i] = 0

        img_i = unnormalized_view1[i]
        path = os.path.join(root, 'GTSRB', 'Validation_Adversarial_v0', 'Images', '{:05d}'.format(class_i))
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(img_i,f'{path}/{0:05d}_{label_counters[class_i]:05d}.ppm')


print((correct_src+0.0)/total_imgs ,(correct_views+0.0)/total_imgs ,total_imgs )



