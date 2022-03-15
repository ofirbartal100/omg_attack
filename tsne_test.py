'''Main transfer script.'''

import hydra
import os
from dabs.src.datasets.catalog import DATASET_DICT
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset, IterableDataset, SubsetRandomSampler,Sampler
from tqdm import tqdm
import torch 

@hydra.main(config_path='conf', config_name='transfer_tsne')
def run(config):

    # Deferred imports for faster tab completion
    import pytorch_lightning as pl

    pl.seed_everything(config.trainer.seed)
    data,targets = getData(config)
    tsneNplot(data, targets,config.exp.name)


def tsneNplot(data, targets,desc):
    embeddings = TSNE(n_jobs=24,verbose=1,perplexity=200).fit_transform(data)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.scatter(vis_x, vis_y, c=targets, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(f'/disk2/ofirb/dabs/tsne_{desc}.png')



def getData(config):
    from dabs.src.systems import enc_transfer

    system = enc_transfer.TransferSystem(config)
    system.setup('tsne')
    system.eval()
    loader = DataLoader(system.train_dataset,batch_size=1,num_workers=config.dataset.num_workers,drop_last=True,pin_memory=True,shuffle=False)
        
    device = 4
    system.to(device)
    embedds,targets = [],[]
    for idx,data,labels in tqdm(loader):
        cu_data = data.to(device)
        cu_data = system.normalize(cu_data)
        # embd = system.model.forward([cu_data], prehead=True).detach().cpu()
        system.model.visualize_attn_heads([cu_data],data)
        # exit()
        # embedds.append(embd)
        # targets.append(labels)


    return torch.cat(embedds,dim=0).numpy() , torch.cat(targets,dim=0).numpy()


if __name__ == '__main__':
    run()