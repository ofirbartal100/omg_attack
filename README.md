# OMG-ATTACK: Self-Supervised On-Manifold Generation for Transferable Adversarial Attacks

This repository contains the code for OMG-ATTACK, a self-supervised, computationally efficient, and transferable adversarial attack model.

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/b6597a51-01ba-4054-88c9-0e15d0136dde)

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/951881c5-a1a2-4c4d-ae68-311cf81389ad)

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/4054b176-4366-475a-a270-1655507e535a)




## Datasets

| Dataset | Link |
|:-----------------|:-----------------|
| MNIST|[Link](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)|
|Traffic Sign|[Link](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
|CU Birds|[Link](http://www.vision.caltech.edu/visipedia/CUB-200.html)|


## Results
Below are results for algorithms trained on each dataset in DABS. The baseline performance is obtained via a randomly initialized encoder.

### Attack success rate for different datasets, each dataset is paired with a threat model. Budgets are equal across methods to ensure a fair comparison between attacks.

| Dataset | Model    | PGD   | FGSM  | MI-FGSM | VMI-FGSM | AdvGAN | OMG   |
| ------- | -------- | ----- | ----- | ------- | -------- | ------ | ----- |
| MNIST   | Model C  | **99.87** | 87.48 | 99.86   | 99.87    | 89.27  | 99.37 |
| GTSRB   | STN-CNN  | 33.53 | 19.71 | 33.83   | 33.91    | 20.25  | **67.48** |
| CUB     | Resnet18 | **94.27** | 92.85 | 93.65   | 93.89    | 75.09  | 46.97 |


### ASR on defended models. The adversarial examples here were generated using the original threat models, from the white-box experiment. The defense strategy used to train the defended models is adversarial training using FGSM.

| Dataset | Def. Model | PGD   | FGSM  | MI-FGSM | VMI-FGSM | AdvGAN | OMG   |
| ------- | ---------- | ----- | ----- | ------- | -------- | ------ | ----- |
| MNIST   | ModelC     | 6.25  | 0.32  | 3.62    | 3.14     | 8.41   | **99.33** |
| GTSRB   | STN-CNN    | 16.75 | 8.71  | 17.41   | 17.65    | 6.99   | **64.58** |
| CUB     | Resnet18   | 12.65 | 14.58 | 19.35   | 25.43    | 4.75   | **38.94** |


### BlackBox Transferability to other models. We used the same augmentations that were generated for the white-box settings and tested their transferability on other stronger black-box target models. The reported numbers are ASR Percentage.

| Dataset | Model         | PGD   | FGSM  | MI-FGSM | VMI-FGSM | AdvGAN | OMG   |
| ------- | ------------- | ----- | ----- | ------- | -------- | ------ | ----- |
| MNIST   | Model A       | 76.98 | 55.84 | 80.65   | 84.15    | 82.64  | **99.29** |
|         | Model B       | 60.59 | 85.06 | 88.12   | 90.27    | 72.35  | **99.56** |
|         | Resnet18      | 70.91 | 56.43 | 72.49   | 73.60    | 72.32  | **89.65** |
| GTSRB   | Resnet50      | 10.74 | 8.70  | 11.65   | 11.76    | 9.01   | **58.89** |
| CUB     | Resnet50      | 31.12 | 37.90 | 45.84   | **54.20**    | 17.63  | 48.33 |
|         | WideResnet50  | 25.14 | 33.42 | 38.67   | 46.02    | 19.93  | **46.84** |


### Domain Transfer 80% to 20%, Train is the ASR percentage of the attack on the truncated dataset, Seen is the ASR percentage on the untruncated dataset but on the subset with only seen classes. Unseen is the ASR percentage on the subset with only unseen classes.

| Dataset | Train AdvGAN | Seen AdvGAN | Unseen AdvGAN | Train OMG | Seen OMG | Unseen OMG |
| ------- | ------------ | ----------- | ------------- | --------- | -------- | ---------- |
| MNIST   | 81.03        | 72.20       | 31.69         | 85.56     | 84.25    | **46.49**      |
| GTSR    | 20.55        | 12.89       | 4.13          | 69.19     | 65.54    | **60.61**      |
| CUB     | 31.11        | 24.21       | 24.42         | 66.98     | 63.90    | **70.09**      |

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/be42122d-c6b8-48b1-822f-d52dd8708f4e)

## Visual Results

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/c0de1c38-3aee-4ded-915a-116b3ba227e1)

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/ca568eb1-d159-4d99-972c-3aa21c9ed950)

![image](https://github.com/ofirbartal100/omg_attack/assets/23661390/f00cc534-9071-4c0d-abd9-716dc6adf2e7)

