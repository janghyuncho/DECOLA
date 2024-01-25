# **DECOLA** Model Zoo

In all our experiments, we used 8 Quadro RTX 6000 and 8 V100 GPUs. 

#### How to read the tables

The "config" column contains a link to the config file. 
To train a model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

#### Third-party ImageNet-21K pre-trained models

Our paper uses ImageNet-21K pretrained models that are not part of Detectron2 (ResNet-50-21K from [MIIL](https://github.com/Alibaba-MIIL/ImageNet21K) and SwinB-21K from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)). Before training, 
please download the models and place them under `DECOLA_ROOT/weights/`, and following [this tool](../tools/convert-thirdparty-pretrained-model-to-d2.py) to convert the format.

#### **DECOLA** and baselines
Here we provide the configs and checkpoints of DECOLA and Detic as our main baseline. 
Please refer to [Detic](https://github.com/facebookresearch/Detic/tree/main) to learn about it. 
The baseline is trained on *detection dataset* (LVIS-base or LVIS) for 4x and further trained on weak dataset (ImageNet-21K) for another 4x. 
DECOLA is trained on the same detection dataset with [*language condition*](https://github.com/janghyuncho/DECOLA/blob/main/docs/TRAINING.md#phase-1-decola-for-language-conditioning) for 4x (phase 1) and finetuned on the same weak dataset for another 4x (phase 2). 
For more training detail, please see [training details](TRAINING.md).

## Open-vocabulary LVIS with Deformable DETR

### ResNet-50 backbone
| name          | box AP_novel | box AP_c | box AP_f | box mAP | model |
|--------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [baseline](../configs/BoxSup-DeformDETR_Lbase_CLIP_R5021k_4x.yaml)       | 9.4 | 33.8 | 40.4 | 32.2 |  [weight](https://utexas.box.com/shared/static/ubry0bcodnd4y59zjatlkpej2bf2eaqr.pth) |
| [baseline + self-train](../configs/BoxSup+ST-DeformDETR_LbaseI_CLIP_R5021k_4x_ft4x.yaml)   | 23.2 | 36.5 | 41.6 | 36.2 |  [weight](https://utexas.box.com/shared/static/jditk3ofcguxvx5ff8zmvwv8aeveee9n.pth) |
| DECOLA    [[Phase 2](../configs/DECOLA_PHASE2_LbaseI_CLIP_R5021k_4x_ft4x.yaml)] | 27.6 | 38.3 | 42.9 | 38.3 |  [weight](https://utexas.box.com/shared/static/t1sos72n3582tqhuy7cw6rtre1p0sdfy.pth) |

### Swin-B backbone
| name      |  box AP_novel | box AP_c | box AP_f | box mAP | model |
|------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------|  
| [baseline](../configs/BoxSup-DeformDETR_Lbase_CLIP_SwinB_4x.yaml)        | 16.2 | 43.8 |49.1 | 41.1 |  [weight](https://utexas.box.com/shared/static/yz05g2x4bsc8q3avv6bskubi22266ca2.pth) |
| [baseline + self-train](../configs/BoxSup+ST-DeformDETR_LbaseI_CLIP_SwinB_4x_ft4x.yaml)   | 30.8 | 43.6 | 45.9 | 42.3 |  [weight](https://utexas.box.com/shared/static/d9vftqopdo56lobunqwrmo34dj5mh917.pth) |
| DECOLA   [[Phase 2](../configs/DECOLA_PHASE2_LbaseI_CLIP_SwinB_4x_ft4x.yaml)] | 35.7 | 47.5 | 49.7 | 46.3 |  [weight](https://utexas.box.com/shared/static/g5qw22h02no3b3h0dyw0hdpxmuiactwa.pth) |

### Swin-L backbone (w/ O365)
| name           |  box AP_novel | box AP_c | box AP_f | box mAP | model |
|---------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------|  
| [baseline](../configs/BoxSup-DeformDETR_Lbase_CLIP_SwinB_4x.yaml)        | 21.9 | 53.3 | 57.7 | 49.6 | [weight](https://utexas.box.com/shared/static/d8ej3y9wyemayx0gvlpnli44d9q39ycp.pth) |
| [baseline + self-train](../configs/BoxSup+ST-DeformDETR_LbaseI_CLIP_SwinB_4x_ft4x.yaml)   | 36.5 | 53.5 | 56.5 | 51.8 |  [weight](https://utexas.box.com/shared/static/ytox0hpx411lgs63vcudg03diov63551.pth) |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_LbaseI_CLIP_SwinB_4x_ft4x.yaml)] | 46.9 | 56.0 | 58.0 | 55.2 |  [weight](https://utexas.box.com/shared/static/0ftx2ywoanv8rw7r8jj1iuifvvs212zv.pth) |


## Standard LVIS with Deformable DETR

### ResNet-50 backbone
| name           | box AP_rare | box AP_c | box AP_f | box mAP | model |
|---------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [baseline](../configs/BoxSup-DeformDETR_L_CLIP_R5021k_4x.yaml)        | 26.3 | 34.1 | 41.3 | 35.6 | [weight](https://utexas.box.com/shared/static/2n608myne7ou0nim3ay3lv0vycm770yd.pth) |
| [baseline + self-train](../configs/Detic_DeformDETR_LI_CLIP_R5021k_4x_ft4x.yaml)   | 30.0 | 35.3 | 41.0 | 36.6 |  [weight](https://utexas.box.com/shared/static/ub5uwggzwqq288pagdpifzxqdgw3benh.pth) |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_LI_CLIP_R5021k_4x_ft4x.yaml)] | 34.8 | 38.7 | 42.5 | 39.6 |  [weight](https://utexas.box.com/shared/static/t12ua5ixg92geoyocwyng0khvopvyh45.pth) |
| DECOLA [[Phase 2 (offline)](../configs/DECOLA_PHASE2_LI_CLIP_R5021k_4x_OFFLINE_ft4x.yaml)] | 35.9 | 38.0 | 42.4 | 39.4 |  [weight](https://utexas.box.com/shared/static/x28jn9sypl9t26d4052d8pyq1xmt4dap.pth) |


### Swin-B backbone
| name           | box AP_rare | box AP_c | box AP_f | box mAP  | model |
|-------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [baseline](../configs/BoxSup-DeformDETR_L_CLIP_SwinB_4x.yaml)        | 38.3 | 43.4 | 48.6 | 44.5 |  [weight](https://utexas.box.com/shared/static/6rqosjkvp3sp13jnkubu8xltbz1r7v94.pth) |
| [baseline + self-train](../configs/BoxSup+ST-DeformDETR_LI_CLIP_SwinB_4x_ft4x.yaml)   | 42.0 | 44.0 | 48.1 | 45.2 |  [weight](https://utexas.box.com/shared/static/jbmuxjhqt3jy1crhqye2lbxu0g10818i.pth) |
| DECOLA  [[Phase 2](../configs/DECOLA_PHASE2_LI_CLIP_SwinB_4x_ft4x.yaml)] | 46.4 | 46.9 | 49.4 | 47.8 |  [weight](https://utexas.box.com/shared/static/pziu6s5hmwuqnivet90ad2k9t7nw8suy.pth) |
| DECOLA  [[Phase 2 (offline)](../configs/DECOLA_PHASE2_LI_CLIP_SwinB_4x_OFFLINE_ft4x.yaml)] | 47.4 | 47.4 | 49.6 | 48.3 |  [weight](https://utexas.box.com/shared/static/l2iba0tofg7pbx6etwer45jpvz2omgqc.pth) |

## Open-vocabulary LVIS with CenterNet2 
For DECOLA training, we use pseudo-labels generated from Phase 1 DECOLA([R50](https://github.com/janghyuncho/DECOLA/blob/main/docs/MODEL_ZOO.md#resnet-50-backbone-4), [SwinB](https://github.com/janghyuncho/DECOLA/blob/main/docs/MODEL_ZOO.md#swin-b-backbone-4)) trained on LVIS-base. See [here](https://github.com/janghyuncho/DECOLA/blob/main/docs/TRAINING.md#offline-self-labeling-and-training) to learn about how to generate pseudo-labels.


### ResNet-50 backbone
| name           | box AP_novel | box mAP | mask AP_novel | mask mAP | model |
|--------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [Detic-base](https://github.com/facebookresearch/Detic/blob/main/configs/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.yaml)         | 17.6 | 33.8 | 16.4 | 30.2 | [weight](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth) |
| [Detic](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml)    | 26.7 | 36.3 | 24.6 | 32.4 | [weight](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LbaseI_CLIP_R5021k_640b64_ft4x.yaml)] | 29.0 | 37.6 | 26.8 | 33.6 | [weight](https://utexas.box.com/shared/static/o02o305iczlm4atzvtl7lm48l7ztppk3.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LbaseI_CLIP_R5021k_640b64_ft4x+Detic.yaml)] | 29.5 | 37.7 | 27.0 | 33.7 | [weight](https://utexas.box.com/shared/static/les9gtjvdufqvgfh5ozxyv9nv50j63ow.pth) |


### Swin-B backbone
| name           | box AP_novel | box mAP | mask AP_novel | mask mAP | model |
|------------------|:----------:|:-------------:|:--------------:|:------------------:|-------|  
| [Detic-base](https://github.com/facebookresearch/Detic/blob/main/configs/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.yaml)         | 24.6 | 43.0 | 21.9 | 38.4 | [weight](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.pth) |
| [Detic](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml)    | 36.6 | 45.7 | 33.8 | 40.7 | [weight](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LbaseI_CLIP_SwinB_896b32_ft4x+Detic.yaml)] | 38.4 | 46.7 | 35.3 | 42.0 | [weight](https://utexas.box.com/shared/static/34dx5g2flhvnnqnw2s73y8jjod2wi94j.pth) |


- *NOTE: `baseline` and `Detic` weights are directly from [Detic's Model-Zoo](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md).*

## Direct zero-shot transfer to LVIS minival

| name | backbone  | data| AP_r | AP_c | AP_f | mAP_fixed | model |
|:--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_O365_CLIP_SwinT.yaml)] | Swin-T | O365 |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_O365IN21k_CLIP_SwinT.yaml)] | Swin-T | O365, IN21K | 32.8 | 32.0 | 31.8 | 32.0 | [weight](https://utexas.box.com/shared/static/rfa041u6i3lx07lz49az051b8xp11ocr.pth) |
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_O365_CLIP_SwinL.yaml)]| Swin-L | O365  |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_O365_OIIN21k_CLIP_SwinL.yaml)]| Swin-L | O365, OID, IN21K | 41.5 | 38.0 | 34.9 | 36.8 | [weight](https://utexas.box.com/shared/static/2k40r5ms1prl6mezbukeyysh2nsw3dzu.pth) |


## Direct zero-shot transfer to LVIS v1.0

| name | backbone  | data| AP_r | AP_c | AP_f | mAP_fixed | model |
|:------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_O365_CLIP_SwinT.yaml)]| Swin-T | O365 | - |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_O365IN21k_CLIP_SwinT.yaml)] | Swin-T | O365, IN21K | 27.2 | 24.9 | 28.0 | 26.6 | [weight](https://utexas.box.com/shared/static/rfa041u6i3lx07lz49az051b8xp11ocr.pth) |
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_O365_CLIP_SwinL.yaml)]| Swin-L | O365 | - |
| DECOLA [[Phase 2](../configs/DECOLA_PHASE2_O365_OIIN21k_CLIP_SwinL.yaml)]| Swin-L | O365, OID, IN21K | 32.9 | 29.1 | 30.3 | 30.2 | [weight](https://utexas.box.com/shared/static/2k40r5ms1prl6mezbukeyysh2nsw3dzu.pth) |


## Standard LVIS with CenterNet2 
For DECOLA training, we use pseudo-labels generated from Phase 1 DECOLA([R50](https://github.com/janghyuncho/DECOLA/blob/main/docs/MODEL_ZOO.md#resnet-50-backbone-4), [SwinB](https://github.com/janghyuncho/DECOLA/blob/main/docs/MODEL_ZOO.md#swin-b-backbone-4)) trained on LVIS.

### ResNet-50 backbone
| name           | box AP_rare | box mAP | mask AP_rare | mask mAP | model |
|-----------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [Detic-base](https://github.com/facebookresearch/Detic/blob/main/configs/BoxSup-C2_L_CLIP_R5021k_640b64_4x.yaml)         | 28.2 | 35.3 | 25.6 | 31.4 | [weight](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_L_CLIP_R5021k_640b64_4x.pth) |
| [Detic](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml)    | 31.4 | 36.8 | 29.7 | 33.2 | [weight](https://dl.fbaipublicfiles.com/detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LI_CLIP_R5021k_640b64_ft4x.yaml)] | 35.6 | 38.6 | 32.1 | 34.4 | [weight](https://utexas.box.com/shared/static/litahdtnkf0u89d2ani47z9o21hkkpj8.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LI_CLIP_R5021k_640b64_ft4x+Detic.yaml)] | 35.4 | 38.3 | 32.1 | 34.2 | [weight](https://utexas.box.com/shared/static/89o96f5ll7o464jvle2d7cidwmcnl9w1.pth) |

### Swin-B backbone
| name           | box AP_rare | box mAP | mask AP_rare | mask mAP | model |
|-------------------------------|:----------:|:-------------:|:--------------:|:------------------:|-------| 
| [Detic-base](https://github.com/facebookresearch/Detic/blob/main/configs/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml)         | 39.9 | 45.4 | 35.9 | 40.7 | [weight](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth) |
| [Detic](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml)    | 45.8 | 46.9 | 41.7 | 41.7 | [weight](https://dl.fbaipublicfiles.com/detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) |
| DECOLA label [[config](../configs/DECOLA_C2_LI_CLIP_SwinB_896b32_ft4x+Detic.yaml)]      | 46.6 | 48.3 | 42.3 | 43.4 | [weight](https://utexas.box.com/shared/static/8ls60yfw0km4ixeteq4a1jigud5yttwv.pth) |

- *NOTE: `baseline` and `Detic` weights are directly from [Detic's Model-Zoo](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md).*

## **DECOLA** phase 1 on *conditioned*-mAP (c-mAP)
Here, we provide the DECOLA checkpoints in phase 1 training (language-condition). The main evaluation metric for these models as well as standard detector (*baseline*) is c-mAP@k, where `k` is per-image detection limit. 

To evaluate a baseline model for c-mAP, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DETR.ORACLE_EVALUATION True TEST.DETECTIONS_PER_IMAGE $k
``` 
To evaluate a Phase 1 DECOLA model for c-mAP, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DECOLA.ORACLE_EVALUATION True MODEL.DECOLA.TEST_CLASS_CONDITIONED True TEST.DETECTIONS_PER_IMAGE $k
``` 

*Change `k` for different per-image detection limits.*

### ResNet-50 backbone
| name | data           | AP_r@10 | AP_r@20 | AP_r@50 | AP_r@100 | AP_r@300 | model |
|-------------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| [baseline](../configs/BoxSup-DeformDETR_Lbase_CLIP_R5021k_4x.yaml)  | LVIS-base  | 6.0  | 11.3 | 19.2 | 26.8 | 31.9 | [weight](https://utexas.box.com/shared/static/ubry0bcodnd4y59zjatlkpej2bf2eaqr.pth) |
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_Lbase_CLIP_R5021k_4x.yaml) ]  | LVIS-base       | 19.4 | 28.5 | 34.1 | 38.7 | 40.0 | [weight](https://utexas.box.com/shared/static/tn9i8w7tuz0elggris4pu4y8pbstf0jw.pth) |
| [baseline](../configs/BoxSup-DeformDETR_L_CLIP_R5021k_4x.yaml) | LVIS             | 21.3 | 29.4 | 36.9 | 41.1 | 44.6 | [weight](https://utexas.box.com/shared/static/2n608myne7ou0nim3ay3lv0vycm770yd.pth) |
| DECOLA [[Phase 1](../configs/DECOLA_PHASE1_L_CLIP_R5021k_4x.yaml) ]  | LVIS                | 26.6 | 39.1 | 45.2 | 47.1 | 48.8 | [weight](https://utexas.box.com/shared/static/5lsfcqjg4gxpzuc90lv5r1lquwl1cojb.pth) |


### Swin-B backbone
| name | data           | AP_r@10 | AP_r@20 | AP_r@50 | AP_r@100 | AP_r@300 | model |
|------------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| [baseline](../configs/BoxSup-DeformDETR_Lbase_CLIP_SwinB_4x.yaml) | LVIS-base    | 7.4 | 16.1 | 27.5 | 33.1 | 41.9  | [weight](https://utexas.box.com/shared/static/yz05g2x4bsc8q3avv6bskubi22266ca2.pth) |
| DECOLA  [[Phase 1](../configs/DECOLA_PHASE1_Lbase_CLIP_SwinB_4x.yaml)] | LVIS-base     | 21.9 | 32.0 | 40.0 | 44.0 | 47.7 | [weight](https://utexas.box.com/shared/static/o7wfuk5r3m4vesrness4ssk37x1aba99.pth) |
| [baseline](../configs/BoxSup-DeformDETR_L_CLIP_SwinB_4x.yaml)   | LVIS          | 30.1 | 38.2 | 45.5 | 49.3 | 53.2 | [weight](https://utexas.box.com/shared/static/6rqosjkvp3sp13jnkubu8xltbz1r7v94.pth) |
| DECOLA  [[Phase 1](../configs/DECOLA_PHASE1_L_CLIP_SwinB_4x.yaml) ] | LVIS                | 33.5 | 43.9 | 51.4 | 53.8 | 55.8 | [weight](https://utexas.box.com/shared/static/t751ml0zayrx89qrj2c22yzcaxqdxvvn.pth) |