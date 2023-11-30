# Analyzing **DECOLA**

We analyze the properties of DECOLA as well as standard detector. 
In this page, we provide inference code to reproduce our analyses in the paper. 

## *Conditioned* mAP 
We consider self-labeling mAP as the main evaluation metric for DECOLA's language-condition. 
Self-labeling mAP measures mAP while providing class names during inference but with low detection limit. 
DECOLA makes use of it [by conditioning](https://github.com/janghyuncho/DECOLA/blob/main/decola/modeling/decola/decola_zero_shot_classifier.py#L107) while *baseline* (standard open-vocabulary detector) [modifies the classification layer](https://github.com/janghyuncho/DECOLA/blob/main/decola/modeling/detic/d2_detic_deformable_detr.py#L335), which is the standard practice. 

Please use the following command to run DECOLA inference for self-labeling mAP 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DECOLA.ORACLE_EVALUATION True MODEL.DECOLA.TEST_PROB_A_OBJECT 0.0 TEST.DETECTIONS_PER_IMAGE $k
``` 
and the following for baseline
```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DETR.ORACLE_EVALUATION True TEST.DETECTIONS_PER_IMAGE $k
``` 
*Change `k` for different per-image detection limits.*

Below are the results:

### ResNet-50 backbone
| name | data |        config          | AP_novel@10 | AP_novel@20 | AP_novel@50 | AP_novel@100 | AP_novel@300 | model |
|------|----------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| baseline | LVIS-base | [BoxSup-DeformDETR_Lbase_CLIP_R5021k_4x](../configs/BoxSup-DeformDETR_Lbase_CLIP_R5021k_4x.yaml)   | 6.0  | 11.3 | 19.2 | 26.8 | 31.9 | [weight](https://utexas.box.com/shared/static/ubry0bcodnd4y59zjatlkpej2bf2eaqr.pth) |
| DECOLA   | LVIS-base | [DECOLA_PHASE1_Lbase_CLIP_R5021k_4x](../configs/DECOLA_PHASE1_Lbase_CLIP_R5021k_4x.yaml)       | 19.4 | 28.5 | 34.1 | 38.7 | 40.0 | [weight](https://utexas.box.com/shared/static/tn9i8w7tuz0elggris4pu4y8pbstf0jw.pth) |
| baseline | LVIS | [BoxSup-DeformDETR_L_CLIP_R5021k_4x](../configs/BoxSup-DeformDETR_L_CLIP_R5021k_4x.yaml)            | 21.3 | 29.4 | 36.9 | 41.1 | 44.6 | [weight](https://utexas.box.com/shared/static/2n608myne7ou0nim3ay3lv0vycm770yd.pth) |
| DECOLA   | LVIS | [DECOLA_PHASE1_L_CLIP_R5021k_4x](../configs/DECOLA_PHASE1_L_CLIP_R5021k_4x.yaml)                | 26.6 | 39.1 | 45.2 | 47.1 | 48.8 | [weight](https://utexas.box.com/shared/static/5lsfcqjg4gxpzuc90lv5r1lquwl1cojb.pth) |


### Swin-B backbone
| name | data |        config          | AP_novel@10 | AP_novel@20 | AP_novel@50 | AP_novel@100 | AP_novel@300 | model |
|------|----------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| baseline | LVIS-base | [BoxSup-DeformDETR_Lbase_CLIP_SwinB_4x](../configs/BoxSup-DeformDETR_Lbase_CLIP_SwinB_4x.yaml)   | 7.4 | 16.1 | 27.5 | 33.1 | 41.9  | [weight](https://utexas.box.com/shared/static/yz05g2x4bsc8q3avv6bskubi22266ca2.pth) |
| DECOLA   | LVIS-base | [DECOLA_PHASE1_Lbase_CLIP_SwinB_4x](../configs/DECOLA_PHASE1_Lbase_CLIP_SwinB_4x.yaml)       | 21.9 | 32.0 | 40.0 | 44.0 | 47.7 | [weight](https://utexas.box.com/shared/static/o7wfuk5r3m4vesrness4ssk37x1aba99.pth) |
| baseline | LVIS | [BoxSup-DeformDETR_L_CLIP_SwinB_4x](../configs/BoxSup-DeformDETR_L_CLIP_SwinB_4x.yaml)            | 30.1 | 38.2 | 45.5 | 49.3 | 53.2 | [weight](https://utexas.box.com/shared/static/6rqosjkvp3sp13jnkubu8xltbz1r7v94.pth) |
| DECOLA   | LVIS | [DECOLA_PHASE1_L_CLIP_SwinB_4x](../configs/DECOLA_PHASE1_L_CLIP_SwinB_4x.yaml)                | 33.5 | 43.9 | 51.4 | 53.8 | 55.8 | [weight](https://utexas.box.com/shared/static/t751ml0zayrx89qrj2c22yzcaxqdxvvn.pth) |

## Box-efficient Object Detector

We further study the box-efficiency of the object detectors. Specifically, we reduce the number of object queries and test the capability of detecting objects without producing too many boxes. Our goal is to avoid gaming on mAP metric and make it more convenient for downstream applications. 

Run the following command to evaluate DECOLA (phase 1)

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DECOLA.ORACLE_EVALUATION True MODEL.DECOLA.TEST_PROB_A_OBJECT 0.0 TEST.DETECTIONS_PER_IMAGE $k MODEL.DECOLA.PER_PROMPT_TOPK $n
``` 
and the following for baseline
```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth MODEL.DETR.ORACLE_EVALUATION True TEST.DETECTIONS_PER_IMAGE $k MODEL.DETR.PER_CLASS_NUM_QUERY $n
``` 
- *Change `k` for different per-image detection limits and `n` for different number of per-class query (different number of boxes).*

- *Here, `n` is the number of *per-prompt/class query*. The total number of object queries is `n x |Cx|`, where `Cx` is the set of object classes in image `x`.*

Below are the results:


### ResNet-50 backbone
| name | data |        config          | n=1 | n=2 | n=5 | n=10 | n=20 | model |
|------|----------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| baseline | LVIS-base | [BoxSup-DefromDETR_Lbase_CLIP_R5021k_4x](../configs/BoxSup-DefromDETR_Lbase_CLIP_R5021k_4x.yaml) | 14.7 | 22.4 | 27.6 | 30.9 | 32.2 | [weight](https://utexas.box.com/shared/static/ubry0bcodnd4y59zjatlkpej2bf2eaqr.pth) |
| DECOLA | LVIS-base | [DECOLA_PHASE1_Lbase_CLIP_R5021k_4x](../configs/DECOLA_PHASE1_Lbase_CLIP_R5021k_4x.yaml)       | 25.2 | 31.4 | 36.0 | 37.9 | 39.9 | [weight](https://utexas.box.com/shared/static/tn9i8w7tuz0elggris4pu4y8pbstf0jw.pth) |
| baseline | LVIS | [BoxSup-DefromDETR_L_CLIP_R5021k_4x](../configs/BoxSup-DefromDETR_L_CLIP_R5021k_4x.yaml)          | 17.8 | 24.8 | 33.0 | 38.7 | 42.3 | [weight](https://utexas.box.com/shared/static/2n608myne7ou0nim3ay3lv0vycm770yd.pth) |
| DECOLA | LVIS | [DECOLA_PHASE1_L_CLIP_R5021k_4x](../configs/DECOLA_PHASE1_L_CLIP_R5021k_4x.yaml)                | 29.7 | 36.7 | 41.8 | 45.9 | 48.3| [weight](https://utexas.box.com/shared/static/5lsfcqjg4gxpzuc90lv5r1lquwl1cojb.pth) |


### Swin-B backbone
| name | data |        config          | n=1 | n=2 | n=5 | n=10 | n=20 | model |
|------|----------------|--------|:----------:|:-------------:|:--------------:|:------------------:|:-------:|--|
| baseline | LVIS-base | [BoxSup-DefromDETR_Lbase_CLIP_SwinB_4x](../configs/BoxSup-DefromDETR_Lbase_CLIP_SwinB_4x.yaml) | 17.8 | 26.0 | 33.7 | 37.6 | 40.9 | [weight](https://utexas.box.com/shared/static/yz05g2x4bsc8q3avv6bskubi22266ca2.pth) |
| DECOLA | LVIS-base | [DECOLA_PHASE1_Lbase_CLIP_SwinB_4x](../configs/DECOLA_PHASE1_Lbase_CLIP_SwinB_4x.yaml)       | 31.1 | 37.3 | 44.1 | 46.2 | 47.2 | [weight](https://utexas.box.com/shared/static/o7wfuk5r3m4vesrness4ssk37x1aba99.pth) |
| baseline | LVIS | [BoxSup-DefromDETR_L_CLIP_SwinB_4x](../configs/BoxSup-DefromDETR_L_CLIP_SwinB_4x.yaml)          | 20.7 | 29.9 | 42.4 | 48.4 | 51.6 | [weight](https://utexas.box.com/shared/static/6rqosjkvp3sp13jnkubu8xltbz1r7v94.pth) |
| DECOLA | LVIS | [DECOLA_PHASE1_L_CLIP_SwinB_4x](../configs/DECOLA_PHASE1_L_CLIP_SwinB_4x.yaml)                | 34.5 | 42.3 | 49.0 | 50.8 | 52.7 | [weight](https://utexas.box.com/shared/static/t751ml0zayrx89qrj2c22yzcaxqdxvvn.pth) |

