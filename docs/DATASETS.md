# Prepare Datasets for DECOLA

We closely follow [Detic's data preparation](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md).
The basic training of our model uses [LVIS](https://www.lvisdataset.org/) (which uses [COCO](https://cocodataset.org/) images) and [ImageNet-21K](https://www.image-net.org/download.php). 
Optionally, we use [Objects365](https://www.objects365.org/) and [OpenImages (Challenge 2019 version)](https://storage.googleapis.com/openimages/web/challenge2019.html) for cross-dataset evaluation. 
Before starting processing, please download the (selected) datasets from the official websites and place or sim-link them under `$DECOLA_ROOT/datasets/`. 

```
$DECOLA_ROOT/datasets/
    metadata/
    lvis/
    coco/
    imagenet/
```
`metadata/` is our preprocessed meta-data (included in the repo). See the below [section](#Metadata) for details.
Please follow the following instruction to pre-process individual datasets.

### COCO and LVIS

First, download COCO and LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        instances_train2017.json 
        instances_val2017.json
```

Next, prepare the open-vocabulary LVIS training set using 

```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

This will generate `datasets/lvis/lvis_v1_train_norare.json`.

### ImageNet-21K

The ImageNet-21K folder should look like:
```
imagenet/
    ImageNet-21K/
        n01593028.tar
        n01593282.tar
        ...
```

We first unzip the overlapping classes of LVIS (we will directly work with the .tar file for the rest classes) and convert them into LVIS annotation format.

~~~
mkdir imagenet/annotations
python tools/unzip_imagenet_lvis.py --dst_path datasets/imagenet/ImageNet-LVIS
python tools/create_imagenetlvis_json.py --imagenet_path datasets/imagenet/ImageNet-LVIS --out_path datasets/imagenet/annotations/imagenet_lvis_image_info.json
~~~
This creates `datasets/imagenet/annotations/imagenet_lvis_image_info.json`.

[Optional] To train with all the 21K classes, run

~~~
python tools/get_imagenet_21k_full_tar_json.py
python tools/create_lvis_21k.py
~~~
This creates `datasets/imagenet/annotations/imagenet-21k_image_info_lvis-21k.json` and `datasets/lvis/lvis_v1_train_lvis-21k.json` (combined LVIS and ImageNet-21K classes in `categories`).

[Optional] To train on combined LVIS and COCO, run

~~~
python tools/merge_lvis_coco.py
~~~
This creates `datasets/lvis/lvis_v1_train+coco_mask.json`

### Objects365
Download Objects365 (v2) from the website. We only need the validation set in this project:
```
objects365/
    annotations/
        zhiyuan_objv2_val.json
    val/
        images/
            v1/
                patch0/
                ...
                patch15/
            v2/
                patch16/
                ...
                patch49/

```

The original annotation has typos in the class names, we first fix them for our following use of language embeddings.

```
python tools/fix_o365_names.py --ann datasets/objects365/annotations/zhiyuan_objv2_val.json
```
This creates `datasets/objects365/zhiyuan_objv2_val_fixname.json`.

To train on Objects365, download the training images and use the command above.  We note some images in the training annotation do not exist.
We use the following command to filter the missing images.
~~~
python tools/fix_0365_path.py
~~~
This creates `datasets/objects365/zhiyuan_objv2_train_fixname_fixmiss.json`.

### OpenImages

We followed the instructions in [UniDet](https://github.com/xingyizhou/UniDet/blob/master/docs/DATASETS.md#openimages) to convert the metadata for OpenImages.

The converted folder should look like

```
oid/
    annotations/
        oid_challenge_2019_train_bbox.json
        oid_challenge_2019_val_expanded.json
    images/
        0/
        1/
        2/
        ...
```

### Metadata

```
metadata/
    lvis_v1_train_cat_info.json
    lvis_v1_clip_a+cname.npy
    o365_clip_a+cnamefix.npy
    oid_clip_a+cname.npy
    imagenet_lvis_wnid.txt
    Objects365_names_fix.csv
    lvis_v1_minival_inserted_image_name.json   # for eval
```

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each datasets.
They are created by (taking LVIS as an example)
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
~~~
Note we do not include the 21K class embeddings due to the large file size.
To create it, run
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val_lvis-21k.json --out_path datasets/metadata/lvis-21k_clip_a+cname.npy
~~~

`imagenet_lvis_wnid.txt` is the list of matched classes between ImageNet-21K and LVIS.

`Objects365_names_fix.csv` is our manual fix of the Objects365 names.