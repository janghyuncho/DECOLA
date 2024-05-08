# Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).


### Example conda environment setup
```
conda create --name decola python=3.10 -y
conda activate decola
<!-- conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -->
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/janghyuncho/DECOLA.git --recurse-submodules
cd DECOLA
pip install -r requirements.txt

# Deformable-module
cd third_party/Deformable-DETR/models/ops
bash ./make.sh
``` 


#### (Optional) Segment Anything Model (SAM)
If you want to use SAM for the mask generation, install SAM following their [official github](https://github.com/facebookresearch/segment-anything#installation). Please download [SAM weight](https://github.com/facebookresearch/segment-anything#model-checkpoints) (ViT-H model) and place it under `weights/sam/` folder. 
