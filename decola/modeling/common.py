import torch 
import torch.nn as nn 
import logging 
import torch.nn.functional as F
from torch.cuda.amp import autocast
from util.misc import NestedTensor
from detectron2.modeling import build_backbone

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        logger = logging.getLogger("decola")
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        logger.info('Backbone output features: {} -> In features: {}'.format(backbone_shape.items(), cfg.MODEL.DETR.IN_FEATURES))
        backbone_shape = {k:v for k, v in backbone_shape.items() if k in cfg.MODEL.DETR.IN_FEATURES}
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[x].channels for x in backbone_shape.keys()]
        self.out_names = cfg.MODEL.DETR.IN_FEATURES
        self.fp16 = cfg.FP16
        
    def forward(self, tensor_list: NestedTensor):
        if self.fp16:
            with autocast():
                xs = self.backbone(tensor_list.tensors.half())
                xs = {k: v.float() for k, v in xs.items()}
        else:
            xs = self.backbone(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            if name in self.out_names:
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
            
        return out