
import copy
from typing import Optional, List
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import load_class_freq, get_fed_loss_inds

class DECOLA_ZeroshotClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 1203, 
        d_model: int = 256, 
        fed_freq_weight: float = 0.5, 
        zs_weight_path: str = "", 
        zs_obj_weight_path: str = "",
        zs_weight_dim: int = 512,
        norm_weight: bool = True,
        use_bias: bool = True,
        norm_temperature: float = 50.0,
        multi_class_second_stage: bool = False,
        use_fed_loss: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.linear = nn.Linear(d_model, zs_weight_dim)
        self.use_bias = use_bias
        self.multi_class_second_stage = multi_class_second_stage
        self.use_fed_loss = use_fed_loss
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * (-math.log((1 - 0.01) / 0.01)))

        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes+1))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32) # N, D
            zs_weight_dim = zs_weight.shape[0]
            zs_obj_weight = torch.tensor(np.load(zs_obj_weight_path), 
                                        dtype=torch.float32)
            zs_weight = torch.cat([zs_weight, zs_obj_weight], dim=0)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=1)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
        self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=fed_freq_weight,
        ))

    @property
    def device(self):
        return self.zs_weight.device 
    
    def generate_prompt_inds_train(
        self,
        targets,
        is_image_label,
        use_a_object,
    ):
        target_classes_o = []
        for target in targets:
            target_classes_o.append(target['labels'])
            if 'image_label' in target:
                target_classes_o.append(target['image_label'])
        target_classes_o = torch.cat(target_classes_o)

        if (is_image_label or not self.use_fed_loss) and use_a_object:
            # case 1: image label data multi-class (no fed. loss).
            inds = target_classes_o.new_tensor(np.arange(self.num_classes))
        elif use_a_object:
            # case 2: detection data multi-class (fed. loss).
            inds = get_fed_loss_inds(
                    gt_classes=target_classes_o,
                    num_sample_cats=50,
                    weight=self.fed_loss_weight,
                    C=1203) # we only use lvis cat info. 
        else:
            # case 3: detection data language-condition (decola training).
            inds = get_fed_loss_inds(
                    gt_classes=target_classes_o,
                    num_sample_cats=1, # NOTE: there can be images w/o fg class. If so, we sample one at random.
                    weight=torch.ones(self.num_classes).to(self.device),
                    C=1203)
        return inds 

    def generate_prompt_inds_test(
        self,
        targets,
        is_image_label,
        use_a_object,
        oracle_evaluation,
    ):
        use_ground_truth_inds = oracle_evaluation or (is_image_label and not use_a_object)
        if use_ground_truth_inds:
            # case 1: for oracle evaluation or self-labeling.
            inds = torch.cat([t['labels'] for t in targets]).unique() 
        else:
            # case 2: standard evaluation.
            inds = torch.arange(self.num_classes).to(self.device)
        return inds 


    def get_prompt_embedding(
        self, 
        x, 
        inds=None, 
        use_a_object=True, 
    ):
        w = self.zs_weight[self.num_classes:] if use_a_object else self.zs_weight[inds] 
        w = w.unsqueeze(0).expand(x.shape[0], *w.shape).permute(0, 2, 1)
        return w

    def forward(
        self, 
        x, 
        training, 
        targets=None, 
        inds=None, 
        second_stage=False, 
        is_image_label=False, 
        use_a_object=True,
        oracle_evaluation=False,
    ):
        if second_stage:
            if use_a_object:
                return self.forward_second_stage_multi_prompt(x)
            else:
                return self.forward_second_stage_single_prompt(x, inds, training)
        else:
            if training:
                return self.forward_first_stage_train(x, targets, is_image_label, use_a_object, oracle_evaluation)
            else:
                return self.forward_first_stage_test(x, targets, is_image_label, use_a_object, oracle_evaluation)


    def forward_first_stage_test(
        self, 
        x, 
        targets, 
        is_image_label=False,
        use_a_object=True, 
        oracle_evaluation=False
    ):
        assert targets is not None
        B = x.shape[0] 
        x = self.linear(x)
        inds = self.generate_prompt_inds_test(
                targets, 
                is_image_label=is_image_label,
                use_a_object=use_a_object,
                oracle_evaluation=oracle_evaluation)
        inds = inds.to(self.device)

        if use_a_object or (len(inds) == 0 and oracle_evaluation):
            # if during eval-oracle and concept-specific queries are used but there is no gt,
            # we switch to standard eval. 
            inds = self.generate_prompt_inds_test(targets, 
                                                  is_image_label=False, 
                                                  use_a_object=True, 
                                                  oracle_evaluation=False)
            inds = inds.to(self.device)
            use_a_object = True 
            w = self.get_prompt_embedding(x, inds, use_a_object=True)  # (B,C,K)
        else:
            w = self.get_prompt_embedding(x, inds, use_a_object=use_a_object)  # (B,C,K)
        
        # print(w.shape, flush=True)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=2)
        out = torch.bmm(x, w) # (B,S,C) x (B,C,K) -> (B,S,K)
        if self.use_bias:
            out = out + self.cls_bias
        inds = inds[None].expand(B, len(inds)) 

        return out, inds, w, use_a_object


    def forward_first_stage_train(
        self, 
        x, 
        targets, 
        is_image_label=False,
        use_a_object=True, 
        oracle_evaluation=False
    ):
        assert targets is not None
        B = x.shape[0] 
        x = self.linear(x)
        inds = self.generate_prompt_inds_train(
                targets, 
                is_image_label=is_image_label,
                use_a_object=use_a_object,
                )
        inds = inds.to(self.device)

        w = self.get_prompt_embedding(x, inds, use_a_object)  # (B,C,K)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=2)
        out = torch.bmm(x, w) # (B,S,C) x (B,C,K) -> (B,S,K)
        if self.use_bias:
            out = out + self.cls_bias
        inds = inds[None].expand(B, len(inds)) 

        return out, inds, w, use_a_object


    def forward_second_stage_multi_prompt(self, x):
        B, S, _ = x.shape 
        x = self.linear(x) 
        w = self.zs_weight[:self.num_classes].permute(1, 0)
        w = w[None].expand(B, w.shape[0], w.shape[1]) # (B,C,N)
        # print(w.shape, self.num_classes, flush=True)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=2)
        out = torch.bmm(x, w) # (B,S,C) x (B,C,N) -> (B,S,N)
        if self.use_bias:
            out = out + self.cls_bias
        return out, out 


    def forward_second_stage_single_prompt(self, x, inds, training):
        B = x.shape[0]
        S = x.shape[1]
        x = self.linear(x)
        w = self.zs_weight[:self.num_classes].permute(1, 0)
        # print(w.shape, self.num_classes, flush=True)
        w = w[None].expand(B, w.shape[0], w.shape[1]) # (B,C,N)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=2)
        out = torch.bmm(x, w) # (B,S,C) x (B,C,N) -> (B,S,N)
        if self.use_bias:
            out = out + self.cls_bias

        out_match = torch.gather(out, 2, inds.unsqueeze(2))   # B,S,1
        out_loss = out if self.multi_class_second_stage else out_match 
        out_match = out_match if training else out_loss 
        return out_loss, out_match  

