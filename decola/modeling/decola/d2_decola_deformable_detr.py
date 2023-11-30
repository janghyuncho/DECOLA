import torch
import os 
import numpy as np 
import torch.nn.functional as F
from torch import nn
import math 
import time 
import copy 
import fvcore.nn.weight_init as weight_init
import detectron2.utils.comm as comm
import logging 
import wandb 
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances
from models.backbone import Joiner
from models.deformable_detr import _get_clones
from models.position_encoding import PositionEmbeddingSine
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .decola_deformable_transformer import DECOLA_DeformableTransformer
from .decola_zero_shot_classifier import DECOLA_ZeroshotClassifier
from .decola_deformable_detr import DECOLA_DETR
from .decola_criterion import Set_DECOLA_Criterion, SetCustomCriterion
from .decola_matcher import DECOLA_HungarianMatcher
from models.matcher import HungarianMatcher
from ..common import MaskedBackbone
from .utils import clone_module, detach_module
from util.misc import nested_tensor_from_tensor_list

__all__ = ["DECOLA_DeformableDETR"]

@META_ARCH_REGISTRY.register()
class DECOLA_DeformableDETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.logger = logging.getLogger("decola")
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.test_topk = cfg.TEST.DETECTIONS_PER_IMAGE
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        if self.mask_on:
            assert 0, 'Mask is not supported yet :('

        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DETR.TWO_STAGE
        with_box_refine = cfg.MODEL.DETR.WITH_BOX_REFINE
        look_forward_twice = cfg.MODEL.DETR.LOOK_FORWARD_TWICE 

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        cls_weight = cfg.MODEL.DETR.CLS_WEIGHT
        focal_alpha = cfg.MODEL.DETR.FOCAL_ALPHA

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))

        # decola-related 
        prob_a_object = cfg.MODEL.DECOLA.PROB_A_OBJECT  # for "general-purpose" (standard) detection, we use "a object." as prompt.
        use_prompt_embed_bias = cfg.MODEL.DECOLA.USE_PROMPT_EMBED_BIAS 
        per_prompt_topk = cfg.MODEL.DECOLA.PER_PROMPT_TOPK
        first_stage_enc_loss_per_prompt_topk = cfg.MODEL.DECOLA.FIRST_STAGE_ENC_LOSS_PER_PROMPT_TOPK
        use_pixel_feature_as_query = cfg.MODEL.DECOLA.USE_PIXEL_FEATURE_AS_QUERY
        use_prompt_embed_as_query = cfg.MODEL.DECOLA.USE_PROMPT_EMBED_AS_QUERY
        fed_freq_weight = cfg.MODEL.DECOLA.FED_FREQ_WEIGHT
        zs_weight_path = cfg.MODEL.DECOLA.ZS_WEIGHT_PATH
        zs_obj_weight_path = cfg.MODEL.DECOLA.ZS_OBJECT_WEIGHT_PATH
        zs_weight_dim = cfg.MODEL.DECOLA.ZS_WEIGHT_DIM 
        first_stage_norm_weight = cfg.MODEL.DECOLA.FIRST_STAGE_EMBED_NORM 
        second_stage_norm_weight = cfg.MODEL.DECOLA.SECOND_STAGE_EMBED_NORM 

        # ablations 
        apply_loss_only_within_image = cfg.MODEL.DECOLA.APPLY_LOSS_ONLY_WITHIN_IMAGE
        self.multi_class_second_stage = cfg.MODEL.DECOLA.MULTI_CLASS_SECOND_STAGE
        self.no_box_loss_for_pseudo_labels = cfg.MODEL.DECOLA.NO_BOX_LOSS_FOR_PSEUDO_LABELS
        self.per_prompt_topk = per_prompt_topk
        self.test_class_conditioned = cfg.MODEL.DECOLA.TEST_CLASS_CONDITIONED
        self.score_thres = cfg.MODEL.DECOLA.TEST_SCORE_THRESHOLD
        self.oracle_evaluation = cfg.MODEL.DECOLA.ORACLE_EVALUATION

        first_stage_class_embed = DECOLA_ZeroshotClassifier(num_classes=self.num_classes, 
                                                            d_model=hidden_dim,
                                                            fed_freq_weight=fed_freq_weight,
                                                            zs_weight_path=zs_weight_path,
                                                            zs_obj_weight_path=zs_obj_weight_path,
                                                            zs_weight_dim=zs_weight_dim,
                                                            norm_weight=first_stage_norm_weight,
                                                            use_bias=use_prompt_embed_bias,
                                                            use_fed_loss=cfg.MODEL.DETR.USE_FED_LOSS,
                                                          )
        
        transformer = DECOLA_DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=two_stage,
            two_stage_num_proposals=num_queries,
            # decola
            num_classes=self.num_classes,
            per_prompt_topk=per_prompt_topk,
            use_pixel_feature_as_query=use_pixel_feature_as_query,
            use_prompt_embed_as_query=use_prompt_embed_as_query,
            zs_weight_dim=zs_weight_dim,
            first_stage_enc_loss_per_prompt_topk=first_stage_enc_loss_per_prompt_topk,
            prob_a_object=prob_a_object,
            oracle_evaluation=self.oracle_evaluation,
            look_forward_twice=look_forward_twice,
            )

        self.detr = DECOLA_DETR(
            backbone, transformer, 
            num_classes=1, 
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            aux_loss=deep_supervision,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
        )
        del self.detr.class_embed 
        del self.detr.transformer.decoder.class_embed 
        class_embed = DECOLA_ZeroshotClassifier(num_classes=self.num_classes, 
                                                d_model=hidden_dim,
                                                zs_weight_path=zs_weight_path,
                                                zs_obj_weight_path=zs_obj_weight_path,
                                                zs_weight_dim=zs_weight_dim,
                                                norm_weight=second_stage_norm_weight,
                                                use_bias=use_prompt_embed_bias,
                                                multi_class_second_stage=self.multi_class_second_stage,
                                                use_fed_loss=cfg.MODEL.DETR.USE_FED_LOSS,
                                                )

        num_pred = self.detr.transformer.decoder.num_layers 
        if with_box_refine:
            class_embed_list = [copy.deepcopy(class_embed) for i in range(num_pred)]
        else:
            class_embed_list = [class_embed for _ in range(num_pred)]

        class_embed_list = class_embed_list + [first_stage_class_embed]
        self.detr.class_embed = nn.ModuleList(class_embed_list)
        if two_stage:
            self.detr.transformer.decoder.class_embed = self.detr.class_embed

        decola_matcher = DECOLA_HungarianMatcher(
            cost_class=cls_weight, cost_bbox=l1_weight, cost_giou=giou_weight)
        obj_matcher = HungarianMatcher(
            cost_class=cls_weight, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": cls_weight, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()}) # NOTE: bug fixed.
            weight_dict.update(aux_weight_dict)
        
        self.logger.info('weight_dict', weight_dict)
        losses = ["labels", "boxes"] 
        self.decola_criterion = Set_DECOLA_Criterion(
            self.multi_class_second_stage,
            loss_per_image=apply_loss_only_within_image,
            num_classes=self.num_classes, 
            matcher=decola_matcher, 
            weight_dict=weight_dict, 
            focal_alpha=focal_alpha, 
            losses=losses)

        self.obj_criterion = SetCustomCriterion(
            num_classes=self.num_classes, 
            matcher=obj_matcher, 
            weight_dict=weight_dict, 
            focal_alpha=focal_alpha, 
            losses=losses)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std 

        # weak loss setting 
        self.pseudo_detr = None 
        self.with_image_labels = cfg.WITH_IMAGE_LABELS 
        self.detr.with_image_labels = self.with_image_labels
        self.weak_loss_pseudo_type = cfg.MODEL.DECOLA.WEAK_LOSS_PSEUDO_TYPE
        self.use_pseudo_labeling = cfg.MODEL.DECOLA.USE_PSEUDO_LABELING_FOR_WEAK_LOSS
        self.pseudo_size_limit = cfg.MODEL.DECOLA.PSEUDO_LOSS_MIN_SIZE_LIMIT
        self.save_offline_pseudo_labels = cfg.TEST.JUST_STORE_PREDICTIONS 

        # wandb visualize 
        self.use_wandb_vis = not cfg.WANDB.DISABLE_WANDB and (cfg.WANDB.VIS_PERIOD > 0)
        self.vis_period = cfg.WANDB.VIS_PERIOD
        self.iter_count = 0
        self.is_demo = False 

        # checkpointing 
        from fairscale.nn.checkpoint import checkpoint_wrapper
        use_encoder_checkpoint = cfg.MODEL.DECOLA.USE_ENCODER_CHECKPOINT
        use_decoder_checkpoint = cfg.MODEL.DECOLA.USE_DECODER_CHECKPOINT
        if use_encoder_checkpoint:
            for layer in self.detr.transformer.encoder.layers:
                layer = checkpoint_wrapper(layer)
        
        if use_decoder_checkpoint:
            for layer in self.detr.transformer.decoder.layers:
                layer = checkpoint_wrapper(layer)


    @torch.no_grad()
    def update_targets_with_pseudo_labels(self, images, targets):
        for t in targets:
            t['labels'] = t['image_label']
        output = self.pseudo_detr(images, targets)

        for b, t in enumerate(targets):
            new_labels = [] 
            new_boxes = []
            if len(t['labels']) > 0:
                boxes_all = output['pred_boxes'][b]
                scores_all = output['pred_logits'][b]
                pred_lbls = output['prompt_inds'][b]
                for l in t['labels']:
                    scores = scores_all[pred_lbls==l]
                    boxes = boxes_all[pred_lbls==l]
                
                    if self.pseudo_size_limit > 0:
                        valid_mask = (boxes[..., 2] * boxes[..., 3] > self.pseudo_size_limit)
                        boxes = boxes[valid_mask]
                        scores = scores[valid_mask]
                    if self.weak_loss_pseudo_type == 'max-score':
                        inds = scores.topk(1, dim=0)[1]  # 1,1
                        boxes = torch.gather(boxes, 0, inds.repeat(1, 4))
                    elif self.weak_loss_pseudo_type == 'max-size':
                        sizes = boxes[..., 2] * boxes[..., 3] 
                        inds = sizes.topk(1, dim=0)[1]
                        boxes = torch.gather(boxes, 0, inds[:, None].repeat(1, 4))
                    else:
                        raise NotImplementedError('{} not supported for pseudo-labeling.'.format(self.weak_loss_pseudo_type))

                    # pseudo-label needs to be at least EPS=1e-5 in size. 
                    if (boxes[:, 2:] > 1e-5).all():  
                        new_labels.append(l)
                        new_boxes.append(boxes)
                t['labels'] = torch.tensor(new_labels).to(t['labels'])
                t['boxes'] = torch.cat(new_boxes) if len(new_boxes) > 0 else boxes.new_zeros((0, 4))
   
        return targets 


    def forward(self, batched_inputs):
        t0 = time.time()
        if self.pseudo_detr is None and self.with_image_labels:
            # FIXME: this makes the code not "resume-able."  
            self.pseudo_detr = clone_module(self.detr)
            self.pseudo_detr.transformer.oracle_evaluation = True 
            self.pseudo_detr.train(False)
            self.pseudo_detr.transformer.per_prompt_topk = 300 
            self.pseudo_detr.transformer.training = False 
            self.pseudo_detr.transformer.use_a_object = False
            detach_module(self.pseudo_detr)

        t1 = time.time()
        self.detr.train(self.training)
        if self.training:
            # NOTE: This is for the "co-training" ablation. 
            use_a_object = torch.multinomial(torch.tensor([1-self.detr.transformer.prob_a_object, 
                                        self.detr.transformer.prob_a_object]), 1).bool()
            use_a_object = comm.all_gather(use_a_object)[0].item()
            comm.synchronize()
        else:
            use_a_object = not self.test_class_conditioned 
        self.detr.transformer.use_a_object = use_a_object
        
        images = self.preprocess_image(batched_inputs)
        t2 = time.time()

        # print('data type: {}\nuse a object: {}\noutput dim: {}'.format([b['ann_type'] for b in batched_inputs], 
        # use_a_object, output['pred_logits'].shape), flush=True)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, batched_inputs)

            output = self.detr(images, targets)
            multi_class_prediction_flag = self.detr.transformer.use_a_object or self.multi_class_second_stage # always after self.detr() call.

            ann_type = batched_inputs[0]['ann_type'] if self.with_image_labels else None
            
            # NOTE: need to use [self.detr.transformer.use_a_object] 
            # since sometimes the flag changes inside. 
            if ann_type in ['image'] and self.use_pseudo_labeling:  
                targets = self.update_targets_with_pseudo_labels(images, targets)
            
            if self.detr.transformer.use_a_object:
                loss_dict = self.obj_criterion(output, targets)
                weight_dict = self.obj_criterion.weight_dict
            else:
                loss_dict = self.decola_criterion(output, targets)
                weight_dict = self.decola_criterion.weight_dict
                
                # class-agnostic proposals for second stage 
                if self.multi_class_second_stage:
                    del output['enc_outputs']
                    loss_dict_obj = self.obj_criterion(output, targets)
                    for k in loss_dict_obj.keys():
                        if 'enc' not in k:
                            loss_dict[k] = loss_dict_obj[k]
            
            for k in list(loss_dict.keys()):
                if k in weight_dict:
                    if ann_type in ['image']:
                        weight_k = weight_dict[k]
                        if self.no_box_loss_for_pseudo_labels and ('bbox' in k or 'giou' in k):
                            weight_k = 0.0
                        loss_dict[k + "_weak_data"] = weight_k * loss_dict[k]
                        loss_dict[k] = images[0].new_zeros(
                                [1], dtype=torch.float32)[0]
                    else:
                        loss_dict[k] *= weight_dict[k]
                        loss_dict[k + "_weak_data"] = images[0].new_zeros(
                                [1], dtype=torch.float32)[0]
            t3 = time.time()

            if comm.is_main_process() \
            and self.use_wandb_vis \
            and (self.iter_count % self.vis_period == 0):
                with torch.no_grad():
                    image_sizes = torch.as_tensor([(t["height"], t["width"]) for t in batched_inputs], device=self.device)
                    batch_image_sizes = torch.as_tensor([t['image'].shape[1:] for t in batched_inputs], device=self.device)
                    pred_instances = self.post_process(output, batch_image_sizes, score_thres=0.1)
                    target_instances = self.target_post_process(targets, batch_image_sizes)
                    self.wandb_visualize(batched_inputs, pred_instances, vis_name=f'{ann_type}-predictions')
                    self.wandb_visualize(batched_inputs, target_instances, vis_name=f'{ann_type}-targets')
                    del pred_instances, target_instances
            self.iter_count += 1
            # print('[deformable detr]\n[time 1]: {:.3f}\n[time 2]: {:.3f}\n[time 3]: {:.3f}'.format(t1-t0, t2-t1, t3-t2), flush=True)
            return loss_dict
        else:
            image_sizes = torch.as_tensor([(t["height"], t["width"]) for t in batched_inputs], device=self.device)

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs] if not self.is_demo else None 
            targets = self.prepare_targets(gt_instances, batched_inputs)

            # for self-labeling.
            if self.save_offline_pseudo_labels:
                if 'image_label' in targets[0]:
                    for t in targets:
                        t['labels'] = t['image_label']
                output = self.pseudo_detr(images, targets)
                return self.prepare_offline_pseudo_labels(output, targets, image_sizes, batched_inputs)
            else:
                output = self.detr(images, targets)
            
            # phase2 or ablation.  
            multi_class_prediction_flag = self.detr.transformer.use_a_object or self.multi_class_second_stage           
            if multi_class_prediction_flag:
                # eval oracle
                if self.detr.transformer.oracle_evaluation:
                    new_pred_logits = torch.full_like(output['pred_logits'], float('-inf'))
                    for b, target in enumerate(targets):
                        new_pred_logits[b][:, target['labels']] = output['pred_logits'][b][:, target['labels']]
                    output['pred_logits'] = new_pred_logits

            results = self.post_process(output, image_sizes) if multi_class_prediction_flag \
                else self.decola_post_process(output, image_sizes)
            return results

    def post_process(self, outputs, target_sizes, score_thres=None):
        """
        Modified post-process to have different query topk and box topk.
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes), "{} != {}".format(len(out_logits), len(target_sizes))
        assert target_sizes.shape[1] == 2
        if score_thres is None:
            score_thres = self.score_thres

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.test_topk, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='trunc')
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = []
        for s_, l_, b_, size in zip(scores, labels, boxes, target_sizes):
            s = s_[s_>score_thres]
            l = l_[s_>score_thres]
            b = b_[s_>score_thres]
            if len(s) > 0:
                r = Instances((size[0], size[1]))
                r.pred_boxes = Boxes(b)
                r.scores = s
                r.pred_classes = l
                results.append({'instances': r})
        return results

    def decola_post_process(self, outputs, target_sizes, score_thres=None):
        """
        Modified post-process to have different query topk and box topk.
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_labels = outputs['prompt_inds']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        if score_thres is None:
            score_thres = self.score_thres

        prob = out_logits.sigmoid()
        prob = prob.view(prob.shape[0], -1)
        test_topk = min(self.test_topk, prob.shape[1])
        topk_values, topk_indexes = torch.topk(prob, test_topk, dim=1) 
        scores = topk_values

        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, 4))

        out_labels = torch.gather(out_labels, 1, topk_indexes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for s_, l_, b_, size in zip(scores, out_labels, boxes, target_sizes):
            s = s_[s_>score_thres]
            l = l_[s_>score_thres]
            b = b_[s_>score_thres]
            if len(s) > 0:
                r = Instances((size[0], size[1]))
                r.pred_boxes = Boxes(b)
                r.scores = s
                r.pred_classes = l
                results.append({'instances': r})
        return results

    def target_post_process(self, targets, image_sizes):
        img_h, img_w = image_sizes.unbind(1)
        scale_factor = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        new_targets = []
        for s, targets_per_image in zip(scale_factor, targets):
            gt_classes = targets_per_image['labels']
            gt_boxes = targets_per_image['boxes']
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes) * s
            scores  = gt_boxes.new_ones(len(gt_classes))

            new_target = Instances(image_sizes)
            new_target.pred_classes = gt_classes 
            new_target.pred_boxes = gt_boxes
            new_target.scores = scores
            new_targets.append({"instances": new_target})
        return new_targets 

    def prepare_targets(self, targets, batched_inputs):
        new_targets = []
        if self.is_demo:
            for _ in range(len(batched_inputs)):
                new_targets.append({"labels": torch.arange(self.num_classes, device=self.device)})
            return new_targets

        for targets_per_image, x in zip(targets, batched_inputs):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if 'ann_type' in x:
                new_targets[-1].update({'ann_type': x['ann_type']})
                if x['ann_type'] in ['image']:
                    new_targets[-1].update({'image_label': gt_classes.new_tensor(list(set(x['pos_category_ids'])))})
            
        return new_targets

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        return images


    def prepare_offline_pseudo_labels(self, output, targets, image_sizes, batched_inputs):
        assert len(targets[0]['labels']) > 0 
        assert not self.training, "pseudo-labeling should be in eval mode."
        assert not self.detr.transformer.use_a_object
        assert self.detr.transformer.oracle_evaluation

        boxes_batch = [] 
        scores_batch = []
        labels_batch = []
        for b, t in enumerate(targets):
            assert len(t['labels']) > 0 
            pseudo_labels = []
            pseudo_boxes = []
            pseudo_scores = []

            boxes_all = output['pred_boxes'][b]
            scores_all = output['pred_logits'][b].sigmoid()
            pred_lbls = output['prompt_inds'][b]   
            for l in t['labels']:
                scores = scores_all[pred_lbls==l]
                boxes = boxes_all[pred_lbls==l]

                if self.pseudo_size_limit > 0:
                    valid_mask = (boxes[..., 2] * boxes[..., 3] > self.pseudo_size_limit)
                    boxes = boxes[valid_mask]
                    scores = scores[valid_mask]
                if self.weak_loss_pseudo_type == 'max-score':
                    s, inds = scores.topk(1, dim=0)  # 1,1
                    boxes = torch.gather(boxes, 0, inds.repeat(1, 4))
                elif self.weak_loss_pseudo_type == 'max-size':
                    sizes = boxes[..., 2] * boxes[..., 3] 
                    inds = sizes.topk(1, dim=0)[1]
                    boxes = torch.gather(boxes, 0, inds[:, None].repeat(1, 4))
                    s = scores[inds]
                else:
                    raise NotImplementedError('{} not supported for pseudo-labeling.'.format(self.weak_loss_pseudo_type))

                # pseudo-label needs to be at least EPS=1e-5 in size. 
                if (boxes[:, 2:] > 1e-5).all():  
                    pseudo_labels.append(l)
                    pseudo_boxes.append(boxes)
                    pseudo_scores.append(s)
        
            pseudo_scores = torch.tensor(pseudo_scores).to(self.device)
            pseudo_labels = torch.tensor(pseudo_labels).to(self.device)
            pseudo_boxes = torch.cat(pseudo_boxes) if len(pseudo_boxes) > 0 else boxes.new_zeros((0, 4))
            new_boxes = box_cxcywh_to_xyxy(pseudo_boxes) 
            if (new_boxes[:, 0] > new_boxes[:, 2]).any() or (new_boxes[:, 1] > new_boxes[:, 3]).any():
                # self.logger.info("pseudo-label with invalid boxes.")
                assert False, "pseudo-label with invalid boxes."
            boxes_batch.append(new_boxes)
            labels_batch.append(pseudo_labels)
            scores_batch.append(pseudo_scores)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = image_sizes.unbind(1)
        scale_factor = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :]

        results = []
        for s_, l_, b_, size, sf in zip(scores_batch, labels_batch, boxes_batch, image_sizes, scale_factor):
            r = Instances((size[0], size[1]))
            b = b_ * sf
            r.pred_boxes = Boxes(b)
            r.scores = s_
            r.pred_classes = l_ 
            results.append({'instances': r})

        return results

    
    def wandb_visualize(self, inputs, processed_results, vis_name, opacity=0.7):
        # NOTE: Hack to use input as visualization image.
        if len(processed_results) > 0:
            images_vis = [x["image"].float().to('cpu') for x in inputs]
            result_vis = [r["instances"].to('cpu') for r in processed_results]

            image, instances = images_vis[0], result_vis[0]
            image = image.permute(1, 2, 0).to(torch.uint8)
            white = np.ones(image.shape) * 255
            image = image * opacity + white * (1-opacity) 

            visualizer = Visualizer(image, None, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            
            image_pd = wandb.Image(vis_output.get_image())
            wandb.log({vis_name: image_pd})