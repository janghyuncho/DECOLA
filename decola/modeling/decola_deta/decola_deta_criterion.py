import math
import torch
import torch.nn.functional as F
import copy
from torch import nn
# from models.segmentation import sigmoid_focal_loss
from third_party.DETA.models.deformable_detr import SetCriterion
from ..utils import load_class_freq, get_fed_loss_inds
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .decola_deta_matcher import DECOLA_Stage1Assigner, DECOLA_Stage2Assigner


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, reduction=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if not reduction:
        return loss 
    return loss.mean(1).sum() / num_boxes


class Set_DECOLA_Criterion(nn.Module):
    def __init__(self, enc_only, num_classes, matcher, weight_dict, losses, 
                loss_per_image=False, focal_alpha=0.25,
                num_queries=300, assign_first_stage=False, assign_second_stage=False):
        super().__init__()
        self.enc_only = enc_only
        self.loss_per_image = loss_per_image

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.assign_first_stage = assign_first_stage
        self.assign_second_stage = assign_second_stage

        if self.assign_first_stage:
            self.stg1_assigner = DECOLA_Stage1Assigner()
        if self.assign_second_stage:
            self.stg2_assigner = DECOLA_Stage2Assigner(num_queries)

    def _get_DECOLA_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(inds, i) for i, (inds, _) in enumerate(indices)])
        src_idx = torch.cat([inds for (inds, _) in indices])
        tgt_idx = torch.cat([inds for (_, inds) in indices])
        return batch_idx, src_idx, tgt_idx

   
    def loss_labels(self, outputs, targets, indices, num_boxes, is_enc=False, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits_loss' in outputs
        src_logits = outputs['pred_logits_loss']

        idx1, idx2, _ = self._get_DECOLA_src_permutation_idx(indices)
        target_classes_onehot = src_logits.new_zeros(src_logits.shape)  # B,Q,1 
        target_classes_onehot[(idx1, idx2)] = 1 

        if self.loss_per_image:
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes, 
                alpha=self.focal_alpha, 
                gamma=2, reduction=False)        # B,Q,1
            prompt_inds = outputs['prompt_inds'] # B,Q 
            query_masks = prompt_inds.new_zeros(prompt_inds.shape)
            for b, target in enumerate(targets):
                mask = (prompt_inds[b][None] == target['labels'][:, None]).sum(dim=0).bool().to(query_masks)
                query_masks[b] = mask 
            loss_ce = (loss_ce * query_masks[..., None]).sum() / num_boxes
        else:
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes, 
                alpha=self.focal_alpha, 
                gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx1, idx2, _ = self._get_DECOLA_src_permutation_idx(indices)
        if len(idx1) == 0:
            return {'loss_bbox': outputs['pred_boxes'].new_tensor(0),
                    'loss_giou': outputs['pred_boxes'].new_tensor(0)} 
        src_boxes = outputs['pred_boxes'][(idx1, idx2)]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        if not self.enc_only:
            # torch.cuda.empty_cache()
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

            # Retrieve the matching between the outputs of the last layer and the targets
            if self.assign_second_stage:
                indices = self.stg2_assigner(outputs_without_aux, targets)
            else:
                indices = self.matcher(outputs_without_aux, targets)
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs and not self.enc_only:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['prompt_inds_all'] = outputs['prompt_inds_all']
                # aux_outputs['multi_class_second_stage'] = outputs['multi_class_second_stage']
                if not self.assign_second_stage:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False 
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            if self.assign_first_stage:
                indices = self.stg1_assigner(enc_outputs, targets)
            else:
                indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                    kwargs['is_enc'] = True 
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses



class SetCustomCriterion(SetCriterion):
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality, 
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
 

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        enc_loss = 'prompt_inds_all' not in outputs
        N = 1 if enc_loss else self.num_classes 
        target_classes = torch.full(src_logits.shape[:2], N,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, 
            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # B x N x C

        if 'prompt_inds_all' in outputs:
            inds = outputs['prompt_inds_all'] # predefined indexs 
            if inds is None: 
                inds = get_fed_loss_inds(
                    gt_classes=target_classes_o, 
                    num_sample_cats=50, # FIXME no hardcode. 
                    weight=self.fed_loss_weight, 
                    C=self.num_classes)
            loss_ce = sigmoid_focal_loss(
                src_logits[:, :, inds], 
                target_classes_onehot[:, :, inds], 
                num_boxes, 
                alpha=self.focal_alpha, 
                gamma=2) * src_logits.shape[1]
        else:
            loss_ce = sigmoid_focal_loss(
                src_logits, 
                target_classes_onehot, 
                num_boxes, 
                alpha=self.focal_alpha, 
                gamma=2) * src_logits.shape[1]


        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.assign_second_stage:
            indices = self.stg2_assigner(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['prompt_inds_all'] = outputs['prompt_inds_all']
                if not self.assign_second_stage:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            if self.assign_first_stage:
                indices = self.stg1_assigner(enc_outputs, bin_targets)
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses