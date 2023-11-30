
import torch
import numpy as np 
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, box_xyxy_to_cxcywh
from third_party.DETA.models.assigner import (Matcher, sample_topk_per_gt, subsample_labels, nonzero_tuple)
import time 


class DECOLA_Stage2Assigner(nn.Module):
    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 2000 # number > 91 to filter out later
        self.batch_size_per_image = num_queries
        self.proposal_matcher = Matcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)


    def forward(self, outputs, targets, return_cost_matrix=False):
        """
        pred_logits: BxNx1  
        pred_boxes : BxNx4  
        prompt_inds: BxN
        labels     : Y      
        boxes      : Yx4     
        """
        with torch.no_grad():
            t1 = time.time()
            bs, num_queries = outputs["pred_logits_match"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits_match"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"]      # [batch_size, num_queries, 4]
            prompt_inds = outputs["prompt_inds"]  # [batch_size, num_queries ]

            # Also concat the target labels and boxes
            tgt_inds = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            res = []
            for b in range(len(targets)):
                proposals_i = outputs['init_reference'][b].detach()
                if len(targets[b]['boxes']) == 0:
                    res.append((torch.tensor([], dtype=torch.long, device=proposals_i.device),
                                torch.tensor([], dtype=torch.long, device=proposals_i.device)))
                    continue

                out_bbox_i = out_bbox[b]
                prompt_inds_i = prompt_inds[b]
                tgt_lbls_i = targets[b]["labels"]
                tgt_bbox_i = targets[b]["boxes"]
                rows, cols = [], []
                for tid in tgt_lbls_i.unique():
                    ptmatch_idx = prompt_inds_i == tid
                    ttmatch_idx = tgt_lbls_i == tid
                    if ptmatch_idx.any():
                        # local indexing
                        iou, _ = box_iou(
                            box_cxcywh_to_xyxy(tgt_bbox_i[ttmatch_idx]),
                            box_cxcywh_to_xyxy(proposals_i[ptmatch_idx]),
                        )
                        matched_idxs, matched_labels = self.proposal_matcher(iou)  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.7, 0 if iou < 0.3, -1 ow]
                        sampled_idxs, sampled_gt_classes = self._sample_proposals(  # list of sampled proposal_ids, sampled_id -> [0, num_classes)+[bg_label]
                            matched_idxs, matched_labels, tgt_lbls_i[ttmatch_idx]
                        )

                        pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
                        pos_gt_inds = matched_idxs[pos_pr_inds]
                        pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)

                        # global indexing 
                        i = torch.where(ptmatch_idx)[0][pos_pr_inds]
                        j = torch.where(ttmatch_idx)[0][pos_gt_inds]
                        rows.append(i)
                        cols.append(j)
                    else:
                        print(ptmatch_idx.shape, ttmatch_idx.shape, tid, flush=True)
                if len(rows) > 0 and len(cols) > 0:
                    rows = torch.cat(rows)
                    cols = torch.cat(cols)
                else:
                    rows = torch.tensor([], device=proposals_i.device, dtype=torch.long)
                    cols = torch.tensor([], device=proposals_i.device, dtype=torch.long)
                res.append((rows, cols))
            t3 = time.time()
            # print('[matcher]\n[time 1]: {:.3f}\n[time 2]: {:.3f}'.format(t2-t1, t3-t2), flush=True)
            return res



class DECOLA_Stage1Assigner(nn.Module):
    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        self.anchor_matcher = Matcher(thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True)

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)


    def forward(self, outputs, targets):
        """
        pred_logits: BxNx1  
        pred_boxes : BxNx4  
        prompt_inds: BxN
        labels     : Y      
        boxes      : Yx4     
        """
        with torch.no_grad():
            t1 = time.time()
            bs, num_queries = outputs["pred_logits_match"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits_match"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"]      # [batch_size, num_queries, 4]
            prompt_inds = outputs["prompt_inds"]  # [batch_size, num_queries ]

            # Also concat the target labels and boxes
            tgt_inds = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            res = []
            for b in range(len(targets)):
                anchors_i = outputs['anchors'][b]
                if len(targets[b]['boxes']) == 0:
                    res.append((torch.tensor([], dtype=torch.long, device=anchors_i.device),
                                torch.tensor([], dtype=torch.long, device=anchors_i.device)))
                    continue
                
                out_bbox_i = out_bbox[b]
                prompt_inds_i = prompt_inds[b]
                tgt_lbls_i = targets[b]["labels"]
                tgt_bbox_i = targets[b]["boxes"]
                rows, cols = [], []
                for tid in tgt_lbls_i.unique():
                    ptmatch_idx = prompt_inds_i == tid
                    ttmatch_idx = tgt_lbls_i == tid
                    if ptmatch_idx.any():
                        # local indexing
                        iou, _ = box_iou(
                            box_cxcywh_to_xyxy(tgt_bbox_i[ttmatch_idx]),
                            box_cxcywh_to_xyxy(anchors_i[ptmatch_idx]),
                        )
                        matched_idxs, matched_labels = self.anchor_matcher(iou)  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.7, 0 if iou < 0.3, -1 ow]
                        matched_labels = self._subsample_labels(matched_labels)

                        all_pr_inds = torch.arange(len(anchors_i[ptmatch_idx]))
                        pos_pr_inds = all_pr_inds[matched_labels == 1]
                        pos_gt_inds = matched_idxs[pos_pr_inds]
                        pos_ious = iou[pos_gt_inds, pos_pr_inds]
                        pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
                        pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors_i.device), pos_gt_inds.to(anchors_i.device)

                        # global indexing 
                        i = torch.where(ptmatch_idx)[0][pos_pr_inds]
                        j = torch.where(ttmatch_idx)[0][pos_gt_inds]
                        rows.append(i)
                        cols.append(j)
                if len(rows) > 0 and len(cols) > 0:
                    rows = torch.cat(rows)
                    cols = torch.cat(cols)
                else:
                    rows = torch.tensor([], device=anchors_i.device, dtype=torch.int64)
                    cols = torch.tensor([], device=anchors_i.device, dtype=torch.int64)
                res.append((rows, cols))

            t3 = time.time()
            # print('[matcher]\n[time 1]: {:.3f}\n[time 2]: {:.3f}'.format(t2-t1, t3-t2), flush=True)
            return res
