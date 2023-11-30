
import torch
import numpy as np 
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import time 

class DECOLA_HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1, 
                 cost_bbox : float = 1, 
                 cost_giou : float = 1):
        super().__init__()
        self.cost_class = cost_class     
        self.cost_bbox = cost_bbox       
        self.cost_giou = cost_giou       
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0" 

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
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            prompt_inds = outputs["prompt_inds"]            # [batch_size, num_queries ]

            # Also concat the target labels and boxes
            tgt_inds = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            # print('cls {}, box {} giou {}'.format(cost_class.shape, cost_bbox.shape, cost_giou.shape))
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1)
            t2 = time.time() 

            sizes = [len(v["boxes"]) for v in targets]
            Clist = [c[i] for i, c in enumerate(C.split(sizes, -1))]
            tlist = [t for t in tgt_inds.split(sizes, -1)]
            res   = []
            for _C, _t, _prompt_inds in zip(Clist, tlist, prompt_inds):
                rows, cols = [], []
                for tid in _t.unique():
                    ptmatch_idx = _prompt_inds == tid
                    ttmatch_idx = _t == tid
                    if ptmatch_idx.any():
                        # local indexing
                        i, j = linear_sum_assignment(_C[ptmatch_idx][:, ttmatch_idx].to('cpu', non_blocking=True))
                        i, j = torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)

                        # global indexing 
                        i = torch.where(ptmatch_idx)[0][i]
                        j = torch.where(ttmatch_idx)[0][j]
                        rows.append(i)
                        cols.append(j)
                if len(rows) > 0 and len(cols) > 0:
                    rows = torch.cat(rows)
                    cols = torch.cat(cols)
                else:
                    rows = torch.tensor([], device=out_prob.device, dtype=torch.int64)
                    cols = torch.tensor([], device=out_prob.device, dtype=torch.int64)
                res.append((rows, cols))

            t3 = time.time()
            # print('[matcher]\n[time 1]: {:.3f}\n[time 2]: {:.3f}'.format(t2-t1, t3-t2), flush=True)
            return res


def build_matcher(args):
    return DECOLA_HungarianMatcher(cost_class=args.set_cost_class,
                                   cost_bbox=args.set_cost_bbox,
                                   cost_giou=args.set_cost_giou)
