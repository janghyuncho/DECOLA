
import copy
from typing import Optional, List
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy 
from models.ops.modules import MSDeformAttn
from torchvision.ops.boxes import batched_nms
from ..utils import load_class_freq, get_fed_loss_inds
from models.deformable_transformer import (DeformableTransformer, 
DeformableTransformerDecoderLayer, DeformableTransformerDecoder, _get_clones)
import detectron2.utils.comm as comm
import time 

class DECOLA_DeformableTransformer(DeformableTransformer):
    def __init__(self, num_classes, 
                 per_prompt_topk=300,  
                 use_pixel_feature_as_query=False, 
                 use_prompt_embed_as_query=False,
                 zs_weight_dim=512,
                 with_image_labels=False,
                 oracle_evaluation=False,
                 first_stage_enc_loss_per_prompt_topk=-1,
                 prob_a_object=1.0,
                 look_forward_twice=False,
                 assign_first_stage=True,
                 pre_nms_per_level_topk=1000,
                 **kwargs):
        super(DECOLA_DeformableTransformer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.per_prompt_topk = per_prompt_topk
        self.use_pixel_feature_as_query = use_pixel_feature_as_query
        self.use_prompt_embed_as_query = use_prompt_embed_as_query
        self.with_image_labels = with_image_labels
        self.first_stage_enc_loss_per_prompt_topk = first_stage_enc_loss_per_prompt_topk
        self.zs_weight_dim = zs_weight_dim
        self.prob_a_object = prob_a_object
        self.oracle_evaluation = oracle_evaluation
        self.look_forward_twice = look_forward_twice
        if use_pixel_feature_as_query:
            self.pix_trans = nn.Linear(self.d_model, self.d_model)
            self.pix_trans_norm = nn.LayerNorm(self.d_model)
        if use_prompt_embed_as_query:
            self.prompt_embed_trans = nn.Linear(zs_weight_dim, self.d_model)
            self.prompt_embed_trans_norm = nn.LayerNorm(self.d_model)

        decoder_layer = DECOLA_DeformableTransformerDecoderLayer(d_model=self.d_model, 
                                                                d_ffn=kwargs['dim_feedforward'],
                                                                dropout=kwargs['dropout'], 
                                                                activation=kwargs['activation'],
                                                                n_levels=kwargs['num_feature_levels'], 
                                                                n_heads=self.nhead, 
                                                                n_points=kwargs['dec_n_points'])
        self.decoder = DECOLA_DeformableTransformerDecoder(decoder_layer, 
                                                        kwargs['num_decoder_layers'], 
                                                        kwargs['return_intermediate_dec'],
                                                        look_forward_twice)
        
        # DETA-specific
        self.assign_first_stage = assign_first_stage
        self.pre_nms_per_level_topk = pre_nms_per_level_topk

    def encoding_stage(self, srcs, masks, pos_embeds):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return (memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios)


    def generate_proposals(self, 
                           output_scores, 
                           output_coords, 
                           output_memory, 
                           enc_prompt_inds, 
                           enc_prompt_embeds, 
                           level_ids, 
                           spatial_shapes,
                           output_proposals,
    ):
        if self.use_a_object:
            return self.generate_object_prompt_proposals(output_scores, 
                                                         output_coords, 
                                                         output_memory, 
                                                         enc_prompt_inds, 
                                                         enc_prompt_embeds, 
                                                         level_ids, 
                                                         spatial_shapes, 
                                                         output_proposals)
        else:
            return self.generate_prompt_specific_proposals(output_scores, 
                                                           output_coords, 
                                                           output_memory, 
                                                           enc_prompt_inds, 
                                                           enc_prompt_embeds, 
                                                           level_ids, 
                                                           spatial_shapes, 
                                                           output_proposals)


    def assignment_based_proposals(self, proposal_logit, proposal_boxes, topk,  B, level_ids, spatial_shapes):
        topk_proposals = []
        for b in range(B):
            prop_boxes_b = proposal_boxes[b]
            prop_logits_b = proposal_logit[b]

            # pre-nms per-level topk
            pre_nms_topk = 1000
            pre_nms_inds = []
            for lvl in range(len(spatial_shapes)):
                lvl_mask = level_ids == lvl
                pre_nms_inds.append(torch.topk(prop_logits_b.sigmoid() * lvl_mask, pre_nms_topk)[1])
            pre_nms_inds = torch.cat(pre_nms_inds)

            # nms on topk indices
            post_nms_inds = batched_nms(prop_boxes_b[pre_nms_inds], prop_logits_b[pre_nms_inds], level_ids[pre_nms_inds], 0.9)
            keep_inds = pre_nms_inds[post_nms_inds]

            if len(keep_inds) < self.two_stage_num_proposals:
                print(f'[WARNING] nms proposals ({len(keep_inds)}) < {self.two_stage_num_proposals}, running naive topk')
                keep_inds = torch.topk(proposal_logit[b], topk)[1]

            # keep top Q/L indices for L levels
            q_per_l = topk // len(spatial_shapes)
            is_level_ordered = level_ids[keep_inds][None] == torch.arange(len(spatial_shapes), device=level_ids.device)[:,None]  # LS
            keep_inds_mask = is_level_ordered & (is_level_ordered.cumsum(1) <= q_per_l)  # LS
            keep_inds_mask = keep_inds_mask.any(0)  # S

            # pad to Q indices (might let ones filtered from pre-nms sneak by... unlikely because we pick high conf anyways)
            if keep_inds_mask.sum() < topk:
                num_to_add = topk - keep_inds_mask.sum()
                pad_inds = (~keep_inds_mask).nonzero()[:num_to_add]
                keep_inds_mask[pad_inds] = True

            # index
            keep_inds_topk = keep_inds[keep_inds_mask]
            topk_proposals.append(keep_inds_topk)

        return topk_proposals


    def generate_object_prompt_proposals(self, 
                                         output_scores, 
                                         output_coords, 
                                         output_memory, 
                                         prompt_inds, 
                                         enc_prompt_embeds, 
                                         level_ids, 
                                         spatial_shapes,
                                         output_proposals,
    ):
        """
        output_scores: (B,S,1)
        output_coords: (B,S,4)
        output_memory: (B,S,C)
        enc_prompt_inds: (B,1)
        enc_prompt_embeds: (B,C,1)
        """
        B, S, _ = output_scores.shape 
        _, _, C = output_memory.shape

        topk = min(self.two_stage_num_proposals, output_scores[..., 0].shape[1])
        assert self.assign_first_stage, "This module only supports DETA."

        proposal_logit = output_scores[..., 0]
        proposal_boxes = box_cxcywh_to_xyxy(output_coords.sigmoid().float()).clamp(0, 1)
        topk_proposals = self.assignment_based_proposals(proposal_logit, proposal_boxes, topk, B, level_ids, spatial_shapes)
        topk_proposals = torch.stack(topk_proposals)
        topk_coords_unact = torch.gather(output_coords, 1, topk_proposals.unsqueeze(2).expand(B, self.two_stage_num_proposals, 4))
        if self.use_pixel_feature_as_query:
            topk_feats = torch.gather(output_memory, 1, topk_proposals.unsqueeze(2).expand(B, self.two_stage_num_proposals, C))
        else:
            topk_feats = None 
        
        if self.use_prompt_embed_as_query:
            prompt_embeds = enc_prompt_embeds.permute(0, 2, 1)
        else:
            prompt_embeds = None 

        return topk_coords_unact, None, topk_feats, prompt_embeds, output_proposals


    def generate_prompt_specific_proposals(
        self, 
        output_scores, 
        output_coords, 
        output_memory, 
        enc_prompt_inds, 
        enc_prompt_embeds,
        level_ids, 
        spatial_shapes,
        output_proposals,
    ):
        """
        output_scores: (B,S,K)
        output_coords: (B,S,4)
        output_memory: (B,S,C)
        enc_prompt_inds: (B,K)
        enc_prompt_embeds: (B,C,K)
        """
        B, S, _ = output_coords.shape
        _, _, K = output_scores.shape
        _, _, C = output_memory.shape

        # per-prompt limit
        if self.per_prompt_topk < 0:
            per_prompt_topk = S 
        else:
            per_prompt_topk = self.per_prompt_topk 
        
        assert self.assign_first_stage
        
        topk_proposals_all = []
        for k in range(K):
            proposal_logit = output_scores[..., k] # B,S
            proposal_boxes = box_cxcywh_to_xyxy(output_coords.sigmoid().float()).clamp(0, 1)
            topk_proposals = self.assignment_based_proposals(proposal_logit, proposal_boxes, per_prompt_topk, B, level_ids, spatial_shapes)
            topk_proposals_all.append(torch.stack(topk_proposals))
        per_prompt_topk_inds = torch.stack(topk_proposals_all, dim=2)
        # per_prompt_topk_scores, per_prompt_topk_inds = output_scores.topk(per_prompt_topk, dim=1) 
        per_prompt_topk_coords = output_coords.unsqueeze(2).expand(B, S, K, 4) # (B,S,K,4)
        per_prompt_topk_coords = torch.gather(per_prompt_topk_coords, 1, \
                                              per_prompt_topk_inds.unsqueeze(3).expand(B, per_prompt_topk, K, 4))  # (B,N,K,4)
        enc_prompt_inds = enc_prompt_inds.unsqueeze(1).expand(B, per_prompt_topk, K)                               # (B,N,K)
        output_proposals = output_proposals.unsqueeze(2).expand(B, S, K, 4)

        topk_coords = per_prompt_topk_coords.flatten(1, 2) # B,NK,4  
        prompt_inds = enc_prompt_inds.flatten(1, 2)   
        if self.use_pixel_feature_as_query:
            per_prompt_topk_feats = torch.gather(output_memory.unsqueeze(2).expand(B, S, K, C), \
                                                1, per_prompt_topk_inds.unsqueeze(3).expand(B, per_prompt_topk, K, C))  # (B,N,K,C)
            topk_feats = per_prompt_topk_feats.flatten(1, 2)  # (B,NK,C)
        else:
            topk_feats = None 
        
        if self.use_prompt_embed_as_query:
            prompt_embeds = enc_prompt_embeds.permute(0, 2, 1).unsqueeze(1).expand(B, per_prompt_topk, K, self.zs_weight_dim)  # B,N,K,C'
            prompt_embeds = prompt_embeds.flatten(1, 2)     # B,NK,C
        else:
            prompt_embeds = None 
            
        # print(topk_coords.shape, prompt_inds.shape, output_proposals.shape, flush=True)
        return topk_coords, prompt_inds, topk_feats, prompt_embeds, output_proposals
        

    def forward_decoder(self, decoder_input):
        memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten, output_memory, output_proposals, level_ids, targets = decoder_input

        self.is_image_label = 'image_label' in targets[0]
        # print('prob label: {}\nuse_a_object: {}\nis_image_label: {}\n'.format(self.prob_a_object, self.use_a_object, self.is_image_label), flush=True)
        t0 = time.time()
        # print(2, self.oracle_evaluation, self.use_a_object, flush=True)
        enc_outputs_class, enc_prompt_inds, enc_prompt_embeds, use_a_object = self.decoder.class_embed[self.decoder.num_layers](
                                                                                        output_memory, 
                                                                                        training=self.training,
                                                                                        targets=targets, 
                                                                                        is_image_label=self.is_image_label,
                                                                                        use_a_object=self.use_a_object,
                                                                                        oracle_evaluation=self.oracle_evaluation)
        t1 = time.time()
        # during eval, this flag may change. 
        self.use_a_object = use_a_object 
        box_output = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = box_output + output_proposals # B,S,4
    
        # per-prompt promposals -> topk 
        topk_coords_unact, topk_prompt_inds, topk_feats, topk_prompt_embeds, output_proposals = \
            self.generate_proposals(enc_outputs_class, 
                                    enc_outputs_coord_unact, 
                                    output_memory, 
                                    enc_prompt_inds, 
                                    enc_prompt_embeds, 
                                    level_ids, 
                                    spatial_shapes, 
                                    output_proposals)
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.split(pos_trans_out, memory.shape[2], dim=2)

        if self.use_pixel_feature_as_query:
            tgt = tgt + self.pix_trans_norm(self.pix_trans(topk_feats.detach()))
        if self.use_prompt_embed_as_query:
            tgt = tgt + self.prompt_embed_trans_norm(self.prompt_embed_trans(topk_prompt_embeds))

        t2 = time.time()
        # 2nd stage decoding 
        hs, inter_references_out = self.decoder(tgt, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios, 
                                                query_embed, mask_flatten,
                                                per_prompt_topk=self.per_prompt_topk,
                                                per_prompt_self_attention=(not self.use_a_object))
        
        t3 = time.time()
        # print('[transformer]\n[time 1]: {:.3f}\n[time 2]: {:.3f}\n[time 3]: {:.3f}'.format(t1-t0, t2-t1, t3-t2), flush=True)
        return hs, init_reference_out, inter_references_out, topk_prompt_inds, enc_outputs_class, enc_outputs_coord_unact, enc_prompt_inds, output_proposals.sigmoid()


    def forward(self, srcs, masks, pos_embeds, query_embed, targets):
        # ddetr encoding stage
        (memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios) = \
            self.encoding_stage(srcs, masks, pos_embeds)

        # prepare input for decoder
        output_memory, output_proposals, level_ids = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        # 1st stage output : B,S,K 
        # enc_prompt_embeds: B,C,K
        decoder_input = (memory, 
                         spatial_shapes, 
                         level_start_index, 
                         valid_ratios, 
                         mask_flatten, 
                         output_memory, 
                         output_proposals, 
                         level_ids,
                         targets
                    )
        return self.forward_decoder(decoder_input)


    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        level_ids = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
            level_ids.append(grid.new_ones(H_ * W_, dtype=torch.long) * lvl)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        level_ids = torch.cat(level_ids)
        return output_memory, output_proposals, level_ids

class DECOLA_DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers, 
        return_intermediate=False, 
        look_forward_twice=False
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self, 
        tgt, 
        reference_points, 
        src, 
        src_spatial_shapes, 
        src_level_start_index, 
        src_valid_ratios,
        query_pos=None, 
        src_padding_mask=None, 
        per_prompt_topk=300, 
        per_prompt_self_attention=False,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, 
                           query_pos, 
                           reference_points_input, 
                           src, src_spatial_shapes, 
                           src_level_start_index, 
                           src_padding_mask,
                           per_prompt_topk=per_prompt_topk,
                           per_prompt_self_attention=per_prompt_self_attention)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
            
class DECOLA_DeformableTransformerDecoderLayer(DeformableTransformerDecoderLayer):
    def roll_embedding(self, embed, per_prompt_topk, per_prompt_self_attention=False):
        b, nk, c = embed.shape 
        if per_prompt_self_attention:
            embed = embed.reshape(b, -1, per_prompt_topk, c).permute(0, 2, 1, 3) # (B,K,N,C)
            embed = embed.flatten(0, 1) # (BK,N,C)
            # print('rolled {} -> {} '.format((b, nk, c), embed.shape), flush=True)
        return embed

    def unroll_embedding(self, embed, per_prompt_topk, per_prompt_self_attention=False):
        bk, n, c = embed.shape 
        if per_prompt_self_attention:
            embed = embed.reshape(-1, per_prompt_topk, n, c).permute(0, 2, 1, 3) # (B,N,K,C)
            embed = embed.flatten(1, 2)
            # print('unrolled {} -> {} '.format((bk, n, c), embed.shape), flush=True)
        return embed

    def forward(self, 
                tgt, query_pos, 
                reference_points, src, 
                src_spatial_shapes, 
                level_start_index, 
                src_padding_mask=None, 
                per_prompt_topk=300, 
                per_prompt_self_attention=False):
        # self attention
        tgt = self.roll_embedding(tgt, per_prompt_topk, per_prompt_self_attention)
        query_pos = self.roll_embedding(query_pos, per_prompt_topk, per_prompt_self_attention)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = self.unroll_embedding(tgt, per_prompt_topk, per_prompt_self_attention)
        query_pos = self.unroll_embedding(query_pos, per_prompt_topk, per_prompt_self_attention)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

