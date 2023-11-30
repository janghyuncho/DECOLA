import copy 
import torch
import time 
import torch.nn.functional as F
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       interpolate, inverse_sigmoid)
from models.deformable_detr import DeformableDETR

class DECOLA_DETR(DeformableDETR):
    def forward_train(self, samples: NestedTensor, targets=None):
        t0 = time.time()
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None 
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        t1 = time.time()
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs, 
            init_reference, 
            inter_references, 
            topk_prompt_inds, 
            enc_outputs_class_raw, 
            enc_outputs_coord_unact_raw, 
            enc_prompt_inds_raw,
            anchors
        ) = self.transformer(srcs, masks, pos, query_embeds, targets)
        t2 = time.time()
        outputs_classes_loss = []
        outputs_classes_match = []
        outputs_coords  = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class_loss, outputs_class_match = self.class_embed[lvl](hs[lvl], 
                                                            inds=topk_prompt_inds,
                                                            training=True,
                                                            second_stage=True,
                                                            is_image_label=self.transformer.is_image_label,
                                                            use_a_object=self.transformer.use_a_object)
                
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_loss.append(outputs_class_loss)
            outputs_classes_match.append(outputs_class_match)
            outputs_coords.append(outputs_coord)
        outputs_classes_loss = torch.stack(outputs_classes_loss)
        outputs_classes_match = torch.stack(outputs_classes_match)
        outputs_coords = torch.stack(outputs_coords)

        outputs_classes_loss[-1] += enc_outputs_class_raw.new_zeros([1])[0]

        t3 = time.time()
        out = {'pred_logits_loss': outputs_classes_loss[-1], 
               'pred_logits_match': outputs_classes_match[-1], 
               'pred_logits': outputs_classes_loss[-1], \
               'pred_boxes': outputs_coords[-1], 
               'prompt_inds': topk_prompt_inds, 
               'prompt_inds_all': enc_prompt_inds_raw[0],
               'init_reference': init_reference
               }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes_loss, outputs_classes_match, \
                                    outputs_coords, topk_prompt_inds, enc_prompt_inds_raw[0])

        if self.transformer.use_a_object:
            enc_outputs_coord = enc_outputs_coord_unact_raw.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class_raw, 'pred_boxes': enc_outputs_coord, 'anchors': anchors}
        else:
            enc_outputs_class = enc_outputs_class_raw.unsqueeze(3)
            enc_prompt_inds = enc_prompt_inds_raw.unsqueeze(1).repeat(1, enc_outputs_class.shape[1], 1)        # B,S,K
            enc_outputs_coord_raw = enc_outputs_coord_unact_raw.sigmoid()
            enc_outputs_coord = enc_outputs_coord_raw.unsqueeze(2).repeat(1, 1, enc_outputs_class.shape[2], 1) # B,S,K,4

            # topk 
            if self.transformer.first_stage_enc_loss_per_prompt_topk > 0:
                topk = min(self.transformer.first_stage_enc_loss_per_prompt_topk, enc_outputs_class.shape[1])
                enc_outputs_class, enc_outputs_inds = enc_outputs_class.topk(topk, dim=1)
                enc_outputs_coord = torch.gather(enc_outputs_coord, 1, enc_outputs_inds.repeat(1, 1, 1, 4))
                enc_prompt_inds = torch.gather(enc_prompt_inds, 1, enc_outputs_inds.squeeze(3))
                anchors = torch.gather(anchors, 1, enc_outputs_inds.repeat(1, 1, 1, 4))
            enc_prompt_inds = enc_prompt_inds.flatten(1, 2)     # B,SK
            enc_outputs_coord = enc_outputs_coord.flatten(1, 2) # B,SK,4
            enc_outputs_class = enc_outputs_class.flatten(1, 2) # B,SK,1 
            anchors = anchors.flatten(1, 2)
            
            out['enc_outputs'] = {'pred_logits_match': enc_outputs_class, 
                                  'pred_logits': enc_outputs_class, 
                                  'pred_logits_loss': enc_outputs_class, 
                                  'pred_boxes': enc_outputs_coord, 
                                  'prompt_inds': enc_prompt_inds,
                                  'anchors': anchors,
                                  }

        # print('[detr]\n[time 1]: {:.3f}\n[time 2]: {:.3f}\n[time 3]: {:.3f}'.format(t1-t0, t2-t1, t3-t2), flush=True)
        return out

    def forward_test(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None 
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        (
            hs, 
            init_reference, 
            inter_references, 
            topk_prompt_inds, 
            enc_outputs_class_raw, 
            enc_outputs_coord_unact_raw, 
            enc_prompt_inds_raw,
            anchors
        ) = self.transformer(srcs, masks, pos, query_embeds, targets)
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class, _ = self.class_embed[lvl](hs[lvl], 
                                                     inds=topk_prompt_inds, 
                                                     training=False,
                                                     second_stage=True,
                                                     is_image_label=self.transformer.is_image_label,
                                                     use_a_object=self.transformer.use_a_object)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'prompt_inds': topk_prompt_inds, 'init_reference': init_reference}

        return out

    def forward(self, samples: NestedTensor, targets=None):
        if self.transformer.training:
            return self.forward_train(samples, targets)
        else:
            return self.forward_test(samples, targets)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class_loss, outputs_class_match, outputs_coord, topk_prompt_inds, prompt_inds_all):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits_loss': a1, 'pred_logits_match': a2,  'pred_logits': a1, 'pred_boxes': b, \
                 'prompt_inds': copy.deepcopy(topk_prompt_inds), 'prompt_inds_all': copy.deepcopy(prompt_inds_all)}
                for a1, a2, b in zip(outputs_class_loss[:-1], outputs_class_match[:-1], outputs_coord[:-1])]