import json 
import os 
import torchvision 
import argparse 
import torch 
from torchvision.ops import nms 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_root', default='datasets/imagenet/pseudo_annotations/')
    parser.add_argument('--annotation_name', default='imagenet_lvis_v1_decola_phase1_r50_21k_zeroshot_4x')
    parser.add_argument('--iou_thres', type=float, default=0.9)
    args = parser.parse_args()

    x = json.load(open(f'{args.annotation_root}/{args.annotation_name}.json', 'r'))

    img2boxes = {} 
    img2scores = {}
    img2cats = {}
    for ann in x['annotations']:
        if ann['image_id'] not in img2boxes:
            img2boxes[ann['image_id']] = [] 
            img2scores[ann['image_id']] = []
            img2cats[ann['image_id']] = []
        img2boxes[ann['image_id']].append(ann['bbox'])
        img2scores[ann['image_id']].append(ann['score'])
        img2cats[ann['image_id']].append(ann['category_id'])
    
    # format 
    for k in img2boxes.keys():
        img2boxes[k] = torch.tensor(img2boxes[k])
        img2scores[k] = torch.tensor(img2scores[k])
        img2cats[k] = torch.tensor(img2cats[k])
    
    # apply nms 
    for k in img2boxes.keys():
        boxes = img2boxes[k] 
        scores = img2scores[k]
        cats = img2cats[k]
        idxs = nms(boxes, scores, args.iou_thres)
        img2boxes[k] = boxes[idxs]
        img2scores[k] = scores[idxs]
        img2cats[k] = cats[idxs]
    
    new_anns = [] 
    i = 1
    for k in img2boxes.keys():
        boxes = img2boxes[k].tolist() 
        scores = img2scores[k].tolist()
        cats = img2cats[k].tolist()
        for box, score, cat in zip(boxes, scores, cats):
            ann = {'image_id': k, 'category_id': cat, 'bbox': box, 'score': score, 'id': i}
            i += 1 
            new_anns.append(ann)
    
    print('filtering done.', len(x['annotations']), len(new_anns))
    x['annotations'] = new_anns 
    
    print('writing', f'{args.annotation_root}/{args.annotation_name}.json ...')
    json.dump(x, open(f'{args.annotation_root}/{args.annotation_name}.json', 'w'))
    print('done.')