import json 
import os 
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', default='datasets/imagenet/ImageNet-LVIS/')
    parser.add_argument('--annotation_path', default='datasets/imagenet/annotations/imagenet_lvis_image_info.json')
    parser.add_argument('--out_root', default='datasets/imagenet/pseudo_annotations/')
    parser.add_argument('--annotation_name', default='imagenet_lvis_v1_decola_phase1_r50_21k_standard_4x')
    parser.add_argument('--sizes', nargs='+', required=True)
    args = parser.parse_args()

    pseudo_anns_list = []
    for size in args.sizes:
        pseudo_anns = json.load(open(f'datasets/imagenet/pseudo_annotations/{args.annotation_name}_minsize_{size}.json', 'r'))
        pseudo_anns_list.append(pseudo_anns)
    
    images = pseudo_anns_list[0]['images']
    annotations = []
    image_to_anns = {}
    for pseudo_anns in pseudo_anns_list:
        for ann in pseudo_anns['annotations']:
            if ann['image_id'] not in image_to_anns:
                image_to_anns[ann['image_id']] = []
            image_to_anns[ann['image_id']].append(ann)
    
    combined_ann_counter = 0 
    for anns in image_to_anns.values():
        for ann in anns:
            combined_ann_counter += 1 # id starts from 1 
            ann['id'] = combined_ann_counter 
            annotations.append(ann)

    categories = pseudo_anns_list[0]['categories']
    multi_res_annotations = {'images': images, 'annotations': annotations, 'categories': categories}

    # save
    if not os.path.exists(os.path.join(args.out_root)):
        os.makedirs(args.out_root)
    save_path = os.path.join(args.out_root, f'{args.annotation_name}.json')
    json.dump(multi_res_annotations, open(save_path, 'w'))