import json 
import os 
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', default='datasets/imagenet/ImageNet-LVIS/')
    parser.add_argument('--annotation_path', default='datasets/imagenet/annotations/imagenet_lvis_image_info.json')
    parser.add_argument('--out_root', default='datasets/imagenet/pseudo_annotations/')
    parser.add_argument('--annotation_name', default='decola_phase1_r50_21k_zeroshot_4x_minsize_400')
    parser.add_argument('--prediction_path', default='output/pseudo_labels/DECOLA_SELF_LABELING_Lbase_CLIP_R5021k_4x/inference_imagenet_lvis_v1/lvis_instances_results.json')

    args = parser.parse_args()

    predictions = json.load(open(args.prediction_path, 'r'))

    # set id
    for i in range(len(predictions)):
        predictions[i]['id'] = i + 1

    # make annotations    
    imagenet_annotations = json.load(open(args.annotation_path, 'r'))
    imagenet_annotations['annotations'] = predictions 
    
    # save
    if not os.path.exists(os.path.join(args.out_root)):
        os.makedirs(args.out_root)
    save_path = os.path.join(args.out_root, f'{args.annotation_name}.json')
    json.dump(imagenet_annotations, open(save_path, 'w'))