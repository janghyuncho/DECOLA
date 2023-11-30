import wandb 
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask, _create_text_labels
import numpy as np  
import torch 


def wandb_visualize(self, inputs, images, processed_results, opacity=0.8):
    # NOTE: Hack to use input as visualization image.
    images_raw = [x["image"].float().to('cpu') for x in inputs]
    images_vis = [retry_if_cuda_oom(sem_seg_postprocess)(img, img_sz, x.get("height", img_sz[0]), x.get("width", img_sz[1])) 
                    for img, img_sz, x in zip(images_raw, images.image_sizes, inputs)]
    images_vis = [img.to('cpu') for img in images_vis]
    result_vis = [r["predictions"].to('cpu') for r in processed_results]
    target_vis = [r["gt_instances"].to('cpu') for r in processed_results]
    image, instances, targets = images_vis[0], result_vis[0], target_vis[0]
    image = image.permute(1, 2, 0).to(torch.uint8)
    white = np.ones(image.shape) * 255
    image = image * opacity + white * (1-opacity) 

    visualizer = Partvisualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    image_pd = wandb.Image(vis_output.get_image())
    wandb.log({"predictions": image_pd})

    visualizer = Partvisualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_instance_predictions(predictions=targets)
    
    image_gt = wandb.Image(vis_output.get_image())
    wandb.log({"ground_truths": image_gt})


class Visualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("part_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.7
        else:
            colors = None
            alpha = 0.6

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.6

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

