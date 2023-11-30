import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.lvis_evaluation import LVISEvaluator


class CustomLVISEvaluator(LVISEvaluator):
    def __init__(
        self,
        *args,
        max_dets_per_class_per_image=-1,
        category_ids=[],
        just_store_predictions=False,
        prediction_save_filename="",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # (vincent) use class ids and per-class limit.
        self._max_dets_per_class_per_image = max_dets_per_class_per_image
        self._category_ids = category_ids
        self._just_store_predictions = just_store_predictions
        self._prediction_save_filename = prediction_save_filename
        

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._just_store_predictions:
            self._logger.info("Just storing predictions.")
            file_path = os.path.join(self._output_dir, "{}.json".format(self._prediction_save_filename))
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                # class_names=self._metadata.get("thing_classes"), # not used in the original code. 
                max_dets_per_class_per_image=self._max_dets_per_class_per_image,
                category_ids=self._category_ids,
            )
            self._results[task] = res


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, max_dets_per_image=None, 
    max_dets_per_class_per_image=-1, category_ids=[]
):
    """
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        category_ids (list[int]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset
    
    from .custom_lvis_results import CustomLVISResults
    from .custom_lvis_eval import CustomLVISEval

    logger.info(f"Evaluating with max detections per image = {max_dets_per_image}")
    logger.info(f"Evaluating with max detections per class per image = {max_dets_per_class_per_image}")
    if len(category_ids) > 0:
        logger.info(f"Evaluating on subset of categories:\n{category_ids}\n")
    lvis_results = CustomLVISResults(lvis_gt, lvis_results, 
                                     max_dets=max_dets_per_image, 
                                     max_dets_per_class=max_dets_per_class_per_image)
    lvis_eval = CustomLVISEval(lvis_gt, lvis_results, iou_type, category_ids=category_ids)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
    return results
