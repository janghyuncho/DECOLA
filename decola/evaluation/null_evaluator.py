from typing import Dict, List
from detectron2.utils.comm import synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator


class NullEvaluator(DatasetEvaluator):
    def reset(self):
        return 

    def process(self, inputs: List[Dict], outputs: Dict):
        return 
            
    def evaluate(self):
        synchronize()
        return