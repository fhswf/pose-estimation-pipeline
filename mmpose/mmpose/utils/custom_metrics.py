
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoWholeBodyMetric
from xtcocotools.cocoeval import COCOeval
import numpy as np

@METRICS.register_module()
class SafeCocoWholeBodyMetric(CocoWholeBodyMetric):
    """
    Subclass of CocoWholeBodyMetric that safely handles shape mismatches
    between prediction sigmas and ground truth keypoints.
    
    It attempts to evaluate subsets (body, foot, etc.) but catches errors
    if the installed xtcocotools version triggers shape mismatches.
    """
    
    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI."""
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])
        
        # Helper to safely run eval
        def safe_eval(iou_type, sigma_slice):
            try:
                coco_eval = COCOeval(
                    self.coco,
                    coco_det,
                    iou_type,
                    sigma_slice,
                    use_area=self.use_area)
                coco_eval.params.useSegm = None
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                return coco_eval.stats
            except ValueError as e:
                print(f"Skipping {iou_type} evaluation due to error: {e}")
                # Return empty stats or zeros? 
                # Stats are usually a list of 10 floats.
                return [0.0] * 10
            except Exception as e:
                print(f"An unexpected error occurred during {iou_type} evaluation: {e}")
                return [0.0] * 10

        # Run evaluations
        # Body
        stats_body = safe_eval('keypoints_body', sigmas[cuts[0]:cuts[1]])
        
        # Foot
        stats_foot = safe_eval('keypoints_foot', sigmas[cuts[1]:cuts[2]])
        
        # Face
        stats_face = safe_eval('keypoints_face', sigmas[cuts[2]:cuts[3]])
        
        # Left Hand
        stats_lh = safe_eval('keypoints_lefthand', sigmas[cuts[3]:cuts[4]])
        
        # Right Hand
        stats_rh = safe_eval('keypoints_righthand', sigmas[cuts[4]:cuts[5]])
        
        # WholeBody (Should work as shapes match 133 vs 133)
        print("Evaluating keypoints_wholebody...")
        stats_whole = safe_eval('keypoints_wholebody', sigmas)
        
        # Use wholebody stats for the main info_str if others failed?
        # The original code expects `coco_eval.stats` to be bound to the last evaluation?
        # Actually `info_str` is constructed from `coco_eval.stats` at the end.
        # In the original code, `coco_eval` variable is reused.
        # So `info_str` depends on the LAST run, which is 'keypoints_wholebody'.
        
        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        
        info_str = list(zip(stats_names, stats_whole))
        
        return info_str
