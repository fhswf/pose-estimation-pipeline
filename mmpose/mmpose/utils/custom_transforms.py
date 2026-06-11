
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
from typing import List, Tuple, Union

@TRANSFORMS.register_module()
class SafeKeypointConverter(BaseTransform):
    """Change the order of keypoints according to the given mapping.
    Safely handles cases where optional keys like keypoints_3d are None.

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
        mapping (list): A list containing mapping indexes. Each element has
            format (source_index, target_index)
    """

    def __init__(self, num_keypoints: int,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]]):
        self.num_keypoints = num_keypoints
        self.mapping = mapping
        if len(mapping):
            source_index, target_index = zip(*mapping)
        else:
            source_index, target_index = [], []

        src1, src2 = [], []
        interpolation = False
        for x in source_index:
            if isinstance(x, (list, tuple)):
                assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                    'length 2'
                src1.append(x[0])
                src2.append(x[1])
                interpolation = True
            else:
                src1.append(x)
                src2.append(x)

        # When paired source_indexes are input,
        # keep a self.source_index2 for interpolation
        if interpolation:
            self.source_index2 = src2

        self.source_index = src1
        self.target_index = list(target_index)
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        num_instances = results['keypoints'].shape[0]

        if 'keypoints_visible' not in results:
            results['keypoints_visible'] = np.ones(
                (num_instances, results['keypoints'].shape[1]))

        if len(results['keypoints_visible'].shape) > 2:
            results['keypoints_visible'] = results['keypoints_visible'][:, :,
                                                                        0]

        # Initialize output arrays
        keypoints = np.zeros((num_instances, self.num_keypoints, 3))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))
        
        # Patch: Check for None explicitly
        key = 'keypoints_3d' if results.get('keypoints_3d') is not None else 'keypoints'
        c = results[key].shape[-1]

        flip_indices = results.get('flip_indices', None)

        # Create a mask to weight visibility loss
        keypoints_visible_weights = keypoints_visible.copy()
        keypoints_visible_weights[:, self.target_index] = 1.0

        # Interpolate keypoints if pairs of source indexes provided
        if self.interpolation:
            keypoints[:, self.target_index, :c] = 0.5 * (
                results[key][:, self.source_index] +
                results[key][:, self.source_index2])
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index] * results[
                    'keypoints_visible'][:, self.source_index2]
            # Flip keypoints if flip_indices provided
            if flip_indices is not None:
                for i, (x1, x2) in enumerate(
                        zip(self.source_index, self.source_index2)):
                    idx = flip_indices[x1] if x1 == x2 else i
                    flip_indices[i] = idx if idx < self.num_keypoints else i
                flip_indices = flip_indices[:len(self.source_index)]
        # Otherwise just copy from the source index
        else:
            keypoints[:,
                      self.target_index, :c] = results[key][:,
                                                            self.source_index]
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index]

        # Update the results dict
        results['keypoints'] = keypoints[..., :2]
        results['keypoints_visible'] = np.stack(
            [keypoints_visible, keypoints_visible_weights], axis=2)
            
        # Patch: Only update 3d if it exists and is not None
        if results.get('keypoints_3d') is not None:
            results['keypoints_3d'] = keypoints
            if 'target_idx' in results:
                 results['lifting_target'] = keypoints[results['target_idx']]
                 results['lifting_target_visible'] = keypoints_visible[
                    results['target_idx']]
        
        results['flip_indices'] = flip_indices

        return results
