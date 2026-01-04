from PIL import Image
import numpy as np
from lang_sam import LangSAM
from typing import List, Dict
import sys
import json

class GeometricRelationAnalyzer:
    def __init__(self):
        self.model = LangSAM()

    
    def analyze_image(self, image_path: str, objects: List[str]):
        image_pil = Image.open(image_path).convert("RGB")
        masks = {}
        
        for obj in objects:
            masks[obj] = []
            # Get multiple instances for each object
            results  = self.model.predict([image_pil], [obj])
            masks_array = results[0]["masks"]
            boxes = results[0]["boxes"]
            labels = results[0]["labels"]
            scores = results[0]["scores"]

            if masks_array is not None and len(scores) > 0:
                max_score_id = np.argmax(scores)
                max_score = scores[max_score_id]
                print(f"Max score for {obj}: {max_score}")
                if max_score > 0.4:  # Confidence threshold
                    mask = masks_array[max_score_id].squeeze()
                    masks[obj].append({
                        'mask': mask,
                        'bbox': boxes[max_score_id],
                        'label': obj,
                        'area': np.sum(mask>0)
                    })

                # Gather all detected mask instances
        mask_data = []
        detected_objects = [obj for obj in objects if len(masks[obj]) > 0]
        for obj in detected_objects:
            mask_data.extend(masks[obj])
        
        # Calculate relationships
        relationships = []
        for i in range(len(mask_data)):
            for j in range(i+1, len(mask_data)):
                relationships += self._get_relationships(mask_data[i], mask_data[j])

        # Write relationships to a JSON file
        with open('relationships.json', 'w') as f:
            json.dump(relationships, f)

        return relationships
    def _are_masks_similar(self, mask1: Dict, mask2: Dict, 
                        size_threshold: float = 0.95, 
                        position_threshold: float = 0.005) -> bool:
        """
        Checks if two masks are similar in both size and position.
        - size_threshold: 0-1 ratio of smaller/larger area (0.8 = 80% similar)
        - position_threshold: Max allowed centroid distance relative to image size
        """
        # Size similarity check
        area_ratio = min(mask1['area'], mask2['area']) / max(mask1['area'], mask2['area'])
        if area_ratio < size_threshold:
            return False

        # Position similarity check
        centroid1 = self._get_mask_metrics(mask1['mask'])[0]
        centroid2 = self._get_mask_metrics(mask2['mask'])[0]
        
        # Calculate Euclidean distance between centroids
        distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        print(f"Centroid distance: {distance}, centroid1: {centroid1}, centroid2: {centroid2}")
        
        # Get image dimensions from mask shape (assuming masks are same size as image)
        height, width = mask1['mask'].shape
        max_possible_distance = np.sqrt(width**2 + height**2)
        print(f"Distance: {distance}, Max possible distance: {max_possible_distance}")
        
        return distance <= (position_threshold * max_possible_distance)

    def _get_relationships(self, mask1: Dict, mask2: Dict, threshold_factor=0.99):
        relations = []
        # Add similarity check first
        if self._are_masks_similar(mask1, mask2):
            relations.append({
                'object1': mask1['label'],
                'relation': 'similar',
                'object2': mask2['label']
            })
            return relations
    
        # Calculate centroids and bounding boxes
        centroid1, bbox1 = self._get_mask_metrics(mask1['mask'])
        centroid2, bbox2 = self._get_mask_metrics(mask2['mask'])
    
        # Dynamic thresholds
        thresh_x = threshold_factor * min(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
        thresh_y = threshold_factor * min(bbox1[3] - bbox1[1], bbox2[3] - bbox2[1])
    
        print(f"Area1 {mask1['label']}: {mask1['area']}, Area2 {mask2['label']}: {mask2['area']}")
        # Check for overlap
        bbox1_contains_bbox2 = self._is_bbox_contained(mask1['bbox'], mask2['bbox'])
        bbox2_contains_bbox1 = self._is_bbox_contained(mask2['bbox'], mask1['bbox'])
        if bbox1_contains_bbox2:
            print(f"bbox1 {mask1['label']} contains bbox2 {mask2['label']}")
            print(bbox1, bbox2)
            relations.append({
                'object1': mask2['label'],
                'relation': 'on',
                'object2': mask1['label']
            })
            return relations
        elif bbox2_contains_bbox1:
            print(f"bbox1 {mask1['label']} is contained in bbox2 {mask2['label']}")
            print(bbox1, bbox2)
            relations.append({
                'object1': mask1['label'],
                'relation': 'on',
                'object2': mask2['label']
            })
            return relations
        elif self._check_mask_overlap(mask1['mask'], mask2['mask']):
            if mask1['area'] > mask2['area']:
                relations.append({
                    'object1': mask2['label'],
                    'relation': 'on',
                    'object2': mask1['label']
                })
            else:
                relations.append({
                    'object1': mask1['label'],
                    'relation': 'on',
                    'object2': mask2['label']
                })
            return relations
    
        # Relative positioning with thresholds
        if abs(centroid1[0] - centroid2[0]) < thresh_x:
            position_x = ""
        elif centroid1[0] < centroid2[0]:
            position_x = "left"
        else:
            position_x = "right"
    
        if abs(centroid1[1] - centroid2[1]) < thresh_y:
            position_y = ""
        elif centroid1[1] < centroid2[1]:
            position_y = "behind"
        else:
            position_y = "in front"

        relations.append({
            'object1': mask1['label'],
            'relation': position_x + ' ' + position_y,
            'object2': mask2['label']
        })
        return relations
    
    def _check_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.15):
        overlap = np.logical_and(mask1 > 0, mask2 > 0)
        return np.mean(overlap) >= threshold

    def _is_bbox_contained(self, bbox_a, bbox_b):
        # bbox: (x_min, y_min, x_max, y_max)
        return (bbox_a[0] <= bbox_b[0] and 
                bbox_a[1] <= bbox_b[1] and 
                bbox_a[2] >= bbox_b[2] and 
                bbox_a[3] >= bbox_b[3])

    def _get_mask_metrics(self, mask: np.ndarray):
        y, x = np.where(mask > 0)
        if len(x) == 0 or len(y) == 0:
            return (0, 0), (0, 0, 0, 0)
        centroid = (x.mean(), y.mean())
        bbox = (x.min(), y.min(), x.max(), y.max())
        return centroid, bbox

    def get_geometric_relations(self, image_path, objects):
        return self.analyze_image(image_path, objects)

# # --------- Script usage example ---------
# if __name__ == "__main__":
#     analyzer = GeometricRelationAnalyzer()
#     # Example: python sam_lang.py path_to_image.jpg
#     image_path = sys.argv[1]
#     objects = ["banana", "lemon", "pear", "table", "red plate", "hammer"]
#     results = analyzer.analyze_image(image_path, objects)
#     print(results)