import numpy as np
import torch
from frame_tools import left_right_frame_padding, split_frame_with_overlap
from mllm import mllm_recognize
from omni_mask_merge import fuse_instances
from oneformer_predictor import OneFormerPredictor
from cropformer_predictor import CropFormerPredictor
import concurrent.futures
from utils import multi_mask_iou_matrix, merge_stuff_masks
global classes_list, stuff_list
from class_tools import COCO_MAP, CITYSCAPES_MAP, ADE20K_MAP, converted_oneformer_label, classes_list, stuff_list


class FramePredictor:
    def __init__(self, device: str,
                 max_workers: int = 8,
                 pad_ratio: float = 0.125,
                 overlap_ratio: float = 0.125,
                 mask_area_threshold: int = 300,
                 match_iou_threshold: float = 0.6,
                 merge_iou_threshold: float = 0.5,
                 other_config: dict = {"CropFormer": {},
                                       "OneFormer": {}}
                 ):
        self.pad_ratio = pad_ratio
        self.overlap_ratio = overlap_ratio
        self.device = device
        self.cropformer_predictor = CropFormerPredictor(**other_config["CropFormer"])
        self.cityscapes_oneformer_predictor = OneFormerPredictor(dataset_name="cityscapes", device=device,
                                                                 **other_config["OneFormer"])
        self.coco_oneformer_predictor = OneFormerPredictor(dataset_name="coco", device=device,
                                                           **other_config["OneFormer"])
        self.ade20k_oneformer_predictor = OneFormerPredictor(dataset_name="ade20k", device=device,
                                                             **other_config["OneFormer"])
        self.max_workers = max_workers
        self.mask_area_threshold = mask_area_threshold
        self.match_iou_threshold = match_iou_threshold
        self.merge_iou_threshold = merge_iou_threshold

    def pad_and_split(self, image, original_width):
        padded_image, pad_size = left_right_frame_padding(image, self.pad_ratio)
        parts, overlap_size = split_frame_with_overlap(padded_image, original_width, pad_size,
                                                       self.overlap_ratio)
        return parts, pad_size, overlap_size

    def predict_and_merge(self, predictor, split_parts: tuple, pad_size: int, overlap_size: int,
                          is_entity: bool = False):
        left_part, middle_part, right_part = split_parts
        if is_entity:
            _, left_masks = predictor(left_part)
            _, middle_masks = predictor(middle_part)
            _, right_masks = predictor(right_part)
            left_labels = torch.zeros(len(left_masks)).to(left_masks.device)
            middle_labels = torch.zeros(len(middle_masks)).to(middle_masks.device)
            right_labels = torch.zeros(len(right_masks)).to(right_masks.device)
        else:
            left_labels, left_masks = predictor(left_part)
            middle_labels, middle_masks = predictor(middle_part)
            right_labels, right_masks = predictor(right_part)
        if left_labels is None or middle_labels is None or right_labels is None:
            return None, None
        left_result = (left_labels, left_masks)
        middle_result = (middle_labels, middle_masks)
        right_result = (right_labels, right_masks)
        if is_entity:
            fused_classes, fused_masks = fuse_instances(
                [left_result, middle_result, right_result],
                pad_size, overlap_size,
                area_threshold=self.mask_area_threshold,
                merge_iou_threshold=self.merge_iou_threshold
            )
        else:
            fused_classes, fused_masks = fuse_instances(
                [left_result, middle_result, right_result],
                pad_size, overlap_size,
                merge_iou_threshold=self.merge_iou_threshold
            )
        return fused_classes, fused_masks

    def process_entity_mask(self, entity_idx, entity_mask, datasets):
        matching_classes = []
        for dataset_classes, dataset_masks in datasets:
            ious = multi_mask_iou_matrix(entity_mask.unsqueeze(0), dataset_masks)
            matching_mask_indices = (ious > self.match_iou_threshold).nonzero(as_tuple=True)
            if len(matching_mask_indices[0]) == 0:
                continue
            max_iou_index = torch.argmax(ious)
            matching_classes.append(dataset_classes[max_iou_index])
        set_classes = set(matching_classes)
        if len(set_classes) == 1 and len(matching_classes) >= 2:
            return True, entity_idx, matching_classes[0]
        else:
            return False, entity_idx, None

    def predict(self, image: np.ndarray, use_llm: bool = True):
        image_size = image.shape[:2]
        split_parts, pad_size, overlap_size = self.pad_and_split(image, image_size[1])
        oneformer_result = []
        _, entity_masks = self.predict_and_merge(self.cropformer_predictor, split_parts,
                                                 pad_size, overlap_size, True)
        cs_classes, cs_masks = self.predict_and_merge(self.cityscapes_oneformer_predictor, split_parts,
                                                      pad_size, overlap_size)
        if cs_classes is not None:
            cs_semantic_classes = self.cityscapes_oneformer_predictor.get_semantic_information(cs_classes)
            cs_classes = converted_oneformer_label(cs_semantic_classes, CITYSCAPES_MAP)
            cs_classes, cs_masks = merge_stuff_masks(cs_classes, cs_masks)
            oneformer_result.append((cs_classes, cs_masks))
        coco_classes, coco_masks = self.predict_and_merge(self.coco_oneformer_predictor, split_parts,
                                                          pad_size, overlap_size)
        if coco_classes is not None:
            coco_semantic_classes = self.coco_oneformer_predictor.get_semantic_information(coco_classes)
            coco_classes = converted_oneformer_label(coco_semantic_classes, COCO_MAP)
            coco_classes, coco_masks = merge_stuff_masks(coco_classes, coco_masks)
            oneformer_result.append((coco_classes, coco_masks))
        ade20k_classes, ade20k_masks = self.predict_and_merge(self.ade20k_oneformer_predictor, split_parts,
                                                              pad_size, overlap_size)
        if ade20k_classes is not None:
            ade20k_semantic_classes = self.ade20k_oneformer_predictor.get_semantic_information(ade20k_classes)
            ade20k_classes = converted_oneformer_label(ade20k_semantic_classes, ADE20K_MAP)
            ade20k_classes, ade20k_masks = merge_stuff_masks(ade20k_classes, ade20k_masks)
            oneformer_result.append((ade20k_classes, ade20k_masks))
        matched_index_list = []
        matched_label_list = []
        unmatched_index_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_entity_mask, entity_idx, entity_mask,
                                oneformer_result)
                for entity_idx, entity_mask in enumerate(entity_masks)
            ]

            for future in concurrent.futures.as_completed(futures):
                if_match, mask_index, semantic_class = future.result()
                if if_match:
                    matched_index_list.append(mask_index)
                    matched_label_list.append(semantic_class)
                else:
                    unmatched_index_list.append(mask_index)
        if use_llm:
            llm_visualize_index_list = []
            llm_visualize_label_label = []
            if len(unmatched_index_list) != 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(mllm_recognize, image, mask, classes_list, entity_index)
                        for entity_index, mask in zip(unmatched_index_list, entity_masks[unmatched_index_list])
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        mask_index, llm_infor = future.result()
                        if mask_index is not None:
                            matched_index_list.append(mask_index)
                            matched_label_list.append(llm_infor["Label"])
                            llm_visualize_index_list.append(mask_index)
                            llm_visualize_label_label.append(llm_infor["Label"])
                            if llm_infor["Label"] not in classes_list:
                                classes_list.append(llm_infor["Label"])
                                if llm_infor["Thing/Stuff"] == "Stuff":
                                    stuff_list.append(llm_infor["Label"])
        else:
            for entity_index, mask in zip(unmatched_index_list, entity_masks[unmatched_index_list]):
                matched_index_list.append(entity_index)
                matched_label_list.append("unknown")
        predicted_masks = entity_masks[matched_index_list]
        matched_label_list, predicted_masks = merge_stuff_masks(matched_label_list, predicted_masks)
        return predicted_masks, matched_label_list
