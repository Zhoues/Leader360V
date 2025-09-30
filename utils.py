import cv2
import numpy as np
import torch
global stuff_list
from class_tools import stuff_list
from pycocotools import mask as coco_mask


def multi_mask_iou_matrix(masks_1: torch.Tensor, masks_2: torch.Tensor):
    masks_1 = masks_1.to(torch.float)
    masks_2 = masks_2.to(torch.float)
    masks_1_area = masks_1.sum(dim=(1, 2))
    masks_2_area = masks_2.sum(dim=(1, 2))
    intersection = torch.einsum('chw,dhw->cd', masks_1, masks_2)
    union = masks_1_area.unsqueeze(1) + masks_2_area.unsqueeze(0) - intersection
    iou_matrix = intersection / union
    iou_matrix = torch.nan_to_num(iou_matrix)
    return iou_matrix


def get_bounding_box(mask: np.ndarray):
    x_indices, y_indices = np.where(mask)
    x_min = x_indices.min()
    x_max = x_indices.max()
    y_min = y_indices.min()
    y_max = y_indices.max()
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def find_closest_point(mask: np.ndarray, centroid: np.ndarray):
    foreground_points = np.argwhere(mask == 1)
    distances = np.linalg.norm(foreground_points - centroid, axis=1)
    closest_index = np.argmin(distances)
    return tuple(foreground_points[closest_index][::-1])


def get_connected_components(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask)
    labels_im = labels_im.astype(np.int16)
    return [labels_im == label for label in range(1, num_labels)]


def center_point(points: np.ndarray):
    return tuple(points.mean(axis=0)[::-1])


def get_cluster_prompts(mask: np.ndarray):
    coords = np.argwhere(mask == 1)
    y_center, x_center = coords.mean(axis=0)

    top = coords[coords[:, 0] == coords[:, 0].min()]
    bottom = coords[coords[:, 0] == coords[:, 0].max()]
    left = coords[coords[:, 1] == coords[:, 1].min()]
    right = coords[coords[:, 1] == coords[:, 1].max()]

    if mask[int(y_center), int(x_center)] == 1:
        points = [
            (x_center, y_center),
            center_point(top),
            center_point(bottom),
            center_point(left),
            center_point(right),
        ]
    else:
        points = [
            find_closest_point(mask, np.array([y_center, x_center])),
            center_point(top),
            center_point(bottom),
            center_point(left),
            center_point(right),
        ]
    points = np.array(points)
    labels = np.ones(len(points))
    return points, labels


def get_masks_point_prompts(masks: np.ndarray):
    point_prompts = []
    for mask in masks:
        components = get_connected_components(mask)
        mask_prompt = []
        for component in components:
            component_prompts = get_cluster_prompts(component)
            mask_prompt.extend(component_prompts)
        mask_prompt = np.array(mask_prompt)
        point_prompts.append(mask_prompt)
    return point_prompts


def match_masks(new_masks: torch.Tensor, sam2_masks: torch.Tensor, matched_iou_threshold: float):
    new_masks_match_idx_list = []
    sam2_masks_match_idx_list = []
    unmatched_new_mask_indices = set(range(len(new_masks)))
    overlap_ratios = multi_mask_iou_matrix(new_masks, sam2_masks)
    filter = overlap_ratios > matched_iou_threshold
    matched_indices = torch.nonzero(filter, as_tuple=False)
    sorted_indices = torch.argsort(overlap_ratios[filter], descending=True)
    matched_indices = matched_indices[sorted_indices]
    for index in matched_indices:
        if index[0] in new_masks_match_idx_list or index[1] in sam2_masks_match_idx_list:
            continue
        new_masks_match_idx_list.append(index[0])
        sam2_masks_match_idx_list.append(index[1])
        unmatched_new_mask_indices.discard(index[0].item())
        #
        # cv2.imwrite(f"mask_dir/new_mask_{index[0]}.jpg", (new_masks[index[0]] * 255).detach().cpu().numpy().astype(np.uint8))
        # cv2.imwrite(f"mask_dir/mask_match_{index[0]}.jpg",
        #             (sam2_masks[index[1]] * 255).detach().cpu().numpy().astype(np.uint8))
    #
    # for index in unmatched_new_mask_indices:
    #     cv2.imwrite(f"mask_dir/un_new_mask_{index}.jpg",
    #                 (new_masks[index] * 255).detach().cpu().numpy().astype(np.uint8))

    return new_masks_match_idx_list, sam2_masks_match_idx_list, list(unmatched_new_mask_indices)


def get_segmented_ratio(masks: torch.Tensor):
    combined_mask = masks.any(dim=0).int()
    nonzero_count_combined = combined_mask.sum()
    total_elements_combined = combined_mask.numel()
    segmented_ratio = nonzero_count_combined / total_elements_combined
    return segmented_ratio


def merge_stuff_masks(label_list: list, masks: torch.Tensor):
    refined_label_list = []
    refined_masks = []
    for label, mask in zip(label_list, masks):
        if label in stuff_list and label in refined_label_list:
            refined_index = refined_label_list.index(label)
            refined_masks[refined_index] = refined_masks[refined_index] | mask
        else:
            refined_label_list.append(label)
            refined_masks.append(mask)
    refined_masks = torch.stack(refined_masks)
    return refined_label_list, refined_masks


def reshape_masks(masks: torch.Tensor):
    B, H, W = masks.shape
    masks_float = masks.to(torch.float)
    intersection = torch.einsum('chw,dhw->cd', masks_float, masks_float)
    areas = masks_float.sum(dim=(1, 2))
    for i in range(B):
        for j in range(i + 1, B):
            if intersection[i, j] > 0:
                overlap = masks[i] & masks[j]
                if areas[i] > areas[j]:
                    masks[i] = masks[i] & ~overlap
                else:
                    masks[j] = masks[j] & ~overlap
    return masks


def get_duplicate_ratio(masks: torch.Tensor):
    masks_float = masks.to(torch.float)
    intersection = torch.einsum('chw,dhw->cd', masks_float, masks_float)
    intersection = intersection.fill_diagonal_(0)
    masks_areas = masks_float.sum(dim=(1, 2)).unsqueeze(1)
    duplicate_ratio = intersection / masks_areas
    duplicate_ratio = duplicate_ratio.nan_to_num()
    return duplicate_ratio


def mask_to_rle(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle


def mask_to_polygons(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(contour, epsilon=1, closed=True) for contour in contours]
    polygons_as_list = [polygon.reshape(-1, 2).tolist() for polygon in polygons]
    return polygons_as_list
