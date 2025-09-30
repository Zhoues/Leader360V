import torch
import torch.nn.functional as F
from utils import multi_mask_iou_matrix, reshape_masks


@torch.no_grad()
def fuse_instances(prediction_result: list, pad_size: int = 256, overlap_size: int = 256,
                   merge_iou_threshold: float = 0.5, area_threshold: int = None):
    _, _, w_1 = prediction_result[0][1].shape
    _, _, w_2 = prediction_result[1][1].shape
    _, _, w_3 = prediction_result[2][1].shape
    original_width = w_1 + w_2 + w_3 - 2 * pad_size - 2 * overlap_size
    transformed_width = original_width + 2 * overlap_size
    fused_classes = prediction_result[0][0]
    fused_masks = prediction_result[0][1].to(torch.float)
    fused_masks = F.pad(fused_masks, (0, transformed_width - w_1))

    for i in range(len(prediction_result) - 1):
        curr_classes = fused_classes
        curr_masks = fused_masks
        next_classes = prediction_result[i + 1][0]
        next_masks = prediction_result[i + 1][1].to(torch.float)

        if i == 0:
            curr_overlap = curr_masks[:, :, w_1 - overlap_size: w_1]
        else:
            curr_overlap = curr_masks[:, :, w_1 + w_2 - 2 * overlap_size: w_1 + w_2 - overlap_size]
        next_overlap = next_masks[:, :, :overlap_size]

        iou = multi_mask_iou_matrix(curr_overlap, next_overlap)

        filter = iou > merge_iou_threshold
        matched_indices = torch.nonzero(filter, as_tuple=False)
        sorted_indices = torch.argsort(iou[filter], descending=True)
        matched_indices = matched_indices[sorted_indices]

        if i == 0:
            next_masks = F.pad(next_masks, (w_1 - pad_size, w_3 - pad_size))
        else:
            next_masks = F.pad(next_masks, (transformed_width - w_3, 0))

        match_curr_idx_list = []
        remove_idx_list = []
        for index in matched_indices:
            if index[0].item() in match_curr_idx_list or index[1].item() in remove_idx_list:
                continue
            if curr_classes[index[0]] == next_classes[index[1]]:
                curr_masks[index[0]] = torch.max(curr_masks[index[0]], next_masks[index[1]])
                remove_idx_list.append(index[1].item())
                match_curr_idx_list.append(index[0].item())

        unmatched_next = set(range(len(next_classes))) - set(remove_idx_list)
        fused_classes = torch.cat((curr_classes, next_classes[list(unmatched_next)]), dim=0)
        fused_masks = torch.cat((curr_masks, next_masks[list(unmatched_next)]), dim=0)

    first_overlap = fused_masks[:, :, :pad_size * 2]
    last_overlap = fused_masks[:, :, -pad_size * 2:]

    iou = multi_mask_iou_matrix(first_overlap, last_overlap)

    filter = iou > merge_iou_threshold
    matched_indices = torch.nonzero(filter, as_tuple=False)
    sorted_indices = torch.argsort(iou[filter], descending=True)
    matched_indices = matched_indices[sorted_indices]

    matched_index = []
    remove_index = []
    for index in matched_indices:
        if index[0].item() in matched_index or index[1].item() in remove_index:
            continue
        if fused_classes[index[0]] == fused_classes[index[1]] and index[0].item() != index[1].item():
            fused_masks[index[0]] = torch.max(fused_masks[index[0]], fused_masks[index[1]])
            remove_index.append(index[1].item())
            matched_index.append(index[0].item())
    indices_to_keep = set(range(len(fused_masks))) - set(remove_index)
    output_classes = fused_classes[list(indices_to_keep)]
    output_masks = fused_masks[:, :, pad_size: -pad_size].to(torch.bool)[
        list(indices_to_keep)]
    if area_threshold is None:
        return output_classes, output_masks
    else:
        output_masks = reshape_masks(output_masks)
        output_areas = output_masks.sum(dim=(-2, -1))
        filtered_areas = output_areas >= area_threshold
        return output_classes[filtered_areas], output_masks[filtered_areas]
