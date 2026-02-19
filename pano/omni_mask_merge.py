import torch
import torch.nn.functional as F
from pano.utils import multi_mask_iou_matrix, reshape_masks


@torch.no_grad()
def fuse_instances(prediction_result: list, pad_size: int = 256, overlap_size: int = 256,
                   merge_iou_threshold: float = 0.5, area_threshold: int = None):
    """
    pad_size：每个切片左右 padding 的宽度（后面会裁掉）。
    overlap_size：切片之间的重叠宽度。
    merge_iou_threshold：两个实例在重叠区域的 IoU 大于该阈值才认为是同一个实例，进行合并。
    area_threshold：如果不为 None，会按面积阈值过滤小实例。
    """
    _, _, w_1 = prediction_result[0][1].shape
    _, _, w_2 = prediction_result[1][1].shape
    _, _, w_3 = prediction_result[2][1].shape
    # 推回原图宽度。直觉是：三段宽度相加，但切片之间有重叠区域（重复算了两次），以及每段可能带 padding（也不是原图内容），因此要减掉相应部分
    original_width = w_1 + w_2 + w_3 - 2 * pad_size - 2 * overlap_size
    # 定义一个“中间融合宽度”（我称为 transformed）：比原图宽一些，多出两侧 overlap; 目的：在融合过程中保留一些边界重叠，最后再统一裁掉 pad。
    transformed_width = original_width + 2 * overlap_size

    # 初始融合类别 = 第 1 段的类别列表。
    fused_classes = prediction_result[0][0]
    # 初始融合 masks = 第 1 段的 masks，并转为 float。
    fused_masks = prediction_result[0][1].to(torch.float)
    # 在 mask 的宽度维右侧做 padding，使它的宽度从 w_1 变成 transformed_width。
    fused_masks = F.pad(fused_masks, (0, transformed_width - w_1))

    # 逐段把第 2 段、第 3 段并进来
    for i in range(len(prediction_result) - 1):
        # 当前融合类别 = 上一次融合的类别
        curr_classes = fused_classes
        # 当前融合 masks = 上一次融合的 masks
        curr_masks = fused_masks
        # 下一段的类别 = 下一段的类别列表
        next_classes = prediction_result[i + 1][0]
        # 下一段的 masks = 下一段的 masks，并转为 float。
        next_masks = prediction_result[i + 1][1].to(torch.float)


        # 取出当前段与下一段的“重叠区域”，计算 IoU 矩阵
        if i == 0:
            # 当前还是第 1 段为主，所以取它的末尾 overlap_size 宽。
            curr_overlap = curr_masks[:, :, w_1 - overlap_size: w_1]
        else:
            # 当前已经包含了前两段，因此用 w_1 + w_2 之类的偏移来取第二个连接处的重叠带。
            curr_overlap = curr_masks[:, :, w_1 + w_2 - 2 * overlap_size: w_1 + w_2 - overlap_size]
        # 从下一段 next_masks 取最左侧 overlap_size 宽，作为它与前一段的重叠带。
        next_overlap = next_masks[:, :, :overlap_size]
        # 计算 IoU 矩阵
        iou = multi_mask_iou_matrix(curr_overlap, next_overlap)

        # 找出 IoU 大于阈值的匹配对
        filter = iou > merge_iou_threshold
        
        # 把所有 True 的位置取出来，得到形如 (K,2) 的索引列表，每行 [curr_idx, next_idx]
        matched_indices = torch.nonzero(filter, as_tuple=False)
        # 把这些候选配对按它们对应的 IoU 值从大到小排序; 这样后面做贪心匹配时，会优先匹配 IoU 最大的组合
        sorted_indices = torch.argsort(iou[filter], descending=True)
        matched_indices = matched_indices[sorted_indices]

        # 把 next_masks pad 到与 curr_masks 同一“全局坐标系”上
        if i == 0:
            next_masks = F.pad(next_masks, (w_1 - pad_size, w_3 - pad_size))
        else:
            next_masks = F.pad(next_masks, (transformed_width - w_3, 0))


        # match_curr_idx_list：记录 curr 里已经被匹配过的实例 index（避免一个 curr 实例匹配多个 next）。
        # remove_idx_list：记录 next 里已经被合并掉的实例 index（后面不再追加到 fused）。
        match_curr_idx_list = []
        remove_idx_list = []
        for index in matched_indices:
            # 如果 curr 这个实例已经匹配过，或 next 这个实例已经被合并过，则跳过（保证一对一的贪心匹配）。
            if index[0].item() in match_curr_idx_list or index[1].item() in remove_idx_list:
                continue
            # 只有当两个实例类别相同，才允许合并（防止不同类别误合并）。
            # 注意： CropFormer 输出的是 class-agnostic mask，因此可以直接合并
            if curr_classes[index[0]] == next_classes[index[1]]:
                # 合并 mask：逐像素取最大值，相当于做 union（如果 mask 是 0/1，max 就是或操作）
                curr_masks[index[0]] = torch.max(curr_masks[index[0]], next_masks[index[1]])
                remove_idx_list.append(index[1].item())
                match_curr_idx_list.append(index[0].item())

        # 计算 next 里哪些实例没有被合并掉。
        unmatched_next = set(range(len(next_classes))) - set(remove_idx_list)
        # 把未匹配的 next 实例当作“新实例”追加到融合结果中； 类别向量按第 0 维拼接； mask 张量按第 0 维拼接（实例维）
        fused_classes = torch.cat((curr_classes, next_classes[list(unmatched_next)]), dim=0)
        fused_masks = torch.cat((curr_masks, next_masks[list(unmatched_next)]), dim=0)

    # 额外处理：首尾重叠（把“最左”和“最右”的重复实例再合一次）
    # 从融合后的 masks 中取出最左侧 2*pad_size 和最右侧 2*pad_size 两个区域。意图：如果图像在水平方向是“环形/拼接首尾相接”（比如全景），或者切片策略导致两端也有重复，需要再做一次合并。
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

    # 计算最终要保留的实例 index（从 fused_masks 中移除已经合并掉的实例）。
    indices_to_keep = set(range(len(fused_masks))) - set(remove_index)
    # 输出类别：只保留需要保留的实例
    output_classes = fused_classes[list(indices_to_keep)]
    # 输出 masks：裁掉两侧 padding，转为 bool 类型（CropFormer 输出是 float，需要转回 bool 方便后续计算），只保留需要保留的实例
    output_masks = fused_masks[:, :, pad_size: -pad_size].to(torch.bool)[
        list(indices_to_keep)]

    # 如果不需要按面积过滤，直接返回融合后的类别与 masks
    if area_threshold is None:
        return output_classes, output_masks
    else:
        # 按面积过滤：先 reshape 成标准 mask 格式（确保每个实例都是独立的），再计算面积，最后过滤掉小实例
        output_masks = reshape_masks(output_masks)
        output_areas = output_masks.sum(dim=(-2, -1))
        filtered_areas = output_areas >= area_threshold
        return output_classes[filtered_areas], output_masks[filtered_areas]
