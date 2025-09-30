import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, colors


def make_colors():
    all_colors = colors.XKCD_COLORS
    rgb_colors = {name: colors.to_rgb(hex_code) for name, hex_code in all_colors.items()}
    rgb_colors_int = [tuple(int(c * 255) for c in rgb) for _, rgb in rgb_colors.items()]
    return rgb_colors_int


def tuple_to_hex_color(rgb_tuple):
    return "#{:02X}{:02X}{:02X}".format(*rgb_tuple)


all_colors = make_colors()


def visualize_masks(masks: torch.Tensor | np.ndarray, save_path: str = None, mask_id_list=None):
    mask_num, m_H, m_W = masks.shape
    if type(masks) is torch.Tensor:
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    else:
        masks_np = masks.astype(np.uint8)
    masks_np = masks_np > 0
    if mask_id_list is None:
        areas = np.sum(masks_np, axis=(1, 2))
        masks_np = masks_np[np.argsort(-areas)]
        mask_id_list = np.arange(len(masks_np))
    color_mask = np.zeros((m_H, m_W, 3), dtype=np.uint8)
    for mask, color_id in zip(masks_np, mask_id_list):
        color_mask[mask] = all_colors[color_id % len(all_colors)]
    if save_path is not None:
        cv2.imwrite(save_path, color_mask)
    return color_mask


def visualize_masks_on_frame(masks: torch.Tensor | np.ndarray, frame: np.ndarray, save_path: str, mask_id_list=None):
    mask_num, m_H, m_W = masks.shape
    if type(masks) is torch.Tensor:
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    else:
        masks_np = masks.astype(np.uint8)
    masks_np = masks_np > 0
    if mask_id_list is None:
        areas = np.sum(masks_np, axis=(1, 2))
        masks_np = masks_np[np.argsort(-areas)]
        mask_id_list = np.arange(len(masks_np))
    colored_mask = np.zeros((m_H, m_W, 4), dtype=np.uint8)
    for mask, color_id in zip(masks_np, mask_id_list):
        color = all_colors[color_id % len(all_colors)]
        colored_mask[mask == 1] = np.concatenate([color, [0.4 * 255]])
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    ax.imshow(colored_mask)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_classes_and_masks(masks: torch.Tensor | np.ndarray, class_list: list, save_path: str, mask_id_list=None):
    if type(masks) is torch.Tensor:
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    else:
        masks_np = masks.astype(np.uint8)
    if mask_id_list is None:
        areas = np.sum(masks_np, axis=(1, 2))
        sorted_id = np.argsort(-areas)
        masks_np = masks_np[sorted_id]
        mask_id_list = np.arange(len(masks_np))
        class_list = [class_list[i] for i in sorted_id]
    _, ax = plt.subplots(figsize=(10, 10))
    colored_mask = np.zeros((masks_np.shape[1], masks_np.shape[2], 3), dtype=np.uint8)
    for mask, color_id, class_name in zip(masks_np, mask_id_list, class_list):
        colored_mask[mask == 1] = all_colors[color_id % len(all_colors)]
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if stats[1:, -1].size == 0:
            continue
        largest_component_id = np.argmax(stats[1:, -1]) + 1
        for cid in range(1, _num_cc):
            if cid == largest_component_id:  # or stats[cid, -1] > 500:
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                ax.text(
                    center[0],
                    center[1],
                    class_name,
                    size=10,
                    family="sans-serif",
                    bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="white",
                    zorder=10,
                    rotation=0,
                )
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_classes_and_masks_on_frame(masks: torch.Tensor | np.ndarray, image: np.ndarray, class_list: list,
                                         save_path: str,
                                         mask_id_list=None):
    if type(masks) is torch.Tensor:
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    else:
        masks_np = masks.astype(np.uint8)
    if mask_id_list is None:
        areas = np.sum(masks_np, axis=(1, 2))
        sorted_id = np.argsort(-areas)
        masks_np = masks_np[sorted_id]
        mask_id_list = np.arange(len(masks_np))
        class_list = [class_list[i] for i in sorted_id]
    _, ax = plt.subplots(figsize=(10, 10))
    colored_mask = np.zeros((image.shape[0], image.shape[1], 4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for mask, color_id, class_name in zip(masks_np, mask_id_list, class_list):
        color = all_colors[color_id % len(all_colors)]
        colored_mask[mask == 1] = np.concatenate([[color_ / 255 for color_ in color], [0.4]])
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if stats[1:, -1].size == 0:
            continue
        largest_component_id = np.argmax(stats[1:, -1]) + 1
        for cid in range(1, _num_cc):
            if cid == largest_component_id:  # or stats[cid, -1] > 500:
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                ax.text(
                    center[0],
                    center[1],
                    class_name,
                    size=10,
                    family="sans-serif",
                    bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="white",
                    zorder=10,
                    rotation=0,
                )
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_mask_on_image(mask: torch.Tensor | np.ndarray, image: np.ndarray, save_path: str):
    if type(mask) == torch.Tensor:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    colored_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    color = all_colors[1]
    colored_mask[mask_np] = np.concatenate([color, [255]])
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    ax.imshow(colored_mask, cmap='jet', alpha=1)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
