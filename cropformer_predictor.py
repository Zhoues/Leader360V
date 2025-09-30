import copy
import cv2
import numpy as np
import torch
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import sys

sys.path.insert(0, "detectron2/projects/CropFormer")
from mask2former.data.dataset_mappers.crop_augmentations import BatchResizeShortestEdge, EntityCrop, EntityCropTransform
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

crop_former_file_dict = {
    "SWIN-TINY": {
        "config_file": "detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_swin_tiny_3x.yaml",
        "weight_file": "pretrained_models/CropFormer/CropFormer_swin_tiny_3x_5cea5e.pth"
    },
    "SWIN-LARGE": {
        "config_file": "detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml",
        "weight_file": "pretrained_models/CropFormer/CropFormer_swin_large_w7_3x_6843ef.pth"
    },
    "HORNET-LARGE": {
        "config_file": "detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        "weight_file": "pretrained_models/CropFormer/CropFormer_hornet_3x_03823a.pth"
    }
}


def setup_coprformer_cfg(config_file, weight_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = weight_file
    cfg.freeze()
    return cfg


def make_colors():
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors


cropformer_colors = make_colors()


class CropFormerPredictor(DefaultPredictor):
    def __init__(self, model_type: str = "HORNET-LARGE", confidence_threshold: float = 0.3):
        cfg = setup_coprformer_cfg(crop_former_file_dict[model_type]["config_file"],
                                   crop_former_file_dict[model_type]["weight_file"])
        super().__init__(cfg)
        self.confidence_threshold = confidence_threshold

    def generate_img_augs(self):
        shortest_side = np.random.choice([self.cfg.INPUT.MIN_SIZE_TEST])

        augs = [
            T.ResizeShortestEdge(
                (shortest_side,),
                self.cfg.INPUT.MAX_SIZE_TEST,
                self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),

        ]

        # Build original image augmentation
        crop_augs = []
        entity_crops = EntityCrop(self.cfg.ENTITY.CROP_AREA_RATIO,
                                  self.cfg.ENTITY.CROP_STRIDE_RATIO,
                                  self.cfg.ENTITY.CROP_SAMPLE_NUM_TEST,
                                  False)
        crop_augs.append(entity_crops)

        entity_resize = BatchResizeShortestEdge((shortest_side,), self.cfg.INPUT.MAX_SIZE_TEST,
                                                self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
        crop_augs.append(entity_resize)

        # augs      = T.AugmentationList(augs)
        crop_augs = T.AugmentationList(crop_augs)
        return augs, crop_augs

    @torch.no_grad()
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]

        # build cropformer augmentations
        augs, crop_augs = self.generate_img_augs()

        height, width = original_image.shape[:2]
        aug_input_ori = T.AugInput(copy.deepcopy(original_image))

        aug_input_ori, _ = T.apply_transform_gens(augs, aug_input_ori)
        image_ori = aug_input_ori.image
        image_ori = torch.as_tensor(image_ori.astype("float32").transpose(2, 0, 1))

        aug_input_crop = T.AugInput(copy.deepcopy(original_image))
        transforms_crop = crop_augs(aug_input_crop)
        image_crop = aug_input_crop.image
        assert len(image_crop.shape) == 4, "the image shape must be [N, H, W, C]"
        image_crop = torch.as_tensor(image_crop.astype("float32").transpose(0, 3, 1, 2))

        for transform_type in transforms_crop:
            if isinstance(transform_type, EntityCropTransform):
                crop_axises = transform_type.crop_axises
                crop_indexes = transform_type.crop_indexes

        inputs = {"image": image_ori,
                  "height": height,
                  "width": width,
                  "image_crop": image_crop,
                  "crop_region": crop_axises,
                  "crop_indexes": crop_indexes
                  }
        # pdb.set_trace()
        predictions = self.model([inputs])[0]
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores
        # select by confidence threshold
        selected_indexes = (pred_scores >= self.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks = pred_masks[selected_indexes]
        return selected_scores, selected_masks


def visualize_CropFormer_masks(masks, scores, save_path):
    _, m_H, m_W = masks.shape
    mask_id = np.zeros((m_H, m_W), dtype=np.uint8)
    selected_scores, ranks = torch.sort(scores)
    ranks = ranks + 1
    for index in ranks:
        mask_id[(masks[index - 1] == 1).cpu().numpy()] = int(index)
    unique_mask_id = np.unique(mask_id)
    color_mask = np.zeros((m_H, m_W, 3), dtype=np.uint8)
    for count in unique_mask_id:
        if count == 0:
            continue
        color_mask[mask_id == count] = cropformer_colors[count]
    cv2.imwrite(save_path, color_mask)
    return color_mask
