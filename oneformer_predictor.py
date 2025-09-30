import imutils
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
import sys
import torch.nn.functional as F

sys.path.insert(0, "OneFormer")
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from demo.defaults import DefaultPredictor
from cityscapesscripts.helpers.labels import labels


class OneFormerConfig:
    def __init__(self):
        self.MODEL_DICT = {
            "SWIN":
                {
                    "cityscapes": {
                        "config": "OneFormer/configs/cityscapes/swin/oneformer_swin_large_bs16_90k.yaml",
                        "checkpoint": "pretrained_models/OneFormer/Cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth"
                    },
                    "coco": {
                        "config": "OneFormer/configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml",
                        "checkpoint": "pretrained_models/OneFormer/COCO/150_16_swin_l_oneformer_coco_100ep.pth"
                    },
                    "ade20k": {
                        "config": "OneFormer/configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml",
                        "checkpoint": "pretrained_models/OneFormer/ADE20K/250_16_swin_l_oneformer_ade20k_160k.pth"
                    }
                },
            "DINAT":
                {
                    "cityscapes": {
                        "config":  "OneFormer/configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml",
                        "checkpoint": "pretrained_models/OneFormer/Cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth"
                    },
                    "coco": {
                        "config": "OneFormer/configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
                        "checkpoint": "pretrained_models/OneFormer/COCO/150_16_dinat_l_oneformer_coco_100ep.pth"
                    },
                    "ade20k": {
                        "config": "OneFormer/configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml",
                        "checkpoint": "pretrained_models/OneFormer/ADE20K/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth"
                    }
                },
            "ConvNeXt": {
                "cityscapes": {
                    "config": "OneFormer/configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_xlarge_bs16_90k.yaml",
                    "checkpoint": "pretrained_models/OneFormer/Cityscapes/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth"
                },
                "ade20k": {
                    "config": "OneFormer/configs/ade20k/convnext/oneformer_convnext_xlarge_bs16_160k.yaml",
                    "checkpoint": "pretrained_models/OneFormer/ADE20K/250_16_convnext_xl_oneformer_ade20k_160k.pth"
                }
            }
        }

    def setup_cfg(self, dataset: str, model_type: str):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        if model_type == "ConvNeXt" and dataset == "coco":
            cfg_path = self.MODEL_DICT["SWIN"][dataset]["config"]
        else:
            cfg_path = self.MODEL_DICT[model_type][dataset]["config"]
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "ConvNeXt" and dataset == "coco":
            cfg.MODEL.WEIGHTS = self.MODEL_DICT["SWIN"][dataset]["checkpoint"]
        else:
            cfg.MODEL.WEIGHTS = self.MODEL_DICT[model_type][dataset]["checkpoint"]
        cfg.freeze()
        return cfg

    def setup_modules(self, dataset: str, model_type: str):
        cfg = self.setup_cfg(dataset, model_type)
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
            stuff_colors = [k.color for k in labels if k.trainId != 255]
            metadata = metadata.set(stuff_colors=stuff_colors)
        return predictor, metadata

    def get_datasets_semantic_classes(self):
        ade20k_config = self.setup_cfg("ade20k", "SWIN")
        coco_config = self.setup_cfg("coco", "SWIN")
        cityscapes_config = self.setup_cfg("cityscapes", "SWIN")
        ade20k_metadata = MetadataCatalog.get(
            ade20k_config.DATASETS.TEST_PANOPTIC[0] if len(ade20k_config.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        coco_metadata = MetadataCatalog.get(
            coco_config.DATASETS.TEST_PANOPTIC[0] if len(coco_config.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        cityscapes_metadata = MetadataCatalog.get(
            cityscapes_config.DATASETS.TEST_PANOPTIC[0] if len(cityscapes_config.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        ade20k_classes = ade20k_metadata.stuff_classes
        coco_classes = coco_metadata.stuff_classes
        cityscapes_classes = cityscapes_metadata.stuff_classes
        return ade20k_classes, coco_classes, cityscapes_classes


oneformer_config = OneFormerConfig()


class OneFormerPredictor:
    def __init__(self, dataset_name: str = "cityscapes", model_type: str = "ConvNeXt",
                 device: str = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.predictor, self.metadata = oneformer_config.setup_modules(dataset_name, model_type)

    @torch.no_grad()
    def __call__(self, image: np.ndarray):
        H, W, _ = image.shape
        image = imutils.resize(image, width=640)
        predictions = self.predictor(image, "panoptic")
        masks, classes = predictions["panoptic_seg"]
        if len(classes) != 0:
            classes = torch.as_tensor([class_['category_id'] for class_ in classes]).to(self.device)
            masks = torch.stack([masks == class_ + 1 for class_ in range(len(classes))]).float()
        else:
            classes = None
            masks = (masks == 0).unsqueeze(0).float()
        masks = F.interpolate(masks.unsqueeze(0), (H, W), mode="nearest")[0]
        return classes, masks

    def get_semantic_information(self, classes: torch.Tensor):
        semantic_classes = []
        for label in classes:
            text = self.metadata.stuff_classes[label]
            semantic_classes.append(text)
        return semantic_classes