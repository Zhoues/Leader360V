import json
import time
import torch
import yaml
import os
import shutil
import cv2
import numpy as np
from pano.frame_predictor import FramePredictor
from pano.utils import mask_to_polygons, mask_to_rle, rle_to_mask
from pano.visualize import visualize_masks, visualize_classes_and_masks
from pano.omni_tools import OmniImage


class PanoramaAnnotator:
    def __init__(self, config_file: str):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # 设置 OpenAI API 密钥和基础 URL
        os.environ["OPENAI_API_KEY"] = config["API"]["openai_api_key"]    
        os.environ["OPENAI_BASE_URL"] = config["API"]["openai_base_url"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_predictor = FramePredictor(
            device=self.device, **config["FramePredictor"], other_config=config
        )

    def annotate_one_frame(
        self,
        frame_path: str,
        work_dir_path: str = "work-dir",
        visualize: bool = False,
        use_llm: bool = True,
    ):
        os.makedirs(work_dir_path, exist_ok=True)
        frame_name = os.path.basename(frame_path)
        frame_name = frame_name[: frame_name.rindex(".")]
        work_dir_path = os.path.join(work_dir_path, frame_name)
        if os.path.exists(work_dir_path):
            shutil.rmtree(work_dir_path)
        os.makedirs(work_dir_path, exist_ok=True)

        print("=" * 20 + f"Start Annotating {frame_name}" + "=" * 20)
        start_time = time.time()

        # 加载图像
        image = cv2.imread(frame_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {frame_path}")

        # 如果最长边大于 3840，则缩放到 3840 x 1920，长宽满足 2:1 比例
        if max(image.shape[:2]) > 3840:
            # 目标最大边为 3840，且 2:1
            new_width, new_height = 3840, 1920
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Image resized to {image.shape[1]}x{image.shape[0]}")

        # 分割并得到带语义的 mask
        predicted_masks, semantic_label_list = self.frame_predictor.predict(
            image, use_llm=use_llm
        )
        if predicted_masks is None or len(predicted_masks) == 0:
            print(f"No masks predicted for {frame_name}")
            return None, None

        # 转为 numpy 便于保存与可视化（保持 [N, H, W]）
        if isinstance(predicted_masks, torch.Tensor):
            masks_np = predicted_masks.detach().cpu().numpy().astype(np.uint8)
        else:
            masks_np = np.asarray(predicted_masks, dtype=np.uint8)
        masks_np = (masks_np > 0).astype(np.uint8)

        segmentation_infor = {"version": "5.6.0", "flags": {}, "shapes": [],
                            "imagePath": f"{frame_path}",
                            "imageData": None,
                            "imageHeight": image.shape[0],
                            "imageWidth": image.shape[1]
                    }

        # 保存标注结果：每张 mask 对应一个 label，并保存多边形（可选）
        for i, (mask, semantic_label) in enumerate(zip(masks_np, semantic_label_list)):
            rle = mask_to_rle(mask)
            segmentation_infor["shapes"].append({
                "label": semantic_label,
                "group_id": i,
                "description": "",
                "shape_type": "rle",
                "flags": {},
                "mask": rle
            })
        ann_path = os.path.join(work_dir_path, "annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(segmentation_infor, f, ensure_ascii=False, indent=2)
        print(f"Annotations saved to {ann_path}")


        if visualize:

            # 测试保存的标注结果是否正确
            with open(ann_path, "r", encoding="utf-8") as f:
                segmentation_infor = json.load(f)
            
            # 从保存的标注中恢复 masks_np 和 semantic_label_list，用于验证保存/读取是否正确
            masks_list = []
            semantic_label_list = []
            for shape in segmentation_infor["shapes"]:
                semantic_label_list.append(shape["label"])
                mask = rle_to_mask(shape["mask"]).astype(np.uint8)
                masks_list.append(mask)
            masks_np = np.stack(masks_list, axis=0)

            # 1) 纯 mask 可视化：只画彩色 mask，不显示原图、不显示类别
            vis_mask_path = os.path.join(work_dir_path, "vis_mask_only.png")
            visualize_masks(masks_np, save_path=vis_mask_path)
            print(f"Pure mask visualization saved to {vis_mask_path}")

            # 2) mask + 类别可视化：彩色 mask + 每个区域中心的 class 文字
            vis_class_path = os.path.join(work_dir_path, "vis_mask_and_class.png")
            visualize_classes_and_masks(
                masks_np, semantic_label_list, save_path=vis_class_path
            )
            print(f"Mask + class visualization saved to {vis_class_path}")

            # 3) mask + 非unknown类别可视化：彩色 mask + 每个区域中心的 class 文字
            vis_non_unknown_class_path = os.path.join(work_dir_path, "vis_mask_and_non_unknown_class.png")
            non_unknown_indices = [i for i, label in enumerate(semantic_label_list) if label != "unknown"]
            non_unknown_semantic_label_list = [semantic_label_list[i] for i in non_unknown_indices]
            non_unknown_masks_np = masks_np[non_unknown_indices]
            visualize_classes_and_masks(
                non_unknown_masks_np, non_unknown_semantic_label_list, save_path=vis_non_unknown_class_path
            )
            print(f"Mask + non-unknown class visualization saved to {vis_non_unknown_class_path}")

            # Debug: 保存 get_front_view_crop_centered_on_mask 的返回结果（外接 + 内接 bbox/crop）
            debug_crop_dir = os.path.join(work_dir_path, "debug_front_view_crops")
            if os.path.exists(debug_crop_dir):
                shutil.rmtree(debug_crop_dir)
            os.makedirs(debug_crop_dir, exist_ok=True)
            omni = OmniImage(img_w=image.shape[1], img_h=image.shape[0])
            for mask_id, (mask_np, label) in enumerate(zip(masks_np, semantic_label_list)):
                result = omni.get_front_view_crop_centered_on_mask(
                    image, mask_np, view_width=640, view_height=480, fov_h=100, expand=50
                )
                if result is None:
                    continue
                safe_label = label.replace(" ", "_").replace("/", "_")[:50]
                prefix = f"{mask_id}_{int(mask_np.sum())}_{safe_label}"
                cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_front_view.png"), result["front_view"])
                # mask 保存为 RGB 可视化：0=黑，255=白
                mask_01 = (result["mask_view"] > 0).astype(np.uint8)
                mask_rgb = (mask_01 * 255)[:, :, np.newaxis].repeat(3, axis=2)
                cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_mask_view.png"), mask_rgb)
                # 外接 bbox
                x1, y1, w, h = result["bbox_2d"]
                bbox_view = result["front_view"].copy()
                cv2.rectangle(bbox_view, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_bbox_2d.png"), bbox_view)
                cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_crop_image.png"), result["crop_image"])
                cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_object_crop_image.png"), result["object_crop_image"])
                # 内接 bbox 及对应 crop
                if result.get("bbox_2d_inner") is not None:
                    ix1, iy1, iw, ih = result["bbox_2d_inner"]
                    bbox_view_inner = result["front_view"].copy()
                    cv2.rectangle(bbox_view_inner, (ix1, iy1), (ix1 + iw, iy1 + ih), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_bbox_2d_inner.png"), bbox_view_inner)
                    cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_crop_image_inner.png"), result["crop_image_inner"])
                    cv2.imwrite(os.path.join(debug_crop_dir, f"{prefix}_object_crop_image_inner.png"), result["object_crop_image_inner"])
            print(f"Debug front-view crops saved to {debug_crop_dir}")

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f}s, {len(masks_np)} instances.")
        return masks_np, semantic_label_list


if __name__ == "__main__":
    config_file = "config.yaml"
    work_dir_path = "first_frame_annotate_result_dir"
    # frame_path_1 = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360VOTS/train/003/image/000000.jpg"
    # frame_path_2 = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360VOTS/train/004/image/000001.jpg"
    # frame_path_1 = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hstar_bench_png/hos_bench/1/001.png"
    # frame_path_2 = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hstar_bench_png/hps_bench/1/002.png"
    frame_path_1 = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/Structured3D/Structured3D/scene_00000/2D_rendering/485142/panorama/full/rgb_coldlight.png"
    panorama_annotator = PanoramaAnnotator(config_file)
    panorama_annotator.annotate_one_frame(
        frame_path=frame_path_1,
        work_dir_path=work_dir_path,
        visualize=True,
        use_llm=True,
    )
    # panorama_annotator.annotate_one_frame(
    #     frame_path=frame_path_2,
    #     work_dir_path=work_dir_path,
    #     visualize=True,
    #     use_llm=True,
    # )
