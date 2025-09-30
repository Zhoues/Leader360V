import json
import math
import time
import torch
from sam2_predictor import SAM2Predictor
import yaml
import os
import shutil
import cv2
from utils import mask_to_polygons
from frame_tools import VideoStreamReader


class VideoSegmentor:
    def __init__(self, config_file: str):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_predictor = SAM2Predictor(**config["SAM2Predictor"],
                                            device=self.device,
                                            other_config=config)

    def segment_video(self, video_path: str, work_dir_path: str = "work-dir", visualize: bool = False):
        os.makedirs(work_dir_path, exist_ok=True)
        video_name = os.path.basename(video_path)
        video_name = video_name[:video_name.rindex(".")]
        work_dir_path = os.path.join(work_dir_path, video_name)
        if os.path.exists(work_dir_path):
            shutil.rmtree(work_dir_path)
        os.makedirs(work_dir_path, exist_ok=True)
        print('=' * 20 + f"Start Segmenting {video_name}" + '=' * 20)
        start_time = time.time()
        self.sam2_predictor.predict(video_path)
        end_time = time.time()
        frame_idx_list = sorted(list(self.sam2_predictor.video_segments.keys()))
        video_reader = VideoStreamReader(video_path)
        for frame_idx in frame_idx_list:
            second_index = str(math.ceil((frame_idx + 1) / video_reader.fps)).zfill(4)
            frame_dir = os.path.join(work_dir_path, f"{second_index}")
            os.makedirs(frame_dir, exist_ok=True)
            frame = video_reader.read_frame(frame_idx)
            cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_idx + 1}.png"), frame)
            masks = self.sam2_predictor.video_segments[frame_idx]["masks"].detach().cpu().numpy()
            obj_ids = self.sam2_predictor.video_segments[frame_idx]["object_ids"]
            segmentation_infor = {"version": "5.6.0", "flags": {}, "shapes": [],
                                  "imagePath": f"frame_{frame_idx + 1}.png",
                                  "imageData": None,
                                  "imageHeight": frame.shape[0],
                                  "imageWidth": frame.shape[1]
                                  }
            for mask, obj_id in zip(masks, obj_ids):
                semantic_label = self.sam2_predictor.semantic_labels[obj_id]
                polygons = mask_to_polygons(mask)
                for polygon in polygons:
                    segmentation_infor["shapes"].append({
                        "label": semantic_label,
                        "points": polygon,
                        "group_id": obj_id,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None
                    })
            json_path = os.path.join(frame_dir, f"frame_{frame_idx + 1}.json")
            with open(json_path, "w") as file:
                json.dump(segmentation_infor, file)
        video_reader.close()
        if visualize:
            video_path = os.path.join(work_dir_path, "visualization.mp4")
            self.sam2_predictor.visualize_result(video_path)
        infor_path = os.path.join(work_dir_path, "information.txt")
        with open(infor_path, "w") as file:
            video_duration_infor = f"Video Duration: {self.sam2_predictor.video_reader.duration}\n"
            file.write(video_duration_infor)
            segment_duration_infor = f"Annotation Duration: {end_time - start_time}\n"
            file.write(segment_duration_infor)
            object_num_infor = f"Object Num: {len(self.sam2_predictor.semantic_labels)}\n"
            file.write(object_num_infor)
            object_list_infor = f"Object List: {self.sam2_predictor.semantic_labels}\n"
            file.write(object_list_infor)
            fps_infor = f"FPS: {video_reader.fps}\n"
            file.write(fps_infor)
        print('=' * 20 + f"End Segmenting {video_name}" + '=' * 20)


if __name__ == "__main__":
    config_file = "config.yaml"
    work_dir_path = "test_video_result_dir"
    video_segmentor = VideoSegmentor(config_file)
    video_segmentor.segment_video("test_video.mp4", work_dir_path=work_dir_path, visualize=True)
