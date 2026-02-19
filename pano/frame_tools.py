import cv2
import numpy as np
import math


class VideoStreamReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = math.ceil(self.cap.get(cv2.CAP_PROP_FPS))
        self.finished = False
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frame / self.fps

    def read_frame(self, frame_idx: int = 0):
        if self.finished:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        while not ret:
            frame_idx -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
        return frame

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.finished = True


def left_right_frame_padding(frame, pad_ratio=0.125):
    """
    把左边1/5加到右边，右边1/5加到左边。
    """
    h, w, _ = frame.shape
    pad_size = int(w * pad_ratio)
    left = frame[:, :pad_size]
    right = frame[:, -pad_size:]
    new_frame = np.hstack((right, frame, left))
    return new_frame, pad_size


def split_frame_with_overlap(frame, original_width, pad_size, overlap_ratio=0.125):
    """
    将frame水平分成三个有重叠区域的子图。
    """
    overlap_size = int(original_width * overlap_ratio)
    stride = (original_width + pad_size * 2 + overlap_size * 2) // 3
    parts = (
        frame[:, 0: stride],
        frame[:, stride - overlap_size: 2 * stride - overlap_size],
        frame[:, 2 * stride - 2 * overlap_size:]
    )
    return parts, overlap_size


def save_frame(save_path, frame):
    cv2.imwrite(save_path, frame)
