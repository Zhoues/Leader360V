"""
测试：根据 delta pitch、delta yaw 和 FOV 从全景图中截取 640x480 的透视 RGB 图像并保存。
"""
import cv2
import numpy as np
from pano.omni_tools import OmniImage

def get_image_from_pano(
    pano_image,
    delta_yaw,
    delta_pitch,
    out_width=640,
    out_height=480,
    fov=60,
    save_path=None,
):
    """
    根据 delta pitch、delta yaw 和 FOV 从全景图（equirectangular）中截取对应的透视 RGB 二维图像并可选保存。

    Args:
        pano_image: 全景图，可为 numpy array (H, W, C) 或图像路径字符串
        delta_yaw: 中心方向的水平偏移角度（度），正为向右
        delta_pitch: 中心方向的俯仰偏移角度（度），正为向上
        out_width: 输出图像宽度，默认 640
        out_height: 输出图像高度，默认 480
        fov: 水平视场角（度），默认 60；垂直 FOV 按宽高比自动计算
        save_path: 若提供则保存图像到此路径（BGR，与 OpenCV 一致）

    Returns:
        out: 透视图像 (out_height, out_width, 3)，numpy uint8，BGR
    """
    if isinstance(pano_image, str):
        img = cv2.imread(pano_image)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {pano_image}")
    else:
        img = np.asarray(pano_image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3 and getattr(img, "dtype", None) != np.uint8:
            img = img.astype(np.uint8)
    h, w = img.shape[:2]
    omni = OmniImage(img_w=w, img_h=h)
    out = omni.get_front_view_crop_from_pano(
        img,
        delta_yaw=delta_yaw,
        delta_pitch=delta_pitch,
        out_width=out_width,
        out_height=out_height,
        fov=fov,
    )
    if save_path:
        cv2.imwrite(save_path, out)
    return out

# 全景图路径
pano_image = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360VOTS/train/003/image/000000.jpg"

# 视角：delta_yaw=0, delta_pitch=0 表示正中心；单位：度
delta_yaw = 0.0
delta_pitch = 0.0
# 水平 FOV（度），默认 60；输出尺寸 640x480
fov = 100
out_width = 640
out_height = 480
save_path = "/share/project/zhouenshen/hpfs/code/ActivePerception/Leader360V/data/out_perspective_640x480.jpg"

# 截取并保存
img = get_image_from_pano(
    pano_image,
    delta_yaw=delta_yaw,
    delta_pitch=delta_pitch,
    out_width=out_width,
    out_height=out_height,
    fov=fov,
    save_path=save_path,
)
print(f"已保存透视图: {save_path}, shape={img.shape}")

# 可选：测试不同视角（例如向右 30 度、向上 10 度）
img2 = get_image_from_pano(
    pano_image,
    delta_yaw=30.0,
    delta_pitch=10.0,
    out_width=640,
    out_height=480,
    fov=100,
    save_path="/share/project/zhouenshen/hpfs/code/ActivePerception/Leader360V/data/out_perspective_yaw30_pitch10.jpg",
)
print(f"已保存另一视角: shape={img2.shape}")
