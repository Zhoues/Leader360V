"""
测试：根据 delta pitch、delta yaw 以及指定的 HFOV 和 VFOV 从全景图中截取透视 RGB 图像并保存。
适用于需要精确匹配相机参数（如 Intel RealSense D435）的场景。

D435 RGB 相机默认规格：HFOV≈69.4°, VFOV≈42.5°, DFOV≈77° (±3°)
"""
import cv2
import numpy as np
from pano.omni_tools import OmniImage, Bfov


# Intel RealSense D435 RGB 相机 FOV 默认值（度）
D435_HFOV = 69.4
D435_VFOV = 42.5


def get_image_from_pano_with_hfov_vfov(
    pano_image,
    delta_yaw,
    delta_pitch,
    out_width=1920,
    out_height=1080,
    hfov=None,
    vfov=None,
    save_path=None,
    interpolation=cv2.INTER_LINEAR,
):
    """
    根据 delta pitch、delta yaw 以及 HFOV、VFOV 从全景图（equirectangular）中截取对应的透视 RGB 图像并可选保存。

    Args:
        pano_image: 全景图，可为 numpy array (H, W, C) 或图像路径字符串
        delta_yaw: 中心方向的水平偏移角度（度），正为向右
        delta_pitch: 中心方向的俯仰偏移角度（度），正为向上
        out_width: 输出图像宽度，默认 640
        out_height: 输出图像高度，默认 480
        hfov: 水平视场角 HFOV（度），与 vfov 二选一必填，或两者都填
        vfov: 垂直视场角 VFOV（度），与 hfov 二选一必填，或两者都填
             若仅填一个，则另一个按宽高比自动计算；两者都填时完全独立指定
        save_path: 若提供则保存图像到此路径（BGR，与 OpenCV 一致）
        interpolation: cv2 插值方式，默认 INTER_LINEAR

    Returns:
        out: 透视图像 (out_height, out_width, 3)，numpy uint8，BGR
    """
    if hfov is None and vfov is None:
        raise ValueError("hfov 和 vfov 至少需指定一个")
    if hfov is None:
        # 仅有 vfov 时，按宽高比反推 hfov
        hfov = vfov * (out_width / out_height)
    if vfov is None:
        # 仅有 hfov 时，按宽高比计算 vfov（与原 0_test_get_image_from_pano_ori 一致）
        vfov = hfov * (out_height / out_width)

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
        fov_h=hfov,
        fov_v=vfov,
        interpolation=interpolation,
    )

    if save_path:
        cv2.imwrite(save_path, out)
    return out


# ============= 示例用法 =============
pano_image = "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360VOTS/train/003/image/000000.jpg"

# 视角：delta_yaw=0, delta_pitch=0 表示正中心；单位：度
delta_yaw = 0.0
delta_pitch = 0.0

# 方式1：指定 HFOV 和 VFOV（D435 风格）
hfov = D435_HFOV   # 69.4 度
vfov = D435_VFOV   # 42.5 度

out_width = 1920
out_height = 1080
save_path = "/share/project/zhouenshen/hpfs/code/ActivePerception/Leader360V/data/out_perspective_d435_hfov_vfov.jpg"

img = get_image_from_pano_with_hfov_vfov(
    pano_image,
    delta_yaw=delta_yaw,
    delta_pitch=delta_pitch,
    out_width=out_width,
    out_height=out_height,
    hfov=hfov,
    vfov=vfov,
    save_path=save_path,
)
print(f"已保存 D435 风格透视图: {save_path}, shape={img.shape}")

# 方式2：仅指定 HFOV，VFOV 按宽高比自动计算（与原脚本行为一致）
img2 = get_image_from_pano_with_hfov_vfov(
    pano_image,
    delta_yaw=30.0,
    delta_pitch=10.0,
    out_width=1920,
    out_height=1080,
    hfov=100,  # 仅指定 HFOV
    save_path="/share/project/zhouenshen/hpfs/code/ActivePerception/Leader360V/data/out_perspective_hfov100.jpg",
)
print(f"已保存仅 HFOV=100 的透视图: shape={img2.shape}")

# 方式3：仅指定 VFOV，HFOV 按宽高比自动计算
img3 = get_image_from_pano_with_hfov_vfov(
    pano_image,
    delta_yaw=0,
    delta_pitch=0,
    out_width=1920,
    out_height=1080,
    vfov=50,  # 仅指定 VFOV
    save_path="/share/project/zhouenshen/hpfs/code/ActivePerception/Leader360V/data/out_perspective_vfov50.jpg",
)
print(f"已保存仅 VFOV=50 的透视图: shape={img3.shape}")
