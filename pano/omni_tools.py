# From github: https://github.com/HuajianUP/360VOT
import cv2
import numpy as np
import shapely.geometry as sgeo
import torch


class Bfov():
    def __init__(self, lon, lat, fov_h, fov_v, rotation=0):
        # center position : lon, lat
        # horizontal and vertical fov : fov_h, fov_v
        # rotation： positive -> anticlockwise； negative -> clockwise
        # in angle
        self.clon = lon
        self.clat = lat
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.rotation = rotation

    def iou(self, target_bfov):
        pass

    def todict(self):
        return {"clon": self.clon, "clat": self.clat,
                "fov_h": self.fov_h, "fov_v": self.fov_v,
                "rotation": self.rotation}

    def tolist(self):
        return (self.clon, self.clat, self.fov_h, self.fov_v, self.rotation)


class Bbox():
    def __init__(self, cx, cy, w, h, rotation=0):
        # center position : cx, cy
        # bbox width and heigh: w, h
        # rotation： positive-> clockwise； negative -> anticlockwise
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.rotation = rotation

        istensor = isinstance(cx, torch.Tensor) or isinstance(cy, torch.Tensor) or \
                   isinstance(w, torch.Tensor) or isinstance(h, torch.Tensor) or \
                   isinstance(rotation, torch.Tensor)

        if not istensor: self._init_corner()

    def _init_corner(self):
        rotation = ang2rad(self.rotation)
        ## rotation matrix R
        # cos -sin
        # sin cos
        # opencv is based on the bottom-left [-w/2, h/2]
        w_2 = self.w / 2
        h_2 = self.h / 2
        w_2cos = w_2 * np.cos(rotation)
        w_2sin = w_2 * np.sin(rotation)
        h_2cos = h_2 * np.cos(rotation)
        h_2sin = h_2 * np.sin(rotation)

        self.bottomLeft = [self.cx - w_2cos - h_2sin, self.cy - w_2sin + h_2cos]  # R@[-w/2, h/2].t()
        self.topLeft = [self.cx - w_2cos + h_2sin, self.cy - h_2cos - w_2sin]  # R@[-w/2, -h/2].t()
        self.topRight = [2 * self.cx - self.bottomLeft[0], 2 * self.cy - self.bottomLeft[1]]
        self.bottomRight = [2 * self.cx - self.topLeft[0], 2 * self.cy - self.topLeft[1]]

    def iou(self, target_bbox):
        a = sgeo.Polygon([self.topLeft, self.topRight, self.bottomRight, self.bottomLeft])
        b = sgeo.Polygon([target_bbox.topLeft, target_bbox.topRight, target_bbox.bottomRight, target_bbox.bottomLeft])
        iou = a.intersection(b).area / a.union(b).area
        return iou

    def todict(self):
        return {"cx": self.cx, "cy": self.cy,
                "w": self.w, "h": self.h,
                "rotation": self.rotation}

    def tolist_xywh(self):
        return (self.topLeft[0], self.topLeft[1], self.w, self.h)

    def tolist(self):
        return (self.cx, self.cy, self.w, self.h, self.rotation)


def convert_mask_to_polygon(mask, max_only=False, integrate=False):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]
    # cv.drawContours(img, contours, -1, (0,255,0), 3)
    # print(len(contours))
    if max_only:
        contours = np.array(max(contours, key=lambda arr: arr.size)).reshape(-1, 2)
        return contours

    if integrate:
        group = []
        for contour in contours:
            if contour.size > 3 * 2:
                group.append(contour)
        contours = np.concatenate(group, axis=0).reshape(-1, 2)
        return contours

    return contours


def ang2rad(a):
    return a / 180 * np.pi


def rad2ang(r):
    return r / np.pi * 180


def rotation_2d(angle):
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def rotate_x(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[1, 1] = cos_a
    r_mat[2, 2] = cos_a
    r_mat[1, 2] = -sin_a
    r_mat[2, 1] = sin_a
    return r_mat


def rotate_y(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[0, 0] = cos_a
    r_mat[2, 2] = cos_a
    r_mat[0, 2] = sin_a
    r_mat[2, 0] = -sin_a
    return r_mat


def rotate_z(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[0, 0] = cos_a
    r_mat[1, 1] = cos_a
    r_mat[0, 1] = -sin_a
    r_mat[1, 0] = sin_a
    return r_mat


def mask_dilate(mask, kernel_size=3):
    mask = mask.astype('uint8')
    objs = np.unique(mask)
    res_mask = np.zeros_like(mask)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for obj in objs[1:]:
        binary_mask = np.where(mask == obj, 1, 0).astype(np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        res_mask[dilated_mask > 0] = obj

    return res_mask


class OmniCam():
    """
    Spherical camera model
    u[0:image_w] -> lon[-pi:pi]
    v[0:image_h] -> lat[pi/2:-pi/2]
    camera coordinate (opencv convention): x_axis: right; y_axis: down; z_axis: outwards
    """

    def __init__(self, img_w=2048, img_h=1024):
        self.img_w = img_w
        self.img_h = img_h

        # 定义“横向焦距/比例系数” fx：把经度范围 -𝜋 到 𝜋 (总共2𝜋)映射到像素宽度 img_w
        self.fx = img_w / (2 * np.pi)
        # 定义“纵向焦距/比例系数” fy：把纬度范围 -𝜋/2 到 𝜋/2 (总共𝜋)映射到像素高度 img_h
        # 这里是负号：因为图像坐标 v 向下增大，但纬度 lat 往“上”是正（𝜋/2），所以要用负比例把方向对齐。
        self.fy = -img_h / np.pi

        # 主点（图像中心）坐标，类似 pinhole 相机里的 (cx, cy)。
        self.cx = img_w / 2
        self.cy = img_h / 2

    def uv2lonlat(self, u, v):
        """
        像素坐标 → 经纬度
            u: 图像坐标 x 坐标，范围 [0, img_w)
            v: 图像坐标 y 坐标，范围 [0, img_h)
        Returns:
            lon: 经度，范围 [-𝜋, 𝜋]
            lat: 纬度，范围 [-𝜋/2, 𝜋/2]
        """
        # (1) u + 0.5：把像素索引当作“像素中心”（常见做法，减少半像素偏差）
        # (2) -self.cx：以图像中心为 0
        # (3) / self.fx：从像素单位变为弧度单位
        # (4) 结果 lon 约在 [-𝜋, 𝜋] 之间
        lon = ((u + 0.5) - self.cx) / self.fx
        # 把像素纵坐标转成纬度，逻辑同上，lat 约在 [-𝜋/2, 𝜋/2] 之间; 这里也有负号，因为 v 向下增大，lat 向上增大，所以也要负比例。
        lat = ((v + 0.5) - self.cy) / self.fy
        return lon, lat

    def lonlat2xyz(self, lon, lat):
        """
        经纬度 → 单位球面三维坐标 (x,y,z)
            lon: 经度，范围 [-𝜋, 𝜋]
            lat: 纬度，范围 [-𝜋/2, 𝜋/2]
        Returns:
            x: 三维坐标 x 坐标
            y: 三维坐标 y 坐标
            z: 三维坐标 z 坐标
        """
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(-lat)
        z = np.cos(lat) * np.cos(lon)
        return x, y, z

    def uv2xyz(self, u, v):
        lon, lat = self.uv2lonlat(u, v)
        return self.lonlat2xyz(lon, lat)

    def xyz2lonlat(self, x, y, z, norm=False):
        lon = np.arctan2(x, z)
        lat = np.arcsin(-y) if norm else np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))
        return lon, lat

    def lonlat2uv(self, lon, lat):
        u = lon * self.fx + self.cx - 0.5
        v = lat * self.fy + self.cy - 0.5
        return u, v

    def xyz2uv(self, x, y, z, norm=False):
        lon, lat = self.xyz2lonlat(x, y, z, norm)
        return self.lonlat2uv(lon, lat)

    def get_inverse_lonlat(self, R, u, v):
        """
        给定旋转矩阵 R（3×3）和像素点 (u,v)，返回“经过旋转后的方向”对应的经纬度
        常用于：把当前图像像素对应的方向，变换到另一个坐标系/相机姿态下去看它的经纬度。
            R: 旋转矩阵，3×3
            u: 图像坐标 x 坐标，范围 [0, img_w)
            v: 图像坐标 y 坐标，范围 [0, img_h)
        Returns:
            lon: 经度，范围 [-𝜋, 𝜋]
            lat: 纬度，范围 [-𝜋/2, 𝜋/2]
        """
        x, y, z = self.uv2xyz(u, v)
        xyz = R @ np.array([x, y, z])
        lon, lat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])
        return lon, lat

    def get_rough_FOV(self, bbox_w, bbox_h):
        """
        粗略 FOV 估计（按像素比例换算角度）
            bbox_w: 物体宽度，单位：像素
            bbox_h: 物体高度，单位：像素
        Returns:
            fov_h: 大致的横向 FOV，单位：度
            fov_v: 大致的纵向 FOV，单位：度
        """
        return bbox_w / self.img_w * 360, bbox_h / self.img_h * 180


class OmniImage(OmniCam):
    def __init__(self, img_w=2048, img_h=1024):
        super().__init__(img_w, img_h)

        # 预先生成一张“全景图每个像素对应的三维方向”的数组 self.xyz，用于后续快速投影。
        self.xyz = self._init_omni_image_cor()

    def _init_omni_image_cor(self, fov_h=360, fov_v=180, num_sample_h=None, num_sample_v=None):
        """
        生成一个形状约为 (num_sample_v, num_sample_h, 3) 的数组，每个位置是一个单位球方向 (x,y,z)。
            fov_h: 横向 FOV，单位：度
            fov_v: 纵向 FOV，单位：度
            num_sample_h: 横向采样点数，默认 None（自动根据 fov_h 和 fov_v 计算）
            num_sample_v: 纵向采样点数，默认 None（自动根据 fov_h 和 fov_v 计算）
        Returns:
            xyz: “全景图每个像素对应的三维方向”的数组，形状 (num_sample_h, num_sample_v, 3)
        """

        # 把水平/垂直视场角从度转为弧度
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        # 默认采样分辨率，若不指定，水平采样数用图像宽 img_w
        if num_sample_h is None:
            num_sample_h = self.img_w
        # 垂直采样数按视场比例计算：让采样点在角度上近似保持同等密度。
        # 对完整全景 360:180=2:1，就会得到 num_sample_v ≈ num_sample_h/2，匹配 2:1
        if num_sample_v is None:
            num_sample_v = int(num_sample_h * (fov_v / fov_h))

        # 视场覆盖是中心对称的
        lon_range = fov_h / 2
        lat_range = fov_v / 2

        lon, lat = np.meshgrid(np.linspace(-lon_range, lon_range, num_sample_h),
                               np.linspace(lat_range, -lat_range, num_sample_v))
        
        # 调用父类方法，把经纬度网格转换成单位球三维方向网格
        x, y, z = self.lonlat2xyz(lon, lat)

        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        return xyz

    def _init_perspective_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v):
        """
        生成透视相机（针孔）采样方向，即生成一个透视投影（pinhole）下，图像平面上每个采样点对应的相机坐标系方向向量（未归一化也没关系）。
            fov_h: 横向 FOV，单位：度
            fov_v: 纵向 FOV，单位：度
            num_sample_h: 横向采样点数
            num_sample_v: 纵向采样点数
        Returns:
            xyz: “透视投影下，图像平面上每个采样点对应的相机坐标系方向向量”的数组，形状 (num_sample_h, num_sample_v, 3)
        """
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        # initi tangent image
        len_x = np.tan(fov_h / 2)
        len_y = np.tan(fov_v / 2)
        if num_sample_v is None:
            num_sample_v = int(num_sample_h * (fov_v / fov_h))

        cx, cy = np.meshgrid(np.linspace(-len_x, len_x, num_sample_h), np.linspace(-len_y, len_y, num_sample_v))
        xyz = np.concatenate([cx[..., None], cy[..., None], np.ones_like(cx)[..., None]], axis=-1)
        return xyz

    def _init_cylindrical_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v):
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        # initi tangent image
        len_x = fov_h / 2
        len_y = np.tan(fov_v / 2)
        if num_sample_v is None:
            num_sample_v = int(num_sample_h * fov_v / fov_h)

        lon, cy = np.meshgrid(np.linspace(-len_x, len_x, num_sample_h), np.linspace(-len_y, len_y, num_sample_v))
        x = np.sin(lon)
        z = np.cos(lon)

        xyz = np.concatenate([x[..., None], cy[..., None], z[..., None]], axis=-1)

        return xyz

    def _get_bfov_regin(self, bfov, projection_type=None, num_sample_h=500, num_sample_v=None):
        c_lon = ang2rad(bfov.clon)
        c_lat = ang2rad(bfov.clat)
        rz = ang2rad(bfov.rotation)
        R = rotate_y(c_lon) @ rotate_x(c_lat) @ rotate_z(rz)
        if projection_type is None:
            if bfov.fov_h > 90 or bfov.fov_v > 90:
                projection_type = 2
            else:
                projection_type = 0

        if projection_type == 0:
            xyz = self._init_perspective_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)
        elif projection_type == 1:
            xyz = self._init_cylindrical_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)
        else:
            xyz = self._init_omni_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)

        xyz_new = xyz @ R.transpose()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2])
        u = u.astype(np.float32) % (self.img_w - 1)
        v = v.astype(np.float32) % (self.img_h - 1)
        return u, v

    def crop_bfov(self, img, bfov, projection_type=None, num_sample_h=1000, num_sample_v=None):
        # supposed the input is in angle
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h, num_sample_v)
        out = cv2.remap(img, u, v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)  # INTER_LINEAR
        # print(out.shape)
        return out, u, v

    def uncrop_bfov(self, cropped_img, bfov, projection_type=None, out_width=3840, out_height=1920, kernel=3):
        # supposed the input is in angle
        u, v = self._get_bfov_regin(bfov, projection_type, cropped_img.shape[1], cropped_img.shape[0])

        u_flat = u.flatten()
        v_flat = v.flatten()

        x_indices = np.tile(np.arange(v.shape[1]), (v.shape[0], 1)).flatten()
        y_indices = np.repeat(np.arange(v.shape[0]), u.shape[1])

        map_x = np.full((out_height, out_width), -1, dtype=np.float32)
        map_y = np.full((out_height, out_width), -1, dtype=np.float32)

        valid_mask = (u_flat >= 0) & (u_flat < out_width) & (v_flat >= 0) & (v_flat < out_height)
        valid_u = u_flat[valid_mask].astype(int)
        valid_v = v_flat[valid_mask].astype(int)
        valid_x = x_indices[valid_mask]
        valid_y = y_indices[valid_mask]

        map_x[valid_v, valid_u] = valid_x
        map_y[valid_v, valid_u] = valid_y

        new_img = cv2.remap(cropped_img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        new_img = mask_dilate(new_img, kernel)

        return new_img, u, v

    def plot_bfov(self, img, bfov, projection_type=None, num_sample_h=1000, num_sample_v=None, border_only=True,
                  color=(255, 0, 0), size=10):
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h, num_sample_v)  # img_h,
        img = self.plot_uv(img, u, v, border_only, color, size)
        return img

    def plot_uv(self, img, u, v, border_only=True, color=(255, 0, 0), size=2):
        if border_only:
            for j in range(u.shape[1]):
                cv2.circle(img, (int(u[0, j]), int(v[0, j])), 1, color, size)
                cv2.circle(img, (int(u[-1, j]), int(v[-1, j])), 1, color, size)

            for i in range(u.shape[0]):
                cv2.circle(img, (int(u[i, 0]), int(v[i, 0])), 1, color, size)
                cv2.circle(img, (int(u[i, -1]), int(v[i, -1])), 1, color, size)
        else:
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    cv2.circle(img, (int(u[i, j]), int(v[i, j])), 1, color, 1)
        return img

    def crop_bbox(self, img, bbox, borderMode=cv2.BORDER_CONSTANT, needBoderValue=False):
        # only consider horizontal rotation, and pad the vertical
        u, v = np.meshgrid(np.linspace(-bbox.w * 0.5, bbox.w * 0.5, int(bbox.w)),
                           np.linspace(-bbox.h * 0.5, bbox.h * 0.5, int(bbox.h)))
        R = rotation_2d(ang2rad(bbox.rotation))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1) @ R.transpose()
        u = uv[..., 0] + bbox.cx
        v = uv[..., 1] + bbox.cy

        u = u.astype(np.float32) % (self.img_w - 1)
        v = v.astype(np.float32)  # %self.img_h

        borderValue = cv2.mean(img) if needBoderValue else [0, 0, 0]
        out = cv2.remap(img, u, v, interpolation=cv2.INTER_NEAREST, borderMode=borderMode, borderValue=borderValue)
        return out, u, v

    def plot_bbox(self, img, bbox, color=(255, 0, 0), size=10):
        # according to the characteristic of 360, it is impossible to cross top and bottom, thus only consider cross left and right
        # print("plot_bbox", bbox.todict(), bbox.topLeft, bbox.topRight, bbox.topRight, bbox.bottomLeft )
        img_h, img_w = img.shape[:2]
        topleft = bbox.topLeft.copy()
        topRight = bbox.topRight.copy()
        bottomRight = bbox.bottomRight.copy()
        bottomLeft = bbox.bottomLeft.copy()

        topLine = sgeo.LineString([topleft, topRight])
        rightLine = sgeo.LineString([topRight, bottomRight])
        bottomLine = sgeo.LineString([bottomRight, bottomLeft])
        leftLine = sgeo.LineString([bottomLeft, topleft])

        leftBorder = sgeo.LineString([(-1, 0), (-1, img_h)])
        rightBorder = sgeo.LineString([(img_w, 0), (img_w, img_h)])

        lines = []

        if topLine.intersects(leftBorder):
            point = topLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                lines.append([topleft, [img_w - 1, point.y]])
                lines.append([[0, point.y], topRight])
        elif topLine.intersects(rightBorder):
            point = topLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                lines.append([topleft, [img_w - 1, point.y]])
                lines.append([[0, point.y], topRight])
        else:
            lines.append([topleft, topRight])

        if bottomLine.intersects(leftBorder):
            point = bottomLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                lines.append([bottomLeft, [img_w - 1, point.y]])
                lines.append([[0, point.y], bottomRight])
        elif bottomLine.intersects(rightBorder):
            point = bottomLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                lines.append([bottomLeft, [img_w - 1, point.y]])
                lines.append([[0, point.y], bottomRight])
        else:
            lines.append([bottomLeft, bottomRight])

        if rightLine.intersects(leftBorder):
            point = rightLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                if topRight[0] < 0:
                    lines.append([topRight, [img_w - 1, point.y]])
                    lines.append([[0, point.y], bottomRight])
                else:
                    lines.append([topRight, [0, point.y]])
                    lines.append([[img_w - 1, point.y], bottomRight])

        elif rightLine.intersects(rightBorder):
            point = rightLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                if topRight[0] < img_w:
                    lines.append([topRight, [img_w - 1, point.y]])
                    lines.append([[0, point.y], bottomRight])
                else:
                    lines.append([topRight, [0, point.y]])
                    lines.append([[img_w - 1, point.y], bottomRight])
        else:
            lines.append([topRight, bottomRight])

        if leftLine.intersects(leftBorder):
            point = leftLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                if topleft[0] < 0:
                    lines.append([topleft, [img_w - 1, point.y]])
                    lines.append([[0, point.y], bottomLeft])
                else:
                    lines.append([topleft, [0, point.y]])
                    lines.append([[img_w - 1, point.y], bottomLeft])

        elif leftLine.intersects(rightBorder):
            point = leftLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                if topleft[0] < img_w:
                    lines.append([topleft, [img_w - 1, point.y]])
                    lines.append([[0, point.y], bottomLeft])
                else:
                    lines.append([topleft, [0, point.y]])
                    lines.append([[img_w - 1, point.y], bottomLeft])
        else:
            lines.append([topleft, bottomLeft])

        for line in lines:
            start, end = line.copy()
            start = np.intp(start)
            end = np.intp(end)
            start[0] %= img_w
            end[0] %= img_w
            cv2.line(img, start, end, color, size)
        return img

    def _get_global_coordinate(self, bbox, ref_u, ref_v):
        # convert local box coordinate to coordinate

        u_init, v_init = np.meshgrid(np.linspace(-bbox.w * 0.5, bbox.w * 0.5, int(bbox.w)),
                                     np.linspace(-bbox.h * 0.5, bbox.h * 0.5, int(bbox.h)))
        R = rotation_2d(ang2rad(bbox.rotation))
        uv_init = np.concatenate([u_init[..., None], v_init[..., None]], axis=-1) @ R.transpose()
        u_local = uv_init[..., 0] + bbox.cx
        v_local = uv_init[..., 1] + bbox.cy
        u_local = u_local.astype(np.float32)
        v_local = v_local.astype(np.float32)
        u_global = cv2.remap(ref_u, u_local, v_local, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        v_global = cv2.remap(ref_v, u_local, v_local, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)

        c_u = ref_u[int(bbox.cy), int(bbox.cx)]
        c_v = ref_v[int(bbox.cy), int(bbox.cx)]
        return c_u, c_v, u_global, v_global

    def uv2Bbox(self, u, v, cx, need_rotation):

        shift = cx - self.img_w * 0.5

        u_new = (u - shift) % (self.img_w - 1)
        v_new = v
        uv = np.concatenate([u_new[..., None], v_new[..., None]], axis=-1).reshape(-1, 2)
        rotation_angle = 0
        if need_rotation:
            rect_xy, rect_wh, rect_ang = rect = cv2.minAreaRect(uv)
            cx, cy = rect_xy
            w, h = rect_wh
            rotation_angle = rect_ang
            # print(cx, cy, w, h, rect_ang)
        else:
            lx, ly, w, h = cv2.boundingRect(uv)
            # print(lx, ly, w, h)
            cx = lx + w * .5
            cy = ly + h * .5

        """
        x1, y1 = np.min(u_new), np.min(v_new)
        x2, y2 = np.max(u_new), np.max(v_new)
        w = (x2 - x1)#%self.img_w 
        h = (y2 - y1)#%self.img_h
        #w = w if w < self.img_w else self.img_
        assert w > 0 and h > 0
        cx = (x2+x1) * 0.5 - shift if w < self.img_w-1 else self.img_w * 0.5
        cy = (y2+y1) * 0.5
        """
        cx = cx + shift if w < self.img_w - 1 else self.img_w * 0.5

        return Bbox(cx, cy, w, h, rotation_angle)

    def localBbox2Bfov(self, bbox, ref_u, ref_v, need_rotation=True):
        # Args:
        # bbox: supposed to be axis-aligned
        # ref_u, ref_v: coordinate of the local region with respect to origin 360 image
        # Return: Bfov
        # consider the case of rotated local bbox
        c_u, c_v, u_global, v_global = self._get_global_coordinate(bbox, ref_u, ref_v)

        c_lon, c_lat = self.uv2lonlat(c_u, c_v)
        R = rotate_y(c_lon) @ rotate_x(c_lat)

        u = np.concatenate([u_global[0, :], u_global[:, -1], u_global[-1, :], u_global[:, 0]])
        v = np.concatenate([v_global[0, :], v_global[:, -1], v_global[-1, :], v_global[:, 0]])

        # print(uv)
        x, y, z = self.uv2xyz(u, v)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ R
        lon, lat = self.xyz2lonlat(xyz[:, 0], xyz[:, 1], xyz[:, 2], True)

        rotation_angle = 0
        if need_rotation:
            shift = 1000
            scale = 360
            lon = np.intp(lon * scale + shift)
            lat = np.intp(lat * scale + shift)
            # print(lon)
            lonlat = np.concatenate([lon[..., None], lat[..., None]], axis=-1).reshape(-1, 2)
            rect_xy, rect_wh, rect_ang = cv2.minAreaRect(lonlat)
            c_lon, c_lat = rect_xy
            c_lon = (c_lon - shift) / scale
            c_lat = (c_lat - shift) / scale
            fov_h, fov_v = rect_wh
            fov_h /= scale
            fov_v /= scale
            rotation_angle = -rect_ang

            if abs(rect_ang + bbox.rotation) > 85:
                temp = fov_v
                fov_v = fov_h
                fov_h = temp
                rotation_angle += 90
                # print("swap")
            # print(c_lon, c_lat, fov_h, fov_v, rect_ang)
        else:
            lon_max, lon_min = np.max(lon), np.min(lon)
            lat_max, lat_min = np.max(lat), np.min(lat)
            fov_h = lon_max - lon_min
            fov_v = lat_max - lat_min
            c_lon = (lon_max + lon_min) * 0.5
            c_lat = (lat_max + lat_min) * 0.5

        x, y, z = self.lonlat2xyz(c_lon, c_lat)
        xyz = R @ np.array([x, y, z])
        c_lon, c_lat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])
        bfov = Bfov(rad2ang(c_lon), rad2ang(c_lat), rad2ang(fov_h), rad2ang(fov_v), rotation_angle)

        return bfov  # , u_global, v_global

    def localBbox2Bbox(self, bbox, ref_u, ref_v, need_rotation=True):
        # get the global bbox to cover the area of local bbox.
        c_u, c_v, u_global, v_global = self._get_global_coordinate(bbox, ref_u, ref_v)
        return self.uv2Bbox(u_global, v_global, c_u, need_rotation)

    def bbox2Bfov(self, bbox, need_rotation=False):
        # Args: the bbox with respect to the 360 image
        # Return: Bfov
        # get the boundary
        topLine = sgeo.LineString([bbox.topLeft, bbox.topRight])
        rightLine = sgeo.LineString([bbox.topRight, bbox.bottomRight])
        bottomLine = sgeo.LineString([bbox.bottomRight, bbox.bottomLeft])
        leftLine = sgeo.LineString([bbox.bottomLeft, bbox.topLeft])

        uv1 = np.array([topLine.interpolate(dis).xy for dis in np.linspace(0, topLine.length, bbox.w)])
        uv2 = np.array([rightLine.interpolate(dis).xy for dis in np.linspace(0, rightLine.length, bbox.h)])
        uv3 = np.array([bottomLine.interpolate(dis).xy for dis in np.linspace(0, bottomLine.length, bbox.w)])
        uv4 = np.array([leftLine.interpolate(dis).xy for dis in np.linspace(0, leftLine.length, bbox.h)])
        # print(uv1.shape, uv2.shape)
        uv = np.concatenate([uv1, uv2, uv3, uv4]).reshape(-1, 2)
        # print(uv.shape)
        u = uv[:, 0] % (self.img_w - 1)
        v = uv[:, 1] % (self.img_h - 1)

        clon, clat = self.uv2lonlat(bbox.cx, bbox.cy)
        rotation_angle = bbox.rotation if need_rotation else 0

        R = rotate_y(clon) @ rotate_x(clat) @ rotate_z(ang2rad(rotation_angle))

        x, y, z = self.uv2xyz(u, v)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ R
        lon, lat = self.xyz2lonlat(xyz[..., 0], xyz[..., 1], xyz[..., 2], True)

        clon = (np.max(lon) + np.min(lon)) * 0.5
        clat = (np.max(lat) + np.min(lat)) * 0.5
        x, y, z = self.lonlat2xyz(clon, clat)
        xyz = R @ np.array([x, y, z])
        clon, clat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])

        clon = rad2ang((clon))
        clat = rad2ang((clat))

        fov_h = rad2ang(np.max(lon) - np.min(lon))
        fov_v = rad2ang(np.max(lat) - np.min(lat))

        return Bfov(clon, clat, fov_h, fov_v, rotation_angle), u, v

    def bfov2Bbox(self, bfov, need_rotation=False, projection_type=None):
        # Args: Bfov
        # Return: the bbox with respect to the 360 image
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h=500)  # img_h,
        cx, _ = self.lonlat2uv(ang2rad(bfov.clon), 0)
        return self.uv2Bbox(u, v, cx, need_rotation)

    def align_center_by_lonlat(self, img, lon, lat, rotation=0):
        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            self.img_w = img.shape[1]
            self.img_h = img.shape[0]
            self._init_image_cor()
        R = rotate_y(lon) @ rotate_x(lat) @ rotate_z(rotation)

        xyz_new = self.xyz @ R.transpose()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2], True)
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        out_img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        return out_img, R

    def align_center_by_lonlatangle(self, img, lon, lat, rotation=0):
        lat = ang2rad(lat)
        lon = ang2rad(lon)
        rotation = ang2rad(rotation)
        return self.align_center_by_lonlat(img, lon, lat, rotation)

    def align_center(self, img, u, v, rotation=0):
        lon, lat = self.uv2lonlat(u, v)
        rotation = ang2rad(rotation)
        return self.align_center_by_lonlat(img, lon, lat, rotation)

    def rot_image(self, img, pitch, yaw, roll=0):
        """
        rot the image by the angle
        positive pitch pulls up the (original center of) image
        positive yaw shifts the (original center of) image to the right
        positive roll clockwise rotates the image along the center
        """
        lat = -ang2rad(pitch)
        lon = -ang2rad(yaw)
        rotation = -ang2rad(roll)
        return self.align_center_by_lonlat(img, lon, lat, rotation)

    def mask2Bfov(self, mask_image, need_rotation=True):
        assert self.img_w == mask_image.shape[1] and self.img_h == mask_image.shape[0]

        if len(mask_image.shape) > 2:
            mask = mask_image[:, :, 0].copy()
        else:
            mask = mask_image.copy()
        test_v, test_u = np.where(mask > 0)
        if len(test_v) < 8:
            return None
        # need to consider disapper case
        contours1 = convert_mask_to_polygon(mask, max_only=True)
        cx, cy = np.mean(contours1, axis=0)
        # rough rotation step 2
        mask_image_rotation, R = self.align_center(mask, cx, cy)
        # adjust the image according to the centroid of the mask. as mask may cross image
        v, u = np.where(mask_image_rotation > 0)
        clon, clat = self.get_inverse_lonlat(R, np.mean(u), np.mean(v))
        # adjust original image again, Step 2.1
        mask_image_rotation2, R2 = self.align_center_by_lonlat(mask, clon, clat)
        # get the rotated bbox, ensure the target not cross edge
        contours2 = convert_mask_to_polygon(mask_image_rotation2, integrate=True)

        # get the final bfov or rbov estimation
        rect_xy, rect_wh, rect_ang = cv2.minAreaRect(contours2)
        clon2, clat2 = self.get_inverse_lonlat(R2, rect_xy[0], rect_xy[1])
        rotation_angle = 0
        if need_rotation:
            rotation_angle = rect_ang if rect_wh[0] > rect_wh[1] else -90 + rect_ang

        mask_image_rotation3, R3 = self.align_center_by_lonlat(mask, clon2, clat2, ang2rad(rotation_angle))
        contours3 = convert_mask_to_polygon(mask_image_rotation3, integrate=True)
        u = contours3[:, 0]
        v = contours3[:, 1]

        min_u, max_u = np.min(u), np.max(u)
        min_v, max_v = np.min(v), np.max(v)

        # based on the bbox, calculate the center and fov of fovbbox
        cu = (min_u + max_u) * 0.5
        cv = (min_v + max_v) * 0.5
        clon3, clat3 = self.get_inverse_lonlat(R3, cu, cv)
        clon = rad2ang(clon3)
        clat = rad2ang(clat3)
        # use a bbox to approximate
        min_lon, min_lat = self.uv2lonlat(min_u, max_v)
        max_lon, max_lat = self.uv2lonlat(max_u, min_v)
        fov_h = rad2ang(max_lon - min_lon)
        fov_v = rad2ang(max_lat - min_lat)

        bfov = Bfov(clon, clat, fov_h, fov_v, rotation_angle)
        return bfov

    def mask2Bbox(self, mask_image, need_rotation=True, expand: int = 100):
        """
        给一张全景(equirect)的二值/灰度 mask，计算一个包围它的 bbox；必要时返回旋转框（minAreaRect），
        并处理全景图的左右拼接边界问题（物体跨越 u=0/W 时普通 bbox 会很大）。
        """
        assert self.img_w == mask_image.shape[1] and self.img_h == mask_image.shape[0]

        if len(mask_image.shape) > 2:
            mask = mask_image[:, :, 0].copy()
        else:
            mask = mask_image.copy()

        test_v, test_u = np.where(mask > 0)
        if len(test_v) < 8:
            return None

        # Step 1: 先从 mask 得到轮廓，并估计目标中心经纬度
        # 把 mask 转换为多边形/轮廓点集合，并计算轮廓的中心点, 并转换为经纬度 (c_lon, c_lat)
        # 这个经纬度后面用于“把目标水平移动到图像中间”，解决跨边界问题
        contours1 = convert_mask_to_polygon(mask, max_only=True)
        cx, cy = np.mean(contours1, axis=0)
        c_lon, c_lat = self.uv2lonlat(cx, cy)

        # Step 2：把目标按经度对齐到图像中心（水平 shift）
        mask_image_rotation, R = self.align_center_by_lonlat(mask, c_lon, 0)
        shift = cx - self.img_w * 0.5
        
        # get the final bbox or rbbox
        contours2 = convert_mask_to_polygon(mask_image_rotation, integrate=True)
        rotation_angle = 0
        if need_rotation:
            rect_xy, rect_wh, rotation_angle = cv2.minAreaRect(contours2)
            cx, cy = rect_xy
            w, h = rect_wh
        else:
            lx, ly, w, h = cv2.boundingRect(contours2)
            cx = lx + w * 0.5
            cy = ly + h * 0.5
        w = w + expand
        h = h + expand
        cx = (cx + shift) % self.img_w if w < self.img_w - 1 else self.img_w * 0.5
        return Bbox(cx, cy, w, h, rotation_angle)

    def get_front_view_crop_from_pano(
        self,
        img,
        delta_yaw=0,
        delta_pitch=0,
        out_width=640,
        out_height=480,
        fov_h=None,
        fov_v=None,
        interpolation=cv2.INTER_LINEAR,
    ):
        """
        从当前全景图中按指定视角截取透视投影的矩形图像。

        Args:
            img: 全景图 (H, W, C)，equirectangular 格式
            delta_yaw: 中心方向的水平偏移角度（度），正为向右
            delta_pitch: 中心方向的俯仰偏移角度（度），正为向上
            out_width: 输出图像宽度
            out_height: 输出图像高度
            fov_h: 水平视场角（度）
            fov_v: 垂直视场角（度），若为 None，则根据 fov_h 和 out_width/out_height 计算
            interpolation: cv2 插值方式，默认 INTER_LINEAR

        Returns:
            out: 透视图像 (out_height, out_width, C)
        """
        if fov_v is None:
            fov_v = fov_h * (out_height / out_width)
        if fov_h is None:
            fov_h = fov_v * (out_width / out_height)
        bfov = Bfov(lon=delta_yaw, lat=delta_pitch, fov_h=fov_h, fov_v=fov_v, rotation=0)
        u, v = self._get_bfov_regin(bfov, projection_type=0, num_sample_h=out_width, num_sample_v=out_height)
        out = cv2.remap(img, u, v, interpolation=interpolation, borderMode=cv2.BORDER_WRAP)
        return out

    def get_mask_center_lonlat_deg(self, mask_image):
        """
        取 mask 最大连通域的中心，返回经纬度（度）。
        Returns:
            (c_lon_deg, c_lat_deg) 或 None（若 mask 无效）
        """
        if len(mask_image.shape) > 2:
            mask = mask_image[:, :, 0].copy()
        else:
            mask = mask_image.copy()
        test_v, test_u = np.where(mask > 0)
        if len(test_v) < 8:
            return None
        contours = convert_mask_to_polygon(mask, max_only=True)
        cx, cy = np.mean(contours, axis=0)
        c_lon, c_lat = self.uv2lonlat(cx, cy)
        return rad2ang(c_lon), rad2ang(c_lat)

    def get_front_view_crop_centered_on_mask(
        self,
        image,
        mask_image,
        view_width=640,
        view_height=480,
        fov_h=100,
        fov_v=None,
        expand=50,
    ):
        """
        将 mask 对应物体转到全景正中心（mask 最大连通域中心），投影到 2D 前向透视图，
        得到该视角下的 bbox/mask，并裁剪；若 bbox 超出视窗则裁到视窗内。

        Args:
            image: 全景图 (H, W, C)
            mask_image: 二值 mask (H, W) 或 (H, W, 1)，与 image 同尺寸
            view_width: 前向透视 view 宽度，默认 640
            view_height: 前向透视 view 高度，默认 480
            fov_h: 水平 FOV（度），默认 100；fov_v: 垂直 FOV（度），若为 None，则根据 fov_h 和 view_width/view_height 计算
            expand: 2D bbox 扩展像素

        Returns:
            dict: {
                "front_view": (view_height, view_width, C),
                "mask_view": (view_height, view_width) uint8,
                "bbox_2d": (x, y, w, h) 外接 2D bbox，已 clip 到视窗内,
                "crop_image": 外接 bbox 对应的 2D 裁剪上下文图,
                "object_crop_image": 外接 bbox 对应的仅物体图（crop 与 mask 相乘）,
                "bbox_2d_inner": (x, y, w, h) 内接 2D bbox（mask 紧包围），或 None,
                "crop_image_inner": 内接 bbox 对应的裁剪图，或 None,
                "object_crop_image_inner": 内接 bbox 对应的仅物体图，或 None,
            }
            若 mask 无效或 2D 视角下无前景，则返回 None。
        """
        assert image.shape[:2] == (self.img_h, self.img_w)
        if len(mask_image.shape) > 2:
            mask_np = mask_image[:, :, 0].astype(np.uint8)
        else:
            mask_np = mask_image.astype(np.uint8)

        center = self.get_mask_center_lonlat_deg(mask_np)
        if center is None:
            return None
        c_lon_deg, c_lat_deg = center

        if fov_v is None:
            fov_v = fov_h * (view_height / view_width)
        bfov = Bfov(lon=c_lon_deg, lat=c_lat_deg, fov_h=fov_h, fov_v=fov_v, rotation=0)
        u, v = self._get_bfov_regin(
            bfov,
            projection_type=0,
            num_sample_h=view_width,
            num_sample_v=view_height,
        )
        u = np.clip(u, 0, self.img_w - 1.01).astype(np.float32)
        v = np.clip(v, 0, self.img_h - 1.01).astype(np.float32)

        front_view = cv2.remap(
            image, u, v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        mask_view = cv2.remap(
            mask_np, u, v,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        ys, xs = np.where(mask_view > 0)
        if len(ys) == 0:
            return None
        x_min, x_max = int(np.min(xs)), int(np.max(xs)) + 1
        y_min, y_max = int(np.min(ys)), int(np.max(ys)) + 1

        x1 = max(0, x_min - expand)
        y1 = max(0, y_min - expand)
        x2 = min(view_width, x_max + expand)
        y2 = min(view_height, y_max + expand)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        bbox_2d = (x1, y1, w, h)

        crop_image = front_view[y1:y2, x1:x2].copy()
        mask_crop = mask_view[y1:y2, x1:x2]
        object_crop_image = crop_image * (mask_crop[:, :, np.newaxis] > 0)

        # 内接 2D bbox：mask 紧包围框（不 expand），clip 到视窗内
        ix1 = max(0, x_min)
        iy1 = max(0, y_min)
        ix2 = min(view_width, x_max)
        iy2 = min(view_height, y_max)
        iw = ix2 - ix1
        ih = iy2 - iy1
        bbox_2d_inner = (ix1, iy1, iw, ih) if iw > 0 and ih > 0 else None
        if bbox_2d_inner is not None:
            crop_image_inner = front_view[iy1:iy2, ix1:ix2].copy()
            mask_crop_inner = mask_view[iy1:iy2, ix1:ix2]
            object_crop_image_inner = crop_image_inner * (mask_crop_inner[:, :, np.newaxis] > 0)
        else:
            crop_image_inner = None
            object_crop_image_inner = None

        return {
            "front_view": front_view,
            "mask_view": mask_view,
            "bbox_2d": bbox_2d,
            "crop_image": crop_image,
            "object_crop_image": object_crop_image,
            "bbox_2d_inner": bbox_2d_inner,
            "crop_image_inner": crop_image_inner,
            "object_crop_image_inner": object_crop_image_inner,
        }