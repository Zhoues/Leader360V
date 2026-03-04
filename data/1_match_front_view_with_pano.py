#!/usr/bin/env python3
"""
将 hos_sft / hps_sft 下的 PNG（命名 {pano_id}_{task_id}_{view_id}.png）与全景视频文件夹中
每个 MP4 的第一帧进行 VLM 匹配；按 pano_id 聚合投票，超过一半 PNG 认为同一 MP4 则记为匹配成功。
输出 JSON：每个 pano_id 的 PNG→MP4 投票列表、匹配到的 MP4、是否匹配成功。
输出格式样例见：data/pano_match_result_example.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pano.mllm import *

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 路径配置
PANORAMIC_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360x_dataset_HR/panoramic")
SFT_BASE_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano")  # 用于生成相对路径，如 hos-sft/hos_sft_sharegpt/1_1_1.png
HOS_SFT_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hos-sft/hos_sft_sharegpt")
HPS_SFT_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hps-sft/hps_sft_sharegpt")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "pano_first_frames_cache"


def png_path_to_relative(png_path: Path | str) -> str:
    """将 PNG 绝对路径转为相对路径，如 hos-sft/hos_sft_sharegpt/1_1_1.png。"""
    p = Path(png_path).resolve()
    try:
        return str(p.relative_to(SFT_BASE_DIR.resolve()))
    except ValueError:
        return f"{p.parent.name}/{p.name}"


def mllm_pano_match_two_images(
    front_view_image: np.ndarray,
    pano_image: np.ndarray,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> bool:
    """判断前视图（第一张图）是否属于第二张全景图。返回 True 表示匹配。"""
    encoded_front = encode_image(front_view_image)
    encoded_pano = encode_image(pano_image)
    prompt = [
        {"type": "text", "text": mllm_pano_match_user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_pano}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_front}"},
        },
    ]
    try:
        message = gpt_4o_complete(
            model=model,
            prompt=prompt,
            system_prompt=mllm_pano_match_system_prompt,
        )
        out = json.loads(message)
        return bool(out.get("match", False))
    except Exception:
        return False

def parse_png_filename(path: Path) -> tuple[str, int, int] | None:
    """解析 PNG 文件名 {pano_id}_{task_id}_{view_id}.png，task_id/view_id 从 1 计数。返回 (pano_id, task_id, view_id)。"""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        view_id = int(parts[-1])
        task_id = int(parts[-2])
        pano_id = "_".join(parts[:-2])
        return (pano_id, task_id, view_id)
    except ValueError:
        return None


def collect_pngs_by_pano_id(
    dirs_with_prefix: list[tuple[Path, str]],
) -> dict[str, list[Path]]:
    """从多个目录收集 PNG，按 pano_id 分组。每个目录带前缀以区分来源（如 hos / hps）。"""
    pano_to_pngs: dict[str, list[Path]] = defaultdict(list)
    for d, prefix in dirs_with_prefix:
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            parsed = parse_png_filename(p)
            if parsed is not None:
                raw_pano_id, _, _ = parsed
                pano_id = f"{prefix}_{raw_pano_id}"
                pano_to_pngs[pano_id].append(p)
    for k in pano_to_pngs:
        pano_to_pngs[k] = sorted(set(pano_to_pngs[k]))
    return dict(pano_to_pngs)


def get_mp4_first_frame(mp4_path: Path) -> tuple[str, object] | None:
    """读取 MP4 第一帧，返回 (mp4_basename, BGR 图像)。失败返回 None。"""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    name = mp4_path.name  # 带 .mp4
    return (name, frame)


def build_pano_first_frame_cache(mp4_dir: Path, cache_dir: Path) -> dict[str, object]:
    """
    构建 mp4_name -> 第一帧图像。优先从 cache_dir 读取已缓存的 jpg，不存在时从 MP4 提取并写入缓存。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp4_list = list(mp4_dir.glob("*.mp4"))
    cache = {}
    for mp4_path in tqdm(mp4_list, desc="MP4 第一帧缓存"):
        name = mp4_path.name
        cache_path = cache_dir / (mp4_path.stem + ".jpg")
        if cache_path.exists():
            frame = cv2.imread(str(cache_path))
            if frame is not None:
                cache[name] = frame
                continue
        res = get_mp4_first_frame(mp4_path)
        if res is not None:
            _, frame = res
            cache[name] = frame
            cv2.imwrite(str(cache_path), frame)
    return cache


def run_one_match(
    pano_id: str,
    png_path: Path,
    mp4_name: str,
    pano_first_frames: dict,
    model: str,
) -> tuple[str, str, str, bool]:
    """单次匹配：加载 PNG，与指定 MP4 的第一帧送 VLM，返回 (pano_id, png_path_str, mp4_name, match)。"""

    img = cv2.imread(str(png_path))
    if img is None:
        return (pano_id, str(png_path), mp4_name, False)
    pano_frame = pano_first_frames.get(mp4_name)
    if pano_frame is None:
        return (pano_id, str(png_path), mp4_name, False)
    match = mllm_pano_match_two_images(img, pano_frame, model=model)
    return (pano_id, str(png_path), mp4_name, match)


def aggregate_votes(
    results: list[tuple[str, str, str, bool]],
    pano_to_pngs: dict[str, list[Path]],
) -> dict[str, dict]:
    """
    按 pano_id 聚合：(pano_id, png_path, mp4_name, match) -> 每个 pano_id 的
    png_votes（每张 PNG 匹配到的 MP4 列表）、mp4_vote_counts、matched_mp4、match_success。
    保证每个 pano_id 下所有 PNG 都出现在 png_votes 中（无匹配时 matched_mp4_names 为空）。
    """
    # pano_id -> png_path -> [mp4_name, ...] (matched)
    pano_png_matches: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for pano_id, png_path, mp4_name, match in results:
        if match:
            pano_png_matches[pano_id][png_path].append(mp4_name)

    out = {}
    for pano_id, png_list in pano_to_pngs.items():
        png_to_mp4s = pano_png_matches.get(pano_id, {})
        png_votes = [
            {"png_path": png_path_to_relative(png_path), "matched_mp4_names": png_to_mp4s.get(str(png_path), [])}
            for png_path in sorted(png_list)
        ]
        # 统计每个 MP4 被多少张 PNG 选中（每张 PNG 可能选多个 MP4）
        mp4_vote_counts: dict[str, int] = defaultdict(int)
        for mp4_list in png_to_mp4s.values():
            for mp4_name in mp4_list:
                mp4_vote_counts[mp4_name] += 1
        n_pngs = len(png_list)
        threshold = (n_pngs / 2) if n_pngs else 0
        # 超过一半 PNG 都选中的 MP4 作为 matched_mp4（取票数最高的一个）
        candidates = [m for m, c in mp4_vote_counts.items() if c > threshold]
        if candidates:
            matched_mp4 = max(candidates, key=lambda m: mp4_vote_counts[m])
            match_success = True
        else:
            matched_mp4 = None
            match_success = False
        out[pano_id] = {
            "png_votes": png_votes,
            "mp4_vote_counts": dict(mp4_vote_counts),
            "matched_mp4": matched_mp4,
            "match_success": match_success,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="前视图 PNG 与全景 MP4 第一帧 VLM 匹配，按 pano_id 投票聚合")
    parser.add_argument(
        "--max-pano-ids",
        type=int,
        default=None,
        help="试跑模式：最多参与匹配的 pano_id 数量（默认不限制）",
    )
    parser.add_argument(
        "--max-mp4s",
        type=int,
        default=None,
        help="试跑模式：最多参与匹配的 MP4 数量；-1 表示使用全部 MP4（默认不限制）",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default=None,
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    os.environ["OPENAI_API_KEY"] = config["API"].get("openai_api_key", "")
    os.environ["OPENAI_BASE_URL"] = config["API"].get("openai_base_url", "http://127.0.0.1:25547/v1")
    model = config["API"].get("model_name", "Qwen/Qwen3-VL-8B-Instruct")

    # 1) 全景第一帧缓存（优先读 data/pano_first_frames_cache，否则从 MP4 提取并写入）
    if not PANORAMIC_DIR.exists():
        raise FileNotFoundError(f"全景视频目录不存在: {PANORAMIC_DIR}")
    pano_first_frames = build_pano_first_frame_cache(PANORAMIC_DIR, CACHE_DIR)
    mp4_names = list(pano_first_frames.keys())
    if not mp4_names:
        raise RuntimeError(f"未在 {PANORAMIC_DIR} 下找到可读的 MP4")

    if args.max_mp4s is not None and args.max_mp4s >= 0:
        mp4_names = mp4_names[: args.max_mp4s]
        print(f"[试跑] 仅使用前 {args.max_mp4s} 个 MP4: {len(mp4_names)} 个")
    # --max-mp4s -1 表示不限制，使用全部 MP4

    # 2) 按 pano_id 收集 PNG（hos / hps 分别加前缀，避免 pano_id 重复）
    pano_to_pngs = collect_pngs_by_pano_id([
        (HOS_SFT_DIR, "hos"),
        (HPS_SFT_DIR, "hps"),
    ])
    if not pano_to_pngs:
        raise RuntimeError("未在 hos_sft / hps_sft 下找到符合命名规则的 PNG")

    print(f"有效 MP4 数量: {len(mp4_names)} 个，有效 pano_id 数量: {len(pano_to_pngs)} 个，有效 PNG 数量: {sum(len(v) for v in pano_to_pngs.values())} 张")


    if args.max_pano_ids is not None:
        pano_to_pngs = dict(list(pano_to_pngs.items())[: args.max_pano_ids])
        print(f"[试跑] 仅使用前 {args.max_pano_ids} 个 pano_id: {list(pano_to_pngs.keys())}")

    # 3) 构建任务：(pano_id, png_path, mp4_name)
    tasks = []
    for pano_id, png_list in pano_to_pngs.items():
        for png_path in png_list:
            for mp4_name in mp4_names:
                tasks.append((pano_id, png_path, mp4_name))
    print(f"当前匹配 pano_id 数量: {len(pano_to_pngs)} 个，构建任务: {len(tasks)} 个")

    max_workers = min(16, os.cpu_count() or 4)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_match,
                pano_id,
                png_path,
                mp4_name,
                pano_first_frames,
                model,
            ): (pano_id, png_path, mp4_name)
            for pano_id, png_path, mp4_name in tasks
        }
        with tqdm(total=len(futures), desc="VLM 匹配进度", unit="pair") as pbar:
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception:
                    pano_id, png_path, mp4_name = futures[future]
                    results.append((pano_id, str(png_path), mp4_name, False))
                pbar.update(1)

    # 4) 聚合：每个 pano_id 的 png_votes、mp4_vote_counts、matched_mp4、match_success
    aggregated = aggregate_votes(results, pano_to_pngs)

    if args.output_json_path is None:
        args.output_json_path = PROJECT_ROOT / "data" / "pano_match_result.json"
    args.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {args.output_json_path}")
    n_success = sum(1 for v in aggregated.values() if v["match_success"])
    print(f"匹配成功 pano_id 数: {n_success} / {len(aggregated)}")

if __name__ == "__main__":
    main()
