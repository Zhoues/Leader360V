#!/usr/bin/env python3
"""
简化版：每个 pano_id 只选一张 PNG 与全景 MP4 第一帧做 VLM 匹配，只要该张匹配成功即视为该 pano_id 匹配成功。
使用多线程，进度条与主脚本一致（按 task 个数）。
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PANORAMIC_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/360x_dataset_HR/panoramic")
SFT_BASE_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano")
HOS_SFT_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hos-sft/hos_sft_sharegpt")
HPS_SFT_DIR = Path("/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/hps-sft/hps_sft_sharegpt")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "pano_first_frames_cache"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "pano_match_result_simple.json"


def png_path_to_relative(png_path: Path | str) -> str:
    p = Path(png_path).resolve()
    try:
        return str(p.relative_to(SFT_BASE_DIR.resolve()))
    except ValueError:
        return f"{p.parent.name}/{p.name}"


def mllm_pano_match_two_images(
    front_view_image: np.ndarray,
    pano_image: np.ndarray,
    model: str,
) -> bool:
    encoded_front = encode_image(front_view_image)
    encoded_pano = encode_image(pano_image)
    prompt = [
        {"type": "text", "text": mllm_pano_match_user_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_front}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_pano}"}},
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
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return (mp4_path.name, frame)


def build_pano_first_frame_cache(mp4_dir: Path, cache_dir: Path) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp4_list = list(mp4_dir.glob("*.mp4"))
    cache = {}
    for mp4_path in tqdm(mp4_list, desc="MP4 第一帧缓存"):
        cache_path = cache_dir / (mp4_path.stem + ".jpg")
        if cache_path.exists():
            frame = cv2.imread(str(cache_path))
            if frame is not None:
                cache[mp4_path.name] = frame
                continue
        res = get_mp4_first_frame(mp4_path)
        if res is not None:
            name, frame = res
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
    """单次匹配：返回 (pano_id, png_path_str, mp4_name, match)。"""
    img = cv2.imread(str(png_path))
    if img is None:
        return (pano_id, str(png_path), mp4_name, False)
    pano_frame = pano_first_frames.get(mp4_name)
    if pano_frame is None:
        return (pano_id, str(png_path), mp4_name, False)
    match = mllm_pano_match_two_images(img, pano_frame, model=model)
    return (pano_id, str(png_path), mp4_name, match)


def aggregate_simple_results(
    results: list[tuple[str, str, str, bool]],
    pano_to_one_png: dict[str, Path],
) -> dict[str, dict]:
    """按 pano_id 聚合：每 pano_id 一张图，有任意 MP4 匹配即 match_success。"""
    pano_matched: dict[str, list[str]] = defaultdict(list)
    for pano_id, png_path, mp4_name, match in results:
        if match:
            pano_matched[pano_id].append(mp4_name)
    out = {}
    for pano_id, png_path in pano_to_one_png.items():
        matched_list = pano_matched.get(pano_id, [])
        out[pano_id] = {
            "png_path": png_path_to_relative(png_path),
            "matched_mp4_names": matched_list,
            "matched_mp4": matched_list[0] if matched_list else None,
            "match_success": len(matched_list) > 0,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="简化版：每个 pano_id 选一张图匹配，匹配成功即成功")
    parser.add_argument("--max-pano-ids", type=int, default=None, help="最多处理的 pano_id 数量")
    parser.add_argument("--max-mp4s", type=int, default=None, help="最多使用的 MP4 数量；-1 表示全部")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    os.environ["OPENAI_API_KEY"] = config["API"].get("openai_api_key", "")
    os.environ["OPENAI_BASE_URL"] = config["API"].get("openai_base_url", "http://127.0.0.1:25547/v1")
    model = config["API"].get("model_name", "Qwen/Qwen3-VL-8B-Instruct")

    # 1) 全景第一帧缓存
    if not PANORAMIC_DIR.exists():
        raise FileNotFoundError(f"全景视频目录不存在: {PANORAMIC_DIR}")
    print("正在加载/生成 MP4 第一帧缓存...")
    pano_first_frames = build_pano_first_frame_cache(PANORAMIC_DIR, CACHE_DIR)
    mp4_names = list(pano_first_frames.keys())
    if not mp4_names:
        raise RuntimeError(f"未在 {PANORAMIC_DIR} 下找到可读的 MP4")
    if args.max_mp4s is not None and args.max_mp4s >= 0:
        mp4_names = mp4_names[: args.max_mp4s]
        print(f"[试跑] 仅使用前 {args.max_mp4s} 个 MP4: {len(mp4_names)} 个")
    print(f"有效 MP4 数量: {len(mp4_names)} 个")

    # 2) 按 pano_id 收集 PNG，每个 pano_id 只取第一张
    pano_to_pngs = collect_pngs_by_pano_id([
        (HOS_SFT_DIR, "hos"),
        (HPS_SFT_DIR, "hps"),
    ])
    if not pano_to_pngs:
        raise RuntimeError("未在 hos_sft / hps_sft 下找到符合命名规则的 PNG")
    if args.max_pano_ids is not None:
        # 现在就强制指定 'hps_1' 作为第一个 pano_id
        # pano_to_pngs = {k: v for k, v in pano_to_pngs.items() if k == 'hps_1'}
        pano_to_pngs = dict(list(pano_to_pngs.items())[: args.max_pano_ids])
        print(f"[试跑] 仅使用前 {args.max_pano_ids} 个 pano_id, pano_id: {list(pano_to_pngs.keys())}")
    pano_to_one_png = {pid: sorted(pngs)[0] for pid, pngs in pano_to_pngs.items()}
    print(f"有效 pano_id 数量: {len(pano_to_one_png)} 个（每 pano_id 取 1 张 PNG）")

    # 3) 构建任务：每个 (pano_id, 该 pano_id 唯一 png, mp4_name) 为一个 task
    tasks = [
        (pano_id, png_path, mp4_name)
        for pano_id, png_path in pano_to_one_png.items()
        for mp4_name in mp4_names
    ]
    print(f"当前匹配 pano_id 数量: {len(pano_to_one_png)} 个，构建任务: {len(tasks)} 个")

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

    # 4) 聚合为简化版输出
    result = aggregate_simple_results(results, pano_to_one_png)

    out_path = Path(args.output) if args.output else OUTPUT_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for v in result.values() if v["match_success"])
    print(f"结果已写入: {out_path}")
    print(f"匹配成功 pano_id 数: {n_ok} / {len(result)}")


if __name__ == "__main__":
    main()
