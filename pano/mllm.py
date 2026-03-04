import base64
import json
import torch
from openai import OpenAI
import cv2
import numpy as np
from openai import APIConnectionError, RateLimitError, InternalServerError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os
import threading
import concurrent.futures

global classes_list
from pano.prompt import *
from pano.class_tools import classes_list
from pano.omni_tools import OmniImage

temperature = 0.2
seed = 42
lock = threading.Lock()
global openai_async_client
openai_async_client = None

omni_image = OmniImage(img_w=5760, img_h=2880)

example_respoonse = json.dumps(
    {"Label": "building", "Thing/Stuff": "stuff"},
    ensure_ascii=False
)


def get_openai_async_client():
    global openai_async_client
    if openai_async_client is not None:
        return openai_async_client
    with lock:
        openai_async_client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        return openai_async_client


def encode_image(image_numpy):
    _, buffer = cv2.imencode('.jpg', image_numpy)
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode('utf-8')


@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=3, max=8),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
)
def gpt_4o_complete_if_cache(model, prompt, system_prompt=None, history_messages=[]) -> str:
    openai_async_client = get_openai_async_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_async_client.chat.completions.create(
        model=model, messages=messages, max_tokens=2048,
        temperature=temperature,
        seed=seed,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def gpt_4o_complete(model, prompt, system_prompt=None, history_messages=[]) -> str:
    return gpt_4o_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
    )


def extract_response_info(response: str):
    object_infor = json.loads(response)
    return object_infor


def mllm_recognize_v1(image: np.ndarray, mask: torch.Tensor | np.ndarray, classes_list: list, mask_id: int = 0, model: str = "/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct", expand: int = 50):
    if type(mask) is torch.Tensor:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    h, w, _ = image.shape
    try:
        bbox = omni_image.mask2Bbox(mask_np, need_rotation=False, expand=expand)
        crop_image = omni_image.crop_bbox(image, bbox)[0]
        image_masked = image * mask_np[:, :, np.newaxis]
        object_crop_image = omni_image.crop_bbox(image_masked, bbox)[0]
        encoded_crop_image = encode_image(crop_image)
        encoded_object_crop_image = encode_image(object_crop_image)
    except:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    if mask_np.sum() > 5e5:
        prompt = [
            {
                "type": "text",
                "text": mllm_recognize_v1_user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_object_crop_image}"
                }
            }
        ]
    else:
        prompt = [
            {
                "type": "text",
                "text": mllm_recognize_v1_user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_object_crop_image}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_crop_image}"
                }
            }
        ]
    try:
        message = gpt_4o_complete(
            model=model,
            prompt=prompt,
            system_prompt=mllm_recognize_v1_system_prompt.format(classes_list=classes_list, example_respoonse=example_respoonse)
        )
        object_infor = extract_response_info(message)
    except:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    return mask_id, object_infor


def mllm_recognize_v2(image: np.ndarray, mask: torch.Tensor | np.ndarray, classes_list: list, mask_id: int = 0, model: str = "Qwen/Qwen3-VL-8B-Instruct", expand: int = 50):
    if type(mask) is torch.Tensor:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    h, w, _ = image.shape
    try:
        bbox = omni_image.mask2Bbox(mask_np, need_rotation=False, expand=expand)
        crop_image = omni_image.crop_bbox(image, bbox)[0]
        image_masked = image * mask_np[:, :, np.newaxis]
        object_crop_image = omni_image.crop_bbox(image_masked, bbox)[0]
        encoded_crop_image = encode_image(crop_image)
        encoded_object_crop_image = encode_image(object_crop_image)
    except:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing", "Detail": "unknown"}
    if mask_np.sum() > 5e5:
        prompt = [
            {
                "type": "text",
                "text": mllm_recognize_v2_user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_object_crop_image}"
                }
            }
        ]
    else:
        prompt = [
            {
                "type": "text",
                "text": mllm_recognize_v2_user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_object_crop_image}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_crop_image}"
                }
            }
        ]
    try:
        message = gpt_4o_complete(
            model=model,
            prompt=prompt,
            system_prompt=mllm_recognize_v2_system_prompt.format(example_respoonse=example_respoonse)
        )
        object_infor = extract_response_info(message)
    except:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    return mask_id, object_infor


def mllm_recognize_v3(image: np.ndarray, mask: torch.Tensor | np.ndarray, classes_list: list, mask_id: int = 0, model: str = "Qwen/Qwen3-VL-8B-Instruct", view_width: int = 640, view_height: int = 480, fov_h: int = 100, expand: int = 50):
    """
    使用「物体居中 + 2D 前向透视」流程的识别：
    1. 将 mask 最大连通域中心转到全景正中心
    2. 投影到 2D 前向透视图 (默认 640x480, FOV 100°)
    3. 在该 2D 视角下取 bbox/mask 并裁剪；若 bbox 超出视窗则裁到视窗内
    """
    if type(mask) is torch.Tensor:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    try:
        result = omni_image.get_front_view_crop_centered_on_mask(
            image,
            mask_np,
            view_width=view_width,
            view_height=view_height,
            fov_h=fov_h,
            expand=expand,
        )
    except Exception:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    if result is None:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}

    crop_image = result["crop_image"]
    object_crop_image = result["object_crop_image"]
    # 大 mask 时用内接 bbox 的 object_crop_image，避免送整块大图
    object_crop_image_for_large = (
        result["object_crop_image_inner"]
        if result.get("object_crop_image_inner") is not None
        else object_crop_image
    )
    encoded_crop_image = encode_image(crop_image)
    encoded_object_crop_image = encode_image(object_crop_image)
    encoded_object_crop_image_inner = (
        encode_image(object_crop_image_for_large) if object_crop_image_for_large is not None else encoded_object_crop_image
    )

    if mask_np.sum()/(mask_np.shape[0]*mask_np.shape[1]) < 1e3/(3840*1920):
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    elif mask_np.sum()/(mask_np.shape[0]*mask_np.shape[1]) > 5e5/(3840*1920):
        prompt = [
            {"type": "text", "text": mllm_recognize_v3_user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_object_crop_image_inner}"},
            },
        ]
    else:
        prompt = [
            {"type": "text", "text": mllm_recognize_v3_user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_object_crop_image}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_crop_image}"},
            },
        ]
    
    try:
        message = gpt_4o_complete(
            model=model,
            prompt=prompt,
            system_prompt=mllm_recognize_v3_system_prompt.format(example_respoonse=example_respoonse),
        )
        object_infor = extract_response_info(message)
    except Exception:
        return None, {"Label": "unknown", "Thing/Stuff": "Thing"}
    return mask_id, object_infor


def mllm_judge_new_object(last_frame: np.ndarray, cur_frame: np.ndarray, mask: torch.Tensor | np.ndarray, mask_id: int = 0, last_expand: int = 200):
    if type(mask) is torch.Tensor:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    h, w, _ = cur_frame.shape
    bbox = omni_image.mask2Bbox(mask_np, need_rotation=False, expand=last_expand)
    crop_last_image = omni_image.crop_bbox(last_frame, bbox)[0]
    image_masked = cur_frame * mask_np[:, :, np.newaxis]
    object_crop_image = omni_image.crop_bbox(image_masked, bbox)[0]
    try:
        encoded_crop_image = encode_image(crop_last_image)
        encoded_object_crop_image = encode_image(object_crop_image)
    except:
        return False, mask_id

    encoded_crop_image = encode_image(encoded_crop_image)
    encoded_object = encode_image(encoded_object_crop_image)
    message = gpt_4o_complete(
        model="gpt-4o",
        prompt=[
            {
                "type": "text",
                "text": f"Does this object appear in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_object}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_crop_image}"
                }
            }
        ],
        system_prompt=f"""You are a helpful assistant in image vision.
        You will be provided an object and an image.
        Please tell me if the object has appeared in this image. 
        Only say yes or no, only one word, no other words.
        """
    )
    return "yes" in message.lower(), mask_id


def llm_judge_new_object(curr_frame: np.ndarray, last_frame: np.ndarray, new_masks: torch.Tensor,
                         match_idx_list: [], match_sam2_masks_idx_list: [], worker_num: int = 2):
    llm_refined_idx_list = []
    llm_refined_sam2_masks_idx_list = []
    unmatched_idx_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = [
            executor.submit(mllm_judge_new_object, last_frame, curr_frame,
                            new_masks[match_idx], match_idx)
            for match_idx in match_idx_list
        ]

        for future in concurrent.futures.as_completed(futures):
            if_new_object, match_idx = future.result()
            if if_new_object:
                id_id = match_idx_list.index(match_idx)
                llm_refined_idx_list.append(match_idx)
                llm_refined_sam2_masks_idx_list.append(match_sam2_masks_idx_list[id_id])
            else:
                unmatched_idx_list.append(match_idx)
    return llm_refined_idx_list, llm_refined_sam2_masks_idx_list, unmatched_idx_list
