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
from pano.class_tools import classes_list
from pano.omni_tools import OmniImage

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:25547/v1"


temperature = 0.2
seed = 42
lock = threading.Lock()
global openai_async_client
openai_async_client = None

omni_image = OmniImage(img_w=3840, img_h=1920)

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


def mllm_recognize(image: np.ndarray, mask: torch.Tensor | np.ndarray, classes_list: list, mask_id: int = 0, expand: int = 50):
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
                "text": f"What are the main object (only one) in the providing mask?"
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
                "text": f"What are the main object (only one) in the providing mask?"
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
            model="/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct",   # FIXME(zhouenshen): Qwen/Qwen3-VL-8B-Instruct
            prompt=prompt,
            system_prompt=f"""You are a helpful assistant to recognize the object of a mask in an image. A cropped image will be provided. 
                The known labels are {classes_list}; do not add other new labels.
                If the label is not in the list, or it is hard to judge what it is, then return "unknown" as the label.
                Your answer should be a json response that includes keys 'Label' and 'Thing/Stuff'.
                For example, if the label is a building, then the response should be: {example_respoonse}.
                """
        )
        object_infor = extract_response_info(message)
    except:
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
