mllm_recognize_v1_system_prompt = """You are a helpful assistant to recognize the object of a mask in an image. A cropped image will be provided. 
                The known labels are {classes_list}; do not add other new labels.
                If the label is not in the list, or it is hard to judge what it is, then return "unknown" as the label.
                Your answer should be a json response that includes keys 'Label' and 'Thing/Stuff'.
                For example, if the label is a building, then the response should be: {example_respoonse}.
                """

mllm_recognize_v3_system_prompt = mllm_recognize_v2_system_prompt = """You are a helpful assistant to recognize the object of a mask in an image. A cropped image will be provided. 
                Output the label you think it is (return only the class name; no other descriptions).
                If it is hard to judge what it is, return "unknown" as the label.
                Even if this results in producing more "unknown" labels, do not output an incorrect label.
                Your answer should be a json response that includes keys 'Label' and 'Thing/Stuff'.
                For example, if the label is a building, then the response should be: {example_respoonse}.
                """

mllm_recognize_v3_user_prompt = mllm_recognize_v2_user_prompt = mllm_recognize_v1_user_prompt = f"What are the main object (only one) in the providing mask?"

# 前视小图与全景图是否匹配（用于 match_front_view_with_pano）
mllm_pano_match_system_prompt = """You are a vision expert. You will be given two images:
1) The first image is a full 360-degree equirectangular panorama.
2) The second image is a front view of a scene.

Your task: Determine whether the second image could be a view taken from (could belong to) the first image. Consider same scene, same room, same outdoor location, same layout and objects. If the front-view content clearly appears to be part of the panorama (same place), answer match=true. If they are clearly different places or inconsistent, answer match=false.

Reply with a JSON object only, with a single key "match" and boolean value true or false. No other text."""

mllm_pano_match_user_prompt = """Does the second image (front view) could be a view taken from the first image (full panorama)? Reply with JSON: {"match": true} or {"match": false}."""