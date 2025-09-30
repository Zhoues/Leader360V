
# Leader360V : A Large-scale, Real-world 360 Video Dataset for Multi-task Learning in Diverse Environments

![Leader360V](https://img.shields.io/badge/Dataset-Leader360V-blue)  ![Large_scale](https://img.shields.io/badge/Feature-Large_Scale-red)  ![Real_world](https://img.shields.io/badge/Feature-Real_World-red) 
![Multi_Task_Learning](https://img.shields.io/badge/Feature-Multi_Task_Learning-red)  ![Diverse_Environments](https://img.shields.io/badge/Feature-Diverse_Environments-red) 
![overview](assets/imgs/Teaser_Figure_1_00.png "overview")

## 📷 Quick Demos with Different Scenarios
More results and details can be found on our [📖 Project Homepage](https://leader360v.github.io/Leader360_Homepage_VUE/). 


<table class="center">
    <tr>
    <td><img src="assets/videos/Basement-Indoor.gif"></td>
    <td><img src="assets/videos/Grassland-Outdoor.gif"></td>
    <td><img src="assets/videos/Gym-Indoor.gif"></td>
    </tr>
</table>
<!-- <p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a><p> -->

<table>
    <tr>
    <td><img src="assets/videos/Nature-Outdoor.gif"></td>
    <td><img src="assets/videos/Road-Outdoor.gif"></td>
    <td><img src="assets/videos/SubwayStation-Indoor.gif"></td>
    </tr>
</table>

## 📷 Quick Demos of Annotation
More results and details can be found on our [📖 Project Homepage](https://leader360v.github.io/Leader360_Homepage_VUE/). 

<table class="center">
    <tr>
    <td>Raw Videos</td>
    <td>Pipeline Annotation</td>
    <td>Manual Annotation</td>
    </tr>
</table>

<table>
    <tr>
    <td><img src="assets/videos/anno1/raw1.gif"></td>
    <td><img src="assets/videos/anno1/pipe1.gif"></td>
    <td><img src="assets/videos/anno1/manual1.gif"></td>
    </tr>
</table>

<table>
    <tr>
    <td><img src="assets/videos/anno2/raw2.gif"></td>
    <td><img src="assets/videos/anno2/pipe2.gif"></td>
    <td><img src="assets/videos/anno2/manual2.gif"></td>
    </tr>
</table>

<table>
    <tr>
    <td><img src="assets/videos/anno3/raw3.gif"></td>
    <td><img src="assets/videos/anno3/pipe3.gif"></td>
    <td><img src="assets/videos/anno3/manual3.gif"></td>
    </tr>
</table>

🌟 For mare details, please refer to our project homepage: 
"[Leader360V : A Large-scale, Real-world 360 Video Dataset for Multi-task Learning in Diverse Environments](https://leader360v.github.io/Leader360V_HomePage)".

[[🍓Project Homepage](https://leader360v.github.io/Leader360V_HomePage)]

[📖 Explore More](https://leader360v.github.io/Leader360_Homepage_VUE/)]

[[📊 Huggingface Dataset](https://huggingface.co/datasets/Leader360V/Leader360V)]

## 📷 Usage of A$^3$360V Auto-Annotation Framework
### Environment
```
python == 3.10.16
torch == 2.6.0+cu124
torchvision == 0.21.0
pycocotools == 2.0.8
numpy == 2.2.5
opencv-python == 4.11.0.86
sam2
```

### Preparation

Install [CropFormer](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md) with detectron2 under current directory.
```
Leader360V
|----detectron2
|----|---...
|----|---projects
|----|---|---CropFormer
```
To prevent module name duplicate error in detectron2 inference pipeline construction: 
1. Modify the value of field `MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME` from `MSDeformAttnPixelDecoder` to `Mask2FormerMSDeformAttnPixelDecoder` in `detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/*.yaml`, except `Base-Mask2Former.yaml`.
2. Modify the value of field `MODEL.BACKBONE.NAME` from `D2SwinTransformer` to `Mask2FormerD2SwinTransformer` in `detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_swin_*.yaml`.
3. Modify the class name `MSDeformAttnPixelDecoder` to `Mask2FormerMSDeformAttnPixelDecoder` in `detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/msdeformattn.py`.
4. Modify the class name `D2SwinTransformer` to `Mask2FormerD2SwinTransformer` in `detectron2/projects/CropFormer/mask2former/modeling/backbone/swin.py`.

### Pretrained Weights
Download pretrained weights in [CropFormer](https://github.com/qqlu/Entity/tree/main/Entityv2), [OneFormer](https://github.com/SHI-Labs/OneFormer), and [SAM2](https://github.com/facebookresearch/sam2) and put them in `pretrained_models` directory:
```
pretrained_models
|----detectron2
|----|---CropFormer
|----|---|---CropFormer_hornet_3x_03823a.pth
|----|---|---...
|----|---OneFormer
|----|---|---ADE20K
|----|---|---|---250_16_convnext_xl_oneformer_ade20k_160k.pth
|----|---|---|---...
|----|---|---Cityscapes
|----|---|---|---250_16_dinat_l_oneformer_cityscapes_90k.pth
|----|---|---|---...
|----|---|---COCO
|----|---|---|---150_16_dinat_l_oneformer_coco_100ep.pth
|----|---|---|---...
|----|---SAM2
|----|---|---sam2.1_hiera_base_plus.pt
|----|---|---...
```

### Inference
You can modify the model components and set up other parameters in `config.yaml` file.
Setup the MLLM in `mllm.py` file.
Then, setup the 360 video (2048 * 1024) path and output dir.
```bash
python3 video_segmentor.py
```


## About Leader360V

- Leader360V is the first large-scale (10K+), labeled real-world 360 video datasets for instance segmentation and tracking. Our datasets enjoy high scene diversity, ranging from indoor and urban settings to natural and dynamic outdoor scenes.

- All videos in this dataset have undergone standardized preprocessing, including video clipping, facial anonymization for privacy protection, and balanced scene distribution across categories.

- Regarding dataset composition, we integrated existing 360 video datasets (either unlabeled or annotated for single tasks) and supplemented them with newly collected self-recorded videos. All content was then re-annotated to support joint segmentation and tracking tasks.

- **Due to the large size of the dataset, we have currently uploaded only a selection of demos. We plan to upload all files at a later date to facilitate community research.** 


