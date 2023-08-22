# Open Pre-Trained Models [![MLIR And NNTC Test](https://github.com/sophgo/model-zoo/actions/workflows/mlir-nntc-test.yml/badge.svg?branch=main&event=schedule)](https://github.com/sophgo/model-zoo/actions/workflows/mlir-nntc-test.yml)

test[ashdiashdfjiosadfg.com/asbdfguyawegf.html]
## Usage - Compile and run
test
test
test
Install [tpu-perf](https://github.com/sophgo/tpu-perf) to build and run model cases.

```bash
# Time only cases
python3 -m tpu_perf.build --list default_cases.txt --time
python3 -m tpu_perf.run --list default_cases.txt

# Precision benchmark
python3 -m tpu_perf.build --list default_cases.txt
python3 -m tpu_perf.precision_benchmark --list default_cases.txt
```

## Usage - Git LFS

On default, cloning this repository will not download any models. Install
Git LFS with `pip install git-lfs`.

To download a specific model:
`git lfs pull --include="path/to/model" --exclude=""`

To download all models:
`git lfs pull --include="*" --exclude=""`

## Usage - Model visualization

You can see visualizations of each model's network architecture by using [Netron](https://github.com/lutzroeder/Netron).

## How to contribute

Please lint in your local repo before PR.

```bash
# Install tools
sudo npm install -g markdownlint-cli
pip3 install yamllint

yamllint -c ./.yaml-lint.yml .
markdownlint '**/*.md'
python3 .github/workflows/check.py
```

## Model Navigation

### Vision

|Model                            |Path                                                                                                  |NNTC                |MLIR                |
|:-                               |:-                                                                                                    |:-                  |:-                  |
|adaface                          |[vision/recognition/adaface](vision/recognition/adaface)                                              |                    |:white\_check\_mark:|
|alexnet                          |[vision/classification/AlexNet-Caffe](vision/classification/AlexNet-Caffe)                            |:white\_check\_mark:|                    |
|arcface                          |[vision/recognition/arcface](vision/recognition/arcface)                                              |                    |:white\_check\_mark:|
|basicvsr_backward                |[vision/super-resolution/basicvsr](vision/super-resolution/basicvsr)                                  |                    |:white\_check\_mark:|
|basicvsr_forward                 |[vision/super-resolution/basicvsr](vision/super-resolution/basicvsr)                                  |                    |:white\_check\_mark:|
|basicvsr_spynet                  |[vision/super-resolution/basicvsr](vision/super-resolution/basicvsr)                                  |                    |:white\_check\_mark:|
|basicvsr_upsample                |[vision/super-resolution/basicvsr](vision/super-resolution/basicvsr)                                  |                    |:white\_check\_mark:|
|big\_transfer                    |[vision/classification/big\_transfer\_mlir](vision/classification/big_transfer_mlir)                  |                    |:white\_check\_mark:|
|BiSeNet                          |[vision/segmentation/BiSeNet](vision/segmentation/BiSeNet)                                            |                    |:white\_check\_mark:|
|c3d                              |[vision/video-recognition/C3D](vision/video-recognition/C3D)                                          |:white\_check\_mark:|:white\_check\_mark:|
|CenterNet                        |[vision/detection/CenterNet-Torch](vision/detection/CenterNet-Torch)                                  |:white\_check\_mark:|                    |
|centernet\_onnx                  |[vision/detection/CenterNet-Onnx](vision/detection/CenterNet-Onnx)                                    |                    |:white\_check\_mark:|
|clip                             |[vision/classification/clip-mlir](vision/classification/clip-mlir)                                    |                    |:white\_check\_mark:|
|CRNN                             |[vision/OCR/CRNN](vision/OCR/CRNN)                                                                    |:white\_check\_mark:|                    |
|cyclegan\_horse2zebra            |[vision/GAN/cyclegan](vision/GAN/cyclegan)                                                            |:white\_check\_mark:|:white\_check\_mark:|
|dbnet                            |[vision/detection/dbnet](vision/detection/dbnet)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|DBNet\_totaltext\_res18\_dcn     |[vision/OCR/DBNet](vision/OCR/DBNet)                                                                  |:white\_check\_mark:|:white\_check\_mark:|
|DBNet\_totaltext\_res50\_dcn     |[vision/OCR/DBNet](vision/OCR/DBNet)                                                                  |:white\_check\_mark:|:white\_check\_mark:|
|deeplabv3p                       |[vision/segmentation/deeplabv3p](vision/segmentation/deeplabv3p)                                      |:white\_check\_mark:|:white\_check\_mark:|
|densenet                         |[vision/classification/DenseNet-Caffe](vision/classification/DenseNet-Caffe)                          |:white\_check\_mark:|                    |
|dpn68                            |[vision/classification/dpn68](vision/classification/dpn68)                                            |:white\_check\_mark:|:white\_check\_mark:|
|ECANet                           |[vision/classification/ECANet-Torch](vision/classification/ECANet-Torch)                              |:white\_check\_mark:|                    |
|efficientnet-B0                  |[vision/classification/EfficientNet-B0](vision/classification/EfficientNet-B0)                        |:white\_check\_mark:|                    |
|efficientnet-B1                  |[vision/classification/EfficientNet-B1](vision/classification/EfficientNet-B1)                        |:white\_check\_mark:|                    |
|efficientnet-B4                  |[vision/classification/EfficientNet-B4](vision/classification/EfficientNet-B4)                        |:white\_check\_mark:|                    |
|efficientnet-B5                  |[vision/classification/EfficientNet-B5](vision/classification/EfficientNet-B5)                        |:white\_check\_mark:|                    |
|efficientnet-B7                  |[vision/classification/EfficientNet-B7](vision/classification/EfficientNet-B7)                        |:white\_check\_mark:|                    |
|efficientnet-lite4               |[vision/classification/efficientnet-lite4](vision/classification/efficientnet-lite4)                  |                    |:white\_check\_mark:|
|efficientnetv2                   |[vision/classification/efficientnetv2](vision/classification/efficientnetv2)                          |:white\_check\_mark:|:white\_check\_mark:|
|ERFNet                           |[vision/segmentation/ERFNet-Caffe](vision/segmentation/ERFNet-Caffe)                                  |:white\_check\_mark:|                    |
|face\_alignment                  |[vision/recognition/face\_alignment](vision/recognition/face_alignment)                               |                    |:white\_check\_mark:|
|facenet                          |[vision/recognition/facenet](vision/recognition/facenet)                                              |:white\_check\_mark:|                    |
|GOTURN                           |[vision/tracking/GOTURN-Caffe](vision/tracking/GOTURN-Caffe)                                          |:white\_check\_mark:|                    |
|hrnet                            |[vision/classification/HRNet-Onnx](vision/classification/HRNet-Onnx)                                  |                    |:white\_check\_mark:|
|HRNet                            |[vision/classification/HRNet-Torch](vision/classification/HRNet-Torch)                                |:white\_check\_mark:|                    |
|huggingface-resnet50             |[vision/classification/resnet50-v1.5](vision/classification/resnet50-v1.5)                            |:white\_check\_mark:|                    |
|inception\_resnet\_v2            |[vision/classification/inception\_resnet\_v2](vision/classification/inception_resnet_v2)              |:white\_check\_mark:|                    |
|inception\_v1                    |[vision/classification/inception\_v1](vision/classification/inception_v1)                             |:white\_check\_mark:|:white\_check\_mark:|
|inception\_v3                    |[vision/classification/inception\_v3](vision/classification/inception_v3)                             |:white\_check\_mark:|                    |
|inception\_v4                    |[vision/classification/inception\_v4](vision/classification/inception_v4)                             |:white\_check\_mark:|                    |
|inception\_v4-tflite             |[vision/classification/inception\_v4](vision/classification/inception_v4)                             |                    |:white\_check\_mark:|
|inceptionBN21k                   |[vision/classification/InceptionBN-21k-for-Caffe](vision/classification/InceptionBN-21k-for-Caffe)    |:white\_check\_mark:|                    |
|kapao                            |[vision/pose-estimation/kapao](vision/pose-estimation/kapao)                                          |                    |:white\_check\_mark:|
|lenet                            |[vision/classification/LeNet](vision/classification/LeNet)                                            |:white\_check\_mark:|                    |
|lprnet                           |[vision/detection/lprnet](vision/detection/lprnet)                                                    |                    |:white\_check\_mark:|
|market\_bot\_R50                 |[vision/reid/market\_bot\_R50](vision/reid/market_bot_R50)                                            |:white\_check\_mark:|:white\_check\_mark:|
|MDNet                            |[vision/tracking/MDNet-Torch](vision/tracking/MDNet-Torch)                                            |:white\_check\_mark:|:white\_check\_mark:|
|mm\_resnet50                     |[vision/classification/mm\_resnet50](vision/classification/mm_resnet50)                               |:white\_check\_mark:|:white\_check\_mark:|
|mobilenet-v2                     |[vision/classification/mobilenet-v2](vision/classification/mobilenet-v2)                              |                    |:white\_check\_mark:|
|mobilenet\_v1                    |[vision/classification/MobileNet-Caffe-v1](vision/classification/MobileNet-Caffe-v1)                  |:white\_check\_mark:|                    |
|mobilenetv2                      |[vision/classification/MobileNet-Caffe-v2](vision/classification/MobileNet-Caffe-v2)                  |:white\_check\_mark:|:white\_check\_mark:|
|mobilenetv3                      |[vision/classification/MobileNet-Caffe-v3](vision/classification/MobileNet-Caffe-v3)                  |:white\_check\_mark:|                    |
|mobileseg                        |[vision/segmentation/mobileseg](vision/segmentation/mobileseg)                                        |:white\_check\_mark:|                    |
|mobileseg-mlir                   |[vision/segmentation/mobileseg](vision/segmentation/mobileseg)                                        |                    |:white\_check\_mark:|
|mtcnn\_onet                      |[vision/detection/mtcnn](vision/detection/mtcnn)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|mtcnn\_pnet                      |[vision/detection/mtcnn](vision/detection/mtcnn)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|mtcnn\_rnet                      |[vision/detection/mtcnn](vision/detection/mtcnn)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|openpose                         |[vision/pose-estimation/openpose](vision/pose-estimation/openpose)                                    |:white\_check\_mark:|:white\_check\_mark:|
|pointpillars                     |[vision/detection/pointpillars](vision/detection/pointpillars/)                                       |                    |:white\_check\_mark:|
|PP-OCRv3\_det                    |[vision/OCR/OCRv3\_paddle](vision/OCR/OCRv3_paddle)                                                   |                    |:white\_check\_mark:|
|PP-OCRv3\_rec                    |[vision/OCR/OCRv3\_paddle](vision/OCR/OCRv3_paddle)                                                   |                    |:white\_check\_mark:|
|PP-OCRv3cls                      |[vision/OCR/PP-OCRv3cls](vision/OCR/PP-OCRv3cls)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|PP-OCRv3det                      |[vision/OCR/PP-OCRv3det](vision/OCR/PP-OCRv3det)                                                      |:white\_check\_mark:|:white\_check\_mark:|
|PP-OCRv3rec                      |[vision/OCR/PP-OCRv3rec](vision/OCR/PP-OCRv3rec)                                                      |                    |:white\_check\_mark:|
|pp\_humansegv1\_lite\_argmax     |[vision/segmentation/pp\_humansegv1\_lite](vision/segmentation/pp_humansegv1_lite)                    |:white\_check\_mark:|                    |
|pp\_humansegv1\_lite\_softmax    |[vision/segmentation/pp\_humansegv1\_lite](vision/segmentation/pp_humansegv1_lite)                    |:white\_check\_mark:|                    |
|pp\_humansegv2\_mobile           |[vision/segmentation/pp_humansegv2_mobile](vision/segmentation/pp_humansegv2_mobile)                  |                    |:white\_check\_mark:|
|pp\_liteseg                      |[vision/segmentation/pp\_liteseg](vision/segmentation/pp_liteseg)                                     |:white\_check\_mark:|:white\_check\_mark:|
|pp\_picodet\_s                   |[vision/detection/pp-picodet](vision/detection/pp-picodet)                                            |                    |:white\_check\_mark:|
|ppocr\_mobile\_v2.0\_cls         |[vision/OCR/OCRv3\_paddle](vision/OCR/OCRv3_paddle)                                                   |                    |:white\_check\_mark:|
|ppyoloe\_crn\_s\_300e\_coco      |[vision/detection/ppyoloe](vision/detection/ppyoloe)                                                  |                    |:white\_check\_mark:|
|ppyoloe\_crn\_x\_300e\_coco      |[vision/detection/ppyoloe](vision/detection/ppyoloe)                                                  |:white\_check\_mark:|                    |
|ppyoloe\_plus\_crn\_x\_80e\_coco |[vision/detection/ppyoloe](vision/detection/ppyoloe)                                                  |:white\_check\_mark:|                    |
|ppyolov2\_r101vd\_dcn\_365e\_coco|[vision/detection/ppyolo](vision/detection/ppyolo)                                                    |:white\_check\_mark:|                    |
|py-R-FCN                         |[vision/detection/py-R-FCN](vision/detection/py-R-FCN)                                                |:white\_check\_mark:|                    |
|res2net101\_26w\_4s              |[vision/classification/res2net101\_26w\_4s](vision/classification/res2net101_26w_4s)                  |:white\_check\_mark:|:white\_check\_mark:|
|res2net50\_26w\_4s               |[vision/classification/res2net50\_26w\_4s](vision/classification/res2net50_26w_4s)                    |:white\_check\_mark:|:white\_check\_mark:|
|resnet101-v1-7                   |[vision/classification/resnet101-v1-7](vision/classification/resnet101-v1-7)                          |:white\_check\_mark:|:white\_check\_mark:|
|resnet152-v1-7                   |[vision/classification/resnet152-v1-7](vision/classification/resnet152-v1-7)                          |:white\_check\_mark:|:white\_check\_mark:|
|resnet18-v1-7                    |[vision/classification/resnet18-v1-7](vision/classification/resnet18-v1-7)                            |:white\_check\_mark:|:white\_check\_mark:|
|resnet18-v2                      |[vision/classification/resnet18-v2](vision/classification/resnet18-v2)                                |                    |:white\_check\_mark:|
|resnet34                         |[vision/classification/ResNet34](vision/classification/ResNet34)                                      |:white\_check\_mark:|                    |
|resnet34-v1-7                    |[vision/classification/resnet34-v1-7](vision/classification/resnet34-v1-7)                            |:white\_check\_mark:|:white\_check\_mark:|
|resnet50-caffe                   |[vision/classification/ResNet50-Caffe](vision/classification/ResNet50-Caffe)                          |:white\_check\_mark:|                    |
|resnet50-v1-7                    |[vision/classification/resnet50-v1-7](vision/classification/resnet50-v1-7)                            |:white\_check\_mark:|:white\_check\_mark:|
|resnet50-v2                      |[vision/classification/resnet50-v2](vision/classification/resnet50-v2)                                |                    |:white\_check\_mark:|
|ResNet50\_vd\_infer              |[vision/classification/ResNet50\_vd\_paddle](vision/classification/ResNet50_vd_paddle)                |                    |:white\_check\_mark:|
|resnext                          |[vision/classification/ResNeXt](vision/classification/ResNeXt)                                        |:white\_check\_mark:|                    |
|resneXt50                        |[vision/classification/ResNeXt50](vision/classification/ResNeXt50)                                    |:white\_check\_mark:|                    |
|resneXt50-v1                     |[vision/classification/ResNeXt50-v1](vision/classification/ResNeXt50-v1)                              |                    |:white\_check\_mark:|
|resneXt50-v2                     |[vision/classification/ResNeXt50-v2](vision/classification/ResNeXt50-v2)                              |                    |:white\_check\_mark:|
|retinaface                       |[vision/detection/retinaface](vision/detection/retinaface)                                            |:white\_check\_mark:|:white\_check\_mark:|
|S-DCNet_SHA_fusepost             |[vision/visual-counting/S-DCNet](vision/visual-counting/S-DCNet)                                      |                    |:white\_check\_mark:|
|S-DCNet_SHB_fusepost             |[vision/visual-counting/S-DCNet](vision/visual-counting/S-DCNet)                                      |                    |:white\_check\_mark:|
|sam-vit-base                     |[vision/segmentation/sam-vit-base](vision/segmentation/sam-vit-base)                                  |                    |:white\_check\_mark:|
|scrfd                            |[vision/detection/scrfd](vision/detection/scrfd)                                                      |                    |:white\_check\_mark:|
|SegFormer                        |[vision/segmentation/SegFormer](vision/segmentation/SegFormer)                                        |                    |:white\_check\_mark:|
|shufflenet\_v2                   |[vision/classification/shufflenet\_v2](vision/classification/shufflenet_v2)                           |                    |:white\_check\_mark:|
|shufflenetv2                     |[vision/classification/shufflenet\_v2\_torch](vision/classification/shufflenet_v2_torch)              |:white\_check\_mark:|                    |
|SiamMask                         |[vision/tracking/SiamMask-Torch](vision/tracking/SiamMask-Torch)                                      |:white\_check\_mark:|:white\_check\_mark:|
|squeezenet                       |[vision/classification/SqueezeNet](vision/classification/SqueezeNet)                                  |:white\_check\_mark:|                    |
|squeezenet1.0                    |[vision/classification/squeezenet1.0](vision/classification/squeezenet1.0)                            |                    |:white\_check\_mark:|
|SRCNN                            |[vision/super-resolution/SRCNN](vision/super-resolution/SRCNN)                                        |:white\_check\_mark:|                    |
|ssd-mobilenet-tflite             |[vision/detection/ssd-mobilenet](vision/detection/ssd-mobilenet)                                      |                    |:white\_check\_mark:|
|StableDiffusion\_decode          |[vision/generative/stable\_diffusion](vision/generative/stable_diffusion)                             |                    |:white\_check\_mark:|
|StableDiffusion\_text\_encoder   |[vision/generative/stable\_diffusion](vision/generative/stable_diffusion)                             |                    |:white\_check\_mark:|
|StableDiffusion\_unet            |[vision/generative/stable\_diffusion](vision/generative/stable_diffusion)                             |                    |:white\_check\_mark:|
|swin\_b                          |[vision/classification/SwinTransformer](vision/classification/SwinTransformer)                        |                    |:white\_check\_mark:|
|swin\_s                          |[vision/classification/SwinTransformer](vision/classification/SwinTransformer)                        |                    |:white\_check\_mark:|
|swin\_t                          |[vision/classification/SwinTransformer](vision/classification/SwinTransformer)                        |:white\_check\_mark:|:white\_check\_mark:|
|tpu-mlir\_S-DCNet\_SHA           |[vision/visual-counting/S-DCNet](vision/visual-counting/S-DCNet)                                      |                    |:white\_check\_mark:|
|tpu-mlir\_S-DCNet\_SHB           |[vision/visual-counting/S-DCNet](vision/visual-counting/S-DCNet)                                      |                    |:white\_check\_mark:|
|tpu-mlir\_yoloface               |[vision/detection/yoloface](vision/detection/yoloface)                                                |                    |:white\_check\_mark:|
|tsm                              |[vision/recognition/tsm](vision/recognition/tsm)                                                      |                    |:white\_check\_mark:|
|TSN                              |[vision/video-recognition/TSN](vision/video-recognition/TSN/)                                         |                    |:white\_check\_mark:|
|ultraface                        |[vision/detection/ultraface](vision/detection/ultraface)                                              |                    |:white\_check\_mark:|
|ultralytics\_yolov3              |[vision/detection/ultralytics-yolov3](vision/detection/ultralytics-yolov3)                            |:white\_check\_mark:|                    |
|unet\_plusplus                   |[vision/segmentation/unet\_plusplus](vision/segmentation/unet_plusplus)                               |:white\_check\_mark:|:white\_check\_mark:|
|VDSR                             |[vision/super-resolution/VDSR](vision/super-resolution/VDSR)                                          |:white\_check\_mark:|                    |
|vgg11                            |[vision/classification/vgg11-torch](vision/classification/vgg11-torch)                                |:white\_check\_mark:|                    |
|vgg16                            |[vision/classification/vgg16](vision/classification/vgg16)                                            |                    |:white\_check\_mark:|
|vgg19                            |[vision/classification/vgg19](vision/classification/vgg19)                                            |:white\_check\_mark:|                    |
|vggssd\_300                      |[vision/detection/vggssd\_300](vision/detection/vggssd_300)                                           |:white\_check\_mark:|                    |
|vision\_OCR\_CRNN\_tpu-mlir      |[vision/OCR/CRNN](vision/OCR/CRNN)                                                                    |                    |:white\_check\_mark:|
|vit-base-patch16-384             |[vision/classification/vit_B_16](vision/classification/vit_B_16)                                      |:white\_check\_mark:|                    |
|vit-large-patch16-384            |[vision/classification/vit_L_16](vision/classification/vit_L_16)                                      |:white\_check\_mark:|                    |
|WRN-50-2                         |[vision/classification/WRN-50-2](vision/classification/WRN-50-2)                                      |:white\_check\_mark:|:white\_check\_mark:|
|wrn50                            |[vision/classification/wrn50](vision/classification/wrn50)                                            |:white\_check\_mark:|:white\_check\_mark:|
|xception                         |[vision/classification/xception](vision/classification/xception)                                      |:white\_check\_mark:|                    |
|Yet-Another-EfficientDet-Pytorch |[vision/detection/Yet-Another-EfficientDet-Pytorch](vision/detection/Yet-Another-EfficientDet-Pytorch)|:white\_check\_mark:|                    |
|yolov3                           |[vision/detection/yolov3-torch](vision/detection/yolov3-torch)                                        |:white\_check\_mark:|                    |
|yolov3\_320                      |[vision/detection/yolov3\_320](vision/detection/yolov3_320)                                           |:white\_check\_mark:|                    |
|yolov3\_608                      |[vision/detection/yolov3\_608](vision/detection/yolov3_608)                                           |:white\_check\_mark:|                    |
|yolov3\_mobilenet\_v3\_270e\_coco|[vision/detection/ppyolov3](vision/detection/ppyolov3)                                                |                    |:white\_check\_mark:|
|yolov3\_spp                      |[vision/detection/yolov3\_spp](vision/detection/yolov3_spp)                                           |:white\_check\_mark:|                    |
|yolov3\_tiny                     |[vision/detection/yolov3\_tiny](vision/detection/yolov3_tiny)                                         |:white\_check\_mark:|                    |
|yolov5l                          |[vision/detection/yolov5](vision/detection/yolov5)                                                    |                    |:white\_check\_mark:|
|yolov5l-5                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|                    |
|yolov5m                          |[vision/detection/yolov5](vision/detection/yolov5)                                                    |                    |:white\_check\_mark:|
|yolov5m-5                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|                    |
|yolov5m-6                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|                    |
|yolov5s                          |[vision/detection/yolov5](vision/detection/yolov5)                                                    |                    |:white\_check\_mark:|
|yolov5s-4                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|                    |
|yolov5s-5                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|:white\_check\_mark:|
|yolov5s-tflite                   |[vision/detection/yolov5](vision/detection/yolov5)                                                    |                    |:white\_check\_mark:|
|yolov5x                          |[vision/detection/yolov5](vision/detection/yolov5)                                                    |                    |:white\_check\_mark:|
|yolov5x-5                        |[vision/detection/yolov5](vision/detection/yolov5)                                                    |:white\_check\_mark:|                    |
|yolov7                           |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7-d6                        |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7-e6                        |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7-e6e                       |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7-tiny                      |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7-w6                        |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov7x                          |[vision/detection/yolov7](vision/detection/yolov7)                                                    |                    |:white\_check\_mark:|
|yolov8l                          |[vision/detection/yolov8](vision/detection/yolov8)                                                    |                    |:white\_check\_mark:|
|yolov8m                          |[vision/detection/yolov8](vision/detection/yolov8)                                                    |                    |:white\_check\_mark:|
|yolov8n                          |[vision/detection/yolov8](vision/detection/yolov8)                                                    |                    |:white\_check\_mark:|
|yolov8s                          |[vision/detection/yolov8](vision/detection/yolov8)                                                    |                    |:white\_check\_mark:|
|yolov8x                          |[vision/detection/yolov8](vision/detection/yolov8)                                                    |                    |:white\_check\_mark:|
|yolox\_s\_300e\_coco             |[vision/detection/ppyolox](vision/detection/ppyolox)                                                  |                    |:white\_check\_mark:|

### Language

|Model                             |Path                                                                          |NNTC                |MLIR                |
|:-                                |:-                                                                            |:-                  |:-                  |
|bert                              |[language/nlp/bert](language/nlp/bert)                                        |:white\_check\_mark:|:white\_check\_mark:|
|bert\_base\_squad\_uncased-2.11.0 |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|                    |:white\_check\_mark:|
|bert\_base\_squad\_uncased-4.23.0 |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|                    |:white\_check\_mark:|
|bert\_base\_transformers-2.11.0   |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|:white\_check\_mark:|                    |
|bert\_base\_transformers-4.23.0   |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|:white\_check\_mark:|                    |
|bert\_large\_transformers-2.11.0  |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|:white\_check\_mark:|                    |
|bert\_large\_transformers-4.23.0  |[language/nlp/Huggingface_bert_squadv1](language/nlp/Huggingface_bert_squadv1)|:white\_check\_mark:|                    |
|bert\_paddle                      |[language/nlp/bert\_paddle](language/nlp/bert_paddle)                         |:white\_check\_mark:|                    |
|ConformerEncoder                  |[language/asr/conformer](language/asr/conformer)                              |:white\_check\_mark:|                    |
|ernie-3.0-medium-zh               |[language/nlp/ernie-3.0-medium-zh](language/nlp/ernie-3.0-medium-zh)          |                    |:white\_check\_mark:|
|gpt2                              |[language/generative/gpt2](language/generative/gpt2)                          |:white\_check\_mark:|                    |
|GRU                               |[language/nlp/GRU](language/nlp/GRU)                                          |:white\_check\_mark:|:white\_check\_mark:|
|LLaMA-2-7B-chat                   |[language/generative/llama2](language/generative/llama2)                      |                    |:white\_check\_mark:|
|LLaMA-7B                          |[language/generative/llama](language/generative/llama)                        |                    |:white\_check\_mark:|
|mobilebert-tflite                 |[language/nlp/mobilebert\_tflite](language/nlp/mobilebert_tflite)             |                    |:white\_check\_mark:|
|ScoringIds                        |[language/asr/conformer](language/asr/conformer)                              |:white\_check\_mark:|                    |
|TransformerDecoder                |[language/asr/conformer](language/asr/conformer)                              |:white\_check\_mark:|                    |
|TransformerLM                     |[language/asr/conformer](language/asr/conformer)                              |:white\_check\_mark:|                    |
|WenetDecoder                      |[language/asr/wenet](language/asr/wenet)                                      |:white\_check\_mark:|:white\_check\_mark:|
|WenetEncoder                      |[language/asr/wenet](language/asr/wenet)                                      |:white\_check\_mark:|:white\_check\_mark:|
|WenetSpeechDecoder                |[language/asr/wenet](language/asr/wenet)                                      |                    |:white\_check\_mark:|
|WenetSpeechNonStreamingEncoder    |[language/asr/wenet](language/asr/wenet)                                      |                    |:white\_check\_mark:|
