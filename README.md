# YOLO For 3D Object Detection

#### Note
I have created a new repository of improvements of YOLO3D wrapped in pytorch lightning and more various object detector backbones, currently on development. Please check [ruhyadi/yolo3d-lightning](https://github.com/ruhyadi/yolo3d-lightning).

Unofficial implementation of [Mousavian et al](https://arxiv.org/abs/1612.00496) in their paper *3D Bounding Box Estimation Using Deep Learning and Geometry*. YOLO3D uses a different approach, as the detector uses YOLOv5 which previously used Faster-RCNN, and Regressor uses ResNet18/VGG11 which was previously VGG19.

![inference](docs/demo.gif)

### Download Pretrained Weights
In order to run inference code or resuming training, you can download pretrained ResNet18 or VGG11 model. I have train model with 10 epoch each. You can download model with `resnet18` or `vgg11` for `--weights` arguments.
```
cd ${YOLO3D_DIR}/weights
python get_weights.py --weights resnet18
```

## Inference
For inference with pretrained model you can run code below. It can be run in conda env or docker container. 
```
python inference.py \
    --weights yolov5s.pt \
    --source eval/image_2 \
    --reg_weights weights/resnet18.pkl \
    --model_select resnet18 \
    --output_path runs/ \
    --show_result --save_result
```

## Training
```
python train.py \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --save_epoch 5 \
    --train_path ./dataset/KITTI/training \
    --model_path ./weights \
    --select_model resnet18 \
    --api_key xxx
```

