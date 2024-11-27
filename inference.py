# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai

import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
import torch
from torchvision.models import resnet18, vgg11

from script.Dataset import generate_bins, DetectedObject
from library.Plotting import *
from script import ClassAverages
from script.Model import ResNet, ResNet18, VGG11

VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes

# model factory to choose model
model_factory = {
    'resnet': resnet18(weights=False),
    'resnet18': resnet18(weights=False),
    'vgg11': vgg11(weights=False)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}


class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_


def detect3d(weights, source, classes, reg_weights, model_select, calib_file, output_path, save_result=None):
    # Directory
    calib = str(calib_file)

    # load YOLO model
    model = YOLO(weights)
    names = model.names

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # Run detection 2d
    results = model(source, classes=classes, stream=True)
    temp_path, video = None, None

    # loop images
    for i, result in enumerate(results):
        path = result.path
        img = result.orig_img
        xyxy = result.boxes.xyxy.int().tolist()
        cls = result.boxes.cls.int().tolist()

        if temp_path != path and os.path.splitext(path)[-1][1:] in VID_FORMATS:
            temp_path = path

            if video:
                video.release()

            cap = cv2.VideoCapture(temp_path)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(output_path + "/" + os.path.split(path)[-1], fourcc, frame_fps, (frame_width, frame_height))

        bbox_list = []
        for bbox, cls in zip(xyxy, cls):
            bbox_list.append(Bbox([bbox[:2], bbox[2:]], names[cls]))

        for det in bbox_list:
            if not averages.recognized_class(det.detected_class):
                continue
            detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            # plot 3d detection
            plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)

        if save_result and output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if video:
                video.write(img)
            else:
                cv2.imwrite(f'{output_path}/{i:03d}.png', img)
    if video:
        video.release()


def plot3d(img, proj_matrix, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location)  # 3d boxes

    return location


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolo11n.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/vgg11.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='vgg11', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt',
                        help='Calibration file or path')
    parser.add_argument('--save_result', action='store_true', default=True, help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    return opt


def main(opt):
    detect3d(weights=opt.weights, source=opt.source, classes=opt.classes, reg_weights=opt.reg_weights,
             model_select=opt.model_select, calib_file=opt.calib_file, output_path=opt.output_path,
             save_result=opt.save_result)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
