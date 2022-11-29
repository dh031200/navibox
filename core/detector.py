from time import time

import torch
import numpy as np

from utils.torch_utils import TracedModel
from utils.general import check_img_size, non_max_suppression, letterbox, scale_coords
from trackers.multi_tracker_zoo import create_tracker
from models.experimental import attempt_load


class Detector:
    def __init__(self, weights, device='cuda:0', imgsz=640, trace=True, conf_thresh=0.25, iou_thresh=0.45, classes=None,
                 agnostic_nms=False):
        device = torch.device(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        self.half = device.type != 'cpu'
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        if trace:
            model = TracedModel(model, device, imgsz)
        if self.half:
            model.half()

        self.names = model.module.names if hasattr(model, 'module') else model.names

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        self.device = device
        self.imgsz = imgsz
        self.stride = stride
        self.model = model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        tracker_method = 'ocsort'
        print(f'tracker method : {tracker_method}')
        self.tracker = create_tracker(tracker_method)
        self.min_box_area = 10

    def detect(self, img):
        it = time()
        original_img = img.copy()
        original_shape = img.shape
        # print(f'image_copy : {time() - it}s')
        it = time()
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # print(f'letterbox : {time() - it}s')
        it = time()
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        # print(f'preprocessing : {time() - it}s')
        it = time()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        # print(f'pred : {time() - it}s')
        it = time()
        detections = []
        if len(pred):
            pred = \
            non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=None, agnostic=self.agnostic_nms)[0]
            boxes = scale_coords(img.shape[2:], pred[:, :4], original_shape)
            boxes = boxes.cpu().numpy()
            detections = pred.cpu().numpy()
            detections[:, :4] = boxes
        # print(f'nms : {time() - it}s')
        it = time()

        online_targets = self.tracker.update(detections, original_img)
        # print(f'tracker : {time() - it}s')
        return online_targets
