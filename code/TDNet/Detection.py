import os
import sys
import cv2
import numpy as np
import torch
from TDNet import Utils
from TDNet import Calibration


class YOLOv5:
    def __init__(self,
                root,
                weights=None,  # model.pt path(s)
                imgsz=640,  # inference size (pixels)
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                half=True,  # use FP16 half-precision inference
                dnn=True,  # use OpenCV DNN for ONNX inference
                ):
        # check_requirements(exclude=('tensorboard', 'thop'))
        sys.path.insert(1, root)
        import torch
        import torch.backends.cudnn as cudnn
        self.torch = torch
    # try:
        from models.common import DetectMultiBackend
        from utils.augmentations import  letterbox
        from utils.general import check_img_size, non_max_suppression, scale_boxes, check_requirements
        from utils.torch_utils import select_device
        self.letterbox = letterbox
        self.check_img_size = check_img_size
        self.non_max_suppression =non_max_suppression
        self.scale_coords = scale_boxes
        self.check_requirements= check_requirements
    # except:
        #     raise ValueError('YOLOv5 Not Found!')
        if weights is None: weights=os.path.join(root, 'yolov5l.pt')
        print('\n <<< Model is running on {} >>> \n'.format('CUDA GPU' if torch.cuda.is_available() else 'CPU'))
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights=weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.half = half & (self.pt or jit or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        self.model.half() if half else self.model.float()
        cudnn.benchmark = True  # set True to speed up constant image size inference

    def detect(self, img, conf_thres=0.02, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=100):
        im = self.letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = self.torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:  im = im[None]  # expand for batch dim
        
        pred = self.model(im) # Inference
        det = self.non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        det = det[0]
        detection = []
        if len(det):
            det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            for *box, conf, cls in reversed(det):
                label = self.names[int(cls)]
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w = xmax - xmin
                h = ymax - ymin
                x = xmin + (w//2)
                y = ymin + (h//2)
                detection.append((label, float(conf), (x, y, w, h)))
        return detection
    def close_cuda(self):
        self.torch.cuda.empty_cache()
        



def ExtractDetection(detections, image, detectorParm, RoadData={}, calibrParm={}, e=None, sysParm={}):
    # img = image.copy()
    detetctedBox = list()
    detetcted = list()
    all_detected = []
    if len(detections) > 0:  
        id = 0
        for detection in detections:
            confident = detection[1]
            name_tag = str(detection[0])
            if name_tag in detectorParm['Classes']:

                if confident < detectorParm['Confidence'][name_tag]: continue

                x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3] 
                xmin, ymin, xmax, ymax = Utils.xywh2cord(float(x), float(y), float(w), float(h))
                x, y, w, h = Utils.cord2xywh(xmin, ymin, xmax, ymax)

                position = (x,ymax)
                onehot = [0,0,0,0,0,0,0]
                
                if sysParm.get('Use Road Mask for Ignore', False):
                    if not (position[1] >= RoadData['ROI Mask'].shape[0] or position[0] >= RoadData['ROI Mask'].shape[1]):
                        if RoadData['ROI Mask'][position[1], position[0], 0] == 0: continue
                
                if sysParm.get('Use BEV Mask for Ignore', False): 
                    position_bird = e.projection_on_bird(Calibration.applyROIxy(position, calibrParm['Region of Interest']))
                    try: 
                        if RoadData['Road Mask'][position_bird[1], position_bird[0]] == 0: continue
                    except: continue

                if name_tag == 'person':
                    if ymax - ymin > 200: continue
                    onehot = [1,0,0,0,0,0,0]
                if name_tag == 'car':onehot = [0,1,0,0,0,0,0]
                if name_tag == 'umbrella':onehot = [0,0,1,0,0,0,0]
                if name_tag == 'truck':onehot = [0,0,0,1,0,0,0]
                if name_tag == 'motorcycle':onehot = [0,0,0,0,1,0,0]
                if name_tag == 'bicycle':onehot = [0,0,0,0,0,1,0]
                if name_tag == 'bus': onehot = [0,0,0,0,0,0,1]

                id +=1
                detetctedBox.append([int(xmin), int(ymin), int(xmax), int(ymax), id, *onehot])
                all_detected.append([int(xmin), int(ymin), int(xmax), int(ymax), name_tag, id, confident])



        detetcted = np.array(detetctedBox) if len(detetctedBox) > 0 else np.empty((0, 12))

        
    return detetcted
