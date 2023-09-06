import cv2
import math
import numpy as np
from TDNet import Calibration

class Video:
    def __init__(self, src=0, batch=1, new_size=None, lenght_of_video=None):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.fps = self.stream .get(5)
        self.width = int(self.stream .get(3))
        self.height = int(self.stream .get(4))
        self.lenght = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)) if self.src != 0 else -1
        if not lenght_of_video is None:
            if self.lenght <= self.lenght:
                self.lenght = lenght_of_video
        self.resize = False
        if not new_size is None:
            if isinstance(new_size, tuple) and len(new_size) == 2:
                self.width, self.height = new_size
                self.resize = True
            if isinstance(new_size, int):
                self.width, self.height = new_size, new_size
                self.resize = True
        self.size_wh = (self.width, self.height)
        self.size_hw = (self.height, self.width)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.batch = batch
        
    def __iter__(self):
        self.current_frame = 0
        return self

    def __next__(self):
        self.frames = []
        for _ in range(self.batch):
            (self.grabbed, self.frame) = self.stream.read()
            if self.grabbed: 
                self.current_frame += 1
                if self.resize: cv2.resize(self.frame,(self.width, self.height), interpolation=cv2.INTER_LINEAR)
                self.frames.append(self.frame)
        if len(self.frames) == 0: raise StopIteration
        return self.current_frame, np.array(self.frames)
                    
    def __len__(self):
        return self.lenght

def xywh2cord(x, y, w, h): 
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cord2xywh(xmin, ymin, xmax, ymax):
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + (w//2)
    y = ymin + (h//2)
    return x, y, w, h
    
def get_VideoDetails(path, ReductionFactor):
    cap = cv2.VideoCapture(path)
    fps = cap.get(5)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    height, width = frame_height // ReductionFactor, frame_width // ReductionFactor
    print("Video Reolution: ",(width, height))
    return cap, fps, width, height

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0: return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def Euclidean_distance (p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) 

def BoxMaker(p, t,maxSize=None):
    x1, y1 = p[0] - t, p[1] - t 
    x2, y2 = p[0] + t, p[1] + t
    x1 = 0 if x1 <0 else x1
    y1 = 0 if y1 <0 else y1
    x2 = 0 if x2 <0 else x2
    y2 = 0 if y2 <0 else y2
    if maxSize:
        x1 = maxSize[0] if x1 > maxSize[0] else x1
        y1 = maxSize[1] if y1 > maxSize[1] else y1
        x2 = maxSize[0] if x2 > maxSize[0] else x2
        y2 = maxSize[1] if y2 > maxSize[1] else y2
    return x1,y1,x2,y2

def xywh2cord(x, y, w, h): 
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cord2xywh(xmin, ymin, xmax, ymax):
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + (w//2)
    y = ymin + (h//2)
    return x, y, w, h

def ColorGenerator(seed=100100, size=10):
    np.random.seed = seed
    color=dict()
    for i in range(size):
        h = int(np.random.uniform() *255)
        color[i]= h
    return color

def meanMedian(list):
    upper =[] ; lower =[]
    mean = np.mean(list) 
    for i in list:
        if i > mean:
            upper.append(i)
        else:
            lower.append(i)
    if len(upper) > len(lower):
        return np.mean(upper) 
    elif len(upper) < len(lower):
        return np.mean(lower) 
    else:
        return mean

def popIDfrom(pool, dicti):
    c = len(pool)
    for i, _ in pool.items():
        if pool[i]:
            pool[i] = False
            return i
        else:
            c -=1
    if c == 0:
        print('overflow ID!')
        max = 0
        maxItem = 0
        for id, _ in dicti.items():

            if max <= dicti[id]['absence']:
                max = dicti[id]['absence']
                maxItem = id
                ID = dicti[maxItem]['pID']

        pool[ID] = True
        del dicti[maxItem]
        return ID

def Speed_estimate(v, sampleRate, pixelSize, speedUint, upper):
    if len(v['location_bird']) > 1:
        pos = v['location_bird'][-1]
        _pos = v['location_bird'][-sampleRate]

        dis = Euclidean_distance(pos, _pos)
        speed = float(dis * pixelSize * speedUint * 3600) / 0.5

        return speed
    else:
        return 0.0

def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

def getRefrenceAngle(edge, TARGET):
    nearest =  find_nearest_white(edge, TARGET)[0]
    distance = Euclidean_distance(nearest, TARGET)
    aroun_of_nearest = BoxMaker(nearest, 10, maxSize=(edge.shape[1], edge.shape[0]))
    scanArea = edge[aroun_of_nearest[1]+1: aroun_of_nearest[3], aroun_of_nearest[0]+1: aroun_of_nearest[2]]
    nonzero = cv2.findNonZero(scanArea)
    minPoint = nonzero[np.argmin(nonzero[:,0,1])][0]
    maxPoint = nonzero[np.argmax(nonzero[:,0,1])][0]
    angle = getAngle(minPoint, maxPoint, maxPoint)
    return angle, distance

def getCorrectAngle(edge, _loc, loc):
    r_angle, distance = getRefrenceAngle(edge, loc)
    s_angle = getAngle(_loc, loc, loc)

    r_angleU = r_angle
    r_angleD = (360 - (r_angle + 180))  if r_angle + 180 > 360 else (r_angle + 180) # Rotated 180
    
    rotatedRefrance = True if abs(s_angle - r_angleU) > abs(s_angle - r_angleD) else False
    r_angle = r_angleD if rotatedRefrance else r_angleU
    # print(r_angleU, r_angleD, r_angle, s_angle, rotatedRefrance) 
    return r_angle, s_angle, distance

def refrencePoint(xmin, ymin, xmax, ymax, w, h):
    x1 = xmin
    y1 = ymin + h//2
    x2 = xmax - h//4
    y2 = ymax
    lineW = (x2 - x1)
    lineH = (y2 - y1)
    centerLineX = xmin + lineW // 2
    centerLineY = ymax - lineH // 2
    return centerLineX-2, centerLineY +4

def refrencePoint2(xmin, ymin, xmax, ymax, w, h):
    x1 = xmin + w//4
    y1 = ymin
    x2 = xmax
    y2 = ymax - h//2 
    lineW = (x2 - x1)
    lineH = (y2 - y1)
    centerLineX = xmin + lineW // 2
    centerLineY = ymax - lineH // 2
    return centerLineX-2, centerLineY +4


def cord2Vertex(refPoint, e, size, angle, roiCoords, topOffset, direction=False, Xoffset=0):
    position_bird =  e.projection_on_bird(refPoint)
    box2D = np.int0(cv2.boxPoints(((position_bird[0] +Xoffset, position_bird[1]), size, angle)))

    bottonEdge = (box2D[0][0], box2D[0][1]), (box2D[1][0], box2D[1][1])
    topEdge = (box2D[2][0], box2D[2][1]), (box2D[3][0], box2D[3][1])

    front_left = Calibration.applyROIxy(e.projection_on_image(topEdge[0]), roiCoords, reverse=True)
    front_right = Calibration.applyROIxy(e.projection_on_image(topEdge[1]), roiCoords, reverse=True)
    back_left = Calibration.applyROIxy(e.projection_on_image(bottonEdge[1]), roiCoords, reverse=True)
    back_right = Calibration.applyROIxy(e.projection_on_image(bottonEdge[0]), roiCoords, reverse=True)

    if direction:
        b = 0.8
        bb = 0.8
        direc = findCubeDirection(front_left, back_left, front_right)
        if direc == 'ul':
            front_left_t = front_left[0], front_left[1] - int(topOffset * b)
            front_right_t = front_right[0], front_right[1] -  int(topOffset * bb)
            back_left_t = back_left[0], back_left[1] - topOffset
            back_right_t = back_right[0], back_right[1]- topOffset
        elif direc == 'ur':
            front_left_t = front_left[0], front_left[1] - int(topOffset * bb)
            front_right_t = front_right[0], front_right[1] - int(topOffset * b)
            back_left_t = back_left[0], back_left[1] - topOffset
            back_right_t = back_right[0], back_right[1]- topOffset
        elif direc == 'dl':
            front_left_t = front_left[0], front_left[1] - topOffset
            front_right_t = front_right[0], front_right[1] - topOffset
            back_left_t = back_left[0], back_left[1] - int(topOffset * bb)
            back_right_t = back_right[0], back_right[1]- int(topOffset * b)
        elif direc == 'dr':
            front_left_t = front_left[0], front_left[1] - topOffset
            front_right_t = front_right[0], front_right[1] - topOffset
            back_left_t = back_left[0], back_left[1] - int(topOffset * b)
            back_right_t = back_right[0], back_right[1]- int(topOffset * bb)
    else:
        front_left_t = front_left[0], front_left[1] - topOffset
        front_right_t = front_right[0], front_right[1] - topOffset
        back_left_t = back_left[0], back_left[1] - topOffset
        back_right_t = back_right[0], back_right[1]- topOffset

    return front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t

def findCubeDirection(front_left, back_left, front_right):
    if front_left[0] < back_left[0]:
        if front_left[0] < front_right[0]:
            return 'ul'
        else:
            return 'dr'
    else:
        if front_left[0] < front_right[0]:
            return 'ur'
        else:
            return 'dl'

def getMask(image, coords, Save=None):
    [sX, sY], [eX, eY] = coords
    mask = np.zeros_like(image)
    mask[sY:eY, sX:eX, :] = 255
    if Save: cv2.imwrite(Save, mask)
    return mask

def drawMask(image, coords, Save=None):
    mask = np.zeros_like(image)
    C = np.array([[coords[0], coords[1], coords[3], coords[2]]])
    cv2.fillPoly(mask, C, (255,255,255))
    if Save: cv2.imwrite(Save, mask)
    return mask[:,:,0]

def blend_with_mask_matrix(src1, src2, mask):
    res_channels = []
    for c in range(0, src1.shape[2]):
        a = src1[:, :, c]
        b = src2[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
           dtype=cv2.CV_8U)
        res_channels += [res]
    res = cv2.merge(res_channels)
    return res