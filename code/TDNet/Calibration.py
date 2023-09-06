import os
import cv2 
import numpy as np

import TDNet.Utils as u


R = 114
ESC_KEY = 27
SPACE = 32

def checkReduction(image):
    reduction = 1
    if image.shape[0]>10000 or image.shape[1]>10000:
        reduction = 10
    elif image.shape[0]>5000 or image.shape[1]>5000:
        reduction = 5
    elif image.shape[0]>2000 or image.shape[1]>2000:
        reduction = 3
    elif image.shape[0]>1100 or image.shape[1]>1100:
        reduction = 2
    return reduction


''''''''''''''' Background Extractor '''''''''''''''
def calcBackgound(VideoPath, reduc, Save=None):
    cap = cv2.VideoCapture(VideoPath)
    _, f = cap.read()
    f= cv2.resize(f, (f.shape[1]// reduc , f.shape[0] // reduc))
    img_bkgd = np.float32(f)
    reduc = checkReduction(img_bkgd)
    print('When you feel the background is good enough, press ESC to terminate and save the background.')
    while True:
        ret, f = cap.read()
        if not ret: break
        cv2.imshow('Main Video', cv2.resize(f, (f.shape[1]// reduc , f.shape[0] // reduc)))
        cv2.accumulateWeighted(f, img_bkgd, 0.01)
        res2 = cv2.convertScaleAbs(img_bkgd)
        cv2.imshow('When you feel the background is good enough, press ESC to terminate and save the background.', cv2.resize(res2, (res2.shape[1]// reduc , res2.shape[0] // reduc)))
        k = cv2.waitKey(20)
        if k == 27: break
    if Save: cv2.imwrite(Save, res2)
    cv2.destroyAllWindows()
    cap.release()
    return res2


''''''''''''''' Region of Interest '''''''''''''''
def getROI(image, Save=None):
    while True:
        roi, coords , roiImage = o.getROI('Select a Region of Interst for caliibration | Actions: Space = OK,  r = Retry |', image).run()
        zeroDim = False
        for i in roi.shape:
            if i ==0: zeroDim = True
        if zeroDim: continue
        cv2.imshow('Your Region of Interrest | Actions: Space = OK,  r = Retry |', roi)
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue
        elif k%256 == SPACE: cv2.destroyAllWindows(); break
    if Save: cv2.imwrite(Save, roiImage)
    return roi, coords

def applyROI(coord, roiCoord, reverse=False):
    x1, y1, x2, y2 = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x1 + sX, y1 + sY, x2 + sX, y2 + sY
    else:
        return x1 - sX, y1 - sY, x2 - sX, y2 - sY

def applyROIxy(coord, roiCoord, reverse=False):
    x, y = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x + sX, y + sY
    else:
        return x - sX, y - sY

def putROI(image, roiCoord):
    [sX, sY], [eX, eY] = roiCoord
    if len(image.shape) > 2:
        roi = image[sY:eY, sX:eX,:]
    else:
        roi = image[sY:eY, sX:eX]
    return roi

def ShowROI(image, roiCoord):
    mask = u.getMask(image, roiCoord)
    [sX, sY], [eX, eY] = roiCoord
    Show = cv2.addWeighted(image, 0.5, np.where(mask > 1, image, 0), 1 - 0.5, 0)
    cv2.rectangle(Show,(sX, sY),(eX, eY), (255,255,255),2)
    cv2.putText(Show, 'Region Of Interest', (sX, sY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return Show


''''''''''''''' Bird's Eye View '''''''''''''''
class birds_eye:
    def __init__(self, image, cordinates, size=None):
        self.original = image.copy()
        self.image =  image
        self.c, self.r = image.shape[0:2]
        if size:self.bc, self.br = size
        else:self.bc, self.br = self.c, self.r
        pst2 = np.float32(cordinates)
        pst1 = np.float32([[0,0], [self.r,0], [0,self.c], [self.r,self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)
        self.bird = self.img2bird()
    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.br, self.bc))
        return self.bird
    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image
    def setImage(self, img):
        self.image = img
    def setBird(self, bird):
        self.bird = bird
    def convrt2Bird(self, img):
        return cv2.warpPerspective(img, self.transferI2B, (self.bird.shape[1], self.bird.shape[0]))
    def convrt2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.image.shape[1], self.image.shape[0]))
    def projection_on_bird(self, p, float_type=False):
        M = self.transferI2B
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
    def projection_on_image(self, p, float_type=False):
        M = self.transferB2I
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
def project(M, p):
    px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    return int(px), int(py)


def getRoadEdge(img, a, b, Save=None):
    edges = cv2.Canny(img,a,b,apertureSize = 3)
    if Save:cv2.imwrite(Save, edges)
    return edges

