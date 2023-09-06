import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

from TDNet import Utils

np.seterr(divide='ignore', invalid='ignore')

def VisualiseResult(_Map, e, removeBack=False):
    Map = np.uint8(_Map)
    histMap = e.convrt2Image(Map)
    visualBird = heatMaker(_Map, removeBack)
    visualMap = e.convrt2Image(visualBird)
    visualShow = cv2.addWeighted(e.image, 0.7, visualMap, 1 - 0.7, 0)
    return visualShow, visualBird, histMap

def VisualiseResultOnVisualMap(_Map, visual):
    if len(visual.shape) != 3: 
        visual = cv2.merge([visual,visual,visual])

    Map = np.uint8(_Map)
    visualBird = heatMaker(_Map, removeBack= True)

    img2gray = cv2.cvtColor(visualBird,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 30, 255, cv2.THRESH_BINARY)
    m = cv2.blur(mask, (11,11))
    mas = cv2.merge([m,m,m])
    visualS = Utils.blend_with_mask_matrix(visualBird, visual, mas)

    visualShow = cv2.addWeighted(visual, 0.25, visualS, 1 - 0.25, 0)
    return visualShow

def heatMaker (heatMap, removeBack):
    heatShow = cv2.applyColorMap(np.uint8(heatMap), cv2.COLORMAP_JET)
    if removeBack:
        b = heatShow[:,:,0]
        g = heatShow[:,:,1]
        r = heatShow[:,:,2]
        mask = np.where(b <= 130, 0, b)
        heatShow = np.dstack((mask,g,r))
        cv2.blur(heatShow, (10,10))
    return heatShow

def counterGraph(frame, vecList, pedList, Rithm, Range, figSize=(9,5), stream=False):
    fig, ax = plt.subplots(figsize=figSize)
    canvas = FigureCanvas(fig)
    
    ax.plot(pedList, 'b', label = 'Pedestrian')
    ax.plot(vecList, 'g', label = 'Vehicle')
    
    if stream:
        ax.set_xbound(0, Range)
        if len(vecList) > Range : ax.set_xbound(frame - Range,  frame)

    ax.legend()
    ax.legend(prop={'size': 10}, loc=1)
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Number', fontsize=10)
    
    canvas.draw()
    plt.close()
    array = np.array(canvas.renderer.buffer_rgba())
    return cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)


def avgSpeedGraph(frame, sList, speedLimit, Rithm, Range, figSize=(9,5)):
    fig, ax = plt.subplots(figsize=figSize)
    canvas = FigureCanvas(fig)
    ax.plot(sList, 'g', label = 'Average Speed')
    
    min = 0; 
    max= Range
    if len(sList) > Range  :
        min = frame - Range
        max = frame

    ax.set_xbound(min,  max)
    ax.axline([min, speedLimit], [max, speedLimit], color='red', ls=':', label='Speed Limitaion')

    ax.legend()
    ax.legend(prop={'size': 10}, loc=1)
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Speed (mph)', fontsize=10)
    
    canvas.draw()
    plt.close()
    array = np.array(canvas.renderer.buffer_rgba())
    return cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)


def calculateAvrageSpeed(_vehicle, precent_VehicleID):
    if len(precent_VehicleID) != 0:
        avgs = 0
        for id, _ in _vehicle.items():
            if _vehicle[id]['present']:
                if not _vehicle[id]['Parked']:
                    if _vehicle[id]['speed'] != 0.0:
                        avgs += _vehicle[id]['showSpeed']
        return avgs / len(precent_VehicleID)
    else:
        return 0


def Apply_crowdMap(centroid_dict, img, _crowdMap, dis, speed):   
    
    _v = []
    for id, _ in centroid_dict.items():
        if centroid_dict[id]['present']:
            if not centroid_dict[id]['Parked']:
                if centroid_dict[id]['speed'][-1] < speed:
                    _v.append(id)
    active = []            
    for vi in range(len(_v)):
        for vj in range(vi, len(_v)):
            if vi != vj:
                center_i = tuple(centroid_dict[_v[vi]]['positionBEV'][-1])
                center_j = tuple(centroid_dict[_v[vj]]['positionBEV'][-1])
                if Utils.Euclidean_distance(center_i, center_j) < dis:
                    cx = (center_i[0] + center_j[0]) // 2
                    cy = (center_i[1] + center_j[1]) // 2
                    active.append(center_i)
                    active.append(center_j)
                    active.append((cx, cy))


    heat = np.zeros((_crowdMap.shape[0], _crowdMap.shape[1]))
    for center_bird in active:
        for i in range(1, 20, 5):
            new = np.zeros_like(heat)
            cv2.circle(new,center_bird, 2*i, 10, -1)
            heat = cv2.add(heat, new)
    heat = cv2.blur(heat,(10,10))
    _crowdMap = cv2.add(_crowdMap, heat)
    return _crowdMap, heat
    
class Cell:
    def __init__(self, gridSize, mapSize, bufferSize, SmoothKernel, RadiationRate, DecayRate):
        self.map_size = mapSize[1], mapSize[0]
        self.grid_size = gridSize
        self.buffer_size = bufferSize
        self.cell_dict = {}
        self.make_Grid()
        self.rad = RadiationRate
        self.decay = DecayRate
        if SmoothKernel != 0:
            self.SmoothResults = 1
            self.SmoothKernel = (SmoothKernel, SmoothKernel)
        else:
            self.SmoothResults = 0

    def update(self, centroid_dict):
        activeCell = []
        for id, box in centroid_dict.items():
            if centroid_dict[id]['present']:
                if not centroid_dict[id]['Parked']:
                    if centroid_dict[id]['speed'][-1] != 0.0:
                        center_bird = int(centroid_dict[id]['positionBEV'][-1][0]), int(centroid_dict[id]['positionBEV'][-1][1])
                        activeCell.append(self.find_Cell(center_bird))

        for i in range(self.grid_size[0]):
            for j in range (self.grid_size[1]):
                position = (i,j)
                flag_cache = self.cell_dict[position]['flag']
                if position in activeCell:
                    if np.sum(self.cell_dict[position]['buffer']) == self.buffer_size:
                        self.cell_dict[position]['flag'] = True
                        self.cell_dict[position]['value'] += .1 * self.rad
                    else:
                        self.cell_dict[position]['flag'] = False
                        self.cell_dict[position]['value'] += 1 * self.rad
                        self.cell_dict[position]['buffer'].pop(0)
                        self.cell_dict[position]['buffer'].append(1)
                    if flag_cache == False and self.cell_dict[position]['flag'] == True:
                        self.cell_dict[position]['value'] -= self.buffer_size * self.decay * self.rad
                        self.cell_dict[position]['refresh'] += 1 

                else:
                    self.cell_dict[position]['buffer'].pop(0)
                    self.cell_dict[position]['buffer'].append(0)
        self.normalise()

    def normalise(self):
        for i in range(self.grid_size[0]):
            for j in range (self.grid_size[1]):
                position = (i,j)
                if 255 < self.cell_dict[position]['value']:
                    self.cell_dict[position]['value'] = 255

    def make_Grid(self):
        for i in range(self.grid_size[0]):
            for j in range (self.grid_size[1]):
                self.cell_dict[(i,j)] = {'value':.0, 'buffer': [0 for x in range(self.buffer_size)], 'flag': False, 'refresh':.0}
    
    def make_Map(self, showGrid=False):
        map = np.zeros(self.map_size, dtype=int)
        h, w  = self.map_size
        cx = w//self.grid_size[0] 
        cy = h//self.grid_size[1] 
        if showGrid :gridShow = np.zeros((self.map_size), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range (self.grid_size[1]):
                map[j*cy:j*cy + cy,  i*cx:i*cx + cx] = self.cell_dict[(i, j)]['value']
                if showGrid:
                    top_left  = ( i*cx, j*cy)
                    botton_right = (i*cx + cx, j*cy + cy)
                    cv2.rectangle(gridShow,top_left, botton_right, 255, 1)
        if self.SmoothResults:
            map = self.smoother(map, self.SmoothKernel)
        return map

    def make_Hist(self):
        tl_cell = (50, 80)
        rb_cell = (0, 60)

        w = tl_cell[1] - tl_cell[0]
        h = rb_cell[1] - rb_cell[0]

        iw = 0
        selected_cell = np.zeros((w, h))
        for i in range(tl_cell[0], tl_cell[1]):
            jh = 0
            for j in range (rb_cell[0], rb_cell[1]):
                selected_cell[iw,jh] = self.cell_dict[(i,j)]['value']
                jh +=1
            iw +=1
        return selected_cell

    def find_Cell(self, point):
        h, w  = self.map_size
        cx = w//self.grid_size[0] 
        cy = h//self.grid_size[1] 
        return point[0]//cx , point[1]//cy

    def smoother(self, array, BlurKernel):
        max = array.max()
        array = cv2.blur(array, BlurKernel)
        array = ((array - array.min()) / (array.max() - array.min())) * max
        return array


class SpeedVioCell(Cell):
    def __init__(self, gridSize, mapSize, bufferSize, SmoothKernel, RadiationRate, DecayRate, SpeedLimitation):
        super(SpeedVioCell, self).__init__(gridSize, mapSize, bufferSize, SmoothKernel, RadiationRate, DecayRate)
        self.speedLimit = SpeedLimitation

    def update(self, centroid_dict):
            activeCell = []
            for id, box in centroid_dict.items():

                if centroid_dict[id]['present']:
                    if not centroid_dict[id]['Parked']:
                        if centroid_dict[id]['speed'][-1] >= self.speedLimit:
                            center_bird = int(centroid_dict[id]['positionBEV'][-1][0]), int(centroid_dict[id]['positionBEV'][-1][1])
                            activeCell.append(self.find_Cell(center_bird))

            for i in range(self.grid_size[0]):
                for j in range (self.grid_size[1]):
                    position = (i,j)
                    flag_cache = self.cell_dict[position]['flag']
                    if position in activeCell:
                        if np.sum(self.cell_dict[position]['buffer']) == self.buffer_size:
                            self.cell_dict[position]['flag'] = True
                            self.cell_dict[position]['value'] += .1 * self.rad
                        else:
                            self.cell_dict[position]['flag'] = False
                            self.cell_dict[position]['value'] += 1 * self.rad
                            self.cell_dict[position]['buffer'].pop(0)
                            self.cell_dict[position]['buffer'].append(1)
                        if flag_cache == False and self.cell_dict[position]['flag'] == True:
                            self.cell_dict[position]['value'] -= self.buffer_size * self.decay * self.rad
                            self.cell_dict[position]['refresh'] += 1 

                    else:
                        self.cell_dict[position]['buffer'].pop(0)
                        self.cell_dict[position]['buffer'].append(0)
            self.normalise()


class nearestPedVecCell(Cell):
    def __init__(self, gridSize, mapSize, bufferSize, SmoothKernel, RadiationRate, DecayRate, Distance):
        super(nearestPedVecCell, self).__init__(gridSize, mapSize, bufferSize, SmoothKernel, RadiationRate, DecayRate)
        self.Distance = Distance

    def update(self, ped, vec):
            activeCell = []
            _v = []; _p = []
            for id, _ in vec.items():
                if vec[id]['present']:
                    if not vec[id]['Parked']:
                        if vec[id]['speed'][-1] != 0.0:
                            _v.append((int(vec[id]['positionBEV'][-1][0]), int(vec[id]['positionBEV'][-1][1])))

            for id, _ in ped.items():
                if ped[id]['present']:
                    _p.append((int(ped[id]['positionBEV'][-1][0]), int(ped[id]['positionBEV'][-1][1])))

            for p in _p:
                for v in _v:
                    if Utils.Euclidean_distance(p, v) <= self.Distance:
                        if not p in activeCell:
                            activeCell.append(self.find_Cell(p))


            for i in range(self.grid_size[0]):
                for j in range (self.grid_size[1]):
                    position = (i,j)
                    flag_cache = self.cell_dict[position]['flag']
                    if position in activeCell:
                        if np.sum(self.cell_dict[position]['buffer']) == self.buffer_size:
                            self.cell_dict[position]['flag'] = True
                            self.cell_dict[position]['value'] += .1 * self.rad
                        else:
                            self.cell_dict[position]['flag'] = False
                            self.cell_dict[position]['value'] += 1 * self.rad
                            self.cell_dict[position]['buffer'].pop(0)
                            self.cell_dict[position]['buffer'].append(1)
                        if flag_cache == False and self.cell_dict[position]['flag'] == True:
                            self.cell_dict[position]['value'] -= self.buffer_size * self.decay * self.rad
                            self.cell_dict[position]['refresh'] += 1 
                    else:
                        self.cell_dict[position]['buffer'].pop(0)
                        self.cell_dict[position]['buffer'].append(0)
            self.normalise()