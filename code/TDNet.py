import cv2
import numpy as np
import time
import os
import json
import argparse
from tqdm import tqdm
import TDNet as TN

class Load:
    def __init__(self, Input, opt, Batch=False):
        StartFrom=opt.start; EndAt=opt.end
        RuntimeCfg={
            'Config load'         : 1,
            'Config update'       : {'General':1, 'Visualizer':1, 'Figure':1, 'Heatmap':1, 'System':1, 'Calibration':0, 'Detector':1},
            'Config save'         : 1}
        self.cfg ={
            'General': {
                'Speed Limitation' : 30,
                'Speed Unit'       : 'mph', #kmph
                'Logo'             : {'1/0': 1, 'Caption': '3D-Net Model', 'Size': (270, 50), 'Font Size': 1, 'Color': (0,200,255)},
                'Real Size (cm)'   : {'person':(70), 'car':(470,190), 'truck':(470,190), 'bus':(1195,255), 'motorcycle':(190,70), 'bicycle':(180,50)}},

            'Visualizer': {'1/0' : 1,
                'Main Stream'    : {'1/0': 0, 'Show':0 , 'ShowBEV':0 , 'Save':0 , 'Rithm':1 , 'Video':0, 'VideoBEV':0},
                '2D Detection'   : {'1/0': 1, 'Show':0 , 'Save':0 , 'Rithm':100 , 'Video':1},
                '3D Detection'   : {'1/0': 1, 'Show':0 , 'Save':0 , 'Rithm':100 , 'Video':1},
                'Environment Map': {'1/0': 1, 'Show':0 , 'Save':0 , 'Rithm':100 , 'Video':1, 'Show Trajectory': 1, 'Show Parcked': 1, 'Show Legend': 1, 'Show Frame Counter':1, 'Show ID': 0},
                'CONFIGS':{
                    'Show Caption'     : 1,
                    'Show Speed'       : 0,
                    'Speed Unit Text'  : ' mph',
                    'Speed Text Color' : (0,0,0),
                    'vSpeed Text Color': (0,0,200),
                    'Map':{'Speed Text Size'  : .43},
                    '2D' :{'Caption Text Size': (.7, .5), 'Speed Text Size': (.6, .45), 'width condition': 90},
                    '3D' :{'Caption Text Size': (.7, .5), 'Speed Text Size': (.6, .45), 'width condition': 50, 'caption width condition':110},
                    'person'    : {'Map':{'color':(220,110,0) ,'size'  :5}, 
                                    '3D':{'color':(250,30,30) ,'tcolor':(250,50,50), 'bcolor':(255,100,100), 'size' :(8,8)},
                                    '2D':{'color':(180,30,30) ,'bcolor':(255,70,70)}},
                    'car'       : {'Map':{'color':(0,150,0)   ,'bcolor':(0,0,0)     ,'vcolor':(0, 0, 255) ,'pcolor':(120,150,120), 'size' :(26,15)}, 
                                    '3D':{'color':(0,150,0)   ,'tcolor':(0,220,0)   ,'bcolor':(0,200,0)   ,'vcolor':(0,0,160), 'vtcolor':(0,0,255) , 'vbcolor':(0,0,255), 'size' :(38,15),'height coef': 0.6 ,'direction': False},
                                    '2D':{'color':(0,150,0)   ,'bcolor':(0,200,0)}},
                    'truck'     : {'Map':{'color':(0,150,0)   ,'bcolor':(0,0,0)     ,'vcolor':(0, 0, 255)  ,'pcolor':(120,150,120), 'size' :(26,15)}, 
                                    '3D':{'color':(0,150,0)   ,'tcolor':(0,220,0)   ,'bcolor':(0,200,0)    ,'vcolor':(0,0,160), 'vtcolor':(0,0,255) , 'vbcolor':(0,0,255), 'size' :(38,15),'height coef': 0.65 ,'direction': False},
                                    '2D':{'color':(0,150,0)   ,'bcolor':(0,200,0)}},
                    'bus'       : {'Map':{'color':(15,90,190) ,'bcolor':(0,0,0)     ,'vcolor':(0, 0, 255)  ,'pcolor':(120,150,120), 'size' :(48,16)},
                                    '3D':{'color':(30,90,160) ,'tcolor':(50,120,200),'bcolor':(15,100,240) ,'vcolor':(0,0,160), 'vtcolor':(0,0,255) , 'vbcolor':(0,0,255), 'size' :(100,22),'height coef': 0.7,'direction': True},
                                    '2D':{'color':(30,90,160) ,'bcolor':(15,100,240)}},
                    'motorcycle': {'Map':{'color':(160,160,34),'bcolor':(0,0,0)     ,'vcolor':(0, 0, 255)  ,'pcolor':(120,150,120), 'size' :(26,15)},
                                    '3D':{'color':(160,160,34),'tcolor':(160,180,54),'bcolor':(0,200,0)    ,'vcolor':(0,0,160), 'vtcolor':(0,0,255) , 'vbcolor':(0,0,255), 'size' :(38,15) ,'height coef': 0.6,'direction': False},
                                    '2D':{'color':(160,160,34),'bcolor':(0,200,0)}},
                    'bicycle'   : {'Map':{'color':(255,170,0) ,'bcolor':(0,0,0)     ,'vcolor':(255,130,0)  ,'pcolor':(255,130,0), 'size'   :(17,8)}, 
                                    '3D':{'color':(255,130,90),'tcolor':(255,150,20),'bcolor':(254,232,125),'vcolor':(255,130,0), 'vtcolor':(255,130,0), 'vbcolor':(254,232,125), 'size' :(15,5) ,'height coef': 0.7,'direction': False},
                                    '2D':{'color':(255,130,0) ,'bcolor':(254,232,125)}}}},
            'Figure':  {'1/0': 1,
                'Counter'  : {'1/0' : 1, 'Show':0 , 'Save':0, 'Video':1 , 'Rithm':1, 'Size':(10, 5), 'Range':10, 'Stream': 0},
                'AVGSpeed' : {'1/0' : 1, 'Show':0 , 'Save':0, 'Video':1 , 'Rithm':1, 'Size':(12, 5), 'Range':100}},
            
            'Heatmap': {'1/0': 1,
                'BEV on Visual map': 1,
                'Remove blue back' : 1,
                'Vehicle'  :{'1/0' : 1, 'Show':0 , 'ShowBEV':0 ,'Save':0 , 'SaveBEV': 0, 'Rithm':100, 'Video':1, 'VideoBEV':1,
                                        'Grid Size' : (120*2, 60*2), 'BufferSize': 50, 'Radiation' : 2.125*4*10, 'Decay' : 0.1, 'Smoothness': 10},   
                'Pedest'   :{'1/0' : 1, 'Show':0 , 'ShowBEV':0 ,'Save':0 , 'SaveBEV': 0, 'Rithm':100, 'Video':1, 'VideoBEV':1,
                                        'Grid Size' : (120*2, 60*2), 'BufferSize': 25, 'Radiation' : 2.125*10,   'Decay' : 0.1, 'Smoothness': 10}, 
                'Speed'    :{'1/0' : 1, 'Show':0 , 'ShowBEV':0 ,'Save':0 , 'SaveBEV': 0, 'Rithm':100, 'Video':1, 'VideoBEV':1,
                                        'Grid Size' : (120*2, 60*2), 'BufferSize': 50, 'Radiation' : 2.125*4*10, 'Decay' : 0.1, 'Smoothness': 20},    
                'Nearest'  :{'1/0' : 1, 'Show':0 , 'ShowBEV':0 ,'Save':0 , 'SaveBEV': 0, 'Rithm':100, 'Video':1, 'VideoBEV':1,
                                        'Grid Size' : (120*2, 60*2), 'BufferSize': 25, 'Radiation' : 2.125*4*10, 'Decay' : 0.1, 'Smoothness': 20, 'Distance' : 25},
                'Crowd'    :{'1/0' : 0, 'Show':0 , 'ShowBEV':0 ,'Save':0 , 'SaveBEV': 0, 'Rithm':100, 'Video':1, 'VideoBEV':1,
                                        'Distance' : 70, 'Lower Speed': 7}},

            'System': {
                'Show Window Size'        : (1,1),
                'Reduction Factor'        : 1,
                'Calibration Mode'        : 'Semi',
                'Environment Mode'        : 'Visual Map', # Satellite, BEV
                ##############################
                'Real Object Size'        : 1,
                'Use Visual Map'          : 1,
                'Use Road Mask for Ignore': 1,
                'Use Mask for make Road'  : 0,
                'Use Road Refrence'       : 1,
                'Vehicle Park Sapce (cm)' : 45,
                ##############################
                'Use ROI BEV Mask'        : 1,
                'Maximum Vehicle Number'  : 50,
                'Maximum Pedest Number'   : 70,
                'ID range for Vehicle'    : 100,
                'Video Format'            : 'DIVX', #MJPG
                'Overlap Search BEV (px)' : 20,
                'Overlap Search (IOU)'    : 0.65,
                'Buffers': {'Vehicle':{
                                'type_buf'     : 20,
                                'location'     : 15,
                                'locationBox'  : 1,
                                'locationBEV'  : 15, 
                                'position'     : 15,
                                'positionBEV'  : 10, # angle sample rate
                                'delta_position_bird':10,
                                'speed'        : 1,
                                'k_velocity'   : 15,
                                'angle'        : 2},
                            'Pedest':{
                                'locationBox'  : 1,
                                'location'     : 1,
                                'locationBEV'  : 5,
                                'position'     : 10,
                                'positionBEV'  : 10}}},

            'Calibration':{
                'Region of Interest' : None,
                'Coordinate'         : None,
                'Pixel Unit'         : None,
                'BEV size'           : None},

            'Detector':{
                'Classes': ['person', 'car', 'umbrella', 'truck', 'motorcycle', 'bicycle', 'bus'],
                'Confidence' : {'person' :.1, 'car':.35, 'umbrella':.4, 'truck':.2, 'motorcycle':.2, 'bicycle':.5, 'bus':.1},
                'SORT'  : {'1/0': 1, 'Max Age': 30, 'Min His': 3, 'IoU the': 0.2},
                'thresh': 0.05,
                'config': "",
                'weight': "",
                'meta'  : ""},
        }
        root = os.path.dirname(os.path.abspath(Input))
        filename = os.path.basename(Input).split('.')[0]
        path = root + '/' + filename 
        _Satellite = path + '/Satellite.png'
        _Background = path + '/Background.bmp'
        _RoI =  path + '/Region of Interest.bmp'
        _RoiMask = path + '/ROI Mask.bmp'
        _BeV = path + '/Bird Eye View.bmp'
        _VisualMap = path + '/Visual Map.bmp'
        _RoadMask = path + '/Road Mask.bmp'
        _RoadEdge = path + '/Road Border.bmp'
        _RoiBevMask = path + '/ROI BEV Mask.bmp'
        _PixelUnit = path + '/Pixel Unit.bmp'
        _Config = path + '/config.json'
        _Video = opt.save + '/Video'
        _Figure = opt.save + '/Figure'
        if not os.path.exists(Input): print('::: Video Not Found!'); return 0
        self.Input = Input
        self.StartFrom = StartFrom
        self.EndAt = EndAt
        self.OutVideo  = _Video
        self.OutFigure = _Figure
        self.Data ={'Background'     : None, # An image with the same size of the Camera
                    'Satellite'      : None, # An Image of Satellite
                    'Visual Map'     : None, # An image with the same size of Satellite image
                    'BEV'            : None, # An Image of BEV in the ROI coordinates
                    'ROI'            : None, # An image of Region of interest
                    'ROI Mask'       : None, # A gray image with size of the Camera
                    'ROI BEV Mask'   : None, # A gray image with the same size of BEV image
                    'Road Mask'      : None, # A gray image in BEV space
                    'Road Border'    : None} # A gray image in BEV space which is edges of Road Mask
        print('::: Run time Settings :' , RuntimeCfg)
        RawRun = False
        HaveConfig = False
        if not os.path.isdir(path): os.mkdir(path); RawRun=True; print(': Data Folder Created.')
        if not os.path.isdir(_Video):os.mkdir(_Video) ; print(': Video Folder Created.')
        if not os.path.isdir(_Figure):os.mkdir(_Figure);print(': Figure Folder Created.')

        print(':::::: Start File Checking ...')
        print(':: Loading Data and Configrations ...')
        if RuntimeCfg['Config load'] and os.path.exists(_Config): 
            HaveConfig = True
            with open(_Config) as jcfg: _cfg = json.load(jcfg)
            for it in RuntimeCfg['Config update']:
                if not RuntimeCfg['Config update'][it]: self.cfg[it] = _cfg[it]

        if os.path.exists(_Background):self.Data['Background'] = cv2.imread(_Background); print(': Background Found.') 
        if os.path.exists(_RoI):       self.Data['ROI']        = TN.Calibration.putROI(self.Data['Background'], self.cfg['Calibration']['Region of Interest'])
        if os.path.exists(_Satellite): self.Data['Satellite']  = cv2.imread(_Satellite) ;print(': Satellite Map Found.')
        if os.path.exists(_BeV):       self.Data['BEV']        = cv2.imread(_BeV); self.cfg['Calibration']['BEV size'] = self.Data['BEV'].shape[0:2]
        if os.path.exists(_VisualMap): self.Data['Visual Map'] = cv2.imread(_VisualMap); print(': Visual Map Found.')                                                
        if os.path.exists(_RoadMask):  self.Data['Road Mask']  = cv2.imread(_RoadMask,0) ; print(': Road Mask Found.')
        if os.path.exists(_RoadEdge):  self.Data['Road Border']= cv2.imread(_RoadEdge,0); print(': Road Boarder Found.')

        self.Data['ROI BEV Mask'] = TN.Utils.drawMask(self.Data['BEV'], self.cfg['Calibration']['Coordinate'], Save=_RoiBevMask)
        self.Data['ROI Mask'] = TN.Utils.getMask(self.Data['Background'], self.cfg['Calibration']['Region of Interest'], Save=_RoiMask)
        self.CameraSize = self.Data['Background'].shape[1], self.Data['Background'].shape[0]
        self.RoiSize = self.Data['ROI'].shape[1], self.Data['ROI'].shape[0]
        self.BevSize = self.cfg['Calibration']['BEV size'][1], self.cfg['Calibration']['BEV size'][0]
        self.cfg['System']['Environment Mode'] = 'BEV'
        if os.path.exists(_RoadMask):
            if self.cfg['System']['Use Mask for make Road']: elf.cfg['System']['Environment Mode'] = 'Road Mask'
        if os.path.exists(_Satellite):
            if self.cfg['System']['Use Mask for make Road']:  self.cfg['System']['Environment Mode'] = 'Road Mask'
            else:
                self.cfg['System']['Environment Mode'] = 'Satellite'
                self.BevSize = (self.Data['Satellite'].shape[1], self.Data['Satellite'].shape[0])
        if os.path.exists(_VisualMap) and self.cfg['System']['Use Visual Map']:
            self.cfg['System']['Environment Mode'] = 'Visual Map'
            self.BevSize = (self.Data['Visual Map'].shape[1], self.Data['Visual Map'].shape[0])
        print(f'::: Camera size : {self.CameraSize}, Region of Interest : {self.RoiSize},  Environment size :{self.BevSize}')
        self.e = TN.Calibration.birds_eye(self.Data['ROI'], self.cfg['Calibration']['Coordinate'], size=self.BevSize)
        self.detector = TN.Detection.YOLOv5(root=opt.root, weights=opt.weights )

        if RuntimeCfg['Config save']:
            if HaveConfig:
                with open(_Config) as jcfg: _cfg = json.load(jcfg)
                for it in RuntimeCfg['Config update']:
                    if not RuntimeCfg['Config update'][it]:
                        self.cfg[it] = _cfg[it]
            with open(_Config, 'w') as jcfg: json.dump(self.cfg, jcfg) ;print('::: Configration Saved.')
    ##########################################################################
    
    
    def Run(self):
        Stream = TN.Utils.Video(self.Input, lenght_of_video = self.EndAt)
        if self.cfg['Detector']['SORT']['1/0']:  mcmot = TN.Trakers.SORT(self.cfg['Detector']['SORT']['Max Age'], 
                                                                         self.cfg['Detector']['SORT']['Min His'], 
                                                                         self.cfg['Detector']['SORT']['IoU the']) 
        _vehicle = dict()
        _pedest = dict()
        Videos = dict()
        Cache = {
            'Num of Vehicle'    : list(),
            'Num of Pedest'     : list(),
            'AVG Vehicle Speed' : list(),
            'Current Vehicles'  : list(),
            'Current Pedests'   : list(),
            'Color Pool'        : TN.Utils.ColorGenerator(size = 3000),
            'Available ID'      : {i : True  for i in range(1, self.cfg['System']['ID range for Vehicle'], 1)},
            'Trajectory on BEV' : np.zeros((self.BevSize[1], self.BevSize[0], 3), dtype=np.uint8),
            'Trajectory on Per' : np.zeros((self.CameraSize[1], self.CameraSize[0], 3), dtype=np.uint8),
        }       
        if self.cfg['Visualizer']['1/0']:
            TN.Visualizer.cfg = self.cfg['Visualizer']['CONFIGS']
            if self.cfg['Visualizer']['Main Stream']['1/0']:
                if self.cfg['Visualizer']['Main Stream']['Video']:     Videos['Main'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Main - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.CameraSize)
                if self.cfg['Visualizer']['Main Stream']['VideoBEV']:  Videos['MainBEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Main BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.CameraSize)
            if self.cfg['Visualizer']['2D Detection']['1/0']:
                if self.cfg['Visualizer']['2D Detection']['Video']:    Videos['2D']   = cv2.VideoWriter(self.OutVideo + f'/TDNet - Detection 2D- [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.CameraSize)
            if self.cfg['Visualizer']['3D Detection']['1/0']:
                if self.cfg['Visualizer']['3D Detection']['Video']:    Videos['3D']   = cv2.VideoWriter(self.OutVideo + f'/TDNet - Detection 3D - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.CameraSize)
            if self.cfg['Visualizer']['Environment Map']['1/0']:
                if self.cfg['Visualizer']['Environment Map']['Video']: Videos['Env']  = cv2.VideoWriter(self.OutVideo + f'/TDNet - Environment Map - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)

        if self.cfg['Figure']['1/0']:
            if self.cfg['Figure']['Counter']:
                Cache['Counter Figure'] = np.ones((self.cfg['Figure']['Counter']['Size'][1]*100, self.cfg['Figure']['Counter']['Size'][0]*100, 3), dtype=np.uint8) *255
                if self.cfg['Figure']['Counter']['Video']:
                    Videos['Cont']  = cv2.VideoWriter(self.OutVideo + f'/TDNet - Figure of Counters - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, (self.cfg['Figure']['Counter']['Size'][0]*100, self.cfg['Figure']['Counter']['Size'][1]*100))
            if self.cfg['Figure']['AVGSpeed']:
                Cache['AVGS Figure'] = np.ones((self.cfg['Figure']['AVGSpeed']['Size'][1]*100, self.cfg['Figure']['AVGSpeed']['Size'][0]*100, 3), dtype=np.uint8) *255
                if self.cfg['Figure']['AVGSpeed']['Video']: 
                    Videos['AVGS']  = cv2.VideoWriter(self.OutVideo + f'/TDNet - Figure of Average Speed - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, (self.cfg['Figure']['AVGSpeed']['Size'][0]*100, self.cfg['Figure']['AVGSpeed']['Size'][1]*100))

        if self.cfg['Heatmap']['1/0']:
            Heatmaps = dict()
            if self.cfg['Heatmap']['Vehicle']['1/0']:
                if self.cfg['Heatmap']['Vehicle']['Video']:    Videos['HeatV']     = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Movements - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.RoiSize)
                if self.cfg['Heatmap']['Vehicle']['VideoBEV']: Videos['HeatV_BEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Movements BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)
                Heatmaps['Vehicle Movements']  = TN.Analyzer.Cell(
                            self.cfg['Heatmap']['Vehicle']['Grid Size'],self.BevSize, 
                            self.cfg['Heatmap']['Vehicle']['BufferSize'],
                            self.cfg['Heatmap']['Vehicle']['Smoothness'],
                            self.cfg['Heatmap']['Vehicle']['Radiation'],
                            self.cfg['Heatmap']['Vehicle']['Decay'])
            if self.cfg['Heatmap']['Pedest']['1/0']:
                if self.cfg['Heatmap']['Pedest']['Video']:     Videos['HeatP']     = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Pedestrian Movements - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.RoiSize)
                if self.cfg['Heatmap']['Pedest']['VideoBEV']:  Videos['HeatP_BEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Pedestrian Movements BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)
                Heatmaps['Pedest Movements']  = TN.Analyzer.Cell(
                            self.cfg['Heatmap']['Pedest']['Grid Size'],self.BevSize, 
                            self.cfg['Heatmap']['Pedest']['BufferSize'],
                            self.cfg['Heatmap']['Pedest']['Smoothness'],
                            self.cfg['Heatmap']['Pedest']['Radiation'],
                            self.cfg['Heatmap']['Pedest']['Decay'])
            if self.cfg['Heatmap']['Speed']['1/0']:
                if self.cfg['Heatmap']['Speed']['Video']:      Videos['HeatS']     = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Speed - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.RoiSize)
                if self.cfg['Heatmap']['Speed']['VideoBEV']:   Videos['HeatS_BEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Speed BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)
                Heatmaps['Vehicle Speed Violation'] = TN.Analyzer.SpeedVioCell(
                            self.cfg['Heatmap']['Speed']['Grid Size'],self.BevSize, 
                            self.cfg['Heatmap']['Speed']['BufferSize'],
                            self.cfg['Heatmap']['Speed']['Smoothness'],
                            self.cfg['Heatmap']['Speed']['Radiation'],
                            self.cfg['Heatmap']['Speed']['Decay'],
                            self.cfg['General']['Speed Limitation'])
            if self.cfg['Heatmap']['Nearest']['1/0']:
                if self.cfg['Heatmap']['Nearest']['Video']:    Videos['HeatN']     = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Nearest - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.RoiSize)
                if self.cfg['Heatmap']['Nearest']['VideoBEV']: Videos['HeatN_BEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Nearest BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)
                Heatmaps['Nearest Points'] = TN.Analyzer.nearestPedVecCell(
                            self.cfg['Heatmap']['Nearest']['Grid Size'],self.BevSize, 
                            self.cfg['Heatmap']['Nearest']['BufferSize'],
                            self.cfg['Heatmap']['Nearest']['Smoothness'],
                            self.cfg['Heatmap']['Nearest']['Radiation'],
                            self.cfg['Heatmap']['Nearest']['Decay'],
                            self.cfg['Heatmap']['Nearest']['Distance'])
            if self.cfg['Heatmap']['Crowd']['1/0']:
                if self.cfg['Heatmap']['Crowd']['Video']:    Videos['Crowd']     = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Crowd - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.RoiSize)
                if self.cfg['Heatmap']['Crowd']['VideoBEV']: Videos['Crowd_BEV'] = cv2.VideoWriter(self.OutVideo + f'/TDNet - Heatmap Vehicle Crowd BEV - [{str(self.StartFrom)} - {str(self.EndAt)}].avi', cv2.VideoWriter_fourcc(*self.cfg['System']['Video Format']), Stream.fps, self.BevSize)
                Cache['Vehicle Crowd'] = np.zeros((self.BevSize[1],self.BevSize[0]))


        for frame, frame_read in tqdm(iter(Stream), desc='Processing', ncols=100, unit=' Frame'):

            if frame < int(self.StartFrom): continue
            if self.EndAt != -1: 
                if frame > self.EndAt: break

            image =  frame_read[0]
            self.e.setImage(TN.Calibration.putROI(image, self.cfg['Calibration']['Region of Interest']))
            self.e.img2bird()

            ''' Model object-Detection '''
            detections = self.detector.detect(image)
            Detections = TN.Detection.ExtractDetection(detections, image, self.cfg['Detector'], self.Data, self.cfg['Calibration'], self.e, self.cfg['System'])

            ''' Simple Online and Realtime Tracking '''
            if self.cfg['Detector']['SORT']: Detections = mcmot.update(Detections)

            Cache['Current Vehicles'] = []
            Cache['Current Pedests']  = []
            for det in Detections:
                box = det[:5]
                id = str(int(box[4]))
                c = self.cfg['Detector']['Classes'][np.argmax(det[5:])]
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x, y, w, h = TN.Utils.cord2xywh(xmin, ymin, xmax, ymax)
                rxmin, rymin, rxmax, rymax = TN.Calibration.applyROI((xmin, ymin, xmax, ymax), self.cfg['Calibration']['Region of Interest'])
                rx, ry, rw, rh = TN.Utils.cord2xywh(rxmin, rymin, rxmax, rymax)

                if c in ['car','bus', 'truck','motorcycle','bicycle']:
                    Cache['Current Vehicles'].append(id)
                    RefPoint = (x, ymax)
                    RefPoint3D = TN.Utils.refrencePoint(xmin, ymin, xmax, ymax, w, h)
                    RoiRefPoint = (rx, rymax)
                    RoiRefPoint3D = TN.Utils.refrencePoint(rxmin, rymin, rxmax, rymax , rw, rh)
                    standPoint_vehicle = self.e.projection_on_bird(RoiRefPoint3D) if c == 'bus' else self.e.projection_on_bird(RoiRefPoint)
                    if id not in _vehicle:
                        _vehicle[id] = {
                            'type'         : c,
                            # 'type_buf'     : list(),
                            'frame'        : frame,
                            'pID'          : TN.Utils.popIDfrom(Cache['Available ID'], _vehicle),
                            'present'      : True,
                            'locationBox'  : list(), # xmin, ymin, xmax, ymax
                            'location'     : list(), # x , y , w , h
                            'locationBEV'  : list(), # bev(roi(x , y))
                            'position'     : list(), # roi(x , y , w , h)
                            'positionBEV'  : list(), # kalman(bev(roi(x , y))) [standPoint_vehicle],
                            'Parked'       : False,
                            'angle'        : list(), # [-1 * angle],
                            'angleType'    : None,   # 'none', 'self' , 'refrence'
                            'speed'        : [0.],
                            'mainSpeed'    : 0,
                            'velocity'     : 0,
                            'counter'      : 0,
                            'absence'      : 0,
                            'KalmanOBJ'    : TN.Trakers.mKalmanFilter(x=standPoint_vehicle[0], y=standPoint_vehicle[1], vx=15, vy=15, ax=0, ay=0, dt=1/9),
                            'k_velocity'   : list(),
                            'k_speed'      : 0,
                            'k_speed_dx'   : 0,
                            '3D_KalmanOBJ' : TN.Trakers.KF_3D(x=RefPoint3D[0], y=RefPoint3D[1], vx=0, vy=0),
                            'showSpeed'    : -1,
                            'show3DData'   : list(),
                            'PredictCount' : 0
                        }
                        if self.cfg['System']['Use Road Refrence']:
                            angle, distance = TN.Utils.getRefrenceAngle(self.Data['Road Border'], standPoint_vehicle)
                            _vehicle[id]['angle'] = [-1 * angle]
                            _vehicle[id]['angleType'] = 'refrence'
                        else: _vehicle[id]['angleType'] = 'none'
                    # print(_vehicle[id]['pID'], _vehicle[id]['type'])
                    _vehicle[id]['type'] = c
                    _vehicle[id]['counter'] += 1
                    _vehicle[id]['present'] = True
                    # _vehicle[id]['type_buf'].append(10 if c == 'car' else -10)
                    _vehicle[id]['location'].append((x, y, w, h))
                    _vehicle[id]['locationBox'].append((xmin, ymin, xmax, ymax))
                    _vehicle[id]['locationBEV'].append(standPoint_vehicle)
                    _vehicle[id]['position'].append((rx, ry, rw, rh))
                    if len(_vehicle[id]['location']) > self.cfg['System']['Buffers']['Vehicle']['location']: _vehicle[id]['location'].pop(0)
                    if len(_vehicle[id]['locationBox']) > self.cfg['System']['Buffers']['Vehicle']['locationBox']: _vehicle[id]['locationBox'].pop(0)
                    if len(_vehicle[id]['locationBEV']) > self.cfg['System']['Buffers']['Vehicle']['locationBEV']: _vehicle[id]['locationBEV'].pop(0)
                    if len(_vehicle[id]['position']) > self.cfg['System']['Buffers']['Vehicle']['position']: _vehicle[id]['position'].pop(0)
                    # if len(_vehicle[id]['type_buf']) > self.cfg['System']['Buffers']['Vehicle']['type_buf']: _vehicle[id]['type_buf'].pop(0)

                if c == 'person' :
                    Cache['Current Pedests'].append(id)
                    standPoint_pedest = self.e.projection_on_bird((rx, rymax))
                    if id not in _pedest:
                        _pedest[id] = {
                            'present'      : True,
                            'locationBox'  : list(),
                            'location'     : list(),
                            'locationBEV'  : list(),
                            'position'     : list(),
                            'positionBEV'  : list(),
                            'counter'      : 0,
                            'absence'      : 0,
                            'KalmanOBJ'    : TN.Trakers.KalmanTracker(np.array(standPoint_pedest).reshape((2,1)), R=100, P=10., Q=0.01),
                            'PredictCount' : 0,
                            'Parked'       : False,
                            'speed'        : [10.],
                        }
                    _pedest[id]['counter'] += 1
                    _pedest[id]['present'] = True
                    _pedest[id]['location'].append((x, y, w, h))
                    _pedest[id]['locationBox'].append((xmin, ymin, xmax, ymax))
                    _pedest[id]['locationBEV'].append(standPoint_pedest)
                    _pedest[id]['position'].append((rx, ry, rw, rh))
                    if len(_pedest[id]['location']) > self.cfg['System']['Buffers']['Pedest']['location']: _pedest[id]['location'].pop(0)
                    if len(_pedest[id]['locationBox']) > self.cfg['System']['Buffers']['Pedest']['locationBox']:_pedest[id]['locationBox'].pop(0)
                    if len(_pedest[id]['locationBEV']) > self.cfg['System']['Buffers']['Pedest']['locationBEV']:_pedest[id]['locationBEV'].pop(0)
                    if len(_pedest[id]['position']) > self.cfg['System']['Buffers']['Pedest']['position']:_pedest[id]['position'].pop(0)

            ''' History '''
            TN.Core.manageHistory(_vehicle, _pedest, Cache['Available ID'], self.cfg['System']['Maximum Vehicle Number'], self.cfg['System']['Maximum Pedest Number'])

            ''' Maching Overlaped '''
            TN.Core.overlapMaching_onPres(_vehicle,  self.cfg['System']['Overlap Search BEV (px)'])

            ''' Diffratiation Core '''
            TN.Core.update_state(frame, Stream.fps, _vehicle, _pedest, Cache, self.e, self.Data,
                                self.cfg['System']['Buffers'], 
                                self.cfg['General']['Speed Unit'], 
                                self.cfg['System'],
                                self.cfg['Calibration'])

            ''' Maching 2 Overlaped '''
            TN.Core.overlapMaching_onBird(_vehicle,  self.cfg['System']['Overlap Search (IOU)'])
    
            if self.cfg['Figure']['1/0']:
                if self.cfg['Figure']['Counter']['1/0']:
                    Cache['Num of Vehicle'].append(len(Cache['Current Vehicles']))
                    Cache['Num of Pedest'].append(len(Cache['Current Pedests']))
                    if frame % self.cfg['Figure']['Counter']['Rithm'] == 0: 
                        Cache['Counter Figure'] = TN.Analyzer.counterGraph(frame, Cache['Num of Vehicle'], Cache['Num of Pedest'], self.cfg['Figure']['Counter']['Rithm'], self.cfg['Figure']['Counter']['Range'], self.cfg['Figure']['Counter']['Size'],self.cfg['Figure']['Counter']['Stream'])                  
                        if self.cfg['Figure']['Counter']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} Counter.png', Cache['Counter Figure'] )
                    if self.cfg['Figure']['Counter']['Video']: Videos['Cont'].write(Cache['Counter Figure'])
                    if self.cfg['Figure']['Counter']['Show']: cv2.imshow('Counter', Cache['Counter Figure'])
                if self.cfg['Figure']['AVGSpeed']['1/0']:
                    Cache['AVG Vehicle Speed'].append(TN.Analyzer.calculateAvrageSpeed(_vehicle, Cache['Current Vehicles']))
                    if frame % self.cfg['Figure']['AVGSpeed']['Rithm'] == 0: 
                        Cache['AVGS Figure'] = TN.Analyzer.avgSpeedGraph(frame, Cache['AVG Vehicle Speed'], self.cfg['General']['Speed Limitation'], self.cfg['Figure']['AVGSpeed']['Rithm'], self.cfg['Figure']['AVGSpeed']['Range'], self.cfg['Figure']['AVGSpeed']['Size'])
                        if self.cfg['Figure']['AVGSpeed']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} AVGS.png', Cache['AVGS Figure'] )
                    if self.cfg['Figure']['AVGSpeed']['Video']: Videos['AVGS'].write(Cache['AVGS Figure'])
                    if self.cfg['Figure']['AVGSpeed']['Show']: cv2.imshow('Average Speed', Cache['AVGS Figure'])

            if self.cfg['Heatmap']['1/0']:
                if self.cfg['Heatmap']['Vehicle']['1/0']:
                    Heatmaps['Vehicle Movements'].update(_vehicle)
                    map = Heatmaps['Vehicle Movements'].make_Map()
                    moveVec_visualShow, moveVec_visualBird, moveVec_histMap = TN.Analyzer.VisualiseResult(map, self.e, self.cfg['Heatmap']['Remove blue back'])
                    if self.cfg['Heatmap']['BEV on Visual map']: moveVec_visualBird = TN.Analyzer.VisualiseResultOnVisualMap(map, self.Data[self.cfg['System']['Environment Mode']].copy())
                    if self.cfg['Heatmap']['Vehicle']['Show']:    cv2.imshow('Vehicle Movements', cv2.resize(moveVec_visualShow, (moveVec_visualShow.shape[1] // self.cfg['System']['Show Window Size'][1], moveVec_visualShow.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Vehicle']['ShowBEV']: cv2.imshow('Vehicle Movements Map', cv2.resize(moveVec_visualBird, (moveVec_visualBird.shape[1] // self.cfg['System']['Show Window Size'][1], moveVec_visualBird.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Vehicle']['Video']:   Videos['HeatV'].write(moveVec_visualShow)
                    if self.cfg['Heatmap']['Vehicle']['VideoBEV']:Videos['HeatV_BEV'].write(moveVec_visualBird)
                    if frame % self.cfg['Heatmap']['Vehicle']['Rithm'] == 0: 
                        if self.cfg['Heatmap']['Vehicle']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} HVM.png', moveVec_visualShow)
                        if self.cfg['Heatmap']['Vehicle']['SaveBEV']: cv2.imwrite(self.OutFigure + f'/{frame} HVM BEV.png', moveVec_visualBird)
                if self.cfg['Heatmap']['Pedest']['1/0']:
                    Heatmaps['Pedest Movements'].update(_pedest)
                    map = Heatmaps['Pedest Movements'].make_Map()
                    movePed_visualShow, movePed_visualBird, movePed_histMap = TN.Analyzer.VisualiseResult(map, self.e, self.cfg['Heatmap']['Remove blue back'])
                    if self.cfg['Heatmap']['BEV on Visual map']: movePed_visualBird = TN.Analyzer.VisualiseResultOnVisualMap(map, self.Data[self.cfg['System']['Environment Mode']].copy())
                    if self.cfg['Heatmap']['Pedest']['Show']:    cv2.imshow('Pedest Movements', cv2.resize(movePed_visualShow, (movePed_visualShow.shape[1] // self.cfg['System']['Show Window Size'][1], movePed_visualShow.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Pedest']['ShowBEV']: cv2.imshow('Pedest Movements Map', cv2.resize(movePed_visualBird, (movePed_visualBird.shape[1] // self.cfg['System']['Show Window Size'][1], movePed_visualBird.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Pedest']['Video']:   Videos['HeatP'].write(movePed_visualShow)
                    if self.cfg['Heatmap']['Pedest']['VideoBEV']:Videos['HeatP_BEV'].write(movePed_visualBird)
                    if frame % self.cfg['Heatmap']['Pedest']['Rithm'] == 0: 
                        if self.cfg['Heatmap']['Pedest']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} HPM.png', movePed_visualShow)
                        if self.cfg['Heatmap']['Pedest']['SaveBEV']: cv2.imwrite(self.OutFigure + f'/{frame} HPM BEV.png', movePed_visualBird)
                if self.cfg['Heatmap']['Speed']['1/0']:
                    Heatmaps['Vehicle Speed Violation'].update(_vehicle)
                    map = Heatmaps['Vehicle Speed Violation'].make_Map()
                    sv_visualShow, sv_visualBird, sv_histMap = TN.Analyzer.VisualiseResult(map, self.e, self.cfg['Heatmap']['Remove blue back'])
                    if self.cfg['Heatmap']['BEV on Visual map']: sv_visualBird = TN.Analyzer.VisualiseResultOnVisualMap(map, self.Data[self.cfg['System']['Environment Mode']].copy())
                    if self.cfg['Heatmap']['Speed']['Show']:    cv2.imshow('Speed Violation', cv2.resize(sv_visualShow, (sv_visualShow.shape[1] // self.cfg['System']['Show Window Size'][1], sv_visualShow.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Speed']['ShowBEV']: cv2.imshow('Speed Violation Map', cv2.resize(sv_visualBird, (sv_visualBird.shape[1] // self.cfg['System']['Show Window Size'][1], sv_visualBird.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Speed']['Video']:   Videos['HeatS'].write(sv_visualShow)
                    if self.cfg['Heatmap']['Speed']['VideoBEV']:Videos['HeatS_BEV'].write(sv_visualBird)
                    if frame % self.cfg['Heatmap']['Speed']['Rithm'] == 0: 
                        if self.cfg['Heatmap']['Speed']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} HSV.png', sv_visualShow)
                        if self.cfg['Heatmap']['Speed']['SaveBEV']: cv2.imwrite(self.OutFigure + f'/{frame} HSV BEV.png', sv_visualBird)
                if self.cfg['Heatmap']['Nearest']['1/0']:
                    Heatmaps['Nearest Points'].update(_pedest, _vehicle)
                    map = Heatmaps['Nearest Points'].make_Map()
                    npv_visualShow, npv_visualBird, npv_histMap = TN.Analyzer.VisualiseResult(map, self.e, self.cfg['Heatmap']['Remove blue back'])
                    if self.cfg['Heatmap']['BEV on Visual map']: npv_visualBird = TN.Analyzer.VisualiseResultOnVisualMap(map, self.Data[self.cfg['System']['Environment Mode']].copy())
                    if self.cfg['Heatmap']['Nearest']['Show']:    cv2.imshow('Nearest Pedest and Vehicle', cv2.resize(npv_visualShow, (npv_visualShow.shape[1] // self.cfg['System']['Show Window Size'][1], npv_visualShow.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Nearest']['ShowBEV']: cv2.imshow('Nearest Pedest and Vehicle Map', cv2.resize(npv_visualBird, (npv_visualBird.shape[1] // self.cfg['System']['Show Window Size'][1], npv_visualBird.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Nearest']['Video']:   Videos['HeatN'].write(npv_visualShow)
                    if self.cfg['Heatmap']['Nearest']['VideoBEV']:Videos['HeatN_BEV'].write(npv_visualBird)
                    if frame % self.cfg['Heatmap']['Nearest']['Rithm'] == 0: 
                        if self.cfg['Heatmap']['Nearest']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} HNP.png', npv_visualShow)
                        if self.cfg['Heatmap']['Nearest']['SaveBEV']: cv2.imwrite(self.OutFigure + f'/{frame} HNP BEV.png', npv_visualBird)
                if self.cfg['Heatmap']['Crowd']['1/0']:
                    Cache['Vehicle Crowd'], crowdMap = TN.Analyzer.Apply_crowdMap(_vehicle, image, Cache['Vehicle Crowd'], self.cfg['Heatmap']['Crowd']['Distance'],  self.cfg['Heatmap']['Crowd']['Lower Speed'])
                    crowd = (crowdMap - crowdMap.min()) / (crowdMap.max() - crowdMap.min())*255
                    crowd_visualShow, crowd_visualBird, crowd_histMap = TN.Analyzer.VisualiseResult(crowd, self.e, self.cfg['Heatmap']['Remove blue back'])
                    _crowd = (Cache['Vehicle Crowd'] - Cache['Vehicle Crowd'].min()) / (Cache['Vehicle Crowd'].max() - Cache['Vehicle Crowd'].min())*255
                    _crowd_visualShow, _crowd_visualBird, _crowd_histMap = TN.Analyzer.VisualiseResult(_crowd, self.e, self.cfg['Heatmap']['Remove blue back'])
                    if self.cfg['Heatmap']['BEV on Visual map']: _crowd_visualBird = TN.Analyzer.VisualiseResultOnVisualMap(_crowd, self.Data[self.cfg['System']['Environment Mode']].copy())
                    if self.cfg['Heatmap']['Crowd']['Show']:    cv2.imshow('Vehicle Crowd', cv2.resize(_crowd_visualShow, (_crowd_visualShow.shape[1] // self.cfg['System']['Show Window Size'][1], crowd_visualShow.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Crowd']['ShowBEV']: cv2.imshow('Vehicle Crowd Map', cv2.resize(_crowd_visualBird, (_crowd_visualBird.shape[1] // self.cfg['System']['Show Window Size'][1], crowd_visualBird.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Heatmap']['Crowd']['Video']:   Videos['Crowd'].write(_crowd_visualShow)
                    if self.cfg['Heatmap']['Crowd']['VideoBEV']:Videos['Crowd_BEV'].write(_crowd_visualBird)
                    if frame % self.cfg['Heatmap']['Crowd']['Rithm'] == 0: 
                        if self.cfg['Heatmap']['Crowd']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} HCrowd.png', _crowd_visualShow)
                        if self.cfg['Heatmap']['Crowd']['SaveBEV']: cv2.imwrite(self.OutFigure + f'/{frame} HCrowd BEV.png', _crowd_visualBird)

            if self.cfg['Visualizer']['1/0']:
                if self.cfg['General']['Logo']['1/0']:
                    cv2.rectangle(image, (0,0), tuple(self.cfg['General']['Logo']['Size']), self.cfg['General']['Logo']['Color'], -1)
                    cv2.putText(image, self.cfg['General']['Logo']['Caption'],(10, 30),cv2.FONT_HERSHEY_SIMPLEX, self.cfg['General']['Logo']['Font Size'] ,(0,0,0),2,cv2.LINE_AA)
                if self.cfg['Visualizer']['Main Stream']['1/0']:
                    if self.cfg['Visualizer']['Main Stream']['Show']: cv2.imshow('Main Stream', cv2.resize(image, (image.shape[1] // self.cfg['System']['Show Window Size'][1], image.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if self.cfg['Visualizer']['Main Stream']['ShowBEV']: cv2.imshow('Main Stream on BEV', self.e.bird)
                    if self.cfg['Visualizer']['Main Stream']['Video']:   Videos['Main'].write(image)
                    if self.cfg['Visualizer']['Main Stream']['VideoBEV']:Videos['MainBEV'].write(self.e.bird)
                    if frame % self.cfg['Visualizer']['Main Stream']['Rithm'] == 0: 
                        if self.cfg['Visualizer']['Main Stream']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} Main.png', image)
                if self.cfg['Visualizer']['Environment Map']['1/0']:
                    roadMap = self.Data[self.cfg['System']['Environment Mode']].copy()
                    roadMap = TN.Visualizer.draw_roadMap(_vehicle, _pedest, roadMap, self.cfg['General'], self.e, self.cfg['Visualizer']['Environment Map'], self.cfg['Calibration'], self.cfg['System']['Real Object Size'], Cache)
                    if self.cfg['Visualizer']['Environment Map']['Show Frame Counter']: cv2.putText(roadMap, 'Frame ' + str(frame),(5, 10), cv2.FONT_HERSHEY_SIMPLEX,.4,(50,50,50),1,cv2.LINE_AA)
                    if self.cfg['Visualizer']['Environment Map']['Video']: Videos['Env'].write(roadMap)
                    if self.cfg['Visualizer']['Environment Map']['Show']: cv2.imshow('Environment Map', cv2.resize(roadMap, (roadMap.shape[1] // TN.Calibration.checkReduction(roadMap), roadMap.shape[0] // TN.Calibration.checkReduction(roadMap))))
                    if frame % self.cfg['Visualizer']['Environment Map']['Rithm'] == 0:
                        if self.cfg['Visualizer']['Environment Map']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} EnvMap.png', roadMap)
                if self.cfg['Visualizer']['2D Detection']['1/0']:
                    imgBox = TN.Visualizer.draw_detectionBoxes2D(_vehicle, _pedest, image, Cache['Current Vehicles'], Cache['Current Pedests'], self.cfg['General']['Speed Limitation'], Transparency = 0.55)
                    if self.cfg['Visualizer']['2D Detection']['Video']: Videos['2D'].write(imgBox)
                    if self.cfg['Visualizer']['2D Detection']['Show']: cv2.imshow('2D Detection', cv2.resize(imgBox, (imgBox.shape[1] // self.cfg['System']['Show Window Size'][1], imgBox.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if frame % self.cfg['Visualizer']['2D Detection']['Rithm'] == 0:
                        if self.cfg['Visualizer']['2D Detection']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} 2D.png', imgBox)
                if self.cfg['Visualizer']['3D Detection']['1/0']:
                    imgBox3D = TN.Visualizer.draw_detectionBoxes3D(_vehicle, _pedest, image, Cache['Current Vehicles'], Cache['Current Pedests'], self.cfg['General'], self.e, self.cfg['Calibration'], self.cfg['System']['Real Object Size'], Transparency = 0.55)
                    if self.cfg['Visualizer']['3D Detection']['Video']: Videos['3D'].write(imgBox3D)
                    if self.cfg['Visualizer']['3D Detection']['Show']: cv2.imshow('3D Detection', cv2.resize(imgBox3D, (imgBox3D.shape[1] // self.cfg['System']['Show Window Size'][1], imgBox3D.shape[0] // self.cfg['System']['Show Window Size'][0])))
                    if frame % self.cfg['Visualizer']['3D Detection']['Rithm'] == 0: 
                        if self.cfg['Visualizer']['3D Detection']['Save']: cv2.imwrite(self.OutFigure + f'/{frame} 3D.png', imgBox3D)

            k=cv2.waitKey(1)
            if k%256 == 27: cv2.destroyAllWindows(); break
        for v in Videos: Videos[v].release()
        print(':::: Done.')
        return 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='file/dir')
    parser.add_argument('--yolo', default=5, type=int, help='Choose YOLO version [4, 5, 7]. (Default is 5)')
    parser.add_argument('--root', default='YOLOv5', type=str, help='Choose detector\'s root folder (e.g. C:/YOLOv5). (Default is the current path)')
    parser.add_argument('--save', default='results', type=str, help='Choose address folder for saving results')
    parser.add_argument('--start', default=0, type=int, help='Start frame')
    parser.add_argument('--end', default=-1, type=int, help='End frame')
    parser.add_argument('--weights', default='yolov5l.pt', type=str, help='Address of trained weights. (Default is [yolov5l.pt])')
    parser.add_argument('--cfg', default=None, type=str, help='Address of YOLOv4 network architecture. (Default is [./cfg/yolov4.cgf])')
    parser.add_argument('--data', default=None, type=str, help='Address of YOLOv4 trained model data. (Default is [./cfg/coco.data]')
    opt = parser.parse_args()
    return opt

def main(opt):
    if opt.source:
        supSuffix =  ['mp4', 'avi', 'mpg', 'mov']
        if os.path.isdir(opt.source):  
            print("\nIt is a directory")  
            if not os.path.exists(opt.source + '/Config'): print('::: Configuration Not Found!'); return 0
            for entry in os.scandir(opt.source):
                if entry.is_file():
                    suffix = os.path.basename(entry.path).split('.')[1]
                    if suffix in supSuffix:
                        print(f">>>>>>>>>> Start item : {entry.name}")
                        Load(entry.path, opt, Batch=True).Run()
                    else: print(f'Format not support for {os.path.basename(entry.path)}, supported formats are {supSuffix}')
        else:
            if os.path.exists(opt.source):
                root = os.path.dirname(os.path.abspath(opt.source))
                filename = os.path.basename(opt.source).split('.')[0]
                paths = root + '/' + filename 
                if not os.path.exists(paths + '/config.json'): print('::: Configuration Not Found!'); return 0
                suffix = os.path.basename(opt.source).split('.')[1]
                if suffix in supSuffix:
                    print(f">>>>>>>>>> Start item : {opt.source}")
                    Load(opt.source, opt).Run()
                else: print(f'Format not support for {os.path.basename(opt.source)}, supported formats are {supSuffix}')                
            else: print(f'{opt.source} Not Found!')
    else: print('Use --source [file/folder]')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
  
    