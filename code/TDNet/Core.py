import numpy as np
from itertools import combinations

from TDNet.Utils import *

def update_state(frame, fps, _vehicle, _pedest, Cache, e, RoadData, Buffers, SpeedUnit, sysParm, calibParm):
    precent_VehicleID = Cache['Current Vehicles']
    precent_PedestID = Cache['Current Pedests']
    useRoadReefrence = sysParm['Use Road Refrence']
    useSystemMask = sysParm['Use ROI BEV Mask']
    PixelUnit = calibParm['Pixel Unit'][0]
    ParkSpace = sysParm['Vehicle Park Sapce (cm)']

    ''' Diffratiation Core '''
    for id, _ in _vehicle.items():
        xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]
        x, y, w, h = _vehicle[id]['location'][-1]
        refPoint_bird = _vehicle[id]['locationBEV'][-1]
        # X, Y, W, H = np.mean(_vehicle[id]['location'], axis=0)
        refPoint3D = refrencePoint(xmin, ymin, xmax, ymax, w, h)

        if id in precent_VehicleID:
            _vehicle[id]['PredictCount'] = 0
            ############ BEV Kalman Filter:
            _vehicle[id]['KalmanOBJ'].predict()
            current_measurement = np.array([[np.float32(refPoint_bird[0])], [np.float32(refPoint_bird[1])]])
            pp = _vehicle[id]['KalmanOBJ'].correct(current_measurement)
            _vehicle[id]['positionBEV'].append(pp[:2])

            _vehicle[id]['k_velocity'].append(np.sqrt(pp[2]**2 + pp[3]**2))
            if len(_vehicle[id]['k_velocity']) > Buffers['Vehicle']['k_velocity']: _vehicle[id]['k_velocity'].pop(0)


            ############ 3-rd Kalman Filter (for 3D output):
            current_measurement3D = np.array([[np.float32(refPoint3D[0])], [np.float32(refPoint3D[1])]])
            _vehicle[id]['3D_KalmanOBJ'].correct(current_measurement3D)
            _vehicle[id]['show3DData'] = _vehicle[id]['3D_KalmanOBJ'].predict()
        else:               
            DepricatePrediction = 20
            if _vehicle[id]['counter'] > 20:
                loc =  int(_vehicle[id]['positionBEV'][-1][1]), int(_vehicle[id]['positionBEV'][-1][0])
                try:
                    if useSystemMask and RoadData['ROI BEV Mask'][loc[0], loc[1]] == 0:
                      DepricatePrediction = 0
                except:
                    pass                
                    # print(_vehicle[id]['pID'], 'ignore for predict')
                if _vehicle[id]['PredictCount'] < DepricatePrediction:
                    _vehicle[id]['PredictCount'] += 1
                    _vehicle[id]['positionBEV'].append(_vehicle[id]['KalmanOBJ'].predict()[:2])
                    _vehicle[id]['show3DData'] = _vehicle[id]['3D_KalmanOBJ'].predict()
                else:
                    _vehicle[id]['present'] = False 
            else:
                _vehicle[id]['absence'] +=1
                _vehicle[id]['present'] = False 
        if len(_vehicle[id]['positionBEV']) > Buffers['Vehicle']['positionBEV']: _vehicle[id]['positionBEV'].pop(0)
        
        ############ Speed Calculation:
        speedUnit = 6.2137e-6 if SpeedUnit == 'mph' else 360000000

        ##### 6-Frame Speed
        sampleRate_dt = 6
        if _vehicle[id]['counter'] > sampleRate_dt and ((frame - _vehicle[id]['frame']) % sampleRate_dt == 0):
            kSpeed = _vehicle[id]['k_velocity'][-1]

            # _vehicle[id]['speed'].append(int(kSpeed_dx))
            _vehicle[id]['speed'].append(int(kSpeed))
            if len(_vehicle[id]['speed']) >= Buffers['Vehicle']['speed']: _vehicle[id]['speed'].pop(0)
        
        ##### 15-Frame Speed
        sampleRate_dt = 15
        if (_vehicle[id]['counter'] > sampleRate_dt and ((frame - _vehicle[id]['frame']) % sampleRate_dt == 0)):
            kSpeed = _vehicle[id]['k_velocity'][-1]
            showSpeed = int(kSpeed)
            if int(kSpeed) < 3: showSpeed = 0
            _vehicle[id]['showSpeed'] = showSpeed

        ############ Stop Condition:
        StopThereshold = 4
        if _vehicle[id]['speed'][-1] < StopThereshold:
            Stop = True 
        else:
            Stop = False

        ############ Reference Check:
        if useRoadReefrence:
            r_angle, distance = getRefrenceAngle(RoadData['Road Border'], _vehicle[id]['locationBEV'][-1])            
            ############ Parck Status:
            if distance / PixelUnit *10 < ParkSpace:
                _vehicle[id]['Parked'] = True if Stop else False
            else:
                _vehicle[id]['Parked'] = False


        ############ Angle Situation:
        AngleVector = 3
        if _vehicle[id]['counter'] > AngleVector and ((frame - _vehicle[id]['frame']) % AngleVector == 0):
            if True:#_vehicle[id]['present']:
                angleType = _vehicle[id]['angleType'] 

                p1, p2 = _vehicle[id]['locationBEV'][-1], _vehicle[id]['locationBEV'][-AngleVector]
                angle = getAngle(p1, p2, p2)
                
                if Stop:
                    if useRoadReefrence:
                        if distance < ParkSpace: 
                            _vehicle[id]['angleType'] = 'refrence'
                            _vehicle[id]['angle'].append( -1 * r_angle)
                    else:
                        _vehicle[id]['angleType'] = 'none'

                if not Stop:
                    _vehicle[id]['angleType'] = 'self'
                    _vehicle[id]['angle'].append( -1 * angle)

                    if not angleType ==  'none':
                        _a = -1 * _vehicle[id]['angle'][-2]
                        a = -1 * _vehicle[id]['angle'][-1]
                        d_angle = a - _a
                        rotation_flag = False
                        
                        c = 7
                        if _vehicle[id]['type']=='car': c = 7
                        if _vehicle[id]['type']=='bus': c = 7

                        if 0 <= abs(d_angle) <= 5 :
                            w_angle = 1/2
                        if 5 < abs(d_angle) <= 10 :
                            w_angle = c/10
                        elif 10 < abs(d_angle) <= 20:
                            w_angle = c/20
                        elif 20 < abs(d_angle) <= 30:
                            w_angle = c/30
                        elif 30 < abs(d_angle) <= 40:
                            w_angle = c/40
                        elif 40 < abs(d_angle) <= 50:
                            w_angle = c/50
                        elif 50 < abs(d_angle) <= 60:
                            w_angle = c/60
                        elif 60 < abs(d_angle) <= 70:
                            w_angle = c/70
                        elif 70 < abs(d_angle) <= 80:
                            w_angle = c/80
                        elif 80 < abs(d_angle) <= 90:
                            w_angle = c/90
                        elif 90 < abs(d_angle) <= 100:
                            w_angle = c/100
                        elif 100< abs(d_angle) <= 110:
                            w_angle = c/110
                        elif 110< abs(d_angle) <= 120:
                            w_angle = c/120
                        elif 120< abs(d_angle) <= 240:
                            rotation_flag = True
                            w_angle = 0
                        elif 240 < abs(d_angle) <= 250:
                            w_angle = c/250
                        elif 250 < abs(d_angle) <= 260:
                            w_angle = c/260
                        elif 260 < abs(d_angle) <= 270:
                            w_angle = c/270
                        elif 270 < abs(d_angle) <= 280:
                            w_angle = c/280
                        elif 280 < abs(d_angle) <= 290:
                            w_angle = c/290
                        elif 290 < abs(d_angle) <= 300:
                            w_angle = c/300
                        elif 300 < abs(d_angle) <= 310:
                            w_angle = c/310
                        elif 310 < abs(d_angle) <= 320:
                            w_angle = c/320
                        elif 320 < abs(d_angle) <= 330:
                            w_angle = c/330
                        elif 330 < abs(d_angle) <= 340:
                            w_angle = c/340
                        elif 340 < abs(d_angle) <= 350:
                            w_angle = c/350
                        elif 350 < abs(d_angle) <= 355:
                            w_angle = c/355
                        elif 355 < abs(d_angle) <= 360:
                            w_angle = 1/2
                        else:
                            w_angle = 0
                        if rotation_flag:
                            if 180 +  _a > 360: 
                                angle =  _a - 180
                            else:
                                angle =  _a + 180
                        else:
                            reverse_angle = _a + 180
                            if reverse_angle > 360:
                                reverse_angle = reverse_angle - 360
                                if 0 <= a <= reverse_angle or _a < a < 360:
                                    angle = _a + w_angle * abs(d_angle)
                                    if angle < 0: angle += 360
                                    if angle > 360: angle -= 360

                                else:
                                    angle = _a - w_angle * abs(d_angle)
                                    if angle < 0: angle+= 360
                                    if angle > 360: angle -= 360

                            else:
                                if  _a < a < reverse_angle:
                                    angle = _a + w_angle * abs(d_angle)
                                    if angle < 0: angle+= 360
                                    if angle > 360: angle -= 360
                                else:
                                    angle = _a - w_angle * abs(d_angle)
                                    if angle < 0: angle+= 360
                                    if angle > 360: angle -= 360
                    
                        _vehicle[id]['angle'].pop(-1)
                        _vehicle[id]['angle'].append( -1 * angle)
            if len(_vehicle[id]['angle']) > Buffers['Vehicle']['angle']:_vehicle[id]['angle'].pop(0)  


    for id, _ in _pedest.items():
        Xbird, Ybird = _pedest[id]['locationBEV'][-1]

        if id in precent_PedestID:
            _pedest[id]['KalmanOBJ'].update((Xbird, Ybird))
            _pedest[id]['PredictCount'] = 0
            _pedest[id]['positionBEV'].append(_pedest[id]['KalmanOBJ'].predict()[:2])
        else:
            _pedest[id]['absence'] +=1
            DepricatePrediction = 3
            if _pedest[id]['PredictCount'] < DepricatePrediction:
                _pedest[id]['PredictCount'] += 1
                _pedest[id]['positionBEV'].append(_pedest[id]['KalmanOBJ'].predict()[:2])
            else:
                _pedest[id]['present'] = False
        if len(_pedest[id]['positionBEV']) > Buffers['Pedest']['positionBEV']: _pedest[id]['positionBEV'].pop(0)    

    
def overlapMaching_onPres(_vehicle, theIoU):

    ''' Maching Overlaped '''
    overlape_trash = list()
    for v1, v2 in combinations(_vehicle, 2):
        if v1 != v2:
                b1 = _vehicle[v1]['locationBox'][-1]
                b2 = _vehicle[v2]['locationBox'][-1]
                iou = IoU(b1, b2)
                if iou > theIoU :
                    if _vehicle[v1]['frame'] > _vehicle[v2]['frame']:
                        n = v1 ; o = v2
                    else:
                        n = v2 ; o = v1
                    if o not in overlape_trash:
                        overlape_trash.append(o)
                    typeC = _vehicle[n]['type']
                    _vehicle[n] = _vehicle[o]
                    _vehicle[n]['type'] = typeC
    for i in overlape_trash:
        print('Delete for overlap', 'pID:', _vehicle[i]['pID'], 'id:', i)
        del _vehicle[i]



def overlapMaching_onBird(_vehicle, searchArea):
    ''' Maching 2 Overlaped '''
    overlape_trash = list()
    for v1, v2 in combinations(_vehicle, 2):
        if v1 != v2:
            if _vehicle[v1]['present'] and _vehicle[v2]['present']:
                b1 = _vehicle[v1]['positionBEV'][-1]
                b2 = _vehicle[v2]['positionBEV'][-1]
                box1, box2 = BoxMaker(b1, searchArea), BoxMaker(b2, searchArea)
                # print('box1:', box1, 'box2:', box2)
                iou = IoU(box1, box2)
                if iou > .25 :
                    if _vehicle[v1]['frame'] > _vehicle[v2]['frame']:
                        a = v1 ; b = v2
                    else:
                        a = v2 ; b = v1
                    if b not in overlape_trash:
                        overlape_trash.append(b)
                    # print('box1:', a, _vehicle[a]['pID'], 'box2:', b, _vehicle[b]['pID'], iou)
                    typeC = _vehicle[b]['type']
                    _vehicle[a] = _vehicle[b]
                    _vehicle[a]['type'] = typeC
                    
    for i in overlape_trash:
        del _vehicle[i]


def manageHistory(_vehicle, _pedest, availID, MaximumVehicleNumber, MaximumPedestNumber):
    if len(_vehicle) >= MaximumVehicleNumber:
        max = 0
        maxItem = 0
        selected_falg = False
        for id, _ in _vehicle.items():
            if max < _vehicle[id]['absence']:
                max = _vehicle[id]['absence']
                maxItem = id
                selected_falg = True
        if not selected_falg:
            maxItem = list(_vehicle)[0]
        availID[_vehicle[maxItem]['pID']] = True
        # print('Delete History for ', _vehicle[maxItem]['pID'], ' - Counter:', _vehicle[maxItem]['counter'], ' - Absence:', _vehicle[maxItem]['absence'])
        del _vehicle[maxItem]
    if len(_pedest) >= MaximumPedestNumber:
        max = 0
        maxItem = 0
        selected_falg = False
        for id, _ in _pedest.items():
            if max < _pedest[id]['absence']:
                max = _pedest[id]['absence']
                maxItem = id
                selected_falg = True
        if not selected_falg:
            maxItem = list(_pedest)[0]
        del _pedest[maxItem]