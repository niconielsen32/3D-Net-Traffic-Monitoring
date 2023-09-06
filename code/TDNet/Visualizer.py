import cv2
import numpy as np

from TDNet.Utils import *
from TDNet.Calibration import *

cfg = dict()

def Apply_trackmap(centroid_dict, trackmap, colorPool, decay):    
    trackmap = cv2.cvtColor(trackmap, cv2.COLOR_RGB2HSV)
    a = trackmap[:,:,2] 
    a[a>0] -= decay
    trackmap[:,:,2] = a
    for id, box in centroid_dict.items():
        if centroid_dict[id]['present']:
          if centroid_dict[id]['speed'][-1] != 0.0:
            center_bird = int(centroid_dict[id]['positionBEV'][-1][0]), int(centroid_dict[id]['positionBEV'][-1][1])
            # print(center_bird)
            color = colorPool[int(centroid_dict[id]['pID'])]
            cv2.circle(trackmap,center_bird, 3, (color, 255, 255), -1)
    trackmap = cv2.cvtColor(trackmap, cv2.COLOR_HSV2RGB)
    return trackmap

    
def draw_roadMap(_vehicle, _pedest, roadMap, Gcfg, e, parms, calibration, realSize, chache):
    # Make Road:
    if len(roadMap.shape) != 3: 
        road = np.ones((roadMap.shape[0], roadMap.shape[1], 3), dtype=np.uint8)
        road[:,:,0] = 222
        road[:,:,1] = 232
        road[:,:,2] = 242
        road[roadMap > 10, :] = 190
        roadMap = road
    if parms['Show Legend']:
        pass
    if parms['Show Trajectory']:
        roadMap = roadMap.copy()
        chache['Trajectory on BEV'] = Apply_trackmap(_vehicle, chache['Trajectory on BEV'], chache['Color Pool'], 5)
        roadMap = cv2.add(chache['Trajectory on BEV'], roadMap) 
    
    SpeedLimitation = Gcfg['Speed Limitation']

    for id, _ in _vehicle.items():
        if _vehicle[id]['present']:
            c = _vehicle[id]['type']
            pID = str(_vehicle[id]['pID'])
            positionBEV = _vehicle[id]['positionBEV'][-1]
            speed = _vehicle[id]['showSpeed']
            if _vehicle[id]['angleType'] == 'none':
                angle = 0
                if realSize:
                    size = round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][1])
                else:
                    size = (cfg[c]['Map']['size'][1], cfg[c]['Map']['size'][1])
            else:
                angle = _vehicle[id]['angle'][-1]
                if realSize:
                    size = round(Gcfg['Real Size (cm)'][c][0] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][1])
                else:
                    size = cfg[c]['Map']['size']

            color = cfg[c]['Map']['color']
            bColor= cfg[c]['Map']['bcolor']
            
            caption = "{:.0f}".format(speed) #  +' K' + str(speed) + ' ' + str(k_speed_dt)
            if speed < 0: caption = ''
            if speed == 0:
                if parms['Show Parcked'] and _vehicle[id]['Parked']: 
                    color = cfg[c]['Map']['pcolor']
                    caption = 'P'
            if speed > SpeedLimitation:
                color = cfg[c]['Map']['vcolor']
                speedColor = cfg['vSpeed Text Color']
            else:
                speedColor = cfg['Speed Text Color']

            
            coords = cv2.boxPoints(((int(positionBEV[0]),int(positionBEV[1]) - 6), size, angle))

            cv2.drawContours(roadMap,[np.int0(coords)],0,color,-1)
            cv2.drawContours(roadMap,[np.int0(coords)],0,bColor,1)

            coords = (int(positionBEV[0]), int(positionBEV[1]-6))
            cv2.circle(roadMap, coords, size[1]//2, (255,255,255), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(caption, font, cfg['Map']['Speed Text Size'], 2)[0]
            textX = int(positionBEV[0] - (textsize[0] / 2))
            textY = int(positionBEV[1]-6 + (textsize[1] / 2))
            cv2.putText(roadMap, caption,(textX, textY), font, cfg['Map']['Speed Text Size'],speedColor,1,cv2.LINE_AA)
            
            capt = pID #+ ' id' + str(id)
            if parms['Show ID']: cv2.putText(roadMap, capt ,(positionBEV[0] - (len(pID) *4), positionBEV[1] -6),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)

    for id, _ in _pedest.items():
        if _pedest[id]['present']:
            positionBEV = _pedest[id]['positionBEV'][-1]
            if realSize:
                size = round(Gcfg['Real Size (cm)']['person'] / calibration['Pixel Unit'][0])
            else:
                size = cfg['person']['Map']['size']
            cv2.circle(roadMap, (int(positionBEV[0]) , int(positionBEV[1])), size, cfg['person']['Map']['color'], -1)
    return roadMap


def draw_detectionBoxes2D(_vehicle, _pedest, image, precent_VehicleID, precent_PedestID, SpeedLimitation, Transparency):

    ''' Draw Detection Boxes (Transparent) '''
    imgBox = image.copy()
    for id, _ in _vehicle.items():
        if id in precent_VehicleID:
            xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]
            cv2.rectangle(imgBox,(xmin, ymin),(xmax, ymax), cfg[_vehicle[id]['type']]['2D']['color'],-1)

    for id, _ in _pedest.items():
        if id in precent_PedestID:
            xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]
            cv2.rectangle(imgBox,(xmin, ymin),(xmax, ymax),cfg['person']['2D']['color'],-1)
    imgBox = cv2.addWeighted(image, Transparency, imgBox, 1 - Transparency, 0)


    ''' Draw Boarder Boxes '''
    for id, _ in _vehicle.items():
        if id in precent_VehicleID:
            xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]
            cv2.rectangle(imgBox,(xmin, ymin),(xmax, ymax),cfg[_vehicle[id]['type']]['2D']['bcolor'],2)
    

    for id, _ in _pedest.items():
        if id in precent_PedestID:
            xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]
            cv2.rectangle(imgBox,(xmin, ymin),(xmax, ymax),cfg['person']['2D']['bcolor'],2)


    ''' Draw Caption Boxes '''
    if cfg['Show Caption'] or cfg['Show Speed']:
        for id, _ in _vehicle.items():
            if id in precent_VehicleID:
                c = _vehicle[id]['type']
                xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]
                x, y, w, h = cord2xywh(xmin, ymin, xmax, ymax)
                pID   = _vehicle[id]['pID']
                speed = _vehicle[id]['showSpeed']
                caption = _vehicle[id]['type'].upper()
                if caption == 'bicycle'.upper(): caption = "CYCLIST"
                
                if speed > SpeedLimitation:
                    if c == 'bus': 
                        speedColor = (0,0,255)
                    else:
                        speedColor = (0,0,220)
                    textSpeedColor = (0,0,150)
                else:
                    speedColor = (0,255,0)
                    textSpeedColor = (0,0,0)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if w > cfg['2D']['width condition'] or caption == "CYCLIST":
                    if cfg['Show Caption']:
                        # ### Top-Box Yellow Caption:
                        textsize = cv2.getTextSize(caption, font, cfg['2D']['Caption Text Size'][0], 2)
                        cv2.rectangle(imgBox,(xmin, ymin-textsize[0][1]-3), (xmin + textsize[0][0]+3, ymin),(0,200,255),-1) 
                        cv2.putText(imgBox, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][0],(0,0,0),2)
                    if cfg['Show Speed'] and speed != -1: #!=0:
                        downCaption = str(speed) + cfg['Speed Unit Text']
                        textsize = cv2.getTextSize(downCaption, font, cfg['2D']['Speed Text Size'][0], 2)
                        cv2.rectangle(imgBox,(xmin-1, ymax),(xmax+1, ymax + textsize[0][1]+ 12),(255,255,255),-1) 
                        cv2.line(imgBox,(xmin+3, ymax),(xmax-3, ymax),speedColor,5) 
                        cv2.putText(imgBox, downCaption,(x - (textsize[0][0] // 2), ymax + textsize[0][1]+6), font, cfg['2D']['Speed Text Size'][0], textSpeedColor,2) 
                else:
                    if cfg['Show Caption']:
                        textsize = cv2.getTextSize(caption, font, cfg['2D']['Caption Text Size'][1], 2)
                        cv2.rectangle(imgBox,(xmin, ymin-textsize[0][1]-3),(xmin + textsize[0][0]+3, ymin),(0,200,255),-1)
                        cv2.putText(imgBox, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][1],(0,0,0),2)
                    if cfg['Show Speed'] and speed != -1: #!=0:
                        downCaption = str(speed)# + ' mph'
                        textsize = cv2.getTextSize(downCaption, font, cfg['2D']['Speed Text Size'][1], 2)
                        cv2.rectangle(imgBox,(xmin-1, ymax),(xmax+1, ymax + textsize[0][1]+ 8),(255,255,255),-1) 
                        cv2.line(imgBox,(xmin+3, ymax),(xmax-3, ymax),speedColor,4) 
                        cv2.putText(imgBox, downCaption,(x - (textsize[0][0] // 2), ymax + textsize[0][1]+5), font, cfg['2D']['Speed Text Size'][1], textSpeedColor,2) 

        if cfg['Show Caption']:
            for id, _ in _pedest.items():
                if id in precent_PedestID:
                    xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]
                    color = (255,0,0)
                    caption = 'P'# + f'{str(id)}'

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    textsize = cv2.getTextSize(caption, font, cfg['2D']['Caption Text Size'][0], 2)
                    cv2.rectangle(imgBox,(xmin, ymin-textsize[0][1]-3), (xmin + textsize[0][0]+3, ymin),(0,200,255),-1) 
                    cv2.putText(imgBox, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][0],(0,0,0),2)
    return imgBox


def draw_detectionBoxes3D(_vehicle, _pedest, image, precent_VehicleID, precent_PedestID, Gcfg, e3d, calibration, realSize, Transparency):
    SpeedLimitation = Gcfg['Speed Limitation']
    
    ''' Draw 3D Surfaces (Transparent) '''
    # e3d = birds_eye(image, calibration['Coordinate'])#cordinates = [[0,0],[957,550],[328,951],[502,983]])
    imgBox3D = image.copy()
    for id, _ in _vehicle.items():
        if id in precent_VehicleID:
            x, y, w, h = _vehicle[id]['location'][-1]
            xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]

            speed = _vehicle[id]['showSpeed']
            c = _vehicle[id]['type']
            x = int(_vehicle[id]['show3DData'][0])
            y = int(_vehicle[id]['show3DData'][1])
            refPoint = applyROIxy((x,y), calibration['Region of Interest'])
            # refPoint = refrencePoint(xmin, ymin, xmax, ymax, w, h)
            if _vehicle[id]['angleType'] == 'none':
                cv2.rectangle(imgBox3D,(xmin, ymin),(xmax, ymax), cfg[c]['3D']['color'],-1)
            else:
                if realSize:
                    size = round(Gcfg['Real Size (cm)'][c][0] / calibration['Pixel Unit'][0]) , round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][1]) 
                else:
                    size = cfg[c]['3D']['size']
                angle = _vehicle[id]['angle'][-1]
                topOffset = int(h * cfg[c]['3D']['height coef'])
                direction = cfg[c]['3D']['direction']
                front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t = cord2Vertex(refPoint, e3d, size, angle, calibration['Region of Interest'], topOffset, direction, Xoffset=5)
                
                downSurface = [np.array([front_left, front_right, back_right, back_left])]
                topSurface = [np.array([front_left_t, front_right_t, back_right_t, back_left_t])]
                frontSurface = [np.array([front_left_t, front_right_t, front_right, front_left])]
                backSurface = [np.array([back_left_t, back_right_t, back_right, back_left])]
                leftSurface = [np.array([front_left_t, back_left_t, back_left, front_left ])]
                rightSurface = [np.array([front_right_t, back_right_t, back_right, front_right])]

                if speed > SpeedLimitation:                 
                    frontColor = cfg[c]['3D']['vcolor']
                    mainColor = frontColor
                    topColor = cfg[c]['3D']['vtcolor']
                else:
                    frontColor = cfg[c]['3D']['color']
                    mainColor = frontColor
                    topColor = cfg[c]['3D']['tcolor']
                    
                cv2.drawContours(imgBox3D, downSurface, 0, mainColor,-1)
                cv2.drawContours(imgBox3D, backSurface, 0, mainColor,-1)
                cv2.drawContours(imgBox3D, leftSurface, 0, mainColor,-1)
                cv2.drawContours(imgBox3D, rightSurface, 0, mainColor,-1)
                cv2.drawContours(imgBox3D, frontSurface, 0, frontColor,-1)
                cv2.drawContours(imgBox3D, topSurface, 0, topColor,-1)

    for id, _ in _pedest.items():
        if id in precent_PedestID:
            x, y, w, h = _pedest[id]['location'][-1]
            xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]

            color = cfg['person']['3D']['color']
            topcolor = cfg['person']['3D']['tcolor']
            # refPoint = (x, ymax)
            refPoint = applyROIxy((x,ymax), calibration['Region of Interest'])
            if realSize:
                size = (round(Gcfg['Real Size (cm)']['person'] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)']['person'] / calibration['Pixel Unit'][1]))
            else:
                size = cfg['person']['3D']['size'] 
            topOffset = h 
            front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t = cord2Vertex(refPoint, e3d, size, roiCoords=calibration['Region of Interest'], angle=0, topOffset=topOffset, direction=False)
            
            downSurface = [np.array([front_left, front_right, back_right, back_left])]
            topSurface = [np.array([front_left_t, front_right_t, back_right_t, back_left_t])]
            frontSurface = [np.array([front_left_t, front_right_t, front_right, front_left])]
            backSurface = [np.array([back_left_t, back_right_t, back_right, back_left])]
            leftSurface = [np.array([front_left_t, back_left_t, back_left, front_left ])]
            rightSurface = [np.array([front_right_t, back_right_t, back_right, front_right])]

            cv2.drawContours(imgBox3D, downSurface, 0, color,-1)
            cv2.drawContours(imgBox3D, backSurface, 0, color,-1)
            cv2.drawContours(imgBox3D, leftSurface, 0, color,-1)
            cv2.drawContours(imgBox3D, rightSurface, 0, color,-1)
            cv2.drawContours(imgBox3D, frontSurface, 0, color,-1)
            cv2.drawContours(imgBox3D, topSurface, 0, topcolor,-1)

    imgBox3D = cv2.addWeighted(image, Transparency, imgBox3D, 1 - Transparency, 0)



    ''' Draw Boarder for 3D Surfaces '''
    for id, _ in _vehicle.items():
        if id in precent_VehicleID:
            x, y, w, h = _vehicle[id]['location'][-1]
            xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]

            speed = _vehicle[id]['showSpeed']
            x = int(_vehicle[id]['show3DData'][0])
            y = int(_vehicle[id]['show3DData'][1])
            # refPoint = refrencePoint(xmin, ymin, xmax, ymax, w, h)
            c = _vehicle[id]['type']
            if _vehicle[id]['angleType'] == 'none':
                cv2.rectangle(imgBox3D,(xmin, ymin),(xmax, ymax),cfg[c]['3D']['bcolor'],2)
            else:
                # refPoint = (x,y) 
                refPoint = applyROIxy((x,y), calibration['Region of Interest'])
                angle = _vehicle[id]['angle'][-1]#np.mean(_vehicle[id]['angle'])#meanMedian(_vehicle[id]['angle'])
                if realSize:
                    size = round(Gcfg['Real Size (cm)'][c][0] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][1])
                else:
                    size = cfg[c]['3D']['size']
                topOffset = int(h * cfg[c]['3D']['height coef'])
                direction = cfg[c]['3D']['direction']
                front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t = cord2Vertex(refPoint, e3d, size, angle, calibration['Region of Interest'], topOffset, direction, Xoffset=5)

                if speed > SpeedLimitation:                 
                    frontColor = cfg[c]['3D']['vbcolor']
                    mainColor = frontColor
                else:
                    frontColor = cfg[c]['3D']['bcolor']
                    mainColor = frontColor

                thinness = 1 if w < cfg['3D']['width condition'] else 2

                # Ground Box
                cv2.line(imgBox3D, back_left, back_right, mainColor, thinness) 
                cv2.line(imgBox3D, front_left, back_left, mainColor, thinness) 
                cv2.line(imgBox3D, front_right, back_right, mainColor, thinness) 
                cv2.line(imgBox3D, front_left, front_right, frontColor, thinness) 
                # Top Box
                cv2.line(imgBox3D, back_left_t, back_right_t, mainColor, thinness) 
                cv2.line(imgBox3D, front_left_t, back_left_t, mainColor, thinness) 
                cv2.line(imgBox3D, front_right_t, back_right_t, mainColor, thinness) 
                cv2.line(imgBox3D, front_left_t, front_right_t, frontColor, thinness) 
                # Around Box
                cv2.line(imgBox3D, back_left_t, back_left, mainColor, thinness) 
                cv2.line(imgBox3D, back_right_t, back_right, mainColor, thinness) 
                cv2.line(imgBox3D, front_left_t, front_left, frontColor, thinness) 
                cv2.line(imgBox3D, front_right_t, front_right, frontColor,thinness) 


    for id, _ in _pedest.items():
        if id in precent_PedestID:
            x, y, w, h = _pedest[id]['location'][-1]
            xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]
            
            color = cfg['person']['3D']['bcolor']
            # refPoint = (x, ymax)
            refPoint = applyROIxy((x,ymax), calibration['Region of Interest'])
            if realSize:
                size = (round(Gcfg['Real Size (cm)']['person'] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)']['person'] / calibration['Pixel Unit'][1]))
            else:
                size = cfg['person']['3D']['size'] 
            topOffset = h 
            front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t = cord2Vertex(refPoint, e3d, size,  angle=0, roiCoords=calibration['Region of Interest'], topOffset=topOffset, direction=False)
            thinness = 1
            # Ground Box
            cv2.line(imgBox3D, back_left, back_right, color, thinness) 
            cv2.line(imgBox3D, front_left, back_left, color, thinness) 
            cv2.line(imgBox3D, front_right, back_right, color, thinness) 
            cv2.line(imgBox3D, front_left, front_right, color, thinness) 
            # Top Box
            cv2.line(imgBox3D, back_left_t, back_right_t, color, thinness) 
            cv2.line(imgBox3D, front_left_t, back_left_t, color, thinness) 
            cv2.line(imgBox3D, front_right_t, back_right_t, color, thinness) 
            cv2.line(imgBox3D, front_left_t, front_right_t, color, thinness) 
            # Around Box
            cv2.line(imgBox3D, back_left_t, back_left, color, thinness) 
            cv2.line(imgBox3D, back_right_t, back_right, color, thinness) 
            cv2.line(imgBox3D, front_left_t, front_left, color, thinness) 
            cv2.line(imgBox3D, front_right_t, front_right, color,thinness) 


    ''' Caption for 3D '''
    if cfg['Show Caption'] or cfg['Show Speed']:
        for id, _ in _vehicle.items():
            if id in precent_VehicleID:
                x, y, w, h = _vehicle[id]['location'][-1]
                xmin, ymin, xmax, ymax = _vehicle[id]['locationBox'][-1]

                x = int(_vehicle[id]['show3DData'][0])
                y = int(_vehicle[id]['show3DData'][1])
                c = _vehicle[id]['type']
                speed = _vehicle[id]['showSpeed']
                speedText = str(speed) + cfg['Speed Unit Text']
                textSpeedColor = (0,0,150) if speed > SpeedLimitation else  (0,0,0)
                caption = _vehicle[id]['type'].upper()
                if caption == 'bicycle'.upper():  caption = "CYCLIST"
                font = cv2.FONT_HERSHEY_SIMPLEX

                if _vehicle[id]['angleType'] == 'none':
                    if speed > SpeedLimitation:
                        if c == 'bus': 
                            speedColor = (0,0,255)
                        else:
                            speedColor = (0,0,220)
                        textSpeedColor = (0,0,150)
                    else:
                        speedColor = (0,255,0)
                        textSpeedColor = (0,0,0)
                    
                    if w > cfg['2D']['width condition'] or caption == "CYCLIST":
                        if cfg['Show Caption']:
                            # ### Top-Box Yellow Caption:
                            textsize = cv2.getTextSize(caption, font, cfg['2D']['Caption Text Size'][0], 2)
                            cv2.rectangle(imgBox3D,(xmin, ymin-textsize[0][1]-3), (xmin + textsize[0][0]+3, ymin),(0,200,255),-1) 
                            cv2.putText(imgBox3D, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][0],(0,0,0),2)
                        if cfg['Show Speed'] and speed != -1: #!=0:
                            downCaption = str(speed) + cfg['Speed Unit Text']
                            textsize = cv2.getTextSize(downCaption, font, cfg['2D']['Speed Text Size'][0], 2)
                            cv2.rectangle(imgBox3D,(xmin-1, ymax),(xmax+1, ymax + textsize[0][1]+ 12),(255,255,255),-1) 
                            cv2.line(imgBox3D,(xmin+3, ymax),(xmax-3, ymax),speedColor,5) 
                            cv2.putText(imgBox3D, downCaption,(x - (textsize[0][0] // 2), ymax + textsize[0][1]+6), font, cfg['2D']['Speed Text Size'][0], textSpeedColor,2) 
                    else:
                        if cfg['Show Caption']:
                            textsize = cv2.getTextSize(caption, font, cfg['2D']['Caption Text Size'][1], 2)
                            cv2.rectangle(imgBox3D,(xmin, ymin-textsize[0][1]-3),(xmin + textsize[0][0]+3, ymin),(0,200,255),-1)
                            cv2.putText(imgBox3D, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][1],(0,0,0),2)
                        if cfg['Show Speed'] and speed != -1: #!=0:
                            downCaption = str(speed)# + ' mph'
                            textsize = cv2.getTextSize(downCaption, font, cfg['2D']['Speed Text Size'][1], 2)
                            cv2.rectangle(imgBox3D,(xmin-1, ymax),(xmax+1, ymax + textsize[0][1]+ 8),(255,255,255),-1) 
                            cv2.line(imgBox3D,(xmin+3, ymax),(xmax-3, ymax),speedColor,4) 
                            cv2.putText(imgBox3D, downCaption,(x - (textsize[0][0] // 2), ymax + textsize[0][1]+5), font, cfg['2D']['Speed Text Size'][1], textSpeedColor,2) 
                else:
                    # refPoint = (x,y)
                    refPoint = applyROIxy((x,y), calibration['Region of Interest'])
                    angle = _vehicle[id]['angle'][-1]
                    if realSize:
                        size = round(Gcfg['Real Size (cm)'][c][0] / calibration['Pixel Unit'][0]), round(Gcfg['Real Size (cm)'][c][1] / calibration['Pixel Unit'][1])
                    else:
                        size = cfg[c]['3D']['size']
                    topOffset = int(h * cfg[c]['3D']['height coef'])
                    direction = cfg[c]['3D']['direction']
                    front_left, front_right, back_left, back_right, front_left_t, front_right_t, back_left_t, back_right_t = cord2Vertex(refPoint, e3d, size, angle, calibration['Region of Interest'], topOffset, direction, Xoffset=5)
                    # caption = caption + findCubeDirection(front_left, back_left, front_right)
                    px = front_left_t  if front_left[0] < front_right[0] else front_right_t
                    if w > cfg['3D']['caption width condition']:
                        # ### Top-Box Yellow Caption:
                        if cfg['Show Caption']:
                            textsize = cv2.getTextSize(caption, font, cfg['3D']['Caption Text Size'][0], 2)
                            cv2.rectangle(imgBox3D,(px[0], px[1]-textsize[0][1]-3),(px[0] + textsize[0][0]+3, px[1]),(0,200,255),-1) 
                            cv2.putText(imgBox3D, caption,(px[0]+2, px[1]-2), font, cfg['3D']['Caption Text Size'][0],(0,0,0),2)
                        if cfg['Show Speed'] and speed != -1:
                            textsize = cv2.getTextSize(speedText, font, cfg['3D']['Speed Text Size'][0], 2)
                            cv2.rectangle(imgBox3D,(px[0], px[1]+1),(px[0] + textsize[0][0], px[1]+ textsize[0][1] + 5),(255,255,255),-1) 
                            cv2.putText(imgBox3D, speedText,(px[0]+2, px[1]+ textsize[0][1] + 2), font, cfg['3D']['Speed Text Size'][0], textSpeedColor,2)
                    else:
                        # ### Yellow top box :
                        if cfg['Show Caption']:
                            textsize = cv2.getTextSize(caption, font, cfg['3D']['Caption Text Size'][1], 2)
                            cv2.rectangle(imgBox3D,(px[0], px[1]-textsize[0][1]-3),(px[0] + textsize[0][0]+3, px[1]),(0,200,255),-1)
                            cv2.putText(imgBox3D, caption,(px[0]+2, px[1]-2), font, cfg['3D']['Caption Text Size'][1],(0,0,0),2)
                        if cfg['Show Speed'] and speed != -1:
                            textsize = cv2.getTextSize(speedText, font, cfg['3D']['Speed Text Size'][1], 2)
                            cv2.rectangle(imgBox3D,(px[0], px[1]+1),(px[0] + textsize[0][0], px[1] + textsize[0][1] *2 -1),(255,255,255),-1) 
                            cv2.putText(imgBox3D, speedText,(px[0]+1, px[1]+ textsize[0][1] + 5), font , cfg['3D']['Speed Text Size'][1], textSpeedColor,2)
        if cfg['Show Caption']:
            for id, _ in _pedest.items():
                if id in precent_PedestID:
                    xmin, ymin, xmax, ymax = _pedest[id]['locationBox'][-1]
                    caption = 'P'# + f'{str(id)}'

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    textsize = cv2.getTextSize(caption, font, cfg['3D']['Caption Text Size'][0], 2)
                    cv2.rectangle(imgBox3D,(xmin, ymin-textsize[0][1]-3), (xmin + textsize[0][0]+3, ymin),(0,200,255),-1) 
                    cv2.putText(imgBox3D, caption,(xmin+2, ymin-2), font, cfg['2D']['Caption Text Size'][0],(0,0,0),2)

    return imgBox3D
