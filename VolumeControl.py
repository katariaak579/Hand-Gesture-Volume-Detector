import cv2
from cv2 import VideoCapture
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import mediapipe as mp
import HandMov as hm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# VARIABLES

wcam=1280
hcam=720
detector=hm.HandDetector(detectionCon=0.7)

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,wcam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,hcam)

# PYCAW
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)
minvol=volRange[0]
maxvol=volRange[1]
vol=0
volBar=400
volPercent=0

while True:
    _,frame=cap.read()
    frame=detector.findHands(frame)
    lmlist=detector.findPosition(frame,draw=True)
    if len(lmlist)!=0:
        # print(lmlist[4],lmlist[8])    

        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv2.circle(frame,(x1,y1),10,(0,255,255),-1)
        cv2.circle(frame,(x2,y2),10,(0,255,255),-1) 
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.circle(frame,(cx,cy),10,(0,0,255),-1) 
        length=math.hypot(x2-x1,y2-y1)
        print(length)

        vol=np.interp(length,[50,300],[minvol,maxvol])
        volBar=np.interp(length,[50,300],[400,150])
        volPercent=np.interp(length,[50,300],[0,100])
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(frame,(cx,cy),10,(0,255,0),-1) 
        
        cv2.rectangle(frame,(50,150),(85,400),(255,0,0),3)
        cv2.rectangle(frame,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
        cv2.putText(frame,f'{int(volPercent)}%',(40,450),cv2,FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
         

    cv2.imshow("Cam",frame)

    if cv2.waitKey(1)==ord("q"):
        break