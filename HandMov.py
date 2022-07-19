from enum import Flag
from unittest import result
import cv2
import mediapipe as mp
import time
import math


class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity=modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.maxHands,self.modelComplexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]


    def findHands(self,img,draw=True):
    
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handl in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handl, self.mphands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self,img,handNo=0,draw=True):

        self.lmlist=[]
        xlist=[]
        ylist=[]
        bbox=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),-1)
            xmin,xmax=min(xlist),max(xlist)
            ymin,ymax=min(ylist),max(ylist)
            bbox=xmin,ymin,xmax,ymax

            if draw:
                cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,255,0),2)

        return self.lmlist,bbox

    def fingerUps(self):
        fingers=[]
        if self.lmlist[self.tipIds[0]][1]>self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2]<self.lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers


    def findDis(self,p1,p2,img,draw=True,r=15,t=3):
        x1,y1=self.lmlist[p1][1:]
        x2,y2=self.lmlist[p2][1:]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),t)
            cv2.circle(img,(x1,y1),r,(255,0,255),-1)
            cv2.circle(img,(x2,y2),r,(255,0,255),-1)
            cv2.circle(img,(cx,cy),r,(0,0,255),-1)
        length=math.hypot(x1-x2,y1-y2)
        return length,img,[x1,x2,y1,y2,cx,cy]
            
            



def main():

    cap = cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        _, frame = cap.read()
        frame=detector.findHands(frame)
        lmlist=detector.findPosition(frame)
        if len(lmlist)!=0:
            print(lmlist[0])

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
