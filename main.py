import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
import urllib.request
import numpy as np
import streamlit as st
st.title("Object Recognition")
frame_place_holder = st.empty()
model = YOLO("yolov10n.pt")
cam1 = 'http://192.168.1.234/320x240.mjpeg'
alpha = 2
beta = 10
cap = cv.VideoCapture(cam1)
cap.set(3,640)
cap.set(4,840)
classnames = [
        "Human", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush","pen","pencil","book","chip","battery"
]
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()
while True:
    img_resp = urllib.request.urlopen(cam1)
    img = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    cam = cv.imdecode(img, -1)
    result = model(cam,stream=True)
    for r in result:
        box = r.boxes
        for b in box:
            x1,y1,x2,y2 = b.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(cam,(x1,y1,w,h))
            conf = math.ceil((b.conf[0]*100))
            cls = int(b.cls[0])
            cvzone.putTextRect(cam,f"{classnames[cls]} {conf}%",(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
    cam = cv.cvtColor(cam,cv.COLOR_BGR2RGB)

    cam = cv.convertScaleAbs(cam,alpha=alpha, beta = beta)
    frame_place_holder.image(cam,channels='RGB')
    cv.waitKey(1)
cap.release()
cv.destroyAllWindows()