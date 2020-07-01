import numpy as np
import cv2
import os

#from detect_tf import Detector
from detect_pytorch import Detector
from face_align import *

model_path = './converted_models/slim/slimv2.pth'
network = 'slim'
long_side = 320
threshold = 0.8


src_dir = "/home/fangyu/fy/face-recognition-benchmarks/IIM/iim_dataset_registration-4827/dataset_112x112/"
dst_dir = "/home/fangyu/fy/face-recognition-benchmarks/IIM/iim_dataset_registration-4827/dataset_slimv2"

detector = Detector(model_path, long_side, network)

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

for root, dirs, files in os.walk(src_dir):
    for name in files:
        img_path = os.path.join(root, name)
        person_name = img_path.split('/')[-2]
        print(person_name)
        person_path = os.path.join(dst_dir, person_name)
        if not os.path.exists(person_path):
            os.mkdir(person_path)

        img = cv2.imread(img_path)
        dets = detector.detect(img, threshold)
        for b in dets:
            bbox = b[0:4]
            points = b[5:15].reshape((5,2))
            nimg = alignFace(img, bbox, points)
            cv2.imwrite(os.path.join(dst_dir,person_name, name), nimg)
# save img
'''
img_raw = cv2.imread("./Face_Detector_tflite/sample.jpg")
detector = Detector(model_path, long_side, network)
dets = detector.detect(img_raw, threshold)
for b in dets:
    bbox = b[0:4]
    points = b[5:15].reshape((5,2))
    #nimg = alignFace(img_raw, bbox, points)

    #cv2.imshow("test", nimg)
    #cv2.waitKey(0)
    text = "{:.4f}".format(b[4])
    b = list(map(int, b))

    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cx = b[0]
    cy = b[1] + 12
    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
cv2.imwrite("test.jpg", img_raw)
'''
