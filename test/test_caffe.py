import os
import caffe
import cv2
import numpy as np

caffe.set_mode_cpu()
net = caffe.Net("./converted_models/mobilenet/mobilenet.prototxt", "./converted_models/mobilenet/mobilenet.caffemodel", caffe.TEST)
net.blobs['input0'].reshape(1, 3, 480, 640)
tmp_batch = np.zeros([1, 3, 480, 640], dtype=np.float32)

img_raw = cv2.imread("./Face_Detector_ncnn/sample.jpg")
img = np.float32(img_raw)
long_side = 640
im_shape = img.shape
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
resize = float(long_side) / float(im_size_min)
if np.round(resize * im_size_max) > long_side:
    resize = float(long_side) / float(im_size_max)
if resize != 1:
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
img -= (104, 117, 123)
img = img.transpose(2,0,1)
tmp_batch[0, :, :, :] = img

net.blobs['input0'].data[...] = tmp_batch
scores = net.forward()['586'][0]
boxes = net.forward()['output0'][0]
landmarks = net.forward()['585'][0]

print(scores.shape)
print(boxes.shape)
print(landmarks.shape)
n, _ = scores.shape
with open("caffe_result.txt", 'w') as fd:
    for i in range(n):
        fd.write(str(scores[i][1]) + " " + str(boxes[i][0]) + " " + str(boxes[i][1]) + " " + str(boxes[i][2]) + " " + str(boxes[i][3]) + "\n")
