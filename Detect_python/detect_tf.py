import tensorflow as tf
import cv2
import numpy as np

from itertools import product as product
from math import ceil

from config import cfg_mnet, cfg_slim, cfg_rfb

def load_graph(file_name):
    with tf.gfile.GFile(file_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

class Detector():
    def __init__(self, model_path, long_side, network):
        if network == 'mobilenet':
            self.cfg = cfg_mnet
        elif network == 'slim':
            self.cfg = cfg_slim
        elif network == 'RFB':
            self.cfg = cfg_rfb
        else:
            print("not supported network!!")
            exit(0)

        self.min_sizes = self.cfg['min_sizes']
        self.steps = self.cfg['steps']
        self.variances = self.cfg['variance']

        self.long_side = long_side
        
        self.graph = load_graph(model_path)
        self.input_tensor = self.graph.get_tensor_by_name('input0:0')
        self.conf_tensor = self.graph.get_tensor_by_name('Concat_223:0')
        self.loc_tensor = self.graph.get_tensor_by_name('Concat_198:0')
        self.landms_tensor = self.graph.get_tensor_by_name('Concat_248:0')

    def detect(self,img_raw, threshold):
        img = np.float32(img_raw)

        target_size = self.long_side
        max_size = self.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)

        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)


        with tf.Session(graph=self.graph) as sess:
            conf, loc, landms = sess.run((self.conf_tensor,self.loc_tensor,self.landms_tensor),
                    feed_dict={self.input_tensor:[img]})


        conf = conf.reshape((-1, 2))
        conf = self.softmax(conf)
        loc = loc.reshape((-1, 4))
        landms = landms.reshape((-1, 10))

        priors = self.createAnchors(image_size=(im_height, im_width))
        boxes = self.decode(loc, priors, self.variances)
        scale = np.array([im_width, im_height, im_width, im_height])
        boxes = boxes * scale / resize

        scores = conf[:, 1]

        landms = self.decode_landm(landms, priors, self.variances)
        scale1 = np.array([im_width, im_height, im_width, im_height, im_width, 
                           im_height, im_width, im_height, im_width, im_height])
        landms = landms * scale1 / resize

        # ignore low socres
        inds = np.where(scores > threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        return dets


    def createAnchors(self,  image_size = None):
        feature_maps = [[ceil(image_size[0]/float(step)), ceil(image_size[1]/float(step))] for step in self.steps]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(int(f[0])), range(int(f[1]))):
                for min_size in min_sizes:
                    s_kx = float(min_size) / float(image_size[1])
                    s_ky = float(min_size) / float(image_size[0])
                    dense_cx = [x * self.steps[k] / float(image_size[1]) for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / float(image_size[0]) for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.array(anchors)
        return anchors.reshape(-1,4)


    def decode(self, loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), 1) 
        return landms

    def py_cpu_nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def softmax(self, x):
        e_x = np.exp(x)
        result = np.zeros(e_x.shape)
        result[:, 0] = e_x[:, 0]/e_x.sum(axis=1)
        result[:, 1] = e_x[:, 1]/e_x.sum(axis=1)
        return result

