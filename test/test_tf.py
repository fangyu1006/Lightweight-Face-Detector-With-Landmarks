import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import math

GRAPH_PB_PATH = '/home/fangyu/git/Face-Detector-1MB-with-landmark/converted_models/mobilenet/mobilenet.pb'


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        #print(f.readline())
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        #tf.import_graph_def(graph_def, name='')
        tf.import_graph_def(graph_def,input_map = None,return_elements = None,name = "",op_dict = None,producer_op_list = None)
    return graph


def softmax(x, y):
    sum_ = float(math.exp(x)+math.exp(y))
    return math.exp(y)/sum_

graph = load_graph(GRAPH_PB_PATH)

for v in graph.as_graph_def().node:
    print(v.name)


#input = np.ones((1,240,320,3))
img_raw = cv2.imread("./Face_Detector_ncnn/sample.jpg")
img = np.float32(img_raw)
img -= (104,117,123)
long_side = 640
im_shape = img.shape
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
resize = float(long_side) / float(im_size_min)
if np.round(resize * im_size_max) > long_side:
    resize = float(long_side) / float(im_size_max)
if resize != 1:
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
input = []
input.append(img)
input = np.array(input)
x = graph.get_tensor_by_name('input0:0')
y = graph.get_tensor_by_name('Concat_223:0')
y1 = graph.get_tensor_by_name('Concat_198:0')
y2 = graph.get_tensor_by_name('Concat_248:0')

with tf.Session(graph=graph) as sess:
    conf, loc, landmks = sess.run((y,y1,y2), feed_dict={x:input})
    print(conf.shape)
    print(conf)
    print(loc.shape)
    print(landmks.shape)
_, n, _ = loc.shape
#loc = loc.reshape(1,1,-1,4)
#conf = conf.reshape(1,1,-1,2)
with open('tf_result.py', 'w') as fd:
    for j in range(n):
        score = softmax(conf[0,j,0], conf[0,j,1])
        fd.write(str(score)+' ' + str(loc[0,j,0]) + ' ' + str(loc[0,j,1]) + ' ' + str(loc[0,j,2]) + ' ' + str(loc[0,j,3]) + '\n')
        #fd.write(str(landmks[0,j,0]) + ' ' + str(landmks[0,j,1]) + ' ' + str(landmks[0,j,2]) + ' ' + str(landmks[0,j,3]) + '\n')
