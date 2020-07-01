import os
import torch
import numpy as np
import cv2

from models.net_rfb import RFB
from models.retinaface import RetinaFace
from data import cfg_rfb, cfg_mnet, cfg_slim


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc:storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,input_map = None,return_elements = None,name = "",op_dict = None,producer_op_list = None)
    return graph

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')

    net = load_model(net, "./converted_models/mobilenet/mobilenet0.25_Final.pth", True)
    net.eval()
    print('Finish loading model!')
    #print(net)
    #cudnn.benchmark = True
    device = torch.device("cpu")
    net = net.to(device)

    img_raw = cv2.imread("./Face_Detector_ncnn/sample.jpg")
    #img = np.ones((3,240,320), dtype=np.float32)
    img = np.float32(img_raw)
    long_side = 320
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
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    loc, conf, landms = net(img)
    
    _, n, _ = loc.shape
    loc_np = loc.data.cpu().numpy()
    conf_np = conf.data.cpu().numpy()
    landms_np = landms.data.cpu().numpy()
    with open("pytorch_result.txt", 'w') as fd:
        for j in range(n):
            fd.write(str(conf_np[0,j,1]) + ' '+str(loc_np[0, j, 0]) + ' ' + str(loc_np[0, j, 1]) + ' ' + str(loc_np[0, j, 2]) + ' ' + str(loc_np[0, j, 3]) + '\n')
            #fd.write(str(landms_np[0,j,0]) + ' ' + str(landms_np[0,j,1]) + ' ' + str(landms[0,j,2]) + ' ' + str(landms_np[0,j,3]) + ' ' + str(landms_np[0,j,4) + ' ' + str(landms_np[0,j,5]) + ' ' + str(landms_np[0,j,6]) + ' ' + str(landms_np[0,j,7]) + ' ' + str(landms_np[0,j,8]) + ' ' + str(landms_np[0,j,9]) + '\n')
            #fd.write(str(landms_np[0,j,0]) + ' ' + str(landms_np[0,j,1]) + str(landms_np[0,j,2]) + ' ' + str(landms_np[0,j,3]) + '\n')
    print(loc.shape)
    print(loc)
    print(conf)
    print(landms)



