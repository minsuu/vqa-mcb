# import from vqa
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from time import time
import cv2
import hashlib
import caffe
import vqa_data_provider_layer
from vqa_data_provider_layer import LoadVQADataProvider
import numpy as np
import os
from skimage.transform import resize

ROOT_PATH = os.getcwd()

# constants from vqa
GPU_ID = 0
RESNET_MEAN_PATH = ROOT_PATH + "/vqa_mcb/preprocess/ResNet_mean.binaryproto"
RESNET_LARGE_PROTOTXT_PATH = ROOT_PATH + "/vqa_mcb/preprocess/ResNet-152-448-deploy.prototxt"
RESNET_CAFFEMODEL_PATH = ROOT_PATH + "/data/models/vqa/ResNet-152-model.caffemodel"
EXTRACT_LAYER = "res5c"
EXTRACT_LAYER_SIZE = (2048, 14, 14)
TARGET_IMG_SIZE = 448
VQA_PROTOTXT_PATH = ROOT_PATH +  "/data/models/vqa/multi_att_2_glove_pretrained/proto_test_batchsize1.prototxt"
VQA_CAFFEMODEL_PATH = ROOT_PATH +  "/data/models/vqa/multi_att_2_glove_pretrained/_iter_190000.caffemodel"
VDICT_PATH = ROOT_PATH + "/data/models/vqa/multi_att_2_glove_pretrained/vdict.json"
ADICT_PATH = ROOT_PATH + "/data/models/vqa/multi_att_2_glove_pretrained/adict.json"
RES_PATH = ROOT_PATH + "/vid"

resnet_mean = None
resnet_net = None
vqa_net = None
feature_cache = {}
vqa_data_provider = LoadVQADataProvider(VDICT_PATH, ADICT_PATH, batchsize=1, \
    mode='test', data_shape=EXTRACT_LAYER_SIZE)

vqa_data_provider_layer.CURRENT_DATA_SHAPE = EXTRACT_LAYER_SIZE

# mean substraction
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( RESNET_MEAN_PATH , 'rb').read()
blob.ParseFromString(data)
resnet_mean = np.array( caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3,224,224)
resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean,(1,2,0)), (448,448)),(2,0,1))

# resnet
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()

resnet_net = caffe.Net(RESNET_LARGE_PROTOTXT_PATH, RESNET_CAFFEMODEL_PATH, caffe.TEST)

# vqa net
vqa_net = caffe.Net(VQA_PROTOTXT_PATH, VQA_CAFFEMODEL_PATH, caffe.TEST)

print 'Finished setup'

def make_rev_adict(adict):
    """
    An adict maps text answers to neuron indices. A reverse adict maps neuron
    indices to text answers.
    """
    rev_adict = {}
    for k,v in adict.items():
        rev_adict[v] = k
    return rev_adict

def softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist

def vqa(d, f, q):
    start = time()
    img_feature = None
    res_path = RES_PATH + '/' + d + '/' + f + '.npz'
    print 'loading ' + res_path
    with np.load( 'res/' + d + '/' + f + '.npz' ) as f:
        img_feature = f['x']
        print 'feature loaded', img_feature.shape

        qvec, cvec, avec, glove_matrix = vqa_data_provider.create_batch(q)
        vqa_net.blobs['data'].data[...] = np.transpose(qvec,(1,0))
        vqa_net.blobs['cont'].data[...] = np.transpose(cvec,(1,0))
        vqa_net.blobs['img_feature'].data[...] = img_feature
        vqa_net.blobs['label'].data[...] = avec
        vqa_net.blobs['glove'].data[...] = np.transpose(glove_matrix, (1,0,2))
        vqa_net.forward()
        scores = vqa_net.blobs['prediction'].data.flatten()
        scores = softmax(scores)
        top_indicies = scores.argsort()[::-1][:5]
        top_answers = [vqa_data_provider.vec_to_answer(i) for i in top_indicies]
        top_scores = [float(scores[i]) for i in top_indicies]

        results = {'answers': top_answers, 'scores': top_scores, 'time': time()-start}
        return json.dumps(results)