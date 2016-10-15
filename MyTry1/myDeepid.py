import caffe
import os
import numpy as np
from caffe import layers as L
from caffe import params as P

def protoConstructor(srcPath, batchSize):
    # net define
    myNet = caffe.NetSpec()
    # Layer: input
    myNet.data, myNet.label = L.Data(batch_size = batchSize, backend = P.Data.LMDB, source = srcPath, transform_param = dict(scale = 1./255), ntop = 2)
    # Layer: conv-maxpool-1st
    myNet.conv1 = L.Convolution(myNet.data, kernel_size = 4, num_output = 20, weight_filler = dict(type='xavier'))
    myNet.pool1 = L.Pooling(myNet.conv1, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
    # Layer: conv-maxpool-2nd
    myNet.conv2 = L.Convolution(myNet.pool1, kernel_size = 3, num_output = 40, weight_filler = dict(type='xavier'))
    myNet.pool2 = L.Pooling(myNet.conv2, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
    # Layer: conv-maxpool-3rd
    myNet.conv3 = L.Convolution(myNet.pool2, kernel_size = 3, num_output = 60, weight_filler = dict(type='xavier'))
    myNet.pool3 = L.Pooling(myNet.conv3, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
    # Layer: conv-4th
    myNet.conv4 = L.Convolution(myNet.pool3, kernel_size = 2, num_output = 80, weight_filler = dict(type='xavier'))
    # Layer: full connect
    myNet.fcconv4 = L.InnerProduct(myNet.conv4, num_output = 160, weight_filler = dict(type='xavier'))
    myNet.fcpool3 = L.InnerProduct(myNet.pool3, num_output = 160, weight_filler = dict(type='xavier'))
    # Layer: deep id, use eltwise to sum up, then regularize
    myNet.rawdeepid = L.Eltwise(myNet.fcconv4, myNet.fcpool3, eltwise_param = {'operation': 1})
    myNet.deepid = L.ReLU(myNet.rawdeepid)
    # Layer: loss
    myNet.fullconnect = L.InnerProduct(myNet.deepid, num_output = 4000, weight_filler = dict(type='xavier'))
    myNet.loss = L.SoftmaxWithLoss(myNet.fullconnect, myNet.label)
    # return the prototxt
    return myNet.to_proto()

def solverConstructor(filename):
    solveParas = dict(train_net='"./lfw_deepid.prototxt"', base_lr=0.010, momentum=0.9, weight_decay=0.001,\
        lr_policy='"inv"', gamma=0.001, power=0.75, display=10, solver_mode="GPU", snapshot=100,\
        snapshot_prefix='"./snapshot/"')
    with open(filename, 'w') as f:
        for key, val in solveParas.iteritems():
            f.write('{}: {}\n'.format(key, val))
    return filename

def dash(solver, predictLayer, N):
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
    test_interval = 100
    # the main solver loop
    for it in range(N):
        # accuracy
        correct = 0
        for i in xrange(10):
            solver.net.forward()
            correct += (solver.net.blobs[predictLayer].data.argmax(1) == solver.net.blobs['label'].data).sum()
        print 'Train: [%04d]: %04d/1000=%.5f%%' % (it, correct, correct/1e3*1e2)
        # SGD by Caffe
        solver.step(test_interval)

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(1)
    with open('./lfw_deepid.prototxt', 'w') as f:
        f.write(str(protoConstructor('./lmdb', 100)))
    mySolver = caffe.SGDSolver(solverConstructor('deepidSolver.prototxt'))
    dash(mySolver, 'fullconnect', 10)

