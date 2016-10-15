import caffe
import os
import sys
import numpy as np

def get_2partdata(filename, transformer):
    x = []
    y = []
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        # the same one
        for label in range(n):
            print '\rList: %d/%d'%(label, 2*n),
            name, num1, num2 = f.readline().strip().split('\t')
            img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/lfw/', name, '%s_%04d.jpg'%(name, int(num1)))))
            img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/lfw/', name, '%s_%04d.jpg'%(name, int(num2)))))
            x.extend([img1, img2])
            y.append(1)
            sys.stdout.flush()
        # the diff face
        for label in range(n):
            print '\rList: %d/%d'%(n+label, 2*n),
            name1, num1, name2, num2 = f.readline().strip().split('\t')
            img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/lfw/', name1, '%s_%04d.jpg'%(name1, int(num1)))))
            img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/lfw/', name2, '%s_%04d.jpg'%(name2, int(num2)))))
            x.extend([img1, img2])
            y.append(0)
            sys.stdout.flush()
    return np.array(x), np.array(y)

def get_feature(net, x):
    print 'Getting feature of data whose shape is {}'.format(x.shape)
    # deepid
    net.blobs['data'].reshape(*((x.shape[0], ) + net.blobs['data'].data.shape[1:]))
    net.blobs['data'].data[...] = x
    net.forward()
    #print net.blobs['deepid'].data
    return net.blobs['deepid'].data

def test(clf, trainx, trainy, testx, testy, **kwargs):
    acc = (clf.predict(trainx, **kwargs) == trainy).sum()
    print 'Train: {}/{}'.format(acc, trainy.shape[0])

    acc = (clf.predict(testx, **kwargs) == testy).sum()
    print 'Test: {}/{}'.format(acc, testy.shape[0])


if __name__ == '__main__':
    net = caffe.Net('./lfw_multiclass.deploy.prototxt', '/home/share/shaofan/lfw_caffe/snapshot/_iter_800.caffemodel', caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    trainx, trainy = get_2partdata('./lfw_rules/pairsDevTrain.txt', transformer)
    testx, testy= get_2partdata('./lfw_rules/pairsDevTest.txt', transformer)
    
    trainx = get_feature(net, trainx).reshape(trainy.shape[0], 2, -1) 
    testx = get_feature(net, testx).reshape(testy.shape[0], 2, -1) 
    print trainx.shape
    print testx.shape

    __trainx = trainx[:, 0, :] - trainx[:, 1, :]
    __testx = testx[:, 0, :] - testx[:, 1, :]
    from sklearn import svm
    clf = svm.SVC(kernel='linear', probability=True, verbose=True, max_iter=3000)
    clf.fit(__trainx, trainy)
    test(clf, __trainx, trainy, __testx, testy)
