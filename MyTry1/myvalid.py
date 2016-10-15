import caffe
import os
import sys
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize

def get_2partdata(filename, transformer, encounter):
    x = []
    y = []
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        # the same one
        for label in range(n):
            print '\rList: %d/%d'%(label, 2*n),
            name, num1, num2 = f.readline().strip().split('\t')
            try:
                img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) + str(encounter) + '.jpg')))
                img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter) + '.jpg')))
                x.extend([img1, img2])
                y.append(1)
            except:
                print 'Image not exist', name, num1, num2
                pass
            # flip part
            try:
                if (encounter >= 30 and encounter <= 35):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) + str(encounter + 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter + 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(1)
                elif (encounter >= 36 and encounter <= 41):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) +  str(encounter - 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter - 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(1)
                elif (encounter >= 48 and encounter <= 53):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) + str(encounter + 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter + 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(1)
                elif (encounter >= 54 and encounter <= 59):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) + str(encounter - 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter - 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(1)
                else:
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num1)) + str(encounter) + '_flip.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name, 'pat_%s_%04d_'%(name, int(num2)) + str(encounter) + '_flip.jpg')))
                    x.extend([img1, img2])
                    y.append(1)
            except:
                print 'Image not exist', name, num1, num2
                pass
            sys.stdout.flush()
        # the diff face
        for label in range(n):
            print '\rList: %d/%d'%(n+label, 2*n),
            name1, num1, name2, num2 = f.readline().strip().split('\t')
            try:
                img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter) + '.jpg')))
                img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter) + '.jpg')))
                x.extend([img1, img2])
                y.append(0)
            except:
                print 'Image not exist', name1, num1, name2, num2
                pass
            # flip part
            try:
                if (encounter >= 30 and encounter <= 35):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter + 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter + 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(0)
                elif (encounter >= 36 and encounter <= 41):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter - 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter - 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(0)
                elif (encounter >= 48 and encounter <= 53):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter + 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter + 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(0)
                elif (encounter >= 54 and encounter <= 59):
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter - 6) + '.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter - 6) + '.jpg')))
                    x.extend([img1, img2])
                    y.append(0)
                else:
                    img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name1, 'pat_%s_%04d_'%(name1, int(num1)) + str(encounter) + '_flip.jpg')))
                    img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join('/home/share/mylfw/', name2, 'pat_%s_%04d_'%(name2, int(num2)) + str(encounter) + '_flip.jpg')))
                    x.extend([img1, img2])
                    y.append(0)
            except:
                print 'Image not exist: ', name1, num1, name2, num2
                pass
            sys.stdout.flush()
    return np.array(x), np.array(y)

def get_feature(net, x):
    print 'Getting feature of data whose shape is {}'.format(x.shape)
    # deepid
    net.blobs['data'].reshape(*((x.shape[0], ) + net.blobs['data'].data.shape[1:]))
    net.blobs['data'].data[...] = x
    net.forward()
    return net.blobs['deepid'].data

def test(clf, trainx, trainy, testx, testy, **kwargs):
    clf.fit(trainx, trainy)
    acc = (clf.predict(trainx, **kwargs) == trainy).sum()
    print 'Train: {}/{}'.format(acc, len(trainy))
    acc = (clf.predict(testx, **kwargs) == testy).sum()
    print 'Test: {}/{}'.format(acc, len(testy))

def preprocess():
    transformerVector = []
    netVector = []
    for i in range(60):
        if i < 30:
            net = caffe.Net('./lfwvalid.deploy.prototxt', './snapshot_lfw_' + str(i) + '/_iter_9500.caffemodel', caffe.TEST)
        else:
            net = caffe.Net('./lfwvalid2.deploy.prototxt', './snapshot_lfw_' + str(i) + '/_iter_9500.caffemodel', caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 1.0)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        netVector.append(net)
        transformerVector.append(transformer)
    return netVector, transformerVector

class Threshold(object):
    def __init__(self):
        pass

    def fit(self, x, y, tol=1e-5):
        sortedx = sorted(x)
        l = 0.0
        r = 1.0
        while l+tol < r:
            lmid = (2*l+r)/3.0
            rmid = (l+2*r)/3.0
            if (self.predict(x, lmid)==y).sum() <= (self.predict(x, rmid)==y).sum():
                l = lmid
            else:
                r = rmid
        self.p = (l+r)/2.0 

    def predict(self, x, p=None):
        if p == None: p = self.p
        ind = int(x.shape[0]*p)
        if ind == x.shape[0]: ind -= 1
        threshold = sorted(x)[ind]
        y = np.zeros((x.shape[0],))
        y[x >= threshold] = 1
        return y

if __name__ == '__main__':
    netVec, transVec = preprocess()
    trainyVec = []
    testyVec = []
    trainFea = []
    testFea = []
    onceflag = False
    for i in range(60):
        print 'Round:', i
        trainx, trainy = get_2partdata('./lfw_rules/pairsDevTrain.txt', transVec[i], i)
        testx, testy = get_2partdata('./lfw_rules/pairsDevTest.txt', transVec[i], i)
        trainx = get_feature(netVec[i], trainx).reshape(trainy.shape[0], 2, -1) 
        testx = get_feature(netVec[i], testx).reshape(testy.shape[0], 2, -1)
        _trainx = normalize(trainx[:, 0, :] - trainx[:, 1, :])
        _testx = normalize(testx[:, 0, :] - testx[:, 1, :])

        for j in range(len(trainFea), len(_trainx)):
            trainFea.append([])
        for j in range(len(testFea), len(_testx)):
            testFea.append([])
        for j in range(0, len(_trainx)):
            trainFea[j].extend(_trainx[j])
        for j in range(0, len(_testx)):
            testFea[j].extend(_testx[j])
        if onceflag == False:
            trainyVec.extend(trainy)
            testyVec.extend(testy)
            onceflag = True

    from sklearn import svm
    print 'SVM Linear:'
    clf = svm.SVC(kernel='linear', probability=True, verbose=False, max_iter=50000)
    test(clf, trainFea, trainyVec, testFea, testyVec)
    try:
        print 'Cosdiff:'
        nptrainx = np.array(trainFea)
        nptrainy = np.array(trainyVec)
        nptestx = np.array(testFea)
        nptesty = np.array(testyVec)

        itrainx = (nptrainx[:, 0, :] * nptrainx[:, 1, :]).sum(1)/np.sqrt((nptrainx[:, 0, :]**2).sum(1))/np.sqrt((nptrainx[:, 1, :]**2).sum(1))
        itestx = (nptestx[:, 0, :] * nptestx[:, 1, :]).sum(1)/np.sqrt((nptestx[:, 0, :]**2).sum(1))/np.sqrt((nptestx[:, 1, :]**2).sum(1))
        clf = Threshold()
        test(clf, itrainx, trainyVec, itestx, testyVec)
    except:
        pass
