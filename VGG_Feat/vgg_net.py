import numpy as np
import os
import caffe
from sklearn.preprocessing import normalize

def init_model(model_def, model_weights):
  # initializing model
  caffe.set_mode_cpu()
  net = caffe.Net(model_def, model_weights, caffe.TEST)
  return net

def init_transformer(net):
  averageImg = np.array([93.5940,104.7624,129.1863]);  # B-G-R
  # initializing transformer
  transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  transformer.set_mean('data', averageImg)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  return transformer

def load_images(transformer, rootPath, filelist):
  print '  [load_image]',len(filelist)
  imagelist = []
  reslist = []
  for file in filelist:
    # loading
    try:
      image = caffe.io.load_image(rootPath + file)
      # transforming
      imagelist.append(transformer.preprocess('data', image))
      reslist.append(file)
    except:
      # print '    [fail to load]',rootPath + file
      pass
      
  print '  total:',len(reslist)
  print '  done'
  return imagelist, reslist

def load_image_pairs(transformer, rootPath, descriptionFile):
  print '  [load_image from descriptionFile:', descriptionFile, ']'
  imagelist = []
  labellist = []
  acceptCounter = 0
  jumpCounter = 0
  with open(descriptionFile, 'r') as f:
    n = int(f.readline().strip())
    for label in range(n):
      print '\r    ->loading same label: %d/%d'%(label, 2 * n),
      name, num1, num2 = f.readline().strip().split('\t')
      try:
        img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join(rootPath, name, '%s_%04d.jpg'%(name, int(num1)))))
        img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join(rootPath, name, '%s_%04d.jpg'%(name, int(num2)))))
        imagelist.extend([img1, img2])
        labellist.append(1)
        acceptCounter += 1
      except:
        print 'passed...'
        jumpCounter += 1
        pass
    print '\r\n    ->finish load same:', acceptCounter, '/', jumpCounter
    acceptCounter = jumpCounter = 0
    for label in range(n):
      print '\r    ->loading diff label: %d/%d'%(n + label, 2 * n),
      name1, num1, name2, num2 = f.readline().strip().split('\t')
      try:
        img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join(rootPath, name1, '%s_%04d.jpg'%(name1, int(num1)))))
        img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join(rootPath, name2, '%s_%04d.jpg'%(name2, int(num2)))))
        imagelist.extend([img1, img2])
        labellist.append(0)
        acceptCounter += 1
      except:
        print 'passed...'
        jumpCounter += 1
        pass
    print '\r\n    ->finish load diff:', acceptCounter, '/', jumpCounter    
  print '  load image OK'
  return np.array(imagelist), np.array(labellist)
  
def extract_feature(net, imagelist, dimx = 224, dimy = 224):
  print '  [get_feature({})]: shape={}'.format(id(imagelist), imagelist.shape)
  print '  [extract_feature]'
  # setting data
  mybatch = []
  featlist = None
  step = 512
  b = 0
  while (True):
    print '    dealing batch where b:', b
    mybatch = imagelist[b:min(b + step, imagelist.shape[0])]  
    net.blobs['data'].reshape(len(mybatch), 3, dimx, dimy)
    net.blobs['data'].data[...] = mybatch
    res = net.forward()
    # getting feature
    #featlist = net.blobs['fc7'].data[0:len(imagelist), :]
    feats = net.blobs['fc7'].data
    if featlist == None:
      featlist = feats
    else:
      featlist = np.concatenate((featlist, feats))
    # forward pointer
    b += step
    if b >= imagelist.shape[0]:
      break
  print '  done'
  return featlist


def test(clf, trainx, trainy, testx, testy, **kwargs):
  clf.fit(trainx, trainy)
  acc = (clf.predict(trainx, **kwargs) == trainy).sum()
  print 'Train: {}/{}'.format(acc, trainy.shape[0])
  acc = (clf.predict(testx, **kwargs) == testy).sum()
  print 'Test: {}/{}'.format(acc, testy.shape[0])  
  
if __name__ == '__main__':
  # init model
  myNet = init_model('VGG_FACE_deploy.prototxt', 'VGG_FACE.caffemodel')
  myTransformer = init_transformer(myNet)
  AGAIN = True
  myFeature = None
  # load data
  if AGAIN == True:
    xList, yList = load_image_pairs(myTransformer, '/home/share/linjia/lfw', 'pairsDevTrain.txt')
    xListTest, yListTest = load_image_pairs(myTransformer, '/home/share/linjia/lfw', 'pairsDevTest.txt')
    myFeature = extract_feature(myNet, xList)
    myFeatureTest = extract_feature(myNet, xListTest)
    print myFeature.shape
    np.savetxt('myFeas.txt', myFeature, delimiter=',')
    myFeature = myFeature.reshape(yList.shape[0], 2, -1)
    myFeatureTest = myFeatureTest.reshape(yListTest.shape[0], 2, -1)
  else:
    print 'begin load feature...'
    myFeature = np.loadtxt('myFeas.txt', delimiter=',')
    print 'load feature OK.'
  # train classifier 
  print 'Minues & SVM...'
  from sklearn import svm
  __trainx = normalize(myFeature[:, 0, :] - myFeature[:, 1, :])
  __testx = normalize(myFeatureTest[:, 0, :] - myFeatureTest[:, 1, :])
  clf = svm.SVC(kernel='linear', probability=True, verbose=False, max_iter=100000)
  test(clf, __trainx, yList, __testx, yListTest)
  
  
  
  