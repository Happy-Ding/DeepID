import os
import caffe
import lmdb
import Image as I
import numpy as np
from numpy import random
from shutil import rmtree

def getImagePackage(descriptFile, dataRoot):
    objs = []
    labels = []
    with open(descriptFile, 'r') as f:
        n = int(f.readline().strip())
        for label in range(n):
            name, encounter = f.readline().strip().split('\t')
            for pic in range(int(encounter)):
                img = np.array(I.open(os.path.join(dataRoot, name, '%s_%04d.jpg'%(name, pic + 1)))).swapaxes(0, 2) # Swap to BGR
                objs.append(img)
                labels.append(label)
    return objs, labels

def pushToLMDB(objs, labels, dbname):
    # shuffle for random the dataset
    zipped = zip(objs, labels)
    random.shuffle(zipped)
    objs, labels = zip(*zipped)
    # write LMDB
    n = len(objs)
    lmdbFile = lmdb.open(dbname, map_size=n*objs[0].nbytes*10) # 10 to enough space
    info = objs[0].shape # 39*31 from DeepID1
    with lmdbFile.begin(write=True) as txn:
        for i in range(n):
            print '\rLMDB: %d/%d'%(i, n),
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels, datum.height, datum.width = info
            datum.data = objs[i].tostring() # numpy >= 1.9: tobytes()
            datum.label = labels[i]
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

if __name__ == '__main__':
    try:
        rmtree('./train_lmdb/')
    except:
        pass
    print('Creating lmdb ...')
    os.mkdir('./train_lmdb/')
    objs, labels = getImagePackage('./lfw_rules/peopleDevTrain.txt', '/home/share/linjia/lfw/')
    pushToLMDB(objs, labels, './train_lmdb/')
    print('\tDone.')
