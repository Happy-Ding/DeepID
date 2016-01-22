import os
import caffe
import lmdb
import Image as I
import numpy as np
from numpy import random
from shutil import rmtree

def getImagePackage(descriptFile, dataRoot, newSize):
    objs = []
    labels = []
    with open(descriptFile, 'r') as f:
        n = int(f.readline().strip())
        for label in range(n):
            pid, name = f.readline().strip().split('\t')
            for parent, dirnames, filenames in os.walk(os.path.join(dataRoot, pid)):
                for filename in filenames:
                    img = np.array(I.open(os.path.join(parent, filename)).resize(newSize)).swapaxes(0, 2) # Swap to BGR
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
        rmtree('./lmdb_s/')
    except:
        pass
    print('Creating sasia lmdb ...')
    os.mkdir('./lmdb/')
    objs, labels = getImagePackage('./sasia_names.txt', '/home/share/sasia/CASIA-WebFace' ,(39, 31))
    pushToLMDB(objs, labels, './lmdb_s/')
    print('\tDone.')
