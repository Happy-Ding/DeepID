import struct
import os
import vgg_net
	
if __name__ == '__main__':
  model_root = '/home/zhijian/code/vgg/'
  model_def = model_root + 'deploy.prototxt'
  model_weights = model_root + 'vgg_face.caffemodel'
  net = vgg_net.init_model(model_def, model_weights)
  transformer = vgg_net.init_transformer(net)
  
  mappinglist = []# [[(personId), (fileId), (image file name), (feature file name)],...]
  person2id = {}
  personCnt = 0
  fileCnt = 0
  image_root = '/home/share/zhijian/vggface_detected/'
  feat_root = '/home/share/zhijian/vggface_feature/'
  peoplelist = os.listdir(image_root)
  # loop through root directory of each person
  for person in peoplelist:
    if person not in person2id:
      person2id[person] = personCnt
      personCnt += 1
    personId = person2id[person]
    print '\nProcess@',personId,person

    # image files'root directory
    curpath = image_root + person + '/'
    # feature files'root directory
    dstpath = feat_root + person + '/'
    filelist = os.listdir(curpath)
    # some directories do not contain file due to download failure
    if len(filelist) == 0:
      continue

    # load and transform image files
    imagelist, filelist = vgg_net.load_images(transformer, curpath, filelist)
    # extract feature
    featlist = vgg_net.extract_feature(net, imagelist)
    
    print '  [writeback feature]'
    # write feature vector into file
    for i in xrange(len(filelist)):
      filename = filelist[i]
      featname = filename[:filename.index('.')] + '.ft'
      # original filename contains face index, we should filter it out
      pre = filename[:filename.index('_')]
      pos = filename[filename.index('.'):]
      filename = pre + pos
      feat = featlist[i]
      f = open(dstpath+featname, 'w')
      for j in feat: f.write('%s ' % j)
      f.close()
      mapping = [personId, fileCnt, filename, featname]
      mappinglist.append(mapping)
      fileCnt = fileCnt + 1
    print '  done'
  print 'done'

  print 'writeback info'
  #write image file information into file
  f = open(feat_root+'featInfo', 'wb')
  tot = struct.pack('i', fileCnt)
  f.write(tot)
  for mapping in mappinglist:
    lenfile = len(mapping[2])
    lenfeat = len(mapping[3])
    fmt = 'iiii%ss%ss' % (lenfile, lenfeat)
    bytes = struct.pack(fmt, mapping[0], mapping[1], lenfile, lenfeat, mapping[2][:lenfile], mapping[3][:lenfeat])
    f.write(bytes)
  f.close()
  print 'done'

  print 'writeback id2name file'
  #write id2name information into file
  f = open(feat_root+'id2name', 'wb')
  tot = struct.pack('i', personCnt)
  f.write(tot)
  for key in person2id.keys():
    lenname = len(key)
    fmt = 'ii%ss' % lenname
    bytes = struct.pack(fmt, person2id[key], lenname, key)
    f.write(bytes)
  f.close()
  print 'done'
