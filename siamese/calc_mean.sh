#!/bin/bash
./../caffe/build/tools/compute_image_mean "siamese_64/webface_pair1_train" "siamese_64/mean1.proto"
./../caffe/build/tools/compute_image_mean "siamese_64/webface_pair2_train" "siamese_64/mean2.proto"
