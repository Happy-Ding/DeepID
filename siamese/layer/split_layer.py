#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np

class SplitLayer(caffe.Layer):

	def setup(self, bottom, top):
		label1, label2 = bottom[0].data, bottom[1].data
		N = label1.shape[0]
		top[0].reshape(N, 1);
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		label1, label2 = bottom[0].data, bottom[1].data
		N = label1.shape[0]
		# Get similarity of each pair.
		sim = label1[:] == label2[:]
		sim = sim.reshape(N, 1)
		top[0].reshape(N, 1)
		top[0].data[...] = sim
		pass

	def backward(self, top, propagate_down, bottom):
		pass
