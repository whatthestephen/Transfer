import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class VGGNet:
    '''构建VGG网络结构，从模型文件中读取参数'''
    def __init__(self, data_dict):  # 读入的模型文件
        self.data_dict = data_dict  # 变成成员变量，在类中共享

    def get_conv_filter(self, name):  # 在data_dict中将卷积参数抽取出来的函数
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name): # 抽取全连接网络
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        '''构建卷积层'''
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h

    def pooling_layer(self, x, name):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        '''构建全连接层'''
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            else:
                return activation(h)

    def flatten_layer(self, x, name):
        with tf.name_scope(name):
            # batch_size, 宽，高，深
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d

            x = tf.reshape(x, [-1, dim])
            return x

    def build(self, x_rgb):
        '''构建VGG'''
        print("开始：")
        start_time = time.time()
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        print("finished: %4ds" % (time.time() - start_time))
'''
        self.flatten5 = self.flatten_layer(self.pool5, 'flatten')
        self.fc6 = self.fc_layer(self.flatten5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        self.prob = tf.nn.softmax(self.fc8, name='prob')
'''






