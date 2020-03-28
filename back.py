from trans import VGGNet
import numpy as np
import tensorflow as tf
import os
from PIL import Image

learning_rate = 10
lambda_c = 0.1
lambda_s = 1000 #500

VGG_PATH = './vgg16.npy'
content_img_path = './pic/sh1.jpg'
style_img_path = './pic/vk1.jpg'

output_dir = './result'  # 输出文件夹，每一步的结果都保存
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

'''
    构建计算图的第一步需要定义一个随机化的初始图片，在这张图片上进行梯度下降最后得到想要的效果
'''


def initial_result(shape, mean, stddev):  # 对初始函数进行操作(图像大小，均值，方差)
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)  # 截断正态分布
    return tf.Variable(initial)  # 用上一步函数的结果创造一个变量


def read_img(img_name):
    img = Image.open(img_name)  # 参数是路径
    np_img = np.array(img)  # 变成np矩阵，224,224,3
    np_img = np.asarray([np_img], dtype=np.int32)  # 变为4维矩阵，封装在列表中，会把列表的列维也加进去：1,224,224,3
    return np_img


def gram_matrix(x):
    '''
    :param x: x是从某个卷积层得到的输出：[1,width,height,channel]
    :return:
    '''
    b, w, h, ch = x.get_shape().as_list()  # b=batch, get_shape(tensor) return元组
    features = tf.reshape(x, [b, h*w, ch])  # tf.reshape(tensor, shape) 将w和h合并成一个维度
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)  # 防止过大
    return gram


result = initial_result((1, 224, 224, 3), 127.5, 20)  # mean=127.5=255/2, stddev=20
content_val = read_img(content_img_path)  # 读取图像
style_val = read_img(style_img_path)  # 读取图像

# 创建placeholder，同result一起投入vggnet提取特征
content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

# 创建VGGNet
data_dict = np.load(VGG_PATH, encoding='latin1').item()  # 变为字典,没有encoding会卡死
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

# bulid函数完成VGGNet计算图的构建
vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)
# vggnet不同卷积层都可以做风格提取

content_features = [
                      vgg_for_content.conv1_1,
                       vgg_for_content.conv2_1,
                     vgg_for_content.conv3_1,
                    # vgg_for_content.conv4_3,
                    # vgg_for_content.conv5_3
                    ]

# 内容和结果提取相同层的特征
result_content_features = [
                            vgg_for_result.conv1_1,
                            vgg_for_result.conv2_1,
                            vgg_for_result.conv3_1,
                           # vgg_for_result.conv4_3,
                          #  vgg_for_result.conv5_3
                          ]

style_features = [
               #  vgg_for_style.conv1_1,
               #  vgg_for_style.conv2_1,
               #  vgg_for_style.conv3_3,
                 vgg_for_style.conv4_1,
                 vgg_for_style.conv5_1
                ]


# 风格和结果提取相同层的特征
result_style_features = [
                   #  vgg_for_result.conv1_1,
                   #  vgg_for_result.conv2_1,
                   #  vgg_for_result.conv3_3,
                     vgg_for_result.conv4_1,
                     vgg_for_result.conv5_1
]

style_gram = [gram_matrix(feature) for feature in style_features]  # 给风格图像计算gram矩阵
result_style_gram = [gram_matrix(feature) for feature in result_style_features]  # 给结果图像的风格特征计算gram矩阵


content_loss = tf.zeros(1, tf.float32)  # 内容损失，初始化为一个标量
# 可抽取多层，所以内容损失是个加和，用for
# 每一层提取的shape是：[1,width,height,channels]，在后三个维度上求平均，所以有[1,2,3]
for c, c_ in zip(content_features, result_content_features):  # zip将2个数组一一对应绑定在一起
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])  # zip([1,2],[3,4]) = [(1,3),(2,4)]

style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])

loss = content_loss * lambda_c + style_loss * lambda_s
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

'''               '''
num_steps = 200
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_value, content_loss_value, style_loss_value, _ = sess.run([loss, content_loss, style_loss, train_op],
                                                                       feed_dict={
                                                                                    content: content_val,
                                                                                    style: style_val,
                                                                                    })
        print('step: %d, loss_value: %8.4f,  content_loss: %8.4f, style_loss:%8.4f' % (step + 1, loss_value[0],
                                                                                       content_loss_value[0],
                                                                                       style_loss_value[0]))

        if step % 5 == 0:
            result_img_path = os.path.join(output_dir, 'result-%05d.jpg' % (step + 1))
            result_val = result.eval(sess)[0]
            result_val = np.clip(result_val, 0, 255)  # clip小于0设成0，大于255设成255
            img_arr = np.asarray(result_val, np.uint8)
            img = Image.fromarray(img_arr)  # np数组转成image数组
            img.save(result_img_path)