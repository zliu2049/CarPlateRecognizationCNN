import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
           'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']


class CharCNN:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])
        # re-arrange the image space to fit unknown samples for 20x20x1 image

        # Convolution layer 1
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, filter=cw1, strides=[1, 1, 1, 1], padding='SAME'), cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        # Convolution layer 2
        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME'), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        # Convolution layer 3
        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, filter=cw3, strides=[1, 1, 1, 1], padding='SAME'), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])  # initiate the output of the convolution

        # initiate forward propagation 1
        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        # initiate forward propagation 2
        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        # initiate forward propagation 3
        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3

    def train(self, data_dir1, save_model_path):
        print('ready load train dataset')
        X, y = self.init_data(data_dir1)  # initial data
        print('success load ' + str(len(y)) + ' datas')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        # split the train data and test data at test size 0.2

        out_put = self.cnn_construct()  # construct CNN
        predicts = tf.nn.softmax(out_put)  # softmax normalization
        predicts = tf.argmax(predicts, axis=1)  # set the predicted axis
        actual_y = tf.argmax(self.y_place, axis=1)  # set the predicted axis
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        # set the rule of accuracy calculation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        # set the rule of the cost calculation
        opt = tf.train.AdamOptimizer(learning_rate=0.001)  # initiate adaptive moment estimation
        train_step = opt.minimize(cost)  # set the train step

        accuracy_plot = []

        with tf.Session() as sess:  # start training
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                _, loss = sess.run([train_step, cost], feed_dict={self.x_place: train_randx, self.y_place: train_randy,
                                                                  self.keep_place: 0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy, feed_dict={self.x_place: test_randx, self.y_place: test_randy,
                                                        self.keep_place: 1.0})
                    accuracy_plot.append(acc)
                    print(step, loss)
                    if step % 50 == 0:
                        print('accuracy:' + str(acc))
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)
                    if acc > 0.99 and step > 500:
                        saver.save(sess, save_model_path, global_step=step)
                        break
        x = np.arange(len(accuracy_plot))
        plt.plot(x, accuracy_plot)
        plt.show()

    def test(self, x_images, model_path1, test_element_list1):
        text_list = []
        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)  # softmax normalization
        predicts = tf.argmax(predicts, axis=1)  # set the predicted axis
        saver = tf.train.Saver()
        incorrect_count = 0

        with tf.Session() as sess:  # start testing
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path1)
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)
                text_list.append(self.dataset[pred])
                if test_element_list1[i] != self.dataset[pred]:
                    print(self.dataset[pred], test_element_list1[i])
                    incorrect_count += 1
            return text_list, incorrect_count, incorrect_count/len(preds), len(preds)

    def list_all_files(self, root):
        files = []
        file_list = os.listdir(root)
        for i in range(len(file_list)):
            element = os.path.join(root, file_list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_data(self, file_dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('directory not found!')
        files = self.list_all_files(file_dir)

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)  # convert each file into gray scale image
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))  # resize the image into 20x20 pixel
            X.append(resize_img)  # add the images to X vector
            file_dir = os.path.dirname(file)  # ./carIdentityData/cnn_char_train\zh_zhe
            dir_name = os.path.split(file_dir)[-1]  # zh_zhe
            vector_y = [0 for _ in range(len(self.dataset))]  # initial y vector
            index_y = self.dataset.index(dir_name)
            vector_y[index_y] = 1  # mark the image in y vector
            y.append(vector_y)  # add the y vector into y

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)  # rebuild X, y matrix

        return X, y

    def init_testData(self):
        test_X1 = []
        test_element_list1 = []
        if not os.path.exists(test_dir):
            raise ValueError('directory not found!')
        files = self.list_all_files(test_dir)  # build training file list

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)  # convert each file into gray scale image
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))  # resize the image into 20x20 pixel
            test_X1.append(resize_img)  # add the images to test set vector

            file_dir = os.path.dirname(file)  # ./carIdentityData/cnn_char_train\zh_zhe
            dir_name = os.path.split(file_dir)[-1]  # zh_zhe
            test_element_list1.append(dir_name)

        test_X1 = np.array(test_X1)  # rebuild test set matrix
        return test_X1, test_element_list1


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = './carIdentityData/cnn_char_train'
    test_dir = './carIdentityData/cnn_char_test'
    train_model_path = './carIdentityData/model/char_recongnize/model.ckpt'
    model_path = './carIdentityData/model/char_recongnize/model.ckpt-600'

    train_flag = 1
    net = CharCNN()

    if train_flag == 0:
        # training
        net.train(data_dir, train_model_path)
    else:
        # testing
        test_X, test_element_list = net.init_testData()
        text, incorrect, incorrect_rate, total = net.test(test_X, model_path, test_element_list)
        print('total:', total, ' incorrect:', incorrect, ' rate:', 1-incorrect_rate)
        print(text)
