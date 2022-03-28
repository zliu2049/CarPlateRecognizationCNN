import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PlateCNN:
    def __init__(self):
        self.img_w, self.img_h = 136, 36
        self.y_size = 2
        self.batch_size = 100
        self.learn_rate = 0.001

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_h, self.img_w, 3], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, self.img_h, self.img_w, 3])
        # re-arrange the image space to fit unknown samples for 36x136x3 image

        # Convolution layer 1
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 32], stddev=0.01), dtype=tf.float32)
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

        conv_out = tf.reshape(conv3, shape=[-1, 17 * 5 * 128])  # initiate the output of the convolution

        # initiate forward propagation 1
        fw1 = tf.Variable(tf.random_normal(shape=[17 * 5 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        # initiate forward propagation 2
        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        # initiate forward propagation 3
        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.y_size], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.y_size]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3

    def train(self, data_dir1, model_save_path):
        print('ready load train dataset')
        X, y = self.init_data(data_dir1)
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
        opt = tf.train.AdamOptimizer(self.learn_rate)  # initiate adaptive moment estimation
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
                _, loss = sess.run([train_step, cost], feed_dict={self.x_place: train_randx,
                                                                  self.y_place: train_randy, self.keep_place: 0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy, feed_dict={self.x_place: test_randx,
                                                        self.y_place: test_randy, self.keep_place: 1.0})

                    accuracy_plot.append(acc)
                    print(step, loss)
                    print('accuracy:' + str(acc))
                    if acc > 0.99 and step > 500:
                        saver.save(sess, model_save_path, global_step=step)
                        break
        x = np.arange(len(accuracy_plot))
        plt.plot(x, accuracy_plot)
        plt.show()

    def test(self, x_images, model_path1):
        result_list = []
        out_put = self.cnn_construct()
        predicts1 = tf.nn.softmax(out_put)  # softmax normalization
        predicts = tf.argmax(predicts1, axis=1)  # set the predicted axis
        saver = tf.train.Saver()
        # probability = tf.reduce_max(predicts, reduction_indices=[1])
        probability = tf.reduce_max(predicts1, axis=1)
        contains = no_contains = 0

        with tf.Session() as sess:  # start testing
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path1)
            preds1, probs1 = sess.run([predicts, probability],
                                      feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for n in range(len(preds1)):
                predict = preds1[n].astype(int)
                current_prob = probs1[n]
                if predict == 1:
                    result_list.append(('plate', current_prob))
                    if n >= 100:
                        print('incorrect:', n, current_prob)
                    else:
                        contains += 1
                else:
                    result_list.append(('no', current_prob))
                    if n < 100:
                        print('incorrect:', n, current_prob)
                    else:
                        no_contains += 1
            return result_list, contains, no_contains

    def list_all_files(self, root):
        files = []
        list_directory = os.listdir(root)
        for n in range(len(list_directory)):
            element = os.path.join(root, list_directory[n])
            if os.path.isdir(element):
                files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_data(self, directory):
        X = []
        y = []
        if not os.path.exists(directory):
            raise ValueError('directory not found!')
        files = self.list_all_files(directory)

        labels = [os.path.split(os.path.dirname(file))[-1] for file in files]

        for n, file in enumerate(files):
            src_img = cv2.imread(file)
            if src_img.ndim != 3:
                continue
            resize_img = cv2.resize(src_img, (136, 36))  # resize the image into 136x36 pixel
            X.append(resize_img)  # add the images to X vector
            y.append([[0, 1] if labels[n] == 'has' else [1, 0]])  # mark the image in y vector

        X = np.array(X)
        y = np.array(y).reshape(-1, 2)  # rebuild X, y matrix
        return X, y

    def init_testData(self, directory):
        test_X1 = []
        if not os.path.exists(directory):
            raise ValueError('directory not found!')
        files = self.list_all_files(directory)  # build training file list

        for file in files:
            src_img = cv2.imread(file)
            if src_img.ndim != 3:
                continue
            resize_img = cv2.resize(src_img, (136, 36))  # resize the image into 136x36 pixel
            test_X1.append(resize_img)  # add the images to test set vector

        test_X1 = np.array(test_X1)  # rebuild test set matrix
        return test_X1


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = './carIdentityData/cnn_plate_train'
    test_dir = './carIdentityData/cnn_plate_test'
    train_model_path = './carIdentityData/model/plate_recongnize/model.ckpt'
    model_path = './carIdentityData/model/plate_recongnize/model.ckpt-510'

    train_flag = 1
    net = PlateCNN()

    if train_flag == 0:
        # training
        net.train(data_dir, train_model_path)
    else:
        # testing
        test_X = net.init_testData(test_dir)
        result, yes, no = net.test(test_X, model_path)
        print('contains car plate:{0}, no car plate:{1}'.format(yes, no))
