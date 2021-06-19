import numpy as np
import tensorflow as tf

class ResnetParam():
    def __init__(self):
        self.input_shape = []
        self.num_classes = 10
        self.num_layers = 18
        self.learning_rate = 1e-4
        self.logdir = ''
        self.num_epochs = 100

class ResnetModel():
    def __init__(self, param):
        self.num_epochs = param.num_epochs
        self.batch_size = param.input_shape[0]
        self.logdir = param.logdir

        self.ph_x = tf.placeholder(tf.float32, param.input_shape)
        self.ph_y = tf.placeholder(tf.int32, [self.batch_size])

        self.pred = prediction(self.ph_x, param.num_classes, param.num_layers)
        self.loss, self.accuracy = fit(self.pred, self.ph_y, param.num_classes)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=param.learning_rate,
            name='Adam').minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    #     self.writer = tf.summary.FileWriter(param.logdir, self.sess.graph)
    #     self.summary_settings()


    # def summary_settings(self):
    #     self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
    #     self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)
    #     self.summary_train = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])

    #     self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
    #     self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)
    #     self.summary_test = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])


    def train(self, train_data, validation_data):
        train_data_length = train_data[0].shape[0]
        train_idx_table = np.arange(train_data_length)

        test_data_length = validation_data[0].shape[0]
        test_idx_table = np.arange(test_data_length)

        iteration = train_data_length // self.batch_size

        for epoch in range(self.num_epochs):
            indeces = train_idx_table.copy()
            np.random.shuffle(indeces)
            x_train, y_train = train_data

            for i in range(iteration):
                s = i * self.batch_size; e = s + self.batch_size
                x = x_train[indeces[s:e]]; y = y_train[indeces[s:e]]

                _, loss, acc = self.sess.run(
                    [self.optimizer, self.loss, self.accuracy],
                    feed_dict={self.ph_x:x, self.ph_y:y})

                print(f'\r{i} / {iteration}, loss:{loss:.3f}, acc:{acc:.3f}', end='')

            print(f'{epoch} / {self.num_epochs}, ', end='')

            indeces = np.random.choice(train_idx_table)
            x_train, y_train = train_data

            x = x_train[indeces]; y = y_train[indeces]
            loss, acc = self.sess.run(
                [self.loss, self.accuracy],
                feed_dict={self.ph_x:x, self.ph_y:y})
            # summary_str, loss, acc = self.sess.run(
            #     [self.summary_train, self.loss, self.accuracy],
            #     feed_dict={self.ph_x:x, self.ph_y:y})
            # self.writer.add_summary(summary_str, epoch)

            print(f'train_loss:{loss:.3f}, train_acc:{acc:.3f}, ', end='')

            indeces = np.random.choice(test_idx_table)
            x_test, y_test = validation_data

            x = x_test[indeces]; y = y_test[indeces]
            loss, acc = self.sess.run(
                [self.loss, self.accuracy],
                feed_dict={self.ph_x:x, self.ph_y:y})
            # summary_str, loss, acc = self.sess.run(
            #     [self.summary_test, self.loss, self.accuracy],
            #     feed_dict={self.ph_x:x, self.ph_y:y})
            # self.writer.add_summary(summary_str, epoch)

            print(f'val_loss:{loss:.3f}, val_acc:{acc:.3f}')

            self.saver.save(self.sess, f'{self.logdir}/model.ckpt', global_step=epoch)


def fit(y, labels, units):
    one_hot_labels = tf.one_hot(labels, units)
    ent = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y)
    loss = tf.reduce_mean(ent)
    pred = tf.equal(tf.argmax(y, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    return loss, accuracy


def prediction(x, num_classes, num_layers=18):
    y_pred = resnet(x, num_classes, num_layers)
    # y_pred = resnet_sample(x, num_classes, num_layers)
    return y_pred

def resnet_sample(x, units, num_layers, base_ch=64):
    x = conv(x, 3, base_ch, scoop='conv_0')
    x = global_average_pooling(x)
    # x = convn(x, 1, base_ch, scoop='convn_0')
    x = conv(x, 1, base_ch, scoop='conv_1')
    x = batch_norm(x)
    x = relu(x)
    x = fully_connected(x, units)
    x = softmax(x)
    return x

def resnet(x, units, num_layers, base_ch=64):
    x = conv(x, 3, base_ch)
    x = max_pooling(x, 3, 1)

    if num_layers == 18:
        iterations = [2, 2, 2, 2]
        for i, ite in enumerate(iterations):
            ch = base_ch * (2**i)
            x = residual_block_plain(x, ch, ite, scoop=f'res_block_{i}')

    elif num_layers == 34:
        iterations = [3, 4, 6, 3]
        for i, ite in enumerate(iterations):
            ch = base_ch * (2**i)
            x = residual_block_plain(x, ch, ite, scoop=f'res_block_{i}')

    elif num_layers == 50:
        iterations = [3, 4, 6, 3]
        for i, ite in enumerate(iterations):
            ch = base_ch * (2**i)
            x = residual_block_bottleneck(x, ch, ite, scoop=f'res_block_{i}')

    elif num_layers == 101:
        iterations = [3, 4, 23, 3]
        for i, ite in enumerate(iterations):
            ch = base_ch * (2**i)
            x = residual_block_bottleneck(x, ch, ite, scoop=f'res_block_{i}')

    elif num_layers == 152:
        iterations = [3, 8, 36, 3]
        for i, ite in enumerate(iterations):
            ch = base_ch * (2**i)
            x = residual_block_bottleneck(x, ch, ite, scoop=f'res_block_{i}')

    else:
        return None

    x = global_average_pooling(x)
    x = fully_connected(x, units)
    x = softmax(x)

    return x


def residual_block_bottleneck(x, o_channel, iteration, scoop='res_block_bottleneck'):
    with tf.variable_scope(scoop):
        for i in range(iteration):
            x = convn(x, 1, o_channel, scoop=f'{scoop}_{i}_f')
            x = convn(x, 3, o_channel, scoop=f'{scoop}_{i}_f')
            x = convn(x, 1, o_channel*4, scoop=f'{scoop}_{i}_f')
    return x


def residual_block_plain(x, o_channel, iteration, scoop='res_block_plain'):
    with tf.variable_scope(scoop):
        for i in range(iteration):
            x = convn(x, 3, o_channel, scoop=f'{scoop}_{i}_f')
            x = convn(x, 3, o_channel, scoop=f'{scoop}_{i}_l')
    return x


def convn(x, k_size, o_channel, scoop='convn'):
    with tf.variable_scope(scoop):
        x = conv(x, k_size, o_channel, scoop)
        x = batch_norm(x)
        x = relu(x)
    return x


def fully_connected(x, units, scoop='fully_connected'):
    with tf.variable_scope(scoop):
        x = tf.compat.v1.layers.dense(
            x,
            units,
            kernel_initializer=tf.initializers.truncated_normal())
    return x


def conv(x, k_size, o_channel, scoop='conv'):
    input_shape = x.get_shape()
    shape = [k_size, k_size, int(input_shape[-1]), o_channel]

    with tf.variable_scope(scoop):
        w_k = tf.get_variable(
            'w', shape=shape, initializer=tf.initializers.variance_scaling(),
            # regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True)
        w_b = tf.get_variable(
            'b', shape=[o_channel], initializer=tf.initializers.constant(0.1),
            trainable=True)
        x = tf.nn.conv2d(x, filter=w_k, padding='SAME') + w_b
    return x


def batch_norm(x, scoop='batch_norm'):
    input_shape = x.get_shape()

    with tf.variable_scope(scoop):
        mean = tf.get_variable(
            'mean',
            shape=[input_shape[-1]],
            initializer=tf.initializers.constant(0),
            trainable=True) 
        variance = tf.get_variable(
            'variance',
            shape=[input_shape[-1]],
            initializer=tf.initializers.constant(1.0),
            trainable=True) 
        gamma = tf.get_variable(
            'gamma',
            shape=[input_shape[-1]],
            initializer=tf.initializers.constant(1.0),
            trainable=True) 
        beta = tf.get_variable(
            'beta',
            shape=[input_shape[-1]],
            initializer=tf.initializers.constant(0),
            trainable=True)
        epsilon = 1e-5

        x = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)
    return x


def relu(x, scoop='relu'):
    with tf.variable_scope(scoop):
        x = tf.nn.relu(x)
    return x


def max_pooling(x, k_size, strides=2, scoop='max_pooling'):
    with tf.variable_scope(scoop):
        x = tf.nn.max_pool2d(x, k_size, strides, padding='VALID')
    return x


def global_average_pooling(x, scoop='global_average_pooling'):
    with tf.variable_scope(scoop):
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return x


def softmax(x, scoop='softmax'):
    with tf.variable_scope(scoop):
        x = tf.nn.softmax(x)
    return x
