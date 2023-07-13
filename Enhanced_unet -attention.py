import tensorflow as tf
from tensorflow import keras
from layers.attention import PAM, CAM
# form keras import backend as K

import numpy as np
import os
# %matplotlib inline
import matplotlib.pyplot as plt
import glob

img = glob.glob('D:\\SAT\\Semantic_Seg_rygh\\remote_sensing_image\\train\\src\\*.png')   # tf.io.glob.glob
label = glob.glob('D:\\SAT\\Semantic_Seg_rygh\\remote_sensing_image\\train\\label\\*.png')

train_count = len(label)
index = np.random.permutation(len(img))

img = np.array(img)[index]
label = np.array(label)[index]

img_val = glob.glob('D:\\SAT\\Semantic_Seg_rygh\\remote_sensing_image\\train\\test_src\\*.png')
label_val = glob.glob('D:\\SAT\\Semantic_Seg_rygh\\remote_sensing_image\\train\\test_label\\*.png')

val_count = len(img_val)

dataset_train = tf.data.Dataset.from_tensor_slices((img, label))
dataset_val = tf.data.Dataset.from_tensor_slices((img_val, label_val))

def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img

def read_png_label(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img

def normal(img, mask):
    img = tf.cast(img, tf.float32)/127.5 -1
    mask = tf.cast(mask, tf.int32)
    return img, mask

def load_image_train(img_path, mask_path):
    img = read_png(img_path)
    mask = read_png_label(mask_path)

    img, mask = normal(img, mask)
    return img, mask


def load_image_val(img_path, mask_path):
    img = read_png(img_path)
    mask = read_png_label(mask_path)

    img = tf.image.resize(img, (256, 256))
    mask = tf.image.resize(mask, (256, 256))

    img, mask = normal(img, mask)

    return img, mask

BCTCH_SIZE = 4
BUFFER_SIZE = 100
step_per_epoch = train_count//BCTCH_SIZE
val_step = val_count//BCTCH_SIZE

auto = tf.data.experimental.AUTOTUNE

dataset_train = dataset_train.map(load_image_train, num_parallel_calls=auto)
dataset_val = dataset_val.map(load_image_val, num_parallel_calls=auto)

dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BCTCH_SIZE)
dataset_val = dataset_val.batch(BCTCH_SIZE)

class Downsample(keras.layers.Layer):
    def __init__(self, units):
        super(Downsample, self).__init__()
        self.conv1 = keras.layers.Conv2D(units, kernel_size=3, padding='same')
        self.conv2 = keras.layers.Conv2D(units, kernel_size=3, padding='same')
        self.pool = keras.layers.MaxPooling2D()
    def call(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        return x


class Upsample(keras.layers.Layer):
    def __init__(self, unit):
        super(Upsample, self).__init__()
        self.conv1 = keras.layers.Conv2D(unit, kernel_size=3, padding='same')
        self.conv2 = keras.layers.Conv2D(unit, kernel_size=3, padding='same')
        self.deconv = keras.layers.Conv2DTranspose(unit // 2,
                                                   kernel_size=3,
                                                   strides=2,
                                                   padding='same')

    def call(self, x, is_pool=True):
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.deconv(x)
        x = tf.nn.relu(x)
        return x


class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.down1 = Downsample(64)
        self.down2 = Downsample(128)
        self.down3 = Downsample(256)
        self.down4 = Downsample(512)
        self.down5 = Downsample(1024)

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_last = Downsample(64)

        self.up = keras.layers.Conv2DTranspose(512,
                                               kernel_size=3,
                                               strides=2,
                                               padding='same')

        self.last = keras.layers.Conv2D(5,
                                        kernel_size=1,
                                        padding='same')


        # self.up4 = keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same')
        self.mask = keras.layers.Conv2D(1,
                                        kernel_size=1,
                                        padding='same')

    def Enhanced_Module(self, input_feature, kernel_size=5):

        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis = -1, keepdims = True))(input_feature)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis= -1, keepdims = True))(input_feature)

         # = tf.keras.layers.concatenate([avg_pool, max_pool])(axis = -1)

        concate = tf.concat([avg_pool, max_pool], axis=-1)
        cmba_feature = keras.layers.Conv2D(1,
                                           kernel_size=kernel_size,
                                           strides=1,
                                           padding='same',
                                           activation='sigmoid',
                                           kernel_initializer='he_normal',
                                           use_bias=False)(concate)
        return tf.keras.layers.multiply([input_feature,cmba_feature])

    def call(self, x):
        x1 = self.down1(x, is_pool=False)  # 256*256*64
        x1 = self.Enhanced_Module(x1)

        x2 = self.down2(x1)  # 128*128*128
        x2 = self.Enhanced_Module(x2)

        x3 = self.down3(x2)  # 64*64*256
        x3 = self.Enhanced_Module(x3)

        x4 = self.down4(x3)  # 32*32*512
        x4 = self.Enhanced_Module(x4)

        x5 = self.down5(x4)  # 16*16*1024

        ###attention
        reduce_conv5_3 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x5)
        reduce_conv5_3 = tf.keras.layers.BatchNormalization(axis=3)(reduce_conv5_3)
        reduce_conv5_3 = tf.keras.layers.Activation('relu')(reduce_conv5_3)

        pam = PAM()(reduce_conv5_3)
        pam = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
        pam = tf.keras.layers.BatchNormalization(axis=3)(pam)
        pam = tf.keras.layers.Activation('relu')(pam)
        pam = tf.keras.layers.Dropout(0.5)(pam)
        pam = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

        cam = CAM()(reduce_conv5_3)
        cam = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
        cam = tf.keras.layers.BatchNormalization(axis=3)(cam)
        cam = tf.keras.layers.Activation('relu')(cam)
        cam = tf.keras.layers.Dropout(0.5)(cam)
        cam = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

        feature_sum = tf.keras.layers.add([pam, cam])
        feature_sum = tf.keras.layers.Dropout(0.5)(feature_sum)
        feature_sum = tf.keras.layers.Conv2D(1024, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(feature_sum)
        feature_sum = tf.keras.layers.Activation('relu')(feature_sum)

        #######
        x5 = self.up(feature_sum)  # 32*32*512

        x5 = tf.concat([x4, x5], axis=-1)  # 32*32*1024
        x5 = self.up1(x5)  # 64*64*256
        x5 = self.Enhanced_Module(x5)

        x5 = tf.concat([x3, x5], axis=-1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = self.Enhanced_Module(x5)

        x5 = tf.concat([x2, x5], axis=-1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = self.Enhanced_Module(x5)

        x5 = tf.concat([x1, x5], axis=-1)  # 256*256*128
        x5 = self.conv_last(x5, is_pool=False)  # 256*256*64
        x5 = self.Enhanced_Module(x5)

        seg = self.last(x5)  # 256*256*5
        mask = self.mask(seg)

        return seg, mask

model = Net()

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_cont = tf.keras.losses.MeanSquaredLogarithmicError( )

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou = MeanIoU(34, name='train_iou')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_iou = MeanIoU(34, name='test_iou')


def train_step(images, labels):
    with tf.GradientTape(persistent=True) as tape:
        Classify, mask = model(images)  # Classify256*256*5  Contant256*256*1

        Classify_loss = loss_object(labels, Classify)

        mask_loss = loss_cont(labels, mask)

        t_loss = Classify_loss + mask_loss

    gradients = tape.gradient(t_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(Classify_loss)
    train_accuracy(labels, Classify)
    train_iou(labels, Classify)


def test_step(images, labels):
    Classify, mask = model(images)

    t_loss = loss_object(labels, Classify)

    test_loss(t_loss)
    test_accuracy(labels, Classify)
    test_iou(labels, Classify)


EPOCHS = 50

acc = np.zeros((50, 6)).astype(np.float32)

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_iou.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_iou.reset_states()

    for images, labels in dataset_train:
        train_step(images, labels)

    for test_images, test_labels in dataset_val:
        test_step(test_images, test_labels)

    template = 'Epoch {:.3f}, Loss: {:.3f}, Accuracy: {:.3f}, \
                IOU: {:.3f}, Test Loss: {:.3f}, \
                Test Accuracy: {:.3f}, Test IOU: {:.3f}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_iou.result(),
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_iou.result()
                          ))
    acc[epoch, :] = [train_loss.result(), train_accuracy.result() * 100, train_iou.result(),
                     test_loss.result(), test_accuracy.result() * 100, test_iou.result()]

gen_weights = 'D:\\SAT\\Semantic_Seg_rygh\\Enhanced_Unet_attention.h5'
model.save_weights(gen_weights)
np.savetxt("acc_Enhanced_unet_attention.txt", acc)
