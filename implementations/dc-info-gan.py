#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reference implementation of DC-Info-GAN for experimentation within the SynerGAN project. This is meant to be as understandable, flexible and clean as possible such that we can easily extend it and experiment with it.

This code is heavily influenced by two reference implementations on the web from

https://github.com/wiseodd/generative-models
https://github.com/jacobgil/keras-dcgan

Requires Tensorflow 1.3

TODO
 - Get it to work.
 - Test InfoGAN
 - Testing...

"""

import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf

# Is there a nicer way to import these without depending on keras install?
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.layers.core import Activation
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Flatten
from tensorflow.contrib.keras.python.keras.optimizers import SGD
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.datasets import mnist


""" Model Definitions """

def generator_model(cat_dim, cont_dim, noise_dim):
    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    gen_input = merge([cat_input, cont_input, noise_input], mode="concat")

    gen_hidden = Dense(input_dim=100, units=1024)(gen_input)
    gen_hidden = Activation('tanh')(gen_hidden)
    gen_hidden = Dense(128*7*7)(gen_hidden)
    gen_hidden = BatchNormalization()(gen_hidden)
    gen_hidden = Activation('tanh')(gen_hidden)
    gen_hidden = Reshape((7, 7, 128), input_shape=(128*7*7,))(gen_hidden)
    gen_hidden = UpSampling2D(size=(2, 2))(gen_hidden)
    gen_hidden = Conv2D(64, (5, 5), padding='same')(gen_hidden)
    gen_hidden = Activation('tanh')(gen_hidden)
    gen_hidden = UpSampling2D(size=(2, 2))(gen_hidden)
    gen_hidden = Conv2D(1, (5, 5), padding='same')(gen_hidden)
    gen_hidden = Activation('tanh')(gen_hidden)

    return Model(input=[cat_input, cont_input, noise_input], output=[gen_hidden], name=model_name)


def shared_dq_model(img_dim=(28, 28)):
    disc_input = Input(shape=img_dim, name="discriminator_input")
    disc_hidden.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1)))(disc_input)
    disc_hidden = Activation('tanh')(disc_hidden)
    disc_hidden = MaxPooling2D(pool_size=(2, 2))(disc_hidden)
    disc_hidden = Conv2D(128, (5, 5))(disc_hidden)
    disc_hidden = Activation('tanh')(disc_hidden)
    disc_hidden = MaxPooling2D(pool_size=(2, 2))(disc_hidden)
    disc_hidden = Flatten()(disc_hidden)
    disc_hidden = Dense(1024)(disc_hidden)
    disc_hidden = Activation('tanh')(disc_hidden)
    return Model(input=disc_input, output=[gen_hidden], name=model_name)

def discriminator_model():
    model = Sequential()
    model.add(shared_dq_model())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def q_model():
    model = Sequential()
    model.add(shared_dq_model())
    model.add(Dense(12))
    model.add(Activation('tanh'))
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


""" Generator Input Noise """

def sample_z(batch_size, z_dim):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1, 1, size=(batch_size, z_dim))

# for InfoGAN
def sample_c(batch_size, latent_spec):
    '''Returns noise according to latent spec [(type, arg), ...] supports categorical and uniform types'''
    c = []

    for distribution, size in latent_spec:

        if distribution == "categorical":
            idxs = np.random.randint(size, size=batch_size)
            onehot = np.zeros((batch_size, size)).astype(np.float32)
            onehot[np.arange(batch_size), idxs] = 1
            c.append(onehot)
        elif distribution == "uniform":
            random = np.random.uniform(-1, 1, size=(batch_size, 1)) # TODO: normal distribution
            c.append(random)
        else:
            raise NotImplementedError

    return np.concatenate(c, axis=1)

def sample_zc(batch_size, z_dim=32, latent_spec=[("categorical", 10), ("uniform", True), ("uniform", True)]):
    return np.concatenate([sample_z(batch_size, z_dim), sample_c(batch_size, latent_spec)], axis=1)


""" Convenience Method """

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

""" Gaussian loss function """

def gaussian_loss(y_true, y_pred):

    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C


""" Main Execution """

def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])

    d = discriminator_model()
    g = generator_model(cat_dim=1, cont_dim=2, noise_dim=100-12)
    d_on_g = generator_containing_discriminator(g, d)

    q = q_model()
    q_on_g = generator_containing_discriminator(g, q)

    d_optim = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    g_optim = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    q_optim = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)

    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    q.compile(loss='categorical_crossentropy', optimizer=q_optim)

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))

        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = sample_zc(BATCH_SIZE, z_dim=100-12)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            generated_images = g.predict(noise, verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)

            q_loss = q.train_on_batch(generated_images, noise[:,100-12:])

            print("batch %d d_loss : %f" % (index, d_loss))
            print("batch %d q_loss : %f" % (index, q_loss))

            noise = sample_zc(BATCH_SIZE, z_dim=100-12)
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True

            print("batch %d g_loss : %f" % (index, g_loss))

            # Demo images
            noise = sample_zc(BATCH_SIZE, z_dim=100-12)
            noise[:,100-12:-2] = 0
            for i in xrange(10):
                noise[i*10:(i+1)*10, -12+i] = 1

            demo_images = g.predict(noise, verbose=0)

            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")

            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)
                q.save_weights('q', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
