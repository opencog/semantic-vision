#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Reference implementation of DC-Info-GAN for experimentation within the SynerGAN project. This is meant to be as understandable, flexible and clean as possible for experimentation and extension.

This code is influenced by several implementations:
https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/InfoGAN
https://github.com/wiseodd/generative-models
https://github.com/jacobgil/keras-dcgan

Requires Tensorflow 1.3 to run!

@elggem

"""

import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import Model, Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Reshape, Input, concatenate
from tensorflow.contrib.keras.python.keras.layers.core import Activation
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Flatten
from tensorflow.contrib.keras.python.keras.optimizers import SGD
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.datasets import mnist

""" Parameters """

LR = 1e-4
EPSILON = 1e-8
BATCH_SIZE = 128

LAMBDA_DISC = 0.5
LAMBDA_CONT = 0.5

IMG_DIM = (28, 28, 1)

Z_DIM = 88

LATENT_SPEC = [("categorical", 10), ("uniform", 1), ("uniform", 1)]
DISC_DIM = sum([t[1] for t in LATENT_SPEC if t[0]=="categorical"])
CONT_DIM = sum([t[1] for t in LATENT_SPEC if t[0]=="uniform"])


""" Model Definitions """

def generator_model():
    noise_input = Input(batch_shape=(BATCH_SIZE, Z_DIM), name='z_input')
    input_list = [noise_input]

    for distribution, size in LATENT_SPEC:
        inp = Input(batch_shape=(BATCH_SIZE, size))
        input_list.append(inp)

    gen_input = concatenate(input_list, axis=1, name="generator_input")

    h = Dense(1024)(gen_input)
    h = Activation('tanh')(h)
    h = Dense(128*7*7)(h)
    h = BatchNormalization()(h)
    h = Activation('tanh')(h)
    h = Reshape((7, 7, 128), input_shape=(128*7*7,))(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Conv2D(64, (5, 5), padding='same')(h)
    h = Activation('tanh')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Conv2D(1, (5, 5), padding='same')(h)
    h = Activation('tanh')(h)

    return Model(inputs=input_list, outputs=[h], name="generator")


def shared_dq_model():
    img_input = Input(shape=IMG_DIM, name="discriminator_input")

    h = Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1))(img_input)
    h = Activation('tanh')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Conv2D(128, (5, 5))(h)
    h = Activation('tanh')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Flatten()(h)
    h = Dense(1024)(h)
    h = Activation('tanh')(h)

    return Model(inputs=img_input, outputs=[h], name="shared_dq")

def discriminator_model():
    model = shared_dq_model()

    h = Dense(1)(model.output)
    h = Activation('sigmoid')(h)

    return Model(inputs=model.input, outputs=[h], name="discriminator")

def q_model():
    model = shared_dq_model()

    h = Dense(128)(model.output)
    h = BatchNormalization()(h)
    h = LeakyReLU()(h)

    outputs = []

    for distribution, size in LATENT_SPEC:

        if distribution == "categorical":
            outputs.append(Dense(size, activation='softmax',  name="%s_%d" % (distribution, len(outputs)))(h))
        elif distribution == "uniform":
            outputs.append(Dense(size, activation='linear',  name="%s_%d" % (distribution, len(outputs)))(h))
        else:
            raise NotImplementedError

    return Model(inputs=model.input, outputs=outputs, name="q_network")


def generator_containing_discriminator(g, d):
    d.trainable = False
    return Model(inputs=g.input, outputs=d(g.output))


""" Generator Input Noise """

def sample_z():
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))

def sample_c():
    '''Returns noise according to latent spec [(type, arg), ...] supports categorical and uniform types'''
    sampled = []

    for distribution, size in LATENT_SPEC:

        if distribution == "categorical":
            idxs = np.random.randint(size, size=BATCH_SIZE)
            onehot = np.zeros((BATCH_SIZE, size)).astype(np.float32)
            onehot[np.arange(BATCH_SIZE), idxs] = 1
            sampled.append(onehot)
        elif distribution == "uniform":
            random = np.random.uniform(-1, 1, size=(BATCH_SIZE, 1))
            sampled.append(random)
        else:
            raise NotImplementedError

    return sampled

def sample_zc():
    array = [sample_z()]
    array.extend(sample_c())
    return array


""" Convenience Methods """

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

def make_image(generator):
    noise = [sample_z()]

    for distribution, size in LATENT_SPEC:

        if distribution == "categorical":
            idxs = np.concatenate([np.repeat(i,size) for i in range(size)])
            idxs = np.append(idxs, np.repeat(0, BATCH_SIZE-(size*size)))
            onehot = np.zeros((BATCH_SIZE, size)).astype(np.float32)
            onehot[np.arange(BATCH_SIZE), idxs] = 1
            noise.append(onehot)
        elif distribution == "uniform":
            random = np.concatenate([[i/10.0 for i in range(10)] for _ in range(size)])
            random = np.append(random, np.repeat(0, BATCH_SIZE-(size*size)))
            noise.append(random)
        else:
            raise NotImplementedError

    demo_images = generator.predict(noise, batch_size=BATCH_SIZE, verbose=0)
    image = combine_images(demo_images[:(DISC_DIM)*10])
    image = image*127.5+127.5

    return image


""" Loss functions """

def combined_mutual_info_loss(true_disc, predicted_disc):
    if predicted_disc.name.find("categorical") > -1:
        return LAMBDA_DISC * disc_mutual_info_loss(true_disc, predicted_disc)
    elif predicted_disc.name.find("uniform") > -1:
        return LAMBDA_CONT * cont_mutual_info_loss(true_disc, predicted_disc)


def disc_mutual_info_loss(c_disc, aux_dist):
    """
    Mutual Information lower bound loss for discrete distribution.
    """
    reg_disc_dim = aux_dist.get_shape().as_list()[-1]
    cross_ent = - K.mean( K.sum( K.log(aux_dist + EPSILON) * c_disc, axis=1 ) )
    ent = - K.mean( K.sum( K.log(1./reg_disc_dim + EPSILON) * c_disc, axis=1 ) )

    return - (ent - cross_ent)

def cont_mutual_info_loss(c_disc, aux_dist):
    """
    Mutual Information lower bound loss for continous distribution.
    Using MSE here.
    """
    return  tf.losses.mean_squared_error(c_disc, aux_dist)



""" Main Execution """

def train():
    # Prepare Training Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    # Initialize Models
    d = discriminator_model()
    g = generator_model()
    q = q_model()
    d_on_g = generator_containing_discriminator(g, d)
    q_on_g = generator_containing_discriminator(g, q)

    # Initialize Optimizers
    d_optim = Adam(lr=LR, beta_1=0.5, beta_2=0.999, epsilon=EPSILON)
    g_optim = Adam(lr=LR, beta_1=0.5, beta_2=0.999, epsilon=EPSILON)
    q_optim = Adam(lr=LR, beta_1=0.5, beta_2=0.999, epsilon=EPSILON)

    # Compile Models with loss functions
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    q_on_g.compile(loss=combined_mutual_info_loss, optimizer=g_optim)

    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    q.trainable = True
    q.compile(loss=combined_mutual_info_loss, optimizer=q_optim)


    try:
        # Main Training Loop
        for epoch in range(100):
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))

            for index in range(int(X_train.shape[0]/BATCH_SIZE)):
                # Get Real and Generated Images
                noise = sample_zc()

                real_images = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = g.predict(noise, batch_size=BATCH_SIZE, verbose=0)

                # Train Discriminator and Q network
                training_images = np.concatenate((real_images, generated_images))
                labels = [1] * BATCH_SIZE + [0] * BATCH_SIZE

                d_loss = d.train_on_batch(training_images, labels)
                q_loss = q.train_on_batch(generated_images, noise[1:])

                # Train Generator using Fake/Real Signal
                noise = sample_zc()

                d.trainable = False
                g_d_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
                d.trainable = True

                # Train Generator using Mutual Information Lower Bound
                noise = sample_zc()

                q.trainable = False
                g_q_loss = q_on_g.train_on_batch(noise, noise[1:])
                q.trainable = True

                print("batch %d d_loss : %.3f q_loss: %s g_loss_d: %.3f g_loss_q: %s" % (index, d_loss, q_loss, g_d_loss, g_q_loss))

                # Generate Sample Images
                if index % 20 == 0:
                    image = make_image(g)

                    Image.fromarray(image.astype(np.uint8)).save(
                        str(epoch)+"_"+str(index)+".png")

                # Save weights
                if index % 10 == 9:
                    g.save_weights('g.kerasweights', True)
                    d.save_weights('d.kerasweights', True)
                    q.save_weights('q.kerasweights', True)

    except KeyboardInterrupt:
      pass

    print("\rFinished training")


def generate(nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('g.kerasweights')

    image = make_image(g)

    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train()
    elif args.mode == "generate":
        generate()
