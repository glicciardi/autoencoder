#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:21:59 2020

@author: glicciardi
"""

from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np
import scipy.io
import random
import numpy as np



mat = scipy.io.loadmat( '/Users/glicciardi/Desktop/corso_HS/PaviaU.mat' )
img = mat[ 'paviaU' ]

rows,cols,bands=img.shape

n_pixels=rows*cols

imm_reshape= np.reshape(img, (n_pixels,bands))
massimo=np.max(imm_reshape)
imm_reshape=imm_reshape/massimo
#imm_reshape=imm_reshape*2-1

l = [random.randint(0,n_pixels) for i in range(10000)]
l1=l[0:8900]
l2=l[8900:10000]








x_train=imm_reshape[l1]

x_test=imm_reshape[l2]



#####################################
#noise_factor = 0.5
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
#
#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#

####################################


input_img = Input(shape=(bands,))
encoded = Dense(45, activation='sigmoid')(input_img)
#encoded = Dense(30, activation='sigmoid')(encoded)
encoded = Dense(5, activation='sigmoid')(encoded)


decoded = Dense(45, activation='sigmoid')(encoded)
#decoded = Dense(60, activation='sigmoid')(decoded)
decoded = Dense(bands, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

#encoded_input = Input(shape=(45,))
## retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
## create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=1000,
                shuffle=True,
                validation_data=(x_test, x_test))



encoded_imgs = encoder.predict(imm_reshape)
result=autoencoder.predict(imm_reshape)
rec=np.reshape(encoded_imgs, (rows,cols,5))
 


