#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range
from matplotlib import pyplot as plt
import os
os.chdir('/media/sander/D1-P1/shared_folder/python/RNN')

from batch_generator import BatchGenerator
from rnn import RNN
from iob_parser import IOB_parser

##########################################
# Create SONAR data
##########################################

#parser = IOB_parser('sonar_ent/IOB/', 'sonar_text.txt', 'sonar_labels.txt')
#parser.create_sonar_data()

sonar_text = 'sonar_text.txt'
sonar_labels = 'sonar_labels.txt'

##########################################
# Create batch_generator
##########################################

wiki_text = 'wiki.csv'
batch_size = 32
num_unrollings = 50    
generator = BatchGenerator(wiki_text, batch_size, num_unrollings, labels=None, output_shape=2)
#generator = BatchGenerator(sonar_text, batch_size, num_unrollings, labels=sonar_labels, output_shape=2)

##########################################
# Create RNN
##########################################

summary_frequency = 10
num_nodes = 128
num_layers = 1
model_path = './model/checkpoint.ckpt'

RNN = RNN(summary_frequency, num_nodes, num_layers, 
          generator, labeled=False, output_shape=2)

RNN.train(100)
RNN.plot()

RNN.save(model_path, full_model=False)
RNN.load(model_path, full_model=False)

#RNN.sample_sentence('japan is al geruime tijd een natie van de erkende naties inderdaad d', 100)
#RNN.sample_sentence(' ' * (num_unrollings+1), 1000)

#x,y=generator._next()
#pred = RNN.predict(x)