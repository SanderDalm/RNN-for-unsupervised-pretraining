#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range
from matplotlib import pyplot as plt
import random
import os
os.chdir('/media/sander/D1-P1/shared_folder/python/RNN')


class RNN(object):
    
    def __init__(self, summary_frequency, num_nodes, num_layers,
                 batch_generator, labeled=False, output_shape=2):
        
        self.batch_generator = batch_generator
        self.vocab_size = self.batch_generator.vocab_size
        self.batch_size = self.batch_generator.batch_size
        self.num_unrollings = self.batch_generator.num_unrollings
        self.num_nodes = num_nodes
        self.labeled = labeled
        self.output_shape = output_shape
        if not self.labeled:
            self.output_shape = self.vocab_size
        self.summary_frequency = summary_frequency        
        self.session=tf.Session()
    
        # Call a basic LSTM/GRU cell from tensorflow module
        cell = tf.contrib.rnn.GRUCell(num_nodes)
    
        cells = [cell]
    
        # Here we add as many layers as desired
        for i in range(num_layers-1):
          higher_layer_cell = tf.contrib.rnn.GRUCell(self.num_nodes)
          cells.append(higher_layer_cell)
    
        #cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = output_keep_prob)
        #           for cell_ in cells]
    
        # These layers are combined into a conventient MultiRNNCell object
        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
    
        # Read input data
        # Note that there is no label for the last prediction, as the labels are simply the
        # letters shifted by 1 position. For the same reason we discard the first label:
        # there is no prediction for the first letter.
    
        self.train_data = list()
        for _ in range(self.num_unrollings + 1):
          self.train_data.append(
            tf.placeholder(tf.float32, shape=[None, self.vocab_size]))
            
    
        # Feed the data to the RNN model
        outputs = tf.Variable(np.zeros([self.num_unrollings, self.batch_size, self.output_shape]))
        outputs, self.state = tf.contrib.rnn.static_rnn(multi_cell, self.train_data, dtype=tf.float32)
    
        # Classifier. For training, we remove the last output, as it has no label.
        # The last output is only used for prediction purposes during sampling.    
        w = tf.Variable(tf.truncated_normal([self.num_nodes, self.output_shape], -0.1, 0.1), name='output_w')
        b = tf.Variable(tf.zeros([self.output_shape]), name='output_b')
        
        if self.labeled:
            logits = tf.matmul(tf.concat(axis=0,values=outputs), w) + b            
        else:
            logits = tf.matmul(tf.concat(axis=0,values=outputs[:-1]), w) + b
                              
        sample_logits = tf.matmul(tf.concat(axis=0,values=outputs), w) + b        
        self.sample_prediction = tf.nn.softmax(sample_logits)
    
        if self.labeled:
            self.train_labels = []
            for _ in range(self.num_unrollings + 1):
                self.train_labels.append(
                        tf.placeholder(tf.float32, [None, self.output_shape]))        
        else:
            self.train_labels = self.train_data[1:]
        self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.train_labels))
    
        # Optimizer.        
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    
        # Train prediction. We keep this to keep track of the model's progress.
        self.train_prediction = tf.nn.softmax(logits)

        self.session=tf.Session()
        with self.session.as_default():
            init_op = tf.global_variables_initializer()
            self.session.run(init_op)                    
            self.saver = tf.train.Saver()
    
    def logprob(self, predictions, labels):
        
        """Log-probability of the true labels in a predicted batch."""
        predictions[predictions < 1e-10] = 1e-10        
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]
    
    
    def train(self, num_steps):
            
        self.minibatch_perp_list = []
    
        with self.session.as_default():
            
            mean_loss = 0
            for step in range(num_steps):
                
                feed_dict = dict()
                
                if self.labeled:
                    x_batch, y_batch = self.batch_generator._next()                                    
                    for i in range(self.num_unrollings + 1):                             
                        feed_dict[self.train_data[i]] = x_batch[i]
                        feed_dict[self.train_labels[i]] = y_batch[i]
                else:
                    x_batch = self.batch_generator._next()                   
                    for i in range(self.num_unrollings + 1):                    
                        feed_dict[self.train_data[i]] = x_batch[i]
                
                _, l, predictions, = self.session.run(
                        [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                mean_loss += l
                if step % self.summary_frequency == 0:
                    if step > 0:
                        mean_loss = mean_loss / self.summary_frequency
                    # The mean loss is an estimate of the loss over the last few batches.
                    print(
                       'Average loss at step %d: %f ' % (step, mean_loss))
                    mean_loss = 0                    
                    labels = np.concatenate(list(x_batch)[1:])
                                        
                    if self.labeled:
                        labels = np.concatenate(list(y_batch))                        
                        print('Minibatch perplexity: %.2f' % float(
                        np.exp(self.logprob(predictions, labels))))
                    self.minibatch_perp_list.append(float(
                        np.exp(self.logprob(predictions, labels))))
                
            
    def sentence_to_batch(self, sentence):
    
        # Sentence to batch van [num_unrollings,batch_size,vocab_size]
        sample_batch = []
        for i in sentence[-self.num_unrollings-1:]:
            vector = np.zeros([self.batch_size, self.vocab_size])
            for j in range(self.batch_size):
                vector[j, self.batch_generator.char2id(i)] = 1.0
            sample_batch.append(vector)
        return sample_batch


    def sample_distribution(self, distribution):
        
        """Sample one element from a distribution assumed to be an array of normalized
        probabilities.
        """    
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
          s += distribution[i]
          if s >= r:
            return i
        return len(distribution) - 1


    def sample(self, prediction):
        
      """Turn a (column) prediction into 1-hot encoded samples."""
      p = np.zeros(shape=[1, self.vocab_size], dtype=np.float)
      p[0, self.sample_distribution(prediction[0])] = 1.0
      return p
    
    
    def random_distribution(self):
        
      """Generate a random column of probabilities."""
      b = np.random.uniform(0.0, 1.0, size=[1, self.vocab_size])
      return b/np.sum(b, 1)[:,None]


    def sample_sentence(self, start_sentence, length):
                
        assert len(start_sentence) >= self.num_unrollings, 'Input length needs to be >= ' + str(self.num_unrollings)
    
        sentence = start_sentence
        
        with self.session.as_default():
    
            # Sample predictions
            for _ in range(length):
    
                batches = self.sentence_to_batch(sentence)    
                feed_dict = dict()
                for i in range(self.num_unrollings+1):                    
                    feed_dict[self.train_data[i]] = batches[i]
    
                prediction = np.array(self.sample_prediction.eval(feed_dict)).reshape([self.num_unrollings+1,
                                     self.batch_size,
                                     self.vocab_size])[-1][-1].reshape([1, self.vocab_size])
                #sentence += id2char(np.argmax(prediction))            
                sentence += self.batch_generator.id2char(np.argmax(self.sample(prediction)))
                
        print(sentence)
    
    
    def create_restore_dict(self):
        
        variable_names = [v for v in tf.trainable_variables()]
        variable_handles = [v.name for v in variable_names]            
        restore_dict = dict(zip(variable_handles, variable_names))                        
        restore_dict.pop('Variable:0')
        restore_dict.pop('output_w:0')
        restore_dict.pop('output_b:0')            
        
        for key in restore_dict:
            print key
            print restore_dict[key]
        return restore_dict
    
    def save(self, checkpointname, full_model=True):
        
        
        if full_model == False:            
            restore_dict = self.create_restore_dict()
            with self.session.as_default():            
                self.saver = tf.train.Saver(restore_dict)
            
        self.saver.save(self.session, checkpointname)
        print('Model saved')
    
    
    def load(self, checkpointname, full_model=True):
        
        
        if full_model == False:                    
            restore_dict = self.create_restore_dict()    
            with self.session.as_default():                            
                self.saver = tf.train.Saver(restore_dict)
                                
        self.saver.restore(self.session, checkpointname)         
        print('Model restored') 
        
        
    def predict(self, inputs):
        
        feed_dict = dict()              
        for i in range(self.num_unrollings + 1):                    
                    feed_dict[self.train_data[i]] = inputs[i]
        _, l, predictions, = self.session.run(
                [self.optimizer, self.loss, self.sample_prediction], feed_dict=feed_dict)
        
        return predictions

                
    def plot(self):
        
        x1=np.array(self.minibatch_perp_list)        
        plt.plot(x1,color='g',alpha=0.4, linewidth=5)
        plt.xlabel('Iterations')
        plt.ylabel('Perplexity')
        plt.show()


        
        