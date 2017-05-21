#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import string
import tensorflow as tf
from six.moves import range
import os
os.chdir('/media/sander/D1-P1/shared_folder/python/RNN')

class BatchGenerator(object):    
    
    def __init__(self, data, batch_size, num_unrollings, labels=None, output_shape=2):
      
                
        # Initiate vocab
        self.vocab = ' ' + '.' + string.digits + string.ascii_lowercase         
        self.vocab_size = ord(self.vocab[-1])
        self.vocab_dict = dict(zip(range(self.vocab_size),self.vocab))
                
        self.labels = labels
        if self.labels is not None:
            self.labeled = True
            self.labels = open(labels).read()
        else:
            self.labeled = False
                    
        self.output_shape = output_shape
        
        if not self.labeled:
            self.output_shape = self.vocab_size
        
        # Initiate and clean data
        self.text = open(data).read()
        self.text = self.text.replace('\n', ' ')
        self.text = self.text.replace('\r', ' ')
        self.text = self.text.replace('\t', ' ')
        self.text = self.text.replace('  ', ' ').replace('  ', ' ')    
        self.text = self.text.lower()
        self.text = tf.compat.as_str(self.text)        
        self.text = self.clean_text(self.text)
        self._text_size = len(self.text)
        print('Data size %d' % self._text_size)
        
        # Set some params
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        segment = self._text_size // self.batch_size
        self._cursor = [offset * segment for offset in range(self.batch_size)]
        self._last_x_batch = self._next_x_batch()
        if self.labeled:
            self._last_y_batch = self._next_y_batch()
        
        
    def clean_text(self, text):
    
        new_text = ''
        for letter in text:
            if letter in self.vocab:
                new_text += letter
            else:
                continue            
                
        return new_text
    
    
    def char2id(self, char):
        
        return self.vocab_dict.keys()[self.vocab_dict.values().index(char)]
  
    
    def id2char(self, dictid):
      
        return self.vocab_dict[dictid]
        

    def _next_x_batch(self):
        
        x_batch = np.zeros(shape=(self.batch_size, self.vocab_size), dtype=np.float)
                   
        for b in range(self.batch_size):          
            x_batch[b, self.char2id(self.text[self._cursor[b]])] = 1.0                   
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return x_batch
    
        
    def _next_y_batch(self):
                
        y_batch = np.zeros(shape=(self.batch_size, self.output_shape), dtype=np.float)
        
        for b in range(self.batch_size):                                  
            y_batch[b, int(self.labels[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return y_batch
        

    def _next(self):
        
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        if self.labeled:
            x_batches = [self._last_x_batch]        
            y_batches = [self._last_y_batch]
            
            for step in range(self.num_unrollings):
              x_batches.append(self._next_x_batch())
              y_batches.append(self._next_y_batch())
              
            self._last_x_batch = x_batches[-1]
            self._last_y_batch = y_batches[-1]
            
            return x_batches, y_batches
        else:
            x_batches = [self._last_x_batch]        
                        
            for step in range(self.num_unrollings):
              x_batches.append(self._next_x_batch())
                          
            self._last_x_batch = x_batches[-1]
                        
            return x_batches
    

    def characters(self, probabilities):
      """Turn a 1-hot encoding or a probability distribution over the possible
      characters back into its (most likely) character representation."""
      return [self.id2char(c) for c in np.argmax(probabilities, 1)]
    
    def batches2string(self, batches):
      """Convert a sequence of batches back into their (most likely) string
      representation."""
      s = [''] * batches[0].shape[0]
      for b in batches:
        s = [''.join(x) for x in zip(s, self.characters(b))]
      return s
