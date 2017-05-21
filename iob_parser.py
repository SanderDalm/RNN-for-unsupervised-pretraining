#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:34:32 2017

@author: sander
"""
import os
os.chdir('/media/sander/D1-P1/shared_folder/python/RNN/')
from glob import glob
from tqdm import tqdm


class IOB_parser(object):
    
    def __init__(self, file_dir, text_filename, label_filename):
        
        self.file_list = glob(file_dir+'*.iob')        
        self.text_filename = text_filename
        self.label_filename = label_filename
        self.label_dict = self.create_label_dict()
        self.neg_label = self.label_dict['O']
        
    
    def create_label_dict(self):
        
        labels = []
        
        for data in tqdm(self.file_list):    
            for i in open(data).readlines():    
                if len(i.split('\t')) > 1:                
                    label = i.split('\t')[1].strip('\n')                                
                    labels.append(label)                
        return dict(zip(set(labels),range(len(set(labels)))))

    
    def create_sonar_data(self):
        
        woorden = ''
        labels = ''
            
        for data in tqdm(self.file_list):
        
            for i in open(data).readlines():
        
                if len(i.split('\t')) > 1:
                    woord = i.split('\t')[0]                
                    label_str = i.split('\t')[1].strip('\n')
                    label = self.label_dict[label_str]      
                    if label != self.neg_label:
                        label = 1
                    else:
                        label = 0                          
                    label = str(label)*len(woord)
                    
                    if woord in ['.', ',', ':', ';', '!' ,'?', "'"]:
                        woorden += woord
                        labels += label
                    else:
                        woorden += ' ' + woord
                        labels += '0' + label                
        assert len(woorden) == len(labels)
        
        with open(self.text_filename, "w") as text_file:
            text_file.write(woorden)
        with open(self.label_filename, "w") as text_file:
            text_file.write(labels)
