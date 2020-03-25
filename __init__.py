'''
Created on 13.03.2020

@author: JanB-4096
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import datetime

# just import stuff to collect all classes in NNTools
from NNTools.NeuroEvolution import NeuroEvolution
from NNTools.NeuralNet import NeuralNet

def save_obj_to_file(obj, file = None):
    if type(file) == None:
        file = 'obj_'+str(datetime.datetime.today())
    with open(file, 'wb') as file_dict:
        pickle.dump(obj, file_dict)
        
def load_obj_from_file(file = None):
    if type(file) == None:
        file = 'obj_'+str(datetime.datetime.today())
    with open(file, 'rb') as file_dict:
        obj = pickle.load(file_dict)
    return obj