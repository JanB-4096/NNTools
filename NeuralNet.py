'''
Created on 13.03.2020

@author: JanB-4096
'''
import numpy as np
from copy import deepcopy


class NeuralNet():
    
    def __init__(self, layers, activation = 'sigmoid'):
        self.neurons = np.asarray(layers)
        self.number_of_layers = self.neurons.__len__()
        if type(activation)==list:
            if len(activation) == len(layers):
                self.activation = activation
            else:
                self.activation = np.tile(activation, int(len(layers)/len(activation))+1)
                self.activation = self.activation[0:len(layers)]
        else:
            self.activation = [activation for _ in range(0,self.neurons.__len__())]
        self.weights = []
        self.biases = []
        for ii in range(0,self.neurons.__len__()-1):
            self.weights.append((np.random.rand(self.neurons[ii], self.neurons[ii+1])-0.5)*2)
            self.biases.append((np.random.rand(self.neurons[ii+1])-0.5)*2)
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self,x):
        return np.maximum(0, x)
    
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x))
        
    def calculate_output_of_layer(self, layer, x):
        if x.__len__() == self.neurons[layer]:
            layer_output = np.add(np.matmul(x, self.weights[layer]), self.biases[layer])
            if self.activation[layer] == 'sigmoid':
                return self.sigmoid(layer_output)
            elif self.activation[layer] == 'relu':
                return self.relu(layer_output)
            elif self.activation[layer] == 'softmax':
                return self.softmax(layer_output)
        else:
            raise Exception('NNTools.NeuralNet - Missmatching dimension of input {} to number of neurons {}!'.format(x.__len__(), self.neurons[layer]))
        
    def calculate_output_to_input(self, layer_output):
        for ii in range(0,self.number_of_layers-1):
            layer_output = self.calculate_output_of_layer(ii, layer_output)
        return layer_output
            
    def mutate_weights_in_layer(self, layer, mutation_rate = 0.1):
        if layer < self.number_of_layers: # only mutate layers which exist
            rand_weights = np.random.rand(self.weights[layer].shape[0], self.weights[layer].shape[1]) < mutation_rate
            rand_change = (np.random.rand(self.weights[layer].shape[0], self.weights[layer].shape[1])*2)-1
            self.weights[layer][rand_weights] = self.weights[layer][rand_weights] + rand_change[rand_weights]
            self.weights[layer] = np.maximum(-1, self.weights[layer])
            self.weights[layer] = np.minimum(1, self.weights[layer])
            
    def mutate_weights_in_all_layers(self, mutation_rate = 0.1):
        # all hidden layers
        for layer in range(0,self.number_of_layers-1):
            self.mutate_weights_in_layer(layer, mutation_rate)
                    
    def mutate_neuron_number_of_layer(self, layer, mutation_rate = 0.1):
        rand = np.random.random()
        if rand < mutation_rate:
            # get random number of neurons between half and double the previous number of neurons
            new_number_of_neurons = np.max([1, int(((np.random.random()*1.5)+0.5) * self.neurons[layer])])
            if new_number_of_neurons == self.neurons[layer]:
                return
            elif new_number_of_neurons < self.neurons[layer]:
                # delete some neurons of layer
                self.remove_neurons_in_layer(layer, self.neurons[layer]-new_number_of_neurons)
            else:
                # add some neurons to layer
                self.add_neurons_to_layer(layer,new_number_of_neurons-self.neurons[layer])
                
    def mutate_neuron_number_of_all_layers(self, mutation_rate = 0.1):
        # all hidden layers
        for layer in range(1, self.number_of_layers-1):
            self.mutate_neuron_number_of_layer(layer, mutation_rate)
                
    def mutate_activation_of_layer(self, layer, mutation_rate = 0.1):
        rand = np.random.random()
        if rand < mutation_rate:
            rand_activation = np.random.randint(0,3)
            if rand_activation == 0:
                self.activation[layer] = 'sigmoid'
            elif rand_activation == 1:
                self.activation[layer] = 'relu' 
            else:
                self.activation[layer] = 'softmax'
            
    def mutate_activation_of_all_layers(self, mutation_rate = 0.1):
        # all hidden layers
        for layer in range(1, self.number_of_layers-1):
            self.mutate_activation_of_layer(layer, mutation_rate)

    def remove_neurons_in_layer(self, layer, number_to_remove):
        if layer == 0 or layer == self.number_of_layers-1:
            return
        else:
            self.weights[layer-1] = np.delete(self.weights[layer-1], np.s_[self.neurons[layer]-number_to_remove:self.neurons[layer]], 1)
            self.biases[layer-1] = self.biases[layer-1][0:self.neurons[layer]-number_to_remove]
            self.weights[layer] = self.weights[layer][0:self.neurons[layer]-number_to_remove]
            self.neurons[layer] = self.neurons[layer]-number_to_remove
                    
    def add_neurons_to_layer(self, layer, number_to_add):
        if layer == 0 or layer == self.number_of_layers-1:
            return
        else:
            self.weights[layer-1] = np.column_stack([self.weights[layer-1], ((np.random.rand(self.weights[layer-1].shape[0], number_to_add)-0.5)*2)])
            self.biases[layer-1] = np.concatenate((self.biases[layer-1], ((np.random.rand(number_to_add)-0.5)*2)))
            self.weights[layer] = np.row_stack([self.weights[layer], ((np.random.rand(number_to_add, self.weights[layer].shape[1])-0.5)*2)])
            self.neurons[layer] += number_to_add
            
    def clone(self):
        return deepcopy(self)