'''
Created on 13.03.2020

@author: JanB-4096
'''
from NNTools.NeuralNet import NeuralNet
import numpy as np
import itertools


class NeuroEvolution():

    def __init__(self, population_count, layers, activation):
        self.population = [NeuralNet(layers, activation) for _ in range(0, population_count)]
        self.generation = 1
        self.generation_overview = {}
        self.best_fitness = 0
        self.fitness_list = np.zeros(population_count)
        self.fraction_best = 0 # fraction of best nn to keep for next generation
        self.fraction_best_crossover_sp = 0.1 # fraction of population of best species to crossover with single point
        self.fraction_best_crossover_dp = 0.1 # fraction of population of best species to crossover with double point
        self.fraction_best_crossover_up = 0.1 # fraction of population of best species to crossover with unique point
        self.fraction_random_crossover = 0.1 # fraction of population of random species to crossover
        self.fraction_mutation_weights = 0
        self.fraction_mutation_neuron_number = 0.1
        self.fraction_mutation_activation = 0
        
    def clone_species(self, id_of_species):
        return self.population[id_of_species].clone()
        
    def get_mother_and_father(self, id_species_a, id_species_b):
        # clone a species as initialization = mother_species and adapt with father_species
        rand_species = np.random.random()
        if rand_species >= 0.5: # take species b
            mother_species = self.population[id_species_b]
            father_species = self.population[id_species_a]
        else:# take species a
            mother_species = self.population[id_species_a]
            father_species = self.population[id_species_b]
        return mother_species, father_species
            
    def crossover_singlepoint(self, id_species_a, id_species_b):
        mother_species, father_species = self.get_mother_and_father(id_species_a, id_species_b)
        # clone mother species as initial net and adapt with father 
        new_nn = mother_species.clone()
        if np.any(mother_species.neurons != father_species.neurons):
            return self.crossover_species(id_species_a, id_species_b)
        else:
            for ii in range(0, father_species.number_of_layers-1): # for each hidden layer
                rand_row = int(np.round(np.random.random() * mother_species.neurons[ii]))
                rand_col = int(np.round(np.random.random() * mother_species.neurons[ii+1]))
                new_nn.weights[ii][rand_row:, rand_col:] = father_species.weights[ii][rand_row:, rand_col:]      
        return new_nn
                
    def crossover_doublepoint(self, id_species_a, id_species_b):
        mother_species, father_species = self.get_mother_and_father(id_species_a, id_species_b)
        new_nn = mother_species.clone()
        if np.any(mother_species.neurons != father_species.neurons):
            return self.crossover_species(id_species_a, id_species_b)
        else:
            for ii in range(0, father_species.number_of_layers-1): # for all weight matrices
                rand_row = int(np.round(np.random.random() * mother_species.neurons[ii]))
                rand_row2 = int(np.round(np.random.random() * mother_species.neurons[ii]))
                rand_row, rand_row2 = np.min([rand_row, rand_row2]), np.max([rand_row, rand_row2])
                
                rand_col = int(np.round(np.random.random() * mother_species.neurons[ii+1]))
                rand_col2 = int(np.round(np.random.random() * mother_species.neurons[ii+1]))
                rand_col, rand_col = np.min([rand_col, rand_col2]), np.max([rand_col, rand_col2])
                
                new_nn.weights[ii][rand_row:rand_row2, rand_col:rand_col2] = father_species.weights[ii][rand_row:rand_row2, rand_col:rand_col2]
        return new_nn
    
    def crossover_uniquepoint(self, id_species_a, id_species_b):
        mother_species, father_species = self.get_mother_and_father(id_species_a, id_species_b)
        new_nn = mother_species.clone()
        if np.any(mother_species.neurons != father_species.neurons):
            return self.crossover_species(id_species_a, id_species_b)
        else:
            for ii in range(0, father_species.number_of_layers-1): # for all weights
                rand_father = np.random.rand(mother_species.neurons[ii], mother_species.neurons[ii+1]) >= 0.5
                new_nn.weights[ii][rand_father] = father_species.weights[ii][rand_father]
        return new_nn
            
        
    def crossover_species(self, id_species_a, id_species_b):
        # function to crossover between different species with different neuron numbers
        mother_species, father_species = self.get_mother_and_father(id_species_a, id_species_b)
        new_nn = mother_species.clone()
            
        layers_different_number_of_neurons = np.not_equal(mother_species.neurons, father_species.neurons)
        if not np.any(layers_different_number_of_neurons): #all layers have equal structure
            new_nn = self.crossover_uniquepoint(id_species_a, id_species_b)
        else:
            rand_neuron_number_of_father = (np.random.rand(1, layers_different_number_of_neurons.size) > 0.5) & layers_different_number_of_neurons
            #new_nn.neurons[rand_neuron_number_of_father[0]] = father_species.neurons[rand_neuron_number_of_father[0]]
            for layer in range(1,new_nn.number_of_layers-1):
                if rand_neuron_number_of_father[0, layer]:
                    change_of_neurons = father_species.neurons[layer] - new_nn.neurons[layer]
                    if change_of_neurons < 0:
                            new_nn.remove_neurons_in_layer(layer, np.abs(change_of_neurons))
                    else:
                        # append neurons with random initial values
                        new_nn.add_neurons_to_layer(layer, change_of_neurons)

                    # weights for input of layer
                    for jj in range(0, new_nn.neurons[layer-1]):
                        for kk in range(0, new_nn.neurons[layer]):
                            rand = np.random.random() >= 0.5
                            if rand:
                                try: # get weights and biases of father if matrix dimension matches
                                    new_nn.weights[layer-1][jj,kk] = father_species.weights[layer-1][jj,kk] 
                                    new_nn.biases[layer-1][kk] = father_species.biases[layer-1][kk]
                                except: # else try to get weights and biases of mother - neccessary because added neurons are initialized randomly
                                    try:
                                        new_nn.weights[layer-1][jj,kk] = mother_species.weights[layer-1][jj,kk]
                                        new_nn.biases[layer-1][kk] = mother_species.biases[layer-1][kk]
                                    except:
                                        continue
                            else: # other way around if random number is smaller than 0.5
                                try:
                                    new_nn.weights[layer-1][jj,kk] = mother_species.weights[layer-1][jj,kk]
                                    new_nn.biases[layer-1][kk] = mother_species.biases[layer-1][kk]
                                except:
                                    try:
                                        new_nn.weights[layer-1][jj,kk] = father_species.weights[layer-1][jj,kk] 
                                        new_nn.biases[layer-1][kk] = father_species.biases[layer-1][kk]
                                    except:
                                        continue
                                    
                    # weights for output of layer
                    for jj in range(0, new_nn.neurons[layer]):
                        for kk in range(0, new_nn.neurons[layer+1]):
                            rand = np.random.random() >= 0.5
                            if rand:
                                try: # get weights and biases of father if matrix dimension matches
                                    new_nn.weights[layer][jj,kk] = father_species.weights[layer][jj,kk] 
                                    new_nn.biases[layer][kk] = father_species.biases[layer][kk]
                                except: # else try to get weights and biases of mother - neccessary because added neurons are initialized randomly
                                    try:
                                        new_nn.weights[layer][jj,kk] = mother_species.weights[layer][jj,kk]
                                        new_nn.biases[layer][kk] = mother_species.biases[layer][kk]
                                    except:
                                        continue
                            else: # other way around if random number is smaller than 0.5
                                try:
                                    new_nn.weights[layer][jj,kk] = mother_species.weights[layer][jj,kk]
                                    new_nn.biases[layer][kk] = mother_species.biases[layer][kk]
                                except:
                                    try:
                                        new_nn.weights[layer][jj,kk] = father_species.weights[layer][jj,kk] 
                                        new_nn.biases[layer][kk] = father_species.biases[layer][kk]
                                    except:
                                        continue
        return new_nn

    def update_fitness(self, fitness, species_id):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
        self.fitness_list[species_id] = fitness
        
    def check_next_gen_fractions(self):
        # clamp fractions
        self.fraction_best = np.min([np.max([self.fraction_best, 1/self.population.__len__()]), 0.2]) 
        self.fraction_best_crossover_sp = np.min([np.max([self.fraction_best_crossover_sp, 0]), 0.5])
        self.fraction_best_crossover_dp = np.min([np.max([self.fraction_best_crossover_dp, 0]), 0.5])
        self.fraction_best_crossover_up = np.min([np.max([self.fraction_best_crossover_up, 0]), 0.5])
        self.fraction_random_crossover = np.min([np.max([self.fraction_random_crossover, 0]), 0.5])
        self.fraction_mutation_weights = np.min([np.max([self.fraction_mutation_weights, 0]), 1])
        self.fraction_mutation_neuron_number = np.min([np.max([self.fraction_mutation_neuron_number, 0]), 0.5])
        self.fraction_mutation_activation = np.min([np.max([self.fraction_mutation_activation, 0]), 0.5])
        
        # check fractions for sum
        sum_fractions = self.fraction_best + self.fraction_best_crossover_sp + self.fraction_best_crossover_dp + \
            self.fraction_best_crossover_up + self.fraction_random_crossover + self.fraction_mutation_weights + \
            self.fraction_mutation_neuron_number + self.fraction_mutation_activation
        if sum_fractions > 1:
            Warning('Sum of fractions for next generation are bigger than 1 - your population will increase with each generation!')
        elif sum_fractions < 1:     
            Warning('Sum of fractions for next generation is smaller than 1 - it filled with more mutation of weights!')
            self.fraction_mutation_weights = 1 - self.fraction_best - self.fraction_best_crossover_sp - \
                self.fraction_best_crossover_dp - self.fraction_best_crossover_up - self.fraction_random_crossover - \
                self.fraction_mutation_neuron_number - self.fraction_mutation_activation
        
    def build_next_generation(self):
        new_gen = []
        
        # number of best nn to use for next gen
        number_for_next_gen = int(np.round(self.fraction_best*self.population.__len__()))
        sorted_fitness = self.fitness_list.argsort()[-1::-1]
        [new_gen.append(self.population[ii]) for ii in sorted_fitness[0:number_for_next_gen]]
        
        if self.fraction_best_crossover_sp > 0:
            number_of_crossover = int(np.round(self.fraction_best_crossover_sp*self.population.__len__()))
            idx_crossover_best = list(itertools.product([sorted_fitness[0]], sorted_fitness[1:]))
            [new_gen.append(self.crossover_singlepoint(ii[0], ii[1])) for ii in idx_crossover_best[0:number_of_crossover]]
        
        if self.fraction_best_crossover_dp > 0:
            number_of_crossover = int(np.round(self.fraction_best_crossover_dp*self.population.__len__()))
            idx_crossover_best = list(itertools.product([sorted_fitness[0]], sorted_fitness[1:]))
            [new_gen.append(self.crossover_doublepoint(ii[0], ii[1])) for ii in idx_crossover_best[0:number_of_crossover]]
            
        if self.fraction_best_crossover_up > 0:
            number_of_crossover = int(np.round(self.fraction_best_crossover_up*self.population.__len__()))
            idx_crossover_best = list(itertools.product([sorted_fitness[0]], sorted_fitness[1:]))
            [new_gen.append(self.crossover_uniquepoint(ii[0], ii[1])) for ii in idx_crossover_best[0:number_of_crossover]]
            
        if self.fraction_random_crossover > 0:
            number_of_crossover= int(np.round(self.fraction_random_crossover*self.population.__len__()))
            idx_random_nn = list(map(int, np.random.rand(number_of_crossover)*self.population.__len__()))
            idx_crossover_random = list(itertools.product(idx_random_nn, idx_random_nn))
            idx_crossover_random = [(ii[0],ii[1]) for ii in idx_crossover_random if ii[0] != ii[1]]
            [new_gen.append(self.crossover_singlepoint(ii[0], ii[1])) for ii in idx_crossover_random[0:number_of_crossover]]
        
        if self.fraction_mutation_neuron_number > 0:
            number_mutate = int(np.round(self.fraction_mutation_neuron_number*self.population.__len__()))
            [new_gen.append(self.population[sorted_fitness[0]]) for _ in range(number_mutate)]
            [new_gen[ii].mutate_neuron_number_of_all_layers() for ii in range(len(new_gen)-number_mutate, len(new_gen))]
            
        if self.fraction_mutation_activation > 0:
            number_mutate = int(np.round(self.fraction_mutation_activation*self.population.__len__()))
            [new_gen.append(self.population[sorted_fitness[0]]) for _ in range(number_mutate)]
            [new_gen[ii].mutate_neuron_number_of_all_layers() for ii in range(len(new_gen)-number_mutate, len(new_gen))]
        
        number_mutate = self.population.__len__() - new_gen.__len__()
        if number_mutate > 0:
            [new_gen.append(self.population[sorted_fitness[0]]) for _ in range(number_mutate)]
            [new_gen[ii].mutate_weights_in_all_layers() for ii in range(len(new_gen)-number_mutate, len(new_gen))]

        
        self.population = new_gen