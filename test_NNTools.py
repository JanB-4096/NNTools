'''
Created on 13.03.2020

@author: JanB-4096
'''
import NNTools
import time

# start main functionality
start_time = time.clock()

# initialize population to play
evolution = NNTools.NeuroEvolution(200, [8,200,100,300,10], ['sigmoid', 'relu'])

### testing ###
input_to_nn = [1., 2., 3., 4., 5., 6., 7., 8., 9., 0.,]

# clone
nn2 = evolution.clone_species(2)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

# mutate
nn2.mutate_weights_in_layer(1, mutation_rate = 1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

nn2.mutate_neuron_number_of_layer(0, mutation_rate = 1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

nn2.mutate_neuron_number_of_layer(1, mutation_rate = 1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

nn2.mutate_neuron_number_of_layer(2, mutation_rate = 1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

nn2.mutate_neuron_number_of_layer(3, mutation_rate = 1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

nn2.mutate_activation_of_layer(2, mutation_rate=1)
print(nn2.calculate_output_to_input(input_to_nn[0:nn2.neurons[0]]))

# crossover
evolution.population[2] = nn2
nn3 = evolution.crossover_singlepoint(4,3)
print(nn3.calculate_output_to_input(input_to_nn[0:nn3.neurons[0]]))

nn3 = evolution.crossover_species(2,4)
print(nn3.calculate_output_to_input(input_to_nn[0:nn3.neurons[0]]))

nn4 = evolution.crossover_species(2,5)
print(nn4.calculate_output_to_input(input_to_nn[0:nn4.neurons[0]]))

# end of main
end_time = time.clock()
print('time elapsed: {}'.format(end_time-start_time))