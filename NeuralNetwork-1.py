from _csv import reader
from random import random
import numpy
import pandas as pd
import sklearn.preprocessing
from numpy import shape
from math import exp

def activate(wts, inputs):
    activation = wts[-1]
    for i in range(len(wts) - 1):
        activation += wts[i] * float(inputs[i])
    return activation


def sigmoid_activation(activation):
    return 1.0 / (1.0 + exp(-activation))


# def Tanh_activation(activaion):
#
#     return (exp(activaion)-exp(-activaion))/(exp(activaion)+exp(-activaion))
def Tanh_activation(activaion):
    tan=numpy.tanh(activaion)
    return -2*tan*(1-tan**2)

# def Relu(activaion):
#
#     return (activaion/(1+exp(-(activaion))))

def forward_propagate(network, row, act_fun):
    inputs = row
    for layer in network:
        # print(layer)
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['wts'], inputs)
            neuron['gd_o'] = activation
            if int(act_fun) == 1:
                neuron['output'] = sigmoid_activation(activation)


            if int(act_fun)==2:
                neuron['output'] = Tanh_activation(activation)
            # if int(act_fun)==3:
            #     neuron['output'] = Relu(activation)

            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_err(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        err = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                errs = 0.0
                for neuron in network[i + 1]:
                    errs += (neuron['wts'][j] * neuron['delta'])
                err.append(errs)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                err.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = err[j] * transfer_derivative(neuron['output'])


def predict(network, row, act_fun):
    outputs = forward_propagate(network, row, act_fun)
    return outputs.index(max(outputs))+3


def initialize_NeuralNetwork(no_inputs, n_hidden, no_outputs, no_hiddden):
    network = list()
    for k in range(no_hiddden):
        hidden_layer = [{'wts': [random() for i in range(no_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'wts': [random() for i in range(n_hidden + 1)]} for i in range(no_outputs)]
    network.append(output_layer)
    return network


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Update network wts with err
def update_wts(network, row, l_rate):
    # print("update")
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['wts'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['wts'][-1] -= l_rate * neuron['delta']


def train_neuralNetwork(network, dataset, l_rate, n_epoch, no_outputs, act_fun):
    for epoch in range(n_epoch):
        sum_err = 0
        c = 0
        for row in dataset:
            outputs = forward_propagate(network, row, act_fun)
            expected = [0 for i in range(no_outputs)]

            expected[int(row[-1])-3] = 1
            # print("expected",expected)
            sum_err += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_err(network, expected)
            update_wts(network, row, l_rate)
            c += 1
        print('>epoch=%d, lrate=%.3f, err=%.3f' % (epoch, l_rate, sum_err))
        count=0
        for row in dataset:
            prediction = predict(network, row, act_fun)
            if prediction == int(row[-1]):
                count += 1
        accuracy = (count / shape(dataset)[0]) * 100
        print("accuracy for epoch-%d: %f"% (epoch,accuracy))



def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (float(row[i]) - float(minmax[i][0])) / (float(minmax[i][1]) - float(minmax[i][0]))


def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# importing dataset


filename = 'winequality-white.csv'
dataset = list()
with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
dataset.pop(0)
dataset=pd.DataFrame(dataset)
min_max=sklearn.preprocessing.MinMaxScaler()
dataset.iloc[:,:-1]=min_max.fit_transform(dataset.iloc[:,:-1])
dataset=dataset.values.tolist()
print("select the activation function you would like to use:"
      "1:sigmoid"
      "2:Tanh")
act_fun = input()
no_inputs = len(dataset[0]) - 1
no_outputs = len(set([row[-1] for row in dataset]))
network = initialize_NeuralNetwork(no_inputs, no_inputs, no_outputs,3)
lenght_dataset=shape(dataset)
i=0.1

count=0
for row in dataset:
    prediction = predict(network, row, act_fun)
    # print('Expected=%d, Got=%d' % (int(row[-1]), prediction))
    if prediction == int(row[-1]):
        count += 1
accuracy=(count/shape(dataset)[0])*100
print("accuracy before training:",accuracy)
train_neuralNetwork(network,dataset, 0.1, 10, no_outputs, act_fun)

print("------------------------------------------------------------------------------------------accuracy after training")
count=0
output= list()
for row in dataset:
    prediction = predict(network, row, act_fun)
    if prediction==int(row[-1]):
        count+=1
    output.append(prediction)
accuracy=(count/shape(dataset)[0])*100
print("accuracy after training:",accuracy)
val_outputs = (set(output))
print("output values",val_outputs)
