import numpy as np
import signal
import CNN
import scipy.signal as sp
import time
from UFE import layer as l
from UFE import model as m

def cnn_to_dense(): 
    step = 0.01
    cnn_layer = l.CNN()
    dense_layer = l.Dense((cnn_layer.output_length, 2), l.Activation.Linear)
    target = [1, 2]
    input = np.ones(cnn_layer.input_shape)

    dense_output = None

    for i in range(1500):
        # forward
        cnn_output = cnn_layer.forward(input)
        dense_output = dense_layer.forward(cnn_output.reshape(-1))
        

        # back prop
        dense_dEdo = dense_layer.backprop(dense_output - target, step)
        pre_cnn_dEdo = dense_dEdo.reshape(cnn_layer.output_shape)
        cnn_layer.backprop(pre_cnn_dEdo, step)

    if (np.sum(np.abs(dense_output - target)) > 0.01):
        raise Exception("Error too big")
    else:
        print("cnn_to_dense passed")

def dense_to_cnn():
    step = 0.01
    cnn_layer = l.CNN(activation=l.Activation.Linear, input_shape=(1, 4, 4))
    dense_layer = l.Dense((2, 4*4))
    input = [1, 2]
    target = np.ones(cnn_layer.output_shape)
    cnn_output = None

    for i in range(1500):
        # forward
        dense_output = dense_layer.forward(input)
        cnn_output = cnn_layer.forward(dense_output.reshape(cnn_layer.input_shape))

        # back prop
        cnn_dEdo = cnn_layer.backprop(cnn_output - target, step)
        pre_dense_dEdo = cnn_dEdo.reshape(dense_layer.output_shape)
        dense_layer.backprop(pre_dense_dEdo, step)

    if (np.sum(np.abs(cnn_output - target)) > 0.01):
        raise Exception("Error too big")
    else:
        print("dense_to_cnn passed")

def model_dense_to_cnn():
    model = m.model([l.Dense((2, 4*4)), 
             l.CNN(activation=l.Activation.Linear, input_shape=(1, 4, 4))])
    target = np.arange(model.output_layer.output_length).reshape(model.output_shape)
    output = None

    for i in range(1500):
        output = model.forward([1, 2])
        model.backprop_target(target, 0.001)

    if (np.sum(np.abs(output - target)) > 0.01):
        raise Exception("Error too big")
    else:
        print("model_dense_to_cnn passed")

def model_cnn_to_dense():
    cnn_layer = l.CNN(input_shape=(3, 12, 5), num_kernel=2, padding=False)
    model = m.model([cnn_layer, 
             l.Dense((cnn_layer.output_length, 2), l.Activation.Linear)])
    target = [1, 2]
    input = np.ones(cnn_layer.input_shape)
    output = None

    for i in range(1500):
        output = model.forward(input)
        model.backprop_target(target, 0.01)

    if (np.sum(np.abs(output - target)) > 0.01):
        raise Exception("Error too big")
    else:
        print("model_cnn_to_dense passed")


def main():
    cnn_to_dense()
    dense_to_cnn()
    model_cnn_to_dense()
    model_dense_to_cnn()


#if __name__ == '__main__':
#    main()

model = m.model([l.Dense((2, 16), l.Activation.Linear), l.CNN((1, 4, 4), padding = True, activation=l.Activation.Linear)])
target = np.random.normal(size=model.output_shape)
for i in range(10000):
    output = model.forward([0.5, -0.9])
    error = np.sum(np.abs(output - target))
    print(error)
    model.backprop_target(target, 0.1)
print(f"Target: \n{target}")




