import enum
import numpy as np
import scipy.signal as sp


class Activation(enum.Enum):
    Relu = 1,
    Linear = 2,
    LeakyRelu = 3,
    Tanh = 4


class Layer:
    """
    Base class of all layers, do not instantiate
    """

    def __init__(self, activation: Activation=Activation.Linear):
        self.activation = activation
        self.input = 0
        self.output = 0
        self.activation_output = 0
        self.output_shape = ()
        self.input_shape = ()
        self.bias = 0
        self.weights = 0

    def forward(self, input):
        self.input = input
        self.output = input
        return self.activate()

    def backprop(self, dEdo, step, clip_value=0.5):
        return dEdo

    def activate(self):
        if (self.activation == Activation.Relu):
            self.activation_output = np.maximum(self.output, 0)
        elif (self.activation == Activation.Linear):
            self.activation_output = self.output
        elif (self.activation == Activation.LeakyRelu):
            self.activation_output = np.maximum(self.output, self.output * 0.1)
        elif(self.activation == Activation.Tanh):
            self.activation_output = np.tanh(self.output)
        else:
            raise Exception(f"{self.activation} is unknown")
        return self.activation_output

    def activation_gradient(self):
        if (self.activation == Activation.Relu):
            return np.piecewise(self.output, [self.output < 0, self.output >= 0], [lambda x: 0, lambda x: 1])
        elif (self.activation == Activation.Linear):
            return 1
        elif (self.activation == Activation.LeakyRelu):
            return np.piecewise(self.output, [self.output < 0, self.output >= 0], [lambda x: 0.1, lambda x: 1])
        elif (self.activation == Activation.Tanh):
            return 1 - np.power(self.activation_output, 2)
        else:
            raise Exception(f"{self.activation} is unknown")

    def serialize(self):
        result = {}
        result["activation_id"] = self.activation.value
        result["output_shape"] = self.output_shape
        result["input_shape"] = self.input_shape
        result["type"] = type(self).__name__

        if (type(self.bias) == np.ndarray):
            result["bias"] = self.bias.tolist()
        else:
            result["bias"] = self.bias
        if (type(self.weights) == np.ndarray):
            result["weights"] = self.weights.tolist()
        else:
            result["weights"] = self.weights
        return result

    def populate(self, data):
        self.activation = Activation(data["activation_id"])
        self.output_shape = data["output_shape"]
        self.input_shape = data["input_shape"]
        if (type(data["bias"]) == list):
            self.bias = np.array(data["bias"])
        else:
            self.bias = data["bias"]

        if (type(data["weights"]) == list):
            self.weights = np.array(data["weights"])
        else:
            self.weights = data["weights"]


class CNN(Layer):
    def __init__(self, input_shape = (1, 10, 10), kernel_size = 3, num_kernel = 1, activation = Activation.Relu, padding=False):
        """
        input_shape: (depth, height, width)
        kernel_size: int, generates individual kernel of shape (size, size)
        num_kernel: int, generates kernel of shape (num_kernel, size, size)
        activation: Activation enum value
        """

        super().__init__(activation)

        self.kernel_momentum = 0
        self.bias_momentum = 0

        self.input_shape = input_shape # (input depth, height, width)
        self.input_depth, input_height, input_width = input_shape
        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.kernel_shape = (num_kernel, self.input_depth, kernel_size, kernel_size)
        self.output_shape = (num_kernel, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.mode = "valid"
        self.dEdx_mode = "full"
        self.padding = padding
        if (self.padding):
            self.mode = "same"
            self.dEdx_mode = "same"
            self.output_shape = (self.num_kernel, input_height, input_width)
        self.output_length = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        self.input_length = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        
        self.kernel = np.random.normal(scale=np.sqrt(2 / (self.kernel_shape[1] * self.kernel_shape[2] * self.kernel_shape[3])), size=self.kernel_shape)
        self.bias = np.zeros(self.output_shape)
        self.weights = self.kernel

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)
        for i in range(self.num_kernel):
            for j in range(self.input_depth):
                self.output[i] += sp.correlate2d(self.input[j], self.kernel[i, j], self.mode)
        return self.activate()

    def backprop(self, dEdo, step, clip_value=0.5):
        dEdo *= self.activation_gradient()
        dEdx = np.zeros(self.input_shape)
        dEdk = np.zeros(self.kernel_shape)
        for i in range(self.num_kernel):
            for j in range(self.input_depth):
                dEdk[i, j] = sp.correlate2d(self.input[j], dEdo[i], "valid")
                dEdx[j] += sp.convolve2d(dEdo[i], self.kernel[i, j], self.dEdx_mode)

                # faster with big matracies and more memory
                #dEdx += sp.fftconvolve([dEdo[i]], self.kernel[i], self.dEdx_mode)
                #dEdx[j] += sp.fftconvolve(dEdo[i], self.kernel[i, j], self.dEdx_mode)
        kernel_update = np.clip(dEdk, -clip_value, clip_value)
        bias_update = np.clip(dEdo, -clip_value, clip_value)

        bias_update = bias_update * step + 0.9 * self.bias_momentum
        self.bias -= bias_update
        self.bias_momentum = bias_update
        
        kernel_update = kernel_update * step + 0.9 * self.kernel_momentum
        self.kernel -= kernel_update
        self.kernel_momentum = kernel_update
        return dEdx

    def serialize(self):
        result = super().serialize()
        result["input_depth"] = self.input_depth
        result["num_kernel"] = self.num_kernel
        result["kernel_size"] = self.kernel_size
        result["kernel_shape"] = self.kernel_shape
        result["mode"] = self.mode
        result["dEdx_mode"] = self.dEdx_mode
        result["padding"] = self.padding
        result["output_length"] = self.output_length
        result["input_length"] = self.input_length
        return result

    def populate(self, data):
        super().populate(data)
        self.kernel = self.weights
        self.input_depth = data["input_depth"]
        self.num_kernel = data["num_kernel"]
        self.kernel_size = data["kernel_size"]
        self.kernel_shape = data["kernel_shape"]
        self.mode = data["mode"]
        self.dEdx_mode = data["dEdx_mode"]
        self.padding = data["padding"]
        self.output_length = data["output_length"]
        self.input_length = data["input_length"]

class Dense(Layer):
    def __init__(self, shape=(), activation = Activation.Relu, isLast=False):
        """
        Shape format: (# of inputs, # of neurons)
        """
        super().__init__(activation)
        self.shape = shape
        self.weights_momentum = 0
        self.bias_momentum = 0
        if (len(self.shape) == 2):
            self.bias = np.zeros(self.shape[1])
            self.weights = np.random.normal(scale=np.sqrt(2 / self.shape[0]), size=self.shape)
            self.output_shape = (1, self.shape[1])
            self.input_shape = (1, self.shape[0])



    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.activate()

    def backprop(self, dEdo, step, clip_value=0.5):
        dEdo *= self.activation_gradient()
        weights_update = np.clip(np.reshape(self.input, (-1, 1)) * dEdo, 
                            -clip_value,
                            clip_value)
        bias_update = np.clip(dEdo, -clip_value, clip_value)

        bias_update = bias_update * step + 0.9 * self.bias_momentum
        self.bias -= bias_update
        self.bias_momentum = bias_update

        weights_update = weights_update * step + 0.9 * self.bias_momentum
        self.weights -= weights_update
        self.weights_momentum = weights_update

        return np.sum(self.weights * dEdo, axis=1)

    def serialize(self):
        result = super().serialize()
        result["shape"] = self.shape
        return result

    def populate(self, data):
        super().populate(data)
        self.shape = data["shape"]

