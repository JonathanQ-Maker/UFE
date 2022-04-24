from UFE import layer as l
import importlib
import numpy as np

class model:

    def __init__(self, layers=[]):
        self.layers = layers
        self.input = None
        self.output = None
        if (len(self.layers) > 0):
            self.output_layer = self.layers[-1]
            self.input_layer = self.layers[0]
            self.output_shape = self.output_layer.output_shape
            self.input_shape = self.input_layer.input_shape

        self.update_adapters()
     
    def update_adapters(self):
        self.adapters = {}
        length = len(self.layers)
        for i in range(length):
            if (i + 1 < length):
                current_layer = self.layers[i]
                next_layer = self.layers[i + 1]
                if (type(current_layer) == l.Dense and type(next_layer) == l.CNN):
                    # Dense to CNN
                   self.adapters[i] = DenseToCNN(current_layer, next_layer)
                elif (type(current_layer) == l.CNN and type(next_layer) == l.Dense):
                    # CNN to Dense
                    self.adapters[i] = CnnToDense(current_layer, next_layer)


    def forward(self, input):
        self.input = input
        output = input
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
            if (self.adapters.get(i) != None):
                output = self.adapters[i].adapte_input(output)
        self.output = output
        return self.output

    def backprop(self, dEdo, step, clip_value=0.51):
        for i in range(len(self.layers)-1, -1, -1):
            dEdo = self.layers[i].backprop(dEdo, step, clip_value)
            next_index = i - 1
            if (next_index >= 0):
                if (self.adapters.get(next_index) != None):
                    dEdo = self.adapters[next_index].adapte_gradient(dEdo)
        return dEdo

    def backprop_target(self, target, step, clip_value=0.5):
        return self.backprop(self.output - target, step, clip_value)

    def save(self, path):
        print(f"Saving model to {path}.npy...")
        data = {}
        layers = []
        for layer in self.layers:
            layers.append(layer.serialize())
        data["layers"] = layers
        data["output_shape"] = self.output_shape
        data["input_shape"] = self.input_shape
        np.save(path, data)
        print(f"Save success")

    def load(self, path):
        print(f"Loading model from {path}.npy...")
        data = np.load(path+".npy", allow_pickle=True).item()
        layers = data["layers"]
        self.layers = []
        for layer in layers:
            layer_class = getattr(importlib.import_module("UFE.layer"), layer["type"])
            layer_instance = layer_class()
            layer_instance.populate(layer)
            self.layers.append(layer_instance)
        self.output_shape = data["output_shape"]
        self.input_shape = data["input_shape"]
        self.output_layer = self.layers[-1]
        self.input_layer = self.layers[0]
        self.update_adapters()
        print(f"Load success")

    

class LayerAdapter:
    """
    Base class of layer adapters, do not instantiate.

    When two different types of layers are connected together,
    some reshaping of input matrix is nessesary. 
    This class aims to handle such operations
    """

    def __init__(self, parent_layer: l.Layer, child_layer: l.Layer):
        self.parent_layer = parent_layer
        self.child_layer = child_layer

    def adapte_input(self, input):
        return input

    def adapte_gradient(self, dEdo):
        return dEdo

class DenseToCNN(LayerAdapter):

    def __init__(self, parent_layer: l.Dense, child_layer: l.CNN):
        super().__init__(parent_layer, child_layer)

    def adapte_input(self, input):
        return input.reshape(self.child_layer.input_shape)

    def adapte_gradient(self, dEdo):
        return dEdo.reshape(self.parent_layer.output_shape)[0]

class CnnToDense(LayerAdapter):

    def __init__(self, parent_layer: l.CNN, child_layer: l.Dense):
        super().__init__(parent_layer, child_layer)

    def adapte_input(self, input):
        return input.flatten()

    def adapte_gradient(self, dEdo):
        return dEdo.reshape(self.parent_layer.output_shape)