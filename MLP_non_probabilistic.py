import tvm
from tvm import relax, te
from tvm.ir.module import IRModule
from tvm.relax.testing import nn
from typing import List
from typing import Tuple
import operators
import schedules
import numpy as np
import torch
import math

import weight_utils as wu

def get_parameters(input_shape: Tuple[int,...], input_shape_flattened: Tuple[int,...],  neurons: List[int], var_biases = True, aleatoric_head: bool = True, dtype:str = "float32"):
    raise NotImplementedError

def load_weights(
    weight_path, layout_weights=[[50, 784], [50, 50], [10, 50]], layout_bias=[50, 50, 10], 
    bias_var=False, aleatoric_head=False,
    layer_names=['input_layer','hidden_layers.0','out_layer'],
    variance_rescale_factor=1.0,
):
    raise NotImplementedError

def wrap_weights(weights, aleatoric_head: bool, var_biases: bool, device, dtype:str):    
    raise NotImplementedError

def create_dummy_weights(input_shape, neurons,  device, dtype:str):

    # inputs shape needs to be flat [mini-batch-size, c*h*w]
    if len(input_shape) > 2:
        m = 1
        for l in input_shape[1:]:
            m *= l
        input_shape = [input_shape[0], m]
    
    layers = len(neurons)
    weights_tvm = []

    for l, n in enumerate(neurons): # TODO use correct shapes per layer and type, copy these shapes from get_parameters
        if l == 0: # input layer
            weight_shape = (input_shape[1],n) # inverse weight order for the deterministic case
        else: # hidden layer, also valid for aleatoric head?
            weight_shape = (neurons[l-1],n) # inverse order

        print(weight_shape)
        bias_shape = (n,)
        weights_tvm.append(tvm.nd.array(np.random.normal(0.0,1,weight_shape).astype(dtype), device=device))# mean weight
        weights_tvm.append(tvm.nd.array(np.random.normal(1,1,bias_shape).astype(dtype), device=device))# mean_bias
    return weights_tvm


def get_net(input_shape: Tuple[int,...], neurons: List[int], activation:str = "relu", dtype:str = "float32") -> IRModule:
    """
    Creates an IRModule that contains a net specified in the arguments.
    Currently only supports an MLP-structure.
    3 instead of 4 for no var_biases
    parameters[4i+1, 4i+2] = weights of a layer
    parameters[4i+3, 4i+4] = biases of a layer
    """

    m = 1
    for l in input_shape[1:]:
        m *= l
    input_shape_flattened = [input_shape[0], m]

    if not activation in ["relu", "sigmoid"]:
        raise ValueError("Unsupported Activation function: " + activation + ". Allowed: Relu, Sigmoid")

    bb = relax.BlockBuilder()
    with bb.function("main"):

        assert activation=='relu'
        assert len(neurons)==3

        model = nn.Sequential(
            nn.Linear(input_shape_flattened[1],neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0],neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1],neurons[2]),
        )
        data = nn.Placeholder(input_shape, name="data")
        data2 = bb.emit(relax.op.reshape( data, (input_shape[0],-1)))
        output = model(data2)
        params = [data] + model.parameters()
        bb.emit_func_output(output, params=params)
    
    module = bb.get()
    #module.show()
    
    return module

