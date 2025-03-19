import tvm
from tvm import relax, te
from tvm.ir.module import IRModule
from typing import List
from typing import Tuple
import operators
import schedules
import numpy as np
import torch
import math

import weight_utils as wu

def split_tuple(bb, in_tuple):
    return bb.emit(relax.TupleGetItem(in_tuple, 0)), bb.emit(relax.TupleGetItem(in_tuple, 1))

def fuse_head(mean:te.Tensor, var:te.Tensor, mean_ah:te.Tensor, var_ah:te.Tensor)-> te.Tensor:
    """
    a = mean and var of output layer
    b = mean and var of aleatoric head
    """
    t = relax.Tuple((mean, var, mean_ah, var_ah))

    return t

def add_dense(data_mean, data_var, weight_mean, weight_var, bb, last_layer=False, bias_mean=None, bias_var=None, v_mode_in = False, v_mode_weight = False, packed = False):
    """
    data_mean, data_var, weight_mean, weight_var: input tensors, if data_var is None calculate first layer
    bb: BlockBuilder to which the layer should be added
    last_layer: defines whether the output will be a Python-tuple or Relax-Tuple
    bias_mean, bias_var: optional bias  
    """
    out = bb.emit_te(operators.dense_pfp, data_mean, data_var, weight_mean, weight_var, bias_mean, bias_var, v_mode_in, v_mode_weight, packed)
    return split_tuple(bb, out)
    
def add_relu(activation_mean, activation_var, bb):
    """
    Adds a ReLU function to the BlockBuilder
    """
    out = bb.emit_te(operators.relu_pfp, activation_mean, activation_var)
    return bb.emit(relax.TupleGetItem(out, 0)), bb.emit(relax.TupleGetItem(out, 1))

def add_sigmoid(activation_mean, activation_var, bb):
    out = bb.emit_te(operators.relu_pfp, activation_mean, activation_var)
    return bb.emit(relax.TupleGetItem(out, 0)), bb.emit(relax.TupleGetItem(out, 1))

def get_parameters(input_shape: Tuple[int,...], input_shape_flattened: Tuple[int,...],  neurons: List[int], var_biases = True, aleatoric_head: bool = True, dtype:str = "float32"):
    """
    creates Relax.Vars of all weights and biases used in the network. This is needed for compilation.
    Parameters = List of all Relax.Vars in use, i.e. input, weight, bias
    Output is created by BlockBuilder via emit_output
    parameters[0] = input
    parameters[4i+1, 4i+2] = weights of a layer
    parameters[4i+3, 4i+4] = biases of a layer

    """
    input = relax.Var("input", relax.TensorStructInfo(input_shape, dtype))
    parameters = [input]

    for i, n in enumerate(neurons):
        if i == 0: # input layer
            shape = (n, input_shape_flattened[1])
        else: # hidden layer
            shape = (n, neurons[i-1])

        bias_shape = (n,) 
        parameters.append(relax.Var("fc" + str(i+1)+"_mean_weight", relax.TensorStructInfo(shape, dtype)))
        parameters.append(relax.Var("fc" + str(i+1)+"_variance_weight", relax.TensorStructInfo(shape, dtype)))
        parameters.append(relax.Var("fc" + str(i+1)+"_mean_bias", relax.TensorStructInfo(bias_shape, dtype)))
        if var_biases:
            parameters.append(relax.Var("fc" + str(i+1)+"_variance_bias", relax.TensorStructInfo(bias_shape, dtype)))

    if aleatoric_head:
        parameters.append(relax.Var("ah_mean_weight", relax.TensorStructInfo(shape, dtype)))
        parameters.append(relax.Var("ah_variance_weight", relax.TensorStructInfo(shape, dtype)))
        parameters.append(relax.Var("ah_mean_bias", relax.TensorStructInfo(bias_shape, dtype)))
        if var_biases:
            parameters.append(relax.Var("ah_variance_bias", relax.TensorStructInfo(bias_shape, dtype)))

    return parameters


def get_net(input_shape: Tuple[int,...], neurons: List[int], activation:str = "relu", var_biases: bool = True, aleatoric_head: bool = False, dtype:str = "float32") -> IRModule:
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

    params = get_parameters(input_shape, input_shape_flattened, neurons, var_biases=var_biases, aleatoric_head=aleatoric_head, dtype=dtype)

    if not activation in ["relu", "sigmoid"]:
        raise ValueError("Unsupported Activation function: " + activation + ". Allowed: Relu, Sigmoid")


    input_mean = params[0]
    input_var = None

    params_per_layer = 4 if var_biases else 3

    # assumption: aleatoric head is a pfp_dense operator, this must be adjusted if it is a regular dense layer
    if (len(params)-1) %params_per_layer !=0:
        raise ValueError("Illegal parameters in get_MLP: Must be " + str(params_per_layer) + " parameters per layer.")
    
    layers = len(params)//params_per_layer
    print('layers = ', layers)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            
            # flatten input first
            input_mean = bb.emit(relax.op.reshape( input_mean, (input_shape[0],-1)))
            
            for l in range(layers):
                last_layer = l == layers - 1
                if not last_layer:
                    activation_mean, activation_var = add_dense(input_mean, input_var, params[params_per_layer*l+1], params[params_per_layer*l+2], bb, bias_mean=params[params_per_layer*l+3], bias_var = params[params_per_layer*l+4] if var_biases else None)
                    if activation == "relu":
                        input_mean, input_var = add_relu(activation_mean, activation_var, bb)
                    elif activation == "sigmoid":
                        input_mean, input_var = add_sigmoid(activation_mean, activation_var, bb)
                    else:
                        raise NotImplementedError("Unknown Activation: " + activation + " in get_MLP!")
                else: # last layer
                    if not aleatoric_head: # last layer returns python tuple of tensors to emit_output, to create a Relax_tuple
                        output = add_dense(input_mean, input_var, params[params_per_layer*l+1], params[params_per_layer*l+2], bb, last_layer=True, bias_mean=params[params_per_layer*l+3], bias_var = params[params_per_layer*l+4] if var_biases else None)
                    else: # the last layer is the aleatoric head
                        mean_output_ah, var_output_ah = add_dense(input_mean, input_var, params[params_per_layer*(l)+1], params[params_per_layer*l+2], bb, last_layer=False, bias_mean=params[params_per_layer*l+3], bias_var = params[params_per_layer*l+4] if var_biases else None)
                        output = fuse_head(activation_mean, activation_var, mean_output_ah, var_output_ah)
            
            R = bb.emit_output(output)
        bb.emit_func_output(R, params=params)
    module = bb.get()
   
    return module

def create_dummy_weights(input_shape, neurons, aleatoric_head: bool, var_biases: bool, device, dtype:str):

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
            weight_shape = (n, input_shape[1])
        else: # hidden layer, also valid for aleatoric head?
            weight_shape = (n, neurons[l-1])

        bias_shape = (n,)
        weights_tvm.append(tvm.nd.array(np.random.normal(10,1,weight_shape).astype(dtype), device=device))# mean weight
        weights_tvm.append(tvm.nd.array(np.random.normal(10,1,weight_shape).astype(dtype), device=device))# var_weight
        weights_tvm.append(tvm.nd.array(np.random.normal(10,1,bias_shape).astype(dtype), device=device))# mean_bias
        if var_biases:
            weights_tvm.append(tvm.nd.array(np.random.normal(10,1,bias_shape).astype(dtype), device=device))# var_bias
        if (l == layers -1) and aleatoric_head:
            weights_tvm.append(tvm.nd.array(np.random.normal(10,1,weight_shape).astype(dtype), device=device))# ah_mean
            weights_tvm.append(tvm.nd.array(np.random.normal(10,1,weight_shape).astype(dtype), device=device))# ah_var
            weights_tvm.append(tvm.nd.array(np.random.normal(10,1,bias_shape).astype(dtype), device=device))# ah_bias_mean
            if var_biases:
                weights_tvm.append(tvm.nd.array(np.random.normal(10,1,bias_shape).astype(dtype), device=device))# ah_bias_var
   
    return weights_tvm

# load pyro VI weights
def load_weights(
    weight_path, layout_weights=[[50, 784], [50, 50], [10, 50]], layout_bias=[50, 50, 10], 
    bias_var=False, aleatoric_head=False,
    layer_names=['input_layer','hidden_layers.0','out_layer'],
    # convert_varrho_to_2rawmoments=True, ### TODO 
    variance_rescale_factor=1.0,
):
    """ load VI weiths to MLP
        For this PFP implementation, the weights has to be stored as second raw moment E[w], but not the weight of the first layer since there we don't have 
        variances on the activation path and need the classical formula
    """
    # 3 layer, 2 for mean and var, 2 for weight and bias
    assert len(layout_weights) == len(layout_bias), "Illegal weight configuration for loading!"
    layer = len(layout_bias)

    if aleatoric_head:
        raise NotImplementedError

    weights = {}
    if weight_path.endswith('pt'):
        # load torch model
        pretrained_model = torch.load(weight_path, map_location=torch.device('cpu'))

        # iterate over layers
        for l in range(layer):
            mean_weights = np.zeros(layout_weights[l])
            var_weights = np.zeros(layout_weights[l])
            mean_bias = np.zeros(layout_bias[l])
            if bias_var:
                var_bias = np.zeros(layout_bias[l])

            # weight means
            mean_weights[:] = wu.load_from_pyro_dict(pretrained_model, layer_names[l], 'locs', 'weight', to_numpy=True, variance_rescale_factor=variance_rescale_factor)
            weights[f"mw{l+1}"] = mean_weights

            # weight scales
            # only the first layer requires the weights in variance format, rest in  2nd raw moment format, 
            second_raw_moment_format = False if l==0 else True
            var_weights[:] = wu.load_from_pyro_dict(pretrained_model, layer_names[l], 'scales', 'weight', to_numpy=True, variance_rescale_factor=variance_rescale_factor)
            if second_raw_moment_format:
                weights[f"vw{l+1}"] = wu.convert_var_to_second_raw_moment(mean_weights, var_weights)
            else:
                weights[f"vw{l+1}"] = var_weights

            # bias
            mean_bias[:] = wu.load_from_pyro_dict(pretrained_model, layer_names[l], 'locs', 'bias', to_numpy=True, variance_rescale_factor=variance_rescale_factor)
            weights[f"mb{l+1}"] = mean_bias
            if bias_var:
                var_bias[:] = wu.load_from_pyro_dict(pretrained_model, layer_names[l], 'scales', 'bias', to_numpy=True, variance_rescale_factor=variance_rescale_factor)
                weights[f"vb{l+1}"] = var_bias

        return weights
    
    else:
        raise NotImplementedError


def wrap_weights(weights, aleatoric_head: bool, var_biases: bool, device, dtype:str):    
    params_per_layer = 4 if var_biases else 3
    layers = len(weights)//params_per_layer
    weights_tvm = []

    for l in range (layers):
        weights_tvm.append(tvm.nd.array(weights["mw" + str(l+1)].astype(dtype), device=device))
        weights_tvm.append(tvm.nd.array(weights["vw" + str(l+1)].astype(dtype), device=device))
        weights_tvm.append(tvm.nd.array(weights["mb" + str(l+1)].astype(dtype), device=device))
        if var_biases:
            weights_tvm.append(tvm.nd.array(weights["vb" + str(l+1)].astype(dtype), device=device))
        if aleatoric_head:
             weights_tvm.append(tvm.nd.array(weights["ah_m"].astype(dtype), device=device))
             weights_tvm.append(tvm.nd.array(weights["ah_v"].astype(dtype), device=device))
   
    return weights_tvm


