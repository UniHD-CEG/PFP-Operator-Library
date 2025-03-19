import tvm
from tvm import relax, te, topi
from tvm.ir.module import IRModule
from typing import List
from typing import Tuple
import operators
import schedules
import numpy as np
import torch
import weight_utils as wu

def pp():
    pp.counter +=1
    return pp.counter
pp.counter = -1

def fuse_head(mean:te.Tensor, var:te.Tensor, mean_ah:te.Tensor, var_ah:te.Tensor)-> te.Tensor:
    """
    a = mean and var of output layer
    b = mean and var of aleatoric head
    """
    t = relax.Tuple((mean, var, mean_ah, var_ah))

    return t

def add_dense(bb, data_mean, data_var, weight_mean, weight_var, bias_mean=None, bias_var=None, convert_var_activations_to_2ndrawmoment = False, convert_var_weights_to_2ndrawmoment = False):
    """
    data_mean, data_var, weight_mean, weight_var: input tensors, if data_var is None calculate first layer
    bb: BlockBuilder to which the layer should be added
    last_layer: defines whether the output will be a Python-tuple or Relax-Tuple
    bias_mean, bias_var: optional bias  
    """
    out = bb.emit_te(operators.dense_pfp, data_mean, data_var, weight_mean, weight_var, bias_mean, bias_var, convert_var_activations_to_2ndrawmoment, convert_var_weights_to_2ndrawmoment)
    return split_tuple(bb, out)
    
def add_relu(bb, activation_mean, activation_var, return_variance=False):
    """
    Adds a ReLU function to the BlockBuilder
    """
    out = bb.emit_te(operators.relu_pfp, activation_mean, activation_var, variance_mode=return_variance)
    return split_tuple(bb, out)

def add_sigmoid(bb, activation_mean, activation_var):
    out = bb.emit_te(operators.sigmoid_pfp, activation_mean, activation_var, False)
    return split_tuple(bb, out)

def add_conv(bb, input_mean, input_var, filter_mean, filter_var, stride, pad, bias_mean=None, bias_var=None, dtype: str = "float32", convert_var_activations_to_2ndrawmoment= False, convert_var_weights_to_2ndrawmoment = False ):
    out = bb.emit_te(operators.conv_pfp_2d, input_mean, input_var, filter_mean, filter_var, stride, pad, bias_mean, bias_var, convert_var_activations_to_2ndrawmoment, convert_var_weights_to_2ndrawmoment, True, dtype)
    return split_tuple(bb, out) 

def add_pool(bb, input_mean, input_var, pool_size=(1,1), stride=(1,1), dilation=(1,1), padding=(0,0), layout='NCHW', ceil_mode=True, dtype='float32'):
    out = bb.emit_te(operators.max_pool_pfp, input_mean, input_var, pool_size, stride, dilation, padding, layout, ceil_mode, dtype)
    return split_tuple(bb, out)

def add_pool_fast_split(bb, input_mean, input_var):
    split = bb.emit_te(operators.lenet_split, input_mean, input_var)
    split_result = [bb.emit(relax.TupleGetItem(split, i)) for i in range(8)]

    out = [add_pool_fast(bb, split_result[i], split_result[i+1]) for i in range(0, len(split_result), 2)]
    m1 = bb.emit(relax.op.concat([out[0][0], out[1][0]], axis = 3))
    m2 = bb.emit(relax.op.concat([out[2][0], out[2][0]], axis = 3))
    v1 = bb.emit(relax.op.concat([out[0][1], out[1][1]], axis = 3))
    v2 = bb.emit(relax.op.concat([out[2][1], out[2][1]], axis = 3))

    m = bb.emit(relax.op.concat([m1, m2], axis = 2))
    v = bb.emit(relax.op.concat([v1, v2], axis = 2))

    return m, v

def add_pool_fast(bb, input_mean, input_var):
    out = bb.emit_te(operators.lenet_pool, input_mean, input_var)
    return split_tuple(bb, out)

def split_tuple(bb, in_tuple):
    return bb.emit(relax.TupleGetItem(in_tuple, 0)), bb.emit(relax.TupleGetItem(in_tuple, 1))

def get_parameters(input_shape: Tuple[int,...], var_biases: bool = False, aleatoric_head: bool = False, dtype:str = "float32"):
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
    # assume that the input has no variance information, change here if there is input var available
    #input_variance = relax.Var("input_variance", relax.TensorStructInfo(input_shape, dtype))
    #parameters = [input, input_variance]

    #shape1 = (6,1,28,28)
    shape1 = (6,1,5,5)
    parameters.append(relax.Var("conv1_mean_weight", relax.TensorStructInfo(shape1, dtype)))
    parameters.append(relax.Var("conv1_variance_weight", relax.TensorStructInfo(shape1, dtype)))
    bias_shape1 = (6, )
    parameters.append(relax.Var("conv1_mean_bias", relax.TensorStructInfo(bias_shape1, dtype)))
    if var_biases:
        parameters.append(relax.Var("conv1_variance_bias", relax.TensorStructInfo(bias_shape1, dtype)))

    #shape2 = (16,6,10,10)
    shape2 = (16,6,5,5)
    parameters.append(relax.Var("conv2_mean_weight", relax.TensorStructInfo(shape2, dtype)))
    parameters.append(relax.Var("conv2_variance_weight", relax.TensorStructInfo(shape2, dtype)))
    bias_shape2 = (16, )
    parameters.append(relax.Var("conv2_mean_bias", relax.TensorStructInfo(bias_shape2, dtype)))
    if var_biases:
        parameters.append(relax.Var("conv2_variance_bias", relax.TensorStructInfo(bias_shape2, dtype)))

    shape3 = (120,400)
    parameters.append(relax.Var("fc1_mean_weight", relax.TensorStructInfo(shape3, dtype)))
    parameters.append(relax.Var("fc1_variance_weight", relax.TensorStructInfo(shape3, dtype)))
    bias_shape3 = (120, )
    parameters.append(relax.Var("fc1_mean_bias", relax.TensorStructInfo(bias_shape3, dtype)))
    if var_biases:
        parameters.append(relax.Var("fc1_variance_bias", relax.TensorStructInfo(bias_shape3, dtype)))

    shape4 = (84,120)
    parameters.append(relax.Var("fc2_mean_weight", relax.TensorStructInfo(shape4, dtype)))
    parameters.append(relax.Var("fc2_variance_weight", relax.TensorStructInfo(shape4, dtype)))
    bias_shape4 = (84, )
    parameters.append(relax.Var("fc2_mean_bias", relax.TensorStructInfo(bias_shape4, dtype)))
    if var_biases:
        parameters.append(relax.Var("fc2_variance_bias", relax.TensorStructInfo(bias_shape4, dtype)))

    shape5 = (10,84)
    parameters.append(relax.Var("fc3_mean_weight", relax.TensorStructInfo(shape5, dtype)))
    parameters.append(relax.Var("fc3_variance_weight", relax.TensorStructInfo(shape5, dtype)))
    bias_shape5 = (10, )
    parameters.append(relax.Var("fc3_mean_bias", relax.TensorStructInfo(bias_shape5, dtype)))
    if var_biases:
        parameters.append(relax.Var("fc3_variance_bias", relax.TensorStructInfo(bias_shape5, dtype)))

    if aleatoric_head:
        parameters.append(relax.Var("ah_mean_weight", relax.TensorStructInfo(shape5, dtype)))
        parameters.append(relax.Var("ah_variance_weight", relax.TensorStructInfo(shape5, dtype)))
        parameters.append(relax.Var("ah_mean_bias", relax.TensorStructInfo(bias_shape5, dtype)))
        if var_biases:
            parameters.append(relax.Var("ah_variance_bias", relax.TensorStructInfo(bias_shape5, dtype)))

    return parameters


def get_net(input_shape: Tuple[int,...], var_biases: bool = False, aleatoric_head: bool =False, dtype:str = "float32", vectorized_maxpool=True) -> IRModule:
    assert dtype=="float32" # other dtypes not distributed to all layer types, inc. dense

    parameters = get_parameters(input_shape, var_biases=var_biases, aleatoric_head=aleatoric_head, dtype=dtype)

    for p in parameters:
        print(p)
    
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            m, v = add_conv(bb, parameters[pp()], None, parameters[pp()], parameters[pp()], stride=1, pad=2, bias_mean=parameters[pp()], dtype=dtype, bias_var=parameters[pp()] if var_biases else None)   
            m, v = add_relu(bb, m, v, return_variance=True)
            if vectorized_maxpool:
                m, v = add_pool_fast_split(bb, m, v)
            else:
                 m, v = add_pool(bb, m, v, pool_size=(2,2), stride=(2,2), dtype=dtype)
            m, v = add_conv(bb, m, v, parameters[pp()], parameters[pp()], stride=1, pad=0, bias_mean=parameters[pp()], dtype=dtype, bias_var=parameters[pp()] if var_biases else None, convert_var_activations_to_2ndrawmoment=True)   
            m, v = add_relu(bb, m, v, return_variance=True)
            if vectorized_maxpool:
                m, v = add_pool_fast(bb, m, v)
            else:
                m, v = add_pool(bb, m, v, pool_size=(2,2), stride=(2,2), dtype=dtype)
            m = bb.emit(relax.op.reshape( m, (input_shape[0],400)))
            v = bb.emit(relax.op.reshape( v, (input_shape[0],400)))
            m, v = add_dense(bb, m, v, parameters[pp()], parameters[pp()], parameters[pp()], parameters[pp()] if var_biases else None, convert_var_activations_to_2ndrawmoment=True)
            m, v = add_relu(bb, m, v)
            m, v = add_dense(bb, m, v, parameters[pp()], parameters[pp()], parameters[pp()], parameters[pp()] if var_biases else None)
            m, v = add_relu(bb, m, v)
            output = add_dense(bb, m, v, parameters[pp()], parameters[pp()], parameters[pp()], parameters[pp()] if var_biases else None) 
            R = bb.emit_output(output)
        bb.emit_func_output(R, params=parameters)

    module = bb.get()

    # test legalize for pool
    module = relax.transform.LegalizeOps()(module)
    # print('low-level TIR model:', module)

    return module

def create_dummy_weights(device, dtype:str = "float32"):

    shape1 = (6,1,5,5)
    shape2 = (16,6,5,5)
    shape3 = (120,400)
    shape4 = (84,120)
    shape5 = (10,84)

    bias_shape1 = (6, )
    bias_shape2 = (16, )
    bias_shape3 = (120, )
    bias_shape4 = (84, )
    bias_shape5 = (10, )



    shapes = [shape1, bias_shape1, shape2, bias_shape2, shape3, bias_shape3, shape4, bias_shape4, shape5, bias_shape5]
    weights_tvm = []
    for s in shapes:
        weights_tvm.append(tvm.nd.array(np.random.normal(0,1,s).astype(dtype), device=device))# mean weight
        weights_tvm.append(tvm.nd.array(weights_tvm[-1].numpy()**2, device=device))


   
    return weights_tvm

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




# load pyro VI weights
def load_weights(
    weight_path, 
    layout_weights=[[6,1,5,5],[16,6,5,5],[120,400],[84,120],[10,84]], 
    layout_bias=[6,16,120,84,10], 
    bias_var=False, aleatoric_head=False,
    layer_names=['conv1','conv2','fc1','fc2','fc3'],
    convert_varrho_to_2rawmoments=False, ### TODO 
    variance_rescale_factor=1.0,
):
    """ load VI weiths to LeNet
        For the PFP impl. here, the weights has to be stored  as second raw moment E[w], but not the weight of the first layer since there we don't have 
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
            print(f"l={l}, second_raw_moment_format={second_raw_moment_format}")
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

