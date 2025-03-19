import tvm
from tvm import relax, te, topi
from tvm.relax.frontend.torch.fx_translator import from_fx
import tvm.relax as relax
from tvm.ir.module import IRModule
from typing import List
from typing import Tuple
import operators
import schedules
import numpy as np
import torch
import torch.nn.functional as F
import weight_utils as wu
import onnx
from tvm.relax.frontend.onnx import from_onnx

def get_parameters(input_shape: Tuple[int,...], var_biases: bool = False, aleatoric_head: bool = False, dtype:str = "float32"):
    raise NotImplementedError

def wrap_weights(weights, aleatoric_head: bool, var_biases: bool, device, dtype:str):    
    raise NotImplementedError

def load_weights(
    weight_path, 
    layout_weights=[[6,1,5,5],[16,6,5,5],[120,400],[84,120],[10,84]], 
    layout_bias=[6,16,120,84,10], 
    bias_var=False, aleatoric_head=False,
    layer_names=['conv1','conv2','fc1','fc2','fc3'],
    convert_varrho_to_2rawmoments=False, ### TODO 
    variance_rescale_factor=1.0,
):
    raise NotImplementedError


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

    shapes = [bias_shape1, shape1, bias_shape2, shape2, bias_shape3, shape3, bias_shape4, shape4, bias_shape5, shape5]
    weights_tvm = []
    for s in shapes:
        weights_tvm.append(tvm.nd.array(np.random.normal(0,1,s).astype(dtype), device=device))# mean weight

    return weights_tvm

# source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.max_pool2d1 = torch.nn.MaxPool2d(2)
        self.max_pool2d2 = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        c1 = self.relu(self.conv1(input))
        s2 = self.max_pool2d1(c1 )
        c3 = self.relu(self.conv2(s2))
        s4 = self.max_pool2d2(c3)
        s4 = torch.flatten(s4, 1)
        f5 = self.relu(self.fc1(s4))
        f6 = self.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output

def get_net(input_shape: Tuple[int,...], dtype:str = "float32") -> IRModule:
    assert dtype=="float32" # other dtypes not distributed to all layer types, inc. dense

    # source: https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html
    # https://discuss.tvm.apache.org/t/import-pytorch-model-by-relax-frontend/17371
    model = LeNet()
    model = model.eval()
    fx_model = torch.fx.symbolic_trace(model)
    
    input_info = [(input_shape, dtype)]
    mod = from_fx(fx_model, input_info, keep_params_as_input=True)
    (mod, params) = tvm.relax.frontend.detach_params(mod)

    # LOWERING, converts high-level relax ops to low-level tir implementation, which is suitable for meta schedules optimizations
    mod = relax.transform.LegalizeOps()(mod)
    print('low-level TIR model:', mod)

    return mod

