import os
from time import time
import pickle
import argparse
import wandb

import numpy as np
import torch # importing torch before tvm prevents segfault at end of program

import tvm
from tvm import relax
from tvm.ir.module import IRModule

import RPC
from RPC import module_upload
from devices import get_device
import schedules

import data_preprocessing_dirtyMNIST as dataDirtyMNIST
import MLP
import MLP_non_probabilistic
import LeNet
import LeNet_non_probabilistic
import uncertainty_utils



def create_args():
    parser = argparse.ArgumentParser()

    # monitroing
    parser.add_argument("--experiment_name", type=str, default=None)

    # monitoring
    parser.add_argument("--monitoring", action="store_true", default=False, dest='monitoring')
    parser.add_argument("--no-monitoring", action="store_false", dest='monitoring')
    parser.add_argument("--monitoring_project", type=str, default="tacoTVM")
    parser.add_argument("--monitoring_name", type=str, default=None)
    parser.add_argument("--monitoring_entity", type=str, default="uhdcsg")

    # model architecture
    parser.add_argument("--model_architecture", default="lenet", choices=["mlp","lenet"]) 
    parser.add_argument("--activation", type=str, default="relu", choices=["relu","sigmoid"]) 
    parser.add_argument("--aleatoric_head",    action="store_true",  dest="aleatoric_head")
    parser.add_argument("--no_aleatoric_head", action="store_false", dest="aleatoric_head")
    parser.set_defaults(aleatoric_head=False)
    parser.add_argument("--vectorized_maxpool", action="store_true",  dest="vectorized_maxpool")        # use vectorized opt. for kernel size 2
    parser.add_argument("--no_vectorized_maxpool", action="store_false",  dest="vectorized_maxpool")    # use generic pool impl. instead
    parser.set_defaults(vectorized_maxpool=True)
    # MLP specific
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=100)

    # model & dataset hyperparameters
    parser.add_argument("--batch_size", type=int, default=10) 
    parser.add_argument("--data_set", default="dirtyMNIST", choices=["noisy_sine","dirtyMNIST"]) 
    parser.add_argument("--dirtyMNIST_train_datasets", type=str,  default=["DirtyMNIST"],
            choices=["MNIST","AmbiguousMNIST","DirtyMNIST","FashionMNIST"]) 
    parser.add_argument("--dirtyMNIST_test_datasets", type=str,  default=["MNIST","AmbiguousMNIST","FashionMNIST"], nargs="+",
            choices=["MNIST","AmbiguousMNIST","DirtyMNIST","FashionMNIST"]) 

    # deterministic or probabilistic model
    parser.add_argument("--probabilistic_model", action="store_true", default=True, dest='probabilistic_model')
    parser.add_argument("--non_probabilistic_model", action="store_false", dest='probabilistic_model')
 

    # load model
    parser.add_argument("--pretrained_model_path", default=None) 
    parser.add_argument("--pretrained_weights_variance_rescale_factor", type=float, default=1.0)

    # noisy sine specfic
    parser.add_argument("--noisy_sine_type", default="taco1", choices=["taco1","M18c1","original"]) 

    # PFP specific
    parser.add_argument("--var_bias", action="store_true", dest="var_bias")
    parser.add_argument("--no_var_bias", action="store_false", dest="var_bias")
    parser.set_defaults(var_bias=True)

    # hardware
    parser.add_argument("--device_name", default="pi4", choices=["dev", "local", "pi3", "pi4","pi5"]) 
    parser.add_argument("--dtype", default="float32", choices=["float32"]) 

    # TVM
    # parser.add_argument("--profile", action="store_true", dest="profile")
    # parser.add_argument("--no_profile", action="store_false", dest="profile")
    # parser.set_defaults(profile=True)
    parser.add_argument("--execution_mode", default="benchmark", choices=["run","profile","benchmark"]) 
    parser.add_argument("--tune", action="store_true", dest="tune")
    parser.add_argument("--no_tune", action="store_false", dest="tune")
    parser.set_defaults(tune=False)
    parser.add_argument("--tvm_tuning_dir", default='./tuning_results') 

    parser.add_argument("--tune_max_trials_global", type=int, default=10000) 
    parser.add_argument("--tune_num_trials_per_iter", type=int, default=5000) 

    # Scheduling OPs
    # stochastic
    parser.add_argument("--tune_dense_stochastic", action="store_true", dest="tune_dense_stochastic")
    parser.add_argument("--no_tune_dense_stochastic", action="store_false", dest="tune_dense_stochastic")
    parser.set_defaults(tune_dense_stochastic=True)
    # reorder
    parser.add_argument("--tune_dense_reorder", action="store_true", dest="tune_dense_reorder")
    parser.add_argument("--no_tune_dense_reorder", action="store_false", dest="tune_dense_reorder")
    parser.set_defaults(tune_dense_reorder=True)
    # vectorize
    parser.add_argument("--tune_dense_vectorize", action="store_true", dest="tune_dense_vectorize")
    parser.add_argument("--no_tune_dense_vectorize", action="store_false", dest="tune_dense_vectorize")
    parser.set_defaults(tune_dense_vectorize=True)
    # unroll
    parser.add_argument("--tune_dense_unroll", action="store_true", dest="tune_dense_unroll")
    parser.add_argument("--no_tune_dense_unroll", action="store_false", dest="tune_dense_unroll")
    parser.set_defaults(tune_dense_unroll=True)
    # parallel 
    parser.add_argument("--tune_dense_parallel", action="store_true", dest="tune_dense_parallel")
    parser.add_argument("--no_tune_dense_parallel", action="store_false", dest="tune_dense_parallel")
    parser.set_defaults(tune_dense_parallel=True)
    # blocking/tiling
    parser.add_argument("--tune_dense_blocking", action="store_true", dest="tune_dense_blocking")
    parser.add_argument("--no_tune_dense_blocking", action="store_false", dest="tune_dense_blocking")
    parser.set_defaults(tune_dense_blocking=False)
    # packed
    parser.add_argument("--tune_dense_packed", action="store_true", dest="tune_dense_packed")
    parser.add_argument("--no_tune_dense_packed", action="store_false", dest="tune_dense_packed")
    parser.set_defaults(tune_dense_packed=False)

    # tune dense with MetaScheduler or Manual, MS makes above flags irrelevant
    parser.add_argument("--tune_dense_custom_schedule", action="store_true", dest="tune_dense_custom_schedule")
    parser.add_argument("--no_tune_dense_custom_schedule", action="store_false", dest="tune_dense_custom_schedule")
    parser.set_defaults(tune_dense_custom_schedule=False)

    # tune Pool
    parser.add_argument("--tune_lenet_pool", action="store_true", dest="tune_lenet_pool")
    parser.add_argument("--no_tune_lenet_pool", action="store_false", dest="tune_lenet_pool")
    parser.set_defaults(tune_lenet_pool=False)

    # benchmark
    parser.add_argument("--benchmark_repeat", type=int, default=10)
    parser.add_argument("--benchmark_number", type=int, default=100)
    parser.add_argument("--benchmark_min_repeat_ms", type=int, default=100)

    # debug
    parser.add_argument("--print_model", action="store_true", dest="print_model")
    parser.add_argument("--no_print_model", action="store_false", dest="print_model")
    parser.set_defaults(print_model=False)

    # get args
    args = parser.parse_args()
    return args

## acc ##
def calc_accuracy(means, true_labels):
    total = true_labels.shape[0]
    pred = np.argmax(means, axis=1)
    correct = np.sum(pred == true_labels)
    return correct/total

### run ##
def run_model(args, vm, parameters, input_data, device):
    # convert torch data to numpy to tvm
    data_np = input_data.detach().numpy().astype(args.dtype)
    data_tvm = tvm.nd.array(data_np, device=device) 
    params = [data_tvm, *parameters]
   
    # Step 8: Run
    if args.execution_mode=='profile':
        profile = vm.profile("main",  *params) # with general TVM runtime
        print(profile)
        return profile
    elif args.execution_mode=='benchmark':
        vm.set_input("main", *params)
        benchmark_fn = vm.time_evaluator(func_name="invoke_stateful", dev=device,  repeat=args.benchmark_repeat, number=args.benchmark_number, min_repeat_ms=args.benchmark_min_repeat_ms) # with general TVM runtime
        benchmark_results = benchmark_fn("main")
        print(benchmark_results)
        return benchmark_results
    elif args.execution_mode=='run':
        vm.set_input("main", *params)
        vm.invoke_stateful("main") # run async
        if args.probabilistic_model:
            m, v = vm.get_outputs("main") # syncs
            m = m.numpy()
            v = v.numpy()
        else:
            m = vm.get_outputs("main") # syncs
            m = m.numpy()
            v = np.zeros_like(m)
        return m, v
    else:
        raise NotImplementedError

 


#### MAIN ####
def main(args):

    # get RPC device info
    target, device_key, device_type = get_device(args.device_name)
    print(f"Device: {args.device_name}, target: {target}, rpc key: {device_key}, device type = {device_type}")

    # input tensor shape
    if args.data_set == "dirtyMNIST":
        input_shape=[args.batch_size,1,28,28]
        output_size = 10
        # flattened input for MLP
        if args.model_architecture=='mlp':
            #input_shape = [args.batch_size,784]
            input_shape_flattened = [args.batch_size,784]
        else:
            input_shape_flattened=None
    elif args.data_set =="noisy_sine":
        input_shape=[args.batch_size,1]
        input_shape_flattened = input_shape
        output_size = 1
    else:
        raise NotImplementedError


    # create neuron/weight list
    if args.model_architecture=='mlp':
        # neurons [50,50,10]
        neuron_list = []
        for i in range(args.hidden_layers+1):
            neuron_list.append(args.hidden_size)
        neuron_list.append(output_size)
        print(f"neurons: {neuron_list}")

        # weights [[50,784],[50,50],[10,50]]
        weight_size_list = []
        weight_size_list.append([args.hidden_size,input_shape_flattened[1]])
        for i in range(1,args.hidden_layers+1):
            weight_size_list.append([args.hidden_size,args.hidden_size])
        weight_size_list.append([output_size,args.hidden_size])
        print(f"weights: {weight_size_list}")

        # bias [[50],[50],[10]]
        bias_size_list = neuron_list
        print(f"bias: {bias_size_list}")

        # layer name list
        layer_names = ['input_layer']
        for i in range(args.hidden_layers):
            layer_names.append(f"hidden_layers.{i}")
        layer_names.append('out_layer')
    else:
        neuron_list=None
        layer_names=None
        bias_size_list=None
        weight_size_list=None

    # monitoring
    if args.monitoring:
        wb = wandb.init(project=args.monitoring_project, name=args.monitoring_name, config=args, save_code=True, entity=args.monitoring_entity)
        try:
            wb.config["TVM_target"] = target
            wb.config["device_type"] = device_type
            wb.config["input_shape"] = input_shape
            wb.config["input_shape_flattened"] = input_shape_flattened
            wb.config["neurons_list"] = neuron_list
            wb.config["weight_size_list"] = weight_size_list
            wb.config["layer_names"] = layer_names
            wb.config["bias_size_list"] = bias_size_list
        except:
            print('some argument is not defined')


    # Step 1: create IRModule of fully connected net
    if args.probabilistic_model:
        if args.model_architecture=='mlp':
           module = MLP.get_net(input_shape, neuron_list, args.activation, args.var_bias, args.aleatoric_head, args.dtype)
        elif args.model_architecture=='lenet':
            module = LeNet.get_net(input_shape, args.var_bias, args.aleatoric_head, args.dtype, vectorized_maxpool=args.vectorized_maxpool)
        else:
            raise NotImplementedError
    else:
        if args.model_architecture=='mlp':
            module = MLP_non_probabilistic.get_net(input_shape, neuron_list, args.activation, args.dtype)
        elif args.model_architecture=='lenet':
            module = LeNet_non_probabilistic.get_net(input_shape, args.dtype)
        else:
            raise NotImplementedError



    if args.print_model:
        module.show()

    # Step 2 (optional): apply optimizations to module
    if args.tune:
        print('# START TUNING #')
        module = schedules.tune(args, module, target, device_key, args.tvm_tuning_dir, max_trials_global=args.tune_max_trials_global, num_trials_per_iter=args.tune_num_trials_per_iter)
        print('# END TUNING #')
    else:
        # try to load tuned configuration
        print('try to load tuned configuration...')
        module = schedules.load_tuning(module, target, args.tvm_tuning_dir)


    # Step 3: compile for target architecture, interchangable with step 3
    t0 = time()
    print('build...')
    with tvm.transform.PassContext(opt_level=3):#  TODO does this do anything for relax?
        vm_exe = relax.build(module, target=target)
    print('...done in ', time() - t0, "s")


    # Step 4: initiate session with target device and upload the compiled Executable
    if args.device_name == "local":
        device = tvm.device(str(target), 0)
    else:
        vm_exe, device = RPC.module_upload(vm_exe, device_key=device_key)

    # Step 4: create VM to run the module
    profile = True if args.execution_mode=='profile' else False
    vm = relax.VirtualMachine(vm_exe, device, profile=profile)

    # Step 5: load weights
    if args.pretrained_model_path is None or args.pretrained_model_path=='None':
        print('use dummy weights')
        if args.model_architecture=='mlp':
            if args.probabilistic_model:
                parameters = MLP.create_dummy_weights(input_shape, neuron_list, args.aleatoric_head, args.var_bias, device, args.dtype)
            else:
                parameters = MLP_non_probabilistic.create_dummy_weights(input_shape, neuron_list, device, args.dtype)
        elif args.model_architecture=='lenet':
            if args.probabilistic_model:
                parameters = LeNet.create_dummy_weights(device, args.dtype)
            else:
                parameters = LeNet_non_probabilistic.create_dummy_weights(device, args.dtype)
        else:
            raise NotImplementedError
    else:
        if os.path.isfile(args.pretrained_model_path):
            # load model
            if args.model_architecture=='mlp':
                weights = MLP.load_weights(args.pretrained_model_path, layout_weights=weight_size_list, layout_bias=bias_size_list, bias_var=args.var_bias, layer_names=layer_names, aleatoric_head=args.aleatoric_head, variance_rescale_factor=args.pretrained_weights_variance_rescale_factor)
                parameters = MLP.wrap_weights(weights, args.aleatoric_head, args.var_bias, device, args.dtype)
            elif args.model_architecture=='lenet':
                weights = LeNet.load_weights(args.pretrained_model_path, bias_var=args.var_bias, aleatoric_head=args.aleatoric_head, variance_rescale_factor=args.pretrained_weights_variance_rescale_factor)
                parameters = LeNet.wrap_weights(weights, args.aleatoric_head, args.var_bias, device, args.dtype)
            else:
                raise NotImplementedError
        else:
            raise FileNotFoundError
     

    # Step 6: load/create input data
    if args.data_set == 'noisy_sine':
        raise NotImplementedError        
    elif args.data_set == 'dirtyMNIST':
        _, test_loader = dataDirtyMNIST.prepare_data(args.dirtyMNIST_train_datasets, args.dirtyMNIST_test_datasets, make_dataloaders=True,
                batch_size=args.batch_size)
    else:
        raise NotImplementedError


    # Step 7a: Profile run!
    if args.execution_mode=='profile':
        for test_set_name in test_loader.keys():
            print(f"## Testing {test_set_name} ##")
            test_set = test_loader[test_set_name]
            inputs, targets = next(iter(test_set)) # get one mini-batch
            m = v = None
            profile = run_model(args, vm, parameters, inputs, device)
            if args.monitoring:
                wb.log({f'{test_set_name}/profile':profile.csv()})
     
    # Setp 7: Iterate over testsets
    elif args.execution_mode=='run':
        variances_dict = {}
        for test_set_name in test_loader.keys():
            print(f"Testing {test_set_name}")
            test_set = test_loader[test_set_name]
            ms = []
            vs = []
            true_labels = []
            for idx, (inputs, targets) in enumerate(test_set):
                if idx%100==0:
                    print(f"run {test_set_name}: {100.0*idx/len(test_set):.2f}%")
                m, v = run_model(args, vm, parameters, inputs, device)
                if np.isnan(v).any():
                    print(f"NAN detected, in variances, mini-batch {idx}")
                if np.isnan(m).any():
                    print(f"NAN detected, in means, mini-batch {idx}")
                ms.append(m)
                vs.append(v)
                true_labels.append(targets.detach().numpy())
            print(f"run {test_set_name}: {100.0*(idx+1)/len(test_set):.2f}%")
            means = np.concatenate(ms)
            variances = np.concatenate(vs)
            true_labels = np.concatenate(true_labels)
            print(f"average variance: {variances.mean():.2E}")
            accuracy = calc_accuracy(means, true_labels)
            print(f"accuracy = {100.0*accuracy:.2f}%")
            variances_dict[f"{test_set_name}"] = variances

            if args.monitoring:
                log_dict = {
                    f"{test_set_name}/accuracy": accuracy,
                    f"{test_set_name}/variances": variances,
                    f"{test_set_name}/means": means,
                    f"{test_set_name}/variance": variances.mean(),
                }
                wb.log(log_dict)

        # Step 8 -- Analysis
        # calc Auroc
        auroc = uncertainty_utils.calculate_AUROC_TVMscript(variances_dict)
        print(f"AUROC = {auroc}")
        if args.monitoring:
            log_dict = {
                f"AUROC": auroc,
            }
            wb.log(log_dict)

    elif args.execution_mode=='benchmark':
        for test_set_name in test_loader.keys():
            print(f"## Testing {test_set_name} ##")
            test_set = test_loader[test_set_name]
            inputs, targets = next(iter(test_set)) # get one mini-batch
            m = v = None
            benchmark_results = run_model(args, vm, parameters, inputs, device)
            if args.monitoring:
                wb.log({
                    f'{test_set_name}/latency/mean':benchmark_results.mean,
                    f'{test_set_name}/latency/mean':benchmark_results.median,
                    f'{test_set_name}/latency/std':benchmark_results.std,
                    f'{test_set_name}/latency_results':benchmark_results.results,
                    f'{test_set_name}/benchmark':benchmark_results.__str__(),
                    })
     
    else:
        raise NotImplementedError
    
    # monitoring finalize
    if args.monitoring:
        wb.finish()
 

#### MAIN ####
if __name__ == "__main__":
    args = create_args()
    print(args)
    main(args)

    exit()
