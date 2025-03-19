# Probabilistic Forward Pass Operator Library

This repository provides an implementation of the **Probabilistic Forward Pass (PFP)** for efficient execution of **Bayesian Neural Networks (BNNs)** using **TVM**. 
The library enables optimized execution of both probabilistic and deterministic neural network models through code generation with TVM.

This repository is published alongside our research paper, where we present and evaluate the approach in detail.

## TVM Hardware Setup

To reproduce the experimental results, **Raspberry Pi devices** with the **TVM runtime** installed are required. 

A **TVM RPC tracker** is used to manage remote execution across multiple devices. The tracker functions as a job scheduler, coordinating all connected TVM devices.

To configure the available devices, modify the file `devices.py`:
- Ensure that the **RPC device names** and **compilation flags** match the available hardware.
- Set the **correct IP address** for the **TVM tracker**.

## Running Models

The following example commands demonstrate how to use the library to execute different model architectures, both for **PFP-based Bayesian Neural Networks** and **deterministic models**:

```
python main.py --model_architecture='mlp' --non_probabilistic_model --no_tune
python main.py --model_architecture='lenet' --probabilistic_model --no_tune
```

## Executing Pretrained BNNs

To run a **pretrained Bayesian Neural Network** based on **Variational Inference (VI)**, specify the model path and the rescale factor for the pretrained weight variances.

Example command:
```
python main.py --model_architecture='lenet' --probabilistic_model --no_tune --execution_mode='run' --pretrained_model_path='./pretrained_model/vi_model_20240730_e40_lenet.pt' --pretrained_weights_variance_rescale_factor=0.7
```

## Profiling and Benchmarking

The execution framework supports **three modes**:

- **run**: Executes the model on the entire dataset, reports **accuracy** and **AUROC**, and saves all predictions.
- **profile**: Profiles individual **TVM operators**, reporting detailed latency statistics.
- **benchmark**: Measures execution latencies precisely using multiple **repetitions**, reporting **mean** and **standard deviation**.

Example commands:
```
python main.py --model_architecture='lenet' --probabilistic_model --no_tune --execution_mode='run'
python main.py --model_architecture='lenet' --probabilistic_model --no_tune --execution_mode='profile'
python main.py --model_architecture='lenet' --probabilistic_model --no_tune --execution_mode='benchmark'
```

## Model Tuning

TVM’s **auto-tuning** can be enabled using the `--tune` flag. The tuning process searches for the best operator configurations and saves them in the specified directory.

### Running auto-tuning:
```
python main.py --model_architecture='lenet' --probabilistic_model --tune --tvm_tuning_dir='./tuning_results'
```

### Using saved tuning results:
Once tuning is completed, subsequent model executions will automatically use the **optimized operators** from the specified tuning folder:
```
python main.py --model_architecture='lenet' --probabilistic_model --no_tune --tvm_tuning_dir='./tuning_results'
```

### Advanced tuning options:
Moreover, many detailed flags exist to further specify the tuning options.

Example with advanced tuning configurations:
```
python main.py --model_architecture='lenet' --probabilistic_model --tune --tvm_tuning_dir='./tuning_results' --tune_max_trials_global=1000 --tune_num_trials_per_iter=500 --tune_dense_stochastic --tune_dense_vectorize --no_tune_lenet_pool
