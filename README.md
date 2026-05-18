# DTFR-LIMF
This is the official Pytorch implementation of the paper:

Enhancing learning efficiency in spiking neural networks via expanded representation and neuronal heterogeneity

## Dependencies
- Python 3
- PyTorch, torchvision
- spikingjelly 0.0.0.0.12
- Python packages: `pip install tqdm progress torchtoolbox thop`


## Training
Experiments on the CIFAR10-DVS dataset are implemented based on the [PSN](https://github.com/fangwei123456/Parallel-Spiking-Neuron) code; for details, please refer to the cifar10dvs folder. All other experiments are implemented in the DTFR-LIMF folder. These two folders should be regarded as two separate projects, and the following instructions are for the DTFR-LIMF folder. Records of the training process can be found in the ./logs folder.


    # CIFAR-10
	  python ./train.py -data_dir ./data_dir -dataset cifar10 -model spiking_resnet18 -T 4 -b 128 -T_max 400 -epochs 400 -weight_decay 5e-5 -neuron ILIF -cutupmix_auto -loss_lambda 0.05
    
    # CIFAR-100
    python ./train.py -data_dir ./data_dir -dataset cifar10 -model spiking_resnet18 -T 4 -b 128 -T_max 400 -epochs 400 -neuron ILIF -cutupmix_auto -lr 0.05 -loss_lambda 0.05
    
    # Tiny-Imagenet
    python ./train.py -data_dir ./data_dir -dataset tiny_imagenet -model vggsnn -T 4 -b 128 -T_max 200 -epochs 200 -neuron ILIF -cutupmix_auto -j 16 -loss_lambda 0.2 -mse_n_reg -lr 0.05 -loss_lambda 0.2
       
    # DVS-CIFAR10 (example)
	  python ./train.py -data_dir ./data_dir -dataset DVSCIFAR10 -T 4 -drop_rate 0.3 -model spiking_vgg11_bn -lr 0.05 -mse_n_reg -neuron ILIF -loss_lambda 0.3
	
	  # DVS-Gesture
    python ./train.py -data_dir ./data_dir -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -drop_rate 0.4 -T_max 200 -epochs 200 -neuron ILIF -lr 0.01 -loss_lambda 0.1


## Inference

    # example:
    python inference.py -data_dir ./data_dir -dataset tiny_imagenet -model spiking_vgg13_bn -b 256 -T 4 -neuron ILIF 
    -resume ./checkpoint_max.pth
