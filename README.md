Multi-Residual Networks
============================
By Masoud Abdi and Saeid Nahavandi


Implementation of Multi-Residual Networks (http://arxiv.org/abs/1609.05672).


#### CIFAR Test Error Rates

| Network       | depth |  k  |  w  | Parameters | CIFAR-10 (%)         |  CIFAR-100 (%)          | 
| ------------- | ----- | --- | --- | ---------- | -------------------- | ----------------------- |
| Pre-Resnet    | 1001  |  1  |  1  |   10.2M    |  4.62(4.69+/-0.20)   |   22.71(22.68+/-0.22)   | 
| Multi-Resnet  |  200  |  5  |  1  |   10.2M    |**4.35**(4.36+/-0.04) | **20.42**(20.44+/-0.15) | 
| Multi-Resnet  |  398  |  5  |  1  |   20.4M    |       3.92           |        20.59            |
| Multi-Resnet  |  26   |  2  |  10 |    72M     |       3.96           |       **19.45**         | 
| Multi-Resnet  |  26   |  4  |  10 |    154M    |       **3.73**       |        19.60            |

##Usage:
```bash
th main.lua -netType multi-resnet -depth 200 -k 5 -batchSize 64 -nGPU 2 -nThreads 4 -dataset cifar10 -nEpochs 200
```




##Notes: 

In order to see the effect of model parallelism use modelParallel (Tested on K80 GPU):

for model parallelism on 2 GPUs use:
```bash
th main.lua -netType mpreresnet -dataset cifar10 -batchSize 128 -depth 110 -k 4 -modelParallel true
```

for data parallelism on 2 GPUs use:
```bash
th main.lua -netType mpreresnet -dataset cifar10 -batchSize 128 -depth 434 -k 1 -nGPU 2
```

It achieves up to 15% speed up and there is room for improvement.


The code is based on https://github.com/facebook/fb.resnet.torch and https://github.com/KaimingHe/resnet-1k-layers


##Contact

Please contact me on mabdi{at}deakin.edu.au

I appreciate any discussion, suggestion or contribution.










