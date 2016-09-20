Multi-Residual Networks
============================
By Masoud Abdi and Saeid Nahavandi


This code is the implementation of Multi-Residual Networks (http://arxiv.org/abs/1609.05672).



Usage:
```bash
th main.lua -netType multi-resnet -depth 200 -k 5 -batchSize 64 -nGPU 2 -nThreads 4 -dataset cifar10 -nEpochs 200 -shareGradInput false
```


Note:
The code is based on
https://github.com/facebook/fb.resnet.torch
and 
https://github.com/KaimingHe/resnet-1k-layers






