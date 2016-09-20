Multi-Residual Networks
============================
By Masoud Abdi and Saeid Nahavandi


Implementation of Multi-Residual Networks (http://arxiv.org/abs/1609.05672).


Note: The code is based on https://github.com/facebook/fb.resnet.torch and https://github.com/KaimingHe/resnet-1k-layers


#### CIFAR-10 Test Error 

| Network       | depth |  k  | Test Error         | 
| ------------- | ----- | --- | ------------------ |
| Pre-Resnet    | 1001  |  1  |  4.62(4.69+/-0.20) | 
| Multi-Resnet  |  200  |  5  |  **4.35**(4.36+/-0.04) | 
| Multi-Resnet  |  398  |  5  |        **3.92**        | 



#Usage:
```bash
th main.lua -netType multi-resnet -depth 200 -k 5 -batchSize 64 -nGPU 2 -nThreads 4 -dataset cifar10 -nEpochs 200 -shareGradInput false
```









