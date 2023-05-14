# Self2Self Pytorch Implementation

## Introduction
This is a pytorch implementation of [Self2Self](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf), "Yuhui Quan, Mingqin Chen, Tongyao Pang, Hui Ji; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1890-1898."

It is a pytorch reimplementation of the [tensorflow version](https://github.com/scut-mingqinchen/self2self). 

You can simply run main.py with the default parameters:
```
sh main.sh
```
The denoised images will be saved in images/, and the logs will be saved in logs/.

## Details of reimplementation

There are some notable details in the conversion of tensorflow to pytorch, which will significantly effect the performance. 

### Partial Convolution 2D layer
Pytorch has a package of the implementation of [Pconv2d](https://github.com/DesignStripe/torch_pconv). However, the implementation details is different from that of the tensorflow version. Specifically, the variable **mask** in the tensorflow is a 4-d tensor with shape (1, channel, width, height), but a 3-d tensor with shape (1, width, height) in the pytorch package. We have implemented a Pconv2d structure consistent with the tensorflow version, by modifying the source code of the pytorch package. 

### Optimizer
The implementation details of Adam between tensorflow and pytorch have slight difference. However, it is widely discussed the suboptimal convergence in PyTorch compared to TensorFlow when using Adam optimizer. 

## Update Log

### 2023-05-14 
- Found and fixed a bug in line 144 of file "network/pconv.py", which enables our implementation to achieve comparable denosing performance with the tensorflow version. 
- Found that changing the optimizer from Adam to AdamW achieves better denosing performance. (We still keep the Adam optimizer, in order to keep up with the original tensorflow version of the implementation)

Thanks to @haimiaozh for all the contributions to improving this project! 
