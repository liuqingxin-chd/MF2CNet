# Remote Sensing Image Scene Classification Using Multiscale Feature Fusion Covariance Network With Octave Convolution

This repository is the implementation of our paper: [Remote Sensing Image Scene Classification Using Multiscale Feature Fusion Covariance Network With Octave Convolution](https://ieeexplore.ieee.org/document/9737532). 

If you find this work helpful, please cite our paper:

    @ARTICLE{9737532,
    author={Bai, Lin and Liu, Qingxin and Li, Cuiling and Ye, Zhen and Hui, Meng and Jia, Xiuping},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Remote Sensing Image Scene Classification Using Multiscale Feature Fusion Covariance Network With Octave Convolution}, 
    year={2022},
    volume={60},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2022.3160492}}
 
 ## Descriptions
 
In this article, a multiscale feature fusion covariance network (MF2CNet) with octave convolution (Oct Conv) is proposed, which can extract multifrequency and multiscale features from RSIs. First, the multifrequency feature extraction (MFE) module is used to obtain fine-grained frequency features by Oct Conv. Then, the features of different layers in MF2CNet are fused by the multiscale feature fusion (MF2) module. Finally, instead of using global average pooling (GAP), global covariance pooling (GCP) extracts high-order information from RSIs to capture richer statistics of deep features. In the proposed MF2CNet, the obtained multifrequency and multiscale features can effectively improve the performance of CNNs. Experimental results on four public RSI datasets show that MF2CNet has advantages in RSSC over current state-of-the-art methods.
 
 ![image](https://github.com/liuqingxin-chd/MF2CNet/blob/main/network.jpg)
