# VA-TransUNet
Study together

```
@article{Jiang2022VATransUNetAU,
  title={VA-TransUNet: A U-shaped Medical Image Segmentation Network with Visual Attention},
  author={Ting Jiang and Tao Xu and Xiaoning Li},
  journal={Proceedings of the 2022 11th International Conference on Computing and Pattern Recognition},
  year={2022},
  url={https://dl.acm.org/doi/10.1145/3581807.3581826}
}

@article{蒋婷2024采用多尺度视觉注意力分割腹部,
  title={采用多尺度视觉注意力分割腹部CT和心脏MR图像},
  author={蒋婷 and 李晓宁},
  journal={中国图象图形学报},
  volume={29},
  number={1},
  pages={268},
}
```
## 安装环境和运行

## 安装conda虚拟环境
```
conda create -n segunet python=3.6 
```
激活环境
```
conda activate segunet
```
从torch官网选择对应版本的torch， 这根据您的服务器具体配置有所不同
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
安装依赖库
```
pip install tqdm h5py opencv-python scipy SimpleITK==2.0.0 medpy monai tensorboardX mmcv timm mmcv-full
```
可能版本有所差异，这需要您仔细选择。有缺少某些库，您记得运行时及时pip对应库即可

## 关于预训练模型，您可以从
```
https://github.com/Visual-Attention-Network
```
中获取对应的预训练模型，我们已经在网络代码中说明加载的模型名称，如果想达到最优的性能，您需要下载对应参数文件。

## 关于数据集
我们从TransUNet的作者那里获取的预处理好的Synapse、ACDC数据，您也可以发送邮件向我们获取。


## 对于networks.py和networks_msvan.py的差异
networks中的模型是那篇英文会议论文的模型；
networks_msvan是那篇中文期刊论文的模型。

因此，您仅简单需要调整导入的模型就可以完成对应的训练过程，以及修改对应的训练好文件的目录就可完成测试过程
```
python train.py
```
```
python test.py
```

# 致谢
非常感谢TransUNet的作者Jieneng Chen等人，向我们提供处理好的数据集以及开源的模型。您可以看出，我们的代码是基于他们的项目的。

非常感谢Visual Attention Network的作者Menghao Guo等人提供非常好的创意。



# 其他
后续待录用论文准备开源：
```
@article{郭逸凡2024采用不对称聚焦加权,
  title={采用不对称聚焦加权Dice损失分割腹部CT图像},
  author={郭逸凡 and 林佳成 and 蒋婷},
  journal={智能计算机与应用},
  volume={?},
  number={?},
  pages={?},
}
```
### 合作者的其他论文（已开源）
```
@article{郭2024hypersegunet,
  title={Hyper-SegUNet:基于超网络的超参自学习医学图像分割模型},
  author={郭逸凡 and 裴瑄 and 王大寒 and 陈培芝},
  journal={四川师范大学学报(自然科学版)},
  volume = {47(01)},
  pages = {127-135},
  year={2024}
}
```
```
https://github.com/MischiefGhostOgre/HyperSegUNet/
```
我们后续的工作将研究配准而不是分割。

