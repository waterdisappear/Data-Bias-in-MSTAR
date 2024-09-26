<h1 align="center"> Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Li Liu (刘丽), Wenpeng Zhang (张文鹏), and Yongxiang (刘永祥) </em></h5>

<p align="center">
<a href="https://arxiv.org/abs/2304.00668"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
<a href="https://ieeexplore.ieee.org/document/10100951"><img src="https://img.shields.io/badge/Paper-IEEE%20GRSL-red"></a>
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Introduction">Catalog</a></li>
    <li><a href="#data">License</a></li>
    <li><a href="#exp1">Acknowledgement</a></li>
    <li><a href="#exp5">Citation</a></li>
  </ol>
</details>

## Introduction
This is the official repository for the paper “Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR”. 

这里是论文 “Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR (发现并解释SAR目标识别中深度学习的非因果性) ”的代码库。

**Abstract:** In recent years, deep learning has been widely used in synthetic aperture radar (SAR) automatic target recognition (ATR) and achieved excellent performance on the moving and stationary target acquisition and recognition (MSTAR) dataset. However, due to constrained imaging conditions, MSTAR has data biases such as background correlation, that is, background clutter properties have a spurious correlation with target classes. Deep learning can overfit clutter to reduce training errors. Therefore, the degree of overfitting for clutter reflects the noncausality of deep learning in SAR ATR. Existing methods only qualitatively analyze this phenomenon. In this letter, we quantify the contributions of different regions to target recognition based on the Shapley value. The Shapley value of clutter measures the degree of overfitting. Moreover, we explain how data bias and model bias contribute to noncausality. Concisely, data bias leads to comparable signal-to-clutter ratios (SCR) and clutter textures in training and test sets. And various model structures have different degrees of overfitting for these biases. The experimental results of various models under standard operating conditions (SOCs) on the MSTAR dataset support our conclusions. 

**摘要：** 近年来，深度学习在合成孔径雷达（SAR）自动目标识别（ATR）中得到了广泛应用，并在移动和静止目标获取与识别（MSTAR）数据集上取得了优异的性能。然而，由于成像条件的限制，MSTAR 存在背景相关性等数据偏差，即背景杂波属性与目标类别存在虚假相关性。使得深度学习可以对杂波进行过拟合，以减少训练误差。因此，杂波的过拟合程度反映了深度学习在 SAR ATR 中的非因果性。相比与现有对该现象的定性分析而言，我们在本文中根据 Shapley 值量化了不同区域对目标识别的贡献，将杂波的 Shapley 值衡量过拟合的程度。此外，我们还解释了数据偏差和模型偏差是如何导致非相关性的。简而言之，数据偏差会导致训练集和测试集中的信噪比（SCR）和杂波纹理具有相似性。而不同的模型结构对这些偏差有不同程度的过拟合。各种模型在 MSTAR 数据集标准操作条件（SOC）的实验结果支持了我们的结论。

We analyze the contributions and interactions of targets, clutter, and shadow regions during training for the MSTAR dataset. The contribution of clutter can be used as a quantitative indicator of the non-causality of deep learning. As shown in the Figure below, an example of clutter and bias is that the blue clutter contribution significantly impacts the classification of most targets. Besides, the SCR for each class in the training and test sets are very similar, indicating a background bias introduced during data collection.

我们分析了目标、杂波和阴影区域在 MSTAR 数据集训练过程中的贡献和相互作用。杂波的贡献可作为深度学习非因果关系的量化指标。如下图所示，杂波和过拟合的一个例子是，蓝色杂波的贡献极大地影响了大多数目标的分类。此外，训练集和测试集中每个类别的 SCR 都非常相似，这表明在数据收集过程中引入了背景偏差。

<figure>
<div align="center">
<img src=example/class_scr_convenext.jpg width="90%">
</div>
</figure>

## data
The folder includes MSTAR images under SOC and SARbake segmentation files. 

该文件夹包括 SOC 下的 MSTAR 图像以及 SARbake 分割文件。

```bash
SOC: Ten classes of target recognition under standard conditions (JPEG-E)  
SARbake: Corresponding segmented dataset  
JPEG: Linear mapping  
JPEG-E: Linear mapping and contrast enhancement  
```

```bash
SOC: 标准条件下的十类目标识别 (JPEG-E)  
SARbake: 对应的目标、阴影和背景分割文件
JPEG: 对于原始复数数据的幅度图像进行了线性映射
JPEG-E: 对于原始复数数据的幅度图像进行了线性映射和对比度增强
```

## exp1
ShapleyValue_Demo.py is a demo of calculating the Shapley value and binary Shapley interaction.

ShapleyValue_Demo.py 是一个计算沙普利值和二元沙普利值交互的demo。

```python
from exp1.DataLoad import load_data, load_test
from exp1.TrainTest import model_train, model_shapely, model_test, class_model_shapely
from model.Model import A_ConvNet

train_all, label_name = load_data(arg.data_path + 'TRAIN')
train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)

model = A_ConvNet(num_classes=len(label_name))
opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

for epoch in tqdm(range(1, arg.epochs + 1)):
    model_train(model=model, data_loader=train_loader, opt=opt)
    # calculate shapely value and binary Shapley interaction of each epoch
    history = model_shapely(model=model, data_loader=train_loader, his=history)
    
# calculate shapely value of each class
clu, tar, sha = class_model_shapely(model=model, data_loader=train_loader, label_length=len(label_name)) 
```

SCR_signal_to_clutter.py calculate the clutter mean and SCR of each class.

SCR_signal_too_clutter.py 用于计算每个类别的杂波均值和 SCR。

## exp5
Add SCR re-weighting during training to investigate whether changing the SCR would affect the degree of overfitting for clutter.

在训练过程中添加 SCR 重加权以研究改变SCR是否会影响对于杂波过拟合程度。

## model
Code for the eight models used for our experiment.

我们实验的八个模型的代码。

## Acknowledgement

Many thanks to the research [SARbake](https://data.mendeley.com/datasets/jxhsg8tj7g/3).


## Statement

- This project is released under the [Attribution-NonCommercial 4.0 International](LICENSE).
- Any questions please contact us at lwj2150508321@sina.com. 
- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

```
@ARTICLE{li2023discovering,
  author={Li, Weijie and Yang, Wei and Liu, Li and Zhang, Wenpeng and Liu, Yongxiang},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Discovering and Explaining the Noncausality of Deep Learning in SAR ATR}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3266493}
}

```

