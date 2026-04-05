<h1 align="center"> 🔍 Discovering and Explaining the Non‑Causality of Deep Learning in SAR ATR </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Li Liu (刘丽), Wenpeng Zhang (张文鹏), and Yongxiang Liu (刘永祥) </em></h5>

<p align="center">
  <a href="#Introduction">📖 Introduction</a> |
  <a href="#Data">📁 Data</a> |
  <a href="#Exp1">🧪 Exp1</a> |
  <a href="#Exp5">⚙️ Exp5</a> |
  <a href="#Model">🤖 Model</a> |
  <a href="#Acknowledgement">🙏 Acknowledgement</a> |
  <a href="#Statement">📜 Statement</a>
</p>


<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10100951"><img src="https://img.shields.io/badge/Paper-IEEE%20GRSL-blue"></a>
  <a href="https://arxiv.org/abs/2304.00668"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
  <a href="https://zhuanlan.zhihu.com/p/630435432"><img src="https://img.shields.io/badge/文章-知乎-blue"></a> 
</p>


---

## 📖 Introduction

This is the official repository for the paper *“Discovering and Explaining the Non‑Causality of Deep Learning in SAR ATR”*. If you find our work useful, please give us a star ⭐ on GitHub and cite our paper using the BibTeX format at the end.

这里是论文 *“Discovering and Explaining the Non‑Causality of Deep Learning in SAR ATR” (发现并解释SAR目标识别中深度学习的非因果性)* 的代码库。如果您觉得我们的工作有价值，请在 GitHub 上给我们点个星星 ⭐，并按页面最后的 BibTeX 格式引用我们的论文。

---

**Abstract:**  
In recent years, deep learning has been widely used in synthetic aperture radar (SAR) automatic target recognition (ATR) and has achieved excellent performance on the moving and stationary target acquisition and recognition (MSTAR) dataset. However, due to constrained imaging conditions, MSTAR has data biases such as background correlation – i.e., background clutter properties are spuriously correlated with target classes. Deep learning can overfit to clutter to reduce training errors. Therefore, the degree of overfitting to clutter reflects the non‑causality of deep learning in SAR ATR. Existing methods only qualitatively analyze this phenomenon. In this letter, we quantify the contributions of different regions to target recognition based on the Shapley value. The Shapley value of clutter measures the degree of overfitting. Moreover, we explain how data bias and model bias contribute to non‑causality. Concisely, data bias leads to comparable signal‑to‑clutter ratios (SCR) and clutter textures in training and test sets, and various model structures have different degrees of overfitting to these biases. The experimental results of various models under standard operating conditions (SOC) on the MSTAR dataset support our conclusions.

**摘要：**  
近年来，深度学习在合成孔径雷达（SAR）自动目标识别（ATR）中得到了广泛应用，并在移动和静止目标获取与识别（MSTAR）数据集上取得了优异的性能。然而，由于成像条件的限制，MSTAR 存在背景相关性等数据偏差，即背景杂波属性与目标类别存在虚假相关性。深度学习可以对杂波进行过拟合以减少训练误差。因此，杂波的过拟合程度反映了深度学习在 SAR ATR 中的非因果性。相比现有对该现象的定性分析，本文根据 Shapley 值量化了不同区域对目标识别的贡献，用杂波的 Shapley 值来衡量过拟合的程度。此外，我们还解释了数据偏差和模型偏差如何导致非因果性。简而言之，数据偏差导致训练集和测试集中具有相似的信杂比（SCR）和杂波纹理，而不同的模型结构对这些偏差有不同程度的过拟合。多种模型在 MSTAR 数据集标准操作条件（SOC）下的实验结果支持了我们的结论。

---

We analyze the contributions and interactions of target, clutter, and shadow regions during training on the MSTAR dataset. The contribution of clutter can serve as a quantitative indicator of the non‑causality of deep learning. As shown in the figure below, an example of clutter and bias is that the blue clutter contribution significantly impacts the classification of most targets. Additionally, the SCR for each class in the training and test sets is very similar, indicating a background bias introduced during data collection.

我们分析了在 MSTAR 数据集训练过程中，目标、杂波和阴影区域的贡献及其相互作用。杂波的贡献可作为深度学习非因果性的量化指标。如下图所示，杂波和偏差的一个例子是：蓝色杂波区域对大多数目标的分类都有显著影响。此外，训练集和测试集中每个类别的 SCR 都非常相似，这表明数据采集过程中引入了背景偏差。

<figure>
<div align="center">
<img src=example/class_scr_convenext.jpg width="90%">
</div>
</figure>


---

## 📁 Data

The folder includes MSTAR images under SOC and the SARbake segmentation files.

该文件夹包含 SOC 条件下的 MSTAR 图像以及 SARbake 分割文件。

```bash
SOC: Ten classes of target recognition under standard conditions (JPEG-E)  
SARbake: Corresponding segmented dataset  
JPEG: Linear mapping  
JPEG-E: Linear mapping and contrast enhancement  
```

```bash
SOC: 标准条件下的十类目标识别 (JPEG-E)  
SARbake: 对应的目标、阴影和背景分割文件
JPEG: 对原始复数数据的幅度图像进行线性映射
JPEG-E: 对原始复数数据的幅度图像进行线性映射和对比度增强
```

------

## 🧪 Exp1

**`ShapleyValue_Demo.py`** – a demo for calculating the Shapley value and binary Shapley interaction.

**`ShapleyValue_Demo.py`** – 计算 Shapley 值和二元 Shapley 交互的示例程序。

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

**`SCR_signal_to_clutter.py`** – calculates the clutter mean and SCR for each class.

**`SCR_signal_to_clutter.py`** – 计算每个类别的杂波均值和信杂比（SCR）。

------

## ⚙️ Exp5

Add SCR re‑weighting during training to investigate whether changing the SCR affects the degree of overfitting to clutter.

在训练过程中加入 SCR 重加权，以研究改变 SCR 是否会影响对杂波的过拟合程度。

------

## 🤖 Model

Code for the eight models used in our experiments.

我们实验中所用的八个模型的代码。

------

## 🙏 Acknowledgement

Many thanks to the research [SARbake](https://data.mendeley.com/datasets/jxhsg8tj7g/3).

衷心感谢 [SARbake](https://data.mendeley.com/datasets/jxhsg8tj7g/3) 研究团队。

------

## 📜 Statement

- This project is released under the [CC BY‑NC 4.0](https://license/) license.
- For any questions, please contact us at: **lwj2150508321@sina.com**.
- If you find our work useful, please give us a star ⭐ on GitHub and cite our paper using the following BibTeX entry:

```bibtex
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
