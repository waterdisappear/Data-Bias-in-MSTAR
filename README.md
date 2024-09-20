<h1 align="center"> Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR </h1> 

<h5 align="center"><em> Weijie Li, Wei Yang, Li Liu, Wenpeng Zhang, and Yongxiang </em></h5>

<p align="center">
<a href="https://arxiv.org/abs/2304.00668"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
<a href="https://ieeexplore.ieee.org/document/10100951"><img src="https://img.shields.io/badge/Paper-IEEE%20GRSL-red"></a>
</p>

This paper</a> analyzes the contributions and interactions of targets, clutter, and shadow regions during training for MSTAR dataset. The contribution of clutter can be used as a quantitative indicator of the non-causality of deep learning. 

An example between clutter and overfitting：
<p align="center">
  <img src="https://github.com/waterdisappear/Data-Bias-in-MSTAR/blob/main/class_scr_convenext.jpg" width="480">
</p>


## data
The folder includes MSTAR data and images under SOC and SARbake segmentation files. Please unzip the files.

```bash
SOC: Ten classes of target recognition under standard conditions (JPEG-E)  
SARbake: Corresponding segmented dataset  
JPEG: Linear mapping  
JPEG-E: Linear mapping and contrast enhancement  
```

## exp1
ShapleyValue_Demo.py is a demo of calculating the Shapley value and binary Shapley interaction.

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

## exp5
Add SCR re-weighting during training.

## Contact us
If you have any questions, please contact us at lwj2150508321@sina.com

```
@ARTICLE{10100951,
  author={Li, Weijie and Yang, Wei and Liu, Li and Zhang, Wenpeng and Liu, Yongxiang},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Discovering and Explaining the Noncausality of Deep Learning in SAR ATR}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3266493}}
```

