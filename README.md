# Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR

This paper first analyzes the contributions and interactions of targets, clutter, and shadow regions during training. The contribution of clutter can be used as a quantitative indicator of the non-causality of deep learning. 
https://arxiv.org/abs/2304.00668

## data
The folder includes MSTAR data and images under SOC and SARbake segmentation files. Please unzip the files.

```bash
SOC: Ten classes of target recognition under standard conditions (JPEG-E)  
SARbake: Corresponding segmented dataset  
JPEG: Linear mapping  
JPEG-E: Linear mapping and contrast enhancement  
```

## exp1
Demo of the Shapley value and binary Shapley interaction

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
  
clu, tar, sha = class_model_shapely(model=model, data_loader=train_loader, label_length=len(label_name)) # calculate shapely value of each classes
```
