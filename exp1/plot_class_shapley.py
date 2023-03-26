#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import rcParams
config = {
    "font.family": 'Times New Roman',
    "font.sans-serif": 'Times New Roman',
    "font.size": 14,
    "mathtext.fontset": 'stix',
    "font.serif": 'Times New Roman',
}
rcParams.update(config)
# 构建数据
blue = '#1f77b4'
yellow = '#ff7f0e'
grey = '#7f7f7f'

label = ['BTR70', 'BMPR2', 'BRDM2', 'BTR60', '2S1', 'T72', 'T62', 'ZIL131', 'D7', 'ZSU234']
x = [1,2,3,4,5,6,7,8,9,10]
model_name = 'Model_1'
# model_name = 'Model_2'
# model_name = 'efficientnet_b0'
# model_name = 'efficientnet_b1'
# model_name = 'Model_3'
# model_name = 'resnet_34'
# model_name = 'resnet_50'
# model_name = 'convenext_1'
data = np.load('../exp1/class_shapely_'+model_name+'.npy.npz')

class_clutter_shapley = data['class_clutter_shapley']
class_target_shapley = data['class_target_shapley']
class_shadow_shapley = data['class_shadow_shapley']

class_clutter_shapley_mean = class_clutter_shapley.mean(axis=0)
class_target_shapley_mean = class_target_shapley.mean(axis=0)
class_shadow_shapley_mean = class_shadow_shapley.mean(axis=0)

print(class_clutter_shapley_mean)
print(class_target_shapley_mean)
print(class_shadow_shapley_mean)

clutter = class_clutter_shapley_mean
target = class_target_shapley_mean
shadow = class_shadow_shapley_mean


y1 = 100*clutter/(np.abs(clutter)+np.abs(target)+np.abs(shadow))
y2 = 100*target/(np.abs(clutter)+np.abs(target)+np.abs(shadow))
y3 = 100*shadow/(np.abs(clutter)+np.abs(target)+np.abs(shadow))

y1_1 = np.maximum(y1, 0)
y1_2 = np.minimum(y1, 0)
y2_1 = np.maximum(y2, 0)
y2_2 = np.minimum(y2, 0)
y3_1 = np.maximum(y3, 0)
y3_2 = np.minimum(y3, 0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(x,y1_1,width=0.4,color=blue, label='Clutter',zorder=5)
ax1.bar(x,y1_2,width=0.4,color=blue,zorder=5)
ax1.bar(x,y2_1,width=0.4,bottom=y1_1,label='Target',color=yellow,zorder=5)
ax1.bar(x,y2_2,width=0.4,bottom=y1_2,color=yellow,zorder=5)
ax1.bar(x,y3_1,width=0.4,bottom=y2_1+y1_1,label='Shadow',color=grey,zorder=5)
ax1.bar(x,y3_2,width=0.4,bottom=y2_2+y1_2,color=grey,zorder=5)

plt.xticks(x,label, fontsize=10, rotation=20)#


# plt.bar(x,y3,width=0.4,bottom=y2,label='NO2',color='#00bfc4',edgecolor='grey',zorder=5)
ax1.set_ylabel('Shapley value ratio (%)')
ax1.set_ylim(-45,130)
# plt.yticks(np.arange(-0.4,1.05,0.2),[f'{i}%' for i in range(-40,105,20)])
plt.grid(axis='y',alpha=0.5,ls='--')

scr_1 = [9.32, 9.42, 9.72, 10.53, 11.00, 11.51, 13.83, 14.27, 16.57, 16.74]
scr_2 = [9.42, 10.01, 10.31, 11.18, 11.18, 12.01, 14.53, 14.79, 17.25, 17.34]
ax2 = ax1.twinx() # this is the important function
ax2.plot(x, scr_1, 'o-', color='deepskyblue', label='Training',zorder=6)
ax2.plot(x, scr_2, 'o-', color='cyan', label='Test',zorder=6)
ax2.set_ylim(8,20)
ax2.set_ylabel('SCR (dB)')
# ax2.plot(x, scr, 'o-', color='coral', label='Params')
# 添加图例，将图例移至图外
ax1.legend(loc='upper left', frameon=False, prop = {'size':12})
ax2.legend(loc='upper right', frameon=False, prop = {'size':12})
plt.tight_layout()
# ax1.legend(loc='lower right', bbox_to_anchor=(1.331, 0.0), frameon=False)
# ax2.legend(loc='lower right', bbox_to_anchor=(1.35, 0.2), frameon=False)

plt.savefig('./class_scr_'+model_name+'.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()
