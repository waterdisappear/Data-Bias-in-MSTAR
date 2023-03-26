import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from matplotlib import rcParams

blue = '#1f77b4'
yellow = '#ff7f0e'
grey = '#7f7f7f'

config = {
    "font.family": 'Times New Roman',
    "font.sans-serif": 'Times New Roman',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": 'Times New Roman',
}
rcParams.update(config)


def plot_ratio(model_name, i=0):
    data = np.load('shapely_'+model_name+'.npz')
    x = data['x']
    clutter_shapley = data['clutter_shapley']
    target_shapley = data['target_shapley']
    shadow_shapley = data['shadow_shapley']
    acc_train = data['acc_train']
    acc_test = data['acc_test']

    TEMP = (clutter_shapley[i]+target_shapley[i]+shadow_shapley[i])
    clutter_shapley_mean = clutter_shapley[i]/TEMP
    target_shapley_mean = target_shapley[i]/TEMP
    shadow_shapley_mean = shadow_shapley[i]/TEMP

    fig, ax1 = plt.subplots()
    # plt.yticks(np.arange(0, 110, 20), [f'{i}%' for i in range(0, 110, 20)])
    ax1.plot(x, clutter_shapley_mean*100, color=blue, label='Clutter')
    ax1.plot(x, target_shapley_mean*100, color=yellow, label='Target')
    ax1.plot(x, shadow_shapley_mean*100, color=grey, label='Shadow')

    ax2 = ax1.twinx()  # mirror the ax1，
    ax2.plot(x, acc_train[i], 'r-', label='Training')
    ax2.plot(x, acc_test[i], 'r--', label='Test')
    # plt.yticks(np.arange(0, 110, 20), [f'{i}%' for i in range(0, 110, 20)])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Shapley value ratio (%)')
    ax2.set_ylabel('Accuracy (%)')
    ax1.legend(bbox_to_anchor=(0.6, 0.0), loc='lower center', frameon=False, prop = {'size':11})
    ax2.legend(loc='lower right', frameon=False, prop = {'size':11})
    # ax1.set_ylim(-10, 110)
    # ax2.set_ylim(-10, 110)
    ax1.grid(alpha=0.5,ls='-.')
    plt.savefig(  model_name+'_shapley_ratio.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

def plot_shapley(model_name, i=0):
    data = np.load('shapely_'+model_name+'.npz')
    x = data['x']
    clutter_shapley = data['clutter_shapley']
    target_shapley = data['target_shapley']
    shadow_shapley = data['shadow_shapley']
    acc_train = data['acc_train']
    acc_test = data['acc_test']

    clutter_shapley_mean = clutter_shapley[i]
    target_shapley_mean = target_shapley[i]
    shadow_shapley_mean = shadow_shapley[i]

    fig, ax1 = plt.subplots()
    ax1.plot(x, clutter_shapley_mean, color=blue, label='Clutter')
    ax1.plot(x, target_shapley_mean, color=yellow, label='Target')
    ax1.plot(x, shadow_shapley_mean, color=grey, label='Shadow')

    ax2 = ax1.twinx()  # mirror the ax1，
    ax2.plot(x, acc_train[i], 'r-', label='Training')
    ax2.plot(x, acc_test[i], 'r--', label='Test')
    # plt.yticks(np.arange(0, 110, 20), [f'{i}%' for i in range(0, 110, 20)])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Shapley value')
    ax2.set_ylabel('Accuracy (%)')
    ax1.legend(bbox_to_anchor=(0.6, 0.0), loc='lower center', frameon=False, prop = {'size':11})
    ax2.legend(loc='lower right', frameon=False, prop = {'size':11})
    # ax1.set_ylim(0, 120)
    # ax2.set_ylim(0, 120)
    ax1.grid(alpha=0.5,linestyle='-.')
    plt.savefig(  model_name+'_shapley.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

def plot_Bshapley(model_name, i=0):
    data = np.load('shapely_'+model_name+'.npz')
    x = data['x']
    clutter_shapley = data['clutter_target']
    target_shapley = data['target_shadow']
    shadow_shapley = data['shadow_clutter']
    acc_train = data['acc_train']
    acc_test = data['acc_test']

    clutter_shapley_mean = clutter_shapley[i]
    target_shapley_mean = target_shapley[i]
    shadow_shapley_mean = shadow_shapley[i]

    fig, ax1 = plt.subplots()


    ax1.plot(x, clutter_shapley_mean, color=blue, label='Clutter&Target')
    ax1.plot(x, target_shapley_mean, color=yellow, label='Target&Shadow')
    ax1.plot(x, shadow_shapley_mean, color=grey, label='Shadow&Clutter')
    ax2 = ax1.twinx()  # mirror the ax1，
    ax2.plot(x, acc_train[i], 'r-', label='Training')
    ax2.plot(x, acc_test[i], 'r--', label='Test')
    # plt.yticks(np.arange(0, 110, 20), [f'{i}%' for i in range(0, 110, 20)])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Bivariate Shapley interaction')
    ax2.set_ylabel('Accuracy (%)')
    ax1.legend(bbox_to_anchor=(0.6, 0.0), loc='lower center', frameon=False, prop = {'size':11})
    ax2.legend(loc='lower right', frameon=False, prop = {'size':11})
    # ax1.set_ylim(0, 120)
    # ax2.set_ylim(0, 110)
    # ax2.set_xlim(-10, 155)
    ax1.grid(alpha=0.5,linestyle='-.')
    plt.savefig(model_name+'_bshapley.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

if __name__ == '__main__':
    model_name = 'Model_1'
    plot_ratio(model_name)
    plot_shapley(model_name)
    plot_Bshapley(model_name)

    data = np.load('shapely_'+model_name+'.npz')

    print(data['acc_test'].mean(axis=0)[-1])
    print(data['acc_train'].mean(axis=0)[-1])
    print(data['clutter_shapley'].mean(axis=0)[-1])
    print(data['target_shapley'].mean(axis=0)[-1])
    print(data['shadow_shapley'].mean(axis=0)[-1])

    print(100*data['clutter_shapley'].mean(axis=0)[-1]/(data['clutter_shapley'].mean(axis=0)[-1]+data['target_shapley'].mean(axis=0)[-1]+data['shadow_shapley'].mean(axis=0)[-1]))
    print(100*data['target_shapley'].mean(axis=0)[-1]/(data['clutter_shapley'].mean(axis=0)[-1]+data['target_shapley'].mean(axis=0)[-1]+data['shadow_shapley'].mean(axis=0)[-1]))
    print(100*data['shadow_shapley'].mean(axis=0)[-1]/(data['clutter_shapley'].mean(axis=0)[-1]+data['target_shapley'].mean(axis=0)[-1]+data['shadow_shapley'].mean(axis=0)[-1]))

    print(data['clutter_target'].mean(axis=0)[-1])
    print(data['target_shadow'].mean(axis=0)[-1])
    print(data['shadow_clutter'].mean(axis=0)[-1])

    data = np.load('class_shapely_'+model_name+'.npy.npz')

    class_clutter_shapley=data['class_clutter_shapley']
    class_target_shapley = data['class_target_shapley']
    class_shadow_shapley = data['class_shadow_shapley']

    class_clutter_shapley_mean = class_clutter_shapley.mean(axis=0)
    class_target_shapley_mean = class_target_shapley.mean(axis=0)
    class_shadow_shapley_mean = class_shadow_shapley.mean(axis=0)

    print(class_clutter_shapley_mean)
    print(class_target_shapley_mean)
    print(class_shadow_shapley_mean)

    print(class_clutter_shapley_mean/(np.abs(class_clutter_shapley_mean)+np.abs(class_target_shapley_mean)+np.abs(class_shadow_shapley_mean))*100)
    print(class_target_shapley_mean/(np.abs(class_clutter_shapley_mean)+np.abs(class_target_shapley_mean)+np.abs(class_shadow_shapley_mean))*100)
    print(class_shadow_shapley_mean/(np.abs(class_clutter_shapley_mean)+np.abs(class_target_shapley_mean)+np.abs(class_shadow_shapley_mean))*100)
