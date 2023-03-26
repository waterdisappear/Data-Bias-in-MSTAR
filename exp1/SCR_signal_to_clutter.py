import sys
sys.path.append('..')
import torch
import numpy as np
import argparse
from exp1.DataLoad import load_data
# caluter  class_scr and class_clutter_mean

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../data/SOC/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)

    train_all, label_name = load_data(arg.data_path + 'TRAIN')
    train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
    class_scr = np.zeros((10, 1))
    class_clutter_mean = np.zeros((10, 1))

    shapely_0_list = np.zeros((1,1))
    shapely_1_list = np.zeros((1,1))
    label_list = np.zeros((1,1))
    for data, mask, label in train_loader:
        d0 = data * (mask[:, 0, :, :].unsqueeze(1))
        # d0 = torch.abs(torch.normal(0, 0.1, data.shape)) * (mask[:, 0, :, :].unsqueeze(1))
        d1 = data * (mask[:, 1, :, :].unsqueeze(1))

        d0_mean = (d0.sum(axis=-1).sum(axis=-1).sum(axis=-1))/(mask[:, 0, :, :].sum(axis=-1).sum(axis=-1))
        d1_mean = (d1.sum(axis=-1).sum(axis=-1).sum(axis=-1)) / (mask[:, 1, :, :].sum(axis=-1).sum(axis=-1))

        shapely_0_list = np.append(shapely_0_list, d0_mean.cpu().detach().numpy())
        shapely_1_list = np.append(shapely_1_list, d1_mean.cpu().detach().numpy())
        label_list = np.append(label_list, label.cpu().detach().numpy())

    shapely_0_list = shapely_0_list[1:]
    shapely_1_list = shapely_1_list[1:]
    label_list = label_list[1:]

    scr_list = 20*np.log10(shapely_1_list/shapely_0_list)

    for j in range(0,len(label_name)):
        class_scr[j] = (scr_list*(label_list==j)).sum()/(label_list==j).sum()
        class_clutter_mean[j] = (shapely_0_list*(label_list==j)).sum()/(label_list==j).sum()

    print(class_scr)
    print(class_clutter_mean)