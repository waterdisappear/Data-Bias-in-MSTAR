import torch
import numpy as np
import argparse
import collections
from tqdm import tqdm

from exp1.DataLoad import load_data, load_test
from exp5.TrainTest import model_train, model_shapely, model_test, class_model_shapely
from model.Model import A_ConvNet

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../data/SOC/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=5,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)

    x = np.array(np.arange(1, arg.epochs+1), dtype=int)
    clutter_shapley = np.zeros((arg.fold, arg.epochs))
    target_shapley = np.zeros((arg.fold, arg.epochs))
    shadow_shapley = np.zeros((arg.fold, arg.epochs))

    clutter_target = np.zeros((arg.fold, arg.epochs))
    target_shadow = np.zeros((arg.fold, arg.epochs))
    shadow_clutter = np.zeros((arg.fold, arg.epochs))

    acc_train = np.zeros((arg.fold, arg.epochs))
    acc_test = np.zeros((arg.fold, arg.epochs))

    class_clutter_shapley = np.zeros((arg.fold, 10))
    class_target_shapley = np.zeros((arg.fold, 10))
    class_shadow_shapley = np.zeros((arg.fold, 10))

    train_all, label_name = load_data(arg.data_path + 'TRAIN')
    test_set = load_test(arg.data_path + 'TEST')

    for k in tqdm(range(arg.fold)):
        history = collections.defaultdict(list)
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

        model = A_ConvNet(num_classes=len(label_name))
        opt = torch.optim.Adam(model.parameters(), lr=arg.lr)
        # sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        total = sum([param.nelement() for param in model.parameters()])

        print("Number of parameter: %.2fM" % (total / 1e6))
        # best_test_accuracy = 0
        for epoch in tqdm(range(1, arg.epochs + 1)):
            print("##### " + " EPOCH " + str(epoch) + "#####")
            acc_train[k, epoch-1] = model_train(model=model, data_loader=train_loader, opt=opt)
            # sch.step()
            acc_test[k, epoch - 1] = model_test(model=model, test_loader=test_loader)
            history = model_shapely(model=model, data_loader=train_loader, his=history)


        clutter_shapley[k, :] = np.expand_dims(np.array(history['clutter']), 0)
        target_shapley[k, :] = np.expand_dims(np.array(history['target']), 0)
        shadow_shapley[k, :] = np.expand_dims(np.array(history['shadow']), 0)

        clutter_target[k, :] = np.expand_dims(np.array(history['clutter_target']), 0)
        target_shadow[k, :] = np.expand_dims(np.array(history['target_shadow']), 0)
        shadow_clutter[k, :] = np.expand_dims(np.array(history['shadow_clutter']), 0)

        clu, tar, sha = class_model_shapely(model=model, data_loader=train_loader, label_length=len(label_name))
        class_clutter_shapley[k, :] = clu
        class_target_shapley[k, :] = tar
        class_shadow_shapley[k, :] = sha

        print(clutter_shapley.mean(axis=0)/(clutter_shapley.mean(axis=0)+target_shapley.mean(axis=0)+shadow_shapley.mean(axis=0)))
        print(clu/(clu+tar+sha)*100)

    clutter_shapley_ratio = clutter_shapley.mean(axis=0)/(clutter_shapley.mean(axis=0)+target_shapley.mean(axis=0)+shadow_shapley.mean(axis=0))
    target_shapley_ratio = target_shapley.mean(axis=0)/(clutter_shapley.mean(axis=0)+target_shapley.mean(axis=0)+shadow_shapley.mean(axis=0))
    shadow_shapley_ratio = shadow_shapley.mean(axis=0)/(clutter_shapley.mean(axis=0)+target_shapley.mean(axis=0)+shadow_shapley.mean(axis=0))
    #
    np.savez('./result/shapely_model_1', clutter_shapley=clutter_shapley, target_shapley=target_shapley, shadow_shapley=shadow_shapley,
             clutter_shapley_ratio=clutter_shapley_ratio, target_shapley_ratio=target_shapley_ratio, shadow_shapley_ratio=shadow_shapley_ratio,
             clutter_target=clutter_target, target_shadow=target_shadow, shadow_clutter=shadow_clutter,
             acc_train=acc_train, acc_test=acc_test, x=x)

    data = np.load('shapely_model_1.npz')

    print(data['clutter_shapley'].mean(axis=0)[-1])
    print(data['target_shapley'].mean(axis=0)[-1])
    print(data['shadow_shapley'].mean(axis=0)[-1])

    print(data['clutter_shapley_ratio'][-1]*100)
    print(data['target_shapley_ratio'][-1]*100)
    print(data['shadow_shapley_ratio'][-1]*100)

    print(data['clutter_target'].mean(axis=0)[-1])
    print(data['target_shadow'].mean(axis=0)[-1])
    print(data['shadow_clutter'].mean(axis=0)[-1])

    np.savez('./result/class_shapely_model_1.npy', class_clutter_shapley=class_clutter_shapley, class_target_shapley=class_target_shapley, class_shadow_shapley=class_shadow_shapley)

    class_clutter_shapley_mean = class_clutter_shapley.mean(axis=0)
    class_target_shapley_mean = class_target_shapley.mean(axis=0)
    class_shadow_shapley_mean = class_shadow_shapley.mean(axis=0)

    print(class_clutter_shapley_mean)
    print(class_target_shapley_mean)
    print(class_shadow_shapley_mean)

    print(class_clutter_shapley_mean/(class_clutter_shapley_mean+class_target_shapley_mean+class_shadow_shapley_mean)*100)
    print(class_target_shapley_mean/(class_clutter_shapley_mean+class_target_shapley_mean+class_shadow_shapley_mean)*100)
    print(class_shadow_shapley_mean/(class_clutter_shapley_mean+class_target_shapley_mean+class_shadow_shapley_mean)*100)

