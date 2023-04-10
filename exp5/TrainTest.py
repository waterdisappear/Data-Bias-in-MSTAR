import torch
import numpy as np
import sys
sys.path.append('..')
from torch import nn
from torchvision import transforms as T
from exp5.DataLoad import adjust

def model_train(model, data_loader, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr = nn.CrossEntropyLoss()
    train_loss = 0
    for i, data in enumerate(data_loader):
        x, mask, y = data
        for j in range(x.shape[0]):
            x[j,:,:] = torch.from_numpy(adjust(x[j,:,:].squeeze().detach().cpu().numpy(), mask[j,:,:,:].squeeze().detach().cpu().permute([1,2,0]).numpy())).to(device)
        out = model(x.to(device))
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()
        loss = cr(out, y.to(device))
        train_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # print("Train loss is:{:.8f}".format(train_loss / len(data_loader)))
    # print("Train accuracy is:{:.2f} % ".format(train_acc / len(data_loader.dataset) * 100.))
    return train_acc / len(data_loader.dataset) * 100.


def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # test_loss = 0
    # pred_all = np.array([[]]).reshape((0, 1))
    # real_all = np.array([[]]).reshape((0, 1))
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def model_shapely(model, data_loader, his):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    shapely_0_list = np.zeros((1,1))
    shapely_1_list = np.zeros((1,1))
    shapely_2_list = np.zeros((1,1))

    Bshapely_0_list = np.zeros((1,1))
    Bshapely_1_list = np.zeros((1,1))
    Bshapely_2_list = np.zeros((1,1))

    with torch.no_grad():
        for data, mask, label in data_loader:
            # dzero = torch.zeros(data.shape)
            dzero = torch.abs(torch.normal(0.0, 0.1, data.shape))

            d0 = data * (mask[:, 0, :, :].unsqueeze(1)) + dzero * (1-mask[:, 0, :, :].unsqueeze(1))
            d1 = data * (mask[:, 1, :, :].unsqueeze(1)) + dzero * (1-mask[:, 1, :, :].unsqueeze(1))
            d2 = data * (mask[:, 2, :, :].unsqueeze(1)) + dzero * (1-mask[:, 2, :, :].unsqueeze(1))

            d01 = data * ((mask[:, 0, :, :] + mask[:, 1, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 0, :, :] + mask[:, 1, :, :]).unsqueeze(1))
            d12 = data * ((mask[:, 1, :, :] + mask[:, 2, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 1, :, :] + mask[:, 2, :, :]).unsqueeze(1))
            d02 = data * ((mask[:, 0, :, :] + mask[:, 2, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 0, :, :] + mask[:, 2, :, :]).unsqueeze(1))

            def out_v(data, model, label):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.eval()
                with torch.no_grad():
                    output = model(data.to(device))
                return output.gather(1, label.to(device).unsqueeze(1)).squeeze()

            vzero = out_v(dzero.to(device), model, label)
            v123 = out_v(data.to(device), model, label)

            v1 = out_v(d0.to(device), model, label)
            v2 = out_v(d1.to(device), model, label)
            v3 = out_v(d2.to(device), model, label)

            v12 = out_v(d01.to(device), model, label)
            v23 = out_v(d12.to(device), model, label)
            v13 = out_v(d02.to(device), model, label)

            shapely_0 = (v1 - vzero + v1 - vzero + v12 - v2 + v123 - v23 + v13 - v3 + v123 - v23) / 6
            shapely_1 = (v12 - v1 + v123 - v13 + v2 - vzero + v2 - vzero + v123 - v13 + v23 - v3) / 6
            shapely_2 = (v123 - v12 + v13 - v1 + v123 - v12 + v23 - v2 + v3 - vzero + v3 - vzero) / 6

            shapely_0_list = np.append(shapely_0_list, shapely_0.cpu().detach().numpy())
            shapely_1_list = np.append(shapely_1_list, shapely_1.cpu().detach().numpy())
            shapely_2_list = np.append(shapely_2_list, shapely_2.cpu().detach().numpy())

            Bshapely_01 = (v12 - v1 - v2 + vzero + v123 - v13 - v23 + v3) / 2
            Bshapely_12 = (v23 - v2 - v3 + vzero + v123 - v12 - v13 + v1) / 2
            Bshapely_20 = (v13 - v1 - v3 + vzero + v123 - v12 - v23 + v2) / 2

            Bshapely_0_list = np.append(Bshapely_0_list, Bshapely_01.cpu().detach().numpy())
            Bshapely_1_list = np.append(Bshapely_1_list, Bshapely_12.cpu().detach().numpy())
            Bshapely_2_list = np.append(Bshapely_2_list, Bshapely_20.cpu().detach().numpy())

    his['clutter_target'].append(Bshapely_0_list[1:].mean())
    his['target_shadow'].append(Bshapely_1_list[1:].mean())
    his['shadow_clutter'].append(Bshapely_2_list[1:].mean())

    his['clutter'].append(
        shapely_0_list[1:].mean())
    his['target'].append(
        shapely_1_list[1:].mean())
    his['shadow'].append(
        shapely_2_list[1:].mean())

    # his['clutter_ratio'].append(shapely_0_list[1:].mean()/(shapely_0_list[1:].mean()+shapely_1_list[1:].mean()+shapely_2_list[1:].mean()))
    # his['target_ratio'].append(shapely_1_list[1:].mean()/(shapely_0_list[1:].mean()+shapely_1_list[1:].mean()+shapely_2_list[1:].mean()))
    # his['shadow_ratio'].append(shapely_2_list[1:].mean()/(shapely_0_list[1:].mean()+shapely_1_list[1:].mean()+shapely_2_list[1:].mean()))

    return his
def class_model_shapely(model, data_loader, label_length=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    clutter = np.zeros((1, label_length))
    target = np.zeros((1, label_length))
    shadow = np.zeros((1, label_length))

    shapely_0_list = np.zeros((1,1))
    shapely_1_list = np.zeros((1,1))
    shapely_2_list = np.zeros((1,1))
    label_list = np.zeros((1,1))
    with torch.no_grad():
        for data, mask, label in data_loader:
            # dzero = torch.zeros(data.shape)
            dzero = torch.abs(torch.normal(0.0, 0.1, data.shape))

            d0 = data * (mask[:, 0, :, :].unsqueeze(1)) + dzero * (1-mask[:, 0, :, :].unsqueeze(1))
            d1 = data * (mask[:, 1, :, :].unsqueeze(1)) + dzero * (1-mask[:, 1, :, :].unsqueeze(1))
            d2 = data * (mask[:, 2, :, :].unsqueeze(1)) + dzero * (1-mask[:, 2, :, :].unsqueeze(1))

            d01 = data * ((mask[:, 0, :, :] + mask[:, 1, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 0, :, :] + mask[:, 1, :, :]).unsqueeze(1))
            d12 = data * ((mask[:, 1, :, :] + mask[:, 2, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 1, :, :] + mask[:, 2, :, :]).unsqueeze(1))
            d02 = data * ((mask[:, 0, :, :] + mask[:, 2, :, :]).unsqueeze(1)) + dzero * (1-(mask[:, 0, :, :] + mask[:, 2, :, :]).unsqueeze(1))

            def out_v(data, model, label):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.eval()
                with torch.no_grad():
                    output = model(data.to(device))
                return output.gather(1, label.to(device).unsqueeze(1)).squeeze()

            vzero = out_v(dzero.to(device), model, label)
            v123 = out_v(data.to(device), model, label)

            v1 = out_v(d0.to(device), model, label)
            v2 = out_v(d1.to(device), model, label)
            v3 = out_v(d2.to(device), model, label)

            v12 = out_v(d01.to(device), model, label)
            v23 = out_v(d12.to(device), model, label)
            v13 = out_v(d02.to(device), model, label)

            shapely_0 = (v1 - vzero + v1 - vzero + v12 - v2 + v123 - v23 + v13 - v3 + v123 - v23) / 6
            shapely_1 = (v12 - v1 + v123 - v13 + v2 - vzero + v2 - vzero + v123 - v13 + v23 - v3) / 6
            shapely_2 = (v123 - v12 + v13 - v1 + v123 - v12 + v23 - v2 + v3 - vzero + v3 - vzero) / 6

            shapely_0_list = np.append(shapely_0_list, shapely_0.cpu().detach().numpy())
            shapely_1_list = np.append(shapely_1_list, shapely_1.cpu().detach().numpy())
            shapely_2_list = np.append(shapely_2_list, shapely_2.cpu().detach().numpy())
            label_list = np.append(label_list, label.cpu().detach().numpy())

    shapely_0_list = shapely_0_list[1:]
    shapely_1_list = shapely_1_list[1:]
    shapely_2_list = shapely_2_list[1:]
    label_list = label_list[1:]


    for j in range(0, label_length):
        clutter[0, j] = (shapely_0_list*(label_list==j)).sum()/(label_list==j).sum()
        target[0, j] = (shapely_1_list*(label_list==j)).sum()/(label_list==j).sum()
        shadow[0, j] = (shapely_2_list*(label_list==j)).sum()/(label_list==j).sum()

    return clutter.squeeze(), target.squeeze(), shadow.squeeze()

