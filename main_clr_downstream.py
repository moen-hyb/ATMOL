#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
from sklearn.metrics import roc_auc_score,r2_score,mean_squared_error
from utils_clr_downstream import *
from torch_geometric.data import DataLoader
from encoder_gat import GATNet
from model_clr_downstream import Model
import torch.nn as nn
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

""""
药物下游任务
"""

def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE= mse ** 0.5
    return mae, mse, RMSE

def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2= round((SSR / SST) ** 2,3)
    return r2

# train for one epoch to learn unique features
def train(model, device, data_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    feature_x = torch.Tensor()
    feature_org = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        output, y, ew, xw = model(data)
        feature_x = torch.cat((feature_x, torch.Tensor(output.cpu().data.numpy())), 0)
        feature_org = torch.cat((feature_org, torch.Tensor(xc16.cpu().data.numpy())), 0)
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())), 0)
        pred = nn.Sigmoid()(output)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(data_loader.dataset),
                                                                           100. * batch_idx / len(data_loader),
                                                                           loss.item()))
    return feature_x, feature_org, edge_weight.numpy(), feature_weight.numpy()

def predicting(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output, y, e_weight, weight = model(data)
            pred = nn.Sigmoid()(output)
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            e_weight = e_weight.to('cpu')
            weight = weight.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            edge_weight = torch.cat((edge_weight, e_weight), 0)
            feature_weight = torch.cat((feature_weight, weight),0)


    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), edge_weight.numpy(), feature_weight.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ATMOL downstream')
    parser.add_argument('--path', default='down_task', help='down_task orginal data for input')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    # args parse
    args = parser.parse_args()

    print(args)
    batch_size, epochs = args.batch_size, args.epochs

    LOG_INTERVAL = 20

    # best_tasks = SIDER scaffold ClinTox SIDER
    clr_tasks = {'BBBP': 1, 'HIV': 1, 'BACE': 1, 'Tox21': 12, 'ClinTox': 2, 'SIDER': 27, 'MUV': 17}
    task = 'BBBP'
    # data prepare
    train_data = TestbedDataset(root=args.path, dataset='train', task=task)
    valid_data = TestbedDataset(root=args.path, dataset='valid', task=task)
    test_data = TestbedDataset(root=args.path, dataset='test', task=task)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=None)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=None)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=None)

    encoder_file = 'model_encoder_gat1_128_20_now_1_re_gat_de25_p1'

    print(encoder_file)
    """
    读取预训练模型的数据
    使用 GAT encoder 来预测药物下游任务
    """
    model_encoder = GATNet().cuda()
    model_encoder.load_state_dict(torch.load('results/model/'+ encoder_file + '.pkl', map_location='cuda:0'))
    model = Model(n_output=clr_tasks[task], encoder=model_encoder).cuda()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.pre.parameters(), lr=0.0001, weight_decay=1e-7)

    save_file ='{}_{}_{}'.format(batch_size, epochs,task)
    if not os.path.exists('results/down_task/clr/model_encoder_'+encoder_file+'_'+task):
        os.makedirs('results/down_task/clr/model_encoder_'+encoder_file+'_'+task)
    save_name = 'results/down_task/clr/model_encoder_' + encoder_file + '_'+task
    result_file_name = save_name+'/'+save_file+'_result.csv'
    valid_AUCs = save_name+'/'+save_file+'_validAUCs.txt'
    test_AUCs = save_name+'/'+save_file+'_testAUCs.txt'
    model_file_name =save_name+'/'+save_file+'_encoder.pkl'
    AUCs = ('Epoch\tAUC\tR2\tMSE\tmse\tmae\trmse\tr2')

    with open(valid_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(test_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    stopping_monitor = 0
    independent_num = []
    for epoch in range(epochs+1):
        feature_x, feature_org,ew,xw = train(model, device, train_data_loader, optimizer, epoch + 1)

        if (epoch + 0) % 10 == 0:
            S, T, ew, xw = predicting(model,device,valid_data_loader)
            # T is correct score
            # S is predict score

            # compute preformence
            mae, mse, rmse = compute_mae_mse_rmse(T, S)
            r2 = compute_rsquared(S, T)

            R2 = r2_score(T, S)
            MSE = mean_squared_error(T, S)
            auc = roc_auc_score(T, S)
            AUCs = [epoch, auc, R2, MSE, mse, mae, rmse, r2]
            print('AUC: ', AUCs)

            if best_auc < auc:
                best_auc = auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                save_AUCs(AUCs, valid_AUCs)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
                independent_num.append(T)
                independent_num.append(S)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break

    model.load_state_dict(torch.load(model_file_name))
    S, T,__,__ = predicting(model, device,test_data_loader)
    mae, mse, rmse = compute_mae_mse_rmse(S, T)
    r2 = compute_rsquared(S, T)

    R2 = r2_score(T, S)
    MSE = mean_squared_error(T, S)
    auc = roc_auc_score(T, S)
    AUCs = [0, auc, R2, MSE, mse, mae, rmse, r2]
    print('test_AUC: ', AUCs)
    # save data
    save_AUCs(AUCs, test_AUCs)

