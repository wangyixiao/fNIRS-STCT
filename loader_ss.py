import torch
import pandas as pd
import numpy as np
from scipy import signal


def Load_Dataset(data_path, start=20, end=278):
    feature = []
    label = []
    for num in range(1, 31): #30个受试者
        name = data_path + '/' + str(num) + '/' + str(num) + '.xls'#fNIRS数据
        Hb_org = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(num) + '/' + str(num) + '_desc.xls'#分类
        desc = pd.read_excel(name, header=None)

        Hb = []
        for i in range(1, 76):#75个工作表，一个人做了75次实验
            name = 'Sheet' + str(i)
            Hb.append(Hb_org[name].values)

        # (75, 347, 40)
        Hb = np.array(Hb)
        desc = np.array(desc)

        HbO_R = []
        HbO_L = []
        HbO_F = []
        HbR_R = []
        HbR_L = []
        HbR_F = []
        for i in range(75):#分类别，分HbOHbR
            if desc[i, 0] == 1:
                HbO_R.append(Hb[i, start:end, :20])
                HbR_R.append(Hb[i, start:end, 20:])
            elif desc[i, 0] == 2:
                HbO_L.append(Hb[i, start:end, :20])
                HbR_L.append(Hb[i, start:end, 20:])
            elif desc[i, 0] == 3:
                HbO_F.append(Hb[i, start:end, :20])
                HbR_F.append(Hb[i, start:end, 20:])

        # (25, 256, 20) --> (25, 1, 256, 20)
        HbO_R = np.array(HbO_R).reshape((25, 1, end - start, 20))
        HbO_L = np.array(HbO_L).reshape((25, 1, end - start, 20))
        HbO_F = np.array(HbO_F).reshape((25, 1, end - start, 20))

        HbR_R = np.array(HbR_R).reshape((25, 1, end - start, 20))
        HbR_L = np.array(HbR_L).reshape((25, 1, end - start, 20))
        HbR_F = np.array(HbR_F).reshape((25, 1, end - start, 20))

        HbO_R = np.concatenate((HbO_R, HbR_R), axis=1)#concatenate数组拼接  axis=1在第二维操作 (25, 2, 20, 256)
        HbO_L = np.concatenate((HbO_L, HbR_L), axis=1)
        HbO_F = np.concatenate((HbO_F, HbR_F), axis=1)

        for i in range(25):
            feature.append(HbO_R[i, :, :, :])
            feature.append(HbO_L[i, :, :, :])
            feature.append(HbO_F[i, :, :, :])

            label.append(0)
            label.append(1)
            label.append(2)

        print(str(num) + '  OK')

    feature = np.array(feature) #(2250, 2, 256, 20) 2250=75*30
    label = np.array(label) #(2250,)
    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]



