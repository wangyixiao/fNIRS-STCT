import numpy as np
import os

all_acc = []
all_pre = []
all_rec = []
all_f1 = []
all_kap = []


for tr in range(1, 31):
    path = os.path.join('', str(tr))
    val_acc = open(path + '/test_acc.txt', "r")
    string = val_acc.readlines()[-1]  # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表 [-1]表示最后一行

    acc = string.split('acc=')[1].split(', pre')[0]
    pre = string.split('pre=')[1].split(', rec')[0]
    rec = string.split('rec=')[1].split(', f1')[0]
    f1 = string.split('f1=')[1].split(', kap')[0]
    kappa = string.split('kap=')[1]

    acc = float(acc)
    pre = float(pre)
    rec = float(rec)
    f1 = float(f1) * 100
    kappa = float(kappa)

    all_acc.append(acc)
    all_pre.append(pre)
    all_rec.append(rec)
    all_f1.append(f1)
    all_kap.append(kappa)


sub_acc = np.array(all_acc)
sub_pre = np.array(all_pre)
sub_rec = np.array(all_rec)
sub_f1 = np.array(all_f1)
sub_kap = np.array(all_kap)
print('acc = %.2f ± %.2f' % (np.mean(all_acc), np.std(all_acc)))
print('pre = %.2f ± %.2f' % (np.mean(all_pre), np.std(all_pre)))
print('rec = %.2f ± %.2f' % (np.mean(all_rec), np.std(all_rec)))
print('f1 = %.2f ± %.2f' % (np.mean(all_f1), np.std(all_f1)))
print('kap = %.2f ± %.2f' % (np.mean(all_kap), np.std(all_kap)))