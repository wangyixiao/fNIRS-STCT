import torch
from sklearn.model_selection import RepeatedKFold
import numpy as np
from model import fNIRS_STCT
from loader_si import Dataset, Load_Dataset, Split_Dataset
import os
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score
from ols import OnlineLabelSmoothing

if __name__ == "__main__":
    # Training epochs
    EPOCH = 120

    # Select the specified path
    data_path = ''

    # Save file and avoid training file overwriting.
    save_path = ''
    assert os.path.exists(save_path) is False, 'path is exist'
    os.makedirs(save_path)

    # Load dataset and set flooding levels. Different models may have different flooding levels.
    flooding_level = [0.45, 0.40, 0.35]
    feature, label = Load_Dataset(data_path)

    _, _, sampling_points, channels = feature.shape  # (2250, 2, 256, 20)

    Subjects = 30
    for sub in range(1, Subjects+1):
        X_train, y_train, X_test, y_test = Split_Dataset(sub, feature, label, sampling_points)

        path = save_path + '/' + str(sub)
        assert os.path.exists(path) is False, 'sub-path is exist'
        os.makedirs(path)

        train_set = Dataset(X_train, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

        # -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        net = fNIRS_STCT(n_class=3, dim=160, depth=1, heads=4, mlp_dim=64).to(device)

        criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=3, smoothing=0.1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.005)
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        # save results
        metrics = open(path + '/metrics.txt', 'w')  # 只写模式。【不可读；不存在则创建；存在则删除内容；】
        # -------------------------------------------------------------------------------------------------------------------- #
        test_max_acc = 0
        for epoch in range(EPOCH):
            net.train()
            criterion.train()
            train_running_acc = 0
            total = 0
            loss_steps = []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                if epoch < 30:
                    b = flooding_level[0]
                elif epoch < 50:
                    b = flooding_level[1]
                else:
                    b = flooding_level[2]

                # flooding
                loss = (loss - b).abs() + b

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total += labels.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            train_running_loss = float(np.mean(loss_steps))
            train_running_acc = 100 * train_running_acc / total
            print('[%d, %d] Train loss: %0.4f' % (sub, epoch, train_running_loss))
            print('[%d, %d] Train acc: %0.3f%%' % (sub, epoch, train_running_acc))

            # -------------------------------------------------------------------------------------------------------------------- #
            net.eval()
            criterion.eval()
            test_running_acc = 0
            total = 0
            loss_steps = []
            y_label = []
            y_pred = []
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    y_label += labels

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    y_pred += pred
                    test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                test_running_acc = 100 * test_running_acc / total
                test_running_loss = float(np.mean(loss_steps))
                print('     [%d, %d] Test loss: %0.4f' % (sub, epoch, test_running_loss))
                print('     [%d, %d] Test acc: %0.3f%%' % (sub, epoch, test_running_acc))

                acc = accuracy_score(y_label, y_pred)

                # macro mode for UFFT
                precision = precision_score(y_label, y_pred, average='macro')
                recall = recall_score(y_label, y_pred, average='macro')
                f1 = f1_score(y_label, y_pred, average='macro')

                kappa_value = cohen_kappa_score(y_label, y_pred)  # 科恩卡帕系数（Cohen’s Kappa Coefficient）
                confusion = confusion_matrix(y_label, y_pred)  # 混淆矩阵confusion_matrix
                metrics.write(
                    "test loss=%.4f,train loss=%.4f,train acc=%.4f,acc=%.4f, pre=%.4f, rec=%.4f, f1=%.4f, kap=%.4f" % (
                        test_running_loss, train_running_loss, test_running_acc, acc * 100, precision * 100,
                        recall * 100, f1, kappa_value))
                metrics.write('\n')
                metrics.flush()  # 关闭输出流

                if test_running_acc > test_max_acc:
                    test_max_acc = test_running_acc
                    torch.save(net.state_dict(), path + '/model.pt')
                    test_save = open(path + '/test_acc.txt', "w")
                    test_save.write("acc=%.4f, pre=%.4f, rec=%.4f, f1=%.4f, kap=%.4f" % (
                        acc * 100, precision * 100, recall * 100, f1, kappa_value))
                    test_save.close()
            criterion.next_epoch()
            lrStep.step()
