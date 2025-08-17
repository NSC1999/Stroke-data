from util import plot_confusion_matrix, plot_accuracy_curve, plot_loss_curve  # 如无需要可在util里保留，这里不再调用
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import config as C
import logging


class EAPCR(nn.Module):

    def __init__(self, num_embed=0, embed_dim=0, dropout_prob=0.7, device='CPU', input_size=10):
        super().__init__()

        self.device = device

        # 注意：这里的 size/new_size 与输入特征矩阵的成形相关
        size, new_size, l, m, = 64, 64, 8, 8

        T = self.Generator_matrix(size, new_size, l, m)
        self.T = torch.tensor(T, dtype=torch.float32).to(C.device)
        np.savetxt('T_numpy.txt', T, fmt='%d')

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.tanh = nn.Tanh()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        dummy = torch.zeros(1, 1, input_size, input_size)
        conv_out = self.conv1(dummy)
        flat_dim = conv_out.numel() * 2

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 5)  # ← 二分类，若多分类请替换为类别数
        )

        flat_dim_2 = input_size * C.embed_dim
        self.res = nn.Sequential(
            nn.Linear(flat_dim_2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 5)  # ← 二分类，若多分类请替换为类别数
        )

        self.weight = nn.Parameter(torch.tensor(0.0))
        self.to(self.device)

    def forward(self, x):
        x = x.long()
        E = self.embedding(x)

        # Residual 分支
        R = E.reshape(E.size(0), -1)
        R = self.res(R)

        # Attention + Permutation 分支
        ET = E.transpose(1, 2)
        A = torch.matmul(E, ET)                 # (B, N, N)
        PA = torch.matmul(self.T, A)
        PA = torch.matmul(PA, self.T.transpose(0, 1))
        PA = self.tanh(PA).unsqueeze(1)         # (B, 1, N, N)
        A  = self.tanh(A).unsqueeze(1)          # (B, 1, N, N)

        C  = self.conv1(A)
        CC = self.conv2(PA)

        C_cat = torch.cat((C, CC), dim=1)
        C_cat = C_cat.reshape(C_cat.size(0), -1)
        C_cat = self.fc(C_cat)

        W = torch.sigmoid(self.weight)
        Output = W * C_cat + (1 - W) * R
        return Output

    def train_model(self, train_loader, test_loader, epochs=10, lr=1, k=0):
        """训练并在每个epoch末评估，但只返回最终一轮 test 上的四个指标。"""
        self.fold_output_dir = f"{C.outpath}/fold_{k+1}"
        os.makedirs(self.fold_output_dir, exist_ok=True)

        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(self.parameters(), lr=lr)

        loss_epoch_list = []
        train_accuracy_list = []
        test_accuracy_list = []

        for epoch in range(epochs):
            self.train()
            loss_epoch = 0.0
            for _, (train_x, train_y) in enumerate(train_loader):
                train_x = train_x.long().to(self.device)
                train_y = train_y.long().to(self.device)
                preds = self(train_x)
                loss_batch = loss_fn(preds, train_y)
                opt.zero_grad()
                loss_batch.backward()
                opt.step()
                loss_epoch += loss_batch.item()

            loss_epoch /= max(1, len(train_loader))
            loss_epoch_list.append(loss_epoch)

            # 可选：每epoch评估一下（这里只留精简输出）
            self.eval()
            with torch.no_grad():
                tr_acc, _, _, _ = self.basic_metrics(train_loader)
                te_acc, _, _, _ = self.basic_metrics(test_loader)
                train_accuracy_list.append(tr_acc)
                test_accuracy_list.append(te_acc)

            print(f'[Fold {k+1}] Epoch {epoch+1}/{epochs} | Loss {loss_epoch:.4f} | '
                  f'TrainAcc {tr_acc*100:.2f}% | TestAcc {te_acc*100:.2f}%')

        # 最终一轮：拿 test 集四指标
        acc, prec, rec, f1 = self.basic_metrics(test_loader)

        # 如需曲线，可在 util 里实现；本需求不保存图，仅保留能力
        # plot_accuracy_curve(train_accuracy_list, test_accuracy_list, self.fold_output_dir)
        # plot_loss_curve(loss_epoch_list, self.fold_output_dir)

        return acc, prec, rec, f1

    @staticmethod
    def give_num(num=1024, l=32, m=32):
        a = np.array([x for x in range(num)])
        a = a.reshape((l, m))
        a = np.transpose(a, (1, 0))
        a = a.reshape([-1, ])
        return a

    @staticmethod
    def Generator_matrix(size, new_size, l=85, m=85):
        num_list = EAPCR.give_num(size, l, m)
        num_list = num_list[num_list < new_size]
        M = np.zeros((new_size, new_size))
        for i in range(len(num_list)):
            num = num_list[i]
            M[i][num] = 1
        return M

    def basic_metrics(self, data_loader):
        """仅返回四个基础指标（macro 平均）：acc, precision, recall, f1"""
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_test, y_test in data_loader:
                x_test = x_test.long().to(self.device)
                y_test = y_test.long().to(self.device)
                outputs = self(x_test)
                probas = outputs.softmax(dim=1)
                top_prob, top_class = torch.max(probas, 1)

                all_preds.extend(top_class.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return acc, prec, rec, f1
