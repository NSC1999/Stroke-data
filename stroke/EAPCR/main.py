from util import CustomDataset, Get_Datas_CSV_all
from model import EAPCR
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import sys
import logging
import config as C
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    os.makedirs(C.outpath, exist_ok=True)
    os.makedirs(C.result_dir, exist_ok=True)
    logpath = os.path.join(C.outpath, 'log')
    os.makedirs(logpath, exist_ok=True)
    log_file = os.path.join(logpath, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_file


if __name__ == '__main__':

    # 仅需要五折四个指标：acc、precision（macro）、recall（macro）、f1（macro）
    # 运行命令：python main.py train_fold
    seed = C.seed
    set_random_seed(seed)
    log_file = ensure_dirs()

    data_path     = C.data_path
    device        = C.device
    epochs        = C.epochs
    batch_size    = C.batch_size
    learning_rate = C.learning_rate

    num_embed     = C.num_embed
    embed_dim     = C.embed_dim
    dropout_prob  = C.dropout_prob

    logging.info(f'seed: {seed}')
    logging.info(f'epochs: {epochs}')
    logging.info(f'batch size: {batch_size}')
    logging.info(f'learning rate: {learning_rate}')
    logging.info(f'dropout probability: {dropout_prob}')

    if len(sys.argv) > 1 and sys.argv[1] == 'train_fold':

        # 读取数据
        X_all, y_all = Get_Datas_CSV_all(data_path)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

        # 记录每折四指标与汇总
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_all, y_all)):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            train_dataset = CustomDataset(torch.tensor(X_train), torch.tensor(y_train))
            test_dataset  = CustomDataset(torch.tensor(X_test),  torch.tensor(y_test))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

            input_size = torch.tensor(X_train).shape[-1]
            model = EAPCR(num_embed, embed_dim, dropout_prob, device, input_size)

            if fold == 0:
                logging.info("Model structure：\n" + str(model))
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total number of parameters: {total_params}")
                logging.info(f'Total number of parameters: {total_params}')

            acc, prec, rec, f1 = model.train_model(
                train_loader, test_loader,
                epochs=epochs, lr=learning_rate, k=fold
            )

            print(f'[Fold {fold+1}] Acc={acc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}')
            logging.info(f'[Fold {fold+1}] Acc={acc:.4f} Prec={prec:.4f} Recall={rec:.4f} F1={f1:.4f}')

            # 每一折结果存CSV
            fold_df = pd.DataFrame([{
                'fold': fold + 1,
                'accuracy': acc,
                'precision_macro': prec,
                'recall_macro': rec,
                'f1_macro': f1
            }])
            fold_csv = os.path.join(C.result_dir, f'result_fold_{fold+1}.csv')
            fold_df.to_csv(fold_csv, index=False)

            fold_rows.append([acc, prec, rec, f1])

        # 汇总均值±标准差
        arr = np.array(fold_rows)  # shape (5,4)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)

        summary_df = pd.DataFrame([{
            'accuracy_mean':         means[0],
            'accuracy_std':          stds[0],
            'precision_macro_mean':  means[1],
            'precision_macro_std':   stds[1],
            'recall_macro_mean':     means[2],
            'recall_macro_std':      stds[2],
            'f1_macro_mean':         means[3],
            'f1_macro_std':          stds[3],
        }])

        summary_csv = os.path.join(C.result_dir, 'summary_cv.csv')
        summary_df.to_csv(summary_csv, index=False)

        print('\n=== Cross-validation Summary ===')
        print(summary_df.to_string(index=False))
        logging.info('=== Cross-validation Summary ===')
        logging.info('\n' + summary_df.to_string(index=False))

    else:
        print("Usage: python main.py train_fold")
