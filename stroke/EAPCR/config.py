import torch

# ========= 路径相关 =========
data_path  = '../data/ID_07_02/data_ID.csv'
outpath    = '../output_EAPCR_01'

# 每折与汇总结果保存目录（按你的要求）
result_dir = '/data/naishuncheng/wenchuang/EAPCR/result'

# ========= 设备与随机种子 =========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed   = 2025

# ========= EAPCR 超参 =========
num_embed     = 49
embed_dim     = 256
dropout_prob  = 0.7

# ========= 训练超参 =========
epochs        = 50
batch_size    = 256
learning_rate = 0.0002
