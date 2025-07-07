import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


# 设置随机种子
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# 强制 PyTorch 使用确定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rc('font',family='Arial')
plt.style.use("ggplot")
# 自己写的函数文件functionfile.py
# 如果需要调整TSlib-test.ipynb文件的路径位置 注意同时调整导入的路径
from models import iTransformer
from utils.timefeatures import time_features
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        net (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        length_size (int): 输出序列的长度。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        print_train (bool, optional): 是否在训练中打印进度，默认为False。
    返回:
        net (torch.nn.Module): 训练好的模型。
        train_loss (list): 训练过程中每个epoch的平均训练损失列表。
        best_epoch (int): 达到最佳验证损失的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均训练损失
    print_frequency = num_epochs / 20  # 计算打印训练状态的频率

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  # 前向传播
            labels = labels[:, -length_size:].squeeze()
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值

        avg_train_loss = total_train_loss / len(train_loader)  # 计算该epoch的平均损失
        train_loss.append(avg_train_loss)  # 将平均损失添加到列表中

        # 如果设置为打印训练状态，则按频率打印
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1


def cal_eval(y_real, y_pred):
    """
    输入参数:
    y_real - numpy数组，表示测试集的真实目标值。
    y_pred - numpy数组，表示预测的结果。

    输出:
    df_eval - pandas DataFrame对象
    """

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])

    return df_eval


# df = pd.read_csv('data\\rlDatatc.csv')
df = pd.read_csv('data\\concatenated_result917_8_9 - 2.csv')
# 注意多变量情况下，目标变量必须为最后一列
data_dim = df[df.columns.drop('date')].shape[1]  # 一共多少个变量
data_target = df['Target']  # 预测的目标变量
data = df[df.columns.drop('date')]  # 选取所有的数据
# print(data_target)


# 时间戳
df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='H')  # 这一步很关键，注意数据的freq

"""
The following frequencies are supported:
    Y   - yearly
        alias: A
    M   - monthly
    W   - weekly
    D   - daily
    B   - business days
    H   - hourly
    T   - minutely
        alias: min
    S   - secondly
"""
# # 无验证集

# # 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.8
window =30# 模型输入序列长度
length_size = 1 # 预测结果的序列长度
batch_size = 32

data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data_inverse[int(train_set * data_length-window):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[int(train_set * data_length-window):, :]
#
# data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
# data_train_mark = data_stamp[:int(train_set * data_length), :]
# data_test = data_inverse[int(train_set * data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
# data_test_mark = data_stamp[int(train_set * data_length):, :]
n_feature = data_dim

print(data_dim)

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)
print(f"Number of samples in the test set: {len(x_test)}")


"""
# 有验证集

# 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))
data_length = len(data_inverse)
train_ratio = 0.6
val_ratio = 0.8
# 6：2：2
window = 30  # 模型输入序列长度 过去30天的数据
length_size = 1  # 预测结果的序列长度  预测未来1天
train_size = int(data_length * train_ratio)
val_size = int(data_length * val_ratio)
data_train = data_inverse[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_inverse[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_inverse[val_size:, :]
data_test_mark = data_stamp[val_size:, :]
batch_size = 32
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(window, length_size, batch_size, data_val,
                                                                     data_val_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)
                                                                          """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 400# 训练迭代次数 300论以后趋于稳定 0.001+500最佳 这个别改 loss维持再0.0010徘徊
learning_rate = 0.001  # 学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.2  # 训练迭代的早停比例 即patience=0.25*num_epochs


class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # input sequence length
        self.label_len = int(window / 2)  # start token length
        self.pred_len = length_size  # 预测序列长度
        self.freq = 'h'  # 时间的频率，
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 64  # 模型维度
        self.n_heads = 4  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 2  # 编码器块的数量
        self.d_layers = 1  # 解码器块的数量
        self.d_ff = 32  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立
        self.top_k = 5  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数
        self.moving_avg = window - 1  # Autoformer中的参数


config = Config()

model_type = 'iTransformer'
net = iTransformer.Model(config).to(device)
#################################
# print("\n" + "=" * 80)
# print("Model Structure Trace (输入输出维度变化)")
# print("=" * 80)
#
# # 创建虚拟输入
# dummy_input = torch.randn(config.batch_size, config.seq_len, config.enc_in).to(device)
# dummy_mark = torch.randn(config.batch_size, config.seq_len, 4).to(device)
#
# print(f"\n初始输入维度: {dummy_input.shape}")
# print(f"时间特征维度: {dummy_mark.shape}")
#
#
# # 定义钩子注册函数
# def register_hooks(net):
#     handles = []
#
#     def hook_fn(module, inputs, output):
#         # 过滤容器模块
#         if isinstance(module, (nn.ModuleList, nn.Sequential)):
#             return
#         input_shapes = [tuple(inp.shape) if isinstance(inp, torch.Tensor) else str(type(inp)) for inp in inputs]
#         output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else str(type(output))
#         print(f"\n▌ Layer: {module.__class__.__name__}")
#         print(f"├─ Input shape : {input_shapes}")
#         print(f"└─ Output shape: {output_shape}")
#         print("-" * 60)
#
#     # 递归遍历所有子模块
#     for name, module in net.named_modules():
#         # 为所有非容器模块注册钩子
#         if not isinstance(module, (nn.ModuleList, nn.Sequential)):
#             handle = module.register_forward_hook(hook_fn)
#             handles.append(handle)
#     return handles
#
#
# # 注册钩子
# handles = register_hooks(net)
#
# # 运行前向传播以触发钩子
# with torch.no_grad():
#     _ = net(dummy_input, dummy_mark, None, None, None)  # 注意参数匹配
#
# # 移除所有钩子
# for handle in handles:
#     handle.remove()

########################################################################
# print("\n" + "=" * 80)
# print("Key Model Architecture (主要模块维度变化)")
# print("=" * 80)
#
# # 定义需要跟踪的模块类型
# track_modules = [
#     'TCN',
#     'DataEmbedding_inverted',
#     'MultiHeadDifferentialAttention',
#     'Encoder',
#     'Linear'
# ]
#
#
# def key_module_hook(module, inputs, output):
#     """只打印关键模块的维度变化"""
#     module_name = module.__class__.__name__
#
#     # 过滤不需要显示的模块
#     if module_name not in track_modules:
#         return
#
#     # 格式化输入输出形状
#     input_shapes = []
#     for inp in inputs:
#         if isinstance(inp, torch.Tensor):
#             input_shapes.append(tuple(inp.shape))
#         elif isinstance(inp, (list, tuple)):
#             input_shapes.append([tuple(i.shape) if isinstance(i, torch.Tensor) else type(i) for i in inp])
#         else:
#             input_shapes.append(str(type(inp)))
#
#     output_shape = []
#     if isinstance(output, torch.Tensor):
#         output_shape = tuple(output.shape)
#     elif isinstance(output, (list, tuple)):
#         output_shape = [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o) for o in output]
#     else:
#         output_shape = str(type(output))
#
#     # 打印关键信息
#     print(f"\n▌ Module: {module_name}")
#     print(f"├─ Input shape  : {input_shapes}")
#     print(f"└─ Output shape : {output_shape}")
#     print("-" * 60)
#
#
# # 注册钩子到指定模块
# handles = []
# for name, module in net.named_modules():
#     if module.__class__.__name__ in track_modules:
#         handle = module.register_forward_hook(key_module_hook)
#         handles.append(handle)
#
# # 运行前向传播以触发钩子
# with torch.no_grad():
#     _ = net(dummy_input, dummy_mark, None, None, None)
#
# # 移除所有钩子
# for handle in handles:
#     handle.remove()
########################################################


criterion = nn.MSELoss().to(device)  # 损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 优化器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)  # 学习率调整策略

trained_model, train_loss, final_epoch = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=True)
"""
trained_model, train_loss, val_loss, final_epoch = model_train_val(
    net=net,
    train_loader=train_loader,
    val_loader=val_loader,
    length_size=length_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
    early_patience=early_patience,
    print_train=False
)
"""


