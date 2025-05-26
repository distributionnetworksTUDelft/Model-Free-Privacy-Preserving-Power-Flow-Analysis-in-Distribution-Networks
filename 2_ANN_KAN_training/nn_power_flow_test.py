# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:54:51 2024

@author: dliu8
"""



# =============================================================================
# a deep NN for power flow calculation based on MLP
''' totally successfully on  Wen Mar 19 10:38:10 2024 '''
# =============================================================================

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np




''' parameters seting'''
train_ratio = 0.8
batch_size = 25
Num_house = 95


''' input dataset'''
# data = pd.read_excel(r'/training_data_pq_v_52.xlsx')
# data = pd.read_excel(r'/training_data_pq_v_52_random.xlsx')
# data = pd.read_excel(r'/training_data_pq_v_64.xlsx')
# data = pd.read_excel(r'training_data_pq_v_64_random.xlsx')
# data = pd.read_excel(r'/training_data_pq_v_95.xlsx')
# data = pd.read_excel(r'/training_data_pq_v_95_random.xlsx')
data = pd.read_excel(r'/training_data_pq_v_95_3phase_random.xlsx')


# 假设数据的第一列是标签，剩余的列是特征
features = data.iloc[96*0:96*30, 0:Num_house*2].values
# features = np.log1p(features) # not good
# features = (features - 1.2)/0.4# for 64  95
features = (features - 1.3)/0.3# for 3-phase 95
# features = (features - 1.3)/0.3 # for 52
# features = np.where(features != 0, (features - 1) / 0.6, features)
labels = data.iloc[96*0:96*30, Num_house*2:].values
# labels = (labels - 0.98)/0.01 # # for 52 used in my cases
# labels = (labels - 0.98)/0.001 # # for 64  95 used in my cases
labels = (labels - 0.99)/0.02 # # for 3-phase 95 used in my cases
# labels = 1/(1+np.exp(-labels))


# mean_f = features.mean(axis=0)
# std_f = features.std(axis=0) + 1e-6
# features = (features - mean_f) / std_f

# mean_l = labels.mean(axis=0)
# std_l = labels.std(axis=0) + 1e-6
# labels = (labels - mean_l) / std_l
# error = error * std_l.reshape(-1,1)


features_tensor = torch.tensor(features, dtype=torch.float32) 
labels_tensor = torch.tensor(labels, dtype=torch.float32) 

# 创建 TensorDataset 对象
dataset = TensorDataset(features_tensor, labels_tensor)
train_size = int(train_ratio * len(dataset)) 
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


''' model construction '''

# NN0 ------------------------------------------------------------------------
# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()
#         self.fc1 = nn.Linear(Num_house * 2, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, Num_house)

#     def forward(self, x):
#         x = x.view(-1, Num_house*2)  # 将输入展平为一维向量
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = F.tanh(self.fc3(x)) 
#         x = F.tanh(self.fc4(x)) 
#         x = self.fc5(x)  
#         return x
# torch.save(model.state_dict(), 'model_nn0_95_encry_30day_1500.pth')


# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()
#         self.fc1 = nn.Linear(Num_house * 2, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, 1024)
#         self.fc5 = nn.Linear(1024, Num_house)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = x.view(-1, Num_house*2)  # 将输入展平为一维向量
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x)) 
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = F.tanh(self.fc3(x)) 
#         x = self.fc5(x)  
#         return x

# # NN---------------------------------------------------------------------------    
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(Num_house * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, Num_house)

    def forward(self, x):
        x = x.view(-1, Num_house * 2)  # 将输入展平为一维向量
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x)) 
        x = F.tanh(self.fc4(x)) 
        x = self.fc5(x)  
        return x 
# torch.save(model.state_dict(), 'model_nn_95_3phase_encry_30day_150.pth')



# NN_improved------------------------------------------------------------------
# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()
#         # 定义四个全连接层
#         self.fc1 = nn.Linear(Num_house * 2, Num_house * 2)
#         self.fc2 = nn.Linear(Num_house * 2, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 512)
#         self.fc5 = nn.Linear(512, 512)
#         self.fc6 = nn.Linear(512, Num_house)

#     def forward(self, x):
#         x = x.view(-1, Num_house * 2)  # 将输入展平为一维向量
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = F.tanh(self.fc3(x)) 
#         x = F.tanh(self.fc4(x)) 
#         x = F.tanh(self.fc5(x)) 
#         x = self.fc6(x)  
#         return x



# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()
#         self.num_nodes = Num_house
#         self.input_dim = Num_house * 2  # 有功+无功
#         self.d_model = 128  # transformer内部维度

#         # 将输入映射为 d_model 维度
#         self.input_proj = nn.Linear(self.input_dim, self.d_model)

#         # 单层 TransformerEncoder（也可以叠加多层）
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=4,
#             dim_feedforward=256,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         # 输出映射回 num_nodes 电压幅值
#         self.output_proj = nn.Sequential(
#             nn.Linear(self.d_model, 128),
#             nn.ReLU(),
#             nn.Linear(128, Num_house)
#         )

#     def forward(self, x):
#         # x shape: [batch_size, num_nodes*2]
#         x = x.unsqueeze(1)  # -> [batch, seq_len=1, input_dim]
#         x = self.input_proj(x)  # -> [batch, 1, d_model]
#         x = self.encoder(x)     # -> [batch, 1, d_model]
#         x = x.squeeze(1)        # -> [batch, d_model]
#         out = self.output_proj(x)  # -> [batch, num_nodes]
#         return out



# # KAN--------------------------------------------------------------------------
# class KANLinear(torch.nn.Module):
#     def __init__(
#         self,
#         in_features,
#         out_features,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         # base_activation=torch.nn.SiLU,
#         base_activation=F.tanh,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(KANLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         h = (grid_range[1] - grid_range[0]) / grid_size
#         grid = (
#             (
#                 torch.arange(-spline_order, grid_size + spline_order + 1) * h
#                 + grid_range[0]
#             )
#             .expand(in_features, -1)
#             .contiguous()
#         )
#         self.register_buffer("grid", grid)

#         self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
#         self.spline_weight = torch.nn.Parameter(
#             torch.Tensor(out_features, in_features, grid_size + spline_order)
#         )

#         self.scale_noise = scale_noise
#         self.scale_base = scale_base
#         self.scale_spline = scale_spline
#         self.base_activation = base_activation
#         self.grid_eps = grid_eps

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.constant_(self.base_weight, self.scale_base)
#         with torch.no_grad():
#             noise = (
#                 (
#                     torch.rand(self.grid_size + 1, self.in_features, self.out_features)
#                     - 1 / 2
#                 )
#                 * self.scale_noise
#                 / self.grid_size
#             )
#             self.spline_weight.data.copy_(
#                 self.scale_spline
#                 * self.curve2coeff(
#                     self.grid.T[self.spline_order : -self.spline_order],
#                     noise,
#                 )
#             )

#     def b_splines(self, x: torch.Tensor):
#         """
#         Compute the B-spline bases for the given input tensor.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, in_features).

#         Returns:
#             torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features

#         grid: torch.Tensor = (
#             self.grid
#         )  # (in_features, grid_size + 2 * spline_order + 1)
#         x = x.unsqueeze(-1)
#         bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
#         for k in range(1, self.spline_order + 1):
#             bases = (
#                 (x - grid[:, : -(k + 1)])
#                 / (grid[:, k:-1] - grid[:, : -(k + 1)])
#                 * bases[:, :, :-1]
#             ) + (
#                 (grid[:, k + 1 :] - x)
#                 / (grid[:, k + 1 :] - grid[:, 1:(-k)])
#                 * bases[:, :, 1:]
#             )

#         assert bases.size() == (
#             x.size(0),
#             self.in_features,
#             self.grid_size + self.spline_order,
#         )
#         return bases.contiguous()

#     def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Compute the coefficients of the curve that interpolates the given points.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, in_features).
#             y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

#         Returns:
#             torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         assert y.size() == (x.size(0), self.in_features, self.out_features)

#         A = self.b_splines(x).transpose(
#             0, 1
#         )  # (in_features, batch_size, grid_size + spline_order)
#         B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
#         solution = torch.linalg.lstsq(
#             A, B
#         ).solution  # (in_features, grid_size + spline_order, out_features)
#         result = solution.permute(
#             2, 0, 1
#         )  # (out_features, in_features, grid_size + spline_order)

#         assert result.size() == (
#             self.out_features,
#             self.in_features,
#             self.grid_size + self.spline_order,
#         )
#         return result.contiguous()

#     def forward(self, x: torch.Tensor):
#         assert x.dim() == 2 and x.size(1) == self.in_features

#         base_output = F.linear(self.base_activation(x), self.base_weight)
#         spline_output = F.linear(
#             self.b_splines(x).view(x.size(0), -1),
#             self.spline_weight.view(self.out_features, -1),
#         )
#         return base_output + spline_output

#     @torch.no_grad()
#     def update_grid(self, x: torch.Tensor, margin=0.01):
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         batch = x.size(0)

#         splines = self.b_splines(x)  # (batch, in, coeff)
#         splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
#         orig_coeff = self.spline_weight  # (out, in, coeff)
#         orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
#         unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
#         unreduced_spline_output = unreduced_spline_output.permute(
#             1, 0, 2
#         )  # (batch, in, out)

#         # sort each channel individually to collect data distribution
#         x_sorted = torch.sort(x, dim=0)[0]
#         grid_adaptive = x_sorted[
#             torch.linspace(
#                 0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
#             )
#         ]

#         uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
#         grid_uniform = (
#             torch.arange(
#                 self.grid_size + 1, dtype=torch.float32, device=x.device
#             ).unsqueeze(1)
#             * uniform_step
#             + x_sorted[0]
#             - margin
#         )

#         grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
#         grid = torch.concatenate(
#             [
#                 grid[:1]
#                 - uniform_step
#                 * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
#                 grid,
#                 grid[-1:]
#                 + uniform_step
#                 * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
#             ],
#             dim=0,
#         )

#         self.grid.copy_(grid.T)
#         self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         l1_fake = self.spline_weight.abs().mean(-1)
#         regularization_loss_activation = l1_fake.sum()
#         p = l1_fake / regularization_loss_activation
#         regularization_loss_entropy = -torch.sum(p * p.log())
#         return (
#             regularize_activation * regularization_loss_activation
#             + regularize_entropy * regularization_loss_entropy
#         )


# class CustomNet(torch.nn.Module):  # KAN networks
#     def __init__(
#         self,
#         layers_hidden,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=F.tanh,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(CustomNet, self).__init__()
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         self.layers = torch.nn.ModuleList()
#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             self.layers.append(
#                 KANLinear(
#                     in_features,
#                     out_features,
#                     grid_size=grid_size,
#                     spline_order=spline_order,
#                     scale_noise=scale_noise,
#                     scale_base=scale_base,
#                     scale_spline=scale_spline,
#                     base_activation=base_activation,
#                     grid_eps=grid_eps,
#                     grid_range=grid_range,
#                 )
#             )

#     def forward(self, x: torch.Tensor, update_grid=False):
#         for layer in self.layers:
#             if update_grid:
#                 layer.update_grid(x)
#             x = layer(x)
#         return x

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         return sum(
#             layer.regularization_loss(regularize_activation, regularize_entropy)
#             for layer in self.layers
#         )
# # ------------------------------------------------------------------------------


# 定义损失函数
def loss_function(output, target):
    loss = F.mse_loss(output, target)
    return loss

# 定义训练函数
def train(model, train_loader, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets) 
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss/len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
    return losses, model


# model = CustomNet(layers_hidden=[Num_house * 2, 512, 512, 512, 512, Num_house])
model = CustomNet()
optimizer = optim.AdamW(model.parameters(), lr=0.000005)
# optimizer = optim.AdamW(model.parameters(), lr=0.001)
# optimizer = optim.ASGD(model.parameters(), lr=0.01)

import time
start_time = time.time()
loss_train, model = train(model, train_loader, optimizer, num_epochs=1500)

end_time = time.time()
execution_time = end_time - start_time
print("代码运行时间为：", execution_time, "s")

'''
torch.save(model.state_dict(), 'model_nn_im_64_raw_30day_1500.pth')
model = CustomNet()
state_dict_numpy = torch.load('/model_nn0_52_raw_30day_1500.pth')
state_dict_torch = {name: torch.tensor(param) for name, param in state_dict_numpy.items()}
model.load_state_dict(state_dict_torch)
model.eval()
'''

''' testing '''
def test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0.0
    total = 0
    losses = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            losses.append(loss.item())
    avg_loss = test_loss / total
    print(f'Test Loss: {avg_loss:.8f}')
    return losses

loss_test = test(model, test_loader, loss_function)


def test_error(model, test_loader, loss_function):
    error = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        targets_array = targets.numpy().reshape(-1)
        outputs_array = outputs.detach().numpy().reshape(-1)
        error.append(targets_array - outputs_array)
    error = np.array(error)
    error = error.T
    return error

error = test_error(model, test_loader, loss_function)


''' one-by-one tesing'''
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_iter = iter(test_loader)
inputs, targets = next(test_iter)
inputs, targets = next(test_iter)
outputs = model(inputs)

# 将 inputs 和 targets 转换为 NumPy 数组
inputs_array = inputs.numpy().reshape(-1)
targets_array = targets.numpy().reshape(-1)
outputs_array = outputs.detach().numpy().reshape(-1)

# 画出 inputs 和 targets 的曲线
plt.figure(figsize=(10, 5))

# 画出 inputs 的曲线
plt.subplot(4, 1, 1)
plt.plot(inputs_array, label='Inputs')
plt.title('Inputs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 画出 targets 的曲线
plt.subplot(4, 1, 2)
plt.plot(targets_array, label='Targets')
plt.title('Targets')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 画出 targets 的曲线
plt.subplot(4, 1, 3)
plt.plot(outputs_array, label='outputs')
plt.title('outputs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 画出 targets 的曲线
plt.subplot(4, 1, 4)
plt.plot(abs(targets_array -outputs_array), label='Errors')
plt.title('Errors')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(targets_array, label='Targets')
plt.title('Targets')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.plot(outputs_array, label='outputs')
plt.title('outputs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(loss_train, label='loss_train')
plt.plot(loss_test, label='loss_test')
plt.xlabel('Iterarion')
plt.ylabel('Loss')
plt.legend()
plt.show()


# totally successfully on  Wen Mar 19 10:38:10 2024
































