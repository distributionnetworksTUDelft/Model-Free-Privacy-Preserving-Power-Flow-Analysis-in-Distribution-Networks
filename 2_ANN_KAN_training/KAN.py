# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:24:27 2024

@author: dliu8
"""

import torch
import torch.nn.functional as F
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
batch_size = 5
Num_house = 95


''' input dataset'''
data = pd.read_excel(r'/training_data_pq_v_95_random.xlsx')

# 假设数据的第一列是标签，剩余的列是特征
features = data.iloc[96*0:96*30, 0:Num_house*2].values
# features = np.log1p(features) # not good
features = (features - 1.2)/0.4
# features = np.where(features != 0, (features - 1) / 0.6, features)
labels = data.iloc[96*0:96*30, Num_house*2:].values
labels = (labels - 0.98)/0.001
# labels = 1/(1+np.exp(-labels))

# indices_to_delete = [0, 1, 2, 3, 16, 17] #  52
# indices_to_delete = [0, 1, 2, 3, 8, 9, 14, 15, 34, 35, 66, 67, 98, 99] #  64
indices_to_delete = [0, 1, 2, 3, 4, 5, 10, 11, 26, 27, 156, 157] #  95
features[:,indices_to_delete] = 0



features_tensor = torch.tensor(features, dtype=torch.float32) 
labels_tensor = torch.tensor(labels, dtype=torch.float32) 

# 创建 TensorDataset 对象
dataset = TensorDataset(features_tensor, labels_tensor)
train_size = int(train_ratio * len(dataset)) 
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        # base_activation=torch.nn.SiLU,
        base_activation=F.tanh,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.base_weight, self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )




# import torch

# # 定义输入数据和标签
# input_data = torch.randn(1, 52 * 2)  # 输入数据是 52*2 维度
# target = torch.randn(1, 52)  # 输出是 52 维度

# # 创建KAN模型实例
# model = KAN(layers_hidden=[52 * 2, 512, 512, 512, 52])

# # 使用模型进行前向传播
# output = model(input_data)


# 打印损失值
# print("损失值:", output.item())


# import torch

# 生成输入数据和标签
# input_data = torch.randn(1000, 52 * 2)  # 输入数据是 1000 个样本，每个样本是 52*2 维度
# labels = torch.randn(1000, 52)  # 标签是 1000 个样本，每个样本是 52 维度
# # 定义批处理大小
# batch_size = 25

# 划分数据集为批处理
# input_data_batches = features_tensor.chunk(features_tensor.size(0) // batch_size)
# labels_batches = labels_tensor.chunk(labels_tensor.size(0) // batch_size)

# 打印数据集的形状
# print("输入数据形状:", input_data_batches.shape)
# print("标签形状:", labels.shape)



import torch
import torch.optim as optim
import torch.nn.functional as F
# from your_module import KAN  # 导入你的KAN网络模块

# 创建KAN网络实例
kan_model = KAN(layers_hidden=[Num_house * 2, 512, Num_house])
# torch.save(kan_model.state_dict(), 'model_kan_52_encry_30day_150.pth')


# 定义优化器
optimizer = optim.AdamW(kan_model.parameters(), lr=0.001)

# 定义损失函数
def loss_function(output, target):
    loss = F.mse_loss(output, target)
    return loss

# 训练模型
num_epochs = 150
loss_train = []

import time
start_time = time.time()
for epoch in range(num_epochs):

    # 将模型设置为训练模式
    kan_model.train()
    total_loss = 0.0
    # 遍历训练数据集
    for i, data in enumerate(train_loader, 0):  # 假设这里有输入数据和标签的批处理
        input_data, target = data
        # 将输入数据和标签转换为张量
        # input_data = torch.tensor(input_data, dtype=torch.float32)
        # target = torch.tensor(target, dtype=torch.float32)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = kan_model(input_data)
        
        # 计算损失
        loss = loss_function(output, target)
        total_loss += loss.item()
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
    loss_train.append(total_loss/len(train_loader))
    # 每个epoch结束后输出损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 保存训练好的模型
# torch.save(kan_model.state_dict(), 'kan_model.pth')
end_time = time.time()
execution_time = end_time - start_time
print("代码运行时间为：", execution_time, "s")


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

loss_test = test(kan_model, test_loader, loss_function)


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

error = test_error(kan_model, test_loader, loss_function)





''' one-by-one tesing'''
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_iter = iter(test_loader)
inputs, targets = next(test_iter)
inputs, targets = next(test_iter)
outputs = kan_model(inputs)

# 将 inputs 和 targets 转换为 NumPy 数组
inputs_array = inputs.numpy().reshape(-1)
targets_array = targets.numpy().reshape(-1)
outputs_array = outputs.detach().numpy().reshape(-1)

# 画出 inputs 和 targets 的曲线
plt.figure(figsize=(10, 5))
plt.plot(targets_array*0.001+0.98, label='Targets')
plt.title('Targets')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.plot(outputs_array*0.001+0.98, label='outputs')
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

















