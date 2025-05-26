# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:36:02 2024

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
# data = pd.read_excel(r'/training_data_pq_v_95_random.xlsx')
# data = data.values
data = pd.read_excel(r'/training_data_pq_v_95_3phase_random.xlsx')


# LV-95
# features = data.iloc[24*170:24*260, 0:Num_house*2].values
# # features = (features - 1.2)/0.4# for 64
# labels = data.iloc[24*170:24*260, Num_house*2:].values
# # labels = (labels - 0.98)/0.001 # # for 64

#  LV-95 3-phase
# features = data.iloc[24*170:24*260, 0:Num_house*2].values
# labels = data.iloc[24*170:24*260, Num_house*2:].values
# features = data.iloc[24*300:24*330, 0:Num_house*2].values
# labels = data.iloc[24*300:24*330, Num_house*2:].values
features = data.iloc[24*50:24*100, 0:Num_house*2].values
labels = data.iloc[24*50:24*100, Num_house*2:].values
features = (features - 1.3)/0.3# for 3-phase 95
labels = (labels - 0.99)/0.02 # # for 3-phase 95 used in my cases

indices_to_delete = [0, 1, 2, 3, 4, 5, 10, 11, 26, 27, 156, 157] #  95 wihtout demands nodes
features[:,indices_to_delete] = 0

features_tensor = torch.tensor(features, dtype=torch.float32) 
labels_tensor = torch.tensor(labels, dtype=torch.float32) 
dataset = TensorDataset(features_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# single LV-95
# # -------------------------------------------------------
# features = data.iloc[24*280:24*300, 0:Num_house*2].values
# features = data.iloc[24*20:24*40, 0:Num_house*2].values
# features = (features - 1.2)/0.4# for 64

# # labels = data.iloc[24*280:24*300, Num_house*2:].values
# labels = data.iloc[24*20:24*40, Num_house*2:].values
# labels = (labels - 0.98)/0.001 # # for 64

# indices_to_delete = [0, 1, 2, 3, 4, 5, 10, 11, 26, 27, 156, 157] #  95
# features[:,indices_to_delete] = 0

# features_tensor = torch.tensor(features, dtype=torch.float32) 
# labels_tensor = torch.tensor(labels, dtype=torch.float32) 
# dataset = TensorDataset(features_tensor, labels_tensor)
# test_loader_0 = DataLoader(dataset, batch_size=5, shuffle=False)  


# # ------------------------------------------------------------
# features = data.iloc[24*330:24*360, 0:Num_house*2].values
# features = data.iloc[24*150:24*200, 0:Num_house*2].values
# features = (features - 1.2)/0.4# for 64

# labels = data.iloc[24*330:24*360, Num_house*2:].values
# labels = data.iloc[24*150:24*200, Num_house*2:].values
# labels = (labels - 0.98)/0.001 # # for 64

# indices_to_delete = [0, 1, 2, 3, 4, 5, 10, 11, 26, 27, 156, 157] #  95
# features[:,indices_to_delete] = 0

# features_tensor = torch.tensor(features, dtype=torch.float32) 
# labels_tensor = torch.tensor(labels, dtype=torch.float32) 
# dataset = TensorDataset(features_tensor, labels_tensor)
# test_loader = DataLoader(dataset, batch_size=1, shuffle=False)  
 

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

# # NN_improved------------------------------------------------------------------
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
# #------------------------------------------------------------------------------

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


model = CustomNet()
# optimizer = optim.AdamW(model.parameters(), lr=0.000005)
# import time
# start_time = time.time()
# loss_train, model = train(model, train_loader, optimizer, num_epochs=1500)

# end_time = time.time()
# execution_time = end_time - start_time
# print("代码运行时间为：", execution_time, "s")

# optimizer = optim.AdamW(model.parameters(), lr=0.00005)
# loss_train, model = train(model, train_loader, optimizer, num_epochs=1500)




'''
# torch.save(model.state_dict(), 'model_nn_im_95_test_90day_1500.pth')
# torch.save(model.state_dict(), 'model_nn_95_3phase_test_1_30day_1000.pth')
model = CustomNet()
# state_dict_numpy = torch.load('/model_nn_im_95_test_90day_1500.pth')

# state_dict_numpy = torch.load(r'\model_nn_95_3phase_encry_30day_1500.pth')
state_dict_numpy = torch.load(r'\model_nn_95_3phase_test_1_30day_1000.pth')

state_dict_torch = {name: torch.tensor(param) for name, param in state_dict_numpy.items()}
model.load_state_dict(state_dict_torch)
for name, param in model.named_parameters():
    if any(sub_str in name for sub_str in ["fc5"]):
        param.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=0.00005)    
loss_train, model = train(model, train_loader, optimizer, num_epochs=1000)
loss_train, model = train(model, test_loader_0, optimizer, num_epochs=1000)
loss_train, model = train(model, test_loader, optimizer, num_epochs=1000)



for name, param in model.named_parameters():
    print(f"Name: {name}, requires_grad: {param.requires_grad}")
    

        
# 假设 model 是你的 PyTorch 模型
for name, param in model.named_parameters():
    numpy_param = param.detach().numpy()
    print(f"参数 {name} 的形状为：{numpy_param.shape}")


model.eval()
'''

import time
start_time = time.time()
# optimizer = optim.AdamW(model.parameters(), lr=0.000005)
# loss_train, model = train(model, test_loader_0, optimizer, num_epochs=1000)

end_time = time.time()
execution_time = end_time - start_time
print("代码运行时间为：", execution_time, "s")


# ''' testing '''
# def test(model, test_loader, loss_function):
#     model.eval()
#     test_loss = 0.0
#     total = 0
#     losses = []
    
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             loss = loss_function(outputs, targets)
#             test_loss += loss.item() * inputs.size(0)
#             total += targets.size(0)
#             losses.append(loss.item())
#     avg_loss = test_loss / total
#     print(f'Test Loss: {avg_loss:.8f}')
#     return losses

# loss_test = test(model, test_loader, loss_function)


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
error1=error*0.001













''' one-by-one tesing'''
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
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








