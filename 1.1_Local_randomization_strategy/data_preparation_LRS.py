# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:55:56 2023

@author: dliu8
"""


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde


# %matplotlib qt #: make a new window for the figure

# =============================================================================
''' generation of input data '''
# =============================================================================

# data = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_95-PD_f.xlsx')
# data = data.values
# data = data[:,2:]


# time_periods = 5664  # 59 days
# nums1 = np.zeros((data.shape[0], 5664))
# for ii in range(len(data)):
#     nums = []
#     for j in range(time_periods):
#         temp = random.gauss(1, 0.2)
#         nums.append(temp)
#     nums1[ii,:] = np.array(nums)
    
# # 多次乘法操作
# nums = np.zeros((data.shape[0], 5664))
# for i in range(59):
#     nums[:, i * 96: (i + 1) * 96] = np.multiply(data, nums1[:, i * 96:(i + 1) * 96])
# # for i in range(52):
# #     plt.plot(nums[i,:])    
# nums = np.around(nums, decimals=3)

# file_name = fr"C:\Users\dliu8\Desktop\PD0.xlsx"
# nums = pd.DataFrame(nums)
# nums.to_excel(file_name, index=False)

# # # #  reactive power-----------------------------------------------------
# pf = 0.95
# pf1 = np.sqrt(1-0.95**2)/0.95
# data = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_52-PD_f.xlsx')
# data = data.values
# data = data * pf1
# data = np.around(data, decimals=3)

# file_name = fr"C:\Users\dliu8\Desktop\PD0.xlsx"
# nums = pd.DataFrame(data)
# nums.to_excel(file_name, index=False)

# #  training data preparation -----------------------------------------------------
P = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_95-PD_NL.xlsx')
# P = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/P_randomization_64.xlsx')
P = P.values
P = P[:,2:8642]

Q = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_95-QD_NL.xlsx')
# Q = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Q_randomization_64.xlsx')
Q = Q.values
Q = Q[:,2:8642]

V = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_95_voltage_NL.xlsx')
V = V.values

input_data = np.zeros((8640, int(P.shape[0]*3)))
for i in range(P.shape[1]):
    p = P[:,i].reshape(1, -1)
    q = Q[:,i].reshape(1, -1)
    v = V[:,i].reshape(1, -1)
    
    merged_array = np.empty(int(P.shape[0]) * 2)
    # 使用循环将两个数组中的元素交替存放在一个数组中
    for j in range(int(P.shape[0])):
        merged_array[j * 2] = p[0][j]
        merged_array[j * 2 + 1] = q[0][j]
    merged_array = merged_array.reshape(1, -1)
    input_data[i,:] = np.concatenate((merged_array, v), axis=1)
    
    # input_data[i,:] = np.concatenate((p, q, v), axis=1)

# # 将DataFrame保存为Excel文件
input_data = pd.DataFrame(input_data)
input_data.to_excel('training_data_pq_v_95.xlsx', index=False)


#  input data of each SM : [P, Q, V, 0, 0, V, 0, 0]

# input_data = np.zeros((5760, int(P.shape[0]*8)))
# for i in range(0, int(P.shape[0]*8), 8):
#     input_data[:,i] = P[int(i/8),:]
#     input_data[:,i+1] = Q[int(i/8),:]
#     input_data[:,i+2] = V[int(i/8),:]
#     input_data[:,i+5] = V[int(i/8),:]
    
  
# # # # 将DataFrame保存为Excel文件
# input_data = pd.DataFrame(input_data)
# input_data.to_excel('output.xlsx', index=False)   

# # -----------------------------------------------------
# array1 = System_Data_Nodes_pa.to_numpy()
# array2 = System_Data_Nodes_pb.to_numpy()
# array3 = System_Data_Nodes_pc.to_numpy()

# 数组相加




# '''
# # # successfully on 19/03/2024--------------power flow calculation step-by-step
# '''
#     sub_model = model
#     time_periods = 96
#   # --------------------------------------------------------------------------
#     PDa = {N[i]: System_Data_Nodes1.loc[i,f'PDa':f'PDa.{91*time_periods-1-time_periods}'].tolist() for i in System_Data_Nodes1.index}      
#     QDa = {N[i]: System_Data_Nodes2.loc[i,f'QDa':f'QDa.{91*time_periods-1-time_periods}'].tolist() for i in System_Data_Nodes2.index}   

#   model.Periods = RangeSet(1, 90*96, doc = 'Time Indices')
#   # Active and Reactive Demand Power
#   model.PDa = Param(model.N, model.Periods, initialize = 1.0, mutable = True)
#   for t in range(len(list(model.N))):
#       # for p in range(len(list(model.Periods))):
#       for p in range(90*96):  
#               model.PDa[t+1, p+1].value = float(PDa[t+1][p])                  
             
#   model.QDa = Param(model.N, model.Periods, initialize = 5.0, mutable = True)
#   for t in range(len(list(model.N))):
#       # for p in range(len(list(model.Periods))):
#       for p in range(90*96):
#               model.QDa[t+1, p+1].value = float(QDa[t+1][p])
                         
            
# #   # successfully on 19/03/2024----------------------------------------------
#     import time

#     time_periods = 1
#     model.Periods = RangeSet(1, time_periods, doc = 'Time Indices')
#     model.Time = RangeSet(1, 90, doc = 'Time Periods')
#     V_a = pd.DataFrame()
#     V_b = pd.DataFrame()
#     V_c = pd.DataFrame()
    
# start_time = time.time() #
#     for dd in range(1+31*96, 90*96+1):
#         print(dd)
#         #  Active and Reactive Demand Power update
#         for t in range(len(list(model.N))):
#             for p in range(len(list(model.Periods))):
#                 sub_model.PDa[t+1, p+1].value = model.PDa[t+1, (dd-1) * len(model.Periods) + p + 1].value
#                 sub_model.QDa[t+1, p+1].value = model.QDa[t+1, (dd-1) * len(model.Periods) + p + 1].value
                
#                 sub_model.PDb[t+1, p+1].value = model.PDb[t+1, (dd-1) * len(model.Periods) + p + 1].value
#                 sub_model.QDb[t+1, p+1].value = model.QDb[t+1, (dd-1) * len(model.Periods) + p + 1].value
                
#                 sub_model.PDc[t+1, p+1].value = model.PDc[t+1, (dd-1) * len(model.Periods) + p + 1].value
#                 sub_model.QDc[t+1, p+1].value = model.QDc[t+1, (dd-1) * len(model.Periods) + p + 1].value
            
#         # solve the model
#         solver = SolverFactory('ipopt')
#         results = solver.solve(sub_model, tee=True)  
        
#         # 循环遍历model.Va并将数据添加到DataFrame中
#         data0 = pd.DataFrame()

#         for i in sub_model.N:
#             for t in model.Periods:
#                 data0.loc[i, t] = float(sub_model.Va[i, t].value) / V0
                
#         V_a = pd.concat([V_a, data0], axis=1)     
        
#         # ----------------------------------
#         data0 = pd.DataFrame()

#         for i in sub_model.N:
#             for t in model.Periods:
#                 data0.loc[i, t] = float(sub_model.Vb[i, t].value) / V0
                
#         V_b = pd.concat([V_b, data0], axis=1)     
        
#         # ------------------------------
#         data0 = pd.DataFrame()

#         for i in sub_model.N:
#             for t in model.Periods:
#                 data0.loc[i, t] = float(sub_model.Vc[i, t].value) / V0
                
#         V_c = pd.concat([V_c, data0], axis=1)     
        

# end_time = time.time()
# execution_time = end_time - start_time
# print("代码运行时间为：", execution_time, "秒")

# # data1 = data.values.tolist()
# # 将DataFrame保存为Excel文件
# V_c.to_excel('outputs.xlsx', index=False)



# -----------------------------------------------------
#  data combination in 3-phase 95-LVDN
voltage_a = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/voltage_a.xlsx')
voltage_b = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/voltage_b.xlsx')
voltage_c = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/voltage_c.xlsx')


indices_a = [1, 2, 3, 4, 6, 8, 9, 14, 15, 19, 26, 27, 28, 32, 38, 39, 40, 46, 47, 51, 52, 58, 62, 63, 64, 72, 73, 74, 78, 80, 85, 86, 89, 90, 20]
indices_b = [7, 10, 11, 16, 24, 25, 29, 30, 36, 37, 42, 43, 44, 48, 49, 55, 56, 59, 60, 67, 68, 69, 75, 79, 81, 82, 91, 92, 94, 17]
indices_c = [5, 12, 13, 18, 22, 31, 33, 34, 41, 45, 50, 53, 54, 57, 61, 65, 66, 70, 71, 76, 83, 84, 87, 88, 93, 95, 21, 35, 77, 23]

indices_a = [i-1 for i in indices_a if i > 0]
indices_b = [i-1 for i in indices_b if i > 0]
indices_c = [i-1 for i in indices_c if i > 0]


# 提取对应bus index的行
selected_a = voltage_a.iloc[indices_a]
selected_b = voltage_b.iloc[indices_b]
selected_c = voltage_c.iloc[indices_c]

# 合并数据，保留原始行索引
voltage_all = pd.concat([selected_a, selected_b, selected_c], axis=0)
voltage_all = voltage_all.sort_index()
V_transposed = voltage_all.T
V_transposed.to_excel('outputs.xlsx', index=False)




# -------------------------------------------------------------------
# power_a = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/Nodes_95_PDa.xlsx')
# power_a = power_a.fillna(0)
# power_a = power_a.iloc[:, 0:8645]
# power_b = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/Nodes_95_PDb.xlsx')
# power_b = power_b.fillna(0)
# power_b = power_b.iloc[:, 0:8645]
# power_c = P = pd.read_excel(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation\unbalanced three phase/Nodes_95_PDc.xlsx')
# power_c = power_c.fillna(0)
# power_c = power_c.iloc[:, 0:8645]


# indices_a = [1, 2, 3, 4, 6, 8, 9, 14, 15, 19, 26, 27, 28, 32, 38, 39, 40, 46, 47, 51, 52, 58, 62, 63, 64, 72, 73, 74, 78, 80, 85, 86, 89, 90, 20]
# indices_b = [7, 10, 11, 16, 24, 25, 29, 30, 36, 37, 42, 43, 44, 48, 49, 55, 56, 59, 60, 67, 68, 69, 75, 79, 81, 82, 91, 92, 94, 17]
# indices_c = [5, 12, 13, 18, 22, 31, 33, 34, 41, 45, 50, 53, 54, 57, 61, 65, 66, 70, 71, 76, 83, 84, 87, 88, 93, 95, 21, 35, 77, 23]

# indices_a = [i-1 for i in indices_a if i > 0]
# indices_b = [i-1 for i in indices_b if i > 0]
# indices_c = [i-1 for i in indices_c if i > 0]

# # 提取对应bus index的行
# selected_a = power_a.iloc[indices_a]
# selected_b = power_b.iloc[indices_b]
# selected_c = power_c.iloc[indices_c]

# P_A = selected_a.iloc[:, 2:].sum()
# P_A = np.array(P_A)
# P_B = selected_b.iloc[:, 2:].sum()
# P_B = np.array(P_B)
# P_C = selected_c.iloc[:, 2:].sum()
# P_C = np.array(P_C)

# P_avg = (P_A + P_B + P_C) / 3
# # 计算LUF_P，形状为 (8640,)
# # max(|P_A - P_avg|, |P_B - P_avg|, |P_C - P_avg|) / P_avg * 100
# abs_diff_A = np.abs(P_A - P_avg)
# abs_diff_B = np.abs(P_B - P_avg)
# abs_diff_C = np.abs(P_C - P_avg)
# max_diff = np.maximum.reduce([abs_diff_A, abs_diff_B, abs_diff_C])
# LUF_P = max_diff / (P_avg + 1e-10) * 100  # 添加1e-10避免除零





 



















































































