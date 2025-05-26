# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:32:28 2024

@author: dliu8
"""

# =============================================================================
# this py code represent ths SM side, which is a verifier
# successfully on 27/09/2024
# =============================================================================

import zmq
from random import randint
import time
import json
from ecpy.curves import Curve, Point
import random
import numpy as np
# import csv
import pandas as pd


cp = Curve.get_curve("secp256k1")

# -----------------------------------------------------------------------------
def create_commit(p, g, h, m, r):
    # Create to scalar points on the curve
    mg = cp.mul_point(m, g)
    rh = cp.mul_point(r, h)

    # # Commitment which is the two points on the curve
    c = cp.add_point(mg, rh) # Combine points (mod p is inherently respected in elliptic curve operations)

    return c, r

def open(p, g, h, m1, c1, s1, b, m2, c2, s2):
    # Verifier's check        
    if b == 0:
        r1 = s1
        r2 = (s2 - (1-b)*m2) % p
    else:
        r1 = (s1 - b*m1) % p
        r2 = s2

    o1, _ = create_commit(p, g, h, m1, r1)
    o2, _ = create_commit(p, g, h, m2, r2)

    if o1 == c1 and o2 == c2:
        return True
    else:
        return False
    
# def open(p, g, h, m1, c1, s1, b, m2, c2, s2, m3, c3, s3):
#     # Verifier's check        
#     if b == 0:
#         r1 = s1
#         r2 = (s2 - (1-b)*m2) % p
#         r3 = s3
#     else:
#         r1 = (s1 - b*m1) % p
#         r2 = s2
#         r3 = (s3 - b*m3) % p

#     o1, _ = create_commit(p, g, h, m1, r1)
#     o2, _ = create_commit(p, g, h, m2, r2)
#     o3, _ = create_commit(p, g, h, m3, r3)


#     if o1 == c1 and o2 == c2 and o3 == c3:
#         return True
#     else:
#         return False

# def open(p, g, h, m1, c1, s1, b, m2, c2, s2, m3, c3, s3, m4, c4, s4):
#     # Verifier's check        
#     if b == 0:
#         r1 = s1
#         r2 = (s2 - (1-b)*m2) % p
#         r3 = s3
#         r4 = (s4 - (1-b)*m4) % p
#     else:
#         r1 = (s1 - b*m1) % p
#         r2 = s2
#         r3 = (s3 - b*m3) % p
#         r4 = s4

#     o1, _ = create_commit(p, g, h, m1, r1)
#     o2, _ = create_commit(p, g, h, m2, r2)
#     o3, _ = create_commit(p, g, h, m3, r3)
#     o4, _ = create_commit(p, g, h, m4, r4)



#     if o1 == c1 and o2 == c2 and o3 == c3 and o4 == c4:
#         return True
#     else:
#         return False    
    

def combinations(k):
    return list((i, j) for i in range(k+1) for j in range(i+1, k+1))
 
def combinations_of_three(k):
    """

    """
    return list((i, j, m) for i in range(k+1) for j in range(i+1, k+1) for m in range(j+1, k+1))

def combinations_of_four(k):
    """

    """
    result = []
    for i in range(k+1):
        for j in range(i+1, k+1):
            for m in range(j+1, k+1):
                for n in range(m+1, k+1):
                    result.append((i, j, m, n))
    return result
    
 
    
 # -------------------------
def generate_random_numbers(data_0, scaler, k, m1, m2):
  """
  """

  data = int(data_0 * scaler)

  data1 = random.randint(int(data * 0.49), int(data * 0.51))
  data2 = data - data1

  min_val = int(min(data1, data2) * 0.99)
  max_val = int(max(data1, data2) * 1.01)

  D = [random.randint(min_val, max_val) for _ in range(k)]

  D[m1] = data1
  D[m2] = data2  

  return D


# def generate_random_numbers(data_0, scaler, k, m1, m2, m3, m4):
#     """
#     生成随机数并构造新向量，将data_0不完全均匀分为4份，替换指定位置
#     Args:
#         data_0: 初始浮点数
#         scaler: 缩放因子
#         k: 生成的随机数个数
#         m1, m2, m3, m4: 要替换的索引位置

#     Returns:
#         list: 新向量D
#     """

#     # 数据预处理
#     data = int(data_0 * scaler)

#     # 不完全均匀划分
#     # 引入随机性，保证每次划分都不完全相同
#     random.seed()
#     data1 = random.randint(int(data * 0.24), int(data * 0.26))
#     data2 = random.randint(int(data * 0.24), int(data * 0.26))
#     data3 = random.randint(int(data * 0.24), int(data * 0.26))
#     data4 = data - data1 - data2 - data3

#     # 生成随机数
#     D = [random.randint(int(min(data1, data2, data3, data4) * 0.99), int(max(data1, data2, data3, data4) * 1.01)) for _ in range(k)]

#     # 替换指定位置
#     D[m1] = data1
#     D[m2] = data2
#     D[m3] = data3
#     D[m4] = data4

# %%
#     return D



    
# -----------------------------------------------------------------------------
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

send = socket.send_json
recv = socket.recv_json

# -----------------------------------------------------------------------------

N = 63       # the number of SM 
scaler = 10**4
time_l = 96*30

data = pd.read_excel(r'C:\Users\dliu8\Desktop\Paper & Data\RQ2\2.0-Code\FL-based-power-flow-model\power flow_nn\data generation/Nodes_95_voltage_NL.xlsx')
data = data.values
data_0 = data[1:, 2:2+time_l]     # except reference bus. 1 p.u.
data_0 = data_0.round (4)

# N = len(data_0)





time0 = []
NN = 20
for k in range(16, 18, 2):
    for N in range(NN, NN+10, 10):
        print(N)
        start_time = time.time()
    
        
        # sm_ind = []
        #------ZKP---------------------------------------------------------------------
        for t in range(0, N):
    
            # -------------------------------------------------------------------------
            p, g, h = recv()
        
            data_dict = json.loads(g)
            g = Point(data_dict['x'], data_dict['y'], cp)
        
            data_dict = json.loads(h)
            h = Point(data_dict['x'], data_dict['y'], cp)
        
            # -------------------------------------------------------------------------
            # k = randint(6,  12)
    
                
            send (k)
        
            list_com = combinations(k-1)
            # list_com = combinations_of_three(k-1)
            # list_com = combinations_of_four(k-1)
            # list_com = [tuple(map(lambda x: x + 1, tup)) for tup in list_com]

            kk = len(list_com)
        
            m_ind = randint(0, kk-1)
            m1 = list_com[m_ind][0] + p
            m2 = list_com[m_ind][1] + p
            # m3 = list_com[m_ind][2] + p
            # m4 = list_com[m_ind][3] + p


            
            print(kk, "------------", m_ind)
             
            for i in range(0, kk):
                print(i)
                # m1, m2 = recv()
                
                c1 = recv()
                data_dict = json.loads(c1)
                c1 = Point(data_dict['x'], data_dict['y'], cp)
                
                c2 = recv()
                data_dict = json.loads(c2)
                c2 = Point(data_dict['x'], data_dict['y'], cp)
                
                # c3 = recv()
                # data_dict = json.loads(c3)
                # c3 = Point(data_dict['x'], data_dict['y'], cp)
                
                # c4 = recv()
                # data_dict = json.loads(c4)
                # c4 = Point(data_dict['x'], data_dict['y'], cp)
                
                b = randint(0, 1)  # V choice
                send(b)
                    
                s1, s2 = recv()

                # s1, s2, s3 = recv()
                # s1, s2, s3, s4 = recv()

                
                is_valid = open(p, g, h, m1, c1, s1, b, m2, c2, s2)
                # is_valid = open(p, g, h, m1, c1, s1, b, m2, c2, s2, m3, c3, s3)
                # is_valid = open(p, g, h, m1, c1, s1, b, m2, c2, s2, m3, c3, s3, m4, c4, s4)

                
                if is_valid:
                    print("Commitment is valid.")
                    send("exit")
                    
                    # =================================================================
                    # sending the SM dataset
                    # generate a distribution of uniform format 
                    # =================================================================
                    for ii in range(0, time_l):
                        send(generate_random_numbers(data_0[t][ii], scaler, k, m1-p, m2-p))
                        # send(generate_random_numbers(data_0[t][ii], scaler, k, m1-p, m2-p, m3-p, m4-p))
        
                    break
                
                else:
                    # print("Commitment is invalid.")
                    send(1)
            
            # sm_ind.append([k, kk, m_ind, m1, m2,is_valid])
        
        
            
        end_time = time.time()
        execution_time = end_time - start_time
        time0.append(execution_time)
        print("running time cost", execution_time, "s")
            
time0 = np.array(time0)
 
time1 = recv()
time1 = np.array(time1)

time10 = time0 + time1



























