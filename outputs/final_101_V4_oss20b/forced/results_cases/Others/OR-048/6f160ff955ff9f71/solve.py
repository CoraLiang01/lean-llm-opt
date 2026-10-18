import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
I = capacity_df['StorageID'].tolist()
J = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
m = gp.Model('AirConditionerAllocation')
x = m.addVars([(i, j) for i in I for j in J], vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in J)) <= capacity_dict[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (non‑zero units) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 0:
                print(f'Storage {i}, Product "{j}": {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')