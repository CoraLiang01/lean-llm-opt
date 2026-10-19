import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
df = pd.read_csv(data_path, sep=',')
ele_s_df = df[df['Product_Reference'].str.startswith('ELE-S')].copy()
products = ele_s_df['Product_Reference'].tolist()
revenue = dict(zip(ele_s_df['Product_Reference'], ele_s_df['Revenue']))
demand = dict(zip(ele_s_df['Product_Reference'], ele_s_df['Demand']))
inventory = dict(zip(ele_s_df['Product_Reference'], ele_s_df['Initial Inventory']))
m = gp.Model('Maximize_ELE_S_Revenue')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
for prod in products:
    m.addConstr(x[prod] <= demand[prod])
for prod in products:
    m.addConstr(x[prod] <= inventory[prod])
m.setObjective(gp.quicksum((revenue[prod] * x[prod] for prod in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nFulfilled units per product:')
    for prod in products:
        print(f'  {prod}: {int(x[prod].X)} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')