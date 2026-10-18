import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
df = pd.read_csv(path, sep=',')
product_names = df['Product Name'].tolist()
revenue = dict(zip(product_names, df['Revenue']))
demand = dict(zip(product_names, df['Demand']))
inv = dict(zip(product_names, df['Initial Inventory']))
m = gp.Model('RetailRevenueOptimization')
x = m.addVars(product_names)
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
for i in product_names:
    m.addConstr(x[i] <= demand[i])
    m.addConstr(x[i] <= inv[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nFulfillment quantities:')
    for i in product_names:
        qty = x[i].X
        if qty > 1e-06:
            print(f'  {i}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')