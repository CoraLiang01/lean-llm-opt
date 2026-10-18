import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
products = df['Product Name'].tolist()
revenue = dict(zip(products, df['Revenue']))
demand = dict(zip(products, df['Demand']))
inventory = dict(zip(products, df['Initial Inventory']))
m = gp.Model('MaximizeRevenue')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= demand[p])
    m.addConstr(x[p] <= inventory[p])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nUnits sold per product:')
    for p in products:
        print(f'  {p}: {int(x[p].X)}')
else:
    print(f'Model status: {m.status}')