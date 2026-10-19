import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
df = pd.read_csv(data_path, sep=',')
products = df['Product Name'].tolist()
revenue = dict(zip(df['Product Name'], df['Revenue']))
demand = dict(zip(df['Product Name'], df['Demand']))
inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
m = gp.Model('PizzaRevenueMaximization')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0)
for p in products:
    m.addConstr(x[p] <= demand[p])
for p in products:
    m.addConstr(x[p] <= inventory[p])
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment plan ---')
    for p in products:
        qty = x[p].X
        if qty > 0:
            print(f'{p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')