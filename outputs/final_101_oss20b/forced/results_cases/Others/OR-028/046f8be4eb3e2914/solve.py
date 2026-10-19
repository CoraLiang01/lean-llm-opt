import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
df = pd.read_csv(data_path, sep=',')
product_ids = df['Product Name'].astype(str).tolist()
revenues = df['Revenue'].astype(int).tolist()
demands = df['Demand'].astype(int).tolist()
inventories = df['Initial Inventory'].astype(int).tolist()
m = gp.Model('RevenueMaximization')
x = m.addVars(len(product_ids), vtype=gp.GRB.INTEGER)
obj = gp.quicksum((revenues[i] * x[i] for i in range(len(product_ids))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demands[i] for i in range(len(product_ids))))
m.addConstrs((x[i] <= inventories[i] for i in range(len(product_ids))))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_rev = m.objVal
    print(f'Optimal total revenue: {total_rev:.2f}')
    print('Fulfilled units per product:')
    for i, pid in enumerate(product_ids):
        print(f'{pid}: {x[i].X:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')