import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
df = pd.read_csv(path, sep=',')
mask = df['Product Name'].str.contains('27in', case=False, na=False)
df27 = df[mask].copy()
products = df27['Product Name'].tolist()
revenue = dict(zip(df27['Product Name'], df27['Revenue']))
demand = dict(zip(df27['Product Name'], df27['Demand']))
inventory = dict(zip(df27['Product Name'], df27['Initial Inventory']))
m = gp.Model('MaximizeRevenue27in')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs([x[p] <= demand[p] for p in products])
m.addConstrs([x[p] <= inventory[p] for p in products])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Sales Plan for 27in products ---')
    for p in products:
        qty = x[p].X
        if qty > 1e-06:
            print(f'{p}: {int(qty)} units (Revenue per unit: {revenue[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')