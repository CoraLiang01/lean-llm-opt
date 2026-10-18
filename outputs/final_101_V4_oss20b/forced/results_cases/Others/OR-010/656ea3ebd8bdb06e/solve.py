import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
df = pd.read_csv(path, sep=',')
products = df['Product Name'].tolist()
revenue = dict(zip(products, df['Revenue']))
demand = dict(zip(products, df['Demand']))
inventory = dict(zip(products, df['Initial Inventory']))
m = gp.Model('MobileSalesRevenueMaximization')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products))
m.addConstrs((x[p] <= inventory[p] for p in products))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nFulfillment plan:')
    for p in products:
        qty = x[p].X
        if qty > 0:
            print(f'  {p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')