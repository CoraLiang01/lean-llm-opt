import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df.loc[0, 'Capacity'])
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',')
products = products_df['ProductName'].tolist()
profit = dict(zip(products, products_df['Value']))
weight = dict(zip(products, products_df['Weight']))
m = gp.Model('BakeryKnapsack')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((profit[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        qty = x[p].X
        if qty > 0:
            print(f'{p}: {int(qty)} units')
    used_capacity = sum((weight[p] * x[p].X for p in products))
    print(f'Used capacity: {used_capacity:.0f} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')