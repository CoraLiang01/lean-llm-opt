import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity = int(capacity_df.iloc[0]['Capacity'])
product_names = products_df['ProductName'].tolist()
weights = dict(zip(product_names, products_df['Weight']))
values = dict(zip(product_names, products_df['Value']))
m = gp.Model('SupermarketOrderOptimization')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((values[p] * x[p] for p in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in product_names)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Optimal Order Plan ---')
    for p in product_names:
        qty = x[p].X
        if qty > 0:
            print(f'{p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')