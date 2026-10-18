import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv(products_path, sep=',')
product_names = products_df['ProductName'].tolist()
weights = dict(zip(product_names, products_df['Weight']))
values = dict(zip(product_names, products_df['Value']))
m = gp.Model('SupermarketRestock')
var_map = {}
for idx, p in enumerate(product_names):
    var_map[p] = m.addVar(vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[p] * var_map[p] for p in product_names))
m.setObjective(obj, gp.GRB.MAXIMIZE)
capacity_constr = gp.quicksum((weights[p] * var_map[p] for p in product_names))
m.addConstr(capacity_constr <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('Order quantities:')
    for p in product_names:
        qty = var_map[p].X
        if qty > 0:
            print(f'  {p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')