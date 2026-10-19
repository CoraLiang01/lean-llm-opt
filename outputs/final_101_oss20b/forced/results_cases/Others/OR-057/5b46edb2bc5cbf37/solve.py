import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
platforms = capacity_df['PlatformID'].tolist()
products = products_df['ProductName'].tolist()
capacity_dict = dict(zip(platforms, capacity_df['Capacity']))
value_dict = dict(zip(products, products_df['Value']))
weight_dict = dict(zip(products, products_df['Weight']))
m = gp.Model('GameListingOptimization')
x = m.addVars(platforms, products, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((value_dict[j] * x[i, j] for i in platforms for j in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in platforms:
    mem_used = gp.quicksum((weight_dict[j] * x[i, j] for j in products))
    m.addConstr(mem_used <= capacity_dict[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\nAllocation per platform:')
    for i in platforms:
        print(f'Platform {i} (Capacity: {capacity_dict[i]}):')
        used_mem = 0
        for j in products:
            qty = x[i, j].X
            if qty > 0:
                print(f'  {j}: {int(qty)} units (Weight {weight_dict[j]} each)')
                used_mem += qty * weight_dict[j]
        print(f'  Total memory used: {used_mem} / {capacity_dict[i]}\n')
else:
    print(f'No optimal solution found. Status: {m.status}')