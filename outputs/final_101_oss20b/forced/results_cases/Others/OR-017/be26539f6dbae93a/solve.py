import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
df = pd.read_csv(csv_path, sep=',')
zz_mask = df['SKU'].str.contains('ZZ', na=False)
zz_df = df.loc[zz_mask].copy()
skus = zz_df['SKU'].tolist()
revenue = zz_df.set_index('SKU')['Revenue'].to_dict()
demand = zz_df.set_index('SKU')['Demand'].to_dict()
inventory = zz_df.set_index('SKU')['Initial Inventory'].to_dict()
m = gp.Model('ZZ_Product_Revenue_Maximization')
x = m.addVars(skus, vtype=gp.GRB.INTEGER)
obj_expr = gp.quicksum((revenue[s] * x[s] for s in skus))
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
m.addConstrs((x[s] <= demand[s] for s in skus))
m.addConstrs((x[s] <= inventory[s] for s in skus))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Sales Plan for ZZ Products ---')
    for s in skus:
        qty = x[s].X
        if qty > 0:
            print(f'SKU {s}: sell {int(qty)} units (Revenue per unit: {revenue[s]:.2f})')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')