import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
df = pd.read_csv(csv_path, sep=',')
organ_mask = df['Sub Category'].str.contains('Organ', case=False, na=False)
df_organ = df.loc[organ_mask].copy()
products = df_organ['Sub Category'].tolist()
revenue = df_organ['Revenue'].to_dict()
demand = df_organ['Demand'].to_dict()
inventory = df_organ['Initial Inventory'].to_dict()
prod_to_idx = {prod: i for i, prod in enumerate(products)}
m = gp.Model('OrganRevenueMaximization')
x = m.addVars(len(products))
obj_expr = gp.quicksum((revenue[products[i]] * x[i] for i in range(len(products))))
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
for i in range(len(products)):
    prod = products[i]
    m.addConstr(x[i] <= demand[prod])
    m.addConstr(x[i] <= inventory[prod])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}\n')
    print('Optimal sales plan:')
    for i in range(len(products)):
        qty = x[i].X
        if qty > 1e-06:
            print(f'  {products[i]}: {qty:.2f} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')