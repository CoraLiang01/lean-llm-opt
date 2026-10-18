import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
df = pd.read_csv(csv_path, sep=',')
mask = df['Product Name'].str.contains('27in', case=False, na=False)
df27 = df.loc[mask].reset_index(drop=True)
products = df27['Product Name'].tolist()
revenues = df27['Revenue'].tolist()
invs = df27['Initial Inventory'].tolist()
demands = df27['Demand'].tolist()
m = gp.Model('MaxRevenue')
x_vars = {}
for idx, prod in enumerate(products):
    var = m.addVar(vtype=gp.GRB.INTEGER, lb=0)
    x_vars[prod] = var
for idx, prod in enumerate(products):
    m.addConstr(x_vars[prod] <= invs[idx])
    m.addConstr(x_vars[prod] <= demands[idx])
obj_expr = gp.quicksum((revenues[idx] * x_vars[prod] for idx, prod in enumerate(products)))
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Sales Plan for 27in Products ---')
    for idx, prod in enumerate(products):
        units_sold = x_vars[prod].X
        print(f'{prod}: {int(units_sold)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')