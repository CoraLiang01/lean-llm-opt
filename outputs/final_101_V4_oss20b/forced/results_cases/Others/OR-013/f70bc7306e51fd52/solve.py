import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv'
df = pd.read_csv(path, sep=',')
mask = df['Product Name'].str.contains('4U', case=False, na=False)
df4u = df[mask].copy()
products = df4u['Product Name'].tolist()
revenues = df4u['Revenue'].tolist()
demands = df4u['Demand'].tolist()
invs = df4u['Initial Inventory'].tolist()
caps = [min(d, inv) for d, inv in zip(demands, invs)]
m = gp.Model('MaxRevenue4U')
x_vars = m.addVars(len(products), vtype=gp.GRB.INTEGER)
obj = gp.quicksum((revenues[i] * x_vars[i] for i in range(len(products))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in range(len(products)):
    m.addConstr(x_vars[i] <= caps[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Optimal Fulfillment Quantities ---')
    for i, prod in enumerate(products):
        qty = x_vars[i].X
        if qty > 0:
            print(f'{prod}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')