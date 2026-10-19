import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
df = pd.read_csv(data_path, sep=',')
products = df['Product Name'].tolist()
revenue_dict = dict(zip(df['Product Name'], df['Revenue']))
demand_dict = dict(zip(df['Product Name'], df['Demand']))
inventory_dict = dict(zip(df['Product Name'], df['Initial Inventory']))
m = gp.Model('MaxRevenue')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
m.addConstrs((x[p] <= demand_dict[p] for p in products))
m.addConstrs((x[p] <= inventory_dict[p] for p in products))
objective = gp.quicksum((revenue_dict[p] * x[p] for p in products))
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}\n')
    print('Optimal sales plan:')
    for p in products:
        qty = x[p].X
        if qty > 0:
            print(f'  {p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')