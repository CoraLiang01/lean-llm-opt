import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
df = pd.read_csv(path, sep=',')
fashion_df = df[df['Product Name'].str.contains('Fashion', case=False, na=False)].copy()
m = gp.Model('FashionRevenueMaximization')
variables = {}
revenue = {}
upper = {}
for idx, row in fashion_df.iterrows():
    prod = row['Product Name']
    rev = row['Revenue']
    dem = row['Demand']
    inv = row['Initial Inventory']
    ub = min(dem, inv)
    var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=ub)
    variables[prod] = var
    revenue[prod] = rev
    upper[prod] = ub
m.setObjective(gp.quicksum((revenue[p] * variables[p] for p in variables)), gp.GRB.MAXIMIZE)
for prod, var in variables.items():
    m.addConstr(var <= upper[prod])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nOptimal sales plan:')
    for prod, var in variables.items():
        if var.X > 1e-06:
            print(f'  {prod}: {var.X:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')