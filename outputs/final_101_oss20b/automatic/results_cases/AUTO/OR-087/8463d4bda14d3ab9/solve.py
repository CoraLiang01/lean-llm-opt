import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path_36_1 = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path_36_1)
df2 = pd.read_csv(path_36_2)
df3 = pd.read_csv(path_36_3)
demand_row = df1[df1['Product'] == 'Maximum Demand (100 kg units)'].iloc[0]
price_row = df1[df1['Product'] == 'Selling Price ($/100 kg)'].iloc[0]
cost_row = df1[df1['Product'] == 'Production Cost ($/100 kg)'].iloc[0]
quota_row = df1[df1['Product'] == 'Production Quota (max per day)'].iloc[0]
activation_row = df2[df2['Product'] == 'Activation Cost ($)'].iloc[0]
minbatch_row = df3[df3['Product'] == 'Minimum Batch Size (100 kg units)'].iloc[0]
product_cols = [col for col in df1.columns if col != 'Product']
Demand = {int(col[1:]): demand_row[col] for col in product_cols}
Price = {int(col[1:]): price_row[col] for col in product_cols}
Cost = {int(col[1:]): cost_row[col] for col in product_cols}
Quota = {int(col[1:]): quota_row[col] for col in product_cols}
FixedCost = {int(col[1:]): activation_row[col] for col in product_cols}
MinBatch = {int(col[1:]): minbatch_row[col] for col in product_cols}
P = list(range(1, 81))
m = gp.Model('ProductionPlan')
x = m.addVars(P, vtype=gp.GRB.INTEGER)
y = m.addVars(P, vtype=gp.GRB.BINARY)
obj = gp.quicksum(((Price[p] - Cost[p]) * x[p] - FixedCost[p] * y[p] for p in P))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for p in P:
    m.addConstr(x[p] <= Demand[p])
    m.addConstr(x[p] <= Quota[p] * 22)
    m.addConstr(x[p] <= Demand[p] * y[p])
    m.addConstr(x[p] <= Quota[p] * 22 * y[p])
    m.addConstr(x[p] >= MinBatch[p] * y[p])
m.addConstr(gp.quicksum((x[p] / Quota[p] for p in P)) <= 22)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in P:
        prod_qty = x[p].X
        activated = y[p].X
        print(f'Product {p}:')
        print(f'  Activated: {('Yes' if activated > 0.5 else 'No')}')
        print(f'  Quantity (100 kg units): {int(prod_qty)}')
else:
    print(f'No optimal solution found. Status: {m.status}')