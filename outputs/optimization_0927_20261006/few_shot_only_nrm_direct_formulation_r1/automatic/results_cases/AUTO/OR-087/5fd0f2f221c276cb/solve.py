import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, label):
    norm = lambda s: str(s).strip().casefold()
    for (idx, val) in enumerate(df['Product']):
        if norm(val) == norm(label):
            return idx
    raise KeyError(f"Row with label '{label}' not found in 'Product' column.")
file1_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file2_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file3_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(file1_path, dtype=str, keep_default_na=False)
df2 = pd.read_csv(file2_path, dtype=str, keep_default_na=False)
df3 = pd.read_csv(file3_path, dtype=str, keep_default_na=False)
products = [f'A{i}' for i in range(1, 81)]
for (df, fname) in [(df1, '36-1.csv'), (df2, '36-2.csv'), (df3, '36-3.csv')]:
    missing = [p for p in products if p not in df.columns]
    if missing:
        raise KeyError(f'Missing products {missing} in {fname}')
idx_demand = find_row_idx(df1, 'Maximum Demand (100 kg units)')
idx_price = find_row_idx(df1, 'Selling Price ($/100 kg)')
idx_cost = find_row_idx(df1, 'Production Cost ($/100 kg)')
idx_quota = find_row_idx(df1, 'Production Quota (max per day)')
demand = {}
price = {}
cost = {}
quota = {}
for p in products:
    demand[p] = int(df1.at[idx_demand, p])
    price[p] = float(df1.at[idx_price, p])
    cost[p] = float(df1.at[idx_cost, p])
    quota[p] = float(df1.at[idx_quota, p])
idx_activation = find_row_idx(df2, 'Activation Cost ($)')
activation_cost = {}
for p in products:
    activation_cost[p] = float(df2.at[idx_activation, p])
idx_minbatch = find_row_idx(df3, 'Minimum Batch Size (100 kg units)')
min_batch = {}
for p in products:
    min_batch[p] = int(df3.at[idx_minbatch, p])
total_days = 22
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(products, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((price[p] - cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= quota[p] * total_days for p in products), name='')
m.addConstrs((x_vars[p] >= min_batch[p] * y_vars[p] for p in products), name='')
m.addConstrs((x_vars[p] <= demand[p] * y_vars[p] for p in products), name='')
m.addConstr(gp.quicksum((x_vars[p] / quota[p] if quota[p] > 0 else 0.0 for p in products)) <= total_days, name='total_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        x_val = x_vars[p].X
        y_val = y_vars[p].X
        if y_val > 0.5:
            print(f'{p}: Activated, Produce {int(round(x_val))} (100kg units)')
        else:
            print(f'{p}: Not activated')
else:
    print(f'No optimal solution found. Status: {m.status}')