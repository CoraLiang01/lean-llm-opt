import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, target):
    for idx, val in df['Product'].items():
        if str(val).strip().casefold() == target.strip().casefold():
            return idx
    raise KeyError(f"Row with Product='{target}' not found in DataFrame.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_ids = [col for col in df1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}.')
idx_demand = find_row_idx(df1, 'Maximum Demand (100 kg units)')
idx_price = find_row_idx(df1, 'Selling Price ($/100 kg)')
idx_cost = find_row_idx(df1, 'Production Cost ($/100 kg)')
idx_quota = find_row_idx(df1, 'Production Quota (max per day)')
max_demand = {pid: float(df1.at[idx_demand, pid]) for pid in product_ids}
selling_price = {pid: float(df1.at[idx_price, pid]) for pid in product_ids}
prod_cost = {pid: float(df1.at[idx_cost, pid]) for pid in product_ids}
quota = {pid: float(df1.at[idx_quota, pid]) for pid in product_ids}
if df2.shape[0] != 1:
    raise ValueError('36-2.csv should have exactly one row.')
activation_cost = {pid: float(df2.at[0, pid]) for pid in product_ids}
if df3.shape[0] != 1:
    raise ValueError('36-3.csv should have exactly one row.')
min_batch = {pid: float(df3.at[0, pid]) for pid in product_ids}
m = gp.Model('ProductionPlan80Products')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, name='')
y = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = gp.quicksum(((selling_price[pid] - prod_cost[pid]) * x[pid] - activation_cost[pid] * y[pid] for pid in product_ids))
m.setObjective(profit_terms, gp.GRB.MAXIMIZE)
m.addConstrs((x[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x[pid] <= 22 * quota[pid] for pid in product_ids), name='')
m.addConstrs((x[pid] >= min_batch[pid] * y[pid] for pid in product_ids), name='')
m.addConstrs((x[pid] <= max_demand[pid] * y[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x[pid] / quota[pid] for pid in product_ids)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for pid in product_ids:
        xi = x[pid].X
        yi = y[pid].X
        if yi > 0.5:
            print(f'{pid}: Produced {int(round(xi))} (100kg units), Line Activated (y=1)')
        else:
            print(f'{pid}: Not produced (x=0), Line Not Activated (y=0)')
else:
    print(f'No optimal solution found. Status: {m.status}')