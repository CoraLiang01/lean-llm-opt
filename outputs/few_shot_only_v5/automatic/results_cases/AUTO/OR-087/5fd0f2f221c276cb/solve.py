import gurobipy as gp
import pandas as pd
import numpy as np
import re

def get_row_by_label(df, label):
    norm = lambda s: str(s).strip().casefold()
    norm_label = norm(label)
    for idx, val in enumerate(df.iloc[:, 0]):
        if norm(val) == norm_label:
            return df.iloc[idx]
    raise KeyError(f"Row label '{label}' not found in DataFrame.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',', dtype=str, keep_default_na=False)
df2 = pd.read_csv(path2, sep=',', dtype=str, keep_default_na=False)
df3 = pd.read_csv(path3, sep=',', dtype=str, keep_default_na=False)
product_ids = [col for col in df1.columns if col.strip() != 'Product']
for df, fname in zip([df2, df3], [path2, path3]):
    cols = [col for col in df.columns if col.strip() != 'Product']
    if cols != product_ids:
        raise ValueError(f'Product columns in {fname} do not match those in {path1}.')
row_demand = get_row_by_label(df1, 'Maximum Demand (100 kg units)')
row_price = get_row_by_label(df1, 'Selling Price ($/100 kg)')
row_cost = get_row_by_label(df1, 'Production Cost ($/100 kg)')
row_quota = get_row_by_label(df1, 'Production Quota (max per day)')
row_activation = get_row_by_label(df2, 'Activation Cost ($)')
row_minbatch = get_row_by_label(df3, 'Minimum Batch Size (100 kg units)')

def to_int_dict(row):
    return {pid: int(str(row[pid]).strip()) for pid in product_ids}

def to_float_dict(row):
    return {pid: float(str(row[pid]).strip()) for pid in product_ids}
demand = to_int_dict(row_demand)
price = to_float_dict(row_price)
cost = to_float_dict(row_cost)
quota = to_int_dict(row_quota)
activation_cost = to_float_dict(row_activation)
min_batch = to_int_dict(row_minbatch)
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((price[pid] - cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / quota[pid] for pid in product_ids)) <= 22, name='total_days')
m.optimize()