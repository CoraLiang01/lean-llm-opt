import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, col, target):
    norm = lambda s: re.sub('\\s+', ' ', str(s)).strip().casefold()
    norm_target = norm(target)
    for (idx, val) in df[col].items():
        if norm(val) == norm_target:
            return idx
    raise KeyError(f"Row with {col} == '{target}' not found.")
df_36_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv', dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
idx_demand = find_row_idx(df_36_1, 'Product', 'Maximum Demand (100 kg units)')
idx_price = find_row_idx(df_36_1, 'Product', 'Selling Price ($/100 kg)')
idx_cost = find_row_idx(df_36_1, 'Product', 'Production Cost ($/100 kg)')
idx_quota = find_row_idx(df_36_1, 'Product', 'Production Quota (max per day)')
max_demand = {pid: float(df_36_1.at[idx_demand, pid]) for pid in product_ids}
price = {pid: float(df_36_1.at[idx_price, pid]) for pid in product_ids}
cost = {pid: float(df_36_1.at[idx_cost, pid]) for pid in product_ids}
daily_quota = {pid: float(df_36_1.at[idx_quota, pid]) for pid in product_ids}
idx_activation = find_row_idx(df_36_2, 'Product', 'Activation Cost ($)')
activation_cost = {pid: float(df_36_2.at[idx_activation, pid]) for pid in product_ids}
idx_minbatch = find_row_idx(df_36_3, 'Product', 'Minimum Batch Size (100 kg units)')
min_batch = {pid: float(df_36_3.at[idx_minbatch, pid]) for pid in product_ids}
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((price[pid] - cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * daily_quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22, name='shared_time')
m.optimize()