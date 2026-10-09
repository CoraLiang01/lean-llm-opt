import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, label_pattern):
    norm = lambda s: re.sub('\\s+', ' ', str(s)).strip().casefold()
    for (idx, val) in enumerate(df['Product']):
        if re.fullmatch(label_pattern, norm(val)):
            return idx
    raise KeyError(f"Row matching '{label_pattern}' not found in 'Product' column.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',', dtype=str, keep_default_na=False)
df2 = pd.read_csv(path2, sep=',', dtype=str, keep_default_na=False)
df3 = pd.read_csv(path3, sep=',', dtype=str, keep_default_na=False)
product_ids = [col for col in df1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}.')

def norm_label(s):
    return re.sub('\\s+', ' ', str(s)).strip().casefold()
row_labels_1 = [norm_label(x) for x in df1['Product']]
row_labels_2 = [norm_label(x) for x in df2['Product']]
row_labels_3 = [norm_label(x) for x in df3['Product']]
idx_demand = find_row_idx(df1, 'maximum demand \\(100 kg units\\)')
idx_price = find_row_idx(df1, 'selling price \\(\\$/100 kg\\)')
idx_cost = find_row_idx(df1, 'production cost \\(\\$/100 kg\\)')
idx_quota = find_row_idx(df1, 'production quota \\(max per day\\)')
idx_activation = find_row_idx(df2, 'activation cost \\(\\$\\)')
idx_minbatch = find_row_idx(df3, 'minimum batch size \\(100 kg units\\)')
max_demand = {pid: int(float(df1.at[idx_demand, pid])) for pid in product_ids}
price = {pid: float(df1.at[idx_price, pid]) for pid in product_ids}
cost = {pid: float(df1.at[idx_cost, pid]) for pid in product_ids}
quota = {pid: int(float(df1.at[idx_quota, pid])) for pid in product_ids}
activation_cost = {pid: float(df2.at[idx_activation, pid]) for pid in product_ids}
min_batch = {pid: int(float(df3.at[idx_minbatch, pid])) for pid in product_ids}
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = ((price[pid] - cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / quota[pid] for pid in product_ids)) <= 22, name='shared_days')
m.optimize()