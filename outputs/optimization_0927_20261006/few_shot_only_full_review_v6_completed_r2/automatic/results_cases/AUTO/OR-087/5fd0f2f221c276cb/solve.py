import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, col, target):
    norm = lambda s: str(s).strip().casefold()
    matches = df[col].apply(norm) == norm(target)
    idxs = np.where(matches)[0]
    if len(idxs) == 0:
        raise KeyError(f"Row '{target}' not found in column '{col}'")
    return idxs[0]
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, sep=',', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(path_36_2, sep=',', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(path_36_3, sep=',', dtype=str, keep_default_na=False)
product_cols = [col for col in df_36_1.columns if col != 'Product']
products = product_cols.copy()
rowkey_36_1 = 'Product'
idx_demand = find_row_idx(df_36_1, rowkey_36_1, 'Maximum Demand (100 kg units)')
idx_price = find_row_idx(df_36_1, rowkey_36_1, 'Selling Price ($/100 kg)')
idx_cost = find_row_idx(df_36_1, rowkey_36_1, 'Production Cost ($/100 kg)')
idx_quota = find_row_idx(df_36_1, rowkey_36_1, 'Production Quota (max per day)')
max_demand = {p: int(df_36_1.at[idx_demand, p]) for p in products}
selling_price = {p: float(df_36_1.at[idx_price, p]) for p in products}
prod_cost = {p: float(df_36_1.at[idx_cost, p]) for p in products}
prod_quota = {p: float(df_36_1.at[idx_quota, p]) for p in products}
idx_activation = find_row_idx(df_36_2, 'Product', 'Activation Cost ($)')
activation_cost = {p: float(df_36_2.at[idx_activation, p]) for p in products}
idx_minbatch = find_row_idx(df_36_3, 'Product', 'Minimum Batch Size (100 kg units)')
min_batch = {p: int(df_36_3.at[idx_minbatch, p]) for p in products}
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[p] - prod_cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= max_demand[p], name=f'demand_{p}')
for p in products:
    m.addConstr(x_vars[p] <= int(22 * prod_quota[p]), name=f'quota_{p}')
for p in products:
    m.addConstr(x_vars[p] >= min_batch[p] * y_vars[p], name=f'minbatch_{p}')
for p in products:
    m.addConstr(x_vars[p] <= max_demand[p] * y_vars[p], name=f'link_{p}')
m.addConstr(gp.quicksum((x_vars[p] / prod_quota[p] if prod_quota[p] > 0 else 0.0 for p in products)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        x_val = x_vars[p].X
        y_val = y_vars[p].X
        if y_val > 0.5:
            print(f'{p}: ACTIVE (y={int(y_val)}), Production: {int(round(x_val))} (100kg units)')
        else:
            print(f'{p}: INACTIVE (y={int(y_val)}), Production: {int(round(x_val))} (100kg units)')
else:
    print(f'No optimal solution found. Status: {m.status}')