import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def find_row_by_regex(df, pattern):
    for (idx, val) in df['Product'].items():
        if re.search(pattern, val, re.IGNORECASE):
            return idx
    raise KeyError(f"Could not find a row matching pattern '{pattern}'")
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(path_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(path_36_3, dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}')
row_demand = find_row_by_regex(df_36_1, 'max.*demand|demand')
row_price = find_row_by_regex(df_36_1, 'price')
row_cost = find_row_by_regex(df_36_1, 'cost')
row_quota = find_row_by_regex(df_36_1, 'quota')
max_demand = {pid: int(df_36_1.at[row_demand, pid]) for pid in product_ids}
selling_price = {pid: float(df_36_1.at[row_price, pid]) for pid in product_ids}
production_cost = {pid: float(df_36_1.at[row_cost, pid]) for pid in product_ids}
daily_quota = {pid: float(df_36_1.at[row_quota, pid]) for pid in product_ids}
row_activation = find_row_by_regex(df_36_2, 'activation|cost')
activation_cost = {pid: float(df_36_2.at[row_activation, pid]) for pid in product_ids}
row_minbatch = find_row_by_regex(df_36_3, 'batch|min')
min_batch_size = {pid: int(df_36_3.at[row_minbatch, pid]) for pid in product_ids}
total_days = 22
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_expr = gp.quicksum(((selling_price[pid] - production_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids))
m.setObjective(profit_expr, gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= int(math.floor(daily_quota[pid] * total_days)) for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch_size[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] if daily_quota[pid] > 0 else 0.0 for pid in product_ids)) <= total_days, name='TotalProductionDays')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for pid in product_ids:
        x_val = x_vars[pid].X
        y_val = y_vars[pid].X
        if y_val > 0.5:
            print(f'{pid}: ACTIVE, Production = {int(round(x_val))} (100kg units)')
        else:
            print(f'{pid}: INACTIVE, Production = 0')
else:
    print(f'No optimal solution found. Status: {m.status}')