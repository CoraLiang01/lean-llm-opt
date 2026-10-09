import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def find_row_idx(df, pattern):
    for (idx, val) in enumerate(df['Product']):
        if re.search(pattern, str(val), re.IGNORECASE):
            return idx
    raise KeyError(f"Could not find a row matching pattern '{pattern}'")
df_36_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv', dtype=str, keep_default_na=False)
product_ids = [f'A{i}' for i in range(1, 81)]
row_demand = find_row_idx(df_36_1, 'maximum\\s*demand')
row_price = find_row_idx(df_36_1, 'selling\\s*price')
row_cost = find_row_idx(df_36_1, 'production\\s*cost')
row_quota = find_row_idx(df_36_1, 'production\\s*quota')
max_demand = {}
selling_price = {}
production_cost = {}
daily_quota = {}
for pid in product_ids:
    max_demand[pid] = int(float(df_36_1.at[row_demand, pid]))
    selling_price[pid] = float(df_36_1.at[row_price, pid])
    production_cost[pid] = float(df_36_1.at[row_cost, pid])
    daily_quota[pid] = float(df_36_1.at[row_quota, pid])
    if daily_quota[pid] <= 0:
        raise ValueError(f'Daily production quota for {pid} must be positive.')
activation_cost = {}
for pid in product_ids:
    activation_cost[pid] = float(df_36_2.at[0, pid])
min_batch_size = {}
for pid in product_ids:
    min_batch_size[pid] = int(float(df_36_3.at[0, pid]))
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[pid] - production_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= int(math.floor(daily_quota[pid] * 22)) for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch_size[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22, name='TotalProdDays')
m.optimize()