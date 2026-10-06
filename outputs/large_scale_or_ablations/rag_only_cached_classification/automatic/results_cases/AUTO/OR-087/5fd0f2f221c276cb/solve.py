import pandas as pd
import numpy as np
from gurobipy import Model, GRB
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, sep=',')
df_36_2 = pd.read_csv(file_36_2, sep=',')
df_36_3 = pd.read_csv(file_36_3, sep=',')
product_cols = [f'A{i}' for i in range(1, 81)]
products = product_cols.copy()

def get_row_by_prefix(df, prefix):
    mask = df['Product'].str.casefold().str.strip().str.startswith(prefix.casefold().strip())
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row starting with '{prefix}', found {len(rows)}")
    return rows.iloc[0]
max_demand_row = get_row_by_prefix(df_36_1, 'Maximum Demand')
selling_price_row = get_row_by_prefix(df_36_1, 'Selling Price')
prod_cost_row = get_row_by_prefix(df_36_1, 'Production Cost')
prod_quota_row = get_row_by_prefix(df_36_1, 'Production Quota')
max_demand = {p: int(max_demand_row[p]) for p in products}
selling_price = {p: float(selling_price_row[p]) for p in products}
prod_cost = {p: float(prod_cost_row[p]) for p in products}
prod_quota = {p: int(prod_quota_row[p]) for p in products}
if df_36_2.shape[0] != 1:
    raise ValueError('36-2.csv must have exactly one row')
activation_cost_row = df_36_2.iloc[0]
activation_cost = {p: float(activation_cost_row[p]) for p in products}
if df_36_3.shape[0] != 1:
    raise ValueError('36-3.csv must have exactly one row')
min_batch_row = df_36_3.iloc[0]
min_batch = {p: int(min_batch_row[p]) for p in products}
num_days = 22
for param_name, param_dict in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
    if set(param_dict.keys()) != set(products):
        raise ValueError(f"Parameter '{param_name}' does not cover all products")
m = Model('ProductionPlanning')
x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=GRB.BINARY, name='')
for p in products:
    m.addConstr(x[p] <= max_demand[p])
for p in products:
    m.addConstr(x[p] <= prod_quota[p] * num_days)
for p in products:
    m.addConstr(x[p] >= min_batch[p] * y[p])
for p in products:
    m.addConstr(x[p] <= prod_quota[p] * num_days * y[p])
obj = sum(((selling_price[p] - prod_cost[p]) * x[p] for p in products)) - sum((activation_cost[p] * y[p] for p in products))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()