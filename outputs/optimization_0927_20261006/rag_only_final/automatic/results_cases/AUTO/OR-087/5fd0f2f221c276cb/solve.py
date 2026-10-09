import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(file_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(file_36_3, dtype=str, keep_default_na=False)
product_cols = [f'A{i}' for i in range(1, 81)]
products = product_cols.copy()

def get_row_by_product(df, row_name):
    mask = df['Product'].str.casefold().str.strip() == row_name.casefold().strip()
    rows = df[mask]
    if rows.shape[0] != 1:
        raise ValueError(f"Row '{row_name}' not found exactly once in file.")
    return rows.iloc[0]
max_demand_row = get_row_by_product(df_36_1, 'Maximum Demand (100 kg units)')
selling_price_row = get_row_by_product(df_36_1, 'Selling Price ($/100 kg)')
prod_cost_row = get_row_by_product(df_36_1, 'Production Cost ($/100 kg)')
prod_quota_row = get_row_by_product(df_36_1, 'Production Quota (max per day)')
max_demand = {p: int(float(max_demand_row[p])) for p in products}
selling_price = {p: float(selling_price_row[p]) for p in products}
prod_cost = {p: float(prod_cost_row[p]) for p in products}
prod_quota = {p: int(float(prod_quota_row[p])) for p in products}
activation_cost_row = get_row_by_product(df_36_2, 'Activation Cost ($)')
activation_cost = {p: float(activation_cost_row[p]) for p in products}
min_batch_row = get_row_by_product(df_36_3, 'Minimum Batch Size (100 kg units)')
min_batch = {p: int(float(min_batch_row[p])) for p in products}
num_days = 22
for (param_name, param_dict) in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
    if set(param_dict.keys()) != set(products):
        raise ValueError(f"Parameter '{param_name}' does not cover all products.")
m = Model('ProductionPlan')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(products, vtype=GRB.BINARY, name='')
for p in products:
    m.addConstr(x_vars[p] <= max_demand[p], name=f'demand_{p}')
for p in products:
    m.addConstr(x_vars[p] <= prod_quota[p] * num_days, name=f'capacity_{p}')
for p in products:
    m.addConstr(x_vars[p] >= y_vars[p] * min_batch[p], name=f'link_lb_{p}')
    upper = min(max_demand[p], prod_quota[p] * num_days)
    m.addConstr(x_vars[p] <= y_vars[p] * upper, name=f'link_ub_{p}')
profit_terms = ((selling_price[p] - prod_cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products)
m.setObjective(quicksum(profit_terms), GRB.MAXIMIZE)
m.optimize()