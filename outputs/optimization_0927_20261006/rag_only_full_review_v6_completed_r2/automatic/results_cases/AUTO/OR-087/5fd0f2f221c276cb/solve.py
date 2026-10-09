import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(file_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(file_36_3, dtype=str, keep_default_na=False)
product_ids = [f'A{i}' for i in range(1, 81)]
df_36_1['Product_norm'] = df_36_1['Product'].str.strip().str.casefold()

def get_row_by_norm(df, col, value):
    idx = df[col].str.strip().str.casefold() == value.strip().casefold()
    if not idx.any():
        raise ValueError(f"Row '{value}' not found in {col}")
    return df[idx].iloc[0]
row_max_demand = get_row_by_norm(df_36_1, 'Product', 'Maximum Demand (100 kg units)')
max_demand = {pid: int(float(row_max_demand[pid])) for pid in product_ids}
row_selling_price = get_row_by_norm(df_36_1, 'Product', 'Selling Price ($/100 kg)')
selling_price = {pid: float(row_selling_price[pid]) for pid in product_ids}
row_prod_cost = get_row_by_norm(df_36_1, 'Product', 'Production Cost ($/100 kg)')
prod_cost = {pid: float(row_prod_cost[pid]) for pid in product_ids}
row_daily_quota = get_row_by_norm(df_36_1, 'Product', 'Production Quota (max per day)')
daily_quota = {pid: int(float(row_daily_quota[pid])) for pid in product_ids}
row_activation_cost = df_36_2.iloc[0]
activation_cost = {pid: float(row_activation_cost[pid]) for pid in product_ids}
row_min_batch = df_36_3.iloc[0]
min_batch_size = {pid: int(float(row_min_batch[pid])) for pid in product_ids}
num_days = 22
m = Model()
x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=GRB.BINARY, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= max_demand[pid])
    m.addConstr(x_vars[pid] <= daily_quota[pid] * num_days)
    m.addConstr(x_vars[pid] >= min_batch_size[pid] * y_vars[pid])
    max_possible = min(max_demand[pid], daily_quota[pid] * num_days)
    m.addConstr(x_vars[pid] <= max_possible * y_vars[pid])
profit_terms = []
for pid in product_ids:
    unit_profit = selling_price[pid] - prod_cost[pid]
    profit_terms.append(unit_profit * x_vars[pid] - activation_cost[pid] * y_vars[pid])
m.setObjective(quicksum(profit_terms), GRB.MAXIMIZE)
m.optimize()