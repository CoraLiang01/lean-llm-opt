import gurobipy as gp
import pandas as pd
import numpy as np
import re

def get_product_ids():
    return [f'A{i}' for i in range(1, 81)]

def extract_row(df, row_label):

    def norm(s):
        return re.sub('\\s+', ' ', str(s)).strip().casefold()
    target = norm(row_label)
    for (idx, val) in enumerate(df['Product']):
        if norm(val) == target:
            return df.iloc[idx]
    raise KeyError(f"Row '{row_label}' not found in DataFrame.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, dtype=str, keep_default_na=False)
df2 = pd.read_csv(path2, dtype=str, keep_default_na=False)
df3 = pd.read_csv(path3, dtype=str, keep_default_na=False)
product_ids = get_product_ids()
row_demand = extract_row(df1, 'Maximum Demand (100 kg units)')
row_price = extract_row(df1, 'Selling Price ($/100 kg)')
row_cost = extract_row(df1, 'Production Cost ($/100 kg)')
row_quota = extract_row(df1, 'Production Quota (max per day)')
row_activation = extract_row(df2, 'Activation Cost ($)')
row_minbatch = extract_row(df3, 'Minimum Batch Size (100 kg units)')
max_demand = {pid: float(row_demand[pid]) for pid in product_ids}
price = {pid: float(row_price[pid]) for pid in product_ids}
cost = {pid: float(row_cost[pid]) for pid in product_ids}
daily_quota = {pid: float(row_quota[pid]) for pid in product_ids}
activation_cost = {pid: float(row_activation[pid]) for pid in product_ids}
min_batch = {pid: float(row_minbatch[pid]) for pid in product_ids}
for pid in product_ids:
    for (d, name) in [(max_demand, 'max_demand'), (price, 'price'), (cost, 'cost'), (daily_quota, 'daily_quota'), (activation_cost, 'activation_cost'), (min_batch, 'min_batch')]:
        if pid not in d:
            raise KeyError(f'Missing {name} for product {pid}')
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((price[pid] - cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * daily_quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22, name='shared_days')
m.optimize()