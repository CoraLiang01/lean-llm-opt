import gurobipy as gp
import pandas as pd
import numpy as np
import re

def get_row_by_label(df, label):
    norm_label = re.sub('\\s+', ' ', label.strip()).casefold()
    for (idx, val) in df['Product'].items():
        norm_val = re.sub('\\s+', ' ', str(val).strip()).casefold()
        if norm_val == norm_label:
            return df.loc[idx]
    raise KeyError(f"Row with label '{label}' not found in DataFrame.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',', dtype=str, keep_default_na=False)
df2 = pd.read_csv(path2, sep=',', dtype=str, keep_default_na=False)
df3 = pd.read_csv(path3, sep=',', dtype=str, keep_default_na=False)
product_ids = [col for col in df1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, got {len(product_ids)}')
row_demand = get_row_by_label(df1, 'Maximum Demand (100 kg units)')
row_price = get_row_by_label(df1, 'Selling Price ($/100 kg)')
row_cost = get_row_by_label(df1, 'Production Cost ($/100 kg)')
row_quota = get_row_by_label(df1, 'Production Quota (max per day)')
row_activation = get_row_by_label(df2, 'Activation Cost ($)')
row_minbatch = get_row_by_label(df3, 'Minimum Batch Size (100 kg units)')

def to_float_dict(row):
    return {pid: float(row[pid]) for pid in product_ids}

def to_int_dict(row):
    return {pid: int(float(row[pid])) for pid in product_ids}
max_demand = to_int_dict(row_demand)
selling_price = to_float_dict(row_price)
production_cost = to_float_dict(row_cost)
daily_quota = to_float_dict(row_quota)
activation_cost = to_float_dict(row_activation)
min_batch = to_int_dict(row_minbatch)
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[pid] - production_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * daily_quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22, name='TotalDays')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for pid in product_ids:
        x_val = x_vars[pid].X
        y_val = y_vars[pid].X
        if y_val > 0.5:
            print(f'{pid}: Produce {int(round(x_val))} (activated, min batch {min_batch[pid]})')
        else:
            print(f'{pid}: Not produced')
else:
    print(f'No optimal solution found. Status: {m.status}')