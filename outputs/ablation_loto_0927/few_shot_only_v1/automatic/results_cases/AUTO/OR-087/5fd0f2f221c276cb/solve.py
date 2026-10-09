import gurobipy as gp
import pandas as pd
import numpy as np
import re

def get_row(df, label):
    idx = [i for (i, v) in enumerate(df['Product'].astype(str).str.strip()) if v.lower() == label.lower()]
    if not idx:
        raise KeyError(f"Row '{label}' not found in file {df}")
    return df.iloc[idx[0]]
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [col for col in df1.columns if col != 'Product']
products = product_cols.copy()
row_demand = get_row(df1, 'Maximum Demand (100 kg units)')
row_price = get_row(df1, 'Selling Price ($/100 kg)')
row_cost = get_row(df1, 'Production Cost ($/100 kg)')
row_quota = get_row(df1, 'Production Quota (max per day)')
row_activation = get_row(df2, 'Activation Cost ($)')
row_minbatch = get_row(df3, 'Minimum Batch Size (100 kg units)')

def to_dict(row):
    return {k: float(row[k]) for k in products}
max_demand = to_dict(row_demand)
selling_price = to_dict(row_price)
production_cost = to_dict(row_cost)
daily_quota = to_dict(row_quota)
activation_cost = to_dict(row_activation)
min_batch = to_dict(row_minbatch)
for pname in products:
    for (d, label) in [(max_demand, 'Maximum Demand'), (selling_price, 'Selling Price'), (production_cost, 'Production Cost'), (daily_quota, 'Production Quota'), (activation_cost, 'Activation Cost'), (min_batch, 'Minimum Batch Size')]:
        if pname not in d:
            raise KeyError(f'Missing {label} for product {pname}')
m = gp.Model('ProductionPlan80')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[p] - production_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= 22 * daily_quota[p] for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / daily_quota[p] for p in products)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        if y[p].X > 0.5:
            print(f'Product {p}:')
            print(f'  Status: ACTIVE (y={int(round(y[p].X))})')
            print(f'  Production: {int(round(x[p].X))} (100kg units)')
        else:
            print(f'Product {p}:')
            print(f'  Status: INACTIVE (y={int(round(y[p].X))})')
            print(f'  Production: {int(round(x[p].X))} (100kg units)')
else:
    print(f'No optimal solution found. Status: {m.status}')