import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_row_idx(df, col, target):
    norm = lambda s: re.sub('\\s+', ' ', str(s)).strip().casefold()
    target_norm = norm(target)
    for idx, val in df[col].items():
        if norm(val) == target_norm:
            return idx
    raise KeyError(f"Row '{target}' not found in column '{col}'.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
products = [f'A{i}' for i in range(1, 81)]
row_demand = find_row_idx(df1, 'Product', 'Maximum Demand (100 kg units)')
row_price = find_row_idx(df1, 'Product', 'Selling Price ($/100 kg)')
row_cost = find_row_idx(df1, 'Product', 'Production Cost ($/100 kg)')
row_quota = find_row_idx(df1, 'Product', 'Production Quota (max per day)')
max_demand = {p: float(df1.at[row_demand, p]) for p in products}
selling_price = {p: float(df1.at[row_price, p]) for p in products}
prod_cost = {p: float(df1.at[row_cost, p]) for p in products}
daily_quota = {p: float(df1.at[row_quota, p]) for p in products}
row_actcost = find_row_idx(df2, 'Product', 'Activation Cost ($)')
activation_cost = {p: float(df2.at[row_actcost, p]) for p in products}
row_minbatch = find_row_idx(df3, 'Product', 'Minimum Batch Size (100 kg units)')
min_batch = {p: int(df3.at[row_minbatch, p]) for p in products}
m = gp.Model('MonthlyProductionPlan')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products))
m.setObjective(profit_terms, gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= 22 * daily_quota[p] for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= min(max_demand[p], 22 * daily_quota[p]) * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / daily_quota[p] for p in products)) <= 22)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        if y[p].X > 0.5:
            print(f'{p}: Produce {int(round(x[p].X))} (activated, min batch {min_batch[p]})')
        else:
            print(f'{p}: Not produced (inactive)')
else:
    print(f'No optimal solution found. Status: {m.status}')