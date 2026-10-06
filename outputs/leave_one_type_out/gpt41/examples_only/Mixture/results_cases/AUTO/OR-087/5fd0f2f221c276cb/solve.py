import gurobipy as gp
import pandas as pd
import numpy as np
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [col for col in df1.columns if col != 'Product']
products = product_cols
row_map_1 = {rowname.strip().casefold(): idx for idx, rowname in enumerate(df1['Product'])}

def find_row_idx(df, search):
    for idx, val in enumerate(df['Product']):
        if val.strip().casefold() == search.strip().casefold():
            return idx
    raise KeyError(f"Row '{search}' not found in DataFrame.")
idx_max_demand = find_row_idx(df1, 'Maximum Demand (100 kg units)')
idx_selling_price = find_row_idx(df1, 'Selling Price ($/100 kg)')
idx_prod_cost = find_row_idx(df1, 'Production Cost ($/100 kg)')
idx_prod_quota = find_row_idx(df1, 'Production Quota (max per day)')
max_demand = {p: float(df1.at[idx_max_demand, p]) for p in products}
selling_price = {p: float(df1.at[idx_selling_price, p]) for p in products}
prod_cost = {p: float(df1.at[idx_prod_cost, p]) for p in products}
prod_quota = {p: float(df1.at[idx_prod_quota, p]) for p in products}
if df2.shape[0] != 1:
    raise ValueError('36-2.csv should have exactly one row of activation costs.')
activation_cost = {p: float(df2.at[0, p]) for p in products}
if df3.shape[0] != 1:
    raise ValueError('36-3.csv should have exactly one row of minimum batch sizes.')
min_batch = {p: float(df3.at[0, p]) for p in products}
num_days = 22
m = gp.Model('ProductionPlan')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
for p in products:
    m.addConstr(x[p] <= max_demand[p], name=f'max_demand_{p}')
    m.addConstr(x[p] <= prod_quota[p] * num_days, name=f'prod_cap_{p}')
    m.addConstr(x[p] >= min_batch[p] * y[p], name=f'min_batch_{p}')
    M_p = min(max_demand[p], prod_quota[p] * num_days)
    m.addConstr(x[p] <= M_p * y[p], name=f'link_{p}')
profit_terms = gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products))
m.setObjective(profit_terms, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('Production Plan (100 kg units):')
    for p in products:
        xval = x[p].X
        yval = y[p].X
        if yval > 0.5:
            print(f'  {p}: {int(round(xval))} units (activated, fixed cost: {activation_cost[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')