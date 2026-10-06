import gurobipy as gp
import pandas as pd
import numpy as np
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [col for col in df1.columns if col.startswith('A')]
if len(product_cols) != 80:
    raise ValueError(f'Expected 80 products (A1-A80), got {len(product_cols)}')
products = product_cols

def get_row_value(df, row_name, products):
    idx = df['Product'].str.casefold().str.strip() == row_name.casefold().strip()
    if not idx.any():
        raise ValueError(f"Row '{row_name}' not found in file.")
    row = df.loc[idx].iloc[0]
    return {p: float(row[p]) for p in products}
max_demand = get_row_value(df1, 'Maximum Demand (100 kg units)', products)
selling_price = get_row_value(df1, 'Selling Price ($/100 kg)', products)
prod_cost = get_row_value(df1, 'Production Cost ($/100 kg)', products)
prod_quota = get_row_value(df1, 'Production Quota (max per day)', products)

def get_row_value_single(df, row_name, products):
    if not df['Product'].iloc[0].casefold().strip() == row_name.casefold().strip():
        raise ValueError(f"Row '{row_name}' not found in file.")
    row = df.iloc[0]
    return {p: float(row[p]) for p in products}
activation_cost = get_row_value_single(df2, 'Activation Cost ($)', products)
min_batch_size = get_row_value_single(df3, 'Minimum Batch Size (100 kg units)', products)
num_days = 22
BigM = {}
for p in products:
    BigM[p] = min(max_demand[p], prod_quota[p] * num_days)
m = gp.Model('ProductionPlan')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= max_demand[p], name=f'demand_{p}')
for p in products:
    m.addConstr(x[p] <= prod_quota[p] * num_days, name=f'capacity_{p}')
for p in products:
    m.addConstr(x[p] >= min_batch_size[p] * y[p], name=f'minbatch_{p}')
for p in products:
    m.addConstr(x[p] <= BigM[p] * y[p], name=f'link_{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('Production Plan (100kg units):')
    for p in products:
        if x[p].X > 1e-06:
            print(f'  {p}: {int(round(x[p].X))} units, Activated: {int(round(y[p].X))}')
    print('Activated lines and their fixed costs:')
    for p in products:
        if y[p].X > 0.5:
            print(f'  {p}: Activation Cost = {activation_cost[p]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')