import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(file_36_1, sep=',')
df2 = pd.read_csv(file_36_2, sep=',')
df3 = pd.read_csv(file_36_3, sep=',')
products = [col for col in df1.columns if col != 'Product']

def get_row(df, row_name):
    row = df[df['Product'].str.casefold().str.strip() == row_name.casefold().strip()]
    if row.empty:
        raise KeyError(f"Row '{row_name}' not found in file.")
    return row.iloc[0]
max_demand_row = get_row(df1, 'Maximum Demand (100 kg units)')
max_demand = {p: float(max_demand_row[p]) for p in products}
selling_price_row = get_row(df1, 'Selling Price ($/100 kg)')
selling_price = {p: float(selling_price_row[p]) for p in products}
prod_cost_row = get_row(df1, 'Production Cost ($/100 kg)')
prod_cost = {p: float(prod_cost_row[p]) for p in products}
quota_row = get_row(df1, 'Production Quota (max per day)')
prod_quota = {p: float(quota_row[p]) for p in products}
if df2.shape[0] != 1:
    raise ValueError('36-2.csv should have exactly one row.')
activation_cost_row = df2.iloc[0]
activation_cost = {p: float(activation_cost_row[p]) for p in products}
if df3.shape[0] != 1:
    raise ValueError('36-3.csv should have exactly one row.')
min_batch_row = df3.iloc[0]
min_batch = {p: float(min_batch_row[p]) for p in products}
for (param_name, param_dict) in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
    missing = set(products) - set(param_dict)
    if missing:
        raise KeyError(f'Missing {param_name} for products: {missing}')
m = gp.Model('ProductionPlan80')
m.Params.MIPGap = 0.0001
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= 22 * prod_quota[p] for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / prod_quota[p] for p in products)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    for p in products:
        print(f'{x[p].VarName}: {x[p].X:.0f}')
        print(f'{y[p].VarName}: {y[p].X:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')