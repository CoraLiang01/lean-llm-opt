import gurobipy as gp
import pandas as pd
import numpy as np
file1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(file1, sep=',', dtype=str, keep_default_na=False)
df2 = pd.read_csv(file2, sep=',', dtype=str, keep_default_na=False)
df3 = pd.read_csv(file3, sep=',', dtype=str, keep_default_na=False)
products = [f'A{i}' for i in range(1, 81)]

def get_row_dict(df, row_label):
    row = df[df['Product'].str.strip() == row_label]
    if row.empty:
        raise KeyError(f"Row '{row_label}' not found in DataFrame.")
    row = row.iloc[0]
    return {p: row[p] for p in products}
max_demand_raw = get_row_dict(df1, 'Maximum Demand (100 kg units)')
selling_price_raw = get_row_dict(df1, 'Selling Price ($/100 kg)')
prod_cost_raw = get_row_dict(df1, 'Production Cost ($/100 kg)')
prod_quota_raw = get_row_dict(df1, 'Production Quota (max per day)')
activation_cost_raw = get_row_dict(df2, 'Activation Cost ($)')
min_batch_raw = get_row_dict(df3, 'Minimum Batch Size (100 kg units)')

def to_int_dict(d):
    return {k: int(d[k]) for k in d}

def to_float_dict(d):
    return {k: float(d[k]) for k in d}
max_demand = to_int_dict(max_demand_raw)
selling_price = to_float_dict(selling_price_raw)
prod_cost = to_float_dict(prod_cost_raw)
prod_quota = to_int_dict(prod_quota_raw)
activation_cost = to_float_dict(activation_cost_raw)
min_batch = to_int_dict(min_batch_raw)
for param_name, param_dict in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
    missing = set(products) - set(param_dict)
    if missing:
        raise KeyError(f'Missing {param_name} for products: {missing}')
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(products, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= 22 * prod_quota[p] for p in products), name='')
m.addConstrs((x_vars[p] >= min_batch[p] * y_vars[p] for p in products), name='')
m.addConstrs((x_vars[p] <= max_demand[p] * y_vars[p] for p in products), name='')
m.addConstr(gp.quicksum((x_vars[p] / prod_quota[p] for p in products)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        x_val = x_vars[p].X
        y_val = y_vars[p].X
        if y_val > 0.5:
            print(f'{p}: ACTIVE, Produce {int(round(x_val))} (100kg units)')
        else:
            print(f'{p}: INACTIVE, Produce 0')
else:
    print(f'No optimal solution found. Status: {m.status}')