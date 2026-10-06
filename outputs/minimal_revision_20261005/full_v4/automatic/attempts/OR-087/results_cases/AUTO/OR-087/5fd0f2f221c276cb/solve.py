import gurobipy as gp
import pandas as pd
import numpy as np
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, sep=',')
df_36_2 = pd.read_csv(path_36_2, sep=',')
df_36_3 = pd.read_csv(path_36_3, sep=',')
product_keys = [f'A{i}' for i in range(1, 81)]

def get_row(df, row_name):
    row = df[df['Product'].str.casefold().str.strip() == row_name.casefold().strip()]
    if row.empty:
        raise KeyError(f"Row '{row_name}' not found in file.")
    return row.iloc[0]
max_demand_row = get_row(df_36_1, 'Maximum Demand (100 kg units)')
selling_price_row = get_row(df_36_1, 'Selling Price ($/100 kg)')
prod_cost_row = get_row(df_36_1, 'Production Cost ($/100 kg)')
quota_row = get_row(df_36_1, 'Production Quota (max per day)')
if not (df_36_2.shape[0] == 1 and df_36_2.iloc[0]['Product'].strip().casefold() == 'activation cost ($)'.casefold()):
    raise KeyError("Row 'Activation Cost ($)' not found in 36-2.csv.")
activation_cost_row = df_36_2.iloc[0]
if not (df_36_3.shape[0] == 1 and df_36_3.iloc[0]['Product'].strip().casefold() == 'minimum batch size (100 kg units)'.casefold()):
    raise KeyError("Row 'Minimum Batch Size (100 kg units)' not found in 36-3.csv.")
min_batch_row = df_36_3.iloc[0]

def build_param_dict(row):
    d = {}
    for k in product_keys:
        if k not in row:
            raise KeyError(f"Missing product key '{k}' in row.")
        d[k] = float(row[k])
    return d
max_demand = build_param_dict(max_demand_row)
selling_price = build_param_dict(selling_price_row)
prod_cost = build_param_dict(prod_cost_row)
quota = build_param_dict(quota_row)
activation_cost = build_param_dict(activation_cost_row)
min_batch = build_param_dict(min_batch_row)
for d in [max_demand, selling_price, prod_cost, quota, activation_cost, min_batch]:
    if set(d.keys()) != set(product_keys):
        raise ValueError('Parameter dictionary keys do not match product_keys.')
total_days = 22

def solve_problem():
    m = gp.Model('ProductionPlan80')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(product_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[i] - prod_cost[i]) * x[i] - activation_cost[i] * y[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[i] <= max_demand[i] for i in product_keys), name='')
    m.addConstrs((x[i] <= total_days * quota[i] for i in product_keys), name='')
    m.addConstrs((x[i] >= min_batch[i] * y[i] for i in product_keys), name='')
    m.addConstrs((x[i] <= max_demand[i] * y[i] for i in product_keys), name='')
    m.addConstr(gp.quicksum((x[i] / quota[i] for i in product_keys)) <= total_days, name='shared_days')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')