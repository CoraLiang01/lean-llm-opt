import gurobipy as gp
import pandas as pd
import numpy as np
import re
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_1 = pd.read_csv(path_36_1, sep=',')
df_2 = pd.read_csv(path_36_2, sep=',')
df_3 = pd.read_csv(path_36_3, sep=',')
product_ids = [f'A{i}' for i in range(1, 81)]

def get_param_row(df, row_name):
    mask = df['Product'].astype(str).str.casefold() == row_name.casefold()
    if not mask.any():
        raise KeyError(f"Row '{row_name}' not found in DataFrame.")
    row = df.loc[mask].iloc[0]
    return {pid: float(row[pid]) for pid in product_ids}

def find_row_name(df, pattern):
    for name in df['Product']:
        if re.search(pattern, str(name), re.IGNORECASE):
            return name
    raise KeyError(f"Could not find a row matching pattern '{pattern}'")
demand_row = find_row_name(df_1, 'demand')
price_row = find_row_name(df_1, 'price')
cost_row = find_row_name(df_1, 'cost')
quota_row = find_row_name(df_1, 'quota')
max_demand = get_param_row(df_1, demand_row)
selling_price = get_param_row(df_1, price_row)
production_cost = get_param_row(df_1, cost_row)
production_quota = get_param_row(df_1, quota_row)
activation_row = find_row_name(df_2, 'activation|fixed')
activation_cost = get_param_row(df_2, activation_row)
minbatch_row = find_row_name(df_3, 'batch|min')
min_batch_size = get_param_row(df_3, minbatch_row)
for pid in product_ids:
    for d in [max_demand, selling_price, production_cost, production_quota, activation_cost, min_batch_size]:
        if pid not in d:
            raise KeyError(f'Missing parameter for product {pid}')

def solve_problem():
    m = gp.Model('ProductionPlan80')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[pid] - production_cost[pid]) * x[pid] - activation_cost[pid] * y[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[pid] <= max_demand[pid] for pid in product_ids), name='')
    m.addConstrs((x[pid] <= 22 * production_quota[pid] for pid in product_ids), name='')
    m.addConstrs((x[pid] >= min_batch_size[pid] * y[pid] for pid in product_ids), name='')
    m.addConstrs((x[pid] <= max_demand[pid] * y[pid] for pid in product_ids), name='')
    m.addConstr(gp.quicksum((x[pid] / production_quota[pid] if production_quota[pid] > 0 else 0.0 for pid in product_ids)) <= 22, name='shared_days')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')