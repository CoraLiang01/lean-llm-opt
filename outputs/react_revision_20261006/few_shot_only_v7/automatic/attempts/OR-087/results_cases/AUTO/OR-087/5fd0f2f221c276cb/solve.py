import gurobipy as gp
import pandas as pd
import numpy as np
import re
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_1 = pd.read_csv(file_36_1, dtype=str, keep_default_na=False)
df_2 = pd.read_csv(file_36_2, dtype=str, keep_default_na=False)
df_3 = pd.read_csv(file_36_3, dtype=str, keep_default_na=False)
product_ids = [f'A{i}' for i in range(1, 81)]

def get_param_row(df, row_name):
    mask = df['Product'].str.casefold() == row_name.casefold()
    if not mask.any():
        raise ValueError(f"Row '{row_name}' not found in DataFrame.")
    row = df.loc[mask].iloc[0]
    return {pid: row[pid] for pid in product_ids}

def find_row_name(df, pattern):
    for rn in df['Product']:
        if re.search(pattern, rn, re.IGNORECASE):
            return rn
    raise ValueError(f"Row matching '{pattern}' not found.")
rowname_demand = find_row_name(df_1, 'demand')
rowname_price = find_row_name(df_1, 'price')
rowname_cost = find_row_name(df_1, 'cost')
rowname_quota = find_row_name(df_1, 'quota')
max_demand = get_param_row(df_1, rowname_demand)
selling_price = get_param_row(df_1, rowname_price)
production_cost = get_param_row(df_1, rowname_cost)
production_quota = get_param_row(df_1, rowname_quota)
for d in [max_demand, selling_price, production_cost, production_quota]:
    for k in d:
        d[k] = float(d[k])
rowname_activation = find_row_name(df_2, 'activation|fixed')
activation_cost = get_param_row(df_2, rowname_activation)
for k in activation_cost:
    activation_cost[k] = float(activation_cost[k])
rowname_minbatch = find_row_name(df_3, 'batch|min')
min_batch_size = get_param_row(df_3, rowname_minbatch)
for k in min_batch_size:
    min_batch_size[k] = float(min_batch_size[k])
for (dname, d) in [('max_demand', max_demand), ('selling_price', selling_price), ('production_cost', production_cost), ('production_quota', production_quota), ('activation_cost', activation_cost), ('min_batch_size', min_batch_size)]:
    missing = set(product_ids) - set(d.keys())
    if missing:
        raise ValueError(f'Missing products in {dname}: {missing}')

def solve_problem():
    m = gp.Model('ProductionPlan80')
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[i] - production_cost[i]) * x_vars[i] - activation_cost[i] * y_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= max_demand[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] <= 22 * production_quota[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] >= min_batch_size[i] * y_vars[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] <= max_demand[i] * y_vars[i] for i in product_ids), name='')
    m.addConstr(gp.quicksum((x_vars[i] / production_quota[i] if production_quota[i] > 0 else 0.0 for i in product_ids)) <= 22, name='shared_days')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')