import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
products = [f'A{i}' for i in range(1, 81)]

def get_param_row(df, param_pattern):
    mask = df['Product'].str.replace('\\s+', '', regex=True).str.casefold().str.contains(param_pattern.replace(' ', '').casefold())
    matches = df[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for parameter '{param_pattern}', found {len(matches)}")
    return matches.iloc[0]
row_price = get_param_row(df1, 'price')
row_cost = get_param_row(df1, 'production cost')
row_demand = get_param_row(df1, 'maximum demand')
row_quota = get_param_row(df1, 'production quota')
row_activation = get_param_row(df2, 'activation cost')
row_minbatch = get_param_row(df3, 'minimum batch size')

def build_param_dict(row):
    return {p: float(row[p]) for p in products}
selling_price = build_param_dict(row_price)
production_cost = build_param_dict(row_cost)
activation_cost = build_param_dict(row_activation)
max_demand = build_param_dict(row_demand)
prod_quota = build_param_dict(row_quota)
min_batch = build_param_dict(row_minbatch)
for (param_name, param_dict) in [('selling_price', selling_price), ('production_cost', production_cost), ('activation_cost', activation_cost), ('max_demand', max_demand), ('prod_quota', prod_quota), ('min_batch', min_batch)]:
    missing = set(products) - set(param_dict)
    if missing:
        raise ValueError(f'Missing {param_name} for products: {missing}')

def solve_problem():
    m = gp.Model('MonthlyProductionPlan')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[p] - production_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
    m.addConstrs((x[p] <= 22 * prod_quota[p] for p in products), name='')
    m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
    m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
    m.addConstr(gp.quicksum((x[p] / prod_quota[p] if prod_quota[p] > 0 else 0.0 for p in products)) <= 22, name='shared_days')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')