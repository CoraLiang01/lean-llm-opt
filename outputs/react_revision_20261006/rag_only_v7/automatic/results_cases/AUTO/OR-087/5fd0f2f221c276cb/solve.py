import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',', dtype=str, keep_default_na=False)
df2 = pd.read_csv(path2, sep=',', dtype=str, keep_default_na=False)
df3 = pd.read_csv(path3, sep=',', dtype=str, keep_default_na=False)
product_cols = [f'A{i}' for i in range(1, 81)]
num_days = 22

def get_row_by_prefix(df, prefix):
    prefix_norm = ' '.join(prefix.casefold().split())
    for (idx, val) in df['Product'].items():
        val_norm = ' '.join(val.casefold().split())
        if val_norm.startswith(prefix_norm):
            return df.loc[idx]
    raise ValueError(f"Row with prefix '{prefix}' not found in DataFrame.")
max_demand_row = get_row_by_prefix(df1, 'Maximum Demand')
selling_price_row = get_row_by_prefix(df1, 'Selling Price')
prod_cost_row = get_row_by_prefix(df1, 'Production Cost')
prod_quota_row = get_row_by_prefix(df1, 'Production Quota')

def row_to_numeric_dict(row):
    return {k: float(row[k]) for k in product_cols}
max_demand = row_to_numeric_dict(max_demand_row)
selling_price = row_to_numeric_dict(selling_price_row)
prod_cost = row_to_numeric_dict(prod_cost_row)
prod_quota = row_to_numeric_dict(prod_quota_row)
activation_cost_row = get_row_by_prefix(df2, 'Activation Cost')
activation_cost = {k: float(activation_cost_row[k]) for k in product_cols}
min_batch_row = get_row_by_prefix(df3, 'Minimum Batch Size')
min_batch = {k: int(min_batch_row[k]) for k in product_cols}
for (param_name, param_dict) in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
    if set(param_dict.keys()) != set(product_cols):
        raise ValueError(f"Parameter '{param_name}' does not cover all products.")

def solve_problem():
    m = gp.Model('production_plan')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(product_cols, vtype=GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(product_cols, vtype=GRB.BINARY, name='')
    for i in product_cols:
        m.addConstr(x_vars[i] <= max_demand[i], name='demand_' + i)
        m.addConstr(x_vars[i] <= prod_quota[i] * num_days, name='quota_' + i)
        m.addConstr(x_vars[i] >= min_batch[i] * y_vars[i], name='minbatch_' + i)
        max_prod = min(max_demand[i], prod_quota[i] * num_days)
        m.addConstr(x_vars[i] <= max_prod * y_vars[i], name='link_' + i)
    profit_expr = gp.quicksum(((selling_price[i] - prod_cost[i]) * x_vars[i] - activation_cost[i] * y_vars[i] for i in product_cols))
    m.setObjective(profit_expr, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')