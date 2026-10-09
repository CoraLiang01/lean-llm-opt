import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'

def solve_problem():
    df_36_1 = pd.read_csv(file_36_1, sep=',', dtype=str, keep_default_na=False)
    df_36_2 = pd.read_csv(file_36_2, sep=',', dtype=str, keep_default_na=False)
    df_36_3 = pd.read_csv(file_36_3, sep=',', dtype=str, keep_default_na=False)
    product_cols = [col for col in df_36_1.columns if col.startswith('A')]
    if len(product_cols) != 80:
        raise ValueError(f'Expected 80 product columns, found {len(product_cols)}')
    products = product_cols

    def find_row_idx(df, row_name):
        target = row_name.strip().casefold()
        for (idx, val) in enumerate(df['Product']):
            if val.strip().casefold() == target:
                return idx
        raise KeyError(f"Row '{row_name}' not found in file.")
    idx_demand = find_row_idx(df_36_1, 'Maximum Demand (100 kg units)')
    idx_price = find_row_idx(df_36_1, 'Selling Price ($/100 kg)')
    idx_cost = find_row_idx(df_36_1, 'Production Cost ($/100 kg)')
    idx_quota = find_row_idx(df_36_1, 'Production Quota (max per day)')
    demand = {p: float(df_36_1.at[idx_demand, p]) for p in products}
    price = {p: float(df_36_1.at[idx_price, p]) for p in products}
    cost = {p: float(df_36_1.at[idx_cost, p]) for p in products}
    quota = {p: float(df_36_1.at[idx_quota, p]) for p in products}
    if df_36_2.shape[0] != 1:
        raise ValueError('36-2.csv should have exactly one row.')
    row_name_36_2 = df_36_2.at[0, 'Product'].strip().casefold()
    if row_name_36_2 != 'activation cost ($)'.casefold():
        raise ValueError("36-2.csv row must be 'Activation Cost ($)'")
    activation_cost = {p: float(df_36_2.at[0, p]) for p in products}
    if df_36_3.shape[0] != 1:
        raise ValueError('36-3.csv should have exactly one row.')
    row_name_36_3 = df_36_3.at[0, 'Product'].strip().casefold()
    if row_name_36_3 != 'minimum batch size (100 kg units)'.casefold():
        raise ValueError("36-3.csv row must be 'Minimum Batch Size (100 kg units)'")
    min_batch = {p: float(df_36_3.at[0, p]) for p in products}
    for p in products:
        for (param, d) in [('demand', demand), ('price', price), ('cost', cost), ('quota', quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
            if p not in d:
                raise KeyError(f'Missing {param} for product {p}')
    m = gp.Model('ProductionPlan80')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(products, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((price[p] - cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
    m.addConstrs((x_vars[p] <= 22 * quota[p] for p in products), name='')
    m.addConstrs((x_vars[p] >= min_batch[p] * y_vars[p] for p in products), name='')
    m.addConstrs((x_vars[p] <= demand[p] * y_vars[p] for p in products), name='')
    m.addConstr(gp.quicksum((x_vars[p] / quota[p] for p in products)) <= 22, name='shared_days')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.4f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')