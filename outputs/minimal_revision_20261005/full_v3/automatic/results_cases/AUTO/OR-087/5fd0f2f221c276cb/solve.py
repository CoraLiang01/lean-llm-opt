import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'

def solve_problem():
    df1 = pd.read_csv(file_36_1, sep=',')
    df2 = pd.read_csv(file_36_2, sep=',')
    df3 = pd.read_csv(file_36_3, sep=',')
    product_cols = [col for col in df1.columns if col.startswith('A')]
    products = product_cols.copy()
    row_map = {}
    for (idx, row) in df1.iterrows():
        label = str(row['Product']).strip().casefold()
        row_map[label] = idx
    demand_row = None
    price_row = None
    cost_row = None
    quota_row = None
    for (k, idx) in row_map.items():
        if 'maximum demand' in k:
            demand_row = idx
        elif 'selling price' in k:
            price_row = idx
        elif 'production cost' in k:
            cost_row = idx
        elif 'production quota' in k:
            quota_row = idx
    if None in [demand_row, price_row, cost_row, quota_row]:
        raise ValueError('Could not find all required parameter rows in 36-1.csv')
    max_demand = {p: float(df1.at[demand_row, p]) for p in products}
    selling_price = {p: float(df1.at[price_row, p]) for p in products}
    prod_cost = {p: float(df1.at[cost_row, p]) for p in products}
    daily_quota = {p: float(df1.at[quota_row, p]) for p in products}
    if df2.shape[0] != 1:
        raise ValueError('36-2.csv should have exactly one row')
    act_row_label = str(df2.at[0, 'Product']).strip().casefold()
    if 'activation cost' not in act_row_label:
        raise ValueError("36-2.csv row label does not match 'Activation Cost'")
    activation_cost = {p: float(df2.at[0, p]) for p in products}
    if df3.shape[0] != 1:
        raise ValueError('36-3.csv should have exactly one row')
    batch_row_label = str(df3.at[0, 'Product']).strip().casefold()
    if 'minimum batch size' not in batch_row_label:
        raise ValueError("36-3.csv row label does not match 'Minimum Batch Size'")
    min_batch = {p: int(df3.at[0, p]) for p in products}
    for p in products:
        for d in [max_demand, selling_price, prod_cost, daily_quota, activation_cost, min_batch]:
            if p not in d:
                raise ValueError(f'Missing parameter for product {p}')
    m = gp.Model('ProductionPlan80')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
    m.addConstrs((x[p] <= 22 * daily_quota[p] for p in products), name='')
    m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
    m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
    m.addConstr(gp.quicksum((x[p] / daily_quota[p] for p in products)) <= 22, name='total_days')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.4f}')
        for p in products:
            print(f'{x[p].VarName} {x[p].X}')
            print(f'{y[p].VarName} {y[p].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()