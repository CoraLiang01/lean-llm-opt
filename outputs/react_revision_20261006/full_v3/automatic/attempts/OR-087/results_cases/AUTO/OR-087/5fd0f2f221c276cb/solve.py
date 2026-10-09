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
    product_cols = [col for col in df1.columns if col != 'Product']
    products = product_cols.copy()
    df1.set_index('Product', inplace=True)
    required_rows_36_1 = ['Maximum Demand (100 kg units)', 'Selling Price ($/100 kg)', 'Production Cost ($/100 kg)', 'Production Quota (max per day)']
    for row in required_rows_36_1:
        if row not in df1.index:
            raise ValueError(f"Row '{row}' missing from 36-1.csv")
    max_demand = df1.loc['Maximum Demand (100 kg units)', products].astype(float).to_dict()
    price = df1.loc['Selling Price ($/100 kg)', products].astype(float).to_dict()
    cost = df1.loc['Production Cost ($/100 kg)', products].astype(float).to_dict()
    daily_quota = df1.loc['Production Quota (max per day)', products].astype(float).to_dict()
    if df2.shape[0] != 1:
        raise ValueError('36-2.csv must have exactly one row')
    activation_cost = df2.iloc[0, 1:].astype(float)
    activation_cost.index = product_cols
    activation_cost = activation_cost.to_dict()
    if df3.shape[0] != 1:
        raise ValueError('36-3.csv must have exactly one row')
    min_batch = df3.iloc[0, 1:].astype(float)
    min_batch.index = product_cols
    min_batch = min_batch.to_dict()
    for p in products:
        for (d, name) in [(max_demand, 'max_demand'), (price, 'price'), (cost, 'cost'), (daily_quota, 'daily_quota'), (activation_cost, 'activation_cost'), (min_batch, 'min_batch')]:
            if p not in d:
                raise ValueError(f'Missing {name} for product {p}')
    m = gp.Model('ProductionPlan80')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((price[p] - cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
    m.addConstrs((x[p] <= 22 * daily_quota[p] for p in products), name='')
    m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
    m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
    m.addConstr(gp.quicksum((x[p] / daily_quota[p] for p in products)) <= 22, name='shared_days')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.2f}')
        for p in products:
            print(f'{x[p].VarName}: {x[p].X:.0f}')
            print(f'{y[p].VarName}: {y[p].X:.0f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()