import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
    path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
    path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
    df1 = pd.read_csv(path_36_1, sep=',')
    df2 = pd.read_csv(path_36_2, sep=',')
    df3 = pd.read_csv(path_36_3, sep=',')
    product_cols = [f'A{i}' for i in range(1, 81)]
    products = product_cols.copy()

    def get_row_by_label(df, label):
        mask = df['Product'].str.casefold().str.replace('\\s+', '', regex=True) == label.casefold().replace(' ', '')
        matches = df[mask]
        if matches.shape[0] != 1:
            raise ValueError(f"Row label '{label}' not found uniquely in 36-1.csv")
        return matches.iloc[0][product_cols]
    max_demand = get_row_by_label(df1, 'Maximum Demand (100 kg units)').astype(float).to_dict()
    selling_price = get_row_by_label(df1, 'Selling Price ($/100 kg)').astype(float).to_dict()
    prod_cost = get_row_by_label(df1, 'Production Cost ($/100 kg)').astype(float).to_dict()
    prod_quota = get_row_by_label(df1, 'Production Quota (max per day)').astype(float).to_dict()
    if df2.shape[0] != 1:
        raise ValueError('36-2.csv must have exactly one row')
    activation_cost = df2.iloc[0][product_cols].astype(float).to_dict()
    if df3.shape[0] != 1:
        raise ValueError('36-3.csv must have exactly one row')
    min_batch = df3.iloc[0][product_cols].astype(float).to_dict()
    for p in products:
        for (param, d) in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
            if p not in d:
                raise ValueError(f'Missing {param} for product {p}')
    num_days = 22
    m = gp.Model('production_plan')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    y = m.addVars(products, vtype=GRB.BINARY, name='')
    expr = gp.LinExpr()
    for p in products:
        expr += (selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p]
    m.setObjective(expr, GRB.MAXIMIZE)
    for p in products:
        m.addConstr(x[p] <= max_demand[p], name='demand_' + p)
        m.addConstr(x[p] <= prod_quota[p] * num_days, name='capacity_' + p)
        m.addConstr(x[p] >= min_batch[p] * y[p], name='minbatch_' + p)
        m.addConstr(x[p] <= prod_quota[p] * num_days * y[p], name='link_' + p)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for p in products:
            print(f'x[{p}] {x[p].X}')
        for p in products:
            print(f'y[{p}] {y[p].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()