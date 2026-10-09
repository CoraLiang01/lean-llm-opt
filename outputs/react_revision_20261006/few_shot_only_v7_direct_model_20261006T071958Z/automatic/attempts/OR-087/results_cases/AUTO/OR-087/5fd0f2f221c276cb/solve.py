import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
    path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
    df1 = pd.read_csv(path1, sep=',', dtype=str, keep_default_na=False)
    df2 = pd.read_csv(path2, sep=',', dtype=str, keep_default_na=False)
    df3 = pd.read_csv(path3, sep=',', dtype=str, keep_default_na=False)
    product_cols = [col for col in df1.columns if col != 'Product']
    products = product_cols.copy()

    def get_row(df, label):
        mask = df['Product'].str.casefold().str.strip() == label.casefold().strip()
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        mask = df['Product'].str.casefold().str.contains(label.casefold().strip())
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        raise KeyError(f"Row '{label}' not found uniquely in {df}")
    row_demand = get_row(df1, 'Maximum Demand (100 kg units)')
    row_price = get_row(df1, 'Selling Price ($/100 kg)')
    row_cost = get_row(df1, 'Production Cost ($/100 kg)')
    row_quota = get_row(df1, 'Production Quota (max per day)')
    row_activation = get_row(df2, 'Activation Cost ($)')
    row_minbatch = get_row(df3, 'Minimum Batch Size (100 kg units)')

    def to_float_dict(row):
        d = {}
        for p in products:
            val = row[p]
            try:
                d[p] = float(val)
            except Exception:
                raise ValueError(f"Non-numeric or missing value for product {p} in row '{row['Product']}'")
        return d
    max_demand = to_float_dict(row_demand)
    selling_price = to_float_dict(row_price)
    prod_cost = to_float_dict(row_cost)
    prod_quota = to_float_dict(row_quota)
    activation_cost = to_float_dict(row_activation)
    min_batch = to_float_dict(row_minbatch)
    for (d, name) in [(max_demand, 'Maximum Demand'), (selling_price, 'Selling Price'), (prod_cost, 'Production Cost'), (prod_quota, 'Production Quota'), (activation_cost, 'Activation Cost'), (min_batch, 'Minimum Batch Size')]:
        missing = set(products) - set(d)
        if missing:
            raise KeyError(f'Missing {name} data for products: {missing}')
    m = gp.Model('MonthlyProductionPlan')
    x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(products, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x_vars[p] <= max_demand[p] for p in products), name='')
    m.addConstrs((x_vars[p] <= 22 * prod_quota[p] for p in products), name='')
    m.addConstrs((x_vars[p] >= min_batch[p] * y_vars[p] for p in products), name='')
    m.addConstrs((x_vars[p] <= max_demand[p] * y_vars[p] for p in products), name='')
    m.addConstr(gp.quicksum((x_vars[p] / prod_quota[p] if prod_quota[p] > 0 else 0.0 for p in products)) <= 22, name='shared_days')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for p in products:
            print(f'{x_vars[p].VarName} {x_vars[p].X}')
            print(f'{y_vars[p].VarName} {y_vars[p].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()