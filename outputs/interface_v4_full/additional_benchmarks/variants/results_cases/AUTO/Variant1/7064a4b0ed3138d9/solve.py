import gurobipy as gp
import pandas as pd
import numpy as np

def solve_lot_sizing():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
    df = pd.read_csv(path, sep=',')
    required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")
    months = list(df['Month'].astype(str))
    n_months = len(months)
    if n_months != 24:
        raise ValueError(f'Expected 24 months, got {n_months}.')
    demand = dict(zip(months, df['Demand'].astype(float)))
    prod_cost = dict(zip(months, df['ProductionCost'].astype(float)))
    setup_cost = dict(zip(months, df['SetupCost'].astype(float)))
    hold_cost = dict(zip(months, df['HoldingCost'].astype(float)))
    prod_cap = dict(zip(months, df['ProductionCapacity'].astype(float)))
    m = gp.Model('LotSizing')
    x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    I_keys = [f'I{i}' for i in range(n_months + 1)]
    I = m.addVars(I_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.addConstr(I['I0'] == 0, name='InitialInventory')
    for idx, month in enumerate(months):
        prev_I = f'I{idx}'
        curr_I = f'I{idx + 1}'
        m.addConstr(I[prev_I] + x[month] == demand[month] + I[curr_I], name=f'InvBal_{month}')
    m.addConstr(I[f'I{n_months}'] == 0, name='FinalInventory')
    for month in months:
        m.addConstr(x[month] <= prod_cap[month] * y[month], name=f'ProdCap_{month}')
    obj = gp.quicksum((prod_cost[month] * x[month] + setup_cost[month] * y[month] + hold_cost[month] * I[f'I{idx + 1}'] for idx, month in enumerate(months)))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_lot_sizing()