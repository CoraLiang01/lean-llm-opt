import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',')
months = list(df['Month'].astype(str))
n_months = len(months)
required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
    if df[col].isnull().any():
        raise ValueError(f'Missing values in column: {col}')
    if len(df[col]) != n_months:
        raise ValueError(f'Column {col} length {len(df[col])} does not match number of months {n_months}')
demand = dict(zip(months, df['Demand'].astype(float)))
prod_cost = dict(zip(months, df['ProductionCost'].astype(float)))
setup_cost = dict(zip(months, df['SetupCost'].astype(float)))
hold_cost = dict(zip(months, df['HoldingCost'].astype(float)))
prod_cap = dict(zip(months, df['ProductionCapacity'].astype(float)))

def solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap):
    m = gp.Model('CapacitatedLotSizing')
    x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    inv = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[t] * x[t] for t in months)) + gp.quicksum((setup_cost[t] * y[t] for t in months)) + gp.quicksum((hold_cost[t] * inv[t] for t in months)), gp.GRB.MINIMIZE)
    for (idx, t) in enumerate(months):
        if idx == 0:
            m.addConstr(x[t] - demand[t] == inv[t], name=f'invbal_{t}')
        else:
            prev = months[idx - 1]
            m.addConstr(inv[prev] + x[t] - demand[t] == inv[t], name=f'invbal_{t}')
    for t in months:
        m.addConstr(x[t] <= prod_cap[t] * y[t], name=f'caplink_{t}')
    m.addConstr(inv[months[-1]] == 0.0, name='final_inv_zero')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.4f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.4f}')
else:
    print(f'Solver status: {m.status}')