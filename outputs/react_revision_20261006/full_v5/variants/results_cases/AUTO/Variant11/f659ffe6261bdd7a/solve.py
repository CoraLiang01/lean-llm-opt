import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',')
months = df['Month'].astype(str).tolist()
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
m = gp.Model('CapacitatedLotSizing')
x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
inv = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[t] * x[t] for t in months)) + gp.quicksum((setup_cost[t] * y[t] for t in months)) + gp.quicksum((hold_cost[t] * inv[t] for t in months)), gp.GRB.MINIMIZE)
for (idx, t) in enumerate(months):
    if idx == 0:
        m.addConstr(x[t] - demand[t] == inv[t], name=f'inv_bal_{t}')
    else:
        prev_t = months[idx - 1]
        m.addConstr(inv[prev_t] + x[t] - demand[t] == inv[t], name=f'inv_bal_{t}')
for t in months:
    m.addConstr(x[t] <= prod_cap[t] * y[t], name=f'cap_link_{t}')
m.addConstr(inv[months[-1]] == 0.0, name='final_inventory_zero')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.2f}')
    for t in months:
        print(f'x[{t}] {x[t].VarName} {x[t].X:.4f}')
    for t in months:
        print(f'inv[{t}] {inv[t].VarName} {inv[t].X:.4f}')
    for t in months:
        print(f'y[{t}] {y[t].VarName} {y[t].X:.0f}')
else:
    print(f'Solver status: {m.status}')