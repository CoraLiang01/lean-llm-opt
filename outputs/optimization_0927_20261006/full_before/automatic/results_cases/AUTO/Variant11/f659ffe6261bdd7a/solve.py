import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',')
months = df['Month'].astype(str).tolist()
n_months = len(months)
demand = dict(zip(months, df['Demand'].astype(float)))
prod_cost = dict(zip(months, df['ProductionCost'].astype(float)))
setup_cost = dict(zip(months, df['SetupCost'].astype(float)))
hold_cost = dict(zip(months, df['HoldingCost'].astype(float)))
capacity = dict(zip(months, df['ProductionCapacity'].astype(float)))
for (colname, param) in [('Demand', demand), ('ProductionCost', prod_cost), ('SetupCost', setup_cost), ('HoldingCost', hold_cost), ('ProductionCapacity', capacity)]:
    if set(param.keys()) != set(months):
        raise ValueError(f'Missing {colname} data for some months.')
m = gp.Model('CapacitatedLotSizing')
x = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[t] * x[t] + setup_cost[t] * y[t] + hold_cost[t] * inv[t] for t in months)), gp.GRB.MINIMIZE)
for (idx, t) in enumerate(months):
    if idx == 0:
        inv_prev = 0.0
    else:
        inv_prev = inv[months[idx - 1]]
    m.addConstr(inv_prev + x[t] - demand[t] == inv[t], name=f'inv_bal_{t}')
for t in months:
    m.addConstr(x[t] <= capacity[t] * y[t], name=f'cap_link_{t}')
m.addConstr(inv[months[-1]] == 0.0, name='final_inv_zero')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for t in months:
        print(f'Month {t}:')
        print(f'  Production (x): {x[t].X:.2f}')
        print(f'  Setup (y): {int(round(y[t].X))}')
        print(f'  Ending Inventory (inv): {inv[t].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')