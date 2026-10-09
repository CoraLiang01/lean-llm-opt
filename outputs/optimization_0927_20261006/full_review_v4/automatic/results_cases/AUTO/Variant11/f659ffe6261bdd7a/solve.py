import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = df['Month'].tolist()
n_months = len(months)

def to_int_series(col):
    return df[col].astype(str).str.strip().astype(int)

def to_float_series(col):
    return df[col].astype(str).str.strip().astype(float)
demand = dict(zip(months, to_int_series('Demand')))
production_cost = dict(zip(months, to_int_series('ProductionCost')))
setup_cost = dict(zip(months, to_int_series('SetupCost')))
holding_cost = dict(zip(months, to_float_series('HoldingCost')))
production_capacity = dict(zip(months, to_int_series('ProductionCapacity')))
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + holding_cost[month] * inv_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(inv_vars[prev_month] + x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
for month in months:
    m.addConstr(x_vars[month] <= production_capacity[month] * y_vars[month], name=f'capacity_link_{month}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='ending_inventory_zero')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for month in months:
        print(f'{month}:')
        print(f'  Production (x): {x_vars[month].X:.2f}')
        print(f'  Setup (y): {int(round(y_vars[month].X))}')
        print(f'  Ending Inventory (inv): {inv_vars[month].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')