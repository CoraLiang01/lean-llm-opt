import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = 'benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df['Month'] = df['Month'].astype(str).str.strip()
months = list(df['Month'])
if len(months) != 12:
    raise ValueError(f'Expected 12 months, got {len(months)} in {csv_path}')
for col in ['Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in {csv_path}")
df['Demand'] = df['Demand'].astype(float)
df['ProductionCost'] = df['ProductionCost'].astype(float)
df['SetupCost'] = df['SetupCost'].astype(float)
df['HoldingCost'] = df['HoldingCost'].astype(float)
df['ProductionCapacity'] = df['ProductionCapacity'].astype(float)
demand = dict(zip(months, df['Demand']))
prod_cost = dict(zip(months, df['ProductionCost']))
setup_cost = dict(zip(months, df['SetupCost']))
hold_cost = dict(zip(months, df['HoldingCost']))
capacity = dict(zip(months, df['ProductionCapacity']))
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
inv_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + hold_cost[month] * inv_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(inv_vars[prev_month] + x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
for month in months:
    m.addConstr(x_vars[month] <= capacity[month] * y_vars[month], name=f'capacity_link_{month}')
m.addConstr(inv_vars[months[-1]] == 0.0, name='ending_inventory_zero')
m.optimize()