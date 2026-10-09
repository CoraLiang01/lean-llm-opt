import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Month'] = df['Month'].str.strip()
months = list(df['Month'])
demand = df['Demand'].astype(float).to_dict()
production_cost = df['ProductionCost'].astype(float).to_dict()
setup_cost = df['SetupCost'].astype(float).to_dict()
holding_cost = df['HoldingCost'].astype(float).to_dict()
production_capacity = df['ProductionCapacity'].astype(float).to_dict()
for (colname, col) in [('Demand', demand), ('ProductionCost', production_cost), ('SetupCost', setup_cost), ('HoldingCost', holding_cost), ('ProductionCapacity', production_capacity)]:
    missing = [m for m in months if m not in col]
    if missing:
        raise ValueError(f'Missing {colname} data for months: {missing}')
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + holding_cost[month] * inv_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    demand_t = demand[month]
    if idx == 0:
        m.addConstr(x_vars[month] - demand_t == inv_vars[month], name=f'inv_balance_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(inv_vars[prev_month] + x_vars[month] - demand_t == inv_vars[month], name=f'inv_balance_{month}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='zero_ending_inventory')
for month in months:
    m.addConstr(x_vars[month] <= production_capacity[month] * y_vars[month], name=f'capacity_link_{month}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for month in months:
        print(f'Month {month}:')
        print(f'  Production (x): {x_vars[month].X:.2f}')
        print(f'  Setup (y): {int(round(y_vars[month].X))}')
        print(f'  Ending Inventory (inv): {inv_vars[month].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')