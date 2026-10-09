import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = df['Month'].astype(str).tolist()
demand = df['Demand'].astype(float).to_dict()
production_cost = df['ProductionCost'].astype(float).to_dict()
setup_cost = df['SetupCost'].astype(float).to_dict()
holding_cost = df['HoldingCost'].astype(float).to_dict()
production_capacity = df['ProductionCapacity'].astype(float).to_dict()
for m in months:
    if m not in demand or m not in production_cost or m not in setup_cost or (m not in holding_cost) or (m not in production_capacity):
        raise ValueError(f'Missing parameter data for month {m}')
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[mth] * x_vars[mth] + setup_cost[mth] * y_vars[mth] + holding_cost[mth] * inv_vars[mth] for mth in months)), gp.GRB.MINIMIZE)
for (idx, mth) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_balance_{mth}')
    else:
        prev_mth = months[idx - 1]
        m.addConstr(inv_vars[prev_mth] + x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_balance_{mth}')
for mth in months:
    m.addConstr(x_vars[mth] <= production_capacity[mth] * y_vars[mth], name=f'prod_cap_link_{mth}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='zero_ending_inventory')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for mth in months:
        print(f'Month {mth}:')
        print(f'  Production (x): {x_vars[mth].X:.2f}')
        print(f'  Setup (y): {int(round(y_vars[mth].X))}')
        print(f'  Ending Inventory (inv): {inv_vars[mth].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')