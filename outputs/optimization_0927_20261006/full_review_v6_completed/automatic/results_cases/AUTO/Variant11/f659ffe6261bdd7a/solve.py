import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Month' not in df.columns:
    raise KeyError("Required column 'Month' not found in CSV.")
months = df['Month'].tolist()
n_months = len(months)

def to_int_series(col):
    return df[col].astype(int).to_dict()

def to_float_series(col):
    return df[col].astype(float).to_dict()
demand = to_int_series('Demand')
production_cost = to_int_series('ProductionCost')
setup_cost = to_int_series('SetupCost')
holding_cost = to_float_series('HoldingCost')
production_capacity = to_int_series('ProductionCapacity')
for m in months:
    for (param, d) in [('Demand', demand), ('ProductionCost', production_cost), ('SetupCost', setup_cost), ('HoldingCost', holding_cost), ('ProductionCapacity', production_capacity)]:
        if m not in d:
            raise KeyError(f"Month '{m}' missing parameter '{param}'.")
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[mon] * x_vars[mon] + setup_cost[mon] * y_vars[mon] + holding_cost[mon] * inv_vars[mon] for mon in months)), gp.GRB.MINIMIZE)
first_month = months[0]
m.addConstr(x_vars[first_month] - demand[first_month] == inv_vars[first_month], name='inv_balance_0')
for idx in range(1, n_months):
    prev_month = months[idx - 1]
    curr_month = months[idx]
    m.addConstr(inv_vars[prev_month] + x_vars[curr_month] - demand[curr_month] == inv_vars[curr_month], name=f'inv_balance_{idx}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='ending_inventory_zero')
for mon in months:
    m.addConstr(x_vars[mon] <= production_capacity[mon] * y_vars[mon], name=f'capacity_link_{mon}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for mon in months:
        print(f'Month {mon}:')
        print(f'  Production (x): {x_vars[mon].X:.2f}')
        print(f'  Setup (y): {int(round(y_vars[mon].X))}')
        print(f'  Ending Inventory (inv): {inv_vars[mon].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')