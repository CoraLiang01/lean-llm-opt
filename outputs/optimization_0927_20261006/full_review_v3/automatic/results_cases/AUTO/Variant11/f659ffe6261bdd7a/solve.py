import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Month'] = df['Month'].astype(str).str.strip()
months = list(df['Month'])
for col in ['Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']:
    df[col] = pd.to_numeric(df[col], errors='raise')
demand = df.set_index('Month')['Demand'].to_dict()
prod_cost = df.set_index('Month')['ProductionCost'].to_dict()
setup_cost = df.set_index('Month')['SetupCost'].to_dict()
hold_cost = df.set_index('Month')['HoldingCost'].to_dict()
prod_cap = df.set_index('Month')['ProductionCapacity'].to_dict()
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[mon] * x_vars[mon] + setup_cost[mon] * y_vars[mon] + hold_cost[mon] * inv_vars[mon] for mon in months)), gp.GRB.MINIMIZE)
for (idx, mon) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[mon] - demand[mon] == inv_vars[mon], name=f'inv_balance_{mon}')
    else:
        prev_mon = months[idx - 1]
        m.addConstr(inv_vars[prev_mon] + x_vars[mon] - demand[mon] == inv_vars[mon], name=f'inv_balance_{mon}')
for mon in months:
    m.addConstr(x_vars[mon] <= prod_cap[mon] * y_vars[mon], name=f'prod_cap_link_{mon}')
last_mon = months[-1]
m.addConstr(inv_vars[last_mon] == 0, name='zero_ending_inventory')
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