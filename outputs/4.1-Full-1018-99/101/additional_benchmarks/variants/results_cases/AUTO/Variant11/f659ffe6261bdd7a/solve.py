import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = df['Month'].tolist()
demand = df.set_index('Month')['Demand'].astype(float).to_dict()
prod_cost = df.set_index('Month')['ProductionCost'].astype(float).to_dict()
setup_cost = df.set_index('Month')['SetupCost'].astype(float).to_dict()
hold_cost = df.set_index('Month')['HoldingCost'].astype(float).to_dict()
prod_cap = df.set_index('Month')['ProductionCapacity'].astype(float).to_dict()
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[t] * x_vars[t] + setup_cost[t] * y_vars[t] + hold_cost[t] * inv_vars[t] for t in months)), gp.GRB.MINIMIZE)
for i, t in enumerate(months):
    if i == 0:
        inv_prev = 0.0
    else:
        inv_prev = inv_vars[months[i - 1]]
    m.addConstr(inv_prev + x_vars[t] - demand[t] == inv_vars[t], name=f'inv_balance_{t}')
for t in months:
    m.addConstr(x_vars[t] <= prod_cap[t] * y_vars[t], name=f'capacity_link_{t}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0.0, name='final_inventory_zero')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for t in months:
        print(f'Month {t}:')
        print(f'  Production (x): {x_vars[t].X:.2f}')
        print(f'  Setup (y): {int(round(y_vars[t].X))}')
        print(f'  Ending Inventory (inv): {inv_vars[t].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')