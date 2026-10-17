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
required_columns = ['Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
demand = {}
prod_cost = {}
setup_cost = {}
hold_cost = {}
prod_cap = {}
for idx, row in df.iterrows():
    month = row['Month']
    try:
        demand[month] = int(row['Demand'])
        prod_cost[month] = float(row['ProductionCost'])
        setup_cost[month] = float(row['SetupCost'])
        hold_cost[month] = float(row['HoldingCost'])
        prod_cap[month] = float(row['ProductionCapacity'])
    except Exception as e:
        raise ValueError(f"Error converting parameters for month '{month}': {e}")
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + hold_cost[month] * inv_vars[month] for month in months)), gp.GRB.MINIMIZE)
for i, month in enumerate(months):
    if i == 0:
        m.addConstr(x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
    else:
        prev_month = months[i - 1]
        m.addConstr(inv_vars[prev_month] + x_vars[month] - demand[month] == inv_vars[month], name=f'inv_balance_{month}')
for month in months:
    m.addConstr(x_vars[month] <= prod_cap[month] * y_vars[month], name=f'prod_cap_link_{month}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='ending_inventory_zero')
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