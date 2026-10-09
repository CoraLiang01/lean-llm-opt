import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
months = df['Month'].tolist()
if len(months) != 24:
    raise ValueError(f'Expected 24 months, found {len(months)}.')

def to_int_series(s):
    return s.astype(str).str.strip().astype(int)

def to_float_series(s):
    return s.astype(str).str.strip().astype(float)
demand = dict(zip(months, to_int_series(df['Demand'])))
production_cost = dict(zip(months, to_int_series(df['ProductionCost'])))
setup_cost = dict(zip(months, to_int_series(df['SetupCost'])))
holding_cost = dict(zip(months, to_float_series(df['HoldingCost'])))
production_capacity = dict(zip(months, to_int_series(df['ProductionCapacity'])))
m = gp.Model('MonthlyLotSizing')
x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
I_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + holding_cost[month] * I_vars[month] for month in months)), gp.GRB.MINIMIZE)
I0 = m.addVar(lb=0.0, ub=0.0, vtype=gp.GRB.CONTINUOUS, name='I0')
for (idx, month) in enumerate(months):
    if idx == 0:
        prev_I = I0
    else:
        prev_I = I_vars[months[idx - 1]]
    m.addConstr(I_vars[month] == prev_I + x_vars[month] - demand[month], name=f'inv_bal_{month}')
for month in months:
    m.addConstr(x_vars[month] <= production_capacity[month] * y_vars[month], name=f'prod_cap_{month}')
m.addConstr(I_vars[months[-1]] == 0.0, name='final_inventory_zero')
m.optimize()