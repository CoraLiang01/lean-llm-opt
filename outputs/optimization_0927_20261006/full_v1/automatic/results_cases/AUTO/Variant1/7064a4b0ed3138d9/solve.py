import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = list(df['Month'])
if len(months) != 24:
    raise ValueError(f'Expected 24 months, got {len(months)}')

def to_int_col(col):
    return {row['Month']: int(row[col]) for (_, row) in df.iterrows()}

def to_float_col(col):
    return {row['Month']: float(row[col]) for (_, row) in df.iterrows()}
demand = to_int_col('Demand')
production_cost = to_int_col('ProductionCost')
setup_cost = to_int_col('SetupCost')
holding_cost = to_float_col('HoldingCost')
production_capacity = to_int_col('ProductionCapacity')
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
I_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + holding_cost[month] * I_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[month] - demand[month] == I_vars[month], name=f'inv_bal_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(I_vars[prev_month] + x_vars[month] - demand[month] == I_vars[month], name=f'inv_bal_{month}')
for month in months:
    m.addConstr(x_vars[month] <= production_capacity[month] * y_vars[month], name=f'prod_cap_{month}')
last_month = months[-1]
m.addConstr(I_vars[last_month] == 0, name='final_inventory_zero')
m.optimize()