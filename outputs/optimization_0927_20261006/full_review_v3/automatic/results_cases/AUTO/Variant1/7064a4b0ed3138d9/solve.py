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
for m in months:
    for (param, dct) in [('Demand', demand), ('ProductionCost', production_cost), ('SetupCost', setup_cost), ('HoldingCost', holding_cost), ('ProductionCapacity', production_capacity)]:
        if m not in dct:
            raise KeyError(f"Month '{m}' missing parameter '{param}'.")
m = gp.Model('MonthlyLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
I_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[mth] * x_vars[mth] + setup_cost[mth] * y_vars[mth] + holding_cost[mth] * I_vars[mth] for mth in months)), gp.GRB.MINIMIZE)
for (idx, mth) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[mth] - demand[mth] == I_vars[mth], name=f'inv_bal_{mth}')
    else:
        prev_mth = months[idx - 1]
        m.addConstr(I_vars[prev_mth] + x_vars[mth] - demand[mth] == I_vars[mth], name=f'inv_bal_{mth}')
for mth in months:
    m.addConstr(x_vars[mth] <= production_capacity[mth] * y_vars[mth], name=f'prod_cap_{mth}')
last_month = months[-1]
m.addConstr(I_vars[last_month] == 0, name='final_inventory_zero')
m.optimize()