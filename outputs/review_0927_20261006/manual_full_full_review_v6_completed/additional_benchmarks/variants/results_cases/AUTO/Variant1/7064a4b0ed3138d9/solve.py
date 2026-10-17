import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
months = list(df['Month'])

def to_int_series(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to int: {e}")

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
demand_dict = dict(zip(months, to_int_series(df['Demand'], 'Demand')))
prod_cost_dict = dict(zip(months, to_int_series(df['ProductionCost'], 'ProductionCost')))
setup_cost_dict = dict(zip(months, to_int_series(df['SetupCost'], 'SetupCost')))
hold_cost_dict = dict(zip(months, to_float_series(df['HoldingCost'], 'HoldingCost')))
prod_cap_dict = dict(zip(months, to_int_series(df['ProductionCapacity'], 'ProductionCapacity')))
m = gp.Model('MonthlyLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
I_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost_dict[month] * x_vars[month] + setup_cost_dict[month] * y_vars[month] + hold_cost_dict[month] * I_vars[month] for month in months)), gp.GRB.MINIMIZE)
for idx, month in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[month] - demand_dict[month] == I_vars[month], name=f'inv_bal_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(I_vars[prev_month] + x_vars[month] - demand_dict[month] == I_vars[month], name=f'inv_bal_{month}')
last_month = months[-1]
m.addConstr(I_vars[last_month] == 0, name='final_inventory_zero')
for month in months:
    m.addConstr(x_vars[month] <= prod_cap_dict[month] * y_vars[month], name=f'prod_cap_{month}')
m.optimize()