import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = 'benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
months = df['Month'].tolist()
n_months = len(months)
if n_months != 24:
    raise ValueError(f'Expected 24 months, found {n_months}.')

def to_int_series(s, colname):
    try:
        return s.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")

def to_float_series(s, colname):
    try:
        return s.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")
demand = dict(zip(months, to_int_series(df['Demand'], 'Demand')))
prod_cost = dict(zip(months, to_int_series(df['ProductionCost'], 'ProductionCost')))
setup_cost = dict(zip(months, to_int_series(df['SetupCost'], 'SetupCost')))
hold_cost = dict(zip(months, to_float_series(df['HoldingCost'], 'HoldingCost')))
prod_cap = dict(zip(months, to_int_series(df['ProductionCapacity'], 'ProductionCapacity')))
m = gp.Model('LotSizing')
x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
I_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((prod_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + hold_cost[month] * I_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[month] - demand[month] == I_vars[month], name=f'inv_bal_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(I_vars[prev_month] + x_vars[month] - demand[month] == I_vars[month], name=f'inv_bal_{month}')
for month in months:
    m.addConstr(x_vars[month] <= prod_cap[month] * y_vars[month], name=f'prod_cap_{month}')
last_month = months[-1]
m.addConstr(I_vars[last_month] == 0, name='final_inventory_zero')
m.optimize()