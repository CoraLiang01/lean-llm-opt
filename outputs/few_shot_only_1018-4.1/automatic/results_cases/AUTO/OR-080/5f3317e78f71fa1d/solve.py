import gurobipy as gp
import pandas as pd
import numpy as np
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing 'truck_id' column in parameters.csv")
truck_ids = df['truck_id'].astype(str).tolist()
periods = [1, 2, 3, 4]

def get_numeric_col(col):
    if col not in df.columns:
        raise KeyError(f"Missing '{col}' column in parameters.csv")
    vals = pd.to_numeric(df[col], errors='raise')
    if len(vals) != len(truck_ids):
        raise ValueError(f"Length mismatch in column '{col}'")
    return dict(zip(truck_ids, vals))
Q = get_numeric_col('Q')
S = get_numeric_col('S')
C = get_numeric_col('C')
for dcol, t in zip(['d1', 'd2', 'd3', 'd4'], periods):
    if dcol not in df.columns:
        raise KeyError(f"Missing '{dcol}' column in parameters.csv")
demand_vals = [float(df[dcol].iloc[0]) for dcol in ['d1', 'd2', 'd3', 'd4']]
demand = dict(zip(periods, demand_vals))
m = gp.Model('TruckTransportSchedule')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_start_4_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(y_vars[i, t] + y_vars[i, t + 1] <= 1 - (prev_y - y_vars[i, t]), name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        prev_x = 0 if t == 2 else x_vars[i, t - 1]
        m.addConstr(x_vars[i, t] - (x_vars[i, t - 1] if t > 1 else 0) <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr((x_vars[i, t - 1] if t > 1 else 0) - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()