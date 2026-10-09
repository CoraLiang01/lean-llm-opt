import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
truck_ids = df['truck_id'].tolist()
periods = [1, 2, 3, 4]

def to_float_dict(col):
    return {row['truck_id']: float(row[col]) for (_, row) in df.iterrows()}
Q_dict = to_float_dict('Q')
S_dict = to_float_dict('S')
C_dict = to_float_dict('C')
demand_cols = ['d1', 'd2', 'd3', 'd4']
demand = []
for (t, col) in enumerate(demand_cols, 1):
    val = df.iloc[0][col]
    demand.append(float(val))
demand_dict = {t: demand[t - 1] for t in periods}
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S_dict[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C_dict[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand_dict[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_capacity_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'truck_capacity_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_x_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_period_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        prev_y = y_vars[i, t - 1]
        curr_y = y_vars[i, t]
        if t + 1 in periods:
            m.addConstr(y_vars[i, t + 1] <= 1 - (prev_y - curr_y), name=f'min_down1_{i}_{t}')
        if t + 2 in periods:
            m.addConstr(y_vars[i, t + 2] <= 1 - (prev_y - curr_y), name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'zero_if_inactive_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_4_{i}')
m.optimize()