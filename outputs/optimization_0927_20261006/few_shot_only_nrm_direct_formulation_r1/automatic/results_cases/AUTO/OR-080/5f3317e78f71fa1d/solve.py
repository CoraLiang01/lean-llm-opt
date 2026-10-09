import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing 'truck_id' column in parameters.csv")
truck_ids = df['truck_id'].tolist()
n_trucks = len(truck_ids)
periods = [1, 2, 3, 4]

def to_float_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing '{col}' column in parameters.csv")
    try:
        return df[col].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Could not convert column '{col}' to float: {e}")
Q_dict = to_float_col(df.set_index('truck_id'), 'Q')
S_dict = to_float_col(df.set_index('truck_id'), 'S')
C_dict = to_float_col(df.set_index('truck_id'), 'C')
for dcol in ['d1', 'd2', 'd3', 'd4']:
    if dcol not in df.columns:
        raise KeyError(f"Missing '{dcol}' column in parameters.csv")
demand = {}
for (t, dcol) in zip(periods, ['d1', 'd2', 'd3', 'd4']):
    try:
        demand[t] = float(df.iloc[0][dcol])
    except Exception as e:
        raise ValueError(f"Could not convert demand column '{dcol}' to float: {e}")
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S_dict[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C_dict[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_capacity_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'truck_cap_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - 0, name=f'startup_def_{i}_{t}')
        else:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [1, 2]:
        if t + 1 in periods:
            m.addConstr(y_vars[i, t + 1] <= 1 - (y_vars[i, t - 1] - y_vars[i, t]), name=f'min_down1_{i}_{t}')
        if t + 2 in periods:
            m.addConstr(y_vars[i, t + 2] <= 1 - (y_vars[i, t - 1] - y_vars[i, t]), name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Truck Schedule ---')
    for i in truck_ids:
        print(f'Truck {i}:')
        for t in periods:
            y_val = int(round(y_vars[i, t].X))
            s_val = int(round(s_vars[i, t].X))
            x_val = x_vars[i, t].X
            print(f'  Period {t}: Active={y_val}, Startup={s_val}, Transported={x_val:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')