import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing required column 'truck_id' in parameters.csv")
trucks = df['truck_id'].tolist()
if len(trucks) != 10:
    raise ValueError(f'Expected 10 trucks, found {len(trucks)} in parameters.csv')

def get_numeric_col(colname):
    if colname not in df.columns:
        raise KeyError(f"Missing required column '{colname}' in parameters.csv")
    try:
        return df.set_index('truck_id')[colname].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
Q_dict = get_numeric_col('Q')
S_dict = get_numeric_col('S')
C_dict = get_numeric_col('C')
periods = [1, 2, 3, 4]
demand_cols = [f'd{t}' for t in periods]
for col in demand_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required demand column '{col}' in parameters.csv")
demand_dict = {}
for t, col in zip(periods, demand_cols):
    try:
        demand_dict[t] = float(df.iloc[0][col])
    except Exception as e:
        raise ValueError(f"Failed to convert demand column '{col}' to float: {e}")
m = gp.Model('TruckScheduling')
y_vars = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(trucks, periods, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((S_dict[i] * s_vars[i, t] for i in trucks for t in periods)) + gp.quicksum((C_dict[i] * x_vars[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in trucks)) >= demand_dict[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in trucks)), name=f'sparecap_{t}')
for i in trucks:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in trucks:
    prev_y = 0
    for t in periods:
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_logic_{i}_{t}')
        prev_y = y_vars[i, t]
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'minup_{i}_{t}')
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t] + y_vars[i, t + 1] <= 1 - (y_vars[i, t - 1] if t > 1 else 0 - y_vars[i, t]), name=f'mindown_{i}_{t}')
for i in trucks:
    prev_x = 0
    for t in periods:
        if t == 1:
            prev_x = 0
        else:
            m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
        prev_x = x_vars[i, t]
m.optimize()