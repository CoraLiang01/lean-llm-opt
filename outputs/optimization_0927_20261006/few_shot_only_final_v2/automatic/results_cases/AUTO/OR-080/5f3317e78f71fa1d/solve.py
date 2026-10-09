import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
params_df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in params_df.columns:
    raise KeyError("Missing required column 'truck_id' in parameters.csv")
truck_ids = params_df['truck_id'].astype(str).tolist()
n_trucks = len(truck_ids)

def get_numeric_col(df, col, key='truck_id'):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in parameters.csv")
    try:
        return pd.Series(df[col].astype(float).values, index=df[key].astype(str))
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
Q_series = get_numeric_col(params_df, 'Q')
S_series = get_numeric_col(params_df, 'S')
C_series = get_numeric_col(params_df, 'C')
periods = [1, 2, 3, 4]
demand = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
for (t, dcol) in zip(periods, ['d1', 'd2', 'd3', 'd4']):
    if dcol in params_df.columns:
        dval = params_df[dcol].astype(float).sum()
I = truck_ids
T = periods
Q = Q_series.to_dict()
S = S_series.to_dict()
C = C_series.to_dict()

def build_model():
    m = gp.Model('TruckScheduling')
    x_vars = m.addVars(I, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, T, vtype=gp.GRB.BINARY, name='')
    s_vars = m.addVars(I, T, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * s_vars[i, t] for i in I for t in T)) + gp.quicksum((C[i] * x_vars[i, t] for i in I for t in T)), gp.GRB.MINIMIZE)
    for t in T:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) >= demand[t], name=f'demand_{t}')
    for t in T:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in I)), name=f'spare_cap_{t}')
    for i in I:
        for t in T:
            m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
    for i in I:
        m.addConstr(s_vars[i, 1] == y_vars[i, 1], name=f'startup_init_{i}')
        for t in [2, 3, 4]:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_logic_{i}_{t}')
        m.addConstr(s_vars[i, 4] == 0, name=f'startup_forbid_{i}_4')
    for i in I:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[i, t] + y_vars[i, t + 1] >= 2 * s_vars[i, t], name=f'min_up_{i}_{t}')
    for i in I:
        y_prev = 0
        for t in [1, 2, 3]:
            m.addConstr(y_prev - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'min_down_{i}_{t}')
            y_prev = y_vars[i, t]
    for i in I:
        x_prev = 0.0
        for t in [1, 2, 3, 4]:
            if t == 1:
                m.addConstr(x_vars[i, 1] <= 300, name=f'ramp_up_{i}_1')
                m.addConstr(x_vars[i, 1] >= 0, name=f'ramp_down_{i}_1')
            else:
                m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
                m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
    for i in I:
        for t in T:
            m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
    return m
m = build_model()
m.optimize()