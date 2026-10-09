import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
    df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
    required_cols = {'item', 'value', 'weight'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f'Missing required columns in value.csv: {missing}')
    items = df['item'].tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate item identifiers found in value.csv.')
    try:
        value_dict = dict(zip(items, df['value'].astype(float)))
        weight_dict = dict(zip(items, df['weight'].astype(float)))
    except Exception as e:
        raise ValueError(f'Error converting value or weight columns to float: {e}')
    for i in items:
        if not np.isfinite(value_dict[i]):
            raise ValueError(f"Non-finite value for item {i} in 'value'.")
        if not np.isfinite(weight_dict[i]):
            raise ValueError(f"Non-finite value for item {i} in 'weight'.")
    m = gp.Model('KnapsackDisplay')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in items:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()