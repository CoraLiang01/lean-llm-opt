import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 time periods, got {df.shape[0]} rows in 44.csv')
periods = list(range(48))
try:
    requirement_col = [col for col in df.columns if re.fullmatch('Requirement', col, re.IGNORECASE)][0]
except IndexError:
    raise KeyError("Could not find 'Requirement' column in 44.csv")
requirements = {}
for (idx, row) in df.iterrows():
    try:
        t = idx
        req = int(row[requirement_col])
        requirements[t] = req
    except Exception as e:
        raise ValueError(f'Error parsing requirement at row {idx}: {e}')
shift_starts = list(range(48))
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in shift_starts if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()