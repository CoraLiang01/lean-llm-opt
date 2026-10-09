import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in 44.csv")
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
try:
    requirements = {t: int(df.loc[t, 'Requirement']) for t in periods}
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' to int: {e}")
shift_starts = periods
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_shifts = [s for s in shift_starts if (t - s) % 48 < shift_length]
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()