import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
if 'Shift' not in df.columns or 'Number Required' not in df.columns:
    raise ValueError('Required columns missing from CSV.')
df['Shift'] = df['Shift'].astype(int)
df['Number Required'] = df['Number Required'].astype(int)
periods = df['Shift'].tolist()
if sorted(periods) != list(range(1, 25)):
    raise ValueError('Expected 24 consecutive periods indexed 1..24.')
r_t = dict(zip(df['Shift'], df['Number Required']))
shift_length = 4
num_periods = 24
m = Model('bus_shift_scheduling')
x_vars = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((x_vars[t] for t in periods)), GRB.MINIMIZE)
for s in periods:
    covering_starts = []
    for t in periods:
        covered = [(t - 1 + offset) % num_periods + 1 for offset in range(shift_length)]
        if s in covered:
            covering_starts.append(t)
    if not covering_starts:
        raise ValueError(f'No shift covers period {s}')
    m.addConstr(quicksum((x_vars[t] for t in covering_starts)) >= r_t[s], name=f'cover_{s}')
m.optimize()