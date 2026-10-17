import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Shift' not in df.columns or 'Number Required' not in df.columns:
    raise ValueError('Required columns missing from CSV.')
df['Shift'] = df['Shift'].astype(int)
df['Number Required'] = df['Number Required'].astype(int)
T = sorted(df['Shift'].unique())
if len(T) != 24 or min(T) != 1 or max(T) != 24:
    raise ValueError('Expected 24 hourly periods indexed 1..24.')
required_staff = dict(zip(df['Shift'], df['Number Required']))
shift_length = 4
num_periods = 24

def covered_periods(t):
    return [(t + i - 1) % num_periods + 1 for i in range(shift_length)]
covering_shifts = {s: [] for s in T}
for t in T:
    for s in covered_periods(t):
        covering_shifts[s].append(t)
m = Model('bus_shift_scheduling')
m.Params.OutputFlag = 0
x_vars = m.addVars(T, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((x_vars[t] for t in T)), GRB.MINIMIZE)
for s in T:
    m.addConstr(quicksum((x_vars[t] for t in covering_shifts[s])) >= required_staff[s], name=f'cover_{s}')
m.optimize()