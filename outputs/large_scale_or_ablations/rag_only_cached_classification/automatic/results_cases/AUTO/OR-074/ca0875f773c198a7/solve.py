import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
shift_length_periods = 16
requirements = df['Requirement'].astype(int).to_dict()
shift_starts = periods
coverage = {t: [] for t in periods}
for s in shift_starts:
    for i in range(shift_length_periods):
        t = (s + i) % len(periods)
        coverage[t].append(s)
if set(requirements.keys()) != set(periods):
    raise ValueError('Mismatch between periods and requirements in CSV.')
for t in periods:
    if len(coverage[t]) == 0:
        raise ValueError(f'Period {t} is not covered by any shift.')
m = Model('Waitstaff_Scheduling')
x = m.addVars(shift_starts, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(x.sum(), GRB.MINIMIZE)
for t in periods:
    m.addConstr(sum((x[s] for s in coverage[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()