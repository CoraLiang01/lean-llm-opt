import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not set(['Shift', 'Number Required']).issubset(df.columns):
    raise KeyError("Required columns 'Shift' and 'Number Required' not found in CSV.")
df['Shift'] = df['Shift'].astype(int)
periods = df['Shift'].tolist()
required_dict = dict(zip(df['Shift'], df['Number Required'].astype(int)))
num_periods = len(periods)
shift_length = 4
m = gp.Model('BusCrewScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), gp.GRB.MINIMIZE)
for s in periods:
    covering_starts = []
    for t in periods:
        covered = [(t + k - 1) % num_periods + 1 for k in range(shift_length)]
        if s in covered:
            covering_starts.append(t)
    if not covering_starts:
        raise ValueError(f'No assignments cover period {s}.')
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= required_dict[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum total assignments)')
    print('--- Assignment Plan ---')
    for t in periods:
        val = x_vars[t].X
        if val > 1e-06:
            print(f"  Start at period {t} ({df.loc[df['Shift'] == t, 'Time'].values[0]}): {int(round(val))} assignments")
else:
    print(f'No optimal solution found. Status: {m.status}')