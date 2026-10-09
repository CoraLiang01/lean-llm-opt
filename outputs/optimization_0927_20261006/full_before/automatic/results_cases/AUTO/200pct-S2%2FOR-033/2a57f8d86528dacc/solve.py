import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = df['Requirement'].astype(int).to_dict()
shift_starts = list(range(n_periods))
shift_length = 16
coverage = {t: [] for t in periods}
for s in shift_starts:
    covered = [(s + i) % n_periods for i in range(shift_length)]
    for t in covered:
        coverage[t].append(s)
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in coverage[t])) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in shift_starts))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Assignments ---')
    for s in shift_starts:
        val = int(round(x[s].X))
        if val > 0:
            shift_time = df.loc[s, 'Time']
            end_idx = (s + shift_length - 1) % n_periods
            end_time = df.loc[end_idx, 'Time']
            print(f'Start at period {s + 1:02d} ({shift_time}): {val} staff (covers to period {end_idx + 1:02d} [{end_time}])')
else:
    print(f'No optimal solution found. Status: {m.status}')