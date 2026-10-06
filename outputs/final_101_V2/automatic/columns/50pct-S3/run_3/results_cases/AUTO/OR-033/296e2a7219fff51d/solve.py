import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_to_time = dict(zip(periods, df['Time']))
requirement = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_starts = []
    for s in periods:
        covered = [(s + offset) % n_periods for offset in range(shift_length)]
        if t in covered:
            covering_starts.append(s)
    if len(covering_starts) == 0:
        raise ValueError(f'No shift covers period {t} ({period_to_time[t]})')
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirement[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        val = x[s].X
        if val > 0.5:
            print(f'Start at {period_to_time[s]}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')