import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [(t - i) % n_periods for i in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for t in periods:
        n_staff = int(round(x[t].X))
        if n_staff > 0:
            time_label = df.loc[t, 'Time']
            print(f'Start at period {t:2d} ({time_label}): {n_staff} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')