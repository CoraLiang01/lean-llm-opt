import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
shifts = list(range(len(df)))
period_idx_to_time = dict(zip(periods, df['Time']))
if df['Requirement'].isnull().any():
    raise ValueError('Missing requirement values in CSV.')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shifts)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_shifts = [s for s in shifts if (t - s) % len(periods) < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirements[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shifts:
        val = x[s].X
        if val > 1e-06:
            print(f'  Start at {period_idx_to_time[s]}: {int(round(val))} waitstaff')
    print('\n--- Coverage per Period ---')
    for t in periods:
        covering_shifts = [s for s in shifts if (t - s) % len(periods) < shift_length]
        coverage = sum((x[s].X for s in covering_shifts))
        print(f'  {period_idx_to_time[t]}: required={requirements[t]}, covered={int(round(coverage))}')
else:
    print(f'No optimal solution found. Status: {m.status}')