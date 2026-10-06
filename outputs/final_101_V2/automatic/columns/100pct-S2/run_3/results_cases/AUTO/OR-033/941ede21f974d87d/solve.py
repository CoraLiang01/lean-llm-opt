import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx = list(range(48))
period_idx_to_time = dict(zip(period_idx, periods))
requirement = df['Requirement'].astype(int).to_dict()
shift_starts = period_idx
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
shift_length = 16
for t in period_idx:
    covering_shifts = [s for s in shift_starts if (t - s) % 48 in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x[s].X)) for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Assignment ---')
    for s in shift_starts:
        num = int(round(x[s].X))
        if num > 0:
            print(f'Shift start: {period_idx_to_time[s]:<20}  Number of waitstaff: {num}')
else:
    print(f'No optimal solution found. Status: {m.status}')