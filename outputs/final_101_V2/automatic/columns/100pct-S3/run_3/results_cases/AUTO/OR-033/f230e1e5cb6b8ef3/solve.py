import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_labels = df['Time'].to_dict()
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
num_periods = 48
shift_length = 16
for t in periods:
    covering_starts = [s for s in periods if (t - s) % num_periods < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        staff_count = int(round(x[s].X))
        if staff_count > 0:
            print(f'Start at {period_labels[s]}: {staff_count} staff')
    print(f'\nTotal waitstaff scheduled: {int(round(total_staff))}')
else:
    print(f'No optimal solution found. Status: {m.status}')