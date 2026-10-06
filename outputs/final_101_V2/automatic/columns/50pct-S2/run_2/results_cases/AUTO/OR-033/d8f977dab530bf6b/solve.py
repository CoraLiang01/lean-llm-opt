import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
time_slots = list(df.index)
n_slots = len(time_slots)
if n_slots != 48:
    raise ValueError(f'Expected 48 time slots, got {n_slots}')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(time_slots, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in time_slots)), gp.GRB.MINIMIZE)
shift_length = 16
for t in time_slots:
    covering_starts = [s for s in time_slots if (t - s) % n_slots < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments (slot index: number of staff) ---')
    for s in time_slots:
        if x[s].X > 1e-06:
            time_label = df.loc[s, 'Time']
            print(f'  Shift starting at slot {s} ({time_label}): {int(round(x[s].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')