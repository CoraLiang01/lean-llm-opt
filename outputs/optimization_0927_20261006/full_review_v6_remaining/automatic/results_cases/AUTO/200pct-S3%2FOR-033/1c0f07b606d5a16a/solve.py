import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in CSV.")
time_slots = list(df.index)
slot_labels = df['Time'].tolist()
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
num_slots = len(time_slots)
shift_length = 16
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(time_slots, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in time_slots)), gp.GRB.MINIMIZE)
for t in time_slots:
    covering_starts = []
    for s in time_slots:
        covered_slots = [(s + i) % num_slots for i in range(shift_length)]
        if t in covered_slots:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'Time slot {t} ({slot_labels[t]}) is not covered by any shift start.')
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in time_slots:
        n = int(round(shift_start_vars[s].X))
        if n > 0:
            print(f'  Start at slot {s:2d} ({slot_labels[s]}): {n} waitstaff')
    print('\n--- Coverage per Time Slot ---')
    for t in time_slots:
        coverage = sum((int(round(shift_start_vars[s].X)) for s in time_slots if t in [(s + i) % num_slots for i in range(shift_length)]))
        print(f'  Slot {t:2d} ({slot_labels[t]}): Required={requirements[t]}, Covered={coverage}')
else:
    print(f'No optimal solution found. Status: {m.status}')