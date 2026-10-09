import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
time_slots = list(df['Time'])
n_slots = len(time_slots)
slot_idx_to_label = {i: time_slots[i] for i in range(n_slots)}
label_to_slot_idx = {label: i for (i, label) in slot_idx_to_label.items()}
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
shift_covers = dict()
for s in range(n_slots):
    covered = [(s + k) % n_slots for k in range(shift_length)]
    shift_covers[s] = set(covered)
slot_covered_by = {t: set() for t in range(n_slots)}
for s in range(n_slots):
    for t in shift_covers[s]:
        slot_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
x = m.addVars(range(n_slots), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in range(n_slots))), gp.GRB.MINIMIZE)
for t in range(n_slots):
    m.addConstr(gp.quicksum((x[s] for s in slot_covered_by[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in range(n_slots)))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(n_slots):
        val = x[s].X
        if val > 1e-06:
            print(f'  Start at {slot_idx_to_label[s]}: {int(round(val))} staff')
    print('\n--- Coverage Check (first 10 slots) ---')
    for t in range(min(10, n_slots)):
        covered = sum((x[s].X for s in slot_covered_by[t]))
        print(f'  Slot {slot_idx_to_label[t]}: Required={requirements[t]}, Covered={int(round(covered))}')
else:
    print(f'No optimal solution found. Status: {m.status}')