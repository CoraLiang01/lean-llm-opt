import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 time slots, found {df.shape[0]}')
if df['Requirement'].isnull().any():
    raise ValueError("Missing values in 'Requirement' column.")
time_slots = list(range(48))
shift_starts = list(range(48))
requirement = df['Requirement'].astype(int).to_dict()
shift_covers = dict()
for s in shift_starts:
    covered = [(s + i) % 48 for i in range(16)]
    shift_covers[s] = set(covered)
slot_covered_by = dict()
for t in time_slots:
    slot_covered_by[t] = [s for s in shift_starts if t in shift_covers[s]]
for t in time_slots:
    if len(slot_covered_by[t]) == 0:
        raise ValueError(f'Time slot {t} is not covered by any shift start.')
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in time_slots:
    m.addConstr(gp.quicksum((x[s] for s in slot_covered_by[t])) >= requirement[t], name=f'cov_{t}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        print(f'{x[s].VarName} {x[s].X}')
else:
    print(f'Solver status: {m.status}')