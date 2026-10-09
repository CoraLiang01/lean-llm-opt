import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_slots = 48
shift_length_slots = 16
if df.shape[0] != num_slots:
    raise ValueError(f'Expected {num_slots} time slots, found {df.shape[0]}')
if 'Requirement' not in df.columns:
    raise ValueError("Missing required column 'Requirement' in CSV.")
requirements = df['Requirement'].astype(int).to_numpy()
if requirements.shape[0] != num_slots:
    raise ValueError('Requirement vector length mismatch.')
shift_start_indices = list(range(num_slots))
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(num_slots):
    covering_starts = []
    for s in shift_start_indices:
        covered_slots = [(s + offset) % num_slots for offset in range(shift_length_slots)]
        if t in covered_slots:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers time slot {t}')
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_start_indices:
        var = shift_start_vars[s]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')