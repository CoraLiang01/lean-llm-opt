import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
time_periods = df['Time'].tolist()
if len(time_periods) != 48:
    raise ValueError(f'Expected 48 time periods, got {len(time_periods)}')
try:
    requirements = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' to int: {e}")
period_idx_to_id = {i: time_periods[i] for i in range(48)}
period_id_to_idx = {time_periods[i]: i for i in range(48)}
period_idx_to_req = {i: requirements[i] for i in range(48)}
shift_start_indices = list(range(48))

def covered_periods_by_shift(s):
    return [(s + offset) % 48 for offset in range(16)]
period_idx_to_covering_shifts = {t: [] for t in range(48)}
for s in shift_start_indices:
    covered = covered_periods_by_shift(s)
    for t in covered:
        period_idx_to_covering_shifts[t].append(s)
m = Model('Waitstaff_Scheduling')
m.Params.OutputFlag = 0
shift_start_vars = m.addVars(shift_start_indices, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((shift_start_vars[s] for s in shift_start_indices)), GRB.MINIMIZE)
for t in range(48):
    covering_shifts = period_idx_to_covering_shifts[t]
    m.addConstr(quicksum((shift_start_vars[s] for s in covering_shifts)) >= period_idx_to_req[t], name=f'cover_{t}')
m.optimize()