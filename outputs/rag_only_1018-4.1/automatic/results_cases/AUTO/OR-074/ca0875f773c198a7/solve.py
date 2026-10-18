import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
df['Time'] = df['Time'].str.strip()
df['Requirement'] = df['Requirement'].astype(int)
time_periods = list(df['Time'])
n_periods = len(time_periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {n_periods}')
period_idx_to_label = {i: time_periods[i] for i in range(n_periods)}
period_label_to_idx = {time_periods[i]: i for i in range(n_periods)}
requirements = df.set_index('Time')['Requirement'].to_dict()
requirements_by_idx = {i: requirements[period_idx_to_label[i]] for i in range(n_periods)}
shift_length = 16
shift_starts = list(range(n_periods))
shift_coverage = dict()
for s in shift_starts:
    covered = [(s + offset) % n_periods for offset in range(shift_length)]
    shift_coverage[s] = set(covered)
period_covered_by_shifts = {t: [] for t in range(n_periods)}
for s in shift_starts:
    for t in shift_coverage[s]:
        period_covered_by_shifts[t].append(s)
m = Model('Waitstaff_Scheduling')
m.Params.OutputFlag = 0
x_vars = m.addVars(shift_starts, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((x_vars[s] for s in shift_starts)), GRB.MINIMIZE)
for t in range(n_periods):
    covering_shifts = period_covered_by_shifts[t]
    if not covering_shifts:
        raise ValueError(f'No shifts cover period {t} ({period_idx_to_label[t]})')
    m.addConstr(quicksum((x_vars[s] for s in covering_shifts)) >= requirements_by_idx[t], name=f'cover_{t}')
m.optimize()