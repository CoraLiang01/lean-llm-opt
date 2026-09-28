import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(csv_path, sep=',')
workstations = df['Workstation'].astype(int).tolist()
if set(workstations) != {1, 2, 3} or len(workstations) != 3:
    raise ValueError('Expected exactly 3 workstations with IDs 1,2,3.')
model_cols = [col for col in df.columns if re.fullmatch('HiFi\\d+_Minutes', col)]
if len(model_cols) != 101:
    raise ValueError(f'Expected 101 HiFi columns, found {len(model_cols)}.')
model_indices = list(range(1, 102))
model_col_map = {i: f'HiFi{i}_Minutes' for i in model_indices}
for i in model_indices:
    if model_col_map[i] not in model_cols:
        raise ValueError(f'Missing expected column: {model_col_map[i]}')
processing_time = {}
for w in workstations:
    row = df.loc[df['Workstation'] == w]
    if row.shape[0] != 1:
        raise ValueError(f'Workstation {w} not found or duplicated in CSV.')
    row = row.iloc[0]
    for m in model_indices:
        val = row[model_col_map[m]]
        if not np.issubdtype(type(val), np.integer):
            raise ValueError(f'Processing time for workstation {w}, model {m} is not integer.')
        processing_time[w, m] = int(val)
maintenance_percent = {}
for w in workstations:
    row = df.loc[df['Workstation'] == w]
    percent = row['Maintenance_Percent'].iloc[0]
    if not np.issubdtype(type(percent), np.integer):
        raise ValueError(f'Maintenance percent for workstation {w} is not integer.')
    maintenance_percent[w] = int(percent)
total_minutes = 1440
effective_capacity = {}
for w in workstations:
    eff = total_minutes * (1 - maintenance_percent[w] / 100)
    effective_capacity[w] = eff
m = gp.Model('MinIdleTime')
x = m.addVars(model_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
idle = m.addVars(workstations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for w in workstations:
    m.addConstr(gp.quicksum((processing_time[w, m] * x[m] for m in model_indices)) + idle[w] == effective_capacity[w], name=f'cap_w{w}')
m.setObjective(gp.quicksum((idle[w] for w in workstations)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total idle time: {m.objVal:.2f} minutes')
    print('\n--- Production Plan (nonzero units only) ---')
    for m_idx in model_indices:
        val = x[m_idx].X
        if val > 1e-06:
            print(f'  HiFi-{m_idx}: {int(round(val))} units')
    print('\n--- Idle Time per Workstation ---')
    for w in workstations:
        print(f'  Workstation {w}: {idle[w].X:.2f} minutes (Effective capacity: {effective_capacity[w]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')