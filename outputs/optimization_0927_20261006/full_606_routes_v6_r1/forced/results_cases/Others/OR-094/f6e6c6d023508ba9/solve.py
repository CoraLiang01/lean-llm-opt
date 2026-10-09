import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
workstations = [1, 2, 3]
hifi_pattern = re.compile('^HiFi(\\d+)_Minutes$')
hifi_cols = []
hifi_nums = []
for col in df.columns:
    m = hifi_pattern.match(col)
    if m:
        hifi_cols.append(col)
        hifi_nums.append(int(m.group(1)))
sorted_hifi = sorted(zip(hifi_nums, hifi_cols), key=lambda x: x[0])
model_nums = [num for (num, col) in sorted_hifi]
model_cols = [col for (num, col) in sorted_hifi]
if len(model_nums) != 101 or model_nums[0] != 1 or model_nums[-1] != 101:
    raise ValueError('Did not find all 101 HiFi model columns from 1 to 101.')
processing_time = {}
for (idx, row) in df.iterrows():
    ws = int(row['Workstation'])
    if ws not in workstations:
        continue
    for (num, col) in zip(model_nums, model_cols):
        val = row[col]
        try:
            processing_time[ws, num] = int(val)
        except Exception:
            raise ValueError(f'Invalid processing time for workstation {ws}, model {num}: {val}')
maintenance_percent = {}
for (idx, row) in df.iterrows():
    ws = int(row['Workstation'])
    if ws not in workstations:
        continue
    val = row['Maintenance_Percent']
    try:
        maintenance_percent[ws] = int(val)
    except Exception:
        raise ValueError(f'Invalid maintenance percent for workstation {ws}: {val}')
total_minutes_per_day = 1440
effective_capacity = {}
for ws in workstations:
    if ws not in maintenance_percent:
        raise ValueError(f'Missing maintenance percent for workstation {ws}')
    eff = total_minutes_per_day * (1 - maintenance_percent[ws] / 100.0)
    effective_capacity[ws] = eff
m = gp.Model('MinimizeTotalIdleTime')
x_vars = m.addVars(model_nums, vtype=gp.GRB.INTEGER, lb=0, name='')
idle_vars = m.addVars(workstations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for ws in workstations:
    used_time_expr = gp.quicksum((processing_time[ws, i] * x_vars[i] for i in model_nums))
    m.addConstr(idle_vars[ws] == effective_capacity[ws] - used_time_expr, name=f'idle_def_ws{ws}')
    m.addConstr(used_time_expr <= effective_capacity[ws], name=f'cap_ws{ws}')
m.setObjective(gp.quicksum((idle_vars[ws] for ws in workstations)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total idle time: {m.objVal:.2f} minutes')
    print('\n--- Production Plan (units per model) ---')
    for i in model_nums:
        val = x_vars[i].X
        if val > 0.5:
            print(f'  HiFi-{i}: {int(round(val))} units')
    print('\n--- Idle Time per Workstation ---')
    for ws in workstations:
        print(f'  Workstation {ws}: {idle_vars[ws].X:.2f} minutes (Effective capacity: {effective_capacity[ws]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')