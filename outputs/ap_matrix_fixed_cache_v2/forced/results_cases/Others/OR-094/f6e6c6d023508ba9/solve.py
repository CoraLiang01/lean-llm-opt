import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
workstations = df['Workstation'].astype(int).tolist()
hifi_cols = [col for col in df.columns if re.fullmatch('HiFi\\d+_Minutes', col)]

def hifi_num(col):
    m = re.match('HiFi(\\d+)_Minutes', col)
    return int(m.group(1)) if m else 0
hifi_cols = sorted(hifi_cols, key=hifi_num)
models = [f'HiFi{n}' for n in range(1, 102)]
expected_cols = [f'HiFi{n}_Minutes' for n in range(1, 102)]
missing_cols = set(expected_cols) - set(hifi_cols)
if missing_cols:
    raise ValueError(f'Missing required columns in CSV: {missing_cols}')
processing_time = {}
for idx, row in df.iterrows():
    w = int(row['Workstation'])
    processing_time[w] = {}
    for col in hifi_cols:
        model = col.replace('_Minutes', '')
        processing_time[w][model] = int(row[col])
maintenance_percent = {}
effective_capacity = {}
for idx, row in df.iterrows():
    w = int(row['Workstation'])
    maint_pct = float(row['Maintenance_Percent'])
    maintenance_percent[w] = maint_pct
    effective_capacity[w] = 1440 * (1 - maint_pct / 100.0)
m = gp.Model('MinimizeTotalIdleTime')
x_vars = m.addVars(models, vtype=gp.GRB.INTEGER, lb=0, name='')
idle_vars = m.addVars(workstations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for w in workstations:
    m.addConstr(gp.quicksum((processing_time[w][model] * x_vars[model] for model in models)) + idle_vars[w] == effective_capacity[w], name=f'workstation_balance_{w}')
m.setObjective(gp.quicksum((idle_vars[w] for w in workstations)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total idle time: {m.objVal:.2f} minutes')
    print('\n--- Production Plan (nonzero units) ---')
    for model in models:
        val = x_vars[model].X
        if val > 1e-06:
            print(f'{model}: {int(round(val))} units')
    print('\n--- Idle Time per Workstation ---')
    for w in workstations:
        print(f'Workstation {w}: {idle_vars[w].X:.2f} minutes (Effective capacity: {effective_capacity[w]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')