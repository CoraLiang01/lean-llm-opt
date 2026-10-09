import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv'
params_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv'
activities_df = pd.read_csv(activities_path, sep=',', dtype=str, keep_default_na=False)
for col in ['NormalDuration', 'CrashDuration', 'CrashCostPerDay']:
    activities_df[col] = activities_df[col].astype(int)
params_df = pd.read_csv(params_path, sep=',', dtype=str, keep_default_na=False)
params_df['Parameter'] = params_df['Parameter'].str.strip().str.casefold()
params_df['Value'] = params_df['Value'].astype(int)
activity_ids = activities_df['Activity'].tolist()
crash_cost_per_day = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
max_crash_days = {aid: normal_duration[aid] - crash_duration[aid] for aid in activity_ids}

def parse_preds(preds):
    preds = preds.strip()
    if preds == '':
        return []
    return [p.strip() for p in re.split('[;,]', preds) if p.strip() != '']
predecessors = {}
for (idx, row) in activities_df.iterrows():
    aid = row['Activity']
    preds = parse_preds(row['Predecessors'])
    predecessors[aid] = preds
deadline_row = params_df[params_df['Parameter'] == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[max_crash_days[aid] for aid in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[aid] * z_vars[aid] for aid in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crashing cost: {m.objVal:.2f}')
    print(f'Project completion time T: {T_var.X:.2f} (Deadline: {project_deadline})')
    print('\n--- Activity Schedule ---')
    for aid in activity_ids:
        print(f'Activity {aid}:')
        print(f'  Start time (s_{aid}): {s_vars[aid].X:.2f}')
        print(f'  Crash days used (z_{aid}): {int(round(z_vars[aid].X))} / max {max_crash_days[aid]}')
        print(f'  Duration: {normal_duration[aid] - int(round(z_vars[aid].X))} days (Normal: {normal_duration[aid]}, Crash: {crash_duration[aid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')