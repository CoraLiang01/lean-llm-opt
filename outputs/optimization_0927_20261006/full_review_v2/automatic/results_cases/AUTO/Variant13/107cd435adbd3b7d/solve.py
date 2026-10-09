import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_predecessors(s):
    if pd.isna(s) or str(s).strip() == '':
        return []
    return [x.strip() for x in str(s).split(';') if x.strip() != '']
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip()
activity_ids_set = set(activity_ids)
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
crash_day_upper = {}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    try:
        nd = int(row['NormalDuration'])
        cd = int(row['CrashDuration'])
        ccpd = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Non-integer duration or cost for activity '{act}': {e}")
    normal_duration[act] = nd
    crash_duration[act] = cd
    crash_cost_per_day[act] = ccpd
    crash_day_upper[act] = nd - cd
    predecessors[act] = parse_predecessors(row['Predecessors'])
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
try:
    project_deadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Non-integer ProjectDeadline value: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[crash_day_upper[act] for act in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for act in activity_ids:
    for pred in predecessors[act]:
        if pred not in activity_ids_set:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in activity list.")
        m.addConstr(s_vars[act] >= s_vars[pred] + (normal_duration[pred] - z_vars[pred]), name=f'prec_{pred}_to_{act}')
for act in activity_ids:
    m.addConstr(T_var >= s_vars[act] + (normal_duration[act] - z_vars[act]), name=f'completion_{act}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crashing cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T_var.X:.2f}')
    print('\n--- Activity Schedule ---')
    for act in activity_ids:
        print(f'Activity {act}:')
        print(f'  Start time (s_{act}): {s_vars[act].X:.2f}')
        print(f'  Crash days used (z_{act}): {int(round(z_vars[act].X))} (max {crash_day_upper[act]})')
        print(f'  Duration: {normal_duration[act] - int(round(z_vars[act].X))} days (Normal: {normal_duration[act]}, Crash: {crash_duration[act]})')
else:
    print(f'No optimal solution found. Status: {m.status}')