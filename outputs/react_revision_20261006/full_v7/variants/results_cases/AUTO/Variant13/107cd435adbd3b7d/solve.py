import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv'
params_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv'
activities_df = pd.read_csv(activities_path, dtype=str, keep_default_na=False)
params_df = pd.read_csv(params_path, dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip().tolist()

def to_int_series(series):
    return series.astype(str).str.strip().astype(int)
normal_duration = dict(zip(activity_ids, to_int_series(activities_df['NormalDuration'])))
crash_duration = dict(zip(activity_ids, to_int_series(activities_df['CrashDuration'])))
crash_cost_per_day = dict(zip(activity_ids, to_int_series(activities_df['CrashCostPerDay'])))
crash_bounds = {i: (0, normal_duration[i] - crash_duration[i]) for i in activity_ids}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    preds_raw = str(row['Predecessors']).strip()
    if preds_raw == '' or preds_raw.lower() == 'nan':
        preds = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
    predecessors[act] = preds
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
for i in activity_ids:
    if i not in normal_duration or i not in crash_duration or i not in crash_cost_per_day:
        raise ValueError(f'Missing duration or cost data for activity {i}')
    if crash_bounds[i][1] < 0:
        raise ValueError(f'CrashDuration exceeds NormalDuration for activity {i}')

def solve_project_crashing(activity_ids, normal_duration, crash_duration, crash_cost_per_day, crash_bounds, predecessors, project_deadline):
    m = gp.Model('ProjectCrashing')
    m.Params.MIPGap = 0.0001
    s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_vars = {}
    for i in activity_ids:
        (lb, ub) = crash_bounds[i]
        z_vars[i] = m.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=f'z_{i}')
    T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
    for i in activity_ids:
        for j in predecessors[i]:
            if j not in activity_ids:
                raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
            m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
    for i in activity_ids:
        m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
    m.addConstr(T_var <= project_deadline, name='deadline')
    m.optimize()
    return m
m = solve_project_crashing(activity_ids=activity_ids, normal_duration=normal_duration, crash_duration=crash_duration, crash_cost_per_day=crash_cost_per_day, crash_bounds=crash_bounds, predecessors=predecessors, project_deadline=project_deadline)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')