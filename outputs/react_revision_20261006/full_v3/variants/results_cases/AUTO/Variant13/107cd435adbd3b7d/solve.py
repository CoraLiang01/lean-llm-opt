import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'].unique())
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    normal_duration[act] = int(row['NormalDuration'])
    crash_duration[act] = int(row['CrashDuration'])
    crash_cost_per_day[act] = int(row['CrashCostPerDay'])
    preds_raw = str(row['Predecessors']).strip()
    if preds_raw == '' or preds_raw.lower() == 'nan':
        preds = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
    predecessors[act] = preds
crash_bounds = {}
for act in activities:
    lb = 0
    ub = normal_duration[act] - crash_duration[act]
    if ub < 0:
        raise ValueError(f'CrashDuration for activity {act} exceeds NormalDuration.')
    crash_bounds[act] = (lb, ub)
successors = {act: [] for act in activities}
for act in activities:
    for pred in predecessors[act]:
        if pred not in activities:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in activity list.")
        successors[pred].append(act)
terminal_activities = [act for act in activities if len(successors[act]) == 0]
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])

def solve_project_crashing():
    m = gp.Model('ProjectCrashing')
    m.setParam('MIPGap', 0.0001)
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb={i: crash_bounds[i][0] for i in activities}, ub={i: crash_bounds[i][1] for i in activities}, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost_per_day[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in predecessors[i]:
            m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for i in terminal_activities:
        m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'complete_{i}')
    m.addConstr(T <= project_deadline, name='deadline')
    m.optimize()
    return m
m = solve_project_crashing()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')