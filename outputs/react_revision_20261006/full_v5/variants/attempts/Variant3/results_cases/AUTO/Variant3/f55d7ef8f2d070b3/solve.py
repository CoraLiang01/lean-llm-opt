import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'])
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = row['Predecessors']
    if preds_raw == '' or pd.isna(preds_raw):
        preds = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
    predecessors[act] = preds
successors = {act: [] for act in activities}
for act in activities:
    for pred in predecessors[act]:
        if pred in successors:
            successors[pred].append(act)
        else:
            raise ValueError(f"Predecessor '{pred}' for activity '{act}' not found in activity list.")
terminal_activities = [act for act in activities if len(successors[act]) == 0]
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
for act in activities:
    if act not in normal_duration or act not in crash_duration or act not in crash_cost:
        raise ValueError(f"Missing duration or cost data for activity '{act}'")
    if normal_duration[act] < crash_duration[act]:
        raise ValueError(f"NormalDuration < CrashDuration for activity '{act}'")

def solve_project_crashing(activities, predecessors, successors, terminal_activities, normal_duration, crash_duration, crash_cost, project_deadline):
    m = gp.Model('ProjectCrashing')
    m.Params.MIPGap = 0.0001
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=0, ub=None, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in predecessors[i]:
            m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for i in activities:
        max_crash = normal_duration[i] - crash_duration[i]
        m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
        m.addConstr(z[i] <= max_crash, name=f'z_ub_{i}')
    for i in terminal_activities:
        m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'T_terminal_{i}')
    m.addConstr(T <= project_deadline, name='T_deadline')
    m.optimize()
    return m
m = solve_project_crashing(activities=activities, predecessors=predecessors, successors=successors, terminal_activities=terminal_activities, normal_duration=normal_duration, crash_duration=crash_duration, crash_cost=crash_cost, project_deadline=project_deadline)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.status}')