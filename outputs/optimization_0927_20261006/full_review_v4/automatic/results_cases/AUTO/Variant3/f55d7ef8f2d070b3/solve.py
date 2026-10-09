import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activities = activities_df['Activity'].tolist()
NormalDuration = {}
CrashDuration = {}
CrashCostPerDay = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    try:
        NormalDuration[act] = int(row['NormalDuration'])
        CrashDuration[act] = int(row['CrashDuration'])
        CrashCostPerDay[act] = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in activity row {idx + 2} for activity '{act}': {e}")
Predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = row['Predecessors'].strip()
    if preds_raw == '':
        Predecessors[act] = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
        Predecessors[act] = preds
Successors = {act: [] for act in activities}
for act in activities:
    for pred in Predecessors[act]:
        if pred not in Successors:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in activities list.")
        Successors[pred].append(act)
terminal_activities = [act for act in activities if len(Successors[act]) == 0]
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if len(deadline_row) != 1:
    raise ValueError('ProjectDeadline parameter not found or not unique in project_parameters.csv')
try:
    ProjectDeadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Invalid ProjectDeadline value: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activities, lb=0, ub={act: NormalDuration[act] - CrashDuration[act] for act in activities}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[act] * z_vars[act] for act in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in Predecessors[i]:
        m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{i}_after_{j}')
for i in terminal_activities:
    m.addConstr(T_var >= s_vars[i] + (NormalDuration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= ProjectDeadline, name='deadline')
m.optimize()