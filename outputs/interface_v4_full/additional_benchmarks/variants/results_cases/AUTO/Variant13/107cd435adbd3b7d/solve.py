import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities = list(activities_df['Activity'])

def parse_predecessors(cell):
    if pd.isna(cell) or str(cell).strip() == '':
        return []
    return [pred.strip() for pred in str(cell).split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_predecessors(row['Predecessors']) for _, row in activities_df.iterrows()}
successors = {act: [] for act in activities}
for act, preds in predecessors.items():
    for pred in preds:
        if pred not in successors:
            raise ValueError(f"Predecessor '{pred}' listed for activity '{act}' not found in activity list.")
        successors[pred].append(act)
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost_per_day = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
max_crash_days = {act: int(normal_duration[act] - crash_duration[act]) for act in activities}
deadline_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
terminal_activities = [act for act in activities if len(successors[act]) == 0]
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=[max_crash_days[act] for act in activities], vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z[act] for act in activities)), gp.GRB.MINIMIZE)
for act in activities:
    for pred in predecessors[act]:
        m.addConstr(s[act] >= s[pred] + (normal_duration[pred] - z[pred]), name=f'prec_{pred}_to_{act}')
for act in activities:
    m.addConstr(z[act] <= max_crash_days[act], name=f'crash_ub_{act}')
    m.addConstr(z[act] >= 0, name=f'crash_lb_{act}')
for act in terminal_activities:
    m.addConstr(T >= s[act] + (normal_duration[act] - z[act]), name=f'completion_{act}')
m.addConstr(T <= project_deadline, name='deadline')
m.optimize()