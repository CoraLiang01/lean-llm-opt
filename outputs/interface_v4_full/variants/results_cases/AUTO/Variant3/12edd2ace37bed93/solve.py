import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant3/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'])
NormalDuration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
CrashDuration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
CrashCostPerDay = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))

def parse_preds(preds):
    preds = preds.strip()
    if preds == '' or preds.lower() == 'nan':
        return []
    return [p.strip() for p in preds.split(';') if p.strip() != '']
Predecessors = {row['Activity']: parse_preds(row['Predecessors']) for _, row in activities_df.iterrows()}
Successors = {a: [] for a in activities}
for i in activities:
    for pred in Predecessors[i]:
        if pred in Successors:
            Successors[pred].append(i)
        else:
            raise ValueError(f"Predecessor '{pred}' for activity '{i}' not found in activity list.")
deadline_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])

def solve_project_crashing():
    m = gp.Model('ProjectCrashing')
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=0, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in Predecessors[i]:
            m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for i in activities:
        max_crash = NormalDuration[i] - CrashDuration[i]
        m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
        m.addConstr(z[i] <= max_crash, name=f'z_ub_{i}')
    terminal_activities = [i for i in activities if len(Successors[i]) == 0]
    for i in terminal_activities:
        m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'T_terminal_{i}')
    m.addConstr(T <= ProjectDeadline, name='deadline')
    m.optimize()
    return m
m = solve_project_crashing()