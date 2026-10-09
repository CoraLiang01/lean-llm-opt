import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',')
activities = activities_df['Activity'].astype(str).tolist()
NormalDuration = dict(zip(activities_df['Activity'].astype(str), activities_df['NormalDuration'].astype(int)))
CrashDuration = dict(zip(activities_df['Activity'].astype(str), activities_df['CrashDuration'].astype(int)))
CrashCostPerDay = dict(zip(activities_df['Activity'].astype(str), activities_df['CrashCostPerDay'].astype(int)))

def parse_preds(cell):
    if pd.isnull(cell) or str(cell).strip() == '':
        return []
    return [pred.strip() for pred in str(cell).split(';') if pred.strip() != '']
Predecessors = dict(zip(activities_df['Activity'].astype(str), activities_df['Predecessors'].apply(parse_preds)))
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
for i in activities:
    if i not in NormalDuration or i not in CrashDuration or i not in CrashCostPerDay or (i not in Predecessors):
        raise ValueError(f'Missing parameter for activity {i}')
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in Predecessors[i]:
        if j not in activities:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{i}_{j}')
for i in activities:
    max_crash = NormalDuration[i] - CrashDuration[i]
    m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z[i] <= max_crash, name=f'z_ub_{i}')
for i in activities:
    m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'T_ge_finish_{i}')
m.addConstr(T <= ProjectDeadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crashing cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T.X:.2f} days')
    print('\n--- Activity Schedule ---')
    for i in activities:
        print(f'Activity {i}:')
        print(f'  Start time (s_{i}): {s[i].X:.2f}')
        print(f'  Days crashed (z_{i}): {int(round(z[i].X))} (of max {NormalDuration[i] - CrashDuration[i]})')
        print(f'  Duration after crashing: {NormalDuration[i] - int(round(z[i].X))} days')
else:
    print(f'No optimal solution found. Status: {m.status}')