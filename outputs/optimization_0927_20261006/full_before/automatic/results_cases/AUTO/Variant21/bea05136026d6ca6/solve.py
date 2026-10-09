import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'])
NormalDuration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
CrashDuration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
CrashCostPerDay = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
CrashBounds = {i: NormalDuration[i] - CrashDuration[i] for i in activities}

def parse_preds(cell):
    cell = str(cell).strip()
    if cell == '' or cell.lower() == 'nan':
        return []
    return [pred.strip() for pred in cell.split(';') if pred.strip() != '']
Predecessors = {row['Activity']: parse_preds(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z[i] <= CrashBounds[i], name=f'z_ub_{i}')
for i in activities:
    for j in Predecessors[i]:
        if j not in activities:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_to_{i}')
for i in activities:
    m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'completion_{i}')
m.addConstr(T <= ProjectDeadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crash cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T.X:.2f}')
    print('--- Activity Schedule ---')
    for i in activities:
        print(f'Activity {i}:')
        print(f'  Start time (s_{i}): {s[i].X:.2f}')
        print(f'  Crash days used (z_{i}): {int(round(z[i].X))} (max {CrashBounds[i]})')
        print(f'  Duration after crashing: {NormalDuration[i] - int(round(z[i].X))} days')
        preds = Predecessors[i]
        if preds:
            print(f"  Predecessors: {', '.join(preds)}")
        else:
            print(f'  Predecessors: None')
else:
    print(f'No optimal solution found. Status: {m.status}')