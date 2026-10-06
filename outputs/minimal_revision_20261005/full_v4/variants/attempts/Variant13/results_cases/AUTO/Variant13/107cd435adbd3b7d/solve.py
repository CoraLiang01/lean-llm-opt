import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'].unique())
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
crash_day_ub = {i: int(normal_duration[i] - crash_duration[i]) for i in activities}

def parse_predecessors(cell):
    cell = str(cell).strip()
    if cell == '' or cell.lower() == 'nan':
        return []
    return [pred.strip() for pred in cell.split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_predecessors(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
for i in activities:
    if i not in normal_duration or i not in crash_duration or i not in crash_cost:
        raise ValueError(f'Missing duration or cost data for activity {i}')
    if crash_day_ub[i] < 0:
        raise ValueError(f'Crash duration exceeds normal duration for activity {i}')
m = gp.Model('ProjectCrashing')
m.setParam('MIPGap', 0.0001)
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=[crash_day_ub[i] for i in activities], vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in predecessors[i]:
        if j not in activities:
            raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
        m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_{i}')
for i in activities:
    m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'compl_{i}')
m.addConstr(T <= project_deadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in activities:
        print(f's[{i}] {s[i].VarName} {s[i].X}')
    for i in activities:
        print(f'z[{i}] {z[i].VarName} {z[i].X}')
    print(f'T {T.VarName} {T.X}')
else:
    print(f'Solver status: {m.status}')