import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities = list(activities_df['Activity'])
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
crash_bounds = {i: (0, normal_duration[i] - crash_duration[i]) for i in activities}

def parse_preds(s):
    if pd.isna(s) or str(s).strip() == '':
        return []
    return [pred.strip() for pred in str(s).split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_preds(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
m.setParam('MIPGap', 0.0001)
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb={i: crash_bounds[i][0] for i in activities}, ub={i: crash_bounds[i][1] for i in activities}, vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in predecessors[i]:
        if j not in activities:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name='prec')
for i in activities:
    m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name='compl')
m.addConstr(T <= project_deadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')