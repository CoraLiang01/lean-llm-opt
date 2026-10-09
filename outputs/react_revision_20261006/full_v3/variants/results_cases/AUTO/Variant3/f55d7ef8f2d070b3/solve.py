import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities = list(activities_df['Activity'])
NormalDuration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
CrashDuration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
CrashCostPerDay = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))

def parse_preds(cell):
    if pd.isna(cell) or str(cell).strip() == '':
        return []
    return [x.strip() for x in str(cell).split(';') if x.strip() != '']
Predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds = parse_preds(row['Predecessors'])
    Predecessors[act] = preds
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
for i in activities:
    if i not in NormalDuration or i not in CrashDuration or i not in CrashCostPerDay:
        raise ValueError(f"Missing duration or cost data for activity '{i}'")
    if not 0 <= CrashDuration[i] <= NormalDuration[i]:
        raise ValueError(f"CrashDuration > NormalDuration for activity '{i}'")

def solve_project_crashing(activities, NormalDuration, CrashDuration, CrashCostPerDay, Predecessors, ProjectDeadline):
    m = gp.Model('ProjectCrashing')
    m.Params.MIPGap = 0.0001
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=0, ub={i: NormalDuration[i] - CrashDuration[i] for i in activities}, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in Predecessors[i]:
            if j not in activities:
                raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activities list")
            m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_{i}')
    for i in activities:
        m.addConstr(z[i] >= 0, name=f'z_nonneg_{i}')
        m.addConstr(z[i] <= NormalDuration[i] - CrashDuration[i], name=f'z_ub_{i}')
    for i in activities:
        m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'T_finish_{i}')
    m.addConstr(T <= ProjectDeadline, name='T_deadline')
    for i in activities:
        m.addConstr(s[i] >= 0, name=f's_nonneg_{i}')
    m.addConstr(T >= 0, name='T_nonneg')
    m.optimize()
    return m
m = solve_project_crashing(activities=activities, NormalDuration=NormalDuration, CrashDuration=CrashDuration, CrashCostPerDay=CrashCostPerDay, Predecessors=Predecessors, ProjectDeadline=ProjectDeadline)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')