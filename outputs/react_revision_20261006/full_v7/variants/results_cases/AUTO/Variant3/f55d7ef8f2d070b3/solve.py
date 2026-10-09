import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv'
params_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv'
activities_df = pd.read_csv(activities_path, sep=',', dtype=str, keep_default_na=False)
params_df = pd.read_csv(params_path, sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
NormalDuration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
CrashDuration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
CrashCostPerDay = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
Predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = row['Predecessors'].strip()
    if preds_raw == '':
        Predecessors[act] = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
        Predecessors[act] = preds
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row.iloc[0]['Value'])
for i in activity_ids:
    if i not in NormalDuration or i not in CrashDuration or i not in CrashCostPerDay or (i not in Predecessors):
        raise ValueError(f'Missing parameter for activity {i}')

def solve_project_crashing(activity_ids, NormalDuration, CrashDuration, CrashCostPerDay, Predecessors, ProjectDeadline):
    m = gp.Model('ProjectCrashing')
    m.Params.MIPGap = 0.0001
    s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_ub = {i: NormalDuration[i] - CrashDuration[i] for i in activity_ids}
    z_vars = m.addVars(activity_ids, lb=0, ub=z_ub, vtype=gp.GRB.INTEGER, name='')
    T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((CrashCostPerDay[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
    for i in activity_ids:
        for j in Predecessors[i]:
            if j not in activity_ids:
                raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
            m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{i}_{j}')
    for i in activity_ids:
        m.addConstr(z_vars[i] >= 0, name=f'z_nonneg_{i}')
        m.addConstr(z_vars[i] <= NormalDuration[i] - CrashDuration[i], name=f'z_ub_{i}')
    for i in activity_ids:
        m.addConstr(s_vars[i] >= 0, name=f's_nonneg_{i}')
    for i in activity_ids:
        m.addConstr(T_var >= s_vars[i] + (NormalDuration[i] - z_vars[i]), name=f'T_finish_{i}')
    m.addConstr(T_var <= ProjectDeadline, name='T_deadline')
    m.optimize()
    return m
m = solve_project_crashing(activity_ids, NormalDuration, CrashDuration, CrashCostPerDay, Predecessors, ProjectDeadline)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')