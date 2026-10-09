import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv'
params_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv'
df_acts = pd.read_csv(activities_path, sep=',')
df_acts['Activity'] = df_acts['Activity'].astype(str).str.strip()
df_params = pd.read_csv(params_path, sep=',')
deadline_row = df_params[df_params['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
Activities = df_acts['Activity'].tolist()
NormalDuration = dict(zip(df_acts['Activity'], df_acts['NormalDuration']))
CrashDuration = dict(zip(df_acts['Activity'], df_acts['CrashDuration']))
CrashCostPerDay = dict(zip(df_acts['Activity'], df_acts['CrashCostPerDay']))
CrashLowerBound = {i: 0 for i in Activities}
CrashUpperBound = {i: NormalDuration[i] - CrashDuration[i] for i in Activities}

def parse_preds(preds):
    if pd.isna(preds) or str(preds).strip() == '':
        return []
    return [p.strip() for p in str(preds).split(';') if p.strip() != '']
Predecessors = {}
for (idx, row) in df_acts.iterrows():
    act = row['Activity']
    preds = parse_preds(row['Predecessors'])
    Predecessors[act] = preds
for (act, preds) in Predecessors.items():
    for pred in preds:
        if pred not in Activities:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in Activities list.")

def solve_project_crashing():
    m = gp.Model('ProjectCrashing')
    s = m.addVars(Activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(Activities, lb=[CrashLowerBound[i] for i in Activities], ub=[CrashUpperBound[i] for i in Activities], vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in Activities)), gp.GRB.MINIMIZE)
    for i in Activities:
        for j in Predecessors[i]:
            m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for i in Activities:
        m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'compl_{i}')
    m.addConstr(T <= ProjectDeadline, name='deadline')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_project_crashing()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')