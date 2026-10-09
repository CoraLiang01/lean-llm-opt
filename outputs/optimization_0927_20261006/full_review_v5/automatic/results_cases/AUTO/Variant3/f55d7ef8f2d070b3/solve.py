import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()
NormalDuration = {}
CrashDuration = {}
CrashCostPerDay = {}
CrashAmountMax = {}
Predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    try:
        ndur = int(row['NormalDuration'])
        cdur = int(row['CrashDuration'])
        ccost = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in row {idx + 2} for activity '{act}': {e}")
    NormalDuration[act] = ndur
    CrashDuration[act] = cdur
    CrashCostPerDay[act] = ccost
    CrashAmountMax[act] = ndur - cdur
    preds_raw = row['Predecessors']
    if preds_raw.strip() == '':
        preds = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
    Predecessors[act] = preds
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
try:
    ProjectDeadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Invalid ProjectDeadline value: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[CrashAmountMax[a] for a in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[a] * z_vars[a] for a in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in Predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{i}_after_{j}')
for i in activity_ids:
    m.addConstr(z_vars[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z_vars[i] <= CrashAmountMax[i], name=f'z_ub_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (NormalDuration[i] - z_vars[i]), name=f'T_after_{i}')
m.addConstr(T_var <= ProjectDeadline, name='deadline')
m.optimize()