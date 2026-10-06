import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant21/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant21/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities = list(activities_df['Activity'])
NormalDuration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
CrashDuration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
CrashCostPerDay = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
CrashBounds = {i: NormalDuration[i] - CrashDuration[i] for i in activities}

def parse_preds(preds):
    if pd.isna(preds) or str(preds).strip() == '':
        return []
    return [p.strip() for p in re.split('[;,]', str(preds)) if p.strip() != '']
Predecessors = {row['Activity']: parse_preds(row['Predecessors']) for _, row in activities_df.iterrows()}
deadline_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=[CrashBounds[i] for i in activities], vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in Predecessors[i]:
        if j not in activities:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_to_{i}')
for i in activities:
    m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'compl_{i}')
m.addConstr(T <= ProjectDeadline, name='deadline')
m.optimize()