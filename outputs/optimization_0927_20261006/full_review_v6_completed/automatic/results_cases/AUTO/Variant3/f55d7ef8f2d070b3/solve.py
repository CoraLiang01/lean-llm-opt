import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip()
activity_set = list(activity_ids)

def to_int_series(df, col):
    return df[col].astype(str).str.strip().astype(int)
NormalDuration = dict(zip(activity_ids, to_int_series(activities_df, 'NormalDuration')))
CrashDuration = dict(zip(activity_ids, to_int_series(activities_df, 'CrashDuration')))
CrashCostPerDay = dict(zip(activity_ids, to_int_series(activities_df, 'CrashCostPerDay')))

def parse_preds(s):
    s = str(s).strip()
    if s == '':
        return []
    return [x.strip() for x in re.split('[;,]', s) if x.strip() != '']
Predecessors = dict(zip(activity_ids, activities_df['Predecessors'].apply(parse_preds)))
deadline_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_set, lb=0, ub={i: NormalDuration[i] - CrashDuration[i] for i in activity_set}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z_vars[i] for i in activity_set)), gp.GRB.MINIMIZE)
for i in activity_set:
    for j in Predecessors[i]:
        if j not in activity_set:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_set:
    m.addConstr(T_var >= s_vars[i] + (NormalDuration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= ProjectDeadline, name='deadline')
m.optimize()