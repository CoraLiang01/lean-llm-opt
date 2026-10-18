import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
NormalDuration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
CrashDuration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
CrashCostPerDay = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
CrashLowerBound = {i: 0 for i in activity_ids}
CrashUpperBound = {i: NormalDuration[i] - CrashDuration[i] for i in activity_ids}

def parse_preds(s):
    s = s.strip()
    if s == '':
        return []
    return [pred.strip() for pred in re.split(';', s) if pred.strip() != '']
Predecessors = {row['Activity']: parse_preds(row['Predecessors']) for _, row in activities_df.iterrows()}
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[CrashUpperBound[i] for i in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in Predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{i}_after_{j}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (NormalDuration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= ProjectDeadline, name='deadline')
m.optimize()