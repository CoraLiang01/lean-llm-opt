import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activities_df['Activity'] = activities_df['Activity'].str.strip()
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
crash_day_ub = {act: normal_duration[act] - crash_duration[act] for act in activity_ids}

def parse_preds(s):
    s = s.strip()
    if not s:
        return []
    return [pred.strip() for pred in re.split(';', s) if pred.strip()]
predecessors = {row['Activity']: parse_preds(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for act in activity_ids:
    m.addConstr(z_vars[act] >= 0, name=f'z_lb_{act}')
    m.addConstr(z_vars[act] <= crash_day_ub[act], name=f'z_ub_{act}')
for act in activity_ids:
    for pred in predecessors[act]:
        if pred not in activity_ids:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in activity list.")
        m.addConstr(s_vars[act] >= s_vars[pred] + (normal_duration[pred] - z_vars[pred]), name=f'prec_{pred}_to_{act}')
for act in activity_ids:
    m.addConstr(T_var >= s_vars[act] + (normal_duration[act] - z_vars[act]), name=f'T_ge_finish_{act}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()